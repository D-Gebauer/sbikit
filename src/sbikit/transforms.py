"""Parameter maps psi -> theta for :func:`~sbikit.sampling.sample_posteriors` (``transform=``).

The flow is conditioned on theta (the labels of the training simulations); the sampler moves in
psi (the parameters of the model being inferred), with the prior on psi. Typical uses:

- a flow trained on one intrinsic-alignment amplitude per tomographic bin, inferred with a
  few-parameter model that predicts those amplitudes;
- reparametrisations (e.g. sample S8 instead of sigma8) without retraining.

A transform acts on torch tensors (..., P_in) -> (..., P_out). It is called under ``no_grad`` (it
does not have to be differentiable) and must return finite values inside the prior support; points
with non-finite outputs are rejected by the sampler.

Example: a flow conditioned on (Om, sigma8), sampled in (Om, S8)::

    to_sigma8 = Columns(["Om", "S8"], keep=["Om"], derived=["sigma8"],
                        fn=lambda psi: psi[..., 1:2] / torch.sqrt(psi[..., 0:1] / 0.3))
    post = sample_posteriors(nle, x_obs, prior_on_Om_S8, transform=to_sigma8, init_centre=...)
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

__all__ = ["Transform", "Identity", "Columns"]


class Transform:
    """Base class: ``names_in`` (psi) and ``names_out`` (theta, the flow's labels).

    Subclasses implement ``__call__(psi) -> theta`` and, if they hold tensors, :meth:`to`.
    """

    names_in: Sequence[str]
    names_out: Sequence[str]

    def __call__(self, psi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device) -> "Transform":
        """The transform on ``device`` (default: itself, for transforms without tensors)."""
        return self

    def check(self, flow_names: Sequence[str]) -> None:
        """Raise unless the output columns are the flow's labels, in order."""
        if list(self.names_out) != list(flow_names):
            raise ValueError(f"transform outputs {list(self.names_out)}, flow is conditioned on {list(flow_names)}")


class Identity(Transform):
    """theta = psi, with the given column names."""

    def __init__(self, names: Sequence[str]):
        self.names_in = self.names_out = list(names)

    def __call__(self, psi):
        return psi


class Columns(Transform):
    """theta = [psi[..., keep], fn(psi)]: keep some psi columns, append derived ones.

    Args:
        names_in: names of the psi columns.
        keep: names of psi columns passed through (in this order).
        derived: names of the columns ``fn`` returns.
        fn: callable (..., P_in) tensor -> (..., len(derived)) tensor.
        order: optional final column order (names from keep + derived); default keep + derived.
    """

    def __init__(self, names_in: Sequence[str], keep: Sequence[str], derived: Sequence[str],
                 fn: Callable[[torch.Tensor], torch.Tensor], order: Optional[Sequence[str]] = None):
        self.names_in = list(names_in)
        self.keep_idx = [self.names_in.index(n) for n in keep]
        self.fn = fn
        cols = list(keep) + list(derived)
        self.names_out = list(order) if order is not None else cols
        if sorted(self.names_out) != sorted(cols):
            raise ValueError("order must be a permutation of keep + derived")
        self.perm = [cols.index(n) for n in self.names_out]

    def __call__(self, psi):
        out = torch.cat([psi[..., self.keep_idx], self.fn(psi)], -1)
        return out[..., self.perm]

    def to(self, device):
        """A copy with ``fn.to(device)`` if ``fn`` has a ``to`` method (e.g. an ``nn.Module``), else itself."""
        if hasattr(self.fn, "to"):
            new = object.__new__(Columns)
            new.__dict__.update(self.__dict__)
            new.fn = self.fn.to(device)
            return new
        return self
