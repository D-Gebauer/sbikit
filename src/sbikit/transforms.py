"""Parameter maps psi -> theta for :func:`~sbikit.sampling.sample_posteriors` (``transform=``).

The flow is conditioned on theta (the labels of the training simulations); the sampler moves in
psi (the parameters of the model being inferred), with the prior on psi. Typical uses:

- a flow trained on one intrinsic-alignment amplitude per tomographic bin, inferred with a
  few-parameter model that predicts those amplitudes;
- reparametrisations (e.g. sample S8 instead of sigma8) without retraining.

A transform acts on torch tensors (..., P_in) -> (..., P_out). It is called under ``no_grad`` (it
does not have to be differentiable) and must return finite values inside the prior support; points
with non-finite outputs are rejected by the sampler. If a transform has ``names_in`` and the prior
has names, :func:`~sbikit.sampling.sample_posteriors` requires them to agree.

Example: a flow conditioned on (Om, sigma8), sampled in (Om, S8)::

    to_sigma8 = Columns(["Om", "S8"], keep=["Om"], derived=["sigma8"],
                        fn=lambda psi: psi[..., 1:2] / torch.sqrt(psi[..., 0:1] / 0.3))
    post = sample_posteriors(nle, x_obs, prior_on_Om_S8, transform=to_sigma8, init_centre=...)
"""
from __future__ import annotations

import copy
from typing import Callable, Optional, Sequence

import torch

from ._utils import did_you_mean, resolve_device

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
        """Raise ``ValueError`` unless the output columns are the flow's labels, in order."""
        if list(self.names_out) != list(flow_names):
            raise ValueError(f"transform outputs {list(self.names_out)}, flow is conditioned on {list(flow_names)}")

    def check_input(self, prior_names: Sequence[str]) -> None:
        """Raise ``ValueError`` unless the input columns are the prior's parameters, in order."""
        if list(self.names_in) != list(prior_names):
            raise ValueError(f"transform expects the parameters {list(self.names_in)}, the prior has "
                             f"{list(prior_names)}")

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(names_in={list(getattr(self, 'names_in', []))}, "
                f"names_out={list(getattr(self, 'names_out', []))})")


class Identity(Transform):
    """theta = psi, with the given column names."""

    def __init__(self, names: Sequence[str]):
        self.names_in = self.names_out = list(names)

    def __call__(self, psi):
        return psi


def _hint(name, valid):
    """A suggestion for a misspelled name; names that differ only in case count as close."""
    same = [v for v in valid if str(v).lower() == str(name).lower()]
    return f" (did you mean {same[0]!r}?)" if same else did_you_mean(name, valid)


def _unique(names, what):
    repeated = sorted({n for n in names if list(names).count(n) > 1})
    if repeated:
        raise ValueError(f"Columns: {what} must be unique; repeated: {repeated}")


class Columns(Transform):
    """theta = [psi[..., keep], fn(psi)]: keep some psi columns, append derived ones.

    Args:
        names_in: names of the psi columns.
        keep: names of psi columns passed through (in this order).
        derived: names of the columns ``fn`` returns.
        fn: callable (..., P_in) tensor -> (..., len(derived)) tensor.
        order: optional final column order (names from keep + derived); default keep + derived.

    Raises:
        ValueError: for repeated names, a name in ``keep`` that is not in ``names_in``, or an
            ``order`` that is not a permutation of keep + derived; on call, if ``fn`` does not
            return a (..., len(derived)) tensor.

    Example:
        >>> tr = Columns(["Om", "S8"], keep=["Om"], derived=["sigma8"],
        ...              fn=lambda psi: psi[..., 1:2] / torch.sqrt(psi[..., 0:1] / 0.3))
        >>> tr(torch.tensor([[0.3, 0.8]]))
        tensor([[0.3000, 0.8000]])
    """

    def __init__(self, names_in: Sequence[str], keep: Sequence[str], derived: Sequence[str],
                 fn: Callable[[torch.Tensor], torch.Tensor], order: Optional[Sequence[str]] = None):
        self.names_in = list(names_in)
        keep, derived = list(keep), list(derived)
        cols = keep + derived
        _unique(self.names_in, "names_in")
        _unique(cols, "the names in keep + derived")
        for n in keep:
            if n not in self.names_in:
                raise ValueError(f"Columns: keep has {n!r}, which is not in names_in{_hint(n, self.names_in)}; "
                                 f"names_in: {self.names_in}")
        self.keep_idx = [self.names_in.index(n) for n in keep]
        self.fn = fn
        self.n_derived = len(derived)
        self.names_out = list(order) if order is not None else cols
        if sorted(map(str, self.names_out)) != sorted(map(str, cols)):
            raise ValueError(f"Columns: order must be a permutation of keep + derived = {cols}; got {self.names_out}")
        self.perm = [cols.index(n) for n in self.names_out]

    def __call__(self, psi):
        if psi.shape[-1] != len(self.names_in):
            raise ValueError(f"Columns: expected {len(self.names_in)} input columns {self.names_in}, got a tensor "
                             f"of shape {tuple(psi.shape)}")
        new = self.fn(psi)
        expected = (*psi.shape[:-1], self.n_derived)
        if not isinstance(new, torch.Tensor) or tuple(new.shape) != expected:
            got = tuple(new.shape) if isinstance(new, torch.Tensor) else type(new).__name__
            raise ValueError(f"Columns: fn must return a tensor of shape {expected} (one column per derived name) "
                             f"for an input of shape {tuple(psi.shape)}; got {got}. For a single column, keep the "
                             "last axis: psi[..., i:i + 1] or result[..., None]")
        out = torch.cat([psi[..., self.keep_idx], new], -1)
        return out[..., self.perm]

    def to(self, device):
        """A copy whose ``fn`` is a deep copy moved to ``device``, if ``fn`` has a ``to`` method (e.g. an
        ``nn.Module``); the given ``fn`` stays where it is. Otherwise the transform itself."""
        if hasattr(self.fn, "to"):
            new = copy.copy(self)
            new.fn = copy.deepcopy(self.fn).to(resolve_device(device))
            return new
        return self
