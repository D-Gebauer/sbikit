"""Common compressor interface and registries."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .._utils import NotFittedError, Registry, as_numpy

__all__ = ["Compressor", "REGISTRY", "register", "compressor_from_state", "NETWORKS", "register_network"]

REGISTRY = Registry("compressor kind")
NETWORKS = Registry("network")


def register(cls):
    """Class decorator: make a compressor loadable by its ``kind`` (:func:`compressor_from_state`,
    :func:`sbikit.io.load`).

    Raises:
        ValueError: if another class is already registered under that ``kind``.
    """
    existing = REGISTRY.get(cls.kind)
    if existing is not None and existing is not cls and existing.__qualname__ != cls.__qualname__:
        raise ValueError(f"compressor kind {cls.kind!r} is already registered for {existing.__name__}; "
                         "give the new class its own kind")
    REGISTRY[cls.kind] = cls
    return cls


def register_network(name: str, factory=None, overwrite: bool = False):
    """Make a network architecture selectable by name (``NetConfig.network``,
    ``IMNNConfig.network``) and restorable from checkpoints.

    Args:
        name: the name used in the configs.
        factory: ``factory(n_in, n_out, config) -> torch.nn.Module`` mapping (N, n_in) to
            (N, n_out); ``config`` is the compressor's config, whose ``network_kwargs`` dict is
            meant for the factory's own options. Without ``factory`` this returns a decorator.
        overwrite: replace an existing network of that name.

    Example:
        >>> @register_network("cnn")
        ... def cnn(n_in, n_out, config):
        ...     return MyCNN(n_in, n_out, **config.network_kwargs)
        >>> vmim = VMIMCompressor(3, {"network": "cnn", "network_kwargs": {"channels": 16}})
    """
    return NETWORKS.register(name, factory, overwrite=overwrite)


class Compressor:
    """Maps data vectors (N, D) to summaries (N, n_out).

    ``fit`` returns a validation loss that is comparable between trials of the same method (what a
    hyperparameter search minimises), or nan for compressors without one (MOPED).

    To add a compressor, subclass this, set a unique ``kind``, implement ``fit``, ``__call__``,
    ``state`` and ``from_state``, and decorate the class with :func:`register`. ``state()`` must
    return plain Python values and tensors only (no numpy arrays or custom objects), so that the
    checkpoint loads with ``weights_only=True``.
    """
    kind = "base"
    n_out: int

    def fit(self, x: Optional[np.ndarray] = None, theta: Optional[np.ndarray] = None,
            x_val: Optional[np.ndarray] = None, theta_val: Optional[np.ndarray] = None) -> float:
        """Train on data vectors ``x`` (N, D) with parameters ``theta`` (N, P), validating on
        ``x_val``, ``theta_val`` (compressors that train networks hold out part of the training
        set if no validation set is given). Compressors trained on a fiducial set ignore the
        arguments. Returns the validation loss."""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Summaries (N, n_out), float32, of the data vectors ``x`` (N, D); a single data vector
        (D,) gives (n_out,)."""
        raise NotImplementedError

    def to(self, device) -> "Compressor":
        """Move the compressor to ``device`` (nothing to do for compressors without networks);
        returns itself."""
        return self

    def state(self) -> dict:
        """Plain serialisable state, with the compressor's ``kind``."""
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict, device="cpu") -> "Compressor":
        """Rebuild a fitted compressor from :meth:`state`."""
        raise NotImplementedError

    # -- helpers for subclasses
    def _not_fitted(self) -> NotFittedError:
        return NotFittedError(f"{type(self).__name__}: call fit() first")

    @staticmethod
    def _batch(x, n_in: Optional[int] = None, name: str = "x") -> tuple:
        """``x`` as a float64 (N, D) array and whether it was a single (D,) vector.

        Raises:
            ValueError: if ``x`` is not 1D or 2D, or does not have ``n_in`` columns.
        """
        x = as_numpy(x)
        single = x.ndim == 1
        if single:
            x = x[None]
        if x.ndim != 2:
            raise ValueError(f"{name} must be (N, D) data vectors or a single (D,) vector, got shape {x.shape}")
        if n_in is not None and x.shape[1] != n_in:
            raise ValueError(f"{name} has {x.shape[1]} columns, the compressor was fitted on {n_in}")
        return x, single


def compressor_from_state(state: dict, device="cpu") -> Compressor:
    """The fitted compressor described by ``state``, whatever its class.

    Raises:
        ValueError: if no compressor is registered for ``state["kind"]`` (import the module that
            defines a custom compressor before loading it).
    """
    return REGISTRY[state["kind"]].from_state(state, device)
