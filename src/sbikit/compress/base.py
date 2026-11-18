"""Common compressor interface and registry."""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["Compressor", "REGISTRY", "register", "compressor_from_state"]

REGISTRY: dict = {}


def register(cls):
    """Class decorator: make a compressor loadable by its ``kind`` (:func:`compressor_from_state`)."""
    REGISTRY[cls.kind] = cls
    return cls


class Compressor:
    """Maps data vectors (N, D) to summaries (N, n_out).

    ``fit`` returns a validation loss that is comparable between trials of the same method (what a
    hyperparameter search minimises), or nan for compressors without one (MOPED).

    To add a compressor, subclass this, set a unique ``kind``, implement the four methods below and
    decorate the class with :func:`register`.
    """
    kind = "base"
    n_out: int

    def fit(self, x_trn: Optional[np.ndarray] = None, t_trn: Optional[np.ndarray] = None,
            x_val: Optional[np.ndarray] = None, t_val: Optional[np.ndarray] = None) -> float:
        """Train on data vectors ``x_trn`` (N, D) with parameters ``t_trn`` (N, P), validating on
        ``x_val``, ``t_val``. Compressors trained on a fiducial set ignore the arguments.
        Returns the validation loss."""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Summaries (N, n_out), float32, of the data vectors ``x`` (N, D)."""
        raise NotImplementedError

    def state(self) -> dict:
        """Plain serialisable state, with the compressor's ``kind``."""
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict, device="cpu") -> "Compressor":
        """Rebuild a fitted compressor from :meth:`state`."""
        raise NotImplementedError


def compressor_from_state(state: dict, device="cpu") -> Compressor:
    """The fitted compressor described by ``state``, whatever its class."""
    return REGISTRY[state["kind"]].from_state(state, device)
