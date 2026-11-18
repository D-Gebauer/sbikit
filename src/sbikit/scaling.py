"""Input/output scalers that work on numpy arrays and torch tensors and serialise to tensors.

Every scaler has ``fit(x)`` (statistics per column of an (N, D) array, returns itself),
``__call__(x)``, ``inverse(z)`` and ``state()`` (a dict that :func:`scaler_from_state` turns back into
the scaler). Calling a scaler returns the type it was given: arrays in, arrays out; tensors in,
tensors on the same device out.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["ZScore", "RobustAsinh", "Identity", "make_scaler", "scaler_from_state"]


def _like(a, ref):
    """``a`` (a tensor) as the type/device of ``ref``."""
    if isinstance(ref, torch.Tensor):
        return a.to(device=ref.device, dtype=ref.dtype)
    return a.cpu().numpy().astype(np.asarray(ref).dtype if np.asarray(ref).dtype.kind == "f" else np.float64)


class Identity:
    """No scaling."""
    kind = "identity"

    def fit(self, x):
        """Nothing to fit; returns itself."""
        return self

    def __call__(self, x):
        return x

    def inverse(self, z):
        """Returns ``z`` unchanged."""
        return z

    def state(self):
        """Serialisable state."""
        return {"kind": self.kind}


class ZScore:
    """(x - mean) / std per column; a zero std is replaced by 1.

    Args:
        mean, std: (D,) statistics, if they are known already; otherwise call :meth:`fit`.
        ddof: delta degrees of freedom of the std in :meth:`fit`.
    """
    kind = "zscore"

    def __init__(self, mean=None, std=None, ddof: int = 0):
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float64)
        self.std = None if std is None else torch.as_tensor(std, dtype=torch.float64)
        self.ddof = ddof

    def fit(self, x):
        """Take mean and std from the columns of ``x`` (N, D); returns itself."""
        x = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x).double()
        self.mean = x.mean(0)
        std = x.std(0, correction=self.ddof)
        self.std = torch.where(std > 0, std, torch.ones_like(std))
        return self

    def __call__(self, x):
        return (x - _like(self.mean, x)) / _like(self.std, x)

    def inverse(self, z):
        """Scaled values back to the original units."""
        return z * _like(self.std, z) + _like(self.mean, z)

    def state(self):
        """Serialisable state."""
        return {"kind": self.kind, "mean": self.mean, "std": self.std, "ddof": self.ddof}


class RobustAsinh:
    """z = (x - median) / (1.4826 MAD), then k asinh(z / k): linear in the bulk, logarithmic tails.

    Robust against the occasional extreme simulation, which would dominate a z-score. Columns with a
    zero MAD fall back to the std, and to 1 if that is zero too.

    Args:
        k: where the tails start to be compressed, in robust sigmas.
        median, scale: (D,) statistics, if they are known already; otherwise call :meth:`fit`.
    """
    kind = "robust_asinh"

    def __init__(self, k: float = 5.0, median=None, scale=None):
        self.k = float(k)
        self.median = None if median is None else torch.as_tensor(median, dtype=torch.float64)
        self.scale = None if scale is None else torch.as_tensor(scale, dtype=torch.float64)

    def fit(self, x):
        """Take median and scale from the columns of ``x`` (N, D); returns itself."""
        x = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x).double()
        self.median = x.median(0).values
        mad = 1.4826 * (x - self.median).abs().median(0).values
        std = x.std(0)
        scale = torch.where(mad > 0, mad, std)
        self.scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return self

    def __call__(self, x):
        z = (x - _like(self.median, x)) / _like(self.scale, x)
        if isinstance(z, torch.Tensor):
            return self.k * torch.asinh(z / self.k)
        return self.k * np.arcsinh(z / self.k)

    def inverse(self, y):
        """Scaled values back to the original units."""
        z = (torch.sinh(y / self.k) if isinstance(y, torch.Tensor) else np.sinh(y / self.k)) * self.k
        return z * _like(self.scale, y) + _like(self.median, y)

    def state(self):
        """Serialisable state."""
        return {"kind": self.kind, "k": self.k, "median": self.median, "scale": self.scale}


def make_scaler(kind: str, **kwargs):
    """A new scaler by name: ``"identity"`` (or ``"none"``), ``"zscore"`` or ``"robust_asinh"``."""
    return {"identity": Identity, "none": Identity, "zscore": ZScore, "robust_asinh": RobustAsinh}[kind](**kwargs)


def scaler_from_state(state):
    """The scaler described by a ``state()`` dict."""
    s = dict(state)
    kind = s.pop("kind")
    return make_scaler(kind, **s)
