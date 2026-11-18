"""Input/output scalers that work on numpy arrays and torch tensors and serialise to tensors.

Every scaler has ``fit(x)`` (statistics per column of an (N, D) array, returns itself),
``__call__(x)``, ``inverse(z)`` and ``state()`` (a dict that :func:`scaler_from_state` turns back into
the scaler). Calling a scaler returns the type it was given: arrays in, arrays out; tensors in,
tensors on the same device out. Integer inputs are converted to floating point.
``log_det(x)`` is the log of the absolute Jacobian determinant of the scaling at the rows of ``x``,
which turns a density of the scaled values into a density of the raw ones:
``log p(x) = log p_scaled(scaler(x)) + scaler.log_det(x)``.

To add a scaler, write a class with a unique ``kind``, the four methods above, and a ``state()``
that returns ``{"kind": kind, **constructor_arguments}``; then :func:`register_scaler` it. A
``log_det`` method is optional (without it the Jacobian is taken to be constant and left out). It can
then be selected by name wherever a scaler name is accepted (e.g. ``NetConfig.input_scaling``) and
is restored when a checkpoint is loaded.
"""
from __future__ import annotations

import numpy as np
import torch

from ._utils import Registry

__all__ = ["ZScore", "RobustAsinh", "Identity", "SCALERS", "register_scaler", "make_scaler", "scaler_from_state"]


def _float(x):
    """``x`` unchanged if it is floating point, else converted (integer tensors -> the default
    float dtype, other array-likes -> float64)."""
    if isinstance(x, torch.Tensor):
        return x if x.is_floating_point() else x.to(torch.get_default_dtype())
    x = np.asarray(x)
    return x if x.dtype.kind == "f" else x.astype(np.float64)


def _like(a, ref):
    """``a`` (a tensor) as the type/device of ``ref`` (which is floating point)."""
    if isinstance(ref, torch.Tensor):
        return a.to(device=ref.device, dtype=ref.dtype)
    return a.cpu().numpy().astype(ref.dtype)


def _stats(x):
    """A float64 CPU tensor of the (N, D) data ``x`` (array-like or tensor on any device)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double()
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


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

    def log_det(self, x):
        """Zeros of shape ``x.shape[:-1]``."""
        return x.new_zeros(x.shape[:-1]) if isinstance(x, torch.Tensor) else np.zeros(np.shape(x)[:-1])

    def state(self):
        """Serialisable state."""
        return {"kind": self.kind}

    def __repr__(self):
        return "Identity()"


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
        x = _stats(x)
        self.mean = x.mean(0)
        std = x.std(0, correction=self.ddof) if len(x) > self.ddof else torch.zeros_like(self.mean)
        self.std = torch.where(std > 0, std, torch.ones_like(std))
        return self

    def __call__(self, x):
        x = _float(x)
        return (x - _like(self.mean, x)) / _like(self.std, x)

    def inverse(self, z):
        """Scaled values back to the original units."""
        z = _float(z)
        return z * _like(self.std, z) + _like(self.mean, z)

    def log_det(self, x):
        """log |det d scaled / d x| = -sum(log std), the same for every row; shape ``x.shape[:-1]``."""
        x = _float(x)
        const = -torch.log(self.std).sum()
        if isinstance(x, torch.Tensor):
            return const.to(device=x.device, dtype=x.dtype).expand(x.shape[:-1])
        return np.full(x.shape[:-1], float(const))

    def state(self):
        """Serialisable state."""
        return {"kind": self.kind, "mean": self.mean, "std": self.std, "ddof": self.ddof}

    def __repr__(self):
        return f"ZScore({'unfitted' if self.mean is None else f'{len(self.mean)} columns'})"


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
        x = _stats(x)
        self.median = x.median(0).values
        mad = 1.4826 * (x - self.median).abs().median(0).values
        std = x.std(0) if len(x) > 1 else torch.zeros_like(self.median)
        scale = torch.where(mad > 0, mad, std)
        self.scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return self

    def __call__(self, x):
        x = _float(x)
        z = (x - _like(self.median, x)) / _like(self.scale, x)
        if isinstance(z, torch.Tensor):
            return self.k * torch.asinh(z / self.k)
        return self.k * np.arcsinh(z / self.k)

    def inverse(self, y):
        """Scaled values back to the original units."""
        y = _float(y)
        z = (torch.sinh(y / self.k) if isinstance(y, torch.Tensor) else np.sinh(y / self.k)) * self.k
        return z * _like(self.scale, y) + _like(self.median, y)

    def log_det(self, x):
        """log |det d scaled / d x| at the rows of ``x``; shape ``x.shape[:-1]``."""
        x = _float(x)
        scale = _like(self.scale, x)
        z = (x - _like(self.median, x)) / scale
        if isinstance(x, torch.Tensor):
            return (-torch.log(scale) - 0.5 * torch.log1p((z / self.k) ** 2)).sum(-1)
        return (-np.log(scale) - 0.5 * np.log1p((z / self.k) ** 2)).sum(-1)

    def state(self):
        """Serialisable state."""
        return {"kind": self.kind, "k": self.k, "median": self.median, "scale": self.scale}

    def __repr__(self):
        return f"RobustAsinh(k={self.k}, {'unfitted' if self.median is None else f'{len(self.median)} columns'})"


SCALERS = Registry("scaler")
SCALERS.update({"identity": Identity, "none": Identity, "zscore": ZScore, "robust_asinh": RobustAsinh})


def register_scaler(cls, name=None, overwrite: bool = False):
    """Make a scaler class selectable by name and loadable from checkpoints.

    Args:
        cls: the scaler class (see the module docstring for what it needs).
        name: the name to register it under (default ``cls.kind``).
        overwrite: replace an existing scaler of that name.

    Returns:
        ``cls``, so this works as a class decorator.
    """
    return SCALERS.register(name or cls.kind, cls, overwrite=overwrite)


def make_scaler(kind, **kwargs):
    """A scaler from a name (``"identity"``/``"none"``, ``"zscore"``, ``"robust_asinh"`` or a
    registered one) and its constructor arguments, or a scaler instance (returned as is).

    Raises:
        ValueError: for an unknown name.
    """
    if not isinstance(kind, str):
        if kwargs:
            raise TypeError("constructor arguments cannot be combined with a scaler instance")
        return kind
    return SCALERS[kind](**kwargs)


def scaler_from_state(state):
    """The scaler described by a ``state()`` dict."""
    s = dict(state)
    kind = s.pop("kind")
    return make_scaler(kind, **s)
