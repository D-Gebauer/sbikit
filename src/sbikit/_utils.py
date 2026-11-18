"""Shared helpers: config dataclasses, seeding, input checks, devices, logging and registries.

Internal module; the public pieces are re-exported from :mod:`sbikit`
(:func:`set_verbosity`, :func:`resolve_device`).
"""
from __future__ import annotations

import contextlib
import dataclasses
import difflib
import logging
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

__all__ = ["ConfigBase", "Registry", "seeded", "resolve_device", "as_numpy", "as_2d", "check_pair",
           "check_finite", "holdout_split", "logger", "set_verbosity", "did_you_mean", "NotFittedError"]

logger = logging.getLogger("sbikit")


# ------------------------------------------------------------------------------ logging
def _default_handler():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.set_name("sbikit-default")
    return handler


def _setup_logging():
    """Give the ``sbikit`` logger a plain stdout handler at INFO unless the application configured
    it already (same approach as optuna)."""
    if not logger.handlers:
        logger.addHandler(_default_handler())
        logger.propagate = False
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


def set_verbosity(level) -> None:
    """Set how much sbikit reports: ``"debug"``, ``"info"`` (default: training and sampling
    summaries), ``"warning"`` (problems only), ``"error"`` or ``"silent"``; or a ``logging`` level.

    All messages go through the ``"sbikit"`` logger, so applications can also attach their own
    handlers to ``logging.getLogger("sbikit")``.
    """
    if isinstance(level, str):
        names = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING,
                 "error": logging.ERROR, "silent": logging.CRITICAL + 1}
        if level.lower() not in names:
            raise ValueError(f"unknown verbosity {level!r}; use one of {sorted(names)}")
        level = names[level.lower()]
    logger.setLevel(int(level))


# ------------------------------------------------------------------------------ errors
class NotFittedError(RuntimeError):
    """An object was used before ``fit()`` (or after a fit that failed)."""


def did_you_mean(name: str, valid: Sequence[str]) -> str:
    """``" (did you mean 'x'?)"`` for the closest valid name, or an empty string. Names that differ
    only in case (``s8`` / ``S8``) always match."""
    name, valid = str(name), [str(v) for v in valid]
    same = [v for v in valid if v.lower() == name.lower()]
    if not same:
        lower = {v.lower(): v for v in valid}
        same = [lower[c] for c in difflib.get_close_matches(name.lower(), list(lower), n=1)]
    return f" (did you mean {same[0]!r}?)" if same else ""


def unknown(kind: str, name, valid: Sequence[str]) -> ValueError:
    """A ValueError for an unknown option, listing the valid ones."""
    return ValueError(f"unknown {kind} {name!r}{did_you_mean(name, valid)}; valid: {sorted(str(v) for v in valid)}")


# ------------------------------------------------------------------------------ configs
class ConfigBase:
    """Mixin for the config dataclasses.

    ``make`` accepts None (defaults), an instance (returned as is), a dict or keyword overrides;
    an unknown key raises a ``TypeError`` that names the closest valid field. Subclasses list the
    allowed values of string options in ``_choices`` (a dict, or a property for choices that come
    from a registry) and implement further checks in ``validate``.
    """

    _choices: Mapping[str, Sequence] = {}

    @classmethod
    def fields(cls) -> list:
        """Names of the config's fields."""
        return [f.name for f in dataclasses.fields(cls)]

    @classmethod
    def make(cls, cfg=None, **overrides):
        """A config from None (defaults), an instance, or a dict; ``overrides`` replace fields.

        Raises:
            TypeError: for a key that is not a field of the config.
        """
        if isinstance(cfg, cls):
            return dataclasses.replace(cfg, **overrides) if overrides else cfg
        if cfg is not None and not isinstance(cfg, Mapping):
            raise TypeError(f"{cls.__name__}: expected None, a dict or a {cls.__name__}, got {type(cfg).__name__}")
        values = {**dict(cfg or {}), **overrides}
        valid = cls.fields()
        bad = [k for k in values if k not in valid]
        if bad:
            hints = "; ".join(f"{k!r}{did_you_mean(k, valid)}" for k in bad)
            raise TypeError(f"{cls.__name__} has no field {hints}. Fields: {valid}")
        return cls(**values)

    def __post_init__(self):
        self._normalise()
        choices = self._choices
        for name, allowed in dict(choices).items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{type(self).__name__}.{name} = {value!r}{did_you_mean(value, allowed)}; "
                                 f"valid: {sorted(str(a) for a in allowed)}")
        self.validate()

    def _normalise(self) -> None:
        """Hook: convert field values to their canonical types (called before the checks)."""

    def validate(self) -> None:
        """Hook: raise ``ValueError`` for invalid values."""

    def to_dict(self) -> dict:
        """The fields as a plain dict (tuples become lists), e.g. for JSON records."""
        def plain(v):
            if isinstance(v, (tuple, list)):
                return [plain(x) for x in v]
            if isinstance(v, Mapping):
                return {k: plain(x) for k, x in v.items()}
            return v
        return {k: plain(v) for k, v in dataclasses.asdict(self).items()}

    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"{type(self).__name__}: {message}")


def parse_widths(widths) -> tuple:
    """Hidden-layer widths as a tuple of ints, from a sequence, an int or a string ``"256,128"``
    (an empty sequence or string means no hidden layer)."""
    if isinstance(widths, str):
        widths = [w for w in widths.replace(" ", "").split(",") if w]
    elif isinstance(widths, (int, np.integer)):
        widths = [widths]
    try:
        out = tuple(int(w) for w in widths)
        exact = all(float(w) == o for w, o in zip(widths, out))
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or any(w < 1 for w in out):
        raise ValueError(f"widths must be positive integers, e.g. (256, 128) or '256,128'; got {widths!r}")
    return out


# ------------------------------------------------------------------------------ registries
class Registry(dict):
    """Name -> object, with helpful errors. ``register`` works as a function or a decorator.

    Args:
        kind: what the entries are, for error messages (e.g. ``"flow"``).
    """

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def register(self, name: str, obj=None, overwrite: bool = False):
        """Add ``obj`` under ``name``; without ``obj`` returns a decorator.

        Raises:
            ValueError: if the name is taken by another object and ``overwrite`` is False.
        """
        def add(o):
            if name in self and self[name] is not o and not overwrite:
                raise ValueError(f"{self.kind} {name!r} is already registered; pass overwrite=True to replace it")
            self[name] = o
            return o
        return add if obj is None else add(obj)

    def get(self, name, default=None):
        """The entry, or ``default``."""
        return dict.get(self, name, default)

    def __missing__(self, name):
        raise unknown(self.kind, name, list(self))


# ------------------------------------------------------------------------------ seeding and devices
@contextlib.contextmanager
def seeded(seed: Optional[int]):
    """Run a block with torch's random generators seeded, then restore their previous state.

    With ``seed=None`` nothing is changed (the block uses the caller's global state). Unlike
    ``torch.manual_seed``, this does not alter the random stream of the calling code, so that e.g.
    training several models in a loop does not make them identical.
    """
    if seed is None:
        yield
        return
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        yield


def resolve_device(device=None) -> str:
    """``"auto"`` (or None) -> ``"cuda"`` if available, else ``"mps"`` if available, else ``"cpu"``;
    anything else is checked and returned as a string.

    Raises:
        RuntimeError: if a CUDA device is requested but CUDA is not available.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    name = str(device)
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is not available; use device='cpu' or device='auto'")
    return name


# ------------------------------------------------------------------------------ arrays
def as_numpy(a, dtype=np.float64) -> np.ndarray:
    """A numpy array from an array-like or a tensor (on any device, with or without grad)."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=dtype)


def as_2d(a, name: str, n_cols: Optional[int] = None, dtype=np.float64, what: str = "rows") -> np.ndarray:
    """``a`` as an (N, C) array, with a clear error otherwise.

    Args:
        a: array-like or tensor.
        name: the argument's name, for the error message.
        n_cols: required number of columns (None: any).
        dtype: dtype of the result.
        what: what the rows are (``"simulations"``, ...), for the error message.

    Raises:
        ValueError: if ``a`` is not 2D, is empty or has the wrong number of columns.
    """
    a = as_numpy(a, dtype)
    if a.ndim != 2:
        hint = " (for a single column use a[:, None]; for a single row a[None])" if a.ndim == 1 else ""
        raise ValueError(f"{name} must be 2D (N {what} x columns), got shape {a.shape}{hint}")
    if n_cols is not None and a.shape[1] != n_cols:
        raise ValueError(f"{name} has {a.shape[1]} columns, expected {n_cols}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} is empty (shape {a.shape})")
    return a


def check_finite(a: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` if ``a`` holds NaN or inf, saying how many rows are affected."""
    ok = np.isfinite(a)
    if not ok.all():
        rows = int((~ok.reshape(len(a), -1).all(1)).sum())
        raise ValueError(f"{name} contains NaN or inf in {rows} of {len(a)} rows; remove or repair those "
                         "simulations first")


def check_pair(x, theta, x_name: str = "x", theta_name: str = "theta", n_x: Optional[int] = None,
               n_theta: Optional[int] = None, finite: bool = True) -> tuple:
    """Validated (x (N, D), theta (N, P)) float64 arrays of matching length.

    Raises:
        ValueError: for wrong dimensions, different numbers of rows, wrong column counts or
            (``finite``) non-finite values.
    """
    x, theta = as_2d(x, x_name, n_x, what="simulations"), as_2d(theta, theta_name, n_theta, what="simulations")
    if len(x) != len(theta):
        raise ValueError(f"{x_name} has {len(x)} rows but {theta_name} has {len(theta)}; they must be the same "
                         "simulations in the same order")
    if finite:
        check_finite(x, x_name)
        check_finite(theta, theta_name)
    return x, theta


def holdout_split(n: int, fraction: float, seed: Optional[int] = None) -> tuple:
    """Random (train, validation) index arrays, with ``round(fraction * n)`` validation rows
    (at least 1, at most n - 1).

    Raises:
        ValueError: if ``n < 2`` or the fraction is not in (0, 1).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {fraction}")
    if n < 2:
        raise ValueError("need at least 2 simulations to hold out a validation set")
    n_val = min(max(int(round(fraction * n)), 1), n - 1)
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def plain(obj: Any):
    """``obj`` with numpy scalars, numpy arrays, tuples and paths replaced by Python scalars,
    tensors, lists and strings, recursively; what ``torch.load(weights_only=True)`` can read back."""
    import pathlib
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "US":
            return obj.tolist()
        if obj.dtype.kind == "O":
            return [plain(o) for o in obj.tolist()]
        return torch.as_tensor(np.ascontiguousarray(obj)).clone()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, Mapping):
        return {(k.item() if isinstance(k, np.generic) else k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset, range)):
        return [plain(o) for o in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return plain(dataclasses.asdict(obj))
    return obj
