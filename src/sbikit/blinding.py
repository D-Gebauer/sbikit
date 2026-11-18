"""Parameter-level blinding of posterior samples.

A key file holds a random seed and, per blinded parameter, a scale; the hidden offset of parameter
p is scale_p * U(-1, 1) drawn from that seed. Pipelines shift posterior samples of the data by the
offsets before anything is written or plotted, and record only the key's fingerprint. Validation on
simulations is unaffected (blind only the data posteriors). Unblinding = :meth:`Blinder.unblind`
once the agreed criteria are met.

Keep the key file outside version control and outside shared result folders, and never edit it: the
offsets depend on the seed and on the set of names in the file, and the fingerprint is a hash of
the file's bytes.

The names are matched exactly (``"s8"`` is not ``"S8"``); a blinded parameter that is missing from
``names`` raises, so that a spelling mismatch cannot leave samples unblinded. Arrays carry no mark
of being blinded: blinding twice shifts twice. Derived parameters must be computed from the blinded
samples (apply the transform to the output of :meth:`Blinder.blind`), not blinded separately.

Example::

    make_key("blind.key", {"S8": 0.05, "w0": 0.3})        # once; the file is never overwritten
    blinder = Blinder("blind.key")
    blinded = blinder.blind(post.samples, ["Om", "S8", "w0"])   # Om is not in the key: unchanged
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import secrets
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ._utils import did_you_mean

__all__ = ["make_key", "Blinder"]


def make_key(path, scales: Mapping[str, float]) -> str:
    """Write a new key and return its fingerprint.

    Args:
        path: key file to create (mode 0600). An existing file is never overwritten.
        scales: parameter name -> scale (finite, positive); the hidden offset is drawn from
            scale * U(-1, 1).

    Returns:
        The fingerprint of the new key (see :class:`Blinder`).

    Raises:
        TypeError: if ``scales`` is not a mapping of names to numbers.
        ValueError: if ``scales`` is empty or a scale is not finite and positive.
        FileExistsError: if ``path`` exists.
    """
    if not isinstance(scales, Mapping):
        raise TypeError(f"scales must be a mapping of parameter name -> scale, e.g. {{'S8': 0.05}}; got "
                        f"{type(scales).__name__}")
    if not scales:
        raise ValueError("scales is empty: name at least one parameter to blind")
    checked = {}
    for name, scale in scales.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"scales: parameter names must be non-empty strings, got {name!r}")
        if isinstance(scale, bool) or not isinstance(scale, (numbers.Real, np.number)):
            raise TypeError(f"scales[{name!r}] must be a number, got {scale!r}")
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(f"scales[{name!r}] must be finite and positive, got {scale!r}")
        checked[str(name)] = float(scale)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a blinding key is never overwritten")
    key = {"seed": secrets.randbits(63), "scales": checked}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(key, f)
    return Blinder(path).fingerprint


class Blinder:
    """Applies the hidden offsets of a key file written by :func:`make_key`.

    Args:
        path: the key file.

    Attributes:
        fingerprint: 16 hex digits of the key file's hash; safe to store with results, to show
            which key blinded them.

    Raises:
        ValueError: if the file is not a blinding key.
    """

    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        try:
            key = json.loads(raw)
            self._scales = {str(k): float(v) for k, v in key["scales"].items()}
            seed = int(key["seed"])
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ValueError(f"{self.path} is not a blinding key written by sbikit.blinding.make_key") from None
        u = np.random.default_rng(seed).uniform(-1.0, 1.0, len(self._scales))
        self._offsets = {p: s * ui for (p, s), ui in zip(sorted(self._scales.items()), u)}
        self.fingerprint = hashlib.sha256(raw).hexdigest()[:16]

    @property
    def params(self) -> list:
        """Names of the blinded parameters."""
        return sorted(self._scales)

    def _shift(self, samples, names, sign: float, missing: str) -> np.ndarray:
        if missing not in ("raise", "ignore"):
            raise ValueError(f"missing must be 'raise' or 'ignore', got {missing!r}")
        a = samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else np.asarray(samples)
        if a.dtype.kind != "f":
            a = a.astype(np.float64)
        if isinstance(names, str) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"names must be a sequence of parameter names, one per column; got {names!r}")
        names = [str(n) for n in names]
        if a.ndim == 0 or a.shape[-1] != len(names):
            raise ValueError(f"samples has shape {a.shape} but names has {len(names)} entries; names must give one "
                             "name per column of the last axis")
        if len(set(names)) != len(names):
            raise ValueError(f"names must be unique, got {names}")
        absent = [p for p in self.params if p not in names]
        if absent and missing == "raise":
            hints = ", ".join(f"{p!r}{did_you_mean(p, names)}" for p in absent)
            raise KeyError(f"blinded parameter(s) {hints} not in names {names}; these columns would stay unblinded. "
                           "Names are matched exactly. For an array that holds only some of the blinded "
                           "parameters on purpose, pass missing='ignore'")
        offsets = np.array([self._offsets.get(n, 0.0) for n in names])
        return a + (sign * offsets).astype(a.dtype)

    def blind(self, samples, names: Sequence[str], *, missing: str = "raise") -> np.ndarray:
        """Shift the blinded parameters by their hidden offsets.

        Args:
            samples: (..., P) array-like or tensor.
            names: the P parameter names of the last axis. Columns whose name is not in the key
                pass unchanged.
            missing: what to do if a parameter of the key is not in ``names``: ``"raise"``, or
                ``"ignore"`` for arrays that hold only some of the blinded parameters.

        Returns:
            A blinded copy as a numpy array of the same shape (and floating dtype).

        Raises:
            ValueError: if ``names`` does not have one unique name per column.
            KeyError: if a parameter of the key is not in ``names`` (``missing="raise"``).
        """
        return self._shift(samples, names, 1.0, missing)

    def unblind(self, samples, names: Sequence[str], *, missing: str = "raise") -> np.ndarray:
        """Inverse of :meth:`blind` (same arguments)."""
        return self._shift(samples, names, -1.0, missing)

    def __repr__(self):
        return f"Blinder({self.path.name}, params={self.params}, fingerprint={self.fingerprint})"
