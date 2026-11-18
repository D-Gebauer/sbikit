"""Parameter-level blinding of posterior samples.

A key file holds a random seed and, per blinded parameter, a scale; the hidden offset of parameter
p is scale_p * U(-1, 1) drawn from that seed. Pipelines shift posterior samples of the data by the
offsets before anything is written or plotted, and record only the key's fingerprint. Validation on
simulations is unaffected (blind only the data posteriors). Unblinding = :meth:`Blinder.unblind`
once the agreed criteria are met.

Keep the key file outside version control and outside shared result folders.

Example::

    make_key("blind.key", {"S8": 0.05, "w0": 0.3})        # once; the file is never overwritten
    blinder = Blinder("blind.key")
    blinded = blinder.blind(post.samples, ["Om", "S8", "w0"])   # Om is not in the key: unchanged
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = ["make_key", "Blinder"]


def make_key(path, scales: Mapping[str, float]) -> str:
    """Write a new key and return its fingerprint.

    Args:
        path: key file to create (mode 0600). An existing file is never overwritten.
        scales: parameter name -> scale; the hidden offset is drawn from scale * U(-1, 1).
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a blinding key is never overwritten")
    key = {"seed": secrets.randbits(63), "scales": {k: float(v) for k, v in dict(scales).items()}}
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
    """

    def __init__(self, path):
        self.path = Path(path)
        key = json.loads(self.path.read_text())
        self._scales = dict(key["scales"])
        u = np.random.default_rng(int(key["seed"])).uniform(-1.0, 1.0, len(self._scales))
        self._offsets = {p: s * ui for (p, s), ui in zip(sorted(self._scales.items()), u)}
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]

    @property
    def params(self) -> list:
        """Names of the blinded parameters."""
        return sorted(self._scales)

    def _vector(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self._offsets.get(n, 0.0) for n in names])

    def blind(self, samples, names: Sequence[str]) -> np.ndarray:
        """samples (..., P) with columns ``names`` -> blinded copy (unknown names pass unchanged)."""
        return np.asarray(samples) + self._vector(names)

    def unblind(self, samples, names: Sequence[str]) -> np.ndarray:
        """Inverse of :meth:`blind`."""
        return np.asarray(samples) - self._vector(names)

    def __repr__(self):
        return f"Blinder({self.path.name}, params={self.params}, fingerprint={self.fingerprint})"
