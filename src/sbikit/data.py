"""Containers for simulations and the fiducial/derivative set, and train/val/test splits.

sbikit never decides how an analysis splits its simulations: pass the splits explicitly (e.g. by
N-body realisation), or let :func:`resolve_splits` draw random ones. Random splits keep every
``group`` (e.g. one N-body run, or one cosmology) in a single split, so that correlated
simulations cannot leak between training and testing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = ["Dataset", "FiducialSet", "Splits", "random_splits", "resolve_splits"]


@dataclass
class Dataset:
    """Simulations for training and testing.

    Args:
        theta: (N, P) parameters the flow is conditioned on (the labels).
        x: (N, D) data vectors (or already compressed summaries).
        param_names: the P names of the columns of ``theta``.
        groups: optional (N,) integer ids; simulations that share an id (e.g. the footprints and
            noise realisations of one N-body run) are never split apart by random splits.
        blocks: optional name -> column indices of ``x``, for hierarchical compression
            (e.g. one block per field, tomographic bin or statistic).
        meta: optional name -> (N, ...) per-simulation arrays that are not labels (implicitly
            marginalised nuisances such as m and Delta z draws, n(z) realisation ids, the
            N-body run); carried through :meth:`subset` for diagnostics and re-labelling.
    """

    theta: np.ndarray
    x: np.ndarray
    param_names: Sequence[str]
    groups: Optional[np.ndarray] = None
    blocks: Optional[Mapping[str, np.ndarray]] = None
    meta: Optional[Mapping[str, np.ndarray]] = None

    def __post_init__(self):
        self.theta = np.asarray(self.theta, dtype=np.float64)
        self.x = np.asarray(self.x, dtype=np.float32)
        self.param_names = list(self.param_names)
        if self.theta.ndim != 2 or self.x.ndim != 2 or len(self.theta) != len(self.x):
            raise ValueError(f"theta {self.theta.shape} and x {self.x.shape} must be (N, P) and (N, D)")
        if self.theta.shape[1] != len(self.param_names):
            raise ValueError(f"{self.theta.shape[1]} parameter columns but {len(self.param_names)} names")
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
            if self.groups.shape != (len(self.x),):
                raise ValueError(f"groups must have shape ({len(self.x)},), got {self.groups.shape}")
        if self.blocks is not None:
            self.blocks = {k: np.asarray(v, dtype=np.int64) for k, v in self.blocks.items()}
            for k, v in self.blocks.items():
                if v.min() < 0 or v.max() >= self.x.shape[1]:
                    raise ValueError(f"block {k!r} indexes outside the {self.x.shape[1]} columns of x")
        if self.meta is not None:
            self.meta = {k: np.asarray(v) for k, v in self.meta.items()}
            for k, v in self.meta.items():
                if len(v) != len(self.x):
                    raise ValueError(f"meta {k!r} has {len(v)} rows, x has {len(self.x)}")

    def __len__(self):
        """Number of simulations."""
        return len(self.x)

    def index(self, names: Sequence[str]) -> list[int]:
        """Column indices of the given parameter names."""
        return [self.param_names.index(n) for n in names]

    def subset(self, idx) -> "Dataset":
        """The simulations ``idx`` (indices or a boolean mask) as a new Dataset; ``groups`` and
        ``meta`` are subset too, ``blocks`` are kept."""
        idx = np.asarray(idx)
        return Dataset(self.theta[idx], self.x[idx], self.param_names,
                       None if self.groups is None else self.groups[idx], self.blocks,
                       None if self.meta is None else {k: v[idx] for k, v in self.meta.items()})


@dataclass
class FiducialSet:
    """Simulations at a fiducial point, and seed-matched derivative simulations.

    Used by MOPED (covariance + mean derivatives) and by the IMNN (trained on these only).

    Args:
        x: (n_fid, D) independent simulations at the fiducial parameters (covariance).
        param_names: the K parameters with derivatives (the compression targets).
        theta_fid: (K,) fiducial values of those parameters.
        x_minus, x_plus: optional (K, n_d, D) simulations at theta_fid -/+ step for each parameter;
            row d of x_minus[k] and x_plus[k] must share the random seed (initial conditions, noise).
        delta: (K,) theta_plus - theta_minus for each parameter.
        dmu: optional (K, D) mean derivatives d mu / d theta, if they are known directly (e.g. from
            noiseless means). Overrides the finite differences of x_minus/x_plus for MOPED.
    """

    x: np.ndarray
    param_names: Sequence[str]
    theta_fid: np.ndarray
    x_minus: Optional[np.ndarray] = None
    x_plus: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None
    dmu: Optional[np.ndarray] = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.param_names = list(self.param_names)
        self.theta_fid = np.asarray(self.theta_fid, dtype=np.float64)
        k, d = len(self.param_names), self.x.shape[1]
        if self.theta_fid.shape != (k,):
            raise ValueError(f"theta_fid must have shape ({k},)")
        if (self.x_minus is None) != (self.x_plus is None):
            raise ValueError("give both x_minus and x_plus, or neither")
        if self.x_minus is not None:
            self.x_minus = np.asarray(self.x_minus, dtype=np.float64)
            self.x_plus = np.asarray(self.x_plus, dtype=np.float64)
            if self.x_minus.shape != self.x_plus.shape or self.x_minus.shape[0] != k or self.x_minus.shape[2] != d:
                raise ValueError(f"x_minus/x_plus must both be ({k}, n_d, {d})")
            if self.delta is None:
                raise ValueError("delta (theta_plus - theta_minus per parameter) is required")
        if self.delta is not None:
            self.delta = np.asarray(self.delta, dtype=np.float64)
            if self.delta.shape != (k,) or np.any(self.delta == 0):
                raise ValueError(f"delta must have shape ({k},) and be non-zero")
        if self.dmu is not None:
            self.dmu = np.asarray(self.dmu, dtype=np.float64)
            if self.dmu.shape != (k, d):
                raise ValueError(f"dmu must have shape ({k}, {d})")
        if self.dmu is None and self.x_minus is None:
            raise ValueError("give dmu or x_minus/x_plus")

    @property
    def has_derivative_sims(self) -> bool:
        """True if seed-matched ``x_minus``/``x_plus`` simulations are present (the IMNN needs them)."""
        return self.x_minus is not None

    def mean_derivative(self) -> np.ndarray:
        """(K, D) d mu / d theta: ``dmu`` if given, else the mean finite difference."""
        if self.dmu is not None:
            return self.dmu
        return (self.x_plus - self.x_minus).mean(1) / self.delta[:, None]

    def subset(self, idx_fid, idx_deriv=None) -> "FiducialSet":
        """The fiducial sims ``idx_fid`` and the derivative pairs ``idx_deriv`` (all if None)."""
        sl = slice(None) if idx_deriv is None else idx_deriv
        return FiducialSet(self.x[idx_fid], self.param_names, self.theta_fid,
                           None if self.x_minus is None else self.x_minus[:, sl],
                           None if self.x_plus is None else self.x_plus[:, sl], self.delta, self.dmu)

    def split(self, groups_fid, groups_deriv, val_groups) -> tuple:
        """(train, validation) FiducialSets by group, e.g. N-body run: rows whose group is in ``val_groups`` go
        to validation, in both the fiducial sims (``groups_fid``, (n_fid,)) and the derivative pairs
        (``groups_deriv``, (n_d,)). Sims that share a group (initial conditions) never sit on both sides."""
        gf, gd = np.asarray(groups_fid), np.asarray(groups_deriv)
        if gf.shape != (len(self.x),) or (self.x_minus is not None and gd.shape != (self.x_minus.shape[1],)):
            raise ValueError("groups_fid must be (n_fid,) and groups_deriv (n_d,)")
        vf, vd = np.isin(gf, val_groups), np.isin(gd, val_groups)
        if not (vf.any() and vd.any() and (~vf).any() and (~vd).any()):
            raise ValueError("the split leaves an empty training or validation part")
        return self.subset(~vf, ~vd), self.subset(vf, vd)


class Splits(dict):
    """Named index arrays, e.g. ``{"train": ..., "val": ..., "test": ...}``.

    Any names are allowed (``comp_train``, ``calib``, ...). :meth:`check` verifies them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in list(self):
            self[k] = np.sort(np.asarray(self[k], dtype=np.int64))

    def check(self, n: int, groups: Optional[np.ndarray] = None, disjoint: bool = True,
              group_strict: bool = True) -> "Splits":
        """Raise if an index is out of range or repeated, if two splits overlap (``disjoint``),
        or if a group is spread over several splits (``group_strict``, needs ``groups``)."""
        for k, v in self.items():
            if v.size and (v.min() < 0 or v.max() >= n):
                raise ValueError(f"split {k!r} has indices outside [0, {n})")
            if len(np.unique(v)) != len(v):
                raise ValueError(f"split {k!r} repeats indices")
        names = list(self)
        if disjoint:
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    common = np.intersect1d(self[a], self[b])
                    if common.size:
                        raise ValueError(f"splits {a!r} and {b!r} share {common.size} indices")
        if groups is not None and group_strict:
            owner = {}
            for k in names:
                for g in np.unique(groups[self[k]]):
                    if owner.setdefault(g, k) != k:
                        raise ValueError(f"group {g} is in splits {owner[g]!r} and {k!r}")
        return self

    def hash(self) -> str:
        """16 hex digits identifying the split names and indices (stored in run records)."""
        h = hashlib.sha256()
        for k in sorted(self):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self[k]).tobytes())
        return h.hexdigest()[:16]

    def save(self, path) -> None:
        """Write the index arrays to an ``.npz`` file."""
        np.savez(Path(path), **self)

    @classmethod
    def load(cls, path) -> "Splits":
        """Read splits written by :meth:`save`."""
        with np.load(Path(path)) as f:
            return cls({k: f[k] for k in f.files})


def random_splits(n: int, fractions: Mapping[str, float], groups: Optional[np.ndarray] = None,
                  seed: int = 0) -> Splits:
    """Random splits by fraction of the groups (or of the simulations without ``groups``).

    Args:
        n: number of simulations.
        fractions: split name -> relative size. The fractions are normalised; the last split takes
            the rounding remainder.
        groups: optional (n,) ids; all simulations of one group go to the same split.
        seed: seed of the permutation.

    Returns:
        :class:`Splits` with one sorted index array per name.
    """
    if not fractions:
        raise ValueError("fractions is empty")
    rng = np.random.default_rng(seed)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    order = rng.permutation(len(uniq))
    w = np.array(list(fractions.values()), dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("fractions must be non-negative and not all zero")
    edges = np.round(np.cumsum(np.concatenate([[0], w / w.sum()])) * len(uniq)).astype(int)
    edges[-1] = len(uniq)
    out = {}
    for name, a, b in zip(fractions, edges[:-1], edges[1:]):
        chosen = np.zeros(len(uniq), bool)
        chosen[order[a:b]] = True
        out[name] = np.flatnonzero(chosen[inverse])
    return Splits(out)


DEFAULT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def resolve_splits(dataset: Dataset, splits: Optional[Mapping[str, np.ndarray]] = None,
                   fractions: Optional[Mapping[str, float]] = None, seed: int = 0,
                   group_strict: bool = True) -> Splits:
    """The given splits (checked), or random splits by group if none are given.

    Args:
        dataset: the simulations; its ``groups`` are used for the check and for random splits.
        splits: name -> indices. Checked for range, repeats, overlap and (``group_strict``) groups
            spread over several splits.
        fractions: sizes of the random splits if ``splits`` is None (default 0.8 / 0.1 / 0.1 for
            train / val / test).
        seed: seed of the random splits.
        group_strict: raise if given splits separate simulations of one group.
    """
    if splits is not None:
        return Splits(splits).check(len(dataset), dataset.groups, group_strict=group_strict)
    return random_splits(len(dataset), fractions or DEFAULT_FRACTIONS, dataset.groups, seed)
