"""Containers for simulations and the fiducial/derivative set, and train/val/test splits.

sbikit never decides how an analysis splits its simulations: pass the splits explicitly (e.g. by
N-body realisation), or let :func:`resolve_splits` draw random ones. Random splits keep every
``group`` (e.g. one N-body run, or one cosmology) in a single split, so that correlated
simulations cannot leak between training and testing.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from ._utils import as_numpy, check_finite, unknown

__all__ = ["Dataset", "FiducialSet", "Splits", "random_splits", "resolve_splits"]


def _array(a) -> np.ndarray:
    """A numpy array of any dtype from an array-like or a tensor (on any device)."""
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def _names(names) -> list:
    """Parameter names as a list; a single string is one name."""
    return [names] if isinstance(names, str) else list(names)


def _as_index(v, what: str, n: Optional[int] = None) -> np.ndarray:
    """Indices as a 1D int64 array, in the given order.

    Args:
        v: integer indices or a boolean mask (array-like or tensor).
        what: what is indexed, for the error messages (e.g. ``"split 'train'"``).
        n: length of the indexed axis, if known (checked for masks and for the index range).

    Raises:
        TypeError: for indices that are neither integer nor boolean.
        ValueError: for a wrong number of dimensions, a mask of the wrong length or indices
            outside [0, n).
    """
    v = _array(v)
    if v.ndim != 1:
        raise ValueError(f"{what} must be a 1D array of indices or a boolean mask, got shape {v.shape}")
    if v.dtype == bool:
        if n is not None and len(v) != n:
            raise ValueError(f"{what} is a boolean mask of length {len(v)}, expected {n}")
        return np.flatnonzero(v)
    if v.size == 0:
        return np.zeros(0, dtype=np.int64)
    if v.dtype.kind not in "iu":
        raise TypeError(f"{what} must hold integer indices or be a boolean mask, got dtype {v.dtype}")
    v = v.astype(np.int64)
    if n is not None and (v.min() < 0 or v.max() >= n):
        raise ValueError(f"{what} has indices outside [0, {n})")
    return v


@dataclass
class Dataset:
    """Simulations for training and testing.

    Arrays may be given as array-likes or tensors. ``theta`` is stored as float64 and ``x`` as
    float32 (the precision the networks work in); data vectors that need more than seven
    significant digits (a mean many orders of magnitude above the scatter) should be centred or
    rescaled first.

    Args:
        theta: (N, P) parameters the flow is conditioned on (the labels).
        x: (N, D) data vectors (or already compressed summaries).
        param_names: the P names of the columns of ``theta`` (a single string is one name).
        groups: optional (N,) integer ids; simulations that share an id (e.g. the footprints and
            noise realisations of one N-body run) are never split apart by random splits.
        blocks: optional name -> columns of ``x`` (integer indices in the wanted order, or a
            boolean mask over the columns), for hierarchical compression (e.g. one block per
            field, tomographic bin or statistic).
        meta: optional name -> (N, ...) per-simulation arrays that are not labels (implicitly
            marginalised nuisances such as m and Delta z draws, n(z) realisation ids, the
            N-body run); carried through :meth:`subset` for diagnostics and re-labelling.
        allow_nonfinite: accept NaN or inf in ``theta`` and ``x`` (keyword only). By default they
            raise, because one such value turns a training loss into NaN.

    Raises:
        ValueError: for wrong dimensions, lengths that do not match, non-finite values (unless
            ``allow_nonfinite``), or blocks that are empty or index outside the columns of ``x``.
        TypeError: for blocks that are neither integer indices nor boolean masks.
    """

    theta: np.ndarray
    x: np.ndarray
    param_names: Sequence[str]
    groups: Optional[np.ndarray] = None
    blocks: Optional[Mapping[str, np.ndarray]] = None
    meta: Optional[Mapping[str, np.ndarray]] = None
    allow_nonfinite: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        self.theta = as_numpy(self.theta, np.float64)
        self.x = as_numpy(self.x, np.float32)
        self.param_names = _names(self.param_names)
        if self.theta.ndim != 2 or self.x.ndim != 2 or len(self.theta) != len(self.x):
            raise ValueError(f"theta {self.theta.shape} and x {self.x.shape} must be (N, P) and (N, D)")
        if self.theta.shape[1] != len(self.param_names):
            raise ValueError(f"{self.theta.shape[1]} parameter columns but {len(self.param_names)} names")
        if not self.allow_nonfinite:
            check_finite(self.theta, "theta")
            check_finite(self.x, "x")
        if self.groups is not None:
            self.groups = _array(self.groups)
            if self.groups.shape != (len(self.x),):
                raise ValueError(f"groups must have shape ({len(self.x)},), got {self.groups.shape}")
        if self.blocks is not None:
            self.blocks = {k: _as_index(v, f"block {k!r}", self.x.shape[1]) for k, v in self.blocks.items()}
            for k, v in self.blocks.items():
                if v.size == 0:
                    raise ValueError(f"block {k!r} selects no column of x")
        if self.meta is not None:
            self.meta = {k: _array(v) for k, v in self.meta.items()}
            for k, v in self.meta.items():
                if v.shape[:1] != (len(self.x),):
                    raise ValueError(f"meta {k!r} has shape {v.shape}, x has {len(self.x)} rows")

    def __len__(self):
        """Number of simulations."""
        return len(self.x)

    def index(self, names: Sequence[str]) -> list[int]:
        """Column indices of the given parameter names (a single string is one name).

        Raises:
            ValueError: for a name that is not in ``param_names``.
        """
        names = _names(names)
        for n in names:
            if n not in self.param_names:
                raise unknown("parameter", n, self.param_names)
        return [self.param_names.index(n) for n in names]

    def subset(self, idx) -> "Dataset":
        """The simulations ``idx`` (indices or a boolean mask) as a new Dataset; ``groups`` and
        ``meta`` are subset too, ``blocks`` are kept."""
        idx = _array(idx)
        return Dataset(self.theta[idx], self.x[idx], self.param_names,
                       None if self.groups is None else self.groups[idx], self.blocks,
                       None if self.meta is None else {k: v[idx] for k, v in self.meta.items()},
                       allow_nonfinite=self.allow_nonfinite)


@dataclass
class FiducialSet:
    """Simulations at a fiducial point, and seed-matched derivative simulations.

    Used by MOPED (covariance + mean derivatives) and by the IMNN (trained on these only).

    Arrays may be given as array-likes or tensors; all are stored as float64.

    Args:
        x: (n_fid, D) independent simulations at the fiducial parameters (covariance).
        param_names: the K parameters with derivatives (the compression targets; a single string
            is one name).
        theta_fid: (K,) fiducial values of those parameters.
        x_minus, x_plus: optional (K, n_d, D) simulations at theta_fid -/+ step for each parameter;
            row d of x_minus[k] and x_plus[k] must share the random seed (initial conditions, noise).
        delta: (K,) theta_plus - theta_minus for each parameter.
        dmu: optional (K, D) mean derivatives d mu / d theta, if they are known directly (e.g. from
            noiseless means). Overrides the finite differences of x_minus/x_plus for MOPED.

    Raises:
        ValueError: for wrong dimensions or shapes, missing derivatives, or NaN/inf in any array
            (they would turn every MOPED vector into NaN and stop an IMNN from training).
    """

    x: np.ndarray
    param_names: Sequence[str]
    theta_fid: np.ndarray
    x_minus: Optional[np.ndarray] = None
    x_plus: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None
    dmu: Optional[np.ndarray] = None

    def __post_init__(self):
        self.x = as_numpy(self.x)
        self.param_names = _names(self.param_names)
        self.theta_fid = np.atleast_1d(as_numpy(self.theta_fid))
        if self.x.ndim != 2 or 0 in self.x.shape:
            raise ValueError(f"x must be (n_fid, D) fiducial simulations, got shape {self.x.shape}")
        k, d = len(self.param_names), self.x.shape[1]
        if self.theta_fid.shape != (k,):
            raise ValueError(f"theta_fid must have shape ({k},), got {self.theta_fid.shape}")
        if (self.x_minus is None) != (self.x_plus is None):
            raise ValueError("give both x_minus and x_plus, or neither")
        if self.x_minus is not None:
            self.x_minus, self.x_plus = as_numpy(self.x_minus), as_numpy(self.x_plus)
            sm, sp = self.x_minus.shape, self.x_plus.shape
            if sm != sp or len(sm) != 3 or sm[0] != k or sm[1] == 0 or sm[2] != d:
                raise ValueError(f"x_minus/x_plus must both be ({k}, n_d, {d}), got {sm} and {sp}")
            if self.delta is None:
                raise ValueError("delta (theta_plus - theta_minus per parameter) is required")
        if self.delta is not None:
            self.delta = np.atleast_1d(as_numpy(self.delta))
            if self.delta.shape != (k,) or np.any(self.delta == 0):
                raise ValueError(f"delta must have shape ({k},) and be non-zero, got shape {self.delta.shape}")
        if self.dmu is not None:
            self.dmu = as_numpy(self.dmu)
            if self.dmu.shape != (k, d):
                raise ValueError(f"dmu must have shape ({k}, {d}), got {self.dmu.shape}")
        if self.dmu is None and self.x_minus is None:
            raise ValueError("give dmu or x_minus/x_plus")
        for name in ("x", "theta_fid", "x_minus", "x_plus", "delta", "dmu"):
            a = getattr(self, name)
            if a is not None:
                check_finite(a.reshape(-1, d) if a.ndim == 3 else a, name)

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

    def split(self, groups_fid, groups_deriv=None, val_groups=None) -> tuple:
        """(train, validation) FiducialSets by group, e.g. N-body run.

        Rows whose group is in ``val_groups`` go to validation, in both the fiducial sims and the
        derivative pairs, so that sims sharing a group (initial conditions) never sit on both
        sides. A set without derivative simulations (``dmu`` only) is split by ``groups_fid``
        alone, and both parts keep the same ``dmu``.

        Args:
            groups_fid: (n_fid,) group of each fiducial simulation.
            groups_deriv: (n_d,) group of each derivative pair; needed if the set has derivative
                simulations.
            val_groups: the groups that form the validation set.

        Returns:
            (train, validation) FiducialSets.

        Raises:
            ValueError: for group arrays of the wrong shape, ``val_groups`` that occur nowhere, or
                a split that leaves a part empty.

        Example:
            >>> trn, val = fid.split(run_id_fid, run_id_deriv, val_groups=[0, 1])
            >>> trn, val = fid_dmu_only.split(run_id_fid, val_groups=[0, 1])
        """
        if val_groups is None:
            raise ValueError("val_groups is required: the groups that form the validation set")
        val_groups = np.atleast_1d(_array(val_groups))
        gf = _array(groups_fid)
        if gf.shape != (len(self.x),):
            raise ValueError(f"groups_fid must have shape ({len(self.x)},), got {gf.shape}")
        vf, vd, seen = np.isin(gf, val_groups), None, gf
        if self.x_minus is not None:
            if groups_deriv is None:
                raise ValueError("groups_deriv is required for a set with derivative simulations")
            gd = _array(groups_deriv)
            if gd.shape != (self.x_minus.shape[1],):
                raise ValueError(f"groups_deriv must have shape ({self.x_minus.shape[1]},), got {gd.shape}")
            vd, seen = np.isin(gd, val_groups), np.concatenate([gf, gd])
        elif groups_deriv is not None:
            warnings.warn("FiducialSet.split: groups_deriv is ignored, the set has no derivative simulations")
        missing = val_groups[~np.isin(val_groups, seen)]
        if missing.size:
            raise ValueError(f"val_groups {missing.tolist()} occur in none of the group arrays")
        if not (vf.any() and (~vf).any()) or (vd is not None and not (vd.any() and (~vd).any())):
            raise ValueError("the split leaves an empty training or validation part")
        return (self.subset(~vf, None if vd is None else ~vd), self.subset(vf, vd))


_KEY = "split:"     # prefix of the array names in a saved file, so that no split name is taken for an np.savez argument


def _npz_path(path) -> Path:
    """``path`` with the ``.npz`` suffix that ``np.savez`` appends."""
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


class Splits(dict):
    """Named index arrays, e.g. ``{"train": ..., "val": ..., "test": ...}``.

    Any names are allowed (``comp_train``, ``calib``, ...). :meth:`check` verifies them.

    Each split is given as integer indices or as a boolean mask over the simulations (array-like
    or tensor) and is stored as a sorted int64 array: the order of the given indices is not kept,
    so index arrays with the split (``x[splits["train"]]``) rather than with the original list.
    Items assigned later (``splits["calib"] = [...]``, ``update``, ``setdefault``) are converted
    the same way.

    Raises:
        TypeError: for indices that are neither integer nor boolean (e.g. floats).
        ValueError: for index arrays that are not 1D.

    Example:
        >>> splits = Splits({"train": [0, 1, 2, 3], "val": [4, 5]})
        >>> splits["test"] = np.arange(6, 8)
        >>> splits.check(8).save("splits.npz")
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, name, idx):
        super().__setitem__(name, np.sort(_as_index(idx, f"split {name!r}")))

    def update(self, *args, **kwargs):
        """Add or replace splits (same arguments as ``dict.update``)."""
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, name, default=()):
        """The split ``name``; it is set to ``default`` first if missing."""
        if name not in self:
            self[name] = default
        return self[name]

    def copy(self) -> "Splits":
        """A copy (with copies of the index arrays)."""
        return Splits({k: v.copy() for k, v in self.items()})

    def __or__(self, other):
        return Splits({**self, **dict(other)})

    def __ror__(self, other):
        return Splits({**dict(other), **self})

    def __ior__(self, other):
        self.update(other)
        return self

    def __repr__(self):
        return "Splits(" + ", ".join(f"{k}: {len(v)}" for k, v in self.items()) + ")"

    def check(self, n: int, groups: Optional[np.ndarray] = None, disjoint: bool = True,
              group_strict: bool = True) -> "Splits":
        """Raise if an index is out of range or repeated, if two splits overlap (``disjoint``),
        or if a group is spread over several splits (``group_strict``, needs ``groups``).

        Args:
            n: number of simulations.
            groups: optional (n,) group ids (array-like).
            disjoint: require that no index is in two splits.
            group_strict: require that no group is in two splits.

        Returns:
            The splits themselves.

        Raises:
            ValueError: if a check fails, or ``groups`` does not have shape (n,).
        """
        if groups is not None:
            groups = _array(groups)
            if groups.shape != (n,):
                raise ValueError(f"groups must have shape ({n},), got {groups.shape}")
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
        """Write the index arrays to an ``.npz`` file (the suffix is appended if ``path`` lacks it).

        Raises:
            TypeError: if a split name is not a string.
        """
        for k in self:
            if not isinstance(k, str):
                raise TypeError(f"split names must be strings to be saved, got {k!r}")
        np.savez(_npz_path(path), **{_KEY + k: v for k, v in self.items()})

    @classmethod
    def load(cls, path) -> "Splits":
        """Read splits written by :meth:`save` (``path`` with or without the ``.npz`` suffix)."""
        path = Path(path)
        with np.load(path if path.is_file() else _npz_path(path)) as f:
            prefixed = all(k.startswith(_KEY) for k in f.files)     # a file with bare split names is read as it is
            return cls({(k[len(_KEY):] if prefixed else k): f[k] for k in f.files})


def random_splits(n: int, fractions: Mapping[str, float], groups: Optional[np.ndarray] = None,
                  seed: int = 0) -> Splits:
    """Random splits by fraction of the groups (or of the simulations without ``groups``).

    Args:
        n: number of simulations.
        fractions: split name -> relative size. The fractions are normalised; the split sizes are
            the rounded cumulative fractions of the number of groups, and the last split takes
            the rounding remainder.
        groups: optional (n,) ids; all simulations of one group go to the same split.
        seed: seed of the permutation.

    Returns:
        :class:`Splits` with one sorted index array per name.

    Raises:
        ValueError: for invalid fractions, ``groups`` that do not have shape (n,), or too few
            groups (simulations) to give every split with a positive fraction at least one.
    """
    if not fractions:
        raise ValueError("fractions is empty")
    rng = np.random.default_rng(seed)
    what = "simulations" if groups is None else "groups"
    groups = np.arange(n) if groups is None else _array(groups)
    if groups.shape != (n,):
        raise ValueError(f"groups must have shape (n,) = ({n},), got {groups.shape}")
    uniq, inverse = np.unique(groups, return_inverse=True)
    order = rng.permutation(len(uniq))
    w = np.array(list(fractions.values()), dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"fractions must be finite, non-negative and not all zero, got {dict(fractions)}")
    edges = np.round(np.cumsum(np.concatenate([[0], w / w.sum()])) * len(uniq)).astype(int)
    edges[-1] = len(uniq)
    empty = [name for name, f, size in zip(fractions, w, np.diff(edges)) if f > 0 and size == 0]
    if empty:
        raise ValueError(f"random_splits: {len(uniq)} {what} are too few for the fractions {dict(fractions)}: "
                         f"{empty} would be empty. Use more {what}, larger fractions for these splits, or pass "
                         "your own splits")
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
        splits: name -> indices (or boolean masks). Checked for range, repeats, overlap and
            (``group_strict``) groups spread over several splits; ``fractions`` and ``seed`` are
            not used then.
        fractions: sizes of the random splits if ``splits`` is None (default 0.8 / 0.1 / 0.1 for
            train / val / test).
        seed: seed of the random splits.
        group_strict: raise if given splits separate simulations of one group.

    Returns:
        :class:`Splits` with one sorted index array per name (the order of given indices is not
        kept).

    Raises:
        ValueError: if the given splits fail a check, or the random splits would leave a split
            empty (see :func:`random_splits`).
    """
    if splits is not None:
        splits = {k: _as_index(v, f"split {k!r}", len(dataset)) for k, v in dict(splits).items()}
        return Splits(splits).check(len(dataset), dataset.groups, group_strict=group_strict)
    return random_splits(len(dataset), fractions or DEFAULT_FRACTIONS, dataset.groups, seed)
