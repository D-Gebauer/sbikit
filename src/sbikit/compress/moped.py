"""MOPED (Heavens, Jimenez & Lahav 2000): b_a = C^-1 dmu/dtheta_a, Gram-Schmidt orthogonalised in
the given parameter order, so the coefficients y_a = b_a^T x are uncorrelated with unit variance at
the fiducial point. One coefficient per parameter; lossless for a Gaussian likelihood with a
parameter-independent covariance and a mean that is linear in the parameters."""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import torch

from ..data import FiducialSet
from .base import Compressor, register

__all__ = ["MOPED", "moped_vectors", "moped_vectors_pivoted", "hartlap_factor"]


def hartlap_factor(n_sims: int, n_dim: int) -> float:
    """(n - d - 2) / (n - 1): debiases the inverse of a covariance estimated from n simulations."""
    return (n_sims - n_dim - 2) / (n_sims - 1)


def moped_vectors(dmu: np.ndarray, inv_cov: np.ndarray, orthogonalize: bool = True,
                  tol: float = 1e-10, names: Optional[Sequence[str]] = None):
    """Compression vectors for the rows of ``dmu``.

    A parameter whose derivative lies in the span of the earlier ones (e.g. a nuisance parameter
    of a part of the data vector that was cut) is dropped with a warning instead of producing NaNs.

    Args:
        dmu: (K, D) mean derivatives, one row per parameter.
        inv_cov: (D, D) inverse covariance.
        orthogonalize: Gram-Schmidt the vectors in the given order (False: plain C^-1 dmu,
            normalised).
        tol: a parameter is degenerate if its residual information is below ``tol`` times its
            total information.
        names: parameter names for the warning.

    Returns:
        (vectors (K', D), kept indices).
    """
    names = list(names) if names is not None else [str(i) for i in range(len(dmu))]
    vecs, kept = [], []
    for k, d in enumerate(np.asarray(dmu, dtype=np.float64)):
        num = inv_cov @ d
        denom_sq = d @ num
        if orthogonalize and vecs:
            proj = np.array([d @ b for b in vecs])
            residual_sq = denom_sq - (proj ** 2).sum()
            num = num - (proj[:, None] * np.array(vecs)).sum(0)
        else:
            residual_sq = denom_sq
        if residual_sq <= tol * max(denom_sq, np.finfo(float).tiny):
            warnings.warn(f"MOPED: dropping degenerate parameter {names[k]!r} (residual {residual_sq:.3e})")
            continue
        vecs.append(num / np.sqrt(residual_sq))
        kept.append(k)
    return np.array(vecs), kept


def moped_vectors_pivoted(dmu: np.ndarray, inv_cov: np.ndarray, scale=None, tol: float = 1e-10,
                          names: Optional[Sequence[str]] = None):
    """Gram-Schmidt MOPED with pivoting: each step takes the remaining parameter with the largest
    sqrt(residual Fisher information) x ``scale`` (e.g. the prior width, so the score is the
    expected number of prior widths constrained), i.e. the best-measured direction first.

    The span of the vectors (and so the information and any likelihood in the full set of
    coefficients) does not depend on the order; the order matters only when coefficients are
    dropped or compressed further.

    Args:
        dmu, inv_cov, tol, names: as in :func:`moped_vectors`.
        scale: (K,) weights of the pivot score (default 1).

    Returns:
        (vectors (K', D), order (indices into ``dmu``), scores).
    """
    dmu = np.asarray(dmu, dtype=np.float64)
    names = list(names) if names is not None else [str(i) for i in range(len(dmu))]
    scale = np.ones(len(dmu)) if scale is None else np.asarray(scale, dtype=np.float64)
    num = dmu @ inv_cov                                   # (K, D) C^-1 dmu_k
    left, vecs, order, scores = list(range(len(dmu))), [], [], []
    while left:
        res2 = np.array([dmu[k] @ num[k] - sum((dmu[k] @ b) ** 2 for b in vecs) for k in left])
        full = np.array([dmu[k] @ num[k] for k in left])
        ok = res2 > tol * np.maximum(full, np.finfo(float).tiny)
        for k in [k for k, good in zip(left, ok) if not good]:
            warnings.warn(f"MOPED: dropping degenerate parameter {names[k]!r}")
        left, res2 = [k for k, good in zip(left, ok) if good], res2[ok]
        if not left:
            break
        j = int(np.argmax(np.sqrt(res2) * scale[left]))
        k = left.pop(j)
        v = num[k] - sum((dmu[k] @ b) * b for b in vecs)
        vecs.append(v / np.sqrt(res2[j]))
        order.append(k)
        scores.append(float(np.sqrt(res2[j]) * scale[k]))
    return np.array(vecs), order, np.array(scores)


@register
class MOPED(Compressor):
    """Linear MOPED compression from a :class:`FiducialSet` (covariance from ``fiducial.x``,
    derivatives from ``fiducial.dmu`` or the finite differences of the derivative sims).

    Args:
        fiducial: the fiducial and derivative simulations.
        params: the parameters to compress for, in Gram-Schmidt order (default: all of
            ``fiducial.param_names``). This can include parameters the flow is not conditioned on.
        hartlap: apply the Hartlap factor to the inverse covariance (rescales all coefficients by
            a constant; matters only for chi^2-type goodness-of-fit tests).
        pivot_scale: optional name -> scale (e.g. prior widths): order the Gram-Schmidt by
            :func:`moped_vectors_pivoted` instead of the order of ``params``.

        orthogonalize, tol: see :func:`moped_vectors`.

    A precomputed compression matrix (e.g. the one used by a likelihood analysis of the same data)
    is wrapped with :meth:`from_vectors`, so that both analyses share the same summaries.

    Attributes:
        vectors: (K', D) compression vectors after :meth:`fit`.
        kept_params: the parameter of each vector, in order (degenerate ones are dropped).
        scores: pivot scores (only with ``pivot_scale``).

    Example:
        >>> fid = FiducialSet(x_fid, ["Om", "s8"], theta_fid, x_minus, x_plus, delta)
        >>> moped = MOPED(fid)
        >>> moped.fit()
        >>> s = moped(x)                                  # (N, 2)
    """
    kind = "moped"

    def __init__(self, fiducial: Optional[FiducialSet] = None, params: Optional[Sequence[str]] = None,
                 hartlap: bool = True, orthogonalize: bool = True, tol: float = 1e-10, pivot_scale=None):
        self.fiducial, self.hartlap, self.orthogonalize, self.tol = fiducial, hartlap, orthogonalize, tol
        self.params = list(params) if params is not None else (list(fiducial.param_names) if fiducial else None)
        self.pivot_scale = None if pivot_scale is None else {k: float(v) for k, v in dict(pivot_scale).items()}
        self.vectors: Optional[np.ndarray] = None
        self.kept_params: list = []
        self.scores: Optional[np.ndarray] = None

    @classmethod
    def from_vectors(cls, vectors, params: Sequence[str]) -> "MOPED":
        """A fitted MOPED from given vectors (K, D), row k for ``params[k]``."""
        m = cls(None, params, hartlap=False)
        m.vectors = np.asarray(vectors, dtype=np.float64)
        if m.vectors.ndim != 2 or len(m.vectors) != len(m.params):
            raise ValueError(f"vectors {m.vectors.shape} must be ({len(m.params)}, D)")
        m.kept_params = list(m.params)
        return m

    @property
    def n_out(self):
        """Number of summaries (the kept parameters once fitted)."""
        return len(self.kept_params) if self.vectors is not None else len(self.params)

    def fit(self, x_trn=None, t_trn=None, x_val=None, t_val=None) -> float:
        """Compute the vectors from the fiducial set (the arguments are ignored). Returns nan:
        MOPED has no validation loss."""
        fid = self.fiducial
        if fid is None:
            raise ValueError("MOPED needs a FiducialSet")
        idx = [fid.param_names.index(p) for p in self.params]
        cov = np.cov(fid.x.T)
        inv = np.linalg.inv(cov)
        if self.hartlap:
            inv = hartlap_factor(*fid.x.shape) * inv
        self.inv_cov = inv
        if self.pivot_scale is not None:
            if not self.orthogonalize:
                raise ValueError("pivoting needs orthogonalize=True")
            scale = [self.pivot_scale[p] for p in self.params]
            vecs, kept, self.scores = moped_vectors_pivoted(fid.mean_derivative()[idx], inv, scale, self.tol,
                                                            self.params)
        else:
            vecs, kept = moped_vectors(fid.mean_derivative()[idx], inv, self.orthogonalize, self.tol, self.params)
        self.vectors, self.kept_params = vecs, [self.params[k] for k in kept]
        return float("nan")

    def __call__(self, x):
        if self.vectors is None:
            raise RuntimeError("call fit() first")
        return (np.asarray(x, dtype=np.float64) @ self.vectors.T).astype(np.float32)

    def state(self):
        """Plain serialisable state (the vectors and settings, not the fiducial set)."""
        return {"kind": self.kind, "vectors": torch.as_tensor(self.vectors), "params": self.params,
                "kept_params": self.kept_params, "hartlap": self.hartlap,
                "orthogonalize": self.orthogonalize, "tol": self.tol, "pivot_scale": self.pivot_scale}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted MOPED from :meth:`state`."""
        m = cls(None, state["params"], state["hartlap"], state["orthogonalize"], state["tol"],
                state.get("pivot_scale"))
        m.vectors = np.asarray(state["vectors"], dtype=np.float64)
        m.kept_params = list(state["kept_params"])
        return m
