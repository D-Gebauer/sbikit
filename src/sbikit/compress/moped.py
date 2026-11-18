"""MOPED (Heavens, Jimenez & Lahav 2000): b_a = C^-1 dmu/dtheta_a, Gram-Schmidt orthogonalised in
the given parameter order, so the coefficients y_a = b_a^T x are uncorrelated at the fiducial
point. One coefficient per parameter; lossless for a Gaussian likelihood with a
parameter-independent covariance and a mean that is linear in the parameters.

Normalisation: the vectors satisfy b_a^T C b_b = delta_ab for the covariance C they were built
with. For the fiducial simulations themselves the coefficients therefore have unit variance with
``hartlap=False``, and a variance equal to the Hartlap factor (n_fid - D - 2) / (n_fid - 1) with
the default ``hartlap=True``, which rescales all vectors by the square root of that factor. On
independent simulations the variance is only approximately that, unless n_fid >> D."""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import torch

from .._utils import as_numpy, did_you_mean, unknown
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

    Raises:
        ValueError: if ``inv_cov`` is not positive definite along a derivative, or a derivative
            or ``inv_cov`` is not finite.
    """
    dmu, inv_cov = np.asarray(dmu, dtype=np.float64), np.asarray(inv_cov, dtype=np.float64)
    names = list(names) if names is not None else [str(i) for i in range(len(dmu))]
    vecs, kept = [], []
    for k, d in enumerate(dmu):
        num = inv_cov @ d
        denom_sq = d @ num
        _check_information(denom_sq, d, inv_cov, names[k])
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
    return np.array(vecs).reshape(len(vecs), len(inv_cov)), kept


def _check_information(info: float, d: np.ndarray, inv_cov: np.ndarray, name: str) -> None:
    """Raise if ``info`` = d^T C^-1 d of a parameter is not finite, or negative beyond rounding."""
    if not np.isfinite(info):
        raise ValueError(f"MOPED: the derivative of parameter {name!r} or the inverse covariance is not finite")
    if info < -len(d) * np.finfo(float).eps * (np.abs(d) @ np.abs(inv_cov) @ np.abs(d)):
        raise ValueError(f"MOPED: the inverse covariance is not positive definite (dmu^T C^-1 dmu = {info:.3e} "
                         f"for parameter {name!r})")


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

    Raises:
        ValueError: as :func:`moped_vectors`, or if ``scale`` does not have one entry per parameter.
    """
    dmu, inv_cov = np.asarray(dmu, dtype=np.float64), np.asarray(inv_cov, dtype=np.float64)
    names = list(names) if names is not None else [str(i) for i in range(len(dmu))]
    scale = np.ones(len(dmu)) if scale is None else np.asarray(scale, dtype=np.float64)
    if scale.shape != (len(dmu),):
        raise ValueError(f"scale must have shape ({len(dmu)},), got {scale.shape}")
    num = dmu @ inv_cov                                   # (K, D) C^-1 dmu_k
    for k in range(len(dmu)):
        _check_information(dmu[k] @ num[k], dmu[k], inv_cov, names[k])
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
    return np.array(vecs).reshape(len(vecs), len(inv_cov)), order, np.array(scores)


@register
class MOPED(Compressor):
    """Linear MOPED compression from a :class:`FiducialSet` (covariance from ``fiducial.x``,
    derivatives from ``fiducial.dmu`` or the finite differences of the derivative sims).

    Args:
        fiducial: the fiducial and derivative simulations.
        params: the parameters to compress for, in Gram-Schmidt order (default: all of
            ``fiducial.param_names``; a single string is one parameter). This can include
            parameters the flow is not conditioned on.
        hartlap: apply the Hartlap factor to the inverse of the sample covariance (rescales all
            coefficients by a constant; matters only for chi^2-type goodness-of-fit tests).
        orthogonalize, tol: see :func:`moped_vectors`.
        pivot_scale: optional name -> scale (e.g. prior widths): order the Gram-Schmidt by
            :func:`moped_vectors_pivoted` instead of the order of ``params``. It needs an entry
            for every parameter in ``params``; entries for other parameters of the fiducial set
            are not used.
        cov: optional (D, D) covariance of the data vector to use instead of the sample covariance
            of ``fiducial.x``, e.g. an analytic or a shrinkage estimate (keyword only). The
            Hartlap factor is not applied to it.

    A precomputed compression matrix (e.g. the one used by a likelihood analysis of the same data)
    is wrapped with :meth:`from_vectors`, so that both analyses share the same summaries.

    Attributes:
        vectors: (K', D) compression vectors after :meth:`fit`.
        kept_params: the parameter of each vector, in order (degenerate ones are dropped).
        scores: pivot scores (only with ``pivot_scale``).
        inv_cov: (D, D) inverse covariance used by :meth:`fit` (None before, and after loading).

    Example:
        >>> fid = FiducialSet(x_fid, ["Om", "s8"], theta_fid, x_minus, x_plus, delta)
        >>> moped = MOPED(fid)
        >>> moped.fit()
        >>> s = moped(x)                                  # (N, 2)
        >>> s1 = moped(x[0])                              # (2,)
        >>> moped = MOPED(fid, cov=analytic_covariance)   # covariance not from the simulations
    """
    kind = "moped"

    def __init__(self, fiducial: Optional[FiducialSet] = None, params: Optional[Sequence[str]] = None,
                 hartlap: bool = True, orthogonalize: bool = True, tol: float = 1e-10, pivot_scale=None, *,
                 cov=None):
        self.fiducial, self.hartlap, self.orthogonalize, self.tol = fiducial, hartlap, orthogonalize, tol
        if isinstance(params, str):
            params = [params]
        self.params = list(params) if params is not None else (list(fiducial.param_names) if fiducial else None)
        self.pivot_scale = None if pivot_scale is None else {k: float(v) for k, v in dict(pivot_scale).items()}
        self.cov = None if cov is None else as_numpy(cov)
        self.vectors: Optional[np.ndarray] = None
        self.kept_params: list = []
        self.scores: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        if fiducial is not None:
            self._check_names()

    def _check_names(self) -> None:
        """Raise ``ValueError`` for parameters or ``pivot_scale`` keys the fiducial set does not have."""
        known = list(self.fiducial.param_names)
        if not self.params:
            raise ValueError("MOPED: params is empty")
        for p in self.params:
            if p not in known:
                raise unknown("parameter", p, known)
        if self.pivot_scale is not None:
            for p in self.pivot_scale:
                if p not in known:
                    raise ValueError(f"pivot_scale: unknown parameter {p!r}{did_you_mean(p, known)}; valid: {known}")
            missing = [p for p in self.params if p not in self.pivot_scale]
            if missing:
                raise ValueError(f"pivot_scale has no entry for {missing}; it needs one for every parameter in params")

    @classmethod
    def from_vectors(cls, vectors, params: Sequence[str]) -> "MOPED":
        """A fitted MOPED from given vectors (K, D), row k for ``params[k]``.

        Raises:
            ValueError: if ``vectors`` is not (K, D) with one row per parameter.
        """
        m = cls(None, params, hartlap=False)
        m.vectors = as_numpy(vectors).copy()
        if m.vectors.ndim != 2 or len(m.vectors) != len(m.params):
            raise ValueError(f"vectors {m.vectors.shape} must be ({len(m.params)}, D)")
        m.kept_params = list(m.params)
        return m

    @property
    def n_out(self):
        """Number of summaries (the kept parameters once fitted)."""
        return len(self.kept_params) if self.vectors is not None else len(self.params)

    def _inverse_covariance(self) -> np.ndarray:
        """The (D, D) inverse covariance: of ``cov`` if given, else of the sample covariance of the
        fiducial simulations (with the Hartlap factor if ``hartlap``)."""
        n, d = self.fiducial.x.shape
        if self.cov is not None:
            cov, source = self.cov, "cov"
            if cov.shape != (d, d):
                raise ValueError(f"cov must have shape ({d}, {d}) like the data vector, got {cov.shape}")
            if not np.isfinite(cov).all():
                raise ValueError("cov contains NaN or inf")
            if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-8 * np.abs(cov).max()):
                raise ValueError("cov is not symmetric")
        else:
            need = d + 2 if self.hartlap else d
            if n <= need:
                raise ValueError(
                    f"MOPED: {n} fiducial simulations are too few for a data vector of length D = {d}: the sample "
                    f"covariance needs more than {need} ({'D + 2 with' if self.hartlap else 'D without'} the Hartlap "
                    "factor). Use more fiducial simulations, a shorter data vector, or pass a covariance with cov=")
            factor = hartlap_factor(n, d)
            if factor < 0.5:
                warnings.warn(f"MOPED: only {n} fiducial simulations for a data vector of length {d} (Hartlap factor "
                              f"{factor:.2f}): the inverse sample covariance is noisy and the compression loses "
                              "information. Consider more simulations, a shorter data vector or cov=", RuntimeWarning)
            cov, source = np.atleast_2d(np.cov(self.fiducial.x.T)), "the sample covariance of fiducial.x"
        var = cov.diagonal()
        if np.any(var <= 0):
            raise ValueError(f"MOPED: {source} has zero or negative variance in column(s) "
                             f"{np.flatnonzero(var <= 0)[:10].tolist()} of the data vector; remove constant entries")
        sd = np.sqrt(var)
        eig = np.linalg.eigvalsh(cov / np.outer(sd, sd))          # of the correlation matrix: no effect of units
        if not np.isfinite(eig).all() or eig[0] <= d * np.finfo(float).eps * eig[-1]:
            raise ValueError(f"MOPED: {source} is singular or not positive definite (smallest / largest eigenvalue "
                             f"of the correlation matrix: {eig[0]:.3e} / {eig[-1]:.3e}); remove linearly dependent "
                             "entries of the data vector")
        inv = np.linalg.inv(cov)
        return hartlap_factor(n, d) * inv if self.hartlap and self.cov is None else inv

    def fit(self, x_trn=None, t_trn=None, x_val=None, t_val=None) -> float:
        """Compute the vectors from the fiducial set (the arguments are ignored).

        Returns:
            nan: MOPED has no validation loss.

        Raises:
            ValueError: without a fiducial set; for unknown parameter or ``pivot_scale`` names; for
                too few fiducial simulations for the sample covariance (``n_fid <= D + 2`` with the
                Hartlap factor, ``n_fid <= D`` without); for a singular, non-finite or wrongly
                shaped covariance; or if every parameter is degenerate.
        """
        fid = self.fiducial
        if fid is None:
            raise ValueError("MOPED needs a FiducialSet")
        self._check_names()
        if self.pivot_scale is not None and not self.orthogonalize:
            raise ValueError("pivoting needs orthogonalize=True")
        idx = [fid.param_names.index(p) for p in self.params]
        inv, dmu, scores = self._inverse_covariance(), fid.mean_derivative()[idx], None
        if self.pivot_scale is not None:
            scale = [self.pivot_scale[p] for p in self.params]
            vecs, kept, scores = moped_vectors_pivoted(dmu, inv, scale, self.tol, self.params)
        else:
            vecs, kept = moped_vectors(dmu, inv, self.orthogonalize, self.tol, self.params)
        if not kept:
            raise ValueError("MOPED: every parameter is degenerate (no derivative information): check dmu or the "
                             "derivative simulations")
        self.inv_cov, self.vectors, self.scores = inv, vecs, scores
        self.kept_params = [self.params[k] for k in kept]
        return float("nan")

    def __call__(self, x):
        """Summaries (N, n_out), float32, of the data vectors ``x`` (N, D); a single data vector
        (D,) gives (n_out,).

        Raises:
            NotFittedError: before :meth:`fit`.
            ValueError: if ``x`` is not 1D or 2D or has the wrong number of columns.
        """
        if self.vectors is None:
            raise self._not_fitted()
        x, single = self._batch(x, self.vectors.shape[1])
        y = (x @ self.vectors.T).astype(np.float32)
        return y[0] if single else y

    def state(self):
        """Plain serialisable state (the vectors and settings, not the fiducial set).

        Raises:
            NotFittedError: before :meth:`fit`.
        """
        if self.vectors is None:
            raise self._not_fitted()
        return {"kind": self.kind, "vectors": torch.tensor(self.vectors), "params": list(self.params),
                "kept_params": list(self.kept_params), "hartlap": bool(self.hartlap),
                "orthogonalize": bool(self.orthogonalize), "tol": float(self.tol),
                "pivot_scale": None if self.pivot_scale is None else dict(self.pivot_scale),
                "scores": None if self.scores is None else torch.tensor(np.asarray(self.scores, dtype=np.float64))}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted MOPED from :meth:`state` (``device`` is not used: MOPED has no network)."""
        m = cls(None, state["params"], state["hartlap"], state["orthogonalize"], state["tol"],
                state.get("pivot_scale"))
        m.vectors = as_numpy(state["vectors"]).copy()
        m.kept_params = list(state["kept_params"])
        m.scores = None if state.get("scores") is None else as_numpy(state["scores"]).copy()
        return m

    def __repr__(self):
        if self.vectors is None:
            return f"MOPED(params={self.params}, unfitted)"
        return f"MOPED(params={self.kept_params}, D={self.vectors.shape[1]})"
