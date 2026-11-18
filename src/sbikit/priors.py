"""Priors in raw parameter space (torch, batched, GPU-capable).

A :class:`Prior` is a box times optional density terms on subsets of the parameters (Gaussian,
KDE); parameters not covered by a term are uniform over the box. Optional support constraints
(convex polygons, bounds along a curve, or any callable returning a boolean mask) set the density
to -inf outside. ``log_prob`` is normalised up to the truncation by the box and the constraints,
which only adds a constant per prior.

Terms and constraints address parameters through ``dims``: column indices, or parameter names that
the :class:`Prior` resolves against its ``names``.

**Density terms** are objects with

- ``kind``: a unique string (class attribute);
- ``dims``: the parameters the term acts on;
- ``log_prob(x)``: normalised log density (n,) of the rows of ``x`` (n, len(dims));
- ``sample(n, generator)``: ``n`` draws (n, len(dims)) on the term's device;
- ``to(device)``: a copy on ``device``, of the same class;
- ``spec()``: ``{"kind": kind, **constructor_arguments}`` with plain values and tensors.

**Constraints** are callables ``theta (..., P) -> bool (...)``, True inside the support. They
receive 2D input (n, P) in :meth:`Prior.sample` and 3D input (targets, walkers, P) in the sampler,
so they must index the last axis only (``theta[..., i]``, never ``theta[:, i]``). A plain function
is enough for sampling; to save a prior, a constraint is an object that also has ``kind``, ``dims``,
``to(device)`` and ``spec()`` as above.

A prior with custom terms or constraints can be saved and loaded once their classes are registered
with :func:`register_term` / :func:`register_constraint`; the constructor is then called with the
entries of ``spec()`` (and ``device``, if it takes one). A subclass of a registered class (also of a
built-in one) needs its own ``kind`` and its own registration: :meth:`Prior.spec` refuses to store
it under the inherited ``kind``, because it would be loaded as its base class.

Example:
    >>> @register_constraint
    ... class Ordered:
    ...     kind = "ordered"
    ...     def __init__(self, dims):
    ...         self.dims = list(dims)
    ...     def __call__(self, theta):
    ...         return theta[..., self.dims[0]] <= theta[..., self.dims[1]]
    ...     def to(self, device):
    ...         return self
    ...     def spec(self):
    ...         return {"kind": self.kind, "dims": self.dims}
    >>> prior = Prior([0, 0], [1, 1], constraints=[Ordered(["a", "b"])], names=["a", "b"])
"""
from __future__ import annotations

import copy
import inspect
import math
import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ._utils import Registry, as_2d, as_numpy, check_finite, did_you_mean, logger, plain, resolve_device, to_tensor

__all__ = ["Prior", "Gaussian", "KDE", "Polygon2D", "CurveBound2D", "loo_bandwidth", "TERMS", "CONSTRAINTS",
           "register_term", "register_constraint"]

TERMS = Registry("prior term")
CONSTRAINTS = Registry("prior constraint")

_MAX_BATCH = 1 << 20           # draws per round of the rejection sampler, at most
_EMPTY_AFTER = 10 * _MAX_BATCH  # give up when not one of this many draws was inside the support
_PAIRS = 1 << 22               # pairwise distances held in memory at once


def _register(registry, cls):
    kind = getattr(cls, "kind", None)
    if not isinstance(kind, str) or not kind:
        raise TypeError(f"{cls!r} needs a class attribute `kind` (a unique string) to be registered")
    existing = registry.get(kind)
    if existing is not None and existing is not cls and existing.__qualname__ != cls.__qualname__:
        raise ValueError(f"{registry.kind} kind {kind!r} is already registered for {existing.__name__}; "
                         "give the new class its own kind")
    registry[kind] = cls
    return cls


def register_term(cls):
    """Class decorator: make a density term loadable by its ``kind`` (:meth:`Prior.from_spec`,
    :func:`sbikit.io.load`). The module docstring lists what a term needs.

    Raises:
        TypeError: if the class has no ``kind``.
        ValueError: if another class is already registered under that ``kind``.
    """
    return _register(TERMS, cls)


def register_constraint(cls):
    """Class decorator: make a constraint class loadable by its ``kind`` (:meth:`Prior.from_spec`,
    :func:`sbikit.io.load`). The module docstring lists what a constraint needs.

    Raises:
        TypeError: if the class has no ``kind``.
        ValueError: if another class is already registered under that ``kind``.
    """
    return _register(CONSTRAINTS, cls)


def _t(a, device=None, dtype=torch.float32):
    return to_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a, dtype=dtype, device=device)


def _as_dims(dims, owner: str, n: Optional[int] = None) -> list:
    """``dims`` as a list of column indices (int) and/or parameter names (str), without repeats."""
    if isinstance(dims, (str, int, np.integer)):
        dims = [dims]
    out = []
    for d in (dims.tolist() if isinstance(dims, (np.ndarray, torch.Tensor)) else dims):
        if isinstance(d, str):
            out.append(str(d))
        elif isinstance(d, (int, np.integer)) and not isinstance(d, bool):
            out.append(int(d))
        else:
            raise ValueError(f"{owner}: dims must be column indices or parameter names, got {d!r}")
    if not out:
        raise ValueError(f"{owner}: dims is empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{owner}: dims {out} repeat a parameter")
    if n is not None and len(out) != n:
        raise ValueError(f"{owner}: dims must give {n} distinct parameters, got {out}")
    return out


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _fmt(values) -> str:
    return "[" + ", ".join(f"{float(v):.4g}" for v in values) + "]"


class _Part:
    """What the built-in terms and constraints share: ``dims`` handling and a ``to`` that keeps the
    class of the object (and any attributes a subclass adds)."""

    def _columns(self) -> list:
        if any(isinstance(d, str) for d in self.dims):
            raise ValueError(f"{type(self).__name__}: dims {self.dims} are parameter names; they become column "
                             "indices when the object is given to Prior(..., names=[...])")
        return self.dims

    def to(self, device):
        """A copy on ``device`` (same class, same values)."""
        device = resolve_device(device)
        new = copy.copy(self)
        new.dims = list(self.dims)
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, key, value.to(device))
        return new


# ------------------------------------------------------------------------------ density terms
@register_term
class Gaussian(_Part):
    """Multivariate normal on ``dims`` (e.g. correlated shifts of the n(z) means).

    Args:
        dims: the parameters the term acts on (column indices, or names resolved by the
            :class:`Prior`).
        mean: (len(dims),) mean.
        cov: (len(dims), len(dims)) covariance (symmetric, positive definite), or (len(dims),)
            variances.
        device: torch device of the term.

    Raises:
        ValueError: if ``mean`` or ``cov`` do not match ``dims``, hold non-finite values, or
            ``cov`` is not symmetric positive definite.
    """
    kind = "gaussian"

    def __init__(self, dims: Sequence, mean, cov, device="cpu"):
        device = resolve_device(device)
        self.dims = _as_dims(dims, "Gaussian")
        d = len(self.dims)
        mean, cov = np.atleast_1d(as_numpy(mean)), np.atleast_1d(as_numpy(cov))
        if mean.shape != (d,):
            raise ValueError(f"Gaussian: mean must have shape ({d},) for dims {self.dims}, got {mean.shape}")
        if cov.shape not in ((d,), (d, d)):
            raise ValueError(f"Gaussian: cov must be ({d},) variances or a ({d}, {d}) covariance for dims "
                             f"{self.dims}, got shape {cov.shape}")
        if not (np.isfinite(mean).all() and np.isfinite(cov).all()):
            raise ValueError("Gaussian: mean and cov must be finite")
        if cov.ndim == 1:
            if not (cov > 0).all():
                raise ValueError(f"Gaussian: variances must be positive, got {cov.tolist()}")
            cov = np.diag(cov)
        if np.abs(cov - cov.T).max() > 1e-6 * np.abs(cov).max():
            raise ValueError("Gaussian: cov is not symmetric")
        try:
            tril = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            raise ValueError("Gaussian: cov is not positive definite") from None
        self._mean64, self._cov64 = mean.copy(), cov.copy()        # private: spec() must not follow the caller's arrays
        self.mean, self.cov, self._tril = _t(mean, device), _t(cov, device), _t(tril, device)
        self.dist = torch.distributions.MultivariateNormal(self.mean, scale_tril=self._tril)

    def log_prob(self, x):
        """Normalised log density of the rows of ``x`` (n, len(dims)); returns (n,)."""
        return self.dist.log_prob(x)

    def sample(self, n, generator=None):
        """``n`` draws, (n, len(dims))."""
        z = torch.randn(n, len(self.dims), generator=generator, device=self.mean.device)
        return self.mean + z @ self.dist.scale_tril.T

    def to(self, device):
        """A copy on ``device`` (same class, same values)."""
        new = super().to(device)
        new.dist = torch.distributions.MultivariateNormal(new.mean, scale_tril=new._tril)
        return new

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`)."""
        return {"kind": self.kind, "dims": list(self.dims), "mean": torch.tensor(self._mean64),
                "cov": torch.tensor(self._cov64)}

    def __repr__(self):
        diagonal = np.allclose(self._cov64, np.diag(np.diag(self._cov64)))
        return (f"{type(self).__name__}(dims={self.dims}, mean={_fmt(self._mean64)}, "
                f"std={_fmt(np.sqrt(np.diag(self._cov64)))}{'' if diagonal else ', correlated'})")


def loo_bandwidth(centres, chol, grid=np.geomspace(0.02, 1.0, 60), *, max_points: int = 2000, seed: int = 0) -> float:
    """Bandwidth factor (in units of the data covariance) with the best leave-one-out likelihood.

    Rows that are exact copies of a centre (several simulations at one parameter point) are left
    out together with it; otherwise the likelihood would grow without bound towards zero bandwidth.
    A grid of more than 12 values is searched in two passes (every third value, then the
    neighbours of the best one), which finds the best value of the whole grid as long as the
    likelihood has a single maximum at that resolution. The cost is about
    ``min(n, max_points) * n * len(grid) / 2.5`` kernel evaluations, in memory chunks of a few MB.

    Args:
        centres: (n, d) points of the KDE.
        chol: (d, d) lower Cholesky factor of their covariance.
        grid: candidate bandwidth factors (positive; they are sorted).
        max_points: with more centres than this, the leave-one-out likelihood is averaged over a
            random subset of ``max_points`` centres (each still against all the others).
        seed: seed of that random subset.

    Returns:
        The best value of ``grid``.

    Raises:
        ValueError: for centres that are not (n >= 2, d), or a grid with non-positive values.
    """
    c = as_2d(centres, "centres", what="points")
    grid = np.atleast_1d(np.asarray(grid, dtype=np.float64))
    if grid.ndim != 1 or not (np.isfinite(grid).all() and (grid > 0).all()):
        raise ValueError("grid must be a 1D sequence of positive bandwidth factors")
    n, d = c.shape
    if n < 2:
        raise ValueError("loo_bandwidth needs at least 2 centres")
    if int(max_points) < 1:
        raise ValueError(f"max_points must be at least 1, got {max_points}")
    z = to_tensor(np.ascontiguousarray(np.linalg.solve(np.asarray(chol, dtype=np.float64), (c - c.mean(0)).T).T))
    rows = (np.arange(n) if n <= max_points
            else np.sort(np.random.default_rng(seed).choice(n, int(max_points), replace=False)))
    grid, step = np.unique(grid), max(1, _PAIRS // 4 // n)

    def scores(idx):
        """Mean leave-one-out log-likelihood of the grid values ``idx``."""
        total = torch.zeros(len(idx), dtype=torch.float64)
        for a in range(0, len(rows), step):
            sq = torch.cdist(z[rows[a:a + step]], z, compute_mode="donot_use_mm_for_euclid_dist").square_()
            sq[sq == 0.0] = torch.inf                 # the point itself and its exact copies
            buf = torch.empty_like(sq)
            for i, h in enumerate(grid[idx]):
                total[i] += torch.log(torch.mul(sq, -0.5 / h ** 2, out=buf).exp_().sum(1) + 1e-300).sum()
        return (total.numpy() / len(rows) - d * np.log(grid[idx])).tolist()

    first = sorted({*range(0, len(grid), 3), len(grid) - 1}) if len(grid) > 12 else list(range(len(grid)))
    score = dict(zip(first, scores(first)))
    top = max(first, key=score.get)
    near = [i for i in range(max(top - 2, 0), min(top + 3, len(grid))) if i not in score]
    if near:
        score.update(zip(near, scores(near)))
    best = max(sorted(score), key=score.get)
    if len(grid) > 1 and best in (0, len(grid) - 1):
        warnings.warn(f"loo_bandwidth: the best bandwidth {grid[best]:.3g} is the "
                      f"{'smallest' if best == 0 else 'largest'} value of the grid, so the optimum may lie beyond "
                      "it; pass a wider grid, or set the bandwidth by hand", RuntimeWarning, stacklevel=2)
    return float(grid[best])


@register_term
class KDE(_Part):
    """Gaussian KDE on ``dims`` (e.g. the density of a simulation design that is not uniform).

    The kernel covariance is ``bandwidth**2`` times the covariance of the centres. A copy of the
    centres is kept in float64, so the density is exactly the same after ``to()`` and after saving
    and loading, also if the array passed in is changed afterwards.

    Args:
        dims: the parameters the term acts on (column indices, or names resolved by the
            :class:`Prior`).
        centres: (n, len(dims)) points, n >= 2, with a non-singular covariance.
        bandwidth: kernel size in units of the data covariance (positive). None: the value with
            the best leave-one-out likelihood (:func:`loo_bandwidth`), which costs
            ``min(n, max_points) * n`` kernel evaluations per candidate value (a few seconds for
            20000 centres).
        device: torch device of the term.
        max_points: with more centres than this, the bandwidth search averages the leave-one-out
            likelihood over a seeded random subset of ``max_points`` centres.

    Raises:
        ValueError: for centres of the wrong shape, non-finite or with a singular covariance, or a
            bandwidth that is not positive.
    """
    kind = "kde"

    def __init__(self, dims: Sequence, centres, bandwidth: Optional[float] = None, device="cpu", *,
                 max_points: int = 2000):
        device = resolve_device(device)
        self.dims = _as_dims(dims, "KDE")
        d = len(self.dims)
        c = as_numpy(centres)
        c = as_2d(c[:, None] if c.ndim == 1 and d == 1 else c, "KDE centres", what="points")
        if c.shape[1] != d:
            raise ValueError(f"KDE: centres must be (n, {d}) for dims {self.dims}, got shape {c.shape}")
        if not np.isfinite(c).all():
            raise ValueError("KDE: centres contain NaN or inf")
        if len(c) < 2:
            raise ValueError("KDE: needs at least 2 centres")
        if bandwidth is not None and not (np.ndim(bandwidth) == 0 and np.isfinite(bandwidth) and bandwidth > 0):
            raise ValueError(f"KDE: bandwidth must be a positive number or None (leave-one-out selection), "
                             f"got {bandwidth!r}")
        mean, cov = c.mean(0), np.atleast_2d(np.cov(c.T))
        std = np.sqrt(np.diag(cov))
        if not (std > 0).all() or np.linalg.eigvalsh(cov / np.outer(std, std))[0] < 1e-10:
            raise ValueError(f"KDE: the covariance of the {len(c)} centres is singular (a constant or linearly "
                             f"dependent column, or not more centres than the {d} dimensions)")
        chol = np.linalg.cholesky(cov)
        if bandwidth is None:
            bandwidth = loo_bandwidth(c, chol, max_points=max_points)
            used = f"{min(len(c), max_points)} of {len(c)}" if len(c) > max_points else f"{len(c)}"
            logger.info(f"KDE: bandwidth {bandwidth:.3g} from the leave-one-out likelihood of {used} centres")
        self.bandwidth = float(bandwidth)
        self._centres64 = c.copy()                  # private: spec() must not follow the caller's array
        self.centres, self.mean = _t(c, device), _t(mean, device)
        self._mean_rest = _t(mean - mean.astype(np.float32), device)       # what float32 loses of the mean
        self.chol = _t(self.bandwidth * chol, device)
        self.white = _t(np.linalg.solve(self.bandwidth * chol, (c - mean).T).T, device)
        self.norm = float(-np.log(len(c)) - 0.5 * d * np.log(2 * np.pi) - np.log(self.bandwidth * np.diag(chol)).sum())

    def _whiten(self, x):
        centred = (x - self.mean.to(x)) - self._mean_rest.to(x)
        return torch.linalg.solve_triangular(self.chol.to(x), centred.T, upper=False).T

    def log_prob(self, x):
        """Normalised log density of the rows of ``x`` (n, len(dims)); returns (n,)."""
        out = []
        for chunk in x.split(max(1, _PAIRS // len(self.white))):        # (chunk, n_centres) matrices
            sq = torch.cdist(self._whiten(chunk), self.white.to(x), compute_mode="donot_use_mm_for_euclid_dist") ** 2
            out.append(torch.logsumexp(-0.5 * sq, dim=1) + self.norm)
        return torch.cat(out) if out else x.new_zeros(0)

    def sample(self, n, generator=None):
        """``n`` draws, (n, len(dims))."""
        dev = self.centres.device
        pick = torch.randint(len(self.centres), (n,), generator=generator, device=dev)
        return self.centres[pick] + torch.randn(n, len(self.dims), generator=generator, device=dev) @ self.chol.T

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`); the centres are stored in float64."""
        return {"kind": self.kind, "dims": list(self.dims), "centres": torch.tensor(self._centres64),
                "bandwidth": self.bandwidth}

    def __repr__(self):
        return (f"{type(self).__name__}(dims={self.dims}, {len(self._centres64)} centres, "
                f"bandwidth={self.bandwidth:.4g})")


# ------------------------------------------------------------------------------ constraints
@register_constraint
class Polygon2D(_Part):
    """Support constraint: inside a convex polygon in the plane of two parameters.

    Args:
        dims: the two parameters spanning the plane (column indices, or names resolved by the
            :class:`Prior`).
        vertices: (n_vertices >= 3, 2) corners of the convex polygon, listed in order around it
            (clockwise or anticlockwise).
        device: torch device of the constraint.

    Raises:
        ValueError: if ``dims`` are not two distinct parameters, or the vertices are not the
            corners of a convex polygon with a non-zero area, listed in order around it.
    """
    kind = "polygon2d"

    def __init__(self, dims: Sequence, vertices, device="cpu"):
        device = resolve_device(device)
        self.dims = _as_dims(dims, "Polygon2D", 2)
        v = as_numpy(vertices)
        if v.ndim != 2 or v.shape[1] != 2 or not np.isfinite(v).all():
            raise ValueError(f"Polygon2D: vertices must be (n >= 3, 2) finite corner points, got shape {v.shape}")
        v = v.astype(np.float32).astype(np.float64)                             # the corners as they are stored
        w = v[(np.roll(v, -1, 0) != v).any(1)]                                  # without repeated corners
        size = np.ptp(w, axis=0) if len(w) else np.zeros(2)
        convex = len(w) >= 3 and bool((size > 0).all())
        if convex:                       # tested in the unit box of the corners: the scales of the two axes drop out
            u = (w - w.min(0)) / size
            edges = np.roll(u, -1, 0) - u
            after = np.roll(edges, -1, 0)
            cross = edges[:, 0] * after[:, 1] - edges[:, 1] * after[:, 0]      # turn at each corner
            turns = np.arctan2(cross, (edges * after).sum(1)).sum() / (2 * np.pi)
            area = 0.5 * (edges[:, 0] * (u[:, 1] + np.roll(u[:, 1], -1))).sum()
            spacing = np.finfo(np.float32).eps * np.abs(w).max(0) / size       # of float32 numbers, per axis
            slack = np.abs(edges)[:, ::-1] @ spacing                           # what rounding can add, per edge
            tol = 1e-6 + 2 * (slack + np.roll(slack, -1))                      # per corner: 1e-6 and the rounding
            convex = (abs(area) > 1e-6 + slack.sum() and ((cross >= -tol).all() or (cross <= tol).all())
                      and abs(abs(turns) - 1) < 1e-3)
        if not convex:
            raise ValueError("Polygon2D: the vertices must be listed in order around a convex polygon with a "
                             "non-zero area (clockwise or anticlockwise); these are fewer than 3 distinct corners, "
                             "not convex, not in order, or lie on a line (as stored, in float32)")
        self.vertices = _t(v, device)
        v = self.vertices
        edges = torch.roll(v, -1, 0) - v
        normals = torch.stack([edges[:, 1], -edges[:, 0]], 1)
        outward = (normals * (v - v.mean(0))).sum(-1) > 0
        self.normals = torch.where(outward[:, None], normals, -normals)
        self.offsets = (self.normals * v).sum(-1)

    def __call__(self, theta):
        pts = theta[..., self._columns()]
        return ((pts @ self.normals.T.to(pts)) - self.offsets.to(pts) <= 0).all(-1)

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`)."""
        return {"kind": self.kind, "dims": list(self.dims), "vertices": self.vertices.detach().cpu().clone()}

    def __repr__(self):
        return f"{type(self).__name__}(dims={self.dims}, {len(self.vertices)} vertices)"


@register_constraint
class CurveBound2D(_Part):
    """Support constraint: one parameter above or below a curve in another parameter.

    The curve is piecewise linear through the nodes and constant beyond the first and last node,
    so the nodes should span the prior box of the x parameter (a :class:`Prior` warns if they do
    not). Any smooth edge can be tabulated (:meth:`from_function`); straight edges are better
    expressed with :class:`Polygon2D`.

    Args:
        dims: the two parameters ``[x, y]`` (column indices, or names resolved by the
            :class:`Prior`).
        x_nodes: (n,) strictly increasing x values, n >= 2.
        y_nodes: (n,) values of the curve at ``x_nodes``.
        side: ``"above"`` keeps ``y >= curve(x) + margin``, ``"below"`` keeps
            ``y <= curve(x) - margin``.
        margin: distance kept from the curve, in y.
        device: torch device of the constraint.

    Raises:
        ValueError: if ``dims`` are not two distinct parameters, the nodes are not finite, of equal
            length >= 2 and increasing in x, or ``side`` is unknown.

    Example:
        >>> bound = CurveBound2D([0, 1], [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], side="below")
        >>> bound(torch.tensor([[0.5, 0.4], [0.5, 0.6], [1.5, 0.4]]))
        tensor([ True, False,  True])
    """
    kind = "curve_bound2d"

    def __init__(self, dims: Sequence, x_nodes, y_nodes, side: str = "above", margin: float = 0.0,
                 device="cpu"):
        device = resolve_device(device)
        self.dims = _as_dims(dims, "CurveBound2D", 2)
        self.x_nodes, self.y_nodes = _t(x_nodes, device).detach().clone(), _t(y_nodes, device).detach().clone()
        self.side, self.margin = str(side), float(margin)
        if self.x_nodes.ndim != 1 or self.x_nodes.shape != self.y_nodes.shape or len(self.x_nodes) < 2:
            raise ValueError("CurveBound2D: x_nodes and y_nodes must be 1D with the same length >= 2")
        if not (torch.isfinite(self.x_nodes).all() and torch.isfinite(self.y_nodes).all()):
            bad = int((~(torch.isfinite(self.x_nodes) & torch.isfinite(self.y_nodes))).sum())
            raise ValueError(f"CurveBound2D: {bad} of {len(self.x_nodes)} nodes are NaN or inf; the curve must be "
                             "finite at every node")
        if not torch.all(self.x_nodes[1:] > self.x_nodes[:-1]):
            raise ValueError("CurveBound2D: x_nodes must be strictly increasing")
        if self.side not in ("above", "below"):
            raise ValueError(f"CurveBound2D: side must be 'above' or 'below', got {side!r}")
        if not math.isfinite(self.margin):
            raise ValueError(f"CurveBound2D: margin must be finite, got {margin!r}")

    @classmethod
    def from_function(cls, dims: Sequence, fn: Callable, x_min: float, x_max: float, n: int = 512,
                      **kwargs) -> "CurveBound2D":
        """Tabulate ``fn`` on ``n`` equally spaced nodes between ``x_min`` and ``x_max``.

        Args:
            dims: the two parameters ``[x, y]``.
            fn: numpy array (n,) -> array (n,), the curve ``y = fn(x)`` (a constant is broadcast).
            x_min, x_max: range of the nodes; use the prior box of the x parameter.
            n: number of nodes. The interpolation error is about ``(x_max - x_min)**2 / (8 n**2)``
                times the second derivative of ``fn``.
            **kwargs: ``side``, ``margin``, ``device``.

        Raises:
            ValueError: if ``fn`` does not return one value per node, or a non-finite one.
        """
        x = np.linspace(x_min, x_max, n)
        y = np.asarray(fn(x), dtype=np.float64)
        if y.shape not in ((), (1,), x.shape):
            raise ValueError(f"CurveBound2D.from_function: fn must map an array of shape {x.shape} to the same "
                             f"shape, got {y.shape}")
        return cls(dims, x, np.broadcast_to(y, x.shape), **kwargs)

    def curve(self, x):
        """The curve at ``x`` (any shape): linear between the nodes, constant outside them."""
        xn, yn = self.x_nodes.to(x), self.y_nodes.to(x)
        i = torch.searchsorted(xn, x.contiguous()).clamp(1, len(xn) - 1)
        w = ((x - xn[i - 1]) / (xn[i] - xn[i - 1])).clamp(0.0, 1.0)
        return yn[i - 1] + w * (yn[i] - yn[i - 1])

    def __call__(self, theta):
        ix, iy = self._columns()
        x, y = theta[..., ix], theta[..., iy]
        if self.side == "above":
            return y >= self.curve(x) + self.margin
        return y <= self.curve(x) - self.margin

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`)."""
        return {"kind": self.kind, "dims": list(self.dims), "x_nodes": self.x_nodes.detach().cpu().clone(),
                "y_nodes": self.y_nodes.detach().cpu().clone(), "side": self.side, "margin": self.margin}

    def __repr__(self):
        return (f"{type(self).__name__}(dims={self.dims}, side={self.side!r}, margin={self.margin:.4g}, "
                f"{len(self.x_nodes)} nodes on {_fmt([self.x_nodes[0], self.x_nodes[-1]])})")


def _from_spec(registry, spec, device):
    """Build a registered term or constraint from its ``spec()``."""
    args = dict(spec)
    kind = args.pop("kind")
    if kind not in registry:
        raise ValueError(f"unknown {registry.kind} kind {kind!r}{did_you_mean(kind, list(registry))}; registered: "
                         f"{sorted(registry)}. A custom class must be registered (register_term / "
                         "register_constraint) before a prior that uses it is loaded")
    cls = registry[kind]
    params = inspect.signature(cls).parameters
    if "device" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
        args["device"] = device
    return cls(**args)


# ------------------------------------------------------------------------------ prior
class Prior:
    """Box x density terms x constraints over P parameters.

    Args:
        low, high: (P,) box bounds. Parameters without a density term are uniform over the box and
            need finite bounds. A parameter covered by a term may have infinite bounds; the sampler
            then needs ``logit_box=False`` (see :attr:`bounded`), so for a Gaussian parameter
            bounds such as mean +- 5 sigma are usually the better choice.
        terms: density terms (:class:`Gaussian`, :class:`KDE`, or any object with the term
            interface of the module docstring) on disjoint parameter subsets.
        constraints: callables theta (..., P) -> bool (...) that are True inside the support
            (:class:`Polygon2D`, :class:`CurveBound2D`, or your own; they must index the last axis
            only).
        names: optional parameter names (P unique strings). With names, terms and constraints
            may give their ``dims`` as names; they are resolved to column indices on copies, the
            objects passed in are not modified.
        device: torch device of the bounds; the terms and constraints are used as given
            (:meth:`to` moves everything).

    Attributes:
        ndim: number of parameters P.
        names: list of the parameter names, or None.
        low, high: (P,) float32 tensors.
        terms, constraints: lists, with ``dims`` as column indices.

    Raises:
        ValueError: for bounds of the wrong shape or order, NaN bounds, infinite bounds on a
            parameter without a term, names of the wrong length or repeated, ``dims`` that are
            out of range, repeated, unknown names or shared between terms.
        TypeError: for a term without ``dims``/``log_prob``/``sample`` or a constraint that is not
            callable.

    Example:
        >>> prior = Prior([0.1, 0.5], [0.5, 1.0], names=["Om", "s8"])
        >>> prior.log_prob(torch.tensor([[0.3, 0.8], [0.6, 0.8]]))    # second point: outside
        tensor([1.6094,   -inf])
        >>> prior = Prior([0.1, 0.5, -0.1], [0.5, 1.0, 0.1], names=["Om", "s8", "dz"],
        ...               terms=[Gaussian(["dz"], [0.0], [0.02 ** 2])])
    """

    def __init__(self, low, high, terms: Sequence = (), constraints: Sequence[Callable] = (),
                 names: Optional[Sequence[str]] = None, device="cpu"):
        device = resolve_device(device)
        lo, hi = as_numpy(low), as_numpy(high)
        if lo.ndim != 1 or hi.ndim != 1 or lo.shape != hi.shape or len(lo) == 0:
            raise ValueError(f"low and high must both be (P,) arrays, got shapes {lo.shape} and {hi.shape}")
        self.ndim = len(lo)
        self.low, self.high = _t(lo, device), _t(hi, device)
        self.device = self.low.device
        if names is not None:
            if isinstance(names, str) or not all(isinstance(n, str) for n in names):
                raise ValueError(f"names must be a sequence of {self.ndim} strings, got {names!r}")
            names = [str(n) for n in names]
            if len(names) != self.ndim:
                raise ValueError(f"names has {len(names)} entries for {self.ndim} parameters")
            if len(set(names)) != len(names):
                raise ValueError(f"names must be unique, got {names}")
        self.names = names
        if np.isnan(lo).any() or np.isnan(hi).any():
            raise ValueError(f"low/high are NaN for {self._labels(np.flatnonzero(np.isnan(lo) | np.isnan(hi)))}")
        order = ~(self.high > self.low).cpu().numpy()
        if order.any():
            raise ValueError(f"high must be above low (in float32) for every parameter; not so for "
                             f"{self._labels(np.flatnonzero(order))}")

        self.terms, covered = [], {}
        for t in terms:
            if not all(hasattr(t, a) for a in ("dims", "log_prob", "sample")):
                raise TypeError(f"prior term {t!r} needs dims, log_prob(x) and sample(n, generator)")
            t = self._resolved(t, f"term {type(t).__name__}")
            for d in t.dims:
                if d in covered:
                    raise ValueError(f"density terms must act on disjoint parameters; {self._labels([d])} is in "
                                     f"{covered[d]} and {type(t).__name__}")
                covered[d] = type(t).__name__
            self.terms.append(t)
        self.constraints = []
        for c in constraints:
            if not callable(c):
                raise TypeError(f"prior constraint {c!r} is not callable")
            c = self._resolved(c, f"constraint {type(c).__name__}") if hasattr(c, "dims") else c
            if isinstance(c, CurveBound2D):
                ix, x0, x1 = c.dims[0], float(c.x_nodes[0]), float(c.x_nodes[-1])
                slack = 1e-6 * (hi[ix] - lo[ix])
                if np.isfinite(slack) and (x0 > lo[ix] + slack or x1 < hi[ix] - slack):
                    warnings.warn(f"the nodes of {c!r} span [{x0:.4g}, {x1:.4g}] but {self._labels([ix])} ranges "
                                  f"over [{lo[ix]:.4g}, {hi[ix]:.4g}]; the curve is held constant beyond its end "
                                  "nodes", stacklevel=2)
            self.constraints.append(c)

        self.uniform_dims = [d for d in range(self.ndim) if d not in covered]
        improper = [d for d in self.uniform_dims if not (np.isfinite(lo[d]) and np.isfinite(hi[d]))]
        if improper:
            raise ValueError(f"infinite bounds for {self._labels(improper)}, which has no density term: a uniform "
                             "prior over an infinite range is improper. Give finite bounds or add a term "
                             "(e.g. Gaussian)")
        self.log_uniform = -torch.log(self.high - self.low)[self.uniform_dims].sum()

    def _labels(self, dims) -> str:
        return ", ".join(repr(self.names[d]) if self.names else f"parameter {d}" for d in dims)

    def _resolved(self, part, owner: str):
        """``part`` with ``dims`` as column indices; a copy if its ``dims`` had to be changed."""
        given, dims = part.dims, []
        for d in _as_dims(given, owner):
            if isinstance(d, str):
                if self.names is None:
                    raise ValueError(f"{owner} gives dims by name ({d!r}) but the prior has no names; pass "
                                     "names=[...] to Prior")
                if d not in self.names:
                    raise ValueError(f"{owner}: unknown parameter {d!r}{did_you_mean(d, self.names)}; the prior's "
                                     f"names are {self.names}")
                d = self.names.index(d)
            elif not -self.ndim <= d < self.ndim:
                raise ValueError(f"{owner}: dims entry {d} is out of range for {self.ndim} parameters")
            dims.append(d % self.ndim)
        if len(set(dims)) != len(dims):
            raise ValueError(f"{owner}: dims {list(given)} repeat a parameter")
        if isinstance(given, list) and given == dims and all(type(d) is int for d in given):
            return part
        part = copy.copy(part)
        part.dims = dims
        return part

    @classmethod
    def box_from_data(cls, theta, pad: float = 0.0, **kwargs) -> "Prior":
        """Uniform box over the range of ``theta`` (N, P), with optional terms/constraints.

        The bounds are stored in float32 and rounded outward, so that every row of ``theta`` is
        inside the box both in float32 and in float64.

        Args:
            theta: (N, P) parameters (array-like or tensor).
            pad: widen the box by this fraction of each parameter's range on both sides.
            **kwargs: ``terms``, ``constraints``, ``names``, ``device`` of :class:`Prior`.

        Raises:
            ValueError: if ``theta`` is not 2D, holds NaN/inf, ``pad`` is negative, or a parameter
                has the same value in every row (zero width).
        """
        t = as_2d(theta, "theta", what="simulations")
        check_finite(t, "theta")
        if not (np.isfinite(pad) and pad >= 0):
            raise ValueError(f"pad must be >= 0, got {pad!r}")
        lo, hi = t.min(0), t.max(0)
        flat = np.flatnonzero(hi == lo)
        if len(flat):
            names = kwargs.get("names")
            cols = [f"{names[i]!r}" if names is not None and len(names) == t.shape[1] else f"column {i}" for i in flat]
            raise ValueError(f"box_from_data: {', '.join(cols)} has the same value in every row (zero width); "
                             "drop fixed parameters or give the box explicitly with Prior(low, high)")
        lo, hi = lo - pad * (hi - lo), hi + pad * (hi - lo)
        lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
        lo32 = np.where(lo32 > lo, np.nextafter(lo32, np.float32(-np.inf)), lo32)
        hi32 = np.where(hi32 < hi, np.nextafter(hi32, np.float32(np.inf)), hi32)
        return cls(lo32, hi32, **kwargs)

    @property
    def bounded(self) -> bool:
        """True if every box bound is finite. With an infinite bound (allowed for parameters that
        have a density term) the sampler cannot work in the logit of the box: use
        ``MCMCConfig(logit_box=False)``."""
        return bool(torch.isfinite(self.low).all() and torch.isfinite(self.high).all())

    def index(self, names):
        """Column index of a parameter name, or a list of indices for a sequence of names.

        Raises:
            ValueError: if the prior has no names or a name is unknown.
        """
        if self.names is None:
            raise ValueError("this prior has no parameter names; pass names=[...] to Prior")
        single = isinstance(names, str)
        out = []
        for n in ([names] if single else names):
            if n not in self.names:
                raise ValueError(f"unknown parameter {n!r}{did_you_mean(str(n), self.names)}; names: {self.names}")
            out.append(self.names.index(n))
        return out[0] if single else out

    def _input(self, theta):
        """(theta as a floating tensor, whether it was given as a tensor)."""
        is_tensor = isinstance(theta, torch.Tensor)
        if not is_tensor:
            theta = _t(np.asarray(theta, dtype=np.float32), self.device)
        elif not theta.is_floating_point():
            theta = theta.to(self.low.dtype)
        if theta.ndim == 0 or theta.shape[-1] != self.ndim:
            raise ValueError(f"theta must have shape (..., {self.ndim}), got {tuple(theta.shape)}")
        return theta, is_tensor

    def _inside(self, theta):
        mask = ((theta >= self.low.to(theta)) & (theta <= self.high.to(theta))).all(-1)
        for c in self.constraints:
            ok = c(theta)
            if not isinstance(ok, torch.Tensor) or ok.dtype != torch.bool or ok.shape != mask.shape:
                got = f"a {ok.dtype} tensor of shape {tuple(ok.shape)}" if isinstance(ok, torch.Tensor) else repr(ok)
                raise ValueError(f"constraint {c!r} returned {got} for theta of shape {tuple(theta.shape)}; a "
                                 f"constraint must return a bool tensor of shape {tuple(mask.shape)} and index the "
                                 "last axis only (theta[..., i], not theta[:, i])")
            mask = mask & ok
        return mask

    def inside(self, theta):
        """True where ``theta`` (..., P) is in the box and passes every constraint.

        Args:
            theta: (..., P) tensor, or array-like.

        Returns:
            Boolean mask (...): a tensor for a tensor, a numpy array otherwise.

        Raises:
            ValueError: if the last dimension is not P, or a constraint returns the wrong shape.
        """
        theta, is_tensor = self._input(theta)
        mask = self._inside(theta)
        return mask if is_tensor else mask.cpu().numpy()

    def log_prob(self, theta):
        """Log density of ``theta`` (..., P); -inf outside the support.

        Args:
            theta: (..., P) tensor (on the prior's device), or array-like.

        Returns:
            (...) log densities: a tensor for a tensor, a numpy array otherwise.

        Raises:
            ValueError: if the last dimension is not P.
        """
        theta, is_tensor = self._input(theta)
        flat = theta.reshape(-1, self.ndim)
        lp = self.log_uniform.to(flat).expand(len(flat)).clone()
        cols = flat.to(self.low.dtype)
        for t in self.terms:
            lp = lp + t.log_prob(cols[:, t.dims]).to(lp)
        lp = lp.reshape(theta.shape[:-1])
        lp = torch.where(self._inside(theta), lp, torch.full_like(lp, -torch.inf))
        return lp if is_tensor else lp.detach().cpu().numpy()

    def sample(self, n: int, generator=None, max_rounds: int = 1000, *, seed: Optional[int] = None):
        """``n`` draws by rejection: uniform in the box, the density terms from their own samplers,
        draws outside the box or the constraints redrawn.

        The number of draws per round follows the measured acceptance (at most about 1e6 per
        round), so small supports are sampled without tuning.

        Args:
            n: number of samples.
            generator: a ``torch.Generator`` on the prior's device, or an int seed. None: torch's
                global random state.
            max_rounds: give up after this many rounds, or as soon as the measured acceptance shows
                that they cannot be enough.
            seed: an int seed (the same as passing it as ``generator``).

        Returns:
            (n, P) tensor on the prior's device.

        Raises:
            ValueError: for a negative ``n``, ``max_rounds`` that is not an int >= 1, or both
                ``generator`` and ``seed``.
            TypeError: for a ``generator`` or ``seed`` that is not an int or a ``torch.Generator``.
            RuntimeError: if ``max_rounds`` rounds were not enough, or none of the first 1e7 draws
                was inside the support; the message gives the measured acceptance.
        """
        n = int(n)
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if not _is_int(max_rounds) or max_rounds < 1:
            raise ValueError(f"max_rounds must be an int >= 1, got {max_rounds!r}")
        if seed is not None and generator is not None:
            raise ValueError("pass either generator or seed, not both")
        name, generator = ("generator", generator) if seed is None else ("seed", seed)
        if _is_int(generator):
            generator = torch.Generator(device=self.device).manual_seed(int(generator))
        elif generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError(f"{name} must be an int seed, a torch.Generator or None, got {generator!r}")
        if n == 0:
            return self.low.new_empty((0, self.ndim))
        finite = torch.isfinite(self.low) & torch.isfinite(self.high)     # the others are drawn by their term
        low = torch.where(finite, self.low, torch.zeros_like(self.low))
        width = torch.where(finite, self.high - self.low, torch.zeros_like(self.low))
        out, have, drawn, m = [], 0, 0, min(max(2 * n, 64), _MAX_BATCH)
        for done in range(1, max_rounds + 1):
            draw = low + torch.rand(m, self.ndim, generator=generator, device=self.device) * width
            for t in self.terms:
                draw[:, t.dims] = t.sample(m, generator)
            draw = draw[self._inside(draw)]
            out.append(draw)
            have, drawn = have + len(draw), drawn + m
            if have >= n:
                return torch.cat(out)[:n]
            if have == 0 and drawn >= _EMPTY_AFTER:
                break
            if have >= 16 and (n - have) * drawn / have > 2 * _MAX_BATCH * (max_rounds - done):
                break                                 # the remaining rounds cannot deliver the rest
            m = min(max(int(1.2 * (n - have) * drawn / max(have, 1)) + 1, 64), _MAX_BATCH)
        if have == 0:
            raise RuntimeError(f"Prior.sample: none of {drawn:,} draws were inside the support (acceptance below "
                               f"{1 / drawn:.0e}). The support is empty or a tiny part of the box: check the "
                               "constraints and density terms against the box bounds")
        need = math.ceil(1.2 * n * drawn / have / _MAX_BATCH) + 3
        raise RuntimeError(f"Prior.sample: only {have} of {n} samples after {drawn:,} draws (acceptance "
                           f"{have / drawn:.2e}); max_rounds={max_rounds} rounds of at most {_MAX_BATCH:,} draws "
                           f"are not enough. Pass max_rounds >= {need}, or shrink the box around the support so "
                           "that fewer draws are rejected")

    def to(self, device) -> "Prior":
        """A copy on ``device``; the class and any extra attributes of a subclass are kept."""
        device = resolve_device(device)
        new = copy.copy(self)
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(new, key, value.to(device))
        new.terms = [t.to(device) if hasattr(t, "to") else t for t in self.terms]
        new.constraints = [c.to(device) if hasattr(c, "to") else c for c in self.constraints]
        new.names = None if self.names is None else list(self.names)
        new.uniform_dims = list(self.uniform_dims)
        new.device = new.low.device
        return new

    def spec(self) -> dict:
        """A serialisable description: Python scalars, strings, lists and tensors only.

        Raises:
            TypeError: if a term or constraint has no ``spec()`` (e.g. a plain function), its
                ``kind`` is not registered (it could not be loaded again), or its ``kind`` is
                registered for another class (e.g. a subclass that inherits the ``kind`` of a
                built-in class would be loaded as that class).
        """
        def describe(part, registry, register):
            what = registry.kind
            if not hasattr(part, "spec"):
                raise TypeError(f"{what} {part!r} has no spec() and cannot be stored; use the built-in classes, or "
                                f"a class with kind, dims, to() and spec() registered with {register}")
            s = plain(part.spec())
            if s.get("kind") not in registry:
                raise TypeError(f"{what} kind {s.get('kind')!r} of {type(part).__name__} is not registered, so the "
                                f"stored prior could not be loaded; decorate the class with @{register}")
            known, own = registry[s["kind"]], type(part)
            if known is not own and (known.__module__, known.__qualname__) != (own.__module__, own.__qualname__):
                raise TypeError(f"{what} {type(part).__name__} has kind {s['kind']!r}, which is registered for "
                                f"{known.__name__}: the stored prior would be loaded with a {known.__name__}. Give "
                                f"{type(part).__name__} its own `kind` and decorate it with @{register}")
            return s
        return {"low": self.low.detach().cpu().clone(), "high": self.high.detach().cpu().clone(),
                "names": None if self.names is None else [str(n) for n in self.names],
                "terms": [describe(t, TERMS, "register_term") for t in self.terms],
                "constraints": [describe(c, CONSTRAINTS, "register_constraint") for c in self.constraints]}

    @classmethod
    def from_spec(cls, spec: dict, device="cpu") -> "Prior":
        """Rebuild a prior from :meth:`spec`. Terms and constraints are built from the registries
        (:func:`register_term`, :func:`register_constraint`).

        Raises:
            ValueError: for a term or constraint kind that is not registered.
        """
        device = resolve_device(device)
        terms = [_from_spec(TERMS, s, device) for s in spec.get("terms", ())]
        cons = [_from_spec(CONSTRAINTS, s, device) for s in spec.get("constraints", ())]
        return cls(spec["low"], spec["high"], terms, cons, spec.get("names"), device)

    def __repr__(self):
        box = ", ".join(f"{self.names[i] if self.names else i}: [{lo:.4g}, {hi:.4g}]"
                        for i, (lo, hi) in enumerate(zip(self.low.tolist(), self.high.tolist())))
        extra = "".join(f", {label}={parts}" for label, parts in (("terms", self.terms),
                                                                    ("constraints", self.constraints)) if parts)
        return f"{type(self).__name__}({box}{extra})"
