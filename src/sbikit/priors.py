"""Priors in raw parameter space (torch, batched, GPU-capable).

A :class:`Prior` is a box (needed by the logit-space sampler) times optional density terms on
subsets of the parameters (Gaussian, KDE); parameters not covered by a term are uniform over the
box. Optional support constraints (convex polygons, bounds along a curve, or any callable
returning a boolean mask) set the density to -inf outside. ``log_prob`` is normalised up to the
truncation by the box and the constraints, which only adds a constant per prior.

Density terms share one interface: ``dims`` (the parameter columns they act on),
``log_prob(x)`` and ``sample(n, generator)`` on the (n, len(dims)) columns, ``to(device)`` and
``spec()`` (a serialisable description). Constraints are callables ``theta (..., P) -> bool (...)``;
the built-in ones also have ``to`` and ``spec`` so that a prior using them can be saved.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

__all__ = ["Prior", "Gaussian", "KDE", "Polygon2D", "CurveBound2D", "loo_bandwidth"]


def _t(a, device=None, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a, dtype=dtype, device=device)


# ------------------------------------------------------------------------------ density terms
class Gaussian:
    """Multivariate normal on ``dims`` (e.g. correlated shifts of the n(z) means).

    Args:
        dims: the parameter columns the term acts on.
        mean: (len(dims),) mean.
        cov: (len(dims), len(dims)) covariance, or a 1D array of variances.
        device: torch device of the term.
    """
    kind = "gaussian"

    def __init__(self, dims: Sequence[int], mean, cov, device="cpu"):
        self.dims = list(dims)
        self.mean, self.cov = _t(mean, device), _t(cov, device)
        if self.cov.ndim == 1:
            self.cov = torch.diag(self.cov)
        self.dist = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.cov)

    def log_prob(self, x):
        """Normalised log density of the rows of ``x`` (n, len(dims)); returns (n,)."""
        return self.dist.log_prob(x)

    def sample(self, n, generator=None):
        """``n`` draws, (n, len(dims))."""
        z = torch.randn(n, len(self.dims), generator=generator, device=self.mean.device)
        return self.mean + z @ self.dist.scale_tril.T

    def to(self, device):
        """A copy on ``device``."""
        return Gaussian(self.dims, self.mean.to(device), self.cov.to(device), device)

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`)."""
        return {"kind": self.kind, "dims": self.dims, "mean": self.mean.cpu(), "cov": self.cov.cpu()}


def loo_bandwidth(centres: np.ndarray, chol: np.ndarray, grid=np.geomspace(0.02, 1.0, 60)) -> float:
    """Bandwidth factor (in units of the data covariance) with the best leave-one-out likelihood.

    Args:
        centres: (n, d) points of the KDE.
        chol: (d, d) lower Cholesky factor of their covariance.
        grid: candidate bandwidth factors.
    """
    z = np.linalg.solve(chol, centres.T).T
    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    d = z.shape[1]
    scores = [np.mean(np.log(np.exp(-0.5 * sq / h ** 2).sum(1) + 1e-300)) - d * np.log(h) for h in grid]
    return float(grid[int(np.argmax(scores))])


class KDE:
    """Gaussian KDE on ``dims`` (e.g. the density of a simulation design that is not uniform).

    The kernel covariance is ``bandwidth**2`` times the covariance of the centres.

    Args:
        dims: the parameter columns the term acts on.
        centres: (n, len(dims)) points.
        bandwidth: kernel size in units of the data covariance; default: the value with the best
            leave-one-out likelihood (:func:`loo_bandwidth`).
        device: torch device of the term.
    """
    kind = "kde"

    def __init__(self, dims: Sequence[int], centres, bandwidth: Optional[float] = None, device="cpu"):
        self.dims = list(dims)
        c = np.asarray(centres.cpu() if isinstance(centres, torch.Tensor) else centres, dtype=np.float64)
        chol = np.linalg.cholesky(np.atleast_2d(np.cov(c.T)))
        self.bandwidth = float(bandwidth) if bandwidth else loo_bandwidth(c, chol)
        self.chol = _t(self.bandwidth * chol, device)
        self.centres = _t(c, device)
        self.white = self._whiten(self.centres)
        self.norm = (-np.log(len(c)) - 0.5 * c.shape[1] * np.log(2 * np.pi)
                     - torch.log(torch.diagonal(self.chol)).sum())

    def _whiten(self, x):
        return torch.linalg.solve_triangular(self.chol, x.T, upper=False).T

    def log_prob(self, x):
        """Normalised log density of the rows of ``x`` (n, len(dims)); returns (n,)."""
        out = []
        for chunk in x.split(1 << 14):        # (chunk, n_centres) matrices
            z = self._whiten(chunk)
            sq = (z ** 2).sum(1, keepdim=True) + (self.white ** 2).sum(1) - 2 * z @ self.white.T
            out.append(torch.logsumexp(-0.5 * sq, dim=1) + self.norm)
        return torch.cat(out) if out else x.new_zeros(0)

    def sample(self, n, generator=None):
        """``n`` draws, (n, len(dims))."""
        dev = self.centres.device
        pick = torch.randint(len(self.centres), (n,), generator=generator, device=dev)
        return self.centres[pick] + torch.randn(n, len(self.dims), generator=generator, device=dev) @ self.chol.T

    def to(self, device):
        """A copy on ``device`` (same bandwidth)."""
        return KDE(self.dims, self.centres.cpu().numpy(), self.bandwidth, device)

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`)."""
        return {"kind": self.kind, "dims": self.dims, "centres": self.centres.cpu(), "bandwidth": self.bandwidth}


# ------------------------------------------------------------------------------ constraints
class Polygon2D:
    """Support constraint: inside a convex polygon in the plane of two parameters.

    Args:
        dims: the two parameter columns spanning the plane.
        vertices: (n_vertices, 2) corners of the convex polygon, clockwise or anticlockwise.
        device: torch device of the constraint.
    """
    kind = "polygon2d"

    def __init__(self, dims: Sequence[int], vertices, device="cpu"):
        self.dims = list(dims)
        self.vertices = _t(vertices, device)
        v = self.vertices
        edges = torch.roll(v, -1, 0) - v
        normals = torch.stack([edges[:, 1], -edges[:, 0]], 1)
        outward = (normals * (v - v.mean(0))).sum(-1) > 0
        self.normals = torch.where(outward[:, None], normals, -normals)
        self.offsets = (self.normals * v).sum(-1)

    def __call__(self, theta):
        pts = theta[..., self.dims]
        return ((pts @ self.normals.T.to(pts)) - self.offsets.to(pts) <= 0).all(-1)

    def to(self, device):
        """A copy on ``device``."""
        return Polygon2D(self.dims, self.vertices.to(device), device)

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`)."""
        return {"kind": self.kind, "dims": self.dims, "vertices": self.vertices.cpu()}


class CurveBound2D:
    """Support constraint: one parameter above or below a curve in another parameter.

    The curve is piecewise linear through the nodes and constant beyond the first and last node,
    so the nodes should span the prior box of the x parameter. Any smooth edge can be tabulated
    (:meth:`from_function`); straight edges are better expressed with :class:`Polygon2D`.

    Args:
        dims: the two parameter columns ``[x, y]``.
        x_nodes: (n,) strictly increasing x values, n >= 2.
        y_nodes: (n,) values of the curve at ``x_nodes``.
        side: ``"above"`` keeps ``y >= curve(x) + margin``, ``"below"`` keeps
            ``y <= curve(x) - margin``.
        margin: distance kept from the curve, in y.
        device: torch device of the constraint.

    Example:
        >>> bound = CurveBound2D([0, 1], [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], side="below")
        >>> bound(torch.tensor([[0.5, 0.4], [0.5, 0.6], [1.5, 0.4]]))
        tensor([ True, False,  True])
    """
    kind = "curve_bound2d"

    def __init__(self, dims: Sequence[int], x_nodes, y_nodes, side: str = "above", margin: float = 0.0,
                 device="cpu"):
        self.dims = list(dims)
        self.x_nodes, self.y_nodes = _t(x_nodes, device), _t(y_nodes, device)
        self.side, self.margin = str(side), float(margin)
        if len(self.dims) != 2:
            raise ValueError("dims must name two parameter columns [x, y]")
        if self.x_nodes.ndim != 1 or self.x_nodes.shape != self.y_nodes.shape or len(self.x_nodes) < 2:
            raise ValueError("x_nodes and y_nodes must be 1D with the same length >= 2")
        if not torch.all(self.x_nodes[1:] > self.x_nodes[:-1]):
            raise ValueError("x_nodes must be strictly increasing")
        if self.side not in ("above", "below"):
            raise ValueError("side must be 'above' or 'below'")

    @classmethod
    def from_function(cls, dims: Sequence[int], fn: Callable, x_min: float, x_max: float, n: int = 512,
                      **kwargs) -> "CurveBound2D":
        """Tabulate ``fn`` on ``n`` equally spaced nodes between ``x_min`` and ``x_max``.

        Args:
            dims: the two parameter columns ``[x, y]``.
            fn: numpy array (n,) -> array (n,), the curve ``y = fn(x)``.
            x_min, x_max: range of the nodes; use the prior box of the x parameter.
            n: number of nodes. The interpolation error is about ``(x_max - x_min)**2 / (8 n**2)``
                times the second derivative of ``fn``.
            **kwargs: ``side``, ``margin``, ``device``.
        """
        x = np.linspace(x_min, x_max, n)
        return cls(dims, x, np.asarray(fn(x), dtype=np.float64), **kwargs)

    def curve(self, x):
        """The curve at ``x`` (any shape): linear between the nodes, constant outside them."""
        xn, yn = self.x_nodes.to(x), self.y_nodes.to(x)
        i = torch.searchsorted(xn, x.contiguous()).clamp(1, len(xn) - 1)
        w = ((x - xn[i - 1]) / (xn[i] - xn[i - 1])).clamp(0.0, 1.0)
        return yn[i - 1] + w * (yn[i] - yn[i - 1])

    def __call__(self, theta):
        x, y = theta[..., self.dims[0]], theta[..., self.dims[1]]
        if self.side == "above":
            return y >= self.curve(x) + self.margin
        return y <= self.curve(x) - self.margin

    def to(self, device):
        """A copy on ``device``."""
        return CurveBound2D(self.dims, self.x_nodes.to(device), self.y_nodes.to(device), self.side, self.margin,
                            device)

    def spec(self):
        """Serialisable description (see :meth:`Prior.spec`)."""
        return {"kind": self.kind, "dims": self.dims, "x_nodes": self.x_nodes.cpu(), "y_nodes": self.y_nodes.cpu(),
                "side": self.side, "margin": self.margin}


_TERMS = {"gaussian": Gaussian, "kde": KDE}
_CONSTRAINTS = {"polygon2d": Polygon2D, "curve_bound2d": CurveBound2D}


# ------------------------------------------------------------------------------ prior
class Prior:
    """Box x density terms x constraints over P parameters.

    Args:
        low, high: (P,) box bounds. Every parameter needs finite bounds (the sampler moves in the
            logit of the box); for Gaussian parameters use e.g. mean +- 5 sigma.
        terms: density terms (:class:`Gaussian`, :class:`KDE`) on disjoint parameter subsets.
        constraints: callables theta (..., P) -> bool (...) that are True inside the support.
        names: optional parameter names (stored in the spec).
        device: torch device of the bounds; the terms and constraints are used as given
            (:meth:`to` moves everything).

    Example:
        >>> prior = Prior([0.1, 0.5], [0.5, 1.0], names=["Om", "s8"])
        >>> prior.log_prob(torch.tensor([[0.3, 0.8], [0.6, 0.8]]))    # second point: outside
        tensor([1.6094,   -inf])
    """

    def __init__(self, low, high, terms: Sequence = (), constraints: Sequence[Callable] = (),
                 names: Optional[Sequence[str]] = None, device="cpu"):
        self.low, self.high = _t(low, device), _t(high, device)
        if self.low.shape != self.high.shape or self.low.ndim != 1 or not torch.all(self.high > self.low):
            raise ValueError("low and high must be (P,) with high > low")
        self.ndim = len(self.low)
        self.terms, self.constraints = list(terms), list(constraints)
        self.names = None if names is None else list(names)
        covered = [d for t in self.terms for d in t.dims]
        if len(covered) != len(set(covered)):
            raise ValueError("density terms must act on disjoint parameters")
        self.uniform_dims = [d for d in range(self.ndim) if d not in covered]
        self.log_uniform = -torch.log(self.high - self.low)[self.uniform_dims].sum()
        self.device = self.low.device

    @classmethod
    def box_from_data(cls, theta, **kwargs) -> "Prior":
        """Uniform box over the range of ``theta`` (N, P), with optional terms/constraints."""
        t = np.asarray(theta)
        return cls(t.min(0), t.max(0), **kwargs)

    def inside(self, theta):
        """Boolean mask (...): True where ``theta`` (..., P) is in the box and passes every constraint."""
        mask = ((theta >= self.low.to(theta)) & (theta <= self.high.to(theta))).all(-1)
        for c in self.constraints:
            mask = mask & c(theta)
        return mask

    def log_prob(self, theta):
        """Log density of ``theta`` (..., P), shape (...); -inf outside the support."""
        flat = theta.reshape(-1, theta.shape[-1])
        lp = self.log_uniform.to(flat).expand(len(flat)).clone()
        for t in self.terms:
            lp = lp + t.log_prob(flat[:, t.dims])
        lp = lp.reshape(theta.shape[:-1])
        return torch.where(self.inside(theta), lp, torch.full_like(lp, -torch.inf))

    def sample(self, n: int, generator=None, max_rounds: int = 1000):
        """``n`` draws (n, P) by rejection: uniform in the box, the density terms from their own
        samplers, draws outside the box or the constraints redrawn (at most ``max_rounds`` times)."""
        out, have = [], 0
        for _ in range(max_rounds):
            m = max(2 * (n - have), 64)
            u = torch.rand(m, self.ndim, generator=generator, device=self.device)
            draw = self.low + u * (self.high - self.low)
            for t in self.terms:
                draw[:, t.dims] = t.sample(m, generator)
            draw = draw[self.inside(draw)]
            out.append(draw)
            have += len(draw)
            if have >= n:
                return torch.cat(out)[:n]
        raise RuntimeError("prior support too small for rejection sampling")

    def to(self, device) -> "Prior":
        """A copy on ``device``."""
        return Prior(self.low.to(device), self.high.to(device), [t.to(device) for t in self.terms],
                     [c.to(device) if hasattr(c, "to") else c for c in self.constraints], self.names, device)

    def spec(self) -> dict:
        """A serialisable description (custom callables can't be stored and raise)."""
        cons = []
        for c in self.constraints:
            if not hasattr(c, "spec"):
                raise TypeError(f"constraint {c!r} has no spec(); use the built-in constraint classes "
                                "to store the prior")
            cons.append(c.spec())
        return {"low": self.low.cpu(), "high": self.high.cpu(), "names": self.names,
                "terms": [t.spec() for t in self.terms], "constraints": cons}

    @classmethod
    def from_spec(cls, spec: dict, device="cpu") -> "Prior":
        """Rebuild a prior from :meth:`spec`."""
        terms = []
        for s in spec["terms"]:
            s = dict(s)
            terms.append(_TERMS[s.pop("kind")](**s, device=device))
        cons = []
        for s in spec["constraints"]:
            s = dict(s)
            cons.append(_CONSTRAINTS[s.pop("kind")](**s, device=device))
        return cls(spec["low"], spec["high"], terms, cons, spec.get("names"), device)
