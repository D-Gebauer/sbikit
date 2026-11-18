"""Priors and blinding: validation, name-based dims, registries, numerics of the KDE, sampling."""
import json
import math
import resource
import time
import warnings

import numpy as np
import pytest
import torch

from sbikit import CurveBound2D, Gaussian, KDE, Polygon2D, Prior, priors
from sbikit.blinding import Blinder, make_key
from sbikit.io import load, save
from sbikit.priors import CONSTRAINTS, TERMS, loo_bandwidth, register_constraint, register_term

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


# ------------------------------------------------------------------ names and validation
def test_dims_by_name_resolved_on_copies():
    gauss = Gaussian(["dz"], [0.0], [0.01])
    poly = Polygon2D(["s8", "Om"], [[0.6, 0.2], [1.0, 0.2], [1.0, 0.4]])
    curve = CurveBound2D(np.array(["Om", "w0"]), [0.2, 0.4], [-1.0, -1.0])
    names = np.array(["Om", "s8", "w0", "dz"])
    p = Prior([0.2, 0.6, -1.5, -0.1], [0.4, 1.0, -0.5, 0.1], [gauss], [poly, curve], names=names)
    assert p.terms[0].dims == [3] and p.constraints[0].dims == [1, 0] and p.constraints[1].dims == [0, 2]
    assert gauss.dims == ["dz"] and poly.dims == ["s8", "Om"]          # the caller's objects are untouched
    assert p.names == ["Om", "s8", "w0", "dz"] and all(type(n) is str for n in p.names)
    by_index = Prior([0.2, 0.6, -1.5, -0.1], [0.4, 1.0, -0.5, 0.1], [Gaussian([-1], [0.0], [0.01])],
                     [Polygon2D([1, 0], [[0.6, 0.2], [1.0, 0.2], [1.0, 0.4]]),
                      CurveBound2D([0, 2], [0.2, 0.4], [-1, -1])])
    th = by_index.sample(200, 0)
    assert torch.equal(p.log_prob(th), by_index.log_prob(th)) and torch.isfinite(p.log_prob(th)).all()
    assert p.index("w0") == 2 and p.index(["dz", "Om"]) == [3, 0]
    assert "Om: [0.2, 0.4]" in repr(p) and "Gaussian(dims=[3]" in repr(p) and "Polygon2D" in repr(p)
    with pytest.raises(ValueError, match="did you mean 's8'"):
        p.index("S8")
    with pytest.raises(ValueError, match="no names"):
        Prior([0, 0], [1, 1], [Gaussian(["a"], [0.0], [1.0])])
    with pytest.raises(ValueError, match="unknown parameter 'omega'.*did you mean 'Omega'"):
        Prior([0, 0], [1, 1], [Gaussian(["omega"], [0.0], [1.0])], names=["Omega", "b"])
    with pytest.raises(ValueError, match="parameter names"):          # a constraint used without a Prior
        poly(torch.zeros(3, 4))
    with pytest.raises(ValueError, match="no parameter names"):
        Prior([0], [1]).index("a")


def test_prior_validation():
    with pytest.raises(ValueError, match=r"\(P,\)"):
        Prior([0, 0], [1, 1, 1])
    with pytest.raises(ValueError, match=r"\(P,\)"):
        Prior([[0, 0]], [[1, 1]])
    with pytest.raises(ValueError, match="above low.*'b'"):
        Prior([0, 1], [1, 1], names=["a", "b"])
    with pytest.raises(ValueError, match="NaN for parameter 0"):
        Prior([np.nan, 0], [1, 1])
    with pytest.raises(ValueError, match="2 strings"):
        Prior([0, 0], [1, 1], names="ab")
    with pytest.raises(ValueError, match="3 entries for 2"):
        Prior([0, 0], [1, 1], names=["a", "b", "c"])
    with pytest.raises(ValueError, match="unique"):
        Prior([0, 0], [1, 1], names=["a", "a"])
    with pytest.raises(ValueError, match="out of range"):
        Prior([0, 0], [1, 1], [Gaussian([2], [0.0], [1.0])])
    with pytest.raises(ValueError, match="out of range"):
        Prior([0, 0], [1, 1], constraints=[Polygon2D([0, 5], SQUARE)])
    with pytest.raises(ValueError, match="repeat"):
        Prior([0, 0], [1, 1], [Gaussian([0, -2], [0.0, 0.0], [1.0, 1.0])])
    with pytest.raises(ValueError, match="disjoint.*'a'"):
        Prior([0, 0], [1, 1], [Gaussian([0], [0.0], [1.0]), Gaussian([-2], [0.0], [1.0])], names=["a", "b"])
    with pytest.raises(TypeError, match="log_prob"):
        Prior([0, 0], [1, 1], [object()])
    with pytest.raises(TypeError, match="not callable"):
        Prior([0, 0], [1, 1], constraints=[3])
    p = Prior([0, 0], [1, 1])
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        p.log_prob(torch.zeros(4, 3))
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        p.inside(np.zeros(3))


def test_infinite_bounds_need_a_term(device):
    with pytest.raises(ValueError, match="infinite bounds for 'b'.*improper"):
        Prior([0, -np.inf], [1, np.inf], names=["a", "b"])
    assert Prior([0, 0], [1, 1]).bounded
    p = Prior([0, -np.inf], [1, np.inf], [Gaussian(["b"], [2.0], [0.25])], names=["a", "b"]).to(device)
    assert not p.bounded
    s = p.sample(4000, 1)
    assert s.shape == (4000, 2) and torch.isfinite(s).all() and p.inside(s).all()
    assert abs(s[:, 1].mean().item() - 2.0) < 0.05 and abs(s[:, 1].std().item() - 0.5) < 0.05
    lp = p.log_prob(np.array([[0.5, 2.0], [0.5, 100.0], [1.5, 2.0]]))
    assert np.isclose(lp[0], -0.5 * np.log(2 * np.pi * 0.25), atol=1e-5) and np.isfinite(lp[1]) and lp[2] == -np.inf
    back = Prior.from_spec(p.spec())
    assert not back.bounded and torch.equal(back.low, p.low.cpu())


def test_gaussian_validation():
    with pytest.raises(ValueError, match=r"mean must have shape \(1,\)"):
        Gaussian([0], [0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="cov must be"):
        Gaussian([0, 1], [0.0, 1.0], np.eye(3))
    with pytest.raises(ValueError, match="positive"):
        Gaussian([0, 1], [0.0, 1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="not symmetric"):
        Gaussian([0, 1], [0.0, 1.0], [[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        Gaussian([0, 1], [0.0, 1.0], [[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        Gaussian([0], [np.nan], [1.0])
    with pytest.raises(ValueError, match="repeat"):
        Gaussian([0, 0], [0.0, 1.0], [1.0, 1.0])
    g = Gaussian(1, 0.5, 0.04)                                          # scalars for a single parameter
    assert g.dims == [1] and np.isclose(g.log_prob(torch.tensor([[0.5]])).item(), -0.5 * np.log(2 * np.pi * 0.04))
    full = Gaussian([0, 1], [0.0, 1.0], [[1.0, 0.3], [0.3, 2.0]])
    ref = torch.distributions.MultivariateNormal(torch.tensor([0.0, 1.0]), torch.tensor([[1.0, 0.3], [0.3, 2.0]]))
    x = torch.randn(50, 2, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(full.log_prob(x), ref.log_prob(x), atol=1e-5) and "correlated" in repr(full)


# ------------------------------------------------------------------ to(), numpy, sampling
class TemperedPrior(Prior):
    """A subclass with its own density and an extra attribute."""
    power = 0.5

    def log_prob(self, theta):
        return self.power * super().log_prob(theta)


class WideGaussian(Gaussian):
    def log_prob(self, x):
        return super().log_prob(x) - 1.0


def test_to_keeps_subclasses_and_attributes(device):
    def rule(theta):
        return theta[..., 0] < 0.9

    term = WideGaussian([1], [0.5], [0.04])
    p = TemperedPrior([0, 0], [1, 1], [term], [Polygon2D([0, 1], SQUARE), rule], names=["a", "b"])
    p.note = "kept"
    q = p.to(device)
    assert type(q) is TemperedPrior and q.note == "kept" and q.names == ["a", "b"] and q is not p
    assert type(q.terms[0]) is WideGaussian and type(q.constraints[0]) is Polygon2D and q.constraints[1] is rule
    assert q.low.device.type == torch.device(device).type == q.terms[0].mean.device.type == q.device.type
    th = torch.tensor([[0.2, 0.4], [0.95, 0.4]])
    expected = 0.5 * (Gaussian([1], [0.5], [0.04]).log_prob(th[:1, 1:]) - 1.0)
    lp = q.log_prob(th.to(device)).cpu()
    assert torch.allclose(lp[0], expected[0]) and lp[1] == -torch.inf
    kde = KDE([0], np.linspace(0.0, 1.0, 30), bandwidth=0.3).to(device)   # 1D centres for one parameter
    assert type(kde) is KDE and kde.centres.device.type == torch.device(device).type and kde.centres.shape == (30, 1)
    curve = CurveBound2D([0, 1], [0.0, 1.0], [0.0, 1.0]).to(device)
    assert type(curve) is CurveBound2D and curve.x_nodes.device.type == torch.device(device).type


def test_numpy_in_numpy_out():
    p = Prior([0.0, 0.0], [1.0, 2.0], constraints=[Polygon2D([0, 1], [[0, 0], [1, 0], [1, 2]])])
    th = np.array([[0.5, 0.5], [0.5, 1.5], [2.0, 0.5]])
    lp, inside = p.log_prob(th), p.inside(th)
    assert isinstance(lp, np.ndarray) and isinstance(inside, np.ndarray) and inside.dtype == bool
    assert inside.tolist() == [True, False, False] and np.isclose(lp[0], -np.log(2.0)) and lp[1] == -np.inf
    assert p.log_prob(th.tolist()).shape == (3,) and p.log_prob(th[0]).shape == ()
    assert p.log_prob(th[None]).shape == (1, 3)
    t = torch.as_tensor(th, dtype=torch.float32)
    assert isinstance(p.log_prob(t), torch.Tensor) and np.array_equal(p.log_prob(t).numpy(), lp)
    assert p.inside(t.reshape(3, 1, 2)).shape == (3, 1) and p.log_prob(t.double()).dtype == torch.float64
    assert torch.equal(p.inside(torch.tensor([[0, 0], [5, 5]])), torch.tensor([True, False]))     # integer tensor


def test_sample_seed_and_generator(device):
    p = Prior([0.0, -1.0], [1.0, 1.0], [Gaussian([1], [0.0], [0.09])], [Polygon2D([0, 1], SQUARE)]).to(device)
    torch.manual_seed(5)
    before = torch.rand(3)
    torch.manual_seed(5)
    a = p.sample(300, 7)
    assert torch.equal(torch.rand(3), before)                           # the global random state is untouched
    assert a.shape == (300, 2) and a.device.type == torch.device(device).type and p.inside(a).all()
    assert torch.equal(a, p.sample(300, seed=7)) and not torch.equal(a, p.sample(300, seed=8))
    assert torch.equal(a, p.sample(300, torch.Generator(device=device).manual_seed(7)))
    assert p.sample(0, 1).shape == (0, 2)
    with pytest.raises(ValueError, match="not both"):
        p.sample(10, torch.Generator(device=device), seed=1)


def test_small_support_is_sampled_and_give_up_reports_acceptance(device):
    corner = Polygon2D([0, 1], [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01]])     # 1e-4 of the box
    p = Prior([0.0, 0.0], [1.0, 1.0], constraints=[corner]).to(device)
    s = p.sample(2000, 0)
    assert s.shape == (2000, 2) and p.inside(s).all() and s.max().item() <= 0.01
    with pytest.raises(RuntimeError, match=r"acceptance \d\.\d+e-0[45]\).*max_rounds >= \d+"):
        p.sample(2000, 0, max_rounds=2)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="none of .* draws were inside the support"):
        Prior([0.0, 0.0], [1.0, 1.0], constraints=[lambda th: th[..., 0] > 2.0]).to(device).sample(10, 0)
    assert time.time() - t0 < 30


def test_sample_arguments_and_early_give_up(device):
    p = Prior([0.0, 0.0], [1.0, 1.0]).to(device)
    for bad in (0, -1, 2.5, None, True):
        with pytest.raises(ValueError, match="max_rounds must be an int >= 1"):
            p.sample(3, 0, max_rounds=bad)
    with pytest.raises(ValueError, match="max_rounds"):
        p.sample(0, 0, max_rounds=0)
    for bad in (True, "abc", 1.5, np.random.default_rng(0)):
        with pytest.raises(TypeError, match="^generator must be an int seed, a torch.Generator or None"):
            p.sample(3, bad)
        with pytest.raises(TypeError, match="^seed must be an int seed, a torch.Generator or None"):
            p.sample(3, seed=bad)
    with pytest.raises(ValueError, match="n must be >= 0"):
        p.sample(-1)
    a = p.sample(4, np.int64(3), max_rounds=np.int64(1))
    assert torch.equal(a, p.sample(4, seed=3)) and torch.equal(a, p.sample(4, None, 5, seed=np.int64(3)))
    assert torch.equal(a, p.sample(4, seed=torch.Generator(device=device).manual_seed(3)))
    empty = p.sample(0)
    assert empty.shape == (0, 2) and empty.dtype == torch.float32 and empty.device == p.low.device
    corner = Polygon2D([0, 1], [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01]])     # 1e-4 of the box
    small = Prior([0.0, 0.0], [1.0, 1.0], constraints=[corner]).to(device)
    t0 = time.time()                    # one round shows that three cannot deliver 1e7 samples: stop after it
    with pytest.raises(RuntimeError, match=rf"only \d+ of 10000000 samples after {priors._MAX_BATCH:,} draws "
                                           r"\(acceptance \d\.\d+e-0[45]\); max_rounds=3 .*max_rounds >= \d{5,}"):
        small.sample(10 ** 7, 0, max_rounds=3)
    assert time.time() - t0 < 5


def test_constraint_with_2d_indexing_is_named():
    def first_above(theta):
        return theta[:, 0] > 0.5

    p = Prior([0, 0], [1, 1], constraints=[first_above])
    assert p.sample(20, 0).shape == (20, 2)
    with pytest.raises(ValueError, match=r"first_above.*shape \(3, 2\).*last axis"):
        p.log_prob(torch.rand(3, 4, 2))
    with pytest.raises(ValueError, match="bool tensor"):
        Prior([0, 0], [1, 1], constraints=[lambda th: th[..., 0]]).inside(torch.rand(3, 2))


def test_box_from_data_contains_its_data():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(5000, 4)) * [1e-3, 1.0, 50.0, 1e4] + [0.965, 0.3, 70.0, 1e6]
    p = Prior.box_from_data(theta, names=list("abcd"))
    assert p.low.dtype == torch.float32
    assert p.inside(torch.as_tensor(theta)).all() and p.inside(torch.as_tensor(theta, dtype=torch.float32)).all()
    assert p.inside(theta).all() and p.inside(theta.astype(np.float32)).all()
    assert np.all(p.high.double().numpy() - theta.max(0) <= 2e-7 * np.abs(theta.max(0)))     # and no wider than needed
    assert Prior.box_from_data(torch.as_tensor(theta)).inside(theta).all()                   # tensors are accepted
    padded = Prior.box_from_data(theta, pad=0.1)
    width = theta.max(0) - theta.min(0)
    np.testing.assert_allclose(padded.low.double().numpy(), theta.min(0) - 0.1 * width, rtol=1e-6)
    np.testing.assert_allclose(padded.high.double().numpy(), theta.max(0) + 0.1 * width, rtol=1e-6)
    fixed = theta.copy()
    fixed[:, 2] = 70.0
    with pytest.raises(ValueError, match="'c' has the same value in every row"):
        Prior.box_from_data(fixed, names=list("abcd"))
    with pytest.raises(ValueError, match="column 2"):
        Prior.box_from_data(fixed)
    with pytest.raises(ValueError, match="2D"):
        Prior.box_from_data(theta[:, 0])
    with pytest.raises(ValueError, match="NaN or inf"):
        Prior.box_from_data(np.where(theta > 1e6 + 3e4, np.nan, theta))
    with pytest.raises(ValueError, match="pad"):
        Prior.box_from_data(theta, pad=-0.1)


# ------------------------------------------------------------------ specs and registries
def test_spec_is_plain_and_round_trips(tmp_path):
    dims = np.array([2, 0])
    p = Prior(np.array([0.0, 0.0, -1.0]), np.array([1.0, 1.0, 1.0]), [Gaussian(dims[:1], [0.0], [0.04])],
              [Polygon2D(dims, [[-1, 0], [1, 0], [1, 1]]),
               CurveBound2D((np.int64(0), np.int64(1)), [0.0, 1.0], [0.0, 0.5])],
              names=np.array(["a", "b", "c"]))

    def check(obj):
        if isinstance(obj, dict):
            assert all(type(k) is str for k in obj)
            for v in obj.values():
                check(v)
        elif isinstance(obj, list):
            for v in obj:
                check(v)
        else:
            assert obj is None or type(obj) in (int, float, str, bool, torch.Tensor), type(obj)

    spec = p.spec()
    check(spec)
    assert spec["names"] == ["a", "b", "c"] and spec["terms"][0]["dims"] == [2]
    assert spec["constraints"][0]["dims"] == [2, 0] and spec["constraints"][1]["dims"] == [0, 1]
    spec["low"][0] = 5.0                                               # the spec holds copies
    assert p.low[0] == 0.0
    save(p, tmp_path / "p.pt")                                         # loads with weights_only=True
    q = load(tmp_path / "p.pt")
    th = p.sample(500, 0)
    assert q.names == p.names and torch.equal(q.log_prob(th), p.log_prob(th))
    assert torch.equal(q.sample(50, 1), p.sample(50, 1))
    assert set(TERMS) >= {"gaussian", "kde"} and set(CONSTRAINTS) >= {"polygon2d", "curve_bound2d"}


@register_term
class Exponential:
    """Custom term: exponential density on one parameter."""
    kind = "test_exponential"

    def __init__(self, dims, rate, device="cpu"):
        self.dims, self.rate, self.device = list(dims), float(rate), device

    def log_prob(self, x):
        return math.log(self.rate) - self.rate * x[:, 0]

    def sample(self, n, generator=None):
        return -torch.log1p(-torch.rand(n, 1, generator=generator, device=self.device)) / self.rate

    def to(self, device):
        return type(self)(self.dims, self.rate, device)

    def spec(self):
        return {"kind": self.kind, "dims": self.dims, "rate": np.float64(self.rate)}


@register_constraint
class Ordered:
    """Custom constraint: the first of two parameters is not above the second."""
    kind = "test_ordered"

    def __init__(self, dims):
        self.dims = list(dims)

    def __call__(self, theta):
        return theta[..., self.dims[0]] <= theta[..., self.dims[1]]

    def to(self, device):
        return self

    def spec(self):
        return {"kind": self.kind, "dims": self.dims}


def test_custom_term_and_constraint_save_and_load(tmp_path, device):
    p = Prior([0.0, 0.0, 0.0], [1.0, 1.0, np.inf], [Exponential(["tau"], 2.0)], [Ordered(["a", "b"])],
              names=["a", "b", "tau"])
    save(p, tmp_path / "custom.pt")
    q = load(tmp_path / "custom.pt", device)
    assert type(q.terms[0]) is Exponential and q.terms[0].rate == 2.0 and q.terms[0].dims == [2]
    assert type(q.constraints[0]) is Ordered and q.constraints[0].dims == [0, 1]
    s = q.sample(3000, 2)
    assert (s[:, 0] <= s[:, 1]).all() and abs(s[:, 2].mean().item() - 0.5) < 0.05
    assert torch.allclose(q.log_prob(s).cpu(), p.log_prob(s.cpu()))
    assert np.isclose(p.log_prob(np.array([0.2, 0.6, 1.0])), math.log(2.0) - 2.0)

    class Unregistered(Ordered):
        kind = "test_unregistered"

    with pytest.raises(TypeError, match="not registered.*register_constraint"):
        Prior([0, 0], [1, 1], constraints=[Unregistered([0, 1])]).spec()
    with pytest.raises(TypeError, match="no spec"):
        Prior([0, 0], [1, 1], constraints=[lambda th: th[..., 0] > 0]).spec()
    with pytest.raises(ValueError, match="unknown prior term kind 'gausian'.*did you mean 'gaussian'"):
        Prior.from_spec({"low": [0.0], "high": [1.0], "names": None, "constraints": [],
                         "terms": [{"kind": "gausian", "dims": [0], "mean": [0.0], "cov": [1.0]}]})
    with pytest.raises(ValueError, match="already registered"):
        register_term(type("Other", (), {"kind": "gaussian"}))
    with pytest.raises(TypeError, match="kind"):
        register_constraint(type("NoKind", (), {}))


@register_term
class ShiftedGaussian(Gaussian):
    """A subclass with its own kind: stored and loaded as itself."""
    kind = "test_shifted_gaussian"

    def log_prob(self, x):
        return super().log_prob(x) - 1.0


def test_subclass_with_an_inherited_kind_is_not_stored_as_its_base(tmp_path):
    th = torch.tensor([[0.5], [0.7]])
    wide = Prior([0.0], [1.0], [WideGaussian([0], [0.5], [0.04])])                    # kind "gaussian", inherited
    with pytest.raises(TypeError, match="WideGaussian has kind 'gaussian', which is registered for Gaussian.*"
                                        "own `kind`.*@register_term"):
        wide.spec()
    with pytest.raises(TypeError, match="registered for Gaussian"):
        save(wide, tmp_path / "wide.pt")

    class Triangle(Polygon2D):
        pass

    with pytest.raises(TypeError, match="Triangle has kind 'polygon2d'.*Polygon2D.*@register_constraint"):
        Prior([0, 0], [1, 1], constraints=[Triangle([0, 1], SQUARE[:3])]).spec()
    shifted = Prior([0.0], [1.0], [ShiftedGaussian([0], [0.5], [0.04])])
    assert shifted.spec()["terms"][0]["kind"] == "test_shifted_gaussian"
    save(shifted, tmp_path / "shifted.pt")
    back = load(tmp_path / "shifted.pt")
    assert type(back.terms[0]) is ShiftedGaussian and torch.equal(back.log_prob(th), shifted.log_prob(th))
    assert torch.equal(back.log_prob(th), Prior([0.0], [1.0], [Gaussian([0], [0.5], [0.04])]).log_prob(th) - 1.0)
    assert Prior.from_spec(Prior([0.0], [1.0], [Gaussian([0], [0.5], [0.04])]).spec()).terms[0].kind == "gaussian"


# ------------------------------------------------------------------ KDE
def test_kde_bandwidth_with_repeated_centres():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(300, 2))
    h = KDE([0, 1], c).bandwidth
    assert 0.2 < h < 0.8
    assert KDE([0, 1], np.concatenate([c, c])).bandwidth == h          # every centre twice
    assert KDE([0, 1], np.repeat(c, 3, axis=0)).bandwidth == h
    full = np.geomspace(0.02, 1.0, 60)                                 # the two-pass search finds the grid optimum
    chol = np.linalg.cholesky(np.cov(c.T))
    z = np.linalg.solve(chol, c.T).T
    sq = ((z[:, None] - z[None]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    scores = [np.mean(np.log(np.exp(-0.5 * sq / g ** 2).sum(1) + 1e-300)) - 2 * np.log(g) for g in full]
    assert h == full[int(np.argmax(scores))] == loo_bandwidth(c, chol)
    with pytest.warns(RuntimeWarning, match="largest value of the grid"):
        loo_bandwidth(c, chol, grid=[0.01, 0.02, 0.05])
    assert loo_bandwidth(c, chol, grid=[0.9, 0.1, 0.4, 2.0]) == 0.4


def test_kde_validation():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(50, 2))
    for bad in (0, 0.0, -0.3, np.nan):
        with pytest.raises(ValueError, match="bandwidth must be a positive number"):
            KDE([0, 1], c, bandwidth=bad)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        KDE([0, 1], rng.normal(size=(50, 3)))
    with pytest.raises(ValueError, match="2D"):
        KDE([0, 1], rng.normal(size=50))
    with pytest.raises(ValueError, match="NaN or inf"):
        KDE([0, 1], np.where(c > 2.0, np.nan, c), bandwidth=0.3)
    with pytest.raises(ValueError, match="at least 2"):
        KDE([0, 1], c[:1])
    with pytest.raises(ValueError, match="singular"):
        KDE([0, 1], c[:2])
    with pytest.raises(ValueError, match="singular"):
        KDE([0, 1], np.stack([c[:, 0], 3.0 * c[:, 0]], 1))
    with pytest.raises(ValueError, match="singular"):
        KDE([0, 1], np.stack([c[:, 0], np.ones(50)], 1))
    assert "50 centres" in repr(KDE([0, 1], torch.as_tensor(c), bandwidth=0.3))


def test_kde_exact_after_save_load_and_accurate_far_from_origin(tmp_path, device):
    rng = np.random.default_rng(1)
    scale, shift = np.array([0.004, 0.5]), np.array([0.965, 70.0])     # means large against the spreads
    c = rng.normal(size=(400, 2)) * scale + shift
    x = (rng.normal(size=(1000, 2)) * scale + shift).astype(np.float32)
    for bandwidth in (None, 0.05):
        kde = KDE([0, 1], c, bandwidth)
        p = Prior([0.9, 60.0], [1.0, 80.0], [kde])
        chol = np.linalg.cholesky(np.cov(c.T)) * kde.bandwidth          # float64 reference
        zx = np.linalg.solve(chol, (x.astype(np.float64) - c.mean(0)).T).T
        zc = np.linalg.solve(chol, (c - c.mean(0)).T).T
        e = -0.5 * ((zx[:, None] - zc[None]) ** 2).sum(-1)
        ref = (e.max(1) + np.log(np.exp(e - e.max(1, keepdims=True)).sum(1)) - np.log(len(c)) - np.log(2 * np.pi)
               - np.log(np.diag(chol)).sum())
        lp = kde.log_prob(torch.as_tensor(x))
        assert np.abs(lp.numpy() - ref).max() < 2e-3
        save(p, tmp_path / "kde.pt")
        q = load(tmp_path / "kde.pt")
        assert q.terms[0].bandwidth == kde.bandwidth and q.spec()["terms"][0]["centres"].dtype == torch.float64
        assert torch.equal(q.terms[0].log_prob(torch.as_tensor(x)), lp)
        assert torch.equal(q.sample(200, 3), p.sample(200, 3))
        moved = kde.to(device)
        assert torch.allclose(moved.log_prob(torch.as_tensor(x, device=device)).cpu(), lp, atol=1e-4)
        assert torch.equal(moved.to("cpu").log_prob(torch.as_tensor(x)), lp)
    s = kde.sample(20000, torch.Generator().manual_seed(0)).double().numpy()
    assert np.allclose(s.mean(0), c.mean(0), atol=4 * scale / np.sqrt(400))
    assert np.allclose(s.std(0), c.std(0), rtol=0.1)


def test_terms_keep_private_copies_of_their_inputs(tmp_path):
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(300, 3)) * [0.05, 0.1, 1.0] + [0.3, 0.8, 0.0]            # float64, as most callers have
    prior = Prior.box_from_data(theta, pad=0.5, terms=[KDE([0, 1], theta[:, :2], 0.3)])
    x = torch.as_tensor(theta[:50], dtype=torch.float32)
    before, centres = prior.log_prob(x), prior.spec()["terms"][0]["centres"].clone()
    theta -= theta.mean(0)                                             # the caller changes its array in place
    assert torch.equal(prior.log_prob(x), before) and torch.equal(prior.spec()["terms"][0]["centres"], centres)
    save(prior, tmp_path / "p.pt")
    assert torch.equal(load(tmp_path / "p.pt").log_prob(x), before)
    tensor = torch.as_tensor(centres.numpy().copy())                   # a float64 tensor shares memory with numpy
    kde = KDE([0, 1], tensor, 0.3)
    tensor += 1.0
    assert torch.equal(kde.spec()["centres"], centres)
    one = np.linspace(0.0, 1.0, 30)                                    # 1D centres for one parameter
    kde = KDE([0], one, 0.3)
    one += 5.0
    assert np.array_equal(kde.spec()["centres"][:, 0].numpy(), np.linspace(0.0, 1.0, 30))

    mean, cov = np.array([0.0, 1.0]), np.array([[1.0, 0.3], [0.3, 2.0]])
    g = Gaussian([0, 1], mean, cov)
    lp, text = g.log_prob(x[:, :2]), repr(g)
    mean += 5.0
    cov *= 4.0
    again = Gaussian(**{k: v for k, v in g.spec().items() if k != "kind"})
    assert repr(g) == text == repr(again) and torch.equal(again.log_prob(x[:, :2]), lp)
    assert torch.equal(g.spec()["mean"], torch.tensor([0.0, 1.0], dtype=torch.float64))
    var = np.array([0.04, 0.09])                                       # variances
    g = Gaussian([0, 1], [0.0, 0.0], var)
    var *= 100.0
    assert torch.equal(g.spec()["cov"], torch.diag(torch.tensor([0.04, 0.09], dtype=torch.float64)))

    for convert in (lambda a: np.array(a, dtype=np.float32), torch.tensor):           # float32: no conversion needed
        xn, yn = convert([0.0, 1.0]), convert([0.0, 0.5])
        curve = CurveBound2D([0, 1], xn, yn)
        xn += 3.0
        yn += 3.0
        assert curve.x_nodes.tolist() == [0.0, 1.0] and curve.spec()["y_nodes"].tolist() == [0.0, 0.5]
        assert curve(torch.tensor([[0.5, 0.3], [0.5, 0.2]])).tolist() == [True, False]
    v = torch.tensor(SQUARE)
    poly = Polygon2D([0, 1], v)
    v += 3.0
    assert torch.equal(poly.vertices, torch.tensor(SQUARE)) and poly(torch.tensor([[0.5, 0.5]])).item()
    low = np.zeros(2, dtype=np.float32)
    p = Prior(low, torch.ones(2))
    low -= 5.0
    assert p.low.tolist() == [0.0, 0.0]


def test_kde_with_many_centres_is_fast_and_small():
    rng = np.random.default_rng(2)
    c = rng.normal(size=(20000, 2))
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    kde = KDE([0, 1], c)
    elapsed = time.time() - t0
    grown = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024       # MB (kB on Linux)
    assert elapsed < 30 and grown < 500, (elapsed, grown)             # all pairs at once would need several GB
    assert 0.15 < kde.bandwidth < 0.4
    lp = kde.log_prob(torch.zeros(1, 2)).item()
    assert abs(lp - (-np.log(2 * np.pi))) < 0.1
    chol = np.linalg.cholesky(np.cov(c[:3000].T))                     # the subset is seeded
    h = loo_bandwidth(c[:3000], chol, max_points=300)
    assert h == loo_bandwidth(c[:3000], chol, max_points=300, seed=0) and 0.15 < h < 0.5
    assert 0.15 < loo_bandwidth(c[:3000], chol, max_points=300, seed=1) < 0.5


# ------------------------------------------------------------------ constraints
def test_polygon_validation():
    th = torch.rand(2000, 2, generator=torch.Generator().manual_seed(0)) * 3 - 1
    tri = [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
    a, b = Polygon2D([0, 1], tri), Polygon2D([0, 1], tri[::-1])       # both orientations
    expected = (th[:, 0] >= 0) & (th[:, 1] >= 0) & (th[:, 0] / 2 + th[:, 1] <= 1)
    assert torch.equal(a(th), b(th)) and (a(th) == expected).float().mean() > 0.999
    assert torch.equal(Polygon2D([0, 1], tri + tri[:1])(th), a(th))    # closed: first corner repeated at the end
    assert torch.equal(Polygon2D([0, 1], [[0, 0], [1, 0], [2, 0], [0, 1]])(th), a(th))   # a corner on an edge
    order = "in order around a convex polygon"
    for bad in ([[0, 0], [1, 1], [1, 0], [0, 1]],                                          # bow-tie (not in order)
                [[0, 0], [2, 0], [2, 2], [1, 0.5], [0, 2]],                                # not convex
                [[0, 0], [1, 1], [2, 2]],                                                  # on a line
                [[0, 0], [1, 0]],                                                          # two corners
                [[0, 0], [0, 0], [0, 0]],
                [[np.cos(a), np.sin(a)] for a in 4 * np.pi * np.arange(5) / 5]):           # pentagram
        with pytest.raises(ValueError, match=order):
            Polygon2D([0, 1], bad)
    with pytest.raises(ValueError, match=r"\(n >= 3, 2\)"):
        Polygon2D([0, 1], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="repeat"):
        Polygon2D([0, 0], SQUARE)
    with pytest.raises(ValueError, match="2 distinct"):
        Polygon2D([0, 1, 2], SQUARE)
    far = np.array([[0, 0], [0.5, 0], [1, 0], [1, 1], [0.5, 1], [0, 1]]) * 0.1 + [70.0, -70.0]   # far from the origin
    poly = Polygon2D([0, 1], far)
    assert poly(torch.tensor([[70.05, -69.95], [70.12, -69.95]])).tolist() == [True, False]
    back = Polygon2D(**{k: v for k, v in poly.spec().items() if k != "kind"})          # stored corners stay valid
    assert torch.equal(back.vertices, poly.vertices)


def test_polygon_with_unequal_axis_scales():
    order = "in order around a convex polygon"
    th = torch.tensor([[0.3, 2e-9], [0.3, 4e-9], [0.6, 2e-9]])
    box = [[0.1, 1e-9], [0.5, 1e-9], [0.5, 3e-9], [0.1, 3e-9]]         # two parameters of very different size
    assert Polygon2D([0, 1], box)(th).tolist() == [True, False, False]
    assert Polygon2D([0, 1], [c[::-1] for c in box[::-1]])(th.flip(-1)).tolist() == [True, False, False]
    assert Polygon2D([1, 0], box)(th.flip(-1)).tolist() == [True, False, False]
    for scale in ([1.0, 1e-7], [1e-9, 1e6], [1e-9, 1e-9], [1e6, 1e6], [400.0, 0.002]):
        for shift in ([0.0, 0.0], [1.0, 3.0], [-2.0, 0.5]):            # in units of the polygon's size
            corners = (np.array(SQUARE) + shift) * scale
            poly = Polygon2D([0, 1], corners)
            inside = (torch.tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5]], dtype=torch.float64) + torch.tensor(shift))
            assert poly((inside * torch.tensor(scale)).float()).tolist() == [True, False, False], (scale, shift)
            dented = (np.array([[0, 0], [1, 0], [1, 1], [0.5, 0.85], [0, 1]]) + shift) * scale
            with pytest.raises(ValueError, match=order):
                Polygon2D([0, 1], dented)
            edge = (np.array([[0, 0], [0.5, 0], [1, 0], [1, 1], [0.25, 0.25]]) + shift) * scale   # corners on edges
            assert len(Polygon2D([0, 1], edge).vertices) == 5
    with pytest.raises(ValueError, match=order):                       # a reflex corner at (300, 0.0017)
        Polygon2D([0, 1], [[100, 0], [500, 0], [500, 0.002], [300, 0.0017], [100, 0.002]])
    with pytest.raises(ValueError, match=order):                       # all corners at one value of a parameter
        Polygon2D([0, 1], [[0.1, 1e-9], [0.5, 1e-9], [0.3, 1e-9]])
    with pytest.raises(ValueError, match=order):                       # narrower than float32 resolves at 70
        Polygon2D([0, 1], np.array(SQUARE) * [1e-6, 1.0] + [70.0, 0.0])
    spec = {"low": torch.tensor([0.1, 1e-9]), "high": torch.tensor([0.5, 3e-9]), "names": None, "terms": [],
            "constraints": [{"kind": "polygon2d", "dims": [0, 1],                       # a stored prior
                             "vertices": torch.tensor([[0.1, 1e-9], [0.5, 1e-9], [0.5, 3e-9], [0.3, 3e-9]])}]}
    prior = Prior.from_spec(spec)
    assert prior.inside(np.array([[0.4, 2e-9], [0.15, 2.5e-9]])).tolist() == [True, False]
    assert torch.equal(Prior.from_spec(prior.spec()).constraints[0].vertices, spec["constraints"][0]["vertices"])


def test_curve_bound_validation():
    with pytest.raises(ValueError, match="1 of 3 nodes are NaN or inf"):
        CurveBound2D([0, 1], [0.0, 1.0, 2.0], [0.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="NaN or inf"):
        CurveBound2D.from_function([0, 1], lambda x: np.where(x < 0.5, np.inf, 0.0), 0.0, 1.0)
    with pytest.raises(ValueError, match="repeat"):
        CurveBound2D([1, 1], [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="2 distinct"):
        CurveBound2D([0], [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        CurveBound2D.from_function([0, 1], lambda x: x[:3], 0.0, 1.0)
    flat = CurveBound2D.from_function([0, 1], lambda x: 0.5, 0.0, 1.0, side="below")      # a constant is broadcast
    assert flat(torch.tensor([[0.3, 0.4], [0.3, 0.6]])).tolist() == [True, False] and "below" in repr(flat)
    with pytest.warns(UserWarning, match=r"span \[0, 1\].*'x' ranges over \[-1, 2\]"):
        Prior([-1.0, 0.0], [2.0, 1.0], constraints=[flat], names=["x", "y"])


# ------------------------------------------------------------------ blinding
def test_blinder_offsets_of_an_existing_key(tmp_path):
    # a key file written by hand with a set seed: its offsets and fingerprint must stay what they are
    key = tmp_path / "old.key"
    key.write_text(json.dumps({"seed": 1234567890123, "scales": {"w0": 0.3, "S8": 0.05, "Om": 0.02}}))
    b = Blinder(key)
    assert b.fingerprint == "a557417b16ccc0ac" and b.params == ["Om", "S8", "w0"]
    shift = b.blind(np.zeros((1, 3)), ["Om", "S8", "w0"])[0]
    np.testing.assert_array_equal(shift, [0.014492801374135294, -0.02522984996842431, -0.20683966132455484])


def test_blinder_checks_names(tmp_path):
    make_key(tmp_path / "k", {"S8": 0.05, "w0": 0.3})
    b = Blinder(tmp_path / "k")
    s = np.random.default_rng(0).normal(size=(4, 50, 3))
    with pytest.raises(KeyError, match="'S8' .did you mean 's8'"):
        b.blind(s, ["Om", "s8", "w0"])
    with pytest.raises(KeyError, match="'w0'"):
        b.unblind(s[..., :2], ["Om", "S8"])
    with pytest.raises(ValueError, match="one name per column"):
        b.blind(s, ["S8"])
    with pytest.raises(ValueError, match="one name per column"):
        b.blind(s[..., :1], ["Om", "S8", "w0"])
    with pytest.raises(ValueError, match="unique"):
        b.blind(s, ["S8", "S8", "w0"])
    with pytest.raises(ValueError, match="sequence of parameter names"):
        b.blind(s, "abc")
    with pytest.raises(ValueError, match="missing must be"):
        b.blind(s, ["Om", "S8", "w0"], missing="warn")
    part = b.blind(s[..., :2], ["Om", "S8"], missing="ignore")        # an array with only some blinded parameters
    full = b.blind(s, ["Om", "S8", "w0"])
    np.testing.assert_array_equal(part, full[..., :2])
    np.testing.assert_array_equal(b.blind(s, ["a", "b", "c"], missing="ignore"), s)
    np.testing.assert_allclose(b.unblind(part, ["Om", "S8"], missing="ignore"), s[..., :2])
    t = torch.as_tensor(s, dtype=torch.float32).requires_grad_()       # tensors in, numpy out, dtype kept
    out = b.blind(t, ["Om", "S8", "w0"])
    assert isinstance(out, np.ndarray) and out.dtype == np.float32 and np.allclose(out, full, atol=1e-6)
    assert b.blind(np.zeros((2, 3), dtype=int), ["Om", "S8", "w0"]).dtype == np.float64
    np.testing.assert_array_equal(s, np.random.default_rng(0).normal(size=(4, 50, 3)))      # the input is not changed


def test_make_key_validation(tmp_path):
    key = tmp_path / "k"
    for bad, error in ((["S8", "w0"], TypeError), ({}, ValueError), ({"S8": 0.0}, ValueError),
                       ({"S8": -0.1}, ValueError),
                       ({"S8": np.nan}, ValueError), ({"S8": np.inf}, ValueError), ({"S8": "0.05"}, TypeError),
                       ({3: 0.05}, TypeError), ({"S8": True}, TypeError)):
        with pytest.raises(error, match="scale"):
            make_key(key, bad)
        assert not key.exists()                                        # nothing is written for a bad key
    fp = make_key(key, {"S8": np.float32(0.05), "w0": 1})
    assert Blinder(key).fingerprint == fp
    assert json.loads(key.read_text())["scales"] == {"S8": float(np.float32(0.05)), "w0": 1.0}
    (tmp_path / "other.json").write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="not a blinding key"):
        Blinder(tmp_path / "other.json")
    (tmp_path / "text").write_text("hello")
    with pytest.raises(ValueError, match="not a blinding key"):
        Blinder(tmp_path / "text")


def test_read_only_arrays_give_no_warning():
    def ro(a):
        a = np.array(a, dtype=np.float32)
        a.setflags(write=False)
        return a

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prior = Prior(ro([0, 0]), ro([1, 1]), names=["a", "b"],
                      terms=[Gaussian(["a"], ro([0.5]), ro([[0.04]]))],
                      constraints=[Polygon2D(["a", "b"], ro([[0, 0], [1, 0], [1, 1], [0, 1]])),
                                   CurveBound2D.from_function(["a", "b"], lambda a: a - 2, 0.0, 1.0)])
        assert np.isfinite(prior.log_prob(ro([[0.5, 0.5]]))).all()
        assert np.isfinite(Prior(ro([0, 0]), ro([1, 1]), terms=[KDE([0, 1], ro(np.random.default_rng(0).uniform(0, 1, (200, 2))),
                                                                bandwidth=0.3)])
                           .log_prob(ro([[0.5, 0.5]]))).all()
