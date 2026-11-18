"""Posterior maxima and goodness of fit on a linear-Gaussian model, where everything is known."""
import numpy as np
import pytest
import torch
from scipy import stats

import sbikit as sk
from sbikit import NLE, Ensemble, Prior, find_map
from sbikit.gof import calibrated_pvalue, likelihood_chi2, predictive_statistic

from conftest import GaussianLikelihood, simulate  # isort: skip


class Likelihood(GaussianLikelihood):
    """The exact likelihood with the ``sample`` method of an NLE."""

    def sample(self, theta, n=1, seed=None):
        mu = np.asarray(theta) @ self.A.cpu().numpy().T.astype(np.float64) + self.b.cpu().numpy()
        return mu + self.sigma * np.random.default_rng(seed).normal(size=(n,) + mu.shape)


@pytest.fixture
def model(linear_model, device):
    m = dict(linear_model)
    m["lik"] = Likelihood(m["A"], m["b"], m["sigma"], device)
    m["cov"] = np.linalg.inv(m["A"].T @ m["A"] / m["sigma"] ** 2)       # posterior covariance for a flat prior
    return m


def _mean(model, x):
    return (model["cov"] @ (model["A"].T @ (x - model["b"]).T / model["sigma"] ** 2)).T


def _posteriors(model, x, n, rng):
    """(B, n, P) samples of the analytic posteriors of the rows of ``x``."""
    z = rng.normal(size=(len(x), n, model["P"]))
    return _mean(model, x)[:, None] + z @ np.linalg.cholesky(model["cov"]).T


def _sims(model, n, rng):
    truth = rng.uniform(-0.5, 0.5, size=(n, model["P"]))
    return truth, simulate(model, truth, rng)


# ------------------------------------------------------------------------------ find_map
def test_find_map_matches_analytic(model):
    rng = np.random.default_rng(0)
    truth, x = _sims(model, 5, rng)
    sd, prior = np.sqrt(model["cov"].diagonal()), Prior([-2, -2], [2, 2], names=["a", "b"])
    samples = _posteriors(model, x, 200, rng)
    theta, logp = find_map(model["lik"], x, prior, samples)
    assert theta.shape == (5, 2) and logp.shape == (5,)
    np.testing.assert_allclose(theta, _mean(model, x), atol=0.05 * sd.max())
    at_samples = np.stack([model["lik"].log_prob(torch.as_tensor(np.repeat(x[b:b + 1], 200, 0), dtype=torch.float32,
                                                                 device=model["lik"].device)[None],
                                                 torch.as_tensor(samples[b], dtype=torch.float32,
                                                                 device=model["lik"].device))[0].cpu().numpy()
                           for b in range(5)])
    assert (logp >= at_samples.max(1) + prior.log_prob(theta[0]) - 1e-3).all()

    for scale in (None, sd):                                               # one start point per target
        one, _ = find_map(model["lik"], x, prior, truth, scale=scale)
        np.testing.assert_allclose(one, _mean(model, x), atol=0.05 * sd.max())
    single, _ = find_map(model["lik"], x[0], prior, samples[0])            # (S, P) for a single target
    np.testing.assert_allclose(single, theta[:1], atol=1e-6)


def test_find_map_on_the_box_edge_with_per_target_priors(model):
    rng = np.random.default_rng(1)
    _, x = _sims(model, 1, rng)
    x = np.repeat(x, 2, 0)
    mean, cov = _mean(model, x)[0], model["cov"]
    sd = np.sqrt(cov.diagonal())
    edge = mean[0] - 3 * sd[0]                                              # the maximum in a lies above the box
    priors = [Prior([-2, -2], [2, 2]), Prior([-2, -2], [edge, 2])]
    samples = _posteriors(model, x, 200, rng)
    samples[1] = np.minimum(samples[1], [edge, 2.0])
    theta, logp = find_map(model["lik"], x, priors, samples)
    assert theta[1, 0] == np.float32(edge)
    np.testing.assert_allclose(theta[1, 1], mean[1] + cov[1, 0] / cov[0, 0] * (edge - mean[0]), atol=0.05 * sd[1])
    assert logp[0] > logp[1]


def test_find_map_with_a_transform(model):
    rng = np.random.default_rng(2)
    _, x = _sims(model, 3, rng)
    samples = _posteriors(model, x, 100, rng) / 2                           # psi = theta / 2
    psi, _ = find_map(model["lik"], x, Prior([-1, -1], [1, 1]), samples, transform=lambda psi: 2 * psi)
    np.testing.assert_allclose(2 * psi, _mean(model, x), atol=0.05 * np.sqrt(model["cov"].diagonal()).max())


def test_find_map_input_checks(model):
    x, prior = np.zeros((3, model["D"])), Prior([-2, -2], [2, 2])
    with pytest.raises(ValueError, match="init must have shape"):
        find_map(model["lik"], x, prior, np.zeros((4, 5, 2)))
    with pytest.raises(ValueError, match="NaN or inf"):
        find_map(model["lik"], x, prior, np.full((3, 2), np.nan))
    outside = Prior([-2, -2], [2, 2], constraints=[lambda theta: theta[..., 0] > 1])
    with pytest.raises(ValueError, match="no start point"):
        find_map(model["lik"], x, outside, np.zeros((3, 4, 2)), scale=0.1, steps=5)
    edge, _ = find_map(model["lik"], x, prior, np.full((3, 2), 3.0), steps=0)   # moved to the edge of the box
    np.testing.assert_array_equal(edge, 2.0)
    with pytest.raises(ValueError, match="scale is needed"):
        find_map(model["lik"], x, Prior([-2, -2], [2, np.inf], terms=[sk.Gaussian([1], [0.0], [1.0])]),
                 np.zeros((3, 2)))
    with pytest.raises(ValueError, match="n_starts"):
        find_map(model["lik"], x, prior, np.zeros((3, 2)), n_starts=0)


# ------------------------------------------------------------------------------ likelihood_chi2
def test_likelihood_chi2_follows_chi2(model):
    rng = np.random.default_rng(3)
    truth, x = _sims(model, 2000, rng)
    chi2 = likelihood_chi2(model["lik"], x, truth, n_draws=500)
    assert chi2.shape == (2000,)
    assert stats.kstest(chi2, "chi2", args=(model["D"],)).pvalue > 1e-3
    np.testing.assert_array_equal(chi2, likelihood_chi2(model["lik"], x, truth, n_draws=500))
    assert not np.array_equal(chi2, likelihood_chi2(model["lik"], x, truth, n_draws=500, seed=1))

    sub = likelihood_chi2(model["lik"], x, truth, n_draws=500, columns=[0, 2])
    assert stats.kstest(sub, "chi2", args=(2,)).pvalue > 1e-3

    wide = truth @ model["A"].T + model["b"] + 1.5 * model["sigma"] * rng.normal(size=x.shape)
    assert likelihood_chi2(model["lik"], wide, truth, n_draws=500).mean() == pytest.approx(2.25 * model["D"], rel=0.1)


def test_likelihood_chi2_at_one_point(model):
    rng = np.random.default_rng(4)
    point = np.array([0.1, -0.2])
    x = simulate(model, np.repeat(point[None], 1000, 0), rng)
    chi2 = likelihood_chi2(model["lik"], x, point)
    assert stats.kstest(chi2, "chi2", args=(model["D"],)).pvalue > 1e-3
    assert likelihood_chi2(model["lik"], x[0], point).shape == (1,)
    assert likelihood_chi2(model["lik"], x[0], np.repeat(point[None], 7, 0)).shape == (7,)


def test_likelihood_chi2_input_checks(model, linear_model):
    x, theta = np.zeros((6, model["D"])), np.zeros((6, 2))
    with pytest.raises(ValueError, match="n_draws"):
        likelihood_chi2(model["lik"], x, theta, n_draws=model["D"] + 2)
    for columns in ([0, 0], [model["D"]], [], [0.5]):
        with pytest.raises(ValueError, match="columns"):
            likelihood_chi2(model["lik"], x, theta, columns=columns)
    with pytest.raises(ValueError, match="rows"):
        likelihood_chi2(model["lik"], x, theta[:4])
    with pytest.raises(ValueError, match="NaN or inf"):
        likelihood_chi2(model["lik"], np.full_like(x, np.nan), theta)
    with pytest.raises(TypeError, match="sample"):
        likelihood_chi2(linear_model["lik"], x, theta)


# ------------------------------------------------------------------------------ predictive tests
def test_predictive_statistic_is_the_mean_log_likelihood(linear_model):
    rng = np.random.default_rng(5)
    _, x = _sims(linear_model, 3, rng)
    samples = rng.uniform(-0.5, 0.5, size=(3, 50, 2))
    t = predictive_statistic(linear_model["lik"], x, samples)              # a likelihood with log_prob only
    mu = samples @ linear_model["A"].T + linear_model["b"]
    expected = (((x[:, None] - mu) / linear_model["sigma"]) ** 2).sum(-1).mean(1)
    np.testing.assert_allclose(t, expected, rtol=1e-4)
    np.testing.assert_allclose(predictive_statistic(linear_model["lik"], x[0], samples[0]), t[:1], rtol=1e-6)


def test_predictive_statistic_of_columns_is_the_gaussian_predictive(model):
    rng = np.random.default_rng(6)
    _, x = _sims(model, 4, rng)
    samples = _posteriors(model, x, 4000, rng)
    A, mean = model["A"], _mean(model, x) @ model["A"].T + model["b"]
    cov = model["sigma"] ** 2 * np.eye(model["D"]) + A @ model["cov"] @ A.T
    for columns in (None, [0, 2]):
        cols = np.arange(model["D"]) if columns is None else np.array(columns)
        r, c = (x - mean)[:, cols], cov[np.ix_(cols, cols)]
        expected = np.einsum("bi,bi->b", r, np.linalg.solve(c, r.T).T) + np.linalg.slogdet(2 * np.pi * c)[1]
        t = predictive_statistic(model["lik"], x, samples, columns=list(cols))
        np.testing.assert_allclose(t, expected, atol=0.5)


@pytest.mark.parametrize("columns", [None, [0, 1, 2, 3]])
def test_calibrated_predictive_test(model, columns):
    rng = np.random.default_rng(7)
    (_, x_ref), (_, x_new) = _sims(model, 300, rng), _sims(model, 300, rng)
    t_ref = predictive_statistic(model["lik"], x_ref, _posteriors(model, x_ref, 200, rng), columns)
    t_new = predictive_statistic(model["lik"], x_new, _posteriors(model, x_new, 200, rng), columns, seed=1)
    p = calibrated_pvalue(t_new, t_ref)
    assert p.shape == (300,) and stats.kstest(p, "uniform").pvalue > 1e-3

    off = np.linalg.svd(model["A"])[0][:, -1]                               # a shift no parameter can absorb
    x_bad = x_new[:1] + 10 * model["sigma"] * off
    t_bad = predictive_statistic(model["lik"], x_bad, _posteriors(model, x_bad, 200, rng), columns)
    assert calibrated_pvalue(t_bad[0], t_ref) == pytest.approx(1 / 301)


def test_predictive_statistic_input_checks(model, linear_model):
    x = np.zeros((3, model["D"]))
    with pytest.raises(ValueError, match="samples must be"):
        predictive_statistic(model["lik"], x, np.zeros((2, 10, 2)))
    with pytest.raises(ValueError, match="NaN or inf"):
        predictive_statistic(model["lik"], x, np.full((3, 10, 2), np.inf))
    with pytest.raises(ValueError, match="posterior samples"):
        predictive_statistic(model["lik"], x, np.zeros((3, 4, 2)), columns=[0, 1, 2])
    with pytest.raises(TypeError, match="sample"):
        predictive_statistic(linear_model["lik"], x, np.zeros((3, 10, 2)), columns=[0])


def test_calibrated_pvalue():
    reference = [1.0, 2.0, 3.0, 4.0]
    assert calibrated_pvalue(2.5, reference) == 0.6 and isinstance(calibrated_pvalue(2.5, reference), float)
    np.testing.assert_allclose(calibrated_pvalue([[0.0, 3.0], [5.0, np.inf]], reference), [[1.0, 0.6], [0.2, 0.2]])
    for bad in ((np.nan, reference), (1.0, [1.0, np.nan]), (1.0, [])):
        with pytest.raises(ValueError):
            calibrated_pvalue(*bad)


# ------------------------------------------------------------------------------ with flows
class _Constant:
    """Stands in for a fitted member whose every draw is ``value``."""

    def __init__(self, value):
        self.value = value

    def log_prob(self, input, condition):
        raise NotImplementedError

    def sample(self, theta, n=1, seed=None):
        return np.full((n, len(theta), 3), self.value)


def test_ensemble_sample_follows_the_weights():
    draws = Ensemble([_Constant(0.0), _Constant(1.0)], weights=[0.25, 0.75]).sample(np.zeros((40, 2)), n=500, seed=0)
    assert draws.shape == (500, 40, 3)
    assert (draws[..., 0] == draws[..., 1]).all()                          # one member per draw
    assert draws.mean() == pytest.approx(0.75, abs=0.02)
    assert 0.6 < draws[0].mean() < 0.9 and 0.5 < draws[:, 0].mean() < 1.0  # mixed along both axes


def test_with_trained_flows(device):
    rng = np.random.default_rng(8)
    theta = rng.uniform(-1, 1, size=(600, 2))
    s = theta @ rng.normal(size=(2, 3)) + 0.3 * rng.normal(size=(600, 3))
    members = [NLE(flow={"model": "maf", "hidden_features": 16, "num_transforms": 2}, device=device, seed=k)
               for k in range(2)]
    for member in members:
        member.fit(s[:500], theta[:500], s[500:], theta[500:], {"max_epochs": 20, "batch_size": 100})
    prior = Prior([-1, -1], [1, 1])
    for likelihood in (members[0], Ensemble(members)):
        draws = likelihood.sample(theta[500:], n=5, seed=0)
        assert draws.shape == (5, 100, 3)
        np.testing.assert_array_equal(draws, likelihood.sample(theta[500:], n=5, seed=0))
        chi2 = likelihood_chi2(likelihood, s[500:], theta[500:], n_draws=200)
        assert chi2.shape == (100,) and np.isfinite(chi2).all() and 1 < np.median(chi2) < 6
        samples = theta[500:520, None] + 0.1 * rng.normal(size=(20, 30, 2))
        for columns in (None, [0, 1]):
            assert np.isfinite(predictive_statistic(likelihood, s[500:520], samples, columns)).all()
        start = np.clip(samples, -1, 1)
        for member in members:
            member.flow.zero_grad()
        point, logp = find_map(likelihood, s[500:520], prior, start, steps=50)
        at_start = likelihood.log_likelihood(np.repeat(s[500:520], 30, 0), start.reshape(-1, 2)).reshape(20, 30)
        assert (logp >= at_start.max(1).values.cpu().numpy() + prior.log_prob(point[0]) - 1e-3).all()
        assert all(p.grad is None for member in members for p in member.flow.parameters())
