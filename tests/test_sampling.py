import numpy as np
import pytest

from sbikit import MCMCConfig, Prior, sample_posteriors


def _analytic(model, x):
    """Posterior mean/cov for a flat prior (the box is wide enough not to truncate)."""
    A, s = model["A"], model["sigma"]
    cov = np.linalg.inv(A.T @ A / s ** 2)
    mean = (cov @ (A.T @ (x - model["b"]).T / s ** 2)).T
    return mean, cov


def _check(post, model, x):
    mean, cov = _analytic(model, x)
    np.testing.assert_allclose(post.samples.mean(1), mean, atol=0.15 * np.sqrt(cov.diagonal()).max())
    np.testing.assert_allclose(post.samples.std(1).mean(0), np.sqrt(cov.diagonal()), rtol=0.1)


def test_batched_posteriors_match_analytic(linear_model):
    from conftest import simulate
    rng = np.random.default_rng(2)
    truth = rng.uniform(-0.5, 0.5, size=(6, 2))
    x = simulate(linear_model, truth, rng)
    prior = Prior([-2, -2], [2, 2])
    for moves, logit in (("de_snooker", True), ("stretch", False)):
        cfg = MCMCConfig(nwalkers=64, moves=moves, logit_box=logit, burn_in=300, block=300, max_steps=6000,
                         thin=2, min_ess=500, n_samples=3000, seed=1)
        post = sample_posteriors(linear_model["lik"], x, prior, cfg, init_centre=truth, init_sd=[0.05, 0.05], verbose=False)
        assert post.samples.shape == (6, 3000, 2)
        assert post.converged.all(), post.diagnostics
        _check(post, linear_model, x)


class _Flat:
    def __init__(self, device):
        self.device = device

    def log_prob(self, input, condition):
        import torch
        return torch.zeros(1, len(condition), device=condition.device)


@pytest.mark.parametrize("moves,logit", [
    ("de_snooker", True), ("de_snooker", False), ("de", True), ("stretch", True), ("walk", True), ("kde", True),
    ([["snooker", 1.0, {}]], True), ([["gaussian", 1.0, {"cov": 0.3}]], True),
    ([["de", 0.6, {}], ["snooker", 0.2, {}], ["stretch", 0.2, {}]], True)])
def test_flat_box_width_exact(moves, logit, device):
    """Posterior = prior = flat 9-d box: sd must be 1/sqrt(12) per parameter. Catches move biases
    (emcee's / torchemcee 0.1.0's snooker move gives x0.97 in the DE/snooker mix, x0.84 alone)."""
    P = 9
    cfg = MCMCConfig(nwalkers=256, moves=moves, logit_box=logit, burn_in=2000, block=1000, max_steps=8000,
                     thin=4, adaptive=False, n_samples=100000, seed=0)
    post = sample_posteriors(_Flat(device), np.zeros((4, 1)), Prior(np.zeros(P), np.ones(P)), cfg,
                             init_centre=np.full((4, P), 0.5), init_sd=np.full(P, 0.3), verbose=False)
    ratio = (post.samples.std(1) * np.sqrt(12)).mean()
    assert abs(ratio - 1) < 0.015, ratio


def test_per_target_priors_and_prior_init(linear_model):
    from conftest import simulate
    rng = np.random.default_rng(3)
    truth = np.array([[0.2, -0.1], [0.0, 0.3]])
    x = simulate(linear_model, truth, rng)
    priors = [Prior([-2, -2], [2, 2]), Prior([-1, -1], [1, 1])]
    cfg = MCMCConfig(nwalkers=64, init="prior", burn_in=500, block=300, max_steps=6000, thin=2, min_ess=500, seed=4)
    post = sample_posteriors(linear_model["lik"], x, priors, cfg, verbose=False)
    assert post.converged.all()
    _check(post, linear_model, x)
    assert (np.abs(post.samples[1]) <= 1).all()
