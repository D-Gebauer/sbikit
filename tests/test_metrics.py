import numpy as np
import pytest

from sbikit import metrics


def _gaussian_posteriors(n_sims, n_samples, P, width_factor, seed=0):
    """Truth ~ N(0, 1), data d = truth + N(0, 1): the exact posterior is N(d / 2, 1 / 2).
    Samples have width width_factor * sqrt(1/2): calibrated for 1, overconfident below."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n_sims, P))
    d = truth + rng.normal(size=(n_sims, P))
    samples = d[:, None] / 2 + width_factor * np.sqrt(0.5) * rng.normal(size=(n_sims, n_samples, P))
    return samples, truth


def test_tarp_matches_package(device):
    tarp = pytest.importorskip("tarp")
    samples, truth = _gaussian_posteriors(300, 500, 3, 1.0)
    for seed in (0, 7):
        f = metrics.tarp_f(samples, truth, n_references=1, seed0=seed, device=device)
        alpha, ecp = metrics.ecp_curve(f)
        ecp_pkg, alpha_pkg = tarp.get_tarp_coverage(samples.transpose(1, 0, 2), truth, norm=True, seed=seed)
        np.testing.assert_allclose(alpha, alpha_pkg)
        np.testing.assert_allclose(ecp, ecp_pkg, atol=1e-12)


def test_tarp_scalars_detect_overconfidence():
    cal = metrics.tarp_scalars(metrics.tarp_f(*_gaussian_posteriors(400, 400, 2, 1.0), n_references=10))
    over = metrics.tarp_scalars(metrics.tarp_f(*_gaussian_posteriors(400, 400, 2, 0.5), n_references=10))
    assert abs(cal["bias"]) < 3 * cal["bias_sd"] + 0.02 and abs(cal["confidence"]) < 3 * cal["confidence_sd"] + 0.02
    assert over["confidence"] < -0.1


def test_marginal_coverage_and_fom():
    samples, truth = _gaussian_posteriors(2000, 400, 2, 1.0)
    mc = metrics.marginal_coverage(samples, truth)
    assert np.allclose(mc["cov68"], 0.68, atol=0.03) and np.allclose(mc["cov95"], 0.95, atol=0.015)
    lf = metrics.log_fom(samples, [0, 1])
    assert np.isclose(np.mean(lf), np.log(2.0), atol=0.02)    # Cov = I / 2: -1/2 log det = log 2


def test_evaluate_record(device):
    samples, truth = _gaussian_posteriors(200, 300, 3, 1.0)
    rec, arr = metrics.evaluate(samples, truth, ["a", "b", "c"], {"ab": ["a", "b"]}, {"ab": ["a", "b"]},
                                n_references=5, n_bootstrap=20, device=device)
    assert set(rec["tarp"]) == {"all", "ab"} and "ab" in rec["fom"] and "cov68" in rec["marginal"]["a"]
    assert arr["tarp_ecp_all"].shape == arr["tarp_alpha"].shape
