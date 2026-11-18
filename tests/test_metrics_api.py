"""Input validation, options and analytic Gaussian cases of sbikit.metrics."""
import json
import warnings

import numpy as np
import pytest
import torch

from sbikit import metrics

NAMES = ["a", "b", "c"]


def _gaussian_posteriors(n_sims, n_samples, P, width_factor=1.0, shift=0.0, seed=0):
    """Truth ~ N(0, 1), data d = truth + N(0, 1): the exact posterior is N(d / 2, 1 / 2).
    Samples have width width_factor * sqrt(1/2) and are shifted by ``shift`` posterior sigmas."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(n_sims, P))
    d = truth + rng.normal(size=(n_sims, P))
    sd = np.sqrt(0.5)
    samples = d[:, None] / 2 + shift * sd + width_factor * sd * rng.normal(size=(n_sims, n_samples, P))
    return samples, truth


@pytest.fixture(scope="module")
def calibrated():
    return _gaussian_posteriors(300, 300, 3)


# ------------------------------------------------------------------------------ analytic cases
def test_calibrated_record(calibrated, device):
    rec, arr = metrics.evaluate(*calibrated, NAMES, {"ab": ["a", "b"]}, {"ab": ["a", "b"]}, n_references=10,
                                n_bootstrap=50, device=device)
    for key in ("all", "ab"):
        t = rec["tarp"][key]
        assert abs(t["bias"]) < 3 * t["bias_sd"] + 0.02 and abs(t["confidence"]) < 3 * t["confidence_sd"] + 0.02
        assert t["max_dev"] < 0.1
    for name in NAMES:
        m = rec["marginal"][name]
        assert abs(m["cov68"] - 0.68) < 3 * m["cov68_sd"] and abs(m["cov95"] - 0.95) < 3 * m["cov95_sd"]
        assert m["ks_pvalue"] > 0.01
        assert abs(rec["z_mean"][name]) < 4 * rec["z_mean_sd"][name] and abs(rec["z_rms"][name] - 1) < 0.15
        assert rec["z_n_nan"][name] == 0
    assert np.isclose(rec["fom"]["ab"]["log_fom_mean"], np.log(2.0), atol=0.03)       # Cov = I / 2
    assert rec["n_sims"] == 300 and rec["n_dropped"] == 0 and rec["ess"] is None and rec["extra"] == {}
    assert arr["kept"].all() and arr["post_mean"].shape == (300, 3) and arr["log_fom_ab"].shape == (300,)


def test_signs_of_miscalibration():
    kw = dict(n_references=10, n_bootstrap=20)
    over, _ = metrics.evaluate(*_gaussian_posteriors(300, 300, 2, width_factor=0.5), ["a", "b"], **kw)
    under, _ = metrics.evaluate(*_gaussian_posteriors(300, 300, 2, width_factor=2.0), ["a", "b"], **kw)
    assert over["tarp"]["all"]["confidence"] < -0.2 and under["tarp"]["all"]["confidence"] > 0.2
    assert over["marginal"]["a"]["cov68"] < 0.5 and under["marginal"]["a"]["cov68"] > 0.85
    assert over["z_rms"]["a"] > 1.5 and under["z_rms"]["a"] < 0.7
    assert over["marginal"]["a"]["ks_pvalue"] < 1e-3 and under["marginal"]["a"]["ks_pvalue"] < 1e-3
    assert over["tarp"]["all"]["max_dev"] > 0.1 and under["tarp"]["all"]["max_dev"] > 0.1


def test_shifted_posteriors_show_in_z_mean():
    for shift in (1.0, -1.0):
        rec, _ = metrics.evaluate(*_gaussian_posteriors(300, 300, 2, shift=shift), ["a", "b"], n_references=10,
                                  n_bootstrap=20)
        for name in ("a", "b"):
            assert abs(rec["z_mean"][name] - shift) < 0.25 and abs(rec["z_mean"][name]) > 5 * rec["z_mean_sd"][name]
        assert rec["tarp"]["all"]["confidence"] < -0.1       # TARP sees a shift as overconfidence, not as bias


def test_tarp_scalars_definitions():
    f = np.random.default_rng(0).uniform(size=(4, 500))
    s = metrics.tarp_scalars(f, n_bootstrap=50, tol_min=0.02)
    assert np.isclose(s["bias"], 1 - 2 * f.mean()) and np.isclose(s["confidence"], 1 - 4 * np.abs(f - 0.5).mean())
    assert s["tol_bias"] == max(2 * s["bias_sd"], 0.02) and s["n_sims"] == 500 and s["n_references"] == 4
    alpha = np.linspace(0, 1, 201)
    ecp = np.mean([(fr[:, None] < alpha).mean(0) for fr in f], 0)
    assert np.isclose(s["max_dev"], np.abs(ecp - alpha).max())
    none = metrics.tarp_scalars(f, n_bootstrap=0, tol_min=0.03)
    assert np.isnan(none["bias_sd"]) and none["tol_bias"] == 0.03 and none["bias"] == s["bias"]
    assert metrics.tarp_scalars(f, seed=3) == metrics.tarp_scalars(f, seed=3)


def test_max_dev_small_sets():
    samples, truth = _gaussian_posteriors(15, 300, 2, width_factor=0.3, seed=2)
    assert metrics.tarp_scalars(metrics.tarp_f(samples, truth, 20))["max_dev"] > 0.15
    # symmetric miscalibration: every f at 0 or 1 has zero bias but the largest possible deviation
    assert np.isclose(metrics.tarp_scalars(np.array([0.0, 1.0] * 6), 0)["max_dev"], 0.5, atol=0.01)


def test_mid_ranks_remove_the_finite_sample_offset():
    n = 10
    f = np.arange(n + 1)[None] / n                       # what calibrated posteriors with n samples give
    assert np.isclose(metrics.tarp_scalars(f, 0)["confidence"], -1 / (n + 1))
    mid = metrics.tarp_scalars(f, 0, n_samples=n)
    assert np.isclose(mid["confidence"], 1 / (n + 1) ** 2) and abs(mid["bias"]) < 1e-12
    samples, truth = _gaussian_posteriors(4000, 49, 1)
    mc = metrics.marginal_coverage(samples, truth)
    assert abs(mc["cov68"][0] - 0.68) < 0.03 and abs(mc["cov95"][0] - 0.95) < 0.02 and mc["ks"][0] < 0.04


# ------------------------------------------------------------------------------ TARP options
def test_constant_truth_needs_bounds(calibrated):
    samples, truth = calibrated
    fixed_t, fixed_s = truth.copy(), samples.copy()
    fixed_t[:, 2], fixed_s[:, :, 2] = 0.3, 0.3           # a parameter that is fixed everywhere
    with pytest.raises(ValueError, match="bounds"):
        metrics.tarp_f(fixed_s, fixed_t)
    with pytest.raises(ValueError, match=r"\['c'\].*bounds"):
        metrics.evaluate(samples, fixed_t, NAMES)
    with pytest.raises(ValueError, match="bounds"):
        metrics.tarp_f(samples[:1], truth[:1])           # a single simulation has no range either
    s = metrics.tarp_scalars(metrics.tarp_f(fixed_s, fixed_t, 10, bounds=([-4, -4, 0], [4, 4, 1])), 50, n_samples=300)
    assert abs(s["bias"]) < 3 * s["bias_sd"] + 0.02 and abs(s["confidence"]) < 3 * s["confidence_sd"] + 0.02
    # fixed truth, free posterior: narrow posteriors around the truth are close to calibrated, not bias = +1
    rng = np.random.default_rng(1)
    t = rng.uniform(0.2, 0.8, size=(300, 2))
    t[:, 1] = 0.4
    post = (t + 0.01 * rng.normal(size=t.shape))[:, None] + 0.01 * rng.normal(size=(300, 200, 2))
    s = metrics.tarp_scalars(metrics.tarp_f(post, t, 10, bounds=(0, 1)), 20, n_samples=200)
    assert abs(s["bias"]) < 0.1 and abs(s["confidence"]) < 0.1
    rec, _ = metrics.evaluate(post[:1], t[:1], ["a", "b"], bounds=(0, 1), n_references=3)
    assert rec["n_sims"] == 1 and json.dumps(rec, allow_nan=False)


def test_bounds_and_references(calibrated):
    samples, truth = calibrated
    default = metrics.tarp_f(samples, truth, 3, seed0=4)
    np.testing.assert_allclose(metrics.tarp_f(samples, truth, 3, seed0=4, bounds=(truth.min(0), truth.max(0))), default)
    box = (torch.full((3,), -5.0), [5.0, 5.0, 5.0])
    refs = np.stack([np.random.RandomState(4 + r).uniform(-5, 5, size=truth.shape) for r in range(3)])
    f = metrics.tarp_f(samples, truth, 3, seed0=4, bounds=box)
    np.testing.assert_allclose(metrics.tarp_f(samples, truth, bounds=box, references=refs), f)
    assert metrics.tarp_f(samples, truth, bounds=box, references=refs[0]).shape == (1, 300)
    for bad in (([0, 0], [1, 1]), ([0, 0, 0], [1, 1, 0]), 3.0, ([0, 0, 0], [1, 1, np.inf])):
        with pytest.raises(ValueError, match="bounds"):
            metrics.tarp_f(samples, truth, bounds=bad)
    with pytest.raises(ValueError, match="references"):
        metrics.tarp_f(samples, truth, references=np.zeros((2, 5, 3)))
    with pytest.raises(ValueError, match="n_references"):
        metrics.tarp_f(samples, truth, n_references=0)


def test_small_test_sets_need_bounds():
    bias = {"default": [], "bounds": []}
    for seed in range(20):
        samples, truth = _gaussian_posteriors(3, 300, 3, seed=seed)
        with pytest.warns(RuntimeWarning, match="only 3 simulations.*bounds"):
            f = metrics.tarp_f(samples, truth, 20)
        bias["default"].append(metrics.tarp_scalars(f, 0, n_samples=300)["bias"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            f = metrics.tarp_f(samples, truth, 20, bounds=(-4, 4))
        bias["bounds"].append(metrics.tarp_scalars(f, 0, n_samples=300)["bias"])
    assert np.mean(bias["default"]) > 0.25 and abs(np.mean(bias["bounds"])) < 0.15
    samples, truth = _gaussian_posteriors(10, 50, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics.tarp_f(samples, truth, 2)                                    # ten simulations: no warning
    with pytest.warns(RuntimeWarning, match="only 9 simulations") as caught:
        metrics.evaluate(samples[:9], truth[:9], NAMES, None, {"ab": ["a", "b"]}, n_references=2, n_bootstrap=0)
    assert len(caught) == 1                                                  # once, not once per TARP set
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics.evaluate(samples[:9], truth[:9], NAMES, bounds=(-4, 4), n_references=2, n_bootstrap=0)


def test_tarp_device_fallback(calibrated):
    samples, truth = calibrated
    f = metrics.tarp_f(samples, truth, 2)
    assert np.array_equal(metrics.tarp_f(samples, truth, 2, device="mps"), f)     # float64: runs on the CPU
    assert np.array_equal(metrics.tarp_f(samples, truth, 2, chunk=7), f)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            metrics.tarp_f(samples, truth, 2, device="cuda")


# ------------------------------------------------------------------------------ input validation
def test_shape_errors(calibrated):
    samples, truth = calibrated
    calls = [lambda: metrics.tarp_f(samples[0], truth),                       # single posterior, many truths
             lambda: metrics.tarp_f(samples, truth[0]),
             lambda: metrics.marginal_coverage(samples, truth[:100]),
             lambda: metrics.marginal_coverage(samples, truth[:, :2]),
             lambda: metrics.posterior_summary(samples[:, :200, 0], truth[:, 0]),  # lost parameter axis
             lambda: metrics.posterior_summary(samples[..., :1], truth[:, 0]),
             lambda: metrics.posterior_summary(samples[0, :, 0]),
             lambda: metrics.log_fom(samples[0, :, 0], [0]),
             lambda: metrics.evaluate(samples[:, :200].transpose(1, 0, 2), truth, NAMES),   # tarp axis order
             lambda: metrics.evaluate(samples[:, :0], truth, NAMES),
             lambda: metrics.importance_weights(samples[0, 0], None, None)]
    for call in calls:
        with pytest.raises(ValueError, match=r"\(n_sims, n_samples, P\)"):
            call()
    for bad in ([0.5, np.nan], [0.5, 1.5], np.zeros((2, 3, 4))):
        with pytest.raises(ValueError, match="tarp_f"):
            metrics.tarp_scalars(bad)
        with pytest.raises(ValueError, match="tarp_f"):
            metrics.ecp_curve(bad)


def test_single_posterior_and_tensors(calibrated):
    samples, truth = calibrated
    one = metrics.posterior_summary(samples[0], truth[0])
    full = metrics.posterior_summary(samples, truth)
    assert one["z"].shape == (1, 3) and all(np.array_equal(one[k][0], full[k][0]) for k in full)
    assert np.array_equal(metrics.log_fom(samples[0], [0, 1]), metrics.log_fom(samples, [0, 1])[:1])
    assert metrics.marginal_coverage(samples[0], truth[0])["cov95"] == metrics.marginal_coverage(samples[:1],
                                                                                              truth[:1])["cov95"]
    ts, tt = torch.tensor(samples, dtype=torch.float32, requires_grad=True), torch.tensor(truth, dtype=torch.float32)
    rec_t, _ = metrics.evaluate(ts, tt, NAMES, {"ab": ["a", "b"]}, n_references=3, n_bootstrap=5)
    rec_n, _ = metrics.evaluate(ts.detach().numpy(), tt.numpy(), tuple(NAMES), {"ab": ("a", "b")}, n_references=3,
                                n_bootstrap=5)
    assert rec_t == rec_n


def test_nonfinite_raises_or_drops(calibrated):
    samples, truth = calibrated
    bad_s, bad_t = samples.copy(), truth.copy()
    bad_s[3, 5, 1], bad_s[7, 0, 0], bad_t[9, 2] = np.nan, np.inf, np.nan
    for call in (lambda: metrics.tarp_f(bad_s, bad_t), lambda: metrics.marginal_coverage(bad_s, truth),
                 lambda: metrics.posterior_summary(bad_s), lambda: metrics.log_fom(bad_s, [0]),
                 lambda: metrics.evaluate(bad_s, bad_t, NAMES)):
        with pytest.raises(ValueError, match="NaN or inf in the samples of 2"):
            call()
    with pytest.raises(ValueError, match="truths of 1 of 300"):
        metrics.evaluate(samples, bad_t, NAMES)
    kw = dict(fom_sets={"ab": ["a", "b"]}, n_references=3, n_bootstrap=5)
    with pytest.warns(RuntimeWarning, match="dropped 3 of 300"):
        rec, arr = metrics.evaluate(bad_s, bad_t, NAMES, drop_nonfinite=True, **kw)
    keep = np.ones(300, bool)
    keep[[3, 7, 9]] = False
    ref, ref_arr = metrics.evaluate(samples[keep], truth[keep], NAMES, **kw)
    assert rec["n_dropped"] == 3 and rec["n_sims"] == 297 and np.array_equal(arr["kept"], keep)
    assert {**rec, "n_dropped": 0} == ref and np.array_equal(arr["post_mean"], ref_arr["post_mean"])
    with pytest.raises(ValueError, match="all 300"):
        metrics.evaluate(samples * np.nan, truth, NAMES, drop_nonfinite=True)
    # the weights of dropped simulations are not looked at; those of the others are
    w = np.random.default_rng(0).uniform(size=samples.shape[:2])
    w_nan = np.where(np.isfinite(bad_s).all(2), w, np.nan)
    assert np.isnan(w_nan[[3, 7]]).any(1).all()
    with pytest.warns(RuntimeWarning, match="dropped 3 of 300"):
        rec, arr = metrics.evaluate(bad_s, bad_t, NAMES, drop_nonfinite=True, weights=w_nan, **kw)
    ref, _ = metrics.evaluate(samples[keep], truth[keep], NAMES, weights=w[keep], **kw)
    assert {**rec, "n_dropped": 0} == ref and arr["ess"].shape == (297,)
    w_nan[20, 0] = np.nan
    for w_bad in (w_nan, w[:5]):
        with pytest.raises(ValueError, match="weights"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            metrics.evaluate(bad_s, bad_t, NAMES, drop_nonfinite=True, weights=w_bad, **kw)


def test_seed_extra_and_option_errors(calibrated):
    samples, truth = calibrated
    kw = dict(n_references=2, n_bootstrap=5)
    a, b = (metrics.evaluate(samples, truth, NAMES, seed=None, **kw)[0]["tarp"]["all"] for _ in range(2))
    assert a["bias"] != b["bias"] and abs(a["bias"]) < 0.2                   # no seed: new reference draws per call
    assert metrics.evaluate(samples, truth, NAMES, seed=np.int64(3), **kw)[0] == metrics.evaluate(
        samples, truth, NAMES, seed=3, **kw)[0]
    assert not np.array_equal(metrics.tarp_f(samples, truth, 1, seed0=None), metrics.tarp_f(samples, truth, 1, None))
    for call in (lambda: metrics.evaluate(samples, truth, NAMES, seed=1.5), lambda: metrics.evaluate(
            samples, truth, NAMES, seed="a"), lambda: metrics.tarp_scalars(np.full(20, 0.5), seed=0.5)):
        with pytest.raises(ValueError, match="seed must be an int"):
            call()
    with pytest.raises(ValueError, match="seed0 must be an int"):
        metrics.tarp_f(samples, truth, seed0=2.0)
    with pytest.raises(ValueError, match="extra must map"):
        metrics.evaluate(samples, truth, NAMES, extra=[lambda s, t: 1], **kw)
    with pytest.raises(ValueError, match="tol_min"):
        metrics.evaluate(samples, truth, NAMES, tol_min=-1, **kw)
    f = np.arange(11)[None] / 10
    with pytest.raises(ValueError, match="grid k / n_samples for n_samples=7"):
        metrics.tarp_scalars(f, 0, n_samples=7)
    assert metrics.tarp_scalars(f, 0, n_samples=20)["n_sims"] == 11          # a multiple is a valid grid


def test_names_and_sets_validation(calibrated):
    samples, truth = calibrated
    names = ["Om", "S8", "h"]
    with pytest.raises(ValueError, match="param_names must be 3 distinct"):
        metrics.evaluate(samples, truth, names[:2])
    with pytest.raises(ValueError, match="param_names must be 3 distinct"):
        metrics.evaluate(samples, truth, ["Om", "S8", "S8"])
    with pytest.raises(ValueError, match=r"fom_sets\['x'\].*'s8'.*did you mean 'S8'"):
        metrics.evaluate(samples, truth, names, {"x": ["Om", "s8"]})
    with pytest.raises(ValueError, match=r"tarp_sets\['y'\].*'sigma8'.*available"):
        metrics.evaluate(samples, truth, names, None, {"y": ["Om", "sigma8"]})
    with pytest.raises(ValueError, match="reserved"):
        metrics.evaluate(samples, truth, names, None, {"all": ["Om"]})
    for bad in ([["Om"]], {"x": []}, {"x": ["Om", "Om"]}):
        with pytest.raises(ValueError, match="fom_sets"):
            metrics.evaluate(samples, truth, names, bad)
    rec, _ = metrics.evaluate(samples, truth, names, {"x": "Om"}, {"all": names, "y": ["S8"]}, n_references=2,
                              n_bootstrap=0)
    assert list(rec["tarp"]) == ["all", "y"] and list(rec["fom"]) == ["x"] and rec["param_names"] == names


def test_hpt_style_positional_call(calibrated, device):
    rec, arr = metrics.evaluate(*calibrated, NAMES, {"fom": ["a", "b"]}, {"fom": ["a", "b"]}, 5, 20, 0.02, 0, device)
    for key in ("all", "fom"):
        t = rec["tarp"][key]
        assert all(isinstance(t[k], float) for k in ("bias", "confidence", "tol_bias", "tol_confidence"))
    assert {"n_sims", "param_names", "tarp", "fom", "marginal", "z_mean", "z_rms", "std_median"} <= set(rec)
    assert {"log_fom_mean", "fom_geomean", "n_nan"} == set(rec["fom"]["fom"])
    assert {"post_mean", "post_std", "post_median", "tarp_alpha", "tarp_ecp_all", "tarp_ecp_fom",
            "log_fom_fom"} <= set(arr)


# ------------------------------------------------------------------------------ marginals and summaries
def test_coverage_levels(calibrated):
    samples, truth = calibrated
    with pytest.warns(RuntimeWarning, match=r"300 samples.*0.9973 \(needs 740\), 0.995 \(needs 399\), 0.999 \(needs"):
        mc = metrics.marginal_coverage(samples, truth, (0.68, 0.9973, 0.995, 0.999))
    assert list(mc) == ["ks", "ks_pvalue", "cov68", "cov68_sd", "cov99.73", "cov99.73_sd", "cov99.5", "cov99.5_sd",
                        "cov99.9", "cov99.9_sd"]
    assert all(len(v) == 3 for v in mc.values()) and np.isclose(mc["cov68_sd"][0], np.sqrt(0.68 * 0.32 / 300))
    assert np.isnan(mc["cov99.73"]).all() and np.isfinite(mc["cov68"]).all()
    assert list(metrics.marginal_coverage(samples, truth, 0.9)) == ["ks", "ks_pvalue", "cov90", "cov90_sd"]
    assert list(metrics.marginal_coverage(samples, truth, torch.tensor([[0.5, 0.9]])))[2::2] == ["cov50", "cov90"]
    for bad in ((68, 95), (0.0, 0.5), (0.5, 1.0), (), (0.9999996, 0.9999997), "0.68", ["0.68"], None, (np.nan,)):
        with pytest.raises(ValueError, match="levels"):
            metrics.marginal_coverage(samples, truth, bad)
    with pytest.warns(RuntimeWarning, match="needs 740"):
        rec, _ = metrics.evaluate(samples, truth, NAMES, levels=(0.5, 0.9973), n_references=2, n_bootstrap=0)
    assert {"cov50", "cov99.73"} <= set(rec["marginal"]["a"]) and "cov68" not in rec["marginal"]["a"]
    assert rec["marginal"]["a"]["cov99.73"] is None and json.dumps(rec, allow_nan=False)
    with pytest.raises(ValueError, match="levels"):
        metrics.evaluate(samples, truth, NAMES, levels=(68,))


def test_coverage_needs_enough_samples_for_the_level():
    # grossly overconfident posteriors: nearly every truth lies outside all samples
    for n_samples, level, key in ((10, 0.95, "cov95"), (300, 0.9973, "cov99.73")):
        samples, truth = _gaussian_posteriors(1000, n_samples, 1, width_factor=0.05, seed=3)
        with pytest.warns(RuntimeWarning, match=f"{n_samples} samples per posterior cannot resolve"):
            assert np.isnan(metrics.marginal_coverage(samples, truth, (level,))[key][0])
    for n_samples, level, key in ((39, 0.95, "cov95"), (740, 0.9973, "cov99.73"), (6, 0.68, "cov68")):
        samples, truth = _gaussian_posteriors(1000, n_samples, 1, width_factor=0.05, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert metrics.marginal_coverage(samples, truth, (level,))[key][0] < 0.2
    # a truth outside every sample is never covered, at any resolved level
    samples = np.random.default_rng(0).normal(size=(50, 40, 2))
    mc = metrics.marginal_coverage(samples, np.where(np.arange(50)[:, None] < 25, -9.0, 9.0) * np.ones((50, 2)))
    assert mc["cov68"] == [0.0, 0.0] and mc["cov95"] == [0.0, 0.0] and np.allclose(mc["ks"], 0.5 - 1 / 41)


def test_coverage_is_unbiased_for_few_samples():
    for n_samples in (6, 10, 20, 39, 50):
        samples, truth = _gaussian_posteriors(20000, n_samples, 1, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                  # the levels that need more samples
            mc = metrics.marginal_coverage(samples, truth, (0.5, 0.68, 0.9, 0.95))
        for level in (0.5, 0.68, 0.9, 0.95):
            key = "cov" + format(100 * level, "g")
            resolved = n_samples >= 2 / (1 - level) - 1 - 1e-9
            assert np.isfinite(mc[key][0]) == resolved
            assert not resolved or abs(mc[key][0] - level) < 3.5 * mc[key + "_sd"][0], (n_samples, level)


def test_ks_pvalue_with_few_samples_and_many_simulations():
    # discrete ranks: calibrated posteriors must not be rejected, miscalibrated ones still are
    for n_sims, n_samples in ((4000, 20), (20000, 10), (2000, 3)):
        samples, truth = _gaussian_posteriors(n_sims, n_samples, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mc = metrics.marginal_coverage(samples, truth)
        assert min(mc["ks_pvalue"]) > 0.05 and max(mc["ks"]) < 2 / np.sqrt(n_sims), (n_sims, n_samples)
    for kw in ({"width_factor": 0.7}, {"width_factor": 1.4}, {"shift": 0.3}):
        samples, truth = _gaussian_posteriors(2000, 10, 1, seed=2, **kw)
        assert metrics.marginal_coverage(samples, truth, (0.5,))["ks_pvalue"][0] < 1e-6
    # many samples: the distance is the usual KS distance of the ranks
    samples, truth = _gaussian_posteriors(300, 2000, 1)
    ranks = np.sort((samples < truth[:, None]).mean(1)[:, 0])
    usual = max(np.abs(ranks - np.arange(1, 301) / 300).max(), np.abs(ranks - np.arange(300) / 300).max())
    assert abs(metrics.marginal_coverage(samples, truth)["ks"][0] - usual) < 2 / 2000


def test_ks_pvalue_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for n in (20, 300, 5000):
        d = np.array([0.1, 0.5, 1.0, 1.36, 2.0]) / np.sqrt(n)
        np.testing.assert_allclose(metrics._ks_pvalue(d, n), stats.kstwo.sf(d, n), atol=0.015)


def test_weights_of_one_reproduce_the_unweighted_results(calibrated):
    samples, truth = calibrated
    kw = dict(fom_sets={"ab": ["a", "b"]}, n_references=3, n_bootstrap=5)
    rec, arr = metrics.evaluate(samples, truth, NAMES, **kw)
    for w in (np.ones(samples.shape[:2]), torch.full(samples.shape[:2], 0.25)):
        rec_w, arr_w = metrics.evaluate(samples, truth, NAMES, weights=w, **kw)
        assert rec_w.pop("ess") == {"min": 300.0, "median": 300.0} and np.all(arr_w.pop("ess") == 300)
        assert rec_w == {k: v for k, v in rec.items() if k != "ess"}
        assert all(np.array_equal(arr[k], arr_w[k]) for k in arr)
    # the general weighted code agrees with it to rounding
    w = np.ones(samples.shape[:2])
    w[:, 0] = 1 + 1e-12
    plain, weighted = metrics.posterior_summary(samples, truth), metrics.posterior_summary(samples, truth, w)
    for k in plain:
        np.testing.assert_allclose(weighted[k], plain[k], rtol=0, atol=1e-9)


def test_weighted_summary(calibrated):
    samples, truth = calibrated
    rng = np.random.default_rng(1)
    w = rng.integers(0, 4, size=samples.shape[:2]).astype(float)
    repeated = np.repeat(samples[5], w[5].astype(int), axis=0)
    a, b = metrics.posterior_summary(samples[5], weights=w[5]), metrics.posterior_summary(repeated)
    np.testing.assert_allclose(a["mean"], b["mean"], atol=1e-12)
    np.testing.assert_allclose(a["std"], b["std"], atol=1e-12)
    np.testing.assert_allclose(a["median"], b["median"], atol=0.02)
    # samples with zero weight have no influence
    mask = rng.uniform(size=samples.shape[1]) < 0.5
    a, b = metrics.posterior_summary(samples[5], truth[5], mask.astype(float)), metrics.posterior_summary(
        samples[5, mask], truth[5])
    for k in b:
        np.testing.assert_allclose(a[k], b[k], atol=1e-9)
    # a known reweighting: N(0, 1) samples to N(0.5, 0.8)
    x = rng.normal(size=(1, 100_000, 1))
    lw = -0.5 * ((x[..., 0] - 0.5) / 0.8) ** 2 + 0.5 * x[..., 0] ** 2
    s = metrics.posterior_summary(x, weights=np.exp(lw - lw.max()))
    expected = {"mean": 0.5, "std": 0.8, "median": 0.5, "q16": 0.5 - 0.8 * 0.9945, "q84": 0.5 + 0.8 * 0.9945,
                "q025": 0.5 - 0.8 * 1.96, "q975": 0.5 + 0.8 * 1.96}
    for k, v in expected.items():
        assert abs(s[k][0, 0] - v) < 0.03, k
    # no weight at all, or all of it on one sample: nan / zero width, flagged
    w = np.ones(samples.shape[:2])
    w[5], w[6, 1:] = 0, 0
    with pytest.warns(RuntimeWarning, match="2 of 300 simulations have a zero posterior std"):
        s = metrics.posterior_summary(samples, truth, w)
    assert np.isnan(s["mean"][5]).all() and np.isnan(s["median"][5]).all() and np.isnan(s["z"][5:7]).all()
    assert np.array_equal(s["median"][6], samples[6, 0]) and np.isfinite(s["z"][7]).all()
    for bad in (np.ones((299, 300)), np.ones(300), -w, w * np.nan, np.ones((300, 3))):
        with pytest.raises(ValueError, match="weights"):
            metrics.posterior_summary(samples, truth, bad)
        with pytest.raises(ValueError, match="weights"):
            metrics.evaluate(samples, truth, NAMES, weights=bad)


def test_effective_sample_size_and_importance_weights():
    from sbikit import Prior
    w = np.zeros((4, 50))
    w[0], w[1, 3], w[2, :10] = 1.0, 0.7, 1e-200
    np.testing.assert_allclose(metrics.effective_sample_size(w), [50, 1, 10, 0])
    np.testing.assert_allclose(metrics.effective_sample_size([1.0, 1.0, 0.0]), [2])
    np.testing.assert_allclose(metrics.effective_sample_size(torch.tensor([[3.0, 1.0]])), [16 / 10])
    for bad in (np.ones((2, 3, 4)), [-1.0, 1.0], [np.nan, 1.0], np.ones((2, 0))):
        with pytest.raises(ValueError, match="weights"):
            metrics.effective_sample_size(bad)
    wide, narrow = Prior([0.0, 0.0], [1.0, 1.0]), Prior([0.2, 0.0], [0.6, 1.0])
    samples = np.random.default_rng(0).uniform(size=(5, 400, 2))
    samples[4, :, 0] = 0.9                               # one posterior entirely outside the narrow box
    iw = metrics.importance_weights(samples, wide, narrow)
    inside = (samples[..., 0] >= 0.2) & (samples[..., 0] <= 0.6)
    assert np.array_equal(iw, inside.astype(float))
    np.testing.assert_allclose(metrics.effective_sample_size(iw), inside.sum(1))
    assert metrics.importance_weights(torch.tensor(samples[0]), wide, narrow).shape == (1, 400)
    with pytest.raises(TypeError, match="one prior"):
        metrics.importance_weights(samples, [wide] * 5, narrow)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # zero-weight simulations: nan without numpy warnings
        lf = metrics.log_fom(samples, [0, 1], iw)
    assert np.isnan(lf[4]) and np.isfinite(lf[:4]).all()
    rec, arr = metrics.evaluate(samples[:4], samples[:4, 0], ["a", "b"], {"ab": ["a", "b"]}, weights=iw[:4],
                                bounds=(0, 1), n_references=2, n_bootstrap=0)
    assert rec["ess"]["min"] == inside[:4].sum(1).min() and arr["ess"].shape == (4,)
    np.testing.assert_allclose(arr["log_fom_ab"], lf[:4])


# ------------------------------------------------------------------------------ figure of merit
def test_log_fom_columns_and_weights(calibrated):
    samples, _ = calibrated
    ref = metrics.log_fom(samples, [0, 2])
    assert np.array_equal(metrics.log_fom(samples, ["a", "c"], names=NAMES), ref)
    assert np.array_equal(metrics.log_fom(samples, np.array([0, -1])), ref)
    assert np.array_equal(metrics.log_fom(samples, 1), metrics.log_fom(samples, "b", names=NAMES))
    np.testing.assert_allclose(metrics.log_fom(samples, 1), -0.5 * np.log(samples[..., 1].var(1, ddof=1)))
    assert np.array_equal(metrics.log_fom(samples, np.array(1)), metrics.log_fom(samples, 1))      # 0-d array
    assert np.array_equal(metrics.log_fom(samples, torch.tensor([0, 2])), ref)
    for idx, kw in ((["a"], {}), ([3], {}), ([0, 0], {}), ([], {}), (["x"], {"names": NAMES}), ([0.5], {}),
                    (["a"], {"names": NAMES[:2]}), (None, {}), (np.array([[0, 1]]), {}), (1.0, {})):
        with pytest.raises(ValueError, match="idx|names"):
            metrics.log_fom(samples, idx, **kw)
    w = np.random.default_rng(0).uniform(size=samples.shape[:2])
    expected = [-0.5 * np.linalg.slogdet(np.cov(s[:, :2].T, aweights=wi))[1] for s, wi in zip(samples[:5], w[:5])]
    np.testing.assert_allclose(metrics.log_fom(samples[:5], [0, 1], w[:5].tolist()), expected)
    for bad in (np.ones((301, 300)), np.ones(300), -w, [[1.0] * 300] * 299):
        with pytest.raises(ValueError, match="weights"):
            metrics.log_fom(samples, [0, 1], bad)
    # too few effective samples for a covariance of two parameters: nan, not an inflated number
    few = np.zeros((4, 300))
    few[0, :3], few[1, :2], few[2, :3], few[3, 0] = 1.0, 1.0, [1.0, 1e-9, 1e-9], 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lf = metrics.log_fom(samples[:4], [0, 1], few)
    assert np.isfinite(lf[0]) and np.isnan(lf[1:]).all()
    assert np.isclose(lf[0], metrics.log_fom(samples[:1, :3], [0, 1])[0])


def test_degenerate_chains_give_nan(calibrated):
    samples, truth = calibrated
    rng = np.random.default_rng(0)
    values = (rng.normal(size=(200, 1, 3)) * 10.0 ** rng.integers(-9, 3, size=(200, 1, 1))).astype(np.float32)
    stuck = np.repeat(values, 100, axis=1)               # every sample of a chain identical
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for idx in ([0], [0, 1], [0, 1, 2]):
            assert np.isnan(metrics.log_fom(stuck, idx)).all()
        collinear = samples.copy()
        collinear[:, :, 2] = collinear[:, :, 0].astype(np.float32) * np.float32(0.3)
        assert np.isnan(metrics.log_fom(collinear, [0, 2])).all()
        assert np.isfinite(metrics.log_fom(collinear, [0, 1])).all()
        assert np.isnan(metrics.log_fom(samples[:, :2], [0, 1, 2])).all()        # fewer samples than parameters
        small = samples.copy()
        small[:, :, 0] = 2e-9 * (1 + 0.05 * small[:, :, 0])                      # tiny scale is not degenerate
        assert np.isfinite(metrics.log_fom(small, [0, 1])).all()
    one_stuck = samples.copy()
    one_stuck[4] = one_stuck[4, :1]
    with pytest.warns(RuntimeWarning, match="1 of 300 simulations have a zero posterior std"):
        rec, arr = metrics.evaluate(one_stuck, truth, NAMES, {"ab": ["a", "b"]}, n_references=2, n_bootstrap=0)
    assert rec["fom"]["ab"]["n_nan"] == 1 and np.isnan(arr["log_fom_ab"][4])
    assert rec["z_n_nan"] == dict.fromkeys(NAMES, 1)
    assert all(abs(rec["z_mean"][n]) < 0.5 and rec["z_rms"][n] < 1.5 for n in NAMES)
    assert abs(rec["fom"]["ab"]["log_fom_mean"] - np.log(2.0)) < 0.05


# ------------------------------------------------------------------------------ record
def test_record_is_strict_json(calibrated):
    samples, truth = calibrated
    rec, _ = metrics.evaluate(samples, truth, NAMES, {"ab": ["a", "b"]}, n_references=2, n_bootstrap=0)
    assert rec["tarp"]["all"]["bias_sd"] is None and rec["tarp"]["all"]["tol_bias"] == 0.02
    assert json.loads(json.dumps(rec, allow_nan=False)) == rec
    stuck = np.repeat(samples[:, :1], 20, axis=1)
    with pytest.warns(RuntimeWarning):
        rec, arr = metrics.evaluate(stuck, truth, NAMES, {"ab": ["a", "b"]}, n_references=2, n_bootstrap=0)
    assert rec["fom"]["ab"] == {"log_fom_mean": None, "fom_geomean": None, "n_nan": 300}
    assert rec["z_mean"] == dict.fromkeys(NAMES) and rec["z_n_nan"]["a"] == 300 and np.isnan(arr["log_fom_ab"]).all()
    assert json.loads(json.dumps(rec, allow_nan=False)) == rec


def test_extra_metrics(calibrated):
    samples, truth = calibrated
    extra = {"rmse": lambda s, t: np.sqrt(((s.mean(1) - t) ** 2).mean(0)), "n": lambda s, t: s.shape[1],
             "nested": lambda s, t: {"x": np.float32(1.5), 3: (1, torch.tensor(2.0)), "bad": np.nan}}
    rec, _ = metrics.evaluate(samples, truth, NAMES, n_references=2, n_bootstrap=0, extra=extra)
    assert rec["extra"]["n"] == 300 and rec["extra"]["nested"] == {"x": 1.5, "3": [1, 2.0], "bad": None}
    np.testing.assert_allclose(rec["extra"]["rmse"], np.sqrt(((samples.mean(1) - truth) ** 2).mean(0)))
    assert json.loads(json.dumps(rec, allow_nan=False)) == rec
    with pytest.raises(TypeError, match=r"extra\['obj'\]"):
        metrics.evaluate(samples, truth, NAMES, n_references=2, extra={"obj": lambda s, t: object()})
    with pytest.raises(ValueError, match=r"extra\['three'\]"):
        metrics.evaluate(samples, truth, NAMES, n_references=2, extra={"three": 3})
