import json

import numpy as np
import pytest

from sbikit import NLE, Ensemble, MCMCConfig, Prior, sample_posteriors
from sbikit.hpt import EvalSet, best_trials, tune_compressor, tune_nle
from sbikit.compress import MSECompressor
from sbikit.io import load, save


def _data(model, n, rng):
    from conftest import simulate
    theta = rng.uniform(-1, 1, size=(n, model["P"]))
    return theta, simulate(model, theta, rng).astype(np.float32)


@pytest.mark.slow
def test_nle_learns_linear_gaussian(linear_model, tmp_path, device):
    rng = np.random.default_rng(0)
    t, x = _data(linear_model, 6000, rng)
    nle = NLE(linear_model["D"], linear_model["P"], {"model": "maf", "hidden_features": 32, "num_transforms": 3},
              device=device)
    val, epochs = nle.fit(x[:5000], t[:5000], x[5000:], t[5000:], {"lr": 3e-3, "max_epochs": 60, "stop_after_epochs": 8})
    assert np.isfinite(val)
    save(nle, tmp_path / "nle.pt")
    nle2 = load(tmp_path / "nle.pt", device)
    np.testing.assert_allclose(nle2.log_likelihood(x[:5], t[:5]).cpu(), nle.log_likelihood(x[:5], t[:5]).cpu(),
                               rtol=1e-4)
    ens = Ensemble([nle, nle2])
    truth, xo = _data(linear_model, 4, rng)
    truth, xo = np.clip(truth, -0.6, 0.6), linear_model["A"] @ np.clip(truth, -0.6, 0.6).T
    xo = (xo.T + linear_model["b"]).astype(np.float32)
    cfg = MCMCConfig(nwalkers=64, burn_in=300, block=300, max_steps=4000, thin=2, min_ess=400, seed=0)
    post = sample_posteriors(ens, xo, Prior([-1, -1], [1, 1]), cfg, init_centre=truth, verbose=False)
    # learned posterior close to the exact one
    exact = sample_posteriors(linear_model["lik"], xo, Prior([-1, -1], [1, 1]), cfg, init_centre=truth, verbose=False)
    sd = exact.samples.std(1)
    assert np.all(np.abs(post.samples.mean(1) - exact.samples.mean(1)) < 0.5 * sd)
    assert np.all(np.abs(post.samples.std(1) / sd - 1) < 0.3)


@pytest.mark.slow
def test_tune_nle_both_objectives(linear_model, tmp_path, device):
    rng = np.random.default_rng(1)
    t, x = _data(linear_model, 3000, rng)
    space = {"model": "maf", "hidden_features": {"choices": [16, 32]}, "num_transforms": 2, "num_blocks": 2,
             "lr": {"low": 1e-3, "high": 5e-3, "log": True}, "batch_size": 128}
    train = {"max_epochs": 15, "stop_after_epochs": 5}
    st = tune_nle(x[:2000], t[:2000], x[2000:2500], t[2000:2500], 2, tmp_path / "s.log", "vl", tmp_path,
                  space=space, train=train, n_startup=2, device=device)
    assert len(best_trials(st, 2)) == 2 and (tmp_path / "vl" / "trials" / "0000.pt").exists()
    ev = EvalSet(x[2500:2560], t[2500:2560], Prior([-1, -1], [1, 1]), ["a", "b"])
    mcmc = {"nwalkers": 32, "burn_in": 200, "block": 200, "max_steps": 1000, "thin": 2, "min_ess": 200, "n_samples": 500}
    st = tune_nle(x[:2000], t[:2000], x[2000:2500], t[2000:2500], 2, tmp_path / "s.log", "ft", tmp_path,
                  space=space, train=train, objective="fom_tarp", evaluation=ev, mcmc=mcmc, fom_params=["a", "b"],
                  n_references=5, n_bootstrap=20, n_startup=2, device=device)
    rec = json.loads((tmp_path / "ft" / "trials" / "0001.json").read_text())
    assert {"objective", "constraints", "metrics"} <= set(rec) and len(rec["constraints"]) == 5
    assert st.direction.name == "MAXIMIZE"


def test_tune_compressor(tmp_path, device):
    rng = np.random.default_rng(2)
    t = rng.uniform(-1, 1, size=(1200, 2))
    x = np.hstack([t, t ** 2, rng.normal(size=(1200, 2))]).astype(np.float32)
    space = {"widths": {"choices": ["32", "64,32"]}, "lr": 3e-3, "epochs": 10, "patience": 3}
    cfg, comp, st = tune_compressor(lambda c: MSECompressor(2, c, device), space, x[:1000], t[:1000], x[1000:], t[1000:],
                                    2, tmp_path / "c.db", "mse", tmp_path, n_startup=2)
    assert comp(x[:3]).shape == (3, 2) and isinstance(cfg["widths"], list)
