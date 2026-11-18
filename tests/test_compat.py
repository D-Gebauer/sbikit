"""Checkpoints written by sbikit 0.1.0 must keep loading and give the outputs recorded with them.

``tests/data/v0.1.0`` holds one checkpoint per saveable class and ``reference.npz`` with the outputs
of that version on fixed inputs.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

import sbikit as sk

DATA = Path(__file__).parent / "data" / "v0.1.0"


@pytest.fixture(scope="module")
def ref():
    with np.load(DATA / "reference.npz") as f:
        return {k: f[k] for k in f.files}


@pytest.mark.parametrize("name", ["moped", "mse", "vmim", "hier", "imnn"])
def test_compressor_checkpoints(name, ref, device):
    comp = sk.load(DATA / f"{name}.pt", device)
    assert np.allclose(comp(ref["x"]), ref[name], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["nle_maf", "nle_nsf"])
def test_nle_checkpoints(name, ref, device):
    nle = sk.load(DATA / f"{name}.pt", device)
    # 0.1.0 left out the constant Jacobian of the summary scaler, -sum(log std)
    expected = ref[name] - ref[f"{name}_logstd"]
    assert np.allclose(nle.log_likelihood(ref["s"], ref["theta"]).cpu().numpy(), expected, rtol=1e-4, atol=1e-4)


def test_ensemble_checkpoint(ref, device):
    ens = sk.load(DATA / "ensemble.pt", device)
    assert len(ens.members) == 2
    members = np.stack([ref[n] - ref[f"{n}_logstd"] for n in ("nle_maf", "nle_nsf")])
    expected = np.logaddexp(*members) - np.log(2.0)
    assert np.allclose(ens.log_likelihood(ref["s"], ref["theta"]).cpu().numpy(), expected, rtol=1e-4, atol=1e-4)


def test_prior_checkpoint(ref, device):
    prior = sk.load(DATA / "prior.pt", device)
    lp = prior.log_prob(torch.as_tensor(ref["prior_pts"], device=device)).cpu().numpy()
    inside = np.isfinite(ref["prior_lp"])
    assert np.array_equal(np.isfinite(lp), inside)
    assert np.allclose(lp[inside], ref["prior_lp"][inside], rtol=1e-4, atol=1e-4)
    assert prior.names == ["a", "b", "c"]
