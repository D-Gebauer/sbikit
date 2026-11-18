"""Transforms in sampling, pivoted / fixed MOPED, Dataset meta, blinding."""
import numpy as np
import pytest
import torch

from sbikit import Columns, Dataset, FiducialSet, MCMCConfig, Prior, sample_posteriors
from sbikit.blinding import Blinder, make_key
from sbikit.compress import MOPED, compressor_from_state, moped_vectors, moped_vectors_pivoted


# ------------------------------------------------------------------ transform: sample psi, flow sees theta(psi)
def test_transform_linear_reparametrisation(linear_model):
    """theta = M psi + c with the likelihood in theta: the psi posterior is the analytic Gaussian."""
    from conftest import simulate
    rng = np.random.default_rng(3)
    M = np.array([[1.0, 0.5], [0.0, 2.0]])
    c = np.array([0.1, -0.2])
    psi_true = rng.uniform(-0.3, 0.3, size=(4, 2))
    x = simulate(linear_model, psi_true @ M.T + c, rng)
    Mt, ct = torch.tensor(M, dtype=torch.float32), torch.tensor(c, dtype=torch.float32)
    tr = Columns(["a", "b"], keep=[], derived=["t0", "t1"], fn=lambda p: p @ Mt.T.to(p.device) + ct.to(p.device))
    cfg = MCMCConfig(nwalkers=64, burn_in=300, block=300, max_steps=6000, thin=2, min_ess=500, n_samples=3000, seed=2)
    post = sample_posteriors(linear_model["lik"], x, Prior([-3, -3], [3, 3]), cfg, init_centre=psi_true,
                             init_sd=[0.05, 0.05], verbose=False, transform=tr)
    assert post.converged.all()
    A, s, b = linear_model["A"] @ M, linear_model["sigma"], linear_model["b"] + linear_model["A"] @ c
    cov = np.linalg.inv(A.T @ A / s ** 2)
    mean = (cov @ (A.T @ (x - b).T / s ** 2)).T
    sd = np.sqrt(cov.diagonal())
    np.testing.assert_allclose(post.samples.mean(1), mean, atol=0.15 * sd.max())
    np.testing.assert_allclose(post.samples.std(1).mean(0), sd, rtol=0.1)
    np.testing.assert_allclose(post.derived, post.samples @ M.T + c, atol=1e-5)


def test_columns_order_and_check():
    tr = Columns(["a", "b", "c"], keep=["c", "a"], derived=["d"], fn=lambda p: p[..., :1] * p[..., 1:2],
                 order=["a", "d", "c"])
    psi = torch.tensor([[2.0, 3.0, 5.0]])
    np.testing.assert_allclose(tr(psi).numpy(), [[2.0, 6.0, 5.0]])
    tr.check(["a", "d", "c"])
    with pytest.raises(ValueError):
        tr.check(["a", "c", "d"])


# ------------------------------------------------------------------ MOPED
def _fid(rng, K=4, D=12):
    L = rng.normal(size=(D, D)) / np.sqrt(D)
    C = L @ L.T + np.eye(D)
    x = rng.multivariate_normal(np.zeros(D), C, size=20000)
    dmu = rng.normal(size=(K, D)) * np.array([0.2, 3.0, 1.0, 0.05])[:, None]
    return FiducialSet(x=x, param_names=list("abcd"), theta_fid=np.zeros(K), dmu=dmu), C


def test_pivoted_moped_properties():
    rng = np.random.default_rng(0)
    fid, _ = _fid(rng)
    C = np.cov(fid.x.T)
    inv = np.linalg.inv(C)
    widths = np.array([1.0, 1.0, 1.0, 10.0])
    B, order, scores = moped_vectors_pivoted(fid.dmu, inv, widths)
    np.testing.assert_allclose(B @ C @ B.T, np.eye(4), atol=1e-10)
    first = int(np.argmax(np.sqrt(np.einsum("kd,de,ke->k", fid.dmu, inv, fid.dmu)) * widths))
    assert order[0] == first
    # same span as the unpivoted vectors: identical chi^2 of any vector in the coefficients
    B0, _ = moped_vectors(fid.dmu, inv)
    v = rng.normal(size=12)
    np.testing.assert_allclose(np.sum((B @ v) ** 2), np.sum((B0 @ v) ** 2), rtol=1e-10)
    # the class reproduces the function (Hartlap off) and stores the order
    m = MOPED(fid, hartlap=False, pivot_scale=dict(zip("abcd", widths)))
    m.fit()
    assert m.kept_params == [list("abcd")[k] for k in order]
    np.testing.assert_allclose(m.vectors, B, atol=1e-10)
    m2 = compressor_from_state(m.state())
    np.testing.assert_allclose(m2(fid.x[:5]), m(fid.x[:5]))


def test_moped_from_vectors():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(3, 7))
    m = MOPED.from_vectors(V, ["p", "q", "r"])
    x = rng.normal(size=(5, 7))
    np.testing.assert_allclose(m(x), (x @ V.T).astype(np.float32), rtol=1e-6)
    assert m.n_out == 3
    np.testing.assert_allclose(compressor_from_state(m.state())(x), m(x))
    with pytest.raises(ValueError):
        MOPED.from_vectors(V, ["p", "q"])


# ------------------------------------------------------------------ Dataset meta
def test_dataset_meta_subset():
    rng = np.random.default_rng(2)
    ds = Dataset(rng.normal(size=(10, 2)), rng.normal(size=(10, 3)), ["a", "b"], groups=np.arange(10) // 2,
                 meta={"dz": rng.normal(size=(10, 6)), "run": np.arange(10)})
    sub = ds.subset([1, 4, 7])
    np.testing.assert_array_equal(sub.meta["run"], [1, 4, 7])
    assert sub.meta["dz"].shape == (3, 6)
    with pytest.raises(ValueError):
        Dataset(np.zeros((4, 1)), np.zeros((4, 2)), ["a"], meta={"m": np.zeros(3)})


# ------------------------------------------------------------------ blinding
def test_blinding_roundtrip(tmp_path):
    key = tmp_path / "key.json"
    fp = make_key(key, {"S8": 0.05, "w0": 0.3})
    assert (key.stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError):
        make_key(key, {"S8": 0.05})
    b = Blinder(key)
    assert b.fingerprint == fp and b.params == ["S8", "w0"]
    s = np.random.default_rng(0).normal(size=(2, 100, 3))
    names = ["Om", "S8", "w0"]
    bl = b.blind(s, names)
    np.testing.assert_array_equal(bl[..., 0], s[..., 0])
    d = (bl - s)[0, 0]
    assert 0 < abs(d[1]) <= 0.05 and 0 < abs(d[2]) <= 0.3
    np.testing.assert_allclose(b.unblind(bl, names), s)
    np.testing.assert_allclose(Blinder(key).blind(s, names), bl)       # deterministic from the key


def test_no_tensorflow_via_tensorboard():
    # sbi -> torch.utils.tensorboard must not pull in TensorFlow (it can segfault next to triton, and is slow)
    import sys
    import sbikit  # noqa: F401
    import sbi.inference  # noqa: F401
    assert "tensorboard.compat.notf" in sys.modules or "tensorflow" in sys.modules
    assert "tensorflow" not in sys.modules


def test_ball_init_in_narrow_corner(device):
    # a truth in a corner of the support where a 0.3-sigma ball almost never lands inside: the ball shrinks
    import numpy as np
    import torch
    from sbikit import MCMCConfig, Prior, sample_posteriors
    from sbikit.priors import Polygon2D

    class Lik:
        def log_prob(self, input, condition):
            return -0.5 * ((condition - 0.5) ** 2).sum(-1).reshape(1, -1) * 100

    poly = Polygon2D([0, 1], np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.002], [0.0, 0.002]]))
    prior = Prior([0.0, 0.0], [1.0, 1.0], constraints=[poly], names=["a", "b"])
    cfg = MCMCConfig.make({"nwalkers": 16, "burn_in": 50, "max_steps": 200, "n_samples": 50, "adaptive": False})
    post = sample_posteriors(Lik(), np.zeros((1, 1), np.float32), prior, cfg, init_centre=np.array([[0.999, 0.001]]),
                             init_sd=np.array([0.3, 0.3]), device=device)
    assert np.isfinite(post.samples).all() and (post.samples[..., 1] <= 0.002 + 1e-6).all()
