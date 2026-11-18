import numpy as np
import pytest

from sbikit import FiducialSet
from sbikit.compress import IMNN, MOPED, HierarchicalVMIM, MSECompressor, VMIMCompressor, moped_vectors
from sbikit.io import load, save


def _linear_fiducial(n_fid=4000, n_d=500, D=6, K=2, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D, K))
    theta_fid, step = np.zeros(K), np.full(K, 0.05)
    noise = lambda n: sigma * rng.normal(size=(n, D))       # noqa: E731
    x = theta_fid @ A.T + noise(n_fid)
    eps = noise(n_d)                                          # seed-matched noise
    x_m = np.stack([(theta_fid - step * e) @ A.T + eps for e in np.eye(K)])
    x_p = np.stack([(theta_fid + step * e) @ A.T + eps for e in np.eye(K)])
    fisher = A.T @ A / sigma ** 2
    return FiducialSet(x, ["a", "b"], theta_fid, x_m, x_p, 2 * step), A, fisher


def test_moped_is_lossless_for_linear_gaussian():
    fid, A, fisher = _linear_fiducial()
    m = MOPED(fid, hartlap=False)
    m.fit()
    y = m(fid.x)
    np.testing.assert_allclose(np.cov(y.T), np.eye(2), atol=1e-6)        # orthonormal at the fiducial
    dy = m.vectors @ A                                                     # d y / d theta
    np.testing.assert_allclose(dy.T @ dy, fisher, rtol=0.08)              # Fisher of the summaries
    with pytest.warns(UserWarning, match="degenerate"):
        vecs, kept = moped_vectors(np.vstack([A.T, A.T[:1]]), np.eye(6))
    assert kept == [0, 1]
    assert np.allclose(MOPED.from_state(m.state())(fid.x[:5]), y[:5])


def test_imnn_recovers_linear_fisher(tmp_path, device):
    fid, A, fisher = _linear_fiducial()
    trn, val = fid.split(np.arange(len(fid.x)) % 5, np.arange(fid.x_minus.shape[1]) % 5, [0])
    # reference network and optimiser; patience as the imnn README (min_iterations 1000, patience 250): the
    # covariance regularisation first lowers F, so short patience stops too early
    imnn = IMNN(trn, {"max_iterations": 3000, "min_iterations": 1000, "patience": 250}, device, seed=0,
                validation=val)
    loss = imnn.fit()
    logdet_true = np.linalg.slogdet(fisher)[1]
    assert -loss > logdet_true - 0.3                       # validation log det F close to the optimum
    assert np.linalg.slogdet(imnn.fisher.numpy())[1] < logdet_true + 0.3
    est = imnn(fid.x)                                      # quasi-MLE: unbiased at the fiducial
    np.testing.assert_allclose(est.mean(0), fid.theta_fid, atol=0.05)
    save(imnn, tmp_path / "imnn.pt")
    np.testing.assert_allclose(load(tmp_path / "imnn.pt", device)(fid.x[:10]), est[:10], rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="no longer exist"):
        IMNN(trn, {"epochs": 10})


def _imnn_reference(x, x_mp, dtheta, lam, eps):
    """imnn 0.3.2 (_imnn.py, numerical_gradient_imnn.py) in numpy, transcribed: loss, F, C, invC, dmu, mu.
    x (n_s, n), x_mp (n_d, 2, K, n) as imnn stores the derivative summaries (index 0: minus, 1: plus)."""
    derivatives = np.swapaxes(x_mp[:, 1] - x_mp[:, 0], 1, 2) / dtheta          # _construct_derivatives
    mu = np.mean(x, axis=0)
    C = np.cov(x, rowvar=False)
    invC = np.linalg.inv(C)
    dmu = np.mean(derivatives, axis=0)
    F = np.einsum("ij,ik,kl->jl", dmu, invC, dmu)
    sign, logdet = np.linalg.slogdet(F)
    L2 = np.linalg.norm(C - np.eye(x.shape[1])) + np.linalg.norm(invC - np.eye(x.shape[1]))
    alpha = -np.log(eps * (lam - 1.0) + eps ** 2.0 / (1 + eps)) / eps
    r = lam * L2 / (L2 + np.exp(-alpha * L2))
    return -sign * logdet + r * L2, F, C, invC, dmu, mu


def test_imnn_matches_reference_formulas():
    import torch
    from sbikit.compress.imnn import fisher_statistics, imnn_loss
    rng = np.random.default_rng(1)
    n_s, n_d, K, n = 400, 50, 3, 4
    x = rng.normal(size=(n_s, n)) @ rng.normal(size=(n, n))
    x_mp = rng.normal(size=(n_d, 2, K, n))
    x_mp[:, 1] += rng.normal(size=(K, n))
    dtheta = np.array([0.1, 0.2, 0.05])
    loss_r, F_r, C_r, invC_r, dmu_r, mu_r = _imnn_reference(x, x_mp, dtheta, 10.0, 0.1)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)                          # noqa: E731
    xp, xm = t(np.moveaxis(x_mp[:, 1], 0, 1)), t(np.moveaxis(x_mp[:, 0], 0, 1))   # sbikit: (K, n_d, n)
    loss, (F, C, invC, _, _) = imnn_loss(t(x), xp, xm, t(dtheta), 10.0, 0.1)
    _, _, _, dmu, mu = fisher_statistics(t(x), xp, xm, t(dtheta))
    np.testing.assert_allclose(float(loss), loss_r, rtol=1e-10)
    for a, b in ((F, F_r), (C, C_r), (invC, invC_r), (dmu, dmu_r), (mu, mu_r)):
        np.testing.assert_allclose(a.numpy(), b, rtol=1e-10, atol=1e-12)
    # estimator (imnn get_estimate): theta_fid + einsum("ij,kj,kl,ml->mi", invF, dmu, invC, x - mu)
    from sbikit.compress import IMNN as _I
    m = _I.__new__(_I)
    m.net, m.config = torch.nn.Identity(), type("C", (), {"output": "mle"})()
    m._summaries = lambda d: t(d)
    m.fisher, m.cov, m.dmu, m.mu, m.theta_fid = t(F_r), t(C_r), t(dmu_r), t(mu_r), t(np.array([1.0, 2.0, 3.0]))
    ref = np.array([1.0, 2.0, 3.0]) + np.einsum("ij,kj,kl,ml->mi", np.linalg.inv(F_r), dmu_r, invC_r, x[:5] - mu_r)
    np.testing.assert_allclose(m(x[:5]), ref, rtol=1e-5)


def _nonlinear_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1, 1, size=(n, 2))
    x = np.hstack([np.sin(2 * t), t ** 2, t[:, :1] * t[:, 1:], rng.normal(size=(n, 3))]) + 0.05 * rng.normal(size=(n, 8))
    return x.astype(np.float32), t


@pytest.mark.parametrize("cls", [MSECompressor, VMIMCompressor])
def test_net_compressors_fit_and_roundtrip(cls, tmp_path, device):
    x, t = _nonlinear_data()
    comp = cls(2, {"widths": (64, 32), "epochs": 40, "patience": 10, "lr": 3e-3}, device, seed=0)
    loss = comp.fit(x[:2500], t[:2500], x[2500:], t[2500:])
    assert np.isfinite(loss)
    s = comp(x[2500:])
    assert s.shape == (500, 2)
    if cls is MSECompressor:
        assert np.corrcoef(s[:, 0], t[2500:, 0])[0, 1] > 0.9
    save(comp, tmp_path / "c.pt")
    np.testing.assert_allclose(load(tmp_path / "c.pt", device)(x[:10]), comp(x[:10]), rtol=1e-4, atol=1e-5)


def test_hierarchical_vmim(tmp_path, device):
    x, t = _nonlinear_data()
    blocks = {"sin": [0, 1], "sq": [2, 3], "cross": [4], "noise": [5, 6, 7]}
    h = HierarchicalVMIM(blocks, anchor=["sin"], orders=[["sq", "cross"], ["cross", "sq", "noise"]], n_out=2,
                         configs={"default": {"widths": (32,), "epochs": 15, "patience": 5, "lr": 3e-3},
                                  "final": {"widths": (32,), "epochs": 15, "patience": 5, "lr": 3e-3, "vicreg": 0.1}},
                         device=device)
    loss = h.fit(x[:2000], t[:2000], x[2000:2500], t[2000:2500],
                 final_data=(x[:2500], t[:2500], x[2500:], t[2500:]))
    assert np.isfinite(loss) and len(h.stages) == 2 + 2 + 3 and "r1/noise" in h.val_losses
    assert h(x[:7]).shape == (7, 2)
    save(h, tmp_path / "h.pt")
    np.testing.assert_allclose(load(tmp_path / "h.pt", device)(x[:10]), h(x[:10]), rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_vmim_atoms_stop_the_spike(device):
    """40 % of the labels sit on an atom (t1 = 0). Validated on atom rows, an MDN trained without
    `atoms` rewards a near-delta spike there; with the labels spread over +-0.1 it cannot."""
    rng = np.random.default_rng(1)
    n = 3000
    t = rng.uniform(-1, 1, size=(n, 2))
    atom = rng.random(n) < 0.4
    t[atom, 1] = 0.0
    x = np.hstack([t[:, :1] + 0.05 * rng.normal(size=(n, 1)), rng.normal(size=(n, 3))]).astype(np.float32)
    trn, val = np.arange(2500), np.arange(2500, n)[atom[2500:]]
    cfg = {"widths": (32,), "epochs": 60, "patience": 60, "lr": 3e-3, "components": 4}
    spike = VMIMCompressor(2, cfg, device, seed=0).fit(x[trn], t[trn], x[val], t[val])
    smooth = VMIMCompressor(2, cfg, device, seed=0)
    loss = smooth.fit(x[trn], t[trn], x[val], t[val], atoms=[(1, 0.0, 0.1)])
    # the spread labels bound the atom-row density: -log q(t1) >= log(0.2 / sd(t1)) - small
    assert loss - spike > 1.0, (spike, loss)
    assert smooth.atoms == [[1, 0.0, 0.1]]
