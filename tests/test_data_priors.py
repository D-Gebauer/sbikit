import numpy as np
import pytest
import torch

from sbikit import CurveBound2D, Dataset, Gaussian, KDE, Polygon2D, Prior, Splits, random_splits, resolve_splits
from sbikit.io import load, save


def test_random_splits_keep_groups_together():
    groups = np.repeat(np.arange(50), 4)                   # 50 N-body runs x 4 footprints
    s = random_splits(len(groups), {"train": 0.7, "val": 0.15, "test": 0.15}, groups, seed=3)
    s.check(len(groups), groups)
    assert sum(len(v) for v in s.values()) == len(groups)
    assert len(s["train"]) == 4 * 35
    assert s.hash() == random_splits(len(groups), {"train": 0.7, "val": 0.15, "test": 0.15}, groups, seed=3).hash()


def test_user_splits_are_checked():
    ds = Dataset(np.zeros((8, 1)), np.zeros((8, 2)), ["a"], groups=np.repeat(np.arange(4), 2))
    ok = resolve_splits(ds, {"train": [0, 1, 2, 3], "test": [4, 5, 6, 7]})
    assert list(ok["test"]) == [4, 5, 6, 7]
    with pytest.raises(ValueError, match="share"):
        resolve_splits(ds, {"train": [0, 1, 2], "test": [2, 3]})
    with pytest.raises(ValueError, match="group"):
        resolve_splits(ds, {"train": [0, 2, 3], "test": [1]})
    resolve_splits(ds, {"train": [0, 2, 3], "test": [1]}, group_strict=False)


def test_splits_save_load(tmp_path):
    s = Splits({"train": [3, 1], "val": [0]})
    s.save(tmp_path / "s.npz")
    assert Splits.load(tmp_path / "s.npz").hash() == s.hash()


def test_prior_box_polygon_curve_gaussian(device):
    # support: the band 0.5 <= a + b <= 1.25 of the unit square, and c above the parabola a^2 - 0.5
    poly = Polygon2D([0, 1], [[0.0, 1.0], [0.25, 1.0], [1.0, 0.25], [1.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    curve = CurveBound2D.from_function([0, 2], lambda a: a ** 2 - 0.5, 0.0, 1.0)
    gauss = Gaussian([3], [0.0], [[0.01]])
    p = Prior([0.0, 0.0, -1.0, -0.5], [1.0, 1.0, 1.0, 0.5], [gauss], [poly, curve])
    th = torch.tensor([[0.5, 0.4, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0], [0.5, 0.4, -0.9, 0.0], [0.5, 0.4, 0.0, 0.6]])
    lp = p.log_prob(th)
    assert torch.isfinite(lp[0]) and not torch.isfinite(lp[1:]).any()   # polygon, curve, box
    expected = -np.log(1.0) - np.log(1.0) - np.log(2.0) + (-0.5 * np.log(2 * np.pi * 0.01))
    assert np.isclose(lp[0].item(), expected, atol=1e-4)
    s = p.sample(5000, torch.Generator().manual_seed(0))
    assert p.inside(s).all() and abs(s[:, 3].std().item() - 0.1) < 0.01
    assert (s[:, 2] >= s[:, 0] ** 2 - 0.5 - 1e-5).all() and (s[:, 2] < s[:, 0] ** 2 - 0.3).any()
    q = Prior.from_spec(p.spec())
    assert torch.allclose(q.log_prob(th), lp, equal_nan=True)
    on_dev = p.to(device)                                               # same density on the test device
    assert torch.allclose(on_dev.log_prob(th.to(device)).cpu(), lp, equal_nan=True)
    s = on_dev.sample(2000, torch.Generator(device=device).manual_seed(0))
    assert s.device.type == torch.device(device).type and on_dev.inside(s).all()


def test_curve_bound(device, tmp_path):
    rng = np.random.default_rng(1)
    pts = rng.uniform([-0.5, -2.0], [1.5, 2.0], size=(20000, 2))       # x also outside the nodes [0, 1]
    th = torch.tensor(pts, dtype=torch.float32, device=device)
    fn = lambda x: np.sin(3.0 * x)
    ref = fn(np.clip(pts[:, 0], 0.0, 1.0))                              # constant beyond the end nodes
    clear = np.abs(pts[:, 1] - ref) > 1e-4                              # away from the interpolation error
    above = CurveBound2D.from_function([0, 1], fn, 0.0, 1.0, device=device)
    assert np.array_equal(above(th).cpu().numpy()[clear], (pts[:, 1] >= ref)[clear])
    assert np.abs(above.curve(th[:, 0]).cpu().numpy() - ref).max() < 1e-5
    clear = np.abs(np.abs(pts[:, 1] - ref) - 0.2) > 1e-4
    below = CurveBound2D.from_function([0, 1], fn, 0.0, 1.0, side="below", margin=0.2, device=device)
    assert np.array_equal(below(th).cpu().numpy()[clear], (pts[:, 1] <= ref - 0.2)[clear])
    assert not (above(th) & below(th)).any()
    coarse = CurveBound2D([1, 0], [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], side="below")   # y is column 0 here
    assert coarse(torch.tensor([[0.4, 0.5], [0.6, 0.5], [0.4, 1.5], [0.1, 3.0]])).tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="increasing"):
        CurveBound2D([0, 1], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="side"):
        CurveBound2D([0, 1], [0.0, 1.0], [0.0, 1.0], side="left")
    prior = Prior([-0.5, -2.0], [1.5, 2.0], constraints=[below.to("cpu")])
    save(prior, tmp_path / "p.pt")
    back = load(tmp_path / "p.pt", device)
    assert torch.equal(back.inside(th).cpu(), prior.inside(th.cpu())) and back.constraints[0].side == "below"


def test_prior_save_load_and_kde(tmp_path):
    rng = np.random.default_rng(0)
    kde = KDE([0, 1], rng.normal(size=(300, 2)))
    p = Prior([-5, -5], [5, 5], [kde])
    save(p, tmp_path / "p.pt")
    q = load(tmp_path / "p.pt")
    th = torch.as_tensor(rng.normal(size=(10, 2)), dtype=torch.float32)
    assert torch.allclose(p.log_prob(th), q.log_prob(th))
    # KDE density integrates to ~1 on a grid
    g = torch.linspace(-5, 5, 201)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    dens = kde.log_prob(torch.stack([xx.flatten(), yy.flatten()], 1)).exp().sum() * (g[1] - g[0]) ** 2
    assert abs(dens.item() - 1) < 0.01
