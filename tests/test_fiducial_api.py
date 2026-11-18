"""Input checks, error messages and extension points of the data containers, MOPED and the IMNN."""
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import sbikit as sk
from sbikit import Dataset, FiducialSet, NotFittedError, Splits, random_splits, resolve_splits
from sbikit.compress import IMNN, MOPED, IMNNConfig, compressor_from_state, moped_vectors, register_network
from sbikit.compress.moped import hartlap_factor, moped_vectors_pivoted
from sbikit.io import load, save


def _fiducial(n_fid=300, n_d=60, D=6, K=2, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D, K))
    step = np.full(K, 0.05)
    x = sigma * rng.normal(size=(n_fid, D))
    eps = sigma * rng.normal(size=(n_d, D))                    # seed-matched noise
    x_m = np.stack([(-step * e) @ A.T + eps for e in np.eye(K)])
    x_p = np.stack([(step * e) @ A.T + eps for e in np.eye(K)])
    return FiducialSet(x, list("abcdefgh"[:K]), np.zeros(K), x_m, x_p, 2 * step)


FAST = {"max_iterations": 30, "min_iterations": 5, "patience": 10, "widths": (16,)}


# ------------------------------------------------------------------ Dataset
def test_dataset_checks_inputs():
    with pytest.raises(ValueError, match=r"must be \(N, P\) and \(N, D\)"):
        Dataset(np.zeros(4), np.zeros((4, 2)), ["a"])                      # 1D theta
    with pytest.raises(ValueError, match=r"must be \(N, P\) and \(N, D\)"):
        Dataset(np.zeros((4, 1)), np.zeros(4), ["a"])                      # 1D x
    x = np.zeros((4, 2))
    x[1, 0] = np.nan
    with pytest.raises(ValueError, match="x contains NaN or inf in 1 of 4 rows"):
        Dataset(np.zeros((4, 1)), x, ["a"])
    with pytest.raises(ValueError, match="theta contains NaN"):
        Dataset(np.full((4, 1), np.inf), np.zeros((4, 2)), ["a"])
    ds = Dataset(np.zeros((4, 1)), x, ["a"], allow_nonfinite=True)
    assert len(ds) == 4 and ds.subset([0, 1]).allow_nonfinite              # the opt-out survives subset()
    ds = Dataset(torch.zeros(4, 1, requires_grad=True), torch.ones(4, 2, dtype=torch.float64), "a",
                 groups=torch.arange(4))
    assert ds.theta.dtype == np.float64 and ds.x.dtype == np.float32 and ds.param_names == ["a"]
    assert ds.groups.shape == (4,)
    with pytest.raises(ValueError, match=r"groups must have shape \(4,\), got \(3,\)"):
        Dataset(np.zeros((4, 1)), np.zeros((4, 2)), ["a"], groups=np.arange(3))
    with pytest.raises(ValueError, match="1 parameter columns but 2 names"):
        Dataset(np.zeros((4, 1)), np.zeros((4, 2)), ["a", "b"])


def test_dataset_index_and_blocks():
    ds = Dataset(np.zeros((4, 2)), np.zeros((4, 5)), ["Om", "s8"],
                 blocks={"mask": [False, False, True, False, True], "idx": [4, 0]})
    assert ds.index(["s8", "Om"]) == [1, 0] and ds.index("s8") == [1]
    with pytest.raises(ValueError, match="did you mean 's8'"):
        ds.index(["s88"])
    assert ds.blocks["mask"].tolist() == [2, 4] and ds.blocks["idx"].tolist() == [4, 0]   # order of indices is kept
    with pytest.raises(TypeError, match="block 'b'"):
        Dataset(np.zeros((4, 2)), np.zeros((4, 5)), ["Om", "s8"], blocks={"b": [0.0, 1.0]})
    with pytest.raises(ValueError, match="mask of length 3"):
        Dataset(np.zeros((4, 2)), np.zeros((4, 5)), ["Om", "s8"], blocks={"b": [True, False, True]})
    with pytest.raises(ValueError, match="outside"):
        Dataset(np.zeros((4, 2)), np.zeros((4, 5)), ["Om", "s8"], blocks={"b": [5]})
    with pytest.raises(ValueError, match="no column"):
        Dataset(np.zeros((4, 2)), np.zeros((4, 5)), ["Om", "s8"], blocks={"b": []})


# ------------------------------------------------------------------ FiducialSet
def test_fiducialset_checks_inputs():
    fid = _fiducial()
    args = (fid.param_names, fid.theta_fid, fid.x_minus, fid.x_plus, fid.delta)
    with pytest.raises(ValueError, match=r"x must be \(n_fid, D\)"):
        FiducialSet(fid.x[:, 0], *args)                                     # 1D x
    with pytest.raises(ValueError, match=r"x_minus/x_plus must both be \(2, n_d, 6\), got"):
        FiducialSet(fid.x, fid.param_names, fid.theta_fid, fid.x_minus[0], fid.x_plus[0], fid.delta)
    with pytest.raises(ValueError, match="theta_fid must have shape"):
        FiducialSet(fid.x, fid.param_names, np.zeros(3), fid.x_minus, fid.x_plus, fid.delta)
    for name in ("x", "x_minus", "x_plus", "dmu"):
        arrays = {"x": fid.x.copy(), "x_minus": fid.x_minus.copy(), "x_plus": fid.x_plus.copy(),
                  "dmu": fid.mean_derivative().copy()}
        arrays[name].flat[3] = np.nan
        with pytest.raises(ValueError, match=f"{name} contains NaN or inf"):
            FiducialSet(arrays["x"], fid.param_names, fid.theta_fid, arrays["x_minus"], arrays["x_plus"], fid.delta,
                        arrays["dmu"])
    t = FiducialSet(torch.as_tensor(fid.x), fid.param_names, torch.zeros(2), torch.as_tensor(fid.x_minus),
                    torch.as_tensor(fid.x_plus), torch.as_tensor(fid.delta))
    np.testing.assert_array_equal(t.mean_derivative(), fid.mean_derivative())
    one = FiducialSet(fid.x, "a", 0.3, dmu=np.ones((1, 6)))                # one parameter: a name and a scalar
    assert one.param_names == ["a"] and one.theta_fid.shape == (1,)
    start = (fid.x, fid.param_names, fid.theta_fid)
    for kwargs, message in ((dict(x_minus=fid.x_minus), "give both x_minus and x_plus, or neither"),
                            (dict(x_minus=fid.x_minus, x_plus=fid.x_plus), "delta .* is required"),
                            (dict(x_minus=fid.x_minus, x_plus=fid.x_plus, delta=[0.1]), r"delta must have shape \(2"),
                            (dict(x_minus=fid.x_minus, x_plus=fid.x_plus, delta=[0.1, 0.0]), "non-zero"),
                            (dict(dmu=np.ones((3, 6))), r"dmu must have shape \(2, 6\), got \(3, 6\)"),
                            ({}, "give dmu or x_minus/x_plus")):
        with pytest.raises(ValueError, match=message):
            FiducialSet(*start, **kwargs)


def test_fiducialset_split():
    fid = _fiducial()
    dmu_only = FiducialSet(fid.x, fid.param_names, fid.theta_fid, dmu=fid.mean_derivative())
    trn, val = dmu_only.split(np.arange(300) % 5, val_groups=[0])
    assert len(trn.x) == 240 and len(val.x) == 60 and val.dmu is dmu_only.dmu
    with pytest.warns(UserWarning, match="groups_deriv is ignored"):
        dmu_only.split(np.arange(300) % 5, np.arange(7), [0])
    trn, val = fid.split(np.arange(300) % 5, np.arange(60) % 5, [0, 1])    # positional order as before
    assert val.x.shape == (120, 6) and val.x_minus.shape == (2, 24, 6) and trn.x_plus.shape == (2, 36, 6)
    with pytest.raises(ValueError, match="groups_deriv is required"):
        fid.split(np.arange(300) % 5, val_groups=[0])
    with pytest.raises(ValueError, match=r"groups_fid must have shape \(300,\)"):
        fid.split(np.arange(10), np.arange(60) % 5, [0])
    with pytest.raises(ValueError, match=r"groups_deriv must have shape \(60,\), got \(7,\)"):
        fid.split(np.arange(300) % 5, np.arange(7), [0])
    with pytest.raises(ValueError, match="val_groups is required"):
        fid.split(np.arange(300) % 5, np.arange(60) % 5)
    with pytest.raises(ValueError, match=r"val_groups \[7\] occur in none"):
        fid.split(np.arange(300) % 5, np.arange(60) % 5, [0, 7])
    with pytest.raises(ValueError, match="empty training or validation"):
        fid.split(np.arange(300) % 5, np.zeros(60, int), [0])


# ------------------------------------------------------------------ Splits
def test_splits_normalise_indices():
    mask = np.array([True, False, True, True, False])
    s = Splits({"train": mask, "val": [4, 1]})
    assert s["train"].tolist() == [0, 2, 3] and s["val"].tolist() == [1, 4]       # mask -> indices; sorted
    with pytest.raises(TypeError, match="split 'train'"):
        Splits({"train": [0.0, 1.0]})
    with pytest.raises(ValueError, match="1D"):
        Splits({"train": [[0, 1]]})
    s = Splits(train=[1, 0])
    s["test"] = [3, 2]                                                             # assigned later: same conversion
    s.update(val=torch.tensor([5, 4]))
    s.setdefault("calib", [7, 6])
    for name, expected in (("test", [2, 3]), ("val", [4, 5]), ("calib", [6, 7])):
        assert s[name].dtype == np.int64 and s[name].tolist() == expected
    assert isinstance(s.copy(), Splits) and isinstance(s | {"x": [9]}, Splits) and (s | {"x": [9, 8]})["x"][0] == 8
    left = {"x": [9, 8], "train": [0]} | s                                         # a dict on the left
    assert isinstance(left, Splits) and left["x"].tolist() == [8, 9] and left["train"].tolist() == [0, 1]
    same = s
    s |= {"extra": np.array([False, True, True])}                                  # in place, same conversion
    assert s is same and s.pop("extra").tolist() == [1, 2]
    s.check(8, groups=[0, 0, 1, 1, 2, 2, 3, 3])                                    # groups as a list
    with pytest.raises(ValueError, match=r"groups must have shape \(8,\)"):
        s.check(8, groups=[0, 1])
    with pytest.raises(ValueError, match=r"split 'calib' has indices outside \[0, 7\)"):
        s.check(7)
    with pytest.raises(ValueError, match="split 'twice' repeats indices"):
        Splits(twice=[1, 1, 2]).check(8)
    assert repr(s) == "Splits(train: 2, test: 2, val: 2, calib: 2)"
    assert Splits({"empty": []})["empty"].dtype == np.int64


def test_splits_hash_is_stable():
    # values of version 0.1.0, which run records refer to
    assert Splits({"train": [3, 1], "val": [0]}).hash() == "85246cdf7c276d6c"
    groups = np.repeat(np.arange(50), 4)
    assert random_splits(200, {"train": 0.7, "val": 0.15, "test": 0.15}, groups, seed=3).hash() == "15acca864305e3a1"
    assert random_splits(1000, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1).hash() == "8d171303f1251706"


def test_splits_save_load(tmp_path):
    s = Splits({"train": [3, 1], "file": [0], "allow_pickle": [2]})                # names of np.savez arguments
    s.save(tmp_path / "splits")                                                    # no suffix
    assert (tmp_path / "splits.npz").is_file()
    for path in (tmp_path / "splits", tmp_path / "splits.npz"):
        back = Splits.load(path)
        assert list(back) == list(s) and back.hash() == s.hash()
    np.savez(tmp_path / "old.npz", train=np.array([1, 3]), val=np.array([0]))      # layout of version 0.1.0
    old = Splits.load(tmp_path / "old.npz")
    assert old.hash() == Splits({"train": [3, 1], "val": [0]}).hash()
    with pytest.raises(TypeError, match="split names must be strings to be saved, got 0"):
        Splits({0: [1, 2]}).save(tmp_path / "numbered")
    assert not (tmp_path / "numbered.npz").exists()


def test_random_splits_checks():
    with pytest.raises(ValueError, match=r"groups must have shape \(n,\) = \(10,\)"):
        random_splits(10, {"train": 0.5, "test": 0.5}, groups=np.arange(8))
    with pytest.raises(ValueError, match=r"5 groups are too few .* \['val'\] would be empty"):
        random_splits(20, {"train": 0.8, "val": 0.1, "test": 0.1}, groups=np.repeat(np.arange(5), 4))
    with pytest.raises(ValueError, match="3 simulations are too few"):
        random_splits(3, {"train": 0.8, "val": 0.1, "test": 0.1})
    with pytest.raises(ValueError, match="finite"):
        random_splits(10, {"train": np.nan, "test": 0.5})
    with pytest.raises(ValueError, match="fractions is empty"):
        random_splits(10, {})
    s = random_splits(10, {"train": 1.0, "unused": 0.0})                          # a zero fraction may be empty
    assert len(s["train"]) == 10 and len(s["unused"]) == 0
    ds = Dataset(np.zeros((6, 1)), np.zeros((6, 2)), ["a"])
    with pytest.raises(ValueError, match="too few"):
        resolve_splits(ds)
    mask = np.arange(6) < 4
    r = resolve_splits(ds, {"train": mask, "test": ~mask})
    assert r["train"].tolist() == [0, 1, 2, 3] and r["test"].tolist() == [4, 5]
    with pytest.raises(ValueError, match="mask of length 5"):
        resolve_splits(ds, {"train": mask[:5]})
    with pytest.raises(TypeError, match="integer"):
        resolve_splits(ds, {"train": [0.5, 1.0]})


# ------------------------------------------------------------------ MOPED
def test_moped_too_few_simulations():
    fid = _fiducial()
    with pytest.raises(ValueError, match=r"8 fiducial simulations are too few .* D = 6.*D \+ 2 with.*cov="):
        MOPED(fid.subset(np.arange(8))).fit()
    with pytest.raises(ValueError, match="6 fiducial simulations are too few .* D = 6.*D without"):
        MOPED(fid.subset(np.arange(6)), hartlap=False).fit()
    with pytest.warns(RuntimeWarning, match="Hartlap factor"):                    # allowed, but noisy
        assert np.isnan(MOPED(fid.subset(np.arange(9))).fit())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MOPED(fid).fit()                                                           # enough simulations: no warning


def test_moped_given_covariance():
    fid = _fiducial()
    cov = 0.09 * np.eye(6)
    m = MOPED(fid.subset(np.arange(3)), cov=torch.as_tensor(cov))                 # few sims are enough with cov=
    m.fit()
    np.testing.assert_allclose(m.vectors @ cov @ m.vectors.T, np.eye(2), atol=1e-10)   # no Hartlap factor
    np.testing.assert_allclose(m.inv_cov, np.eye(6) / 0.09)
    with pytest.raises(ValueError, match=r"cov must have shape \(6, 6\)"):
        MOPED(fid, cov=np.eye(5)).fit()
    with pytest.raises(ValueError, match="singular"):
        MOPED(fid, cov=np.ones((6, 6))).fit()
    with pytest.raises(ValueError, match="NaN"):
        MOPED(fid, cov=np.full((6, 6), np.nan)).fit()
    with pytest.raises(ValueError, match="symmetric"):
        MOPED(fid, cov=np.eye(6) + np.triu(np.ones((6, 6)), 1)).fit()


def test_moped_singular_sample_covariance():
    fid = _fiducial()
    x = fid.x.copy()
    x[:, 5] = x[:, 0] + x[:, 1]                                                   # linearly dependent column
    with pytest.raises(ValueError, match="singular"):
        MOPED(FiducialSet(x, fid.param_names, fid.theta_fid, dmu=fid.mean_derivative())).fit()
    x[:, 5] = 1.0                                                                 # constant column
    with pytest.raises(ValueError, match=r"variance in column\(s\) \[5\]"):
        MOPED(FiducialSet(x, fid.param_names, fid.theta_fid, dmu=fid.mean_derivative())).fit()
    with pytest.raises(ValueError, match="not positive definite"):
        moped_vectors(fid.mean_derivative(), -np.eye(6))


def test_moped_names():
    fid = _fiducial(K=3)
    m = MOPED(fid, params="b")                                                    # one parameter as a string
    m.fit()
    assert m.params == ["b"] and m.kept_params == ["b"] and m(fid.x).shape == (300, 1)
    with pytest.raises(ValueError, match="unknown parameter 'bb' .did you mean 'b'"):
        MOPED(fid, params=["a", "bb"])
    with pytest.raises(ValueError, match="pivot_scale: unknown parameter 'cc' .did you mean 'c'"):
        MOPED(fid, pivot_scale={"a": 1.0, "b": 1.0, "c": 1.0, "cc": 2.0})
    with pytest.raises(ValueError, match=r"pivot_scale has no entry for \['c'\]"):
        MOPED(fid, pivot_scale={"a": 1.0, "b": 1.0})
    m = MOPED(fid, params=["a", "b"], pivot_scale={"a": 1.0, "b": 5.0, "c": 2.0})   # widths of all parameters
    m.params.append("zz")                                                           # checked again in fit()
    with pytest.raises(ValueError, match="unknown parameter 'zz'"):
        m.fit()
    with pytest.raises(ValueError, match="MOPED: params is empty"):
        MOPED(fid, params=[])
    with pytest.raises(ValueError, match="MOPED needs a FiducialSet"):
        MOPED().fit()
    with pytest.raises(ValueError, match="pivoting needs orthogonalize=True"):
        MOPED(fid, orthogonalize=False, pivot_scale={"a": 1.0, "b": 1.0, "c": 1.0}).fit()


def test_moped_degenerate_and_unfitted(tmp_path):
    fid = _fiducial()
    zero = FiducialSet(fid.x, fid.param_names, fid.theta_fid, dmu=np.zeros((2, 6)))
    m = MOPED(zero)
    with pytest.warns(UserWarning, match="degenerate"):
        with pytest.raises(ValueError, match="every parameter is degenerate"):
            m.fit()
    for use in (lambda: m(fid.x), m.state, lambda: save(m, tmp_path / "m.pt")):
        with pytest.raises(NotFittedError, match="MOPED: call fit"):
            use()
    assert "unfitted" in repr(m) and m.to("cpu") is m
    with pytest.warns(UserWarning, match="degenerate"):
        vecs, kept = moped_vectors(np.zeros((2, 6)), np.eye(6))
    assert vecs.shape == (0, 6) and kept == []


def test_moped_pivoted_degenerate_and_nonfinite():
    fid = _fiducial(K=3)
    dmu = fid.mean_derivative().copy()
    dmu[2] = 2.0 * dmu[0] - dmu[1]                                                # c lies in the span of a and b
    m = MOPED(FiducialSet(fid.x, fid.param_names, fid.theta_fid, dmu=dmu), pivot_scale={"a": 1.0, "b": 1.0, "c": 1.0})
    with pytest.warns(UserWarning, match="dropping degenerate parameter"):
        m.fit()
    assert len(m.kept_params) == 2 and m.vectors.shape == (2, 6) and m.scores.shape == (2,)
    assert m(fid.x).shape == (300, 2) and np.all(np.diff(m.scores) <= 0)          # best-measured direction first
    with pytest.warns(UserWarning, match="dropping degenerate parameter '1'"):   # nothing is left: empty result
        vecs, order, scores = moped_vectors_pivoted(np.zeros((2, 6)), np.eye(6))
    assert vecs.shape == (0, 6) and order == [] and scores.shape == (0,)
    with pytest.raises(ValueError, match=r"scale must have shape \(3,\), got \(2,\)"):
        moped_vectors_pivoted(dmu, np.eye(6), scale=[1.0, 2.0])
    bad = fid.mean_derivative().copy()
    bad[1, 2] = np.inf
    for vectors in (moped_vectors, moped_vectors_pivoted):
        with np.errstate(invalid="ignore"):
            with pytest.raises(ValueError, match="derivative of parameter 'b' or the inverse covariance is not "
                                                 "finite"):
                vectors(bad, np.eye(6), names=fid.param_names)
            with pytest.raises(ValueError, match="not finite"):
                vectors(fid.mean_derivative(), np.full((6, 6), np.nan))


def test_moped_shapes_and_state():
    fid = _fiducial()
    m = MOPED(fid, pivot_scale={"a": 1.0, "b": 3.0})
    m.fit()
    y = m(fid.x)
    assert m(fid.x[0]).shape == (2,) and np.array_equal(m(fid.x[0]), y[0])        # a single data vector
    assert m(torch.as_tensor(fid.x[:4])).shape == (4, 2) and m(fid.x[:0]).shape == (0, 2)
    with pytest.raises(ValueError, match="has 5 columns, the compressor was fitted on 6"):
        m(fid.x[:, :5])
    with pytest.raises(ValueError, match="got shape"):
        m(fid.x[None])
    state = m.state()
    back = compressor_from_state(state)
    np.testing.assert_array_equal(back.scores, m.scores)
    np.testing.assert_array_equal(back(fid.x), y)
    state["vectors"].zero_()                                                       # the state holds copies
    assert np.abs(m.vectors).sum() > 0
    old = {k: v for k, v in m.state().items() if k != "scores"}                    # a state of version 0.1.0
    assert compressor_from_state(old).scores is None
    np.testing.assert_array_equal(compressor_from_state(old)(fid.x), y)
    assert repr(m) == f"MOPED(params={m.kept_params}, D=6)"


def test_moped_hartlap_variance_and_one_column():
    fid = _fiducial(n_fid=40)
    m = MOPED(fid, params=["b"])                                                   # default hartlap=True, a subset
    m.fit()
    np.testing.assert_allclose(np.var(m(fid.x).astype(np.float64), ddof=1), hartlap_factor(40, 6), rtol=1e-5)
    m = MOPED(fid, hartlap=False)
    m.fit()
    np.testing.assert_allclose(np.cov(m(fid.x).T.astype(np.float64)), np.eye(2), atol=1e-5)
    one = FiducialSet(fid.x[:, :1], ["a"], [0.0], dmu=[[2.0]])                     # a one-column data vector
    m = MOPED(one, hartlap=False)
    m.fit()
    np.testing.assert_allclose(np.var(m(one.x).astype(np.float64), ddof=1), 1.0, rtol=1e-5)


# ------------------------------------------------------------------ IMNN
def test_imnn_config_validation():
    with pytest.raises(TypeError, match="did you mean 'patience'"):
        IMNNConfig.make({"patiense": 3})
    with pytest.raises(ValueError, match="no longer exist"):
        IMNNConfig.make({"val_fraction": 0.1})
    with pytest.raises(ValueError, match="no longer exist"):
        IMNNConfig.make(None, epochs=10)
    with pytest.raises(ValueError, match="output = 'MLE'.*valid: .'mle', 'raw'."):
        IMNNConfig(output="MLE")
    with pytest.raises(ValueError, match="unknown input_scaling 'zscor' .did you mean 'zscore'"):
        IMNNConfig(input_scaling="zscor")
    with pytest.raises(ValueError, match="unknown network"):
        IMNNConfig(network="no_such_net")
    for bad in ({"lam": 0.5}, {"eps": 0.0}, {"patience": 0}, {"max_iterations": 0}, {"dropout": 1.0}, {"lr": 0.0},
                {"n_summaries": 0}, {"batch_fid": 0}, {"batch_deriv": 0}, {"grad_clip": -1.0}, {"asinh_k": 0.0},
                {"widths": (16, 0)}):
        with pytest.raises(ValueError):
            IMNNConfig.make(bad)
    for bad, message in (({"lr": None}, "IMNNConfig.lr must be a number, got None"),
                         ({"lam": "ten"}, "IMNNConfig.lam must be a number, got 'ten'"),
                         ({"dropout": None}, "IMNNConfig.dropout must be a number"),
                         ({"grad_clip": "x"}, "IMNNConfig.grad_clip must be a number"),
                         ({"lr": True}, "IMNNConfig.lr must be a number"),
                         ({"patience": 2.5}, "IMNNConfig.patience must be an integer, got 2.5"),
                         ({"min_iterations": None}, "IMNNConfig.min_iterations must be an integer"),
                         ({"max_iterations": "100"}, "IMNNConfig.max_iterations must be an integer"),
                         ({"n_summaries": 2.5}, "IMNNConfig.n_summaries must be None or an integer"),
                         ({"batch_fid": True}, "IMNNConfig.batch_fid must be None or an integer"),
                         ({"widths": "a,b"}, "IMNNConfig.widths must be positive integers"),
                         ({"widths": None}, "IMNNConfig.widths must be positive integers"),
                         ({"widths": (16.5,)}, "IMNNConfig.widths must be positive integers"),
                         ({"network_kwargs": [1]}, "IMNNConfig.network_kwargs must be a dict, got list"),
                         ({"lr": float("inf")}, "lr must be positive")):
        with pytest.raises(ValueError, match=message):
            IMNNConfig.make(bad)
    c = IMNNConfig.make({"lr": "1e-3", "eps": np.float32(0.5), "grad_clip": "2", "patience": 20.0, "widths": 32,
                         "n_summaries": np.int64(3), "batch_fid": 50.0, "network_kwargs": None})
    values = (c.lr, c.eps, c.grad_clip, c.patience, c.n_summaries, c.batch_fid)
    assert values == (1e-3, 0.5, 2.0, 20, 3, 50) and [type(v) for v in values] == [float] * 3 + [int] * 3
    assert c.widths == (32,) and c.network_kwargs == {}
    c = IMNNConfig.make({"widths": "32,16", "grad_clip": 0})
    assert c.widths == (32, 16) and c.asinh_k == 5.0 and c.network == "imnn_mlp" and c.network_kwargs == {}
    assert IMNNConfig.make(c) is c and IMNNConfig.make(c, lr=1e-2).lr == 1e-2
    assert IMNNConfig.make(c.to_dict()) == c


def test_imnn_set_and_batch_sizes():
    fid = _fiducial()
    trn, val = fid.split(np.arange(300) % 5, np.arange(60) % 5, [0])
    with pytest.raises(ValueError, match="n_summaries = 1 is below the number of parameters"):
        IMNN(fid, {"n_summaries": 1})
    with pytest.raises(ValueError, match="n_summaries = 1 is below"):
        IMNN(fid, {"n_summaries": 1, "output": "raw"})
    with pytest.raises(ValueError, match=r"the validation set has 3 fiducial simulations.*n_summaries \+ 1 = 3"):
        IMNN(trn, validation=val.subset(np.arange(3)))
    with pytest.raises(ValueError, match="the training set has 5 fiducial"):
        IMNN(trn.subset(np.arange(5)), {"n_summaries": 4})
    with pytest.raises(ValueError, match="batch_fid has 3 fiducial"):
        IMNN(trn, {"batch_fid": 3})
    with pytest.raises(ValueError, match="batch_deriv = 49 exceeds the 48 derivative pairs"):
        IMNN(trn, {"batch_deriv": 49})
    with pytest.raises(ValueError, match="data vectors of length 5"):
        IMNN(trn, validation=FiducialSet(val.x[:, :5], val.param_names, val.theta_fid, val.x_minus[..., :5],
                                         val.x_plus[..., :5], val.delta))
    with pytest.raises(ValueError, match="theta_fid"):
        IMNN(trn, validation=FiducialSet(val.x, val.param_names, val.theta_fid + 1, val.x_minus, val.x_plus,
                                         val.delta))
    with pytest.raises(ValueError, match="validation set: different parameters or derivative steps"):
        IMNN(trn, validation=FiducialSet(val.x, ["a", "z"], val.theta_fid, val.x_minus, val.x_plus, val.delta))
    with pytest.raises(ValueError, match="validation set: different parameters or derivative steps"):
        IMNN(trn, validation=FiducialSet(val.x, val.param_names, val.theta_fid, val.x_minus, val.x_plus,
                                         2 * val.delta))
    with pytest.raises(ValueError, match="derivative sims"):
        IMNN(FiducialSet(fid.x, fid.param_names, fid.theta_fid, dmu=fid.mean_derivative()))
    same = FiducialSet(np.ones((50, 6)), fid.param_names, fid.theta_fid, fid.x_minus, fid.x_plus, fid.delta)
    with pytest.raises(ValueError, match="fiducial simulations of the training set are all identical"):
        IMNN(same, FAST)
    with pytest.raises(ValueError, match="fiducial simulations of the validation set are all identical"):
        IMNN(trn, FAST, validation=same)
    small = IMNN(trn, FAST, validation=val.subset(np.arange(12)))                 # allowed, but noisy
    with pytest.warns(RuntimeWarning, match="the validation set has only 12"):
        small.fit()
    with pytest.raises(ValueError, match="needs a FiducialSet"):
        IMNN().fit()


def test_imnn_seed_and_global_rng(device):
    fid = _fiducial()
    trn, val = fid.split(np.arange(300) % 5, np.arange(60) % 5, [0])
    imnn = IMNN(trn, {**FAST, "batch_fid": 100, "batch_deriv": 20, "dropout": 0.1}, device, seed=4, validation=val)
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    loss = imnn.fit()
    assert torch.equal(torch.rand(3), expected)                                   # the caller's stream is untouched
    first, history = imnn(fid.x[:20]), list(imnn.train_history)
    assert imnn.fit() == loss and np.array_equal(imnn(fid.x[:20]), first)         # the same model again
    assert imnn.train_history == history and len(history) == imnn.iterations == len(imnn.history)
    assert imnn.val_loss == loss and "fitted: " in repr(imnn)
    other = IMNN(trn, imnn.config, device, seed=5, validation=val)
    other.fit()
    assert not np.array_equal(other(fid.x[:20]), first)


def test_imnn_call_shapes_and_state(device, tmp_path):
    fid = _fiducial()
    for output, n_out in (("mle", 2), ("raw", 3)):
        imnn = IMNN(fid, {**FAST, "n_summaries": 3, "output": output, "input_scaling": "robust_asinh",
                          "asinh_k": 3.0}, device, seed=0)
        assert "unfitted" in repr(imnn)
        for use in (lambda: imnn(fid.x), imnn.state):
            with pytest.raises(NotFittedError, match="IMNN: call fit"):
                use()
        imnn.fit()
        assert imnn.in_scaler.k == 3.0 and imnn.n_out == n_out
        y = imnn(fid.x[:8])
        assert y.shape == (8, n_out) and y.dtype == np.float32
        assert imnn(fid.x[0]).shape == (n_out,) and np.allclose(imnn(fid.x[0]), y[0], atol=1e-5)
        assert imnn(torch.as_tensor(fid.x[:8])).shape == (8, n_out)
        assert imnn(fid.x[:0]).shape == (0, n_out)                                 # an empty batch
        with pytest.raises(ValueError, match="has 5 columns, the compressor was fitted on 6"):
            imnn(fid.x[:, :5])
        state = imnn.state()
        assert state["iterations"] == imnn.iterations and state["seed"] == 0 and "history" not in state
        assert state["val_loss"] == imnn.val_loss and state["config"]["asinh_k"] == 3.0
        save(imnn, tmp_path / "imnn.pt")
        back = load(tmp_path / "imnn.pt", device)
        np.testing.assert_allclose(back(fid.x[:8]), y, rtol=1e-4, atol=1e-5)
        assert back.iterations == imnn.iterations and back.seed == 0 and back.to(device) is back
        for v in state["net"].values():                                            # the state holds copies
            v.zero_()
        state["fisher"].zero_()
        np.testing.assert_array_equal(imnn(fid.x[:8]), y)
        # a state of version 0.1.0: without the keys and config fields added since
        old = imnn.state()
        for k in ("iterations", "val_loss", "seed"):
            del old[k]
        for k in ("asinh_k", "network", "network_kwargs"):
            del old["config"][k]
        old["config"]["batch_fid"] = 0
        back = compressor_from_state(old, device)
        np.testing.assert_allclose(back(fid.x[:8]), y, rtol=1e-4, atol=1e-5)
        assert back.iterations is None and back.config.batch_fid is None and "fitted" in repr(back)
        again = back.state()                                                       # and can be saved again
        assert again["iterations"] is None and again["val_loss"] is None and again["seed"] is None
        save(back, tmp_path / "again.pt")
        np.testing.assert_array_equal(load(tmp_path / "again.pt", device)(fid.x[:8]), back(fid.x[:8]))
    old["config"]["output"] = "score"                                              # a name no version knows: "mle"
    assert compressor_from_state(old, device).config.output == "mle"


def test_imnn_old_checkpoint_saves_again(device, tmp_path):
    data = Path(__file__).parent / "data" / "v0.1.0"
    with np.load(data / "reference.npz") as f:
        x, expected = f["x"], f["imnn"]
    old = load(data / "imnn.pt", device)
    assert old.iterations is None and old.val_loss is None and old.seed is None
    np.testing.assert_allclose(old(x), expected, rtol=1e-4, atol=1e-5)
    save(old, tmp_path / "imnn.pt")
    back = load(tmp_path / "imnn.pt", device)
    np.testing.assert_array_equal(back(x), old(x))
    assert back.state()["iterations"] is None and repr(back).endswith("fitted)")


def test_imnn_state_is_plain_and_independent(device, tmp_path):
    fid = _fiducial()
    imnn = IMNN(fid, {**FAST, "grad_clip": 1.0}, device, seed=np.int64(3))        # e.g. a seed from rng.integers()
    assert np.isfinite(imnn.fit())
    state = imnn.state()
    assert type(state["seed"]) is int and state["seed"] == 3
    torch.save(state, tmp_path / "state.pt")                                       # plain values and tensors only
    assert torch.load(tmp_path / "state.pt", weights_only=True)["seed"] == 3
    y = imnn(fid.x[:8])
    first, second = IMNN.from_state(state, device), IMNN.from_state(state, device)
    stats = ("mu", "cov", "dmu", "fisher", "theta_fid")
    assert not any(getattr(first, k) is state[k] or getattr(first, k) is getattr(second, k) for k in stats)
    assert all(getattr(first, k).device.type == "cpu" and getattr(first, k).dtype == torch.float64 for k in stats)
    for k in stats:                                                                # changing the state afterwards
        state[k].add_(1.0)
    first.mu.add_(1.0)                                                             # ... or one of the copies
    np.testing.assert_allclose(second(fid.x[:8]), y, rtol=1e-4, atol=1e-5)
    unclipped = IMNN(fid, {**FAST, "grad_clip": 0}, device, seed=3)
    reference = IMNN(fid, FAST, device, seed=3)
    assert unclipped.fit() == reference.fit()                                      # 0 and None: no clipping
    tiny = IMNN(fid, {**FAST, "lr": 1.0, "grad_clip": 1e-12}, device, seed=3)       # the clipping is applied
    large = IMNN(fid, {**FAST, "lr": 1.0}, device, seed=3)
    tiny.fit(), large.fit()
    assert tiny.train_history[0] == large.train_history[0] and tiny.train_history[5] != large.train_history[5]


def test_imnn_custom_networks(device, tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self, n_in, n_out, hidden=8):
            super().__init__()
            self.body = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(),
                                            torch.nn.Linear(hidden, n_out))

        def forward(self, x):
            assert x.ndim == 2                                                     # networks see (N, n_in)
            return self.body(x)

    register_network("tiny_test_net", lambda n_in, n_out, c: Tiny(n_in, n_out, **c.network_kwargs), overwrite=True)
    fid = _fiducial()
    imnn = IMNN(fid, {**FAST, "network": "tiny_test_net", "network_kwargs": {"hidden": 12}}, device, seed=0)
    assert np.isfinite(imnn.fit()) and isinstance(imnn.net, Tiny) and imnn.net.body[0].out_features == 12
    save(imnn, tmp_path / "tiny.pt")
    back = load(tmp_path / "tiny.pt", device)
    assert isinstance(back.net, Tiny)
    np.testing.assert_allclose(back(fid.x[:8]), imnn(fid.x[:8]), rtol=1e-4, atol=1e-5)

    direct = IMNN(fid, FAST, device, seed=0, net=lambda n_in, n_out: Tiny(n_in, n_out))   # the expert shortcut
    assert np.isfinite(direct.fit()) and direct(fid.x[:8]).shape == (8, 2) and "custom net" in repr(direct)
    with pytest.raises(TypeError, match="register_network"):
        direct.state()
    wrong = IMNN(fid, FAST, device, seed=0, net=lambda n_in, n_out: Tiny(n_in, n_out + 1))
    with pytest.raises(ValueError, match="the network must map"):
        wrong.fit()


def test_imnn_callback_and_interruption(device):
    fid = _fiducial()
    seen = []

    def stop_at_5(info):
        seen.append(info)
        return info["epoch"] == 5

    imnn = IMNN(fid, FAST, device, seed=0)
    loss = imnn.fit(callback=stop_at_5)
    assert imnn.iterations == 6 and [i["epoch"] for i in seen] == list(range(6))
    assert set(seen[0]) >= {"epoch", "train_loss", "val_loss", "best", "model"} and seen[0]["model"] is imnn
    assert loss == min(i["val_loss"] for i in seen) == seen[-1]["best"]
    y = imnn(fid.x[:8])                                                            # the best weights, usable

    lines, handler = [], logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    logging.getLogger("sbikit").addHandler(handler)
    try:
        IMNN(fid, {**FAST, "max_iterations": 100, "patience": 100}, device, seed=0).fit(verbose=True)
        n_verbose = len(lines)
        IMNN(fid, FAST, device, seed=0).fit()                                      # silent by default
    finally:
        logging.getLogger("sbikit").removeHandler(handler)
    assert len(lines) == n_verbose == 2 and lines[0].startswith("IMNN iteration 100: loss ")
    assert lines[1].startswith("IMNN: stopped after 100 iterations (max_iterations), log det F = ")
    assert lines[1].endswith("on the training set")

    def interrupt_at_5(info):
        if info["epoch"] == 5:
            raise KeyboardInterrupt

    again = IMNN(fid, FAST, device, seed=0)
    with pytest.raises(KeyboardInterrupt):
        again.fit(callback=interrupt_at_5)
    assert again.iterations == 6 and again.val_loss == loss
    np.testing.assert_array_equal(again(fid.x[:8]), y)                             # best weights were restored


def test_imnn_failed_fit_and_warnings(device):
    fid = _fiducial()
    flat = FiducialSet(fid.x, fid.param_names, fid.theta_fid, fid.x_minus, fid.x_minus, fid.delta)   # no signal
    imnn = IMNN(flat, FAST, device, seed=0)
    with pytest.warns(RuntimeWarning, match="never positive definite"):
        assert imnn.fit() == float("inf")
    with pytest.raises(NotFittedError):
        imnn(fid.x)
    with pytest.raises(NotFittedError):
        imnn.state()

    class Flat(torch.nn.Module):
        """Linear, but with constant outputs (a singular summary covariance) in eval mode
        (``in_eval``) or from the ``after``-th training call on."""

        def __init__(self, n_in, n_out, in_eval=False, after=None):
            super().__init__()
            self.lin, self.in_eval, self.after, self.calls = torch.nn.Linear(n_in, n_out), in_eval, after, 0

        def forward(self, x):
            self.calls += self.training
            flat = self.in_eval if not self.training else (self.after is not None and self.calls > self.after)
            return self.lin(x) * (0.0 if flat else 1.0)

    trn, val = fid.split(np.arange(300) % 5, np.arange(60) % 5, [0])
    cases = {"training loss, first iteration": (dict(after=0), None, "never positive definite in 0 iterations and "
                                                                     "the covariance of the summaries became singular"),
             "validation, every iteration": (dict(in_eval=True), val, "never positive definite in"),
             "final statistics": (dict(in_eval=True), None, "whole training set is singular")}
    failed = {}
    for case, (kw, validation, message) in cases.items():
        imnn = failed[case] = IMNN(trn, FAST, device, seed=0, validation=validation,
                                   net=lambda i, o, kw=kw: Flat(i, o, **kw))
        with pytest.warns(RuntimeWarning, match=message):
            assert imnn.fit() == float("inf"), case                                # pruned like a diverged fit
        assert imnn.net is None and "unfitted" in repr(imnn) and imnn.iterations == len(imnn.history), case
        with pytest.raises(NotFittedError):
            imnn(fid.x)
    history = failed["validation, every iteration"].history                        # trained until patience ran out
    assert 10 < len(history) < 30 and all(np.isnan(h["val_logdetF"]) and np.isfinite(h["loss"]) for h in history)
    assert failed["final statistics"].iterations > 0
    late = IMNN(trn, FAST, device, seed=0, net=lambda i, o: Flat(i, o, after=9))   # three calls per iteration
    with pytest.warns(RuntimeWarning, match="stopped in iteration 4 because the covariance of the summaries became"):
        loss = late.fit()
    assert np.isfinite(loss) and late.iterations == 3 and late(fid.x[:4]).shape == (4, 2)   # the best weights are kept
    big = FiducialSet(1e4 * fid.x, fid.param_names, fid.theta_fid, 1e4 * fid.x_minus, 1e4 * fid.x_plus, fid.delta)
    with pytest.warns(RuntimeWarning, match="input_scaling='zscore'"):
        IMNN(big, {**FAST, "max_iterations": 2}, device, seed=0).fit()
    with pytest.warns(RuntimeWarning, match="without a validation set"):
        IMNN(fid, {**FAST, "max_iterations": 2, "batch_fid": 100}, device, seed=0).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                             # the reference setup: no warning
        IMNN(fid, {**FAST, "max_iterations": 2}, device, seed=0).fit()
    assert sk.compress.NETWORKS["imnn_mlp"](6, 2, IMNNConfig()).layers[0].in_features == 6
