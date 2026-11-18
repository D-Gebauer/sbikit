"""API, validation and failure handling of the network compressors (MSE, VMIM, hierarchical VMIM)."""
import logging
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from sbikit._utils import NotFittedError
from sbikit.compress import (NETWORKS, HierarchicalVMIM, MSECompressor, NetConfig, VMIMCompressor,
                             compressor_from_state, register_network)
from sbikit.io import load, save

CFG = {"widths": (16,), "epochs": 6, "patience": 6, "lr": 3e-3}
NEW_KEYS = ("atoms", "seed", "history", "train_history", "best_val_loss")
NEW_FIELDS = ("val_fraction", "min_delta", "optimizer", "schedule", "network", "network_kwargs")


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1, 1, size=(n, 2))
    x = np.hstack([t, t ** 2, rng.normal(size=(n, 2))]) + 0.05 * rng.normal(size=(n, 6))
    return x.astype(np.float32), t


class _Messages(logging.Handler):
    """Collects the messages of the sbikit logger."""

    def __enter__(self):
        self.lines = []
        logging.getLogger("sbikit").addHandler(self)
        return self.lines

    def __exit__(self, *exc):
        logging.getLogger("sbikit").removeHandler(self)

    def emit(self, record):
        self.lines.append(record.getMessage())


def _old_style(state):
    """A net-compressor state as the first release wrote it (no training record, fewer config fields)."""
    state = {k: v for k, v in state.items() if k not in NEW_KEYS}
    state["config"] = {k: v for k, v in state["config"].items() if k not in NEW_FIELDS}
    state["config"]["widths"] = tuple(state["config"]["widths"])
    return state


# ------------------------------------------------------------------ config
def test_config_typos_and_invalid_values():
    with pytest.raises(TypeError, match="did you mean 'epochs'"):
        NetConfig.make({"epoch": 5})
    with pytest.raises(TypeError, match="did you mean 'widths'"):
        VMIMCompressor(2, {"width": (8,)})
    for bad in ({"dropout": 1.0}, {"lr": 0.0}, {"lr": float("nan")}, {"batch_size": 0}, {"epochs": 2.5},
                {"patience": 0}, {"grad_clip": -1.0}, {"pct_start": 1.0}, {"components": 0}, {"vicreg": -0.1},
                {"val_fraction": 0.0}, {"min_delta": -1.0}, {"optimizer": "sgd"}, {"schedule": "cosine"},
                {"widths": (16, 0)}, {"widths": "16,a"}, {"network_kwargs": 3}, {"weight_decay": "x"},
                {"widths": (2.5,)}, {"widths": 2.5}, {"widths": "2.5"}, {"widths": (float("inf"),)},
                {"widths": (float("nan"),)}, {"batch_size": float("inf")}, {"epochs": float("nan")}):
        with pytest.raises(ValueError):
            NetConfig.make(bad)
    with pytest.raises(ValueError, match="NetConfig.epochs must be an integer, got inf"):
        NetConfig(epochs=float("inf"))
    with pytest.raises(ValueError, match=r"NetConfig.widths must be positive integers.*got \(16, 2.5\)"):
        NetConfig(widths=(16, 2.5))
    with pytest.raises(ValueError, match="did you mean 'zscore'"):
        NetConfig(input_scaling="zscor")
    with pytest.raises(ValueError, match="did you mean 'mlp'.*register_network"):
        NetConfig(network="mpl")
    with pytest.raises(ValueError, match="n_out"):
        MSECompressor(0)


def test_config_normalisation():
    c = NetConfig.make({"widths": "64, 32", "epochs": 6.0, "lr": np.float32(1e-3), "grad_clip": None})
    assert c.widths == (64, 32) and c.epochs == 6 and isinstance(c.epochs, int) and type(c.lr) is float
    assert NetConfig(widths=8).widths == (8,) and NetConfig(widths=[4, 2]).widths == (4, 2)
    for widths, expected in [((16.0, np.int64(8)), (16, 8)), (["16", "8"], (16, 8)), (np.array([8, 4]), (8, 4)),
                             ((w for w in (4, 2)), (4, 2)), ((), ())]:      # integral values of any numeric type
        got = NetConfig(widths=widths).widths
        assert got == expected and all(type(w) is int for w in got)
    d = c.to_dict()
    assert d["widths"] == [64, 32] and d["grad_clip"] is None and NetConfig.make(d) == c
    assert NetConfig.make(c) is c and NetConfig.make(None) == NetConfig()


# ------------------------------------------------------------------ training options
@pytest.mark.parametrize("grad_clip", [None, 0])
def test_grad_clip_off_trains(grad_clip, device):
    x, t = _data()
    comp = MSECompressor(2, {**CFG, "epochs": 15, "patience": 15, "grad_clip": grad_clip, "schedule": "constant"},
                         device, seed=0)
    loss = comp.fit(x[:250], t[:250], x[250:], t[250:])
    assert np.isfinite(loss) and comp.history[-1] < 0.7 * comp.history[0]      # the weights do move


@pytest.mark.parametrize("n, batch_size", [(257, 64), (65, 64), (3, 1)])
def test_single_row_last_batch_with_vicreg(n, batch_size, device):
    x, t = _data(300)
    comp = VMIMCompressor(2, {**CFG, "batch_size": batch_size, "vicreg": 0.5, "epochs": 3}, device, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss = comp.fit(x[:n], t[:n], x[260:], t[260:])
    assert np.isfinite(loss) and np.isfinite(comp.train_history).all() and np.isfinite(comp(x)).all()


@pytest.mark.parametrize("epochs, batch_size", [(1, 300), (2, 300), (3, 300), (10, 300), (5, 150), (4, 100)])
def test_short_onecycle_schedules(epochs, batch_size, device):
    x, t = _data(340)
    comp = MSECompressor(2, {**CFG, "epochs": epochs, "batch_size": batch_size}, device, seed=0)   # pct_start 0.1
    assert np.isfinite(comp.fit(x[:300], t[:300], x[300:], t[300:])) and len(comp.history) == epochs


@pytest.mark.parametrize("extra", [{"optimizer": "adam"}, {"schedule": "constant"}, {"widths": ()}])
def test_training_variants(extra, device):
    x, t = _data()
    comp = VMIMCompressor(2, {**CFG, **extra}, device, seed=0)
    assert np.isfinite(comp.fit(x[:250], t[:250], x[250:], t[250:]))
    assert len(comp.history) == len(comp.train_history) == 6 and comp.best_val_loss == min(comp.history)


# ------------------------------------------------------------------ inputs
def test_fit_input_errors(device):
    x, t = _data(100)
    comp = MSECompressor(2, CFG, device, seed=0)
    for args, match in [((x[:80], t[:79], x[80:], t[80:]), "80 rows but theta has 79"),
                        ((x[:80], t[:80, 0], x[80:], t[80:]), "theta must be 2D"),
                        ((x[0], t[:1], x[80:], t[80:]), "x must be 2D"),
                        ((x[:80], t[:80], x[80:, :5], t[80:]), "x_val has 5 columns"),
                        ((x[:80], t[:80], x[80:], t[80:, :1]), "theta_val has 1 columns"),
                        ((x[:80], t[:80], x[80:], t[81:]), "x_val has 20 rows"),
                        ((x[:80], t[:80], x[80:], None), "given together"),
                        ((x[:80], t[:80], x[:0], t[:0]), "x_val is empty"),
                        ((x[:80], t[:80, :1], x[80:], t[80:, :1]), "n_out = 1")]:
        with pytest.raises(ValueError, match=match):
            comp.fit(*args)
    bad = x.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="x contains NaN or inf in 1 of 80 rows"):
        comp.fit(bad[:80], t[:80], x[80:], t[80:])
    bad_t = t.copy()
    bad_t[90, 0] = np.inf
    with pytest.raises(ValueError, match="theta_val contains NaN or inf"):
        comp.fit(x[:80], t[:80], x[80:], bad_t[80:])
    with pytest.raises(NotFittedError, match="MSECompressor: call fit"):   # none of these left a half-fitted object
        comp(x)
    with pytest.raises(NotFittedError):
        comp.state()


def test_tensor_inputs(device):
    x, t = _data(120)
    as_t = lambda a: torch.tensor(a, device=device, requires_grad=True, dtype=torch.float32)     # noqa: E731
    a = MSECompressor(2, CFG, device, seed=0)
    a.fit(as_t(x[:100]), as_t(t[:100]), as_t(x[100:]), as_t(t[100:]))
    b = MSECompressor(2, CFG, device, seed=0)
    b.fit(x[:100], t[:100].astype(np.float32), x[100:].tolist(), t[100:].astype(np.float32))
    np.testing.assert_allclose(a(as_t(x)), b(x), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="5 columns, the compressor was fitted on 6"):
        a(x[:, :5])
    with pytest.raises(ValueError, match="single"):
        a(x[None])
    assert a(x[:0]).shape == (0, 2)


def test_auto_validation_split(device):
    x, t = _data(300)
    with _Messages() as lines:
        comp = VMIMCompressor(2, {**CFG, "val_fraction": 0.2}, device, seed=4)
        loss = comp.fit(x, t)
    assert np.isfinite(loss) and any("holding out 60 of 300" in line for line in lines)
    again = VMIMCompressor(2, {**CFG, "val_fraction": 0.2}, device, seed=4)
    assert again.fit(x, t) == loss                                   # the split is drawn with the seed
    with pytest.raises(ValueError, match="at least 2"):
        VMIMCompressor(2, CFG, device).fit(x[:1], t[:1])


def test_seed_reproducible_and_global_stream_untouched(device):
    x, t = _data()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    comp = VMIMCompressor(2, CFG, device, seed=0)
    first = comp.fit(x[:250], t[:250], x[250:], t[250:])
    s1 = comp(x)
    assert torch.equal(torch.rand(3), expected)                      # fit did not touch the caller's stream
    assert comp.fit(x[:250], t[:250], x[250:], t[250:]) == first     # a second fit starts afresh
    np.testing.assert_array_equal(comp(x), s1)
    other = VMIMCompressor(2, CFG, device, seed=1)
    other.fit(x[:250], t[:250], x[250:], t[250:])
    assert not np.allclose(other(x), s1)
    a, b = (VMIMCompressor(2, CFG, device) for _ in range(2))        # unseeded: successive fits differ
    a.fit(x[:250], t[:250], x[250:], t[250:])
    b.fit(x[:250], t[:250], x[250:], t[250:])
    assert not np.allclose(a(x), b(x))


# ------------------------------------------------------------------ failures, callbacks, interruption
@pytest.mark.parametrize("cls", [MSECompressor, VMIMCompressor])
def test_failed_fit_leaves_unfitted(cls, device):
    x, t = _data()
    comp = cls(2, CFG, device, seed=0)
    comp.fit(x[:250], t[:250], x[250:], t[250:])                     # fitted first: the failure must undo that
    comp.config = NetConfig.make({**CFG, "input_scaling": "identity"})
    with pytest.warns(RuntimeWarning, match="left unfitted"):
        loss = comp.fit(x[:250] * 1e30, t[:250], x[250:] * 1e30, t[250:])
    assert loss == float("inf") and comp.net is None and comp.best_val_loss == float("inf")
    assert len(comp.history) == 1 and not hasattr(comp, "mdn")       # stopped at the first non-finite epoch
    with pytest.raises(NotFittedError):
        comp(x)
    with pytest.raises(NotFittedError):
        comp.state()
    assert "unfitted" in repr(comp)


def test_divergence_keeps_best_weights(device):
    class Diverging(MSECompressor):
        calls = 0

        def loss(self, s, t):
            self.calls += self.net.training
            return super().loss(s, t) * (float("nan") if self.calls > 8 else 1.0)

    x, t = _data()
    comp = Diverging(2, {**CFG, "batch_size": 125}, device, seed=0)  # 2 steps per epoch: NaN in epoch 5
    with pytest.warns(RuntimeWarning, match="non-finite in epoch 5"):
        loss = comp.fit(x[:250], t[:250], x[250:], t[250:])
    assert np.isfinite(loss) and loss == min(comp.history[:4]) and len(comp.history) == 5
    assert np.isfinite(comp(x)).all()


def test_callback_stop_and_interrupt(device):
    x, t = _data()
    seen = []

    def stop_at_3(info):
        seen.append(info)
        return info["epoch"] == 3

    comp = MSECompressor(2, {**CFG, "epochs": 20, "patience": 20}, device, seed=0)
    loss = comp.fit(x[:250], t[:250], x[250:], t[250:], callback=stop_at_3)
    assert [i["epoch"] for i in seen] == [0, 1, 2, 3] and len(comp.history) == 4
    assert {"epoch", "train_loss", "val_loss", "best", "model"} <= set(seen[-1]) and seen[-1]["model"] is comp
    assert seen[-1]["best"] == loss == min(comp.history) and seen[-1]["train_loss"] == comp.train_history[-1]

    def interrupt(info):
        if info["epoch"] == 3:
            raise KeyboardInterrupt

    other = MSECompressor(2, {**CFG, "epochs": 20, "patience": 20}, device, seed=0)
    with pytest.raises(KeyboardInterrupt):
        other.fit(x[:250], t[:250], x[250:], t[250:], callback=interrupt)
    np.testing.assert_array_equal(other(x), comp(x))                 # the best weights were restored
    assert other.best_val_loss == loss


def test_verbose_logs_each_epoch(device):
    x, t = _data(120)
    with _Messages() as lines:
        MSECompressor(2, {**CFG, "epochs": 3}, device, seed=0).fit(x[:100], t[:100], x[100:], t[100:], verbose=True)
    assert sum("epoch" in line for line in lines) == 3
    with _Messages() as lines:
        MSECompressor(2, {**CFG, "epochs": 3}, device, seed=0).fit(x[:100], t[:100], x[100:], t[100:])
    assert not lines


# ------------------------------------------------------------------ calling
def test_single_vector(device):
    x, t = _data()
    comp = VMIMCompressor(3, CFG, device, seed=0)
    comp.fit(x[:250], t[:250], x[250:], t[250:])
    one = comp(x[7])
    assert one.shape == (3,) and one.dtype == np.float32
    np.testing.assert_allclose(one, comp(x[7:8])[0], rtol=1e-6)
    rng = np.random.default_rng(0)
    wide = rng.normal(size=(12, 8200)).astype(np.float32)             # longer than one evaluation chunk
    comp = VMIMCompressor(2, {**CFG, "widths": (4,), "epochs": 1}, device, seed=0)
    comp.fit(wide[:10], t[:10], wide[10:], t[10:12])
    assert comp(wide[0]).shape == (2,) and comp(wide).shape == (12, 2)
    np.testing.assert_allclose(comp(wide[0]), comp(wide)[0], rtol=1e-5, atol=1e-6)


def test_many_rows_are_chunked(device):
    x, t = _data(200)
    comp = MSECompressor(2, {**CFG, "epochs": 2}, device, seed=0)
    comp.fit(x[:150], t[:150], x[150:], t[150:])
    big = np.tile(x, (45, 1))                                         # 9000 rows: three chunks
    np.testing.assert_allclose(comp(big)[-200:], comp(x), rtol=1e-6, atol=1e-7)


def test_mse_predict_in_parameter_units(device):
    rng = np.random.default_rng(0)
    theta = rng.uniform([10.0, -3.0], [30.0, -1.0], size=(800, 2))   # far from unit scale
    x = (theta @ rng.normal(size=(2, 5)) + 0.01 * rng.normal(size=(800, 5))).astype(np.float32)
    comp = MSECompressor(2, {"widths": (), "epochs": 60, "patience": 60, "lr": 3e-2, "input_scaling": "zscore"},
                         device, seed=0)
    with pytest.raises(NotFittedError, match="MSECompressor: call fit"):
        comp.predict(x[700:])
    comp.fit(x[:700], theta[:700], x[700:], theta[700:])
    est = comp.predict(x[700:])
    assert est.shape == (100, 2) and comp.predict(x[700]).shape == (2,)
    assert np.abs(est - theta[700:]).max() < 0.05 * np.ptp(theta, axis=0).min()
    np.testing.assert_allclose(comp(x[700:]), (est - theta[:700].mean(0)) / theta[:700].std(0), atol=1e-4)


# ------------------------------------------------------------------ networks
def test_registered_network_trains_saves_loads(tmp_path, device):
    def small(n_in, n_out, config):
        hidden = config.network_kwargs["hidden"]
        return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, n_out))

    register_network("test_tanh", small, overwrite=True)
    x, t = _data()
    comp = VMIMCompressor(2, {**CFG, "network": "test_tanh", "network_kwargs": {"hidden": 12}}, device, seed=0)
    assert np.isfinite(comp.fit(x[:250], t[:250], x[250:], t[250:]))
    assert isinstance(comp.net, nn.Sequential) and comp.net[0].out_features == 12
    save(comp, tmp_path / "c.pt")
    back = load(tmp_path / "c.pt", device)
    assert isinstance(back.net, nn.Sequential) and back.config.network_kwargs == {"hidden": 12}
    np.testing.assert_allclose(back(x), comp(x), rtol=1e-5, atol=1e-6)
    del NETWORKS["test_tanh"]                                         # as when the registering module is not imported
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                # one clear error, no "loading it unchecked"
        with pytest.raises(ValueError, match="network 'test_tanh' of this state is not registered.*Import the "
                                             "module.*register_network"):
            VMIMCompressor.from_state(comp.state(), device)
        with pytest.raises(ValueError, match="'test_tanh' of this state is not registered"):
            load(tmp_path / "c.pt", device)
    register_network("test_tanh", small)
    np.testing.assert_allclose(load(tmp_path / "c.pt", device)(x), comp(x), rtol=1e-5, atol=1e-6)
    register_network("test_wrong", lambda n_in, n_out, config: nn.Linear(n_in, n_out + 1), overwrite=True)
    with pytest.raises(ValueError, match=r"expected \(2, 2\)"):
        VMIMCompressor(2, {**CFG, "network": "test_wrong"}, device).fit(x[:250], t[:250], x[250:], t[250:])


def test_net_factory_trains_but_is_not_saveable(device):
    x, t = _data()
    comp = MSECompressor(2, CFG, device, seed=0, net=lambda n_in, n_out: nn.Linear(n_in, n_out))
    assert np.isfinite(comp.fit(x[:250], t[:250], x[250:], t[250:]))
    assert isinstance(comp.net, nn.Linear) and comp(x).shape == (300, 2) and "custom" in repr(comp)
    with pytest.raises(TypeError, match="register_network"):
        comp.state()
    with pytest.raises(TypeError, match="factory"):
        MSECompressor(2, CFG, device, net=nn.Linear(6, 2))


def test_state_roundtrip_and_old_style_state(tmp_path, device):
    x, t = _data()
    t[:60, 1] = 0.0
    comp = VMIMCompressor(2, CFG, device, seed=5)
    loss = comp.fit(x[:250], t[:250], x[250:], t[250:], atoms=[(1, 0.0, 0.1)])
    state = comp.state()
    state["net"]["head.bias"] += 1.0                                  # a copy, not a view of the live weights
    np.testing.assert_array_equal(compressor_from_state(comp.state(), device)(x), comp(x))
    save(comp, tmp_path / "c.pt")
    back = load(tmp_path / "c.pt", device)                            # weights_only: plain values and tensors
    assert back.atoms == [[1, 0.0, 0.1]] and back.seed == 5 and back.best_val_loss == loss
    assert back.history == comp.history and back.train_history == comp.train_history
    assert back.config == comp.config
    old = VMIMCompressor.from_state(_old_style(comp.state()), device)
    np.testing.assert_array_equal(old(x), comp(x))
    assert old.atoms == [] and old.seed is None and old.history == [] and old.best_val_loss is None
    with pytest.raises(ValueError, match="kind 'vmim'"):
        MSECompressor.from_state(comp.state(), device)
    unchecked = _old_style(comp.state())                              # a config from before the range checks
    unchecked["config"]["patience"] = 0
    with pytest.warns(UserWarning, match="patience must be at least 1.*loading it unchecked"):
        old = VMIMCompressor.from_state(unchecked, device)
    np.testing.assert_array_equal(old(x), comp(x))
    assert old.config.patience == 0 and old.config.widths == (16,)


def test_subclass_with_own_config(device):
    from dataclasses import dataclass

    @dataclass
    class HuberConfig(NetConfig):
        delta: float = 1.0

    class Huber(MSECompressor):
        kind = "test_huber"
        config_class = HuberConfig

        def loss(self, s, t):
            return nn.functional.huber_loss(s, t, delta=self.config.delta)

    x, t = _data()
    comp = Huber(2, {**CFG, "delta": 0.5}, device, seed=0)
    assert np.isfinite(comp.fit(x[:250], t[:250], x[250:], t[250:]))
    back = Huber.from_state(comp.state(), device)
    assert back.config.delta == 0.5
    np.testing.assert_array_equal(back(x), comp(x))
    with pytest.raises(TypeError, match="delta"):
        MSECompressor(2, {"delta": 0.5})


# ------------------------------------------------------------------ atoms and degenerate labels
def test_atoms_validation_and_warning(device):
    x, t = _data()
    t = t.astype(np.float32).astype(np.float64)
    t[:80, 1] = np.float32(0.3)                                       # labels stored in single precision
    comp = VMIMCompressor(2, CFG, device, seed=0)
    fit = lambda atoms: comp.fit(x[:250], t[:250], x[250:], t[250:], atoms=atoms)     # noqa: E731
    for atoms, match in [((1, 0.0, 0.1), "triples"), ([(1, 0.0)], "triples"), ([(2, 0.0, 0.1)], "out of range"),
                         ([(1.5, 0.0, 0.1)], "triples"), ([(1, 0.0, -0.1)], "half_width")]:
        with pytest.raises(ValueError, match=match):
            fit(atoms)
    with pytest.warns(UserWarning, match=r"no training row has theta\[:, 1\] == 0.31 \(the closest label is 0.3000"):
        fit([(1, 0.31, 0.1)])
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                # 0.3 matches the single-precision labels
        assert np.isfinite(fit([(1, 0.3, 0.1)]))
        comp.fit(x[:250], t[:250], x[250:], t[250:], atoms=[(1, np.float64(0.3), 0.1)],
                 callback=lambda info: seen.append(info["train_loss"]))
    assert comp.atoms == [[1, 0.3, 0.1]] and seen == comp.train_history
    plain = VMIMCompressor(2, CFG, device, seed=0)
    plain.fit(x[:250], t[:250], x[250:], t[250:])
    assert plain.train_history != comp.train_history                 # the labels were spread


def test_constant_label_column_warns(device):
    x, t = _data()
    t[:, 1] = -1.0
    with pytest.warns(UserWarning, match=r"theta column\(s\) \[1\] are constant"):
        VMIMCompressor(2, {**CFG, "epochs": 2}, device, seed=0).fit(x[:250], t[:250], x[250:], t[250:])


# ------------------------------------------------------------------ hierarchical VMIM
BLOCKS = {"lin": [0, 1], "sq": [2, 3], "noise": [4, 5]}
HCFG = {"default": {"widths": (8,), "epochs": 3, "patience": 3, "lr": 3e-3}}


def test_hierarchical_constructor_validation():
    def make(**kw):
        return HierarchicalVMIM(**{"blocks": BLOCKS, "anchor": ["lin"], "orders": [["sq"]], "n_out": 2, **kw})

    mask = np.zeros(6, bool)
    mask[:2] = True
    for kw, match in [({"blocks": {**BLOCKS, "lin": mask}}, "flatnonzero"),
                      ({"blocks": {**BLOCKS, "lin": [0.0, 1.0]}}, "integer column indices"),
                      ({"blocks": {**BLOCKS, "lin": []}}, "non-empty"),
                      ({"blocks": {**BLOCKS, "lin": [[0, 1]]}}, "1D"),
                      ({"blocks": {}}, "non-empty dict"),
                      ({"blocks": {**BLOCKS, "final": [0]}}, "reserved"),
                      ({"blocks": {**BLOCKS, "anchor": [0]}}, "reserved"),
                      ({"blocks": {**BLOCKS, "default": [0]}}, "reserved"),
                      ({"anchor": ["line"]}, "unknown block 'line' \\(did you mean 'lin'"),
                      ({"orders": [["sq", "nois"]]}, "did you mean 'noise'"),
                      ({"anchor": []}, "anchor is empty"),
                      ({"orders": []}, "orders is empty"),
                      ({"orders": ["sq", ["noise"]]}, "one list of block names per repeat"),
                      ({"configs": {"Final": {"vicreg": 0.1}}}, "did you mean 'final'"),
                      ({"configs": {"sqq": {}}}, "did you mean 'sq'"),
                      ({"configs": {"epochs": 5, "lr": 1e-3}}, "configs=\\{'default'"),
                      ({"n_out": 0}, "n_out"),
                      ({"n_out": 2.5}, "n_out must be a positive integer, got 2.5"),
                      ({"n_anchor": 0}, "n_anchor must be a positive integer, got 0"),
                      ({"n_anchor": -1}, "n_anchor must be a positive integer"),
                      ({"n_anchor": "4"}, "n_anchor must be a positive integer"),
                      ({"n_block": 0}, "n_block must be a positive integer, got 0"),
                      ({"n_block": 2.7}, "n_block must be a positive integer, got 2.7"),
                      ({"n_block": float("inf")}, "n_block must be a positive integer")]:
        with pytest.raises(ValueError, match=match):
            make(**kw)
    h = make(n_out=3)
    assert (h.n_anchor, h.n_block) == (6, 3)                          # the defaults: 2 n_out and n_out
    h = make(n_anchor=5.0, n_block=np.int64(1))
    assert (h.n_anchor, h.n_block) == (5, 1) and type(h.n_anchor) is int and type(h.n_block) is int
    with pytest.raises(TypeError, match="did you mean 'epochs'"):
        make(configs={"final": {"epoch": 5}})
    with pytest.raises(TypeError, match="configs must be a dict"):
        make(configs=[{"epochs": 5}])


def test_hierarchical_shorthands():
    h = HierarchicalVMIM(BLOCKS, anchor="lin", orders=["sq", "noise"], n_out=2, configs=NetConfig(epochs=7),
                         seed=None)
    assert h.anchor == ["lin"] and h.orders == [["sq", "noise"]] and h.seed is None
    assert h.config_for("final").epochs == 7 and h.config_for("sq") is h.default_config
    h = HierarchicalVMIM(BLOCKS, "lin", "sq", 2, configs={"default": {"epochs": 7}, "final": {"vicreg": 0.1}})
    assert h.config_for("sq").epochs == h.config_for("anchor").epochs == 7
    assert h.config_for("final") == NetConfig(vicreg=0.1)            # a stage's entry replaces the default as a whole
    assert HierarchicalVMIM(BLOCKS, "lin", "sq", 2).orders == [["sq"]]
    assert HierarchicalVMIM(BLOCKS, "lin", [[]], 2).orders == [[]]                      # the anchor alone
    assert HierarchicalVMIM({"1": [0], "2": [1, 2]}, "1", ["2", "1"], 2).orders == [["2", "1"]]
    h = HierarchicalVMIM({"lin": range(2), "sq": np.arange(2, 4)}, ["lin"], [["sq"]], 2)
    assert h.blocks["lin"].tolist() == [0, 1] and h.blocks["sq"].dtype == np.int64
    assert "unfitted" in repr(h)
    with pytest.raises(NotFittedError, match="HierarchicalVMIM: call fit"):
        h(np.zeros((2, 6)))
    with pytest.raises(NotFittedError):
        h.state()


def test_hierarchical_fit_roundtrip(tmp_path, device):
    x, t = _data()
    configs = {**HCFG, "final": {**HCFG["default"], "vicreg": 0.1}, "sq": {**HCFG["default"], "epochs": 2}}
    h = HierarchicalVMIM(BLOCKS, "lin", [["sq", "noise"], ["noise", "sq"]], 2, configs=configs, device=device, seed=3)
    seen = []
    with _Messages() as lines:
        loss = h.fit(x, t, callback=lambda info: seen.append((info["stage"], info["epoch"], info["model"] is h,
                                                              type(info["compressor"]).__name__)), verbose=True)
    assert np.isfinite(loss) and loss == h.val_losses["final"] and len(h.stages) == 6 and h.n_in == 6
    assert sum("holding out 30 of 300" in line for line in lines) == 1          # one split for all stages
    assert sum("validation loss" in line for line in lines) == 7                # one line per stage
    assert [s for s, e, _, _ in seen if e == 0] == ["r0/anchor", "r0/sq", "r0/noise", "r1/anchor", "r1/noise",
                                                    "r1/sq", "final"]
    assert all(m and c == "VMIMCompressor" for _, _, m, c in seen) and len(h.stages[1][2].history) == 2
    s = h(x)
    assert s.shape == (300, 2) and h(x[5]).shape == (2,)
    np.testing.assert_allclose(h(x[5]), s[5], rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="5 columns"):
        h(x[:, :5])
    again = HierarchicalVMIM(BLOCKS, "lin", [["sq", "noise"], ["noise", "sq"]], 2, configs=configs, device=device,
                             seed=3)
    assert again.fit(x, t) == loss                                    # same seed, same split, same model
    np.testing.assert_array_equal(again(x), s)

    save(h, tmp_path / "h.pt")
    back = load(tmp_path / "h.pt", device)
    np.testing.assert_allclose(back(x), s, rtol=1e-5, atol=1e-6)
    assert back.config_for("final").vicreg == 0.1 and back.config_for("sq").epochs == 2
    assert back.default_config == h.default_config and back.configs == h.configs and back.n_in == 6
    assert back.val_losses == h.val_losses and "fitted" in repr(back)
    assert back.to(device) is back and back.device == device and back.final.device == device
    assert all(comp.device == device for _, _, comp in back.stages)
    np.testing.assert_allclose(back(x), s, rtol=1e-5, atol=1e-6)

    state = h.state()                                                 # a state as the first release wrote it
    old = {k: v for k, v in state.items() if k not in ("configs", "default_config", "n_in")}
    old["stages"] = [(r, n, _old_style(st)) for r, n, st in state["stages"]]
    old["final"] = _old_style(state["final"])
    old = HierarchicalVMIM.from_state(old, device)
    np.testing.assert_array_equal(old(x), s)
    assert old.default_config == NetConfig() and old.configs == {} and old.n_in is None
    with pytest.raises(ValueError, match="column index 5"):
        old(x[:, :5])
    with pytest.raises(ValueError, match="HierarchicalVMIM.from_state: the state is of kind 'vmim'.*"
                                         "compressor_from_state"):
        HierarchicalVMIM.from_state(h.final.state(), device)
    with pytest.raises(ValueError, match="kind 'hierarchical_vmim'"):
        VMIMCompressor.from_state(h.state(), device)
    named = h.state()                                                 # block names the constructor now refuses
    named["blocks"] = {"anchor": [0, 1], "sq": [2, 3], "noise": [4, 5], "unused": []}
    named["anchor"] = ["anchor"]
    np.testing.assert_array_equal(HierarchicalVMIM.from_state(named, device)(x), s)


def test_hierarchical_fit_checks_and_final_data(device):
    x, t = _data()
    h = HierarchicalVMIM({**BLOCKS, "noise": [4, 6]}, "lin", ["sq"], 2, configs=HCFG, device=device)
    with pytest.raises(ValueError, match=r"blocks\['noise'\] has the column index 6, but x has 6 columns"):
        h.fit(x[:250], t[:250], x[250:], t[250:])
    h = HierarchicalVMIM(BLOCKS, "lin", ["sq"], 2, configs=HCFG, device=device, seed=None)
    for kw, match in [({"final_data": (x, t)}, "final_data must be the tuple"),
                      ({"final_data": (x[:, :5], t, None, None)}, "final_data x has 5 columns"),
                      ({"final_data": (x, t, x, None)}, "given together"),
                      ({"x_val": x[250:]}, "given together")]:
        with pytest.raises(ValueError, match=match):
            h.fit(x[:250], t[:250], **kw)
    with pytest.raises(ValueError, match="x has 250 rows but theta has 249"):
        h.fit(x[:250], t[:249], x[250:], t[250:])
    assert h.final is None
    loss = h.fit(x[:200], t[:200], x[200:250], t[200:250], final_data=(x[:250], t[:250], x[250:], t[250:]))
    assert np.isfinite(loss) and h(x).shape == (300, 2)               # seed=None trains
    assert np.isfinite(h.fit(x[:200], t[:200], x[200:250], t[200:250], final_data=(x, t, None, None)))


def test_hierarchical_failed_stage(device):
    x, t = _data()
    configs = {"default": {**HCFG["default"], "input_scaling": "identity"}}
    h = HierarchicalVMIM(BLOCKS, "lin", ["sq"], 2, configs=configs, device=device)
    assert np.isfinite(h.fit(x[:250], t[:250], x[250:], t[250:]))
    bad = x.copy()
    bad[:, 2:4] *= 1e30                                               # the "sq" stage overflows
    with pytest.warns(RuntimeWarning, match="stage 'r0/sq' reached no finite validation loss"):
        loss = h.fit(bad[:250], t[:250], bad[250:], t[250:])
    assert loss == float("inf") and h.final is None and h.stages == [] and h.val_losses["r0/sq"] == float("inf")
    with pytest.raises(NotFittedError):
        h(x)
