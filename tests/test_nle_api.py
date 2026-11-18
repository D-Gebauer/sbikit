"""NLE and Ensemble: configs, flow registry, seeding, input checks, training options, scalers, states."""
import logging
import warnings

import numpy as np
import pytest
import torch

from sbikit import NLE, Ensemble, FlowConfig, NotFittedError, TrainConfig
from sbikit.io import load, save
from sbikit.nle import FLOWS, register_flow
from sbikit.scaling import RobustAsinh, ZScore

from sbi.neural_nets.net_builders import build_zuko_maf  # isort: skip  (after sbikit, which keeps TensorFlow out)

SMALL = {"model": "maf", "hidden_features": 16, "num_transforms": 2}
FAST = {"max_epochs": 3, "batch_size": 100}


def _data(n=400, n_x=3, n_theta=2, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1, 1, size=(n, n_theta))
    s = theta @ np.random.default_rng(99).normal(size=(n_theta, n_x)) + 0.3 * rng.normal(size=(n, n_x))
    return s, theta


def _fit(device, seed=0, flow=SMALL, train=FAST, n=400, **kwargs):
    s, theta = _data(n)
    nle = NLE(flow=flow, device=device, **kwargs)
    nle.fit(s[:-80], theta[:-80], s[-80:], theta[-80:], train, seed=seed)
    return nle


@pytest.fixture(scope="module")
def trained(device):
    return _fit(device)


@pytest.fixture
def log(caplog):
    """Records of the sbikit logger (which does not propagate to the root logger)."""
    logger = logging.getLogger("sbikit")
    logger.addHandler(caplog.handler)
    level = logger.level
    logger.setLevel(logging.DEBUG)
    yield caplog
    logger.setLevel(level)
    logger.removeHandler(caplog.handler)


class GaussianEstimator(torch.nn.Module):
    """Minimal estimator: independent Gaussians whose mean is linear in the condition."""

    def __init__(self, n_x, n_theta):
        super().__init__()
        self.mean = torch.nn.Linear(n_theta, n_x)
        self.log_sd = torch.nn.Parameter(torch.zeros(n_x))

    def log_prob(self, input, condition):
        z = (input.reshape(-1, input.shape[-1]) - self.mean(condition)) / self.log_sd.exp()
        return (-0.5 * z ** 2 - self.log_sd - 0.5 * np.log(2 * np.pi)).sum(-1).reshape(1, -1)

    def loss(self, input, condition):
        return -self.log_prob(input, condition)[0]


# ------------------------------------------------------------------ configs
def test_unknown_config_keys_raise_with_suggestion():
    with pytest.raises(TypeError, match="did you mean 'hidden_features'"):
        FlowConfig.make({"hiden_features": 32})
    with pytest.raises(TypeError, match="did you mean 'num_transforms'"):
        NLE(3, 2, {"model": "maf", "num_transform": 3})
    with pytest.raises(TypeError, match="did you mean 'batch_size'"):
        TrainConfig.make({"batchsize": 64})
    s, theta = _data(50)
    with pytest.raises(TypeError, match="did you mean 'max_epochs'"):
        NLE(3, 2, SMALL).fit(s, theta, train={"max_epoch": 5})
    with pytest.raises(TypeError, match="Fields"):
        TrainConfig.make({"learning_rate": 1e-3})


@pytest.mark.parametrize("cls, bad, match", [
    (FlowConfig, {"model": "nfs"}, "did you mean 'nsf'"),
    (FlowConfig, {"hidden_features": 0}, "hidden_features"),
    (FlowConfig, {"num_blocks": 1.5}, "num_blocks"),
    (FlowConfig, {"num_transforms": -1}, "num_transforms"),
    (FlowConfig, {"kwargs": "residual"}, "kwargs"),
    (TrainConfig, {"lr": 0}, "lr"),
    (TrainConfig, {"lr": float("nan")}, "lr"),
    (TrainConfig, {"batch_size": 0}, "batch_size"),
    (TrainConfig, {"max_epochs": 0}, "max_epochs"),
    (TrainConfig, {"stop_after_epochs": 0}, "stop_after_epochs"),
    (TrainConfig, {"sched_factor": 1.0}, "sched_factor"),
    (TrainConfig, {"grad_clip": -1}, "grad_clip"),
    (TrainConfig, {"val_fraction": 1.0}, "val_fraction"),
    (TrainConfig, {"optimizer": "adamm"}, "did you mean 'adam'"),
    (TrainConfig, {"scheduler": "cosine"}, "valid"),
])
def test_invalid_config_values_raise(cls, bad, match):
    with pytest.raises(ValueError, match=match):
        cls.make(bad)


def test_config_normalisation_and_storage():
    flow = FlowConfig.make({"hidden_features": np.int64(32), "num_transforms": 3.0, "kwargs": {"residual": True}})
    assert type(flow.hidden_features) is int and type(flow.num_transforms) is int
    assert flow.to_dict()["kwargs"] == {"residual": True}
    train = TrainConfig.make({"batch_size": np.int64(64), "lr": np.float32(1e-3), "grad_clip": 0})
    assert type(train.batch_size) is int and type(train.lr) is float and not train.grad_clip
    assert TrainConfig(grad_clip=None).grad_clip is None
    assert NLE(flow="maf").flow_config.model == "maf"
    assert TrainConfig.make(train) is train and TrainConfig.make(None) == TrainConfig()


# ------------------------------------------------------------------ flows
def test_every_registered_flow_trains(device):
    s, theta = _data(200)
    assert {"maf", "nsf", "naf", "unaf", "nice", "gf", "bpf", "ncsf", "sospf"} <= set(FLOWS)
    for model in ("maf", "nsf", "naf", "unaf", "nice", "gf", "bpf", "ncsf", "sospf"):
        nle = NLE(3, 2, {"model": model, "hidden_features": 8, "num_transforms": 2}, device, seed=0)
        val, epochs = nle.fit(s, theta, train={"max_epochs": 2, "batch_size": 100})
        ll = nle.log_likelihood(s[:7], theta[:7])
        assert np.isfinite(val) and epochs == 2 and ll.shape == (7,) and torch.isfinite(ll).all(), model
        assert nle.sample(theta[:2], n=3, seed=0).shape == (3, 2, 3), model


def test_flow_kwargs_reach_the_builder():
    plain_maf, residual = NLE(3, 2, SMALL), NLE(3, 2, {**SMALL, "kwargs": {"residual": True}})
    assert set(plain_maf.flow.state_dict()) != set(residual.flow.state_dict())
    widths = NLE(3, 2, {"model": "maf", "num_transforms": 1, "kwargs": {"hidden_features": [24, 12]}})
    assert sorted({p.shape[0] for p in widths.flow.parameters() if p.ndim == 2}) == [6, 12, 24]
    with pytest.raises(TypeError):
        NLE(3, 2, {"model": "maf", "kwargs": {"no_such_option": 1}})


def test_register_flow_save_load(tmp_path, device):
    def tiny(batch_x, batch_theta, config):
        return build_zuko_maf(batch_x, batch_theta, z_score_x="none", z_score_y="none", num_transforms=1,
                              hidden_features=[config.kwargs.get("width", 8)])

    register_flow("test_tiny", tiny)
    try:
        register_flow("test_tiny", tiny)                               # the same builder again is fine
        with pytest.raises(ValueError, match="already registered"):
            register_flow("test_tiny", lambda *a: None)
        register_flow("test_tiny2", overwrite=True)(tiny)              # decorator form
        assert FLOWS["test_tiny2"] is tiny
        nle = _fit(device, flow={"model": "test_tiny", "kwargs": {"width": 12}})
        assert any(p.shape[0] == 12 for p in nle.flow.parameters())
        save(nle, tmp_path / "nle.pt")
        torch.manual_seed(3)
        expected = torch.rand(3)
        torch.manual_seed(3)
        back = load(tmp_path / "nle.pt", device)
        assert torch.equal(torch.rand(3), expected)                    # loading leaves the global RNG alone
        s, theta = _data(20, seed=5)
        assert torch.equal(back.log_likelihood(s, theta), nle.log_likelihood(s, theta))
        assert back.flow_config == nle.flow_config and back.train_config == nle.train_config
        assert back.history == nle.history and back.train_history == nle.train_history
        assert (back.best_val_loss, back.epochs, back.seed, back.fitted) == (nle.best_val_loss, 3, None, True)
    finally:
        FLOWS.pop("test_tiny", None)
        FLOWS.pop("test_tiny2", None)
    with pytest.raises(ValueError, match="register_flow"):
        load(tmp_path / "nle.pt", device)                              # the flow is not registered any more


def test_builder_callable_and_instance(device):
    s, theta = _data(300)

    def builder(batch_x, batch_theta):
        return build_zuko_maf(batch_x, batch_theta, z_score_x="none", z_score_y="none", hidden_features=[8],
                              num_transforms=1)

    nle = NLE(flow=builder, device=device, seed=0)
    assert nle.flow is None and nle.flow_config is None and "builder" in repr(nle)
    val, _ = nle.fit(s, theta, train=FAST)
    assert np.isfinite(val) and nle.fitted and nle.log_likelihood(s[:4], theta[:4]).shape == (4,)
    with pytest.raises(TypeError, match="register_flow"):
        nle.state()

    est = GaussianEstimator(3, 2)
    before = est.mean.weight.detach().clone()
    nle = NLE(flow=est, device=device)
    val1, _ = nle.fit(s, theta, train=FAST, seed=0)
    assert nle.flow is est and not torch.equal(est.mean.weight.detach().cpu(), before)
    val2, _ = nle.fit(s, theta, train=FAST, seed=0)                    # an instance is trained on, not re-initialised
    assert val2 < val1 and (nle.n_x, nle.n_theta) == (3, 2)
    with pytest.raises(TypeError, match="register_flow"):
        nle.state()
    with pytest.raises(NotImplementedError):
        nle.sample(theta[:2])
    with pytest.raises(TypeError, match="torch.nn.Module"):
        NLE(flow=torch.nn.Linear(2, 3))
    with pytest.raises(TypeError, match="torch.nn.Module"):
        NLE(3, 2, flow=lambda bx, bt: "not an estimator")

    replaced = _fit(device)
    replaced.flow = GaussianEstimator(3, 2)                            # a replaced flow does not match the config   
    assert replaced.flow_config is None
    with pytest.raises(TypeError, match="register_flow"):
        replaced.state()
    fancy = _fit(device, flow={**SMALL, "kwargs": {"activation": torch.nn.ELU}})
    with pytest.raises(TypeError, match="kwargs"):
        fancy.state()


# ------------------------------------------------------------------ seeding
def test_fit_seed_fixes_initialisation_and_training(device):
    s, theta = _data(300)
    a, b = NLE(flow=SMALL, device=device), NLE(flow=SMALL, device=device)
    assert a.fit(s, theta, train=FAST, seed=4) == b.fit(s, theta, train=FAST, seed=4)
    ll = a.log_likelihood(s, theta)
    assert torch.equal(ll, b.log_likelihood(s, theta))
    a.fit(s, theta, train=FAST, seed=4)                                # the same object again: not stacked
    assert torch.equal(a.log_likelihood(s, theta), ll)
    a.fit(s, theta, train=FAST, seed=5)
    assert not torch.equal(a.log_likelihood(s, theta), ll)
    c = NLE(3, 2, SMALL, device, seed=4)                               # constructor seed: built at construction
    d = NLE(3, 2, SMALL, device, seed=4)
    assert all(torch.equal(p, q) for p, q in zip(c.flow.parameters(), d.flow.parameters()))
    c.fit(s, theta, train=FAST)
    assert torch.equal(c.log_likelihood(s, theta), ll)


def test_seeded_fit_leaves_global_rng_alone():
    s, theta = _data(200)
    torch.manual_seed(11)
    expected = torch.rand(5)
    torch.manual_seed(11)
    NLE(3, 2, SMALL, seed=0).fit(s, theta, train=FAST, seed=1)
    assert torch.equal(torch.rand(5), expected)


def test_members_trained_in_a_loop_differ(device):
    s, theta = _data(300)
    lls = []
    for seed in (0, 1, None, None):
        nle = NLE(flow=SMALL, device=device)
        nle.fit(s[:250], theta[:250], s[250:], theta[250:], FAST, seed)
        lls.append(nle.log_likelihood(s, theta))
    assert all(not torch.equal(lls[i], lls[j]) for i in range(4) for j in range(i))


def test_unseeded_fit_follows_the_global_torch_seed(device):
    s, theta = _data(300)

    def run():                                                         # no seed and no validation set
        nle = NLE(flow=SMALL, device=device)
        nle.fit(s, theta, train=FAST)
        return nle

    with torch.random.fork_rng():
        torch.manual_seed(7)
        a, b = run(), run()
        torch.manual_seed(7)
        c = run()
    assert torch.equal(a.s_scaler.mean, c.s_scaler.mean) and a.history == c.history
    assert torch.equal(a.log_likelihood(s, theta), c.log_likelihood(s, theta))
    assert not torch.equal(a.s_scaler.mean, b.s_scaler.mean)           # each unseeded fit holds out other rows


# ------------------------------------------------------------------ fit: inputs
def test_auto_validation_split_and_inferred_dimensions(device, log):
    s, theta = _data(400)
    nle = NLE(flow=SMALL, device=device, seed=0)
    assert (nle.n_x, nle.n_theta, nle.flow, nle.train_config, nle.fitted) == (None, None, None, None, False)
    val, epochs = nle.fit(s, theta, train={**FAST, "val_fraction": 0.25})
    assert "holding out 100 of 400" in log.text
    assert (nle.n_x, nle.n_theta, nle.fitted, epochs) == (3, 2, True, 3)
    assert nle.train_config.val_fraction == 0.25 and len(nle.s_scaler.mean) == 3
    again = NLE(flow=SMALL, device=device, seed=0)
    assert again.fit(s, theta, train={**FAST, "val_fraction": 0.25})[0] == val        # the split is seeded too
    with pytest.raises(ValueError, match="both s_val and theta_val"):
        nle.fit(s, theta, s[:10])
    with pytest.raises(ValueError, match="3"):
        nle.fit(s[:, :2], theta)                                       # the inferred dimensions are kept            
    assert NLE(n_x=np.int64(3)).n_x == 3
    with pytest.raises(ValueError, match="n_x"):
        NLE(n_x=0)


def test_fit_shape_and_nan_errors(device):
    s, theta = _data(100)
    nle = NLE(3, 2, SMALL, device)
    bad = s.copy()
    bad[3, 1] = np.nan
    for args, match in [((s, theta[:90]), "rows"), ((s[:, :2], theta), "columns"), ((s, theta[:, :1]), "columns"),
                        ((s[:, 0], theta), "2D"), ((bad, theta), "s contains NaN"), ((s[:0], theta[:0]), "empty"),
                        ((s, theta, bad, theta), "s_val contains NaN"), ((s, theta, s[:5], theta[:4]), "rows"),
                        ((s, theta, s[:5, :2], theta[:5]), "columns")]:
        with pytest.raises(ValueError, match=match):
            nle.fit(*args, train=FAST)
    assert not nle.fitted
    with pytest.raises(ValueError, match="ZZ|unknown scaler"):
        NLE(s_scaler="ZZ")


def test_fit_accepts_tensors_with_grad(device):
    s, theta = _data(200)
    st = torch.tensor(s, dtype=torch.float32, device=device, requires_grad=True)
    tt = torch.tensor(theta, device=device)
    a, b = NLE(flow=SMALL, device=device), NLE(flow=SMALL, device=device)
    a.fit(st * 1.0, tt, train=FAST, seed=0)
    b.fit(st.detach().cpu().numpy(), theta, train=FAST, seed=0)
    assert torch.equal(a.log_likelihood(st, tt), b.log_likelihood(s.astype(np.float32), theta))


def test_constant_column_warns(device):
    s, theta = _data(100)
    s[:, 1] = 2.5
    with pytest.warns(RuntimeWarning, match=r"columns \[1\] are constant"):
        NLE(flow=SMALL, device=device).fit(s, theta, train={"max_epochs": 1})


# ------------------------------------------------------------------ fit: failure, warm start, options
def test_use_before_fit():
    nle = NLE(3, 2, SMALL)
    s, theta = _data(5)
    for call in (lambda: nle.log_likelihood(s, theta), lambda: nle.val_loss(s, theta), lambda: nle.state(),
                 lambda: nle.log_prob(torch.zeros(1, 5, 3), torch.zeros(5, 2)), lambda: nle.sample(theta),
                 lambda: nle.fit(s, theta, warm_start=True)):
        with pytest.raises(NotFittedError, match="call fit"):
            call()
    assert issubclass(NotFittedError, RuntimeError) and "not fitted" in repr(nle)
    with pytest.raises(ValueError, match="not fitted"):
        Ensemble([nle])


def test_failed_fit_warns_and_leaves_unfitted(device):
    s, theta = _data(200)
    nle = _fit(device)
    with pytest.warns(RuntimeWarning, match="no epoch reached a finite validation loss"):
        val, epochs = nle.fit(s, theta, train={"max_epochs": 5, "grad_clip": None},
                              optimizer=lambda params: torch.optim.SGD(params, lr=1e30))
    assert val == float("inf") and epochs == 1 and not nle.fitted and nle.best_val_loss == float("inf")
    with pytest.raises(NotFittedError):
        nle.log_likelihood(s, theta)
    with pytest.raises(NotFittedError):
        nle.state()


def test_warm_start(device):
    s, theta = _data(400)
    trn, val = (s[:320], theta[:320]), (s[320:], theta[320:])
    nle = NLE(flow=SMALL, device=device, seed=0)
    first, _ = nle.fit(*trn, *val, FAST)
    scaler = nle.s_scaler
    second, epochs = nle.fit(*trn, *val, FAST, warm_start=True)
    assert second < first and epochs == 3 and len(nle.history) == 3 and nle.s_scaler is scaler
    # a warm start that only gets worse keeps the weights it started from
    weights = {k: v.clone() for k, v in nle.flow.state_dict().items()}
    with pytest.warns(RuntimeWarning, match="non-finite"):
        third, _ = nle.fit(*trn, *val, {"max_epochs": 3, "grad_clip": None}, warm_start=True,
                           optimizer=lambda params: torch.optim.SGD(params, lr=1e30))
    assert third == pytest.approx(second, abs=1e-5) and nle.fitted and nle.s_scaler is scaler
    assert all(torch.equal(v, weights[k]) for k, v in nle.flow.state_dict().items())
    nle.fit(*trn, *val, FAST)                                          # a plain fit starts over
    assert nle.s_scaler is not scaler and nle.best_val_loss == first


def test_custom_optimizer_scheduler_and_callback(device, log):
    s, theta = _data(300)
    seen, infos = [], []

    def optimizer(params):
        seen.append("optimizer")
        return torch.optim.SGD(params, lr=0.05, momentum=0.9)

    def callback(info):
        infos.append(info)
        return info["epoch"] == 4

    nle = NLE(flow=SMALL, device=device, seed=0)
    val, epochs = nle.fit(s, theta, train={"max_epochs": 50}, optimizer=optimizer, callback=callback, verbose=True,
                          scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5))
    assert seen == ["optimizer"] and epochs == 4 and len(nle.history) == len(nle.train_history) == 4
    assert {"epoch", "train_loss", "val_loss", "best", "lr", "model"} <= set(infos[0]) and infos[0]["model"] is nle
    np.testing.assert_allclose([i["lr"] for i in infos], [0.05, 0.025, 0.0125, 0.00625])
    assert [i["val_loss"] for i in infos] == nle.history and val == min(nle.history) == infos[-1]["best"]
    assert log.text.count("NLE epoch") == 4 and "stopped by the callback" in log.text
    # the config's own choices
    lrs = []
    nle.fit(s, theta, train={**FAST, "optimizer": "adamw", "scheduler": "none", "lr": 2e-3, "weight_decay": 0.01},
            callback=lambda info: lrs.append(info["lr"]))
    assert lrs == [2e-3] * 3


def test_wrong_typed_training_hooks_and_scaler_classes_raise(device):
    s, theta = _data(100)
    nle = NLE(flow=SMALL, device=device)
    for kwargs, match in [({"optimizer": "adam"}, r"optimizer must be None or a function.*train=\{'optimizer': ...\}"),
                          ({"scheduler": "none"}, r"scheduler must be None or a function.*train=\{'scheduler': ...\}"),
                          ({"scheduler": 3}, "scheduler must be None or a function"),
                          ({"callback": 3}, "callback must be None or a function"),
                          ({"optimizer": lambda params: None}, "optimizer must return")]:
        with pytest.raises(TypeError, match=match):
            nle.fit(s, theta, train=FAST, **kwargs)
        assert not nle.fitted and nle.epochs is None and nle.history == []     # refused before any training
    for kwargs in ({"s_scaler": ZScore}, {"t_scaler": RobustAsinh}):
        with pytest.raises(TypeError, match=r"got the class \w+; pass an instance, e.g. \w+\(\)"):
            NLE(flow=SMALL, **kwargs)


def test_plateau_scheduler_matches_early_stopping(device, monkeypatch):
    made = []

    class Spy(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def __init__(self, *args, **kwargs):
            made.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", Spy)
    _fit(device, train={**FAST, "min_delta": 1e-3, "sched_patience": 2})
    assert made == [{"factor": TrainConfig().sched_factor, "patience": 2, "threshold": 1e-3, "threshold_mode": "abs"}]


def test_interrupt_restores_best_weights(device):
    s, theta = _data(300)
    best = {}

    def callback(info):
        if info["val_loss"] == info["best"]:
            best.update({k: v.clone() for k, v in info["model"].flow.state_dict().items()})
            best["val"] = info["model"].val_loss(s[250:], theta[250:])         # the model is usable in a callback
        if info["epoch"] == 3:
            raise KeyboardInterrupt

    nle = NLE(flow=SMALL, device=device, seed=0)
    with pytest.raises(KeyboardInterrupt):
        nle.fit(s[:250], theta[:250], s[250:], theta[250:], {"max_epochs": 20}, callback=callback,
                optimizer=lambda params: torch.optim.SGD(params, lr=0.05),     # the third epoch has a far too high lr
                scheduler=lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 if e < 2 else 200.0))
    assert nle.fitted and nle.epochs == 3 and nle.best_val_loss == min(nle.history[:2])
    assert not nle.history[2] < nle.best_val_loss + 0.1                        # the last epoch was not the best
    assert all(torch.equal(v, best[k]) for k, v in nle.flow.state_dict().items())
    assert nle.val_loss(s[250:], theta[250:]) == pytest.approx(best["val"])
    assert best["val"] == pytest.approx(nle.best_val_loss, abs=1e-4)


def test_interrupt_before_the_first_epoch_leaves_unfitted(device):
    s, theta = _data(100)

    class Interrupted(torch.optim.SGD):
        def step(self, *args, **kwargs):
            raise KeyboardInterrupt

    nle = NLE(flow=SMALL, device=device, seed=0)
    with pytest.raises(KeyboardInterrupt):
        nle.fit(s, theta, train=FAST, optimizer=lambda params: Interrupted(params, lr=0.1))
    assert not nle.fitted and nle.history == [] and nle.best_val_loss == np.inf and not nle.flow.training
    with pytest.raises(NotFittedError):
        nle.log_likelihood(s, theta)
    nle.fit(s, theta, train=FAST)                                      # and it can be fitted afterwards
    assert nle.fitted and len(nle.history) == 3


def test_loss_is_density_of_raw_summaries(device):
    s, theta = _data(400)
    fits = []
    for scale in (1.0, 10.0):
        nle = NLE(flow=SMALL, device=device, seed=0)
        val, _ = nle.fit(scale * s[:320], theta[:320], scale * s[320:], theta[320:], FAST)
        assert val == min(nle.history) == nle.best_val_loss
        assert val == pytest.approx(nle.val_loss(scale * s[320:], theta[320:]), abs=1e-4)
        fits.append((val, nle.train_history[-1]))
    # the same flow on the same scaled values: the losses differ by the log-volume factor of the units
    np.testing.assert_allclose(np.subtract(fits[1], fits[0]), 3 * np.log(10.0), atol=2e-3)


def test_scaling_in_float64(device):
    s, theta = _data(300)
    a, b = NLE(flow=SMALL, device=device), NLE(flow=SMALL, device=device)
    a.fit(s, theta, train=FAST, seed=0)
    b.fit(s + 1e7, theta, train=FAST, seed=0)                         # float32 resolves 1e7 only to 1, the std is ~0.5
    np.testing.assert_allclose(b.log_likelihood(s + 1e7, theta).cpu(), a.log_likelihood(s, theta).cpu(), atol=2e-3)


# ------------------------------------------------------------------ evaluation
def test_log_likelihood_shapes(trained):
    s, theta = _data(30, seed=3)
    ll = trained.log_likelihood(s, theta)
    assert ll.shape == (30,) and ll.dtype == torch.float32 and not ll.requires_grad
    assert torch.allclose(trained.log_likelihood(torch.tensor(s), torch.tensor(theta, requires_grad=True)), ll)
    one = trained.log_likelihood(s[0], theta[0])
    assert one.shape == (1,) and torch.allclose(one, ll[:1])
    every = trained.log_likelihood(s[0], theta)                       # one summary vector at every theta
    assert every.shape == (30,) and torch.allclose(every, trained.log_likelihood(np.repeat(s[:1], 30, 0), theta))
    assert torch.allclose(trained.log_likelihood(s[:1], theta), every)
    assert trained.val_loss(s, theta) == pytest.approx(-ll.mean().item())
    for args, match in [((s, theta[0]), "shapes"), ((s[:, :2], theta), "columns"), ((s, theta[:, :1]), "columns"),
                        ((s[:5], theta), "rows"), ((s[0, :2], theta[0]), "columns"), ((s[None], theta), "shapes"),
                        ((s[:0], theta[:0]), "empty")]:
        with pytest.raises(ValueError, match=match):
            trained.log_likelihood(*args)
    with pytest.raises(ValueError, match="expected input"):
        trained.log_prob(torch.zeros(1, 4, 1, device=trained.device), torch.zeros(4, 2, device=trained.device))


def test_log_prob_layout_and_jacobian(trained):
    from torchemcee.integrations.sbi import flow_log_prob_fn
    s, theta = _data(6, seed=4)
    st, tt = (torch.tensor(a, dtype=torch.float32, device=trained.device) for a in (s, theta))
    walkers = tt[:, None, :].repeat(1, 5, 1).requires_grad_()
    lp = flow_log_prob_fn(trained, st)(walkers)                        # (targets, walkers)
    assert lp.shape == (6, 5) and lp.requires_grad
    ll = trained.log_likelihood(s, theta)
    assert torch.allclose(lp[:, 0], ll, atol=1e-5) and torch.allclose(trained.log_prob(st, tt), ll[None], atol=1e-5)
    flow_only = trained.flow.log_prob(trained.s_scaler(st)[None], condition=trained.t_scaler(tt))[0]
    assert torch.allclose(ll, flow_only - float(torch.log(trained.s_scaler.std).sum()), atol=1e-4)


def test_log_prob_shape_errors(trained):
    s, theta = _data(6, seed=4)
    st, tt = (torch.tensor(a, dtype=torch.float32, device=trained.device) for a in (s, theta))
    ll = trained.log_likelihood(s, theta)
    for inp, cond, shape in [(st, tt, (1, 6)), (st[None], tt, (1, 6)), (st[:1], tt, (1, 6)), (st, tt[:1], (1, 6)),
                             (st.reshape(2, 3, 3), tt[:3], (2, 3))]:           # sbi's layouts stay
        assert trained.log_prob(inp, cond).shape == shape
    assert torch.allclose(trained.log_prob(s, theta), ll[None], atol=1e-5)     # arrays are accepted too
    for inp, cond, match in [(st[0], tt[0], "expected input"), (st, tt[0], "expected input"),
                             (torch.tensor(1.0), tt, "expected input"), (st[None, None], tt, "expected input"),
                             (st, tt[None], "expected input"), (st[:, :2], tt, "expected input"),
                             (st, tt[:, :1], "expected input"), (st[None, :4], tt[:3], "input has 4 rows but"),
                             (st[:4], tt[:3], "input has 4 rows but condition has 3")]:
        with pytest.raises(ValueError, match=match):
            trained.log_prob(inp, cond)


@pytest.mark.parametrize("scaler", ["zscore", "robust_asinh", "identity", None])
def test_density_is_normalised_for_every_scaler(scaler, device):
    rng = np.random.default_rng(0)
    theta = rng.uniform(-1, 1, size=(500, 1))
    s = 40.0 + 5.0 * theta + 3.0 * rng.standard_t(4, size=(500, 1))
    nle = NLE(flow=SMALL, device=device, seed=0, s_scaler=scaler, t_scaler=scaler)
    nle.fit(s, theta, train=FAST)
    grid = np.linspace(-400, 480, 40001)
    p = np.exp(nle.log_likelihood(grid[:, None], np.full((len(grid), 1), 0.3)).cpu().numpy().astype(np.float64))
    assert np.sum(p) * (grid[1] - grid[0]) == pytest.approx(1.0, abs=5e-3)


def test_scaler_instances_and_roundtrip(tmp_path, device):
    mine = RobustAsinh(k=3.0)
    nle = _fit(device, s_scaler=mine, t_scaler=ZScore())
    assert mine.median is None and nle.s_scaler is not mine                # the instance is copied, then fitted
    assert nle.s_scaler.k == 3.0 and nle.s_scaler.median is not None
    save(nle, tmp_path / "nle.pt")
    back = load(tmp_path / "nle.pt", device)
    s, theta = _data(20, seed=6)
    assert isinstance(back.s_scaler, RobustAsinh)
    assert torch.equal(back.log_likelihood(s, theta), nle.log_likelihood(s, theta))
    back.fit(s, theta, train=FAST)                                     # a loaded model refits with its kind of scaler
    assert isinstance(back.s_scaler, RobustAsinh) and back.s_scaler.k == 3.0
    with pytest.raises(TypeError, match="s_scaler"):
        NLE(s_scaler=3)


def test_sample(trained):
    s, theta = _data(400)
    draws = trained.sample(theta[:50], n=20, seed=1)
    assert draws.shape == (20, 50, 3) and np.array_equal(draws, trained.sample(theta[:50], n=20, seed=1))
    assert not np.array_equal(draws, trained.sample(theta[:50], n=20, seed=2))
    assert np.all(np.abs(draws.mean((0, 1)) - s[:50].mean(0)) < s.std(0))          # raw units
    with pytest.raises(ValueError, match="2D"):
        trained.sample(theta[0])


def test_more_rows_than_one_evaluation_chunk(trained, device):
    from sbikit import nle as nle_module
    s, theta = _data(90, seed=5)
    reps = nle_module._CHUNK // 90 + 2
    big_s, big_theta = np.tile(s, (reps, 1)), np.tile(theta, (reps, 1))
    assert len(big_s) > nle_module._CHUNK
    ll = trained.log_likelihood(s, theta)
    big = trained.log_likelihood(big_s, big_theta)                     # several chunks, joined in order
    assert big.shape == (len(big_s),) and torch.allclose(big, ll.repeat(reps), atol=1e-5)
    assert torch.allclose(big, trained.log_prob(big_s, big_theta)[0].float(), atol=1e-5)       # one direct call
    every = trained.log_likelihood(s[0], big_theta)                    # one summary vector at many theta rows
    assert every.shape == (len(big_s),) and torch.allclose(every[:90], trained.log_likelihood(s[0], theta), atol=1e-5)
    assert trained.val_loss(big_s, big_theta) == pytest.approx(trained.val_loss(s, theta), abs=1e-5)
    draws = trained.sample(big_theta, n=2, seed=0)
    assert draws.shape == (2, len(big_s), 3) and np.isfinite(draws).all()
    assert np.array_equal(draws, trained.sample(big_theta, n=2, seed=0))
    first = trained.sample(big_theta[:nle_module._CHUNK], n=2, seed=0)             # chunks are joined along the rows
    assert np.array_equal(draws[:, :nle_module._CHUNK], first)
    assert not np.array_equal(draws[:, nle_module._CHUNK:], first[:, :len(big_s) - nle_module._CHUNK])
    # a validation set of several chunks gives the loss of the same rows in one chunk
    small = NLE(flow=SMALL, device=device, seed=0)
    val, _ = small.fit(s, theta, s, theta, FAST)
    many = NLE(flow=SMALL, device=device, seed=0)
    assert many.fit(s, theta, big_s, big_theta, FAST)[0] == pytest.approx(val, abs=1e-5)
    np.testing.assert_allclose(many.history, small.history, atol=1e-5)


def test_devices_and_repr(trained, device):
    assert repr(trained) == f"NLE(model='maf', n_x=3, n_theta=2, device={device!r}, fitted)"
    assert trained.to("cpu").device == "cpu" and next(trained.flow.parameters()).device.type == "cpu"
    assert trained.to(device) is trained and trained.device == device
    auto = NLE(3, 2, SMALL, device="auto")
    assert auto.device == ("cuda" if torch.cuda.is_available() else auto.device) and auto.device != "auto"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            NLE(3, 2, SMALL, device="cuda")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            trained.to("cuda")


# ------------------------------------------------------------------ states
def test_state_does_not_alias_live_weights(device):
    nle = _fit(device)
    state = nle.state()
    frozen = {k: v.clone() for k, v in state["flow"].items()}
    mean = state["s_scaler"]["mean"].clone()
    with torch.no_grad():
        for p in nle.flow.parameters():
            p.add_(1.0)
        nle.s_scaler.mean.add_(1.0)
    assert all(torch.equal(state["flow"][k], v) for k, v in frozen.items())
    assert torch.equal(state["s_scaler"]["mean"], mean)
    nle.fit(*_data(100), train=FAST)                                   # a refit leaves an earlier state alone too
    assert all(torch.equal(state["flow"][k], v) for k, v in frozen.items())


def test_old_style_state_loads(trained):
    s, theta = _data(20, seed=7)
    state = trained.state()
    assert state["train_config"]["max_epochs"] == 3 and len(state["history"]) == 3 and state["epochs"] == 3
    for key in ("train_config", "history", "train_history", "best_val_loss", "epochs", "seed"):
        state.pop(key)
    state["flow_config"].pop("kwargs")
    old = NLE.from_state(state, trained.device)
    assert torch.equal(old.log_likelihood(s, theta), trained.log_likelihood(s, theta)) and old.fitted
    assert (old.train_config, old.history, old.train_history, old.best_val_loss, old.epochs, old.seed) == \
        (None, [], [], None, None, None)


# ------------------------------------------------------------------ ensembles
def test_ensemble_is_a_mixture_of_proper_densities(tmp_path, device):
    s, theta = _data(1500)
    wide = _fit(device, seed=0)
    narrow = NLE(flow=SMALL, device=device, seed=1)                    # trained on other simulations: other scalers
    keep = np.abs(theta).max(1) < 0.4
    narrow.fit(s[keep], theta[keep], train=FAST)
    assert torch.log(wide.s_scaler.std / narrow.s_scaler.std).sum() > 1.0
    st, tt = (torch.tensor(a, dtype=torch.float32, device=device) for a in (s[:40], theta[:40]))

    def by_hand(member):
        with torch.no_grad():
            z, y = (st - member.s_scaler.mean.to(st)) / member.s_scaler.std.to(st), member.t_scaler(tt)
            return (member.flow.log_prob(z[None], condition=y)[0] - torch.log(member.s_scaler.std).sum().to(st)).cpu()

    members = torch.stack([by_hand(wide), by_hand(narrow)])
    ens = Ensemble([wide, narrow])
    mixture = torch.logsumexp(members, 0) - np.log(2)
    assert torch.allclose(ens.log_likelihood(s[:40], theta[:40]).cpu(), mixture, atol=1e-4)
    assert (len(ens), ens.n_x, ens.n_theta, ens.fitted, ens.device) == (2, 3, 2, True, wide.device)
    assert "2 members" in repr(ens) and ens.val_loss(s[:40], theta[:40]) == pytest.approx(
        -ens.log_likelihood(s[:40], theta[:40]).mean().item())
    weighted = Ensemble([wide, narrow], weights=[3, 1])
    np.testing.assert_allclose(weighted.weights, [0.75, 0.25])
    expected = torch.logsumexp(members + torch.log(torch.tensor([0.75, 0.25]))[:, None], 0)
    assert torch.allclose(weighted.log_likelihood(s[:40], theta[:40]).cpu(), expected, atol=1e-4)
    first = Ensemble([wide, narrow], [1, 0]).log_likelihood(s[:40], theta[:40]).cpu()
    assert torch.allclose(first, members[0], atol=1e-4)
    assert weighted.log_prob(st[None], tt).shape == (1, 40)

    save(Ensemble([weighted, wide], [1, 2]), tmp_path / "ens.pt")     # an ensemble of an ensemble and an NLE
    back = load(tmp_path / "ens.pt", device)
    assert isinstance(back.members[0], Ensemble) and isinstance(back.members[1], NLE)
    np.testing.assert_allclose(back.weights, [1 / 3, 2 / 3])
    np.testing.assert_allclose(back.members[0].weights, [0.75, 0.25])
    nested = torch.logsumexp(torch.stack([expected, members[0]]) + torch.log(torch.tensor([1 / 3, 2 / 3]))[:, None], 0)
    assert torch.allclose(back.log_likelihood(s[:40], theta[:40]).cpu(), nested, atol=1e-4)
    state = ens.state()
    state.pop("weights")                                               # a state written without weights
    np.testing.assert_allclose(Ensemble.from_state(state, device).weights, [0.5, 0.5])


def test_ensemble_validation(trained, device):
    other = NLE(flow=SMALL, device=device)
    s, theta = _data(100, n_x=2)
    other.fit(s, theta, train={"max_epochs": 1})
    for members, weights, match in [([], None, "at least one"), ([trained, other], None, "n_x"),
                                    ([trained, object()], None, "log_prob"), ([trained, trained], [1.0], "weights"),
                                    ([trained, trained], [1.0, -1.0], "weights"), ([trained], [0.0], "weights")]:
        with pytest.raises(ValueError, match=match):
            Ensemble(members, weights)


def test_ensemble_with_other_likelihoods(trained):
    class Flat:                                                        # anything with log_prob in sbi's layout
        def log_prob(self, input, condition):
            assert input.dtype == condition.dtype == torch.float32
            return torch.zeros(1, len(condition), device=condition.device)

    s, theta = _data(10, seed=8)
    ens = Ensemble([trained, Flat()])
    expected = torch.logaddexp(trained.log_likelihood(s, theta), torch.zeros(10, device=trained.device)) - np.log(2)
    assert (ens.n_x, ens.n_theta, ens.fitted) == (3, 2, True)
    assert torch.allclose(ens.log_likelihood(s, theta), expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_only_arrays_give_no_warning(dtype, device):
    # torch warns (once per process) about tensors made from read-only arrays
    from sbikit import Prior, sample_posteriors

    def ro(a):
        a = np.array(a, dtype=dtype)
        a.setflags(write=False)
        return a

    s, theta = (ro(a) for a in _data())
    always = torch.is_warn_always_enabled()
    torch.set_warn_always(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            nle = NLE(flow=SMALL, device=device, seed=0)
            nle.fit(s, theta, train=FAST)
            assert torch.isfinite(nle.log_likelihood(s, theta)).all()
            cfg = {"nwalkers": 16, "burn_in": 10, "max_steps": 40, "thin": 1, "adaptive": False, "n_samples": 20}
            prior = Prior(ro([-1, -1]), ro([1, 1]))
            for start in ({"init_centre": theta[:2], "init_sd": ro([0.1, 0.1])}, {"init_centre": theta[0]},
                          {"init_walkers": ro(np.random.default_rng(0).uniform(-0.5, 0.5, (16, 2)))}):
                post = sample_posteriors(nle, s[:2], prior, cfg, verbose=False, **start)
                assert post.samples.shape == (2, 20, 2)
    finally:
        torch.set_warn_always(always)
