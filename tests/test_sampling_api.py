"""sample_posteriors: step accounting, options, input checks, diagnostics and the Posteriors container."""
import json
import logging
import warnings

import numpy as np
import pytest
import torch
import torchemcee

from sbikit import Columns, Identity, MCMCConfig, Posteriors, Prior, sample_posteriors
from sbikit.sampling import MOVE_PRESETS, MOVES, LogitBox, diagnose, grouped_rhat, register_move

pytestmark = pytest.mark.filterwarnings("ignore:.*did not converge")


class Normal:
    """Standard normal in theta, whatever the data; counts the evaluated rows."""

    def __init__(self, device="cpu", n_x=None, n_theta=None):
        self.device, self.rows = device, 0
        if n_x is not None:
            self.n_x, self.n_theta = n_x, n_theta

    def log_prob(self, input, condition):
        self.rows += len(condition)
        return (-0.5 * (condition ** 2).sum(-1)).reshape(1, -1)


def _cfg(**kw):
    base = dict(nwalkers=16, burn_in=20, block=40, max_steps=80, thin=2, n_samples=100, seed=1)
    return MCMCConfig(**{**base, **kw})


def _run(device, B=2, P=2, prior=None, lik=None, **kw):
    call = {k: kw.pop(k) for k in list(kw) if k in ("init_centre", "init_sd", "init_walkers", "keep_chain",
                                                    "callback", "transform")}
    prior = prior if prior is not None else Prior([-5.0] * P, [5.0] * P)
    return sample_posteriors(lik or Normal(device), np.zeros((B, 1)), prior, _cfg(**kw), verbose=False, **call)


# ------------------------------------------------------------------ step accounting
@pytest.mark.parametrize("adaptive", [True, False])
@pytest.mark.parametrize("thin,block,max_steps,burn_in", [(1, 50, 200, 30), (5, 50, 200, 30), (4, 50, 203, 17)])
def test_max_steps_counts_steps_after_burn_in(adaptive, thin, block, max_steps, burn_in, device):
    lik, B, W = Normal(device), 3, 16
    cfg = MCMCConfig(nwalkers=W, moves="stretch", burn_in=burn_in, block=block, max_steps=max_steps, thin=thin,
                     adaptive=adaptive, n_samples=None, rhat=1 + 1e-9, logit_box=False)
    post = sample_posteriors(lik, np.zeros((B, 1)), Prior([-5, -5], [5, 5]), cfg, verbose=False, keep_chain=True,
                             init_centre=np.zeros(2))
    steps = max_steps // thin * thin                       # never converges: runs to max_steps
    assert post.diagnostics["steps"] == [steps] and post.diagnostics["burn_in"] == [burn_in]
    # every step evaluates each walker once; two more evaluations check and set up the ball's start positions
    assert lik.rows == (burn_in + steps + 2) * B * W
    kept = steps // thin
    assert post.chain.shape == (kept - kept // 2, B, W, 2)
    assert post.samples.shape == (B, (kept - kept // 2) * W, 2)


def test_thinning_is_regular_across_blocks(device):
    def run(**kw):
        return _run(device, B=3, burn_in=30, max_steps=200, n_samples=None, rhat=1 + 1e-9, seed=3, keep_chain=True,
                    **kw)

    full = run(thin=1, block=200, adaptive=False)          # all 200 steps; the chain is its second half
    for kw in (dict(thin=5, block=50), dict(thin=5, block=35), dict(thin=5, block=200, adaptive=False)):
        post = run(**kw)
        kept_at = np.arange(5, 201, 5)[-len(post.chain):]  # step numbers after burn-in of the kept states
        np.testing.assert_array_equal(post.chain, full.chain[kept_at - 101])
        np.testing.assert_array_equal(post.log_prob, full.log_prob[kept_at - 101])


def test_seed_reproducible_and_global_rng_untouched(device):
    state = torch.get_rng_state()
    a, b, c = _run(device), _run(device), _run(device, seed=2)
    assert torch.equal(state, torch.get_rng_state())
    np.testing.assert_array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples, c.samples)
    torch.manual_seed(0)
    d = _run(device, seed=None)
    torch.manual_seed(0)
    e = _run(device, seed=None)
    np.testing.assert_array_equal(d.samples, e.samples)    # seed=None follows the caller's global state
    assert not np.array_equal(d.samples, _run(device, seed=None).samples)


def test_batches_have_their_own_seeds(device):
    seen = []
    post = _run(device, B=5, targets_per_batch=2, callback=lambda info: seen.append(info["batch"]))
    assert post.samples.shape == (5, 100, 2) and len(post.diagnostics["steps"]) == 3
    assert sorted(set(seen)) == [0, 1, 2]
    np.testing.assert_array_equal(post.samples[:2], _run(device, B=2).samples)       # batch 0 = the unbatched run
    assert not np.array_equal(post.samples[2:4], post.samples[:2])


# ------------------------------------------------------------------ config
def test_config_rejects_typos_and_invalid_values():
    with pytest.raises(TypeError, match="nwalkers"):
        MCMCConfig.make({"nwalker": 32})
    with pytest.raises(TypeError, match="max_steps"):
        MCMCConfig.make({"max_step": 100})
    with pytest.raises(ValueError, match="prior"):
        MCMCConfig(init="Prior")
    for bad in (dict(init="map"), dict(nwalkers=0), dict(nwalkers=15), dict(thin=0), dict(block=3, thin=5),
                dict(burn_in=-1), dict(max_steps=100), dict(rhat=1.0), dict(rhat_groups=0), dict(min_acceptance=1.0),
                dict(init_scale=0.0), dict(n_samples=0), dict(targets_per_batch=0), dict(targets_per_batch=-1),
                dict(seed=-1), dict(thin=2.5), dict(moves="de_snoker"), dict(moves=[["dee", 1.0]]),
                dict(moves=[["de", -1.0]]), dict(moves=[["de", {"sigma": 1e-4}]]), dict(moves=[["de", "0.5"]]),
                dict(moves=[]), dict(moves=3), dict(moves=[["de", 1.0, {"nope": 1}]]),
                dict(moves=[[torchemcee.DEMove(), 1.0, {"sigma": 1.0}]]), dict(moves="snooker", nwalkers=6),
                dict(moves=[["stretch", 1.0]], stretch_a=0.5)):
        with pytest.raises(ValueError):
            MCMCConfig(**bad)
    with pytest.raises(ValueError, match="did you mean 'de_snooker'"):
        MCMCConfig(moves="de_snoker")
    # too few kept steps for the diagnostics: the message names what to change
    with pytest.raises(ValueError, match="raise block"):
        MCMCConfig(block=20, thin=5)
    with pytest.raises(ValueError, match="raise max_steps"):
        MCMCConfig(max_steps=20, thin=5, adaptive=False)
    cfg = MCMCConfig(max_steps=1e4, nwalkers=np.int64(32), adaptive=False, block=500)
    assert cfg.max_steps == 10000 and type(cfg.max_steps) is int and type(cfg.nwalkers) is int
    assert MCMCConfig(adaptive=False, max_steps=200).max_steps == 200          # block is not used then
    assert MCMCConfig.make(None, init="prior").init == "prior" and MCMCConfig().init == "auto"


class _HalfDE(torchemcee.DEMove):
    def __init__(self, gamma0=0.5):
        super().__init__(gamma0=gamma0)


@pytest.mark.parametrize("moves", [
    "de", "snooker", torchemcee.DEMove(), [torchemcee.DEMove(), torchemcee.StretchMove(a=1.5)],
    [[torchemcee.DEMove(), 0.5], ["stretch", 0.5]], ("de", "walk"),
    [["de", 0.8, {}], ["gaussian", 0.2, {"cov": 0.01}]], [["half_de_test", 1.0, {"gamma0": 0.3}], ["de", 1.0]],
    "my_preset_test"])
def test_moves_forms(moves, device):
    register_move("half_de_test", _HalfDE, overwrite=True)
    MOVE_PRESETS["my_preset_test"] = [["half_de_test", 0.5], [torchemcee.WalkMove(), 0.5]]
    try:
        post = _run(device, moves=moves)
        assert np.isfinite(post.samples).all() and (post.diagnostics["acceptance"] > 0).all()
        record = json.loads(json.dumps(post.diagnostics["config"]))           # JSON-friendly, moves by repr
        assert record[0]["nwalkers"] == 16
    finally:
        del MOVES["half_de_test"], MOVE_PRESETS["my_preset_test"]


def test_moves_registry_and_config_record(device):
    assert {"stretch", "walk", "de", "snooker", "kde", "gaussian"} <= set(MOVES)
    with pytest.raises(ValueError, match="already registered"):
        register_move("de", _HalfDE)
    cfg = MCMCConfig(moves=[[torchemcee.DEMove(sigma=1e-4), 1.0], ["stretch", 1.0, {"a": 1.5}]])
    record = cfg.to_dict()["moves"]                        # instances by a repr of their settings, names as given
    assert record[0][0].startswith("DEMove(sigma=0.0001, ") and record[0][1] == 1.0
    assert record[1] == ["stretch", 1.0, {"a": 1.5}] and json.loads(json.dumps(record)) == record
    # a config written with names only rebuilds from its record
    post = _run(device, moves=[["de", 0.8], ["stretch", 0.2]])
    assert MCMCConfig.make(post.diagnostics["config"][0]) == _cfg(moves=[["de", 0.8], ["stretch", 0.2]])
    # stretch_a applies to stretch moves given by name without their own a
    from sbikit.sampling import _weighted_moves
    moves = _weighted_moves(MCMCConfig(moves=[["stretch", 1.0], ["stretch", 1.0, {"a": 3.0}]], stretch_a=1.5))
    assert [m.a for m, _ in moves] == [1.5, 3.0]
    assert _weighted_moves(MCMCConfig(moves="stretch", stretch_a=2.5))[0][0].a == 2.5


# ------------------------------------------------------------------ initial positions
class Recorder(Normal):
    """Keeps the first positions it is evaluated at: the start positions of the walkers."""

    first = None

    def log_prob(self, input, condition):
        if self.first is None:
            self.first = condition.detach().cpu().numpy().copy()
        return super().log_prob(input, condition)


def _start(device, B, **kw):
    """(B, W, P) start positions of a short run."""
    lik = Recorder(device)
    base = dict(burn_in=0, block=8, max_steps=8, thin=1, adaptive=False, logit_box=False, n_samples=None)
    _run(device, B=B, lik=lik, **{**base, **kw})
    return lik.first.reshape(B, -1, lik.first.shape[-1])


@pytest.mark.parametrize("B", [2, 3])
def test_init_centre_single_point(B, device):
    centre = np.array([1.0, -1.0])                         # (P,): the same point for every target, also if B == P
    start = _start(device, B, init_centre=centre, init_sd=[1e-3, 1e-3])
    np.testing.assert_allclose(start.mean(1), np.tile(centre, (B, 1)), atol=1e-3)
    per_target = np.array([[1.0, -1.0], [2.0, 3.0], [-2.0, 0.5]])[:B]
    start = _start(device, B, init_centre=torch.tensor(per_target, requires_grad=True), init_sd=np.full((B, 2), 1e-3))
    np.testing.assert_allclose(start.mean(1), per_target, atol=1e-3)
    for bad in (np.zeros((B, 3)), np.zeros((B + 1, 2)), np.zeros(3), np.zeros((1, B, 2)), 0.0):
        with pytest.raises(ValueError, match="init_centre must have shape"):
            _start(device, B, init_centre=bad)
    with pytest.raises(ValueError, match="init_sd must have shape"):
        _start(device, B, init_centre=centre, init_sd=np.ones(3))
    with pytest.raises(ValueError, match="init_sd must be positive"):
        _start(device, B, init_centre=centre, init_sd=[0.1, 0.0])
    with pytest.raises(ValueError, match="init_centre contains NaN"):
        _start(device, B, init_centre=[np.nan, 0.0])
    with pytest.warns(UserWarning, match="outside the prior box"):
        start = _start(device, B, init_centre=[7.0, 0.0], init_sd=[1e-3, 1e-3])
    assert (start[..., 0] <= 5).all() and start[..., 0].min() > 4.99


def test_default_init_sd_is_per_target(device):
    priors = [Prior([-0.5, -0.5], [0.5, 0.5]), Prior([-500.0, -500.0], [500.0, 500.0])]
    spread = _start(device, 2, prior=priors, init_centre=np.zeros(2), nwalkers=128).std(1)       # (B, P)
    np.testing.assert_allclose(spread[0], 0.3 / np.sqrt(12), rtol=0.25)      # init_scale x box width / sqrt(12)
    np.testing.assert_allclose(spread[1], 300 / np.sqrt(12), rtol=0.25)
    spread = _start(device, 2, prior=priors, init_centre=np.zeros(2), init_sd=[[0.01, 0.01], [10.0, 10.0]],
                    init_scale=1.0, nwalkers=128).std(1)
    np.testing.assert_allclose(spread, [[0.01, 0.01], [10.0, 10.0]], rtol=0.25)
    spread = _start(device, 2, prior=priors, init_centre=np.zeros(2), init_sd=0.02, init_scale=1.0, nwalkers=128)
    np.testing.assert_allclose(spread.std(1), 0.02, rtol=0.25)


def test_init_auto_prior_and_ball(device, caplog):
    logger = logging.getLogger("sbikit")
    logger.addHandler(caplog.handler)
    try:
        post = sample_posteriors(Normal(device), np.zeros((2, 1)), Prior([1.0, 1.0], [3.0, 3.0]),
                                 _cfg(burn_in=0, block=8, max_steps=8, thin=1, n_samples=None), verbose=True)
    finally:
        logger.removeHandler(caplog.handler)
    assert "prior draws" in caplog.text and "mcmc 2 x 16" in caplog.text
    # the best of the prior draws lie towards the mode (the corner at 1), but still spread over the box
    assert post.samples.min() >= 1 and post.samples.max() <= 3 and post.samples.std() > 0.1
    with pytest.raises(ValueError, match="init='ball' needs init_centre"):
        _run(device, init="ball")
    assert _run(device, init="prior").samples.shape == (2, 100, 2)
    # start arguments that the chosen init does not use are reported
    with pytest.warns(UserWarning, match="init_centre and init_sd ignored: with MCMCConfig.init = 'prior'"):
        _run(device, init="prior", init_centre=np.zeros(2), init_sd=np.ones(2))
    with pytest.warns(UserWarning, match="init_sd ignored: with MCMCConfig.init = 'auto' the walkers start from prior"):
        _run(device, init_sd=np.ones(2))


def test_init_walkers(device):
    rng = np.random.default_rng(0)
    walkers = (np.array([2.0, -3.0]) + 1e-3 * rng.normal(size=(16, 2))).astype(np.float32)
    np.testing.assert_array_equal(_start(device, 2, init_walkers=walkers), np.stack([walkers, walkers]))
    per_target = np.stack([walkers, walkers + 1.0])
    np.testing.assert_array_equal(_start(device, 2, init_walkers=torch.tensor(per_target)), per_target)
    with pytest.raises(ValueError, match="init_walkers must have shape"):
        _start(device, 2, init_walkers=walkers[:8])
    with pytest.raises(ValueError, match="init_walkers inside the support"):
        _start(device, 2, init_walkers=walkers + 10.0)
    with pytest.raises(ValueError, match="do not span"):
        _start(device, 2, init_walkers=np.tile([2.0, -3.0], (16, 1)))
    outside = walkers.copy()
    outside[:3, 0] += 10.0                                 # the logit map would move these onto the edge of the box
    with pytest.raises(ValueError, match=r"6 of 32 start positions .*\(6 outside the prior support"):
        _start(device, 2, init_walkers=outside, logit_box=True)
    with pytest.warns(UserWarning, match="init_walkers ignored: with MCMCConfig.init = 'prior'"):
        _start(device, 2, init="prior", init_walkers=walkers)
    with pytest.warns(UserWarning, match="init_centre ignored: with MCMCConfig.init = 'auto' .* at init_walkers"):
        _start(device, 2, init_walkers=walkers, init_centre=np.zeros(2))


def test_nonfinite_start_positions_raise_clearly(device):
    class Holes(Normal):
        def log_prob(self, input, condition):
            out = super().log_prob(input, condition)
            return torch.where(condition[:, 0].reshape(1, -1) > 0, out, torch.nan)

    # prior draws and ball draws in the bad half are redrawn
    assert (_run(device, lik=Holes(device), init="prior").samples[..., 0] > 0).all()
    assert (_run(device, lik=Holes(device), init_centre=[0.0, 0.0]).samples[..., 0] > 0).all()
    with pytest.raises(ValueError, match=r"non-finite likelihood\), in targets \[0, 1\].*init='prior'") as err:
        _run(device, lik=Holes(device), init_centre=[-3.0, 0.0], init_sd=[0.01, 0.01])
    assert "allow_nonfinite" not in str(err.value)
    tr = Columns(["a", "b"], keep=["b"], derived=["c"], fn=lambda p: torch.log(p[..., :1]))
    with pytest.raises(ValueError, match="likelihood or transform"):
        _run(device, transform=tr, init_centre=[-3.0, 0.0], init_sd=[0.01, 0.01])


def test_infinite_bounds(device):
    from sbikit.priors import Gaussian
    prior = Prior([-np.inf, -1.0], [np.inf, 1.0], terms=[Gaussian([0], [0.0], [1.0])])
    with pytest.raises(ValueError, match="logit_box=False"):
        _run(device, prior=prior)
    with pytest.raises(ValueError, match="init_sd is needed"):
        _run(device, prior=prior, logit_box=False, init_centre=[0.0, 0.0])
    for kw in (dict(init_centre=[0.3, 0.0], init_sd=[0.5, 0.5]), dict(init="prior")):
        post = _run(device, prior=prior, logit_box=False, **kw)
        assert np.isfinite(post.samples).all() and np.abs(post.samples[..., 1]).max() <= 1


# ------------------------------------------------------------------ results and diagnostics
def test_unconverged_targets_warn(device):
    with pytest.warns(RuntimeWarning, match=r"2 of 2 targets did not converge after 80 steps \(worst R-hat"):
        post = _run(device)
    assert not post.converged.any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        post = _run(device, nwalkers=32, burn_in=200, block=400, max_steps=4000, min_ess=300, rhat=1.05)
    assert post.converged.all() and post.diagnostics["steps"][0] < 4000


def test_n_samples(device):
    with pytest.warns(RuntimeWarning, match="n_samples = 1000 is more than the 320 points"):
        post = _run(device, n_samples=1000)
    assert post.samples.shape == (2, 1000, 2)
    post = _run(device, n_samples=320, keep_chain=True)                      # the whole pool, without replacement
    flat = post.chain.transpose(1, 0, 2, 3).reshape(2, -1, 2)
    np.testing.assert_array_equal(np.sort(post.samples, 1), np.sort(flat, 1))
    post = _run(device, n_samples=None, keep_chain=True)                     # all of it, in chain order
    np.testing.assert_array_equal(post.samples, post.chain.transpose(1, 0, 2, 3).reshape(2, -1, 2))


@pytest.mark.parametrize("logit_box", [True, False])
def test_keep_chain_and_log_prob(logit_box, device):
    post = _run(device, keep_chain=True, logit_box=logit_box)
    assert post.chain.shape == (20, 2, 16, 2) and post.log_prob.shape == (20, 2, 16)
    density = -0.5 * (post.chain ** 2).sum(-1) - 2 * np.log(10.0)            # likelihood x flat prior, in theta
    np.testing.assert_allclose(post.log_prob, density, atol=1e-4)
    assert _run(device).chain is None and _run(device).log_prob is None
    again = _run(device, burn_in=0, init_walkers=post.chain[-1], logit_box=logit_box)      # continue from the last step
    assert again.diagnostics["burn_in"] == [0] and np.isfinite(again.samples).all()


def test_callback_can_stop(device):
    seen = []

    def callback(info):
        seen.append(info)
        return len(seen) == 2

    post = _run(device, max_steps=400, callback=callback)
    assert [s["steps"] for s in seen] == [40, 80] and post.diagnostics["steps"] == [80]
    assert set(seen[0]) == {"steps", "converged", "rhat_max", "ess_min", "tau_max", "seconds", "batch"}
    assert seen[0]["converged"].shape == (2,) and seen[0]["converged"].dtype == bool
    assert not post.converged.any()
    seen.clear()
    _run(device, adaptive=False, callback=callback)                          # one check at the end
    assert [s["steps"] for s in seen] == [80]


def test_short_chains_do_not_crash():
    rng = torch.Generator().manual_seed(0)
    accepted = torch.ones((2, 8), dtype=torch.long)
    for n in (1, 2, 3, 4, 5, 8):
        chain = torch.randn((n, 2, 8, 3), generator=rng)
        rhat = grouped_rhat(chain, 4)
        assert rhat.shape == (2, 3) and (torch.isnan(rhat).all() if n < 4 else torch.isfinite(rhat).all())
        diag = diagnose(chain, accepted, max(n, 1), {"min_ess": 1, "min_tau": 1})
        for key in ("tau", "ess"):
            assert diag[key].shape == (2, 3) and not (diag[key] <= 0).any()
        assert diag["converged"].shape == (2,) and (n >= 4 or not diag["converged"].any())
    assert torch.isnan(grouped_rhat(torch.zeros((6, 1, 4, 2)), 2)).all()      # constant chain: undefined, no error
    assert not diagnose(torch.zeros((6, 1, 4, 2)), accepted[:1, :4], 6)["converged"].any()


@pytest.mark.parametrize("kw", [dict(block=7, max_steps=7, thin=1), dict(block=14, max_steps=15, thin=2),
                                dict(max_steps=7, thin=1, adaptive=False), dict(block=7, max_steps=20, thin=1)])
def test_shortest_allowed_runs(kw, device):
    post = _run(device, burn_in=0, n_samples=None, min_ess=1, min_tau=1, rhat=1 + 1e-9, keep_chain=True, **kw)
    d = post.diagnostics
    assert d["steps"] == [kw["max_steps"] // kw["thin"] * kw["thin"]] and len(post.chain) >= 4
    for key in ("tau", "ess"):                             # an estimate or nan, never zero or negative
        assert not (d[key] <= 0).any()
    undefined = np.isnan(d["tau"]).any(1) | np.isnan(d["rhat"]).any(1)
    assert not d["converged"][undefined].any() and np.isfinite(post.samples).all()


def test_diagnostics_units(device):
    post = _run(device, B=3, burn_in=50, block=200, max_steps=1000, thin=4, adaptive=False, nwalkers=32)
    d = post.diagnostics
    assert d["rhat"].shape == d["ess"].shape == d["tau"].shape == (3, 2)
    np.testing.assert_array_equal(d["rhat_max"], d["rhat"].max(1))
    np.testing.assert_array_equal(d["ess_min"], d["ess"].min(1))
    np.testing.assert_array_equal(d["tau_max"], d["tau"].max(1))
    # ESS of the second-half chain (125 kept steps x 32 walkers); tau in steps, not in kept steps
    np.testing.assert_allclose(d["ess"] * d["tau"], 125 * 32 * 4, rtol=1e-6)
    assert (d["tau"] > 4).all() and ((d["acceptance"] > 0) & (d["acceptance"] < 1)).all()


# ------------------------------------------------------------------ input checks
def test_shape_mismatches_raise(linear_model, device):
    lik, P, D = linear_model["lik"], linear_model["P"], linear_model["D"]
    lik.n_x, lik.n_theta = D, P
    prior, x = Prior([-2.0] * P, [2.0] * P, names=["a", "b"]), np.zeros((3, D))
    cfg = _cfg()

    def run(x=x, prior=prior, cfg=cfg, **kw):
        return sample_posteriors(lik, x, prior, cfg, verbose=False, **kw)

    assert run().names == ["a", "b"]
    assert run(x=np.zeros(D)).samples.shape == (1, 100, P)                    # (n_x,): a single target
    with pytest.raises(ValueError, match="x_obs must be"):
        run(x=np.zeros((0, D)))
    with pytest.raises(ValueError, match="x_obs must be"):
        run(x=np.zeros((3, 2, D)))
    with pytest.raises(ValueError, match=f"x_obs has 1 columns but the likelihood takes {D}"):
        run(x=np.zeros((3, 1)))
    with pytest.raises(ValueError, match=r"x_obs contains NaN or inf in 1 of 3 targets: \[1\]"):
        run(x=np.array([[0.0] * D, [np.nan] * D, [0.0] * D]))
    with pytest.raises(ValueError, match="prior has 1 parameters but the likelihood is conditioned on 2"):
        run(prior=Prior([-2.0], [2.0]))
    with pytest.raises(ValueError, match="got 2 priors for 3 targets"):
        run(prior=[prior, prior])
    with pytest.raises(ValueError, match="same number of parameters"):
        run(prior=[prior, prior, Prior([-2.0], [2.0])])
    with pytest.raises(ValueError, match="different parameter names"):
        run(prior=[prior, prior, Prior([-2.0] * P, [2.0] * P, names=["b", "a"])])
    with pytest.raises(TypeError, match="prior must be a sbikit Prior"):
        run(prior=torch.distributions.Normal(0.0, 1.0))
    with pytest.raises(TypeError, match="log_prob"):
        sample_posteriors(object(), x, prior, cfg)
    with pytest.raises(ValueError, match="nwalkers = 4 must be larger than twice"):
        run(cfg=_cfg(nwalkers=4, moves="stretch"))
    # transforms: output width against the likelihood, names against the prior, output shape
    three = Columns(["a", "b"], keep=["a", "b"], derived=["c"], fn=lambda p: p[..., :1])
    with pytest.raises(ValueError, match="transform returns 3 columns but the likelihood is conditioned on 2"):
        run(transform=three)
    with pytest.raises(ValueError, match=r"transform expects the parameters \['b', 'a'\], the prior has \['a', 'b'\]"):
        run(transform=Identity(["b", "a"]))
    with pytest.raises(ValueError, match="transform must map"):
        run(transform=lambda psi: psi.sum(-1))
    assert run(transform=Identity(["a", "b"])).derived.shape == (3, 100, P)
    assert run(prior=Prior([-2.0] * P, [2.0] * P), transform=Identity(["b", "a"])).derived.shape == (3, 100, P)


def test_likelihood_without_device_attribute():
    class Bare:
        def log_prob(self, input, condition):
            return (-0.5 * (condition ** 2).sum(-1)).reshape(1, -1)

    post = sample_posteriors(Bare(), torch.zeros((2, 1), requires_grad=True), Prior([-5, -5], [5, 5]), _cfg(),
                             verbose=False)
    assert post.samples.shape == (2, 100, 2)
    post = sample_posteriors(Bare(), np.zeros((1, 1)), Prior([-5, -5], [5, 5]), _cfg(), device="auto", verbose=False)
    assert post.samples.shape == (1, 100, 2)


def test_prior_subclass_log_prob_is_used(device):
    class Half(Prior):
        """The box, restricted to theta_0 > 0 through its own log_prob."""

        def log_prob(self, theta):
            lp = super().log_prob(theta)
            return torch.where(theta[..., 0] > 0, lp, torch.full_like(lp, -torch.inf))

    half = Half([-5.0, -5.0], [5.0, 5.0])
    for kw in (dict(init="prior"), dict(init_centre=[1.0, 0.0])):
        post = _run(device, prior=half, burn_in=100, **kw)
        assert (post.samples[..., 0] > 0).all()
        post = _run(device, prior=[half, Prior([-5.0, -5.0], [5.0, 5.0])], burn_in=100, **kw)
        assert (post.samples[0, :, 0] > 0).all() and (post.samples[1, :, 0] < 0).any()


def test_equal_priors_as_list_give_the_same_chain(device):
    kw = dict(n_samples=None, init_centre=np.zeros(2), init_sd=[0.5, 0.5])
    one = _run(device, B=3, prior=Prior([-5.0, -4.0], [5.0, 4.0]), **kw)
    many = _run(device, B=3, prior=[Prior([-5.0, -4.0], [5.0, 4.0]) for _ in range(3)], **kw)
    np.testing.assert_array_equal(one.samples, many.samples)
    # different boxes per target: the vectorised box density equals the per-prior evaluation
    boxes = [Prior([-5.0, -4.0], [5.0, 4.0]), Prior([-1.0, -1.0], [2.0, 2.0]), Prior([-0.3, -3.0], [0.3, 3.0])]
    low = torch.stack([p.low for p in boxes])[:, None]
    high = torch.stack([p.high for p in boxes])[:, None]
    theta = torch.randn((3, 50, 2), generator=torch.Generator().manual_seed(0)) * 2
    expected = torch.stack([p.log_prob(t) for p, t in zip(boxes, theta)])
    inside = ((theta >= low) & (theta <= high)).all(-1)
    np.testing.assert_array_equal(torch.where(inside, -torch.log(high - low).sum(-1), -torch.inf), expected)
    post = _run(device, B=3, prior=boxes, n_samples=None)
    for b, p in enumerate(boxes):
        assert (post.samples[b] >= p.low.numpy()).all() and (post.samples[b] <= p.high.numpy()).all()


# ------------------------------------------------------------------ transforms
class _Scale(torch.nn.Module):
    """psi -> 2 psi[..., :1], recording every .to() call on this instance."""

    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(2.0))
        self.to_calls = []

    def forward(self, psi):
        return self.factor * psi[..., :1]

    def to(self, *args, **kwargs):
        self.to_calls.append(args)
        return super().to(*args, **kwargs)


def test_transform_is_not_moved(device):
    module = _Scale()
    tr = Columns(["a", "b"], keep=["b"], derived=["c"], fn=module)
    moved = tr.to(device)
    assert moved is not tr and moved.fn is not module and isinstance(moved, Columns) and module.to_calls == []
    assert moved.names_out == ["b", "c"] and moved.fn.to_calls != []
    post = _run(device, transform=tr)
    assert tr.fn is module and module.to_calls == []
    np.testing.assert_allclose(post.derived, np.stack([post.samples[..., 1], 2 * post.samples[..., 0]], -1), rtol=1e-6)
    whole = _Scale()                                                         # a bare module as the transform

    class Both(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = whole

        def forward(self, psi):
            return torch.cat([self.inner(psi), psi[..., 1:]], -1)

    post = _run(device, transform=Both())
    assert whole.to_calls == [] and post.derived.shape == (2, 100, 2)
    plain = Columns(["a", "b"], keep=["a"], derived=["c"], fn=lambda p: p[..., 1:] ** 2)
    assert plain.to(device) is plain
    shift = torch.tensor([[0.0], [1.0]], device=device)[:, None]            # (B, 1, 1): one constant per target
    post = _run(device, transform=lambda psi: psi + shift.to(psi.device))
    np.testing.assert_allclose(post.derived[1], post.samples[1] + 1.0, rtol=1e-6)


def test_columns_validation():
    def fn(p):
        return p[..., :1]

    with pytest.raises(ValueError, match="keep has 'om', which is not in names_in.*did you mean 'Om'"):
        Columns(["Om", "S8"], keep=["om"], derived=["s"], fn=fn)
    with pytest.raises(ValueError, match="names_in must be unique"):
        Columns(["a", "a"], keep=["a"], derived=["s"], fn=fn)
    with pytest.raises(ValueError, match="keep \\+ derived must be unique"):
        Columns(["a", "b"], keep=["a"], derived=["a"], fn=fn)
    with pytest.raises(ValueError, match="order must be a permutation"):
        Columns(["a", "b"], keep=["a"], derived=["s"], fn=fn, order=["a", "t"])
    psi = torch.ones((4, 2))
    for bad in (lambda p: p[..., 0], lambda p: p, lambda p: p[:2, :1], lambda p: p[..., :1].numpy()):
        with pytest.raises(ValueError, match="fn must return a tensor of shape \\(4, 1\\)"):
            Columns(["a", "b"], keep=["a"], derived=["s"], fn=bad)(psi)
    with pytest.raises(ValueError, match="expected 2 input columns"):
        Columns(["a", "b"], keep=["a"], derived=["s"], fn=fn)(torch.ones((4, 3)))
    tr = Columns(["a", "b"], keep=["b"], derived=["s"], fn=fn, order=["s", "b"])
    assert tr(torch.tensor([[[1.0, 2.0]]])).tolist() == [[[1.0, 2.0]]]
    assert repr(tr) == "Columns(names_in=['a', 'b'], names_out=['s', 'b'])"
    assert repr(Identity(["a"])) == "Identity(names_in=['a'], names_out=['a'])"
    tr.check_input(["a", "b"])
    with pytest.raises(ValueError, match="transform expects the parameters"):
        tr.check_input(["b", "a"])


# ------------------------------------------------------------------ Posteriors
def test_posteriors_helpers_and_save_load(tmp_path, device):
    tr = Columns(["a", "b"], keep=["a"], derived=["c"], fn=lambda p: p[..., 1:] ** 2)
    post = _run(device, B=3, prior=Prior([-5.0, -5.0], [5.0, 5.0], names=["a", "b"]), transform=tr, keep_chain=True)
    assert len(post) == 3 and post.names == ["a", "b"]
    assert repr(post) == ("Posteriors(targets=3, samples=100, parameters=2, converged=0/3, names=['a', 'b'], "
                          "derived=2, chain=(20, 3, 16, 2))")
    np.testing.assert_allclose(post.mean(), post.samples.mean(1))
    np.testing.assert_allclose(post.std(), post.samples.std(1))
    assert post.quantile(0.5).shape == (3, 2) and post.quantile([0.16, 0.5, 0.84]).shape == (3, 3, 2)
    s = post.summary()
    assert set(s) == {"mean", "std", "median", "q16", "q84", "converged"}
    np.testing.assert_allclose(s["median"], np.median(post.samples, 1))
    assert (s["q16"] < s["median"]).all() and (s["median"] < s["q84"]).all() and s["converged"].shape == (3,)

    path = post.save(tmp_path / "post")                                      # the suffix is added
    assert path.name == "post.npz" and post.save(tmp_path / "post.npz") == path
    for back in (Posteriors.load(path), Posteriors.load(tmp_path / "post")):
        for key in ("samples", "derived", "chain", "log_prob"):
            np.testing.assert_array_equal(getattr(back, key), getattr(post, key))
        assert back.names == post.names and set(back.diagnostics) == set(post.diagnostics)
        for key, value in post.diagnostics.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(back.diagnostics[key], value)
                assert back.diagnostics[key].dtype == value.dtype
            else:
                assert back.diagnostics[key] == value
        assert back.converged.dtype == bool and repr(back) == repr(post)

    bare = Posteriors(np.zeros((2, 5, 3)), {"converged": np.array([True, False])}, None)   # positional construction
    assert bare.names is None and bare.chain is None and bare.log_prob is None
    assert repr(bare) == "Posteriors(targets=2, samples=5, parameters=3, converged=1/2)"
    assert repr(Posteriors(np.zeros((2, 5, 3)))) == "Posteriors(targets=2, samples=5, parameters=3)"
    back = Posteriors.load(bare.save(tmp_path / "bare.npz"))
    assert back.derived is None and back.names is None and back.converged.tolist() == [True, False]
    np.savez(tmp_path / "other.npz", a=np.zeros(3))
    with pytest.raises(ValueError, match="not written by Posteriors.save"):
        Posteriors.load(tmp_path / "other.npz")


def test_batches_of_different_length(device):
    # the first batch is stopped at its first check, the second runs to max_steps: chain and samples keep the
    # last steps the batches have in common
    with pytest.warns(RuntimeWarning, match="stopped after different numbers of steps"):
        post = _run(device, B=2, targets_per_batch=1, n_samples=None, keep_chain=True,
                    callback=lambda info: info["batch"] == 0)
    assert post.diagnostics["steps"] == [40, 80]
    assert post.chain.shape == (10, 2, 16, 2) and post.log_prob.shape == (10, 2, 16)
    assert post.samples.shape == (2, 160, 2)
    np.testing.assert_array_equal(post.samples, post.chain.transpose(1, 0, 2, 3).reshape(2, -1, 2))
    whole = _run(device, B=2, targets_per_batch=1, n_samples=None, keep_chain=True)
    np.testing.assert_array_equal(post.chain[:, 1], whole.chain[-10:, 1])


# ------------------------------------------------------------------ logging
def test_torchemcee_logger_is_left_alone(device, caplog):
    log = logging.getLogger("torchemcee")
    assert log.level == logging.NOTSET                                       # importing sbikit does not set it
    with caplog.at_level(logging.WARNING, logger="torchemcee"):
        level = log.level
        _run(device)                                                         # short chain: torchemcee would warn
        assert log.level == level and "autocorrelation" not in caplog.text
        torchemcee.integrated_autocorr_time(torch.randn((8, 1, 4, 1)).cumsum(0), tol=50, quiet=True)
        assert "autocorrelation" in caplog.text                              # direct use still reports


def test_logit_box_jacobian():
    box = LogitBox(torch.tensor([-1.0, 2.0]), torch.tensor([3.0, 2.5]))
    u = torch.tensor([[0.3, -1.2], [2.0, 0.1]], dtype=torch.float64, requires_grad=True)
    jac = torch.stack([torch.autograd.functional.jacobian(box.to_theta, row).diagonal() for row in u])
    np.testing.assert_allclose(box.log_jacobian(u).detach(), torch.log(jac).sum(-1), rtol=1e-6)
    np.testing.assert_allclose(box.to_u(box.to_theta(u)).detach(), u.detach(), rtol=1e-6)
