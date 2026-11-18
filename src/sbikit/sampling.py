"""Batched posterior sampling with torchemcee.

All targets (observations or test simulations) are sampled in one ensemble with a separate set of
walkers each, so that hundreds of posteriors cost one batched flow evaluation per step. By default
walkers move in the logit of each target's prior box (they never leave it), with
differential-evolution and snooker moves (ter Braak & Vrugt 2008), and the chain grows in blocks
after burn-in until, on its second half, every target has grouped split R-hat < ``rhat``,
ESS >= ``min_ess`` and length >= ``min_tau`` tau in every parameter (or ``max_steps``). Samples are
drawn from that second half.

The snooker move needs torchemcee >= 0.1.1. The ``DESnookerMove`` of emcee 3.1.6 and of torchemcee
0.1.0 does not follow ter Braak & Vrugt (2008): it scales the proposal by ``(x - z) / sqrt|x - z|``
and uses the exponent (d - 1) / 2, where the paper has the unit vector and d - 1. On a flat 9-d box
it gives a standard deviation 16 % too small alone, and 3 % too small in the 0.8 DE / 0.2 snooker
mix. torchemcee 0.1.1 has the corrected move; the test suite checks every move on that box.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchemcee
from torchemcee.integrations.sbi import flow_log_prob_fn

from .priors import Prior

__all__ = ["MCMCConfig", "Posteriors", "sample_posteriors", "LogitBox", "grouped_rhat", "diagnose"]

logging.getLogger("torchemcee").setLevel(logging.ERROR)


@dataclass
class MCMCConfig:
    """Settings of :func:`sample_posteriors`.

    Attributes:
        nwalkers: walkers per target.
        moves: a preset name (``"de_snooker"``, ``"de"``, ``"stretch"``, ``"walk"``, ``"kde"``; see
            ``MOVE_PRESETS``) or a weighted list ``[[name, weight, {kwargs}], ...]`` of the
            torchemcee moves ``stretch | walk | de | snooker | kde | gaussian``, e.g.
            ``[["de", 0.8, {}], ["gaussian", 0.2, {"cov": 0.01}]]``.
        stretch_a: scale of the stretch move (``"stretch"`` preset).
        logit_box: sample u = logit((theta - low) / (high - low)) instead of theta. Walkers then
            cannot leave the prior box, and posteriors that pile up at a bound mix better.
        init: ``"ball"`` (Gaussian ball around ``init_centre``) or ``"prior"`` (prior draws).
        init_scale: ball standard deviation in units of ``init_sd``.
        burn_in: steps discarded at the start.
        block: steps between convergence checks.
        max_steps: maximum number of steps after burn-in.
        thin: keep every ``thin``-th step.
        adaptive: stop as soon as every target has converged; False runs exactly ``max_steps``.
        rhat: convergence threshold of the grouped split R-hat.
        rhat_groups: number of walker groups used as chains in R-hat.
        min_ess: required effective sample size per parameter.
        min_tau: required chain length in integrated autocorrelation times.
        min_acceptance: required mean acceptance fraction.
        n_samples: samples returned per target, drawn from the second half of the chain.
        targets_per_batch: split very many targets into several ensembles of this size (memory).
        seed: seed of the sampler.
        progress: print a line per convergence check (for long single runs).
    """

    nwalkers: int = 256
    # a preset name (see MOVE_PRESETS) or a weighted list [[name, weight, {kwargs}], ...] of the
    # torchemcee moves stretch | walk | de | snooker | kde | gaussian (kwargs e.g. {"cov": 0.01})
    moves: Union[str, list] = "de_snooker"
    stretch_a: float = 2.0          # used by the "stretch" preset
    logit_box: bool = True          # sample u = logit((theta - low) / (high - low))
    init: str = "ball"              # ball (around init_centre) | prior
    init_scale: float = 0.3         # ball sd in units of init_sd
    burn_in: int = 1000
    block: int = 500                # steps between convergence checks
    max_steps: int = 10000          # steps after burn-in
    thin: int = 5
    adaptive: bool = True           # False: run exactly max_steps after burn-in, no early stop
    rhat: float = 1.01
    rhat_groups: int = 8
    min_ess: float = 2000
    min_tau: float = 20
    min_acceptance: float = 0.05
    n_samples: int = 2000           # returned per target, drawn from the second half of the chain
    targets_per_batch: Optional[int] = None   # split very many targets into several ensembles
    seed: int = 0
    progress: bool = False          # print a line per convergence check (long single runs)

    @classmethod
    def make(cls, cfg):
        """A config from None (defaults), an instance (returned as is) or a dict."""
        if cfg is None:
            return cls()
        return cfg if isinstance(cfg, cls) else cls(**dict(cfg))


@dataclass
class Posteriors:
    """Result of :func:`sample_posteriors` for B targets.

    Attributes:
        samples: (B, n_samples, P) posterior samples, in the sampled parameters.
        diagnostics: per-target arrays (B,) ``converged``, ``tau_max``, ``rhat_max``, ``ess_min``,
            ``acceptance``, and per-ensemble lists ``steps``, ``seconds``, ``config``.
        derived: (B, n_samples, P') = transform(samples) if a transform was given, else None.
    """

    samples: np.ndarray                  # (B, n_samples, P) in the sampled parameters
    diagnostics: dict = field(default_factory=dict)   # per target arrays (B,) + run info
    derived: Optional[np.ndarray] = None  # (B, n_samples, P') = transform(samples), if a transform was given

    @property
    def converged(self) -> np.ndarray:
        """(B,) bool: which targets met every convergence criterion."""
        return self.diagnostics["converged"]


class LogitBox:
    """The map u = logit((theta - low) / (high - low)) between a box and the real line.

    Args:
        low, high: box bounds, broadcastable to theta (e.g. (B, 1, P) for per-target boxes).
    """

    def __init__(self, low, high):
        self.low, self.width = low, high - low

    def to_theta(self, u):
        """Unbounded u -> theta inside the box."""
        return self.low + self.width * torch.sigmoid(u)

    def to_u(self, theta):
        """theta -> u; points on the bounds are moved 1e-6 of the width inside."""
        return torch.logit(((theta - self.low) / self.width).clamp(1e-6, 1 - 1e-6))

    def wrap(self, log_prob_fn):
        """The density in u of a log density in theta: ``log_prob_fn(theta(u)) + log |d theta / d u|``."""
        log_width = torch.log(self.width).sum(-1)

        def log_prob(u):
            log_jac = (F.logsigmoid(u) + F.logsigmoid(-u)).sum(-1) + log_width
            return log_prob_fn(self.to_theta(u)) + log_jac

        return log_prob


def grouped_rhat(chain, k):
    """Split R-hat (Gelman-Rubin) with k groups of walkers as the chains.

    The walkers of an ensemble are not independent chains, so they are pooled into ``k`` groups,
    and each group's first and second half are compared.

    Args:
        chain: (n, B, W, P) tensor of n steps, B targets, W walkers.
        k: number of walker groups (at most W // 2).

    Returns:
        (B, P) tensor.
    """
    n, n_targets, nwalkers, ndim = chain.shape
    k = max(1, min(k, nwalkers // 2))
    half = n // 2
    x = chain[:2 * half, :, : (nwalkers // k) * k].reshape(2, half, n_targets, k, nwalkers // k, ndim)
    x = x.permute(2, 5, 0, 3, 1, 4).reshape(n_targets, ndim, 2 * k, -1)
    size = x.shape[-1]
    within = x.var(-1).mean(-1)
    between = size * x.mean(-1).var(-1)
    return torch.sqrt(((size - 1) / size * within + between / size) / within)


def diagnose(chain, accepted, nsteps, cfg: MCMCConfig):
    """Convergence diagnostics per target.

    Args:
        chain: (n, B, W, P) thinned chain.
        accepted: accepted-proposal counts of the sampler state.
        nsteps: number of (unthinned) steps taken.
        cfg: thresholds.

    Returns:
        dict of tensors: ``tau`` (integrated autocorrelation time, in thinned steps), ``rhat``,
        ``ess`` (each (B, P)), ``acceptance`` (B, W) and ``converged`` (B,).
    """
    n, n_targets, nwalkers, ndim = chain.shape
    chunk = max(1, int(1e8 // max(1, n * nwalkers * ndim)))
    tau, rhat = [], []
    for i in range(0, n_targets, chunk):
        part = chain[:, i:i + chunk]
        tau.append(torchemcee.integrated_autocorr_time(part, tol=cfg.min_tau, quiet=True))
        rhat.append(grouped_rhat(part, cfg.rhat_groups))
    tau, rhat = torch.cat(tau), torch.cat(rhat)
    ess = n * nwalkers / tau
    acceptance = torchemcee.acceptance_fraction(accepted, nsteps)
    converged = ((rhat < cfg.rhat).all(-1) & (ess >= cfg.min_ess).all(-1)
                 & (n >= cfg.min_tau * tau).all(-1) & (acceptance.mean(-1) > cfg.min_acceptance))
    return {"tau": tau, "rhat": rhat, "ess": ess, "acceptance": acceptance, "converged": converged}


MOVES = {
    "stretch": torchemcee.StretchMove, "walk": torchemcee.WalkMove, "de": torchemcee.DEMove,
    "snooker": torchemcee.DESnookerMove, "kde": torchemcee.KDEMove, "gaussian": torchemcee.GaussianMove,
}

MOVE_PRESETS = {
    "de_snooker": [["de", 0.8, {}], ["snooker", 0.2, {}]],   # ter Braak & Vrugt 2008
    "de": [["de", 1.0, {}]],
    "stretch": None,                                         # StretchMove(a=stretch_a)
    "walk": [["walk", 1.0, {}]],
    "kde": [["kde", 1.0, {}]],
}


def _check_torchemcee():
    """torchemcee < 0.1.1 has emcee's biased DESnookerMove (too narrow posteriors)."""
    v = tuple(int(p) for p in torchemcee.__version__.split(".")[:3] if p.isdigit())
    if v < (0, 1, 1):
        raise RuntimeError(f"torchemcee {torchemcee.__version__} has the biased emcee snooker move; "
                           "install torchemcee >= 0.1.1 (git tag v0.1.1)")


def _moves(cfg: MCMCConfig):
    spec = cfg.moves
    if isinstance(spec, str):
        if spec not in MOVE_PRESETS:
            raise ValueError(f"unknown moves preset {spec!r}; presets: {sorted(MOVE_PRESETS)}")
        if spec == "stretch":
            return torchemcee.StretchMove(a=cfg.stretch_a)
        spec = MOVE_PRESETS[spec]
    out = []
    for item in spec:
        name, weight, kwargs = (list(item) + [{}])[:3]
        if name not in MOVES:
            raise ValueError(f"unknown move {name!r}; moves: {sorted(MOVES)}")
        if name == "snooker":
            _check_torchemcee()
        out.append((MOVES[name](**dict(kwargs)), float(weight)))
    return out[0][0] if len(out) == 1 else out


def _as_priors(prior, n):
    priors = list(prior) if isinstance(prior, (list, tuple)) else [prior] * n
    if len(priors) != n:
        raise ValueError(f"{len(priors)} priors for {n} targets")
    return priors


def sample_posteriors(likelihood, x_obs, prior: Union[Prior, Sequence[Prior]], config=None,
                      init_centre=None, init_sd=None, device=None, verbose: bool = True,
                      transform=None) -> Posteriors:
    """Sample p(theta | x_b) for every row b of ``x_obs``, all in one batched ensemble.

    Args:
        likelihood: an :class:`~sbikit.nle.NLE` or :class:`~sbikit.nle.Ensemble` (anything with
            ``log_prob(input, condition)`` in sbi's layout).
        x_obs: (B, n_x) summaries to condition on.
        prior: one :class:`~sbikit.priors.Prior` for all targets, or one per target.
        config: :class:`MCMCConfig` or dict.
        init_centre: (B, P) ball centres (truths for test sims, the fiducial point or a MAP for
            data); required for ``init="ball"``.
        init_sd: (P,) scale of the ball (e.g. the std of the training parameters); default: the
            prior box width / sqrt(12).
        transform: optional map psi -> theta (a :class:`~sbikit.transforms.Transform` or any callable
            on (..., P) tensors). The sampler then moves in psi (``prior``, ``init_centre`` and
            ``init_sd`` are in psi) and the likelihood is evaluated at theta = transform(psi). Use it
            to infer the parameters of a model the flow was not trained on, e.g. a model that
            predicts the per-bin amplitudes the flow is conditioned on. ``Posteriors.derived`` holds
            transform(samples).
        device: torch device of the sampler (default: the likelihood's device).
        verbose: print one summary line per ensemble.

    Returns:
        :class:`Posteriors`. Check ``.converged`` before using the samples: targets that reached
        ``max_steps`` without converging are returned as well.

    Example:
        >>> prior = Prior.box_from_data(theta_train)
        >>> post = sample_posteriors(nle, s_test, prior, MCMCConfig(nwalkers=128),
        ...                          init_centre=theta_test, init_sd=theta_train.std(0))
        >>> post.samples.shape, post.converged.all()
        ((n_test, 2000, P), True)
    """
    cfg = MCMCConfig.make(config)
    device = torch.device(device or likelihood.device)
    x_obs = torch.as_tensor(np.asarray(x_obs) if not isinstance(x_obs, torch.Tensor) else x_obs,
                            dtype=torch.float32, device=device)
    if x_obs.ndim == 1:
        x_obs = x_obs[None]
    B = len(x_obs)
    priors = _as_priors(prior, B)
    if init_centre is not None:
        init_centre = np.asarray(init_centre, dtype=np.float64).reshape(B, -1)
    per = cfg.targets_per_batch or B
    samples, diags = [], []
    for start in range(0, B, per):
        sl = slice(start, min(B, start + per))
        s, d = _sample_batch(likelihood, x_obs[sl], priors[sl], cfg,
                             None if init_centre is None else init_centre[sl], init_sd, device,
                             seed=cfg.seed + start, verbose=verbose, transform=transform)
        samples.append(s)
        diags.append(d)
    keys = [k for k in diags[0] if isinstance(diags[0][k], np.ndarray)]
    out = {k: np.concatenate([d[k] for d in diags]) for k in keys}
    out.update({k: [d[k] for d in diags] for k in diags[0] if k not in keys})
    samples = np.concatenate(samples)
    derived = None
    if transform is not None:
        tr = transform.to("cpu") if hasattr(transform, "to") else transform
        with torch.no_grad():
            derived = tr(torch.as_tensor(samples, dtype=torch.float32)).numpy()
    return Posteriors(samples, out, derived)


def _sample_batch(likelihood, x_obs, priors, cfg, init_centre, init_sd, device, seed, verbose, transform=None):
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    B = len(x_obs)
    # group identical prior objects so each is evaluated once on its targets
    uniq, which = [], []
    for p in priors:
        for i, q in enumerate(uniq):
            if q is p:
                which.append(i)
                break
        else:
            uniq.append(p)
            which.append(len(uniq) - 1)
    uniq = [p.to(device) for p in uniq]
    which = torch.as_tensor(which, device=device)
    ndim = uniq[0].ndim
    low = torch.stack([uniq[w].low for w in which.tolist()])[:, None]     # (B, 1, P)
    high = torch.stack([uniq[w].high for w in which.tolist()])[:, None]

    def prior_lp(theta):                                                   # (B, n, P) -> (B, n)
        out = torch.empty(theta.shape[:2], device=device, dtype=theta.dtype)
        for i, p in enumerate(uniq):
            m = which == i
            out[m] = p.log_prob(theta[m])
        return out

    like = flow_log_prob_fn(likelihood, x_obs)
    if transform is not None:
        tr = transform.to(device) if hasattr(transform, "to") else transform
        like_theta = like

        def like(psi):
            return like_theta(tr(psi))

    def log_prob(theta):
        # -inf outside the support; a non-finite flow value (far outside the training range) is
        # treated the same way, since torchemcee rejects nan
        lp, ll = prior_lp(theta), like(theta)
        ok = torch.isfinite(lp) & torch.isfinite(ll)
        return torch.where(ok, ll + lp, torch.full_like(lp, -torch.inf))

    # initial walkers, inside each target's support
    if cfg.init == "prior":
        init = torch.stack([uniq[w].sample(cfg.nwalkers, generator=gen) for w in which.tolist()])
    else:
        if init_centre is None:
            raise ValueError("init='ball' needs init_centre (e.g. truths or the fiducial point)")
        sd = (torch.as_tensor(np.asarray(init_sd), dtype=torch.float32, device=device) if init_sd is not None
              else ((high - low)[0, 0] / np.sqrt(12.0)))
        centre = torch.as_tensor(init_centre, dtype=torch.float32, device=device)
        centre = torch.minimum(torch.maximum(centre, low[:, 0] + 1e-4 * (high - low)[:, 0]), high[:, 0] - 1e-4 * (high - low)[:, 0])
        # a centre in a narrow corner of the support (box edge and constraints at once) can leave too few ball
        # draws inside it: retry with a smaller ball (it only sets the start; burn-in is discarded)
        for k in range(5):
            try:
                init = torchemcee.ball(centre, cfg.init_scale * 0.3 ** k * sd, cfg.nwalkers, log_prob_fn=prior_lp,
                                       device=device, generator=gen)
                break
            except ValueError:
                if k == 4:
                    raise
    if cfg.logit_box:
        box = LogitBox(low, high)
        target_fn, start = box.wrap(log_prob), box.to_u(init)
    else:
        box, target_fn, start = None, log_prob, init

    sampler = torchemcee.EnsembleSampler(target_fn, ndim, cfg.nwalkers, B, moves=_moves(cfg),
                                         device=device, generator=gen)
    thin = max(1, cfg.thin)
    block = max(1, cfg.block // thin)
    with torch.no_grad():
        if cfg.adaptive:
            sampler.run_mcmc(start, block, discard=cfg.burn_in, thin_by=thin, progress=False)
            while True:
                chain = sampler.get_chain(numpy=False)
                chain = chain[len(chain) // 2:]
                diag = diagnose(chain, sampler.state.accepted, sampler.iteration, cfg)
                if verbose and cfg.progress:
                    print(f"      block: {sampler.iteration} steps, converged {diag['converged'].float().mean():.3f}, "
                          f"ess_min {diag['ess'].min():.0f}, rhat_max {diag['rhat'].max():.4f}, "
                          f"{time.time() - t0:.0f}s", flush=True)
                if bool(diag["converged"].all()) or sampler.iteration >= cfg.max_steps:
                    break
                sampler.run_mcmc(None, block, thin_by=thin, progress=False)
        else:
            sampler.run_mcmc(start, max(1, cfg.max_steps // thin), discard=cfg.burn_in, thin_by=thin, progress=False)
            chain = sampler.get_chain(numpy=False)
            chain = chain[len(chain) // 2:]
            diag = diagnose(chain, sampler.state.accepted, sampler.iteration, cfg)

    flat = chain.permute(1, 0, 2, 3).reshape(B, -1, ndim)
    pick = torch.randint(flat.shape[1], (cfg.n_samples,), generator=gen, device=device)
    picked = flat[:, pick]
    samples = (box.to_theta(picked) if box is not None else picked).cpu().numpy()
    tau = (diag["tau"] * thin).cpu().numpy()
    out = {
        "converged": diag["converged"].cpu().numpy(),
        "tau_max": tau.max(1), "rhat_max": diag["rhat"].max(1).values.cpu().numpy(),
        "ess_min": diag["ess"].min(1).values.cpu().numpy(),
        "acceptance": diag["acceptance"].mean(1).cpu().numpy(),
        "steps": int(sampler.iteration), "seconds": time.time() - t0,
        "config": asdict(cfg),
    }
    if verbose:
        print(f"    mcmc {B} x {cfg.nwalkers}: {sampler.iteration} steps, converged {out['converged'].mean():.3f}, "
              f"tau_max {tau.max():.0f}, rhat_max {out['rhat_max'].max():.4f}, ess_min {out['ess_min'].min():.0f}, "
              f"{out['seconds']:.0f}s", flush=True)
    return samples, out
