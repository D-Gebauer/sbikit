"""Batched posterior sampling with torchemcee.

All targets (observations or test simulations) are sampled in one ensemble with a separate set of
walkers each, so that hundreds of posteriors cost one batched flow evaluation per step. By default
walkers move in the logit of each target's prior box (they never leave it), with
differential-evolution and snooker moves (ter Braak & Vrugt 2008), and the chain grows in blocks
after burn-in until, on its second half, every target has grouped split R-hat < ``rhat``,
ESS >= ``min_ess`` and length >= ``min_tau`` tau in every parameter (or ``max_steps``). Samples are
drawn from that second half.

Step accounting (one step = one move of every walker): ``burn_in`` steps are taken and discarded,
then at most ``max_steps`` further steps, of which every ``thin``-th is kept (the first kept state
is ``thin`` steps after burn-in). ``block`` and ``max_steps`` are rounded down to multiples of
``thin``; a new block continues from the last state of the previous one, so no step is thrown away
and the thinning is regular across blocks. This is the same with ``adaptive=True`` (convergence
check after every block) and ``adaptive=False`` (one check after ``max_steps``).

The snooker move needs torchemcee >= 0.1.1. The ``DESnookerMove`` of emcee 3.1.6 and of torchemcee
0.1.0 does not follow ter Braak & Vrugt (2008): it scales the proposal by ``(x - z) / sqrt|x - z|``
and uses the exponent (d - 1) / 2, where the paper has the unit vector and d - 1. On a flat 9-d box
it gives a standard deviation 16 % too small alone, and 3 % too small in the 0.8 DE / 0.2 snooker
mix. torchemcee 0.1.1 has the corrected move; the test suite checks every move on that box.
"""
from __future__ import annotations

import contextlib
import copy
import itertools
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchemcee
from torchemcee.integrations.sbi import flow_log_prob_fn

from ._utils import ConfigBase, Registry, as_numpy, did_you_mean, logger, resolve_device
from .priors import Prior

__all__ = ["MCMCConfig", "Posteriors", "sample_posteriors", "LogitBox", "grouped_rhat", "diagnose",
           "MOVES", "MOVE_PRESETS", "register_move"]


# ------------------------------------------------------------------------------ moves
MOVES = Registry("move")
for _name, _cls in (("stretch", torchemcee.StretchMove), ("walk", torchemcee.WalkMove), ("de", torchemcee.DEMove),
                    ("snooker", torchemcee.DESnookerMove), ("kde", torchemcee.KDEMove),
                    ("gaussian", torchemcee.GaussianMove)):
    MOVES.register(_name, _cls)

# name -> anything ``MCMCConfig.moves`` accepts; add entries to define further presets
MOVE_PRESETS = {
    "de_snooker": [["de", 0.8, {}], ["snooker", 0.2, {}]],   # ter Braak & Vrugt 2008
    "de": [["de", 1.0, {}]],
    "stretch": [["stretch", 1.0, {}]],                       # StretchMove(a=stretch_a)
    "walk": [["walk", 1.0, {}]],
    "kde": [["kde", 1.0, {}]],
}


def register_move(name: str, cls=None, overwrite: bool = False):
    """Make a torchemcee ``Move`` subclass available by name in ``MCMCConfig.moves``.

    Args:
        name: the name used in move lists, e.g. ``[["mymove", 1.0, {"scale": 0.1}]]``.
        cls: the class (called with the keyword arguments of the list entry); without it, a
            decorator is returned.
        overwrite: replace an existing entry.

    Example:
        >>> @register_move("small_de")
        ... class SmallDE(torchemcee.DEMove):
        ...     def __init__(self):
        ...         super().__init__(gamma0=0.3)
        >>> MCMCConfig(moves=[["de", 0.7], ["small_de", 0.3]])
    """
    return MOVES.register(name, cls, overwrite)


def _check_torchemcee():
    """torchemcee < 0.1.1 has emcee's biased DESnookerMove (too narrow posteriors)."""
    v = tuple(int(p) for p in torchemcee.__version__.split(".")[:3] if p.isdigit())
    if v < (0, 1, 1):
        raise RuntimeError(f"torchemcee {torchemcee.__version__} has the biased emcee snooker move; "
                           "install torchemcee >= 0.1.1 (git tag v0.1.1)")


_MOVES_FORMAT = ("a preset name, a torchemcee Move, or a list whose items are a Move, [move_or_name, weight] or "
                 "[move_or_name, weight, kwargs]")


def _weighted_moves(cfg) -> list:
    """``cfg.moves`` as a list of (Move instance, weight)."""
    spec = cfg.moves
    if isinstance(spec, str):
        if spec in MOVE_PRESETS:
            spec = MOVE_PRESETS[spec]
        elif spec in MOVES:
            spec = [[spec, 1.0]]
        else:
            valid = list(MOVE_PRESETS) + list(MOVES)
            raise ValueError(f"unknown moves {spec!r}{did_you_mean(spec, valid)}; presets: {sorted(MOVE_PRESETS)}, "
                             f"moves: {sorted(MOVES)}")
    if isinstance(spec, (torchemcee.Move, str)):
        spec = [spec]
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        raise ValueError(f"moves must be {_MOVES_FORMAT}; got {spec!r}")
    out = []
    for item in spec:
        if isinstance(item, (torchemcee.Move, str)):
            item = [item, 1.0]
        if not isinstance(item, (list, tuple)) or not 2 <= len(item) <= 3:
            raise ValueError(f"moves must be {_MOVES_FORMAT}; got the item {item!r}")
        move, weight, kwargs = (list(item) + [{}])[:3]
        kwargs = dict(kwargs or {})
        if isinstance(move, str):
            cls = MOVES[move]
            if cls is torchemcee.StretchMove:
                kwargs.setdefault("a", cfg.stretch_a)
            try:
                move = cls(**kwargs)
            except TypeError as err:
                raise ValueError(f"move {item[0]!r}: {err}") from err
            if not isinstance(move, torchemcee.Move):
                raise ValueError(f"move {item[0]!r} is registered as {getattr(cls, '__name__', cls)!r}, which does "
                                 f"not build a torchemcee Move (got a {type(move).__name__}); register a Move subclass")
        elif not isinstance(move, torchemcee.Move):
            raise ValueError(f"moves must be {_MOVES_FORMAT}; got the item {item!r}")
        elif kwargs:
            raise ValueError(f"move {move!r} is an instance already; keyword arguments {kwargs} only apply to names")
        if isinstance(weight, bool) or not isinstance(weight, (int, float, np.integer, np.floating)) or not weight > 0:
            raise ValueError(f"move weights must be positive numbers; got {weight!r} for {item[0]!r}")
        if isinstance(move, torchemcee.DESnookerMove):
            _check_torchemcee()
        out.append((move, float(weight)))
    return out


def _moves(cfg):
    """``cfg.moves`` in the form ``torchemcee.EnsembleSampler(moves=)`` takes."""
    out = _weighted_moves(cfg)
    return out[0][0] if len(out) == 1 else out


def _describe(value):
    """JSON-friendly form of a config value: moves by their repr, arrays as lists."""
    if isinstance(value, torchemcee.Move):
        if type(value).__repr__ is not object.__repr__:
            return repr(value)
        args = ", ".join(f"{k}={v!r}" for k, v in vars(value).items()
                         if not k.startswith("_") and isinstance(v, (bool, int, float, str, type(None))))
        return f"{type(value).__name__}({args})"
    if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _describe(v) for k, v in value.items()}
    return value


# ------------------------------------------------------------------------------ config and result
_INT_FIELDS = ("nwalkers", "burn_in", "block", "max_steps", "thin", "rhat_groups", "n_samples", "targets_per_batch",
               "seed")
_FLOAT_FIELDS = ("stretch_a", "init_scale", "rhat", "min_ess", "min_tau", "min_acceptance")
_BOOL_FIELDS = ("logit_box", "adaptive", "progress")


@dataclass
class MCMCConfig(ConfigBase):
    """Settings of :func:`sample_posteriors`.

    One step is one move of every walker. ``burn_in`` steps are discarded, then at most
    ``max_steps`` further steps are run, of which every ``thin``-th is kept.

    Attributes:
        nwalkers: walkers per target (even, and more than twice the number of parameters).
        moves: a preset name (``"de_snooker"``, ``"de"``, ``"stretch"``, ``"walk"``, ``"kde"``; see
            ``MOVE_PRESETS``); a torchemcee ``Move`` instance; or a list whose items are a ``Move``
            instance, ``[move_or_name, weight]`` or ``[move_or_name, weight, {kwargs}]``, with the
            names ``stretch | walk | de | snooker | kde | gaussian`` or any name added with
            :func:`register_move`, e.g. ``[["de", 0.8], ["gaussian", 0.2, {"cov": 0.01}]]``.
        stretch_a: scale ``a`` of stretch moves given by name without their own ``a``.
        logit_box: sample u = logit((theta - low) / (high - low)) instead of theta. Walkers then
            cannot leave the prior box, and posteriors that pile up at a bound mix better. Needs
            finite bounds; use False for priors with infinite bounds.
        init: ``"auto"`` (``init_walkers`` if given, else a ball if ``init_centre`` is given, else
            prior draws), ``"ball"`` (Gaussian ball around ``init_centre``) or ``"prior"`` (per
            target, the ``nwalkers`` prior draws with the highest posterior density out of
            20 x ``nwalkers``; this costs 20 likelihood evaluations of the ensemble). To start from
            plain prior draws pass them as ``init_walkers``.
        init_scale: ball standard deviation in units of ``init_sd``.
        burn_in: steps discarded at the start.
        block: steps between convergence checks (rounded down to a multiple of ``thin``).
        max_steps: maximum number of steps after burn-in (rounded down to a multiple of ``thin``).
        thin: keep every ``thin``-th step.
        adaptive: stop as soon as every target has converged; False runs exactly ``max_steps``
            after burn-in (``block`` is not used).
        rhat: convergence threshold of the grouped split R-hat.
        rhat_groups: number of walker groups used as chains in R-hat.
        min_ess: required effective sample size per parameter.
        min_tau: required chain length in integrated autocorrelation times.
        min_acceptance: required mean acceptance fraction.
        n_samples: samples returned per target, drawn without replacement from the second half of
            the chain (with replacement, and a warning, if it holds fewer); None returns all of it.
        targets_per_batch: split very many targets into several ensembles of this size (memory).
        seed: seed of the sampler; None draws one from torch's global random state.
        progress: log a line per convergence check (for long single runs).

    Raises:
        TypeError: for an unknown field (from :meth:`make`).
        ValueError: for an invalid value.

    Example:
        >>> MCMCConfig(nwalkers=128, moves=[["de", 0.8], ["snooker", 0.2]], max_steps=20000)
        >>> MCMCConfig.make({"nwalkers": 128, "init": "prior"})
    """

    nwalkers: int = 256
    moves: Union[str, list, torchemcee.Move] = "de_snooker"
    stretch_a: float = 2.0          # for stretch moves given by name
    logit_box: bool = True          # sample u = logit((theta - low) / (high - low))
    init: str = "auto"              # auto | ball (around init_centre) | prior
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
    n_samples: Optional[int] = 2000           # per target, from the second half of the chain; None: all of it
    targets_per_batch: Optional[int] = None   # split very many targets into several ensembles
    seed: Optional[int] = 0
    progress: bool = False          # log a line per convergence check (long single runs)

    _choices = {"init": ("auto", "ball", "prior")}

    def _normalise(self) -> None:
        for name in _INT_FIELDS:
            v = getattr(self, name)
            if isinstance(v, (np.integer, float, np.floating)) and float(v).is_integer():
                setattr(self, name, int(v))
        for name in _FLOAT_FIELDS:
            v = getattr(self, name)
            if isinstance(v, (np.integer, np.floating)):
                setattr(self, name, v.item())
        for name in _BOOL_FIELDS:
            if isinstance(getattr(self, name), np.bool_):
                setattr(self, name, bool(getattr(self, name)))
        if isinstance(self.moves, tuple):
            self.moves = list(self.moves)

    def validate(self) -> None:
        r = self._require
        for name in _BOOL_FIELDS:
            v = getattr(self, name)
            r(isinstance(v, bool), f"{name} must be True or False, got {v!r}")
        for name in _FLOAT_FIELDS:
            v = getattr(self, name)
            r(isinstance(v, (int, float)) and not isinstance(v, bool), f"{name} must be a number, got {v!r}")
        for name in _INT_FIELDS:
            v = getattr(self, name)
            optional = name in ("n_samples", "targets_per_batch", "seed")
            r((v is None and optional) or (isinstance(v, int) and not isinstance(v, bool)),
              f"{name} must be an integer{' or None' if optional else ''}, got {v!r}")
        r(self.nwalkers >= 2 and self.nwalkers % 2 == 0, f"nwalkers must be even and >= 2, got {self.nwalkers}")
        r(self.thin >= 1, f"thin must be >= 1, got {self.thin}")
        r(self.block >= self.thin, f"block ({self.block}) must be >= thin ({self.thin})")
        r(self.burn_in >= 0, f"burn_in must be >= 0, got {self.burn_in}")
        if self.adaptive:
            r(self.max_steps >= self.block, f"max_steps ({self.max_steps}) must be >= block ({self.block}); "
              "lower block, raise max_steps, or use adaptive=False to run a fixed number of steps")
        else:
            r(self.max_steps >= self.thin, f"max_steps ({self.max_steps}) must be >= thin ({self.thin})")
        r(self.rhat > 1, f"rhat must be > 1, got {self.rhat}")
        r(self.rhat_groups >= 1, f"rhat_groups must be >= 1, got {self.rhat_groups}")
        r(self.min_ess >= 0, f"min_ess must be >= 0, got {self.min_ess}")
        r(self.min_tau >= 0, f"min_tau must be >= 0, got {self.min_tau}")
        r(0 <= self.min_acceptance < 1, f"min_acceptance must be in [0, 1), got {self.min_acceptance}")
        r(self.init_scale > 0, f"init_scale must be > 0, got {self.init_scale}")
        r(self.stretch_a > 1, f"stretch_a must be > 1, got {self.stretch_a}")
        r(self.n_samples is None or self.n_samples >= 1, f"n_samples must be >= 1 or None, got {self.n_samples}")
        r(self.targets_per_batch is None or self.targets_per_batch >= 1,
          f"targets_per_batch must be >= 1 or None, got {self.targets_per_batch}")
        r(self.seed is None or self.seed >= 0, f"seed must be >= 0 or None, got {self.seed}")
        option = "block" if self.adaptive else "max_steps"
        kept = getattr(self, option) // self.thin
        r(kept - kept // 2 >= 4, f"{option} = {getattr(self, option)} with thin = {self.thin} keeps {kept} steps "
          f"before the {'first ' if self.adaptive else ''}convergence check, but the diagnostics need at least 7 "
          f"(4 in the second half): raise {option} to >= {7 * self.thin} or lower thin")
        for move, _ in _weighted_moves(self):
            need = 2 * int(getattr(move, "nsplits", 1))
            r(self.nwalkers >= need, f"nwalkers = {self.nwalkers} is too few for {type(move).__name__}, which "
              f"splits the walkers into {need // 2} groups: use nwalkers >= {need}")

    def to_dict(self) -> dict:
        """The fields as a plain, JSON-friendly dict. ``Move`` instances in ``moves`` are given by
        their repr, so such a dict documents the run but cannot rebuild the config."""
        return {name: _describe(getattr(self, name)) for name in self.fields()}


def _npz_path(path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


@dataclass(repr=False, eq=False)
class Posteriors:
    """Result of :func:`sample_posteriors` for B targets.

    Attributes:
        samples: (B, n_samples, P) posterior samples, in the sampled parameters.
        diagnostics: convergence diagnostics, computed on the second half of the kept chain
            (see Units below).
        derived: (B, n_samples, P') = transform(samples) if a transform was given, else None.
        names: names of the sampled parameters (the prior's names), or None.
        chain: with ``keep_chain=True``, the (n, B, W, P) second half of the thinned chain in the
            sampled parameters (n kept steps, W walkers), else None.
        log_prob: with ``keep_chain=True``, the (n, B, W) log posterior density (log prior + log
            likelihood, unnormalised) of ``chain`` in the sampled parameters, without the Jacobian
            of the logit map; else None.

    Units of ``diagnostics``:
        - per target, (B,): ``converged`` (bool); ``rhat_max``, ``ess_min``, ``tau_max`` (worst
          parameter); ``acceptance`` (mean acceptance fraction over all steps, burn-in included).
        - per target and parameter, (B, P): ``rhat`` (grouped split R-hat), ``ess`` (effective
          sample size of the second-half chain, all walkers together), ``tau`` (integrated
          autocorrelation time in steps, not in kept steps). Values that cannot be estimated are
          nan, and the target then counts as not converged.
        - per ensemble (lists with one entry per ``targets_per_batch`` batch): ``steps`` (steps
          taken after burn-in), ``burn_in``, ``seconds``, ``config``.

    With ``targets_per_batch`` and ``adaptive=True`` the batches can stop after different numbers
    of steps; ``chain`` / ``log_prob`` (and ``samples`` for ``n_samples=None``) then hold the last
    steps of each batch, as many as the shortest batch has.

    Example:
        >>> post = sample_posteriors(nle, s_obs, prior)
        >>> post
        Posteriors(targets=20, samples=2000, parameters=3, converged=20/20)
        >>> post.mean().shape, post.quantile([0.16, 0.84]).shape
        ((20, 3), (2, 20, 3))
        >>> post.save("posteriors.npz"); post = Posteriors.load("posteriors.npz")
    """

    samples: np.ndarray                  # (B, n_samples, P) in the sampled parameters
    diagnostics: dict = field(default_factory=dict)   # per target arrays (B,) / (B, P) + run info
    derived: Optional[np.ndarray] = None  # (B, n_samples, P') = transform(samples), if a transform was given
    names: Optional[list] = None          # names of the sampled parameters
    chain: Optional[np.ndarray] = None    # (n, B, W, P) second half of the thinned chain (keep_chain=True)
    log_prob: Optional[np.ndarray] = None  # (n, B, W) log posterior density of chain (keep_chain=True)

    @property
    def converged(self) -> np.ndarray:
        """(B,) bool: which targets met every convergence criterion."""
        return self.diagnostics["converged"]

    def __len__(self) -> int:
        """Number of targets."""
        return len(self.samples)

    def __repr__(self) -> str:
        B, n, P = np.shape(self.samples)
        parts = [f"targets={B}", f"samples={n}", f"parameters={P}"]
        conv = self.diagnostics.get("converged")
        if conv is not None:
            parts.append(f"converged={int(np.sum(conv))}/{B}")
        if self.names is not None:
            parts.append(f"names={list(self.names)}")
        if self.derived is not None:
            parts.append(f"derived={np.shape(self.derived)[-1]}")
        if self.chain is not None:
            parts.append(f"chain={tuple(np.shape(self.chain))}")
        return f"Posteriors({', '.join(parts)})"

    def mean(self) -> np.ndarray:
        """(B, P) posterior means."""
        return self.samples.mean(1)

    def std(self) -> np.ndarray:
        """(B, P) posterior standard deviations."""
        return self.samples.std(1)

    def quantile(self, q) -> np.ndarray:
        """Posterior quantiles: (B, P) for a number ``q`` in [0, 1], (len(q), B, P) for a sequence."""
        return np.quantile(self.samples, q, axis=1)

    def summary(self) -> dict:
        """Per-target summaries: ``mean``, ``std``, ``median``, ``q16``, ``q84`` (each (B, P)) and
        ``converged`` ((B,) bool, if the diagnostics have it)."""
        q16, median, q84 = self.quantile([0.16, 0.5, 0.84])
        out = {"mean": self.mean(), "std": self.std(), "median": median, "q16": q16, "q84": q84}
        if "converged" in self.diagnostics:
            out["converged"] = np.asarray(self.diagnostics["converged"])
        return out

    def save(self, path) -> Path:
        """Write everything to one ``.npz`` file (the suffix is added if missing).

        The arrays are stored as they are; ``diagnostics`` and ``names`` as JSON strings, so the
        file loads without pickle. Returns the path written.
        """
        def encode(v):
            if isinstance(v, np.ndarray):
                return {"__array__": v.tolist(), "dtype": str(v.dtype)}
            return v.item() if isinstance(v, np.generic) else v

        def jsonable(v):
            described = _describe(v)
            return repr(v) if described is v else described

        path = _npz_path(path)
        arrays = {k: np.asarray(getattr(self, k)) for k in ("samples", "derived", "chain", "log_prob")
                  if getattr(self, k) is not None}
        info = {"names": None if self.names is None else [str(n) for n in self.names],
                "diagnostics": {k: encode(v) for k, v in self.diagnostics.items()}}
        np.savez(path, **arrays, info=np.array(json.dumps(info, default=jsonable)))
        return path

    @classmethod
    def load(cls, path) -> "Posteriors":
        """Read a file written by :meth:`save`."""
        def decode(v):
            return np.asarray(v["__array__"], dtype=v["dtype"]) if isinstance(v, dict) and "__array__" in v else v

        with np.load(_npz_path(path), allow_pickle=False) as f:
            if "info" not in f or "samples" not in f:
                raise ValueError(f"{path} was not written by Posteriors.save")
            info = json.loads(str(f["info"]))
            arrays = {k: f[k] for k in ("samples", "derived", "chain", "log_prob") if k in f}
        return cls(arrays["samples"], {k: decode(v) for k, v in info["diagnostics"].items()}, arrays.get("derived"),
                   info["names"], arrays.get("chain"), arrays.get("log_prob"))


# ------------------------------------------------------------------------------ logit box and diagnostics
class LogitBox:
    """The map u = logit((theta - low) / (high - low)) between a box and the real line.

    Args:
        low, high: finite box bounds, broadcastable to theta (e.g. (B, 1, P) for per-target boxes).
    """

    def __init__(self, low, high):
        self.low, self.width = low, high - low

    def to_theta(self, u):
        """Unbounded u -> theta inside the box."""
        return self.low + self.width * torch.sigmoid(u)

    def to_u(self, theta):
        """theta -> u; points on the bounds are moved 1e-6 of the width inside."""
        return torch.logit(((theta - self.low) / self.width).clamp(1e-6, 1 - 1e-6))

    def log_jacobian(self, u):
        """log |d theta / d u|, summed over the parameters: (..., P) -> (...)."""
        return (F.logsigmoid(u) + F.logsigmoid(-u)).sum(-1) + torch.log(self.width).sum(-1)

    def wrap(self, log_prob_fn):
        """The density in u of a log density in theta: ``log_prob_fn(theta(u)) + log |d theta / d u|``."""
        def log_prob(u):
            return log_prob_fn(self.to_theta(u)) + self.log_jacobian(u)

        return log_prob


def _as_chain(chain) -> torch.Tensor:
    """``chain`` as a floating (n, B, W, P) tensor (arrays are converted, tensors stay on their device)."""
    chain = torch.as_tensor(chain)
    if not chain.is_floating_point():
        chain = chain.to(torch.float32)
    if chain.dim() != 4:
        raise ValueError(f"chain must have shape (steps, targets, walkers, parameters); got {tuple(chain.shape)}")
    return chain


def grouped_rhat(chain, k):
    """Split R-hat (Gelman-Rubin) with k groups of walkers as the chains.

    The walkers of an ensemble are not independent chains, so they are pooled into ``k`` groups,
    and each group's first and second half are compared.

    Args:
        chain: (n, B, W, P) tensor or array of n steps, B targets, W walkers (e.g. ``Posteriors.chain``).
        k: number of walker groups (at most W // 2).

    Returns:
        (B, P) tensor; nan for chains of fewer than 4 steps (each half needs 2).

    Raises:
        ValueError: if ``chain`` is not 4D.
    """
    chain = _as_chain(chain)
    n, n_targets, nwalkers, ndim = chain.shape
    k = max(1, min(int(k), nwalkers // 2))
    half = n // 2
    if half < 2 or nwalkers < 1:
        return torch.full((n_targets, ndim), torch.nan, dtype=chain.dtype, device=chain.device)
    x = chain[:2 * half, :, : (nwalkers // k) * k].reshape(2, half, n_targets, k, nwalkers // k, ndim)
    x = x.permute(2, 5, 0, 3, 1, 4).reshape(n_targets, ndim, 2 * k, -1)
    size = x.shape[-1]
    within = x.var(-1).mean(-1)
    between = size * x.mean(-1).var(-1)
    return torch.sqrt(((size - 1) / size * within + between / size) / within)


def diagnose(chain, accepted, nsteps, cfg=None):
    """Convergence diagnostics per target.

    Args:
        chain: (n, B, W, P) thinned chain, a tensor or an array (e.g. ``Posteriors.chain``).
        accepted: (B, W) accepted-proposal counts of the sampler state, a tensor or an array.
        nsteps: number of (unthinned) steps the counts were accumulated over.
        cfg: thresholds, a :class:`MCMCConfig` or dict (default: the defaults).

    Returns:
        dict of tensors (on the device of ``chain``, also for array inputs): ``tau`` (integrated
        autocorrelation time, in steps of ``chain``, i.e. thinned steps), ``rhat``, ``ess`` (each
        (B, P)), ``acceptance`` (B, W) and ``converged`` (B,). Where the chain is too short for an
        estimate (non-positive tau, fewer than 4 steps for R-hat), the value is nan and the target
        is not converged.

    Raises:
        ValueError: if ``chain`` is not 4D or ``accepted`` is not (B, W).

    Example:
        >>> post = sample_posteriors(nle, s_obs, prior, keep_chain=True)
        >>> grouped_rhat(post.chain, 4).shape                 # your own diagnostics on the kept chain
        torch.Size([B, P])
    """
    cfg = MCMCConfig.make(cfg)
    chain = _as_chain(chain)
    n, n_targets, nwalkers, ndim = chain.shape
    accepted = torch.as_tensor(accepted, device=chain.device)
    if accepted.shape != (n_targets, nwalkers):
        raise ValueError(f"accepted must have shape ({n_targets}, {nwalkers}) (targets x walkers of chain); got "
                         f"{tuple(accepted.shape)}")
    if n < 2:
        tau = torch.full((n_targets, ndim), torch.nan, dtype=torch.float64, device=chain.device)
        rhat = tau.to(chain.dtype)
    else:
        chunk = max(1, int(1e8 // max(1, n * nwalkers * ndim)))
        tau, rhat = [], []
        for i in range(0, n_targets, chunk):
            part = chain[:, i:i + chunk]
            tau.append(torchemcee.integrated_autocorr_time(part, tol=cfg.min_tau, quiet=True))
            rhat.append(grouped_rhat(part, cfg.rhat_groups))
        tau, rhat = torch.cat(tau), torch.cat(rhat)
        tau = torch.where(tau > 0, tau, torch.full_like(tau, torch.nan))
    ess = n * nwalkers / tau
    acceptance = torchemcee.acceptance_fraction(accepted, max(1, int(nsteps)))
    converged = ((rhat < cfg.rhat).all(-1) & (ess >= cfg.min_ess).all(-1)
                 & (n >= cfg.min_tau * tau).all(-1) & (acceptance.mean(-1) > cfg.min_acceptance))
    return {"tau": tau, "rhat": rhat, "ess": ess, "acceptance": acceptance, "converged": converged}


@contextlib.contextmanager
def _quiet_torchemcee():
    """Silence torchemcee's short-chain warnings inside the block (they are reported through the
    convergence flags), then restore the logger's level."""
    log = logging.getLogger("torchemcee")
    previous = log.level
    log.setLevel(max(previous, logging.ERROR))
    try:
        yield
    finally:
        log.setLevel(previous)


# ------------------------------------------------------------------------------ input handling
_BALL_ROUNDS, _BALL_SHRINKS, _REDRAWS = 100, 5, 10
# start from prior draws: the walkers of a target are the best of _PRIOR_ROUNDS x nwalkers draws, and in the logit
# box the draws keep a distance of _PRIOR_EDGE box widths from the bounds
_PRIOR_ROUNDS, _PRIOR_EDGE = 20, 0.02


def _group(objects):
    """The distinct objects (by identity) and, for every item, the index of its object."""
    uniq, index, which = [], {}, []
    for o in objects:
        if id(o) not in index:
            index[id(o)] = len(uniq)
            uniq.append(o)
        which.append(index[id(o)])
    return uniq, which


def _as_priors(prior, n):
    priors = list(prior) if isinstance(prior, (list, tuple)) else [prior] * n
    if len(priors) != n:
        raise ValueError(f"prior: got {len(priors)} priors for {n} targets (rows of x_obs); pass one prior for all "
                         "targets or one per target")
    for p in _group(priors)[0]:
        missing = [a for a in ("low", "high", "ndim", "log_prob", "sample", "to") if not hasattr(p, a)]
        if missing:
            raise TypeError(f"prior must be a sbikit Prior or a list of them; got {type(p).__name__}, which has no "
                            f"{', '.join(missing)}")
    return priors


def _width(likelihood, name):
    """``likelihood.n_x`` / ``n_theta`` as an int, or None if it has none."""
    n = getattr(likelihood, name, None)
    return None if n is None else int(n)


def _per_target(a, name, B, P, scalar=False):
    """``a`` as a (B, P) float64 array of its own from (P,) (the same for every target) or (B, P)."""
    a = as_numpy(a)
    if scalar and a.ndim == 0:
        a = np.full(P, float(a))
    shape = a.shape
    if a.shape == (P,):
        a = np.broadcast_to(a, (B, P)).copy()               # broadcast_to gives a read-only view
    if a.shape != (B, P):
        raise ValueError(f"{name} must have shape ({P},) (one point for all targets) or ({B}, {P}) "
                         f"(targets x parameters); got {shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or inf")
    return a


def _on_device(transform, device):
    """The transform to sample with on ``device``; the caller's object is left where it is."""
    if transform is None:
        return None
    if not callable(transform):
        raise TypeError(f"transform must be callable (psi -> theta), got {type(transform).__name__}")
    if isinstance(transform, torch.nn.Module):
        return copy.deepcopy(transform).to(device)
    return transform.to(device) if hasattr(transform, "to") else transform


def _check_transform(tr, priors, n_targets, n_theta, device, seed):
    """Names and output shape of the transform against the priors and the likelihood."""
    names_in = getattr(tr, "names_in", None)
    for p in priors:
        names = getattr(p, "names", None)
        if names_in is not None and names is not None and [str(n) for n in names_in] != [str(n) for n in names]:
            raise ValueError(f"transform expects the parameters {list(names_in)}, the prior has {list(names)}; the "
                             "prior must be on the transform's inputs, in the same order")
    gen = torch.Generator(device=device).manual_seed(int(seed))
    psi = torch.as_tensor(priors[0].sample(2 * n_targets, generator=gen), dtype=torch.float32, device=device)
    psi = psi.reshape(n_targets, 2, -1)                    # the layout of the sampler: (targets, walkers, P)
    with torch.no_grad():
        theta = tr(psi)
    if not isinstance(theta, torch.Tensor) or theta.shape[:-1] != psi.shape[:-1]:
        got = tuple(theta.shape) if isinstance(theta, torch.Tensor) else type(theta).__name__
        raise ValueError(f"transform must map a (..., {psi.shape[-1]}) tensor to a (..., n_theta) tensor; got {got} "
                         f"for an input of shape {tuple(psi.shape)}")
    if n_theta is not None and theta.shape[-1] != n_theta:
        raise ValueError(f"transform returns {theta.shape[-1]} columns but the likelihood is conditioned on "
                         f"{n_theta} parameters")


def _tails(arrays, axis):
    """The arrays cut to their shortest length along ``axis`` (keeping the end), and whether any was cut."""
    n = min(a.shape[axis] for a in arrays)
    cut = any(a.shape[axis] != n for a in arrays)
    return [np.take(a, range(a.shape[axis] - n, a.shape[axis]), axis=axis) if a.shape[axis] != n else a
            for a in arrays], cut


def _some(indices, n=8):
    indices = [int(i) for i in indices]
    return f"{indices[:n]}{' ...' if len(indices) > n else ''}"


# ------------------------------------------------------------------------------ sampling
def sample_posteriors(likelihood, x_obs, prior: Union[Prior, Sequence[Prior]], config=None,
                      init_centre=None, init_sd=None, device=None, verbose: bool = True,
                      transform=None, *, init_walkers=None, keep_chain: bool = False,
                      callback: Optional[Callable[[dict], Optional[bool]]] = None) -> Posteriors:
    """Sample p(theta | x_b) for every row b of ``x_obs``, all in one batched ensemble.

    Args:
        likelihood: an :class:`~sbikit.nle.NLE` or :class:`~sbikit.nle.Ensemble`, or anything with
            ``log_prob(input, condition)`` in sbi's layout (``input`` (1, M, n_x) summaries,
            ``condition`` (M, n_theta) parameters, returning M values). It is evaluated on
            ``device`` and is not moved.
        x_obs: (B, n_x) summaries to condition on, or (n_x,) for a single target.
        prior: one :class:`~sbikit.priors.Prior` for all targets, or a list with one per target.
            Pass the same object for targets that share a prior: priors with density terms or
            constraints are evaluated once per distinct object.
        config: :class:`MCMCConfig` or dict.
        init_centre: ball centres, (B, P) or one (P,) point for all targets (truths for test
            sims, the fiducial point or a MAP for data); required for ``init="ball"``. Centres
            outside the prior box are moved to its edge.
        init_sd: scale of the ball, (P,) or (B, P) (e.g. the std of the training parameters);
            default: each target's own prior box width / sqrt(12).
        device: torch device of the sampler, ``"auto"`` included (default: the likelihood's
            ``device`` attribute, else ``"cpu"``).
        verbose: log one summary line per ensemble (and, with ``config.progress``, one per
            convergence check) through the ``sbikit`` logger.
        transform: optional map psi -> theta (a :class:`~sbikit.transforms.Transform` or any callable
            on (..., P) tensors; it is called with (targets of one ensemble, n, P) tensors on
            ``device``). The sampler then moves in psi (``prior``, ``init_centre``, ``init_sd``
            and ``init_walkers`` are in psi) and the likelihood is evaluated at
            theta = transform(psi). Use it to infer the parameters of a model the flow was not
            trained on, e.g. a model that predicts the per-bin amplitudes the flow is conditioned
            on. ``Posteriors.derived`` holds transform(samples). The object is not moved: the
            sampler works with what ``transform.to(device)`` returns (a deep copy for a bare
            ``nn.Module``).
        init_walkers: explicit start positions in the sampled parameters, (B, W, P) or (W, P) for
            all targets, with W = ``config.nwalkers``; used with ``init="auto"``. The last step of
            an earlier run (``post.chain[-1]`` with ``keep_chain=True``) continues that run.
            A warning names start arguments that the chosen ``init`` does not use.
        keep_chain: also return the second half of the thinned chain and its log posterior density
            (``Posteriors.chain``, ``Posteriors.log_prob``).
        callback: called after every convergence check with a dict: ``steps`` (after burn-in),
            ``converged`` ((B,) bool), ``rhat_max``, ``ess_min``, ``tau_max`` ((B,) arrays, as in
            ``Posteriors.diagnostics``), ``seconds`` and ``batch`` (index of the
            ``targets_per_batch`` ensemble the arrays belong to). If it returns True, that ensemble
            stops and is returned as it is, with the flags as diagnosed.

    Returns:
        :class:`Posteriors`. Targets that reached ``max_steps`` without converging are returned as
        well, flagged in ``.converged``, and a ``RuntimeWarning`` reports them.

    Raises:
        ValueError: for inputs whose shapes do not fit together (``x_obs`` against the likelihood,
            the number and size of the priors, the transform, the initial positions), for priors
            with infinite bounds combined with ``logit_box=True``, and for start positions where
            the prior, the likelihood or the transform is not finite.

    Example:
        >>> prior = Prior.box_from_data(theta_train)
        >>> post = sample_posteriors(nle, s_test, prior, MCMCConfig(nwalkers=128),
        ...                          init_centre=theta_test, init_sd=theta_train.std(0))
        >>> post.samples.shape, post.converged.all()
        ((n_test, 2000, P), True)
        >>> post = sample_posteriors(nle, s_obs, prior)          # walkers start from the best prior draws
    """
    cfg = MCMCConfig.make(config)
    device = resolve_device(device or getattr(likelihood, "device", "cpu"))
    if not callable(getattr(likelihood, "log_prob", None)):
        raise TypeError(f"likelihood must have a log_prob(input, condition) method; got {type(likelihood).__name__}")

    x = as_numpy(x_obs, np.float32)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or x.size == 0:
        raise ValueError(f"x_obs must be (B targets, n_x summaries) with at least one row; got shape {x.shape}")
    bad = np.flatnonzero(~np.isfinite(x).all(1))
    if len(bad):
        raise ValueError(f"x_obs contains NaN or inf in {len(bad)} of {len(x)} targets: {_some(bad)}")
    n_x = _width(likelihood, "n_x")
    if n_x is not None and x.shape[1] != n_x:
        raise ValueError(f"x_obs has {x.shape[1]} columns but the likelihood takes {n_x} summaries")
    B = len(x)

    priors = _as_priors(prior, B)
    uniq, which = _group(priors)
    sizes = sorted({int(p.ndim) for p in uniq})
    if len(sizes) > 1:
        raise ValueError(f"prior: all priors must have the same number of parameters; got {sizes}")
    P = sizes[0]
    named = [[str(n) for n in p.names] for p in uniq if getattr(p, "names", None) is not None]
    if any(n != named[0] for n in named):
        raise ValueError("prior: the per-target priors have different parameter names")
    names = named[0] if named else None
    if cfg.nwalkers <= 2 * P:
        raise ValueError(f"MCMCConfig.nwalkers = {cfg.nwalkers} must be larger than twice the number of parameters "
                         f"({P})")

    uniq = [p.to(device) for p in uniq]
    f32 = dict(dtype=torch.float32, device=device)
    low = torch.stack([torch.as_tensor(uniq[w].low, **f32) for w in which])[:, None]      # (B, 1, P)
    high = torch.stack([torch.as_tensor(uniq[w].high, **f32) for w in which])[:, None]
    bounded = torch.isfinite(high - low)[:, 0]                                             # (B, P)
    if cfg.logit_box and not bool(bounded.all()):
        cols = torch.nonzero(~bounded.all(0)).flatten().tolist()
        raise ValueError(f"the prior has infinite bounds (parameter columns {cols}), which the logit box cannot "
                         "map: sample with MCMCConfig(logit_box=False)")

    tr = _on_device(transform, device)
    n_theta = _width(likelihood, "n_theta")
    base_seed = cfg.seed if cfg.seed is not None else int(torch.randint(2 ** 31 - 1, (1,)))
    per = min(B, cfg.targets_per_batch or B)               # targets per ensemble
    if tr is not None:
        _check_transform(tr, uniq, per, n_theta, device, base_seed)
    elif n_theta is not None and P != n_theta:
        raise ValueError(f"prior has {P} parameters but the likelihood is conditioned on {n_theta}; to sample other "
                         "parameters pass transform=")

    mode = cfg.init
    if mode == "auto":
        mode = "walkers" if init_walkers is not None else "ball" if init_centre is not None else "prior"
    given = (("init_walkers", init_walkers, "walkers"), ("init_centre", init_centre, "ball"),
             ("init_sd", init_sd, "ball"))
    unused = [name for name, value, used_by in given if value is not None and mode != used_by]
    if unused:
        where = {"walkers": "at init_walkers", "ball": "in a ball around init_centre", "prior": "from prior draws"}
        warnings.warn(f"{' and '.join(unused)} ignored: with MCMCConfig.init = {cfg.init!r} the walkers start "
                      f"{where[mode]}" + ("" if cfg.init == "auto" else "; use init='auto'"), stacklevel=2)
    centre = sd = walkers = None
    if mode == "walkers":
        walkers = as_numpy(init_walkers, np.float32)
        shape = walkers.shape
        if walkers.ndim == 2:
            walkers = np.broadcast_to(walkers, (B,) + walkers.shape).copy()
        if walkers.shape != (B, cfg.nwalkers, P):
            raise ValueError(f"init_walkers must have shape ({cfg.nwalkers}, {P}) (MCMCConfig.nwalkers x parameters, "
                             f"used for every target) or ({B}, {cfg.nwalkers}, {P}); got {shape}")
        if not np.isfinite(walkers).all():
            raise ValueError("init_walkers contains NaN or inf")
    elif mode == "ball":
        if init_centre is None:
            raise ValueError("init='ball' needs init_centre (e.g. truths or the fiducial point); init='prior' or "
                             "init='auto' start from prior draws")
        centre = _per_target(init_centre, "init_centre", B, P)
        lo, hi = low[:, 0].cpu().numpy().astype(np.float64), high[:, 0].cpu().numpy().astype(np.float64)
        tol = 1e-6 * np.where(np.isfinite(hi - lo), hi - lo, 0.0)
        outside = ((centre < lo - tol) | (centre > hi + tol)).any(1)
        if outside.any():
            warnings.warn(f"init_centre lies outside the prior box for {int(outside.sum())} of {B} targets "
                          f"({_some(np.flatnonzero(outside))}); those centres were moved to the edge of the box",
                          stacklevel=2)
        if init_sd is not None:
            sd = _per_target(init_sd, "init_sd", B, P, scalar=True)
            if not (sd > 0).all():
                raise ValueError("init_sd must be positive")
        elif not bool(bounded.all()):
            cols = torch.nonzero(~bounded.all(0)).flatten().tolist()
            raise ValueError(f"init_sd is needed: the prior box is infinite in parameter columns {cols}, so the "
                             "ball has no default width")
    elif verbose and init_centre is None and cfg.init == "auto":
        logger.info(f"sample_posteriors: no init_centre given, the walkers of each target start from the best "
                    f"{cfg.nwalkers} of {_PRIOR_ROUNDS * cfg.nwalkers} prior draws")

    x = torch.as_tensor(x, **f32)
    results = []
    for i, start in enumerate(range(0, B, per)):
        sl = slice(start, min(B, start + per))
        seed = int(np.random.SeedSequence([base_seed, i]).generate_state(1)[0])
        results.append(_sample_batch(
            likelihood, x[sl], [uniq[w] for w in which[sl]], low[sl], high[sl], cfg, mode,
            None if centre is None else centre[sl], None if sd is None else sd[sl],
            None if walkers is None else walkers[sl], device, seed, verbose, tr, keep_chain, callback, i))

    first = results[0]["diagnostics"]
    keys = [k for k in first if isinstance(first[k], np.ndarray)]
    diagnostics = {k: np.concatenate([r["diagnostics"][k] for r in results]) for k in keys}
    diagnostics.update({k: [r["diagnostics"][k] for r in results] for k in first if k not in keys})

    def join(key, axis):
        if results[0][key] is None:
            return None
        parts, cut = _tails([r[key] for r in results], axis)
        if cut:
            warnings.warn(f"the targets_per_batch ensembles stopped after different numbers of steps; "
                          f"Posteriors.{key} holds the last {parts[0].shape[axis]} entries of each", RuntimeWarning,
                          stacklevel=3)
        return np.concatenate(parts, axis=1 - axis)      # the targets are on the other of the first two axes

    samples, derived = join("samples", 1), join("derived", 1)
    chain, log_prob = join("chain", 0), join("log_prob", 0)

    pool = min(r["pool"] for r in results)
    if cfg.n_samples is not None and cfg.n_samples > pool:
        warnings.warn(f"n_samples = {cfg.n_samples} is more than the {pool} points in the second half of the chain: "
                      "the samples were drawn with replacement and hold duplicates. Lower n_samples (None returns "
                      "the whole second half), or raise max_steps / nwalkers or lower thin", RuntimeWarning,
                      stacklevel=2)
    conv = diagnostics["converged"]
    if not conv.all():
        rhat, ess = diagnostics["rhat_max"][~conv], diagnostics["ess_min"][~conv]
        worst = f"{np.nanmax(rhat):.3f}" if np.isfinite(rhat).any() else "nan"
        least = f"{np.nanmin(ess):.0f}" if np.isfinite(ess).any() else "nan"
        warnings.warn(f"sample_posteriors: {int((~conv).sum())} of {B} targets did not converge after "
                      f"{max(diagnostics['steps'])} steps (worst R-hat {worst}, smallest ESS {least}; thresholds "
                      f"rhat < {cfg.rhat}, min_ess = {cfg.min_ess:g}). They are flagged in Posteriors.converged; "
                      "raise max_steps or check the diagnostics", RuntimeWarning, stacklevel=2)
    return Posteriors(samples, diagnostics, derived, names, chain, log_prob)


@torch.no_grad()
def _sample_batch(likelihood, x_obs, priors, low, high, cfg, mode, centre, sd, walkers, device, seed, verbose, tr,
                  keep_chain, callback, index):
    """One ensemble: B targets x ``cfg.nwalkers`` walkers. ``priors`` are on ``device``, ``low`` /
    ``high`` their (B, 1, P) bounds; returns numpy results and the diagnostics."""
    t0 = time.perf_counter()
    gen = torch.Generator(device=device).manual_seed(int(seed))
    B, W, ndim = len(x_obs), cfg.nwalkers, low.shape[-1]
    uniq, which = _group(priors)
    which = torch.as_tensor(which, device=device)
    members = [which == i for i in range(len(uniq))]                        # (B,) masks: the targets of each prior

    if len(uniq) == 1:
        prior_lp = uniq[0].log_prob                                         # (B, n, P) -> (B, n)
    elif all(type(p) is Prior and not p.terms and not p.constraints for p in uniq):
        log_density = -torch.log(high - low).sum(-1)                        # (B, 1): plain boxes, no loop

        def prior_lp(theta):
            inside = ((theta >= low) & (theta <= high)).all(-1)
            return torch.where(inside, log_density.to(theta.dtype), -torch.inf)
    else:
        def prior_lp(theta):                                                # one call per distinct prior object
            out = torch.empty(theta.shape[:2], device=theta.device, dtype=theta.dtype)
            for p, m in zip(uniq, members):
                out[m] = p.log_prob(theta[m])
            return out

    like = flow_log_prob_fn(likelihood, x_obs)
    if tr is not None:
        like_theta = like

        def like(psi):
            return like_theta(tr(psi))

    def log_prob(theta):
        # -inf outside the support; a non-finite flow value (far outside the training range) is
        # treated the same way, since torchemcee rejects nan
        lp, ll = prior_lp(theta), like(theta)
        ok = torch.isfinite(lp) & torch.isfinite(ll)
        return torch.where(ok, ll + lp, -torch.inf)

    if cfg.logit_box:
        box = LogitBox(low, high)
        target_fn, to_start = box.wrap(log_prob), box.to_u
    else:
        box, target_fn, to_start = None, log_prob, (lambda theta: theta)

    # initial walkers: proposals inside each target's prior support, redrawn where the full
    # log-probability (prior, likelihood, transform) is not finite
    if mode == "walkers":
        init = torch.as_tensor(walkers, dtype=torch.float32, device=device)
        propose = None
    elif mode == "prior":
        def propose():
            out = torch.empty((B, W, ndim), device=device)
            for p, m in zip(uniq, members):
                n = int(m.sum())
                out[m] = torch.as_tensor(p.sample(n * W, generator=gen), dtype=out.dtype).reshape(n, W, ndim)
            if box is not None:
                # a walker within ~1e-3 box widths of a bound lies far out in the logit, where a step of the
                # ensemble's size hardly changes theta: it would need thousands of steps to reach a narrow posterior
                centre_of_box = 0.5 * (low + high)
                out = centre_of_box + (1 - 2 * _PRIOR_EDGE) * (out - centre_of_box)
            return out
    else:
        width = high - low
        pad = torch.where(torch.isfinite(width), 1e-4 * width, torch.zeros_like(width))
        mid = torch.as_tensor(centre, dtype=torch.float32, device=device)[:, None]
        mid = torch.minimum(torch.maximum(mid, low + pad), high - pad)
        scale = cfg.init_scale * (width / 12 ** 0.5 if sd is None
                                  else torch.as_tensor(sd, dtype=torch.float32, device=device)[:, None])
        shrink = torch.ones((B, 1, 1), device=device)

        def propose():
            # a centre in a narrow corner of the support (box edge and constraints at once) can leave too few
            # ball draws inside it: the ball of such a target shrinks (it only sets the start; burn-in is discarded)
            nonlocal shrink

            def draw():
                return mid + torch.randn((B, W, ndim), generator=gen, device=device) * scale * shrink

            out = draw()
            bad = ~torch.isfinite(prior_lp(out))
            for level in range(_BALL_SHRINKS):
                for _ in range(_BALL_ROUNDS):
                    if not bool(bad.any()):
                        return out
                    out = torch.where(bad[..., None], draw(), out)
                    bad = ~torch.isfinite(prior_lp(out))
                if level < _BALL_SHRINKS - 1:
                    shrink = torch.where(bad.any(1)[:, None, None], 0.3 * shrink, shrink)
            return out

    def check(init):
        """The sampler's start positions, their log-probability and which of them are unusable. The prior is
        tested in theta, since the logit map would move a point outside the box onto its edge."""
        start = to_start(init)
        value = target_fn(start)
        return start, value, ~(torch.isfinite(prior_lp(init)) & torch.isfinite(value))

    if propose is not None:
        init = propose()
    start, value, bad = check(init)
    for _ in range(_REDRAWS if propose is not None else 0):
        if not bool(bad.any()):
            break
        init = torch.where(bad[..., None], propose(), init)
        start, value, bad = check(init)
    if bool(bad.any()):
        outside = bad & ~torch.isfinite(prior_lp(init))
        hint = {"walkers": "give init_walkers inside the support",
                "prior": "check x_obs for those targets, or restrict the prior to where the likelihood is defined",
                "ball": "try init='prior', a smaller init_scale, or an init_centre inside the support"}[mode]
        raise ValueError(f"{int(bad.sum())} of {B * W} start positions have a non-finite log-probability "
                         f"({int(outside.sum())} outside the prior support, {int((bad & ~outside).sum())} with a "
                         f"non-finite likelihood{'' if tr is None else ' or transform'}), in targets "
                         f"{_some(index * (cfg.targets_per_batch or 0) + torch.nonzero(bad.any(1)).flatten())}: "
                         f"{hint}")
    if mode == "prior":
        # single prior draws far from a narrow posterior can stay behind when the ensemble contracts and then
        # widen the result unnoticed, so every target starts from its best W of _PRIOR_ROUNDS x W draws
        for _ in range(_PRIOR_ROUNDS - 1):
            more, more_value, more_bad = check(propose())
            value = torch.cat([value, torch.where(more_bad, -torch.inf, more_value)], 1)
            value, best = value.topk(W, dim=1)
            start = torch.cat([start, more], 1).gather(1, best[..., None].expand(-1, -1, ndim))
        order = torch.rand((B, W), generator=gen, device=device).argsort(1)   # no ranking along the walker axis
        start = start.gather(1, order[..., None].expand(-1, -1, ndim))
    dependent = ~torchemcee.walkers_independent(start)
    if bool(dependent.any()):
        raise ValueError(f"the start positions of targets "
                         f"{_some(index * (cfg.targets_per_batch or 0) + torch.nonzero(dependent).flatten())} do "
                         "not span the parameter space (a parameter has the same value for all walkers, or the "
                         "walkers are linearly dependent): check init_sd / init_walkers")

    sampler = torchemcee.EnsembleSampler(target_fn, ndim, W, B, moves=_moves(cfg), device=device, generator=gen)
    thin = cfg.thin
    n_block, n_max = cfg.block // thin, cfg.max_steps // thin
    kept, state = 0, None
    while True:
        n = min(n_block, n_max - kept) if cfg.adaptive else n_max
        # the first kept state of a block is `thin` steps after the last state of the previous block (or after
        # burn-in), and the generator is left at its n-th kept state: every step taken counts
        steps = sampler.sample(start if state is None else None, n, thin_by=thin, progress=False,
                               discard=(cfg.burn_in if state is None else 0) + thin - 1)
        for state in itertools.islice(steps, n):
            pass
        steps.close()
        kept += n
        chain = sampler.get_chain(numpy=False)
        chain = chain[len(chain) // 2:]
        with _quiet_torchemcee():
            diag = diagnose(chain, state.accepted, state.step, cfg)
        tau = (diag["tau"] * thin).cpu().numpy()
        out = {
            "converged": diag["converged"].cpu().numpy(),
            "tau_max": tau.max(1), "rhat_max": diag["rhat"].cpu().numpy().max(1),
            "ess_min": diag["ess"].cpu().numpy().min(1),
            "acceptance": diag["acceptance"].mean(1).cpu().numpy(),
            "rhat": diag["rhat"].cpu().numpy(), "ess": diag["ess"].cpu().numpy(), "tau": tau,
            "steps": int(state.step - cfg.burn_in), "burn_in": int(cfg.burn_in),
            "seconds": time.perf_counter() - t0, "config": cfg.to_dict(),
        }
        if verbose and cfg.progress:
            logger.info(f"      block: {out['steps']} steps, converged {out['converged'].mean():.3f}, "
                        f"ess_min {out['ess_min'].min():.0f}, rhat_max {out['rhat_max'].max():.4f}, "
                        f"{out['seconds']:.0f}s")
        stop = callback is not None and bool(callback(
            {k: out[k] for k in ("steps", "converged", "rhat_max", "ess_min", "tau_max", "seconds")}
            | {"batch": index}))
        if stop or not cfg.adaptive or out["converged"].all() or kept >= n_max:
            break

    flat = chain.permute(1, 0, 2, 3).reshape(B, -1, ndim)
    pool = flat.shape[1]
    if cfg.n_samples is not None:
        pick = (torch.randperm(pool, generator=gen, device=device)[:cfg.n_samples] if cfg.n_samples <= pool
                else torch.randint(pool, (cfg.n_samples,), generator=gen, device=device))
        flat = flat[:, pick]
    theta = box.to_theta(flat) if box is not None else flat
    result = {"samples": theta.cpu().numpy(), "diagnostics": out, "pool": pool, "derived": None, "chain": None,
              "log_prob": None}
    if tr is not None:
        cols = max(1, (1 << 20) // B)                      # chunks of samples; every call sees all B targets
        result["derived"] = torch.cat([tr(theta[:, i:i + cols]) for i in range(0, theta.shape[1], cols)],
                                      1).cpu().numpy()
    if keep_chain:
        lp = sampler.get_log_prob(numpy=False)[-len(chain):]
        result["chain"] = (box.to_theta(chain) if box is not None else chain).cpu().numpy()
        result["log_prob"] = (lp - box.log_jacobian(chain) if box is not None else lp).cpu().numpy()
    out["seconds"] = time.perf_counter() - t0
    if verbose:
        logger.info(f"    mcmc {B} x {W}: {out['steps']} steps after {cfg.burn_in} burn-in, "
                    f"converged {out['converged'].mean():.3f}, tau_max {tau.max():.0f}, "
                    f"rhat_max {out['rhat_max'].max():.4f}, ess_min {out['ess_min'].min():.0f}, "
                    f"{out['seconds']:.0f}s")
    return result
