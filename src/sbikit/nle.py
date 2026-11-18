"""Neural likelihood estimation.

A conditional normalising flow learns the likelihood p(s | theta) of summaries s given parameters
theta from simulations. The flow is chosen by name from the registry :data:`FLOWS` (the zuko flows
that sbi builds, plus anything added with :func:`register_flow`), or given as a builder function or
a ready estimator. Summaries and parameters are scaled inside :class:`NLE` (z-scores by default) and
the scalers are stored with the weights, so a saved model takes raw values and ``log_prob`` is the
density of the raw summaries. :class:`Ensemble` is a weighted mixture of the likelihoods of several
flows.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import torch
from sbi.neural_nets import net_builders as _sbi_builders

from ._utils import (ConfigBase, NotFittedError, Registry, as_2d, as_numpy, check_pair, did_you_mean, holdout_split,
                     logger, plain, resolve_device, seeded, to_tensor)
from .scaling import make_scaler, scaler_from_state

__all__ = ["FlowConfig", "TrainConfig", "NLE", "Ensemble", "FLOWS", "register_flow"]

_CHUNK = 8192                      # rows per forward pass in evaluation

FLOWS = Registry("flow")


def register_flow(name: str, builder=None, overwrite: bool = False):
    """Make a flow selectable by name (``FlowConfig.model``) and restorable from checkpoints.

    Args:
        name: the name used in :class:`FlowConfig`.
        builder: ``builder(batch_x, batch_theta, config) -> estimator``. ``batch_x`` (64, n_x) and
            ``batch_theta`` (64, n_theta) are float32 tensors that only carry the dimensions; the
            data the estimator sees are scaled by the :class:`NLE` already, so a builder must not add
            its own z-scoring. ``config`` is the :class:`FlowConfig`, whose ``kwargs`` dict is meant
            for the builder's own options. The estimator is a ``torch.nn.Module`` with
            ``loss(x, condition=...)`` (negative log density per row) and
            ``log_prob(x, condition=...)`` in sbi's layout (as sbi's ``ZukoFlow``); a
            ``sample(sample_shape, condition=...)`` method enables :meth:`NLE.sample`. Without
            ``builder`` this returns a decorator.
        overwrite: replace an existing flow of that name.

    Raises:
        ValueError: if the name is taken by another builder and ``overwrite`` is False.

    Example:
        >>> @register_flow("wide_maf")
        ... def wide_maf(batch_x, batch_theta, config):
        ...     return build_zuko_maf(batch_x, batch_theta, z_score_x="none", z_score_y="none",
        ...                           hidden_features=[256] * 3, **config.kwargs)
        >>> nle = NLE(flow={"model": "wide_maf", "kwargs": {"residual": True}})
    """
    return FLOWS.register(name, builder, overwrite=overwrite)


def _zuko(name: str, bins: bool = False):
    """The registry builder for sbi's ``build_zuko_<name>``, or None if this sbi does not have it."""
    build = getattr(_sbi_builders, f"build_zuko_{name}", None)
    if build is None:
        return None

    def builder(batch_x, batch_theta, config):
        kwargs = {"z_score_x": "none", "z_score_y": "none", "num_transforms": config.num_transforms,
                  "hidden_features": [config.hidden_features] * config.num_blocks}
        if bins:
            kwargs["num_bins"] = config.num_bins
        return build(batch_x, batch_theta, **{**kwargs, **config.kwargs})
    return builder


for _name, _bins in (("maf", False), ("nsf", True), ("naf", False), ("unaf", False), ("nice", False), ("gf", False),
                     ("bpf", False), ("ncsf", True), ("sospf", False)):
    if (_builder := _zuko(_name, _bins)) is not None:
        FLOWS[_name] = _builder


def _whole(v):
    """``v`` as an int if it is a whole number, else unchanged."""
    number = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
    return int(v) if number and float(v).is_integer() else v


def _real(v):
    """``v`` as a float if it is a number, else unchanged."""
    return float(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else v


@dataclass
class FlowConfig(ConfigBase):
    """Architecture of the flow.

    Attributes:
        model: a name in :data:`FLOWS`. Built in (zuko flows, built by sbi; all work for any
            ``n_x >= 1``):

            - ``"nsf"``: neural spline flow (autoregressive, rational-quadratic splines);
            - ``"maf"``: masked autoregressive flow (affine);
            - ``"naf"``, ``"unaf"``: neural autoregressive flows (monotonic networks; slow to sample);
            - ``"nice"``: affine coupling layers;
            - ``"gf"``: Gaussianisation flow (can return -inf far outside the training data);
            - ``"bpf"``: Bernstein-polynomial flow;
            - ``"sospf"``: sum-of-squares polynomial flow;
            - ``"ncsf"``: circular spline flow, for periodic summaries only: scaled values are
              wrapped to [-pi, pi), so use it with ``NLE(s_scaler="identity")`` and angles in radians.

            Add your own with :func:`register_flow`.
        hidden_features: width of the hidden layers of each transform's network.
        num_blocks: number of hidden layers.
        num_transforms: number of transforms.
        num_bins: spline bins (``"nsf"`` and ``"ncsf"`` only).
        kwargs: further keyword arguments for the builder. The built-in builders pass them to sbi's
            ``build_zuko_*`` and on to the zuko flow, e.g. ``{"residual": True}``,
            ``{"randperm": True}``, ``{"hidden_features": [128, 64]}`` (per-layer widths, replacing
            ``hidden_features`` / ``num_blocks``), ``{"components": 16}`` for ``"gf"``. Keep the
            values plain (numbers, strings, lists, dicts), or the model cannot be saved; anything
            else (activation classes, embedding networks) belongs in a registered builder.

    Raises:
        TypeError: from :meth:`make` for a key that is not a field.
        ValueError: for an unknown ``model`` or a size that is not a positive integer.
    """

    model: str = "nsf"
    hidden_features: int = 64
    num_blocks: int = 2
    num_transforms: int = 5
    num_bins: int = 8
    kwargs: dict = field(default_factory=dict)

    def _normalise(self):
        for name in ("hidden_features", "num_blocks", "num_transforms", "num_bins"):
            setattr(self, name, _whole(getattr(self, name)))
        if isinstance(self.kwargs, Mapping):
            self.kwargs = dict(self.kwargs)

    def validate(self):
        if not isinstance(self.model, str) or self.model not in FLOWS:
            raise ValueError(f"FlowConfig.model = {self.model!r}{did_you_mean(self.model, list(FLOWS))}; "
                             f"valid: {sorted(FLOWS)} (add your own with register_flow)")
        for name in ("hidden_features", "num_blocks", "num_transforms", "num_bins"):
            v = getattr(self, name)
            self._require(type(v) is int and v >= 1, f"{name} must be a positive integer, got {v!r}")
        self._require(isinstance(self.kwargs, dict) and all(isinstance(k, str) for k in self.kwargs),
                      f"kwargs must be a dict of keyword arguments for the flow builder, got {self.kwargs!r}")


@dataclass
class TrainConfig(ConfigBase):
    """Training settings of :meth:`NLE.fit`.

    Attributes:
        lr: learning rate.
        batch_size: minibatch size.
        max_epochs: upper limit of epochs.
        stop_after_epochs: early-stopping patience, in epochs without improvement.
        min_delta: improvement of the validation loss that resets the patience (and the patience of
            the plateau scheduler).
        sched_factor, sched_patience: with ``scheduler="plateau"`` the learning rate is multiplied by
            ``sched_factor`` after ``sched_patience`` epochs without improvement.
        grad_clip: maximum gradient norm (None or 0: no clipping).
        weight_decay: weight decay of the optimizer.
        optimizer: ``"adam"`` or ``"adamw"``.
        scheduler: ``"plateau"`` (``ReduceLROnPlateau``) or ``"none"`` (constant learning rate).
        val_fraction: fraction of the simulations held out for validation when :meth:`NLE.fit` is
            given no validation set.

    Raises:
        TypeError: from :meth:`make` for a key that is not a field.
        ValueError: for a value outside its range.
    """

    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 1000
    stop_after_epochs: int = 20
    min_delta: float = 1e-5
    sched_factor: float = 0.1 ** 0.5
    sched_patience: int = 5
    grad_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    optimizer: str = "adam"
    scheduler: str = "plateau"
    val_fraction: float = 0.1

    _choices = {"optimizer": ("adam", "adamw"), "scheduler": ("plateau", "none")}

    def _normalise(self):
        for name in ("batch_size", "max_epochs", "stop_after_epochs", "sched_patience"):
            setattr(self, name, _whole(getattr(self, name)))
        for name in ("lr", "min_delta", "sched_factor", "grad_clip", "weight_decay", "val_fraction"):
            setattr(self, name, _real(getattr(self, name)))

    def validate(self):
        def check(name, ok, what):
            v = getattr(self, name)
            self._require(isinstance(v, float) and np.isfinite(v) and ok(v), f"{name} must be {what}, got {v!r}")

        for name in ("batch_size", "max_epochs", "stop_after_epochs"):
            v = getattr(self, name)
            self._require(type(v) is int and v >= 1, f"{name} must be a positive integer, got {v!r}")
        v = self.sched_patience
        self._require(type(v) is int and v >= 0, f"sched_patience must be a non-negative integer, got {v!r}")
        check("lr", lambda v: v > 0, "positive")
        check("min_delta", lambda v: v >= 0, "non-negative")
        check("sched_factor", lambda v: 0 < v < 1, "in (0, 1)")
        check("weight_decay", lambda v: v >= 0, "non-negative")
        check("val_fraction", lambda v: 0 < v < 1, "in (0, 1)")
        if self.grad_clip is not None:
            check("grad_clip", lambda v: v >= 0, "None or non-negative")


def _dim(n, name: str) -> Optional[int]:
    if n is None:
        return None
    m = _whole(n)
    if type(m) is not int or m < 1:
        raise ValueError(f"{name} must be a positive integer or None, got {n!r}")
    return m


def _own_scaler(spec, name: str):
    """A scaler of the NLE's own from a name, None (no scaling) or an instance (copied)."""
    if isinstance(spec, type):
        raise TypeError(f"{name} must be a scaler name or a scaler instance, got the class {spec.__name__}; "
                        f"pass an instance, e.g. {spec.__name__}()")
    scaler = make_scaler(spec or "none") if spec is None or isinstance(spec, str) else copy.deepcopy(spec)
    if not (callable(scaler) and callable(getattr(scaler, "fit", None))):
        raise TypeError(f"{name} must be a scaler name or a scaler (with fit and __call__), got {type(spec).__name__}")
    return scaler


def _check_estimator(est):
    if not (isinstance(est, torch.nn.Module) and callable(getattr(est, "loss", None))
            and callable(getattr(est, "log_prob", None))):
        raise TypeError("a flow must be a torch.nn.Module with loss(x, condition=...) and log_prob(x, condition=...) "
                        f"(like sbi's ZukoFlow), got {type(est).__name__}")
    return est


def _check_hook(fn, name: str, what: str, choices=None):
    """Raise ``TypeError`` unless the ``fit`` argument ``name`` is None or callable."""
    if fn is None or callable(fn):
        return
    hint = f"; to choose a built-in one use train={{{name!r}: ...}} with one of {list(choices)}" \
        if choices and isinstance(fn, str) else ""
    raise TypeError(f"{name} must be None or a function {what}, got {fn!r}{hint}")


def _scaled(scaler, x):
    """``scaler(x)`` as a float32 tensor, computed in float64 where the device has it."""
    if x.dtype != torch.float64 and x.device.type != "mps":
        x = x.double()
    return scaler(x).float()


def _log_det(scaler, x):
    """log |det d scaled / d x| per row of ``x``; 0.0 for a scaler without ``log_det``."""
    fn = getattr(scaler, "log_det", None)
    return 0.0 if fn is None else fn(x)


def _tensor(a, device):
    return to_tensor(as_numpy(a, np.float32), device=device)


def _weights(module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _mean_loss(flow, x, y) -> float:
    with torch.no_grad():
        return torch.cat([flow.loss(x[i:i + _CHUNK], condition=y[i:i + _CHUNK])
                          for i in range(0, len(x), _CHUNK)]).mean().item()


def _is_plain(v) -> bool:
    if isinstance(v, dict):
        return all(isinstance(k, (str, int)) and _is_plain(x) for k, x in v.items())
    if isinstance(v, list):
        return all(_is_plain(x) for x in v)
    return v is None or isinstance(v, (bool, int, float, str, torch.Tensor))


class _Likelihood:
    """``log_likelihood`` and ``val_loss`` for anything with ``log_prob`` (sbi layout), ``device``,
    ``n_x``, ``n_theta`` and ``fitted``."""

    _eval_dtype = torch.float32        # dtype of the raw values handed to log_prob by log_likelihood

    def _check_fitted(self):
        if not self.fitted:
            raise NotFittedError(f"{type(self).__name__}: call fit() first")

    def log_likelihood(self, s, theta):
        """log p(s_i | theta_i) for matching rows, without gradients.

        Args:
            s: (M, n_x) summaries (array-like or tensor), or a single summary vector ((n_x,) or
                (1, n_x)), which is evaluated at every row of ``theta``.
            theta: (M, n_theta) parameters, or a single (n_theta,) vector together with a single
                summary vector.

        Returns:
            (M,) float32 tensor on the flow's device; (1,) for a single pair of vectors.

        Raises:
            NotFittedError: before ``fit``.
            ValueError: for other shapes, wrong column counts or different numbers of rows.
        """
        self._check_fitted()
        s, theta = as_numpy(s), as_numpy(theta)
        if s.ndim == 1:
            s, theta = s[None], (theta[None] if theta.ndim == 1 else theta)
        if s.ndim != 2 or theta.ndim != 2:
            raise ValueError("s must be (M, n_x) or a single (n_x,) vector and theta (M, n_theta) (or (n_theta,) with "
                             f"a single s), got shapes {s.shape} and {theta.shape}")
        for a, name, n in ((s, "s", self.n_x), (theta, "theta", self.n_theta)):
            if n is not None and a.shape[1] != n:
                raise ValueError(f"{name} has {a.shape[1]} columns, expected {n}")
        if len(theta) == 0:
            raise ValueError(f"theta is empty (shape {theta.shape})")
        if len(s) == 1:
            s = np.repeat(s, len(theta), 0)
        if len(s) != len(theta):
            raise ValueError(f"s has {len(s)} rows but theta has {len(theta)}; give matching rows or a single s")
        device, dtype = torch.device(self.device), self._eval_dtype
        with torch.no_grad():
            out = [self.log_prob(to_tensor(s[i:i + _CHUNK], dtype=dtype, device=device).unsqueeze(0),
                                 to_tensor(theta[i:i + _CHUNK], dtype=dtype, device=device)).reshape(-1)
                   for i in range(0, len(theta), _CHUNK)]
        return torch.cat(out).float()

    def val_loss(self, s, theta) -> float:
        """Mean -log p(s_i | theta_i) over the rows: the loss that :meth:`NLE.fit` reports."""
        return -self.log_likelihood(s, theta).mean().item()


class NLE(_Likelihood):
    """Conditional density p(s | theta) of summaries given parameters.

    The quickest use is ``NLE().fit(s, theta)``: the dimensions come from the data, a tenth of the
    simulations is held out for validation and the defaults of :class:`FlowConfig` and
    :class:`TrainConfig` apply.

    Args:
        n_x: dimension of the summaries (None: taken from the data at the first :meth:`fit`).
        n_theta: number of parameters the flow is conditioned on (None: taken from the data).
        flow: what to train. None (defaults), a :class:`FlowConfig`, a dict of its fields or a model
            name select a registered flow. For full control pass a builder
            ``builder(batch_x, batch_theta) -> estimator`` or a ready estimator (see
            :func:`register_flow` for what an estimator is); such a model can be trained and used,
            but not saved (register the builder for that). An estimator instance is trained in
            place and never re-initialised.
        device: torch device of the flow (``"auto"``: GPU if there is one).
        seed: default seed of the weight initialisation and of training (hold-out split and
            batch order); ``fit(seed=...)`` overrides it. None draws all of it from torch's global
            random state, so ``torch.manual_seed`` before the fit makes it repeatable.
        s_scaler: scaler of the summaries: a name (``"zscore"``, ``"robust_asinh"``,
            ``"identity"``, or one added with :func:`sbikit.scaling.register_scaler`), None (no
            scaling) or a scaler instance, which is copied. It is fitted on the training set in
            :meth:`fit`.
        t_scaler: the same for the parameters.

    Attributes:
        flow: the estimator; None until the dimensions are known. Assigning to it replaces the
            estimator by an instance (trained in place, not saveable).
        flow_config: the :class:`FlowConfig`; None for a builder function or an instance.
        train_config: the :class:`TrainConfig` of the last :meth:`fit`; None before.
        s_scaler, t_scaler: the scalers (fitted by :meth:`fit`).
        history, train_history: validation and training loss per epoch of the last :meth:`fit`.
        best_val_loss, epochs: its best validation loss and number of epochs; None before (and for
            checkpoints that did not store them).
        n_x, n_theta, device, seed: as given or inferred.
        fitted: whether the model can be evaluated.

    Example:
        >>> nle = NLE(flow={"model": "maf", "hidden_features": 64}, seed=0)
        >>> val_loss, epochs = nle.fit(s_train, theta_train, s_val, theta_val, train={"lr": 3e-4})
        >>> nle.log_likelihood(s_test, theta_test)        # (n_test,) tensor
    """

    def __init__(self, n_x: Optional[int] = None, n_theta: Optional[int] = None, flow=None, device="cpu",
                 seed: Optional[int] = None, *, s_scaler="zscore", t_scaler="zscore"):
        self.n_x, self.n_theta = _dim(n_x, "n_x"), _dim(n_theta, "n_theta")
        self.device = resolve_device(device)
        self.seed = None if seed is None else int(seed)
        self.s_scaler, self.t_scaler = _own_scaler(s_scaler, "s_scaler"), _own_scaler(t_scaler, "t_scaler")
        self.flow_config, self._builder, self._flow = None, None, None
        if isinstance(flow, torch.nn.Module):
            self._source, self._flow = "instance", _check_estimator(flow).to(self.device)
        elif callable(flow):
            self._source, self._builder = "builder", flow
        else:
            self._source = "config"
            self.flow_config = FlowConfig(model=flow) if isinstance(flow, str) else FlowConfig.make(flow)
        self.train_config: Optional[TrainConfig] = None
        self.history: list = []
        self.train_history: list = []
        self.best_val_loss: Optional[float] = None
        self.epochs: Optional[int] = None
        self._fitted = False
        if self._source != "instance" and self.n_x is not None and self.n_theta is not None:
            with seeded(self.seed):
                self._flow = self._build(self.n_x, self.n_theta)

    def _build(self, n_x: int, n_theta: int):
        """A freshly initialised estimator on the NLE's device."""
        batch_x, batch_theta = torch.randn(64, n_x), torch.randn(64, n_theta)
        if self._source == "builder":
            est = self._builder(batch_x, batch_theta)
        else:
            est = FLOWS[self.flow_config.model](batch_x, batch_theta, self.flow_config)
        return _check_estimator(est).to(self.device)

    @property
    def flow(self):
        """The estimator (a ``torch.nn.Module``); None until the dimensions are known."""
        return self._flow

    @flow.setter
    def flow(self, estimator):
        self._flow = _check_estimator(estimator).to(self.device)
        self._source, self._builder, self.flow_config = "instance", None, None

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has succeeded (or the model was loaded from a checkpoint)."""
        return self._fitted

    @property
    def _eval_dtype(self):
        # raw values stay float64 until they are scaled, where the device has float64
        return torch.float32 if torch.device(self.device).type == "mps" else torch.float64

    def __repr__(self):
        if self.flow_config is not None:
            model = self.flow_config.model
        else:
            source = self._builder if self._source == "builder" else self._flow
            model = getattr(source, "__name__", None) or type(source).__name__
        return (f"{type(self).__name__}(model={model!r}, n_x={self.n_x}, n_theta={self.n_theta}, "
                f"device={self.device!r}, {'fitted' if self.fitted else 'not fitted'})")

    # ------------------------------------------------------------------ training
    def fit(self, s, theta, s_val=None, theta_val=None, train=None, seed: Optional[int] = None, *,
            warm_start: bool = False, optimizer=None, scheduler=None, callback=None, verbose: bool = False):
        """Train with early stopping on the validation loss; the weights of the best epoch are kept.

        Every call starts from freshly initialised weights and refits the scalers, so repeated
        calls do not build on each other and the same seed and data give the same model. The two
        exceptions: ``warm_start=True``, and an estimator instance given as ``flow``, which is
        trained in place from its current weights.

        Args:
            s, theta: (N, n_x) summaries and (N, n_theta) parameters of the training set
                (array-likes or tensors).
            s_val, theta_val: the same for the validation set. Without one,
                ``TrainConfig.val_fraction`` of the given simulations is held out at random.
            train: :class:`TrainConfig` or dict of its fields.
            seed: seed of the weight initialisation, the batch order and the hold-out split
                (default: the ``seed`` of the constructor). If both are None, all three are drawn
                from torch's global random state (repeatable after ``torch.manual_seed``);
                otherwise that state is left untouched.
            warm_start: continue from the current weights with the current scalers; the result is
                never worse on the validation set than the weights it started from.
            optimizer: None (``TrainConfig.optimizer``) or a function
                ``params -> torch.optim.Optimizer`` (``TrainConfig.lr`` and ``weight_decay`` are then
                not used).
            scheduler: None (``TrainConfig.scheduler``) or a function ``optimizer -> scheduler``;
                the scheduler is stepped once per epoch (with the validation loss if it is a
                ``ReduceLROnPlateau``).
            callback: called after every epoch with a dict ``{"epoch", "train_loss", "val_loss",
                "best", "lr", "model"}`` (``lr`` is the epoch's learning rate, ``model`` this object,
                usable for evaluation). If it returns something true, training stops and the best
                weights so far are kept.
            verbose: log one line per epoch.

        Returns:
            (best validation loss, number of epochs run). The loss is minus the mean log density
            of the raw summaries, the same units as ``history`` and :meth:`val_loss`. It is inf, with
            a ``RuntimeWarning``, if no epoch gave a finite validation loss; the model is then left
            unfitted (or, for a warm start, as it was).

        Raises:
            ValueError: for inputs that are not 2D, are empty, do not match in length or columns,
                or contain NaN or inf.
            TypeError: if ``optimizer``, ``scheduler`` or ``callback`` is neither None nor callable
                (the built-in optimizers and schedulers are chosen by name in ``train``), or if
                ``optimizer`` does not return an optimizer.
            NotFittedError: for ``warm_start=True`` on a model that is not fitted.

        If training is interrupted (``KeyboardInterrupt`` or an error in a callback), the best
        weights so far are restored before the exception is passed on.
        """
        tr = TrainConfig.make(train)
        _check_hook(optimizer, "optimizer", "params -> torch.optim.Optimizer", TrainConfig._choices["optimizer"])
        _check_hook(scheduler, "scheduler", "optimizer -> scheduler", TrainConfig._choices["scheduler"])
        _check_hook(callback, "callback", "info dict -> stop or not")
        seed = self.seed if seed is None else int(seed)
        name, dev = type(self).__name__, self.device
        if warm_start and not self.fitted:
            raise NotFittedError(f"{name}: warm_start=True continues a fitted model; call fit() first")
        s, theta = check_pair(s, theta, "s", "theta", self.n_x, self.n_theta)
        if (s_val is None) != (theta_val is None):
            raise ValueError("give both s_val and theta_val, or neither (then some of the simulations are held out)")
        if s_val is None:
            # an unseeded split follows torch's global random state, like the initialisation and the batch order
            split_seed = seed if seed is not None else int(torch.randint(2 ** 31 - 1, ()).item())
            trn, held = holdout_split(len(s), tr.val_fraction, split_seed)
            logger.info(f"{name}: no validation set given, holding out {len(held)} of {len(s)} simulations")
            s, theta, s_val, theta_val = s[trn], theta[trn], s[held], theta[held]
        else:
            s_val, theta_val = check_pair(s_val, theta_val, "s_val", "theta_val", s.shape[1], theta.shape[1])
        n_x, n_theta = s.shape[1], theta.shape[1]
        with seeded(seed):
            if warm_start:
                flow, s_scaler, t_scaler = self._flow, self.s_scaler, self.t_scaler
            else:
                constant = np.flatnonzero(np.ptp(s, axis=0) == 0)
                if len(constant):
                    warnings.warn(f"{name}.fit: summary columns {constant.tolist()} are constant in the training set; "
                                  "the density has no meaning in them", RuntimeWarning, stacklevel=2)
                s_scaler, t_scaler = copy.deepcopy(self.s_scaler).fit(s), copy.deepcopy(self.t_scaler).fit(theta)
                flow = self._flow if self._source == "instance" else self._build(n_x, n_theta)
            # scaled in float64, then cast: keeps columns with a large mean / std ratio resolved
            x, y = _tensor(s_scaler(s), dev), _tensor(t_scaler(theta), dev)
            xv, yv = _tensor(s_scaler(s_val), dev), _tensor(t_scaler(theta_val), dev)
            # mean log-Jacobian of the summary scaling: turns the flow's loss into that of the raw summaries
            jac, jac_val = (float(np.mean(as_numpy(_log_det(s_scaler, a)))) for a in (s, s_val))
            if optimizer is not None:
                opt = optimizer(flow.parameters())
                if not (callable(getattr(opt, "step", None)) and callable(getattr(opt, "zero_grad", None))):
                    raise TypeError("optimizer must return a torch.optim.Optimizer for the given parameters, got "
                                    f"{type(opt).__name__}")
            else:
                opt = (torch.optim.AdamW if tr.optimizer == "adamw" else torch.optim.Adam)(
                    flow.parameters(), lr=tr.lr, weight_decay=tr.weight_decay)
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
            if scheduler is not None:
                sched = scheduler(opt)
            elif tr.scheduler == "plateau":
                sched = plateau(opt, factor=tr.sched_factor, patience=tr.sched_patience, threshold=tr.min_delta,
                                threshold_mode="abs")
            else:
                sched = None
            # weights to return to if no epoch improves on them: only where training continues existing weights
            start = _weights(flow) if warm_start or self._source == "instance" else None
            previous = (self.s_scaler, self.t_scaler, self._fitted)
            best, best_state, wait, epoch, reason = np.inf, None, 0, 0, "max_epochs reached"
            if warm_start:
                flow.eval()
                val = _mean_loss(flow, xv, yv) - jac_val
                if np.isfinite(val):
                    best, best_state = val, start
            self._flow, self.s_scaler, self.t_scaler, self.n_x, self.n_theta = flow, s_scaler, t_scaler, n_x, n_theta
            self.train_config, self.history, self.train_history = tr, [], []
            self._fitted = True                                  # a callback can evaluate the model
            bs = min(tr.batch_size, len(x))
            try:
                for epoch in range(1, tr.max_epochs + 1):
                    flow.train()
                    perm = torch.randperm(len(x), device=dev)
                    total = torch.zeros((), device=dev)
                    for i in range(0, len(x), bs):
                        idx = perm[i:i + bs]
                        loss = flow.loss(x[idx], condition=y[idx]).mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        if tr.grad_clip:
                            torch.nn.utils.clip_grad_norm_(flow.parameters(), tr.grad_clip)
                        opt.step()
                        total += loss.detach() * len(idx)
                    flow.eval()
                    train_loss, val = total.item() / len(x) - jac, _mean_loss(flow, xv, yv) - jac_val
                    self.train_history.append(train_loss)
                    self.history.append(val)
                    lr = opt.param_groups[0]["lr"]
                    if isinstance(sched, plateau):
                        sched.step(val)
                    elif sched is not None:
                        sched.step()
                    if np.isfinite(val) and val < best - tr.min_delta:
                        best, best_state, wait = val, _weights(flow), 0
                    else:
                        wait += 1
                    if verbose:
                        logger.info(f"{name} epoch {epoch:4d}  train {train_loss:.4f}  val {val:.4f}  "
                                    f"best {best:.4f}  lr {lr:.2e}")
                    if callback is not None and callback({"epoch": epoch, "train_loss": train_loss, "val_loss": val,
                                                          "best": float(best), "lr": lr, "model": self}):
                        reason = "stopped by the callback"
                        break
                    if not np.isfinite(train_loss):
                        reason = "the training loss became non-finite"
                        break
                    if wait >= tr.stop_after_epochs:
                        reason = "early stopping"
                        break
            finally:                                             # also when interrupted: keep the best weights
                if best_state is not None:
                    flow.load_state_dict(best_state)
                elif start is not None:
                    flow.load_state_dict(start)
                    self.s_scaler, self.t_scaler, self._fitted = previous
                else:
                    self._fitted = False
                flow.eval()
                self.best_val_loss, self.epochs = float(best), epoch
        logger.debug(f"{name}.fit: {epoch} epochs ({reason}), best validation loss {best:.4f}")
        if best_state is None:
            warnings.warn(f"{name}.fit: no epoch reached a finite validation loss ({reason} after {epoch} epochs); the "
                          f"model is left {'as it was before' if start is not None else 'unfitted'}. Check the data "
                          "for outliers and try a lower learning rate", RuntimeWarning, stacklevel=2)
        elif reason.startswith("the training loss"):
            warnings.warn(f"{name}.fit: {reason} in epoch {epoch}; the weights of the best epoch before are kept",
                          RuntimeWarning, stacklevel=2)
        return float(best), epoch

    # ------------------------------------------------------------------ evaluation
    def log_prob(self, input, condition):
        """log p(input | condition) of raw summaries, in sbi's layout.

        The density of the scaled summaries from the flow plus the log-Jacobian of the summary
        scaler (``s_scaler.log_det``; for a z-score the constant ``-sum(log std)``, and 0 for a
        scaler without ``log_det``). Gradients are tracked; use :meth:`log_likelihood` for plain
        evaluation.

        Args:
            input: (1, M, n_x) or (M, n_x) summaries in raw units, on the flow's device. More
                generally (S, M, n_x): S sets of summaries, each evaluated at the M parameter rows.
            condition: (M, n_theta) parameters in raw units. A single row is used for every
                summary, and a single summary (M = 1 in ``input``) for every row.

        Returns:
            (1, M) tensor ((S, M) for an ``input`` with S sets).

        Raises:
            NotFittedError: before :meth:`fit`.
            ValueError: if ``input`` is not 2D or 3D or ``condition`` not 2D, if the last dimensions
                are not ``n_x`` and ``n_theta``, or if the numbers of rows differ and neither is 1.
        """
        self._check_fitted()
        input, condition = to_tensor(input, device=self.device), to_tensor(condition, device=self.device)
        if (input.ndim not in (2, 3) or condition.ndim != 2 or input.shape[-1] != self.n_x
                or condition.shape[-1] != self.n_theta):
            raise ValueError(f"expected input (1, M, {self.n_x}) or (M, {self.n_x}) and condition (M, {self.n_theta}), "
                             f"got shapes {tuple(input.shape)} and {tuple(condition.shape)}")
        rows = (input.shape[-2], len(condition))
        if rows[0] != rows[1] and 1 not in rows:
            raise ValueError(f"input has {rows[0]} rows but condition has {rows[1]}; they must match, or one of them "
                             "be a single row")
        lp = self._flow.log_prob(_scaled(self.s_scaler, input), condition=_scaled(self.t_scaler, condition))
        jac = _log_det(self.s_scaler, input)
        return lp + (jac.to(lp.dtype) if isinstance(jac, torch.Tensor) else jac)

    def sample(self, theta, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Summaries drawn from the learned likelihood, e.g. for posterior predictive checks.

        Args:
            theta: (M, n_theta) parameters in raw units.
            n: number of draws per row of ``theta``.
            seed: torch seed of the draws (None: torch's global random state).

        Returns:
            (n, M, n_x) float64 array of summaries in raw units.

        Raises:
            NotFittedError: before :meth:`fit`.
            NotImplementedError: if the estimator has no ``sample`` method.
            ValueError: if ``theta`` is not (M, n_theta).
        """
        self._check_fitted()
        if not callable(getattr(self._flow, "sample", None)):
            raise NotImplementedError(f"{type(self._flow).__name__} has no sample(sample_shape, condition=...) method")
        y = _tensor(self.t_scaler(as_2d(theta, "theta", self.n_theta)), self.device)
        with seeded(seed), torch.no_grad():
            z = torch.cat([self._flow.sample(torch.Size((int(n),)), condition=y[i:i + _CHUNK])
                           for i in range(0, len(y), _CHUNK)], 1)
        return as_numpy(self.s_scaler.inverse(z.cpu().numpy().astype(np.float64)))

    def to(self, device):
        """Move the flow to ``device`` (``"auto"``: GPU if there is one); returns itself."""
        self.device = resolve_device(device)
        if self._flow is not None:
            self._flow.to(self.device)
        return self

    # ------------------------------------------------------------------ io
    def state(self) -> dict:
        """Plain serialisable state (configs, weights, scalers, training record) for
        :func:`sbikit.io.save`.

        Raises:
            NotFittedError: before :meth:`fit`.
            TypeError: if the flow was given as a builder function or an instance (it could not be
                rebuilt on loading), or if ``FlowConfig.kwargs`` holds values a checkpoint cannot
                store.
        """
        self._check_fitted()
        if self.flow_config is None:
            raise TypeError(f"{type(self).__name__}: a flow given as a builder function or an estimator instance "
                            "cannot be saved, because it could not be rebuilt on loading. Register the builder with "
                            "register_flow(name, builder) and select it with flow=FlowConfig(model=name)")
        flow_config = plain(self.flow_config.to_dict())
        if not _is_plain(flow_config):
            raise TypeError(f"{type(self).__name__}: FlowConfig.kwargs = {self.flow_config.kwargs!r} holds values a "
                            "checkpoint cannot store; keep them to numbers, strings, lists and dicts, and create "
                            "anything else in a builder registered with register_flow")
        return {"kind": "nle", "n_x": self.n_x, "n_theta": self.n_theta, "flow_config": flow_config,
                "flow": {k: v.detach().cpu().clone() for k, v in self._flow.state_dict().items()},
                "s_scaler": plain(self.s_scaler.state()), "t_scaler": plain(self.t_scaler.state()),
                "train_config": None if self.train_config is None else self.train_config.to_dict(),
                "history": [float(v) for v in self.history], "train_history": [float(v) for v in self.train_history],
                "best_val_loss": self.best_val_loss, "epochs": self.epochs, "seed": self.seed}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a trained NLE from :meth:`state`. A flow registered with :func:`register_flow`
        must be registered again (import its module) before loading."""
        with seeded(0):                                          # the initial weights are replaced right away
            nle = cls(state["n_x"], state["n_theta"], state["flow_config"], device, state.get("seed"))
        nle._flow.load_state_dict(state["flow"])
        nle._flow.eval()
        nle.s_scaler, nle.t_scaler = scaler_from_state(state["s_scaler"]), scaler_from_state(state["t_scaler"])
        if state.get("train_config") is not None:
            nle.train_config = TrainConfig.make(state["train_config"])
        nle.history, nle.train_history = list(state.get("history", [])), list(state.get("train_history", []))
        nle.best_val_loss, nle.epochs = state.get("best_val_loss"), state.get("epochs")
        nle._fitted = True
        return nle


class Ensemble(_Likelihood):
    """Mixture of likelihoods: p(s | theta) = sum_k w_k p_k(s | theta).

    It can be used wherever an :class:`NLE` is (sampling, evaluation, saving). Because every member
    returns the density of the raw summaries, members may have been trained on different
    simulations with different scalers.

    Args:
        members: fitted :class:`NLE` objects (or ensembles) for the same summaries and parameters.
        weights: one non-negative weight per member (default: equal); they are normalised to sum
            to 1.

    Attributes:
        members: the list of members.
        weights: (K,) array of normalised weights.

    Raises:
        ValueError: for no members, a member without ``log_prob``, an unfitted member, members
            with different ``n_x`` or ``n_theta``, or invalid weights.

    Example:
        >>> members = [NLE(seed=k) for k in range(4)]
        >>> for member in members:
        ...     member.fit(s_train, theta_train, s_val, theta_val)
        >>> ens = Ensemble(members)
        >>> ens.log_likelihood(s_test, theta_test)        # (n_test,) tensor
    """

    def __init__(self, members, weights=None):
        self.members = list(members)
        if not self.members:
            raise ValueError("Ensemble needs at least one member")
        for i, m in enumerate(self.members):
            if not callable(getattr(m, "log_prob", None)):
                raise ValueError(f"Ensemble member {i} ({type(m).__name__}) has no log_prob(input, condition)")
            if not getattr(m, "fitted", True):
                raise ValueError(f"Ensemble member {i} ({type(m).__name__}) is not fitted; call its fit() first")
        for name in ("n_x", "n_theta"):
            dims = [getattr(m, name, None) for m in self.members]
            if len(set(dims) - {None}) > 1:
                raise ValueError(f"Ensemble members must share {name}, got {dims}")
        k = len(self.members)
        w = np.full(k, 1.0 / k) if weights is None else as_numpy(weights)
        if w.shape != (k,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"weights must be {k} non-negative numbers (one per member) with a positive sum, "
                             f"got {weights!r}")
        self.weights = w / w.sum()

    def _dim(self, name):
        return next((n for m in self.members if (n := getattr(m, name, None)) is not None), None)

    @property
    def n_x(self) -> Optional[int]:
        """Dimension of the summaries."""
        return self._dim("n_x")

    @property
    def n_theta(self) -> Optional[int]:
        """Number of parameters."""
        return self._dim("n_theta")

    @property
    def fitted(self) -> bool:
        """Whether every member is fitted."""
        return all(getattr(m, "fitted", True) for m in self.members)

    @property
    def device(self):
        """Device of the first member."""
        return self.members[0].device

    @property
    def _eval_dtype(self):
        # float64 only if every member scales its own inputs; other likelihoods get float32
        return torch.float64 if all(getattr(m, "_eval_dtype", None) == torch.float64 for m in self.members) \
            else torch.float32

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return (f"{type(self).__name__}({len(self)} members, n_x={self.n_x}, n_theta={self.n_theta}, "
                f"weights={np.round(self.weights, 4).tolist()})")

    def log_prob(self, input, condition):
        """Log of the weighted mean of the member likelihoods; same layout as :meth:`NLE.log_prob`."""
        lp = torch.stack([m.log_prob(input, condition) for m in self.members])
        log_w = torch.log(to_tensor(self.weights, dtype=lp.dtype, device=lp.device))
        return torch.logsumexp(lp + log_w.reshape(-1, *[1] * (lp.ndim - 1)), 0)

    def to(self, device):
        """Move every member to ``device``; returns itself."""
        for m in self.members:
            m.to(device)
        return self

    def state(self) -> dict:
        """Plain serialisable state (member states and weights) for :func:`sbikit.io.save`."""
        return {"kind": "ensemble", "members": [m.state() for m in self.members], "weights": self.weights.tolist()}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild an ensemble from :meth:`state`; members that are ensembles are restored as such."""
        members = [(Ensemble if s.get("kind") == "ensemble" else NLE).from_state(s, device) for s in state["members"]]
        return cls(members, state.get("weights"))
