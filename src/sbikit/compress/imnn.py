"""Information maximising neural networks (Charnock, Lavaux & Wandelt 2018; Makinen et al. 2021).

Follows the authors' reference implementation, ``imnn`` 0.3.2 (``NumericalGradientIMNN``, seed-matched
finite-difference derivatives), not the 2018 paper's original loss (-|F|^2/2 with a |C| penalty), which the
authors replaced. Conventions taken from imnn 0.3.2 (``_imnn.py``, ``numerical_gradient_imnn.py``):

- summaries x = f(d); mu = mean over fiducial sims; C = sample covariance (n_s - 1); dmu/dtheta =
  mean_i [f(d_i^+) - f(d_i^-)] / dtheta with dtheta = theta^+ - theta^-; F = dmu^T C^-1 dmu;
- loss = -slogdet(F) + r Lambda2, Lambda2 = ||C - 1||_F + ||C^-1 - 1||_F (Frobenius norms, not squared),
  r = lam Lambda2 / (Lambda2 + exp(-alpha Lambda2)), alpha = -ln(eps (lam - 1) + eps^2 / (1 + eps)) / eps,
  where slogdet = sign * log|det| (imnn's ``_slogdet``);
- one Adam step per iteration on the whole training set (fiducial + derivative sims);
- after each step, F on the validation set (if given, else the training F of that step); the best weights are
  kept; the patience counter grows only after ``min_iterations``; stop at ``patience`` or ``max_iterations``;
- final mu, C, dmu, F (for the estimator) from the validation set if given, else from the training set;
- estimator theta_fid + F^-1 dmu^T C^-1 (x - mu) (``get_estimate``);
- network as in the imnn examples: stax.serial(Dense(128), LeakyRelu, x3, Dense(n_summaries)) with stax's
  initialisers (truncated glorot-normal weights, N(0, 0.01^2) biases), LeakyReLU slope 0.01, Adam lr 1e-3.

Options beyond the reference (all off by default): ``batch_fid``/``batch_deriv`` (random subsets per
iteration, as the 2018 paper's batches), ``dropout``, ``weight_decay``, ``grad_clip``, ``input_scaling``
(a fixed preprocessing of the inputs, fitted on the training fiducial sims; the reference feeds the data
vectors to the network as they are, which trains poorly unless they are of order one), and other network
architectures (``network``, see :func:`~sbikit.compress.register_network`). Comparing log det F instead of
det F for early stopping is the same ordering (and does not overflow).

Validation: the reference expects separate validation sims for early stopping ("with a fixed data set ...
there *will* be accidental correlations which look like they are due to parameters"). Pass a validation
:class:`FiducialSet`; :meth:`FiducialSet.split` makes one by group (e.g. N-body run), so that copies sharing
initial conditions never sit on both sides.

Like MOPED, an IMNN is optimal only near the fiducial point it was trained at.
"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from .._utils import ConfigBase, logger, parse_widths, plain, resolve_device, seeded, unknown
from ..data import FiducialSet
from ..scaling import SCALERS, Identity, make_scaler, scaler_from_state
from .base import NETWORKS, Compressor, register

__all__ = ["IMNNConfig", "IMNN", "IMNNNet", "imnn_loss", "fisher_statistics", "regularisation", "get_alpha"]

_LEGACY = {"epochs", "steps_per_epoch", "val_fraction"}


@dataclass
class IMNNConfig(ConfigBase):
    """Network and training settings of the :class:`IMNN` (defaults: the imnn 0.3.2 examples).

    Unknown keys raise ``TypeError``, invalid values ``ValueError``. Numbers may be given in any numeric
    type, and the real-valued options also as strings (``"1e-3"``, as a YAML file may deliver them).

    Attributes:
        n_summaries: network outputs (default: the number of parameters; at least that many).
        widths: hidden layer widths (a sequence, an int or a string like ``"128,128"``).
        negative_slope: LeakyReLU slope.
        lr: Adam learning rate.
        lam: coupling strength of the covariance regularisation (imnn's lambda).
        eps: closeness criterion of the regularisation (imnn's epsilon).
        patience: iterations without a better validation det F before stopping.
        min_iterations: iterations before the patience counter starts.
        max_iterations: upper limit of iterations.
        batch_fid: random subset of the fiducial sims per iteration (None: all, as the reference).
            It must exceed ``n_summaries + 1``; a value above the number of sims uses all.
        batch_deriv: random subset of the derivative pairs per iteration (None: all, as the
            reference); between 1 and the number of pairs.
        dropout, weight_decay: optional regularisation (off as in the reference).
        grad_clip: maximum gradient norm (None or 0: no clipping, as in the reference).
        input_scaling: name of a scaler in :mod:`sbikit.scaling` (``"identity"`` (reference),
            ``"zscore"``, ``"robust_asinh"`` or a registered one), fitted on the training fiducial
            sims. Use ``"zscore"`` for data vectors that are not of order one.
        output: ``"mle"`` (the quasi maximum-likelihood estimate theta_fid + F^-1 dmu^T C^-1
            (x - mu), one summary per parameter) or ``"raw"`` (the network outputs).
        asinh_k: the ``k`` of ``"robust_asinh"``.
        network: name of the architecture (``"imnn_mlp"``: the reference network, or one added
            with :func:`~sbikit.compress.register_network`).
        network_kwargs: options for a registered network's factory (plain values).

    Example:
        >>> IMNNConfig.make({"patience": 250, "min_iterations": 1000, "input_scaling": "zscore"})
    """

    n_summaries: Optional[int] = None      # network outputs; default = number of parameters
    widths: tuple = (128, 128, 128)
    negative_slope: float = 0.01
    lr: float = 1e-3
    lam: float = 10.0                      # coupling strength of the covariance regularisation (imnn: λ)
    eps: float = 0.1                       # closeness criterion of the regularisation (imnn: ϵ)
    patience: int = 100
    min_iterations: int = 100
    max_iterations: int = 100_000
    batch_fid: Optional[int] = None        # None: all training fiducial sims every iteration (reference)
    batch_deriv: Optional[int] = None      # None: all training derivative pairs every iteration (reference)
    dropout: float = 0.0
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    input_scaling: str = "identity"        # identity (reference) | zscore | robust_asinh | a registered scaler
    output: str = "mle"                    # mle | raw
    asinh_k: float = 5.0
    network: str = "imnn_mlp"
    network_kwargs: dict = field(default_factory=dict)

    _choices = {"output": ("mle", "raw")}
    _ints = ("patience", "min_iterations", "max_iterations")
    _optional_ints = ("n_summaries", "batch_fid", "batch_deriv")
    _floats = ("negative_slope", "lr", "lam", "eps", "dropout", "weight_decay", "asinh_k")

    @classmethod
    def make(cls, cfg=None, **overrides) -> "IMNNConfig":
        """A config from None (defaults), an instance or a dict; ``overrides`` replace fields.

        Raises:
            TypeError: for a key that is not a field of the config.
            ValueError: for the removed keys ``epochs``, ``steps_per_epoch`` and ``val_fraction``,
                and for invalid values.
        """
        old = _LEGACY & ((set(cfg) if isinstance(cfg, Mapping) else set()) | set(overrides))
        if old:
            raise ValueError(f"IMNNConfig: {sorted(old)} no longer exist (iterations as in imnn 0.3.2: "
                             "max_iterations, patience, min_iterations; validation set: IMNN(..., validation=)")
        return super().make(cfg, **overrides)

    def _normalise(self):
        name = type(self).__name__
        try:
            widths = self.widths
            if not isinstance(widths, str):
                widths = [widths] if np.isscalar(widths) else list(widths)
                if any(isinstance(w, bool) or (not isinstance(w, str) and int(w) != w) for w in widths):
                    raise ValueError
            self.widths = parse_widths(widths)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name}.widths must be positive integers, e.g. (128, 128) or '128,128', got "
                             f"{self.widths!r}") from None
        for key in self._ints + tuple(k for k in self._optional_ints if getattr(self, k) is not None):
            value = getattr(self, key)
            try:
                if isinstance(value, (bool, str)) or int(value) != value:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                allowed = "None or an integer" if key in self._optional_ints else "an integer"
                raise ValueError(f"{name}.{key} must be {allowed}, got {value!r}") from None
            setattr(self, key, int(value))
        for key in self._floats + (("grad_clip",) if self.grad_clip is not None else ()):
            value = getattr(self, key)
            try:
                if isinstance(value, bool):
                    raise ValueError
                setattr(self, key, float(value))
            except (TypeError, ValueError):
                raise ValueError(f"{name}.{key} must be a number, got {value!r}") from None
        if self.network_kwargs is None:
            self.network_kwargs = {}
        if not isinstance(self.network_kwargs, Mapping):
            raise ValueError(f"{name}.network_kwargs must be a dict, got {type(self.network_kwargs).__name__}")
        self.network_kwargs = dict(self.network_kwargs)

    def validate(self):
        req = self._require
        req(self.n_summaries is None or self.n_summaries >= 1,
            f"n_summaries must be None or an integer >= 1, got {self.n_summaries}")
        req(self.negative_slope >= 0, f"negative_slope must be >= 0, got {self.negative_slope}")
        req(np.isfinite(self.lr) and self.lr > 0, f"lr must be positive, got {self.lr}")
        req(self.eps > 0 and self.eps * (self.lam - 1.0) + self.eps ** 2 / (1 + self.eps) > 0,
            f"lam = {self.lam}, eps = {self.eps}: need eps > 0 and eps (lam - 1) + eps^2 / (1 + eps) > 0")
        req(self.patience >= 1, f"patience must be an integer >= 1, got {self.patience}")
        req(self.max_iterations >= 1, f"max_iterations must be an integer >= 1, got {self.max_iterations}")
        req(self.min_iterations >= 0, f"min_iterations must be >= 0, got {self.min_iterations}")
        for name in ("batch_fid", "batch_deriv"):
            value = getattr(self, name)
            req(value is None or value >= 1, f"{name} must be None (all) or an integer >= 1, got {value!r}")
        req(0 <= self.dropout < 1, f"dropout must be in [0, 1), got {self.dropout}")
        req(self.weight_decay >= 0, f"weight_decay must be >= 0, got {self.weight_decay}")
        req(self.grad_clip is None or self.grad_clip >= 0,
            f"grad_clip must be None, 0 (both: no clipping) or positive, got {self.grad_clip}")
        req(self.asinh_k > 0, f"asinh_k must be positive, got {self.asinh_k}")
        if not isinstance(self.input_scaling, str) or self.input_scaling not in SCALERS:
            raise unknown("input_scaling", self.input_scaling, list(SCALERS))
        if not isinstance(self.network, str) or self.network not in NETWORKS:
            raise unknown("network", self.network, list(NETWORKS))


class IMNNNet(nn.Module):
    """stax.serial(Dense(w_1), LeakyRelu, ..., Dense(n_out)) with stax's default initialisers.

    Args:
        n_in, n_out: input and output sizes.
        widths: hidden layer widths.
        negative_slope: LeakyReLU slope.
        dropout: dropout probability after each hidden layer (0: none).
    """

    def __init__(self, n_in, n_out, widths=(128, 128, 128), negative_slope=0.01, dropout=0.0):
        super().__init__()
        layers, d = [], n_in
        for w in widths:
            layers += [nn.Linear(d, w), nn.LeakyReLU(negative_slope)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = w
        layers.append(nn.Linear(d, n_out))
        self.layers = nn.Sequential(*layers)
        for m in self.layers:
            if isinstance(m, nn.Linear):
                # jax glorot_normal = variance_scaling(1, "fan_avg", "truncated_normal"): truncated at +-2 sd,
                # rescaled so the variance is 1 / fan_avg; stax normal() bias: N(0, 0.01^2)
                fan_avg = 0.5 * (m.in_features + m.out_features)
                sd = math.sqrt(1.0 / fan_avg) / 0.87962566103423978
                nn.init.trunc_normal_(m.weight, std=sd, a=-2 * sd, b=2 * sd)
                nn.init.normal_(m.bias, std=0.01)

    def forward(self, x):
        """(..., n_in) -> (..., n_out)."""
        return self.layers(x)


NETWORKS.register("imnn_mlp", lambda n_in, n_out, c: IMNNNet(n_in, n_out, c.widths, getattr(c, "negative_slope", 0.01), c.dropout),
                  overwrite=True)


def fisher_statistics(x, x_plus, x_minus, delta):
    """imnn ``_calculate_F_statistics``: F (K, K), C, C^-1 (n, n), dmu (n, K), mu (n,).

    x (n_s, n), x_plus/x_minus (K, n_d, n), delta (K,) = theta^+ - theta^-.
    """
    mu = x.mean(0)
    z = x - mu
    cov = z.T @ z / (x.shape[0] - 1)
    cinv = torch.linalg.inv(cov)
    dmu = ((x_plus - x_minus) / delta[:, None, None]).mean(1).T
    fisher = dmu.T @ cinv @ dmu
    return fisher, cov, cinv, dmu, mu


def regularisation(cov, cinv):
    """imnn ``_get_regularisation``: ||C - 1||_F + ||C^-1 - 1||_F."""
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return torch.linalg.norm(cov - eye) + torch.linalg.norm(cinv - eye)


def get_alpha(lam, eps):
    """imnn ``get_α``."""
    return -math.log(eps * (lam - 1.0) + eps ** 2 / (1 + eps)) / eps


def _slogdet(m):
    sign, logdet = torch.linalg.slogdet(m)
    return sign * logdet


def imnn_loss(x, x_plus, x_minus, delta, lam=10.0, eps=0.1):
    """imnn ``_calculate_loss``: (loss, (F, C, C^-1, Lambda2, r))."""
    fisher, cov, cinv, _, _ = fisher_statistics(x, x_plus, x_minus, delta)
    lam2 = regularisation(cov, cinv)
    r = lam * lam2 / (lam2 + torch.exp(-get_alpha(lam, eps) * lam2))
    return -_slogdet(fisher) + r * lam2, (fisher, cov, cinv, lam2, r)


@register
class IMNN(Compressor):
    """IMNN compressor trained on a :class:`FiducialSet` with seed-matched derivative simulations.

    Args:
        fiducial: training set, with ``x_minus`` / ``x_plus``.
        config: :class:`IMNNConfig` or dict.
        device: torch device for training and evaluation (``"auto"``: a GPU if there is one).
        seed: seed of the network initialisation and the batches in :meth:`fit`; the same seed
            gives the same result, and the caller's random state is left untouched. None: use
            the global torch random state.
        validation: a separate FiducialSet for early stopping and the final statistics
            (recommended; see the module docstring).
        net: expert shortcut (keyword only): a factory ``net(n_in, n_summaries) -> torch.nn.Module``
            mapping (N, n_in) to (N, n_summaries), used instead of ``config.network``. Such an
            IMNN trains and compresses normally but cannot be saved; to make an architecture
            saveable, add it with :func:`~sbikit.compress.register_network` and select it with
            ``config.network``.

    Attributes:
        fisher, cov, dmu, mu: Fisher matrix, covariance, mean derivatives and mean of the network
            summaries after :meth:`fit` (from the validation set if given).
        history: per iteration, a dict with ``loss``, ``logdetF``, ``Lambda2``, ``r`` and, with a
            validation set, ``val_logdetF`` and ``val_Lambda2`` (NaN in an iteration whose validation
            covariance was singular).
        train_history: the training loss per iteration.
        iterations: number of iterations of the last :meth:`fit`.
        val_loss: what the last :meth:`fit` returned (both None after loading a checkpoint that
            does not record them).

    Raises:
        ValueError: for sets without derivative simulations, a validation set that does not match
            the training set, ``n_summaries`` below the number of parameters, or too few
            simulations (see :meth:`fit`).

    Example:
        >>> trn, val = fid.split(run_id_fid, run_id_deriv, val_groups=[0, 1])
        >>> imnn = IMNN(trn, {"patience": 250, "min_iterations": 1000}, validation=val)
        >>> imnn.fit()                                    # -log det F on the validation set
        >>> s = imnn(x)                                   # (N, K) quasi-MLE of the parameters
    """
    kind = "imnn"

    def __init__(self, fiducial: Optional[FiducialSet] = None, config=None, device="cpu",
                 seed: Optional[int] = None, validation: Optional[FiducialSet] = None, *, net=None):
        self.fiducial, self.validation = fiducial, validation
        self.config, self.device, self.seed = IMNNConfig.make(config), resolve_device(device), seed
        self._net_factory = net
        self.net: Optional[nn.Module] = None
        self.history: list = []
        self.iterations, self.val_loss = 0, float("inf")
        if fiducial is not None:
            self.params = list(fiducial.param_names)
            self.n_params = len(self.params)
            self.n_out = self.n_params if self.config.output == "mle" else self._n_summaries()
            self._check_sets(warn=False)

    def _n_summaries(self):
        return self.config.n_summaries or self.n_params

    def _check_sets(self, warn: bool = True) -> None:
        """Raise ``ValueError`` if the sets and the config cannot give a well-defined Fisher matrix;
        ``warn`` about sets that are large enough for that but small."""
        trn, val, c, n = self.fiducial, self.validation, self.config, self._n_summaries()
        for s in (trn, val):
            if s is not None and not s.has_derivative_sims:
                raise ValueError("the IMNN needs FiducialSets with seed-matched derivative sims (x_minus/x_plus)")
        if val is not None:
            if list(val.param_names) != self.params or not np.allclose(val.delta, trn.delta):
                raise ValueError("validation set: different parameters or derivative steps")
            if val.x.shape[1] != trn.x.shape[1]:
                raise ValueError(f"validation set: data vectors of length {val.x.shape[1]}, the training set has "
                                 f"{trn.x.shape[1]}")
            if not np.allclose(val.theta_fid, trn.theta_fid):
                raise ValueError("validation set: different fiducial parameters (theta_fid)")
        if n < self.n_params:
            raise ValueError(f"n_summaries = {n} is below the number of parameters ({self.n_params}): the Fisher "
                             "matrix of the summaries would be singular. The IMNN needs at least one summary per "
                             "parameter")
        sizes = {"the training set": len(trn.x)}
        if val is not None:
            sizes["the validation set"] = len(val.x)
        if c.batch_fid is not None:
            sizes["batch_fid"] = min(c.batch_fid, len(trn.x))
        for name, size in sizes.items():
            if size <= n + 1:
                raise ValueError(f"{name} has {size} fiducial simulations, but the covariance of {n} summaries "
                                 f"needs more than n_summaries + 1 = {n + 1} (and many more to be accurate)")
            if warn and size < 10 * n:
                warnings.warn(f"IMNN: {name} has only {size} fiducial simulations for {n} summaries; the summary "
                              "covariance is noisy and the Fisher information is overestimated", RuntimeWarning)
        for name, s in (("training", trn), ("validation", val)):
            if s is not None and not np.ptp(s.x, axis=0).any():
                raise ValueError(f"the fiducial simulations of the {name} set are all identical (no scatter in any "
                                 "column), so the covariance of the summaries would be singular")
        n_d = trn.x_minus.shape[1]
        if c.batch_deriv is not None and c.batch_deriv > n_d:
            raise ValueError(f"batch_deriv = {c.batch_deriv} exceeds the {n_d} derivative pairs of the training set")

    def _make_net(self, n_in: int, n_summaries: int) -> nn.Module:
        if self._net_factory is not None:
            return self._net_factory(n_in, n_summaries)
        return NETWORKS[self.config.network](n_in, n_summaries, self.config)

    def _scale(self, x):
        return torch.as_tensor(np.asarray(self.in_scaler(np.asarray(x, dtype=np.float64))), dtype=torch.float32)

    def _tensors(self, s):
        return (self._scale(s.x).to(self.device), self._scale(s.x_plus).to(self.device),
                self._scale(s.x_minus).to(self.device))

    def _forward(self, x):
        """Net outputs for a (..., D) tensor; the network sees (N, D)."""
        return self.net(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1)

    def _summaries(self, x):
        """Net outputs for a (..., D) array, in chunks."""
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        out = [self.net(self._scale(b).to(self.device)) for b in np.array_split(flat, max(1, len(flat) // 8192))]
        out = torch.cat(out)
        return out.reshape(*shape, out.shape[-1])

    def fit(self, x_trn=None, t_trn=None, x_val=None, t_val=None, *, callback=None, verbose: bool = False) -> float:
        """Train on the fiducial set (the positional arguments are ignored).

        Args:
            x_trn, t_trn, x_val, t_val: not used; the IMNN trains on its fiducial set(s).
            callback: called after each iteration with a dict ``{"epoch": iteration, "train_loss":
                the training loss, "val_loss": -log det F of this iteration (inf if F is not
                positive definite), "best": the lowest ``val_loss`` so far, "model": this IMNN}``.
                If it returns True, training stops and the best weights so far are kept.
            verbose: log the progress every 100 iterations and a summary at the end.

        Returns:
            -log det F of the best weights on the validation set (on the training set without one),
            so the IMNN can be tuned like the other compressors; inf if F was never positive
            definite (with a ``RuntimeWarning``; the IMNN is then left unfitted). A singular
            covariance of the summaries counts as an F that is not positive definite; if it occurs
            in the training loss, training stops there (with a ``RuntimeWarning``) and the best
            weights before it are kept.

        Raises:
            ValueError: without a fiducial set; if the training set, the validation set or
                ``batch_fid`` has no more than ``n_summaries + 1`` fiducial simulations; if
                ``batch_deriv`` exceeds the number of derivative pairs; if ``n_summaries`` is
                below the number of parameters; or if the network returns the wrong shape.

        If training is interrupted (``KeyboardInterrupt`` or an error in the callback), the best
        weights so far are restored before the exception is raised again.
        """
        trn, val, c, dev = self.fiducial, self.validation, self.config, self.device
        if trn is None:
            raise ValueError("IMNN.fit needs a FiducialSet: IMNN(fiducial, ...)")
        self._check_sets()
        n_sum = self._n_summaries()
        if val is None and (c.batch_fid or c.batch_deriv or c.dropout > 0):
            warnings.warn("IMNN: without a validation set, early stopping and the returned loss use the Fisher "
                          "matrix of each training batch (with dropout active), which is biased high with "
                          "batch_fid, batch_deriv or dropout; pass validation=", RuntimeWarning)
        kw = {"k": c.asinh_k} if c.input_scaling == "robust_asinh" else {}
        self.in_scaler = make_scaler(c.input_scaling, **kw).fit(trn.x)
        if isinstance(self.in_scaler, Identity):
            scale = float(np.median(trn.x.std(0)))
            if not 0.1 <= scale <= 100.0:
                warnings.warn(f"IMNN: the data vector has a typical scatter of {scale:.1e} and is given to the "
                              "network unscaled (input_scaling='identity', the reference behaviour), which "
                              "trains poorly; consider input_scaling='zscore'", RuntimeWarning)
        self.net, self.history, self.iterations, self.val_loss = None, [], 0, float("inf")
        best, best_state = -np.inf, None
        with seeded(self.seed):
            delta = torch.as_tensor(trn.delta, dtype=torch.float32, device=dev)
            xf, xp, xm = self._tensors(trn)
            xv = self._tensors(val) if val is not None else None
            self.n_in = xf.shape[1]
            self.net = self._make_net(self.n_in, n_sum).to(dev)
            self.net.eval()
            with torch.no_grad():
                shape = tuple(self.net(xf[:2]).shape)
            if shape != (2, n_sum):
                self.net = None
                raise ValueError(f"the network must map (N, {self.n_in}) to (N, n_summaries = {n_sum}); it returned "
                                 f"shape {shape} for 2 rows")
            opt = (torch.optim.Adam(self.net.parameters(), lr=c.lr) if c.weight_decay == 0
                   else torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay))
            n_f, n_d = len(xf), xp.shape[1]
            patience_counter, stop = 0, "max_iterations"
            try:
                for it in range(c.max_iterations):
                    if patience_counter >= c.patience:
                        stop = "patience"
                        break
                    self.net.train()
                    i_f = torch.randperm(n_f, device=dev)[:c.batch_fid] if c.batch_fid else slice(None)
                    i_d = torch.randperm(n_d, device=dev)[:c.batch_deriv] if c.batch_deriv else slice(None)
                    try:
                        loss, (f_trn, _, _, lam2, r) = imnn_loss(self._forward(xf[i_f]), self._forward(xp[:, i_d]),
                                                                 self._forward(xm[:, i_d]), delta, c.lam, c.eps)
                    except torch.linalg.LinAlgError:        # no loss, so no further step: keep the best so far
                        stop = "singular"
                        break
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if c.grad_clip:
                        nn.utils.clip_grad_norm_(self.net.parameters(), c.grad_clip)
                    opt.step()
                    rec = {"loss": float(loss.detach()), "logdetF": float(_slogdet(f_trn.detach())),
                           "Lambda2": float(lam2.detach()), "r": float(r.detach())}
                    f_score = f_trn.detach()
                    if xv is not None:
                        self.net.eval()
                        try:
                            with torch.no_grad():
                                f_score, cov_v, cinv_v, _, _ = fisher_statistics(
                                    self._forward(xv[0]), self._forward(xv[1]), self._forward(xv[2]), delta)
                            rec.update(val_logdetF=float(_slogdet(f_score)),
                                       val_Lambda2=float(regularisation(cov_v, cinv_v)))
                        except torch.linalg.LinAlgError:    # singular validation covariance: no valid F
                            f_score = None
                            rec.update(val_logdetF=float("nan"), val_Lambda2=float("nan"))
                    sign, logdet = torch.linalg.slogdet(f_score) if f_score is not None else (0.0, 0.0)
                    score = float(logdet) if sign > 0 else -np.inf
                    self.history.append(rec)
                    if score > best:
                        best, patience_counter = score, 0
                        best_state = copy.deepcopy(self.net.state_dict())
                    elif it > c.min_iterations:
                        patience_counter += 1
                    if verbose and (it + 1) % 100 == 0:
                        logger.info(f"IMNN iteration {it + 1}: loss {rec['loss']:.4f}, log det F {score:.4f} "
                                    f"(best {best:.4f})")
                    if callback is not None and callback({"epoch": it, "train_loss": rec["loss"], "val_loss": -score,
                                                          "best": -best, "model": self}):
                        stop = "callback"
                        break
            except BaseException:
                self._keep_best(best_state, best)
                raise
            self._keep_best(best_state, best)
        singular = ("the covariance of the summaries became singular (the network returns constant or diverged "
                    "values)")
        if best_state is None:
            why = f" and {singular}" if stop == "singular" else ""
            warnings.warn(f"IMNN.fit: the Fisher matrix was never positive definite in {self.iterations} iterations"
                          f"{why}; the IMNN is left unfitted. Check the simulations and the config (input_scaling, "
                          "lr)", RuntimeWarning)
        elif self.net is not None and stop == "singular":
            warnings.warn(f"IMNN.fit: stopped in iteration {self.iterations + 1} because {singular}; the best "
                          f"weights before that are kept (log det F = {best:.4f})", RuntimeWarning)
        elif self.net is not None and verbose:
            logger.info(f"IMNN: stopped after {self.iterations} iterations ({stop}), log det F = {best:.4f} on the "
                        f"{'validation' if val is not None else 'training'} set")
        return self.val_loss

    def _keep_best(self, best_state, best) -> None:
        """Load the best weights and compute the estimator's statistics; unfitted if there are none."""
        self.iterations = len(self.history)
        if best_state is None:
            self.net, self.val_loss = None, float("inf")
            return
        self.net.load_state_dict(best_state)
        self.net.eval()
        try:
            self._finalise()
        except torch.linalg.LinAlgError:
            where = "validation" if self.validation is not None else "training"
            warnings.warn(f"IMNN.fit: with the best weights, the covariance of the summaries of the whole {where} "
                          "set is singular; the IMNN is left unfitted", RuntimeWarning)
            self.net, self.val_loss = None, float("inf")
            return
        self.val_loss = -float(best)

    def _finalise(self):
        """mu, C, dmu, F for the estimator: on the validation set if given (imnn ``set_F_statistics``)."""
        s = self.validation if self.validation is not None else self.fiducial
        delta = torch.as_tensor(s.delta, dtype=torch.float64)
        with torch.no_grad():
            t = [self._summaries(a).double().cpu() for a in (s.x, s.x_plus, s.x_minus)]
        self.fisher, self.cov, _, self.dmu, self.mu = fisher_statistics(*t, delta)
        self.theta_fid = torch.as_tensor(s.theta_fid, dtype=torch.float64).clone()

    @property
    def train_history(self) -> list:
        """The training loss of each iteration."""
        return [h["loss"] for h in self.history]

    def __call__(self, x):
        """Summaries (N, n_out), float32, of the data vectors ``x`` (N, D); a single data vector
        (D,) gives (n_out,), an empty batch (0, D) an empty (0, n_out) array.

        Raises:
            NotFittedError: before :meth:`fit`, or after a fit that failed.
            ValueError: if ``x`` is not 1D or 2D or has the wrong number of columns.
        """
        if self.net is None:
            raise self._not_fitted()
        x, single = self._batch(x, getattr(self, "n_in", None))
        if len(x) == 0:
            return np.empty((0, self.n_out), dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            t = self._summaries(x).double().cpu()
        if self.config.output == "raw":
            out = t.numpy().astype(np.float32)
        else:
            # imnn get_estimate: theta_fid + F^-1 dmu^T C^-1 (x - mu)
            proj = torch.linalg.solve(self.fisher, self.dmu.T @ torch.linalg.inv(self.cov))   # (K, n)
            out = (self.theta_fid + (t - self.mu) @ proj.T).numpy().astype(np.float32)
        return out[0] if single else out

    def to(self, device):
        """Move the network to ``device``; returns itself."""
        self.device = resolve_device(device)
        if self.net is not None:
            self.net.to(self.device)
        return self

    def state(self):
        """Plain serialisable state (config, weights, scaler and the estimator's statistics).

        Raises:
            NotFittedError: before :meth:`fit`, or after a fit that failed.
            TypeError: for an IMNN built with ``net=``, whose architecture cannot be rebuilt from
                the state.
        """
        if self.net is None:
            raise self._not_fitted()
        if self._net_factory is not None:
            raise TypeError("an IMNN built with net= cannot be saved, because the network cannot be rebuilt from the "
                            "checkpoint. Add the architecture with sbikit.compress.register_network(name, factory) "
                            "and select it with config={'network': name}")
        copy_ = lambda t: t.detach().cpu().clone()          # noqa: E731
        return {"kind": self.kind, "params": list(self.params), "n_out": int(self.n_out),
                "n_summaries": int(self._n_summaries()), "n_in": int(self.n_in), "config": plain(self.config.to_dict()),
                "net": {k: copy_(v) for k, v in self.net.state_dict().items()},
                "in_scaler": plain(self.in_scaler.state()),
                "mu": copy_(self.mu), "cov": copy_(self.cov), "dmu": copy_(self.dmu), "fisher": copy_(self.fisher),
                "theta_fid": copy_(self.theta_fid),
                "iterations": None if self.iterations is None else int(self.iterations),
                "val_loss": None if self.val_loss is None else float(self.val_loss),
                "seed": None if self.seed is None else int(self.seed),
                "reference": "imnn 0.3.2 NumericalGradientIMNN"}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted IMNN from :meth:`state`.

        Raises:
            ValueError: if the state's network is not registered (register a custom architecture
                with :func:`~sbikit.compress.register_network` before loading).
        """
        cfg = dict(state["config"])
        for k in ("batch_fid", "batch_deriv"):              # states written with 0 for "all simulations"
            cfg[k] = cfg.get(k) or None
        if cfg.get("output") not in IMNNConfig._choices["output"]:
            cfg["output"] = "mle"                           # what such a state's estimator computes
        m = cls(None, cfg, device, seed=state.get("seed"))
        m.params, m.n_out = list(state["params"]), state["n_out"]
        m.n_params, m.n_in = len(m.params), state["n_in"]
        m.net = m._make_net(state["n_in"], state["n_summaries"])
        m.net.load_state_dict(state["net"])
        m.net.to(m.device).eval()
        m.in_scaler = scaler_from_state(state["in_scaler"])
        for k in ("mu", "cov", "dmu", "fisher", "theta_fid"):      # copies, on the CPU as __call__ uses them
            setattr(m, k, torch.as_tensor(state[k], dtype=torch.float64).detach().cpu().clone())
        m.iterations, m.val_loss = state.get("iterations"), state.get("val_loss")
        return m

    def __repr__(self):
        c, params = self.config, getattr(self, "params", None)
        network = "custom net" if self._net_factory is not None else f"network={c.network!r}"
        fitted = "unfitted" if self.net is None else "fitted"
        if self.net is not None and self.val_loss is not None:
            fitted += f": {self.iterations} iterations, val_loss={self.val_loss:.4g}"
        return (f"IMNN(params={params}, n_summaries={c.n_summaries or (len(params) if params else None)}, "
                f"output={c.output!r}, {network}, device={self.device!r}, {fitted})")
