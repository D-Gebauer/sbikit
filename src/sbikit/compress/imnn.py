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
(a fixed preprocessing of the inputs, fitted on the training fiducial sims). Comparing log det F instead of
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
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..data import FiducialSet
from ..scaling import make_scaler, scaler_from_state
from .base import Compressor, register

__all__ = ["IMNNConfig", "IMNN", "IMNNNet", "imnn_loss", "fisher_statistics", "regularisation", "get_alpha"]

_LEGACY = {"epochs", "steps_per_epoch", "val_fraction"}


@dataclass
class IMNNConfig:
    """Network and training settings of the :class:`IMNN` (defaults: the imnn 0.3.2 examples).

    Attributes:
        n_summaries: network outputs (default: the number of parameters).
        widths: hidden layer widths.
        negative_slope: LeakyReLU slope.
        lr: Adam learning rate.
        lam: coupling strength of the covariance regularisation (imnn's lambda).
        eps: closeness criterion of the regularisation (imnn's epsilon).
        patience: iterations without a better validation det F before stopping.
        min_iterations: iterations before the patience counter starts.
        max_iterations: upper limit of iterations.
        batch_fid, batch_deriv: random subsets of the fiducial sims / derivative pairs per
            iteration (None: all, as the reference).
        dropout, weight_decay, grad_clip: optional regularisation (off as in the reference).
        input_scaling: ``"identity"`` (reference), ``"zscore"`` or ``"robust_asinh"``, fitted on
            the training fiducial sims.
        output: ``"mle"`` (the quasi maximum-likelihood estimate theta_fid + F^-1 dmu^T C^-1
            (x - mu), one summary per parameter) or ``"raw"`` (the network outputs).
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
    input_scaling: str = "identity"        # identity (reference) | zscore | robust_asinh
    output: str = "mle"                    # mle | raw

    @classmethod
    def make(cls, cfg) -> "IMNNConfig":
        """A config from None (defaults), an instance (returned as is) or a dict."""
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        cfg = dict(cfg)
        old = _LEGACY & set(cfg)
        if old:
            raise ValueError(f"IMNNConfig: {sorted(old)} no longer exist (iterations as in imnn 0.3.2: "
                             "max_iterations, patience, min_iterations; validation set: IMNN(..., validation=)")
        if "widths" in cfg:
            cfg["widths"] = tuple(cfg["widths"])
        return cls(**cfg)


class IMNNNet(nn.Module):
    """stax.serial(Dense(w_1), LeakyRelu, ..., Dense(n_out)) with stax's default initialisers."""

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
        device: torch device for training and evaluation.
        seed: torch seed set at the start of :meth:`fit`.
        validation: a separate FiducialSet for early stopping and the final statistics
            (recommended; see the module docstring).

    Attributes:
        fisher, cov, dmu, mu: Fisher matrix, covariance, mean derivatives and mean of the network
            summaries after :meth:`fit` (from the validation set if given).
        history: per iteration, a dict with ``loss``, ``logdetF``, ``Lambda2``, ``r`` and, with a
            validation set, ``val_logdetF`` and ``val_Lambda2``.

    Example:
        >>> trn, val = fid.split(run_id_fid, run_id_deriv, val_groups=[0, 1])
        >>> imnn = IMNN(trn, {"patience": 250, "min_iterations": 1000}, validation=val)
        >>> imnn.fit()                                    # -log det F on the validation set
        >>> s = imnn(x)                                   # (N, K) quasi-MLE of the parameters
    """
    kind = "imnn"

    def __init__(self, fiducial: Optional[FiducialSet] = None, config=None, device="cpu",
                 seed: Optional[int] = None, validation: Optional[FiducialSet] = None):
        self.fiducial, self.validation = fiducial, validation
        self.config, self.device, self.seed = IMNNConfig.make(config), device, seed
        self.net: Optional[IMNNNet] = None
        self.history: list = []
        if fiducial is not None:
            self.params = list(fiducial.param_names)
            self.n_params = len(self.params)
            self.n_out = self.n_params if self.config.output == "mle" else self._n_summaries()
            for s in (fiducial, validation):
                if s is not None and not s.has_derivative_sims:
                    raise ValueError("the IMNN needs FiducialSets with seed-matched derivative sims (x_minus/x_plus)")
            if validation is not None and (list(validation.param_names) != self.params
                                           or not np.allclose(validation.delta, fiducial.delta)):
                raise ValueError("validation set: different parameters or derivative steps")

    def _n_summaries(self):
        return self.config.n_summaries or self.n_params

    def _scale(self, x):
        return torch.as_tensor(np.asarray(self.in_scaler(np.asarray(x, dtype=np.float64))), dtype=torch.float32)

    def _tensors(self, s):
        return (self._scale(s.x).to(self.device), self._scale(s.x_plus).to(self.device),
                self._scale(s.x_minus).to(self.device))

    def _summaries(self, x):
        """Net outputs for a (..., D) array, in chunks."""
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        out = [self.net(self._scale(b).to(self.device)) for b in np.array_split(flat, max(1, len(flat) // 8192))]
        return torch.cat(out).reshape(*shape, -1)

    def fit(self, x_trn=None, t_trn=None, x_val=None, t_val=None) -> float:
        """Train on the fiducial set (the arguments are ignored). Returns -log det F of the best
        weights on the validation set (on the training set without one), so the IMNN can be tuned
        like the other compressors; inf if F was never positive definite."""
        trn, val, c, dev = self.fiducial, self.validation, self.config, self.device
        if trn is None:
            raise ValueError("no FiducialSet")
        if len(trn.x) <= self._n_summaries() + 1:
            raise ValueError("too few fiducial training sims for the summary covariance")
        if self.seed is not None:
            torch.manual_seed(self.seed)
        kw = {"k": 5.0} if c.input_scaling == "robust_asinh" else {}
        self.in_scaler = make_scaler(c.input_scaling, **kw).fit(trn.x)
        delta = torch.as_tensor(trn.delta, dtype=torch.float32, device=dev)
        xf, xp, xm = self._tensors(trn)
        xv = self._tensors(val) if val is not None else None
        self.net = IMNNNet(xf.shape[1], self._n_summaries(), c.widths, c.negative_slope, c.dropout).to(dev)
        opt = (torch.optim.Adam(self.net.parameters(), lr=c.lr) if c.weight_decay == 0
               else torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay))
        n_f, n_d = len(xf), xp.shape[1]
        best, best_state, patience_counter = -np.inf, None, 0
        self.history = []
        for it in range(c.max_iterations):
            if patience_counter >= c.patience:
                break
            self.net.train()
            i_f = torch.randperm(n_f, device=dev)[:c.batch_fid] if c.batch_fid else slice(None)
            i_d = torch.randperm(n_d, device=dev)[:c.batch_deriv] if c.batch_deriv else slice(None)
            loss, (f_trn, _, _, lam2, r) = imnn_loss(self.net(xf[i_f]), self.net(xp[:, i_d]), self.net(xm[:, i_d]),
                                                     delta, c.lam, c.eps)
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
                with torch.no_grad():
                    f_val, cov_v, cinv_v, _, _ = fisher_statistics(self.net(xv[0]), self.net(xv[1]), self.net(xv[2]),
                                                                   delta)
                rec.update(val_logdetF=float(_slogdet(f_val)), val_Lambda2=float(regularisation(cov_v, cinv_v)))
                f_score = f_val
            sign, logdet = torch.linalg.slogdet(f_score)
            score = float(logdet) if sign > 0 else -np.inf
            self.history.append(rec)
            if score > best:
                best, patience_counter = score, 0
                best_state = copy.deepcopy(self.net.state_dict())
            elif it > c.min_iterations:
                patience_counter += 1
        self.iterations = len(self.history)
        if best_state is None:
            return float("inf")
        self.net.load_state_dict(best_state)
        self.net.eval()
        self._finalise()
        return -float(best)

    def _finalise(self):
        """mu, C, dmu, F for the estimator: on the validation set if given (imnn ``set_F_statistics``)."""
        s = self.validation if self.validation is not None else self.fiducial
        delta = torch.as_tensor(s.delta, dtype=torch.float64)
        with torch.no_grad():
            t = [self._summaries(a).double().cpu() for a in (s.x, s.x_plus, s.x_minus)]
        self.fisher, self.cov, _, self.dmu, self.mu = fisher_statistics(*t, delta)
        self.theta_fid = torch.as_tensor(s.theta_fid, dtype=torch.float64)

    def __call__(self, x):
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.eval()
        with torch.no_grad():
            t = self._summaries(np.asarray(x)).double().cpu()
        if self.config.output == "raw":
            return t.numpy().astype(np.float32)
        # imnn get_estimate: theta_fid + F^-1 dmu^T C^-1 (x - mu)
        proj = torch.linalg.solve(self.fisher, self.dmu.T @ torch.linalg.inv(self.cov))   # (K, n)
        return (self.theta_fid + (t - self.mu) @ proj.T).numpy().astype(np.float32)

    def to(self, device):
        """Move the network to ``device``; returns itself."""
        self.device = device
        if self.net is not None:
            self.net.to(device)
        return self

    def state(self):
        """Plain serialisable state (config, weights, scaler and the estimator's statistics)."""
        return {"kind": self.kind, "params": self.params, "n_out": self.n_out, "n_summaries": self._n_summaries(),
                "n_in": self.net.layers[0].in_features, "config": asdict(self.config),
                "net": {k: v.cpu() for k, v in self.net.state_dict().items()}, "in_scaler": self.in_scaler.state(),
                "mu": self.mu, "cov": self.cov, "dmu": self.dmu, "fisher": self.fisher, "theta_fid": self.theta_fid,
                "reference": "imnn 0.3.2 NumericalGradientIMNN"}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted IMNN from :meth:`state`."""
        m = cls(None, state["config"], device)
        m.params, m.n_out = list(state["params"]), state["n_out"]
        m.n_params = len(m.params)
        c = m.config
        m.net = IMNNNet(state["n_in"], state["n_summaries"], c.widths, c.negative_slope, c.dropout)
        m.net.load_state_dict(state["net"])
        m.net.to(device).eval()
        m.in_scaler = scaler_from_state(state["in_scaler"])
        for k in ("mu", "cov", "dmu", "fisher", "theta_fid"):
            setattr(m, k, state[k])
        return m
