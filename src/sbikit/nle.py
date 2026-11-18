"""Neural likelihood estimation.

A conditional normalising flow (a zuko MAF or NSF, built with sbi's builders) learns the likelihood
p(s | theta) of summaries s given parameters theta from simulations. Summaries and parameters are
z-scored inside :class:`NLE`, and the scalers are stored with the weights, so a saved model takes
and returns raw values. :class:`Ensemble` averages the likelihoods of several flows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from sbi.neural_nets.net_builders import build_zuko_maf, build_zuko_nsf

from .scaling import ZScore, scaler_from_state

__all__ = ["FlowConfig", "TrainConfig", "NLE", "Ensemble"]

_BUILDERS = {"maf": build_zuko_maf, "nsf": build_zuko_nsf}


@dataclass
class FlowConfig:
    """Architecture of the flow.

    Attributes:
        model: ``"nsf"`` (neural spline flow) or ``"maf"`` (masked autoregressive flow).
        hidden_features: width of the hidden layers of each transform's network.
        num_blocks: number of hidden layers.
        num_transforms: number of autoregressive transforms.
        num_bins: spline bins (``"nsf"`` only).
    """

    model: str = "nsf"             # nsf | maf
    hidden_features: int = 64
    num_blocks: int = 2
    num_transforms: int = 5
    num_bins: int = 8              # nsf only

    @classmethod
    def make(cls, cfg):
        """A config from None (defaults), an instance (returned as is) or a dict (unknown keys are ignored)."""
        if cfg is None:
            return cls()
        return cfg if isinstance(cfg, cls) else cls(**{k: v for k, v in dict(cfg).items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    """Training settings of :meth:`NLE.fit`.

    Attributes:
        lr: Adam learning rate.
        batch_size: minibatch size.
        max_epochs: upper limit of epochs.
        stop_after_epochs: early-stopping patience, in epochs without improvement.
        min_delta: improvement of the validation loss that resets the patience.
        sched_factor, sched_patience: the learning rate is multiplied by ``sched_factor`` after
            ``sched_patience`` epochs without improvement (``ReduceLROnPlateau``).
        grad_clip: maximum gradient norm (None: no clipping).
        weight_decay: Adam weight decay.
    """

    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 1000
    stop_after_epochs: int = 20    # early stopping patience
    min_delta: float = 1e-5        # improvement needed to reset the patience
    sched_factor: float = 0.1 ** 0.5
    sched_patience: int = 5
    grad_clip: Optional[float] = 1.0
    weight_decay: float = 0.0

    @classmethod
    def make(cls, cfg):
        """A config from None (defaults), an instance (returned as is) or a dict (unknown keys are ignored)."""
        if cfg is None:
            return cls()
        return cfg if isinstance(cfg, cls) else cls(**{k: v for k, v in dict(cfg).items() if k in cls.__dataclass_fields__})


def _t(a, device):
    return torch.as_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a, dtype=torch.float32, device=device)


class NLE:
    """Conditional density p(s | theta) of summaries given parameters.

    ``log_prob(input, condition)`` takes raw summaries (``(1, M, n_x)`` or ``(M, n_x)``) and raw
    parameters (M, P) in the layout of sbi estimators, so ``torchemcee.integrations.sbi.flow_log_prob_fn``
    can use it. It is the density of the z-scored summaries; the constant Jacobian of the scaling is
    left out (irrelevant for posteriors, consistent between evaluations of one model).

    Args:
        n_x: dimension of the summaries.
        n_theta: number of parameters the flow is conditioned on.
        flow: :class:`FlowConfig` or dict.
        device: torch device of the flow.

    Attributes:
        flow: the zuko flow (an sbi density estimator).
        s_scaler, t_scaler: z-scores of summaries and parameters, fitted on the training set.
        history: validation loss per epoch of the last :meth:`fit`.

    Example:
        >>> nle = NLE(n_x=5, n_theta=3, flow={"model": "maf", "hidden_features": 64})
        >>> val_loss, epochs = nle.fit(s_train, theta_train, s_val, theta_val)
        >>> nle.log_likelihood(s_test, theta_test)        # (n_test,) tensor
    """

    def __init__(self, n_x: int, n_theta: int, flow=None, device="cpu"):
        self.n_x, self.n_theta, self.device = int(n_x), int(n_theta), device
        self.flow_config = FlowConfig.make(flow)
        c = self.flow_config
        kwargs = {"z_score_x": "none", "z_score_y": "none", "hidden_features": [c.hidden_features] * c.num_blocks,
                  "num_transforms": c.num_transforms}
        if c.model == "nsf":
            kwargs["num_bins"] = c.num_bins
        self.flow = _BUILDERS[c.model](torch.randn(64, self.n_x), torch.randn(64, self.n_theta), **kwargs).to(device)
        self.s_scaler: Optional[ZScore] = None
        self.t_scaler: Optional[ZScore] = None
        self.history: list = []

    # ------------------------------------------------------------------ training
    def fit(self, s_trn, t_trn, s_val, t_val, train=None, seed: Optional[int] = None):
        """Train with Adam, plateau decay of the learning rate and early stopping on the
        validation loss; the weights of the best epoch are kept.

        Args:
            s_trn, t_trn: (N, n_x) summaries and (N, n_theta) parameters of the training set.
            s_val, t_val: the same for the validation set.
            train: :class:`TrainConfig` or dict.
            seed: torch seed for the batch order (the weights are initialised at construction).

        Returns:
            (best validation loss, number of epochs run); the loss is inf if no epoch gave a
            finite one.
        """
        tr = TrainConfig.make(train)
        self.train_config = tr
        if seed is not None:
            torch.manual_seed(seed)
        dev = self.device
        self.s_scaler = ZScore().fit(np.asarray(s_trn, dtype=np.float64))
        self.t_scaler = ZScore().fit(np.asarray(t_trn, dtype=np.float64))
        x, y = self._scale(_t(s_trn, dev), _t(t_trn, dev))
        xv, yv = self._scale(_t(s_val, dev), _t(t_val, dev))
        opt = torch.optim.Adam(self.flow.parameters(), lr=tr.lr, weight_decay=tr.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tr.sched_factor, patience=tr.sched_patience)
        bs = min(tr.batch_size, len(x))
        best, best_state, wait, epoch = np.inf, None, 0, 0
        self.history = []
        for epoch in range(1, tr.max_epochs + 1):
            self.flow.train()
            perm = torch.randperm(len(x), device=dev)
            for i in range(0, len(x), bs):
                idx = perm[i:i + bs]
                loss = self.flow.loss(x[idx], condition=y[idx]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if tr.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.flow.parameters(), tr.grad_clip)
                opt.step()
            self.flow.eval()
            with torch.no_grad():
                val = torch.cat([self.flow.loss(xv[i:i + 8192], condition=yv[i:i + 8192])
                                 for i in range(0, len(xv), 8192)]).mean().item()
            self.history.append(val)
            sched.step(val)
            if np.isfinite(val) and val < best - tr.min_delta:
                best, wait = val, 0
                best_state = {k: v.detach().clone() for k, v in self.flow.state_dict().items()}
            else:
                wait += 1
                if wait >= tr.stop_after_epochs:
                    break
        if best_state is None:
            return float("inf"), epoch
        self.flow.load_state_dict(best_state)
        self.flow.eval()
        return float(best), epoch

    # ------------------------------------------------------------------ evaluation
    def _scale(self, s, t):
        return self.s_scaler(s), self.t_scaler(t)

    def log_prob(self, input, condition):
        """log p(input | condition) for tensors in sbi's layout: ``input`` (1, M, n_x) or (M, n_x),
        ``condition`` (M, n_theta), both in raw units and on the flow's device. Gradients are
        tracked; use :meth:`log_likelihood` for plain evaluation."""
        x, y = self._scale(input, condition)
        return self.flow.log_prob(x, condition=y)

    def log_likelihood(self, s, theta):
        """log p(s_i | theta_i) for matching rows, (M,) tensor."""
        s, theta = _t(s, self.device), _t(theta, self.device)
        with torch.no_grad():
            return self.log_prob(s.unsqueeze(0), theta).reshape(-1)

    def val_loss(self, s, t) -> float:
        """Mean -log p(s_i | t_i) over the rows, as minimised in training."""
        with torch.no_grad():
            return -self.log_likelihood(s, t).mean().item()

    def to(self, device):
        """Move the flow to ``device``; returns itself."""
        self.flow.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------ io
    def state(self):
        """Plain serialisable state (config, weights, scalers) for :func:`sbikit.io.save`."""
        return {"kind": "nle", "n_x": self.n_x, "n_theta": self.n_theta, "flow_config": asdict(self.flow_config),
                "flow": {k: v.cpu() for k, v in self.flow.state_dict().items()},
                "s_scaler": self.s_scaler.state(), "t_scaler": self.t_scaler.state()}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a trained NLE from :meth:`state`."""
        nle = cls(state["n_x"], state["n_theta"], state["flow_config"], device)
        nle.flow.load_state_dict(state["flow"])
        nle.flow.eval()
        nle.s_scaler = scaler_from_state(state["s_scaler"])
        nle.t_scaler = scaler_from_state(state["t_scaler"])
        return nle


class Ensemble:
    """Likelihood ensemble: log p(s | theta) = log mean_k p_k(s | theta).

    Args:
        members: trained :class:`NLE` objects with the same summaries and parameters. It can be
            used wherever an NLE is (sampling, saving).
    """

    def __init__(self, members):
        self.members = list(members)

    @property
    def device(self):
        """Device of the first member."""
        return self.members[0].device

    def log_prob(self, input, condition):
        """Log of the mean member likelihood; same layout as :meth:`NLE.log_prob`."""
        lp = torch.stack([m.log_prob(input, condition) for m in self.members])
        return torch.logsumexp(lp, 0) - np.log(len(self.members))

    def to(self, device):
        """Move every member to ``device``; returns itself."""
        for m in self.members:
            m.to(device)
        return self

    def state(self):
        """Plain serialisable state for :func:`sbikit.io.save`."""
        return {"kind": "ensemble", "members": [m.state() for m in self.members]}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild an ensemble from :meth:`state`."""
        return cls([NLE.from_state(s, device) for s in state["members"]])
