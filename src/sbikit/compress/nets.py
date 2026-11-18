"""Neural compressors trained on (x, theta) pairs: MSE regression and VMIM.

Both use a LeakyReLU MLP trunk plus a linear skip connection, AdamW with a one-cycle schedule,
gradient clipping and early stopping on the validation loss. Inputs are scaled inside the
compressor (median/MAD + asinh by default), targets are z-scored.

Atoms in the design (``fit(..., atoms=[(column, value, half_width), ...])``): training labels that
sit exactly on a point mass of the parameter distribution (e.g. LCDM simulations at w0 = -1 among
wCDM ones) are spread uniformly over value +- half_width, redrawn every batch. Without that, the MDN
of a VMIM can put a near-delta component on the atom, which is worth several nats per atom row and
grows with training length, so the loss measures that spike rather than information. Keep atom
rows out of the validation set as well (that is a choice of splits, made by the caller).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scaling import ZScore, make_scaler, scaler_from_state
from .base import Compressor, register

__all__ = ["Net", "MDN", "NetConfig", "NetCompressor", "MSECompressor", "VMIMCompressor"]


class Net(nn.Module):
    """LeakyReLU trunk plus a linear skip connection from the input to the output.

    Args:
        n_in, n_out: input and output sizes.
        widths: hidden layer widths.
        dropout: dropout probability after each hidden layer (0: none).
    """

    def __init__(self, n_in, n_out, widths=(256, 128, 64), dropout=0.0):
        super().__init__()
        layers, d = [], n_in
        for w in widths:
            layers += [nn.Linear(d, w), nn.LeakyReLU(0.1)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = w
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)
        self.skip = nn.Linear(n_in, n_out)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.skip.bias)

    def forward(self, x):
        """(N, n_in) -> (N, n_out)."""
        return self.head(self.trunk(x)) + self.skip(x)


class MDN(nn.Module):
    """Mixture density network: a mixture of diagonal Gaussians q(theta | s).

    Args:
        n_in: size of the summaries s.
        n_out: number of parameters theta.
        hidden: width of the two hidden layers.
        components: number of mixture components.
    """

    def __init__(self, n_in, n_out, hidden=64, components=6):
        super().__init__()
        self.n_out, self.k = n_out, components
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.LeakyReLU(0.1),
                                  nn.Linear(hidden, hidden), nn.LeakyReLU(0.1))
        self.logits = nn.Linear(hidden, components)
        self.means = nn.Linear(hidden, components * n_out)
        self.logsig = nn.Linear(hidden, components * n_out)

    def nll(self, s, theta):
        """Mean -log q(theta | s) over the batch (log sigma is clamped to [-7, 7])."""
        h = self.body(s)
        log_w = torch.log_softmax(self.logits(h), dim=1)
        means = self.means(h).view(-1, self.k, self.n_out)
        logsig = self.logsig(h).view(-1, self.k, self.n_out).clamp(-7.0, 7.0)
        log_comp = -0.5 * (((theta[:, None] - means) / logsig.exp()) ** 2
                           + 2 * logsig + np.log(2 * np.pi)).sum(-1)
        return -torch.logsumexp(log_w + log_comp, dim=1).mean()


@dataclass
class NetConfig:
    """Network and training settings of the MSE and VMIM compressors.

    Attributes:
        widths: hidden layer widths of the compression network.
        dropout: dropout probability.
        lr: maximum learning rate of the one-cycle schedule.
        weight_decay: AdamW weight decay.
        batch_size: minibatch size.
        epochs: length of the one-cycle schedule (the upper limit with early stopping).
        patience: early-stopping patience in epochs.
        grad_clip: maximum gradient norm.
        pct_start: fraction of the schedule spent increasing the learning rate.
        input_scaling: ``"robust_asinh"``, ``"zscore"`` or ``"identity"``
            (:mod:`sbikit.scaling`).
        asinh_k: the ``k`` of ``"robust_asinh"``.
        mdn_hidden: hidden width of the VMIM's mixture density network.
        components: mixture components of the VMIM's mixture density network.
        vicreg: weight of the VICReg variance and covariance terms in the VMIM training loss
            (0: off).
        vicreg_var, vicreg_cov: relative weights of the two terms.
    """

    widths: tuple = (256, 128, 64)
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 300              # one-cycle schedule length (upper limit with early stopping)
    patience: int = 30
    grad_clip: float = 1.0
    pct_start: float = 0.1
    input_scaling: str = "robust_asinh"   # robust_asinh | zscore | identity
    asinh_k: float = 5.0
    # VMIM only
    mdn_hidden: int = 64
    components: int = 6
    vicreg: float = 0.0            # weight of the VICReg terms (0: off)
    vicreg_var: float = 25.0
    vicreg_cov: float = 1.0

    @classmethod
    def make(cls, cfg) -> "NetConfig":
        """A config from None (defaults), an instance (returned as is) or a dict."""
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg
        cfg = dict(cfg)
        if "widths" in cfg:
            cfg["widths"] = tuple(cfg["widths"])
        return cls(**cfg)


class NetCompressor(Compressor):
    """Base class of the network compressors trained on (x, theta) pairs.

    Subclasses define the loss (and optional extra heads that are only needed in training).

    Args:
        n_out: number of summaries.
        config: :class:`NetConfig` or dict.
        device: torch device for training and evaluation.
        seed: torch seed set at the start of :meth:`fit` (None: leave the global state).

    Attributes:
        net: the compression network after :meth:`fit`.
        history: validation loss per epoch.
    """
    kind = "net"

    def __init__(self, n_out: int, config=None, device="cpu", seed: Optional[int] = None):
        self.n_out, self.config, self.device, self.seed = int(n_out), NetConfig.make(config), device, seed
        self.net: Optional[Net] = None
        self.n_in: Optional[int] = None
        self.history: list = []

    # -- subclass hooks
    def build_heads(self, n_theta):
        """Create extra modules trained with the network (e.g. the VMIM's density head) and
        return them as a list."""
        return []

    def drop_heads(self):
        """Discard the extra modules after training."""
        pass

    def loss(self, s, t):
        """Training loss for summaries ``s`` and z-scored parameters ``t`` (tensors)."""
        raise NotImplementedError

    def val_loss(self, s, t):
        """Validation loss as a float (default: the training loss)."""
        return self.loss(s, t).item()

    # --
    def _scale_in(self, x):
        return torch.as_tensor(np.asarray(self.in_scaler(np.asarray(x, dtype=np.float64))), dtype=torch.float32)

    def fit(self, x_trn, t_trn, x_val, t_val, atoms: Optional[Sequence] = None) -> float:
        """Train with early stopping on the validation set; the best weights are kept.

        Args:
            x_trn, t_trn: (N, D) data vectors and (N, P) parameters of the training set.
            x_val, t_val: the same for the validation set.
            atoms: (column, value, half_width) per point mass of the training labels, in raw
                units (see the module docstring).

        Returns:
            The best validation loss (inf if no epoch gave a finite one).
        """
        c = self.config
        if self.seed is not None:
            torch.manual_seed(self.seed)
        kw = {"k": c.asinh_k} if c.input_scaling == "robust_asinh" else {}
        self.in_scaler = make_scaler(c.input_scaling, **kw).fit(np.asarray(x_trn, dtype=np.float64))
        self.t_scaler = ZScore().fit(np.asarray(t_trn, dtype=np.float64))
        dev = self.device
        xt, xv = self._scale_in(x_trn).to(dev), self._scale_in(x_val).to(dev)
        tt = torch.as_tensor(np.asarray(self.t_scaler(np.asarray(t_trn, np.float64))), dtype=torch.float32, device=dev)
        tv = torch.as_tensor(np.asarray(self.t_scaler(np.asarray(t_val, np.float64))), dtype=torch.float32, device=dev)
        jitter = []                      # (mask over training rows, column, half width in z-units)
        for col, value, half in (atoms or []):
            m = torch.as_tensor(np.asarray(t_trn)[:, col] == value, device=dev)
            if m.any():
                jitter.append((m, int(col), float(half / self.t_scaler.std[col])))
        self.atoms = [[int(a), float(v), float(h)] for a, v, h in (atoms or [])]
        self.n_in = xt.shape[1]
        self.net = Net(self.n_in, self.n_out, c.widths, c.dropout).to(dev)
        extra = self.build_heads(tt.shape[1])
        params = list(self.net.parameters()) + [p for m in extra for p in m.parameters()]
        bs = min(c.batch_size, len(xt))
        steps = -(-len(xt) // bs)
        opt = torch.optim.AdamW(params, lr=c.lr, weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c.lr, epochs=c.epochs, steps_per_epoch=steps,
                                                    pct_start=c.pct_start, anneal_strategy="cos")
        modules = [self.net, *extra]
        best, best_state, wait = np.inf, None, 0
        self.history = []
        for _ in range(c.epochs):
            for m in modules:
                m.train()
            perm = torch.randperm(len(xt), device=dev)
            for i in range(steps):
                idx = perm[i * bs:(i + 1) * bs]
                tb = tt[idx]
                if jitter:
                    tb = tb.clone()
                    for m, col, half in jitter:
                        mb = m[idx]
                        if mb.any():
                            tb[mb, col] += (2 * torch.rand(int(mb.sum()), device=dev) - 1) * half
                loss = self.loss(self.net(xt[idx]), tb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, c.grad_clip)
                opt.step()
                sched.step()
            for m in modules:
                m.eval()
            with torch.no_grad():
                val = self.val_loss(self.net(xv), tv)
            self.history.append(val)
            if np.isfinite(val) and val < best - 1e-7:
                best, wait = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= c.patience:
                    break
        if best_state is None:
            return float("inf")
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.drop_heads()
        return float(best)

    def __call__(self, x):
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.eval()
        x = np.asarray(x)
        out = []
        with torch.no_grad():
            for b in np.array_split(x, max(1, len(x) // 4096)):
                out.append(self.net(self._scale_in(b).to(self.device)).cpu())
        return torch.cat(out).numpy()

    def to(self, device):
        """Move the network to ``device``; returns itself."""
        self.device = device
        if self.net is not None:
            self.net.to(device)
        return self

    def state(self):
        """Plain serialisable state (config, weights, scalers)."""
        return {"kind": self.kind, "n_out": self.n_out, "n_in": self.n_in, "config": asdict(self.config),
                "net": {k: v.cpu() for k, v in self.net.state_dict().items()},
                "in_scaler": self.in_scaler.state(), "t_scaler": self.t_scaler.state()}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted compressor from :meth:`state`."""
        comp = cls(state["n_out"], state["config"], device)
        comp.n_in = state["n_in"]
        comp.net = Net(comp.n_in, comp.n_out, comp.config.widths, comp.config.dropout)
        comp.net.load_state_dict(state["net"])
        comp.net.to(device).eval()
        comp.in_scaler = scaler_from_state(state["in_scaler"])
        comp.t_scaler = scaler_from_state(state["t_scaler"])
        return comp


@register
class MSECompressor(NetCompressor):
    """Regression onto the z-scored parameters with a mean-squared-error loss, so the summaries
    estimate the posterior mean. ``n_out`` must be the number of parameters. Arguments as
    :class:`NetCompressor`."""
    kind = "mse"

    def fit(self, x_trn, t_trn, x_val, t_val, atoms=None):
        """See :meth:`NetCompressor.fit`."""
        if np.shape(t_trn)[1] != self.n_out:
            raise ValueError(f"MSE compression needs n_out = {np.shape(t_trn)[1]} (the number of parameters)")
        return super().fit(x_trn, t_trn, x_val, t_val, atoms)

    def loss(self, s, t):
        """Mean squared error."""
        return F.mse_loss(s, t)


@register
class VMIMCompressor(NetCompressor):
    """Variational mutual information maximisation (Jeffrey et al. 2021): the net and a diagonal MDN
    q(theta | s) are trained jointly on -log q. With ``config.vicreg > 0`` VICReg's variance and
    covariance terms on the batch summaries are added during training (no invariance term); the
    validation loss stays -log q so trials with different weights remain comparable. Arguments as
    :class:`NetCompressor`; ``n_out`` is free (the number of parameters is a common choice).

    Example:
        >>> vmim = VMIMCompressor(n_out=3, config={"widths": (128, 64), "epochs": 100}, device="cuda")
        >>> vmim.fit(x_train, theta_train, x_val, theta_val)
        >>> s = vmim(x)                                   # (N, 3)
    """
    kind = "vmim"

    def build_heads(self, n_theta):
        """The mixture density network q(theta | s)."""
        c = self.config
        self.mdn = MDN(self.n_out, n_theta, c.mdn_hidden, c.components).to(self.device)
        return [self.mdn]

    def drop_heads(self):
        """Discard the mixture density network (only the compression network is kept)."""
        del self.mdn

    def val_loss(self, s, t):
        """-log q(t | s), without the VICReg terms."""
        return self.mdn.nll(s, t).item()

    def loss(self, s, t):
        """-log q(t | s), plus the VICReg terms in training if ``config.vicreg > 0``."""
        loss = self.mdn.nll(s, t)
        c = self.config
        if c.vicreg > 0 and self.net.training:
            std = torch.sqrt(s.var(0) + 1e-4)
            var = F.relu(1.0 - std).mean()
            z = s - s.mean(0)
            cov = (z.T @ z) / (len(s) - 1)
            off = cov - torch.diag(torch.diagonal(cov))
            loss = loss + c.vicreg * (c.vicreg_var * var + c.vicreg_cov * (off ** 2).sum() / s.shape[1])
        return loss
