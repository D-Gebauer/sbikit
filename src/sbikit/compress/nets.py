"""Neural compressors trained on (x, theta) pairs: MSE regression and VMIM.

By default both use a LeakyReLU MLP trunk plus a linear skip connection (:class:`Net`), AdamW with a
one-cycle schedule, gradient clipping and early stopping on the validation loss. Inputs are scaled
inside the compressor (median/MAD + asinh by default), targets are z-scored.

Customisation, from light to heavy: the fields of :class:`NetConfig`; another architecture, registered
with :func:`sbikit.compress.register_network` and selected with ``NetConfig.network`` (it is then
saved and loaded like the built-in one) or passed directly as ``net=`` (not saveable); another loss,
by subclassing :class:`NetCompressor` and overriding its hooks.

Atoms in the design (``fit(..., atoms=[(column, value, half_width), ...])``): training labels that
sit exactly on a point mass of the parameter distribution (e.g. LCDM simulations at w0 = -1 among
wCDM ones) are spread uniformly over value +- half_width, redrawn every batch. Without that, the MDN
of a VMIM can put a near-delta component on the atom, which is worth several nats per atom row and
grows with training length, so the loss measures that spike rather than information. Keep atom
rows out of the validation set as well (that is a choice of splits, made by the caller).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .._utils import (ConfigBase, as_2d, check_pair, did_you_mean, holdout_split, logger, parse_widths, plain,
                      resolve_device, seeded)
from ..scaling import SCALERS, ZScore, make_scaler, scaler_from_state
from .base import NETWORKS, Compressor, register

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


def _mlp(n_in, n_out, config):
    """The built-in network ``"mlp"``: :class:`Net` with the config's ``widths`` and ``dropout``."""
    return Net(n_in, n_out, config.widths, config.dropout)


NETWORKS.register("mlp", _mlp, overwrite=True)


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
class NetConfig(ConfigBase):
    """Network and training settings of the MSE and VMIM compressors.

    ``NetConfig.make`` (and every ``config=`` argument) takes None, an instance or a dict; a
    misspelled key raises ``TypeError``, an invalid value ``ValueError``.

    Attributes:
        widths: hidden layer widths of the compression network; a sequence, an int or a string
            such as ``"256,128,64"``.
        dropout: dropout probability, in [0, 1).
        lr: learning rate (the maximum of the one-cycle schedule).
        weight_decay: weight decay of the optimizer.
        batch_size: minibatch size. A last minibatch of a single row is left out of its epoch.
        epochs: maximum number of epochs (the length of the one-cycle schedule).
        patience: early stopping after this many epochs without improvement.
        grad_clip: maximum gradient norm; None or 0: no clipping.
        pct_start: fraction of the one-cycle schedule spent increasing the learning rate.
        input_scaling: name of the input scaler: ``"robust_asinh"``, ``"zscore"``, ``"identity"``
            or one added with :func:`sbikit.scaling.register_scaler`.
        asinh_k: the ``k`` of ``"robust_asinh"``.
        mdn_hidden: hidden width of the VMIM's mixture density network.
        components: mixture components of the VMIM's mixture density network.
        vicreg: weight of the VICReg variance and covariance terms in the VMIM training loss
            (0: off).
        vicreg_var, vicreg_cov: relative weights of the two terms.
        val_fraction: fraction of the simulations held out for validation when ``fit`` gets no
            validation set.
        min_delta: smallest decrease of the validation loss that counts as an improvement.
        optimizer: ``"adamw"`` or ``"adam"``.
        schedule: ``"onecycle"`` (warm-up, then cosine decay over ``epochs``) or ``"constant"``.
            A one-cycle schedule of fewer than 3 steps runs at constant ``lr``.
        network: name of the architecture: ``"mlp"`` (:class:`Net`) or one added with
            :func:`sbikit.compress.register_network`.
        network_kwargs: options of a registered architecture (its factory reads them from the
            config); unused by ``"mlp"``.
    """

    widths: tuple = (256, 128, 64)
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 300              # one-cycle schedule length (upper limit with early stopping)
    patience: int = 30
    grad_clip: Optional[float] = 1.0      # None or 0: off
    pct_start: float = 0.1
    input_scaling: str = "robust_asinh"   # a name in sbikit.scaling.SCALERS
    asinh_k: float = 5.0
    # VMIM only
    mdn_hidden: int = 64
    components: int = 6
    vicreg: float = 0.0            # weight of the VICReg terms (0: off)
    vicreg_var: float = 25.0
    vicreg_cov: float = 1.0
    # training
    val_fraction: float = 0.1
    min_delta: float = 1e-7
    optimizer: str = "adamw"       # adamw | adam
    schedule: str = "onecycle"     # onecycle | constant
    network: str = "mlp"           # a name in sbikit.compress.NETWORKS
    network_kwargs: dict = field(default_factory=dict)

    _choices = {"optimizer": ("adamw", "adam"), "schedule": ("onecycle", "constant")}
    _ints = ("batch_size", "epochs", "patience", "mdn_hidden", "components")
    _floats = ("dropout", "lr", "weight_decay", "pct_start", "asinh_k", "vicreg", "vicreg_var", "vicreg_cov",
               "val_fraction", "min_delta")

    def _normalise(self) -> None:
        name = type(self).__name__
        try:
            widths = self.widths
            if not isinstance(widths, str):
                widths = [widths] if np.isscalar(widths) else list(widths)
                if any(not isinstance(w, str) and int(w) != w for w in widths):
                    raise ValueError
            self.widths = parse_widths(widths)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name}.widths must be positive integers, e.g. (256, 128) or '256,128', got "
                             f"{self.widths!r}") from None
        for key in self._ints:
            value = getattr(self, key)
            try:
                if int(value) != value:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name}.{key} must be an integer, got {value!r}") from None
            setattr(self, key, int(value))
        for key in self._floats + (("grad_clip",) if self.grad_clip is not None else ()):
            value = getattr(self, key)
            try:
                setattr(self, key, float(value))
            except (TypeError, ValueError):
                raise ValueError(f"{name}.{key} must be a number, got {value!r}") from None
        if not isinstance(self.network_kwargs, Mapping):
            raise ValueError(f"{name}.network_kwargs must be a dict, got {type(self.network_kwargs).__name__}")
        self.network_kwargs = dict(self.network_kwargs)

    def validate(self) -> None:
        r = self._require
        r(0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}")
        r(np.isfinite(self.lr) and self.lr > 0, f"lr must be positive, got {self.lr}")
        r(np.isfinite(self.weight_decay) and self.weight_decay >= 0, f"weight_decay must be >= 0, got "
          f"{self.weight_decay}")
        for key in self._ints:
            r(getattr(self, key) >= 1, f"{key} must be at least 1, got {getattr(self, key)}")
        r(self.grad_clip is None or self.grad_clip >= 0, f"grad_clip must be positive, or None / 0 for no "
          f"clipping; got {self.grad_clip}")
        r(0.0 < self.pct_start < 1.0, f"pct_start must be in (0, 1), got {self.pct_start}")
        r(np.isfinite(self.asinh_k) and self.asinh_k > 0, f"asinh_k must be positive, got {self.asinh_k}")
        for key in ("vicreg", "vicreg_var", "vicreg_cov", "min_delta"):
            r(np.isfinite(getattr(self, key)) and getattr(self, key) >= 0, f"{key} must be >= 0, got "
              f"{getattr(self, key)}")
        r(0.0 < self.val_fraction < 1.0, f"val_fraction must be in (0, 1), got {self.val_fraction}")
        for key, registry, how in (("input_scaling", SCALERS, "sbikit.scaling.register_scaler"),
                                   ("network", NETWORKS, "sbikit.compress.register_network")):
            value = getattr(self, key)
            if not isinstance(value, str) or value not in registry:
                hint = did_you_mean(value, list(registry)) if isinstance(value, str) else ""
                r(False, f"{key} = {value!r}{hint}; registered: {sorted(registry)} (add one with {how})")


def _check_atoms(atoms, n_theta: int) -> list:
    """``atoms`` as a list of [column, value, half_width], validated against ``n_theta`` columns."""
    out = []
    for atom in ([] if atoms is None else atoms):
        try:
            col, value, half = atom
            ok = int(col) == col and np.isfinite(float(value)) and np.isfinite(float(half))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"atoms must be a list of (column, value, half_width) triples, e.g. [(1, -1.0, 0.1)]; "
                             f"got the entry {atom!r}")
        if not 0 <= int(col) < n_theta:
            raise ValueError(f"atoms: column {int(col)} is out of range for theta with {n_theta} columns")
        if not float(half) > 0:
            raise ValueError(f"atoms: half_width must be positive, got {half!r} for column {int(col)}")
        out.append([int(col), float(value), float(half)])
    return out


class NetCompressor(Compressor):
    """Base class of the network compressors trained on (x, theta) pairs.

    Subclasses define the loss (and optional extra heads that are only needed in training) by
    overriding the hooks :meth:`loss`, :meth:`val_loss`, :meth:`build_heads`, :meth:`drop_heads` and
    :meth:`build_net`. A subclass with settings of its own sets ``config_class`` to a dataclass
    derived from :class:`NetConfig`.

    Args:
        n_out: number of summaries.
        config: :class:`NetConfig` or dict.
        device: torch device for training and evaluation (``"auto"``: a GPU if there is one).
        seed: seed of the initialisation, the minibatches and the automatic validation split.
            The global random state is restored after :meth:`fit`. None: use the caller's global
            state.
        net: expert shortcut, a factory ``(n_in, n_out) -> torch.nn.Module`` used instead of
            ``config.network``. Such a compressor trains and compresses, but cannot be saved; for
            that, add the architecture with :func:`sbikit.compress.register_network` and select
            it with ``config.network``.

    Attributes:
        net: the compression network after :meth:`fit` (None before, and after a failed fit).
        n_in: length of the data vectors it was fitted on.
        history: validation loss per epoch.
        train_history: mean training loss per epoch.
        best_val_loss: the validation loss of the kept weights (None before :meth:`fit`).
        atoms: the atoms of the last :meth:`fit`, as [column, value, half_width] lists.
    """
    kind = "net"
    config_class = NetConfig

    def __init__(self, n_out: int, config=None, device="cpu", seed: Optional[int] = None,
                 net: Optional[Callable] = None):
        if int(n_out) != n_out or n_out < 1:
            raise ValueError(f"n_out must be a positive integer, got {n_out!r}")
        if net is not None and (isinstance(net, nn.Module) or not callable(net)):
            raise TypeError("net must be a factory (n_in, n_out) -> torch.nn.Module, e.g. "
                            "net=lambda n_in, n_out: MyNet(n_in, n_out), not a module instance")
        self.n_out, self.config = int(n_out), self.config_class.make(config)
        self.device, self.seed = resolve_device(device), seed
        self._net_factory = net
        self.net: Optional[nn.Module] = None
        self.n_in: Optional[int] = None
        self.history: list = []
        self.train_history: list = []
        self.best_val_loss: Optional[float] = None
        self.atoms: list = []

    # -- subclass hooks
    def build_net(self, n_in: int) -> nn.Module:
        """The compression network for data vectors of length ``n_in``: the ``net`` factory if one
        was given, else the registered architecture ``config.network``."""
        if self._net_factory is not None:
            net = self._net_factory(n_in, self.n_out)
        else:
            net = NETWORKS[self.config.network](n_in, self.n_out, self.config)
        if not isinstance(net, nn.Module):
            raise TypeError(f"the network factory must return a torch.nn.Module, got {type(net).__name__}")
        return net

    def build_heads(self, n_theta):
        """Create extra modules trained with the network (e.g. the VMIM's density head) and
        return them as a list."""
        return []

    def drop_heads(self):
        """Discard the extra modules after training."""
        pass

    def loss(self, s, t):
        """Training loss (a scalar tensor) of one minibatch.

        Args:
            s: (B, n_out) tensor, the network output for the batch.
            t: (B, P) tensor of z-scored parameters.
        """
        raise NotImplementedError

    def val_loss(self, s, t):
        """Validation loss as a float, for the tensors ``s`` (network output) and ``t`` (z-scored
        parameters) of the whole validation set (default: the training loss). A training hook;
        it does not take raw arrays."""
        return self.loss(s, t).item()

    # --
    def _scale_in(self, x):
        return torch.as_tensor(np.asarray(self.in_scaler(np.asarray(x, dtype=np.float64))), dtype=torch.float32)

    def _schedule(self, opt, total_steps: int):
        """The learning-rate scheduler (stepped once per minibatch), or None for a constant rate."""
        c = self.config
        if c.schedule != "onecycle" or total_steps < 3:
            return None
        pct = max(c.pct_start, 2.0 / total_steps)            # a warm-up of at least one step
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c.lr, total_steps=total_steps, pct_start=pct,
                                                   anneal_strategy="cos")

    def _finish(self, best_state, heads: bool = True) -> bool:
        """Drop the training heads (if they were built) and load the best weights; without any,
        mark the compressor as unfitted."""
        if heads:
            self.drop_heads()
        if best_state is None:
            self.net = None
            return False
        self.net.load_state_dict(best_state)
        self.net.eval()
        return True

    def fit(self, x, theta, x_val=None, theta_val=None, atoms: Optional[Sequence] = None, *,
            callback: Optional[Callable] = None, verbose: bool = False) -> float:
        """Train with early stopping on the validation set; the best weights are kept.

        Args:
            x, theta: (N, D) data vectors and (N, P) parameters of the training set (array-likes
                or tensors).
            x_val, theta_val: the same for the validation set. Without one, a random
                ``config.val_fraction`` of the given simulations is held out (drawn with ``seed``).
            atoms: (column, value, half_width) per point mass of the training labels, in raw
                units (see the module docstring). Rows are matched to single precision; an atom
                that matches no training row gives a ``UserWarning``. The automatic hold-out does
                not treat atom rows specially.
            callback: called after every epoch with a dict ``{"epoch" (counted from 0),
                "train_loss", "val_loss", "best" (best validation loss so far), "model" (this
                compressor)}``; if it returns True, training stops and the best weights are kept.
            verbose: log one line per epoch.

        Returns:
            The best validation loss. ``inf`` if no epoch gave a finite one: a ``RuntimeWarning``
            is emitted and the compressor is left unfitted.

        Raises:
            ValueError: for inputs of the wrong shape, different numbers of rows, NaN or inf, or
                invalid ``atoms``.
        """
        c, name = self.config, type(self).__name__
        x, theta = check_pair(x, theta)
        if (x_val is None) != (theta_val is None):
            raise ValueError("x_val and theta_val must be given together (or both left out)")
        if x_val is not None:
            x_val, theta_val = check_pair(x_val, theta_val, "x_val", "theta_val", n_x=x.shape[1],
                                          n_theta=theta.shape[1])
        else:
            trn, val = holdout_split(len(x), c.val_fraction, self.seed)
            logger.info(f"{name}: no validation set given, holding out {len(val)} of {len(x)} simulations "
                        f"(val_fraction = {c.val_fraction})")
            x, theta, x_val, theta_val = x[trn], theta[trn], x[val], theta[val]
        atoms = _check_atoms(atoms, theta.shape[1])
        span = np.ptp(theta, axis=0)
        constant = np.flatnonzero(span <= 1e-12 * np.maximum(1.0, np.abs(theta.mean(0))))
        if len(constant):
            warnings.warn(f"{name}: theta column(s) {constant.tolist()} are constant in the training set. They "
                          "carry no information, and a density head can fit them arbitrarily well, which makes "
                          "the loss meaningless; drop them from the labels.", UserWarning, stacklevel=2)
        with seeded(self.seed):
            return self._fit(x, theta, x_val, theta_val, atoms, callback, verbose)

    def _fit(self, x, theta, x_val, theta_val, atoms, callback, verbose) -> float:
        """The training loop of :meth:`fit`, on validated float64 arrays."""
        c, name, dev = self.config, type(self).__name__, self.device
        kw = {"k": c.asinh_k} if c.input_scaling == "robust_asinh" else {}
        self.in_scaler = make_scaler(c.input_scaling, **kw).fit(x)
        self.t_scaler = ZScore().fit(theta)
        xt, xv = self._scale_in(x).to(dev), self._scale_in(x_val).to(dev)
        tt = torch.as_tensor(np.asarray(self.t_scaler(theta)), dtype=torch.float32, device=dev)
        tv = torch.as_tensor(np.asarray(self.t_scaler(theta_val)), dtype=torch.float32, device=dev)
        jitter = []                      # (mask over training rows, column, half width in z-units)
        for col, value, half in atoms:
            mask = theta[:, col].astype(np.float32) == np.float32(value)      # equal to single precision
            if mask.any():
                jitter.append((torch.as_tensor(mask, device=dev), col, float(half / self.t_scaler.std[col])))
            else:
                nearest = float(theta[np.argmin(np.abs(theta[:, col] - value)), col])
                warnings.warn(f"{name}: atoms: no training row has theta[:, {col}] == {value!r} (the closest label "
                              f"is {nearest!r}); this atom is ignored.", UserWarning, stacklevel=3)
        self.atoms = atoms
        self.history, self.train_history, self.best_val_loss = [], [], float("inf")
        self.n_in = xt.shape[1]
        self.net = None
        net = self.build_net(self.n_in).to(dev).eval()
        with torch.no_grad():
            shape = tuple(net(xt[:2]).shape)
        if shape != (len(xt[:2]), self.n_out):
            raise ValueError(f"{name}: the network maps {tuple(xt[:2].shape)} to {shape}, expected "
                             f"({len(xt[:2])}, {self.n_out}) = (rows, n_out)")
        n = len(xt)
        bs = min(c.batch_size, n)
        steps = -(-n // bs)
        if steps > 1 and n % bs == 1:    # a last minibatch of one row has no batch statistics: leave it out
            steps -= 1
        self.net, extra, best, best_state, wait = net, None, np.inf, None, 0
        try:
            extra = self.build_heads(tt.shape[1])
            modules = [self.net, *extra]
            params = [p for m in modules for p in m.parameters()]
            opt = (torch.optim.Adam if c.optimizer == "adam" else torch.optim.AdamW)(
                params, lr=c.lr, weight_decay=c.weight_decay)
            sched = self._schedule(opt, c.epochs * steps)
            for epoch in range(c.epochs):
                for m in modules:
                    m.train()
                perm = torch.randperm(n, device=dev)
                total = torch.zeros((), device=dev)
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
                    if c.grad_clip:
                        nn.utils.clip_grad_norm_(params, c.grad_clip)
                    opt.step()
                    if sched is not None:
                        sched.step()
                    total += loss.detach()
                train = float(total) / steps
                for m in modules:
                    m.eval()
                with torch.no_grad():
                    val = float(self.val_loss(self.net(xv), tv))
                self.history.append(val)
                self.train_history.append(train)
                if np.isfinite(val) and val < best - c.min_delta:
                    best, wait = val, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    wait += 1
                if verbose:
                    logger.info(f"{name} epoch {epoch + 1}/{c.epochs}: train {train:.5f}, val {val:.5f}, "
                                f"best {best:.5f}")
                stop = callback is not None and bool(callback({"epoch": epoch, "train_loss": train, "val_loss": val,
                                                               "best": float(best), "model": self}))
                if not np.isfinite(train):
                    if best_state is not None:
                        warnings.warn(f"{name}: the training loss became non-finite in epoch {epoch + 1}; stopped "
                                      f"and kept the best weights (validation loss {best:.5g}).", RuntimeWarning,
                                      stacklevel=3)
                    break
                if stop or wait >= c.patience:
                    break
        except BaseException:            # also KeyboardInterrupt: keep the best weights so far
            self._finish(best_state, heads=extra is not None)
            self.best_val_loss = float(best)
            raise
        self.best_val_loss = float(best)
        if not self._finish(best_state):
            warnings.warn(f"{name}: no epoch gave a finite validation loss, the compressor is left unfitted. "
                          "Lower lr, set grad_clip, or check the scale of the inputs.", RuntimeWarning, stacklevel=3)
        return float(best)

    def __call__(self, x):
        """Summaries (N, n_out), float32, of the data vectors ``x`` (N, D); a single data vector
        (D,) gives (n_out,).

        Raises:
            NotFittedError: before :meth:`fit`.
            ValueError: if ``x`` does not have the fitted number of columns.
        """
        if self.net is None:
            raise self._not_fitted()
        x, single = self._batch(x, self.n_in)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, max(len(x), 1), 4096):
                out.append(self.net(self._scale_in(x[i:i + 4096]).to(self.device)).float().cpu())
        s = torch.cat(out).numpy()
        return s[0] if single else s

    def to(self, device):
        """Move the network to ``device``; returns itself."""
        self.device = resolve_device(device)
        if self.net is not None:
            self.net.to(self.device)
        return self

    def state(self):
        """Plain serialisable state (config, weights, scalers, training record).

        Raises:
            NotFittedError: before :meth:`fit`.
            TypeError: if the network came from a ``net=`` factory, which cannot be rebuilt.
        """
        if self.net is None:
            raise self._not_fitted()
        if self._net_factory is not None:
            raise TypeError(f"{type(self).__name__}: a network given as net= cannot be saved. Register the "
                            "architecture with sbikit.compress.register_network(name, factory) and select it "
                            "with config={'network': name} to make the compressor saveable.")
        return {"kind": self.kind, "n_out": self.n_out, "n_in": self.n_in, "config": plain(self.config.to_dict()),
                "net": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
                "in_scaler": plain(self.in_scaler.state()), "t_scaler": plain(self.t_scaler.state()),
                "atoms": [list(a) for a in self.atoms], "seed": plain(self.seed), "history": list(self.history),
                "train_history": list(self.train_history), "best_val_loss": self.best_val_loss}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted compressor from :meth:`state`.

        Raises:
            ValueError: if the state belongs to another kind of compressor, or its network is not
                registered (import the module that registers it first).
        """
        if state.get("kind", cls.kind) != cls.kind:
            raise ValueError(f"{cls.__name__}.from_state: the state is of kind {state['kind']!r}, not {cls.kind!r}; "
                             "sbikit.compress.compressor_from_state picks the class from the state")
        network = state["config"].get("network", "mlp")
        if not isinstance(network, str) or network not in NETWORKS:
            raise ValueError(f"{cls.__name__}.from_state: the network {network!r} of this state is not registered "
                             f"(registered: {sorted(NETWORKS)}). Import the module that adds it with "
                             "sbikit.compress.register_network before loading.")
        try:
            config = cls.config_class.make(state["config"])
        except ValueError as err:        # stored values that fail the checks: the weights are valid all the same
            warnings.warn(f"{cls.__name__}.from_state: the stored config is not valid ({err}); loading it unchecked",
                          UserWarning, stacklevel=2)
            config = cls.config_class.make(None)
            vars(config).update({**state["config"], "widths": tuple(state["config"]["widths"])})
        comp = cls(state["n_out"], config, device, state.get("seed"))
        comp.n_in = int(state["n_in"])
        comp.net = comp.build_net(comp.n_in)
        comp.net.load_state_dict(state["net"])
        comp.net.to(comp.device).eval()
        comp.in_scaler = scaler_from_state(state["in_scaler"])
        comp.t_scaler = scaler_from_state(state["t_scaler"])
        comp.atoms = [list(a) for a in state.get("atoms", [])]
        comp.history = list(state.get("history", []))
        comp.train_history = list(state.get("train_history", []))
        comp.best_val_loss = state.get("best_val_loss")
        return comp

    def __repr__(self):
        network = "custom" if self._net_factory is not None else repr(self.config.network)
        if self.net is None:
            fitted = "unfitted"
        else:
            best = "" if self.best_val_loss is None else f", best_val_loss={self.best_val_loss:.4g}"
            fitted = f"n_in={self.n_in}{best}"
        return f"{type(self).__name__}(n_out={self.n_out}, network={network}, device={self.device!r}, {fitted})"


@register
class MSECompressor(NetCompressor):
    """Regression onto the z-scored parameters with a mean-squared-error loss, so the summaries
    estimate the posterior mean in z-scored units (mean and std of the training parameters);
    :meth:`predict` gives the estimate in parameter units. ``n_out`` must be the number of
    parameters. Arguments as :class:`NetCompressor`."""
    kind = "mse"

    def fit(self, x, theta, x_val=None, theta_val=None, atoms=None, *, callback=None, verbose=False):
        """See :meth:`NetCompressor.fit`."""
        n_theta = as_2d(theta, "theta", what="simulations").shape[1]
        if n_theta != self.n_out:
            raise ValueError(f"MSE compression needs n_out = {n_theta} (the number of parameters), got "
                             f"n_out = {self.n_out}")
        return super().fit(x, theta, x_val, theta_val, atoms, callback=callback, verbose=verbose)

    def loss(self, s, t):
        """Mean squared error."""
        return F.mse_loss(s, t)

    def predict(self, x):
        """The regression output in parameter units: the summaries with the z-scoring of the
        targets undone.

        Args:
            x: (N, D) data vectors, or a single one (D,).

        Returns:
            (N, P) float64 parameter estimates, or (P,) for a single data vector.

        Raises:
            NotFittedError: before :meth:`fit`.
        """
        s = self(x)
        return np.asarray(self.t_scaler.inverse(s.astype(np.float64)))


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
        self.__dict__.pop("mdn", None)

    def val_loss(self, s, t):
        """-log q(t | s), without the VICReg terms."""
        return self.mdn.nll(s, t).item()

    def loss(self, s, t):
        """-log q(t | s), plus the VICReg terms in training if ``config.vicreg > 0`` (they need at
        least two rows)."""
        loss = self.mdn.nll(s, t)
        c = self.config
        if c.vicreg > 0 and self.net.training and len(s) > 1:
            std = torch.sqrt(s.var(0) + 1e-4)
            var = F.relu(1.0 - std).mean()
            z = s - s.mean(0)
            cov = (z.T @ z) / (len(s) - 1)
            off = cov - torch.diag(torch.diagonal(cov))
            loss = loss + c.vicreg * (c.vicreg_var * var + c.vicreg_cov * (off ** 2).sum() / s.shape[1])
        return loss
