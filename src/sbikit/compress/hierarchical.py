"""Hierarchical VMIM (Williamson et al. 2026), for any partition of the data vector into blocks
(fields, tomographic bins, statistics, ...).

    anchor  : VMIM on the anchor blocks                                   -> n_anchor summaries
    blocks  : for each block in the repeat's order, VMIM on [block data, anchor and earlier
              block summaries]                                            -> n_block each
    repeats : the anchor + block chain again with another order and seed
    final   : VMIM on all stage summaries                                 -> n_out

The final stage adds VICReg variance/covariance terms only if its config asks for them
(``configs={"final": {"vicreg": 0.1}}``); ``configs["default"]`` applies to every stage without an
entry of its own. A stage's entry is its complete config: it is not merged with the default.

The stages train on one set of simulations; the final VMIM can train on its own set (pass
``final_data``), which keeps it from fitting to stage summaries that are overfitted on the stage
training set. Stage configs are fixed per stage name; tune them with
:func:`sbikit.hpt.tune_compressor` on repeat 0 and pass the result here.
"""
from __future__ import annotations

import warnings
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .._utils import check_pair, did_you_mean, holdout_split, logger, plain, resolve_device, unknown
from .base import Compressor, register
from .nets import NetConfig, VMIMCompressor

__all__ = ["HierarchicalVMIM"]

_RESERVED = ("anchor", "final", "default")


class _StageFailed(Exception):
    """A stage ended without a finite validation loss."""


@register
class HierarchicalVMIM(Compressor):
    """Hierarchical VMIM over named blocks of the data vector (see the module docstring).

    Args:
        blocks: block name -> integer column indices of x (a list, range or integer array; convert
            a boolean mask with ``np.flatnonzero``). The names ``"anchor"``, ``"final"`` and
            ``"default"`` are reserved for the stages.
        anchor: names of the blocks the anchor stage compresses (a single name is accepted).
        orders: one block order per repeat, ``[["b", "c"], ["c", "b"]]`` (blocks not listed are
            not used after the anchor); a flat list of names ``["b", "c"]`` is one repeat.
        n_out: final summary size (typically the number of parameters P).
        n_anchor, n_block: stage sizes, positive integers (default 2 ``n_out`` and ``n_out``).
        configs: :class:`~sbikit.compress.NetConfig` (or dict) per stage name: ``"anchor"``, a
            block name, ``"final"``, and ``"default"`` for all stages without their own entry. A
            stage's own entry replaces the default as a whole (options it leaves out take the
            package defaults, not those of ``"default"``), so repeat the shared options in it. A
            single NetConfig instance is the default for every stage. VICReg terms are used where a
            stage's config sets ``vicreg > 0`` (usually only ``"final"``).
        device: torch device for training and evaluation (``"auto"``: a GPU if there is one).
        seed: base seed; every stage and repeat derives its own. None: unseeded.

    Attributes:
        stages: (repeat, stage name, :class:`VMIMCompressor`) in training order.
        final: the final VMIM (None before :meth:`fit`).
        val_losses: validation loss per stage (``"r<repeat>/<name>"``) and ``"final"``.
        configs, default_config: the stage configs.

    Raises:
        ValueError: for blocks that are not integer indices, unknown or reserved block names, an
            empty ``anchor`` or ``orders``, sizes that are not positive integers, or ``configs``
            keys that are not stage names.

    Example:
        >>> blocks = {"first": range(0, 20), "second": range(20, 40), "third": range(40, 55)}
        >>> shared = {"epochs": 200}
        >>> h = HierarchicalVMIM(blocks, anchor=["first"], n_out=3,
        ...                      orders=[["second", "third"], ["third", "second"]],
        ...                      configs={"default": shared, "final": {**shared, "vicreg": 0.1}})
        >>> h.fit(x_train, theta_train, x_val, theta_val)
        >>> s = h(x)                                      # (N, 3)
    """
    kind = "hierarchical_vmim"

    def __init__(self, blocks: Mapping[str, Sequence[int]], anchor: Sequence[str],
                 orders: Sequence[Sequence[str]], n_out: int, n_anchor: Optional[int] = None,
                 n_block: Optional[int] = None, configs: Optional[Mapping] = None, device="cpu",
                 seed: Optional[int] = 0):
        self._setup(blocks, anchor, orders, n_out, n_anchor, n_block, configs, device, seed)

    def _setup(self, blocks, anchor, orders, n_out, n_anchor, n_block, configs, device, seed, stored: bool = False):
        """Validate and store the constructor arguments. ``stored``: they come from a saved state,
        whose block names and (possibly empty) index lists are taken as they are."""
        if not isinstance(blocks, Mapping) or not blocks:
            raise ValueError("blocks must be a non-empty dict: block name -> column indices of x")
        if stored:
            self.blocks = {name: np.asarray(cols, dtype=np.int64).reshape(-1) for name, cols in blocks.items()}
        else:
            self.blocks = {name: self._indices(name, cols) for name, cols in blocks.items()}
        reserved = [] if stored else [name for name in self.blocks if name in _RESERVED]
        if reserved:
            raise ValueError(f"block name(s) {reserved} are reserved for the stages {list(_RESERVED)}; rename them")
        self.anchor = [anchor] if isinstance(anchor, str) or not np.iterable(anchor) else list(anchor)
        if not self.anchor:
            raise ValueError("anchor is empty: name at least one block for the anchor stage")
        self.orders = self._orders(orders)
        for name in self.anchor + [b for o in self.orders for b in o]:
            if name not in self.blocks:
                raise unknown("block", name, list(self.blocks))
        self.n_out = self._size("n_out", n_out)
        self.n_anchor = self._size("n_anchor", 2 * self.n_out if n_anchor is None else n_anchor)
        self.n_block = self._size("n_block", self.n_out if n_block is None else n_block)
        if isinstance(configs, NetConfig):
            configs = {"default": configs}
        elif configs is not None and not isinstance(configs, Mapping):
            raise TypeError(f"configs must be a dict stage name -> NetConfig or dict, got {type(configs).__name__}")
        configs = dict(configs or {})
        valid = [*_RESERVED, *self.blocks]
        bad = [k for k in configs if k not in valid]
        if bad:
            hints = ", ".join(f"{k!r}{did_you_mean(k, valid)}" for k in bad)
            flat = " To set options for every stage use configs={'default': {...}}." \
                if all(k in NetConfig.fields() for k in bad) else ""
            raise ValueError(f"configs: {hints} is not a stage name; valid: {valid}.{flat}")
        self.default_config = NetConfig.make(configs.pop("default", None))
        self.configs = {k: NetConfig.make(v) for k, v in configs.items()}
        self.device, self.seed = resolve_device(device), seed
        self.stages: list = []      # (repeat, name, VMIMCompressor)
        self.final: Optional[VMIMCompressor] = None
        self.val_losses: dict = {}
        self.n_in: Optional[int] = None

    @staticmethod
    def _size(name, value) -> int:
        """A summary size as an int; anything but a positive integer is an error naming the argument."""
        try:
            ok = int(value) == value and value >= 1
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return int(value)

    @staticmethod
    def _indices(name, cols) -> np.ndarray:
        """A block's column indices as a 1D int64 array."""
        a = np.asarray(cols)
        if a.ndim != 1 or a.size == 0:
            raise ValueError(f"blocks[{name!r}] must be a non-empty 1D list of column indices, got shape {a.shape}")
        if a.dtype.kind not in "iu":
            hint = " (convert a boolean mask with np.flatnonzero(mask))" if a.dtype.kind == "b" else ""
            raise ValueError(f"blocks[{name!r}] must hold integer column indices, got dtype {a.dtype}{hint}")
        return a.astype(np.int64)

    @staticmethod
    def _orders(orders) -> list:
        """``orders`` as a list of lists of block names; a flat list of names is one repeat."""
        if isinstance(orders, str) or not np.iterable(orders):
            orders = [[orders]]
        orders = list(orders)
        flat = [isinstance(o, str) or not np.iterable(o) for o in orders]
        if not orders:
            raise ValueError("orders is empty: give one block order per repeat, e.g. orders=[['b', 'c']] "
                             "(orders=[[]] for the anchor stage alone)")
        if all(flat):
            return [orders]
        if any(flat):
            raise ValueError(f"orders must be one list of block names per repeat, [['b', 'c'], ['c', 'b']], or a "
                             f"single flat list ['b', 'c']; got {orders!r}")
        return [list(o) for o in orders]

    def config_for(self, name):
        """The stage's NetConfig: its own entry in ``configs`` if it has one, else the default (the
        two are not merged). The final stage uses VICReg only if its config sets ``vicreg``."""
        return self.configs.get(name, self.default_config)

    def _check_columns(self, n_cols: int) -> None:
        for name, cols in self.blocks.items():
            if cols.size and (cols.min() < -n_cols or cols.max() >= n_cols):
                worst = cols.max() if cols.max() >= n_cols else cols.min()
                raise ValueError(f"blocks[{name!r}] has the column index {int(worst)}, but x has {n_cols} columns")

    def _anchor_input(self, x):
        return x[:, np.concatenate([self.blocks[b] for b in self.anchor])]

    def _stage_outputs(self, x):
        """Summaries of every stage of every repeat, in training order (list of (N, k) arrays)."""
        outs, chain = [], None
        for _, name, comp in self.stages:
            if name == "anchor":                 # a new repeat starts
                if chain is not None:
                    outs.extend(chain)
                chain = [comp(self._anchor_input(x))]
            else:
                chain.append(comp(np.hstack([x[:, self.blocks[name]], *chain])))
        if chain is not None:
            outs.extend(chain)
        return outs

    def fit(self, x, theta, x_val=None, theta_val=None, final_data: Optional[Sequence] = None,
            atoms: Optional[Sequence] = None, *, callback: Optional[Callable] = None, verbose: bool = False) -> float:
        """Train all stages, then the final VMIM.

        Args:
            x, theta: (N, D) data vectors and (N, P) parameters the stages train on.
            x_val, theta_val: their validation set. Without one, a random
                ``configs["default"].val_fraction`` of the given simulations is held out once (drawn
                with ``seed``) and used by all stages.
            final_data: ``(x, theta, x_val, theta_val)`` for the final stage (default: the stage
                data). Its validation pair may be ``None, None``; the final VMIM then holds out
                its own.
            atoms: passed to every stage (see :mod:`sbikit.compress.nets`).
            callback: called after every epoch of every stage with the dict of
                :meth:`NetCompressor.fit <sbikit.compress.nets.NetCompressor.fit>`, where
                ``"model"`` is this object, plus ``"stage"`` (the key in ``val_losses``) and
                ``"compressor"`` (the stage's VMIM). Returning True ends the current stage; the
                remaining stages still train.
            verbose: log one line per stage.

        Returns:
            The validation loss of the final stage. ``inf`` if a stage reached no finite
            validation loss: a ``RuntimeWarning`` names it and the object is left unfitted.

        Raises:
            ValueError: for inputs of the wrong shape, different numbers of rows, NaN or inf, or
                block indices outside the columns of ``x``.
        """
        x, theta = check_pair(x, theta)
        n_in, n_theta = x.shape[1], theta.shape[1]
        self._check_columns(n_in)
        if (x_val is None) != (theta_val is None):
            raise ValueError("x_val and theta_val must be given together (or both left out)")
        if x_val is not None:
            x_val, theta_val = check_pair(x_val, theta_val, "x_val", "theta_val", n_x=n_in, n_theta=n_theta)
        if final_data is not None:
            if isinstance(final_data, (np.ndarray, str)) or len(final_data) != 4:
                raise ValueError("final_data must be the tuple (x, theta, x_val, theta_val) of the final stage")
            xt_f, tt_f, xv_f, tv_f = final_data
            xt_f, tt_f = check_pair(xt_f, tt_f, "final_data x", "final_data theta", n_x=n_in, n_theta=n_theta)
            if (xv_f is None) != (tv_f is None):
                raise ValueError("final_data: x_val and theta_val must be given together (or both be None)")
            if xv_f is not None:
                xv_f, tv_f = check_pair(xv_f, tv_f, "final_data x_val", "final_data theta_val", n_x=n_in,
                                        n_theta=n_theta)
        if x_val is None:
            fraction = self.default_config.val_fraction
            trn, val = holdout_split(len(x), fraction, self.seed)
            logger.info(f"HierarchicalVMIM: no validation set given, holding out {len(val)} of {len(x)} simulations "
                        f"for all stages (val_fraction = {fraction})")
            x, theta, x_val, theta_val = x[trn], theta[trn], x[val], theta[val]
        if final_data is None:
            xt_f, tt_f, xv_f, tv_f = x, theta, x_val, theta_val

        def train(key, n_out, stage, seed, xt, tt, xv, tv):
            comp = VMIMCompressor(n_out, self.config_for(stage), self.device, seed)
            each = None if callback is None else (
                lambda info: callback({**info, "model": self, "stage": key, "compressor": comp}))
            with warnings.catch_warnings():      # a failed stage is reported once, below, with its name
                warnings.filterwarnings("ignore", message=".*left unfitted", category=RuntimeWarning)
                self.val_losses[key] = loss = comp.fit(xt, tt, xv, tv, atoms, callback=each)
            if verbose:
                logger.info(f"HierarchicalVMIM {key}: validation loss {loss:.5f} after {len(comp.history)} epochs")
            if not np.isfinite(loss):
                raise _StageFailed(key)
            return comp

        self.stages, self.final, self.val_losses, self.n_in = [], None, {}, n_in
        try:
            for r, order in enumerate(self.orders):
                seed = None if self.seed is None else self.seed + 1000 * r
                xt, xv = self._anchor_input(x), self._anchor_input(x_val)
                comp = train(f"r{r}/anchor", self.n_anchor, "anchor", seed, xt, theta, xv, theta_val)
                self.stages.append((r, "anchor", comp))
                prev_t, prev_v = [comp(xt)], [comp(xv)]
                for k, name in enumerate(order):
                    seen = order[:k].count(name)
                    key = f"r{r}/{name}" + (f"#{seen + 1}" if seen else "")
                    xt = np.hstack([x[:, self.blocks[name]], *prev_t])
                    xv = np.hstack([x_val[:, self.blocks[name]], *prev_v])
                    comp = train(key, self.n_block, name, None if seed is None else seed + k + 1, xt, theta, xv,
                                 theta_val)
                    self.stages.append((r, name, comp))
                    prev_t.append(comp(xt))
                    prev_v.append(comp(xv))
            xv_f = None if xv_f is None else np.hstack(self._stage_outputs(xv_f))
            final = train("final", self.n_out, "final", None if self.seed is None else self.seed + 999,
                          np.hstack(self._stage_outputs(xt_f)), tt_f, xv_f, tv_f)
        except _StageFailed as failed:
            self.stages = []
            warnings.warn(f"HierarchicalVMIM: stage {failed.args[0]!r} reached no finite validation loss; the "
                          "compressor is left unfitted. Lower that stage's lr or check its inputs.", RuntimeWarning,
                          stacklevel=2)
            return float("inf")
        except BaseException:
            self.stages = []
            raise
        self.final = final
        return self.val_losses["final"]

    def __call__(self, x):
        """Summaries (N, n_out), float32, of the data vectors ``x`` (N, D); a single data vector
        (D,) gives (n_out,).

        Raises:
            NotFittedError: before :meth:`fit`.
            ValueError: if ``x`` does not have the columns the blocks refer to.
        """
        if self.final is None:
            raise self._not_fitted()
        x, single = self._batch(x, self.n_in)
        self._check_columns(x.shape[1])
        s = self.final(np.hstack(self._stage_outputs(x)))
        return s[0] if single else s

    def to(self, device):
        """Move every stage and the final VMIM to ``device``; returns itself."""
        self.device = resolve_device(device)
        for _, _, comp in self.stages:
            comp.to(self.device)
        if self.final is not None:
            self.final.to(self.device)
        return self

    def state(self):
        """Plain serialisable state (all stages, the final VMIM and the stage configs).

        Raises:
            NotFittedError: before :meth:`fit`.
        """
        if self.final is None:
            raise self._not_fitted()
        return {"kind": self.kind, "blocks": {k: v.tolist() for k, v in self.blocks.items()},
                "anchor": list(self.anchor), "orders": [list(o) for o in self.orders], "n_out": self.n_out,
                "n_anchor": self.n_anchor, "n_block": self.n_block, "seed": plain(self.seed), "n_in": self.n_in,
                "val_losses": {k: float(v) for k, v in self.val_losses.items()},
                "default_config": self.default_config.to_dict(),
                "configs": {k: c.to_dict() for k, c in self.configs.items()},
                "stages": [(r, n, c.state()) for r, n, c in self.stages], "final": self.final.state()}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted compressor from :meth:`state`.

        Raises:
            ValueError: if the state belongs to another kind of compressor.
        """
        if state.get("kind", cls.kind) != cls.kind:
            raise ValueError(f"{cls.__name__}.from_state: the state is of kind {state['kind']!r}, not {cls.kind!r}; "
                             "sbikit.compress.compressor_from_state picks the class from the state")
        configs = {**state.get("configs", {}), "default": state.get("default_config")}
        h = cls.__new__(cls)
        h._setup(state["blocks"], state["anchor"], state["orders"], state["n_out"], state["n_anchor"],
                 state["n_block"], configs, device, state["seed"], stored=True)
        h.stages = [(r, n, VMIMCompressor.from_state(s, h.device)) for r, n, s in state["stages"]]
        h.final = VMIMCompressor.from_state(state["final"], h.device)
        h.val_losses = dict(state["val_losses"])
        h.n_in = state.get("n_in")
        return h

    def __repr__(self):
        fitted = "unfitted" if self.final is None else f"fitted, final val loss {self.val_losses.get('final'):.4g}"
        return (f"HierarchicalVMIM(blocks={list(self.blocks)}, anchor={self.anchor}, repeats={len(self.orders)}, "
                f"n_out={self.n_out}, device={self.device!r}, {fitted})")
