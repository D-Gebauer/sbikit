"""Hierarchical VMIM (Williamson et al. 2026), for any partition of the data vector into blocks
(fields, tomographic bins, statistics, ...).

    anchor  : VMIM on the anchor blocks                                   -> n_anchor summaries
    blocks  : for each block in the repeat's order, VMIM on [block data, anchor and earlier
              block summaries]                                            -> n_block each
    repeats : the anchor + block chain again with another order and seed
    final   : VMIM (+ VICReg variance/covariance terms) on all stage summaries -> n_out

The stages train on one set of simulations; the final VMIM can train on its own set (pass
``final_data``), which keeps it from fitting to stage summaries that are overfitted on the stage
training set. Stage configs are fixed per stage name; tune them with
:func:`sbikit.hpt.tune_compressor` on repeat 0 and pass the result here.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .base import Compressor, register
from .nets import NetConfig, VMIMCompressor

__all__ = ["HierarchicalVMIM"]


@register
class HierarchicalVMIM(Compressor):
    """Hierarchical VMIM over named blocks of the data vector (see the module docstring).

    Args:
        blocks: block name -> column indices of x.
        anchor: names of the blocks the anchor stage compresses.
        orders: one block order per repeat (blocks not listed are not used after the anchor).
        n_out: final summary size (typically the number of parameters P).
        n_anchor, n_block: stage sizes (default 2P and P).
        configs: NetConfig (or dict) per stage name ("anchor", a block name, "final"), or "default"
            for all stages without their own entry.
        device: torch device for training and evaluation.
        seed: base seed; every stage and repeat derives its own.

    Attributes:
        stages: (repeat, stage name, :class:`VMIMCompressor`) in training order.
        final: the final VMIM.
        val_losses: validation loss per stage (``"r<repeat>/<name>"``) and ``"final"``.

    Example:
        >>> blocks = {"xi_plus": range(0, 20), "xi_minus": range(20, 40), "peaks": range(40, 55)}
        >>> h = HierarchicalVMIM(blocks, anchor=["xi_plus"], n_out=3,
        ...                      orders=[["xi_minus", "peaks"], ["peaks", "xi_minus"]])
        >>> h.fit(x_train, theta_train, x_val, theta_val)
        >>> s = h(x)                                      # (N, 3)
    """
    kind = "hierarchical_vmim"

    def __init__(self, blocks: Mapping[str, Sequence[int]], anchor: Sequence[str],
                 orders: Sequence[Sequence[str]], n_out: int, n_anchor: Optional[int] = None,
                 n_block: Optional[int] = None, configs: Optional[Mapping] = None, device="cpu", seed: int = 0):
        self.blocks = {k: np.asarray(v, dtype=np.int64) for k, v in blocks.items()}
        self.anchor, self.orders = list(anchor), [list(o) for o in orders]
        for name in self.anchor + [b for o in self.orders for b in o]:
            if name not in self.blocks:
                raise ValueError(f"unknown block {name!r}")
        self.n_out = int(n_out)
        self.n_anchor = int(n_anchor or 2 * n_out)
        self.n_block = int(n_block or n_out)
        configs = dict(configs or {})
        default = NetConfig.make(configs.pop("default", None))
        self.configs = {k: NetConfig.make(v) for k, v in configs.items()}
        self.default_config = default
        self.device, self.seed = device, seed
        self.stages: list = []      # (repeat, name, VMIMCompressor)
        self.final: Optional[VMIMCompressor] = None
        self.val_losses: dict = {}

    def config_for(self, name):
        """The stage's NetConfig. The final stage uses VICReg only if its config sets ``vicreg``."""
        return self.configs.get(name, self.default_config)

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

    def fit(self, x_trn, t_trn, x_val, t_val, final_data: Optional[Sequence[np.ndarray]] = None,
            atoms: Optional[Sequence] = None) -> float:
        """Train all stages, then the final VMIM.

        Args:
            x_trn, t_trn, x_val, t_val: training and validation data of the stages.
            final_data: ``(x_trn_f, t_trn_f, x_val_f, t_val_f)`` for the final stage (default: the
                stage data).
            atoms: passed to every stage (see :mod:`sbikit.compress.nets`).

        Returns:
            The validation loss of the final stage.
        """
        self.stages, self.val_losses = [], {}
        for r, order in enumerate(self.orders):
            seed = self.seed + 1000 * r
            comp = VMIMCompressor(self.n_anchor, self.config_for("anchor"), self.device, seed)
            self.val_losses[f"r{r}/anchor"] = comp.fit(self._anchor_input(x_trn), t_trn, self._anchor_input(x_val), t_val, atoms)
            self.stages.append((r, "anchor", comp))
            prev_t, prev_v = [comp(self._anchor_input(x_trn))], [comp(self._anchor_input(x_val))]
            for k, name in enumerate(order):
                comp = VMIMCompressor(self.n_block, self.config_for(name), self.device, seed + k + 1)
                xt = np.hstack([x_trn[:, self.blocks[name]], *prev_t])
                xv = np.hstack([x_val[:, self.blocks[name]], *prev_v])
                self.val_losses[f"r{r}/{name}"] = comp.fit(xt, t_trn, xv, t_val, atoms)
                self.stages.append((r, name, comp))
                prev_t.append(comp(xt))
                prev_v.append(comp(xv))
        xt_f, tt_f, xv_f, tv_f = final_data if final_data is not None else (x_trn, t_trn, x_val, t_val)
        self.final = VMIMCompressor(self.n_out, self.config_for("final"), self.device, self.seed + 999)
        loss = self.final.fit(np.hstack(self._stage_outputs(xt_f)), tt_f, np.hstack(self._stage_outputs(xv_f)), tv_f,
                               atoms)
        self.val_losses["final"] = loss
        return loss

    def __call__(self, x):
        return self.final(np.hstack(self._stage_outputs(np.asarray(x))))

    def state(self):
        """Plain serialisable state (all stages and the final VMIM)."""
        return {"kind": self.kind, "blocks": {k: v.tolist() for k, v in self.blocks.items()},
                "anchor": self.anchor, "orders": self.orders, "n_out": self.n_out, "n_anchor": self.n_anchor,
                "n_block": self.n_block, "seed": self.seed, "val_losses": self.val_losses,
                "stages": [(r, n, c.state()) for r, n, c in self.stages], "final": self.final.state()}

    @classmethod
    def from_state(cls, state, device="cpu"):
        """Rebuild a fitted compressor from :meth:`state`."""
        h = cls(state["blocks"], state["anchor"], state["orders"], state["n_out"], state["n_anchor"],
                state["n_block"], None, device, state["seed"])
        h.stages = [(r, n, VMIMCompressor.from_state(s, device)) for r, n, s in state["stages"]]
        h.final = VMIMCompressor.from_state(state["final"], device)
        h.val_losses = dict(state["val_losses"])
        return h
