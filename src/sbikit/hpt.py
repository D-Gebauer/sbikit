"""Optuna hyperparameter searches for compressors and NLE flows.

Search spaces are dicts: ``name -> fixed value | {"choices": [...]} | {"low", "high", "log", "int"}``.
Network widths are given as strings ("256,128,64") because optuna categoricals must be scalars.

NLE objectives:
- ``"val_loss"`` (default): minimise the flow's validation loss.
- ``"fom_tarp"``: sample the posteriors of an evaluation set (simulations with known truths), and
  maximise the mean log FoM of ``fom_params`` subject to TARP constraints (bias and confidence not
  below -tol, on all parameters and on ``fom_params``) and to every chain having converged
  (optuna's constrained TPE). Flows with a validation loss above ``prune_quantile`` of the finished
  trials are pruned before the MCMC.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import optuna
import torch
from optuna.trial import TrialState

from .metrics import evaluate, importance_weights, log_fom
from .nle import NLE, TrainConfig
from .priors import Prior
from .sampling import MCMCConfig, sample_posteriors

__all__ = ["SPACES", "suggest", "make_study", "run_study", "tune_compressor", "EvalSet", "tune_nle", "best_trials"]

WIDTHS = ["128,64", "256,128,64", "512,256,128", "512,256,128,64"]

SPACES = {
    "nle": {"model": {"choices": ["maf", "nsf"]}, "num_transforms": {"low": 3, "high": 8, "int": True},
            "hidden_features": {"choices": [32, 64, 128, 256]}, "num_blocks": {"choices": [2, 3]},
            "num_bins": {"low": 5, "high": 16, "int": True}, "lr": {"low": 1e-5, "high": 3e-3, "log": True},
            "batch_size": {"choices": [64, 128, 256]}},
    "mse": {"widths": {"choices": WIDTHS}, "lr": {"low": 1e-4, "high": 3e-3, "log": True},
            "weight_decay": {"low": 1e-5, "high": 1e-1, "log": True}, "dropout": {"choices": [0.0, 0.1]}},
    "vmim": {"widths": {"choices": WIDTHS}, "lr": {"low": 1e-4, "high": 3e-3, "log": True},
             "mdn_hidden": {"choices": [64, 128]}, "components": {"choices": [4, 6, 8]}},
    "imnn": {"widths": {"choices": WIDTHS}, "lr": {"low": 1e-4, "high": 3e-3, "log": True}},
}


def suggest(trial: optuna.Trial, space: Mapping) -> dict:
    """Draw one configuration from ``space`` (see the module docstring for its format).
    ``"widths"`` strings such as ``"256,128"`` are returned as tuples of ints."""
    cfg = {}
    for name, spec in space.items():
        if isinstance(spec, Mapping) and "choices" in spec:
            v = trial.suggest_categorical(name, list(spec["choices"]))
        elif isinstance(spec, Mapping) and "low" in spec:
            if spec.get("int"):
                v = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log")))
            else:
                v = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
        else:
            v = spec
        if name == "widths" and isinstance(v, str):
            v = tuple(int(w) for w in v.split(","))
        cfg[name] = v
    return cfg


def make_study(storage: Union[str, Path], name: str, direction: str = "minimize", n_startup: int = 10,
               seed: int = 0, constrained: bool = False) -> optuna.Study:
    """TPE study in a sqlite file (``*.db``) or an optuna journal file (anything else; several
    processes can work on one journal study)."""
    storage = Path(storage)
    storage.parent.mkdir(parents=True, exist_ok=True)
    if storage.suffix == ".db":
        backend = f"sqlite:///{storage}"
    else:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        backend = JournalStorage(JournalFileBackend(str(storage)))
    kw = {}
    if constrained:
        kw["constraints_func"] = lambda t: t.user_attrs.get("constraints", [1.0])
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=seed, multivariate=True, **kw)
    return optuna.create_study(study_name=name, storage=backend, direction=direction, sampler=sampler,
                               load_if_exists=True)


def run_study(study: optuna.Study, objective: Callable, n_trials: int) -> optuna.Study:
    """Run until the study has ``n_trials`` finished (complete or pruned) trials; resumes."""
    done = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.trials)
    if n_trials - done > 0:
        study.optimize(objective, n_trials=n_trials - done, gc_after_trial=True)
    return study


def _save(path: Path, obj) -> None:
    from .io import save
    save(obj, path)


def tune_compressor(factory: Callable[[dict], object], space: Mapping, x_trn, t_trn, x_val, t_val,
                    n_trials: int, storage, name: str, outdir=None, n_startup: int = 25, seed: int = 0,
                    fixed: Optional[dict] = None):
    """Minimise a compressor's validation loss.

    Args:
        factory: ``factory(config) -> compressor``, e.g. ``lambda c: VMIMCompressor(n_out, c)``.
        space: search space; ``SPACES["mse"]``, ``SPACES["vmim"]`` and ``SPACES["imnn"]`` are
            starting points.
        x_trn, t_trn, x_val, t_val: training and validation data, passed to ``compressor.fit``.
        n_trials: number of finished trials to reach (the study resumes).
        storage: study file (``*.db``: sqlite, else an optuna journal file).
        name: study name; trials are saved to ``outdir/name/trials/NNNN.pt``.
        outdir: folder for the trial checkpoints (None: keep them in memory only).
        n_startup: random trials before TPE.
        seed: seed of the sampler and of each trial's training.
        fixed: run a single trial with this configuration instead of a search (e.g. to retrain a
            tuned stage with another seed).

    Returns:
        (best config, best compressor, study).
    """
    study = make_study(storage, name, "minimize", n_startup, seed)
    outdir = Path(outdir) if outdir is not None else None
    if fixed is not None and not study.trials:
        study.enqueue_trial({k: (",".join(map(str, v)) if k == "widths" else v) for k, v in fixed.items()
                             if k in space and isinstance(space[k], Mapping)})
    kept = {}

    def objective(trial):
        torch.manual_seed(seed * 1000 + trial.number)
        cfg = suggest(trial, space)
        comp = factory(cfg)
        t0 = time.time()
        loss = comp.fit(x_trn, t_trn, x_val, t_val)
        trial.set_user_attr("seconds", time.time() - t0)
        trial.set_user_attr("config", {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()})
        if not np.isfinite(loss):
            raise optuna.TrialPruned()
        if outdir is not None:
            _save(outdir / name / "trials" / f"{trial.number:04d}.pt", comp)
        kept[trial.number] = comp
        return loss

    run_study(study, objective, 1 if fixed is not None else n_trials)
    best = study.best_trial
    comp = kept.get(best.number)
    if comp is None and outdir is not None:
        from .io import load
        comp = load(outdir / name / "trials" / f"{best.number:04d}.pt")
    return best.user_attrs["config"], comp, study


@dataclass
class EvalSet:
    """Simulations with known truths whose posteriors score an NLE trial (``fom_tarp``).

    Attributes:
        s: (B, n_x) summaries.
        truth: (B, P) true parameters.
        prior: the sampling prior, or one per simulation.
        param_names: the P names.
        init_sd: (P,) scale of the MCMC starting ball (default: prior width / sqrt 12).
        fom_prior: reweight the posteriors to this prior before computing the figure of merit
            (e.g. a flat box when the sampling prior follows a non-uniform simulation design).
    """

    s: np.ndarray                        # (B, n_x) summaries
    truth: np.ndarray                    # (B, P)
    prior: Union[Prior, Sequence[Prior]] # sampling prior(s)
    param_names: Sequence[str]
    init_sd: Optional[np.ndarray] = None # (P,) MCMC ball scale (default: prior width / sqrt 12)
    fom_prior: Optional[Prior] = None    # reweight to this prior for the FoM objective (e.g. a flat box)


def best_trials(study: optuna.Study, k: int = 3, feasible_only: bool = True) -> list:
    """The k best complete trials (feasible ones only for constrained studies)."""
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if feasible_only:
        done = [t for t in done if all(c <= 0 for c in t.user_attrs.get("constraints", [0.0]))]
    rev = study.direction == optuna.study.StudyDirection.MAXIMIZE
    return sorted(done, key=lambda t: t.value, reverse=rev)[:k]


def _objective_fom(samples, ev: EvalSet, fom_idx):
    """Mean log FoM over the eval set; reweighted to ``ev.fom_prior`` and restricted to truths in its
    support if given (a NaN FoM there counts as the worst finite one, not as missing)."""
    if ev.fom_prior is None:
        lf = log_fom(samples, fom_idx)
    else:
        if isinstance(ev.prior, (list, tuple)):
            raise ValueError("fom_prior reweighting needs a single sampling prior")
        w = importance_weights(samples, ev.prior, ev.fom_prior)
        inside = ev.fom_prior.inside(torch.as_tensor(ev.truth, dtype=torch.float32)).numpy()
        lf = log_fom(samples, fom_idx, w)[inside]
    nan = np.isnan(lf)
    if nan.all():
        return -np.inf
    return float(np.where(nan, np.nanmin(lf), lf).mean())


def tune_nle(s_trn, t_trn, s_val, t_val, n_trials: int, storage, name: str, outdir,
             space: Optional[Mapping] = None, train: Optional[Mapping] = None, objective: str = "val_loss",
             evaluation: Optional[EvalSet] = None, mcmc=None, fom_params: Sequence[str] = (),
             tarp_params: Optional[Sequence[str]] = None, prune_quantile: float = 0.75, prune_min_trials: int = 10,
             n_references: int = 50, n_bootstrap: int = 200, tol_min: float = 0.02, n_startup: int = 15,
             device="cpu", seed: int = 0) -> optuna.Study:
    """Hyperparameter search for an :class:`~sbikit.nle.NLE`.

    Every trial's flow is saved to ``outdir/name/trials/NNNN.pt`` with a JSON record (config,
    validation loss and, for ``fom_tarp``, the standard metrics record and posterior arrays).

    Args:
        s_trn, t_trn, s_val, t_val: training and validation summaries and parameters.
        n_trials: number of finished trials to reach (the study resumes).
        storage: study file (``*.db``: sqlite, else an optuna journal file).
        name: study name and output subfolder.
        outdir: output folder.
        space: search space over :class:`~sbikit.nle.FlowConfig` and
            :class:`~sbikit.nle.TrainConfig` fields (default ``SPACES["nle"]``).
        train: fixed training settings (searched ones override them).
        objective: ``"val_loss"`` or ``"fom_tarp"`` (see the module docstring).
        evaluation: the :class:`EvalSet` (``fom_tarp``).
        mcmc: :class:`~sbikit.sampling.MCMCConfig` or dict for the evaluation posteriors.
        fom_params: parameters of the figure of merit to maximise (``fom_tarp``).
        tarp_params: parameters of the second TARP constraint (default ``fom_params``); the first
            is always on all parameters.
        prune_quantile, prune_min_trials: once ``prune_min_trials`` trials have finished, flows
            with a validation loss above this quantile skip the MCMC and are pruned.
        n_references, n_bootstrap, tol_min: TARP settings (:func:`sbikit.metrics.evaluate`).
        n_startup: random trials before TPE.
        device: torch device for training and sampling.
        seed: seed of the sampler and of each trial's training.

    Returns:
        The optuna study; :func:`best_trials` lists its best (feasible) trials.
    """
    if objective not in ("val_loss", "fom_tarp"):
        raise ValueError(f"unknown objective {objective!r}")
    if objective == "fom_tarp" and (evaluation is None or not fom_params):
        raise ValueError("objective 'fom_tarp' needs an EvalSet and fom_params")
    space = dict(SPACES["nle"] if space is None else space)
    base_train = asdict(TrainConfig.make(train))
    mcmc = MCMCConfig.make(mcmc)
    folder = Path(outdir) / name
    (folder / "trials").mkdir(parents=True, exist_ok=True)
    constrained = objective == "fom_tarp"
    study = make_study(storage, name, "maximize" if constrained else "minimize", n_startup, seed, constrained)
    n_x, n_theta = np.shape(s_trn)[1], np.shape(t_trn)[1]
    if constrained:
        names = list(evaluation.param_names)
        fom_idx = [names.index(p) for p in fom_params]
        tarp_sets = {"fom": list(tarp_params or fom_params)}

    def obj(trial):
        cfg = suggest(trial, space)
        flow_cfg = {k: v for k, v in cfg.items() if k in ("model", "hidden_features", "num_blocks", "num_transforms", "num_bins")}
        tr = {**base_train, **{k: v for k, v in cfg.items() if k in base_train}}
        t0 = time.time()
        nle = NLE(n_x, n_theta, flow_cfg, device)
        val, epochs = nle.fit(s_trn, t_trn, s_val, t_val, tr, seed=seed * 1000 + trial.number)
        rec = {"trial": trial.number, "config": cfg, "train": tr, "val_loss": val, "epochs": epochs,
               "train_seconds": time.time() - t0}
        trial.set_user_attr("val_loss", val)
        if not np.isfinite(val):
            raise optuna.TrialPruned()
        _save(folder / "trials" / f"{trial.number:04d}.pt", nle)
        if not constrained:
            _write_json(folder / "trials" / f"{trial.number:04d}.json", rec)
            return val
        vals = [t.user_attrs["val_loss"] for t in study.trials
                if t.state == TrialState.COMPLETE and "val_loss" in t.user_attrs]
        if len(vals) >= prune_min_trials and val > np.quantile(vals, prune_quantile):
            rec["pruned"] = f"val_loss above the {prune_quantile} quantile"
            _write_json(folder / "trials" / f"{trial.number:04d}.json", rec)
            raise optuna.TrialPruned()
        post = sample_posteriors(nle, evaluation.s, evaluation.prior, mcmc, init_centre=evaluation.truth,
                                 init_sd=evaluation.init_sd, device=device, verbose=False)
        metrics, arrays = evaluate(post.samples, evaluation.truth, names, {"fom": list(fom_params)}, tarp_sets,
                                   n_references, n_bootstrap, tol_min, seed, device)
        value = _objective_fom(post.samples, evaluation, fom_idx)
        conv = float(post.converged.mean())
        cons = []
        for key in ("all", "fom"):
            t = metrics["tarp"][key]
            cons += [-t["confidence"] - t["tol_confidence"], -t["bias"] - t["tol_bias"]]
        cons.append(0.0 if conv >= 1.0 else 1.0 - conv + 1e-3)
        trial.set_user_attr("constraints", cons)
        rec.update({"objective": value, "constraints": cons, "converged": conv, "metrics": metrics,
                    "mcmc_steps": post.diagnostics["steps"], "seconds": time.time() - t0})
        _write_json(folder / "trials" / f"{trial.number:04d}.json", rec)
        np.savez(folder / "trials" / f"{trial.number:04d}_posteriors.npz", **arrays,
                 converged=post.converged, rhat_max=post.diagnostics["rhat_max"])
        return value if np.isfinite(value) else -1e30

    return run_study(study, obj, n_trials)


def _write_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return np.asarray(o).tolist()
        if isinstance(o, tuple):
            return list(o)
        return str(o)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=default))
    tmp.replace(path)
