"""Optuna hyperparameter searches for compressors and NLE flows.

Search spaces are dicts ``name -> spec``, where a spec is one of

- a fixed value (``{"value": x}`` for a fixed value that is itself a dict);
- ``{"choices": [...]}``: a categorical over numbers, strings, booleans or None;
- ``{"low": a, "high": b}`` with the optional entries ``"log"`` (sample in the log), ``"int"``
  (integers) and ``"step"`` (a grid of that spacing).

Anything else raises a ``ValueError``. Network widths are given as strings (``"256,128,64"``)
because optuna categoricals must be scalars; :func:`suggest` returns them as tuples of ints.

NLE objectives (:func:`tune_nle`):

- ``"val_loss"`` (default): minimise the flow's validation loss.
- ``"fom_tarp"``: sample the posteriors of an evaluation set (simulations with known truths), and
  maximise the mean log FoM of ``fom_params`` subject to TARP constraints (bias and confidence not
  below -tol, on all parameters and on ``fom_params``) and to every chain having converged
  (optuna's constrained TPE). Flows with a validation loss above the ``prune_quantile`` of the
  validation losses of all trials finished so far skip the MCMC and are pruned.
- a callable ``(nle, trial) -> value`` or ``-> (value, constraints)``: your own objective, minimised.

Studies live in memory, in a sqlite file, in an optuna journal file (which several processes can
work on at once) or in any optuna storage. A search called again on an existing study resumes it.
Give parallel workers different ``seed`` values: workers that start together with the same seed
propose the same configurations.

With an output folder every trial's model is saved; in any case the best models trained by the
current process stay in memory (the best compressor, the :data:`KEEP_IN_MEMORY` best flows), and
:func:`load_trial` returns the model of a trial from either place.

The searches use multivariate TPE and, for constrained objectives, ``constraints_func``; both are
experimental optuna features, whose ``ExperimentalWarning`` is suppressed here.
"""
from __future__ import annotations

import contextlib
import os
import time
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import optuna
import torch
from optuna.trial import TrialState

from ._utils import (as_2d, as_numpy, check_finite, check_pair, did_you_mean, logger, parse_widths, resolve_device,
                     seeded, to_tensor, unknown)
from .io import _jsonable, _write_json, load, save
from .metrics import evaluate, importance_weights, log_fom
from .nle import NLE, FlowConfig, TrainConfig
from .priors import Prior
from .sampling import MCMCConfig, sample_posteriors

__all__ = ["SPACES", "KEEP_IN_MEMORY", "suggest", "make_study", "run_study", "tune_compressor", "EvalSet", "tune_nle",
           "best_trials", "load_trial"]

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

KEEP_IN_MEMORY = 5                       # flows per study that tune_nle keeps in memory (the best ones)
_MAX_FAILURES = 10                       # failed trials in a row after which run_study gives up
_RETRAIN_RTOL = 1e-4                     # a retrained best compressor reproduces its trial if the losses agree to this
_FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)
_RANGE_KEYS = ("low", "high", "log", "int", "step")
_MODELS = weakref.WeakKeyDictionary()    # study -> {trial number: model trained by this process}


@contextlib.contextmanager
def _quiet():
    """Suppress optuna's ExperimentalWarning inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        yield


def _trial_seed(seed: Optional[int], number: int) -> Optional[int]:
    """Training seed of a trial (None without a seed)."""
    return None if seed is None else int(seed) * 1000 + int(number)


# ------------------------------------------------------------------------------ search spaces
def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _parse_spec(name, spec) -> tuple:
    """A spec as ``("fixed", value)``, ``("choices", list)`` or ``("range", dict)``.

    Raises:
        ValueError: for a mapping that is none of the documented forms.
    """
    if not isinstance(spec, Mapping):
        return "fixed", spec
    where = f"search space entry {name!r}"
    forms = ("a fixed value, {'choices': [...]}, {'low': a, 'high': b} with the optional entries 'log', 'int' and "
             "'step', or {'value': x} for a fixed value that is a dict")
    keys = set(spec)
    if "value" in keys:
        if keys != {"value"}:
            raise ValueError(f"{where}: 'value' cannot be combined with {sorted(keys - {'value'}, key=str)}; use "
                             f"{forms}")
        return "fixed", spec["value"]
    if "choices" in keys:
        if keys != {"choices"}:
            raise ValueError(f"{where}: 'choices' cannot be combined with {sorted(keys - {'choices'}, key=str)}; use "
                             f"{forms}")
        choices = spec["choices"]
        if isinstance(choices, (str, Mapping)) or not hasattr(choices, "__iter__") or len(list(choices)) == 0:
            raise ValueError(f"{where}: 'choices' must be a non-empty list, got {choices!r}")
        choices = [c.item() if isinstance(c, np.generic) else c for c in choices]
        bad = [c for c in choices if not (c is None or isinstance(c, (bool, int, float, str)))]
        if bad:
            raise ValueError(f"{where}: choices must be numbers, strings, booleans or None, got {bad[0]!r} (write "
                             "network widths as strings such as '256,128')")
        return "choices", choices
    extra = sorted(keys - set(_RANGE_KEYS), key=str)
    if extra or not {"low", "high"} <= keys:
        hints = "".join(f" {k!r}{did_you_mean(k, [*_RANGE_KEYS, 'choices', 'value'])}" for k in extra)
        problem = f"unknown keys{hints}" if extra else "'low' and 'high' are both needed for a range"
        raise ValueError(f"{where}: {dict(spec)!r} is not a valid spec ({problem}); use {forms}")
    low, high, step = spec["low"], spec["high"], spec.get("step")
    log, integer = bool(spec.get("log", False)), bool(spec.get("int", False))
    if not (_is_number(low) and _is_number(high) and np.isfinite(low) and np.isfinite(high) and low <= high):
        raise ValueError(f"{where}: 'low' and 'high' must be finite numbers with low <= high, got {low!r}, {high!r}")
    if log and low <= 0:
        raise ValueError(f"{where}: a log range needs low > 0, got {low!r}")
    if step is not None and (not _is_number(step) or step <= 0 or log):
        raise ValueError(f"{where}: 'step' must be a positive number and cannot be combined with 'log'")
    if integer:
        if any(float(v) != int(v) for v in (low, high) + (() if step is None else (step,))):
            raise ValueError(f"{where}: an integer range needs integer 'low', 'high' and 'step', got {dict(spec)!r}")
        return "range", {"low": int(low), "high": int(high), "log": log, "int": True,
                         "step": None if step is None else int(step)}
    return "range", {"low": float(low), "high": float(high), "log": log, "int": False,
                     "step": None if step is None else float(step)}


def _check_space(space) -> dict:
    """``name -> parsed spec`` of a search space; raises for a malformed one."""
    if not isinstance(space, Mapping):
        raise TypeError(f"a search space is a dict name -> spec, got {type(space).__name__}")
    return {name: _parse_spec(name, spec) for name, spec in space.items()}


def _suggest_one(trial, name, kind, arg):
    if kind == "choices":
        return trial.suggest_categorical(name, arg)
    if kind == "range" and arg["int"]:
        return trial.suggest_int(name, arg["low"], arg["high"], step=arg["step"] or 1, log=arg["log"])
    if kind == "range":
        return trial.suggest_float(name, arg["low"], arg["high"], step=arg["step"], log=arg["log"])
    return arg


def _widths(name, value):
    """``"256,128"`` -> ``(256, 128)`` for the ``widths`` entry; other values unchanged."""
    return parse_widths(value) if name == "widths" and isinstance(value, str) else value


def suggest(trial: optuna.Trial, space: Mapping) -> dict:
    """Draw one configuration from ``space`` (see the module docstring for its format).

    Args:
        trial: the optuna trial to draw with.
        space: ``name -> spec``.

    Returns:
        ``name -> value``; ``"widths"`` strings such as ``"256,128"`` become tuples of ints.

    Raises:
        ValueError: for a malformed spec (unknown keys, a dict without ``choices`` or
            ``low``/``high``, non-scalar choices, inconsistent bounds).
    """
    parsed = _check_space(space)
    return {name: _widths(name, _suggest_one(trial, name, kind, arg)) for name, (kind, arg) in parsed.items()}


def _candidates(kind, arg) -> list:
    """The values of a parsed spec that are worth validating: all choices, or the ends of a range."""
    if kind == "choices":
        return list(arg)
    return [arg["low"], arg["high"]] if kind == "range" else [arg]


def _entry_error(e: Exception, name, value) -> Exception:
    """A ValueError or TypeError (as ``e``) whose message names the search-space entry that caused ``e``."""
    return (ValueError if isinstance(e, ValueError) else TypeError)(f"search space entry {name!r} = {value!r}: {e}")


def _in_spec(name, kind, arg, value):
    """``value`` as the search would record it if the parsed spec can produce it, else None."""
    if kind == "choices":
        if name == "widths" and isinstance(value, (list, tuple)):
            value = ",".join(str(int(w)) for w in value)
        same = [c for c in arg if c == value and (_is_number(c) and _is_number(value) or type(c) is type(value))]
        return same[0] if same else None
    if kind != "range" or not _is_number(value) or not arg["low"] <= value <= arg["high"]:
        return None
    if arg["int"] and float(value) != int(value):
        return None
    if arg["step"] is not None:
        steps = (value - arg["low"]) / arg["step"]
        if abs(steps - round(steps)) > 1e-9:
            return None
    return int(value) if arg["int"] else float(value)


# ------------------------------------------------------------------------------ studies
def _storage(storage):
    """An optuna storage (or None) from None, a path, a URL or a storage object."""
    if storage is None or isinstance(storage, optuna.storages.BaseStorage):
        return storage
    if isinstance(storage, str) and "://" in storage:
        return storage
    if not isinstance(storage, (str, os.PathLike)):
        raise TypeError("storage must be None, a path, a storage URL or an optuna storage, got "
                        f"{type(storage).__name__}")
    path = Path(storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".db":
        return f"sqlite:///{path}"
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    return JournalStorage(JournalFileBackend(str(path)))


def make_study(storage=None, name: str = "study", direction: str = "minimize", n_startup: int = 10,
               seed: Optional[int] = 0, constrained: bool = False, *, sampler=None, pruner=None) -> optuna.Study:
    """Create a study, or reopen the one of that name in ``storage``.

    Args:
        storage: None (in memory), a file (``*.db``: sqlite; anything else: an optuna journal file,
            which several processes can work on at once), a storage URL (containing ``"://"``,
            e.g. ``"postgresql://..."``) or an optuna storage object.
        name: study name.
        direction: ``"minimize"`` or ``"maximize"``.
        n_startup: random trials before the default sampler starts modelling.
        seed: seed of the default sampler. A new study uses it as given; a reopened one uses a seed
            derived from it and the number of trials the study already has, so that a resumed
            search does not propose the configurations of its first run again. Parallel workers
            must be given different seeds (None: unseeded).
        constrained: the default sampler treats the trials' ``user_attrs["constraints"]`` (a list
            of numbers, feasible if all are <= 0) as constraints.
        sampler: an optuna sampler to use instead of the default, a multivariate
            ``TPESampler``; ``n_startup``, ``seed`` and ``constrained`` are then not used.
        pruner: an optuna pruner (for objectives that report intermediate values).

    Returns:
        The study.

    Raises:
        ValueError: if a study of that name exists with another direction (e.g. a ``val_loss``
            search and a ``fom_tarp`` search under one name).
    """
    if direction not in ("minimize", "maximize"):
        raise unknown("direction", direction, ("minimize", "maximize"))
    with _quiet():
        study = optuna.create_study(study_name=name, storage=_storage(storage), direction=direction, pruner=pruner,
                                    load_if_exists=True)
        if [d.name.lower() for d in study.directions] != [direction]:
            raise ValueError(f"study {name!r} exists with direction {[d.name.lower() for d in study.directions]}, "
                             f"not {direction!r}: it belongs to a search with another objective; use another name "
                             "or storage")
        n_old = len(study.get_trials(deepcopy=False))
        if n_old:
            done = len(study.get_trials(deepcopy=False, states=_FINISHED))
            logger.info(f"study {name!r}: resuming ({done} finished trials)")
        if sampler is None:
            if seed is not None and n_old:
                seed = int(np.random.SeedSequence([int(seed), n_old]).generate_state(1)[0])
            kw = {"constraints_func": lambda t: t.user_attrs.get("constraints", [1.0])} if constrained else {}
            sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=seed, multivariate=True, **kw)
        study.sampler = sampler
    return study


def run_study(study: optuna.Study, objective: Callable, n_trials: int, *, catch=()) -> optuna.Study:
    """Run trials until the study has ``n_trials`` finished (complete or pruned) ones.

    Trials that other processes finish in the meantime count as well, so several workers on one
    study stop together; the trials still running at that moment are completed, so W workers end
    with at most ``n_trials + W - 1``.

    Args:
        study: the study (:func:`make_study`).
        objective: ``objective(trial) -> value``.
        n_trials: number of finished trials to reach; nothing runs if the study has them already.
        catch: exception classes that fail a trial without stopping the search (the trial is
            recorded as failed and does not count). After 10 failed trials in a row the search
            stops with a warning.

    Returns:
        The study.
    """
    if len(study.get_trials(deepcopy=False, states=_FINISHED)) >= n_trials:
        return study
    failed = [0]

    def guard(study, trial):
        failed[0] = failed[0] + 1 if trial.state == TrialState.FAIL else 0
        if failed[0] >= _MAX_FAILURES:
            warnings.warn(f"study {study.study_name!r}: {failed[0]} trials failed in a row; stopping the search",
                          RuntimeWarning, stacklevel=2)
            study.stop()

    with _quiet():
        study.optimize(objective, callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=_FINISHED), guard],
                       catch=catch, gc_after_trial=True)
    return study


def best_trials(study: optuna.Study, k: int = 3, feasible_only: bool = True) -> list:
    """The k best complete trials (feasible ones only for constrained studies)."""
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if feasible_only:
        done = [t for t in done if all(c <= 0 for c in t.user_attrs.get("constraints", [0.0]))]
    rev = study.direction == optuna.study.StudyDirection.MAXIMIZE
    return sorted(done, key=lambda t: t.value, reverse=rev)[:k]


# ------------------------------------------------------------------------------ trial models
def _trial_path(outdir, name, number) -> Path:
    return Path(outdir) / name / "trials" / f"{number:04d}.pt"


def _remember(study, number, model) -> None:
    _MODELS.setdefault(study, {})[number] = model


def _trim(study, k: int) -> None:
    """Keep only the ``k`` best of the models this process holds for ``study`` (feasible trials first)."""
    store = _MODELS.get(study)
    if not store or len(store) <= k:
        return
    n = len(study.get_trials(deepcopy=False))
    ranked = [t.number for t in best_trials(study, n)]
    ranked += [t.number for t in best_trials(study, n, feasible_only=False) if t.number not in ranked]
    keep = [number for number in ranked if number in store][:k]
    for number in [number for number in store if number not in keep]:
        del store[number]


def _find_model(study, number, outdir, name, device):
    """A trial's model from memory (moved to ``device`` unless that is None) or from ``outdir`` (loaded on
    ``device``, None: the CPU); None if it is in neither."""
    model = _MODELS.get(study, {}).get(number)
    if model is not None:
        return model.to(device) if device is not None and hasattr(model, "to") else model
    if outdir is not None and _trial_path(outdir, name, number).is_file():
        return load(_trial_path(outdir, name, number), "cpu" if device is None else device)
    return None


def load_trial(study: optuna.Study, number: Optional[int] = None, outdir=None, name: Optional[str] = None,
               device=None):
    """The model a trial trained: the NLE of a :func:`tune_nle` trial or the compressor of a
    :func:`tune_compressor` trial.

    It is taken from memory if this process trained it and still holds it (see the module
    docstring), otherwise from ``outdir/name/trials/NNNN.pt``.

    Args:
        study: the study returned by the search.
        number: trial number (default: the best feasible trial).
        outdir: the search's output folder, if it had one.
        name: the search's name (default: the study's name).
        device: torch device to put the model on. None (default): a model in memory stays on the
            device it is on, a model read from disk is loaded on the CPU. A model in memory is
            the object the search holds (and :func:`tune_compressor` returned), not a copy: with a
            device it is moved in place.

    Returns:
        The model.

    Raises:
        ValueError: if the study has no such trial (or, without ``number``, no complete feasible
            one), or the model is not in memory and no ``outdir`` is given.
        FileNotFoundError: if the model is neither in memory nor in ``outdir``.

    Example:
        >>> study = tune_nle(s_train, theta_train, n_trials=30, outdir="hpt")
        >>> ensemble = Ensemble([load_trial(study, t.number, "hpt") for t in best_trials(study, 3)])
    """
    device = None if device is None else resolve_device(device)
    name = study.study_name if name is None else name
    if number is None:
        best = best_trials(study, 1)
        if not best:
            raise ValueError(f"study {name!r} has no complete feasible trial; pass a trial number (see "
                             "best_trials(study, feasible_only=False))")
        number = best[0].number
    elif number not in [t.number for t in study.get_trials(deepcopy=False)]:
        raise ValueError(f"study {name!r} has no trial {number}")
    model = _find_model(study, number, outdir, name, device)
    if model is None:
        if outdir is None:
            raise ValueError(f"the model of trial {number} of study {name!r} is not in memory (this process did not "
                             "train it, or it is not among the best that are kept); pass the search's outdir to "
                             "load it from disk")
        raise FileNotFoundError(f"the model of trial {number} of study {name!r} is neither in memory nor at "
                                f"{_trial_path(outdir, name, number)} (pruned and failed trials have none)")
    return model


# ------------------------------------------------------------------------------ compressors
def tune_compressor(factory: Callable[[dict], object], space: Mapping, x_trn=None, t_trn=None, x_val=None, t_val=None,
                    n_trials: int = 20, storage=None, name: str = "compressor", outdir=None, n_startup: int = 25,
                    seed: Optional[int] = 0, fixed: Optional[Mapping] = None, *, sampler=None, catch=(),
                    fit_kwargs: Optional[Mapping] = None):
    """Minimise a compressor's validation loss.

    Before the first trial (and before the study is created) the factory is called with the first
    value of every search-space entry and then once for each further value an entry can take (all
    choices, both ends of a range), so a misspelt key or an invalid value fails at once.

    Args:
        factory: ``factory(config) -> compressor``, e.g. ``lambda c: VMIMCompressor(n_out, c)``.
        space: search space; ``SPACES["mse"]``, ``SPACES["vmim"]`` and ``SPACES["imnn"]`` are
            starting points.
        x_trn, t_trn, x_val, t_val: training and validation data, passed to ``compressor.fit``
            (all None for compressors trained on a fiducial set; without a validation set the
            compressor holds one out).
        n_trials: number of finished trials to reach (an existing study is resumed).
        storage: where the study lives, see :func:`make_study` (None: in memory).
        name: study name; trials are saved to ``outdir/name/trials/``.
        outdir: folder for the trials' checkpoints (``NNNN.pt``) and records (``NNNN.json``:
            configuration, validation loss, seconds, seed). None: nothing is written.
        n_startup: random trials before TPE.
        seed: seed of the sampler and of the trials: trial ``n`` trains with the torch seed
            ``1000 * seed + n``, so a search is reproducible (None: unseeded). A compressor that
            the factory builds without a seed of its own (its ``seed`` attribute is None) is given
            the trial's seed, which makes the validation set it holds out reproducible as well.
            Each trial's seed is stored in its ``user_attrs["seed"]``.
        fixed: train exactly this configuration as one new trial instead of searching (e.g. to
            retrain a tuned stage with another seed): the fixed values of ``space``, overridden by
            ``fixed``, whose keys need not be in ``space``. This also works on a study that has
            trials already.
        sampler: an optuna sampler instead of the default (:func:`make_study`).
        catch: exception classes that fail a trial without stopping the search (:func:`run_study`).
        fit_kwargs: further keyword arguments of ``compressor.fit``, e.g. ``{"atoms": atoms}``.

    Returns:
        (config, compressor, study): of the best trial, or of the new trial with ``fixed``. The
        compressor comes from memory if this process trained it, else from ``outdir`` (on the CPU);
        if it is in neither (a resumed study without ``outdir``), the configuration is trained
        again with the seed its trial was trained with, whatever ``seed`` is now; a
        ``RuntimeWarning`` says so if that does not reproduce the trial's validation loss.

    Raises:
        ValueError: for a malformed space, a search-space value that the factory rejects, or if no
            trial reached a finite validation loss.
        TypeError: for a search-space or ``fixed`` key that the factory rejects.
    """
    parsed = _check_space(space)
    fit_kwargs = dict(fit_kwargs or {})
    if fixed is not None and not isinstance(fixed, Mapping):
        raise TypeError(f"fixed must be a dict name -> value, got {type(fixed).__name__}")
    constants = {k: _widths(k, arg) for k, (kind, arg) in parsed.items() if kind == "fixed"}
    if fixed is None:
        cfg = {k: _widths(k, _candidates(kind, arg)[0]) for k, (kind, arg) in parsed.items()}
    else:
        cfg = {**constants, **{k: _widths(k, v) for k, v in fixed.items()}}
        unset = [k for k, (kind, _) in parsed.items() if kind != "fixed" and k not in fixed]
        if unset:
            warnings.warn(f"fixed gives no value for the searched keys {unset}; they are left at the compressor's "
                          "defaults", stacklevel=2)
    with torch.random.fork_rng():                           # a configuration the factory rejects fails here
        factory(dict(cfg))
        for k, (kind, arg) in parsed.items() if fixed is None else ():
            for v in _candidates(kind, arg)[1:]:            # every other value of the entry, the rest unchanged
                try:
                    factory({**cfg, k: _widths(k, v)})
                except (TypeError, ValueError) as e:
                    raise _entry_error(e, k, v) from e
    study = make_study(storage, name, "minimize", n_startup, seed, sampler=sampler)

    def train(cfg, trial_seed):
        t0 = time.time()
        with seeded(trial_seed):
            comp = factory(dict(cfg))
            if trial_seed is not None and getattr(comp, "seed", 0) is None:
                with contextlib.suppress(AttributeError):   # the seed of the compressor's own draws (validation split)
                    comp.seed = trial_seed
            loss = float(comp.fit(x_trn, t_trn, x_val, t_val, **fit_kwargs))
        return comp, loss, time.time() - t0

    def finish(trial, cfg, comp, loss, seconds, **extra):
        """Record a trained trial; True if its loss is finite."""
        trial.set_user_attr("seconds", seconds)
        trial.set_user_attr("config", _jsonable(cfg))
        trial.set_user_attr("seed", _trial_seed(seed, trial.number))
        ok = bool(np.isfinite(loss))
        if ok:
            _remember(study, trial.number, comp)
        if outdir is not None:
            if ok:
                save(comp, _trial_path(outdir, name, trial.number))
            _write_json(_trial_path(outdir, name, trial.number).with_suffix(".json"),
                        {"trial": trial.number, "config": cfg, "val_loss": loss, "seconds": seconds,
                         "seed": _trial_seed(seed, trial.number), **extra})
        return ok

    if fixed is not None:
        params = {k: _in_spec(k, *parsed[k], v) for k, v in fixed.items() if k in parsed and parsed[k][0] != "fixed"}
        params = {k: v for k, v in params.items() if v is not None}
        with _quiet():
            study.enqueue_trial(params, user_attrs={"fixed": True})
            trial = study.ask()
            if trial.user_attrs.get("fixed"):                # the searched keys become the trial's parameters
                for k in params:
                    _suggest_one(trial, k, *parsed[k])
        try:
            comp, loss, seconds = train(cfg, _trial_seed(seed, trial.number))
            ok = finish(trial, cfg, comp, loss, seconds, fixed=True)
        except BaseException:
            study.tell(trial, state=TrialState.FAIL)
            raise
        if ok:
            study.tell(trial, loss)
        else:
            study.tell(trial, state=TrialState.PRUNED)
            warnings.warn(f"the fixed configuration gave the validation loss {loss}", RuntimeWarning, stacklevel=2)
        _trim(study, 1)
        return _jsonable(cfg), comp, study

    def objective(trial):
        _trim(study, 1)
        cfg = suggest(trial, space)
        comp, loss, seconds = train(cfg, _trial_seed(seed, trial.number))
        if not finish(trial, cfg, comp, loss, seconds):
            raise optuna.TrialPruned()
        return loss

    run_study(study, objective, n_trials, catch=catch)
    _trim(study, 1)
    best = best_trials(study, 1, feasible_only=False)
    if not best:
        raise ValueError(f"study {name!r}: none of the {len(study.trials)} trials reached a finite validation loss")
    best, cfg = best[0], best[0].user_attrs["config"]
    comp = _find_model(study, best.number, outdir, name, None)
    if comp is None:
        logger.info(f"study {name!r}: the compressor of the best trial {best.number} is neither in memory nor on "
                    "disk; training its configuration again")
        trial_seed = best.user_attrs.get("seed", _trial_seed(seed, best.number))   # the seed the trial trained with
        comp, loss, _ = train(cfg, trial_seed)
        same = bool(np.isclose(loss, best.value, rtol=_RETRAIN_RTOL, atol=0.0))
        if not same:
            warnings.warn(f"study {name!r}: training the configuration of the best trial {best.number} again (seed "
                          f"{trial_seed}) gave the validation loss {loss}, the trial had {best.value}: the returned "
                          "compressor is not the one the search found (other data, package versions or hardware, or "
                          "an unseeded trial). Give the search an outdir to keep the compressors it trains",
                          RuntimeWarning, stacklevel=2)
        _remember(study, best.number, comp)
        if outdir is not None and same:
            save(comp, _trial_path(outdir, name, best.number))
    return cfg, comp, study


# ------------------------------------------------------------------------------ NLE
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


def _names(params, names, what) -> list:
    """``params`` (a name or a list of names) as a list, checked against ``names``."""
    params = [params] if isinstance(params, str) else [str(p) for p in params]
    for p in params:
        if p not in names:
            raise unknown(f"parameter in {what}", p, names)
    return params


def _check_eval(ev, n_x: int, n_theta: int, fom_params, tarp_params) -> tuple:
    """Validate the ``fom_tarp`` arguments; returns (parameter names, fom_params, tarp_params)."""
    if not isinstance(ev, EvalSet):
        raise TypeError(f"evaluation must be an EvalSet, got {type(ev).__name__}")
    s = as_2d(ev.s, "evaluation.s", n_x, what="simulations")
    truth = as_2d(ev.truth, "evaluation.truth", n_theta, what="simulations")
    if len(s) != len(truth):
        raise ValueError(f"evaluation.s has {len(s)} rows but evaluation.truth has {len(truth)}")
    check_finite(s, "evaluation.s")
    check_finite(truth, "evaluation.truth")
    names = [str(n) for n in ev.param_names]
    if len(names) != n_theta or len(set(names)) != len(names):
        raise ValueError(f"evaluation.param_names must be {n_theta} different names (one per column of "
                         f"evaluation.truth), got {names}")
    constant = [names[i] for i in np.flatnonzero(~(truth.max(0) > truth.min(0)))]
    if constant:
        raise ValueError(f"evaluation.truth has the same value in all {len(truth)} simulation(s) for the parameter(s) "
                         f"{constant}: the TARP constraints scale each parameter by the range of its truths, so the "
                         "evaluation set needs different truths for every parameter (some tens of simulations for a "
                         "meaningful test)")
    priors = list(ev.prior) if isinstance(ev.prior, (list, tuple)) else [ev.prior]
    if isinstance(ev.prior, (list, tuple)) and len(priors) != len(s):
        raise ValueError(f"evaluation.prior holds {len(priors)} priors for {len(s)} simulations")
    for p in priors + ([] if ev.fom_prior is None else [ev.fom_prior]):
        if getattr(p, "ndim", n_theta) != n_theta:
            raise ValueError(f"a prior of the evaluation set has {p.ndim} parameters, the flow has {n_theta}")
    if ev.fom_prior is not None and isinstance(ev.prior, (list, tuple)):
        raise ValueError("evaluation.fom_prior (reweighting) needs a single sampling prior, not one per simulation")
    if ev.init_sd is not None and np.shape(ev.init_sd) not in ((n_theta,), (len(s), n_theta)):
        raise ValueError(f"evaluation.init_sd must have shape ({n_theta},) or ({len(s)}, {n_theta}), got "
                         f"{np.shape(ev.init_sd)}")
    fom_params = _names(fom_params, names, "fom_params")
    if not fom_params:
        raise ValueError("objective 'fom_tarp' needs fom_params, the parameters of the figure of merit")
    tarp_params = [] if tarp_params is None else _names(tarp_params, names, "tarp_params")
    return names, fom_params, tarp_params or fom_params


def _objective_fom(samples, ev: EvalSet, fom_idx):
    """Mean log FoM over the eval set; reweighted to ``ev.fom_prior`` and restricted to truths in its
    support if given (a NaN FoM there counts as the worst finite one, not as missing)."""
    if ev.fom_prior is None:
        lf = log_fom(samples, fom_idx)
    else:
        w = importance_weights(samples, ev.prior, ev.fom_prior)
        truth = to_tensor(as_numpy(ev.truth), dtype=torch.float32, device=getattr(ev.fom_prior, "device", "cpu"))
        lf = log_fom(samples, fom_idx, w)[as_numpy(ev.fom_prior.inside(truth), bool)]
    nan = np.isnan(lf)
    if nan.all():
        return -np.inf
    return float(np.where(nan, np.nanmin(lf), lf).mean())


def _prune_threshold(study, quantile: Optional[float], min_trials: int) -> float:
    """Validation loss above which a ``fom_tarp`` trial is pruned: the ``quantile`` of the finite
    validation losses of all finished trials, pruned ones included; inf while there are fewer than
    ``min_trials`` of them or with ``quantile=None``."""
    if quantile is None:
        return np.inf
    vals = [t.user_attrs.get("val_loss") for t in study.get_trials(deepcopy=False, states=_FINISHED)]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.quantile(vals, quantile)) if len(vals) >= max(min_trials, 1) else np.inf


def _violation(value) -> float:
    """A constraint value as a float; undefined ones (None, NaN) count as violated."""
    value = np.nan if value is None else float(value)
    return value if np.isfinite(value) else 1.0


def _tarp_violation(value, tol) -> float:
    """The constraint ``value >= -tol`` in optuna's form (feasible if <= 0)."""
    return 1.0 if value is None or tol is None else _violation(-float(value) - float(tol))


def tune_nle(s_trn, t_trn, s_val=None, t_val=None, n_trials: int = 20, storage=None, name: str = "nle", outdir=None,
             space: Optional[Mapping] = None, train: Optional[Mapping] = None,
             objective: Union[str, Callable] = "val_loss", evaluation: Optional[EvalSet] = None, mcmc=None,
             fom_params: Sequence[str] = (), tarp_params: Optional[Sequence[str]] = None,
             prune_quantile: Optional[float] = 0.75, prune_min_trials: int = 10, n_references: int = 50,
             n_bootstrap: int = 200, tol_min: float = 0.02, n_startup: int = 15, device="cpu",
             seed: Optional[int] = 0, *, sampler=None, catch=()) -> optuna.Study:
    """Hyperparameter search for an :class:`~sbikit.nle.NLE`.

    The search space, the training settings and (for ``fom_tarp``) the evaluation set are checked
    before the first trial: a key that is not a field of :class:`~sbikit.nle.FlowConfig` or
    :class:`~sbikit.nle.TrainConfig` raises instead of being searched without effect.

    With ``outdir``, every trial's flow is saved to ``outdir/name/trials/NNNN.pt`` with a JSON
    record (config, validation loss and, for ``fom_tarp``, the standard metrics record and
    posterior arrays). :func:`load_trial` returns a trial's flow.

    Args:
        s_trn, t_trn: (N, n_x) summaries and (N, n_theta) parameters of the training set.
        s_val, t_val: the validation set (None: every flow holds one out of the training set).
        n_trials: number of finished trials to reach (an existing study is resumed).
        storage: where the study lives, see :func:`make_study` (None: in memory).
        name: study name and output subfolder.
        outdir: output folder. None: nothing is written and the best flows stay in memory.
        space: search space over :class:`~sbikit.nle.FlowConfig` and
            :class:`~sbikit.nle.TrainConfig` fields (default ``SPACES["nle"]``).
        train: fixed training settings (searched ones override them).
        objective: ``"val_loss"``, ``"fom_tarp"`` (see the module docstring), or a callable
            ``(nle, trial) -> value`` or ``-> (value, constraints)`` that is minimised; the trial
            is feasible if every constraint is <= 0, and a non-finite value prunes it.
        evaluation: the :class:`EvalSet` (``fom_tarp``).
        mcmc: :class:`~sbikit.sampling.MCMCConfig` or dict for the evaluation posteriors.
        fom_params: parameters of the figure of merit to maximise (``fom_tarp``).
        tarp_params: parameters of the second TARP constraint (default ``fom_params``); the first
            is always on all parameters.
        prune_quantile, prune_min_trials: (``fom_tarp``) once ``prune_min_trials`` trials have
            finished, a flow whose validation loss is above this quantile of the validation losses
            of all finished trials (pruned ones included) skips the MCMC and is pruned, so about
            ``1 - prune_quantile`` of the trials are. None: no pruning.
        n_references, n_bootstrap, tol_min: TARP settings (:func:`sbikit.metrics.evaluate`).
        n_startup: random trials before TPE.
        device: torch device for training and sampling.
        seed: seed of the sampler and of the trials: trial ``n`` is initialised and trained with
            the torch seed ``1000 * seed + n``, so a search is reproducible (None: unseeded).
        sampler: an optuna sampler instead of the default (:func:`make_study`); with constraints it
            has to read them from ``trial.user_attrs["constraints"]`` itself.
        catch: exception classes that fail a trial without stopping the search (:func:`run_study`).

    Returns:
        The optuna study; :func:`best_trials` lists its best (feasible) trials.

    Raises:
        TypeError: for a search-space or ``train`` key that no config has.
        ValueError: for a malformed space, invalid config values, or inconsistent data or
            evaluation arguments.
    """
    custom = callable(objective)
    if not custom and objective not in ("val_loss", "fom_tarp"):
        raise unknown("objective", objective, ("val_loss", "fom_tarp"))
    constrained = objective == "fom_tarp"
    device = resolve_device(device)
    s2, t2 = check_pair(s_trn, t_trn, "s_trn", "t_trn")
    n_x, n_theta = s2.shape[1], t2.shape[1]
    del s2, t2
    if (s_val is None) != (t_val is None):
        raise ValueError("give both s_val and t_val, or neither")
    if s_val is not None:
        check_pair(s_val, t_val, "s_val", "t_val", n_x, n_theta)

    # the search space: every key must be a config field, every value it can take a valid one
    flow_fields, train_fields = FlowConfig.fields(), TrainConfig.fields()
    parsed = _check_space(SPACES["nle"] if space is None else space)
    space = dict(SPACES["nle"] if space is None else space)
    valid = [*flow_fields, *train_fields]
    bad = [k for k in parsed if k not in valid]
    if bad:
        raise TypeError("tune_nle: the search space has keys that are neither FlowConfig nor TrainConfig fields: "
                        + "; ".join(f"{k!r}{did_you_mean(k, valid)}" for k in bad) + f". Fields: {valid}")
    base_train = TrainConfig.make(train).to_dict()

    def split(cfg):
        return ({k: v for k, v in cfg.items() if k in flow_fields},
                {**base_train, **{k: v for k, v in cfg.items() if k in train_fields}})

    for k, (kind, arg) in parsed.items():
        if kind == "fixed" and isinstance(arg, (list, tuple, set, np.ndarray)):
            raise ValueError(f"search space entry {k!r} is the sequence {arg!r}, which would be one fixed value; to "
                             "search over these values write {'choices': [...]}")

    def check(cfg, k=None):
        """Build both configs from ``cfg``; an error names the entry ``k`` if one is given."""
        flow_cfg, tr = split(cfg)
        try:
            FlowConfig.make(flow_cfg), TrainConfig.make(tr)
        except (TypeError, ValueError) as e:
            if k is None:
                raise
            raise _entry_error(e, k, cfg[k]) from e

    first = {k: _candidates(kind, arg)[0] for k, (kind, arg) in parsed.items()}
    try:
        check(first)
    except (TypeError, ValueError):                         # blame the entry whose first value breaks the configuration
        for i, k in enumerate(first):
            check(dict(list(first.items())[:i + 1]), k)
        raise
    for k, (kind, arg) in parsed.items():
        for v in _candidates(kind, arg)[1:]:                # every other value of the entry, the rest unchanged
            check({**first, k: v}, k)

    if constrained:
        if evaluation is None:
            raise ValueError("objective 'fom_tarp' needs evaluation, an EvalSet")
        names, fom_params, tarp_params = _check_eval(evaluation, n_x, n_theta, fom_params, tarp_params)
        fom_idx = [names.index(p) for p in fom_params]
        mcmc = MCMCConfig.make(mcmc)
        if prune_quantile is not None and not 0.0 < prune_quantile <= 1.0:
            raise ValueError(f"prune_quantile must be in (0, 1] or None, got {prune_quantile}")
    elif evaluation is not None or mcmc is not None or len(fom_params) or tarp_params is not None:
        warnings.warn("evaluation, mcmc, fom_params and tarp_params are used only by objective='fom_tarp'; they are "
                      "ignored here", stacklevel=2)

    folder = None if outdir is None else Path(outdir) / name / "trials"
    study = make_study(storage, name, "maximize" if constrained else "minimize", n_startup, seed,
                       constrained or custom, sampler=sampler)

    def run(trial):
        cfg = suggest(trial, space)
        flow_cfg, tr = split(cfg)
        t0 = time.time()
        nle = NLE(n_x, n_theta, flow_cfg, device)
        val, epochs = nle.fit(s_trn, t_trn, s_val, t_val, tr, seed=_trial_seed(seed, trial.number))
        rec = {"trial": trial.number, "config": cfg, "train": tr, "val_loss": val, "epochs": epochs,
               "train_seconds": time.time() - t0, "seed": _trial_seed(seed, trial.number)}

        def record(**more):
            rec.update(more)
            if folder is not None:
                _write_json(folder / f"{trial.number:04d}.json", rec)

        trial.set_user_attr("val_loss", _jsonable(val))
        if not np.isfinite(val):
            record(pruned="no finite validation loss")
            raise optuna.TrialPruned()
        if folder is not None:
            save(nle, folder / f"{trial.number:04d}.pt")
        if custom:
            out = objective(nle, trial)
            value, cons = out if isinstance(out, (tuple, list)) and len(out) == 2 else (out, None)
            value = float(value)
            cons = [0.0] if cons is None else [_violation(c) for c in np.atleast_1d(as_numpy(cons))]
            trial.set_user_attr("constraints", cons)
            if not np.isfinite(value):
                record(objective=value, constraints=cons, pruned="the objective is not finite")
                raise optuna.TrialPruned()
            record(objective=value, constraints=cons, seconds=time.time() - t0)
        elif not constrained:
            value = val
            record()
        else:
            if val > _prune_threshold(study, prune_quantile, prune_min_trials):
                record(pruned=f"val_loss above the {prune_quantile} quantile")
                raise optuna.TrialPruned()
            with warnings.catch_warnings():       # non-convergence is a constraint of the search, recorded below
                warnings.filterwarnings("ignore", message=".*did not converge.*", category=RuntimeWarning)
                post = sample_posteriors(nle, evaluation.s, evaluation.prior, mcmc, init_centre=evaluation.truth,
                                         init_sd=evaluation.init_sd, device=device, verbose=False)
            metrics, arrays = evaluate(post.samples, evaluation.truth, names, {"fom": fom_params},
                                       {"fom": tarp_params}, n_references, n_bootstrap, tol_min,
                                       0 if seed is None else seed, device)
            value = _objective_fom(post.samples, evaluation, fom_idx)
            conv = float(post.converged.mean())
            cons = []
            for key in ("all", "fom"):
                t = metrics["tarp"][key]
                cons += [_tarp_violation(t["confidence"], t["tol_confidence"]),
                         _tarp_violation(t["bias"], t["tol_bias"])]
            cons.append(0.0 if conv >= 1.0 else 1.0 - conv + 1e-3)
            trial.set_user_attr("constraints", cons)
            record(objective=value, constraints=cons, converged=conv, metrics=metrics,
                   mcmc_steps=post.diagnostics["steps"], seconds=time.time() - t0)
            if folder is not None:
                np.savez(folder / f"{trial.number:04d}_posteriors.npz", **arrays, converged=post.converged,
                         rhat_max=post.diagnostics["rhat_max"])
            value = value if np.isfinite(value) else -1e30
        _remember(study, trial.number, nle)
        return value

    def obj(trial):
        _trim(study, KEEP_IN_MEMORY)
        with seeded(_trial_seed(seed, trial.number)):
            return run(trial)

    run_study(study, obj, n_trials, catch=catch)
    _trim(study, KEEP_IN_MEMORY)
    return study
