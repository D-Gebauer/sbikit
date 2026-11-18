"""Checkpoints and run records.

Checkpoints are ``torch.save`` files of plain states (dicts of tensors, numbers, strings and lists),
so they load with ``weights_only=True`` and never execute pickled code. Each stores the package
versions it was written with; :func:`read_meta` shows them without building the object.

:func:`load` finds the class of a checkpoint by its ``kind``: the built-in ones (``"nle"``,
``"ensemble"``, ``"prior"``), every registered compressor (:func:`sbikit.compress.register`) and
scaler (:func:`sbikit.scaling.register_scaler`), and anything added with :func:`register_loader`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import tomllib
import warnings
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from ._utils import Registry, did_you_mean, plain, resolve_device

__all__ = ["FORMAT", "LOADERS", "register_loader", "save", "load", "read_meta", "versions", "run_record", "file_md5"]

FORMAT = "sbikit/1"
LOADERS = Registry("checkpoint kind")
_SAVEABLE = ("NLE, Ensemble, Prior, the compressors, the scalers, or any object with a state() method that returns "
             "a dict with a 'kind' entry")


def versions() -> dict:
    """Versions of Python, torch, numpy, sbikit and its dependencies (None if not importable)."""
    out = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__)}
    for mod in ("sbikit", "sbi", "zuko", "torchemcee", "optuna"):
        try:
            out[mod] = str(__import__(mod).__version__)
        except Exception:                                   # pragma: no cover
            out[mod] = None
    return out


# ------------------------------------------------------------------------------ loaders
def register_loader(kind: str, fn: Optional[Callable] = None, overwrite: bool = False):
    """Tell :func:`load` how to rebuild objects whose ``state()["kind"]`` is ``kind``.

    Compressors registered with :func:`sbikit.compress.register` and scalers registered with
    :func:`sbikit.scaling.register_scaler` need no loader; this is for other classes.

    Args:
        kind: the ``kind`` entry of the object's state.
        fn: ``fn(state, device) -> object``. Without ``fn`` this returns a decorator.
        overwrite: replace an existing loader of that kind.

    Example:
        >>> class Calibration:
        ...     def __init__(self, shift): self.shift = shift
        ...     def state(self): return {"kind": "calibration", "shift": self.shift}
        >>> register_loader("calibration", lambda state, device: Calibration(state["shift"]))
        >>> save(Calibration(0.1), "cal.pt"); cal = load("cal.pt")
    """
    return LOADERS.register(kind, fn, overwrite=overwrite)


@register_loader("nle")
def _load_nle(state, device):
    from .nle import NLE
    return NLE.from_state(state, device)


@register_loader("ensemble")
def _load_ensemble(state, device):
    from .nle import Ensemble
    return Ensemble.from_state(state, device)


@register_loader("prior")
def _load_prior(state, device):
    from .priors import Prior
    return Prior.from_spec(state["spec"], device)


def _loader(kind) -> Optional[Callable]:
    """The function that rebuilds a state of this kind, or None."""
    from .compress import REGISTRY
    from .scaling import SCALERS, scaler_from_state
    if kind in LOADERS:
        return LOADERS[kind]
    if kind in REGISTRY:
        return REGISTRY[kind].from_state
    if kind in SCALERS:
        return lambda state, device: scaler_from_state(state)
    return None


def _known_kinds() -> list:
    from .compress import REGISTRY
    from .scaling import SCALERS
    return sorted({*LOADERS, *REGISTRY, *SCALERS})


# ------------------------------------------------------------------------------ checkpoints
def _not_plain(obj, where: str) -> list:
    """Descriptions of the entries of a nested state that a checkpoint cannot hold."""
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes, torch.Tensor)):
        return []
    if isinstance(obj, dict):
        return [bad for k, v in obj.items() for bad in _not_plain(v, f"{where}[{k!r}]")]
    if isinstance(obj, list):
        return [bad for i, v in enumerate(obj) for bad in _not_plain(v, f"{where}[{i}]")]
    return [f"{where} ({type(obj).__module__}.{type(obj).__qualname__})"]


def save(obj, path, meta: Optional[dict] = None) -> None:
    """Save an NLE, an Ensemble, a Prior, a compressor, a scaler, or any other object with a
    ``state()`` (see :func:`register_loader` for making such an object loadable).

    numpy scalars, numpy arrays, tuples, sets and paths in the state or in ``meta`` are stored as
    Python scalars, tensors, lists and strings, and dataclasses as dicts, so :func:`load` and
    :func:`read_meta` return them in that form.

    Args:
        obj: the object.
        path: file to write (parent folders are created; the write is atomic).
        meta: optional dict stored next to the state (e.g. the data cuts or the run's name); read
            it back with :func:`read_meta` or ``load(path, return_meta=True)``.

    Raises:
        TypeError: if ``obj`` cannot be saved, or if the state or ``meta`` hold values that a
            checkpoint cannot store (custom objects; the message names them).
    """
    from .priors import Prior
    if isinstance(obj, Prior):
        state = {"kind": "prior", "spec": obj.spec()}
    elif callable(getattr(obj, "state", None)):
        state = obj.state()
    else:
        raise TypeError(f"cannot save a {type(obj).__name__}: sbikit.io.save takes {_SAVEABLE}")
    if not isinstance(state, Mapping) or not isinstance(state.get("kind"), str):
        raise TypeError(f"cannot save a {type(obj).__name__}: its state() must return a dict with a 'kind' string; "
                        f"sbikit.io.save takes {_SAVEABLE}")
    if meta is not None and not isinstance(meta, Mapping):
        raise TypeError(f"meta must be a dict, got {type(meta).__name__}")
    state, meta = plain(state), plain(dict(meta or {}))
    if _loader(state["kind"]) is None:
        warnings.warn(f"no loader is registered for kind {state['kind']!r}: sbikit.io.load can read this checkpoint "
                      "only after sbikit.io.register_loader (or sbikit.compress.register) for that kind", stacklevel=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": FORMAT, "kind": state["kind"], "state": state, "versions": versions(), "meta": meta}, tmp)
    try:                                                    # what was written must be readable by load()
        torch.load(tmp, map_location="cpu", weights_only=True)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        bad = _not_plain(state, "state") + _not_plain(meta, "meta")
        detail = "; ".join(bad) if bad else str(e).splitlines()[0]
        raise TypeError(f"cannot save this {type(obj).__name__}: a checkpoint holds only numbers, strings, lists, "
                        f"dicts and tensors, not {detail}") from e
    tmp.replace(path)


def _read(path) -> dict:
    """The raw checkpoint dict, checked to be an sbikit checkpoint in a format this version reads."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such checkpoint: {path}")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        try:                                                # the pickled types that weights_only refuses
            found = sorted(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        except Exception:
            found = []
        first = (str(e).splitlines() or [""])[0]
        holds = (f"it holds objects of the types {found}; a checkpoint saved with numpy values or paths in its meta "
                 "or state is such a file, and torch.load(path, weights_only=False), for a file you trust, still "
                 "returns its contents" if found else "torch cannot unpickle it" if "Weights only load failed" in first
                 else f"{type(e).__name__}: {first[:200]}")
        raise ValueError(f"{path} cannot be read as an sbikit checkpoint, a file of plain values and tensors "
                         f"({holds})") from e
    fmt = ck.get("format") if isinstance(ck, dict) else None
    if not isinstance(fmt, str) or not fmt.startswith("sbikit/") or "kind" not in ck or "state" not in ck:
        what = f"a dict with the keys {sorted(map(str, ck))[:8]}" if isinstance(ck, dict) else f"a {type(ck).__name__}"
        raise ValueError(f"{path} is not an sbikit checkpoint (it holds {what}); checkpoints are written by "
                         "sbikit.io.save")
    if fmt != FORMAT:
        number = fmt.split("/", 1)[1]
        newer = number.isdigit() and int(number) > int(FORMAT.split("/")[1])
        raise ValueError(f"{path} has the checkpoint format {fmt!r}, written by sbikit "
                         f"{(ck.get('versions') or {}).get('sbikit')}; the installed sbikit "
                         f"{versions()['sbikit']} reads {FORMAT!r}" + (". Update sbikit to load it" if newer else ""))
    return ck


def _info(ck: dict) -> dict:
    return {"kind": ck["kind"], "format": ck["format"], "versions": dict(ck.get("versions") or {}),
            "meta": dict(ck.get("meta") or {})}


def read_meta(path) -> dict:
    """What a checkpoint holds, without building the object.

    Args:
        path: a file written by :func:`save`.

    Returns:
        A dict with ``kind`` (e.g. ``"nle"``), ``format``, ``versions`` (the package versions the
        file was written with) and ``meta`` (the dict given to :func:`save`).

    Raises:
        ValueError: if the file is not an sbikit checkpoint or has another format (a newer sbikit
            wrote it).
    """
    return _info(_read(path))


def _version_note(written: Mapping) -> str:
    """How the versions a checkpoint was written with differ from the installed ones."""
    now = versions()
    diff = [k for k in now if k in written and written[k] != now[k]]
    if not diff:
        return "it was written with the installed package versions"
    return ("it was written with " + ", ".join(f"{k} {written[k]}" for k in diff)
            + "; installed: " + ", ".join(f"{k} {now[k]}" for k in diff))


def load(path, device="cpu", *, return_meta: bool = False):
    """Load an object saved with :func:`save`; networks are put on ``device`` in eval mode.

    Args:
        path: the checkpoint file.
        device: torch device (``"auto"``: CUDA if available).
        return_meta: also return the checkpoint's description (see :func:`read_meta`).

    Returns:
        The object, or ``(object, info)`` with ``return_meta=True``.

    Raises:
        ValueError: if the file is not an sbikit checkpoint, has another format (a newer sbikit
            wrote it), or holds a kind that no loaded module knows (import the module that defines
            a custom class first).
        Exception: if the object cannot be rebuilt from its state, the error of the class is
            raised again with the package versions of the checkpoint next to the installed ones.
    """
    device = resolve_device(device)
    ck = _read(path)
    kind, state = ck["kind"], ck["state"]
    fn = _loader(kind)
    if fn is None:
        known = _known_kinds()
        raise ValueError(f"{path} holds an object of kind {kind!r}, which is not known here"
                         f"{did_you_mean(kind, known)} (known: {known}). For a custom class, import the module that "
                         "defines and registers it (sbikit.compress.register, sbikit.io.register_loader) before "
                         "loading")
    try:
        obj = fn(state, device)
    except Exception as e:
        msg = (f"{path}: could not rebuild the {kind!r} object ({type(e).__name__}: {e}); "
               f"{_version_note(_info(ck)['versions'])}")
        try:
            err = type(e)(msg)
        except Exception:
            err = RuntimeError(msg)
        raise err from e
    return (obj, _info(ck)) if return_meta else obj


# ------------------------------------------------------------------------------ run records
def file_md5(path, chunk: int = 1 << 24) -> str:
    """md5 hex digest of a file, read in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _run_git(path, *args) -> Optional[str]:
    """Output of a git command run in ``path``, or None if it fails."""
    try:
        r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                       # pragma: no cover
        return None


def _git(path, package: Optional[str] = None) -> Optional[str]:
    """Short hash of the checkout that holds ``path`` (``+dirty`` with uncommitted changes), or None.

    With ``package``, ``path`` is an installed package folder and the hash is given only if the
    folder belongs to that package's own checkout: the checkout's ``pyproject.toml`` names the
    project and tracks the package's ``__init__.py``. A package installed inside some other
    repository (e.g. in a virtual environment in a project folder) gives None.
    """
    if package is not None:
        top = _run_git(path, "rev-parse", "--show-toplevel")
        try:
            project = tomllib.loads((Path(top) / "pyproject.toml").read_text())["project"]["name"]
        except Exception:
            return None
        if (project.lower().replace("_", "-") != package.lower().replace("_", "-")
                or _run_git(path, "ls-files", "--error-unmatch", "__init__.py") is None):
            return None
    head = _run_git(path, "rev-parse", "--short", "HEAD")
    if not head:
        return None
    return head + ("+dirty" if _run_git(path, "status", "--porcelain", "--untracked-files=no") else "")


def _jsonable(obj):
    """``obj`` as values that strict JSON can hold: configs and dataclasses become dicts, arrays and
    tensors lists, numpy scalars Python scalars, non-finite floats None, anything else a string."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Mapping):
        return {(k if isinstance(k, str) else str(_jsonable(k))): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset, range)):
        return [_jsonable(o) for o in obj]
    if not isinstance(obj, type):
        if callable(getattr(obj, "to_dict", None)):
            return _jsonable(obj.to_dict())
        if dataclasses.is_dataclass(obj):
            return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return str(obj)


def _write_json(path, obj) -> dict:
    """Write ``obj`` as strict JSON (see :func:`_jsonable`) through a temporary file; returns what
    was written."""
    path = Path(path)
    obj = _jsonable(obj)
    text = json.dumps(obj, indent=1, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return obj


def run_record(path, config=None, data_files: Union[str, os.PathLike, Sequence] = (), splits=None,
               repo: Optional[Path] = None, extra: Optional[dict] = None) -> dict:
    """Write a JSON record of a run, to make results traceable and comparable.

    The file is strict JSON: arrays become lists, config objects dicts, and NaN or infinite numbers
    ``null``.

    Args:
        path: file to write, e.g. ``run.json``.
        config: the run's configuration: a dict (which may hold sbikit configs, dataclasses and
            arrays) or a single config object.
        data_files: an input file, or a list of them, whose md5 sums are recorded.
        splits: a :class:`~sbikit.data.Splits` or a plain dict of index arrays; its hash is recorded.
        repo: the analysis repository whose git hash is recorded (default: the working directory).
        extra: further top-level entries.

    Returns:
        The record as written: time, host, argv, versions, git hashes (analysis repository and
        sbikit, with ``+dirty`` for uncommitted changes; the sbikit entry is None unless sbikit runs
        from its own git checkout), GPU name, matmul-precision settings, data md5s, split hash and
        config.

    Raises:
        ValueError: if ``extra`` has a key that the record uses itself.
        FileNotFoundError: if a data file does not exist.
    """
    import sbikit
    if isinstance(data_files, (str, os.PathLike)):
        data_files = [data_files]
    if splits is not None and not callable(getattr(splits, "hash", None)):
        from .data import Splits
        splits = Splits(splits)
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(), "argv": sys.argv,
        "versions": versions(),
        "git": {"project": _git(repo or Path.cwd()),
                "sbikit": _git(Path(sbikit.__file__).parent, package="sbikit")},
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "data_files": {str(p): file_md5(p) for p in data_files}, "splits": None if splits is None else splits.hash(),
        "config": {} if config is None else config,
    }
    clash = sorted(set(extra or {}) & set(rec))
    if clash:
        raise ValueError(f"extra keys {clash} are entries of the record itself; rename them")
    return _write_json(path, {**rec, **(extra or {})})
