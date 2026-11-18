"""Checkpoints and run records.

Checkpoints are ``torch.save`` files of plain states (dicts of tensors, numbers, strings and lists),
so they load with ``weights_only=True`` and never execute pickled code. Each stores the package
versions it was written with.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

__all__ = ["FORMAT", "save", "load", "versions", "run_record", "file_md5"]

FORMAT = "sbikit/1"


def versions() -> dict:
    """Versions of Python, torch, numpy, sbikit and its dependencies (None if not importable)."""
    out = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__)}
    for mod in ("sbikit", "sbi", "zuko", "torchemcee", "optuna"):
        try:
            out[mod] = str(__import__(mod).__version__)
        except Exception:                                   # pragma: no cover
            out[mod] = None
    return out


def save(obj, path, meta: Optional[dict] = None) -> None:
    """Save any sbikit object with a ``state()`` (NLE, Ensemble, compressors) or a Prior.

    Args:
        obj: the object.
        path: file to write (parent folders are created; the write is atomic).
        meta: optional dict of plain values stored next to the state.
    """
    from .priors import Prior
    state = {"kind": "prior", "spec": obj.spec()} if isinstance(obj, Prior) else obj.state()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": FORMAT, "kind": state["kind"], "state": state, "versions": versions(),
                "meta": meta or {}}, tmp)
    tmp.replace(path)


def load(path, device="cpu"):
    """Load an object saved with :func:`save`; networks are put on ``device`` in eval mode."""
    from .compress import REGISTRY
    from .nle import NLE, Ensemble
    from .priors import Prior
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path} is not an {FORMAT} checkpoint")
    kind, state = ck["kind"], ck["state"]
    if kind == "nle":
        return NLE.from_state(state, device)
    if kind == "ensemble":
        return Ensemble.from_state(state, device)
    if kind == "prior":
        return Prior.from_spec(state["spec"], device)
    return REGISTRY[kind].from_state(state, device)


def file_md5(path, chunk: int = 1 << 24) -> str:
    """md5 hex digest of a file, read in chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _git(path) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (r.stdout.strip() + ("+dirty" if dirty else "")) if r.returncode == 0 else None
    except Exception:                                       # pragma: no cover
        return None


def run_record(path, config: Optional[dict] = None, data_files: Sequence = (), splits=None,
               repo: Optional[Path] = None, extra: Optional[dict] = None) -> dict:
    """Write a JSON record of a run, to make results traceable and comparable.

    Args:
        path: file to write, e.g. ``run.json``.
        config: the run's configuration (any JSON-serialisable dict; arrays are converted).
        data_files: input files whose md5 sums are recorded.
        splits: a :class:`~sbikit.data.Splits`; its hash is recorded.
        repo: the analysis repository whose git hash is recorded (default: the working directory).
        extra: further top-level entries.

    Returns:
        The record: time, host, argv, versions, git hashes (analysis repository and sbikit, with
        ``+dirty`` for uncommitted changes), GPU name, matmul-precision settings, data md5s, split
        hash and config.
    """
    import sbikit
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(), "argv": sys.argv,
        "versions": versions(), "git": {"project": _git(repo or Path.cwd()), "sbikit": _git(Path(sbikit.__file__).parent)},
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "data_files": {str(p): file_md5(p) for p in data_files}, "splits": None if splits is None else splits.hash(),
        "config": config or {}, **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return rec
