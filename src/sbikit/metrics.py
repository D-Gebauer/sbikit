"""Calibration and constraining-power metrics for posteriors of simulations with known truths.

Shapes: ``samples`` (n_sims, n_samples, P) and ``truth`` (n_sims, P) (note: the ``tarp`` package
uses (n_samples, n_sims, P)). A single posterior (n_samples, P) with its truth (P,) is accepted
by every function and treated as one simulation. Array-likes and tensors are accepted; any other
shape, and NaN or inf in the samples or truths, raise ``ValueError``.

TARP is the definition of Lemos et al. (2023) as in the ``tarp`` package with ``norm=True,
references="random"``: parameters scaled to [0, 1] by the min/max of the truths, one uniform
reference point per simulation, f = fraction of samples closer to the reference than the truth.
``tarp_f`` reproduces ``tarp.get_tarp_coverage(..., norm=True, seed=s)`` exactly for every
reference seed s, and averages several reference draws (one draw is noisy).

Things to keep in mind when reading TARP numbers:

- the normalisation uses the min/max of the test truths, so curves from test sets with different
  parameter ranges are not on an identical footing; compare the scalars and their bootstrap errors,
  or pass ``bounds`` (e.g. the prior box) to fix the normalisation. A parameter whose truth is the
  same in every simulation has no range and needs ``bounds``;
- for the same reason a small test set needs ``bounds``: the range of a handful of truths is much
  narrower than the prior, the references then fall too close to the posteriors, and calibrated
  posteriors show a positive ``bias`` (in a Gaussian test case about +0.6 for 2 simulations, +0.2
  for 5, +0.07 for 10, +0.03 for 30). ``tarp_f`` warns below 10 simulations;
- uniform references are mostly far from the posteriors, especially in many dimensions, which
  makes the test insensitive to miscalibration in weakly constrained directions. :func:`evaluate`
  therefore reports TARP on all parameters and on named subsets, plus 1D coverage per parameter;
- the ECP histogram is coarse (n_sims // 10 bins); :func:`tarp_scalars` works on f directly;
- the TARP ``bias`` says whether the coverage curve lies above or below the diagonal. It is not
  the bias of the posterior means: read ``z_mean`` (:func:`evaluate`) for that.
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from ._utils import as_numpy, did_you_mean, resolve_device, to_tensor

__all__ = ["tarp_f", "ecp_curve", "tarp_scalars", "marginal_coverage", "posterior_summary", "log_fom",
           "importance_weights", "effective_sample_size", "evaluate"]

_DEGENERATE = 1e-12     # a posterior std below this fraction of the largest |sample| counts as zero
_SINGULAR = 1e-10       # smallest / largest eigenvalue of the correlation matrix that counts as singular
_FEW_SIMS = 10          # TARP without bounds warns below this number of simulations


# ------------------------------------------------------------------------------ input checks
def _nonfinite(samples: np.ndarray, truth: Optional[np.ndarray]) -> tuple:
    """Boolean (n_sims,) masks: simulations with NaN or inf samples, and with NaN or inf truths."""
    bad_s = ~np.isfinite(samples).all((1, 2))
    return bad_s, np.zeros(len(samples), bool) if truth is None else ~np.isfinite(truth).all(1)


def _check(samples, truth=None, finite: bool = True) -> tuple:
    """Validated float64 ``samples`` (n_sims, n_samples, P) and ``truth`` (n_sims, P) or None.

    A single posterior (n_samples, P), with a truth (P,) if one is given, becomes one simulation.

    Raises:
        ValueError: for any other shapes, empty arrays or (``finite``) NaN or inf values.
    """
    s, t = as_numpy(samples), None if truth is None else as_numpy(truth)
    shapes = f"samples {s.shape}" + ("" if t is None else f" and truth {t.shape}")
    if s.ndim == 2 and (t is None or t.ndim == 1):
        s, t = s[None], None if t is None else t[None]
    if s.ndim != 3 or 0 in s.shape or (t is not None and t.shape != (s.shape[0], s.shape[2])):
        with_t = ("", "") if truth is None else (" and truth (n_sims, P)", " with its truth (P,)")
        raise ValueError(f"expected samples (n_sims, n_samples, P){with_t[0]}, or one posterior (n_samples, P)"
                         f"{with_t[1]}; got {shapes}. Mind the axis order (the tarp package uses (n_samples, "
                         "n_sims, P)) and keep the last axis for a single parameter (samples[..., None], "
                         "truth[:, None])")
    if finite:
        bad_s, bad_t = _nonfinite(s, t)
        if bad_s.any() or bad_t.any():
            raise ValueError(f"NaN or inf in the samples of {int(bad_s.sum())} and in the truths of "
                             f"{int(bad_t.sum())} of {len(s)} simulations; remove those simulations first "
                             "(evaluate(..., drop_nonfinite=True) does that)")
    return s, t


def _check_weights(weights, n_sims: int, n_samples: int, values: bool = True) -> np.ndarray:
    """Validated float64 weights (n_sims, n_samples); (n_samples,) is accepted for one simulation.
    ``values=False`` checks the shape only."""
    w = as_numpy(weights)
    if w.ndim == 1 and n_sims == 1:
        w = w[None]
    if w.shape != (n_sims, n_samples):
        raise ValueError(f"weights must be (n_sims, n_samples) = ({n_sims}, {n_samples}), got {w.shape}")
    if values and (not np.isfinite(w).all() or (w < 0).any()):
        raise ValueError("weights must be finite and non-negative")
    return w


def _weights(weights, n_sims: int, n_samples: int) -> Optional[np.ndarray]:
    """Validated weights, or None if there are none or they are equal within every simulation
    (which is the unweighted case)."""
    if weights is None:
        return None
    w = _check_weights(weights, n_sims, n_samples)
    return None if (w[:, 0] > 0).all() and (w == w[:, :1]).all() else w


def _check_f(f) -> np.ndarray:
    """Validated TARP statistic (n_references, n_sims); a 1D array is one reference draw."""
    f = np.atleast_2d(as_numpy(f))
    if f.ndim != 2 or f.size == 0 or not np.isfinite(f).all() or f.min() < 0 or f.max() > 1:
        raise ValueError(f"f must be (n_references, n_sims) with values in [0, 1], as tarp_f returns it; got shape "
                         f"{f.shape}")
    return f


def _check_seed(seed, what: str = "seed") -> None:
    """Raise ``ValueError`` unless ``seed`` is an int or None."""
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise ValueError(f"{what} must be an int, or None for draws that differ from call to call; got {seed!r}")


def _columns(idx, n_params: int, names: Optional[Sequence] = None, what: str = "idx") -> list:
    """Column indices from one or several column indices or (with ``names``) parameter names.

    Raises:
        ValueError: for unknown names, indices out of range, duplicates, an empty selection or
            anything that is not a column index, a name or a sequence of them.
    """
    if isinstance(idx, (np.ndarray, torch.Tensor)):
        idx = idx.tolist()                                  # a 0-d array becomes one index
    try:
        items = [idx] if isinstance(idx, (str, int, np.integer)) else list(idx)
    except TypeError:
        raise ValueError(f"{what} must be column indices or parameter names, got {idx!r}") from None
    out = []
    for i in items:
        if isinstance(i, str):
            if names is None:
                raise ValueError(f"{what}: parameter name {i!r} given without names")
            if i not in names:
                same = [n for n in names if str(n).lower() == i.lower()]
                hint = f" (did you mean {same[0]!r}?)" if same else did_you_mean(i, names)
                raise ValueError(f"{what}: unknown parameter {i!r}{hint}; available: {list(names)}")
            out.append(list(names).index(i))
        elif isinstance(i, (int, np.integer)) and not isinstance(i, bool) and -n_params <= i < n_params:
            out.append(int(i) % n_params)
        else:
            raise ValueError(f"{what}: {i!r} is not a column of the {n_params} parameters")
    if not out or len(set(out)) != len(out):
        raise ValueError(f"{what}: need at least one parameter and no duplicates, got {items}")
    return out


def _bounds(bounds, n_params: int) -> tuple:
    """Validated (low, high) float64 (P,) arrays; scalars apply to every parameter."""
    try:
        low, high = (np.broadcast_to(as_numpy(b), (n_params,)) for b in bounds)
    except (TypeError, ValueError):
        raise ValueError(f"bounds must be (low, high) with one value per parameter ({n_params}), e.g. the prior "
                         "box") from None
    if not (np.isfinite(low).all() and np.isfinite(high).all() and (high > low).all()):
        raise ValueError("bounds must be finite with high > low for every parameter")
    return low, high


def _constant_truth(truth: np.ndarray, names: Optional[Sequence] = None) -> None:
    """Raise ``ValueError`` if a parameter has the same truth in every simulation."""
    flat = np.flatnonzero(~(truth.max(0) > truth.min(0)))
    if len(flat):
        which = flat.tolist() if names is None else [names[i] for i in flat]
        raise ValueError(f"TARP scales the parameters by the range of the truths, but parameter(s) {which} have the "
                         f"same truth in all {len(truth)} simulation(s); pass bounds=(low, high), e.g. the prior box")


# ------------------------------------------------------------------------------ TARP
def tarp_f(samples, truth, n_references: int = 50, seed0: Optional[int] = 0, device="cpu", chunk: int = 256, *,
           bounds=None, references=None) -> np.ndarray:
    """TARP statistic f[r, i] for reference seeds r = seed0 .. seed0 + n_references - 1.

    A small test set needs ``bounds``: see that argument.

    With ``bounds=None`` and ``references=None`` it equals the f inside
    ``tarp.get_tarp_coverage(samples.transpose(1, 0, 2), truth, norm=True, seed=r)`` for every r.

    Args:
        samples: (n_sims, n_samples, P) posterior samples.
        truth: (n_sims, P) true parameters.
        n_references: number of reference draws.
        seed0: seed of the first reference draw (None: draws that differ from call to call).
        device: torch device of the distance computation (``"auto"`` picks one). The distances
            need float64, so an MPS device is replaced by the CPU.
        chunk: simulations per distance block (memory).
        bounds: (low, high), each (P,) or a scalar: the box that is scaled to [0, 1] and in which
            the references are drawn, e.g. the prior box. Default: the min/max of the truths, as
            the ``tarp`` package does. That range is too narrow for a handful of simulations:
            calibrated posteriors then show a positive ``bias`` (in a Gaussian test case about
            +0.6 for 2 simulations, +0.2 for 5, +0.07 for 10, +0.03 for 30), so pass ``bounds``
            for fewer than about 30 simulations; below 10 a ``RuntimeWarning`` says so.
            Note that TARP assumes truths drawn from the prior of the posteriors: for a parameter
            that is fixed in the test set but free in the posteriors, f is only approximately
            uniform (a positive ``bias`` of the order of the posterior width over the box width).
        references: (n_references, n_sims, P) or (n_sims, P) reference points in parameter units,
            used instead of the uniform draws (``n_references`` and ``seed0`` are then ignored).

    Returns:
        (n_references, n_sims) array: per simulation, the fraction of samples closer to the
        reference point than the truth. For calibrated posteriors f is uniform on [0, 1].

    Raises:
        ValueError: for wrong shapes or non-finite values, and without ``bounds`` if a parameter
            has the same truth in every simulation (always the case for a single simulation):
            the normalisation is then undefined and every posterior would look overconfident.
    """
    samples, truth = _check(samples, truth)
    _check_seed(seed0, "seed0")
    n_sims, n_samples, d = samples.shape
    if bounds is None:
        _constant_truth(truth)
        if n_sims < _FEW_SIMS:
            warnings.warn(f"tarp_f: the range of the truths of only {n_sims} simulations is too narrow to scale the "
                          "parameters by, which makes calibrated posteriors look biased (bias > 0); pass "
                          "bounds=(low, high), e.g. the prior box", RuntimeWarning, stacklevel=2)
        low = truth.min(0)
        width = truth.max(0) - low + 1e-10          # the offset is part of the tarp package's definition
    else:
        low, high = _bounds(bounds, d)
        width = high - low
    if references is not None:
        refs = as_numpy(references)
        refs = refs[None] if refs.ndim == 2 else refs
        if refs.ndim != 3 or refs.shape[1:] != (n_sims, d) or len(refs) == 0 or not np.isfinite(refs).all():
            raise ValueError(f"references must be finite with shape (n_references, {n_sims}, {d}) or "
                             f"({n_sims}, {d}), got {refs.shape}")
        refs, n_references = (refs - low) / width, len(refs)
    if n_references < 1 or chunk < 1:
        raise ValueError(f"n_references and chunk must be at least 1, got {n_references} and {chunk}")
    device = resolve_device(device)
    device = "cpu" if torch.device(device).type == "mps" else device
    s = to_tensor((samples - low) / width, device=device)
    t = to_tensor((truth - low) / width, device=device)
    out = np.empty((n_references, n_sims))
    for r in range(n_references):
        rng = np.random.RandomState(None if seed0 is None else seed0 + r)
        ref = to_tensor(refs[r] if references is not None else rng.uniform(0, 1, size=(n_sims, d)), device=device)
        d_t = torch.sqrt(((ref - t) ** 2).sum(-1))
        for i in range(0, n_sims, chunk):
            d_s = torch.sqrt(((ref[i:i + chunk, None] - s[i:i + chunk]) ** 2).sum(-1))
            out[r, i:i + chunk] = (d_s < d_t[i:i + chunk, None]).sum(1).cpu().numpy() / n_samples
    return out


def ecp_curve(f: np.ndarray, num_alpha_bins: Optional[int] = None):
    """Expected coverage probability ECP(alpha) as the ``tarp`` package computes it (histogram of
    f, n_sims // 10 bins by default), averaged over the reference draws.

    Args:
        f: (n_references, n_sims) from :func:`tarp_f` (or (n_sims,) for one draw).
        num_alpha_bins: number of bins (default ``max(1, n_sims // 10)``).

    Returns:
        (alpha, ecp): credibility levels and coverage; calibrated posteriors lie on the diagonal.
    """
    f = _check_f(f)
    nb = num_alpha_bins or max(1, f.shape[1] // 10)
    ecps = []
    for fr in f:
        h, alpha = np.histogram(fr, density=True, bins=nb, range=(0, 1))
        ecps.append(np.concatenate([[0], np.cumsum(h) * (alpha[1] - alpha[0])]))
    return alpha, np.mean(ecps, 0)


def tarp_scalars(f: np.ndarray, n_bootstrap: int = 200, tol_min: float = 0.02, seed: Optional[int] = 0, *,
                 n_samples: Optional[int] = None) -> dict:
    """Scalars of the TARP curve, from f directly (no binning):

    - bias = 2 int (ECP - alpha) d alpha = 1 - 2 E[f]: < 0 means the coverage curve lies below the
      diagonal, > 0 above it. Despite its name it does not measure a shift of the posterior means:
      shifted posteriors mostly show up as a negative ``confidence``, with a ``bias`` of either
      sign. Use the z-scores of the posterior means (``z_mean`` in :func:`evaluate`) for that;
    - confidence = 4 int sign(alpha - 1/2) (ECP - alpha) = 1 - 4 E|f - 1/2|: < 0 means an
      overconfident S-shape (posteriors too narrow, or off-centre), > 0 underconfident (too wide);
    - max_dev = max |ECP(alpha) - alpha| on a grid of 201 alpha values, where ECP(alpha) is the
      fraction of f below alpha (ties count half), averaged over the reference draws. For
      calibrated posteriors it is of order 1 / sqrt(n_sims).

    The ``*_sd`` errors come from bootstrapping the simulations, and ``tol_* = max(2 sd, tol_min)``
    is the tolerance the constrained hyperparameter search (:func:`sbikit.hpt.tune_nle`) allows.

    Args:
        f: (n_references, n_sims) from :func:`tarp_f`.
        n_bootstrap: bootstrap resamples (0: no errors; the ``*_sd`` are then nan).
        tol_min: smallest tolerance (>= 0).
        seed: seed of the bootstrap (None: resamples that differ from call to call).
        n_samples: number of posterior samples behind f. f only takes the values k / n_samples,
            which makes ``confidence`` of calibrated posteriors -1 / (n_samples + 1) rather than
            0. If given, the scalars use the mid-ranks (k + 1/2) / (n_samples + 1), which removes
            that offset (it matters below about 100 samples per posterior).

    Returns:
        dict with ``bias``, ``confidence``, ``bias_sd``, ``confidence_sd``, ``tol_bias``,
        ``tol_confidence``, ``max_dev``, ``n_sims`` and ``n_references``.

    Raises:
        ValueError: if f is not what :func:`tarp_f` returns, for a negative ``n_bootstrap`` or
            ``tol_min``, or if f does not lie on the grid k / ``n_samples``.
    """
    f = _check_f(f)
    _check_seed(seed)
    if n_bootstrap < 0 or not tol_min >= 0 or (n_samples is not None and n_samples < 1):
        raise ValueError(f"n_bootstrap and tol_min must be >= 0 and n_samples >= 1, got {n_bootstrap}, {tol_min} "
                         f"and {n_samples}")
    if n_samples is not None:
        if np.abs(f * n_samples - np.round(f * n_samples)).max() > 1e-6:
            raise ValueError(f"f is not on the grid k / n_samples for n_samples={n_samples}: pass the number of "
                             "posterior samples that f was computed from")
        f = (f * n_samples + 0.5) / (n_samples + 1)
    n = f.shape[1]
    bias = lambda g: 1 - 2 * g.mean()                      # noqa: E731
    conf = lambda g: 1 - 4 * np.abs(g - 0.5).mean()        # noqa: E731
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, n, n) for _ in range(n_bootstrap)]
    sb = float(np.std([bias(f[:, b]) for b in boots])) if n_bootstrap else float("nan")
    sc = float(np.std([conf(f[:, b]) for b in boots])) if n_bootstrap else float("nan")
    alpha = np.linspace(0, 1, 201)
    ecp = np.mean([np.searchsorted(fr, alpha, "left") + np.searchsorted(fr, alpha, "right")
                   for fr in np.sort(f, axis=1)], 0) / (2 * n)
    return {"bias": float(bias(f)), "confidence": float(conf(f)), "bias_sd": sb, "confidence_sd": sc,
            "tol_bias": max(2 * sb, tol_min) if n_bootstrap else tol_min,
            "tol_confidence": max(2 * sc, tol_min) if n_bootstrap else tol_min,
            "max_dev": float(np.abs(ecp - alpha).max()), "n_sims": int(n), "n_references": int(f.shape[0])}


# ------------------------------------------------------------------------------ marginals
def _levels(levels) -> tuple:
    """Validated coverage levels and their record keys (0.68 -> ``"cov68"``)."""
    text = isinstance(levels, str) or (isinstance(levels, (list, tuple)) and any(isinstance(l, str) for l in levels))
    values = [] if text or levels is None else [float(l) for l in np.ravel(as_numpy(levels))]
    if not values or not all(0 < l < 1 for l in values):
        raise ValueError(f"levels must be numbers in (0, 1), e.g. 0.68 (not 68), got {levels!r}")
    levels = values
    keys = ["cov" + format(100 * l, "g") for l in levels]
    if len(set(keys)) != len(keys):
        raise ValueError(f"levels {levels} do not have distinct keys ({keys})")
    return levels, keys


def _ks_pvalue(d: np.ndarray, n: int) -> np.ndarray:
    """P-value of KS distances ``d`` from ``n`` points: the asymptotic Kolmogorov distribution with
    the small-sample correction of Stephens (1970)."""
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * np.asarray(d, dtype=np.float64)[..., None]
    j = np.arange(1, 101)
    series = 2 * ((-1.0) ** (j - 1) * np.exp(-2 * (j * lam) ** 2)).sum(-1)
    return np.where(lam[..., 0] < 0.2, 1.0, np.clip(series, 0.0, 1.0))


def _min_samples(level: float) -> int:
    """Fewest posterior samples that resolve the central interval at ``level``: each tail
    (1 - level) / 2 must hold at least one of the n_samples + 1 equally probable rank cells."""
    return max(1, math.ceil(2 / (1 - level) - 1 - 1e-9))


def marginal_coverage(samples, truth, levels: Sequence[float] = (0.68, 0.95)) -> dict:
    """Per parameter: fraction of truths inside the central credible interval at each level, and
    the KS distance of the truth ranks from uniform (simulation-based calibration in 1D).

    Both use the rank count k of a truth: the number of its n_samples posterior samples below it.
    For calibrated posteriors k is uniform on 0 .. n_samples, i.e. the posterior probability below
    the truth lies in the cell [k, k + 1] / (n_samples + 1).

    - Coverage: each simulation counts with the fraction of its cell inside
      [(1 - level) / 2, (1 + level) / 2] (1 or 0 except in the two cells at the interval ends).
      Its expectation for calibrated posteriors is exactly ``level`` for any number of samples.
      A truth below or above every sample never counts as covered. That needs each tail
      (1 - level) / 2 to hold a whole cell: n_samples >= 2 / (1 - level) - 1 (6 samples for
      0.68, 39 for 0.95, 740 for 0.9973). A level that the samples cannot resolve gives nan and a
      ``RuntimeWarning`` that names the number of samples it needs.
    - KS: the largest difference between the cumulative distribution of the rank counts and that
      of the uniform distribution on 0 .. n_samples.

    Args:
        samples: (n_sims, n_samples, P) posterior samples.
        truth: (n_sims, P) true parameters.
        levels: credibility levels as fractions in (0, 1).

    Returns:
        dict of lists over the P parameters: ``"ks"``, ``"ks_pvalue"`` (probability of a KS
        distance at least this large for calibrated posteriors, from the Kolmogorov distribution:
        accurate for many samples per posterior and conservative, i.e. too large, for few, so
        rank discreteness alone never rejects calibrated posteriors), and per level
        ``"cov<level in %>"`` (0.68 -> ``"cov68"``, 0.9973 -> ``"cov99.73"``) with
        ``"cov<level in %>_sd"``, the binomial scatter sqrt(level (1 - level) / n_sims) expected
        for calibrated posteriors.

    Raises:
        ValueError: for wrong shapes, non-finite values, levels outside (0, 1) (e.g. 68 instead
            of 0.68) or levels that share a key.
    """
    samples, truth = _check(samples, truth)
    levels, keys = _levels(levels)
    n, n_samples, n_params = samples.shape
    k = (samples < truth[:, None, :]).sum(1)                                     # (n_sims, P), 0 .. n_samples
    cdf = np.stack([np.bincount(kp, minlength=n_samples + 1) for kp in k.T], 1).cumsum(0) / n
    ks = np.abs(cdf - np.arange(1, n_samples + 2)[:, None] / (n_samples + 1)).max(0)
    out = {"ks": ks.tolist(), "ks_pvalue": _ks_pvalue(ks, n).tolist()}
    unresolved = []
    for l, key in zip(levels, keys):
        lo, hi = (1 - l) / 2 * (n_samples + 1), (1 + l) / 2 * (n_samples + 1)     # the interval in units of cells
        cov = np.clip(np.minimum(k + 1, hi) - np.maximum(k, lo), 0, 1).mean(0)
        if n_samples < _min_samples(l):
            cov = np.full(n_params, np.nan)
            unresolved.append(f"{l:g} (needs {_min_samples(l)})")
        out[key] = cov.tolist()
        out[key + "_sd"] = [math.sqrt(l * (1 - l) / n)] * n_params
    if unresolved:
        warnings.warn(f"marginal_coverage: {n_samples} samples per posterior cannot resolve the level(s) "
                      f"{', '.join(unresolved)}; their coverage is nan", RuntimeWarning, stacklevel=2)
    return out


def posterior_summary(samples, truth=None, weights=None) -> dict:
    """Mean, median, std, 16/84 and 2.5/97.5 percentiles per simulation and parameter, and
    z = (mean - truth) / std if truths are given.

    The quantiles interpolate linearly between the sorted samples (numpy's default). With weights,
    each sample sits at the mid-point of its weight interval and the positions are rescaled so
    that the first and last sample with weight are the 0 and 100 % quantiles; for equal weights
    this is the same definition, and the result is identical to the unweighted one.

    Args:
        samples: (n_sims, n_samples, P) posterior samples.
        truth: optional (n_sims, P) true parameters.
        weights: optional (n_sims, n_samples) importance weights (finite, >= 0). A simulation
            whose weights are all zero gets nan.

    Returns:
        dict of (n_sims, P) arrays ``mean``, ``std``, ``q025``, ``q16``, ``median``, ``q84``,
        ``q975`` and, with truths, ``z``. Where the posterior std is zero (a stuck chain) z is
        nan and a ``RuntimeWarning`` is emitted.

    Raises:
        ValueError: for wrong shapes, non-finite values or invalid weights.
    """
    samples, truth = _check(samples, truth)
    w = _weights(weights, *samples.shape[:2])
    if w is None:
        q = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=1)
        mean, std = samples.mean(1), samples.std(1)
    else:
        total = w.sum(1, keepdims=True)
        w = w / np.where(total > 0, total, 1.0)
        mean = (w[..., None] * samples).sum(1)
        std = np.sqrt((w[..., None] * (samples - mean[:, None]) ** 2).sum(1))
        q = np.full((5,) + mean.shape, np.nan)
        for i in np.flatnonzero(total[:, 0] > 0):
            keep = w[i] > 0
            q[:, i] = _weighted_quantiles(samples[i, keep], w[i, keep], (0.025, 0.16, 0.5, 0.84, 0.975))
        mean[total[:, 0] <= 0], std[total[:, 0] <= 0] = np.nan, np.nan
    out = {"mean": mean, "std": std, "q025": q[0], "q16": q[1], "median": q[2], "q84": q[3], "q975": q[4]}
    if truth is not None:
        flat = ~(std > _DEGENERATE * np.abs(samples).max(1))
        if flat.any():
            warnings.warn(f"posterior_summary: {int(flat.any(1).sum())} of {len(samples)} simulations have a zero "
                          "posterior std (a stuck chain, or no weight); their z is nan", RuntimeWarning)
        out["z"] = np.where(flat, np.nan, (mean - truth) / np.where(flat, 1.0, std))
    return out


def _weighted_quantiles(x: np.ndarray, w: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    """Quantiles (len(probs), P) of one posterior ``x`` (n, P) with positive weights ``w`` (n,)."""
    order = np.argsort(x, axis=0)
    xs, ws = np.take_along_axis(x, order, 0), w[order]
    mid = np.cumsum(ws, axis=0) - ws / 2
    span = mid[-1] - mid[0]
    pos = (mid - mid[0]) / np.where(span > 0, span, 1.0)
    return np.stack([np.interp(probs, pos[:, j], xs[:, j]) for j in range(x.shape[1])], 1)


# ------------------------------------------------------------------------------ FoM
def log_fom(samples, idx, weights=None, *, names: Optional[Sequence[str]] = None) -> np.ndarray:
    """Log figure of merit, -1/2 log det Cov(theta[idx]), per simulation.

    Args:
        samples: (n_sims, n_samples, P).
        idx: the parameters in the figure of merit: column indices (or one index) or, with
            ``names``, parameter names.
        weights: optional (n_sims, n_samples) importance weights (finite, >= 0). Check their
            :func:`effective_sample_size`: a covariance from a handful of effective samples is
            too small, which inflates the figure of merit.
        names: the P parameter names, needed if ``idx`` holds names.

    Returns:
        (n_sims,) array; nan where the covariance is not positive definite or numerically
        singular (a stuck chain, exactly degenerate parameters) or not determined by the samples
        (fewer samples, or effective samples of the weights, than parameters + 1).

    Raises:
        ValueError: for wrong shapes, non-finite values, invalid weights, unknown parameters or
            an ``idx`` that is not column indices or names.

    Example:
        >>> log_fom(samples, ["Om", "S8"], names=["Om", "S8", "h"])
    """
    samples, _ = _check(samples)
    n_sims, n_samples, n_params = samples.shape
    if names is not None and len(names) != n_params:
        raise ValueError(f"names has {len(names)} entries but the samples have {n_params} parameters")
    cols = _columns(idx, n_params, names)
    w = _weights(weights, n_sims, n_samples)
    x, d = samples[..., cols], len(cols)
    covs = np.full((n_sims, d, d), np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)    # weights dominated by one sample: no covariance
        for i in range(n_sims):
            if w is None and n_samples > 1:
                covs[i] = np.cov(x[i].T)
            elif w is not None and (w[i] > 0).sum() > 1:
                covs[i] = np.cov(x[i].T, aweights=w[i])
    var = np.einsum("ijj->ij", covs)
    good = np.isfinite(covs).all((1, 2)) & (var > (_DEGENERATE * np.abs(x).max(1)) ** 2).all(1)
    if w is not None:
        good &= effective_sample_size(w) > d + 1 - 1e-9
    sd = np.sqrt(np.where(good[:, None], var, 1.0))
    covs = np.where(good[:, None, None], covs, np.eye(d))
    eig = np.linalg.eigvalsh(covs / (sd[:, :, None] * sd[:, None, :]))            # of the correlation matrices
    good &= eig[:, 0] > _SINGULAR * eig[:, -1]
    sign, logdet = np.linalg.slogdet(covs)
    return np.where(good & (sign > 0), -0.5 * logdet, np.nan)


def importance_weights(samples, from_prior, to_prior) -> np.ndarray:
    """Weights to_prior / from_prior per sample (n_sims, n_samples), normalised to max 1 per sim.
    Use them to turn posteriors under the sampling prior into posteriors under another one.

    Samples outside the support of ``to_prior`` get weight 0. Few samples may carry most of the
    weight if the priors differ a lot: check :func:`effective_sample_size` before trusting
    weighted summaries or figures of merit.

    Args:
        samples: (n_sims, n_samples, P) posterior samples drawn under ``from_prior``.
        from_prior, to_prior: priors with ``log_prob`` and ``to`` (e.g. :class:`sbikit.Prior`);
            one prior each, shared by all simulations.

    Returns:
        (n_sims, n_samples) weights in [0, 1]; all zero for a simulation without samples in the
        support of ``to_prior``.

    Raises:
        ValueError: for wrong shapes.
        TypeError: if a prior is not a single prior object (e.g. a list of priors).
    """
    samples, _ = _check(samples, finite=False)
    for name, prior in (("from_prior", from_prior), ("to_prior", to_prior)):
        if not hasattr(prior, "log_prob"):
            raise TypeError(f"importance_weights: {name} must be one prior with log_prob, got "
                            f"{type(prior).__name__}; for a prior per simulation, call it per simulation")
    th = to_tensor(samples.reshape(-1, samples.shape[-1]), dtype=torch.float32)
    fp, tp = from_prior.to("cpu"), to_prior.to("cpu")
    with torch.no_grad():
        lw = torch.cat([tp.log_prob(c) - fp.log_prob(c) for c in th.split(1 << 16)])
    lw = lw.reshape(samples.shape[:2]).double().numpy()
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    mx = np.max(np.where(np.isfinite(lw), lw, -np.inf), axis=1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.nan_to_num(np.exp(lw - mx))


def effective_sample_size(weights) -> np.ndarray:
    """Kish effective sample size (sum w)^2 / sum w^2 per simulation.

    Args:
        weights: (n_sims, n_samples) importance weights, e.g. from :func:`importance_weights`
            ((n_samples,) for one simulation).

    Returns:
        (n_sims,) array between 0 (all weights zero) and n_samples (equal weights).

    Raises:
        ValueError: for a wrong shape or negative or non-finite weights.
    """
    w = as_numpy(weights)
    w = w[None] if w.ndim == 1 else w
    if w.ndim != 2 or w.size == 0:
        raise ValueError(f"weights must be (n_sims, n_samples), got {w.shape}")
    w = _check_weights(w, *w.shape)
    top = w.max(1, keepdims=True)
    w = w / np.where(top > 0, top, 1.0)
    s1, s2 = w.sum(1), (w ** 2).sum(1)
    return np.where(s2 > 0, s1 ** 2 / np.where(s2 > 0, s2, 1.0), 0.0)


# ------------------------------------------------------------------------------ standard record
def _jsonable(obj):
    """``obj`` in plain JSON types: arrays and tuples become lists, numpy scalars Python numbers,
    keys strings, and NaN or inf None."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, (np.ndarray, np.generic)):
        obj = obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return None if isinstance(obj, float) and not math.isfinite(obj) else obj


def _nan(fn: Callable, a: np.ndarray, **kwargs):
    """A numpy nan-reduction (``np.nanmean``, ...) without its warning for all-nan input."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, **kwargs)


def evaluate(samples, truth, param_names: Sequence[str],
             fom_sets: Mapping[str, Sequence[str]] = None,
             tarp_sets: Mapping[str, Sequence[str]] = None,
             n_references: int = 50, n_bootstrap: int = 200, tol_min: float = 0.02, seed: Optional[int] = 0,
             device="cpu", *, weights=None, levels: Sequence[float] = (0.68, 0.95), bounds=None,
             extra: Optional[Mapping[str, Callable]] = None, drop_nonfinite: bool = False) -> tuple[dict, dict]:
    """The standard metrics record for posteriors of simulations with known truths.

    Using the same record everywhere makes runs, compressions and analyses comparable.

    Args:
        samples: (n_sims, n_samples, P) posterior samples.
        truth: (n_sims, P) true parameters.
        param_names: the P names (exactly one per parameter, no duplicates).
        fom_sets: name -> parameter names of a figure of merit, e.g. ``{"om_s8": ["Om", "S8"]}``.
        tarp_sets: name -> parameter names of a TARP test; ``"all"`` (all parameters) is always
            included and cannot be redefined.
        n_references, n_bootstrap, tol_min: see :func:`tarp_f` and :func:`tarp_scalars`.
        seed: seed of the TARP reference draws and of the bootstrap (None: draws that differ from
            call to call).
        device: torch device of the TARP distances.
        weights: optional (n_sims, n_samples) importance weights (:func:`importance_weights`),
            used for the posterior summaries (means, stds, z-scores) and the figures of merit.
            The calibration tests (TARP, marginal coverage) always use the unweighted samples,
            because the truths follow the sampling prior.
        levels: credibility levels of the marginal coverage, as fractions. A level that
            ``n_samples`` cannot resolve (0.95 needs 39 samples, see :func:`marginal_coverage`)
            is reported as ``None``, with a warning.
        bounds: (low, high) over the P parameters that the TARP tests normalise by, e.g. the
            prior box (see :func:`tarp_f`). Needed if a parameter has the same truth in every
            simulation, and for small test sets: without it, calibrated posteriors of fewer than
            about 30 simulations show a positive TARP ``bias`` (a warning below 10).
        extra: name -> callable ``(samples, truth) -> value`` for additional metrics; the
            (JSON-serialisable) values are stored under ``record["extra"][name]``.
        drop_nonfinite: drop the simulations with NaN or inf samples or truths (e.g. failed
            chains) with a warning, instead of raising. Their weights are not looked at.

    Returns:
        (record, arrays). ``record`` passes ``json.dumps(record, allow_nan=False)``; a number that
        is undefined (no bootstrap, every figure of merit nan, ...) is ``None``. Its entries:
        ``n_sims`` (simulations used), ``n_dropped``, ``param_names``, ``tarp[set]``
        (:func:`tarp_scalars`, with mid-ranks), ``fom[set]`` (``log_fom_mean``, ``fom_geomean``,
        ``n_nan``), ``marginal[param]`` (:func:`marginal_coverage`), per parameter ``z_mean`` and
        ``z_rms`` of (posterior mean - truth) / std over the simulations with a non-zero std,
        ``z_mean_sd`` (= ``z_rms`` / sqrt(n), the error of ``z_mean``), ``z_n_nan`` (simulations
        without a z) and ``std_median``, ``ess`` (``min`` and ``median`` effective sample size
        of the weights, or ``None`` without weights) and ``extra``.
        ``arrays`` holds numpy arrays: posterior means, stds and medians, the TARP curves
        (``tarp_alpha``, ``tarp_ecp_<set>``), ``log_fom_<set>`` per simulation, ``kept`` (mask
        over the given simulations: False for dropped ones) and, with weights, ``ess``.

    Raises:
        ValueError: for wrong shapes, a wrong number of (or duplicate) ``param_names``, unknown
            parameters in ``fom_sets`` / ``tarp_sets``, non-finite input (unless
            ``drop_nonfinite``), invalid ``weights``, ``levels``, ``bounds``, ``seed``,
            ``tol_min`` or ``extra``, or a parameter with the same truth in every simulation when
            ``bounds`` is not given.

    Example:
        >>> record, arrays = evaluate(samples, truth, ["Om", "S8", "h"], fom_sets={"om_s8": ["Om", "S8"]})
        >>> record["tarp"]["all"]["confidence"], record["marginal"]["Om"]["cov68"]
    """
    samples, truth = _check(samples, truth, finite=False)
    n_given, n_samples, n_params = samples.shape
    names = [param_names] if isinstance(param_names, str) else list(param_names)
    if len(names) != n_params or len(set(names)) != n_params:
        raise ValueError(f"param_names must be {n_params} distinct names, one per parameter (samples are "
                         f"{samples.shape}); got {len(names)}: {names}")

    def resolve(sets, what):
        if sets is not None and not isinstance(sets, Mapping):
            raise ValueError(f"{what} must map a name to parameter names, got {type(sets).__name__}")
        return {key: _columns(ps, n_params, names, f"{what}[{key!r}]") for key, ps in dict(sets or {}).items()}

    fom_cols, tarp_cols = resolve(fom_sets, "fom_sets"), resolve(tarp_sets, "tarp_sets")
    everything = list(range(n_params))
    if tarp_cols.get("all", everything) != everything:
        raise ValueError("tarp_sets['all'] is reserved for the test on all parameters; use another name")
    tarp_cols = {"all": everything, **tarp_cols}
    w = None if weights is None else _check_weights(weights, n_given, n_samples, values=False)
    levels, _ = _levels(levels)
    box = None if bounds is None else _bounds(bounds, n_params)
    _check_seed(seed)
    if extra is not None and not isinstance(extra, Mapping):
        raise ValueError(f"extra must map a name to a callable (samples, truth) -> value, got {type(extra).__name__}")
    for name, fn in dict(extra or {}).items():
        if not callable(fn):
            raise ValueError(f"extra[{name!r}] must be a callable (samples, truth) -> value")

    bad_s, bad_t = _nonfinite(samples, truth)
    kept = ~(bad_s | bad_t)
    if not kept.all():
        if not drop_nonfinite:
            _check(samples, truth)
        if not kept.any():
            raise ValueError(f"all {n_given} simulations have NaN or inf samples or truths")
        warnings.warn(f"evaluate: dropped {int((~kept).sum())} of {n_given} simulations with NaN or inf samples or "
                      "truths", RuntimeWarning)
        samples, truth, w = samples[kept], truth[kept], None if w is None else w[kept]
    w = None if w is None else _check_weights(w, len(samples), n_samples)
    if box is None:
        _constant_truth(truth, names)

    summ = posterior_summary(samples, truth, w)
    z = summ["z"]
    n_z = np.isfinite(z).sum(0)
    z_rms = np.sqrt(_nan(np.nanmean, z ** 2, axis=0))
    record = {"n_sims": int(len(truth)), "n_dropped": int((~kept).sum()), "param_names": names, "tarp": {},
              "fom": {},
              "z_mean": dict(zip(names, _nan(np.nanmean, z, axis=0).round(4))),
              "z_rms": dict(zip(names, z_rms.round(4))),
              "z_mean_sd": dict(zip(names, (z_rms / np.sqrt(np.maximum(n_z, 1))).round(4))),
              "z_n_nan": dict(zip(names, len(z) - n_z)),
              "std_median": dict(zip(names, _nan(np.nanmedian, summ["std"], axis=0)))}
    mc = marginal_coverage(samples, truth, levels)
    record["marginal"] = {n: {k: v[p] for k, v in mc.items()} for p, n in enumerate(names)}
    arrays = {"post_mean": summ["mean"], "post_std": summ["std"], "post_median": summ["median"], "kept": kept}
    for key, cols in tarp_cols.items():
        with warnings.catch_warnings():
            if key != "all":
                warnings.filterwarnings("ignore", message="tarp_f: the range")      # said once, for "all"
            f = tarp_f(samples[..., cols], truth[:, cols], n_references, seed, device,
                       bounds=None if box is None else (box[0][cols], box[1][cols]))
        record["tarp"][key] = tarp_scalars(f, n_bootstrap, tol_min, seed, n_samples=n_samples)
        arrays["tarp_alpha"], arrays[f"tarp_ecp_{key}"] = ecp_curve(f)
    for key, cols in fom_cols.items():
        lf = log_fom(samples, cols, w)
        mean = _nan(np.nanmean, lf)
        with np.errstate(over="ignore"):
            record["fom"][key] = {"log_fom_mean": float(mean), "fom_geomean": float(np.exp(mean)),
                                  "n_nan": int(np.isnan(lf).sum())}
        arrays[f"log_fom_{key}"] = lf
    record["ess"] = None
    if w is not None:
        arrays["ess"] = effective_sample_size(w)
        record["ess"] = {"min": float(arrays["ess"].min()), "median": float(np.median(arrays["ess"]))}
    record["extra"] = {}
    for name, fn in dict(extra or {}).items():
        value = _jsonable(fn(samples, truth))
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            raise TypeError(f"extra[{name!r}] must return a JSON-serialisable value, got "
                            f"{type(value).__name__}") from None
        record["extra"][name] = value
    return _jsonable(record), arrays
