"""Calibration and constraining-power metrics for posteriors of simulations with known truths.

Shapes: ``samples`` (n_sims, n_samples, P) and ``truth`` (n_sims, P) (note: the ``tarp`` package
uses (n_samples, n_sims, P)).

TARP is the definition of Lemos et al. (2023) as in the ``tarp`` package with ``norm=True,
references="random"``: parameters scaled to [0, 1] by the min/max of the truths, one uniform
reference point per simulation, f = fraction of samples closer to the reference than the truth.
``tarp_f`` reproduces ``tarp.get_tarp_coverage(..., norm=True, seed=s)`` exactly for every
reference seed s, and averages several reference draws (one draw is noisy).

Things to keep in mind when reading TARP numbers:

- the normalisation uses the min/max of the test truths, so curves from test sets with different
  parameter ranges are not on an identical footing; compare the scalars and their bootstrap errors;
- uniform references are mostly far from the posteriors, especially in many dimensions, which
  makes the test insensitive to miscalibration in weakly constrained directions. :func:`evaluate`
  therefore reports TARP on all parameters and on named subsets, plus 1D coverage per parameter;
- the ECP histogram is coarse (n_sims // 10 bins); :func:`tarp_scalars` works on f directly.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import torch

__all__ = ["tarp_f", "ecp_curve", "tarp_scalars", "marginal_coverage", "posterior_summary", "log_fom",
           "importance_weights", "evaluate"]


# ------------------------------------------------------------------------------ TARP
def tarp_f(samples, truth, n_references: int = 50, seed0: int = 0, device="cpu", chunk: int = 256) -> np.ndarray:
    """TARP statistic f[r, i] for reference seeds r = seed0 .. seed0 + n_references - 1.

    Equals the f inside ``tarp.get_tarp_coverage(samples.transpose(1, 0, 2), truth, norm=True,
    seed=r)`` for every r.

    Args:
        samples: (n_sims, n_samples, P) posterior samples.
        truth: (n_sims, P) true parameters.
        n_references: number of reference draws.
        seed0: seed of the first reference draw.
        device: torch device of the distance computation.
        chunk: simulations per distance block (memory).

    Returns:
        (n_references, n_sims) array: per simulation, the fraction of samples closer to the
        reference point than the truth. For calibrated posteriors f is uniform on [0, 1].
    """
    samples, truth = np.asarray(samples, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    n_sims, n_samples, d = samples.shape
    if truth.shape != (n_sims, d):
        raise ValueError(f"truth must be ({n_sims}, {d}), got {truth.shape}")
    low, high = truth.min(0), truth.max(0)
    s = torch.as_tensor((samples - low) / (high - low + 1e-10), device=device)
    t = torch.as_tensor((truth - low) / (high - low + 1e-10), device=device)
    out = np.empty((n_references, n_sims))
    for r in range(n_references):
        ref = torch.as_tensor(np.random.RandomState(seed0 + r).uniform(0, 1, size=(n_sims, d)), device=device)
        d_t = torch.sqrt(((ref - t) ** 2).sum(-1))
        for i in range(0, n_sims, chunk):
            d_s = torch.sqrt(((ref[i:i + chunk, None] - s[i:i + chunk]) ** 2).sum(-1))
            out[r, i:i + chunk] = (d_s < d_t[i:i + chunk, None]).sum(1).cpu().numpy() / n_samples
    return out


def ecp_curve(f: np.ndarray, num_alpha_bins: Optional[int] = None):
    """Expected coverage probability ECP(alpha) as the ``tarp`` package computes it (histogram of
    f, n_sims // 10 bins by default), averaged over the reference draws.

    Returns:
        (alpha, ecp): credibility levels and coverage; calibrated posteriors lie on the diagonal.
    """
    f = np.atleast_2d(f)
    nb = num_alpha_bins or max(1, f.shape[1] // 10)
    ecps = []
    for fr in f:
        h, alpha = np.histogram(fr, density=True, bins=nb, range=(0, 1))
        ecps.append(np.concatenate([[0], np.cumsum(h) * (alpha[1] - alpha[0])]))
    return alpha, np.mean(ecps, 0)


def tarp_scalars(f: np.ndarray, n_bootstrap: int = 200, tol_min: float = 0.02, seed: int = 0) -> dict:
    """Scalars of the TARP curve, from f directly (no binning):

    - bias = 2 int (ECP - alpha) d alpha = 1 - 2 E[f]: < 0 means the curve lies below the diagonal;
    - confidence = 4 int sign(alpha - 1/2) (ECP - alpha) = 1 - 4 E|f - 1/2|: < 0 means an
      overconfident S-shape;
    - max_dev = max |ECP - alpha| of the averaged curve.

    The ``*_sd`` errors come from bootstrapping the simulations, and ``tol_* = max(2 sd, tol_min)``
    is the tolerance the constrained hyperparameter search (:func:`sbikit.hpt.tune_nle`) allows.

    Args:
        f: (n_references, n_sims) from :func:`tarp_f`.
        n_bootstrap: bootstrap resamples (0: no errors).
        tol_min: smallest tolerance.
        seed: seed of the bootstrap.
    """
    f = np.atleast_2d(f)
    bias = lambda g: 1 - 2 * g.mean()                      # noqa: E731
    conf = lambda g: 1 - 4 * np.abs(g - 0.5).mean()        # noqa: E731
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, f.shape[1], f.shape[1]) for _ in range(n_bootstrap)]
    sb = float(np.std([bias(f[:, b]) for b in boots])) if n_bootstrap else float("nan")
    sc = float(np.std([conf(f[:, b]) for b in boots])) if n_bootstrap else float("nan")
    alpha, ecp = ecp_curve(f)
    return {"bias": float(bias(f)), "confidence": float(conf(f)), "bias_sd": sb, "confidence_sd": sc,
            "tol_bias": max(2 * sb, tol_min) if n_bootstrap else tol_min,
            "tol_confidence": max(2 * sc, tol_min) if n_bootstrap else tol_min,
            "max_dev": float(np.abs(ecp - alpha).max()), "n_sims": int(f.shape[1]), "n_references": int(f.shape[0])}


# ------------------------------------------------------------------------------ marginals
def marginal_coverage(samples, truth, levels: Sequence[float] = (0.68, 0.95)) -> dict:
    """Per parameter: fraction of truths inside the central credible interval at each level, and
    the KS distance of the truth ranks from uniform (simulation-based calibration in 1D).

    Returns:
        dict with ``"ks"`` and one ``"cov<level in %>"`` entry per level (e.g. ``"cov68"``), each
        a list over the P parameters.
    """
    samples, truth = np.asarray(samples), np.asarray(truth)
    ranks = (samples < truth[:, None, :]).mean(1)                 # (n_sims, P)
    out = {"ks": [], **{f"cov{int(round(100 * l))}": [] for l in levels}}
    n = len(ranks)
    for p in range(ranks.shape[1]):
        r = np.sort(ranks[:, p])
        grid = np.arange(1, n + 1) / n
        out["ks"].append(float(max(np.abs(r - grid).max(), np.abs(r - grid + 1.0 / n).max())))
        for l in levels:
            out[f"cov{int(round(100 * l))}"].append(float(((r > (1 - l) / 2) & (r < (1 + l) / 2)).mean()))
    return out


def posterior_summary(samples, truth=None, weights=None) -> dict:
    """Mean, median, std, 16/84 and 2.5/97.5 percentiles per sim and parameter ((n_sims, P) each),
    and z = (mean - truth) / std if truths are given. Weighted if ``weights`` (n_sims, n_samples)."""
    samples = np.asarray(samples, dtype=np.float64)
    if weights is None:
        q = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=1)
        mean, std = samples.mean(1), samples.std(1)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum(1, keepdims=True)
        mean = (w[..., None] * samples).sum(1)
        std = np.sqrt((w[..., None] * (samples - mean[:, None]) ** 2).sum(1))
        q = np.stack([_weighted_quantile(samples, w, p) for p in (0.025, 0.16, 0.5, 0.84, 0.975)])
    out = {"mean": mean, "std": std, "q025": q[0], "q16": q[1], "median": q[2], "q84": q[3], "q975": q[4]}
    if truth is not None:
        out["z"] = (mean - np.asarray(truth)) / std
    return out


def _weighted_quantile(samples, w, p):
    order = np.argsort(samples, axis=1)
    s = np.take_along_axis(samples, order, 1)
    cw = np.cumsum(np.take_along_axis(np.broadcast_to(w[..., None], s.shape), order, 1), axis=1)
    idx = (cw < p).sum(1).clip(0, s.shape[1] - 1)
    return np.take_along_axis(s, idx[:, None, :], 1)[:, 0]


# ------------------------------------------------------------------------------ FoM
def log_fom(samples, idx: Sequence[int], weights=None) -> np.ndarray:
    """Log figure of merit, -1/2 log det Cov(theta[idx]), per simulation.

    Args:
        samples: (n_sims, n_samples, P).
        idx: columns of the parameters in the figure of merit.
        weights: optional (n_sims, n_samples) importance weights.

    Returns:
        (n_sims,) array; nan where the covariance is not positive definite.
    """
    x = np.asarray(samples)[..., list(idx)].astype(np.float64)
    covs = []
    for i, s in enumerate(x):
        if weights is None:
            covs.append(np.atleast_2d(np.cov(s.T)))
        else:
            w = weights[i]
            covs.append(np.atleast_2d(np.cov(s.T, aweights=w)) if w.sum() > 0 else np.full((len(idx),) * 2, np.nan))
    sign, logdet = np.linalg.slogdet(np.array(covs))
    return np.where(sign > 0, -0.5 * logdet, np.nan)


def importance_weights(samples, from_prior, to_prior) -> np.ndarray:
    """Weights to_prior / from_prior per sample (n_sims, n_samples), normalised to max 1 per sim.
    Use them to turn posteriors under the sampling prior into posteriors under another one."""
    th = torch.as_tensor(np.asarray(samples).reshape(-1, np.shape(samples)[-1]), dtype=torch.float32)
    fp, tp = from_prior.to("cpu"), to_prior.to("cpu")
    with torch.no_grad():
        lw = torch.cat([tp.log_prob(c) - fp.log_prob(c) for c in th.split(1 << 16)])
    lw = lw.reshape(np.shape(samples)[:2]).double().numpy()
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    mx = np.max(np.where(np.isfinite(lw), lw, -np.inf), axis=1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.nan_to_num(np.exp(lw - mx))


# ------------------------------------------------------------------------------ standard record
def evaluate(samples, truth, param_names: Sequence[str],
             fom_sets: Mapping[str, Sequence[str]] = None,
             tarp_sets: Mapping[str, Sequence[str]] = None,
             n_references: int = 50, n_bootstrap: int = 200, tol_min: float = 0.02, seed: int = 0,
             device="cpu") -> tuple[dict, dict]:
    """The standard metrics record for posteriors of simulations with known truths.

    Using the same record everywhere makes runs, compressions and analyses comparable.

    Args:
        samples: (n_sims, n_samples, P) posterior samples.
        truth: (n_sims, P) true parameters.
        param_names: the P names.
        fom_sets: name -> parameter names of a figure of merit, e.g. ``{"om_s8": ["Om", "S8"]}``.
        tarp_sets: name -> parameter names of a TARP test; ``"all"`` is always included.
        n_references, n_bootstrap, tol_min, seed: see :func:`tarp_f` and :func:`tarp_scalars`.
        device: torch device of the TARP distances.

    Returns:
        (record, arrays). ``record`` is JSON-serialisable: ``tarp[set]`` (:func:`tarp_scalars`),
        ``fom[set]`` (mean log FoM, geometric-mean FoM, number of nan), ``marginal[param]``
        (:func:`marginal_coverage`), ``z_mean`` / ``z_rms`` of (posterior mean - truth) / std and
        ``std_median`` per parameter. ``arrays`` holds posterior means, stds and medians, the TARP
        curves (``tarp_alpha``, ``tarp_ecp_<set>``) and ``log_fom_<set>`` per simulation.
    """
    names = list(param_names)
    samples, truth = np.asarray(samples), np.asarray(truth)
    fom_sets = dict(fom_sets or {})
    tarp_sets = {"all": names, **dict(tarp_sets or {})}
    summ = posterior_summary(samples, truth)
    record = {"n_sims": int(len(truth)), "param_names": names, "tarp": {}, "fom": {},
              "z_mean": dict(zip(names, summ["z"].mean(0).round(4).tolist())),
              "z_rms": dict(zip(names, np.sqrt((summ["z"] ** 2).mean(0)).round(4).tolist())),
              "std_median": dict(zip(names, np.median(summ["std"], 0).tolist()))}
    mc = marginal_coverage(samples, truth)
    record["marginal"] = {n: {k: v[p] for k, v in mc.items()} for p, n in enumerate(names)}
    arrays = {"post_mean": summ["mean"], "post_std": summ["std"], "post_median": summ["median"]}
    for key, ps in tarp_sets.items():
        idx = [names.index(p) for p in ps]
        f = tarp_f(samples[..., idx], truth[:, idx], n_references, seed, device)
        record["tarp"][key] = tarp_scalars(f, n_bootstrap, tol_min, seed)
        alpha, ecp = ecp_curve(f)
        arrays["tarp_alpha"], arrays[f"tarp_ecp_{key}"] = alpha, ecp
    for key, ps in fom_sets.items():
        lf = log_fom(samples, [names.index(p) for p in ps])
        record["fom"][key] = {"log_fom_mean": float(np.nanmean(lf)), "fom_geomean": float(np.exp(np.nanmean(lf))),
                              "n_nan": int(np.isnan(lf).sum())}
        arrays[f"log_fom_{key}"] = lf
    return record, arrays
