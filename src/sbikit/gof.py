"""Goodness of fit: is a summary vector a plausible draw from the learned likelihood?

- :func:`likelihood_chi2`: distance of summaries from the likelihood at given parameters (the
  truths of test simulations, or the posterior maximum of an observation);
- :func:`predictive_statistic`: the likelihood of summaries averaged over posterior samples, as in
  the posterior predictive tests of Doux et al. (2021) with the likelihood of the measurement as
  the statistic;
- :func:`calibrated_pvalue`: the place of a statistic among the same statistic of simulations.

All statistics grow with the misfit. Only the chi2 of a Gaussian likelihood has a known
distribution, so a statistic of an observation is read against the same statistic of simulations
that went through the same steps. Which simulations matters: realisations at one parameter point
make a sharper reference than simulations from the whole prior, whose distribution includes the
regions where the likelihood is learned poorly.

A test only sees what the summaries keep. In directions the likelihood has not learned (summaries
of parameters it is not conditioned on, or that the data hardly constrain) a statistic measures
the distance from the centre of the training set; leave them out with ``columns``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ._utils import as_numpy, check_finite, to_tensor

__all__ = ["likelihood_chi2", "predictive_statistic", "calibrated_pvalue"]

_VALUES = 1 << 24       # values of likelihood draws held in memory at a time
_ROWS = 1 << 16         # likelihood evaluations per call


# ------------------------------------------------------------------------------ helpers
def _rows(a, name: str, n_cols: Optional[int] = None) -> np.ndarray:
    """``a`` as a finite (M, C) float64 array; a single vector becomes one row."""
    a = as_numpy(a)
    a = a[None] if a.ndim == 1 else a
    if a.ndim != 2 or a.size == 0 or (n_cols is not None and a.shape[1] != n_cols):
        expected = "columns" if n_cols is None else f"{n_cols} columns"
        raise ValueError(f"{name} must be (M rows, {expected}) or a single vector; got shape {a.shape}")
    check_finite(a, name)
    return a


def _cols(columns, n: int) -> np.ndarray:
    """Validated column indices of the ``n`` summaries (None: all)."""
    if columns is None:
        return np.arange(n)
    cols = np.atleast_1d(np.asarray(columns))
    if (cols.ndim != 1 or cols.size == 0 or not np.issubdtype(cols.dtype, np.integer)
            or (cols < -n).any() or (cols >= n).any() or len(set((cols % n).tolist())) != cols.size):
        raise ValueError(f"columns must be distinct column indices of the {n} summaries, got {columns!r}")
    return cols % n


def _seed(seed: Optional[int], i: int) -> Optional[int]:
    """The seed of chunk ``i``."""
    return None if seed is None else int(np.random.SeedSequence([int(seed), i]).generate_state(1)[0])


def _draw(likelihood, theta: np.ndarray, n: int, seed: Optional[int]) -> np.ndarray:
    """(n, M, n_x) summaries drawn from the likelihood at the M rows of ``theta``."""
    if not callable(getattr(likelihood, "sample", None)):
        raise TypeError(f"likelihood must have a sample(theta, n, seed=) method, as NLE and Ensemble have; got "
                        f"{type(likelihood).__name__}")
    return as_numpy(likelihood.sample(theta, n, seed=seed))


def _log_likelihood(likelihood, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """log p(s_i | theta_i) for matching rows, (M,)."""
    if callable(getattr(likelihood, "log_likelihood", None)):
        return as_numpy(likelihood.log_likelihood(s, theta))
    device = getattr(likelihood, "device", "cpu")
    with torch.no_grad():
        return as_numpy(likelihood.log_prob(to_tensor(s, dtype=torch.float32, device=device).unsqueeze(0),
                                            to_tensor(theta, dtype=torch.float32, device=device)).reshape(-1))


def _gaussian(x: np.ndarray, draws: np.ndarray) -> tuple:
    """``(x - mean)^T C^-1 (x - mean)`` and ``log det(2 pi C)`` for the mean and covariance of the
    draws: ``x`` (M, d), ``draws`` (n, M, d) or (n, 1, d) (the same draws for every row)."""
    n, d = draws.shape[0], draws.shape[-1]
    mean = draws.mean(0)
    r = draws - mean
    cov = np.einsum("nmi,nmj->mij", r, r) / (n - 1)
    sd = np.sqrt(np.einsum("mii->mi", cov))
    if not (sd > 0).all():
        raise ValueError("a summary has the same value in every draw of the likelihood")
    v = (x - mean) / sd                                    # distances in units of the scatter of each summary
    corr = cov / (sd[:, :, None] * sd[:, None, :])
    chi2 = np.einsum("mi,mi->m", v, np.linalg.solve(corr, v[..., None])[..., 0])
    logdet = np.linalg.slogdet(corr)[1] + 2 * np.log(sd).sum(-1) + d * np.log(2 * np.pi)
    return chi2, np.broadcast_to(logdet, chi2.shape)


# ------------------------------------------------------------------------------ statistics
def likelihood_chi2(likelihood, s, theta, n_draws: int = 1000, columns=None, seed: Optional[int] = 0) -> np.ndarray:
    """Squared Mahalanobis distance of every summary vector from the likelihood p(s | theta_i).

    The mean and covariance of each likelihood are estimated from ``n_draws`` draws, and the
    inverse covariance carries the Hartlap factor ``(n_draws - d - 2) / (n_draws - 1)``. For a
    Gaussian likelihood this is a chi2 with d degrees of freedom, so that over test simulations at
    their true parameters it shows whether the learned likelihoods are too wide or too narrow and
    where they fail. For an observation at its posterior maximum (:func:`~sbikit.sampling.find_map`)
    it is a goodness of fit, to be read against test simulations at their own maxima.

    Args:
        likelihood: an :class:`~sbikit.nle.NLE` or :class:`~sbikit.nle.Ensemble`, or anything
            with ``sample(theta, n, seed=)`` returning (n, M, n_x) draws.
        s: (M, n_x) summaries, or a single (n_x,) vector.
        theta: (M, n_theta) parameters, or one point for all rows of ``s`` (e.g. the fiducial
            point of noise realisations).
        n_draws: draws per likelihood (more than d + 2).
        columns: indices of the summaries to use (default: all d = n_x).
        seed: seed of the draws (None: torch's global random state).

    Returns:
        (M,) array.

    Raises:
        ValueError: for shapes that do not fit together, invalid ``columns``, NaN or inf in the
            inputs, or too few draws.
        TypeError: for a likelihood without ``sample``.

    Example:
        >>> chi2 = likelihood_chi2(nle, s_test, theta_test)            # compare with chi2(n_x)
        >>> theta_map, _ = find_map(nle, s_obs, prior, post.samples)
        >>> likelihood_chi2(nle, s_obs, theta_map)
    """
    s, theta = _rows(s, "s"), _rows(theta, "theta")
    cols = _cols(columns, s.shape[1])
    d = len(cols)
    if not (isinstance(n_draws, (int, np.integer)) and n_draws > d + 2):
        raise ValueError(f"n_draws must be an int larger than the number of summaries + 2 = {d + 2}, got {n_draws!r}")
    if len(theta) == 1:
        chi2 = _gaussian(s[:, cols], _draw(likelihood, theta, n_draws, seed)[..., cols])[0]
    else:
        if len(s) == 1:
            s = np.repeat(s, len(theta), 0)
        if len(s) != len(theta):
            raise ValueError(f"s has {len(s)} rows but theta has {len(theta)}; give matching rows, a single s or a "
                             "single theta")
        step = max(1, _VALUES // (n_draws * s.shape[1]))
        chi2 = np.concatenate([_gaussian(s[i:i + step, cols],
                                         _draw(likelihood, theta[i:i + step], n_draws, _seed(seed, k))[..., cols])[0]
                               for k, i in enumerate(range(0, len(s), step))])
    return chi2 * (n_draws - d - 2) / (n_draws - 1)


def predictive_statistic(likelihood, s, samples, columns=None, seed: Optional[int] = 0) -> np.ndarray:
    """Posterior predictive statistic T_b of every summary vector s_b, given posterior samples.

    Without ``columns``, ``T_b = -2 <log p(s_b | theta)>`` over the samples of target b, with the
    density of the likelihood itself. With ``columns``, the posterior predictive distribution of
    those summaries (one draw of the likelihood per sample) is approximated by a Gaussian and
    ``T_b = -2 log`` of its density at s_b, since a flow has no marginals.

    With the posterior of the same summaries this tests the goodness of fit. With the posterior of
    another analysis (part of the data vector removed, another statistic) it tests whether the two
    are consistent. The values mean nothing on their own: compute the statistic in the same way for
    simulations and compare with :func:`calibrated_pvalue`.

    Args:
        likelihood: an :class:`~sbikit.nle.NLE` or :class:`~sbikit.nle.Ensemble`, or anything
            with ``log_prob(input, condition)`` in sbi's layout (and, for ``columns``,
            ``sample(theta, n, seed=)``).
        s: (B, n_x) summaries, or a single (n_x,) vector.
        samples: (B, n, n_theta) posterior samples in the parameters the likelihood is conditioned
            on (``Posteriors.samples``, or ``Posteriors.derived`` if a transform was used), or
            (n, n_theta) for a single target.
        columns: indices of the summaries to use (default: all, with the exact density).
        seed: seed of the likelihood draws used with ``columns`` (None: torch's global random
            state).

    Returns:
        (B,) array.

    Raises:
        ValueError: for shapes that do not fit together, invalid ``columns``, NaN or inf in the
            inputs, or (with ``columns``) fewer samples than summaries + 2.
        TypeError: with ``columns``, for a likelihood without ``sample``.

    Example:
        >>> t_obs = predictive_statistic(nle, s_obs, post_obs.samples)
        >>> t_ref = predictive_statistic(nle, s_test, post_test.samples)
        >>> calibrated_pvalue(t_obs, t_ref)
    """
    s, theta = _rows(s, "s"), as_numpy(samples)
    theta = theta[None] if theta.ndim == 2 else theta
    if theta.ndim != 3 or theta.shape[0] != len(s) or 0 in theta.shape:
        raise ValueError(f"samples must be ({len(s)}, n, n_theta) (one set of posterior samples per row of s), or "
                         f"(n, n_theta) for a single s; got shape {as_numpy(samples).shape}")
    if not np.isfinite(theta).all():
        raise ValueError("samples contain NaN or inf")
    B, n, P = theta.shape
    step = max(1, _ROWS // n)
    out = []
    if columns is None:
        for i in range(0, B, step):
            lp = _log_likelihood(likelihood, np.repeat(s[i:i + step], n, 0), theta[i:i + step].reshape(-1, P))
            out.append(-2.0 * lp.reshape(-1, n).mean(1))
        return np.concatenate(out)
    cols = _cols(columns, s.shape[1])
    if n < len(cols) + 2:
        raise ValueError(f"the Gaussian of {len(cols)} summaries needs at least {len(cols) + 2} posterior samples "
                         f"per target, got {n}")
    for k, i in enumerate(range(0, B, step)):
        draws = _draw(likelihood, theta[i:i + step].reshape(-1, P), 1, _seed(seed, k))[0]      # (targets * n, n_x)
        chi2, logdet = _gaussian(s[i:i + step, cols], draws.reshape(-1, n, draws.shape[-1]).swapaxes(0, 1)[..., cols])
        out.append(chi2 + logdet)
    return np.concatenate(out)


def calibrated_pvalue(statistic, reference):
    """The fraction of reference simulations with a statistic at least as large.

    ``p = (R + 1) / (n + 1)`` for R of the n reference values >= the statistic: uniform if the
    statistic comes from the same distribution as the reference, and small for a misfit. For
    quantities that fall with the misfit, such as a log posterior density, pass their negatives.

    Args:
        statistic: a value or an array of values of a statistic that grows with the misfit.
        reference: the same statistic for n simulations that are consistent by construction.

    Returns:
        p-values with the shape of ``statistic`` (a float for a single value).

    Raises:
        ValueError: for an empty reference or NaN in either argument.

    Example:
        >>> _, logp_obs = find_map(nle, s_obs, prior, post_obs.samples)
        >>> _, logp_ref = find_map(nle, s_test, prior, post_test.samples)
        >>> calibrated_pvalue(-logp_obs, -logp_ref)
    """
    t, ref = as_numpy(statistic), as_numpy(reference).ravel()
    if ref.size == 0 or np.isnan(ref).any() or np.isnan(t).any():
        raise ValueError("statistic and reference must not hold NaN, and reference must not be empty")
    p = ((ref[None] >= t.reshape(-1, 1)).sum(1) + 1) / (ref.size + 1)
    return float(p[0]) if t.ndim == 0 else p.reshape(t.shape)
