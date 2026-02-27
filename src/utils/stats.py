import numpy as np
import pandas as pd
import math

from scipy.stats import binom
from scipy.stats import beta as beta_dist
from scipy.special import betaln  # log Beta function

import matplotlib.pyplot as plt


"""
Utilities for binomial calibration: simulation, p-values, e-values, and rejection rules.

This module provides:
- RNG helpers (consistent seeding)
- Binomial simulation draws
- e-value: Beta-mixture e-value for H0: p = p0
- p-values:
    * PMF-based exact two-sided ("as or more extreme")
    * Clopper–Pearson two-sided (equal-tail)
- Jeffreys interval + Jeffreys test via CI inversion
- One-stop calibration_test_binom returning consistent keys used by simulations
"""


# -------------------------
# Randomness / simulation
# -------------------------
def make_rng(random_state=None):
    """
    Create a NumPy Generator for reproducibility.

    Parameters
    ----------
    random_state : None | int | np.random.Generator
        - None: new unpredictable generator
        - int: seed
        - Generator: returned as-is

    Returns
    -------
    np.random.Generator
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def binomial_draw(n, p, rng=None, random_state=None):
    """
    One draw from Binomial(n, p).

    Use either `rng` OR `random_state` (not both).
    Accepts boundary values p in [0,1].
    """
    if rng is not None and random_state is not None:
        raise ValueError("Use either rng or random_state, not both.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if not (isinstance(n, (int, np.integer)) and n > 0):
        raise ValueError("n must be a strictly positive integer.")

    rng = make_rng(random_state) if rng is None else rng
    return int(rng.binomial(int(n), float(p)))


def binomial_sample(n, p, size=1, rng=None, random_state=None):
    """
    Multiple independent draws from Binomial(n, p).

    Use either `rng` OR `random_state` (not both).
    Accepts boundary values p in [0,1].

    Returns
    -------
    np.ndarray (shape = (size,))
    """
    if rng is not None and random_state is not None:
        raise ValueError("Use either rng or random_state, not both.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if not (isinstance(n, (int, np.integer)) and n > 0):
        raise ValueError("n must be a strictly positive integer.")
    if not (isinstance(size, (int, np.integer)) and size > 0):
        raise ValueError("size must be a strictly positive integer.")

    rng = make_rng(random_state) if rng is None else rng
    return rng.binomial(int(n), float(p), size=int(size))


# -------------------------
# E-values
# -------------------------
def evalue_beta_mixture_binom(d, n, p0, a, b):
    """
    Beta(a,b)-mixture e-value for testing H0: p = p0 (binomial data).

    E(d) = B(d+a, n-d+b) / ( B(a,b) * p0^d * (1-p0)^(n-d) )

    Parameters
    ----------
    d : int
    n : int
    p0 : float in (0,1)
    a, b : float > 0

    Returns
    -------
    float
    """
    if not (0 < p0 < 1):
        raise ValueError("p0 must be in (0,1).")
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0.")
    if d < 0 or d > n:
        return np.nan

    log_num = betaln(d + a, (n - d) + b)
    log_den = betaln(a, b) + d * math.log(p0) + (n - d) * math.log(1 - p0)
    return float(np.exp(log_num - log_den))


# -------------------------
# P-values / Confidence intervals
# -------------------------
def pvalue_exact_binom_two_sided_pmf(d, n, p0):
    """
    Two-sided exact binomial p-value using the "as or more extreme" PMF criterion:
        p = sum_{k: pmf(k) <= pmf(d)} pmf(k)   under Binom(n,p0)
    """
    if not (0 < p0 < 1):
        raise ValueError("p0 must be in (0,1).")
    if d < 0 or d > n:
        return np.nan

    pmf_obs = binom.pmf(d, n, p0)
    ks = np.arange(n + 1)
    pmf_all = binom.pmf(ks, n, p0)
    return float(pmf_all[pmf_all <= pmf_obs + 1e-15].sum())


def _safe_beta_ppf(q, a, b):
    """Beta quantile helper with edge handling."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    return float(beta_dist.ppf(q, a, b))


def clopper_pearson_ci(d, n, alpha=0.05):
    """
    Clopper–Pearson exact (1-alpha) CI for p.

    Lower = Beta^{-1}(alpha/2; d,   n-d+1)   if d>0 else 0
    Upper = Beta^{-1}(1-alpha/2; d+1, n-d)   if d<n else 1
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    if d < 0 or d > n:
        return (np.nan, np.nan)

    if d == 0:
        lo = 0.0
    else:
        lo = _safe_beta_ppf(alpha / 2.0, d, n - d + 1)

    if d == n:
        hi = 1.0
    else:
        hi = _safe_beta_ppf(1.0 - alpha / 2.0, d + 1, n - d)

    return (lo, hi)


def pvalue_clopper_pearson_two_sided(d, n, p0):
    """
    Two-sided Clopper–Pearson p-value (equal-tail).
    Often written:
        p = 2 * min( P_{p0}(D <= d), P_{p0}(D >= d) ), clipped at 1.
    """
    if not (0 < p0 < 1):
        raise ValueError("p0 must be in (0,1).")
    if d < 0 or d > n:
        return np.nan

    left = binom.cdf(d, n, p0)       # P(D <= d)
    right = binom.sf(d - 1, n, p0)   # P(D >= d)
    return float(min(1.0, 2.0 * min(left, right)))


def jeffreys_ci(d, n, alpha=0.05):
    """
    Jeffreys Bayesian equal-tail (1-alpha) credible interval for p
    with Jeffreys prior Beta(1/2, 1/2).

    Posterior: Beta(d+1/2, n-d+1/2).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    if d < 0 or d > n:
        return (np.nan, np.nan)

    a_post = d + 0.5
    b_post = (n - d) + 0.5
    lo = _safe_beta_ppf(alpha / 2.0, a_post, b_post)
    hi = _safe_beta_ppf(1.0 - alpha / 2.0, a_post, b_post)
    return (lo, hi)


def reject_from_ci(p0, ci):
    """Reject H0: p=p0 if p0 is outside the interval."""
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return False
    return bool((p0 < lo) or (p0 > hi))


# -------------------------
# Rejection rules
# -------------------------
def reject_from_e(e, alpha=0.05):
    """
    Reject H0 using e-values at level alpha via Markov's inequality:
        Reject if e >= 1/alpha
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    return bool(e >= (1.0 / alpha))


def reject_from_p(p, alpha=0.05):
    """Reject H0 using p-values: reject if p <= alpha."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    return bool(p <= alpha)


# -------------------------
# One-stop calibration test helper (consistent API)
# -------------------------
def calibration_test_binom(d, n, p0, alpha=0.05, e_params=(0.5, 0.5)):
    """
    Compute multiple tests + e-value and return consistent keys.

    Keys returned (used by simulation code):
      - reject_e
      - reject_pmf
      - reject_clopper_pearson
      - reject_jeffreys_via_ci

    Also returns the underlying p-values / CI / thresholds.
    """
    a, b = e_params

    # e-value
    e_val = evalue_beta_mixture_binom(d, n, p0, a, b)

    # PMF exact p-value
    p_pmf = pvalue_exact_binom_two_sided_pmf(d, n, p0)
    rej_pmf = reject_from_p(p_pmf, alpha)

    # Clopper–Pearson p-value + CI
    p_cp = pvalue_clopper_pearson_two_sided(d, n, p0)
    ci_cp = clopper_pearson_ci(d, n, alpha=alpha)
    rej_cp = reject_from_p(p_cp, alpha)

    # Jeffreys CI inversion
    ci_j = jeffreys_ci(d, n, alpha=alpha)
    rej_j = reject_from_ci(p0, ci_j)

    return {
        "d": int(d),
        "n": int(n),
        "p0": float(p0),
        "alpha": float(alpha),

        # e-value
        "e_value": float(e_val),
        "reject_e": reject_from_e(e_val, alpha),
        "threshold_e": 1.0 / alpha,
        "e_params": (float(a), float(b)),

        # PMF p-value
        "p_value_pmf": float(p_pmf),
        "reject_pmf": bool(rej_pmf),

        # Clopper–Pearson
        "p_value_clopper_pearson": float(p_cp),
        "reject_clopper_pearson": bool(rej_cp),
        "ci_clopper_pearson": tuple(map(float, ci_cp)),
        "reject_cp_via_ci": reject_from_ci(p0, ci_cp),

        # Jeffreys
        "ci_jeffreys": tuple(map(float, ci_j)),
        "reject_jeffreys_via_ci": bool(rej_j),
    }