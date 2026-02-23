# Cell 1 — Imports
import numpy as np
import pandas as pd
import math

from scipy.stats import binom, norm
from scipy.special import betaln  # log Beta function

import matplotlib.pyplot as plt


"""

Script for the stats function e-values, etc...

-> To define better latter

"""


def evalue_beta_mixture_binom(d, n, pd0, a, b):
    """
    Beta(a,b)-mixture e-value:
    E = B(d+a, n-d+b) / (B(a,b) * pd0^d * (1-pd0)^(n-d)).
    """
    if pd0 <= 0 or pd0 >= 1:
        raise ValueError("pd0 must be in (0,1).")
    if a <= 0 or b <= 0:
        raise ValueError("a,b must be > 0.")
    if d < 0 or d > n:
        return np.nan

    log_num = betaln(d + a, (n - d) + b)
    log_den = betaln(a, b) + d * math.log(pd0) + (n - d) * math.log(1 - pd0)
    return float(np.exp(log_num - log_den))


def pvalue_exact_binom_two_sided(d, n, pd0):
    """
    Two-sided exact binomial p-value (Clopper-Pearson style tail sum, "as or more extreme").
    We compute p = P( Binom(n,pd0) has pmf <= pmf(d) ).
    This is a standard exact two-sided definition used in practice.
    """
    pmf_obs = binom.pmf(d, n, pd0)
    ks = np.arange(n + 1)
    pmf_all = binom.pmf(ks, n, pd0)
    return float(pmf_all[pmf_all <= pmf_obs + 1e-15].sum())


def reject_from_e(e, alpha=0.05):
    return e >= (1 / alpha)


def reject_from_p(p, alpha=0.05):
    return p <= alpha


