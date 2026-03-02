# src/simulation/run_sim_coverage.py
"""
Classic (frequentist) pointwise coverage simulation via tests.

Definition
----------
For each p_true:
    - simulate D ~ Binomial(n, p_true)
    - test H0: p = p_true  (i.e., p0 := p_true at this grid point)
Coverage(p_true) = P_{p_true}(do NOT reject H0) = 1 - rejection_rate.

Experiments
-----------
1) Fixed n, vary p_true in [p_min, p_max]
   -> coverage vs p_true (should be ~ 1 - alpha for all p_true, up to discreteness)

2) Fixed p_true, vary n from n_min to n_max (log-spaced)
   -> coverage vs n (should be ~ 1 - alpha for all n, up to discreteness)

Outputs
-------
Saved to:
    outputs/simulation/coverage/

Run examples
------------
PYTHONPATH=src python src/simulation/run_sim_coverage.py --n-mc 3000 --alpha 0.05
PYTHONPATH=src python src/simulation/run_sim_coverage.py --n-fixed 2000 --p-min 0.0 --p-max 0.03 --p-steps 61
PYTHONPATH=src python src/simulation/run_sim_coverage.py --p-true-fixed 0.01 --n-min 100 --n-max 200000 --n-steps 20
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils.stats import make_rng, binomial_draw, calibration_test_binom


# ============================================================
# Monte Carlo engines
# ============================================================

def run_mc_grid_over_p(n, p_grid, n_mc, alpha, e_params, rng):
    """
    Fix n, vary p_true over p_grid.

    Pointwise null:
        at each p_true, test H0: p = p_true (so p0 := p_true).

    Returns a DataFrame with rejection rates + coverage rates (= 1 - rejection).
    """
    rows = []
    for p_true in p_grid:
        rejs_e = 0
        rejs_pmf = 0
        rejs_cp = 0
        rejs_j = 0

        for _ in range(n_mc):
            d = binomial_draw(n, p_true, rng=rng)
            out = calibration_test_binom(
                d=d,
                n=n,
                p0=p_true,          # <-- pointwise null
                alpha=alpha,
                e_params=e_params,
            )

            rejs_e += int(out["reject_e"])
            rejs_pmf += int(out["reject_pmf"])
            rejs_cp += int(out["reject_clopper_pearson"])
            rejs_j += int(out["reject_jeffreys_via_ci"])

        rej_e = rejs_e / n_mc
        rej_pmf = rejs_pmf / n_mc
        rej_cp = rejs_cp / n_mc
        rej_j = rejs_j / n_mc

        rows.append({
            "p_true": float(p_true),
            "n": int(n),
            "alpha": float(alpha),
            "n_mc": int(n_mc),

            "rej_rate_e": rej_e,
            "rej_rate_pmf": rej_pmf,
            "rej_rate_cp": rej_cp,
            "rej_rate_jeffreys": rej_j,

            "cov_rate_e": 1.0 - rej_e,
            "cov_rate_pmf": 1.0 - rej_pmf,
            "cov_rate_cp": 1.0 - rej_cp,
            "cov_rate_jeffreys": 1.0 - rej_j,
        })

    return pd.DataFrame(rows)


def run_mc_grid_over_n(p_true, n_grid, n_mc, alpha, e_params, rng):
    """
    Fix p_true, vary n over n_grid.

    Pointwise null:
        test H0: p = p_true for each n (so p0 := p_true).

    Returns a DataFrame with rejection rates + coverage rates (= 1 - rejection).
    """
    rows = []
    for n in n_grid:
        n = int(n)
        rejs_e = 0
        rejs_pmf = 0
        rejs_cp = 0
        rejs_j = 0

        for _ in range(n_mc):
            d = binomial_draw(n, p_true, rng=rng)
            out = calibration_test_binom(
                d=d,
                n=n,
                p0=p_true,          # <-- pointwise null
                alpha=alpha,
                e_params=e_params,
            )

            rejs_e += int(out["reject_e"])
            rejs_pmf += int(out["reject_pmf"])
            rejs_cp += int(out["reject_clopper_pearson"])
            rejs_j += int(out["reject_jeffreys_via_ci"])

        rej_e = rejs_e / n_mc
        rej_pmf = rejs_pmf / n_mc
        rej_cp = rejs_cp / n_mc
        rej_j = rejs_j / n_mc

        rows.append({
            "p_true": float(p_true),
            "n": int(n),
            "alpha": float(alpha),
            "n_mc": int(n_mc),

            "rej_rate_e": rej_e,
            "rej_rate_pmf": rej_pmf,
            "rej_rate_cp": rej_cp,
            "rej_rate_jeffreys": rej_j,

            "cov_rate_e": 1.0 - rej_e,
            "cov_rate_pmf": 1.0 - rej_pmf,
            "cov_rate_cp": 1.0 - rej_cp,
            "cov_rate_jeffreys": 1.0 - rej_j,
        })

    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo pointwise coverage simulation (binomial calibration)."
    )

    # Core parameters
    p.add_argument("--alpha", type=float, default=0.05, help="Test level.")
    p.add_argument("--n-mc", type=int, default=3000, help="Monte Carlo repetitions per grid point.")
    p.add_argument("--seed", type=int, default=123, help="RNG seed.")

    # e-value mixture params (passed to calibration_test_binom)
    p.add_argument("--e-a", type=float, default=0.5, help="Beta-mixture a parameter.")
    p.add_argument("--e-b", type=float, default=0.5, help="Beta-mixture b parameter.")

    # Experiment 1: vary p_true with fixed n
    p.add_argument("--n-fixed", type=int, default=2000, help="n for experiment 1.")
    p.add_argument("--p-min", type=float, default=0.0, help="Min p_true for experiment 1.")
    p.add_argument("--p-max", type=float, default=0.2, help="Max p_true for experiment 1.")
    p.add_argument("--p-steps", type=int, default=41, help="Number of p_true grid points.")

    # Experiment 2: vary n with fixed p_true
    p.add_argument("--p-true-fixed", type=float, default=0.01, help="p_true for experiment 2.")
    p.add_argument("--n-min", type=int, default=100, help="Min n for experiment 2.")
    p.add_argument("--n-max", type=int, default=100000, help="Max n for experiment 2.")
    p.add_argument("--n-steps", type=int, default=18, help="Number of n grid points (log-spaced).")

    # Outputs
    p.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("outputs", "simulation", "coverage"),
        help="Output directory for CSV results.",
    )

    return p.parse_args()


def _validate_args(args):
    if not (0 < args.alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    if args.n_mc <= 0:
        raise ValueError("--n-mc must be > 0.")
    if args.n_fixed <= 0:
        raise ValueError("--n-fixed must be > 0.")
    if args.p_steps <= 1:
        raise ValueError("--p-steps must be > 1.")
    if args.n_min <= 0 or args.n_max <= 0 or args.n_min > args.n_max:
        raise ValueError("Need 0 < n-min <= n-max.")
    if args.n_steps <= 1:
        raise ValueError("--n-steps must be > 1.")
    if args.p_min < 0 or args.p_max > 1 or args.p_min > args.p_max:
        raise ValueError("Need 0 <= p-min <= p-max <= 1.")
    if not (0 < args.p_true_fixed < 1):
        raise ValueError("--p-true-fixed must be in (0,1).")


def main():
    args = parse_args()
    _validate_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    rng = make_rng(args.seed)
    e_params = (args.e_a, args.e_b)

    # ------------------------
    # Experiment 1: vary p_true (pointwise coverage)
    # ------------------------
    eps = 1e-12
    p_grid = np.linspace(args.p_min, args.p_max, args.p_steps)
    p_grid = np.clip(p_grid, eps, 1.0 - eps)
    p_grid = np.unique(p_grid)  # avoid duplicates after clipping

    df_p = run_mc_grid_over_p(
        n=args.n_fixed,
        p_grid=p_grid,
        n_mc=args.n_mc,
        alpha=args.alpha,
        e_params=e_params,
        rng=rng,
    )

    path_p = os.path.join(args.outdir, "coverage_vary_p.csv")
    df_p.to_csv(path_p, index=False)
    print(f"Saved: {path_p}")

    # ------------------------
    # Experiment 2: vary n (pointwise coverage)
    # ------------------------
    n_grid = np.unique(
        np.round(np.logspace(np.log10(args.n_min), np.log10(args.n_max), args.n_steps)).astype(int)
    )

    df_n = run_mc_grid_over_n(
        p_true=args.p_true_fixed,
        n_grid=n_grid,
        n_mc=args.n_mc,
        alpha=args.alpha,
        e_params=e_params,
        rng=rng,
    )

    path_n = os.path.join(args.outdir, "coverage_vary_n.csv")
    df_n.to_csv(path_n, index=False)
    print(f"Saved: {path_n}")


if __name__ == "__main__":
    main()