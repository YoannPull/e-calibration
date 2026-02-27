# src/simulation/make_coverage.py
"""
Monte Carlo coverage simulations.

Experiments:
1) Fixed n, vary p_true in [p_min, p_max]
2) Fixed p_true, vary n from n_min to n_max (log-spaced)

Outputs saved to:
    outputs/simulation/coverage/

Run example:
    PYTHONPATH=src python src/simulation/make_coverage.py --n-mc 3000
"""

import os
import argparse
import numpy as np
import pandas as pd

from utils.stats import make_rng, binomial_draw, calibration_test_binom


# ============================================================
# Monte Carlo engines
# ============================================================

def run_mc_grid_over_p(n, p0, p_grid, n_mc, alpha, e_params, rng):
    rows = []
    for p_true in p_grid:
        rejs_e = rejs_pmf = rejs_cp = rejs_j = 0

        for _ in range(n_mc):
            d = binomial_draw(n, p_true, rng=rng)
            out = calibration_test_binom(d=d, n=n, p0=p0, alpha=alpha, e_params=e_params)

            rejs_e += int(out["reject_e"])
            rejs_pmf += int(out["reject_pmf"])
            rejs_cp += int(out["reject_clopper_pearson"])
            rejs_j += int(out["reject_jeffreys_via_ci"])

        rows.append({
            "p_true": float(p_true),
            "n": int(n),
            "p0": float(p0),
            "alpha": float(alpha),
            "n_mc": int(n_mc),
            "rej_rate_e": rejs_e / n_mc,
            "rej_rate_pmf": rejs_pmf / n_mc,
            "rej_rate_cp": rejs_cp / n_mc,
            "rej_rate_jeffreys": rejs_j / n_mc,
        })

    return pd.DataFrame(rows)


def run_mc_grid_over_n(p_true, p0, n_grid, n_mc, alpha, e_params, rng):
    rows = []
    for n in n_grid:
        rejs_e = rejs_pmf = rejs_cp = rejs_j = 0

        for _ in range(n_mc):
            d = binomial_draw(n, p_true, rng=rng)
            out = calibration_test_binom(d=d, n=n, p0=p0, alpha=alpha, e_params=e_params)

            rejs_e += int(out["reject_e"])
            rejs_pmf += int(out["reject_pmf"])
            rejs_cp += int(out["reject_clopper_pearson"])
            rejs_j += int(out["reject_jeffreys_via_ci"])

        rows.append({
            "p_true": float(p_true),
            "n": int(n),
            "p0": float(p0),
            "alpha": float(alpha),
            "n_mc": int(n_mc),
            "rej_rate_e": rejs_e / n_mc,
            "rej_rate_pmf": rejs_pmf / n_mc,
            "rej_rate_cp": rejs_cp / n_mc,
            "rej_rate_jeffreys": rejs_j / n_mc,
        })

    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo coverage simulations (binomial calibration).")

    # core
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--p0", type=float, default=0.01)
    p.add_argument("--n-mc", type=int, default=3000)
    p.add_argument("--seed", type=int, default=123)

    # e-value mixture params
    p.add_argument("--e-a", type=float, default=0.5)
    p.add_argument("--e-b", type=float, default=0.5)

    # exp1: vary p
    p.add_argument("--n-fixed", type=int, default=2000)
    p.add_argument("--p-min", type=float, default=0.0)
    p.add_argument("--p-max", type=float, default=0.2)
    p.add_argument("--p-steps", type=int, default=41)

    # exp2: vary n
    p.add_argument("--p-true-fixed", type=float, default=0.01)
    p.add_argument("--n-min", type=int, default=100)
    p.add_argument("--n-max", type=int, default=100000)
    p.add_argument("--n-steps", type=int, default=18)

    # outputs
    p.add_argument("--outdir", type=str, default=os.path.join("outputs", "simulation", "coverage"))

    return p.parse_args()


def main():
    args = parse_args()

    if not (0 < args.alpha < 1):
        raise ValueError("alpha must be in (0,1).")
    if not (0 < args.p0 < 1):
        raise ValueError("p0 must be in (0,1).")
    if args.n_mc <= 0:
        raise ValueError("--n-mc must be > 0.")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    rng = make_rng(args.seed)
    e_params = (args.e_a, args.e_b)

    # ------------------------
    # Experiment 1
    # ------------------------
    p_grid = np.linspace(args.p_min, args.p_max, args.p_steps)
    df_p = run_mc_grid_over_p(
        n=args.n_fixed,
        p0=args.p0,
        p_grid=p_grid,
        n_mc=args.n_mc,
        alpha=args.alpha,
        e_params=e_params,
        rng=rng,
    )
    path_p = os.path.join(outdir, "coverage_vary_p.csv")
    df_p.to_csv(path_p, index=False)
    print(f"Saved: {path_p}")

    # ------------------------
    # Experiment 2
    # ------------------------
    n_grid = np.unique(np.round(np.logspace(np.log10(args.n_min), np.log10(args.n_max), args.n_steps)).astype(int))
    df_n = run_mc_grid_over_n(
        p_true=args.p_true_fixed,
        p0=args.p0,
        n_grid=n_grid,
        n_mc=args.n_mc,
        alpha=args.alpha,
        e_params=e_params,
        rng=rng,
    )
    path_n = os.path.join(outdir, "coverage_vary_n.csv")
    df_n.to_csv(path_n, index=False)
    print(f"Saved: {path_n}")


if __name__ == "__main__":
    main()