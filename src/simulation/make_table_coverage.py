# src/simulation/make_table_coverage.py

import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build summary tables from pointwise coverage simulation CSVs.")
    p.add_argument("--indir", type=str, required=True,
                   help="Folder containing coverage_vary_p.csv and coverage_vary_n.csv")
    p.add_argument("--outdir", type=str, required=True,
                   help="Folder where summary tables will be written")
    p.add_argument(
        "--points",
        type=str,
        default="0.001,0.005,0.01,0.015,0.02,0.05,0.10,0.20",
        help="Comma-separated list of p_true values where we want a coverage snapshot (nearest grid point used).",
    )
    return p.parse_args()


def _nearest_rows(df, col, targets):
    rows = []
    xs = df[col].to_numpy()
    for t in targets:
        idx = int(np.argmin(np.abs(xs - t)))
        row = df.iloc[idx].copy()
        row["_target"] = float(t)
        row["_picked"] = float(xs[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    path_p = os.path.join(args.indir, "coverage_vary_p.csv")
    path_n = os.path.join(args.indir, "coverage_vary_n.csv")

    if not os.path.isfile(path_p):
        raise FileNotFoundError(f"Missing: {path_p}")
    if not os.path.isfile(path_n):
        raise FileNotFoundError(f"Missing: {path_n}")

    df_p = pd.read_csv(path_p)
    df_n = pd.read_csv(path_n)

    # -------------------------
    # (A) Full table for vary-n
    # -------------------------
    vary_n_tbl = df_n[[
        "n", "p_true", "alpha", "n_mc",
        "cov_rate_e", "cov_rate_pmf", "cov_rate_cp", "cov_rate_jeffreys",
        "rej_rate_e", "rej_rate_pmf", "rej_rate_cp", "rej_rate_jeffreys",
    ]].sort_values("n")

    out_vary_n = os.path.join(args.outdir, "coverage_summary_vary_n.csv")
    vary_n_tbl.to_csv(out_vary_n, index=False)

    # -------------------------
    # (B) Coverage snapshots for vary-p
    # -------------------------
    targets = [float(x.strip()) for x in args.points.split(",") if x.strip() != ""]
    picked = _nearest_rows(df_p.sort_values("p_true"), col="p_true", targets=targets)

    snap_tbl = picked[[
        "_target", "_picked",
        "n", "alpha", "n_mc",
        "cov_rate_e", "cov_rate_pmf", "cov_rate_cp", "cov_rate_jeffreys",
        "rej_rate_e", "rej_rate_pmf", "rej_rate_cp", "rej_rate_jeffreys",
    ]].rename(columns={"_target": "target_p_true", "_picked": "grid_p_true"})

    out_snap = os.path.join(args.outdir, "coverage_snapshot_vary_p.csv")
    snap_tbl.to_csv(out_snap, index=False)

    print("Saved tables:")
    print(f" - {out_vary_n}")
    print(f" - {out_snap}")


if __name__ == "__main__":
    main()