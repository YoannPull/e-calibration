# src/simulation/make_table_coverage.py

import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build summary tables from coverage simulation CSVs.")
    p.add_argument("--indir", type=str, required=True, help="Folder containing coverage_vary_p.csv and coverage_vary_n.csv")
    p.add_argument("--outdir", type=str, required=True, help="Folder where summary tables will be written")
    p.add_argument(
        "--power-points",
        type=str,
        default="0.005,0.01,0.0125,0.015,0.02,0.05,0.10,0.20",
        help="Comma-separated list of true p values where we want a power snapshot (nearest grid point used).",
    )
    return p.parse_args()


def _nearest_rows(df, col, targets):
    """
    For each target value, pick the row in df minimizing |df[col] - target|.
    """
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
    # Type I summary (use vary-n)
    # -------------------------
    # If p_true == p0 in df_n, that rejection rate estimates type-I error.
    # Otherwise, we still produce the table but it should be interpreted as power.
    type1 = df_n.copy()
    type1["is_null"] = np.isclose(type1["p_true"], type1["p0"])

    type1_tbl = type1[[
        "n", "p_true", "p0", "alpha", "n_mc", "is_null",
        "rej_rate_e", "rej_rate_pmf", "rej_rate_cp", "rej_rate_jeffreys"
    ]].sort_values("n")

    out_type1 = os.path.join(args.outdir, "type1_summary.csv")
    type1_tbl.to_csv(out_type1, index=False)

    # -------------------------
    # Power snapshots (use vary-p)
    # -------------------------
    targets = [float(x.strip()) for x in args.power_points.split(",") if x.strip() != ""]
    picked = _nearest_rows(df_p.sort_values("p_true"), col="p_true", targets=targets)

    power_tbl = picked[[
        "_target", "_picked",
        "n", "p0", "alpha", "n_mc",
        "rej_rate_e", "rej_rate_pmf", "rej_rate_cp", "rej_rate_jeffreys"
    ]].rename(columns={"_target": "target_p", "_picked": "grid_p"})

    out_power = os.path.join(args.outdir, "power_summary.csv")
    power_tbl.to_csv(out_power, index=False)

    print("Saved tables:")
    print(f" - {out_type1}")
    print(f" - {out_power}")


if __name__ == "__main__":
    main()