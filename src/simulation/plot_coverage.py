# src/simulation/plot_coverage.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot coverage simulation results (CSV -> PNG).")
    p.add_argument("--indir", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--zoom-width", type=float, default=0.02)
    return p.parse_args()


def plot_rejection_vs_p(df: pd.DataFrame, out_png: str, zoom: bool, zoom_width: float):
    p0 = float(df["p0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    n = int(df["n"].iloc[0])

    if zoom:
        lo = max(0.0, p0 - zoom_width)
        hi = min(1.0, p0 + zoom_width)
        df = df[(df["p_true"] >= lo) & (df["p_true"] <= hi)].copy()

    plt.figure()
    plt.plot(df["p_true"], df["rej_rate_e"], label="e-value (Beta-mixture)")
    plt.plot(df["p_true"], df["rej_rate_pmf"], label="Exact p-value (PMF)")
    plt.plot(df["p_true"], df["rej_rate_cp"], label="Clopper–Pearson")
    plt.plot(df["p_true"], df["rej_rate_jeffreys"], label="Jeffreys CI inversion")

    plt.axvline(p0, linestyle="--", label="p0")
    plt.axhline(alpha, linestyle=":", label="alpha")

    plt.xlabel("True default probability p")
    plt.ylabel("Rejection rate")
    title = f"Rejection rate vs true p (n={n})"
    if zoom:
        title += f" — zoom around p0 (±{zoom_width})"
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_rejection_vs_n(df: pd.DataFrame, out_png: str):
    p_true = float(df["p_true"].iloc[0])
    p0 = float(df["p0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])

    plt.figure()
    plt.plot(df["n"], df["rej_rate_e"], marker="o", label="e-value (Beta-mixture)")
    plt.plot(df["n"], df["rej_rate_pmf"], marker="o", label="Exact p-value (PMF)")
    plt.plot(df["n"], df["rej_rate_cp"], marker="o", label="Clopper–Pearson")
    plt.plot(df["n"], df["rej_rate_jeffreys"], marker="o", label="Jeffreys CI inversion")

    plt.axhline(alpha, linestyle=":", label="alpha")

    plt.xscale("log")
    plt.xlabel("Sample size n (log scale)")
    plt.ylabel("Rejection rate")
    plt.title(f"Rejection rate vs n (true p={p_true}, p0={p0})")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


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

    plot_rejection_vs_p(df_p, os.path.join(args.outdir, "coverage_vary_p.png"), zoom=False, zoom_width=args.zoom_width)
    plot_rejection_vs_p(df_p, os.path.join(args.outdir, "coverage_vary_p_zoom.png"), zoom=True, zoom_width=args.zoom_width)
    plot_rejection_vs_n(df_n, os.path.join(args.outdir, "coverage_vary_n.png"))

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()