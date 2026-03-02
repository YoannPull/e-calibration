# src/visualization/plot_coverage.py

from __future__ import annotations

import os
import argparse
import pandas as pd

from utils.plot_style import (
    apply_plot_style,
    new_figure,
    finalize_ax,
    save_figure,
    get_method_style,
)


def parse_args():
    p = argparse.ArgumentParser(description="Plot pointwise coverage simulation results (CSV -> PNG/PDF).")
    p.add_argument("--indir", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)

    p.add_argument(
        "--zoom-width",
        type=float,
        default=0.02,
        help="Zoom half-width around p-zoom-center for vary-p zoom plots.",
    )
    p.add_argument(
        "--p-zoom-center",
        type=float,
        default=None,
        help="Center of zoom for vary-p. If None, uses median p_true.",
    )

    p.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "draft"],
        help="Plot style mode.",
    )
    p.add_argument(
        "--also-pdf",
        action="store_true",
        help="If set, also write PDF files next to PNG.",
    )

    return p.parse_args()


METHODS = [
    ("e", "cov_rate_e"),
    ("pmf", "cov_rate_pmf"),
    ("cp", "cov_rate_cp"),
    ("jeffreys", "cov_rate_jeffreys"),
]


def _resolve_zoom_center(df: pd.DataFrame, requested: float | None) -> float:
    """
    Choose zoom center:
      - if requested is None: use median p_true
      - else: use nearest available grid point (so zoom isn't empty due to floating rounding)
    """
    df = df.sort_values("p_true")
    if requested is None:
        return float(df["p_true"].median())

    xs = df["p_true"].to_numpy()
    idx = int((abs(xs - requested)).argmin())
    return float(xs[idx])


def plot_coverage_vs_p_one_method(
    df: pd.DataFrame,
    out_path: str,
    method_key: str,
    col: str,
    zoom: bool,
    zoom_width: float,
    zoom_center: float | None,
    also_pdf: bool,
):
    df = df.sort_values("p_true")
    alpha = float(df["alpha"].iloc[0])
    n = int(df["n"].iloc[0])

    if zoom:
        center = _resolve_zoom_center(df, zoom_center)
        lo = max(0.0, center - zoom_width)
        hi = min(1.0, center + zoom_width)
        df = df[(df["p_true"] >= lo) & (df["p_true"] <= hi)].copy()
        xlim = (lo, hi)
    else:
        center = _resolve_zoom_center(df, zoom_center)
        xlim = (float(df["p_true"].min()), float(df["p_true"].max()))

    st = get_method_style(method_key)
    fig, ax = new_figure()

    ax.plot(
        df["p_true"].to_numpy(),
        df[col].to_numpy(),
        color=st["color"],
        label=st["label"],
        zorder=3,
    )

    finalize_ax(
        ax,
        xlabel="True default probability (p)",
        ylabel="Coverage probability",
        title=None,  # journal style
        xlim=xlim,
        ylim=None,
        reference_line=1.0 - alpha,
        reference_label=f"Nominal level ({1.0 - alpha:.0%})",
        add_legend=False,  # one method per plot -> legend optional
    )

    # If you WANT a tiny caption-like text inside the plot (journal-friendly), uncomment:
    # ax.text(
    #     0.02, 0.02,
    #     f"{st['label']} — n={n}",
    #     transform=ax.transAxes,
    #     ha="left", va="bottom",
    #     fontsize=max(9, float(ax.figure.dpi) * 0 + 9),  # stable
    # )

    save_figure(fig, out_path, also_pdf=also_pdf)


def plot_coverage_vs_n_one_method(
    df: pd.DataFrame,
    out_path: str,
    method_key: str,
    col: str,
    also_pdf: bool,
):
    df = df.sort_values("n")
    alpha = float(df["alpha"].iloc[0])
    p_true = float(df["p_true"].iloc[0])

    st = get_method_style(method_key)
    fig, ax = new_figure()

    ax.plot(
        df["n"].to_numpy(),
        df[col].to_numpy(),
        color=st["color"],
        label=st["label"],
        zorder=3,
    )
    ax.set_xscale("log")

    finalize_ax(
        ax,
        xlabel="Sample size n (log scale)",
        ylabel="Coverage probability",
        title=None,
        xlim=(float(df["n"].min()), float(df["n"].max())),
        ylim=None,
        reference_line=1.0 - alpha,
        reference_label=f"Nominal level ({1.0 - alpha:.0%})",
        add_legend=False,  # one method per plot
    )

    save_figure(fig, out_path, also_pdf=also_pdf)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    apply_plot_style(mode=args.mode)

    path_p = os.path.join(args.indir, "coverage_vary_p.csv")
    path_n = os.path.join(args.indir, "coverage_vary_n.csv")

    if not os.path.isfile(path_p):
        raise FileNotFoundError(f"Missing: {path_p}")
    if not os.path.isfile(path_n):
        raise FileNotFoundError(f"Missing: {path_n}")

    df_p = pd.read_csv(path_p)
    df_n = pd.read_csv(path_n)

    for method_key, col in METHODS:
        # vary-p (full)
        plot_coverage_vs_p_one_method(
            df=df_p,
            out_path=os.path.join(args.outdir, f"coverage_vary_p__{method_key}.png"),
            method_key=method_key,
            col=col,
            zoom=False,
            zoom_width=args.zoom_width,
            zoom_center=args.p_zoom_center,
            also_pdf=args.also_pdf,
        )

        # vary-p (zoom)
        plot_coverage_vs_p_one_method(
            df=df_p,
            out_path=os.path.join(args.outdir, f"coverage_vary_p__{method_key}_zoom.png"),
            method_key=method_key,
            col=col,
            zoom=True,
            zoom_width=args.zoom_width,
            zoom_center=args.p_zoom_center,
            also_pdf=args.also_pdf,
        )

        # vary-n
        plot_coverage_vs_n_one_method(
            df=df_n,
            out_path=os.path.join(args.outdir, f"coverage_vary_n__{method_key}.png"),
            method_key=method_key,
            col=col,
            also_pdf=args.also_pdf,
        )

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()