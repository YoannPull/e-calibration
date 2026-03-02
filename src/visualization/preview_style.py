# src/visualization/preview_style.py
from __future__ import annotations

import os
from pathlib import Path
import argparse

import numpy as np

from utils.plot_style import (
    apply_plot_style,
    new_figure,
    finalize_ax,
    save_figure,
    annotate_min,
    annotate_min_below_ylim_at_crossing,
    get_method_style,
)


def parse_args():
    p = argparse.ArgumentParser(description="Preview the global Matplotlib style (journal-grade).")
    p.add_argument("--mode", type=str, default="paper", choices=["paper", "draft"])
    p.add_argument("--outdir", type=str, default=os.path.join("outputs", "style_preview"))
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    apply_plot_style(mode=args.mode)

    # --------------------------------------------------
    # 1) Coverage vs p_true (single method + min annotation)
    # --------------------------------------------------
    p = np.linspace(0.001, 0.10, 220)
    conf = 0.95

    y = conf - 0.06 * np.exp(-((p - 0.03) / 0.012) ** 2) + 0.004 * np.sin(60 * p)
    y = np.clip(y, 0.70, 1.02)

    fig, ax = new_figure()
    st = get_method_style("jeffreys")
    ax.plot(p, y, color=st["color"], zorder=3)

    finalize_ax(
        ax,
        xlabel="True default probability (p)",
        ylabel="Coverage probability",
        title=None,  # journal style: no title
        reference_line=conf,
        reference_label=f"Nominal level ({conf:.0%})",
        xlim=(0.0, 0.10),
        ylim=(0.80, 1.02),
        add_legend=False,
    )

    idx_min = int(np.argmin(y))
    annotate_min(
        ax,
        float(p[idx_min]),
        float(y[idx_min]),
        f"Minimum coverage: {float(y[idx_min]):.3f} (p={float(p[idx_min]):.3f})",
    )

    save_figure(fig, outdir / "preview_single_method.png", also_pdf=True)

    # --------------------------------------------------
    # 2) Coverage vs p_true (multi-method overlay)
    # --------------------------------------------------
    fig, ax = new_figure()
    keys = ["e", "pmf", "cp", "jeffreys"]

    for i, key in enumerate(keys):
        st = get_method_style(key, idx=i)
        # Slightly different shapes so the overlay is meaningful
        yy = conf - (0.028 + 0.008 * (key == "e")) * np.exp(-((p - 0.03) / 0.014) ** 2)
        yy = yy + 0.003 * np.sin((40 + 10 * (key == "pmf")) * p)
        yy = np.clip(yy, 0.78, 1.02)

        ax.plot(p, yy, color=st["color"], label=st["label"], zorder=3)

    finalize_ax(
        ax,
        xlabel="True default probability (p)",
        ylabel="Coverage probability",
        title=None,
        reference_line=conf,
        reference_label=f"Nominal level ({conf:.0%})",
        xlim=(0.0, 0.10),
        ylim=(0.80, 1.02),
        add_legend=True,
        legend_loc="lower right",
    )

    save_figure(fig, outdir / "preview_overlay.png", also_pdf=True)

    # --------------------------------------------------
    # 3) Minimum below y-lim (crossing annotation)
    # --------------------------------------------------
    p2 = np.linspace(0.001, 0.03, 260)
    y2 = conf - 0.30 * np.exp(-((p2 - 0.012) / 0.0045) ** 2)  # dips below 0.8
    y2 = np.clip(y2, 0.0, 1.02)

    fig, ax = new_figure()
    st = get_method_style("cp")
    ax.plot(p2, y2, color=st["color"], zorder=3)

    xlim = (0.000, 0.03)
    ylim = (0.80, 1.02)

    finalize_ax(
        ax,
        xlabel="True default probability (p)",
        ylabel="Coverage probability",
        title=None,
        reference_line=conf,
        reference_label=f"Nominal level ({conf:.0%})",
        xlim=xlim,
        ylim=ylim,
        add_legend=False,
    )

    mask = (p2 >= xlim[0]) & (p2 <= xlim[1])
    p_m = p2[mask]
    y_m = y2[mask]
    idx_min = int(np.argmin(y_m))

    if float(y_m[idx_min]) < float(ylim[0]):
        annotate_min_below_ylim_at_crossing(
            ax,
            x=p_m,
            y=y_m,
            idx_min=idx_min,
            ylim_low=float(ylim[0]),
            text_color="red",
        )
    else:
        annotate_min(ax, float(p_m[idx_min]), float(y_m[idx_min]), f"Minimum: {float(y_m[idx_min]):.3f}")

    save_figure(fig, outdir / "preview_below_ylim.png", also_pdf=True)

    # --------------------------------------------------
    # 4) Non-coverage example (generic): statistic vs n (log-x)
    # --------------------------------------------------
    n = np.unique(np.round(np.logspace(2, 5, 28)).astype(int))
    stat = 0.2 + 1.2 / np.sqrt(n / 100.0)  # just a decreasing curve

    fig, ax = new_figure()
    st = get_method_style("generic_stat", idx=0, label="Example statistic")
    ax.plot(n, stat, marker="o", color=st["color"], label=st["label"], zorder=3)
    ax.set_xscale("log")

    finalize_ax(
        ax,
        xlabel="Sample size n (log scale)",
        ylabel="Statistic value",
        title=None,
        reference_line=None,  # no reference line here
        xlim=(n.min(), n.max()),
        ylim=None,
        add_legend=True,
        legend_loc="upper right",
    )

    save_figure(fig, outdir / "preview_generic_logx.png", also_pdf=True)

    print(f"[OK] Style preview written to: {outdir}")


if __name__ == "__main__":
    main()