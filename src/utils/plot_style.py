# src/utils/plot_style.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Colors / styles
# ============================================================

# A stable, publication-friendly palette (close to matplotlib defaults, readable in print)
BASE_COLORS: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Optional semantic styles for known keys (you can extend this across the project)
METHOD_STYLES: dict[str, dict[str, Any]] = {
    # Coverage / calibration
    "e": {"label": "e-value (Beta-mixture)", "color": "#1f77b4"},
    "pmf": {"label": "Exact p-value (PMF)", "color": "#ff7f0e"},
    "cp": {"label": "Clopper–Pearson", "color": "#2ca02c"},
    "jeffreys": {"label": "Jeffreys CI inversion", "color": "#d62728"},
}


def get_method_style(key: str, idx: int | None = None, *, label: str | None = None) -> dict[str, Any]:
    """
    Return a style dict with at least: {"label": ..., "color": ...}

    - If key is known in METHOD_STYLES, returns it (with optional label override).
    - Otherwise returns a stable fallback color from BASE_COLORS.
    """
    if key in METHOD_STYLES:
        st = dict(METHOD_STYLES[key])
        if label is not None:
            st["label"] = label
        return st

    if idx is None:
        idx = 0
    color = BASE_COLORS[int(idx) % len(BASE_COLORS)]
    return {"label": label if label is not None else str(key), "color": color}


# ============================================================
# Global style
# ============================================================

def apply_plot_style(mode: str = "paper") -> None:
    """
    Apply a consistent Matplotlib style across the whole project.

    mode:
      - "paper": compact, journal-friendly
      - "draft": larger fonts for quick review
    """
    if mode not in {"paper", "draft"}:
        raise ValueError("mode must be in {'paper', 'draft'}")

    if mode == "paper":
        fontsize = 10.5
        labelsize = 10.5
        ticksize = 9.5
        legendsize = 9.5
        figsize = (6.2, 4.0)   # ~single-column friendly; adjust if you target 2-col layouts
        save_dpi = 300         # journal-grade raster; PDFs are vector anyway
        linewidth = 1.9
        markersize = 4.8
        grid_alpha = 0.18
    else:
        fontsize = 13
        labelsize = 13
        ticksize = 12
        legendsize = 12
        figsize = (7.2, 4.8)
        save_dpi = 220
        linewidth = 2.2
        markersize = 5.5
        grid_alpha = 0.22

    plt.rcParams.update({
        # Typography
        "font.size": fontsize,
        "axes.labelsize": labelsize,
        "xtick.labelsize": ticksize,
        "ytick.labelsize": ticksize,
        "legend.fontsize": legendsize,

        # Figure
        "figure.figsize": figsize,
        "figure.dpi": 120,
        "savefig.dpi": save_dpi,
        "savefig.bbox": "tight",

        # Axes appearance
        "axes.grid": True,
        "grid.alpha": grid_alpha,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10,

        # Lines / markers
        "lines.linewidth": linewidth,
        "lines.solid_capstyle": "round",
        "lines.markersize": markersize,

        # Legend
        "legend.frameon": False,
    })


# ============================================================
# Figure helpers
# ============================================================

def new_figure(figsize: tuple[float, float] | None = None):
    """
    Create (fig, ax) with consistent defaults.
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig, out_path: Path | str, *, also_pdf: bool = True) -> None:
    """
    Save the figure to PNG (and optionally PDF). Closes the figure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path)
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))

    plt.close(fig)


def finalize_ax(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    reference_line: float | None = None,
    reference_label: str | None = None,
    reference_style: dict[str, Any] | None = None,
    add_legend: bool = True,
    legend_loc: str = "best",
) -> None:
    """
    Standard axis finalization:
      - labels, limits, optional title (often None for journals)
      - optional reference horizontal line (nominal level, alpha, threshold, etc.)
      - legend control
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if reference_line is not None:
        st = {
            "linestyle": ":",
            "linewidth": 1.35,
            "color": "black",
            "alpha": 0.85,
            "zorder": 2,
        }
        if reference_style is not None:
            st.update(reference_style)

        ax.axhline(
            reference_line,
            label=reference_label if reference_label is not None else None,
            **st,
        )

    # Grid behind data
    ax.set_axisbelow(True)

    if add_legend:
        ax.legend(loc=legend_loc)


# ============================================================
# Annotation helpers (clean, journal-friendly)
# ============================================================

def annotate_min(
    ax,
    x_min: float,
    y_min: float,
    text: str,
    *,
    text_color: str = "black",
) -> None:
    """
    Annotate a minimum point with a small arrow, avoiding clutter.
    """
    ax.scatter([x_min], [y_min], s=18, zorder=4, color=text_color)

    # offset relative to plot window
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = 0.02 * (x1 - x0)
    dy = 0.06 * (y1 - y0)

    ax.annotate(
        text,
        xy=(x_min, y_min),
        xytext=(x_min + dx, y_min + dy),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=max(9, float(plt.rcParams["font.size"]) - 1),
        color=text_color,
        arrowprops=dict(arrowstyle="->", lw=1.0, color=text_color, alpha=0.9),
        zorder=5,
    )


def annotate_min_below_ylim_at_crossing(
    ax,
    *,
    x: np.ndarray,
    y: np.ndarray,
    idx_min: int,
    ylim_low: float,
    text_color: str = "red",
) -> None:
    """
    If the minimum is below the plotting ylim, annotate a point where the curve
    crosses back above ylim_low, and indicate min is below axis.
    """
    n = len(y)
    j = None

    # Search to the right
    for k in range(idx_min, n):
        if y[k] >= ylim_low:
            j = k
            break
    # If not found, search to the left
    if j is None:
        for k in range(idx_min, -1, -1):
            if y[k] >= ylim_low:
                j = k
                break
    if j is None:
        j = idx_min

    xj = float(x[j])

    ax.scatter([xj], [ylim_low], s=18, zorder=4, color=text_color)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = 0.02 * (x1 - x0)
    dy = 0.08 * (y1 - y0)

    ax.annotate(
        "Minimum below axis",
        xy=(xj, ylim_low),
        xytext=(xj + dx, ylim_low + dy),
        ha="left",
        va="bottom",
        fontsize=max(9, float(plt.rcParams["font.size"]) - 1),
        color=text_color,
        arrowprops=dict(arrowstyle="->", lw=1.0, color=text_color, alpha=0.9),
        zorder=5,
    )