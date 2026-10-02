"""Shared figure style so every report chart reads as one system."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Categorical slots 1-3 of the reference palette (validated all-pairs for CVD).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
EQUIPMENT_COLORS = {"Dry Van": BLUE, "Flatbed": ORANGE, "Reefer": AQUA}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    }
)


def new(nrows=1, ncols=1, size=(9.0, 4.0), **kwargs):
    return plt.subplots(nrows, ncols, figsize=size, dpi=170, **kwargs)


def end_label(ax, x, y, text, color, dx=4):
    """Direct label at the end of a line, in text ink with a colored marker."""
    ax.plot([x], [y], "o", color=color, ms=4.5, zorder=5)
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK_2)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
