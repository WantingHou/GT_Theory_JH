"""Shared data, depth coordinates, and figure styling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

INK = "#171717"
MUTED = "#5c5c5c"
GRID = "#d8d8d8"
BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#6F4C9B"
GOLD = "#B8860B"
SURFACE = "#ffffff"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_data() -> np.lib.npyio.NpzFile:
    return np.load(repository_root() / "data/jh_figure_inputs.npz", allow_pickle=False)


def style() -> dict[str, object]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
    }


def clean_axis(ax: plt.Axes, *, grid: str | None = "both") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=2.5, colors=INK)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.45, alpha=0.8, zorder=0)


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=8.0)


def depth_coordinate(depth_m: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.log10(1.0 + np.asarray(depth_m, dtype=float))


def depth_ticks(max_depth_m: float = 100_000.0) -> tuple[np.ndarray, list[str]]:
    values = np.array([0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 35_000.0, 50_000.0, 75_000.0, 100_000.0])
    values = values[values <= max_depth_m]
    labels = []
    for value in values:
        if value == 0:
            labels.append("0")
        elif value < 1_000:
            labels.append(f"{value:g} m")
        else:
            labels.append(f"{value / 1000:g} km")
    return depth_coordinate(values), labels


def configure_depth_axis(ax: plt.Axes, max_depth_m: float = 100_000.0, *, label: bool = True) -> None:
    ticks, labels = depth_ticks(max_depth_m)
    ax.set_ylim(depth_coordinate(max_depth_m), depth_coordinate(0.0))
    ax.set_yticks(ticks, labels)
    if label:
        ax.set_ylabel("Depth below surface")


def save_pdf(fig: plt.Figure, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        metadata={"Title": title, "Author": "WantingHou", "Creator": "GT Theory", "Subject": "JH theory figure"},
    )
    plt.close(fig)


def finite(*arrays: np.ndarray) -> None:
    for array in arrays:
        if not np.all(np.isfinite(np.asarray(array, dtype=float))):
            raise ValueError("non-finite figure data")


def rc_context():
    return mpl.rc_context(style())
