"""Build Figure 7: dimensionless regime coordinates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .common import BLUE, ORANGE, PURPLE, TEAL, clean_axis, finite, load_data, panel, rc_context, save_pdf

ENV_COLORS = {
    "cold/permafrost column": BLUE,
    "arid basin–stable craton": ORANGE,
    "geothermal basin": TEAL,
    "hydrated active orogen/shear zone": PURPLE,
}
ENV_LABELS = {
    "cold/permafrost column": "Cold / permafrost",
    "arid basin–stable craton": "Arid / craton",
    "geothermal basin": "Geothermal",
    "hydrated active orogen/shear zone": "Hydrated orogen",
}


def _track_arrows(ax, x, y, env):
    for name in np.unique(env):
        mask = env == name
        xx, yy = x[mask], y[mask]
        good = np.isfinite(xx) & np.isfinite(yy)
        xx, yy = xx[good], yy[good]
        if len(xx) < 2:
            continue
        order = np.argsort(xx)
        ax.plot(xx[order], yy[order], color=ENV_COLORS.get(str(name), "#555555"), linewidth=0.8, alpha=0.85)
        ax.annotate("", xy=(xx[order][-1], yy[order][-1]), xytext=(xx[order][-2], yy[order][-2]), arrowprops={"arrowstyle": "->", "color": ENV_COLORS.get(str(name), "#555555"), "lw": 0.8})


def build(output: Path) -> Path:
    with load_data() as d:
        arrays = {key: np.asarray(d[key]) for key in d.files if key.startswith("f7_")}

    with rc_context():
        fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.15), constrained_layout=True)

        ax = axes[0]
        left, right = arrays["f7_bg_a_x_left"], arrays["f7_bg_a_x_right"]
        bottom, top = arrays["f7_bg_a_y_bottom"], arrays["f7_bg_a_y_top"]
        colors = ("#dceaf5", "#eaf2e8", "#f7ebdd", "#eee4f3")
        for i, (x0, x1, y0, y1) in enumerate(zip(left, right, bottom, top, strict=True)):
            if np.all(np.isfinite([x0, x1, y0, y1])):
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=colors[i % len(colors)], edgecolor="none", zorder=0))
        rx, ry = arrays["f7_rep_a_x_log10"], arrays["f7_rep_a_y_log10"]
        if len(rx):
            ax.scatter(rx, ry, s=32, facecolor=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
        _track_arrows(ax, arrays["f7_track_a_x_log10"], arrays["f7_track_a_y_log10"], arrays["f7_track_a_environment"])
        ax.axhline(0, color="#777777", linestyle=":", linewidth=0.7)
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.7)
        ax.set_xlim(-4, 0.6)
        ax.set_ylim(-0.25, 1.1)
        ax.set_xlabel(r"$\log_{10}(Pe_T)$")
        ax.set_ylabel(r"$\log_{10}(\mathcal{L})$")
        clean_axis(ax)
        panel(ax, "a")

        ax = axes[1]
        x = arrays["f7_bg_b_x_log10"]
        y = arrays["f7_bg_b_y_log10"]
        value = arrays["f7_bg_b_fill_value"]
        valid = arrays["f7_bg_b_valid"] & np.isfinite(x) & np.isfinite(y) & np.isfinite(value)
        mesh = ax.tricontourf(x[valid], y[valid], value[valid], levels=np.linspace(0, 1, 21), cmap="viridis")
        ax.tricontour(x[valid], y[valid], value[valid], levels=(0.25, 0.5, 0.75), colors="white", linewidths=0.5)
        rx, ry = arrays["f7_rep_b_x_log10"], arrays["f7_rep_b_y_log10"]
        if len(rx):
            ax.scatter(rx, ry, s=24, facecolor=BLUE, edgecolor="white", linewidth=0.7)
        _track_arrows(ax, arrays["f7_track_b_x_log10"], arrays["f7_track_b_y_log10"], arrays["f7_track_b_environment"])
        ax.set_xlabel(r"$\log_{10}(N_{cryo})$")
        ax.set_ylabel(r"$\log_{10}(\mathcal{L})$")
        clean_axis(ax)
        panel(ax, "b")
        fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.15, label="Static/local $R/R_{max}$")

        ax = axes[2]
        left, right = arrays["f7_bg_c_x_left"], arrays["f7_bg_c_x_right"]
        bottom, top = arrays["f7_bg_c_y_bottom"], arrays["f7_bg_c_y_top"]
        fill = arrays["f7_bg_c_fill_value"]
        finite_fill = fill[np.isfinite(fill)]
        lo, hi = (float(np.min(finite_fill)), float(np.max(finite_fill))) if finite_fill.size else (-2.4, -1.8)
        cmap = plt.get_cmap("Greys")
        for x0, x1, y0, y1, value in zip(left, right, bottom, top, fill, strict=True):
            if np.all(np.isfinite([x0, x1, y0, y1, value])):
                color = cmap((value-lo)/(hi-lo) if hi>lo else 0.5)
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=color, edgecolor="white", linewidth=0.3, alpha=0.55))
        rx, ry = arrays["f7_rep_c_x_log10"], arrays["f7_rep_c_y_log10"]
        env = arrays["f7_rep_c_environment"]
        depth = arrays["f7_rep_c_depth_m"]
        for name in np.unique(env):
            mask = env == name
            color = ENV_COLORS.get(str(name), "#555555")
            sizes = np.where(depth[mask] <= 10, 24, np.where(depth[mask] <= 2_500, 38, 54))
            ax.scatter(rx[mask], ry[mask], s=sizes, facecolor="none", edgecolor=color, linewidth=1.0, label=ENV_LABELS.get(str(name), str(name)))
        _track_arrows(ax, arrays["f7_track_c_x_log10"], arrays["f7_track_c_y_log10"], arrays["f7_track_c_environment"])
        ax.set_xlim(-3.2, -1.5)
        ax.set_ylim(-2.35, -0.8)
        ax.set_xlabel(r"$\log_{10}(Da)$")
        ax.set_ylabel(r"$\log_{10}(N_{rx})$")
        ax.legend(frameon=False, fontsize=5.0, loc="best")
        clean_axis(ax)
        panel(ax, "c")

    path = Path(output) / "Fig07.pdf"
    save_pdf(fig, path, "Figure 7 dimensionless regime coordinates")
    return path
