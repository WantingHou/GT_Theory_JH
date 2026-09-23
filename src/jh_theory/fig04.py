"""Build Figure 4: local freeze-thaw regulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .common import BLUE, ORANGE, PURPLE, TEAL, clean_axis, finite, load_data, panel, rc_context, save_pdf
from .theory import normalized


def build(output: Path) -> Path:
    with load_data() as d:
        depth = np.asarray(d["f4_depth_m"])
        labels = ("Pre-freeze", "Early freezing", "Strong freezing", "Thaw-side probe")
        colors = ("#666666", BLUE, PURPLE, TEAL)
        styles = ("-", "--", "-.", ":")
        cryo = [np.asarray(d[f"f4_{name}_cryosuction_strength_Pa"]) / 1e6 for name in ("pre_freeze", "early_freezing", "strong_freezing", "thaw_side")]
        mobility = [np.asarray(d[f"f4_{name}_liquid_mobility_m2_Pa_s"]) for name in ("pre_freeze", "early_freezing", "strong_freezing", "thaw_side")]
        delta = np.asarray(d["f45_b_deltaT_K"])
        alpha = np.asarray(d["f45_b_alpha_Pa_inv"])
        chi = np.asarray(d["f45_b_chi"])
        mr = np.asarray(d["f45_b_mobility_ratio"])
        rr = np.asarray(d["f45_b_R_over_Rmax"])
        valid = np.asarray(d["f45_b_valid"], dtype=bool)
    finite(depth, delta, alpha, *cryo, *mobility)
    row = int(np.argmin(np.abs(alpha - 2.0e-4)))

    with rc_context():
        fig = plt.figure(figsize=(7.35, 5.1), constrained_layout=True)
        outer = fig.add_gridspec(2, 2)
        top = outer[0, :].subgridspec(1, 2, wspace=0.25)
        ax1 = fig.add_subplot(top[0, 0])
        ax2 = fig.add_subplot(top[0, 1], sharey=ax1)
        for p, m, label, color, style in zip(cryo, mobility, labels, colors, styles, strict=True):
            ax1.plot(p, depth, color=color, linestyle=style, label=label)
            ax2.plot(np.clip(np.abs(m / np.nanmax(np.abs(m))), 0, 1), depth, color=color, linestyle=style)
        ax1.set_ylim(3, 0)
        ax1.set_xlabel("$p_{cryo}^{+}$ (MPa)")
        ax1.set_ylabel("$z$ (m)")
        ax1.legend(frameon=False, fontsize=5.8, loc="center right")
        clean_axis(ax1)
        panel(ax1, "a")
        ax2.set_ylim(3, 0)
        ax2.set_xlabel("$M_w/M_0$")
        ax2.tick_params(labelleft=False)
        clean_axis(ax2)

        ax = fig.add_subplot(outer[1, 0])
        x = np.clip(chi[row], 1e-6, None)
        ax.semilogx(x, np.clip(mr[row], 0, 1.05), color=ORANGE, linestyle="--", label="$M_w/M_0$")
        ax.semilogx(x, np.clip(rr[row], 0, 1.05), color=BLUE, label="$R/R_{max}$")
        ax.axvline(1, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_xlim(1e-3, 1e3)
        ax.set_ylim(0, 1.06)
        ax.set_xlabel(r"$\chi$")
        ax.set_ylabel("Normalized response")
        ax.legend(frameon=False)
        clean_axis(ax)
        panel(ax, "b")

        ax = fig.add_subplot(outer[1, 1])
        z = np.ma.masked_where(~valid, np.clip(rr, 1e-4, 1.0))
        mesh = ax.pcolormesh(delta, alpha * 1e4, z, shading="auto", cmap="viridis", norm=LogNorm(vmin=1e-4, vmax=1))
        ax.axhline(2.0, color="white", linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Freezing intensity $\Delta T_f$ (K)")
        ax.set_ylabel(r"$\alpha_{vG}$ ($10^{-4}$ Pa$^{-1}$)")
        fig.colorbar(mesh, ax=ax, label="$R/R_{max}$", pad=0.02)
        clean_axis(ax, grid=None)
        panel(ax, "c")

    path = Path(output) / "Fig04.pdf"
    save_pdf(fig, path, "Figure 4 freeze-thaw regulation")
    return path
