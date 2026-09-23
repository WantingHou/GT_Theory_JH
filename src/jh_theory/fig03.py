"""Build Figure 3: shallow transient vapour filtering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

from .common import BLUE, ORANGE, SURFACE, TEAL, clean_axis, finite, load_data, panel, rc_context, save_pdf

FACTOR = 1_000.0 * 86_400.0


def build(output: Path) -> Path:
    with load_data() as d:
        time_h = np.asarray(d["f3_time_s"]) / 3600.0
        depth = np.asarray(d["f3_face_depth_m"])
        vapour = np.asarray(d["f3_face_vapour_diffusive_kg_m2_s"]) * FACTOR
        liquid = np.asarray(d["f3_face_liquid_h2o_kg_m2_s"]) * FACTOR
        thermal = np.asarray(d["f3_face_vapour_thermal_kg_m2_s"]) * FACTOR
        moisture = np.asarray(d["f3_face_vapour_moisture_kg_m2_s"]) * FACTOR
    finite(time_h, depth, vapour, liquid, thermal, moisture)
    shallow = depth <= 2.5
    snap = int(np.argmin(np.abs(time_h - 30.0)))

    with rc_context():
        fig = plt.figure(figsize=(7.35, 4.9), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1.0))
        axa = fig.add_subplot(grid[0, :])
        cmap = LinearSegmentedColormap.from_list("signed_vapour", (BLUE, SURFACE, ORANGE), N=257)
        mesh = axa.pcolormesh(time_h, depth[shallow], vapour[:, shallow].T, shading="auto", cmap=cmap, norm=SymLogNorm(linthresh=1e-2, vmin=-10, vmax=10))
        axa.set_xlim(0, 48)
        axa.set_ylim(2.5, 0)
        axa.set_xlabel("Elapsed time (h)")
        axa.set_ylabel("Depth below surface (m)")
        clean_axis(axa, grid=None)
        panel(axa, "a")
        cbar = fig.colorbar(mesh, ax=axa, pad=0.015)
        cbar.set_label("$J_v^{diff}$ (g/(m² day))")

        axb = fig.add_subplot(grid[1, 0])
        for target, color, marker in ((0.1, BLUE, "o"), (0.5, ORANGE, "s"), (1.0, TEAL, "^")):
            i = int(np.argmin(np.abs(depth - target)))
            axb.plot(time_h, vapour[:, i], color=color, marker=marker, markevery=max(1, len(time_h)//12), markersize=2.6, markerfacecolor=SURFACE, label=f"{depth[i]:.2f} m")
        axb.set_yscale("symlog", linthresh=1e-2)
        axb.axhline(0, color="#777777", linewidth=0.6)
        axb.axvline(30, color="#777777", linestyle=":", linewidth=0.7)
        axb.set_xlim(0, 48)
        axb.set_xlabel("Elapsed time (h)")
        axb.set_ylabel("Signed $J_v^{diff}$ (g/(m² day))")
        axb.legend(frameon=False)
        clean_axis(axb, grid="y")
        panel(axb, "b")

        axc = fig.add_subplot(grid[1, 1])
        for values, color, style, label in ((liquid, BLUE, "-", "Liquid H₂O"), (thermal, ORANGE, "--", "Thermal vapour"), (moisture, TEAL, "-.", "Moisture-gradient vapour")):
            axc.plot(values[snap, shallow], depth[shallow], color=color, linestyle=style, label=label)
        axc.set_xscale("symlog", linthresh=1e-2)
        axc.set_ylim(2.5, 0)
        axc.axvline(0, color="#777777", linewidth=0.6)
        axc.set_xlabel("Signed H₂O flux (g/(m² day))")
        axc.set_ylabel("Depth below surface (m)")
        axc.legend(frameon=False, loc="lower right")
        clean_axis(axc, grid="y")
        panel(axc, "c")

    path = Path(output) / "Fig03.pdf"
    save_pdf(fig, path, "Figure 3 shallow transient")
    return path
