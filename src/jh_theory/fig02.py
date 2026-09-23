"""Build Figure 2: static thermal and hydrological reference."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import BLUE, GOLD, INK, ORANGE, PURPLE, TEAL, clean_axis, configure_depth_axis, depth_coordinate, finite, load_data, panel, rc_context, save_pdf


def build(output: Path) -> Path:
    with load_data() as d:
        z = np.asarray(d["f2_cell_depth_m"])
        y = depth_coordinate(z)
        zf = np.asarray(d["f2_face_depth_m"])
        yf = depth_coordinate(zf)
        temp_c = np.asarray(d["f2_cell_temperature_K"]) - 273.15
        sw = np.asarray(d["f2_cell_liquid_saturation"])
        sg = np.asarray(d["f2_cell_gas_saturation"])
        porosity = np.asarray(d["f2_cell_porosity"])
        intrinsic = np.asarray(d["f2_cell_intrinsic_permeability_m2"])
        effective = np.asarray(d["f2_cell_effective_liquid_permeability_m2"])
        diffusivity = np.asarray(d["f2_cell_vapour_effective_diffusivity_m2_s"]) * 1e6
        flux = np.asarray(d["f2_face_conductive_flux_W_m2"]) * 1e3
        heating = np.asarray(d["f2_cell_radiogenic_heat_W_m3"]) * 1e6
    finite(z, temp_c, sw, sg, porosity, intrinsic, effective, diffusivity, zf, flux, heating)

    with rc_context():
        fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.2), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(temp_c, y, color=INK, linewidth=1.35)
        configure_depth_axis(ax)
        ax.set_xlabel("Temperature (°C)")
        ax.axhline(depth_coordinate(15_000), color=GOLD, linestyle=":", linewidth=0.8)
        ax.axhline(depth_coordinate(35_000), color=PURPLE, linestyle=":", linewidth=0.8)
        ax.axhline(depth_coordinate(100_000), color=TEAL, linestyle=":", linewidth=0.8)
        clean_axis(ax)
        panel(ax, "a")

        ax = axes[0, 1]
        ax.plot(sw, y, color=BLUE, label="$S_w$")
        ax.plot(sg, y, color=TEAL, linestyle="--", label="$S_g$")
        ax.plot(porosity, y, color=PURPLE, linestyle=":", label=r"$\phi$")
        configure_depth_axis(ax, label=False)
        ax.set_xlabel("Saturation / porosity")
        ax.set_xlim(0, 1.05)
        ax.legend(frameon=False, loc="center right")
        clean_axis(ax)
        panel(ax, "b")

        ax = axes[1, 0]
        ax.semilogx(effective, y, color=BLUE, label="$k_{eff}$")
        ax.semilogx(intrinsic, y, color=INK, linestyle="--", label="$K_{zz}$")
        configure_depth_axis(ax)
        ax.set_xlabel("Permeability (m²)")
        ax.legend(frameon=False, loc="center left")
        clean_axis(ax)
        twin = ax.twiny()
        twin.plot(diffusivity, y, color=TEAL, linestyle="-.")
        twin.set_xlabel("$D_{v,eff}$ (mm²/s)", color=TEAL)
        twin.tick_params(axis="x", colors=TEAL, labelsize=6.2)
        panel(ax, "c")

        ax = axes[1, 1]
        ax.plot(flux, yf, color=INK)
        configure_depth_axis(ax, label=False)
        ax.set_xlabel("$q_{cond}$ (mW/m²)")
        clean_axis(ax)
        twin = ax.twiny()
        twin.plot(heating, y, color=GOLD, linestyle="--")
        twin.set_xlabel("$A(z)$ (µW/m³)", color=GOLD)
        twin.tick_params(axis="x", colors=GOLD, labelsize=6.2)
        panel(ax, "d")

    path = Path(output) / "Fig02.pdf"
    save_pdf(fig, path, "Figure 2 static reference")
    return path
