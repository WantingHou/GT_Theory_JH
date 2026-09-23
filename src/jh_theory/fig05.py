"""Build Figure 5: static local deep-water rheology derivative."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .common import BLUE, ORANGE, TEAL, clean_axis, finite, load_data, panel, rc_context, save_pdf
from .theory import rheology_temperature_sensitivity


def build(output: Path) -> Path:
    with load_data() as d:
        depth = np.asarray(d["f5_cell_depth_m"]) / 1000.0
        temperature = np.asarray(d["f5_temperature_K"])
        pressure = np.asarray(d["f5_lithostatic_pressure_Pa"])
        bound = np.asarray(d["f5_initial_Cb"])
        tdeh = np.asarray(d["f5_T_deh_K"])
        solidus = np.asarray(d["f5_wet_solidus_K"])
        omega = np.asarray(d["f5_omega_rheo_mask_u8"], dtype=bool)
        cdepth = np.asarray(d["f45_c_depth_km"])
        cb_grid = np.asarray(d["f45_c_Cb_over_Cbmax"])
        net_grid = np.asarray(d["f45_c_net_K_inv"])
        accepted_cb = np.asarray(d["f45_c_accepted_Cb_over_Cbmax"])
    direct, hardening, net = rheology_temperature_sensitivity(temperature, pressure, bound)
    finite(depth, temperature, pressure, bound, tdeh, solidus, direct, hardening, net, cdepth, cb_grid, net_grid, accepted_cb)

    with rc_context():
        fig = plt.figure(figsize=(7.35, 5.45), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.15))
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(temperature[omega] - 273.15, depth[omega], color=BLUE, label="$T_0$")
        ax.plot(tdeh[omega] - 273.15, depth[omega], color=ORANGE, linestyle="--", label="$T_{deh}$")
        ax.plot(solidus[omega] - 273.15, depth[omega], color="#666666", linestyle="-.", label="Wet solidus")
        ax.set_ylim(101, 33)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Depth (km)")
        ax.legend(frameon=False)
        clean_axis(ax)
        panel(ax, "a")

        ax = fig.add_subplot(grid[0, 1])
        scale = 1e3
        ax.plot(direct[omega] * scale, depth[omega], color=BLUE, label="Direct thermal softening")
        ax.plot(hardening[omega] * scale, depth[omega], color=ORANGE, linestyle="--", label="Dehydration hardening")
        ax.plot(net[omega] * scale, depth[omega], color=TEAL, linestyle="-.", label="Net sensitivity")
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.8)
        ax.set_ylim(101, 33)
        ax.set_xlabel(r"$\partial\ln\eta_{eff}/\partial T$ ($10^{-3}$ K$^{-1}$)")
        ax.tick_params(labelleft=False)
        ax.legend(frameon=False, fontsize=5.7)
        clean_axis(ax)
        panel(ax, "b")

        ax = fig.add_subplot(grid[1, :])
        mesh = ax.pcolormesh(cdepth, cb_grid, net_grid.T, shading="auto", cmap="magma", norm=Normalize(vmin=-0.022, vmax=0.0))
        ax.contour(cdepth, cb_grid, net_grid.T, levels=(-0.015, -0.01, -0.005, -0.002), colors="white", linewidths=0.55)
        ax.plot(cdepth, accepted_cb, color="#19A7AE", linewidth=1.5, label="Reference water state")
        ax.set_yscale("log")
        ax.set_xlim(35, 100)
        ax.set_ylim(1e-6, 1.0)
        ax.set_xlabel("Depth (km)")
        ax.set_ylabel("Normalized bound-water content $C_b/C_b^{max}$")
        ax.legend(frameon=False, loc="lower left")
        fig.colorbar(mesh, ax=ax, label="Net sensitivity $S+H$ (K$^{-1}$)", pad=0.015)
        clean_axis(ax, grid=None)
        panel(ax, "c")

    path = Path(output) / "Fig05.pdf"
    save_pdf(fig, path, "Figure 5 deep-water rheology derivative")
    return path
