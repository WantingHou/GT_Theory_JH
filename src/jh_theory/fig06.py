"""Build Figure 6: timescale-separated reduced-order column."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import BLUE, GOLD, INK, ORANGE, PURPLE, TEAL, clean_axis, configure_depth_axis, depth_coordinate, finite, load_data, panel, rc_context, save_pdf

ZONES = ("surface", "upper_crust", "lower_crust", "lithosphere")


def _cell_series(d, field: str) -> tuple[np.ndarray, np.ndarray]:
    zparts, vparts = [], []
    for zone in ZONES:
        zkey = f"f6_{zone}__cell_z_m"
        vkey = f"f6_{zone}__{field}"
        if zkey in d.files and vkey in d.files:
            zparts.append(np.asarray(d[zkey]))
            vparts.append(np.asarray(d[vkey]))
    z = np.concatenate(zparts)
    v = np.concatenate(vparts)
    order = np.argsort(z)
    return z[order], v[order]


def _face_series(d, field: str) -> tuple[np.ndarray, np.ndarray]:
    zparts, vparts = [], []
    for zone in ZONES:
        zkey = f"f6_{zone}__face_z_m"
        vkey = f"f6_{zone}__{field}"
        if zkey in d.files and vkey in d.files:
            zparts.append(np.asarray(d[zkey]))
            vparts.append(np.asarray(d[vkey]))
    z = np.concatenate(zparts)
    v = np.concatenate(vparts)
    order = np.argsort(z)
    return z[order], v[order]


def build(output: Path) -> Path:
    with load_data() as d:
        zt, temperature = _cell_series(d, "temperature_K")
        reservoirs = {name: _cell_series(d, f"water_{name}_kg_m3") for name in ("liquid", "ice", "vapour", "bound", "melt")}
        exchange_fields = ("exchange_ice_to_liquid_kg_m3_s", "exchange_vapour_to_liquid_kg_m3_s", "exchange_net_hydration_kg_m3_s")
        exchanges = {name: _cell_series(d, name) for name in exchange_fields}
        energy = {
            "energy total": (*_face_series(d, "face_total_energy_W_m2"), INK, "-"),
            "conductive": (*_face_series(d, "face_conductive_W_m2"), BLUE, "--"),
            "aqueous": (*_face_series(d, "face_aqueous_W_m2"), TEAL, "-."),
            "vapour": (*_face_series(d, "face_vapour_W_m2"), GOLD, ":"),
        }
        water = {
            "water total": (*_face_series(d, "face_total_h2o_kg_m2_s"), INK, "-"),
            "water liquid": (*_face_series(d, "face_liquid_h2o_kg_m2_s"), BLUE, "--"),
            "water vapour": (*_face_series(d, "face_vapour_h2o_kg_m2_s"), GOLD, ":"),
        }
    finite(zt, temperature)
    for z, values, _, _ in (*energy.values(), *water.values()):
        finite(z, values)

    with rc_context():
        fig = plt.figure(figsize=(7.35, 5.3), constrained_layout=True)
        outer = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12))
        top_left = outer[0, 0].subgridspec(1, 2, wspace=0.42)
        ax_t = fig.add_subplot(top_left[0, 0])
        ax_r = fig.add_subplot(top_left[0, 1])
        ax_t.plot(temperature - 273.15, depth_coordinate(zt), color=ORANGE, linewidth=1.35)
        configure_depth_axis(ax_t)
        ax_t.set_xlabel("Temperature (°C)")
        clean_axis(ax_t)
        panel(ax_t, "a")
        for name, color, style in (("liquid", BLUE, "-"), ("ice", TEAL, "--"), ("vapour", GOLD, ":"), ("bound", PURPLE, "-."), ("melt", "#777777", "--")):
            z, values = reservoirs[name]
            ax_r.plot(values, depth_coordinate(z), color=color, linestyle=style, label=name)
        ax_r.set_xscale("symlog", linthresh=1e-7)
        ax_r.set_xlim(-1e-8, 500.0)
        ax_r.set_xticks((0.0, 1e-3, 1e2), ("0", "$10^{-3}$", "$10^2$"))
        configure_depth_axis(ax_r, label=False)
        ax_r.set_xlabel("Water-reservoir mass density (kg/m³)")
        ax_r.legend(frameon=False, fontsize=5.3)
        clean_axis(ax_r)

        ax = fig.add_subplot(outer[0, 1])
        for name, color, style, label in ((exchange_fields[0], TEAL, "--", "thaw / freeze"), (exchange_fields[1], BLUE, "-.", "condense / evaporate"), (exchange_fields[2], GOLD, ":", "hydrate / dehydrate")):
            z, values = exchanges[name]
            ax.plot(values, depth_coordinate(z), color=color, linestyle=style, label=label)
        ax.axvline(0, color="#777777", linewidth=0.6)
        ax.set_xscale("symlog", linthresh=1e-12)
        ax.set_xlim(-5e-5, 5e-6)
        ax.set_xticks((-1e-5, -1e-8, 0.0, 1e-8, 1e-6), ("$-10^{-5}$", "$-10^{-8}$", "0", "$10^{-8}$", "$10^{-6}$"))
        configure_depth_axis(ax, max_depth_m=35_000)
        ax.set_xlabel("Terminal exchange rate (kg/(m³ s))")
        ax.legend(frameon=False, fontsize=5.4)
        clean_axis(ax)
        panel(ax, "b")

        bottom = outer[1, :].subgridspec(1, 3, width_ratios=(1.1, 0.8, 1.0), wspace=0.42)
        ax = fig.add_subplot(bottom[0, 0])
        for label, (z, values, color, style) in energy.items():
            ax.plot(values * 1e3, depth_coordinate(z), color=color, linestyle=style, label=label)
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlim(-1.5e4, 2.0e4)
        ax.set_xticks((-1e4, -10.0, 0.0, 1e3, 1e4), ("$-10^4$", "$-10$", "0", "$10^3$", "$10^4$"))
        configure_depth_axis(ax)
        ax.set_xlabel("Upward energy flux (mW/m²)")
        ax.legend(frameon=False, fontsize=5.2)
        clean_axis(ax)
        panel(ax, "c")

        ax = fig.add_subplot(bottom[0, 1])
        ze, total_energy, _, _ = energy["energy total"]
        zc, conductive, _, _ = energy["conductive"]
        deep_total = ze >= 35_000
        deep_cond = zc >= 35_000
        ax.plot(total_energy[deep_total] * 1e3, ze[deep_total] / 1000.0, color=INK)
        ax.plot(conductive[deep_cond] * 1e3, zc[deep_cond] / 1000.0, color=BLUE, linestyle="--")
        ax.set_ylim(101, 34)
        ax.set_xlabel("Deep energy flux (mW/m²)")
        ax.set_ylabel("Depth (km)")
        clean_axis(ax)

        ax = fig.add_subplot(bottom[0, 2])
        for label, (z, values, color, style) in water.items():
            ax.plot(values, depth_coordinate(z), color=color, linestyle=style, label=label)
        ax.set_xscale("symlog", linthresh=1e-14)
        ax.set_xlim(-1e-7, 5e-6)
        ax.set_xticks((-1e-7, 0.0, 1e-8, 1e-6), ("$-10^{-7}$", "0", "$10^{-8}$", "$10^{-6}$"))
        configure_depth_axis(ax, max_depth_m=35_000)
        ax.set_xlabel("Upward water flux (kg/(m² s))")
        ax.legend(frameon=False, fontsize=5.2)
        clean_axis(ax)

    path = Path(output) / "Fig06.pdf"
    save_pdf(fig, path, "Figure 6 reduced-order column")
    return path
