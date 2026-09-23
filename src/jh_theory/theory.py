"""Equation-level helpers used by the JH figure builders."""

from __future__ import annotations

import numpy as np

GAS_CONSTANT = 8.31446261815324
SECONDS_PER_YEAR = 31_557_600.0


def rheology_temperature_sensitivity(
    temperature_K: np.ndarray,
    pressure_Pa: np.ndarray,
    bound_fraction: np.ndarray,
    *,
    activation_energy_J_mol: float = 480_000.0,
    activation_volume_m3_mol: float = 11.0e-6,
    stress_exponent: float = 3.5,
    water_exponent: float = 1.2,
    maximum_bound_fraction: float = 0.01,
    dehydration_width_K: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return direct softening, dehydration hardening, and their local sum."""
    temperature = np.asarray(temperature_K, dtype=float)
    pressure = np.asarray(pressure_Pa, dtype=float)
    bound = np.asarray(bound_fraction, dtype=float)
    direct = -(activation_energy_J_mol + pressure * activation_volume_m3_mol) / (
        stress_exponent * GAS_CONSTANT * temperature**2
    )
    hardening = (water_exponent / stress_exponent) * (
        1.0 - bound / maximum_bound_fraction
    ) / dehydration_width_K
    return direct, hardening, direct + hardening


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = np.nanmax(np.abs(values))
    if not np.isfinite(scale) or scale == 0.0:
        return np.zeros_like(values)
    return values / scale


def signed_log_ticks(limit: float) -> tuple[np.ndarray, list[str]]:
    ticks = np.array([-limit, -limit / 100.0, -limit / 10_000.0, 0.0, limit / 10_000.0, limit / 100.0, limit])
    labels = [f"−10$^{{{int(np.log10(abs(v)))}}}$" if v < 0 else ("0" if v == 0 else f"10$^{{{int(np.log10(v))}}}$") for v in ticks]
    return ticks, labels
