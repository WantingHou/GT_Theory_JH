from __future__ import annotations

import numpy as np

from jh_theory.common import load_data
from jh_theory.theory import normalized, rheology_temperature_sensitivity


def test_packaged_arrays_are_finite_or_explicitly_masked() -> None:
    with load_data() as data:
        assert len(data.files) >= 200
        for name in data.files:
            array = np.asarray(data[name])
            if array.dtype.kind in "fci":
                assert not np.any(np.isinf(array)), name


def test_rheology_derivative_has_expected_signs() -> None:
    with load_data() as data:
        temperature = np.asarray(data["f5_temperature_K"])
        pressure = np.asarray(data["f5_lithostatic_pressure_Pa"])
        bound = np.asarray(data["f5_initial_Cb"])
        active = np.asarray(data["f5_omega_rheo_mask_u8"], dtype=bool)
    direct, hardening, net = rheology_temperature_sensitivity(temperature, pressure, bound)
    assert direct.shape == temperature.shape == hardening.shape == net.shape
    assert np.all(direct[active] < 0.0)
    assert np.all(hardening[active] >= 0.0)
    assert np.allclose(net, direct + hardening)


def test_normalized_response_is_bounded() -> None:
    result = normalized(np.array([-2.0, 0.0, 1.0]))
    assert np.max(np.abs(result)) == 1.0


def test_figure5_derivative_matches_accepted_static_state() -> None:
    with load_data() as data:
        temperature = np.asarray(data["f5_temperature_K"])
        pressure = np.asarray(data["f5_lithostatic_pressure_Pa"])
        bound = np.asarray(data["f5_initial_Cb"])
        active = np.asarray(data["f5_omega_rheo_mask_u8"], dtype=bool)
        expected = np.asarray(data["f45_c_accepted_net_K_inv"])
    _, _, actual = rheology_temperature_sensitivity(temperature, pressure, bound)
    assert actual[active].shape == expected.shape
    assert np.allclose(actual[active], expected, rtol=1.0e-12, atol=1.0e-15)
