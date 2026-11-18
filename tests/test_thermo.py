"""
Unit tests for finite-difference vibrations, quasi-harmonic, and free energy.

Validates normal mode analysis, linearity detection, transition state mode
splitting, Truhlar quasi-harmonic frequency correction, and
temperature-dependent Gibbs free energy calculations.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from nrnet.engines import attach, load_engine_settings
from nrnet.optimize import relax
from nrnet.thermo import (
    ModeSplit,
    VibResult,
    free_energy_ev,
    is_linear,
    quasi_harmonic,
    split_modes,
    vibrate,
)


def test_vib_result_dataclass() -> None:
    """
    Verify VibResult dataclass creation and field access.

    :return: None.
    :rtype: None
    """
    freqs = np.array([100.0, 200.0, 300.0])
    modes = np.zeros((3, 1, 3))
    res = VibResult(frequencies_cm=freqs, modes=modes, linear=False)
    assert np.array_equal(res.frequencies_cm, freqs)
    assert res.modes.shape == (3, 1, 3)
    assert res.linear is False


def test_vibrate_without_calculator_raises() -> None:
    """
    Verify vibrate raises ValueError when calculator is absent.

    :return: None.
    :rtype: None
    """
    water = molecule("H2O")
    with pytest.raises(
        ValueError, match="Atoms object must have an attached calculator"
    ):
        vibrate(water)


def test_is_linear() -> None:
    """
    Verify linearity detection for diatomic, linear, bent, and monatomic.

    :return: None.
    :rtype: None
    """
    assert is_linear(molecule("H2")) is True
    assert is_linear(molecule("CO2")) is True
    assert is_linear(molecule("H2O")) is False
    assert is_linear(Atoms("He")) is False


def test_quasi_harmonic_correction() -> None:
    """
    Verify quasi-harmonic floor raises low modes and keeps high modes.

    :return: None.
    :rtype: None
    """
    test_freqs = np.array([-50.0, 50.0, 500.0])
    corrected = quasi_harmonic(test_freqs, floor_cm=100.0)
    expected = np.array([100.0, 100.0, 500.0])
    assert np.allclose(corrected, expected)


def test_split_modes_fake_ts() -> None:
    """
    Verify split_modes extracts imaginary mode for a transition state.

    :return: None.
    :rtype: None
    """
    fake_freqs = np.array(
        [-800.0, -10.0, -5.0, 0.0, 2.0, 8.0, 15.0, 1500.0, 3500.0]
    )
    fake_modes = np.zeros((9, 3, 3))
    result = VibResult(
        frequencies_cm=fake_freqs, modes=fake_modes, linear=False
    )
    split = split_modes(result, transition_state=True)
    assert isinstance(split, ModeSplit)
    assert split.imaginary_cm == pytest.approx(-800.0)
    assert len(split.vibrational_cm) == 2
    assert np.allclose(split.vibrational_cm, [1500.0, 3500.0])
    assert len(split.rigid_cm) == 6


def test_split_modes_fake_minimum() -> None:
    """
    Verify split_modes leaves imaginary_cm as None for a local minimum.

    :return: None.
    :rtype: None
    """
    fake_freqs = np.array(
        [-10.0, -5.0, 0.0, 2.0, 8.0, 15.0, 1500.0, 3500.0, 3600.0]
    )
    fake_modes = np.zeros((9, 3, 3))
    result = VibResult(
        frequencies_cm=fake_freqs, modes=fake_modes, linear=False
    )
    split = split_modes(result, transition_state=False)
    assert isinstance(split, ModeSplit)
    assert split.imaginary_cm is None
    assert len(split.vibrational_cm) == 3
    assert np.allclose(split.vibrational_cm, [1500.0, 3500.0, 3600.0])
    assert len(split.rigid_cm) == 6


@pytest.fixture(scope="module")
def relaxed_water() -> tuple[Atoms, float]:
    """
    Relax water molecule with xTB and compute electronic potential energy.

    :return: Relaxed water atoms and electronic potential energy in eV.
    :rtype: tuple[Atoms, float]
    """
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    water = molecule("H2O")
    attached = attach(water, settings)
    opt_result = relax(attached, fmax_ev_per_ang=0.001)
    relaxed_eval = attach(opt_result.atoms, settings)
    return opt_result.atoms, float(relaxed_eval.get_potential_energy())


@pytest.fixture(scope="module")
def water_vib_result(relaxed_water: tuple[Atoms, float]) -> VibResult:
    """
    Compute finite-difference vibrations for relaxed water molecule.

    :param relaxed_water: Relaxed water atoms and electronic energy.
    :type relaxed_water: tuple[Atoms, float]
    :return: Vibrational calculation outcome.
    :rtype: VibResult
    """
    atoms, _ = relaxed_water
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    return vibrate(attach(atoms, settings))


def test_water_vibrations_linearity(water_vib_result: VibResult) -> None:
    """
    Verify relaxed water has non-linear geometry.

    :param water_vib_result: Vibrational calculation outcome.
    :type water_vib_result: VibResult
    :return: None.
    :rtype: None
    """
    assert water_vib_result.linear is False


def test_water_vibrational_modes(water_vib_result: VibResult) -> None:
    """
    Verify relaxed water yields three positive vibrational modes.

    :param water_vib_result: Vibrational calculation outcome.
    :type water_vib_result: VibResult
    :return: None.
    :rtype: None
    """
    split = split_modes(water_vib_result, transition_state=False)
    assert len(split.vibrational_cm) == 3
    assert np.all(split.vibrational_cm > 0.0)


def test_water_vibrational_frequencies(
    water_vib_result: VibResult,
) -> None:
    """
    Verify relaxed water bending and stretching frequency ranges.

    :param water_vib_result: Vibrational calculation outcome.
    :type water_vib_result: VibResult
    :return: None.
    :rtype: None
    """
    split = split_modes(water_vib_result, transition_state=False)
    bend = split.vibrational_cm[0]
    stretch_sym = split.vibrational_cm[1]
    stretch_asym = split.vibrational_cm[2]
    assert 1300.0 < bend < 1800.0
    assert 3300.0 < stretch_sym < 4000.0
    assert 3300.0 < stretch_asym < 4000.0


def test_water_free_energy_temperature_scaling(
    relaxed_water: tuple[Atoms, float],
    water_vib_result: VibResult,
) -> None:
    """
    Verify water Gibbs free energy decreases with increasing temperature.

    :param relaxed_water: Relaxed water atoms and electronic energy.
    :type relaxed_water: tuple[Atoms, float]
    :param water_vib_result: Vibrational calculation outcome.
    :type water_vib_result: VibResult
    :return: None.
    :rtype: None
    """
    atoms, energy_ev = relaxed_water
    split = split_modes(water_vib_result, transition_state=False)
    qh_modes = quasi_harmonic(split.vibrational_cm)
    g_298 = free_energy_ev(
        atoms, energy_ev, qh_modes, temperature_k=298.15, multiplicity=1
    )
    g_600 = free_energy_ev(
        atoms, energy_ev, qh_modes, temperature_k=600.0, multiplicity=1
    )
    assert g_600 < g_298


def test_co2_linear_vibrations() -> None:
    """
    Verify CO2 is recognized as linear and yields four vibrational modes.

    :return: None.
    :rtype: None
    """
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    co2 = molecule("CO2")
    attached = attach(co2, settings)
    opt_result = relax(attached, fmax_ev_per_ang=0.001)
    relaxed_eval = attach(opt_result.atoms, settings)

    vib_result = vibrate(relaxed_eval)
    assert vib_result.linear is True

    split = split_modes(vib_result, transition_state=False)
    assert len(split.rigid_cm) == 5
    assert len(split.vibrational_cm) == 4
    assert split.imaginary_cm is None
