"""
Vibrational frequency calculations, quasi-harmonic correction, and free energy.

Provides normal mode analysis via central finite differences, principal moment
of inertia linearity detection, mode decomposition, Truhlar quasi-harmonic
frequency shifting, and standard Gibbs free energy evaluation under the ideal
gas rigid-rotor harmonic-oscillator approximation.
"""

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field

import ase.units as units
import numpy as np
from ase import Atoms
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations

__all__ = [
    "FD_STEP_ANG",
    "LINEAR_MOMENT_TOL",
    "QH_FLOOR_CM",
    "STANDARD_PRESSURE_PA",
    "ModeSplit",
    "VibResult",
    "calculate_gibbs_energy",
    "calculate_vibrations",
    "extract_signed_frequencies",
    "free_energy_ev",
    "ideal_gas_thermo",
    "is_linear",
    "multiplicity_to_spin",
    "partition_rigid_modes",
    "quasi_harmonic",
    "split_modes",
    "thermo_geometry",
    "vibrate",
    "vibrational_energies_ev",
]

QH_FLOOR_CM: float = 100.0
FD_STEP_ANG: float = 0.01
STANDARD_PRESSURE_PA: float = 101325.0
LINEAR_MOMENT_TOL: float = 0.01


@dataclass(frozen=True, slots=True)
class VibResult:
    """
    Vibrational frequency calculation outcome for an atomic system.

    :param frequencies_cm: Signed frequencies in cm^-1 (all 3N normal modes).
    :type frequencies_cm: np.ndarray
    :param modes: Mode displacement vectors with shape (3N, N, 3).
    :type modes: np.ndarray
    :param linear: Whether the atomic system has linear geometry.
    :type linear: bool
    """

    frequencies_cm: np.ndarray = field(compare=False)
    modes: np.ndarray = field(compare=False)
    linear: bool


@dataclass(frozen=True, slots=True)
class ModeSplit:
    """
    Decomposition of normal modes into rigid, vibrational, and imaginary.

    :param rigid_cm: Rigid body translational and rotational modes in cm^-1.
    :type rigid_cm: np.ndarray
    :param vibrational_cm: True vibrational normal modes in cm^-1.
    :type vibrational_cm: np.ndarray
    :param imaginary_cm: Transition state imaginary frequency if present.
    :type imaginary_cm: float or None
    """

    rigid_cm: np.ndarray = field(compare=False)
    vibrational_cm: np.ndarray = field(compare=False)
    imaginary_cm: float | None = None


def is_linear(atoms: Atoms, tol: float = LINEAR_MOMENT_TOL) -> bool:
    """
    Determine whether an atomic structure has linear geometry.

    :param atoms: Atomic configuration to evaluate.
    :type atoms: Atoms
    :param tol: Principal moment of inertia threshold in amu*Å^2.
    :type tol: float
    :return: True if structure is linear, False otherwise.
    :rtype: bool
    """
    if len(atoms) < 2:
        return False
    if len(atoms) == 2:
        return True
    moments = atoms.get_moments_of_inertia()
    return bool(float(np.min(moments)) < tol)


def extract_signed_frequencies(raw_freqs: np.ndarray) -> np.ndarray:
    """
    Convert complex ASE frequencies into signed real values in cm^-1.

    :param raw_freqs: Complex frequencies returned by Hessian diagonalization.
    :type raw_freqs: np.ndarray
    :return: Real array where imaginary frequencies carry negative signs.
    :rtype: np.ndarray
    """
    is_imag = np.abs(raw_freqs.imag) > np.abs(raw_freqs.real)
    return np.where(is_imag, -np.abs(raw_freqs.imag), raw_freqs.real).astype(
        float
    )


def calculate_vibrations(
    atoms: Atoms, delta_ang: float = FD_STEP_ANG
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate vibrational frequencies and normal modes using ASE Vibrations.

    :param atoms: Atomic structure with attached calculator.
    :type atoms: Atoms
    :param delta_ang: Displacement step size in Angstrom.
    :type delta_ang: float
    :return: Signed frequencies in cm^-1 and mode displacement vectors.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        vib = Vibrations(atoms, name=f"{tmp_dir}/vib", delta=delta_ang, nfree=2)
        vib.run()
        vdata = vib.get_vibrations()
        freqs_cm = extract_signed_frequencies(vdata.get_frequencies())
        return freqs_cm, vdata.get_modes()


def vibrate(atoms: Atoms, delta_ang: float = FD_STEP_ANG) -> VibResult:
    """
    Compute finite-difference vibrational frequencies and normal modes.

    :param atoms: Atomic structure carrying an attached calculator.
    :type atoms: Atoms
    :param delta_ang: Displacement step size in Angstrom.
    :type delta_ang: float
    :return: Vibrational calculation outcome with all 3N normal modes.
    :rtype: VibResult
    :raises ValueError: If atoms does not carry an attached calculator.
    """
    if atoms.calc is None:
        raise ValueError("Atoms object must have an attached calculator.")
    linear = is_linear(atoms)
    target = atoms.copy()
    target.calc = atoms.calc
    freqs_cm, modes = calculate_vibrations(target, delta_ang=delta_ang)
    return VibResult(frequencies_cm=freqs_cm, modes=modes, linear=linear)


def partition_rigid_modes(
    frequencies_cm: np.ndarray, linear: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate rigid-body modes from remaining modes by frequency magnitude.

    :param frequencies_cm: Signed normal mode frequencies in cm^-1.
    :type frequencies_cm: np.ndarray
    :param linear: Whether the atomic system has linear geometry.
    :type linear: bool
    :return: Tuple of rigid-body modes and sorted remaining modes.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    n_rigid = 5 if linear else 6
    abs_freqs = np.abs(frequencies_cm)
    order = np.argsort(abs_freqs)
    rigid_cm = frequencies_cm[order[:n_rigid]]
    remaining = np.sort(frequencies_cm[order[n_rigid:]])
    return rigid_cm, remaining


def split_modes(result: VibResult, transition_state: bool) -> ModeSplit:
    """
    Split all 3N normal modes into rigid, vibrational, and imaginary subsets.

    :param result: Vibrational calculation outcome.
    :type result: VibResult
    :param transition_state: Whether the structure is a transition state.
    :type transition_state: bool
    :return: Mode decomposition dataclass.
    :rtype: ModeSplit
    """
    rigid_cm, remaining = partition_rigid_modes(
        result.frequencies_cm, result.linear
    )
    if not transition_state or len(remaining) == 0:
        return ModeSplit(rigid_cm=rigid_cm, vibrational_cm=remaining)
    return ModeSplit(
        rigid_cm=rigid_cm,
        vibrational_cm=remaining[1:],
        imaginary_cm=float(remaining[0]),
    )


def quasi_harmonic(
    frequencies_cm: np.ndarray | Sequence[float],
    floor_cm: float = QH_FLOOR_CM,
) -> np.ndarray:
    """
    Apply quasi-harmonic floor to vibrational frequencies.

    :param frequencies_cm: Vibrational frequencies in cm^-1.
    :type frequencies_cm: np.ndarray or Sequence[float]
    :param floor_cm: Minimum frequency floor in cm^-1.
    :type floor_cm: float
    :return: Array of frequencies with values below floor raised to it.
    :rtype: np.ndarray
    """
    freqs = np.asarray(frequencies_cm, dtype=float)
    return np.maximum(freqs, floor_cm)


def thermo_geometry(atoms: Atoms) -> str:
    """
    Determine molecular geometry classification for thermochemistry.

    :param atoms: Atomic configuration.
    :type atoms: Atoms
    :return: Classification ('monatomic', 'linear', or 'nonlinear').
    :rtype: str
    """
    if len(atoms) == 1:
        return "monatomic"
    if is_linear(atoms):
        return "linear"
    return "nonlinear"


def multiplicity_to_spin(multiplicity: int) -> float:
    """
    Convert spin multiplicity to total spin quantum number.

    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Total spin quantum number S.
    :rtype: float
    :raises ValueError: If multiplicity is less than 1.
    """
    if multiplicity < 1:
        raise ValueError("Multiplicity must be >= 1.")
    return (multiplicity - 1) / 2.0


def vibrational_energies_ev(
    vibrational_cm: np.ndarray | Sequence[float],
) -> list[float]:
    """
    Convert vibrational frequencies in cm^-1 to energies in eV.

    :param vibrational_cm: Vibrational frequencies in cm^-1.
    :type vibrational_cm: np.ndarray or Sequence[float]
    :return: Vibrational mode energies in eV.
    :rtype: list[float]
    """
    return [float(f) * units.invcm for f in vibrational_cm]


def ideal_gas_thermo(
    atoms: Atoms, energy_ev: float, vib_energies: Sequence[float], spin: float
) -> IdealGasThermo:
    """
    Construct ASE IdealGasThermo model for an atomic structure.

    :param atoms: Atomic configuration.
    :type atoms: Atoms
    :param energy_ev: Electronic potential energy in eV.
    :type energy_ev: float
    :param vib_energies: Vibrational mode energies in eV.
    :type vib_energies: Sequence[float]
    :param spin: Total spin quantum number S.
    :type spin: float
    :return: Configured IdealGasThermo instance.
    :rtype: IdealGasThermo
    """
    return IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=energy_ev,
        atoms=atoms,
        geometry=thermo_geometry(atoms),
        symmetrynumber=1,
        spin=spin,
        natoms=len(atoms),
        vib_selection="all",
    )


def calculate_gibbs_energy(
    thermo: IdealGasThermo, temperature_k: float
) -> float:
    """
    Evaluate standard Gibbs free energy in eV for an IdealGasThermo model.

    :param thermo: Configured IdealGasThermo instance.
    :type thermo: IdealGasThermo
    :param temperature_k: Temperature in Kelvin.
    :type temperature_k: float
    :return: Gibbs free energy G(T) in eV.
    :rtype: float
    """
    return float(
        thermo.get_gibbs_energy(
            temperature=temperature_k,
            pressure=STANDARD_PRESSURE_PA,
            verbose=False,
        )
    )


def free_energy_ev(
    atoms: Atoms,
    energy_ev: float,
    vibrational_cm: np.ndarray | Sequence[float],
    temperature_k: float,
    multiplicity: int,
) -> float:
    """
    Compute standard Gibbs free energy in eV under ideal gas approximation.

    :param atoms: Atomic configuration.
    :type atoms: Atoms
    :param energy_ev: Electronic potential energy in eV.
    :type energy_ev: float
    :param vibrational_cm: Vibrational frequencies in cm^-1.
    :type vibrational_cm: np.ndarray or Sequence[float]
    :param temperature_k: Temperature in Kelvin.
    :type temperature_k: float
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Gibbs free energy G(T) in eV.
    :rtype: float
    :raises ValueError: If multiplicity is less than 1.
    """
    spin = multiplicity_to_spin(multiplicity)
    vib_energies = vibrational_energies_ev(vibrational_cm)
    thermo = ideal_gas_thermo(atoms, energy_ev, vib_energies, spin)
    return calculate_gibbs_energy(thermo, temperature_k=temperature_k)
