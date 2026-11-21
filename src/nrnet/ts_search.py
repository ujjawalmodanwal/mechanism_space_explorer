"""
Transition state search, NEB optimization, saddle refinement, and verification.

Implements the Stage 4 reaction transition state search pipeline: endpoint
geometry relaxation, climbing-image nudged elastic band (NEB) interpolation,
Sella first-order saddle point refinement, finite-difference vibrational normal
mode analysis, and displaced endpoint confirmation.
"""

import signal
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import CalculationFailed
from ase.mep import NEB
from ase.optimize import FIRE
from sella import Sella

from nrnet.chem import state_key
from nrnet.engines import EngineSettings, UnsupportedSpecies, attach
from nrnet.nanoreactor.candidates import Candidate, EndpointPair
from nrnet.optimize import relax
from nrnet.thermo import (
    partition_rigid_modes,
    split_modes,
    vibrate,
)

__all__ = [
    "SEARCH_STEPS",
    "SEED_STEPS",
    "AttemptTimeLimit",
    "SearchSettings",
    "SearchState",
    "TSResult",
    "TSRoute",
    "TSSeed",
    "TSStatus",
    "aligned_band",
    "attempt_time_limit",
    "check_imaginary",
    "check_neb_band_status",
    "confirm_endpoints",
    "confirm_step",
    "endpoints_match_expected",
    "frequency_step",
    "highest_neb_image",
    "is_barrierless_band",
    "load_search_settings",
    "neb_guess",
    "neb_step",
    "raise_time_limit",
    "refine_saddle",
    "relax_endpoint",
    "relax_endpoints_step",
    "relax_single_endpoint",
    "result_from_state",
    "run_band",
    "run_search_steps",
    "saddle_step",
    "search_candidate",
    "search_pair",
    "seed_step",
]


class TSStatus(StrEnum):
    """
    Termination status for transition state search attempts.
    """

    confirmed = "confirmed"
    barrierless = "barrierless"
    endpoint_changed = "endpoint_changed"
    neb_failed = "neb_failed"
    saddle_not_converged = "saddle_not_converged"
    wrong_imaginary_count = "wrong_imaginary_count"
    endpoint_mismatch = "endpoint_mismatch"
    unsupported = "unsupported"
    engine_error = "engine_error"
    time_limit = "time_limit"


class AttemptTimeLimit(BaseException):
    """
    Raised when a search attempt runs past its wall-time limit.

    It derives from BaseException so that the stages' own handlers (Sella's
    RuntimeError in saddle_step, engine failures) cannot swallow it.
    """


def raise_time_limit(signum: int, frame: object) -> None:
    """
    SIGALRM handler that ends the running attempt.

    :param signum: Signal number (SIGALRM).
    :type signum: int
    :param frame: Interrupted stack frame.
    :type frame: object
    :raises AttemptTimeLimit: Always.
    """
    raise AttemptTimeLimit


@contextmanager
def attempt_time_limit(seconds: float | None) -> Iterator[None]:
    """
    Raise AttemptTimeLimit in the main thread once seconds have passed.

    SIGALRM is handled between Python steps, so a native call already
    running (one SCF) finishes first. Engines that start subprocesses
    (ORCA) would leave them running, so set the limit only for in-process
    engines. None or 0 disables it.

    :param seconds: Wall-time limit in seconds, or None.
    :type seconds: float or None
    :return: Context manager yielding None.
    :rtype: Iterator[None]
    """
    if not seconds:
        yield
        return
    previous = signal.signal(signal.SIGALRM, raise_time_limit)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class TSRoute(StrEnum):
    """
    How the saddle guess was made: the engine's own band, or a seed TS.
    """

    band = "band"
    seed = "seed"


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """
    Configuration parameters for transition state search pipeline.
    """

    min_fmax_ev_per_ang: float
    min_max_steps: int
    neb_images: int
    neb_relax_fmax_ev_per_ang: float
    neb_relax_max_steps: int
    neb_fmax_ev_per_ang: float
    neb_max_steps: int
    saddle_fmax_ev_per_ang: float
    saddle_max_steps: int
    imaginary_threshold_cm: float
    confirm_step_ang: float
    barrierless_tol_ev: float = 0.04


@dataclass(frozen=True, slots=True)
class TSResult:
    """
    Outcome of transition state search for an individual endpoint pair.
    """

    status: TSStatus
    pair_index: int
    ts_atoms: Atoms | None = field(default=None, compare=False)
    ts_energy_ev: float | None = None
    reactant_energy_ev: float | None = None
    product_energy_ev: float | None = None
    imaginary_cm: float | None = None
    vibrational_cm: tuple[float, ...] | None = None
    neb_converged: bool = False
    force_calls: int = 0
    note: str = ""
    route: TSRoute = TSRoute.band


@dataclass(frozen=True, slots=True)
class TSSeed:
    """
    Another engine's confirmed TS, used as this engine's first guess.

    Atom order matches only the endpoint pair the seed was found on, so
    the seed carries that pair's index.

    :param pair_index: Endpoint pair the seed TS belongs to.
    :type pair_index: int
    :param atoms: Seed TS geometry.
    :type atoms: Atoms
    :param engine: Engine that found the seed.
    :type engine: str
    """

    pair_index: int
    atoms: Atoms = field(compare=False)
    engine: str


@dataclass(slots=True)
class SearchState:
    """
    Mutable state shared across transition state search pipeline steps.
    """

    pair: EndpointPair
    pair_index: int
    engine_settings: EngineSettings
    search_settings: SearchSettings
    expected_keys: tuple[tuple[str, ...], ...] = ()
    relaxed_reactant: Atoms | None = None
    relaxed_product: Atoms | None = None
    reactant_energy_ev: float | None = None
    product_energy_ev: float | None = None
    ts_guess: Atoms | None = None
    neb_converged: bool = False
    ts_atoms: Atoms | None = None
    ts_energy_ev: float | None = None
    imaginary_cm: float | None = None
    vibrational_cm: tuple[float, ...] | None = None
    mode: np.ndarray | None = None
    force_calls: int = 0
    note: str = ""
    seed: Atoms | None = None

    def __post_init__(self) -> None:
        """
        Initialize expected state keys from pair endpoints if not provided.

        :return: None.
        :rtype: None
        """
        if not self.expected_keys:
            self.expected_keys = (
                state_key(self.pair.before, stereo=True),
                state_key(self.pair.after, stereo=True),
            )


def load_search_settings(path: str | Path) -> SearchSettings:
    """
    Load transition state search settings from a YAML configuration file.

    :param path: Path to the YAML configuration file.
    :type path: str or Path
    :return: Validated transition state search settings dataclass.
    :rtype: SearchSettings
    :raises KeyError: If the 'search' configuration section is missing.
    :raises TypeError: If the 'search' configuration section is not a mapping.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping) or "search" not in data:
        raise KeyError(f"Missing 'search' section in {path}")
    section = data["search"]
    if not isinstance(section, Mapping):
        raise TypeError(f"'search' section in {path} must be a mapping")
    return SearchSettings(**section)


def relax_endpoint(
    atoms: Atoms,
    engine_settings: EngineSettings,
    fmax_ev_per_ang: float = 0.01,
    max_steps: int = 1000,
    expected_state: tuple[str, ...] | None = None,
) -> tuple[Atoms, float, int, bool]:
    """
    Relax an endpoint structure and verify its canonical state key.

    :param atoms: Input atomic configuration to relax.
    :type atoms: Atoms
    :param engine_settings: Computational chemistry engine settings.
    :type engine_settings: EngineSettings
    :param fmax_ev_per_ang: Force threshold in eV/Å.
    :type fmax_ev_per_ang: float
    :param max_steps: Maximum allowable geometry optimization steps.
    :type max_steps: int
    :param expected_state: Expected state key tuple; inferred if None.
    :type expected_state: tuple[str, ...] or None
    :return: Relaxed geometry, energy in eV, steps taken, and match flag.
    :rtype: tuple[Atoms, float, int, bool]
    """
    target = (
        state_key(atoms, stereo=True)
        if expected_state is None
        else expected_state
    )
    eval_atoms = attach(atoms, engine_settings)
    res = relax(
        eval_atoms,
        fmax_ev_per_ang=fmax_ev_per_ang,
        max_steps=max_steps,
    )
    energy = float(attach(res.atoms, engine_settings).get_potential_energy())
    matched = bool(state_key(res.atoms, stereo=True) == target)
    return res.atoms, energy, res.steps, matched


def highest_neb_image(
    images: Sequence[Atoms], energies: tuple[float, ...]
) -> Atoms:
    """
    Extract highest-energy inner image from optimized NEB band.

    :param images: Sequence of optimized NEB band atomic configurations.
    :type images: Sequence[Atoms]
    :param energies: Potential energies across all images in eV.
    :type energies: tuple[float, ...]
    :return: Copy of highest inner image without attached calculator.
    :rtype: Atoms
    """
    finite = all(np.isfinite(val) for val in energies)
    idx = (
        1 + int(np.argmax(energies[1:-1])) if finite and len(images) > 2 else 1
    )
    image = images[idx].copy()
    image.calc = None
    return image


def aligned_band(
    reactant: Atoms,
    product: Atoms,
    neb_images: int,
    engine_settings: EngineSettings,
) -> list[Atoms]:
    """
    Build an IDPP band with the product turned onto the reactant.

    The endpoints are relaxed apart, so they may sit rotated or shifted
    against each other; interpolating without alignment adds that rigid
    motion to every image and the band takes a detour.

    :param reactant: Relaxed initial endpoint atomic structure.
    :type reactant: Atoms
    :param product: Relaxed final endpoint atomic structure.
    :type product: Atoms
    :param neb_images: Total number of images including both endpoints.
    :type neb_images: int
    :param engine_settings: Chemistry engine settings.
    :type engine_settings: EngineSettings
    :return: Interpolated images, each with a calculator attached.
    :rtype: list[Atoms]
    """
    start, end = reactant.copy(), product.copy()
    minimize_rotation_and_translation(start, end)
    images = [start.copy() for _ in range(neb_images - 1)] + [end]
    NEB(images, method="improvedtangent").interpolate("idpp")
    return [attach(img, engine_settings) for img in images]


def run_band(
    images: list[Atoms],
    climb: bool,
    fmax_ev_per_ang: float,
    max_steps: int,
) -> tuple[bool, int]:
    """
    Optimise an NEB band in place with FIRE.

    :param images: Band images with calculators, changed in place.
    :type images: list[Atoms]
    :param climb: Whether the highest image climbs to the saddle.
    :type climb: bool
    :param fmax_ev_per_ang: Force threshold in eV/Å.
    :type fmax_ev_per_ang: float
    :param max_steps: Maximum allowable optimisation steps.
    :type max_steps: int
    :return: Converged flag and steps taken.
    :rtype: tuple[bool, int]
    """
    neb = NEB(
        images,
        climb=climb,
        method="improvedtangent",
        allow_shared_calculator=True,
    )
    opt = FIRE(neb, logfile=None)
    converged = bool(opt.run(fmax=fmax_ev_per_ang, steps=max_steps))
    return converged, int(opt.get_number_of_steps())


def neb_guess(
    reactant: Atoms,
    product: Atoms,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
) -> tuple[Atoms, bool, int, tuple[float, ...]]:
    """
    Generate a transition state guess with a two-stage NEB.

    A plain NEB first settles the band onto the reaction path; only then
    does the highest image climb.

    :param reactant: Relaxed initial endpoint atomic structure.
    :type reactant: Atoms
    :param product: Relaxed final endpoint atomic structure.
    :type product: Atoms
    :param engine_settings: Chemistry engine settings.
    :type engine_settings: EngineSettings
    :param search_settings: Image count and both stages' thresholds.
    :type search_settings: SearchSettings
    :return: Highest image, climbing-stage converged flag, total steps,
        and image energies.
    :rtype: tuple[Atoms, bool, int, tuple[float, ...]]
    """
    cfg = search_settings
    images = aligned_band(reactant, product, cfg.neb_images, engine_settings)
    stage_one = (cfg.neb_relax_fmax_ev_per_ang, cfg.neb_relax_max_steps)
    _, relax_steps = run_band(images, False, *stage_one)
    stage_two = (cfg.neb_fmax_ev_per_ang, cfg.neb_max_steps)
    converged, climb_steps = run_band(images, True, *stage_two)
    ens = tuple(float(img.get_potential_energy()) for img in images)
    image = highest_neb_image(images, ens)
    return image, converged, relax_steps + climb_steps, ens


def refine_saddle(
    atoms: Atoms,
    engine_settings: EngineSettings | None = None,
    fmax_ev_per_ang: float = 0.01,
    max_steps: int = 500,
) -> tuple[Atoms, bool, int, float]:
    """
    Refine transition state guess to first-order saddle point using Sella.

    :param atoms: Initial saddle guess structure.
    :type atoms: Atoms
    :param engine_settings: Engine settings if calculator is absent.
    :type engine_settings: EngineSettings or None
    :param fmax_ev_per_ang: Maximum force threshold in eV/Å.
    :type fmax_ev_per_ang: float
    :param max_steps: Maximum allowable Sella optimization steps.
    :type max_steps: int
    :return: Refined saddle geometry, convergence flag, steps, and energy.
    :rtype: tuple[Atoms, bool, int, float]
    :raises ValueError: If atoms has no calculator and settings is None.
    """
    if atoms.calc is None and engine_settings is None:
        raise ValueError("Atoms object must have an attached calculator.")
    target = atoms if atoms.calc is not None else attach(atoms, engine_settings)
    opt = Sella(target, order=1, internal=True, logfile=None)
    conv = opt.run(fmax=fmax_ev_per_ang, steps=max_steps)
    refined = target.copy()
    refined.calc = None
    energy = float(target.get_potential_energy())
    return refined, bool(conv), int(opt.get_number_of_steps()), energy


def check_imaginary(
    atoms: Atoms,
    engine_settings: EngineSettings | None = None,
    threshold_cm: float = -50.0,
) -> tuple[bool, float | None, np.ndarray | None, tuple[float, ...] | None]:
    """
    Verify transition state has exactly one imaginary frequency below cutoff.

    :param atoms: Refined transition state atomic configuration.
    :type atoms: Atoms
    :param engine_settings: Engine settings if calculator is absent.
    :type engine_settings: EngineSettings or None
    :param threshold_cm: Imaginary frequency cutoff in cm^-1.
    :type threshold_cm: float
    :return: Validity flag, imaginary frequency, mode vector, and real modes.
    :rtype: tuple[bool, float or None, np.ndarray or None, tuple or None]
    :raises ValueError: If atoms has no calculator and settings is None.
    """
    if atoms.calc is None and engine_settings is None:
        raise ValueError("Atoms object must have an attached calculator.")
    target = atoms if atoms.calc is not None else attach(atoms, engine_settings)
    vib = vibrate(target)
    _, rem = partition_rigid_modes(vib.frequencies_cm, vib.linear)
    below = [val for val in rem if val < threshold_cm]
    if len(below) != 1:
        fallback = float(rem[0]) if len(rem) > 0 else None
        return False, fallback, None, None
    imag_val = float(below[0])
    idx = int(np.argmin(np.abs(vib.frequencies_cm - imag_val)))
    mode = np.asarray(vib.modes[idx], dtype=float)
    split = split_modes(vib, transition_state=True)
    vib_cm = tuple(float(val) for val in split.vibrational_cm)
    return True, imag_val, mode, vib_cm


def confirm_endpoints(
    ts_atoms: Atoms,
    mode: np.ndarray,
    expected_keys: Sequence[tuple[str, ...]] | set[tuple[str, ...]],
    engine_settings: EngineSettings,
    confirm_step_ang: float = 0.1,
    fmax_ev_per_ang: float = 0.01,
    max_steps: int = 1000,
) -> tuple[bool, int]:
    """
    Confirm transition state endpoints by displacing along imaginary mode.

    :param ts_atoms: Transition state structure without calculator.
    :type ts_atoms: Atoms
    :param mode: Normal mode displacement vectors with shape (N, 3).
    :type mode: np.ndarray
    :param expected_keys: Expected reactant and product state keys.
    :type expected_keys: Sequence[tuple[str, ...]] or set[tuple[str, ...]]
    :param engine_settings: Computational chemistry engine settings.
    :type engine_settings: EngineSettings
    :param confirm_step_ang: Displacement step size in Angstroms.
    :type confirm_step_ang: float
    :param fmax_ev_per_ang: Force threshold in eV/Å.
    :type fmax_ev_per_ang: float
    :param max_steps: Maximum allowable relaxation steps.
    :type max_steps: int
    :return: Confirmation flag and combined relaxation step count.
    :rtype: tuple[bool, int]
    """
    max_disp = float(np.max(np.linalg.norm(mode, axis=1)))
    scale = (confirm_step_ang / max_disp) if max_disp > 0.0 else 1.0
    total_steps, states = 0, []
    for sign in (1.0, -1.0):
        disp = ts_atoms.copy()
        disp.set_positions(ts_atoms.get_positions() + sign * scale * mode)
        res = relax(
            attach(disp, engine_settings),
            fmax_ev_per_ang=fmax_ev_per_ang,
            max_steps=max_steps,
        )
        states.append(state_key(res.atoms, stereo=True))
        total_steps += res.steps
    return bool(set(states) == set(expected_keys)), total_steps


def relax_single_endpoint(
    state: SearchState, is_product: bool
) -> TSStatus | None:
    """
    Relax single endpoint and record results in search state.

    :param state: Active search state dataclass.
    :type state: SearchState
    :param is_product: Whether relaxing product or reactant.
    :type is_product: bool
    :return: Failure status if invalid, None if verified.
    :rtype: TSStatus or None
    """
    atoms = state.pair.after if is_product else state.pair.before
    expected = state.expected_keys[1] if is_product else state.expected_keys[0]
    cfg = state.search_settings
    params = (cfg.min_fmax_ev_per_ang, cfg.min_max_steps, expected)
    relaxed, energy, steps, matched = relax_endpoint(
        atoms, state.engine_settings, *params
    )
    state.force_calls += steps
    if not np.isfinite(energy):
        return TSStatus.engine_error
    if is_product:
        state.relaxed_product, state.product_energy_ev = relaxed, energy
    else:
        state.relaxed_reactant, state.reactant_energy_ev = relaxed, energy
    return None if matched else TSStatus.endpoint_changed


def relax_endpoints_step(state: SearchState) -> TSStatus | None:
    """
    Relax reactant and product endpoints and verify canonical state keys.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :return: Failure status if endpoint changes, None to continue.
    :rtype: TSStatus or None
    """
    for is_product in (False, True):
        status = relax_single_endpoint(state, is_product)
        if status is not None:
            return status
    return None


def is_barrierless_band(
    energies: Sequence[float],
    tol_ev: float,
) -> bool:
    """
    Check if no interior image energy exceeds endpoints by more than tolerance.

    :param energies: Sequence of image potential energies along NEB band in eV.
    :type energies: Sequence[float]
    :param tol_ev: Maximum allowable elevation above endpoints in eV.
    :type tol_ev: float
    :return: True if band has no interior maximum above endpoints + tol_ev.
    :rtype: bool
    """
    if len(energies) < 3:
        return False
    max_endpoints = max(energies[0], energies[-1])
    max_interior = max(energies[1:-1])
    return bool(max_interior <= max_endpoints + tol_ev)


def endpoints_match_expected(state: SearchState) -> bool:
    """
    Verify both relaxed endpoints match the expected canonical state keys.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :return: True if both relaxed endpoints match expected keys.
    :rtype: bool
    """
    if state.relaxed_reactant is None or state.relaxed_product is None:
        return False
    keys = (
        state_key(state.relaxed_reactant, stereo=True),
        state_key(state.relaxed_product, stereo=True),
    )
    return bool(keys == state.expected_keys)


def check_neb_band_status(
    state: SearchState,
    energies: Sequence[float],
) -> TSStatus | None:
    """
    Assess NEB band energies for numerical divergence or barrierless pathway.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :param energies: Sequence of image energies along NEB band.
    :type energies: Sequence[float]
    :return: Failure status if NEB fails or barrierless, None to continue.
    :rtype: TSStatus or None
    """
    if not all(np.isfinite(val) for val in energies):
        return TSStatus.neb_failed
    tol = state.search_settings.barrierless_tol_ev
    if is_barrierless_band(energies, tol) and endpoints_match_expected(state):
        return TSStatus.barrierless
    return None


def neb_step(state: SearchState) -> TSStatus | None:
    """
    Generate transition state guess using climbing-image NEB with FIRE.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :return: Failure status if NEB fails, None to continue.
    :rtype: TSStatus or None
    """
    assert state.relaxed_reactant and state.relaxed_product
    cfg = state.search_settings
    image, conv, steps, ens = neb_guess(
        state.relaxed_reactant,
        state.relaxed_product,
        state.engine_settings,
        cfg,
    )
    state.force_calls += steps * (cfg.neb_images - 2)
    state.neb_converged, state.ts_guess = conv, image
    return check_neb_band_status(state, ens)


def saddle_step(state: SearchState) -> TSStatus | None:
    """
    Refine transition state guess to first-order saddle point using Sella.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :return: Failure status if saddle refinement fails, None to continue.
    :rtype: TSStatus or None
    """
    assert state.ts_guess is not None
    cfg = state.search_settings
    params = (cfg.saddle_fmax_ev_per_ang, cfg.saddle_max_steps)
    try:
        refined, conv, steps, energy = refine_saddle(
            state.ts_guess, state.engine_settings, *params
        )
    except CalculationFailed:
        raise
    except RuntimeError as err:
        # Sella gives up when its internal-coordinate step will not solve
        # ("Geometry update ODE is taking too long to converge!").
        state.note = str(err)
        return TSStatus.saddle_not_converged
    state.force_calls += steps
    if not np.isfinite(energy):
        return TSStatus.engine_error
    state.ts_atoms, state.ts_energy_ev = refined, energy
    if not conv:
        return TSStatus.saddle_not_converged
    return None


def frequency_step(state: SearchState) -> TSStatus | None:
    """
    Verify transition state normal modes have one imaginary frequency.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :return: Failure status if frequency check fails, None to continue.
    :rtype: TSStatus or None
    """
    assert state.ts_atoms is not None
    state.force_calls += 6 * len(state.ts_atoms)
    valid, imag_val, mode, vib_cm = check_imaginary(
        state.ts_atoms,
        state.engine_settings,
        state.search_settings.imaginary_threshold_cm,
    )
    state.imaginary_cm = imag_val
    state.mode = mode
    state.vibrational_cm = vib_cm
    if not valid or mode is None:
        return TSStatus.wrong_imaginary_count
    return None


def confirm_step(state: SearchState) -> TSStatus | None:
    """
    Verify transition state connects expected reactant and product endpoints.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :return: Termination status (confirmed or endpoint_mismatch).
    :rtype: TSStatus or None
    """
    assert state.ts_atoms is not None and state.mode is not None
    cfg = state.search_settings
    params = (cfg.confirm_step_ang, cfg.min_fmax_ev_per_ang, cfg.min_max_steps)
    confirmed, steps = confirm_endpoints(
        state.ts_atoms,
        state.mode,
        state.expected_keys,
        state.engine_settings,
        *params,
    )
    state.force_calls += steps
    if not confirmed:
        return TSStatus.endpoint_mismatch
    return TSStatus.confirmed


def seed_step(state: SearchState) -> TSStatus | None:
    """
    Take the seed TS as the saddle guess, in place of a band.

    :param state: Active search state carrying a seed geometry.
    :type state: SearchState
    :return: None, so the search continues to the saddle step.
    :rtype: TSStatus or None
    """
    assert state.seed is not None
    state.ts_guess = state.seed.copy()
    return None


SEARCH_STEPS: tuple[Callable[[SearchState], TSStatus | None], ...] = (
    relax_endpoints_step,
    neb_step,
    saddle_step,
    frequency_step,
    confirm_step,
)

# Same checks as SEARCH_STEPS; only the guess differs. The endpoints are
# still relaxed, because the confirm step compares against them.
SEED_STEPS: tuple[Callable[[SearchState], TSStatus | None], ...] = (
    relax_endpoints_step,
    seed_step,
    saddle_step,
    frequency_step,
    confirm_step,
)


def result_from_state(state: SearchState, status: TSStatus) -> TSResult:
    """
    Build a transition state search result from search state.

    :param state: Accumulated search state dataclass.
    :type state: SearchState
    :param status: Termination status of search.
    :type status: TSStatus
    :return: Transition state search result.
    :rtype: TSResult
    """
    return TSResult(
        status=status,
        pair_index=state.pair_index,
        ts_atoms=state.ts_atoms,
        ts_energy_ev=state.ts_energy_ev,
        reactant_energy_ev=state.reactant_energy_ev,
        product_energy_ev=state.product_energy_ev,
        imaginary_cm=state.imaginary_cm,
        vibrational_cm=state.vibrational_cm,
        neb_converged=state.neb_converged,
        force_calls=state.force_calls,
        note=state.note,
        route=TSRoute.band if state.seed is None else TSRoute.seed,
    )


def search_pair(
    pair: EndpointPair,
    pair_index: int,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
    seed: Atoms | None,
) -> TSResult:
    """
    Execute full transition state search pipeline on an endpoint pair.

    :param pair: Candidate reaction endpoint pair geometry.
    :type pair: EndpointPair
    :param pair_index: Index of endpoint pair within candidate reaction.
    :type pair_index: int
    :param engine_settings: Computational chemistry engine settings.
    :type engine_settings: EngineSettings
    :param search_settings: Search and convergence settings dataclass.
    :type search_settings: SearchSettings
    :param seed: Seed TS geometry to start from, or None to run a band.
    :type seed: Atoms or None
    :return: Transition state search result dataclass.
    :rtype: TSResult
    """
    state = SearchState(pair, pair_index, engine_settings, search_settings)
    state.seed = seed
    steps = SEARCH_STEPS if seed is None else SEED_STEPS
    limit = engine_settings.extra.get("attempt_time_limit_s")
    try:
        with attempt_time_limit(limit):
            status = run_search_steps(state, steps)
    except UnsupportedSpecies:
        status = TSStatus.unsupported
    except CalculationFailed as err:
        # Keep the engine's message (e.g. SCF not converged) in the record.
        status, state.note = TSStatus.engine_error, str(err)
    except AttemptTimeLimit:
        status, state.note = TSStatus.time_limit, f"attempt over {limit} s"
    return result_from_state(state, status)


def run_search_steps(
    state: SearchState,
    steps: Sequence[Callable[[SearchState], TSStatus | None]],
) -> TSStatus:
    """
    Run search stages in order; the first status a stage returns ends it.

    :param state: Active transition state search state dataclass.
    :type state: SearchState
    :param steps: Stages to run, SEARCH_STEPS or SEED_STEPS.
    :type steps: Sequence[Callable[[SearchState], TSStatus | None]]
    :return: The ending status, confirmed when every stage passes.
    :rtype: TSStatus
    """
    for step in steps:
        res = step(state)
        if res is not None:
            return res
    return TSStatus.confirmed


def search_candidate(
    candidate: Candidate,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
    seed: TSSeed | None,
) -> list[TSResult]:
    """
    Search transition states on every attempt of a candidate reaction.

    With a seed, the seeded pair is tried first, then the engine's own band
    on every pair. All attempts run because pairs can confirm different
    saddles between the same ends; refine keeps the lowest, and prefers
    any confirmed saddle to a barrierless band.

    :param candidate: Candidate reaction containing ordered endpoint pairs.
    :type candidate: Candidate
    :param engine_settings: Computational chemistry engine settings.
    :type engine_settings: EngineSettings
    :param search_settings: Search and convergence settings dataclass.
    :type search_settings: SearchSettings
    :param seed: Another engine's confirmed TS, or None.
    :type seed: TSSeed or None
    :return: TS results of every attempt.
    :rtype: list[TSResult]
    """
    attempts = [] if seed is None else [(seed.pair_index, seed.atoms)]
    attempts += [(idx, None) for idx in range(len(candidate.pairs))]
    results: list[TSResult] = []
    for pair_idx, seed_atoms in attempts:
        pair = candidate.pairs[pair_idx]
        res = search_pair(
            pair, pair_idx, engine_settings, search_settings, seed_atoms
        )
        results.append(res)
    return results
