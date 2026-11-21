"""
Stage 4 reaction network refinement with computational chemistry engines.

Optimizes frozen species geometries, verifies vibrational minima, computes
ideal gas thermochemistry, searches candidate reaction transition states with
climbing-image NEB and Sella, deduplicates identical reaction signatures, and
logs engine performance counters.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.io
import yaml
from ase import Atoms

from nrnet.chem import multiplicity, state_key
from nrnet.engines import (
    EngineSettings,
    UnsupportedSpecies,
    attach,
    load_engine_settings,
)
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import Candidate, CandidateSet, Species
from nrnet.optimize import RelaxResult, relax
from nrnet.records import (
    CallRecord,
    RunContext,
    config_hash,
    git_commit,
    record_call,
)
from nrnet.store import (
    RefinePaths,
    append_extxyz,
    append_jsonl,
    latest_rows,
    make_refine_paths,
    read_jsonl,
    temperature_label,
)
from nrnet.thermo import (
    ModeSplit,
    free_energy_ev,
    quasi_harmonic,
    split_modes,
    vibrate,
)
from nrnet.ts_search import (
    SearchSettings,
    TSResult,
    TSRoute,
    TSSeed,
    TSStatus,
    load_search_settings,
    search_candidate,
)

__all__ = [
    "DEFAULT_CANDIDATES_DIR",
    "DEFAULT_ENGINES_CONFIG",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_SYSTEMS_DIR",
    "STATUS_ENDPOINT_CHANGED",
    "STATUS_IMAGINARY_MINIMUM",
    "STATUS_OK",
    "STATUS_UNSUPPORTED",
    "RefineJob",
    "attach_species",
    "build_reaction_row",
    "build_refine_context",
    "build_refine_job",
    "build_seed_map",
    "build_species_ok_row",
    "calculate_species_g",
    "calculate_ts_free_energies",
    "candidate_signature",
    "check_relaxed_state",
    "confirmed_free_energy",
    "evaluate_species_modes",
    "execute_reaction_search",
    "execute_species_refine",
    "kept_result",
    "load_job_context",
    "load_job_settings",
    "load_ranking_temperatures",
    "load_refine_runtime",
    "load_seeds",
    "load_thermo_temperatures",
    "process_species_modes",
    "reaction_energy_fields",
    "reaction_meta_fields",
    "read_reaction_records",
    "read_seed_frames",
    "read_species_keys",
    "record_reaction_outcome",
    "record_species_imaginary",
    "record_species_ok",
    "refine",
    "refine_candidate_reaction",
    "refine_reaction_stage",
    "refine_single_species",
    "refine_species_stage",
    "relax_and_process_species",
    "reuse_confirmed_reaction",
    "run_refine_stages",
    "select_candidates",
    "update_confirmed_ts",
]

DEFAULT_RESULTS_DIR: str = "results"
DEFAULT_SEARCH_CONFIG: str = "configs/search.yaml"
DEFAULT_ENGINES_CONFIG: str = "configs/engines.yaml"
DEFAULT_SYSTEMS_DIR: str = "systems"
DEFAULT_CANDIDATES_DIR: str = "candidates"

STATUS_OK: str = "ok"
STATUS_UNSUPPORTED: str = "unsupported"
STATUS_ENDPOINT_CHANGED: str = "endpoint_changed"
STATUS_IMAGINARY_MINIMUM: str = "imaginary_minimum"


@dataclass(frozen=True, slots=True)
class RefineJob:
    """
    Execution parameters and input state for network refinement stages.

    :param paths: Configured refinement filesystem destination paths.
    :type paths: RefinePaths
    :param candidate_set: Frozen species and candidate reactions dataset.
    :type candidate_set: CandidateSet
    :param engine_settings: Computational chemistry engine settings.
    :type engine_settings: EngineSettings
    :param search_settings: Transition state search configuration.
    :type search_settings: SearchSettings
    :param temperatures_k: Evaluation temperatures in Kelvin.
    :type temperatures_k: tuple[float, ...]
    :param context: Invariant execution context for call records.
    :type context: RunContext
    :param seeds: Transition state guesses keyed by reaction ID.
    :type seeds: dict[str, TSSeed]
    :param retry: Status strings to re-search instead of skipping.
    :type retry: tuple[str, ...]
    :param reaction_ids: Optional subset of reaction IDs to refine.
    :type reaction_ids: tuple[str, ...] or None
    :param skip_species: Whether to skip refining frozen species minimums.
    :type skip_species: bool
    """

    paths: RefinePaths
    candidate_set: CandidateSet
    engine_settings: EngineSettings
    search_settings: SearchSettings
    temperatures_k: tuple[float, ...]
    context: RunContext
    seeds: dict[str, TSSeed]
    retry: tuple[str, ...] = ()
    reaction_ids: tuple[str, ...] | None = None
    skip_species: bool = False


def load_thermo_temperatures(path: str | Path) -> list[float]:
    """
    Load ranking temperatures from system YAML configuration.

    :param path: Path to system YAML configuration file.
    :type path: str or Path
    :return: List of evaluation temperatures in Kelvin.
    :rtype: list[float]
    :raises ValueError: If ranking.temperatures_k key is missing.
    """
    file_path = Path(path)
    if file_path.is_dir():
        file_path = file_path / "system.yaml"
    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if (
        not isinstance(data, dict)
        or "ranking" not in data
        or not isinstance(data["ranking"], dict)
        or "temperatures_k" not in data["ranking"]
        or data["ranking"]["temperatures_k"] is None
    ):
        raise ValueError(f"Missing ranking.temperatures_k in {file_path}")
    return [float(val) for val in data["ranking"]["temperatures_k"]]


load_ranking_temperatures = load_thermo_temperatures


def build_refine_context(
    paths: RefinePaths,
    system: str,
    engine: str,
    engine_settings: EngineSettings,
    system_yaml: Path,
    engines_yaml: Path,
    search_yaml: Path,
) -> RunContext:
    """
    Assemble invariant execution context for engine call recording.

    :param paths: Configured refinement filesystem paths.
    :type paths: RefinePaths
    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param engine_settings: Computational chemistry engine configuration.
    :type engine_settings: EngineSettings
    :param system_yaml: Path to system configuration file.
    :type system_yaml: Path
    :param engines_yaml: Path to engine configuration file.
    :type engines_yaml: Path
    :param search_yaml: Path to search configuration file.
    :type search_yaml: Path
    :return: Configured RunContext instance.
    :rtype: RunContext
    """
    return RunContext(
        log_path=paths.calls_csv,
        engine=engine,
        system=system,
        device=engine_settings.device,
        config_hash=config_hash(engines_yaml, search_yaml, system_yaml),
        git_commit=git_commit(),
    )


def read_species_keys(path: Path) -> set[str]:
    """
    Read completed species canonical keys from existing JSONL file.

    :param path: Path to species JSONL file.
    :type path: Path
    :return: Set of canonical species keys previously recorded.
    :rtype: set[str]
    """
    return {str(row["key"]) for row in read_jsonl(path)}


def read_reaction_records(
    path: Path,
    retry: Sequence[str] = (),
) -> tuple[
    set[str], dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]]
]:
    """
    Read completed reaction IDs and confirmed signatures from JSONL file.

    :param path: Path to reactions JSONL file.
    :type path: Path
    :param retry: Status strings to re-search instead of skipping.
    :type retry: Sequence[str]
    :return: Tuple of completed reaction IDs and confirmed rows by signature.
    :rtype: tuple[set[str], dict]
    """
    retry_set = set(retry)
    done_ids: set[str] = set()
    confirmed: dict[
        tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]
    ] = {}
    for rxn_id, row in latest_rows(path, "reaction_id").items():
        status = str(row.get("status", ""))
        if status in retry_set:
            continue
        done_ids.add(rxn_id)
        if status == TSStatus.confirmed.value and "signature" in row:
            sig_raw = row["signature"]
            sig = tuple(sorted([tuple(sig_raw[0]), tuple(sig_raw[1])]))
            confirmed[sig] = row
    return done_ids, confirmed


def candidate_signature(
    candidate: Candidate,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Compute canonical unordered reactant-product state signature.

    :param candidate: Candidate reaction to inspect.
    :type candidate: Candidate
    :return: Sorted pair of sorted reactant and product species keys.
    :rtype: tuple[tuple[str, ...], tuple[str, ...]]
    """
    r_keys = tuple(sorted(candidate.reactant_keys))
    p_keys = tuple(sorted(candidate.product_keys))
    return tuple(sorted([r_keys, p_keys]))


def evaluate_species_modes(
    atoms: Atoms,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
    steps: int,
) -> tuple[ModeSplit, Atoms, int, bool]:
    """
    Vibrate relaxed species and verify absence of imaginary frequencies.

    :param atoms: Relaxed atomic configuration without calculator.
    :type atoms: Atoms
    :param engine_settings: Computational chemistry engine settings.
    :type engine_settings: EngineSettings
    :param search_settings: Transition state and threshold settings.
    :type search_settings: SearchSettings
    :param steps: Optimization steps taken during geometry relaxation.
    :type steps: int
    :return: Mode split, attached atoms, total force calls, and imag flag.
    :rtype: tuple[ModeSplit, Atoms, int, bool]
    """
    attached = attach(atoms, engine_settings)
    vib_res = vibrate(attached)
    split = split_modes(vib_res, transition_state=False)
    calls = steps + 6 * len(atoms)
    imag = any(
        m < search_settings.imaginary_threshold_cm for m in split.vibrational_cm
    )
    return split, attached, calls, imag


def calculate_species_g(
    atoms: Atoms,
    energy_ev: float,
    vibrational_cm: Sequence[float],
    temperatures_k: Sequence[float],
    mult: int,
) -> dict[str, float]:
    """
    Compute standard Gibbs free energy across evaluation temperatures.

    :param atoms: Molecular configuration without calculator.
    :type atoms: Atoms
    :param energy_ev: Electronic potential energy in eV.
    :type energy_ev: float
    :param vibrational_cm: Vibrational normal mode frequencies in cm^-1.
    :type vibrational_cm: Sequence[float]
    :param temperatures_k: Sequence of evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :param mult: Spin multiplicity (2S + 1).
    :type mult: int
    :return: Map of temperature strings to Gibbs free energy in eV.
    :rtype: dict[str, float]
    """
    qh_modes = quasi_harmonic(vibrational_cm)
    return {
        temperature_label(t): free_energy_ev(
            atoms, energy_ev, qh_modes, t, mult
        )
        for t in temperatures_k
    }


def build_species_ok_row(
    key: str,
    energy_ev: float,
    g_ev: dict[str, float],
    mult: int,
    split: ModeSplit,
    calls: int,
) -> dict[str, Any]:
    """
    Build result dictionary for confirmed species minimum.

    :param key: Canonical species key.
    :type key: str
    :param energy_ev: Electronic potential energy in eV.
    :type energy_ev: float
    :param g_ev: Gibbs free energy values mapped by temperature string.
    :type g_ev: dict[str, float]
    :param mult: Spin multiplicity (2S + 1).
    :type mult: int
    :param split: Partitioned vibrational modes.
    :type split: ModeSplit
    :param calls: Total force call count.
    :type calls: int
    :return: Formatted species dictionary.
    :rtype: dict[str, Any]
    """
    lowest = (
        float(min(split.vibrational_cm))
        if len(split.vibrational_cm) > 0
        else 0.0
    )
    return {
        "key": key,
        "status": STATUS_OK,
        "energy_ev": energy_ev,
        "g_ev": g_ev,
        "multiplicity": mult,
        "lowest_cm": lowest,
        "force_calls": calls,
    }


def record_species_ok(
    atoms: Atoms,
    energy_ev: float,
    split: ModeSplit,
    calls: int,
    key: str,
    temperatures_k: Sequence[float],
    paths: RefinePaths,
) -> None:
    """
    Record confirmed species minimum data and save relaxed geometry.

    :param atoms: Relaxed molecular configuration.
    :type atoms: Atoms
    :param energy_ev: Electronic potential energy in eV.
    :type energy_ev: float
    :param split: Partitioned vibrational normal modes.
    :type split: ModeSplit
    :param calls: Combined geometry optimization and frequency force calls.
    :type calls: int
    :param key: Canonical species key.
    :type key: str
    :param temperatures_k: Evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :param paths: Refinement destination paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    mult = multiplicity(atoms)
    g_ev = calculate_species_g(
        atoms, energy_ev, split.vibrational_cm, temperatures_k, mult
    )
    row = build_species_ok_row(key, energy_ev, g_ev, mult, split, calls)
    append_jsonl(paths.species_jsonl, row)
    frame = atoms.copy()
    frame.info["key"] = key
    append_extxyz(paths.species_extxyz, frame)


def check_relaxed_state(
    res: RelaxResult,
    key: str,
    rec: CallRecord,
    paths: RefinePaths,
) -> bool:
    """
    Verify canonical state key of relaxed structure matches expected key.

    :param res: Optimization outcome dataclass.
    :type res: RelaxResult
    :param key: Expected species canonical key.
    :type key: str
    :param rec: Mutable engine call record.
    :type rec: CallRecord
    :param paths: Refinement destination paths.
    :type paths: RefinePaths
    :return: True if state key matches, False if endpoint changed.
    :rtype: bool
    """
    if state_key(res.atoms, stereo=True) != (key,):
        rec.status = STATUS_ENDPOINT_CHANGED
        rec.force_calls = res.steps
        row = {
            "key": key,
            "status": STATUS_ENDPOINT_CHANGED,
            "force_calls": res.steps,
        }
        append_jsonl(paths.species_jsonl, row)
        return False
    return True


def record_species_imaginary(
    key: str,
    energy_ev: float,
    split: ModeSplit,
    calls: int,
    paths: RefinePaths,
) -> None:
    """
    Persist species record when relaxation yielded an imaginary frequency mode.

    :param key: Canonical species key.
    :type key: str
    :param energy_ev: Electronic potential energy in eV.
    :type energy_ev: float
    :param split: Partitioned vibrational normal modes.
    :type split: ModeSplit
    :param calls: Combined force call count.
    :type calls: int
    :param paths: Refinement destination paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    lowest = (
        float(min(split.vibrational_cm))
        if len(split.vibrational_cm) > 0
        else 0.0
    )
    row = {
        "key": key,
        "status": STATUS_IMAGINARY_MINIMUM,
        "energy_ev": energy_ev,
        "lowest_cm": lowest,
        "force_calls": calls,
    }
    append_jsonl(paths.species_jsonl, row)


def process_species_modes(
    atoms: Atoms,
    key: str,
    steps: int,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
    temperatures_k: Sequence[float],
    rec: CallRecord,
    paths: RefinePaths,
) -> None:
    """
    Vibrate species and route outcome to imaginary or confirmed recorder.

    :param atoms: Relaxed atomic structure.
    :type atoms: Atoms
    :param key: Canonical species key.
    :type key: str
    :param steps: Geometry optimization steps taken.
    :type steps: int
    :param engine_settings: Engine configuration settings.
    :type engine_settings: EngineSettings
    :param search_settings: Search settings with thresholds.
    :type search_settings: SearchSettings
    :param temperatures_k: Evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :param rec: Mutable call record.
    :type rec: CallRecord
    :param paths: Refinement destination paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    split, rel_att, calls, imag = evaluate_species_modes(
        atoms, engine_settings, search_settings, steps
    )
    rec.force_calls = calls
    energy = float(rel_att.get_potential_energy())
    if imag:
        rec.status = STATUS_IMAGINARY_MINIMUM
        record_species_imaginary(key, energy, split, calls, paths)
        return
    rec.status = STATUS_OK
    record_species_ok(atoms, energy, split, calls, key, temperatures_k, paths)


def attach_species(
    spec: Species,
    engine_settings: EngineSettings,
    rec: CallRecord,
    paths: RefinePaths,
) -> Atoms | None:
    """
    Attach engine calculator to species structure with exception trapping.

    :param spec: Candidate species configuration.
    :type spec: Species
    :param engine_settings: Computational engine settings.
    :type engine_settings: EngineSettings
    :param rec: Mutable engine call record.
    :type rec: CallRecord
    :param paths: Refinement destination file paths.
    :type paths: RefinePaths
    :return: Attached Atoms copy, or None if species unsupported.
    :rtype: Atoms or None
    """
    try:
        return attach(spec.atoms, engine_settings)
    except UnsupportedSpecies:
        rec.status = STATUS_UNSUPPORTED
        row = {"key": spec.key, "status": STATUS_UNSUPPORTED}
        append_jsonl(paths.species_jsonl, row)
        return None


def relax_and_process_species(
    atoms: Atoms,
    spec_key: str,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
    temperatures_k: Sequence[float],
    rec: CallRecord,
    paths: RefinePaths,
) -> None:
    """
    Relax species structure and process normal modes if key is preserved.

    :param atoms: Input atomic structure with attached calculator.
    :type atoms: Atoms
    :param spec_key: Expected species key.
    :type spec_key: str
    :param engine_settings: Computational engine settings.
    :type engine_settings: EngineSettings
    :param search_settings: Geometry search settings.
    :type search_settings: SearchSettings
    :param temperatures_k: Evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :param rec: Mutable call record.
    :type rec: CallRecord
    :param paths: Refinement destination file paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    cfg = search_settings
    res = relax(atoms, cfg.min_fmax_ev_per_ang, cfg.min_max_steps)
    if check_relaxed_state(res, spec_key, rec, paths):
        process_species_modes(
            res.atoms,
            spec_key,
            res.steps,
            engine_settings,
            search_settings,
            temperatures_k,
            rec,
            paths,
        )


def execute_species_refine(
    spec: Species,
    engine_settings: EngineSettings,
    search_settings: SearchSettings,
    temperatures_k: Sequence[float],
    rec: CallRecord,
    paths: RefinePaths,
) -> None:
    """
    Run relaxation, mode analysis, and thermochemistry under call record.

    :param spec: Candidate species configuration.
    :type spec: Species
    :param engine_settings: Computational engine settings.
    :type engine_settings: EngineSettings
    :param search_settings: Geometry search settings.
    :type search_settings: SearchSettings
    :param temperatures_k: Evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :param rec: Mutable call record.
    :type rec: CallRecord
    :param paths: Refinement destination file paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    attached = attach_species(spec, engine_settings, rec, paths)
    if attached is not None:
        relax_and_process_species(
            attached,
            spec.key,
            engine_settings,
            search_settings,
            temperatures_k,
            rec,
            paths,
        )


def refine_single_species(
    spec: Species,
    job: RefineJob,
) -> None:
    """
    Optimize and evaluate thermochemistry for a single chemical species.

    :param spec: Candidate species configuration to refine.
    :type spec: Species
    :param job: Refinement configuration and input job specification.
    :type job: RefineJob
    :return: None.
    :rtype: None
    """
    with record_call(job.context, "species", spec.key) as rec:
        execute_species_refine(
            spec,
            job.engine_settings,
            job.search_settings,
            job.temperatures_k,
            rec,
            job.paths,
        )


def calculate_ts_free_energies(
    ts_atoms: Atoms,
    energy_ev: float,
    vibrational_cm: Sequence[float],
    temperatures_k: Sequence[float],
) -> dict[str, float]:
    """
    Compute transition state Gibbs free energies across temperatures.

    :param ts_atoms: Transition state configuration without calculator.
    :type ts_atoms: Atoms
    :param energy_ev: Potential energy in eV.
    :type energy_ev: float
    :param vibrational_cm: Real vibrational modes in cm^-1.
    :type vibrational_cm: Sequence[float]
    :param temperatures_k: Sequence of evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :return: Map of temperature strings to Gibbs free energy in eV.
    :rtype: dict[str, float]
    """
    qh_vib = quasi_harmonic(vibrational_cm)
    mult = multiplicity(ts_atoms)
    return {
        temperature_label(t): free_energy_ev(
            ts_atoms, energy_ev, qh_vib, t, mult
        )
        for t in temperatures_k
    }


def confirmed_free_energy(res: TSResult, temperature_k: float) -> float:
    """
    Compute the Gibbs free energy of a confirmed saddle at one temperature.

    :param res: Confirmed TS result with geometry and frequencies.
    :type res: TSResult
    :param temperature_k: Evaluation temperature in Kelvin.
    :type temperature_k: float
    :return: Gibbs free energy in eV.
    :rtype: float
    """
    assert res.ts_atoms is not None and res.ts_energy_ev is not None
    vib = res.vibrational_cm or ()
    g_map = calculate_ts_free_energies(
        res.ts_atoms, res.ts_energy_ev, vib, [temperature_k]
    )
    return g_map[temperature_label(temperature_k)]


def kept_result(results: Sequence[TSResult], temperature_k: float) -> TSResult:
    """
    Choose the attempt a reaction record keeps.

    The confirmed saddle with the lowest G at the first ranking
    temperature wins, because pairs can confirm different saddles between
    the same ends. A confirmed saddle beats a barrierless band: the saddle
    passed the frequency and endpoint checks, while a band reads as
    barrierless whenever its own product end lies high. Without a saddle,
    a barrierless attempt wins; otherwise the last attempt.

    :param results: TS results of every attempt.
    :type results: Sequence[TSResult]
    :param temperature_k: First ranking temperature in Kelvin.
    :type temperature_k: float
    :return: The attempt to record.
    :rtype: TSResult
    """
    confirmed = [
        res
        for res in results
        if res.status == TSStatus.confirmed and res.ts_atoms is not None
    ]
    if not confirmed:
        statuses = [res.status for res in results]
        if TSStatus.barrierless in statuses:
            return results[statuses.index(TSStatus.barrierless)]
        return results[-1]
    return min(
        confirmed, key=lambda res: confirmed_free_energy(res, temperature_k)
    )


def reaction_energy_fields(last_res: TSResult) -> dict[str, Any]:
    """
    Extract electronic and complex energy fields from TSResult.

    :param last_res: Terminal TSResult from search pipeline.
    :type last_res: TSResult
    :return: Dictionary of energy values and search notes.
    :rtype: dict[str, Any]
    """
    return {
        "ts_energy_ev": last_res.ts_energy_ev,
        "g_ts_ev": None,
        "reactant_complex_ev": last_res.reactant_energy_ev,
        "product_complex_ev": last_res.product_energy_ev,
        "imaginary_cm": last_res.imaginary_cm,
        "note": last_res.note,
    }


def reaction_meta_fields(
    candidate: Candidate,
    last_res: TSResult,
    seed: TSSeed | None,
) -> dict[str, Any]:
    """
    Extract routing, signature, and identity metadata for reaction row.

    :param candidate: Evaluated candidate reaction.
    :type candidate: Candidate
    :param last_res: Terminal TSResult from search pipeline.
    :type last_res: TSResult
    :param seed: Transition state seed used for search, or None.
    :type seed: TSSeed or None
    :return: Dictionary of metadata fields.
    :rtype: dict[str, Any]
    """
    seed_engine = (
        seed.engine if seed and last_res.route == TSRoute.seed else None
    )
    sig = [list(candidate.reactant_keys), list(candidate.product_keys)]
    return {
        "reaction_id": candidate.reaction_id,
        "signature": sig,
        "pair_index": last_res.pair_index,
        "route": last_res.route.value,
        "seed_engine": seed_engine,
        "source": candidate.source,
        "literature_name": candidate.literature_name,
    }


def build_reaction_row(
    candidate: Candidate,
    last_res: TSResult,
    attempts: list[str],
    force_calls: int,
    seed: TSSeed | None,
) -> dict[str, Any]:
    """
    Build baseline reaction result record dictionary from search outcome.

    :param candidate: Evaluated candidate reaction.
    :type candidate: Candidate
    :param last_res: Terminal TSResult from search pipeline.
    :type last_res: TSResult
    :param attempts: Sequence of termination status strings per endpoint pair.
    :type attempts: list[str]
    :param force_calls: Cumulative force evaluations across all attempts.
    :type force_calls: int
    :param seed: Transition state seed used for search, or None.
    :type seed: TSSeed or None
    :return: Populated reaction record dictionary.
    :rtype: dict[str, Any]
    """
    row = {
        "status": last_res.status.value,
        "attempts": attempts,
        "force_calls": force_calls,
        "reused_from": None,
    }
    row.update(reaction_meta_fields(candidate, last_res, seed))
    row.update(reaction_energy_fields(last_res))
    return row


def update_confirmed_ts(
    row: dict[str, Any],
    candidate: Candidate,
    last_res: TSResult,
    temperatures_k: Sequence[float],
    ts_extxyz: Path,
) -> None:
    """
    Evaluate thermochemistry and serialize transition state frame if confirmed.

    :param row: Mutable reaction record dictionary to update.
    :type row: dict[str, Any]
    :param candidate: Evaluated candidate reaction.
    :type candidate: Candidate
    :param last_res: Terminal TSResult from search pipeline.
    :type last_res: TSResult
    :param temperatures_k: Evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :param ts_extxyz: Path to transition states extxyz file.
    :type ts_extxyz: Path
    :return: None.
    :rtype: None
    """
    if last_res.status != TSStatus.confirmed or last_res.ts_atoms is None:
        return
    vib = last_res.vibrational_cm or ()
    nrg = last_res.ts_energy_ev or 0.0
    row["g_ts_ev"] = calculate_ts_free_energies(
        last_res.ts_atoms, nrg, vib, temperatures_k
    )
    frame = last_res.ts_atoms.copy()
    frame.info["reaction_id"] = candidate.reaction_id
    frame.info["pair_index"] = last_res.pair_index
    append_extxyz(ts_extxyz, frame)


def record_reaction_outcome(
    candidate: Candidate,
    results: list[TSResult],
    temperatures_k: Sequence[float],
    paths: RefinePaths,
    seed: TSSeed | None,
) -> dict[str, Any]:
    """
    Assemble and persist reaction transition state search outcome.

    :param candidate: Evaluated candidate reaction.
    :type candidate: Candidate
    :param results: Sequence of TS results across attempted pairs.
    :type results: list[TSResult]
    :param temperatures_k: Evaluation temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :param paths: Refinement destination file paths.
    :type paths: RefinePaths
    :param seed: Transition state seed used for search, or None.
    :type seed: TSSeed or None
    :return: Serialized reaction record dictionary.
    :rtype: dict[str, Any]
    """
    kept = kept_result(results, temperatures_k[0])
    attempts = [r.status.value for r in results]
    force_calls = sum(r.force_calls for r in results)
    row = build_reaction_row(candidate, kept, attempts, force_calls, seed)
    row["attempt_ts_energy_ev"] = [
        r.ts_energy_ev if r.status == TSStatus.confirmed else None
        for r in results
    ]
    update_confirmed_ts(row, candidate, kept, temperatures_k, paths.ts_extxyz)
    append_jsonl(paths.reactions_jsonl, row)
    return row


def execute_reaction_search(
    candidate: Candidate,
    job: RefineJob,
) -> dict[str, Any]:
    """
    Execute transition state search for candidate under record_call context.

    :param candidate: Candidate reaction to search.
    :type candidate: Candidate
    :param job: Refinement configuration and input job specification.
    :type job: RefineJob
    :return: Serialized reaction record dictionary.
    :rtype: dict[str, Any]
    """
    seed = job.seeds.get(candidate.reaction_id)
    with record_call(job.context, "reaction", candidate.reaction_id) as rec:
        results = search_candidate(
            candidate, job.engine_settings, job.search_settings, seed
        )
        rec.force_calls = sum(r.force_calls for r in results)
        row = record_reaction_outcome(
            candidate, results, job.temperatures_k, job.paths, seed
        )
        rec.status, rec.note = row["status"], row["note"]
        return row


def reuse_confirmed_reaction(
    candidate: Candidate,
    earlier_row: Mapping[str, Any],
    reactions_path: Path,
) -> dict[str, Any]:
    """
    Persist reused reaction record copied from earlier confirmed result.

    :param candidate: Candidate reaction matching earlier signature.
    :type candidate: Candidate
    :param earlier_row: Existing confirmed reaction record.
    :type earlier_row: Mapping[str, Any]
    :param reactions_path: Path to reactions JSONL file.
    :type reactions_path: Path
    :return: Reused reaction record dictionary.
    :rtype: dict[str, Any]
    """
    reused = dict(earlier_row)
    reused["reaction_id"] = candidate.reaction_id
    reused["reused_from"] = earlier_row["reaction_id"]
    reused["source"] = candidate.source
    reused["literature_name"] = candidate.literature_name
    append_jsonl(reactions_path, reused)
    return reused


def refine_candidate_reaction(
    candidate: Candidate,
    confirmed_by_sig: dict[
        tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]
    ],
    job: RefineJob,
) -> None:
    """
    Refine candidate reaction or reuse confirmed match without engine call.

    :param candidate: Candidate reaction to evaluate.
    :type candidate: Candidate
    :param confirmed_by_sig: Map of confirmed reaction records by signature.
    :type confirmed_by_sig: dict
    :param job: Refinement configuration and input job specification.
    :type job: RefineJob
    :return: None.
    :rtype: None
    """
    sig = candidate_signature(candidate)
    if sig in confirmed_by_sig:
        reuse_confirmed_reaction(
            candidate, confirmed_by_sig[sig], job.paths.reactions_jsonl
        )
        return
    row = execute_reaction_search(candidate, job)
    if row["status"] == TSStatus.confirmed.value:
        confirmed_by_sig[sig] = row


def refine_species_stage(
    species_map: Mapping[str, Species],
    existing_keys: set[str],
    job: RefineJob,
) -> None:
    """
    Execute Stage 4 species optimization and thermochemistry pipeline.

    :param species_map: Mapping of canonical species keys to Species.
    :type species_map: Mapping[str, Species]
    :param existing_keys: Set of species keys already processed on disk.
    :type existing_keys: set[str]
    :param job: Refinement configuration and input job specification.
    :type job: RefineJob
    :return: None.
    :rtype: None
    """
    for key, spec in species_map.items():
        if key in existing_keys:
            continue
        refine_single_species(spec, job)


def refine_reaction_stage(
    candidates: Sequence[Candidate],
    existing_ids: set[str],
    confirmed_by_sig: dict[
        tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]
    ],
    job: RefineJob,
) -> None:
    """
    Execute Stage 4 reaction transition state search and deduplication.

    :param candidates: Sequence of frozen candidate reactions.
    :type candidates: Sequence[Candidate]
    :param existing_ids: Set of reaction IDs already processed on disk.
    :type existing_ids: set[str]
    :param confirmed_by_sig: Map of confirmed reaction records by signature.
    :type confirmed_by_sig: dict
    :param job: Refinement configuration and input job specification.
    :type job: RefineJob
    :return: None.
    :rtype: None
    """
    for cand in candidates:
        if cand.reaction_id in existing_ids:
            continue
        refine_candidate_reaction(cand, confirmed_by_sig, job)


def read_seed_frames(path: Path) -> dict[str, Atoms]:
    """
    Read transition state frames from extxyz file keyed by reaction ID.

    :param path: Path to transition states extxyz file.
    :type path: Path
    :return: Mapping of reaction ID to atomic configuration frame.
    :rtype: dict[str, Atoms]
    """
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    frames: list[Atoms] = ase.io.read(path, index=":", format="extxyz")
    return {
        str(f.info["reaction_id"]): f for f in frames if "reaction_id" in f.info
    }


def build_seed_map(
    path: Path,
    frames: Mapping[str, Atoms],
    engine: str,
) -> dict[str, TSSeed]:
    """
    Construct TSSeed mapping from confirmed non-reused reactions in JSONL.

    :param path: Path to seed reactions JSONL file.
    :type path: Path
    :param frames: Mapping of reaction ID to transition state frame.
    :type frames: Mapping[str, Atoms]
    :param engine: Seed chemistry engine identifier.
    :type engine: str
    :return: Mapping of reaction ID to TSSeed dataclass.
    :rtype: dict[str, TSSeed]
    """
    seeds: dict[str, TSSeed] = {}
    for rxn_id, row in latest_rows(path, "reaction_id").items():
        if row["status"] != TSStatus.confirmed.value:
            continue
        if row["reused_from"] is not None:
            continue
        # A searched, confirmed row always wrote its TS frame; a KeyError
        # here means reactions.jsonl and ts.extxyz are out of step.
        atoms = frames[rxn_id].copy()
        atoms.calc = None
        pair_idx = int(row["pair_index"])
        seeds[rxn_id] = TSSeed(pair_idx, atoms, engine)
    return seeds


def load_seeds(
    results_dir: str | Path,
    system: str,
    seed_engine: str | None,
) -> dict[str, TSSeed]:
    """
    Load confirmed transition state seeds from another engine's refine run.

    :param results_dir: Base directory for refinement results.
    :type results_dir: str or Path
    :param system: Chemical system identifier.
    :type system: str
    :param seed_engine: Seed engine identifier, or None for no seeding.
    :type seed_engine: str or None
    :return: Mapping of reaction ID to TSSeed dataclass instances.
    :rtype: dict[str, TSSeed]
    :raises FileNotFoundError: If seed engine reactions.jsonl does not exist.
    """
    if seed_engine is None:
        return {}
    paths = make_refine_paths(results_dir, system, seed_engine)
    if not paths.reactions_jsonl.is_file():
        raise FileNotFoundError(
            f"Seed reactions file not found: {paths.reactions_jsonl}"
        )
    frames = read_seed_frames(paths.ts_extxyz)
    return build_seed_map(paths.reactions_jsonl, frames, seed_engine)


def load_job_settings(
    engines_path: str | Path,
    search_path: str | Path,
    engine: str,
    system_yaml: str | Path | None = None,
) -> tuple[EngineSettings, SearchSettings, tuple[float, ...]]:
    """
    Load engine settings, search settings, and temperatures for a job.

    :param engines_path: Path to engines YAML configuration.
    :type engines_path: str or Path
    :param search_path: Path to search YAML configuration.
    :type search_path: str or Path
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param system_yaml: Path to system YAML configuration file.
    :type system_yaml: str or Path or None
    :return: Tuple of engine settings, search settings, and temperatures.
    :rtype: tuple[EngineSettings, SearchSettings, tuple[float, ...]]
    :raises ValueError: If system_yaml is not provided.
    """
    esettings = load_engine_settings(engines_path, engine)
    ssettings = load_search_settings(search_path)
    if system_yaml is None:
        raise ValueError("system_yaml is required to load ranking temperatures")
    temps = tuple(load_thermo_temperatures(system_yaml))
    return esettings, ssettings, temps


def load_job_context(
    paths: RefinePaths,
    system: str,
    engine: str,
    esettings: EngineSettings,
    systems_dir: str | Path,
    engines_path: str | Path,
    search_path: str | Path,
) -> RunContext:
    """
    Construct run context from configuration files and engine settings.

    :param paths: Refinement paths instance.
    :type paths: RefinePaths
    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param esettings: Computational chemistry engine settings.
    :type esettings: EngineSettings
    :param systems_dir: Systems configuration base directory.
    :type systems_dir: str or Path
    :param engines_path: Path to engines YAML configuration.
    :type engines_path: str or Path
    :param search_path: Path to search and thermo YAML configuration.
    :type search_path: str or Path
    :return: Configured RunContext instance.
    :rtype: RunContext
    """
    sys_yaml = Path(systems_dir) / system / "system.yaml"
    return build_refine_context(
        paths,
        system,
        engine,
        esettings,
        sys_yaml,
        Path(engines_path),
        Path(search_path),
    )


def load_refine_runtime(
    paths: RefinePaths,
    system: str,
    engine: str,
    systems_dir: str | Path,
    engines_path: str | Path,
    search_path: str | Path,
) -> tuple[EngineSettings, SearchSettings, tuple[float, ...], RunContext]:
    """
    Load settings, ranking temperatures, and runtime context for refinement.

    :param paths: Refinement output paths.
    :type paths: RefinePaths
    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param systems_dir: Systems configuration base directory.
    :type systems_dir: str or Path
    :param engines_path: Path to engines YAML configuration.
    :type engines_path: str or Path
    :param search_path: Path to search and thermo YAML configuration.
    :type search_path: str or Path
    :return: Engine settings, search settings, temperatures, and context.
    :rtype: tuple[EngineSettings, SearchSettings, tuple[float, ...], RunContext]
    """
    sys_yaml = Path(systems_dir) / system / "system.yaml"
    eset, sset, temps = load_job_settings(
        engines_path, search_path, engine, system_yaml=sys_yaml
    )
    ctx = load_job_context(
        paths,
        system,
        engine,
        eset,
        systems_dir,
        engines_path,
        search_path,
    )
    return eset, sset, temps, ctx


def select_candidates(
    candidates: Sequence[Candidate],
    reaction_ids: Sequence[str] | None,
) -> list[Candidate]:
    """
    Filter candidates by reaction identifier while preserving frozen order.

    :param candidates: Sequence of candidate reactions to filter.
    :type candidates: Sequence[Candidate]
    :param reaction_ids: Reaction identifiers to retain, or None for all.
    :type reaction_ids: Sequence[str] or None
    :return: Filtered list of candidate reactions in frozen order.
    :rtype: list[Candidate]
    :raises ValueError: If any requested reaction identifier is not found.
    """
    if reaction_ids is None:
        return list(candidates)
    available_ids = {cand.reaction_id for cand in candidates}
    missing_ids = [
        rxn_id for rxn_id in reaction_ids if rxn_id not in available_ids
    ]
    if missing_ids:
        missing_names = ", ".join(dict.fromkeys(missing_ids))
        raise ValueError(
            f"Reaction IDs not found in candidate set: {missing_names}"
        )
    target_ids = set(reaction_ids)
    return [cand for cand in candidates if cand.reaction_id in target_ids]


def build_refine_job(
    system: str,
    engine: str,
    systems_dir: str | Path,
    candidates_dir: str | Path,
    results_dir: str | Path,
    engines_path: str | Path,
    search_path: str | Path,
    seed_engine: str | None = None,
    retry: Sequence[str] = (),
    reactions: Sequence[str] | None = None,
    skip_species: bool = False,
) -> RefineJob:
    """
    Assemble complete refinement job including configuration and seeds.

    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param systems_dir: Systems configuration base directory.
    :type systems_dir: str or Path
    :param candidates_dir: Frozen candidates base directory.
    :type candidates_dir: str or Path
    :param results_dir: Base directory for refinement results.
    :type results_dir: str or Path
    :param engines_path: Path to engines YAML configuration.
    :type engines_path: str or Path
    :param search_path: Path to search and thermo YAML configuration.
    :type search_path: str or Path
    :param seed_engine: Identifier of engine used for TS seeds, or None.
    :type seed_engine: str or None
    :param retry: Status strings to re-search instead of skipping.
    :type retry: Sequence[str]
    :param reactions: Specific reaction IDs to refine, or None for all.
    :type reactions: Sequence[str] or None
    :param skip_species: Whether to skip refining frozen species minimums.
    :type skip_species: bool
    :return: Configured RefineJob instance.
    :rtype: RefineJob
    """
    paths = make_refine_paths(results_dir, system, engine)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    cset = load_candidates(Path(candidates_dir) / system)
    rxn_ids = tuple(reactions) if reactions is not None else None
    select_candidates(cset.candidates, rxn_ids)
    eset, sset, temps, ctx = load_refine_runtime(
        paths, system, engine, systems_dir, engines_path, search_path
    )
    seeds = load_seeds(results_dir, system, seed_engine)
    return RefineJob(
        paths,
        cset,
        eset,
        sset,
        temps,
        ctx,
        seeds,
        tuple(retry),
        rxn_ids,
        skip_species,
    )


def run_refine_stages(job: RefineJob) -> None:
    """
    Execute species and reaction refinement stages in sequence.

    :param job: Refinement configuration and input job specification.
    :type job: RefineJob
    :return: None.
    :rtype: None
    """
    if not job.skip_species:
        done_keys = read_species_keys(job.paths.species_jsonl)
        refine_species_stage(job.candidate_set.species, done_keys, job)
    done_ids, confirmed = read_reaction_records(
        job.paths.reactions_jsonl, job.retry
    )
    selected = select_candidates(
        job.candidate_set.candidates, job.reaction_ids
    )
    refine_reaction_stage(selected, done_ids, confirmed, job)


def refine(
    system: str,
    engine: str,
    systems_dir: str | Path = DEFAULT_SYSTEMS_DIR,
    candidates_dir: str | Path = DEFAULT_CANDIDATES_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    engines_path: str | Path = DEFAULT_ENGINES_CONFIG,
    search_path: str | Path = DEFAULT_SEARCH_CONFIG,
    seed_engine: str | None = None,
    retry: Sequence[str] = (),
    reactions: Sequence[str] | None = None,
    skip_species: bool = False,
) -> None:
    """
    Refine all frozen species and candidate reactions with an engine.

    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param systems_dir: Systems configuration base directory.
    :type systems_dir: str or Path
    :param candidates_dir: Frozen candidates base directory.
    :type candidates_dir: str or Path
    :param results_dir: Base directory for refinement results.
    :type results_dir: str or Path
    :param engines_path: Path to engines YAML configuration.
    :type engines_path: str or Path
    :param search_path: Path to search and thermo YAML configuration.
    :type search_path: str or Path
    :param seed_engine: Identifier of engine used for TS seeds, or None.
    :type seed_engine: str or None
    :param retry: Status strings to re-search instead of skipping.
    :type retry: Sequence[str]
    :param reactions: Specific reaction IDs to refine, or None for all.
    :type reactions: Sequence[str] or None
    :param skip_species: Whether to skip refining frozen species minimums.
    :type skip_species: bool
    :return: None.
    :rtype: None
    """
    job = build_refine_job(
        system,
        engine,
        systems_dir,
        candidates_dir,
        results_dir,
        engines_path,
        search_path,
        seed_engine,
        retry,
        reactions=reactions,
        skip_species=skip_species,
    )
    run_refine_stages(job)
