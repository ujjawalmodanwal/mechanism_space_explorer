"""
Parameter sweep runner and metrics logging for nanoreactor MD simulations.

Screens combinations of simulation temperatures, metadynamics push strengths,
and replicate groups, logs convergence metrics, and detects literature products.
"""

import csv
import itertools
import json
import math
import multiprocessing
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import ase.io
from ase import Atoms

from nrnet.evaluate import compute_jaccard
from nrnet.literature import (
    LiteratureProduct,
    candidate_set_holds_product,
    load_literature_products,
)
from nrnet.nanoreactor.candidate_store import freeze, load_candidates
from nrnet.nanoreactor.candidates import Candidate, CandidateSet
from nrnet.nanoreactor.md import (
    MDSettings,
    build_all_equilibrated_tasks,
    build_equilibration_tasks,
    build_optimization_tasks,
    build_replicate_tasks,
    execute_equilibration_task,
    execute_optimization_task,
    execute_replicate_task,
    find_replicate_trajectories,
    first_failed_frame,
    load_all_trajectories,
    load_md_settings,
    load_system_reactants,
    read_trajectory,
)

__all__ = [
    "SweepSetting",
    "assemble_metrics_dict",
    "build_base_metrics",
    "build_setting_tasks",
    "build_sweep_records",
    "build_sweep_row",
    "candidate_identity",
    "collect_sweep_replicate_tasks",
    "compute_group_overlap",
    "compute_replicate_fractions",
    "compute_setting_overlaps",
    "count_failed_replicates",
    "ensure_equilibration_trajectories",
    "ensure_optimized_starts",
    "ensure_temperature_equilibration",
    "equilibrate_all_temperatures",
    "execute_sweep_pool",
    "expand_sweep_grid",
    "format_kpush_label",
    "format_temperature_label",
    "freeze_all_groups",
    "freeze_sweep_group",
    "is_candidate_dir_ready",
    "is_group_complete",
    "load_sweep_system_context",
    "merge_sweep_csv_rows",
    "prepare_sweep_settings",
    "process_sweep_results",
    "read_existing_csv_rows",
    "run_sweep",
    "run_sweep_md_pool",
    "save_sweep_results",
    "sweep_candidate_entry",
    "sweep_group_dir",
    "sweep_row_key",
    "sweep_temp_dir",
    "update_sweep_candidates_json",
    "update_sweep_csv",
    "write_sweep_csv",
]


@dataclass(frozen=True, slots=True)
class SweepSetting:
    """
    Specification for a single nanoreactor molecular dynamics sweep setting.

    :param temperature_k: Target simulation bath temperature in Kelvin.
    :type temperature_k: float
    :param kpush_per_atom: Metadynamics push parameter per atom in Hartree.
    :type kpush_per_atom: float
    :param kpush_eh: Total metadynamics push strength in Hartree.
    :type kpush_eh: float
    :param group: Independent replicate group identifier.
    :type group: int
    """

    temperature_k: float
    kpush_per_atom: float
    kpush_eh: float
    group: int


def expand_sweep_grid(
    temperatures: Sequence[float],
    kpush_per_atom: Sequence[float],
    groups: int,
    first_group: int,
    n_atoms: int,
) -> list[SweepSetting]:
    """
    Expand combinations of temperatures, push strengths, and groups.

    :param temperatures: Target temperatures in Kelvin.
    :type temperatures: Sequence[float]
    :param kpush_per_atom: Metadynamics push strength per atom in Hartree.
    :type kpush_per_atom: Sequence[float]
    :param groups: Number of independent replicate groups to generate.
    :type groups: int
    :param first_group: Initial group index offset.
    :type first_group: int
    :param n_atoms: Total atom count in starting system.
    :type n_atoms: int
    :return: List of configured SweepSetting instances.
    :rtype: list[SweepSetting]
    """
    settings: list[SweepSetting] = []
    group_range = range(first_group, first_group + groups)
    for temp, per_atom, grp in itertools.product(
        temperatures, kpush_per_atom, group_range
    ):
        kpush_eh = round(per_atom * n_atoms, 4)
        settings.append(
            SweepSetting(
                temperature_k=float(temp),
                kpush_per_atom=float(per_atom),
                kpush_eh=kpush_eh,
                group=int(grp),
            )
        )
    return settings


def format_temperature_label(temperature_k: float) -> str:
    """
    Format temperature value into directory label string.

    :param temperature_k: Temperature in Kelvin.
    :type temperature_k: float
    :return: Formatted temperature label (e.g. 'T300').
    :rtype: str
    """
    int_val = int(temperature_k)
    val = int_val if int_val == temperature_k else temperature_k
    return f"T{val}"


def format_kpush_label(kpush_per_atom: float) -> str:
    """
    Format kpush per atom into directory label string.

    :param kpush_per_atom: Metadynamics push per atom.
    :type kpush_per_atom: float
    :return: Formatted kpush label.
    :rtype: str
    """
    return f"{kpush_per_atom:g}"


def sweep_temp_dir(
    runs_dir: Path,
    system: str,
    temperature_k: float,
) -> Path:
    """
    Construct path to temperature equilibration directory for a system.

    :param runs_dir: Root calculation runs directory.
    :type runs_dir: Path
    :param system: Chemical system identifier name.
    :type system: str
    :param temperature_k: Target temperature in Kelvin.
    :type temperature_k: float
    :return: Path to temperature equilibration directory.
    :rtype: Path
    """
    t_label = format_temperature_label(temperature_k)
    return runs_dir / "sweep" / system / t_label


def sweep_group_dir(
    runs_dir: Path,
    system: str,
    setting: SweepSetting,
) -> Path:
    """
    Construct path to replicate execution directory for a sweep setting.

    :param runs_dir: Root calculation runs directory.
    :type runs_dir: Path
    :param system: Chemical system identifier name.
    :type system: str
    :param setting: Active sweep parameter configuration.
    :type setting: SweepSetting
    :return: Path to setting and group directory.
    :rtype: Path
    """
    t_label = format_temperature_label(setting.temperature_k)
    k_label = format_kpush_label(setting.kpush_per_atom)
    group_name = f"{t_label}_k{k_label}"
    return runs_dir / "sweep" / system / group_name / f"g{setting.group}"


def is_group_complete(
    group_dir: Path,
    expected_replicates: int,
) -> bool:
    """
    Check whether all expected trajectory replicates exist in group directory.

    :param group_dir: Group root execution directory.
    :type group_dir: Path
    :param expected_replicates: Expected number of replicate trajectories.
    :type expected_replicates: int
    :return: True if at least expected replicate trajectories exist.
    :rtype: bool
    """
    if not group_dir.is_dir():
        return False
    trajectories = find_replicate_trajectories(group_dir)
    return len(trajectories) >= expected_replicates


def count_failed_replicates(group_dir: Path) -> int:
    """
    Count replicate trajectories that halted after an xTB energy failure.

    :param group_dir: Group root execution directory.
    :type group_dir: Path
    :return: Number of replicate trajectories with failed energy frames.
    :rtype: int
    """
    return sum(
        1
        for trj in find_replicate_trajectories(group_dir)
        if first_failed_frame(trj) is not None
    )


def prepare_sweep_settings(
    base_settings: MDSettings,
    setting: SweepSetting,
) -> MDSettings:
    """
    Derive simulation settings specialized for a specific sweep group.

    :param base_settings: Base system MD parameters.
    :type base_settings: MDSettings
    :param setting: Active sweep grid point parameters.
    :type setting: SweepSetting
    :return: Updated MDSettings instance for the sweep group.
    :rtype: MDSettings
    """
    group_seed = base_settings.base_seed + 10000 * setting.group
    return replace(
        base_settings,
        temperature_k=setting.temperature_k,
        kpush=setting.kpush_eh,
        base_seed=group_seed,
    )


def ensure_optimized_starts(
    starts: Sequence[tuple[str, Atoms]],
    settings: MDSettings,
    temp_dir: Path,
    pool: multiprocessing.pool.Pool | None,
) -> list[tuple[str, Atoms]]:
    """
    Ensure all start structures are optimized inside wall constraint.

    :param starts: Sequence of named starting reactant geometries.
    :type starts: Sequence[tuple[str, Atoms]]
    :param settings: Molecular dynamics settings with temperature.
    :type settings: MDSettings
    :param temp_dir: Directory holding opt_<k> folders.
    :type temp_dir: Path
    :param pool: Optional multiprocessing pool.
    :type pool: multiprocessing.pool.Pool or None
    :return: List of (name, optimized Atoms) pairs.
    :rtype: list[tuple[str, Atoms]]
    """
    opt_paths = [
        temp_dir / f"opt_{i}" / "start_optimized.xyz"
        for i in range(len(starts))
    ]
    if not all(p.is_file() for p in opt_paths):
        opt_tasks = build_optimization_tasks(starts, settings, temp_dir)
        runner = pool.map if pool else map
        list(runner(execute_optimization_task, opt_tasks))
    return [
        (starts[i][0], ase.io.read(str(opt_paths[i])))
        for i in range(len(starts))
    ]


def ensure_equilibration_trajectories(
    opt_starts: Sequence[tuple[str, Atoms]],
    settings: MDSettings,
    temp_dir: Path,
    pool: multiprocessing.pool.Pool | None,
) -> list[list[Atoms]]:
    """
    Ensure equilibration trajectories are computed inside wall constraint.

    :param opt_starts: Sequence of named optimized reactant geometries.
    :type opt_starts: Sequence[tuple[str, Atoms]]
    :param settings: Molecular dynamics settings with temperature.
    :type settings: MDSettings
    :param temp_dir: Directory holding equil_<k> folders.
    :type temp_dir: Path
    :param pool: Optional multiprocessing pool.
    :type pool: multiprocessing.pool.Pool or None
    :return: List of equilibration trajectory frame lists.
    :rtype: list[list[Atoms]]
    """
    eq_paths = [
        temp_dir / f"equil_{i}" / "xtb.trj" for i in range(len(opt_starts))
    ]
    if not all(p.is_file() for p in eq_paths):
        eq_tasks = build_equilibration_tasks(opt_starts, settings, temp_dir)
        runner = pool.map if pool else map
        list(runner(execute_equilibration_task, eq_tasks))
    return [read_trajectory(p) for p in eq_paths]


def ensure_temperature_equilibration(
    starts: Sequence[tuple[str, Atoms]],
    settings: MDSettings,
    temp_dir: Path,
    pool: multiprocessing.pool.Pool | None = None,
) -> tuple[list[tuple[str, Atoms]], list[list[Atoms]]]:
    """
    Ensure start structures are optimized and equilibrated at temperature.

    :param starts: Sequence of named starting reactant geometries.
    :type starts: Sequence[tuple[str, Atoms]]
    :param settings: Molecular dynamics settings with temperature.
    :type settings: MDSettings
    :param temp_dir: Directory holding opt_<k> and equil_<k>.
    :type temp_dir: Path
    :param pool: Optional multiprocessing pool for task execution.
    :type pool: multiprocessing.pool.Pool or None
    :return: Tuple of optimized starts and equilibration trajectory lists.
    :rtype: tuple[list[tuple[str, Atoms]], list[list[Atoms]]]
    """
    if settings.equilibrate_ps <= 0.0:
        return list(starts), []
    temp_dir.mkdir(parents=True, exist_ok=True)
    opt_starts = ensure_optimized_starts(starts, settings, temp_dir, pool)
    trajectories = ensure_equilibration_trajectories(
        opt_starts, settings, temp_dir, pool
    )
    return opt_starts, trajectories


def build_setting_tasks(
    starts: Sequence[tuple[str, Atoms]],
    trajectories: Sequence[Sequence[Atoms]],
    group_settings: MDSettings,
    group_dir: Path,
) -> list[tuple[Any, ...]]:
    """
    Build replicate task tuples for an individual sweep setting and group.

    :param starts: Starting reactant structures (optimized if equilibrated).
    :type starts: Sequence[tuple[str, Atoms]]
    :param trajectories: Pre-equilibration trajectory frames for each start.
    :type trajectories: Sequence[Sequence[Atoms]]
    :param group_settings: Molecular dynamics settings for this group.
    :type group_settings: MDSettings
    :param group_dir: Replicate destination directory.
    :type group_dir: Path
    :return: List of worker task tuples for parallel execution.
    :rtype: list[tuple[Any, ...]]
    """
    if group_settings.equilibrate_ps > 0.0:
        return build_all_equilibrated_tasks(
            starts, trajectories, group_settings, group_dir
        )
    return build_replicate_tasks(starts, group_settings, group_dir)


def collect_sweep_replicate_tasks(
    grid: Sequence[SweepSetting],
    starts_by_temp: Mapping[float, Sequence[tuple[str, Atoms]]],
    trjs_by_temp: Mapping[float, Sequence[Sequence[Atoms]]],
    base_settings: MDSettings,
    runs_dir: Path,
    system: str,
) -> list[tuple[Any, ...]]:
    """
    Collect all pending replicate execution tasks across settings and groups.

    :param grid: Sequence of all sweep grid points.
    :type grid: Sequence[SweepSetting]
    :param starts_by_temp: Mapping of temperature to optimized starts.
    :type starts_by_temp: Mapping[float, Sequence[tuple[str, Atoms]]]
    :param trjs_by_temp: Mapping of temperature to equilibration trajectories.
    :type trjs_by_temp: Mapping[float, Sequence[Sequence[Atoms]]]
    :param base_settings: Base simulation configuration.
    :type base_settings: MDSettings
    :param runs_dir: Root execution directory.
    :type runs_dir: Path
    :param system: Chemical system identifier name.
    :type system: str
    :return: Combined master list of pending replicate execution tasks.
    :rtype: list[tuple[Any, ...]]
    """
    all_tasks: list[tuple[Any, ...]] = []
    for setting in grid:
        group_dir = sweep_group_dir(runs_dir, system, setting)
        if is_group_complete(group_dir, base_settings.replicates):
            continue
        g_settings = prepare_sweep_settings(base_settings, setting)
        temp_k = setting.temperature_k
        starts = starts_by_temp[temp_k]
        trjs = trjs_by_temp[temp_k]
        tasks = build_setting_tasks(starts, trjs, g_settings, group_dir)
        all_tasks.extend(tasks)
    return all_tasks


def is_candidate_dir_ready(cand_dir: Path) -> bool:
    """
    Check whether candidate set files exist in destination directory.

    :param cand_dir: Target candidate directory path.
    :type cand_dir: Path
    :return: True if manifest and candidates JSON files exist.
    :rtype: bool
    """
    return (cand_dir / "manifest.json").is_file() and (
        (cand_dir / "candidates.json").is_file()
    )


def freeze_sweep_group(
    reactant: Atoms,
    group_settings: MDSettings,
    group_dir: Path,
    engines_path: str,
) -> CandidateSet:
    """
    Extract and freeze candidate reactions for an individual sweep group.

    :param reactant: Base reactant atomic geometry.
    :type reactant: Atoms
    :param group_settings: Simulation settings for the group.
    :type group_settings: MDSettings
    :param group_dir: Group execution directory containing replicates.
    :type group_dir: Path
    :param engines_path: Filesystem path to engines YAML configuration.
    :type engines_path: str
    :return: Frozen CandidateSet instance.
    :rtype: CandidateSet
    """
    cand_dir = group_dir / "candidates"
    if is_candidate_dir_ready(cand_dir):
        return load_candidates(cand_dir)
    mf = math.ceil(group_settings.min_lifetime_fs / group_settings.dump_fs)
    return freeze(
        reactant=reactant,
        trajectories=load_all_trajectories(group_dir),
        min_frames=mf,
        dump_fs=group_settings.dump_fs,
        engine_settings=engines_path,
        out_dir=cand_dir,
    )


def candidate_identity(candidate: Candidate) -> str:
    """
    Name a candidate reaction by its unordered pair of state keys.

    :param candidate: Frozen candidate reaction.
    :type candidate: Candidate
    :return: Key that is the same for A <-> B and B <-> A.
    :rtype: str
    """
    sides = sorted(
        "|".join(sorted(state))
        for state in (candidate.state_a, candidate.state_b)
    )
    return " <> ".join(sides)


def compute_group_overlap(
    group_candidate_sets: Mapping[int, CandidateSet],
) -> float | None:
    """
    Compute mean pairwise Jaccard similarity of candidate reaction keys.

    Reactions are compared by their unordered pair of states, not by
    mapped SMILES, whose atom numbers differ between trajectories.

    :param group_candidate_sets: Map of group index to CandidateSet.
    :type group_candidate_sets: Mapping[int, CandidateSet]
    :return: Average Jaccard similarity across group pairs, or None if < 2.
    :rtype: float or None
    """
    groups = sorted(group_candidate_sets.keys())
    if len(groups) < 2:
        return None
    jaccards: list[float] = []
    for i, g_a in enumerate(groups):
        for g_b in groups[i + 1 :]:
            cands_a = group_candidate_sets[g_a].candidates
            cands_b = group_candidate_sets[g_b].candidates
            keys_a = {candidate_identity(c) for c in cands_a}
            keys_b = {candidate_identity(c) for c in cands_b}
            val = compute_jaccard(keys_a, keys_b)
            jaccards.append(0.0 if val is None else val)
    return sum(jaccards) / len(jaccards) if jaccards else 0.0


def compute_replicate_fractions(
    events: int,
    radicals: int,
    reactive: int,
    replicates: int,
) -> tuple[float, float]:
    """
    Compute reactive replicate share and radical event fraction.

    :param events: Total reaction event count.
    :type events: int
    :param radicals: Multiple radical event count.
    :type radicals: int
    :param reactive: Count of reactive replicates.
    :type reactive: int
    :param replicates: Total replicates count.
    :type replicates: int
    :return: Tuple of reactive fraction and radical fraction.
    :rtype: tuple[float, float]
    """
    react_frac = round(reactive / replicates, 4) if replicates else 0.0
    rad_frac = round(radicals / events, 4) if events else 0.0
    return react_frac, rad_frac


def assemble_metrics_dict(
    setting: SweepSetting,
    counts: tuple[int, int, int, float, int, int, int, float],
    sizes: tuple[int, int],
) -> dict[str, Any]:
    """
    Assemble metrics dictionary from precomputed numerical values.

    :param setting: Active sweep parameter configuration.
    :type setting: SweepSetting
    :param counts: Precomputed simulation and event counts.
    :type counts: tuple[int, int, int, float, int, int, int, float]
    :param sizes: Candidates and species counts tuple.
    :type sizes: tuple[int, int]
    :return: Metrics dictionary.
    :rtype: dict[str, Any]
    """
    keys = (
        "T kpush_per_atom group n_atoms kpush_eh replicates "
        "reactive_replicates reactive_fraction xtb_failed_replicates events "
        "rejected_multiple_radicals radical_fraction candidates species"
    ).split()
    s = (setting.temperature_k, setting.kpush_per_atom, setting.group)
    vals = (*s, counts[0], setting.kpush_eh, *counts[1:8], *sizes)
    return dict(zip(keys, vals, strict=True))


def build_base_metrics(
    setting: SweepSetting,
    n_atoms: int,
    replicates: int,
    candidate_set: CandidateSet,
    xtb_failed_replicates: int = 0,
) -> dict[str, Any]:
    """
    Build core simulation and candidate extraction metrics dictionary.

    :param setting: Active sweep parameter configuration.
    :type setting: SweepSetting
    :param n_atoms: Total atom count in chemical system.
    :type n_atoms: int
    :param replicates: Total number of replicates executed.
    :type replicates: int
    :param candidate_set: Frozen candidate reactions and species.
    :type candidate_set: CandidateSet
    :param xtb_failed_replicates: Replicates halted by xTB failure.
    :type xtb_failed_replicates: int
    :return: Dictionary containing numerical metrics.
    :rtype: dict[str, Any]
    """
    ev = candidate_set.stats.get("events", 0)
    rad = candidate_set.stats.get("rejected_multiple_radicals", 0)
    rep_ids = {p.replicate for c in candidate_set.candidates for p in c.pairs}
    react = candidate_set.stats.get("reactive_replicates", len(rep_ids))
    react_frac, rad_frac = compute_replicate_fractions(
        ev, rad, react, replicates
    )
    counts = (
        n_atoms, replicates, react, react_frac, xtb_failed_replicates,
        ev, rad, rad_frac,
    )
    sizes = (len(candidate_set.candidates), len(candidate_set.species))
    return assemble_metrics_dict(setting, counts, sizes)


def build_sweep_row(
    setting: SweepSetting,
    n_atoms: int,
    replicates_count: int,
    candidate_set: CandidateSet,
    lit_products: Sequence[LiteratureProduct],
    wall_seconds: float,
    group_overlap: float | None = None,
    xtb_failed_replicates: int = 0,
) -> dict[str, Any]:
    """
    Construct summary metrics row dictionary for a sweep setting and group.

    :param setting: Active sweep parameter configuration.
    :type setting: SweepSetting
    :param n_atoms: System atom count.
    :type n_atoms: int
    :param replicates_count: Total replicate simulations executed.
    :type replicates_count: int
    :param candidate_set: Frozen CandidateSet for the setting.
    :type candidate_set: CandidateSet
    :param lit_products: Literature reference products to match.
    :type lit_products: Sequence[LiteratureProduct]
    :param wall_seconds: Execution wall time in seconds.
    :type wall_seconds: float
    :param group_overlap: Optional Jaccard similarity across replicate groups.
    :type group_overlap: float or None
    :param xtb_failed_replicates: Replicates halted by xTB failure.
    :type xtb_failed_replicates: int
    :return: Summary metrics dictionary.
    :rtype: dict[str, Any]
    """
    row = build_base_metrics(
        setting,
        n_atoms,
        replicates_count,
        candidate_set,
        xtb_failed_replicates,
    )
    for prod in lit_products:
        row[prod.name] = candidate_set_holds_product(candidate_set, prod)
    row["wall_seconds"] = round(wall_seconds, 2)
    if group_overlap is not None:
        row["group_overlap"] = round(group_overlap, 4)
    return row


def read_existing_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    """
    Read existing CSV records from file if it exists.

    :param csv_path: Filesystem path to target CSV file.
    :type csv_path: Path
    :return: List of row dictionaries read from file.
    :rtype: list[dict[str, Any]]
    """
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sweep_row_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
    """
    Extract (T, kpush_per_atom, group) key tuple from row dictionary.

    :param row: Row metrics dictionary.
    :type row: Mapping[str, Any]
    :return: Numerical tuple identifier.
    :rtype: tuple[float, float, int]
    """
    return (float(row["T"]), float(row["kpush_per_atom"]), int(row["group"]))


def merge_sweep_csv_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge existing and new sweep rows, replacing on (T, kpush_per_atom, group).

    :param existing_rows: Previously recorded metrics rows.
    :type existing_rows: Sequence[Mapping[str, Any]]
    :param new_rows: Newly generated metrics rows.
    :type new_rows: Sequence[Mapping[str, Any]]
    :return: Merged list of row dictionaries.
    :rtype: list[dict[str, Any]]
    """
    indexed = {sweep_row_key(r): dict(r) for r in existing_rows}
    indexed.update({sweep_row_key(r): dict(r) for r in new_rows})
    return list(indexed.values())


def write_sweep_csv(
    csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """
    Write sequence of row dictionaries to CSV file with aligned headers.

    :param csv_path: Filesystem path to destination CSV file.
    :type csv_path: Path
    :param rows: Sequence of metrics rows to write.
    :type rows: Sequence[Mapping[str, Any]]
    :return: None.
    :rtype: None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def update_sweep_csv(
    csv_path: Path,
    new_rows: Sequence[Mapping[str, Any]],
) -> None:
    """
    Update sweep CSV file by merging new rows with existing records.

    :param csv_path: Destination CSV file path.
    :type csv_path: Path
    :param new_rows: Newly generated metrics rows.
    :type new_rows: Sequence[Mapping[str, Any]]
    :return: None.
    :rtype: None
    """
    existing = read_existing_csv_rows(csv_path)
    merged = merge_sweep_csv_rows(existing, new_rows)
    write_sweep_csv(csv_path, merged)


def update_sweep_candidates_json(
    json_path: Path,
    new_entries: Mapping[str, Sequence[str]],
) -> None:
    """
    Update sweep candidates JSON dictionary mapping row keys to labels.

    :param json_path: Destination JSON file path.
    :type json_path: Path
    :param new_entries: Mapping of row keys to lists of candidate labels.
    :type new_entries: Mapping[str, Sequence[str]]
    :return: None.
    :rtype: None
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, list[str]] = {}
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
            if isinstance(raw, dict):
                data = {str(k): list(v) for k, v in raw.items()}
    for key, labels in new_entries.items():
        data[key] = list(labels)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def equilibrate_all_temperatures(
    temperatures: Sequence[float],
    starts: Sequence[tuple[str, Atoms]],
    base_settings: MDSettings,
    runs_path: Path,
    system: str,
    pool: multiprocessing.pool.Pool,
) -> tuple[
    dict[float, Sequence[tuple[str, Atoms]]],
    dict[float, Sequence[Sequence[Atoms]]],
]:
    """
    Ensure all unique sweep temperatures are optimized and equilibrated.

    :param temperatures: Target temperatures in Kelvin.
    :type temperatures: Sequence[float]
    :param starts: Starting reactant structures.
    :type starts: Sequence[tuple[str, Atoms]]
    :param base_settings: Base molecular dynamics configuration.
    :type base_settings: MDSettings
    :param runs_path: Root execution directory.
    :type runs_path: Path
    :param system: Chemical system identifier name.
    :type system: str
    :param pool: Active worker multiprocessing pool.
    :type pool: multiprocessing.pool.Pool
    :return: Tuple of mappings from temperature to starts and trajectories.
    :rtype: tuple
    """
    st_by_t: dict[float, Sequence[tuple[str, Atoms]]] = {}
    trj_by_t: dict[float, Sequence[Sequence[Atoms]]] = {}
    for temp_k in set(temperatures):
        t_dir = sweep_temp_dir(runs_path, system, temp_k)
        t_settings = replace(base_settings, temperature_k=float(temp_k))
        opt_st, trjs = ensure_temperature_equilibration(
            starts, t_settings, t_dir, pool
        )
        st_by_t[temp_k], trj_by_t[temp_k] = opt_st, trjs
    return st_by_t, trj_by_t


def freeze_all_groups(
    grid: Sequence[SweepSetting],
    base_settings: MDSettings,
    starts: Sequence[tuple[str, Atoms]],
    runs_path: Path,
    system: str,
    engines: str,
) -> dict[SweepSetting, CandidateSet]:
    """
    Freeze candidate reactions across all sweep groups into local directories.

    :param grid: Sequence of all sweep grid points.
    :type grid: Sequence[SweepSetting]
    :param base_settings: Base simulation settings.
    :type base_settings: MDSettings
    :param starts: System starting reactant configurations.
    :type starts: Sequence[tuple[str, Atoms]]
    :param runs_path: Root execution directory path.
    :type runs_path: Path
    :param system: System identifier name.
    :type system: str
    :param engines: Path to engines YAML configuration.
    :type engines: str
    :return: Dictionary mapping SweepSetting to CandidateSet.
    :rtype: dict[SweepSetting, CandidateSet]
    """
    csets: dict[SweepSetting, CandidateSet] = {}
    for setting in grid:
        g_dir = sweep_group_dir(runs_path, system, setting)
        g_settings = prepare_sweep_settings(base_settings, setting)
        csets[setting] = freeze_sweep_group(
            starts[0][1], g_settings, g_dir, engines
        )
    return csets


def compute_setting_overlaps(
    grid: Sequence[SweepSetting],
    csets: Mapping[SweepSetting, CandidateSet],
) -> dict[tuple[float, float], float | None]:
    """
    Compute pairwise group overlap for each distinct (temperature, kpush).

    :param grid: Sequence of sweep settings.
    :type grid: Sequence[SweepSetting]
    :param csets: Mapping of sweep setting to frozen candidate set.
    :type csets: Mapping[SweepSetting, CandidateSet]
    :return: Mapping of (temperature, kpush) to group overlap Jaccard.
    :rtype: dict[tuple[float, float], float or None]
    """
    overlaps: dict[tuple[float, float], float | None] = {}
    for (t, k), group_items in itertools.groupby(
        grid, key=lambda s: (s.temperature_k, s.kpush_per_atom)
    ):
        group_list = list(group_items)
        if len(group_list) >= 2:
            subset = {s.group: csets[s] for s in group_list}
            overlaps[(t, k)] = compute_group_overlap(subset)
    return overlaps


def sweep_candidate_entry(
    setting: SweepSetting,
    candidate_set: CandidateSet,
) -> tuple[str, list[str]]:
    """
    Format candidates dictionary entry key and reaction SMILES list.

    :param setting: Active sweep parameter configuration.
    :type setting: SweepSetting
    :param candidate_set: Frozen candidate reactions and species.
    :type candidate_set: CandidateSet
    :return: Tuple of row key identifier and mapped SMILES list.
    :rtype: tuple[str, list[str]]
    """
    t_lbl = format_temperature_label(setting.temperature_k)
    k_lbl = format_kpush_label(setting.kpush_per_atom)
    key = f"{t_lbl}_k{k_lbl}_g{setting.group}"
    return key, [c.mapped_smiles for c in candidate_set.candidates]


def build_sweep_records(
    grid: Sequence[SweepSetting],
    n_atoms: int,
    replicates: int,
    csets: Mapping[SweepSetting, CandidateSet],
    lit_products: Sequence[LiteratureProduct],
    wall_time: float,
    overlaps: Mapping[tuple[float, float], float | None],
    failed_counts: Mapping[SweepSetting, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """
    Compile CSV metrics rows and candidates JSON dictionary for sweep.

    :param grid: Sequence of all sweep grid points.
    :type grid: Sequence[SweepSetting]
    :param n_atoms: System atom count.
    :type n_atoms: int
    :param replicates: Replicate count per setting.
    :type replicates: int
    :param csets: Mapping of sweep setting to CandidateSet.
    :type csets: Mapping[SweepSetting, CandidateSet]
    :param lit_products: Target literature products to match.
    :type lit_products: Sequence[LiteratureProduct]
    :param wall_time: Simulation wall time in seconds.
    :type wall_time: float
    :param overlaps: Mapping of (T, kpush) to group overlap score.
    :type overlaps: Mapping[tuple[float, float], float or None]
    :param failed_counts: Optional mapping of setting to failed replicate count.
    :type failed_counts: Mapping[SweepSetting, int] or None
    :return: Tuple of metrics row dictionaries and candidate label entries.
    :rtype: tuple[list[dict[str, Any]], dict[str, list[str]]]
    """
    rows: list[dict[str, Any]] = []
    cands_json: dict[str, list[str]] = {}
    failed = failed_counts or {}
    for setting in grid:
        cset = csets[setting]
        ov = overlaps.get((setting.temperature_k, setting.kpush_per_atom))
        row = build_sweep_row(
            setting, n_atoms, replicates, cset, lit_products, wall_time, ov,
            failed.get(setting, 0),
        )
        rows.append(row)
        key, smis = sweep_candidate_entry(setting, cset)
        cands_json[key] = smis
    return rows, cands_json


def execute_sweep_pool(
    temperatures: Sequence[float],
    starts: Sequence[tuple[str, Atoms]],
    base_settings: MDSettings,
    grid: Sequence[SweepSetting],
    runs_path: Path,
    system: str,
    workers: int,
) -> float:
    """
    Run temperature equilibration and all pending replicate tasks in pool.

    :param temperatures: Simulation bath temperatures.
    :type temperatures: Sequence[float]
    :param starts: Starting reactant structures.
    :type starts: Sequence[tuple[str, Atoms]]
    :param base_settings: Base simulation configuration.
    :type base_settings: MDSettings
    :param grid: Sweep setting combinations.
    :type grid: Sequence[SweepSetting]
    :param runs_path: Calculation runs directory.
    :type runs_path: Path
    :param system: Chemical system identifier name.
    :type system: str
    :param workers: Number of parallel workers.
    :type workers: int
    :return: Replicate execution wall time in seconds.
    :rtype: float
    """
    with multiprocessing.Pool(processes=workers) as pool:
        st_by_t, trj_by_t = equilibrate_all_temperatures(
            temperatures, starts, base_settings, runs_path, system, pool
        )
        tasks = collect_sweep_replicate_tasks(
            grid, st_by_t, trj_by_t, base_settings, runs_path, system
        )
        t0 = time.perf_counter()
        if tasks:
            pool.map(execute_replicate_task, tasks)
        return time.perf_counter() - t0


def save_sweep_results(
    data_dir: str | Path,
    system: str,
    rows: Sequence[Mapping[str, Any]],
    cands_json: Mapping[str, Sequence[str]],
) -> Path:
    """
    Persist generated sweep CSV metrics and candidate labels JSON.

    :param data_dir: Data root directory path.
    :type data_dir: str or Path
    :param system: System identifier name.
    :type system: str
    :param rows: Metrics table rows.
    :type rows: Sequence[Mapping[str, Any]]
    :param cands_json: Mapping of row keys to candidate labels.
    :type cands_json: Mapping[str, Sequence[str]]
    :return: Path to updated md_sweep.csv.
    :rtype: Path
    """
    data_path = Path(data_dir) / system
    csv_file = data_path / "md_sweep.csv"
    update_sweep_csv(csv_file, rows)
    update_sweep_candidates_json(
        data_path / "md_sweep_candidates.json", cands_json
    )
    return csv_file


def load_sweep_system_context(
    system: str,
    temperatures: Sequence[float],
    kpush_per_atom: Sequence[float],
    groups: int,
    first_group: int,
    systems_dir: str | Path,
) -> tuple[
    list[tuple[str, Atoms]],
    MDSettings,
    list[LiteratureProduct],
    list[SweepSetting],
]:
    """
    Load system inputs, configuration, literature products, and sweep grid.

    :param system: Chemical system identifier name.
    :type system: str
    :param temperatures: Simulation bath temperatures.
    :type temperatures: Sequence[float]
    :param kpush_per_atom: Metadynamics push strengths per atom.
    :type kpush_per_atom: Sequence[float]
    :param groups: Replicate groups count.
    :type groups: int
    :param first_group: First group index identifier.
    :type first_group: int
    :param systems_dir: Systems configuration directory.
    :type systems_dir: str or Path
    :return: Tuple of starts, base settings, literature products, and grid.
    :rtype: tuple
    """
    sys_dir = Path(systems_dir) / system
    starts = load_system_reactants(sys_dir)
    base_settings = load_md_settings(sys_dir / "system.yaml")
    lit_products = load_literature_products(sys_dir / "literature.yaml")
    n_atoms = len(starts[0][1])
    grid = expand_sweep_grid(
        temperatures, kpush_per_atom, groups, first_group, n_atoms
    )
    return starts, base_settings, lit_products, grid


def process_sweep_results(
    grid: Sequence[SweepSetting],
    starts: Sequence[tuple[str, Atoms]],
    base_settings: MDSettings,
    runs_path: Path,
    system: str,
    engines: str,
    lit_products: Sequence[LiteratureProduct],
    wall_time: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """
    Freeze all groups, evaluate overlaps, and construct metrics records.

    :param grid: Sequence of evaluated sweep settings.
    :type grid: Sequence[SweepSetting]
    :param starts: Sequence of start structure tuples.
    :type starts: Sequence[tuple[str, Atoms]]
    :param base_settings: Base molecular dynamics configuration.
    :type base_settings: MDSettings
    :param runs_path: Sweep root directory path.
    :type runs_path: Path
    :param system: Target chemical system name.
    :type system: str
    :param engines: Path to engines YAML configuration file.
    :type engines: str
    :param lit_products: Target literature product specifications.
    :type lit_products: Sequence[LiteratureProduct]
    :param wall_time: Total elapsed MD pool execution time in seconds.
    :type wall_time: float
    :return: Tuple of row dictionaries and candidate label mapping.
    :rtype: tuple[list[dict[str, Any]], dict[str, list[str]]]
    """
    csets = freeze_all_groups(
        grid, base_settings, starts, runs_path, system, engines
    )
    overlaps = compute_setting_overlaps(grid, csets)
    failed = {
        s: count_failed_replicates(sweep_group_dir(runs_path, system, s))
        for s in grid
    }
    return build_sweep_records(
        grid, len(starts[0][1]), base_settings.replicates, csets, lit_products,
        wall_time, overlaps, failed,
    )


def run_sweep_md_pool(
    system: str,
    kpush_per_atom: Sequence[float],
    temperatures: Sequence[float],
    groups: int,
    first_group: int,
    workers: int,
    systems_dir: str | Path,
    runs_path: Path,
) -> tuple[
    Sequence[tuple[str, Atoms]],
    MDSettings,
    Sequence[LiteratureProduct],
    Sequence[SweepSetting],
    float,
]:
    """
    Load system inputs and execute parallel MD replicate pool across sweep grid.

    :param system: Chemical system identifier name.
    :type system: str
    :param kpush_per_atom: Sequence of push strengths per atom.
    :type kpush_per_atom: Sequence[float]
    :param temperatures: Target temperatures in Kelvin.
    :type temperatures: Sequence[float]
    :param groups: Replicate groups count.
    :type groups: int
    :param first_group: First group index identifier.
    :type first_group: int
    :param workers: Multiprocessing worker count.
    :type workers: int
    :param systems_dir: System definition directory path.
    :type systems_dir: str or Path
    :param runs_path: Sweep root execution path.
    :type runs_path: Path
    :return: Tuple of starts, settings, lit products, grid, and wall time.
    :rtype: tuple
    """
    starts, base_settings, lit_products, grid = load_sweep_system_context(
        system, temperatures, kpush_per_atom, groups, first_group, systems_dir
    )
    wall_time = execute_sweep_pool(
        temperatures, starts, base_settings, grid, runs_path, system, workers
    )
    return starts, base_settings, lit_products, grid, wall_time


def run_sweep(
    system: str,
    kpush_per_atom: Sequence[float],
    temperatures: Sequence[float],
    groups: int = 1,
    first_group: int = 1,
    workers: int = 1,
    systems_dir: str | Path = "systems",
    runs_dir: str | Path = "runs",
    data_dir: str | Path = "data",
    engines: str = "configs/engines.yaml",
) -> Path:
    """
    Execute full molecular dynamics parameter sweep and record metrics table.

    :param system: Chemical system identifier name.
    :type system: str
    :param kpush_per_atom: Sequence of metadynamics push strengths per atom.
    :type kpush_per_atom: Sequence[float]
    :param temperatures: Sequence of target simulation temperatures in Kelvin.
    :type temperatures: Sequence[float]
    :param groups: Total number of independent groups per setting.
    :type groups: int
    :param first_group: First group index identifier.
    :type first_group: int
    :param workers: Number of parallel worker processes.
    :type workers: int
    :param systems_dir: Systems configuration directory path.
    :type systems_dir: str or Path
    :param runs_dir: Directory where calculation runs are stored.
    :type runs_dir: str or Path
    :param data_dir: Directory where summary CSV and JSON tables are written.
    :type data_dir: str or Path
    :param engines: Path to engines YAML configuration file.
    :type engines: str
    :return: Path to generated md_sweep.csv file.
    :rtype: Path
    """
    runs_path = Path(runs_dir)
    starts, settings, lits, grid, w_sec = run_sweep_md_pool(
        system,
        kpush_per_atom,
        temperatures,
        groups,
        first_group,
        workers,
        systems_dir,
        runs_path,
    )
    rows, cands = process_sweep_results(
        grid, starts, settings, runs_path, system, engines, lits, w_sec
    )
    return save_sweep_results(data_dir, system, rows, cands)
