"""
Product census across nanoreactor MD trajectories using GFN2-xTB relaxation.

Samples frames periodically from trajectories, relaxes each frame with
GFN2-xTB (via tblite and BFGS), fragments the structure into species keyed
by graph topology, checks coverage against frozen candidate sets, and
writes census metrics and species discovery records to CSV.
"""

import argparse
import csv
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

from ase import Atoms
from ase.optimize import BFGS
from rdkit import RDLogger
from tblite.ase import TBLite

from nrnet.chem import (
    connectivity_graph,
    fragments,
    species_labels,
    state_key,
)
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.md import (
    find_replicate_trajectories,
    read_trajectory,
)

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "CENSUS_CSV_FIELDNAMES",
    "CLI_ARGUMENTS",
    "DEFAULT_FMAX_EV_PER_ANG",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_WORKERS",
    "FRAMES_PER_PS",
    "base_species_key",
    "build_census_rows",
    "build_parser",
    "census_trajectory",
    "collect_run_trajectories",
    "export_census_results",
    "init_census_state",
    "inspect_relaxed_fragments",
    "load_known_species",
    "main",
    "print_census_table",
    "record_census_hit",
    "relax_atomic_frame",
    "report_and_export_census",
    "run_census_pool",
    "sample_frame_indices",
    "update_census_discovery",
    "write_census_csv",
]

FRAMES_PER_PS: int = 200
DEFAULT_WORKERS: int = 10
DEFAULT_FMAX_EV_PER_ANG: float = 0.05
DEFAULT_MAX_STEPS: int = 400

CENSUS_CSV_FIELDNAMES: tuple[str, ...] = (
    "key",
    "formula",
    "smiles",
    "inchikey14",
    "runs_with",
    "in_frozen_sets",
    "first_run",
    "first_frame",
)

CLI_ARGUMENTS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("--system",),
        {
            "type": str,
            "required": True,
            "help": "System name for candidate matching and default CSV path.",
        },
    ),
    (
        ("--runs",),
        {
            "nargs": "+",
            "required": True,
            "help": "Run directory paths or glob expressions.",
        },
    ),
    (
        ("--workers",),
        {
            "type": int,
            "default": DEFAULT_WORKERS,
            "help": "Worker process pool size (default: 10).",
        },
    ),
    (
        ("--csv",),
        {
            "type": Path,
            "default": None,
            "help": (
                "Optional path to output CSV (default: data/<system>/...)."
            ),
        },
    ),
)


def base_species_key(full_key: str) -> str:
    """
    Extract the base formula-hash portion of a species key.

    :param full_key: Canonical species key with optional stereochemistry.
    :type full_key: str
    :return: Base key containing the first two hyphen-separated parts.
    :rtype: str
    """
    return "-".join(full_key.split("-")[:2])


def relax_atomic_frame(frame: Atoms) -> Atoms | None:
    """
    Relax atomic frame coordinates with GFN2-xTB and BFGS optimizer.

    :param frame: Atomic frame configuration to relax.
    :type frame: Atoms
    :return: Relaxed configuration or None if optimization fails.
    :rtype: Atoms or None
    """
    relaxed = frame.copy()
    relaxed.calc = TBLite(method="GFN2-xTB", verbosity=0)
    try:
        optimizer = BFGS(relaxed, logfile=None)
        optimizer.run(
            fmax=DEFAULT_FMAX_EV_PER_ANG,
            steps=DEFAULT_MAX_STEPS,
        )
    except Exception:
        return None
    return relaxed


def sample_frame_indices(total_frames: int) -> list[int]:
    """
    Determine periodic frame indices and the terminal frame to audit.

    :param total_frames: Total number of frames in the trajectory.
    :type total_frames: int
    :return: List of chosen frame indices.
    :rtype: list[int]
    """
    picks = list(range(FRAMES_PER_PS, total_frames, FRAMES_PER_PS))
    if total_frames > 0 and (not picks or picks[-1] != total_frames - 1):
        picks.append(total_frames - 1)
    return picks


def inspect_relaxed_fragments(
    relaxed: Atoms,
    frame_idx: int,
) -> dict[str, tuple[str, str, int]]:
    """
    Extract species labels and keys from relaxed frame fragments.

    :param relaxed: Relaxed atomic configuration of the box.
    :type relaxed: Atoms
    :param frame_idx: Frame index within the parent trajectory.
    :type frame_idx: int
    :return: Map of base key to (smiles, inchikey14, frame_idx).
    :rtype: dict[str, tuple[str, str, int]]
    """
    discovered: dict[str, tuple[str, str, int]] = {}
    graph = connectivity_graph(relaxed)
    for frag in fragments(graph):
        sub_atoms = relaxed[frag]
        key = base_species_key(state_key(sub_atoms)[0])
        if key not in discovered:
            labels = species_labels(sub_atoms)
            smiles_val = labels.smiles or ""
            inchi14_val = labels.inchikey[:14] if labels.inchikey else ""
            discovered[key] = (smiles_val, inchi14_val, frame_idx)
    return discovered


def census_trajectory(
    path: Path,
) -> tuple[str, dict[str, tuple[str, str, int]]]:
    """
    Scan trajectory frames and collect all relaxed chemical species.

    :param path: Path to trajectory file.
    :type path: Path
    :return: Tuple of trajectory path string and map of discovered species.
    :rtype: tuple[str, dict[str, tuple[str, str, int]]]
    """
    frames = read_trajectory(path)
    picks = sample_frame_indices(len(frames))
    found: dict[str, tuple[str, str, int]] = {}
    for frame_idx in picks:
        relaxed = relax_atomic_frame(frames[frame_idx])
        if relaxed is None:
            continue
        discovered = inspect_relaxed_fragments(relaxed, frame_idx)
        for key, record in discovered.items():
            if key not in found:
                found[key] = record
    return str(path), found


def collect_run_trajectories(run_patterns: Sequence[str]) -> list[Path]:
    """
    Expand run glob expressions and find all replicate trajectory paths.

    :param run_patterns: Sequence of directory globs or paths.
    :type run_patterns: Sequence[str]
    :return: Unique list of trajectory file paths.
    :rtype: list[Path]
    """
    trajectories: list[Path] = []
    for pattern in run_patterns:
        for dir_path in sorted(Path().glob(pattern)):
            if dir_path.is_dir():
                trajectories.extend(find_replicate_trajectories(dir_path))
            elif dir_path.is_file() and dir_path.name.endswith(".trj"):
                trajectories.append(dir_path)
    return list(dict.fromkeys(trajectories))


def load_known_species(system: str) -> set[str]:
    """
    Collect known base species keys from candidates and sweep directories.

    :param system: Chemical system name.
    :type system: str
    :return: Set of known base species keys.
    :rtype: set[str]
    """
    keys: set[str] = set()
    roots: list[Path] = [Path("candidates") / system]
    sweep_dir = Path("runs/sweep") / system
    if sweep_dir.is_dir():
        roots.extend(sweep_dir.glob("*/g*/candidates"))
    for root in roots:
        if (root / "manifest.json").is_file():
            cset = load_candidates(root)
            keys.update(base_species_key(key) for key in cset.species)
    return keys


def print_census_table(
    system: str,
    path_count: int,
    runs_with: Counter[str],
    known_keys: set[str],
    smiles_map: Mapping[str, str],
) -> None:
    """
    Print formatted species census table to stdout.

    :param system: Chemical system name.
    :type system: str
    :param path_count: Total number of trajectories audited.
    :type path_count: int
    :param runs_with: Map of base key to count of runs containing species.
    :type runs_with: Counter[str]
    :param known_keys: Set of species keys present in frozen sets.
    :type known_keys: set[str]
    :param smiles_map: Map of base key to representative SMILES.
    :type smiles_map: Mapping[str, str]
    :return: None
    :rtype: None
    """
    print(f"{system}: {path_count} runs, {len(runs_with)} distinct species")
    print("runs  in-our-sets  species")
    for key, count in runs_with.most_common():
        tag = "yes" if key in known_keys else "NO "
        display_label = smiles_map.get(key) or key
        print(f"{count:5d}  {tag:11s}  {display_label}")


def write_census_csv(
    csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """
    Export census results and first-observation records to CSV.

    :param csv_path: Output CSV file destination path.
    :type csv_path: Path
    :param rows: Sequence of census row dictionaries.
    :type rows: Sequence[Mapping[str, Any]]
    :return: None
    :rtype: None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CENSUS_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct argument parser for trajectory species census.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Census chemical species across MD runs via xTB."
    )
    for flags, options in CLI_ARGUMENTS:
        parser.add_argument(*flags, **options)
    return parser


def record_census_hit(
    key: str,
    hit_info: tuple[str, str, int],
    rank: int,
    run_path: str,
    smiles_map: dict[str, str],
    inchi_map: dict[str, str],
    first_seen: dict[str, tuple[int, str, int]],
) -> None:
    """
    Update global discovery metadata with newly observed species hit.

    :param key: Canonical base species key.
    :type key: str
    :param hit_info: Tuple of (smiles, inchikey14, frame_idx).
    :type hit_info: tuple[str, str, int]
    :param rank: Priority index of trajectory within input sequence.
    :type rank: int
    :param run_path: String path of the parent trajectory.
    :type run_path: str
    :param smiles_map: Accumulated dictionary of canonical SMILES.
    :type smiles_map: dict[str, str]
    :param inchi_map: Accumulated dictionary of 14-char InChIKeys.
    :type inchi_map: dict[str, str]
    :param first_seen: Map of key to (rank, run_path, frame_idx).
    :type first_seen: dict[str, tuple[int, str, int]]
    :return: None
    :rtype: None
    """
    smi, inchi14, frame_idx = hit_info
    if key not in smiles_map or (not smiles_map[key] and smi):
        smiles_map[key] = smi
    if key not in inchi_map or (not inchi_map[key] and inchi14):
        inchi_map[key] = inchi14
    if key not in first_seen or rank < first_seen[key][0]:
        first_seen[key] = (rank, run_path, frame_idx)


def update_census_discovery(
    run_path: str,
    discovered: Mapping[str, tuple[str, str, int]],
    rank: int,
    state: tuple[
        Counter[str],
        dict[str, str],
        dict[str, str],
        dict[str, tuple[int, str, int]],
    ],
) -> None:
    """
    Incorporate species discovered in a trajectory into global census state.

    :param run_path: String path of the parent trajectory.
    :type run_path: str
    :param discovered: Mapping of base key to (smiles, inchikey14, frame_idx).
    :type discovered: Mapping[str, tuple[str, str, int]]
    :param rank: Priority index of trajectory within input sequence.
    :type rank: int
    :param state: Aggregated census state quadruple.
    :type state: tuple[Counter[str], dict[str, str], dict[str, str],
        dict[str, tuple[int, str, int]]]
    :return: None
    :rtype: None
    """
    runs_with, smiles_map, inchi_map, first_seen = state
    for key, hit_info in discovered.items():
        runs_with[key] += 1
        record_census_hit(
            key, hit_info, rank, run_path, smiles_map, inchi_map, first_seen
        )


def init_census_state() -> tuple[
    Counter[str],
    dict[str, str],
    dict[str, str],
    dict[str, tuple[int, str, int]],
]:
    """
    Initialize empty containers for aggregating census discovery metrics.

    :return: Quadruple of runs_with, smiles_map, inchi_map, and first_seen.
    :rtype: tuple[Counter[str], dict[str, str], dict[str, str],
        dict[str, tuple[int, str, int]]]
    """
    return Counter(), {}, {}, {}


def run_census_pool(
    paths: Sequence[Path],
    workers: int,
) -> tuple[
    Counter[str],
    dict[str, str],
    dict[str, str],
    dict[str, tuple[int, str, int]],
]:
    """
    Execute parallel species census across located trajectories.

    :param paths: Sequence of trajectory paths to audit.
    :type paths: Sequence[Path]
    :param workers: Number of worker processes in the pool.
    :type workers: int
    :return: Tuple of runs_with, smiles_map, inchi_map, and first_seen.
    :rtype: tuple[Counter[str], dict[str, str], dict[str, str],
        dict[str, tuple[int, str, int]]]
    """
    ranks = {str(path): idx for idx, path in enumerate(paths)}
    state = init_census_state()
    with Pool(workers) as pool:
        for run_path, discovered in pool.imap_unordered(
            census_trajectory, paths, chunksize=4
        ):
            update_census_discovery(
                run_path, discovered, ranks[run_path], state
            )
    return state


def build_census_rows(
    runs_with: Counter[str],
    known_keys: set[str],
    smiles_map: Mapping[str, str],
    inchi_map: Mapping[str, str],
    first_seen: Mapping[str, tuple[int, str, int]],
) -> list[dict[str, Any]]:
    """
    Format census output rows for CSV serialization.

    :param runs_with: Map of base key to count of runs containing species.
    :type runs_with: Counter[str]
    :param known_keys: Set of species keys present in frozen sets.
    :type known_keys: set[str]
    :param smiles_map: Map of base key to representative SMILES.
    :type smiles_map: Mapping[str, str]
    :param inchi_map: Map of base key to 14-character InChIKey.
    :type inchi_map: Mapping[str, str]
    :param first_seen: Map of key to first observation tuple.
    :type first_seen: Mapping[str, tuple[int, str, int]]
    :return: List of structured dictionary rows for CSV export.
    :rtype: list[dict[str, Any]]
    """
    return [
        {
            "key": key,
            "formula": key.split("-")[0],
            "smiles": smiles_map.get(key, ""),
            "inchikey14": inchi_map.get(key, ""),
            "runs_with": count,
            "in_frozen_sets": key in known_keys,
            "first_run": first_seen[key][1],
            "first_frame": first_seen[key][2],
        }
        for key, count in runs_with.most_common()
    ]


def export_census_results(
    system: str,
    custom_path: Path | None,
    csv_rows: Sequence[Mapping[str, Any]],
) -> None:
    """
    Write census records to user path or default system checks path.

    :param system: Chemical system name.
    :type system: str
    :param custom_path: User-specified CSV path, if provided.
    :type custom_path: Path or None
    :param csv_rows: Structured census records for serialization.
    :type csv_rows: Sequence[Mapping[str, Any]]
    :return: None
    :rtype: None
    """
    csv_path = custom_path or (Path("data") / system / "checks" / "census.csv")
    write_census_csv(csv_path, csv_rows)


def report_and_export_census(
    system: str,
    path_count: int,
    custom_path: Path | None,
    census_data: tuple[
        Counter[str],
        dict[str, str],
        dict[str, str],
        dict[str, tuple[int, str, int]],
    ],
) -> None:
    """
    Print summary census table and export discovery records to CSV.

    :param system: Chemical system name.
    :type system: str
    :param path_count: Total number of trajectories audited.
    :type path_count: int
    :param custom_path: User-specified CSV path, if provided.
    :type custom_path: Path or None
    :param census_data: Census state quadruple from worker pool.
    :type census_data: tuple[Counter[str], dict[str, str], dict[str, str],
        dict[str, tuple[int, str, int]]]
    :return: None
    :rtype: None
    """
    runs_with, smiles_map, inchi_map, first_seen = census_data
    known_keys = load_known_species(system)
    print_census_table(system, path_count, runs_with, known_keys, smiles_map)
    csv_rows = build_census_rows(
        runs_with, known_keys, smiles_map, inchi_map, first_seen
    )
    export_census_results(system, custom_path, csv_rows)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute CLI entry point for trajectory species census.

    :param argv: Optional command-line arguments list.
    :type argv: Sequence[str] or None
    :return: Process exit status code.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = collect_run_trajectories(args.runs)
    census_data = run_census_pool(paths, args.workers)
    report_and_export_census(args.system, len(paths), args.csv, census_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
