"""
Inspect saddle geometry, atom DFT forces, and energy offsets (PLAN D23).

For each engine with a confirmed saddle geometry in ts.extxyz, computes
the N shortest interatomic distances labelled like C0-H3, atom DFT force
norms and fmax from results/dft_sp/.../points.jsonl, and the DFT energy
difference above the lowest saddle energy across engines in kcal/mol.
Prints one line per engine and writes
data/<system>/checks/saddle_geometry_<reaction>.csv.
"""

import argparse
import csv
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from dft_single_points import DEFAULT_ENGINES

from nrnet.store import latest_rows

__all__ = [
    "EV_TO_KCAL",
    "EngineGeometry",
    "atom_force_norms",
    "build_geometry_item",
    "build_parser",
    "collect_geometry_rows",
    "compute_energy_offsets",
    "format_engine_line",
    "load_dft_sp_records",
    "load_saddle_frame",
    "load_saddle_frames",
    "main",
    "shortest_distances",
    "write_geometry_csv",
]

EV_TO_KCAL: float = 23.060548


@dataclass(frozen=True, slots=True)
class EngineGeometry:
    """
    Geometry, forces, and relative energy for one engine.

    :param engine: Engine identifier.
    :param pairs: Shortest interatomic distances as (label, distance_a).
    :param forces: Force norms per atom as (atom_label, force_norm_ev_a).
    :param fmax: DFT maximum atom force norm in eV/A.
    :param energy_ev: Absolute DFT energy in eV.
    :param de_lowest_kcal: Energy above lowest DFT saddle in kcal/mol.
    """

    engine: str
    pairs: list[tuple[str, float]]
    forces: list[tuple[str, float]]
    fmax: float | None
    energy_ev: float | None
    de_lowest_kcal: float | None


def load_saddle_frame(path: Path, reaction_id: str) -> Atoms | None:
    """
    Find the saddle frame for a reaction in an extxyz file.

    :param path: Path to ts.extxyz.
    :type path: Path
    :param reaction_id: Reaction identifier.
    :type reaction_id: str
    :return: The Atoms frame, or None if missing.
    :rtype: Atoms or None
    """
    if not path.is_file():
        return None
    matches = [
        frame
        for frame in read(path, ":")
        if frame.info.get("reaction_id") == reaction_id
        or frame.info.get("id") == reaction_id
    ]
    return matches[-1] if matches else None


def load_saddle_frames(
    results_dir: Path,
    system: str,
    reaction_id: str,
    engines: Sequence[str],
) -> dict[str, Atoms]:
    """
    Load saddle frames for all engines that found one.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System identifier.
    :type system: str
    :param reaction_id: Reaction identifier.
    :type reaction_id: str
    :param engines: Engines to query.
    :type engines: Sequence[str]
    :return: Mapping of engine name to Atoms object.
    :rtype: dict[str, Atoms]
    """
    frames: dict[str, Atoms] = {}
    for engine in engines:
        ts_path = results_dir / "refine" / system / engine / "ts.extxyz"
        frame = load_saddle_frame(ts_path, reaction_id)
        if frame is not None:
            frames[engine] = frame
    return frames


def shortest_distances(
    atoms: Atoms, count: int
) -> list[tuple[str, float]]:
    """
    Compute the N shortest pairwise interatomic distances.

    :param atoms: Atomic structure.
    :type atoms: Atoms
    :param count: Maximum number of pairs to return.
    :type count: int
    :return: Pairs sorted by distance ascending as (label, distance_a).
    :rtype: list[tuple[str, float]]
    """
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    pairs: list[tuple[str, float]] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = float(
                np.linalg.norm(atoms.positions[i] - atoms.positions[j])
            )
            pairs.append((f"{symbols[i]}{i}-{symbols[j]}{j}", round(dist, 3)))
    pairs.sort(key=lambda item: item[1])
    return pairs[:count]


def atom_force_norms(
    symbols: Sequence[str], forces: Sequence[Sequence[float]] | None
) -> list[tuple[str, float]]:
    """
    Compute force vector 2-norm for each atom.

    :param symbols: Chemical symbols in atom order.
    :type symbols: Sequence[str]
    :param forces: Force 3-vectors in eV/A per atom.
    :type forces: Sequence[Sequence[float]] or None
    :return: List of (atom_label, force_norm_ev_a).
    :rtype: list[tuple[str, float]]
    """
    if not forces:
        return []
    norms: list[tuple[str, float]] = []
    for index, (symbol, force_vec) in enumerate(
        zip(symbols, forces, strict=False)
    ):
        norm = math.sqrt(sum(comp**2 for comp in force_vec))
        norms.append((f"{symbol}{index}", round(norm, 2)))
    return norms


def load_dft_sp_records(
    results_dir: Path, system: str, engine: str
) -> dict[str, dict[str, Any]]:
    """
    Load single-point DFT records for one system and engine.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Mapping of uid to record dictionary.
    :rtype: dict[str, dict[str, Any]]
    """
    path = results_dir / "dft_sp" / system / engine / "points.jsonl"
    return latest_rows(path, "uid") if path.is_file() else {}


def compute_energy_offsets(
    energies: Mapping[str, float | None]
) -> dict[str, float | None]:
    """
    Compute energy offset above lowest DFT saddle in kcal/mol.

    :param energies: Mapping of engine name to DFT energy in eV.
    :type energies: Mapping[str, float or None]
    :return: Mapping of engine name to offset in kcal/mol.
    :rtype: dict[str, float or None]
    """
    valid = [val for val in energies.values() if val is not None]
    if not valid:
        return {eng: None for eng in energies}
    lowest_ev = min(valid)
    return {
        eng: (
            round((val - lowest_ev) * EV_TO_KCAL, 2)
            if val is not None
            else None
        )
        for eng, val in energies.items()
    }


def build_geometry_item(
    engine: str,
    atoms: Atoms,
    sp_row: Mapping[str, Any] | None,
    de_kcal: float | None,
    pairs_count: int,
) -> EngineGeometry:
    """
    Construct one EngineGeometry record from structural and DFT data.

    :param engine: Engine name.
    :type engine: str
    :param atoms: Saddle geometry.
    :type atoms: Atoms
    :param sp_row: DFT single point record if available.
    :type sp_row: Mapping[str, Any] or None
    :param de_kcal: Energy above lowest DFT saddle in kcal/mol.
    :type de_kcal: float or None
    :param pairs_count: Number of shortest distances.
    :type pairs_count: int
    :return: EngineGeometry record.
    :rtype: EngineGeometry
    """
    is_ok = bool(sp_row and sp_row.get("ok"))
    forces_raw = sp_row.get("forces_ev_a") if is_ok and sp_row else None
    fmax_raw = sp_row.get("fmax_ev_a") if is_ok and sp_row else None
    energy_raw = sp_row.get("energy_ev") if is_ok and sp_row else None
    return EngineGeometry(
        engine=engine,
        pairs=shortest_distances(atoms, pairs_count),
        forces=atom_force_norms(atoms.get_chemical_symbols(), forces_raw),
        fmax=round(fmax_raw, 2) if fmax_raw is not None else None,
        energy_ev=energy_raw,
        de_lowest_kcal=de_kcal,
    )


def collect_geometry_rows(
    results_dir: Path,
    system: str,
    reaction_id: str,
    engines: Sequence[str],
    pairs_count: int,
) -> list[EngineGeometry]:
    """
    Gather geometry and DFT data for engines that have a saddle.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System identifier.
    :type system: str
    :param reaction_id: Reaction identifier.
    :type reaction_id: str
    :param engines: Engines to inspect.
    :type engines: Sequence[str]
    :param pairs_count: Number of shortest distances.
    :type pairs_count: int
    :return: List of EngineGeometry objects.
    :rtype: list[EngineGeometry]
    """
    frames = load_saddle_frames(results_dir, system, reaction_id, engines)
    sp_data = {
        eng: load_dft_sp_records(results_dir, system, eng) for eng in frames
    }
    uid = f"ts:{reaction_id}"
    energies = {
        eng: (
            sp_data[eng].get(uid, {}).get("energy_ev")
            if sp_data[eng].get(uid, {}).get("ok")
            else None
        )
        for eng in frames
    }
    offsets = compute_energy_offsets(energies)
    return [
        build_geometry_item(
            eng,
            frames[eng],
            sp_data[eng].get(uid),
            offsets.get(eng),
            pairs_count,
        )
        for eng in frames
    ]


def format_engine_line(item: EngineGeometry) -> str:
    """
    Format single summary line for one engine.

    :param item: EngineGeometry object.
    :type item: EngineGeometry
    :return: Formatted text line.
    :rtype: str
    """
    pair_str = ", ".join(f"{label} {dist:.3f}" for label, dist in item.pairs)
    force_str = ", ".join(
        f"{label} {norm:.2f}" for label, norm in item.forces
    )
    fmax_str = f"{item.fmax:.2f}" if item.fmax is not None else "n/a"
    de_str = (
        f"{item.de_lowest_kcal:.2f}"
        if item.de_lowest_kcal is not None
        else "n/a"
    )
    return (
        f"{item.engine}: pairs {pair_str} | forces {force_str} | "
        f"fmax {fmax_str} | energy above lowest {de_str} kcal/mol"
    )


def write_geometry_csv(
    path: Path, items: Sequence[EngineGeometry]
) -> None:
    """
    Write geometry and DFT comparison rows to CSV.

    :param path: Destination CSV file path.
    :type path: Path
    :param items: EngineGeometry records.
    :type items: Sequence[EngineGeometry]
    :return: None.
    :rtype: None
    """
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in items:
        row: dict[str, Any] = {
            "engine": item.engine,
            "energy_above_lowest_kcal": item.de_lowest_kcal,
            "dft_fmax_ev_a": item.fmax,
        }
        for label, norm in item.forces:
            row[f"force_{label}_ev_a"] = norm
        for idx, (label, dist) in enumerate(item.pairs, start=1):
            row[f"pair_{idx}_label"] = label
            row[f"pair_{idx}_dist_a"] = dist
        for label, dist in item.pairs:
            row[label] = dist
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    :return: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        required=True,
        help="System name.",
    )
    parser.add_argument(
        "--reaction",
        required=True,
        help="Reaction identifier (e.g. R006).",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=5,
        help="Number of shortest distances.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
        help="Engine name (can be repeated).",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        dest="engines_list",
        help="One or more engine names.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Inspect saddle geometry, atom forces, and energy offsets.

    :param argv: Optional command-line argument list.
    :type argv: Sequence[str] or None
    :return: Exit status code (0).
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    engines = args.engines_list or args.engines or list(DEFAULT_ENGINES)
    items = collect_geometry_rows(
        results_dir=args.results_dir,
        system=args.system,
        reaction_id=args.reaction,
        engines=engines,
        pairs_count=args.pairs,
    )
    if not items:
        return 0
    for item in items:
        print(format_engine_line(item))
    out_path = (
        args.data_dir
        / args.system
        / "checks"
        / f"saddle_geometry_{args.reaction}.csv"
    )
    write_geometry_csv(out_path, items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
