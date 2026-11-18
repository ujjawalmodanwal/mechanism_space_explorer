"""
Persistent storage and serialization helpers for refinement outputs.

Manages filesystem paths, JSON Lines serialization, and extended XYZ structure
caching for species and transition state refinement results.
"""

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.io
from ase import Atoms

__all__ = [
    "RefinePaths",
    "append_extxyz",
    "append_jsonl",
    "latest_rows",
    "make_refine_paths",
    "read_jsonl",
    "temperature_label",
    "write_csv",
    "write_jsonl",
]


@dataclass(frozen=True, slots=True)
class RefinePaths:
    """
    Filesystem destination paths for refined outputs.

    :param out_dir: Base directory for refinement results.
    :type out_dir: Path
    :param species_jsonl: Path to species JSON Lines file.
    :type species_jsonl: Path
    :param species_extxyz: Path to relaxed species extxyz file.
    :type species_extxyz: Path
    :param reactions_jsonl: Path to candidate reactions JSON Lines file.
    :type reactions_jsonl: Path
    :param ts_extxyz: Path to transition state geometries extxyz file.
    :type ts_extxyz: Path
    :param calls_csv: Path to engine call performance log.
    :type calls_csv: Path
    """

    out_dir: Path
    species_jsonl: Path
    species_extxyz: Path
    reactions_jsonl: Path
    ts_extxyz: Path
    calls_csv: Path


def make_refine_paths(
    results_dir: str | Path,
    system: str,
    engine: str,
) -> RefinePaths:
    """
    Construct standardized refinement output paths.

    :param results_dir: Base results directory path.
    :type results_dir: str or Path
    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :return: RefinePaths instance with resolved destination paths.
    :rtype: RefinePaths
    """
    base = Path(results_dir) / "refine" / system / engine
    return RefinePaths(
        out_dir=base,
        species_jsonl=base / "species.jsonl",
        species_extxyz=base / "species.extxyz",
        reactions_jsonl=base / "reactions.jsonl",
        ts_extxyz=base / "ts.extxyz",
        calls_csv=base / "calls.csv",
    )


def append_jsonl(path: Path, data: Mapping[str, Any]) -> None:
    """
    Append JSON-serialized mapping as a single line and flush immediately.

    :param path: Destination file path.
    :type path: Path
    :param data: Data mapping to serialize.
    :type data: Mapping[str, Any]
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data) + "\n")
        handle.flush()


def append_extxyz(path: Path, atoms: Atoms) -> None:
    """
    Append atomic configuration frame to extxyz file.

    :param path: Destination extxyz file path.
    :type path: Path
    :param atoms: Atomic configuration to serialize.
    :type atoms: Atoms
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(path, atoms, format="extxyz", append=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read all rows from a JSON Lines file in order.

    :param path: Path to the JSONL file.
    :type path: Path
    :return: List of deserialized JSON records, or empty if nonexistent.
    :rtype: list[dict[str, Any]]
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def latest_rows(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    """
    Map record identifier to its latest row from a JSON Lines file.

    :param path: Path to the JSONL file.
    :type path: Path
    :param id_field: Key name holding the unique record identifier.
    :type id_field: str
    :return: Mapping of identifier to most recent deserialized row.
    :rtype: dict[str, dict[str, Any]]
    """
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        rows[str(row[id_field])] = row
    return rows


def temperature_label(temperature_k: float) -> str:
    """
    Format temperature into standard key for thermochemical free energy maps.

    Single source of truth that turns a temperature into the key of the g_ev
    maps in species.jsonl and reactions.jsonl.

    :param temperature_k: Temperature in Kelvin.
    :type temperature_k: float
    :return: Canonical string representation of float temperature.
    :rtype: str
    """
    return str(float(temperature_k))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """
    Write rows as JSON Lines, replacing destination file completely.

    :param path: Destination file path.
    :type path: Path
    :param rows: Data mappings to serialize.
    :type rows: Iterable[Mapping[str, Any]]
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Write sequence of mappings as CSV, replacing destination completely.

    :param path: Destination file path.
    :type path: Path
    :param rows: Data mappings to serialize.
    :type rows: Sequence[Mapping[str, Any]]
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
