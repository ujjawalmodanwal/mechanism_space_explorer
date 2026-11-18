"""
Unit tests for persistent store helpers, JSONL serialization, and paths.

Verifies reading and appending JSON Lines files, deduplicating records by
identifier to preserve the latest state, formatting temperature labels,
and constructing standardized refinement paths.
"""

import csv
from pathlib import Path

from ase import Atoms

from nrnet.store import (
    append_extxyz,
    append_jsonl,
    latest_rows,
    make_refine_paths,
    read_jsonl,
    temperature_label,
    write_csv,
    write_jsonl,
)


def test_read_jsonl_missing_file_returns_empty(tmp_path: Path) -> None:
    """
    Verify reading a nonexistent JSONL file returns an empty list.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    missing_file = tmp_path / "nonexistent.jsonl"
    assert read_jsonl(missing_file) == []


def test_append_and_read_jsonl_roundtrip(tmp_path: Path) -> None:
    """
    Verify appending and reading JSONL round-trips two rows in order.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    path = tmp_path / "records.jsonl"
    row1 = {"id": "R001", "status": "confirmed", "energy": -12.5}
    row2 = {"id": "R002", "status": "rejected", "energy": -10.0}
    append_jsonl(path, row1)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")
    append_jsonl(path, row2)
    rows = read_jsonl(path)
    assert rows == [row1, row2]


def test_latest_rows_keeps_last_row_per_id(tmp_path: Path) -> None:
    """
    Verify latest_rows maps identifier to the most recent record.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    path = tmp_path / "resume.jsonl"
    append_jsonl(path, {"reaction_id": "R001", "step": 1})
    append_jsonl(path, {"reaction_id": "R002", "step": 1})
    append_jsonl(path, {"reaction_id": "R001", "step": 2})
    latest = latest_rows(path, "reaction_id")
    assert latest == {
        "R001": {"reaction_id": "R001", "step": 2},
        "R002": {"reaction_id": "R002", "step": 1},
    }


def test_temperature_label() -> None:
    """
    Verify temperature_label converts temperatures to canonical string keys.

    :return: None.
    :rtype: None
    """
    assert temperature_label(298.15) == "298.15"
    assert temperature_label(600) == "600.0"
    assert temperature_label(600.0) == "600.0"


def test_make_refine_paths_and_append_extxyz(tmp_path: Path) -> None:
    """
    Verify make_refine_paths constructs standard paths and append_extxyz writes.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    paths = make_refine_paths(tmp_path, "sys1", "xtb")
    expected_base = tmp_path / "refine" / "sys1" / "xtb"
    assert paths.out_dir == expected_base
    assert paths.species_jsonl == expected_base / "species.jsonl"
    mol = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    append_extxyz(paths.species_extxyz, mol)
    assert paths.species_extxyz.is_file()


def test_write_jsonl_replaces_existing_content(tmp_path: Path) -> None:
    """
    Verify write_jsonl replaces existing file content on a second write.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    target = tmp_path / "subdir" / "data.jsonl"
    first_rows = [{"id": "A", "val": 1}, {"id": "B", "val": 2}]
    write_jsonl(target, first_rows)
    assert read_jsonl(target) == first_rows
    second_rows = [{"id": "C", "val": 3}]
    write_jsonl(target, second_rows)
    assert read_jsonl(target) == second_rows


def test_write_csv_roundtrip_and_replaces(tmp_path: Path) -> None:
    """
    Verify write_csv writes header and rows, making folders and replacing.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    target = tmp_path / "deep" / "nested" / "table.csv"
    first_rows = [
        {"col1": "val1", "col2": "10"},
        {"col1": "val2", "col2": "20"},
    ]
    write_csv(target, first_rows)
    assert target.is_file()
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = list(csv.DictReader(handle))
    assert reader == first_rows

    second_rows = [{"other": "new_val"}]
    write_csv(target, second_rows)
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader2 = list(csv.DictReader(handle))
    assert reader2 == second_rows


def test_write_csv_empty_rows_creates_headerless_file(tmp_path: Path) -> None:
    """
    Verify write_csv with empty rows creates an empty file without header.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    target = tmp_path / "empty_dir" / "empty.csv"
    write_csv(target, [])
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == ""
