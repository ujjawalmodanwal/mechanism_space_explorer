"""
Unit tests for reaction network ranking, JSONL export, and settings.

Validates assembling networks across thermochemical temperatures, running
rankers R0 through R4, writing ranks.jsonl and network.jsonl summary files,
and parsing ranking configuration parameters from search YAML.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from nrnet.nanoreactor.candidates import CandidateSet
from nrnet.ranking import (
    NETWORK_FILE,
    RANKS_FILE,
    load_rank_settings,
    make_rank_dir,
    rank_candidates,
)
from nrnet.store import RefinePaths, read_jsonl


def write_search_yaml(path: Path) -> Path:
    """
    Write standard search YAML configuration file into destination path.

    :param path: Destination file path for YAML configuration.
    :type path: Path
    :return: Resolved path to written search YAML.
    :rtype: Path
    """
    data = {
        "thermo": {"temperatures_k": [298.15, 600.0]},
        "rank": {"t_obs_s": 3600.0, "dg_threshold_ev": 0.0},
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return path


def assert_rank_rows(rows: list[dict[str, Any]]) -> None:
    """
    Assert rank rows contain expected rankers and positions starting at 1.

    :param rows: Deserialized rows from ranks JSONL file.
    :type rows: list[dict[str, Any]]
    :return: None.
    :rtype: None
    """
    expected_rankers = {"R0", "R1", "R2", "R3", "R4"}
    by_temp: dict[float, set[str]] = {298.15: set(), 600.0: set()}
    for row in rows:
        by_temp[row["temperature_k"]].add(row["ranker"])
        assert row["position"] >= 1
    assert by_temp[298.15] == expected_rankers
    assert by_temp[600.0] == expected_rankers
    assert all(row["position"] == 1 for row in rows)


def assert_network_rows(rows: list[dict[str, Any]]) -> None:
    """
    Assert network rows contain 2 temperature records with topology metrics.

    :param rows: Deserialized rows from network JSONL file.
    :type rows: list[dict[str, Any]]
    :return: None.
    :rtype: None
    """
    assert len(rows) == 2
    temps = [row["temperature_k"] for row in rows]
    assert temps == [298.15, 600.0]
    for row in rows:
        assert row["source"] == "A + B + S"
        assert row["states"] > 0
        assert row["reactions"] == 1
        assert "no_ts" in row["dropped"]


def test_make_rank_dir(tmp_path: Path) -> None:
    """
    Verify make_rank_dir constructs standardized output directory path.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    out_dir = make_rank_dir(tmp_path, "toy", "xtb")
    assert out_dir == tmp_path / "rank" / "toy" / "xtb"


def test_rank_candidates_writes_outputs(
    tmp_path: Path,
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify rank_candidates writes ranks and network summary JSONL files.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :param fake_refine_results: Fake candidate set and refinement paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    search_path = write_search_yaml(tmp_path / "search.yaml")
    out_dir = make_rank_dir(tmp_path, "toy", "xtb")
    rank_candidates(candidate_set, paths, search_path, out_dir)
    assert (out_dir / RANKS_FILE).is_file()
    assert (out_dir / NETWORK_FILE).is_file()
    assert_network_rows(read_jsonl(out_dir / NETWORK_FILE))
    assert_rank_rows(read_jsonl(out_dir / RANKS_FILE))


def test_rank_candidates_rewrites_files(
    tmp_path: Path,
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify running rank_candidates twice rewrites instead of appending.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :param fake_refine_results: Fake candidate set and refinement paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    search_path = write_search_yaml(tmp_path / "search.yaml")
    out_dir = make_rank_dir(tmp_path, "toy", "xtb")
    rank_candidates(candidate_set, paths, search_path, out_dir)
    ranks_1 = read_jsonl(out_dir / RANKS_FILE)
    net_1 = read_jsonl(out_dir / NETWORK_FILE)
    rank_candidates(candidate_set, paths, search_path, out_dir)
    assert read_jsonl(out_dir / RANKS_FILE) == ranks_1
    assert read_jsonl(out_dir / NETWORK_FILE) == net_1


def test_load_rank_settings_missing_key_raises(tmp_path: Path) -> None:
    """
    Verify load_rank_settings raises KeyError when t_obs_s is missing.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cfg_file = tmp_path / "incomplete_search.yaml"
    with cfg_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"rank": {"dg_threshold_ev": 0.0}}, handle)
    with pytest.raises(KeyError):
        load_rank_settings(cfg_file, 298.15)
