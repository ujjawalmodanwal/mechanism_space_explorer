"""
Unit tests for reaction network ranking, JSONL export, and settings.

Validates assembling networks across thermochemical temperatures, running
rankers R0 through R4, writing ranks.jsonl and network.jsonl summary files,
and parsing ranking configuration parameters from search YAML.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import yaml

from nrnet.nanoreactor.candidate_store import write_candidate_files
from nrnet.nanoreactor.candidates import CandidateSet
from nrnet.network import ReactionBarrier, build_network
from nrnet.rankers import RankSettings, rank_by_kinetics
from nrnet.ranking import (
    NETWORK_FILE,
    RANKS_FILE,
    graph_without_barrierless,
    load_rank_settings,
    make_rank_dir,
    rank,
    rank_candidates,
)
from nrnet.refine import load_thermo_temperatures
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
        "rank": {"t_obs_s": 3600.0, "dg_threshold_ev": 0.0},
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return path


def write_system_yaml(
    path: Path,
    temperatures_k: Sequence[float] = (298.15, 600.0),
) -> Path:
    """
    Write system YAML configuration file with ranking temperatures.

    :param path: Destination file path for YAML configuration.
    :type path: Path
    :param temperatures_k: Ranking temperatures in Kelvin.
    :type temperatures_k: Sequence[float]
    :return: Resolved path to written system YAML.
    :rtype: Path
    """
    data = {
        "ranking": {"temperatures_k": list(temperatures_k)},
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return path


def assert_rank_rows(
    rows: list[dict[str, Any]],
    expected_temps: Sequence[float] = (298.15, 600.0),
) -> None:
    """
    Assert rank rows contain expected rankers and positions starting at 1.

    :param rows: Deserialized rows from ranks JSONL file.
    :type rows: list[dict[str, Any]]
    :param expected_temps: Expected evaluation temperatures in Kelvin.
    :type expected_temps: Sequence[float]
    :return: None.
    :rtype: None
    """
    expected_rankers = {"R0", "R1", "R2", "R3", "R4", "R4_no_barrierless"}
    by_temp: dict[float, set[str]] = {t: set() for t in expected_temps}
    for row in rows:
        by_temp[row["temperature_k"]].add(row["ranker"])
        assert row["position"] >= 1
    for t in expected_temps:
        assert by_temp[t] == expected_rankers
    assert all(row["position"] == 1 for row in rows)


def assert_network_rows(
    rows: list[dict[str, Any]],
    expected_temps: Sequence[float] = (298.15, 600.0),
) -> None:
    """
    Assert network rows contain expected temperature records with topology.

    :param rows: Deserialized rows from network JSONL file.
    :type rows: list[dict[str, Any]]
    :param expected_temps: Expected evaluation temperatures in Kelvin.
    :type expected_temps: Sequence[float]
    :return: None.
    :rtype: None
    """
    assert len(rows) == len(expected_temps)
    temps = [row["temperature_k"] for row in rows]
    assert temps == list(expected_temps)
    for row in rows:
        assert row["source"] == "A + B + S"
        assert row["states"] > 0
        assert row["reactions"] == 1
        assert row["barrierless"] == 0
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
    system_path = write_system_yaml(tmp_path / "system.yaml")
    out_dir = make_rank_dir(tmp_path, "toy", "xtb")
    rank_candidates(
        candidate_set, paths, search_path, out_dir, system_yaml=system_path
    )
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
    system_path = write_system_yaml(tmp_path / "system.yaml")
    out_dir = make_rank_dir(tmp_path, "toy", "xtb")
    rank_candidates(
        candidate_set, paths, search_path, out_dir, system_yaml=system_path
    )
    ranks_1 = read_jsonl(out_dir / RANKS_FILE)
    net_1 = read_jsonl(out_dir / NETWORK_FILE)
    rank_candidates(
        candidate_set, paths, search_path, out_dir, system_yaml=system_path
    )
    assert read_jsonl(out_dir / RANKS_FILE) == ranks_1
    assert read_jsonl(out_dir / NETWORK_FILE) == net_1


def test_rank_end_to_end(
    tmp_path: Path,
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify rank coordinates candidate loading, refine paths, and output files.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :param fake_refine_results: Fake candidate set and refinement paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    assert paths.out_dir.is_dir()
    cand_dir = tmp_path / "candidates"
    toy_cand = cand_dir / "toy"
    toy_cand.mkdir(parents=True, exist_ok=True)
    write_candidate_files(candidate_set, toy_cand)
    search_path = write_search_yaml(tmp_path / "search.yaml")
    system_path = write_system_yaml(tmp_path / "system.yaml")
    out_dir = rank("toy", "xtb", cand_dir, tmp_path, search_path, system_path)
    assert (out_dir / RANKS_FILE).is_file()
    assert (out_dir / NETWORK_FILE).is_file()


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


def test_load_ranking_temperatures_one_and_two_values(tmp_path: Path) -> None:
    """
    Verify temperatures loaded correctly with one and with two values.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    one_path = tmp_path / "one.yaml"
    two_path = tmp_path / "two.yaml"
    write_system_yaml(one_path, [1000.0])
    write_system_yaml(two_path, [298.15, 493.15])
    assert load_thermo_temperatures(one_path) == [1000.0]
    assert load_thermo_temperatures(two_path) == [298.15, 493.15]


def test_load_ranking_temperatures_missing_key_raises(tmp_path: Path) -> None:
    """
    Verify missing ranking.temperatures_k raises ValueError naming the file.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    missing_file = tmp_path / "missing_key.yaml"
    with missing_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"ranking": {}}, handle)
    with pytest.raises(ValueError) as exc_info:
        load_thermo_temperatures(missing_file)
    assert missing_file.name in str(exc_info.value)


def make_barrierless_toy_network() -> nx.DiGraph:
    """
    Construct toy reaction network with one barrierless elementary step.

    :return: Reaction network graph with source attribute set to 'R'.
    :rtype: nx.DiGraph
    """
    energies = {"R": 0.0, "A": -0.1, "P1": -0.2}
    barriers = [
        ReactionBarrier("R", "A", 0.5, barrierless=False),
        ReactionBarrier("A", "P1", 0.0, barrierless=True),
    ]
    graph = build_network(energies, barriers, 298.15)
    graph.graph["source"] = "R"
    return graph


def kinetic_yield_map(
    graph: nx.DiGraph,
    source: str,
    settings: RankSettings,
) -> dict[str, float]:
    """
    Compute mapping of state to kinetic yield score.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Initial reactant source state identifier.
    :type source: str
    :param settings: Ranking configuration settings.
    :type settings: RankSettings
    :return: Mapping of state identifiers to kinetic yield scores.
    :rtype: dict[str, float]
    """
    return {
        item.state: item.score
        for item in rank_by_kinetics(graph, source, settings)
    }


def test_toy_network_barrierless_channel_and_r4_no_barrierless() -> None:
    """
    Verify barrierless arrow gets max(G_A, G_B) and R4_no_barrierless loses it.

    :return: None.
    :rtype: None
    """
    graph = make_barrierless_toy_network()
    assert graph["A"]["P1"]["ts_ev"] == pytest.approx(-0.1)
    assert graph["P1"]["A"]["ts_ev"] == pytest.approx(-0.1)
    assert graph["A"]["P1"]["barrierless"] is True
    settings = RankSettings(temperature_k=298.15, t_obs_s=3600.0)
    prods_r4 = kinetic_yield_map(graph, "R", settings)
    assert "P1" in prods_r4 and prods_r4["P1"] > 0.0
    no_barr_graph = graph_without_barrierless(graph)
    prods_nb = kinetic_yield_map(no_barr_graph, "R", settings)
    assert "P1" not in prods_nb
