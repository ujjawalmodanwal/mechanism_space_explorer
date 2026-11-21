"""
Reaction network construction and product ranking at thermo temperatures.

Builds directed reaction networks for refined chemical engines at all standard
thermochemical temperatures, evaluates rankers R0 through R4, and serializes
ranked product sequences and network summary statistics to JSON Lines files.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from nrnet.assemble import NetworkInputs, build_engine_network
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import CandidateSet
from nrnet.rankers import (
    RANKERS,
    RankedProduct,
    RankSettings,
    rank_by_kinetics,
)
from nrnet.refine import load_thermo_temperatures
from nrnet.store import RefinePaths, make_refine_paths, write_jsonl

__all__ = [
    "NETWORK_FILE",
    "RANKS_FILE",
    "count_barrierless_edges",
    "evaluate_network_temperatures",
    "graph_without_barrierless",
    "load_rank_settings",
    "make_rank_dir",
    "network_row",
    "rank",
    "rank_candidates",
    "rank_rows",
    "ranked_product_row",
]

RANKS_FILE: str = "ranks.jsonl"
NETWORK_FILE: str = "network.jsonl"


def load_rank_settings(
    path: str | Path,
    temperature_k: float,
) -> RankSettings:
    """
    Load ranking configuration parameters for evaluation temperature.

    :param path: Path to search YAML configuration file.
    :type path: str or Path
    :param temperature_k: Evaluation temperature in Kelvin.
    :type temperature_k: float
    :return: Configured RankSettings instance.
    :rtype: RankSettings
    :raises KeyError: If the rank section or required keys are missing.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    rank_cfg = data["rank"]
    return RankSettings(
        temperature_k=float(temperature_k),
        t_obs_s=float(rank_cfg["t_obs_s"]),
        dg_threshold_ev=float(rank_cfg["dg_threshold_ev"]),
    )


def make_rank_dir(
    results_dir: str | Path,
    system: str,
    engine: str,
) -> Path:
    """
    Construct standardized ranking output directory path.

    :param results_dir: Base results directory path.
    :type results_dir: str or Path
    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :return: Destination Path for ranking results.
    :rtype: Path
    """
    return Path(results_dir) / "rank" / system / engine


def graph_without_barrierless(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Construct a copy of reaction network graph omitting barrierless edges.

    :param graph: Assembled reaction network graph.
    :type graph: nx.DiGraph
    :return: Graph copy containing only non-barrierless edges.
    :rtype: nx.DiGraph
    """
    copy_graph = graph.copy()
    barrierless_edges = [
        (src, tgt)
        for src, tgt, data in copy_graph.edges(data=True)
        if data.get("barrierless", False)
    ]
    copy_graph.remove_edges_from(barrierless_edges)
    return copy_graph


def count_barrierless_edges(graph: nx.DiGraph) -> int:
    """
    Count directed edges carrying the barrierless attribute in reaction network.

    :param graph: Assembled reaction network graph.
    :type graph: nx.DiGraph
    :return: Number of directed edges with barrierless=True.
    :rtype: int
    """
    return sum(
        1
        for _, _, data in graph.edges(data=True)
        if data.get("barrierless", False)
    )


def ranked_product_row(
    item: RankedProduct,
    ranker: str,
    position: int,
    temperature_k: float,
) -> dict[str, Any]:
    """
    Serialize RankedProduct instance into dictionary record for JSONL storage.

    :param item: Ranked product result instance.
    :type item: RankedProduct
    :param ranker: Name of the ranking strategy.
    :type ranker: str
    :param position: Rank position starting at 1.
    :type position: int
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: Formatted dictionary record.
    :rtype: dict[str, Any]
    """
    return {
        "temperature_k": float(temperature_k),
        "ranker": ranker,
        "position": position,
        "state": item.state,
        "score": item.score,
        "path": list(item.path),
    }


def rank_rows(
    graph: nx.DiGraph,
    settings: RankSettings,
) -> list[dict[str, Any]]:
    """
    Evaluate all rankers on reaction network and collect product records.

    :param graph: Assembled reaction network graph.
    :type graph: nx.DiGraph
    :param settings: Ranking parameters including temperature and thresholds.
    :type settings: RankSettings
    :return: List of serialized ranked product records.
    :rtype: list[dict[str, Any]]
    """
    source = graph.graph["source"]
    temp = settings.temperature_k
    rows: list[dict[str, Any]] = []
    for name, ranker in RANKERS.items():
        for idx, item in enumerate(ranker(graph, source, settings), start=1):
            rows.append(ranked_product_row(item, name, idx, temp))
    sub = graph_without_barrierless(graph)
    for idx, item in enumerate(
        rank_by_kinetics(sub, source, settings), start=1
    ):
        rows.append(ranked_product_row(item, "R4_no_barrierless", idx, temp))
    return rows


def network_row(
    graph: nx.DiGraph,
    inputs: NetworkInputs,
    temperature_k: float,
) -> dict[str, Any]:
    """
    Summarize reaction network topology, reaction counts, and drop metrics.

    :param graph: Assembled reaction network graph.
    :type graph: nx.DiGraph
    :param inputs: Assembled network input parameters and drop statistics.
    :type inputs: NetworkInputs
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: Summary dictionary of network metrics.
    :rtype: dict[str, Any]
    """
    return {
        "temperature_k": float(temperature_k),
        "source": inputs.source,
        "states": graph.number_of_nodes(),
        "reactions": len(inputs.barriers),
        "clamped": int(graph.graph["clamped"]),
        "barrierless": count_barrierless_edges(graph),
        "dropped": dict(inputs.dropped),
    }


def evaluate_network_temperatures(
    candidate_set: CandidateSet,
    paths: RefinePaths,
    search_path: str | Path,
    temperatures: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Evaluate networks and ranking rows across specified temperatures.

    :param candidate_set: Frozen candidate set extracted from nanoreactor.
    :type candidate_set: CandidateSet
    :param paths: Refinement result paths for species and reactions.
    :type paths: RefinePaths
    :param search_path: Path to search YAML configuration file.
    :type search_path: str or Path
    :param temperatures: Evaluation temperatures in Kelvin.
    :type temperatures: Sequence[float]
    :return: Tuple of rank data rows and network summary data rows.
    :rtype: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    """
    rank_data: list[dict[str, Any]] = []
    net_data: list[dict[str, Any]] = []
    for temp in temperatures:
        graph, inputs = build_engine_network(candidate_set, paths, temp)
        settings = load_rank_settings(search_path, temp)
        net_data.append(network_row(graph, inputs, temp))
        rank_data.extend(rank_rows(graph, settings))
    return rank_data, net_data


def rank_candidates(
    candidate_set: CandidateSet,
    paths: RefinePaths,
    search_path: str | Path,
    out_dir: Path,
    system_yaml: str | Path,
) -> Path:
    """
    Assemble networks, evaluate rankers across temperatures, and write JSONL.

    :param candidate_set: Frozen candidate set extracted from nanoreactor.
    :type candidate_set: CandidateSet
    :param paths: Refinement result paths for species and reactions.
    :type paths: RefinePaths
    :param search_path: Path to search YAML configuration file.
    :type search_path: str or Path
    :param out_dir: Destination directory for ranking outputs.
    :type out_dir: Path
    :param system_yaml: Path to system YAML configuration file.
    :type system_yaml: str or Path
    :return: Output directory path.
    :rtype: Path
    """
    temperatures = load_thermo_temperatures(system_yaml)
    rank_data, net_data = evaluate_network_temperatures(
        candidate_set, paths, search_path, temperatures
    )
    write_jsonl(out_dir / RANKS_FILE, rank_data)
    write_jsonl(out_dir / NETWORK_FILE, net_data)
    return out_dir


def rank(
    system: str,
    engine: str,
    candidates_dir: str | Path,
    results_dir: str | Path,
    search_path: str | Path,
    system_yaml: str | Path,
) -> Path:
    """
    Execute end-to-end network ranking for refined system and engine.

    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param candidates_dir: Root directory of frozen candidate sets.
    :type candidates_dir: str or Path
    :param results_dir: Root results directory.
    :type results_dir: str or Path
    :param search_path: Path to search YAML configuration file.
    :type search_path: str or Path
    :param system_yaml: Path to system YAML configuration file.
    :type system_yaml: str or Path
    :return: Path to destination ranking output directory.
    :rtype: Path
    """
    candidate_set = load_candidates(Path(candidates_dir) / system)
    paths = make_refine_paths(results_dir, system, engine)
    out_dir = make_rank_dir(results_dir, system, engine)
    return rank_candidates(
        candidate_set,
        paths,
        search_path,
        out_dir,
        system_yaml,
    )
