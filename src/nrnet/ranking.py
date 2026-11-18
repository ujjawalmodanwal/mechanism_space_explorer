"""
Reaction network construction and product ranking at thermo temperatures.

Builds directed reaction networks for refined chemical engines at all standard
thermochemical temperatures, evaluates rankers R0 through R4, and serializes
ranked product sequences and network summary statistics to JSON Lines files.
"""

from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from nrnet.assemble import NetworkInputs, build_engine_network
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import CandidateSet
from nrnet.rankers import RANKERS, RankSettings
from nrnet.refine import load_thermo_temperatures
from nrnet.store import RefinePaths, make_refine_paths, write_jsonl

__all__ = [
    "NETWORK_FILE",
    "RANKS_FILE",
    "load_rank_settings",
    "make_rank_dir",
    "network_row",
    "rank",
    "rank_candidates",
    "rank_rows",
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
    rows: list[dict[str, Any]] = []
    for name, ranker in RANKERS.items():
        for idx, item in enumerate(ranker(graph, source, settings), start=1):
            rows.append(
                {
                    "temperature_k": settings.temperature_k,
                    "ranker": name,
                    "position": idx,
                    "state": item.state,
                    "score": item.score,
                    "path": list(item.path),
                }
            )
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
        "dropped": dict(inputs.dropped),
    }


def rank_candidates(
    candidate_set: CandidateSet,
    paths: RefinePaths,
    search_path: str | Path,
    out_dir: Path,
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
    :return: Output directory path.
    :rtype: Path
    """
    temperatures = load_thermo_temperatures(search_path)
    rank_data: list[dict[str, Any]] = []
    net_data: list[dict[str, Any]] = []
    for temp in temperatures:
        graph, inputs = build_engine_network(candidate_set, paths, temp)
        settings = load_rank_settings(search_path, temp)
        net_data.append(network_row(graph, inputs, temp))
        rank_data.extend(rank_rows(graph, settings))
    write_jsonl(out_dir / RANKS_FILE, rank_data)
    write_jsonl(out_dir / NETWORK_FILE, net_data)
    return out_dir


def rank(
    system: str,
    engine: str,
    candidates_dir: str | Path,
    results_dir: str | Path,
    search_path: str | Path,
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
    :return: Path to destination ranking output directory.
    :rtype: Path
    """
    candidate_set = load_candidates(Path(candidates_dir) / system)
    paths = make_refine_paths(results_dir, system, engine)
    out_dir = make_rank_dir(results_dir, system, engine)
    return rank_candidates(candidate_set, paths, search_path, out_dir)
