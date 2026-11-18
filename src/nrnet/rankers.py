"""
Product ranking algorithms (R0 to R4) for reaction networks.

Implements thermodynamic ranking (R0), summed barrier Dijkstra (R1),
bottleneck barrier Dijkstra (R2), path branching probability Dijkstra (R3),
and master equation kinetic simulation (R4).
"""

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import networkx as nx

from nrnet.kinetics import rate_matrix, simulate

__all__ = [
    "RANKERS",
    "BottleneckState",
    "RankSettings",
    "RankedProduct",
    "branch_weight",
    "collect_dijkstra_products",
    "collect_probabilities",
    "dijkstra_bottleneck",
    "dijkstra_branching",
    "evaluate_free_energy",
    "evaluate_kinetics_products",
    "init_bottleneck_state",
    "product_free_energy",
    "push_candidate",
    "rank_by_bottleneck",
    "rank_by_free_energy",
    "rank_by_kinetics",
    "rank_by_probability",
    "rank_by_summed_barriers",
    "reachable_products",
    "step_bottleneck",
]


@dataclass(frozen=True, slots=True)
class RankSettings:
    """
    Configuration settings for network product rankers.

    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :param t_obs_s: Observation time in seconds for kinetic simulations.
    :type t_obs_s: float
    :param dg_threshold_ev: Maximum reaction free energy in eV for R0.
    :type dg_threshold_ev: float
    """

    temperature_k: float
    t_obs_s: float
    dg_threshold_ev: float = 0.0


@dataclass(frozen=True, slots=True)
class RankedProduct:
    """
    Ranked reaction product with state key, numerical score, and path.

    :param state: Destination state identifier.
    :type state: str
    :param score: Numerical evaluation metric from ranking strategy.
    :type score: float
    :param path: Sequence of visited state identifiers from source.
    :type path: tuple[str, ...]
    """

    state: str
    score: float
    path: tuple[str, ...]


@dataclass(slots=True)
class BottleneckState:
    """
    Priority search state for minimax bottleneck Dijkstra.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param g_source_ev: Reactant source free energy.
    :type g_source_ev: float
    :param best: Map of minimum bottleneck cost per state.
    :type best: dict[str, float]
    :param paths: Map of optimal paths per state.
    :type paths: dict[str, tuple[str, ...]]
    :param queue: Priority queue for Dijkstra frontier.
    :type queue: list[Any]
    """

    graph: nx.DiGraph
    g_source_ev: float
    best: dict[str, float]
    paths: dict[str, tuple[str, ...]]
    queue: list[Any]


def reachable_products(graph: nx.DiGraph, source: str) -> set[str]:
    """
    Find all states reachable from source excluding source itself.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :return: Set of reachable state identifiers.
    :rtype: set[str]
    """
    return set(nx.descendants(graph, source))


def product_free_energy(
    graph: nx.DiGraph, g_source_ev: float, product: str
) -> RankedProduct:
    """
    Construct RankedProduct with reaction free energy score.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param g_source_ev: Reactant source free energy in eV.
    :type g_source_ev: float
    :param product: Product state identifier.
    :type product: str
    :return: Ranked product instance.
    :rtype: RankedProduct
    """
    return RankedProduct(
        product, graph.nodes[product]["g_ev"] - g_source_ev, ()
    )


def evaluate_free_energy(
    graph: nx.DiGraph, source: str, threshold_ev: float
) -> list[RankedProduct]:
    """
    Collect products below free energy threshold.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :param threshold_ev: Maximum allowed reaction free energy in eV.
    :type threshold_ev: float
    :return: List of evaluated products.
    :rtype: list[RankedProduct]
    """
    g_source_ev = graph.nodes[source]["g_ev"]
    items = [
        product_free_energy(graph, g_source_ev, product)
        for product in reachable_products(graph, source)
    ]
    return [item for item in items if item.score <= threshold_ev]


def rank_by_free_energy(
    graph: nx.DiGraph, source: str, settings: RankSettings
) -> list[RankedProduct]:
    """
    Rank reachable products by reaction free energy change (R0).

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :param settings: Ranking configuration settings.
    :type settings: RankSettings
    :return: Ranked products sorted ascending by dg_rxn.
    :rtype: list[RankedProduct]
    """
    products = evaluate_free_energy(graph, source, settings.dg_threshold_ev)
    return sorted(products, key=lambda item: item.score)


def collect_dijkstra_products(
    dists: dict[str, float], paths: dict[str, Any], targets: set[str]
) -> list[RankedProduct]:
    """
    Collect RankedProduct instances from Dijkstra distances and paths.

    :param dists: Shortest distances keyed by state identifier.
    :type dists: dict[str, float]
    :param paths: Optimal paths keyed by state identifier.
    :type paths: dict[str, Any]
    :param targets: Set of reachable target state identifiers.
    :type targets: set[str]
    :return: List of ranked products.
    :rtype: list[RankedProduct]
    """
    return [
        RankedProduct(product, dists[product], tuple(paths[product]))
        for product in targets
        if product in dists
    ]


def rank_by_summed_barriers(
    graph: nx.DiGraph, source: str, settings: RankSettings
) -> list[RankedProduct]:
    """
    Rank reachable products by summed activation barriers (R1).

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :param settings: Ranking configuration settings.
    :type settings: RankSettings
    :return: Ranked products sorted ascending by summed barrier.
    :rtype: list[RankedProduct]
    """
    dists, paths = nx.single_source_dijkstra(graph, source, weight="dg_act_ev")
    products = collect_dijkstra_products(
        dists, paths, reachable_products(graph, source)
    )
    return sorted(products, key=lambda item: item.score)


def push_candidate(
    candidate_cost: float,
    target: str,
    path: tuple[str, ...],
    state: BottleneckState,
) -> None:
    """
    Record updated bottleneck candidate if cost improves.

    :param candidate_cost: Candidate bottleneck barrier cost in eV.
    :type candidate_cost: float
    :param target: Successor state identifier.
    :type target: str
    :param path: Candidate path sequence from source.
    :type path: tuple[str, ...]
    :param state: Search state tracking costs and priority queue.
    :type state: BottleneckState
    :return: None.
    :rtype: None
    """
    if candidate_cost < state.best.get(target, math.inf):
        state.best[target] = candidate_cost
        state.paths[target] = (*path, target)
        heapq.heappush(
            state.queue,
            (candidate_cost, len(state.queue), target, state.paths[target]),
        )


def step_bottleneck(
    source: str, cost: float, path: tuple[str, ...], state: BottleneckState
) -> None:
    """
    Relax outgoing edges from current node using bottleneck cost.

    :param source: Current state being expanded.
    :type source: str
    :param cost: Known bottleneck cost to reach source.
    :type cost: float
    :param path: Optimal path reaching source.
    :type path: tuple[str, ...]
    :param state: Search state tracking costs and priority queue.
    :type state: BottleneckState
    :return: None.
    :rtype: None
    """
    graph, g_source_ev = state.graph, state.g_source_ev
    for target in graph.successors(source):
        candidate_cost = max(cost, graph[source][target]["ts_ev"] - g_source_ev)
        push_candidate(candidate_cost, target, path, state)


def init_bottleneck_state(graph: nx.DiGraph, source: str) -> BottleneckState:
    """
    Initialize search state for bottleneck Dijkstra.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :return: Initialized search state.
    :rtype: BottleneckState
    """
    g_source_ev = graph.nodes[source]["g_ev"]
    state = BottleneckState(
        graph, g_source_ev, {source: -math.inf}, {source: (source,)}, []
    )
    heapq.heappush(state.queue, (-math.inf, 0, source, (source,)))
    return state


def dijkstra_bottleneck(
    graph: nx.DiGraph, source: str
) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """
    Run bottleneck Dijkstra finding paths minimizing highest TS barrier.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :return: Tuple of minimum bottleneck costs and corresponding paths.
    :rtype: tuple[dict[str, float], dict[str, tuple[str, ...]]]
    """
    state = init_bottleneck_state(graph, source)
    while state.queue:
        cost, entry_idx, current_node, path = heapq.heappop(state.queue)
        del entry_idx
        if cost <= state.best.get(current_node, math.inf):
            step_bottleneck(current_node, cost, path, state)
    return state.best, state.paths


def rank_by_bottleneck(
    graph: nx.DiGraph, source: str, settings: RankSettings
) -> list[RankedProduct]:
    """
    Rank reachable products by minimum bottleneck TS height (R2).

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :param settings: Ranking configuration settings.
    :type settings: RankSettings
    :return: Ranked products sorted ascending by bottleneck cost.
    :rtype: list[RankedProduct]
    """
    best, paths = dijkstra_bottleneck(graph, source)
    products = collect_dijkstra_products(
        best, paths, reachable_products(graph, source)
    )
    return sorted(products, key=lambda item: item.score)


def branch_weight(
    graph: nx.DiGraph, source: str, target: str, edge_data: dict[str, Any]
) -> float:
    """
    Compute negative log branching probability weight for directed edge.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Predecessor state identifier.
    :type source: str
    :param target: Successor state identifier.
    :type target: str
    :param edge_data: Edge attributes dictionary containing rate_s.
    :type edge_data: dict[str, Any]
    :return: Non-negative edge weight.
    :rtype: float
    """
    del target
    total = sum(
        graph[source][neighbor]["rate_s"]
        for neighbor in graph.successors(source)
    )
    return -math.log(float(edge_data["rate_s"]) / total)


def dijkstra_branching(
    graph: nx.DiGraph, source: str
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Run Dijkstra on graph weighted by negative log branching probability.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :return: Distances and paths from source.
    :rtype: tuple[dict[str, float], dict[str, Any]]
    """
    return nx.single_source_dijkstra(
        graph,
        source,
        weight=lambda src_node, tgt_node, edge_dict: branch_weight(
            graph, src_node, tgt_node, edge_dict
        ),
    )


def collect_probabilities(
    graph: nx.DiGraph, source: str
) -> list[RankedProduct]:
    """
    Evaluate product probabilities via shortest path on -ln p.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :return: List of ranked product candidates.
    :rtype: list[RankedProduct]
    """
    distances, paths = dijkstra_branching(graph, source)
    return [
        RankedProduct(
            product, math.exp(-distances[product]), tuple(paths[product])
        )
        for product in reachable_products(graph, source)
        if product in distances
    ]


def rank_by_probability(
    graph: nx.DiGraph, source: str, settings: RankSettings
) -> list[RankedProduct]:
    """
    Rank reachable products by path transition probability (R3).

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :param settings: Ranking configuration settings.
    :type settings: RankSettings
    :return: Ranked products sorted descending by path probability score.
    :rtype: list[RankedProduct]
    """
    products = collect_probabilities(graph, source)
    return sorted(products, key=lambda item: item.score, reverse=True)


def evaluate_kinetics_products(
    graph: nx.DiGraph, source: str, t_obs_s: float
) -> list[RankedProduct]:
    """
    Simulate master equation and collect reachable product yields.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :param t_obs_s: Observation time in seconds.
    :type t_obs_s: float
    :return: List of evaluated kinetic products.
    :rtype: list[RankedProduct]
    """
    states = rate_matrix(graph)[1]
    populations = simulate(graph, source, [t_obs_s])[0]
    targets = reachable_products(graph, source)
    pairs = zip(states, populations, strict=True)
    return [
        RankedProduct(state, float(population), ())
        for state, population in pairs
        if state in targets
    ]


def rank_by_kinetics(
    graph: nx.DiGraph, source: str, settings: RankSettings
) -> list[RankedProduct]:
    """
    Rank reachable products by transient kinetic population (R4).

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :param settings: Ranking configuration settings.
    :type settings: RankSettings
    :return: Ranked products sorted descending by population at t_obs_s.
    :rtype: list[RankedProduct]
    """
    products = evaluate_kinetics_products(graph, source, settings.t_obs_s)
    return sorted(products, key=lambda item: item.score, reverse=True)


RANKERS: dict[
    str, Callable[[nx.DiGraph, str, RankSettings], list[RankedProduct]]
] = {
    "R0": rank_by_free_energy,
    "R1": rank_by_summed_barriers,
    "R2": rank_by_bottleneck,
    "R3": rank_by_probability,
    "R4": rank_by_kinetics,
}
