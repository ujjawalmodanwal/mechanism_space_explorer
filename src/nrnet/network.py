"""
Reaction network construction and Eyring transition state theory rates.

Builds directed graphs where nodes represent reaction states with Gibbs free
energies and edges represent elementary steps with forward activation barriers
and Eyring rate constants.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import ase.units as units
import networkx as nx
import numpy as np

__all__ = [
    "KCAL_MOL_EV",
    "ReactionBarrier",
    "add_reaction_edge",
    "apply_barrier",
    "build_network",
    "eyring_rate",
    "populate_nodes",
]

KCAL_MOL_EV: float = units.kcal / units.mol


@dataclass(frozen=True, slots=True)
class ReactionBarrier:
    """
    Activation barrier specification for reaction between two states.

    :param state_a: Reactant state identifier.
    :type state_a: str
    :param state_b: Product state identifier.
    :type state_b: str
    :param dg_act_ev: Forward activation barrier from state_a in eV.
    :type dg_act_ev: float
    """

    state_a: str
    state_b: str
    dg_act_ev: float


def eyring_rate(dg_act_ev: float, temperature_k: float) -> float:
    """
    Calculate Eyring transition state theory rate constant.

    :param dg_act_ev: Forward activation barrier in eV.
    :type dg_act_ev: float
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: First-order rate constant in s^-1.
    :rtype: float
    """
    prefactor = (units._k * temperature_k) / units._hplanck
    thermal_energy = units.kB * temperature_k
    return float(prefactor * np.exp(-dg_act_ev / thermal_energy))


def add_reaction_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    ts_ev: float,
    temperature_k: float,
) -> None:
    """
    Add single directed reaction edge with kinetic rate attributes.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant node key.
    :type source: str
    :param target: Product node key.
    :type target: str
    :param ts_ev: Transition state energy in eV.
    :type ts_ev: float
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: None.
    :rtype: None
    """
    dg_act_ev = ts_ev - graph.nodes[source]["g_ev"]
    dg_rxn_ev = graph.nodes[target]["g_ev"] - graph.nodes[source]["g_ev"]
    rate_constant = eyring_rate(dg_act_ev, temperature_k)
    graph.add_edge(
        source,
        target,
        ts_ev=ts_ev,
        dg_act_ev=dg_act_ev,
        dg_rxn_ev=dg_rxn_ev,
        rate_s=rate_constant,
    )


def apply_barrier(
    graph: nx.DiGraph, barrier: ReactionBarrier, temperature_k: float
) -> int:
    """
    Add bidirectional edges for barrier and return 1 if clamped else 0.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param barrier: Reaction barrier specification.
    :type barrier: ReactionBarrier
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: 1 if clamped else 0.
    :rtype: int
    """
    g_state_a_ev = graph.nodes[barrier.state_a]["g_ev"]
    g_state_b_ev = graph.nodes[barrier.state_b]["g_ev"]
    ts_raw_ev = g_state_a_ev + float(barrier.dg_act_ev)
    ts_ev = max(ts_raw_ev, g_state_a_ev, g_state_b_ev)
    add_reaction_edge(
        graph, barrier.state_a, barrier.state_b, ts_ev, temperature_k
    )
    add_reaction_edge(
        graph, barrier.state_b, barrier.state_a, ts_ev, temperature_k
    )
    return int(ts_ev > ts_raw_ev)


def populate_nodes(graph: nx.DiGraph, state_g_ev: Mapping[str, float]) -> None:
    """
    Populate graph nodes with Gibbs free energies.

    :param graph: Reaction network graph to populate.
    :type graph: nx.DiGraph
    :param state_g_ev: Mapping of state identifiers to free energies in eV.
    :type state_g_ev: Mapping[str, float]
    :return: None.
    :rtype: None
    """
    for state, g_ev in state_g_ev.items():
        graph.add_node(state, g_ev=float(g_ev))


def build_network(
    state_g_ev: Mapping[str, float],
    barriers: Sequence[ReactionBarrier],
    temperature_k: float,
) -> nx.DiGraph:
    """
    Construct reaction network with node free energies and kinetic rates.

    :param state_g_ev: Mapping of state identifiers to free energies in eV.
    :type state_g_ev: Mapping[str, float]
    :param barriers: Sequence of reaction barrier specifications.
    :type barriers: Sequence[ReactionBarrier]
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: Directed reaction network.
    :rtype: nx.DiGraph
    """
    graph = nx.DiGraph(temperature_k=float(temperature_k), clamped=0)
    populate_nodes(graph, state_g_ev)
    graph.graph["clamped"] = sum(
        apply_barrier(graph, barrier, temperature_k) for barrier in barriers
    )
    return graph
