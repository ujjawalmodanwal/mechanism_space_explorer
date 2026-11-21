"""
Unit tests for reaction network construction, Eyring rates, and clamping.

Validates ReactionBarrier data models, Eyring transition state theory rates,
toy network topology, branching probabilities, and barrier clamping.
"""

import dataclasses

import networkx as nx
import pytest

from nrnet.network import (
    KCAL_MOL_EV,
    ReactionBarrier,
    build_network,
    eyring_rate,
)


def test_reaction_barrier_dataclass() -> None:
    """
    Verify ReactionBarrier frozen dataclass attributes and immutability.

    :return: None.
    :rtype: None
    """
    barrier = ReactionBarrier("R", "A", 0.5)
    assert barrier.state_a == "R"
    assert barrier.state_b == "A"
    assert barrier.dg_act_ev == 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        barrier.state_a = "B"  # type: ignore[misc]


def test_eyring_rate_at_room_temperature() -> None:
    """
    Verify Eyring rate calculation on 15 kcal/mol barrier at 298.15 K.

    :return: None.
    :rtype: None
    """
    rate = eyring_rate(15.0 * KCAL_MOL_EV, 298.15)
    assert rate == pytest.approx(62.832, rel=1e-3)


def test_toy_network_construction(toy_network: nx.DiGraph) -> None:
    """
    Verify toy network node attributes, edge counts, and graph metadata.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    assert len(toy_network.nodes) == 5
    assert len(toy_network.edges) == 8
    assert toy_network.graph["temperature_k"] == 298.15
    assert toy_network.graph["clamped"] == 0
    assert toy_network.nodes["R"]["g_ev"] == pytest.approx(0.0)
    assert toy_network.nodes["A"]["g_ev"] == pytest.approx(-5.0 * KCAL_MOL_EV)


def test_toy_network_branching_probabilities(toy_network: nx.DiGraph) -> None:
    """
    Verify branching probabilities p(R->A) and p(A->P1) on toy network.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    rate_ra = toy_network["R"]["A"]["rate_s"]
    rate_rb = toy_network["R"]["B"]["rate_s"]
    assert (rate_ra / (rate_ra + rate_rb)) == pytest.approx(0.99978, abs=1e-4)
    rate_ap1 = toy_network["A"]["P1"]["rate_s"]
    rate_ar = toy_network["A"]["R"]["rate_s"]
    assert (rate_ap1 / (rate_ap1 + rate_ar)) == pytest.approx(0.9669, abs=1e-3)


def test_clamp_barrier_below_endpoints() -> None:
    """
    Verify barrier below endpoints is clamped to max(g_a, g_b) and counted.

    :return: None.
    :rtype: None
    """
    energies = {"S1": 1.0, "S2": 2.0}
    barriers = [ReactionBarrier("S1", "S2", -2.0)]
    graph = build_network(energies, barriers, 298.15)
    assert graph.graph["clamped"] == 1
    assert graph["S1"]["S2"]["ts_ev"] == 2.0
    assert graph["S1"]["S2"]["dg_act_ev"] == 1.0
    assert graph["S2"]["S1"]["dg_act_ev"] == 0.0


def test_barrierless_barrier_clamped_to_max() -> None:
    """
    Verify barrierless barrier yields TS = max(G_A, G_B) with edge attributes.

    :return: None.
    :rtype: None
    """
    energies = {"S1": 1.0, "S2": 3.0}
    barriers = [ReactionBarrier("S1", "S2", 0.0, barrierless=True)]
    graph = build_network(energies, barriers, 298.15)
    assert graph["S1"]["S2"]["ts_ev"] == 3.0
    assert graph["S2"]["S1"]["ts_ev"] == 3.0
    assert graph["S1"]["S2"]["barrierless"] is True
    assert graph["S2"]["S1"]["barrierless"] is True
    assert graph["S1"]["S2"]["dg_act_ev"] == 2.0
    assert graph["S2"]["S1"]["dg_act_ev"] == 0.0


def test_barrierless_arrow_does_not_increase_clamped_count() -> None:
    """
    Verify a barrierless arrow does not increment the clamped barrier count.

    :return: None.
    :rtype: None
    """
    energies = {"S1": 1.0, "S2": 3.0, "S3": 0.5}
    barriers = [
        ReactionBarrier("S1", "S2", 0.0, barrierless=True),
        ReactionBarrier("S1", "S3", -2.0, barrierless=False),
    ]
    graph = build_network(energies, barriers, 298.15)
    assert graph.graph["clamped"] == 1
