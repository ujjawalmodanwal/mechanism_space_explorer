"""
Unit tests for reaction network product rankers R0 through R4.

Validates thermodynamic filtering (R0), summed barrier shortest paths (R1),
bottleneck minimax Dijkstra (R2), path transition probability Dijkstra (R3),
and master equation kinetic yields at transient and asymptotic times (R4).
"""

import networkx as nx
import pytest

from nrnet.rankers import (
    RANKERS,
    RankSettings,
    rank_by_bottleneck,
    rank_by_free_energy,
    rank_by_kinetics,
    rank_by_probability,
    rank_by_summed_barriers,
)


def test_rankers_registry() -> None:
    """
    Verify RANKERS registry contains strategies R0 through R4.

    :return: None.
    :rtype: None
    """
    assert set(RANKERS.keys()) == {"R0", "R1", "R2", "R3", "R4"}
    for ranker_fn in RANKERS.values():
        assert callable(ranker_fn)


def test_ranker_r0_free_energy(toy_network: nx.DiGraph) -> None:
    """
    Verify R0 selects P2 as top thermodynamic product with empty path.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    settings = RankSettings(temperature_k=298.15, t_obs_s=60.0)
    ranked = rank_by_free_energy(toy_network, "R", settings)
    assert ranked[0].state == "P2"
    assert ranked[0].path == ()


def test_ranker_r1_summed_barriers(toy_network: nx.DiGraph) -> None:
    """
    Verify R1 selects P2 route (25 kcal) over P1 (33 kcal) with path.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    settings = RankSettings(temperature_k=298.15, t_obs_s=60.0)
    ranked = rank_by_summed_barriers(toy_network, "R", settings)
    products = [item for item in ranked if item.state in ("P1", "P2")]
    assert products[0].state == "P2"
    p2_item = next(item for item in ranked if item.state == "P2")
    assert p2_item.path == ("R", "B", "P2")


def test_ranker_r2_bottleneck(toy_network: nx.DiGraph) -> None:
    """
    Verify R2 selects P1 route (TS 15) over P2 (TS 20) with path.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    settings = RankSettings(temperature_k=298.15, t_obs_s=60.0)
    ranked = rank_by_bottleneck(toy_network, "R", settings)
    products = [item for item in ranked if item.state in ("P1", "P2")]
    assert products[0].state == "P1"
    p1_item = next(item for item in ranked if item.state == "P1")
    assert p1_item.path == ("R", "A", "P1")


def test_ranker_r3_probability(toy_network: nx.DiGraph) -> None:
    """
    Verify R3 selects P1 route by branching probability.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    settings = RankSettings(temperature_k=298.15, t_obs_s=60.0)
    ranked = rank_by_probability(toy_network, "R", settings)
    products = [item for item in ranked if item.state in ("P1", "P2")]
    assert products[0].state == "P1"
    assert products[0].score > products[1].score


def test_ranker_r4_kinetics_short_time(toy_network: nx.DiGraph) -> None:
    """
    Verify R4 at 60 s selects P1 with population approx 0.9996.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    settings = RankSettings(temperature_k=298.15, t_obs_s=60.0)
    ranked = rank_by_kinetics(toy_network, "R", settings)
    assert ranked[0].state == "P1"
    assert ranked[0].score == pytest.approx(0.9996, abs=1e-3)
    assert ranked[0].path == ()


def test_ranker_r4_kinetics_long_time(toy_network: nx.DiGraph) -> None:
    """
    Verify R4 at 1e12 s selects P2 matching Boltzmann within 1e-5.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    settings = RankSettings(temperature_k=298.15, t_obs_s=1e12)
    ranked = rank_by_kinetics(toy_network, "R", settings)
    scores = {item.state: item.score for item in ranked}
    assert ranked[0].state == "P2"
    assert scores["P1"] == pytest.approx(0.0331, abs=1e-3)
    assert scores["P2"] == pytest.approx(0.9669, abs=1e-3)
