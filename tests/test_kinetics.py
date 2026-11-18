"""
Unit tests for kinetic rate matrix construction, LSODA integration, Boltzmann.

Validates column sum conservation, matrix exponential agreement at short times,
asymptotic approach to Boltzmann equilibrium, and solver error handling.
"""

from typing import Any

import networkx as nx
import numpy as np
import pytest
import scipy.linalg as linalg

from nrnet.kinetics import boltzmann, rate_matrix, simulate


def test_rate_matrix_columns_sum_zero(toy_network: nx.DiGraph) -> None:
    """
    Verify every column of rate matrix sums to zero within 1e-12 relative.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    k_mat = rate_matrix(toy_network)[0]
    col_sums = np.abs(np.sum(k_mat, axis=0))
    col_scales = np.sum(np.abs(k_mat), axis=0)
    for column_sum, scale in zip(col_sums, col_scales, strict=True):
        limit = 1e-12 * scale if scale > 0.0 else 1e-12
        assert column_sum <= limit


def test_simulate_at_60s_matches_expm(toy_network: nx.DiGraph) -> None:
    """
    Verify numerical integration at 60 s matches matrix exponential.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    k_mat, states = rate_matrix(toy_network)
    initial_populations = np.zeros(len(states))
    initial_populations[states.index("R")] = 1.0
    sim_pop = simulate(toy_network, "R", [60.0])[0]
    expm_pop = linalg.expm(k_mat * 60.0) @ initial_populations
    assert np.allclose(sim_pop, expm_pop, atol=2e-6)


def test_simulate_at_long_time_matches_boltzmann(
    toy_network: nx.DiGraph,
) -> None:
    """
    Verify numerical integration at 1e12 s matches Boltzmann distribution.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    sim_pop = simulate(toy_network, "R", [1e12])[0]
    boltz_pop = boltzmann(toy_network)
    assert np.allclose(sim_pop, boltz_pop, atol=1e-5)


def test_boltzmann_toy_populations(toy_network: nx.DiGraph) -> None:
    """
    Verify toy network Boltzmann populations: P1 ~0.0331 and P2 ~0.9669.

    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    states = rate_matrix(toy_network)[1]
    boltz = dict(zip(states, boltzmann(toy_network), strict=True))
    assert boltz["P1"] == pytest.approx(0.0331, abs=1e-3)
    assert boltz["P2"] == pytest.approx(0.9669, abs=1e-3)


def failure_mock_result() -> Any:
    """
    Construct mock integration outcome with failure status.

    :return: Mock solver result with status -1.
    :rtype: Any
    """
    return type("Result", (), {"status": -1, "message": "solver error"})()


def test_simulate_failure_raises(
    monkeypatch: pytest.MonkeyPatch, toy_network: nx.DiGraph
) -> None:
    """
    Verify simulate raises RuntimeError when integration status is nonzero.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param toy_network: Worked toy reaction network graph.
    :type toy_network: nx.DiGraph
    :return: None.
    :rtype: None
    """
    monkeypatch.setattr(
        "nrnet.kinetics.solve_master_equation",
        lambda rate_matrix_k, init_y, eval_t: failure_mock_result(),
    )
    with pytest.raises(RuntimeError, match="LSODA integration failed"):
        simulate(toy_network, "R", [1.0])
