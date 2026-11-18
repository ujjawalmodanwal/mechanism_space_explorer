"""
Kinetic simulation and Markov master equation analysis for reaction networks.

Constructs rate matrices from reaction network graphs and integrates population
dynamics over time using scipy LSODA with analytical Jacobians.
"""

from collections.abc import Sequence
from typing import Any

import ase.units as units
import networkx as nx
import numpy as np
import scipy.integrate as integrate

__all__ = [
    "balance_columns",
    "boltzmann",
    "fill_off_diagonals",
    "integrate_kinetics",
    "lsoda_options",
    "rate_matrix",
    "simulate",
    "solve_master_equation",
]


def fill_off_diagonals(
    k_mat: np.ndarray, graph: nx.DiGraph, state_to_idx: dict[str, int]
) -> None:
    """
    Fill off-diagonal transition rates into rate matrix.

    :param k_mat: Rate matrix of shape (n_states, n_states).
    :type k_mat: np.ndarray
    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param state_to_idx: Map of state identifier to matrix index.
    :type state_to_idx: dict[str, int]
    :return: None.
    :rtype: None
    """
    for source, target, edge_data in graph.edges(data=True):
        k_mat[state_to_idx[target], state_to_idx[source]] = float(
            edge_data["rate_s"]
        )


def balance_columns(k_mat: np.ndarray) -> None:
    """
    Set diagonal elements so that every column sums to zero.

    :param k_mat: Rate matrix of shape (n_states, n_states).
    :type k_mat: np.ndarray
    :return: None.
    :rtype: None
    """
    for column_idx in range(k_mat.shape[1]):
        k_mat[column_idx, column_idx] = -np.sum(k_mat[:, column_idx])


def rate_matrix(graph: nx.DiGraph) -> tuple[np.ndarray, list[str]]:
    """
    Construct continuous-time Markov transition rate matrix K.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :return: Tuple of rate matrix K and list of state identifiers.
    :rtype: tuple[np.ndarray, list[str]]
    """
    states = list(graph.nodes)
    k_mat = np.zeros((len(states), len(states)), dtype=float)
    state_to_idx = {state: state_idx for state_idx, state in enumerate(states)}
    fill_off_diagonals(k_mat, graph, state_to_idx)
    balance_columns(k_mat)
    return k_mat, states


def lsoda_options(
    rate_matrix_k: np.ndarray, time_eval: np.ndarray
) -> dict[str, Any]:
    """
    Build kwargs dictionary for master equation solve_ivp.

    :param rate_matrix_k: Transition rate matrix K.
    :type rate_matrix_k: np.ndarray
    :param time_eval: Evaluation time points array.
    :type time_eval: np.ndarray
    :return: Keyword arguments dictionary for solve_ivp.
    :rtype: dict[str, Any]
    """
    return {
        "method": "LSODA",
        "jac": lambda time_val, population_vector: rate_matrix_k,
        "rtol": 1e-8,
        "atol": 1e-20,
        "t_eval": time_eval,
    }


def solve_master_equation(
    rate_matrix_k: np.ndarray,
    initial_populations: np.ndarray,
    time_eval: np.ndarray,
) -> Any:
    """
    Run scipy solve_ivp on master equation using LSODA.

    :param rate_matrix_k: Transition rate matrix K.
    :type rate_matrix_k: np.ndarray
    :param initial_populations: Initial state populations vector.
    :type initial_populations: np.ndarray
    :param time_eval: Time points for solution evaluation.
    :type time_eval: np.ndarray
    :return: OdeResult integration outcome.
    :rtype: Any
    """
    span = (0.0, max(float(np.max(time_eval)), 1e-6))
    return integrate.solve_ivp(
        lambda time_val, population_vector: rate_matrix_k @ population_vector,
        span,
        initial_populations,
        **lsoda_options(rate_matrix_k, time_eval),
    )


def integrate_kinetics(
    rate_matrix_k: np.ndarray,
    initial_populations: np.ndarray,
    time_eval: np.ndarray,
) -> np.ndarray:
    """
    Integrate master equation using scipy LSODA solver.

    :param rate_matrix_k: Transition rate matrix K.
    :type rate_matrix_k: np.ndarray
    :param initial_populations: Initial state populations vector.
    :type initial_populations: np.ndarray
    :param time_eval: Time points for solution evaluation.
    :type time_eval: np.ndarray
    :return: Transposed populations array of shape (len(times), n_states).
    :rtype: np.ndarray
    """
    solution = solve_master_equation(
        rate_matrix_k, initial_populations, time_eval
    )
    if solution.status != 0:
        raise RuntimeError(f"LSODA integration failed: {solution.message}")
    return solution.y.T


def simulate(
    graph: nx.DiGraph, source: str, times_s: Sequence[float] | np.ndarray
) -> np.ndarray:
    """
    Simulate state populations using LSODA numerical integration.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Initial reactant state key.
    :type source: str
    :param times_s: Sequence of evaluation time points in seconds.
    :type times_s: Sequence[float] | np.ndarray
    :return: Population matrix of shape (len(times), n_states).
    :rtype: np.ndarray
    """
    k_mat, states = rate_matrix(graph)
    initial_populations = np.zeros(len(states), dtype=float)
    initial_populations[states.index(source)] = 1.0
    times = np.asarray(times_s, dtype=float)
    return integrate_kinetics(k_mat, initial_populations, times)


def boltzmann(graph: nx.DiGraph) -> np.ndarray:
    """
    Compute normalized equilibrium Boltzmann populations for states.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :return: Array of normalized equilibrium populations.
    :rtype: np.ndarray
    """
    temp_k = float(graph.graph["temperature_k"])
    energies = np.array([graph.nodes[state]["g_ev"] for state in graph.nodes])
    beta = 1.0 / (units.kB * temp_k)
    shifted = np.exp(-(energies - np.min(energies)) * beta)
    return shifted / np.sum(shifted)
