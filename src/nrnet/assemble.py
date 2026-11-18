"""
Assemble refinement results into reaction network inputs and NetworkX graphs.

A state's G is the sum of its species' G; a barrier is G_TS minus the G of
the reacting molecules only, because bystanders add the same G to both ends.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from nrnet.nanoreactor.candidates import Candidate, CandidateSet
from nrnet.network import ReactionBarrier, build_network
from nrnet.refine import STATUS_OK
from nrnet.store import RefinePaths, latest_rows, temperature_label
from nrnet.ts_search import TSStatus

__all__ = [
    "DROP_NO_TS",
    "DROP_SPECIES_NOT_OK",
    "STATE_SEPARATOR",
    "NetworkInputs",
    "assemble_inputs",
    "build_engine_network",
    "collect_state_energies",
    "confirmed_ts_energy",
    "evaluate_candidates",
    "reaction_barrier",
    "species_free_energies",
    "state_free_energy",
    "state_id",
]

STATE_SEPARATOR: str = " + "
DROP_NO_TS: str = "no_ts"
DROP_SPECIES_NOT_OK: str = "species_not_ok"


@dataclass(frozen=True, slots=True)
class NetworkInputs:
    """
    Network inputs assembled from refinement calculations at fixed temperature.

    :param source: Identifier of the initial reactant source state.
    :type source: str
    :param state_g_ev: Mapping of state identifier to free energy in eV.
    :type state_g_ev: dict[str, float]
    :param barriers: Tuple of forward activation barriers for kept reactions.
    :type barriers: tuple[ReactionBarrier, ...]
    :param dropped: Counts of rejected candidate reactions grouped by reason.
    :type dropped: dict[str, int]
    """

    source: str
    state_g_ev: dict[str, float]
    barriers: tuple[ReactionBarrier, ...]
    dropped: dict[str, int]


def state_id(keys: Sequence[str]) -> str:
    """
    Construct canonical state identifier string from species keys.

    :param keys: Sequence of chemical species keys composing the state.
    :type keys: Sequence[str]
    :return: Canonical state identifier with sorted species keys.
    :rtype: str
    """
    return STATE_SEPARATOR.join(sorted(keys))


def species_free_energies(
    species_rows: Mapping[str, Mapping[str, Any]],
    temperature_k: float,
) -> dict[str, float]:
    """
    Extract Gibbs free energies for confirmed ok species at temperature.

    :param species_rows: Mapping of species keys to serialized record dicts.
    :type species_rows: Mapping[str, Mapping[str, Any]]
    :param temperature_k: Evaluation temperature in Kelvin.
    :type temperature_k: float
    :return: Mapping of species keys to Gibbs free energy in eV.
    :rtype: dict[str, float]
    """
    label = temperature_label(temperature_k)
    # An ok row without this temperature means search.yaml changed after
    # refine; the KeyError stops the run instead of dropping the species.
    return {
        key: float(row["g_ev"][label])
        for key, row in species_rows.items()
        if row["status"] == STATUS_OK
    }


def state_free_energy(
    keys: Sequence[str],
    species_g: Mapping[str, float],
) -> float | None:
    """
    Sum species free energies for a state key sequence.

    :param keys: Sequence of chemical species keys composing the state.
    :type keys: Sequence[str]
    :param species_g: Mapping of species keys to free energies in eV.
    :type species_g: Mapping[str, float]
    :return: Total free energy in eV, or None if any species key is missing.
    :rtype: float or None
    """
    if any(key not in species_g for key in keys):
        return None
    return float(sum(species_g[key] for key in keys))


def confirmed_ts_energy(
    row: Mapping[str, Any] | None,
    label: str,
) -> float | None:
    """
    Extract transition state free energy from confirmed search record.

    :param row: Deserialized transition state record dictionary, or None.
    :type row: Mapping[str, Any] or None
    :param label: Temperature label string for free energy lookup.
    :type label: str
    :return: Transition state free energy in eV, or None if unconfirmed.
    :rtype: float or None
    """
    if row is None or row["status"] != TSStatus.confirmed.value:
        return None
    # A confirmed row always carries G_TS for every refine temperature.
    return float(row["g_ts_ev"][label])


def reaction_barrier(
    candidate: Candidate,
    row: Mapping[str, Any] | None,
    species_g: Mapping[str, float],
    temperature_k: float,
) -> tuple[ReactionBarrier | None, str | None]:
    """
    Compute forward activation barrier for candidate reaction if validated.

    :param candidate: Candidate reaction to evaluate.
    :type candidate: Candidate
    :param row: Deserialized transition state search record, or None.
    :type row: Mapping[str, Any] or None
    :param species_g: Mapping of species keys to free energies in eV.
    :type species_g: Mapping[str, float]
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: Tuple of ReactionBarrier (or None) and drop reason (or None).
    :rtype: tuple[ReactionBarrier or None, str or None]
    """
    label = temperature_label(temperature_k)
    ts_energy = confirmed_ts_energy(row, label)
    if ts_energy is None:
        return None, DROP_NO_TS
    g_react = state_free_energy(candidate.reactant_keys, species_g)
    g_a = state_free_energy(candidate.state_a, species_g)
    g_b = state_free_energy(candidate.state_b, species_g)
    if g_react is None or g_a is None or g_b is None:
        return None, DROP_SPECIES_NOT_OK
    barrier = ReactionBarrier(
        state_id(candidate.state_a),
        state_id(candidate.state_b),
        ts_energy - g_react,
    )
    return barrier, None


def evaluate_candidates(
    candidates: Sequence[Candidate],
    reaction_rows: Mapping[str, Mapping[str, Any]],
    species_g: Mapping[str, float],
    temperature_k: float,
) -> tuple[tuple[ReactionBarrier, ...], tuple[Candidate, ...], dict[str, int]]:
    """
    Evaluate candidate reactions against refinement transition state records.

    :param candidates: Sequence of candidate reactions to evaluate.
    :type candidates: Sequence[Candidate]
    :param reaction_rows: Mapping of reaction IDs to search outcome records.
    :type reaction_rows: Mapping[str, Mapping[str, Any]]
    :param species_g: Mapping of species keys to free energies in eV.
    :type species_g: Mapping[str, float]
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: Tuple of kept barriers, kept candidates, and dropped reason counts.
    :rtype: tuple[tuple, tuple, dict]
    """
    barriers: list[ReactionBarrier] = []
    kept: list[Candidate] = []
    dropped: Counter[str] = Counter()
    for candidate in candidates:
        row = reaction_rows.get(candidate.reaction_id)
        barrier, reason = reaction_barrier(
            candidate, row, species_g, temperature_k
        )
        if barrier is not None:
            barriers.append(barrier)
            kept.append(candidate)
        elif reason is not None:
            dropped[reason] += 1
    return tuple(barriers), tuple(kept), dict(dropped)


def collect_state_energies(
    source_state: Sequence[str],
    candidates: Sequence[Candidate],
    species_g: Mapping[str, float],
) -> dict[str, float]:
    """
    Collect free energies for source state and kept reaction states.

    :param source_state: Chemical species keys composing the source state.
    :type source_state: Sequence[str]
    :param candidates: Sequence of kept candidate reactions.
    :type candidates: Sequence[Candidate]
    :param species_g: Mapping of species keys to free energies in eV.
    :type species_g: Mapping[str, float]
    :return: Mapping of state identifier to free energy in eV.
    :rtype: dict[str, float]
    :raises ValueError: If the source state has missing species free energy.
    """
    source_id_str = state_id(source_state)
    source_g = state_free_energy(source_state, species_g)
    if source_g is None:
        raise ValueError(
            f"Source state '{source_id_str}' has missing species free energy"
        )
    state_g_ev: dict[str, float] = {source_id_str: source_g}
    for candidate in candidates:
        g_a = state_free_energy(candidate.state_a, species_g)
        g_b = state_free_energy(candidate.state_b, species_g)
        if g_a is not None and g_b is not None:
            state_g_ev[state_id(candidate.state_a)] = g_a
            state_g_ev[state_id(candidate.state_b)] = g_b
    return state_g_ev


def assemble_inputs(
    candidate_set: CandidateSet,
    paths: RefinePaths,
    temperature_k: float,
) -> NetworkInputs:
    """
    Assemble network inputs from candidate set and engine refinement records.

    :param candidate_set: Frozen candidate set extracted from nanoreactor.
    :type candidate_set: CandidateSet
    :param paths: Filesystem paths to refinement output files.
    :type paths: RefinePaths
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: NetworkInputs data model containing states, barriers, and drops.
    :rtype: NetworkInputs
    :raises ValueError: If the reactant source state has no free energy.
    """
    species_rows = latest_rows(paths.species_jsonl, "key")
    reaction_rows = latest_rows(paths.reactions_jsonl, "reaction_id")
    species_g = species_free_energies(species_rows, temperature_k)
    barriers, kept, dropped = evaluate_candidates(
        candidate_set.candidates, reaction_rows, species_g, temperature_k
    )
    state_g_ev = collect_state_energies(
        candidate_set.reactant_state, kept, species_g
    )
    return NetworkInputs(
        source=state_id(candidate_set.reactant_state),
        state_g_ev=state_g_ev,
        barriers=barriers,
        dropped=dropped,
    )


def build_engine_network(
    candidate_set: CandidateSet,
    paths: RefinePaths,
    temperature_k: float,
) -> tuple[nx.DiGraph, NetworkInputs]:
    """
    Build directed reaction network graph from engine refinement results.

    :param candidate_set: Frozen candidate set extracted from nanoreactor.
    :type candidate_set: CandidateSet
    :param paths: Filesystem paths to refinement output files.
    :type paths: RefinePaths
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: Tuple containing built NetworkX DiGraph and NetworkInputs.
    :rtype: tuple[nx.DiGraph, NetworkInputs]
    """
    inputs = assemble_inputs(candidate_set, paths, temperature_k)
    graph = build_network(inputs.state_g_ev, inputs.barriers, temperature_k)
    graph.graph["source"] = inputs.source
    return graph, inputs
