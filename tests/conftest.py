"""
Pytest configuration and shared test fixtures for reaction networks.

Provides the canonical five-state worked toy reaction network as a plain
helper function and as a pytest fixture.
"""

from pathlib import Path

import networkx as nx
import pytest

from nrnet.nanoreactor.candidates import Candidate, CandidateSet
from nrnet.network import KCAL_MOL_EV, ReactionBarrier, build_network
from nrnet.refine import STATUS_OK
from nrnet.store import (
    RefinePaths,
    append_jsonl,
    make_refine_paths,
    temperature_label,
)
from nrnet.ts_search import TSStatus


def toy_barriers() -> list[ReactionBarrier]:
    """
    Construct reaction barriers for worked toy network in eV.

    :return: List of ReactionBarrier objects.
    :rtype: list[ReactionBarrier]
    """
    raw_barriers = [
        ("R", "A", 15.0),
        ("A", "P1", 18.0),
        ("R", "B", 20.0),
        ("B", "P2", 5.0),
    ]
    return [
        ReactionBarrier(state_a, state_b, barrier_kcal * KCAL_MOL_EV)
        for state_a, state_b, barrier_kcal in raw_barriers
    ]


def build_toy_network() -> nx.DiGraph:
    """
    Construct standard worked toy reaction network at 298.15 K.

    :return: Built reaction network graph.
    :rtype: nx.DiGraph
    """
    energies_kcal = {
        "R": 0.0,
        "A": -5.0,
        "B": 0.0,
        "P1": -10.0,
        "P2": -12.0,
    }
    energies_ev = {
        state: energy * KCAL_MOL_EV for state, energy in energies_kcal.items()
    }
    return build_network(energies_ev, toy_barriers(), 298.15)


@pytest.fixture
def toy_network() -> nx.DiGraph:
    """
    Provide standard worked toy reaction network at 298.15 K.

    :return: Built reaction network graph.
    :rtype: nx.DiGraph
    """
    return build_toy_network()


def build_fake_candidates() -> tuple[Candidate, ...]:
    """
    Construct candidate reactions for fake refinement fixture.

    :return: Tuple of candidate reaction objects.
    :rtype: tuple[Candidate, ...]
    """
    r1 = Candidate(
        "R1", ("A", "B", "S"), ("C", "S"), ("A", "B"), ("C",), (), "", 1, 0.0
    )
    r2 = Candidate("R2", ("C", "S"), ("D", "S"), ("C",), ("D",), (), "", 1, 0.0)
    r3 = Candidate("R3", ("C", "S"), ("X", "S"), ("C",), ("X",), (), "", 1, 0.0)
    return r1, r2, r3


# Fake species: key -> (E, G at 298.15 K, G at 600 K), all in eV.
FAKE_SPECIES: dict[str, tuple[float, float, float]] = {
    "A": (-0.7, -1.0, -1.2),
    "B": (-1.7, -2.0, -2.2),
    "C": (-3.6, -4.0, -4.3),
    "S": (-0.3, -0.5, -0.6),
    "D": (-4.6, -5.0, -5.4),
}

# Fake reactions: id -> (status, E_TS, G_TS at 298.15 K, G_TS at 600 K).
FAKE_REACTIONS: dict[str, tuple[str, float | None, float, float]] = {
    "R1": (TSStatus.confirmed.value, 0.9, 0.5, 0.8),
    "R2": ("endpoint_mismatch", None, 0.0, 0.0),
    "R3": (TSStatus.confirmed.value, 1.8, 1.5, 1.9),
}


def write_fake_species(paths: RefinePaths) -> None:
    """
    Write species JSON Lines records for fake refinement fixture.

    Species X is unsupported, as a doublet is for a closed-shell engine.

    :param paths: Refinement destination paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    labels = (temperature_label(298.15), temperature_label(600.0))
    for key, (energy, *g_values) in FAKE_SPECIES.items():
        row = {
            "key": key,
            "status": STATUS_OK,
            "energy_ev": energy,
            "g_ev": dict(zip(labels, g_values, strict=True)),
        }
        append_jsonl(paths.species_jsonl, row)
    append_jsonl(paths.species_jsonl, {"key": "X", "status": "unsupported"})


def write_fake_reactions(paths: RefinePaths) -> None:
    """
    Write reaction JSON Lines records for fake refinement fixture.

    :param paths: Refinement destination paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    labels = (temperature_label(298.15), temperature_label(600.0))
    for rid, (status, ts_energy, *g_values) in FAKE_REACTIONS.items():
        confirmed = status == TSStatus.confirmed.value
        g_ts = dict(zip(labels, g_values, strict=True)) if confirmed else None
        row = {
            "reaction_id": rid,
            "status": status,
            "ts_energy_ev": ts_energy,
            "g_ts_ev": g_ts,
        }
        append_jsonl(paths.reactions_jsonl, row)


@pytest.fixture
def fake_refine_results(
    tmp_path: Path,
) -> tuple[CandidateSet, RefinePaths]:
    """
    Construct fake refinement results fixture with CandidateSet and RefinePaths.

    :param tmp_path: Temporary directory provided by pytest.
    :type tmp_path: Path
    :return: Tuple of CandidateSet and populated RefinePaths.
    :rtype: tuple[CandidateSet, RefinePaths]
    """
    paths = make_refine_paths(tmp_path, "toy", "xtb")
    write_fake_species(paths)
    write_fake_reactions(paths)
    candidates = build_fake_candidates()
    candidate_set = CandidateSet(
        reactant_state=("A", "B", "S"),
        species={},
        candidates=candidates,
    )
    return candidate_set, paths
