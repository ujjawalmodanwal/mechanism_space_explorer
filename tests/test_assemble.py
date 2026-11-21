"""
Unit tests for refinement assembly, state energies, and network construction.

Validates state identification, species thermochemistry filtering, activation
barrier derivation, candidate filtering, and NetworkX graph generation.
"""

import re

import pytest

from nrnet.assemble import (
    DROP_NO_TS,
    DROP_SPECIES_NOT_OK,
    STATE_SEPARATOR,
    assemble_inputs,
    build_engine_network,
    reaction_barrier,
    species_free_energies,
    state_free_energy,
    state_id,
)
from nrnet.nanoreactor.candidates import Candidate, CandidateSet
from nrnet.refine import STATUS_OK
from nrnet.store import RefinePaths, append_jsonl, temperature_label


def test_state_id_formatting() -> None:
    """
    Verify state_id sorts species keys and joins with STATE_SEPARATOR.

    :return: None.
    :rtype: None
    """
    assert state_id(["B", "A", "C"]) == "A + B + C"
    assert state_id(["A"]) == "A"
    assert state_id([]) == ""
    assert STATE_SEPARATOR in state_id(["A", "B"])


def test_species_free_energies_filtering() -> None:
    """
    Verify species_free_energies retains only STATUS_OK records.

    :return: None.
    :rtype: None
    """
    t_label = temperature_label(298.15)
    rows = {
        "A": {"status": STATUS_OK, "g_ev": {t_label: -1.0}},
        "B": {"status": "unsupported", "g_ev": None},
    }
    energies = species_free_energies(rows, 298.15)
    assert energies == {"A": -1.0}


def test_species_free_energies_rejects_missing_temperature() -> None:
    """
    Verify an ok species without the asked temperature stops the run.

    :return: None.
    :rtype: None
    """
    rows = {"A": {"status": STATUS_OK, "g_ev": {"298.15": -1.0}}}
    with pytest.raises(KeyError):
        species_free_energies(rows, 600.0)


def test_state_free_energy_calculation() -> None:
    """
    Verify state_free_energy sums repeated keys and returns None if missing.

    :return: None.
    :rtype: None
    """
    species_g = {"H": -0.5, "O": -1.5}
    assert state_free_energy(["H", "H", "O"], species_g) == pytest.approx(-2.5)
    assert state_free_energy(["H", "Missing"], species_g) is None


def test_reaction_barrier_drop_reasons() -> None:
    """
    Verify reaction_barrier drops unconfirmed and missing species records.

    :return: None.
    :rtype: None
    """
    cand = Candidate("R1", ("A",), ("B",), ("A",), ("B",), (), "", 1, 0.0)
    b_none, r_none = reaction_barrier(cand, None, {"A": 1.0, "B": 2.0}, 298.15)
    assert b_none is None and r_none == DROP_NO_TS
    unconf = {"status": "neb_failed"}
    b_unconf, r_unconf = reaction_barrier(
        cand, unconf, {"A": 1.0, "B": 2.0}, 298.15
    )
    assert b_unconf is None and r_unconf == DROP_NO_TS


def test_reaction_barrier_missing_species() -> None:
    """
    Verify reaction_barrier returns DROP_SPECIES_NOT_OK when species missing.

    :return: None.
    :rtype: None
    """
    cand = Candidate("R1", ("A",), ("B",), ("A",), ("B",), (), "", 1, 0.0)
    t_label = temperature_label(298.15)
    row = {"status": "confirmed", "g_ts_ev": {t_label: 5.0}}
    barrier, reason = reaction_barrier(cand, row, {"A": 1.0}, 298.15)
    assert barrier is None
    assert reason == DROP_SPECIES_NOT_OK


def test_reaction_barrier_barrierless() -> None:
    """
    Verify reaction_barrier returns barrierless barrier with zero activation.

    :return: None.
    :rtype: None
    """
    cand = Candidate("R1", ("A",), ("B",), ("A",), ("B",), (), "", 1, 0.0)
    row = {"status": "barrierless"}
    species_g = {"A": 1.0, "B": 2.0}
    b, reason = reaction_barrier(cand, row, species_g, 298.15)
    assert reason is None and b is not None
    assert b.barrierless is True and b.dg_act_ev == 0.0
    assert b.state_a == "A" and b.state_b == "B"


def test_r1_kept_at_room_temperature(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify R1 barrier and state energies assembled at 298.15 K.

    :param fake_refine_results: Fixture returning CandidateSet and RefinePaths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    inputs = assemble_inputs(candidate_set, paths, 298.15)
    assert len(inputs.barriers) == 1
    barrier = inputs.barriers[0]
    assert barrier.state_a == state_id(("A", "B", "S"))
    assert barrier.state_b == state_id(("C", "S"))
    assert barrier.dg_act_ev == pytest.approx(0.5 - (-1.0 + -2.0))
    assert inputs.state_g_ev[state_id(("A", "B", "S"))] == pytest.approx(-3.5)
    assert inputs.state_g_ev[state_id(("C", "S"))] == pytest.approx(-4.5)


def test_r1_kept_at_elevated_temperature(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify R1 barrier and state energies assembled at 600.0 K.

    :param fake_refine_results: Fixture returning CandidateSet and RefinePaths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    inputs = assemble_inputs(candidate_set, paths, 600.0)
    assert len(inputs.barriers) == 1
    barrier = inputs.barriers[0]
    assert barrier.state_a == state_id(("A", "B", "S"))
    assert barrier.state_b == state_id(("C", "S"))
    assert barrier.dg_act_ev == pytest.approx(0.8 - (-1.2 + -2.2))
    assert inputs.state_g_ev[state_id(("A", "B", "S"))] == pytest.approx(-4.0)
    assert inputs.state_g_ev[state_id(("C", "S"))] == pytest.approx(-4.9)


def test_dropped_reasons_counter(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify dropped counts map exactly recorded rejection reasons.

    :param fake_refine_results: Fixture returning CandidateSet and RefinePaths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    inputs = assemble_inputs(candidate_set, paths, 298.15)
    assert inputs.dropped == {
        DROP_NO_TS: 1,
        DROP_SPECIES_NOT_OK: 1,
    }


def test_later_species_row_wins(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify later appended species row overrides earlier row in assembly.

    :param fake_refine_results: Fixture returning CandidateSet and RefinePaths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    t298 = temperature_label(298.15)
    new_row = {"key": "A", "status": STATUS_OK, "g_ev": {t298: -1.5}}
    append_jsonl(paths.species_jsonl, new_row)
    inputs = assemble_inputs(candidate_set, paths, 298.15)
    assert inputs.barriers[0].dg_act_ev == pytest.approx(0.5 - (-1.5 + -2.0))
    assert inputs.state_g_ev[state_id(("A", "B", "S"))] == pytest.approx(-4.0)


def test_missing_source_species_raises_value_error(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify assemble_inputs raises ValueError naming source state on missing G.

    :param fake_refine_results: Fixture returning CandidateSet and RefinePaths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    bad_row = {"key": "A", "status": "unsupported", "g_ev": {}}
    append_jsonl(paths.species_jsonl, bad_row)
    source_name = state_id(candidate_set.reactant_state)
    with pytest.raises(ValueError, match=re.escape(source_name)):
        assemble_inputs(candidate_set, paths, 298.15)


def test_build_engine_network(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify build_engine_network sets source graph attribute and adds 2 edges.

    :param fake_refine_results: Fixture returning CandidateSet and RefinePaths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidate_set, paths = fake_refine_results
    graph, inputs = build_engine_network(candidate_set, paths, 298.15)
    src = state_id(("A", "B", "S"))
    assert graph.graph["source"] == src
    assert inputs.source == src
    assert len(graph.edges) == 2
    assert (src, state_id(("C", "S"))) in graph.edges
    assert (state_id(("C", "S")), src) in graph.edges
