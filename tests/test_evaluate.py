"""
Unit tests for evaluation metrics, reaction energies, RMSD, and cost logging.

Tests hand-computed reaction and activation energies in kcal/mol, handling of
unconfirmed transition states and missing species, mean absolute error summary
statistics, transition state geometry RMSDs under rotation, and call costs.
"""

import csv
import shutil
from pathlib import Path

import pytest
import yaml
from ase import Atoms

from nrnet.evaluate import (
    NO_REACTION,
    cost_row,
    evaluate,
    load_yield_cutoff,
    major_product,
    network_agreement,
    r4_yields,
    ranker_agreement,
    ranker_rows,
    reaction_energies,
    species_energies,
    step_errors,
    step_table,
    ts_rmsd,
)
from nrnet.nanoreactor.candidate_store import write_candidate_files
from nrnet.nanoreactor.candidates import Candidate, CandidateSet
from nrnet.network import KCAL_MOL_EV
from nrnet.ranking import make_rank_dir, rank_candidates
from nrnet.refine import STATUS_OK
from nrnet.store import (
    RefinePaths,
    append_extxyz,
    make_refine_paths,
    temperature_label,
    write_csv,
    write_jsonl,
)

LABEL: str = temperature_label(298.15)


def make_test_candidate() -> Candidate:
    """
    Construct test candidate reaction representing A + B -> C.

    :return: Candidate reaction item.
    :rtype: Candidate
    """
    return Candidate(
        reaction_id="R001",
        state_a=("A", "B"),
        state_b=("C",),
        reactant_keys=("A", "B"),
        product_keys=("C",),
        pairs=(),
        mapped_smiles="",
        n_events=1,
        first_seen_fs=0.0,
    )


def test_reaction_energies_hand_computed_values() -> None:
    """
    Verify hand-computed reaction and activation energies including kcal units.

    :return: None.
    :rtype: None
    """
    candidate = make_test_candidate()
    species_e = {"A": -10.0, "B": -20.0, "C": -32.0}
    species_g = {"A": -10.5, "B": -20.5, "C": -33.5}
    row = {
        "reaction_id": "R001",
        "status": "confirmed",
        "ts_energy_ev": -28.0,
        "g_ts_ev": {LABEL: -29.0},
    }
    result = reaction_energies(candidate, row, species_e, species_g, LABEL)

    assert result["reaction_id"] == "R001"
    assert result["status"] == "confirmed"
    assert result["de_rxn_kcal"] == pytest.approx(-2.0 / KCAL_MOL_EV)
    assert result["dg_rxn_kcal"] == pytest.approx(-2.5 / KCAL_MOL_EV)
    assert result["de_act_kcal"] == pytest.approx(2.0 / KCAL_MOL_EV)
    assert result["dg_act_kcal"] == pytest.approx(2.0 / KCAL_MOL_EV)


def test_reaction_energies_unconfirmed_and_missing_rows() -> None:
    """
    Verify unconfirmed row has None for activation while rxn fields remain.

    :return: None.
    :rtype: None
    """
    candidate = make_test_candidate()
    species_e = {"A": -10.0, "B": -20.0, "C": -32.0}
    species_g = {"A": -10.5, "B": -20.5, "C": -33.5}
    unconfirmed = {
        "reaction_id": "R001",
        "status": "endpoint_mismatch",
        "ts_energy_ev": -28.0,
        "g_ts_ev": {LABEL: -29.0},
    }
    res_unconfirmed = reaction_energies(
        candidate, unconfirmed, species_e, species_g, LABEL
    )
    assert res_unconfirmed["status"] == "endpoint_mismatch"
    assert res_unconfirmed["de_rxn_kcal"] == pytest.approx(-2.0 / KCAL_MOL_EV)
    assert res_unconfirmed["dg_rxn_kcal"] == pytest.approx(-2.5 / KCAL_MOL_EV)
    assert res_unconfirmed["de_act_kcal"] is None
    assert res_unconfirmed["dg_act_kcal"] is None

    res_missing = reaction_energies(
        candidate, None, species_e, species_g, LABEL
    )
    assert res_missing["status"] == "missing"
    assert res_missing["de_rxn_kcal"] == pytest.approx(-2.0 / KCAL_MOL_EV)
    assert res_missing["dg_rxn_kcal"] == pytest.approx(-2.5 / KCAL_MOL_EV)
    assert res_missing["de_act_kcal"] is None
    assert res_missing["dg_act_kcal"] is None

    partial_e = {"A": -10.0, "C": -32.0}
    res_partial = reaction_energies(
        candidate, unconfirmed, partial_e, species_g, LABEL
    )
    assert res_partial["de_rxn_kcal"] is None
    assert res_partial["de_act_kcal"] is None
    assert res_partial["dg_rxn_kcal"] == pytest.approx(-2.5 / KCAL_MOL_EV)


def test_species_energies_filters_ok_status() -> None:
    """
    Verify species_energies extracts energy_ev only for STATUS_OK rows.

    :return: None.
    :rtype: None
    """
    rows = {
        "A": {"status": STATUS_OK, "energy_ev": -10.0},
        "B": {"status": "unsupported", "energy_ev": -20.0},
        "C": {"status": STATUS_OK, "energy_ev": -30.0},
    }
    energies = species_energies(rows)
    assert energies == {"A": -10.0, "C": -30.0}


def test_step_errors_mae_and_counts() -> None:
    """
    Verify step_errors computes correct MAE, counts, and ts_success fraction.

    :return: None.
    :rtype: None
    """
    engine_rows = [
        {
            "reaction_id": "R1",
            "status": "confirmed",
            "de_rxn_kcal": 10.0,
            "dg_rxn_kcal": 12.0,
            "de_act_kcal": 20.0,
            "dg_act_kcal": 22.0,
        },
        {
            "reaction_id": "R2",
            "status": "endpoint_mismatch",
            "de_rxn_kcal": 5.0,
            "dg_rxn_kcal": 6.0,
            "de_act_kcal": None,
            "dg_act_kcal": None,
        },
    ]
    ref_rows = [
        {
            "reaction_id": "R1",
            "status": "confirmed",
            "de_rxn_kcal": 12.0,
            "dg_rxn_kcal": 15.0,
            "de_act_kcal": 24.0,
            "dg_act_kcal": 25.0,
        },
        {
            "reaction_id": "R2",
            "status": "confirmed",
            "de_rxn_kcal": 4.0,
            "dg_rxn_kcal": 8.0,
            "de_act_kcal": 15.0,
            "dg_act_kcal": 16.0,
        },
    ]
    errors = step_errors(engine_rows, ref_rows)
    assert errors["de_rxn_kcal_mae"] == pytest.approx(1.5)
    assert errors["de_rxn_kcal_n"] == 2
    assert errors["dg_rxn_kcal_mae"] == pytest.approx(2.5)
    assert errors["dg_rxn_kcal_n"] == 2
    assert errors["de_act_kcal_mae"] == pytest.approx(4.0)
    assert errors["de_act_kcal_n"] == 1
    assert errors["dg_act_kcal_mae"] == pytest.approx(3.0)
    assert errors["dg_act_kcal_n"] == 1
    assert errors["ts_success"] == pytest.approx(0.5)


def test_step_errors_empty_and_zero_matches() -> None:
    """
    Verify step_errors handles empty input sequences gracefully with None MAE.

    :return: None.
    :rtype: None
    """
    errors = step_errors([], [])
    assert errors["de_rxn_kcal_mae"] is None
    assert errors["de_rxn_kcal_n"] == 0
    assert errors["ts_success"] is None


def test_ts_rmsd_alignment_and_rotation(tmp_path: Path) -> None:
    """
    Verify ts_rmsd is 0 for identical, positive for displaced, ignores rotation.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    eng_path = tmp_path / "engine_ts.extxyz"
    ref_path = tmp_path / "reference_ts.extxyz"

    frame1 = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58]],
    )
    frame1.info["reaction_id"] = "R1"

    frame2_ref = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    frame2_ref.info["reaction_id"] = "R2"
    frame2_eng = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frame2_eng.info["reaction_id"] = "R2"

    frame3_ref = frame1.copy()
    frame3_ref.info["reaction_id"] = "R3"
    frame3_eng = frame1.copy()
    frame3_eng.rotate(90, "x")
    frame3_eng.info["reaction_id"] = "R3"

    append_extxyz(ref_path, frame1)
    append_extxyz(ref_path, frame2_ref)
    append_extxyz(ref_path, frame3_ref)

    append_extxyz(eng_path, frame1)
    append_extxyz(eng_path, frame2_eng)
    append_extxyz(eng_path, frame3_eng)

    rmsd_map = ts_rmsd(eng_path, ref_path)
    assert rmsd_map["R1"] == pytest.approx(0.0, abs=1e-6)
    assert rmsd_map["R2"] > 0.0
    assert rmsd_map["R3"] == pytest.approx(0.0, abs=1e-6)


def test_ts_rmsd_missing_files(tmp_path: Path) -> None:
    """
    Verify ts_rmsd returns empty dictionary when either extxyz file is missing.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    missing1 = tmp_path / "missing1.extxyz"
    missing2 = tmp_path / "missing2.extxyz"
    assert ts_rmsd(missing1, missing2) == {}


def test_cost_row_metrics(tmp_path: Path) -> None:
    """
    Verify cost_row calculates wall hours, force calls, tasks, and error counts.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    calls_csv = tmp_path / "calls.csv"
    calls = [
        {
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "engine": "xtb",
            "system": "toy",
            "task": "species",
            "item_id": "A",
            "status": "ok",
            "wall_s": "360.0",
            "force_calls": "100",
            "device": "cpu",
            "git_commit": "abc",
            "config_hash": "def",
            "note": "",
        },
        {
            "timestamp_utc": "2026-01-01T00:01:00Z",
            "engine": "xtb",
            "system": "toy",
            "task": "species",
            "item_id": "B",
            "status": "engine_error",
            "wall_s": "180.0",
            "force_calls": "50",
            "device": "cpu",
            "git_commit": "abc",
            "config_hash": "def",
            "note": "ValueError",
        },
        {
            "timestamp_utc": "2026-01-01T00:02:00Z",
            "engine": "xtb",
            "system": "toy",
            "task": "reaction",
            "item_id": "R1",
            "status": "ok",
            "wall_s": "720.0",
            "force_calls": "200",
            "device": "cpu",
            "git_commit": "abc",
            "config_hash": "def",
            "note": "",
        },
    ]
    write_csv(calls_csv, calls)
    cost = cost_row("xtb", calls_csv)
    assert cost["engine"] == "xtb"
    assert cost["wall_h"] == pytest.approx((360.0 + 180.0 + 720.0) / 3600.0)
    assert cost["force_calls"] == 350
    assert cost["species_calls"] == 2
    assert cost["reaction_calls"] == 1
    assert cost["engine_errors"] == 1


def populate_complete_refine_records(paths: RefinePaths) -> None:
    """
    Populate species.jsonl and reactions.jsonl with energy_ev and ts_energy_ev.

    :param paths: Refinement result filesystem paths.
    :type paths: RefinePaths
    :return: None.
    :rtype: None
    """
    t298 = temperature_label(298.15)
    t600 = temperature_label(600.0)
    species_data = [
        {
            "key": "A",
            "status": STATUS_OK,
            "energy_ev": -1.0,
            "g_ev": {t298: -1.2, t600: -1.4},
        },
        {
            "key": "B",
            "status": STATUS_OK,
            "energy_ev": -2.0,
            "g_ev": {t298: -2.2, t600: -2.4},
        },
        {
            "key": "C",
            "status": STATUS_OK,
            "energy_ev": -4.0,
            "g_ev": {t298: -4.3, t600: -4.5},
        },
        {
            "key": "D",
            "status": STATUS_OK,
            "energy_ev": -5.0,
            "g_ev": {t298: -5.4, t600: -5.6},
        },
        {
            "key": "X",
            "status": "unsupported",
            "energy_ev": 0.0,
            "g_ev": {t298: 0.0, t600: 0.0},
        },
    ]
    write_jsonl(paths.species_jsonl, species_data)
    reaction_data = [
        {
            "reaction_id": "R1",
            "status": "confirmed",
            "ts_energy_ev": 0.2,
            "g_ts_ev": {t298: 0.5, t600: 0.8},
        },
        {
            "reaction_id": "R2",
            "status": "endpoint_mismatch",
            "ts_energy_ev": None,
            "g_ts_ev": None,
        },
        {
            "reaction_id": "R3",
            "status": "confirmed",
            "ts_energy_ev": 1.2,
            "g_ts_ev": {t298: 1.5, t600: 1.9},
        },
    ]
    write_jsonl(paths.reactions_jsonl, reaction_data)


def test_step_table_integration(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify step_table generates reaction rows with temperature_k for all items.

    :param fake_refine_results: Pytest fixture returning candidates and paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidates, paths = fake_refine_results
    populate_complete_refine_records(paths)
    rows = step_table(candidates, paths, 298.15)
    assert len(rows) == 3

    row_r1 = next(row for row in rows if row["reaction_id"] == "R1")
    assert row_r1["status"] == "confirmed"
    assert row_r1["temperature_k"] == 298.15
    # Reactants A (-1.0) + B (-2.0) = -3.0. Product C (-4.0).
    # de_rxn = -4.0 - (-3.0) = -1.0 eV.
    assert row_r1["de_rxn_kcal"] == pytest.approx(-1.0 / KCAL_MOL_EV)
    # Reactants A (-1.2) + B (-2.2) = -3.4. Product C (-4.3).
    # dg_rxn = -4.3 - (-3.4) = -0.9 eV.
    assert row_r1["dg_rxn_kcal"] == pytest.approx(-0.9 / KCAL_MOL_EV)
    # de_act = 0.2 - (-3.0) = 3.2 eV.
    assert row_r1["de_act_kcal"] == pytest.approx(3.2 / KCAL_MOL_EV)
    # dg_act = 0.5 - (-3.4) = 3.9 eV.
    assert row_r1["dg_act_kcal"] == pytest.approx(3.9 / KCAL_MOL_EV)

    row_r2 = next(row for row in rows if row["reaction_id"] == "R2")
    assert row_r2["status"] == "endpoint_mismatch"
    assert row_r2["de_act_kcal"] is None
    assert row_r2["dg_act_kcal"] is None

    row_r3 = next(row for row in rows if row["reaction_id"] == "R3")
    assert row_r3["status"] == "confirmed"
    # X is unsupported, so product is missing: rxn fields None
    assert row_r3["de_rxn_kcal"] is None
    assert row_r3["dg_rxn_kcal"] is None
    # Reactant C is ok, so act fields are present!
    # de_act = 1.2 - (-4.0) = 5.2 eV.
    assert row_r3["de_act_kcal"] == pytest.approx(5.2 / KCAL_MOL_EV)


def test_load_yield_cutoff_valid_and_missing(tmp_path: Path) -> None:
    """
    Verify load_yield_cutoff loads threshold or raises KeyError on missing.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cutoff = load_yield_cutoff("configs/search.yaml")
    assert cutoff == pytest.approx(0.01)

    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("evaluate: {}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_yield_cutoff(bad_cfg)


def test_ranker_rows_and_r4_yields() -> None:
    """
    Verify ranker_rows filters and sorts, and r4_yields extracts scores.

    :return: None.
    :rtype: None
    """
    ranks = [
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 2,
            "state": "P2",
            "score": 0.4,
        },
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 1,
            "state": "P1",
            "score": 0.6,
        },
        {
            "temperature_k": 600.0,
            "ranker": "R4",
            "position": 1,
            "state": "P1",
            "score": 0.9,
        },
    ]
    rows = ranker_rows(ranks, "R4", 298.15)
    assert [r["position"] for r in rows] == [1, 2]
    assert [r["state"] for r in rows] == ["P1", "P2"]

    yields = r4_yields(ranks, 298.15)
    assert yields == {"P1": 0.6, "P2": 0.4}


def test_network_agreement_hand_checked_tiny_lists() -> None:
    """
    Verify network_agreement Jaccard 1/3, tau sign, and top product/path match.

    :return: None.
    :rtype: None
    """
    eng_ranks = [
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 1,
            "state": "P1",
            "score": 0.6,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 2,
            "state": "P2",
            "score": 0.4,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R3",
            "position": 1,
            "state": "P1",
            "score": 1.0,
            "path": ["R", "A", "P1"],
        },
    ]
    ref_ranks = [
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 1,
            "state": "P1",
            "score": 0.7,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 2,
            "state": "P3",
            "score": 0.3,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R3",
            "position": 1,
            "state": "P1",
            "score": 1.0,
            "path": ["R", "A", "P1"],
        },
    ]
    agree = network_agreement(eng_ranks, ref_ranks, 298.15, 0.01)
    assert agree["jaccard"] == pytest.approx(1 / 3)
    assert agree["kendall_tau"] is not None and agree["kendall_tau"] > 0
    assert agree["same_major_product"] is True
    assert agree["same_top_path"] is True

    diff_ranks = [
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 1,
            "state": "P3",
            "score": 0.7,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 2,
            "state": "P1",
            "score": 0.1,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R3",
            "position": 1,
            "state": "P3",
            "score": 1.0,
            "path": ["R", "B", "P3"],
        },
    ]
    agree_diff = network_agreement(eng_ranks, diff_ranks, 298.15, 0.01)
    assert agree_diff["same_major_product"] is False
    assert agree_diff["same_top_path"] is False
    assert agree_diff["kendall_tau"] is not None
    assert agree_diff["kendall_tau"] < 0


def test_network_agreement_with_r4_no_barrierless() -> None:
    """
    Verify network_agreement reports _no_barrierless metrics when present.

    :return: None.
    :rtype: None
    """
    row = {
        "temperature_k": 298.15,
        "position": 1,
        "state": "P1",
        "score": 0.5,
        "path": [],
    }
    eng = [{**row, "ranker": "R4"}, {**row, "ranker": "R4_no_barrierless"}]
    ref = [{**row, "ranker": "R4"}, {**row, "ranker": "R4_no_barrierless"}]
    agree = network_agreement(eng, ref, 298.15, 0.01)
    assert agree["jaccard"] == 1.0
    assert agree["jaccard_no_barrierless"] == 1.0
    assert agree["same_major_product_no_barrierless"] is True


def test_network_agreement_none_cases_empty_inputs() -> None:
    """
    Verify network_agreement returns None values when inputs are empty.

    :return: None.
    :rtype: None
    """
    empty = network_agreement([], [], 298.15, 0.01)
    assert empty["jaccard"] is None
    assert empty["kendall_tau"] is None
    # Two networks in which nothing reacts agree on the outcome.
    assert empty["major_product"] == NO_REACTION
    assert empty["same_major_product"] is True
    assert empty["same_top_path"] is None

    single = [
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 1,
            "state": "P1",
            "score": 0.5,
            "path": [],
        }
    ]
    res_single = network_agreement(single, single, 298.15, 0.01)
    assert res_single["jaccard"] == 1.0
    assert res_single["kendall_tau"] is None
    assert res_single["same_major_product"] is True
    assert res_single["same_top_path"] is None


def test_ranker_agreement_comparison() -> None:
    """
    Verify ranker_agreement correctly scores R0-R3 against reference R4 ranks.

    :return: None.
    :rtype: None
    """
    ref_ranks = [
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 1,
            "state": "P1",
            "score": 0.7,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 2,
            "state": "P2",
            "score": 0.2,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R4",
            "position": 3,
            "state": "P3",
            "score": 0.1,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R1",
            "position": 1,
            "state": "P1",
            "score": -1.0,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R1",
            "position": 2,
            "state": "P2",
            "score": -0.5,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R1",
            "position": 3,
            "state": "P3",
            "score": -0.2,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R0",
            "position": 1,
            "state": "P2",
            "score": -2.0,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R0",
            "position": 2,
            "state": "P1",
            "score": -1.5,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R0",
            "position": 3,
            "state": "P3",
            "score": -0.1,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R2",
            "position": 1,
            "state": "P1",
            "score": -1.0,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R2",
            "position": 2,
            "state": "P4",
            "score": -0.5,
            "path": [],
        },
        {
            "temperature_k": 298.15,
            "ranker": "R2",
            "position": 3,
            "state": "P5",
            "score": -0.2,
            "path": [],
        },
    ]
    rows = ranker_agreement(ref_ranks, 298.15)
    by_ranker = {r["ranker"]: r for r in rows}
    assert by_ranker["R1"]["top1_agree"] is True
    assert by_ranker["R1"]["top3_overlap"] == 1.0
    assert by_ranker["R0"]["top1_agree"] is False
    assert by_ranker["R0"]["top3_overlap"] == 1.0
    assert by_ranker["R2"]["top1_agree"] is True
    assert by_ranker["R2"]["top3_overlap"] == pytest.approx(1 / 3)

    empty_rows = ranker_agreement([], 298.15)
    assert all(r["top1_agree"] is None for r in empty_rows)
    assert all(r["top3_overlap"] is None for r in empty_rows)


def test_evaluate_end_to_end(
    tmp_path: Path,
    fake_refine_results: tuple[CandidateSet, RefinePaths],
) -> None:
    """
    Verify evaluate writes 5 CSV files and produces jaccard 1.0 for identical.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :param fake_refine_results: Fake candidate set and refinement paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :return: None.
    :rtype: None
    """
    candidates, paths = fake_refine_results
    refine_2 = tmp_path / "refine" / "toy" / "xtb2"
    shutil.copytree(paths.out_dir, refine_2)
    paths2 = make_refine_paths(tmp_path, "toy", "xtb2")

    cand_dir = tmp_path / "candidates"
    toy_cand = cand_dir / "toy"
    toy_cand.mkdir(parents=True, exist_ok=True)
    write_candidate_files(candidates, toy_cand)

    sys_dir = tmp_path / "systems" / "toy"
    sys_dir.mkdir(parents=True, exist_ok=True)
    with (sys_dir / "system.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"ranking": {"temperatures_k": [298.15, 600.0]}}, handle)

    search_path = tmp_path / "search.yaml"
    search_data = {
        "rank": {"t_obs_s": 3600.0, "dg_threshold_ev": 0.0},
        "evaluate": {"yield_cutoff": 0.0},
    }
    with search_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(search_data, handle)

    rank_candidates(
        candidates,
        paths,
        search_path,
        make_rank_dir(tmp_path, "toy", "xtb"),
        system_yaml=sys_dir / "system.yaml",
    )
    rank_candidates(
        candidates,
        paths2,
        search_path,
        make_rank_dir(tmp_path, "toy", "xtb2"),
        system_yaml=sys_dir / "system.yaml",
    )

    eval_dir = evaluate(
        system="toy",
        reference="xtb",
        engines=["xtb2"],
        candidates_dir=cand_dir,
        results_dir=tmp_path,
        search_path=search_path,
        systems_dir=tmp_path / "systems",
    )

    assert (eval_dir / "e1_steps.csv").is_file()
    assert (eval_dir / "e1_summary.csv").is_file()
    assert (eval_dir / "e2_networks.csv").is_file()
    assert (eval_dir / "e3_rankers.csv").is_file()
    assert (eval_dir / "e4_cost.csv").is_file()

    with (eval_dir / "e2_networks.csv").open("r", encoding="utf-8") as handle:
        e2_rows = list(csv.DictReader(handle))
    assert len(e2_rows) == 2
    assert all(r["engine"] == "xtb2" for r in e2_rows)
    assert all(float(r["jaccard"]) == 1.0 for r in e2_rows)
    assert all(r["same_major_product"] == "True" for r in e2_rows)


def test_trace_product_is_not_a_major_product() -> None:
    """
    Verify a product below the yield cutoff does not count as major.

    At 600 K xTB turned 99% into aminomethanol while AIMNet2 turned
    0.0009%; R4 lists aminomethanol first for both, yet the outcomes
    differ.

    :return: None.
    :rtype: None
    """
    assert major_product({"P": 0.99}, 0.01) == "P"
    assert major_product({"P": 9e-6}, 0.01) == NO_REACTION
    rows = [
        {
            "temperature_k": 600.0,
            "ranker": "R4",
            "position": 1,
            "state": "P",
            "score": score,
            "path": [],
        }
        for score in (0.99, 9e-6)
    ]
    agree = network_agreement(rows[1:], rows[:1], 600.0, 0.01)
    assert agree["same_major_product"] is False
    assert agree["reference_major_product"] == "P"
