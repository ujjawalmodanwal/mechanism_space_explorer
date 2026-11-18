"""
Unit tests for Stage 4 refinement pipeline, resume, and signature dedup.

Verifies species geometry optimization, vibrational analysis, thermochemical
free energies across multiple temperatures, climbing-image NEB transition
state searches, signature deduplication, and restart idempotency.
"""

import json
from pathlib import Path
from typing import Any

import ase.io
import pytest
from ase import Atoms

from nrnet.chem import state_key
from nrnet.nanoreactor.candidate_store import write_candidate_files
from nrnet.nanoreactor.candidates import (
    Candidate,
    CandidateSet,
    EndpointPair,
    Species,
)
from nrnet.records import read_calls
from nrnet.refine import load_seeds, refine


def build_h2co_geometry() -> Atoms:
    """
    Construct planar formaldehyde H2CO atomic structure in YZ plane.

    :return: 3D atomic structure of formaldehyde.
    :rtype: Atoms
    """
    positions = [
        [0.0, 0.0, 0.6835],
        [0.0, 0.0, -0.5366],
        [0.0, 0.9344, -1.1242],
        [0.0, -0.9344, -1.1242],
    ]
    return Atoms(symbols=["O", "C", "H", "H"], positions=positions)


def build_hcoh_geometry() -> Atoms:
    """
    Construct planar trans-hydroxycarbene HCOH atomic structure in YZ plane.

    :return: 3D atomic structure of trans-hydroxycarbene.
    :rtype: Atoms
    """
    positions = [
        [0.0, -0.011, 0.612],
        [0.0, -0.022, -0.693],
        [0.0, 1.051, -0.993],
        [0.0, -0.917, 0.974],
    ]
    return Atoms(symbols=["O", "C", "H", "H"], positions=positions)


def build_test_species_map(
    key_r: str,
    key_p: str,
    h2co: Atoms,
    hcoh: Atoms,
) -> dict[str, Species]:
    """
    Construct Species mapping for formaldehyde and hydroxycarbene.

    :param key_r: Reactant species canonical key.
    :type key_r: str
    :param key_p: Product species canonical key.
    :type key_p: str
    :param h2co: Formaldehyde atomic structure.
    :type h2co: Atoms
    :param hcoh: Hydroxycarbene atomic structure.
    :type hcoh: Atoms
    :return: Dictionary mapping canonical keys to Species dataclasses.
    :rtype: dict[str, Species]
    """
    spec_r = Species(key_r, h2co, "C=O", None, 1)
    spec_p = Species(key_p, hcoh, "[CH]O", None, 1)
    return {key_r: spec_r, key_p: spec_p}


def build_test_candidate(
    reaction_id: str,
    key_r: str,
    key_p: str,
    pair: EndpointPair,
    first_seen_fs: float,
) -> Candidate:
    """
    Construct Candidate reaction with single 1,2-H shift endpoint pair.

    :param reaction_id: Reaction identifier string.
    :type reaction_id: str
    :param key_r: Reactant canonical species key.
    :type key_r: str
    :param key_p: Product canonical species key.
    :type key_p: str
    :param pair: Reaction endpoint pair dataclass.
    :type pair: EndpointPair
    :param first_seen_fs: Timestamp in femtoseconds.
    :type first_seen_fs: float
    :return: Candidate dataclass instance.
    :rtype: Candidate
    """
    return Candidate(
        reaction_id=reaction_id,
        state_a=(key_r,),
        state_b=(key_p,),
        reactant_keys=(key_r,),
        product_keys=(key_p,),
        pairs=(pair,),
        mapped_smiles="[O:1]=[C:2]([H:3])[H:4]>>[O:1]([H:4])-[C:2]-[H:3]",
        n_events=1,
        first_seen_fs=first_seen_fs,
    )


def setup_refine_test_system(tmp_path: Path) -> CandidateSet:
    """
    Set up candidates and system directories for refinement integration test.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: Generated CandidateSet containing R001 and duplicate R002.
    :rtype: CandidateSet
    """
    h2co, hcoh = build_h2co_geometry(), build_hcoh_geometry()
    key_r, key_p = state_key(h2co)[0], state_key(hcoh)[0]
    pair = EndpointPair(0, 100.0, (0, 1, 2, 3), h2co, hcoh)
    cand_1 = build_test_candidate("R001", key_r, key_p, pair, 100.0)
    cand_2 = build_test_candidate("R002", key_r, key_p, pair, 200.0)
    species_map = build_test_species_map(key_r, key_p, h2co, hcoh)
    cset = CandidateSet((key_r,), species_map, (cand_1, cand_2))
    cand_dir = tmp_path / "candidates" / "test_sys"
    cand_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_files(cset, cand_dir)
    sys_dir = tmp_path / "systems" / "test_sys"
    sys_dir.mkdir(parents=True, exist_ok=True)
    (sys_dir / "system.yaml").write_text("name: test_sys\n", encoding="utf-8")
    return cset


def run_test_refine(tmp_path: Path) -> None:
    """
    Execute refine runner on temporary test system with xTB engine.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    refine(
        system="test_sys",
        engine="xtb",
        systems_dir=tmp_path / "systems",
        candidates_dir=tmp_path / "candidates",
        results_dir=tmp_path / "results",
        engines_path="configs/engines.yaml",
        search_path="configs/search.yaml",
    )


def assert_species_refine_ok(tmp_path: Path, cset: CandidateSet) -> None:
    """
    Assert species rows are confirmed with free energy at both temperatures.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :param cset: Active CandidateSet instance.
    :type cset: CandidateSet
    :return: None.
    :rtype: None
    """
    res_dir = tmp_path / "results" / "refine" / "test_sys" / "xtb"
    jsonl_path = res_dir / "species.jsonl"
    assert jsonl_path.is_file()

    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == len(cset.species)
    for row in rows:
        assert row["status"] == "ok"
        assert "298.15" in row["g_ev"] and "600.0" in row["g_ev"]
        assert row["force_calls"] > 0
        assert row["g_ev"]["600.0"] < row["g_ev"]["298.15"]


def jsonl_records(path: Path) -> list[dict[str, Any]]:
    """
    Read non-empty JSON Lines from file into record list.

    :param path: Path to JSONL file.
    :type path: Path
    :return: List of parsed JSON records.
    :rtype: list[dict[str, Any]]
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def assert_reaction_record_fields(
    r001: dict[str, Any],
    r002: dict[str, Any],
    r_spec: dict[str, Any],
) -> None:
    """
    Assert status, route, and free energy properties of reaction records.

    :param r001: Confirmed R001 reaction record.
    :type r001: dict[str, Any]
    :param r002: Reused R002 reaction record.
    :type r002: dict[str, Any]
    :param r_spec: Reactant species record.
    :type r_spec: dict[str, Any]
    :return: None.
    :rtype: None
    """
    assert r001["status"] == "confirmed" and r001["route"] == "band"
    assert r001["seed_engine"] is None
    assert isinstance(r001["pair_index"], int)
    assert r001["g_ts_ev"]["298.15"] > r_spec["g_ev"]["298.15"]
    assert r001["g_ts_ev"]["600.0"] > r_spec["g_ev"]["600.0"]
    assert r002["reused_from"] == "R001" and r002["status"] == "confirmed"
    assert r002["route"] == "band" and r002["seed_engine"] is None
    assert isinstance(r002["pair_index"], int)


def assert_reactions_refine_ok(tmp_path: Path, cset: CandidateSet) -> None:
    """
    Assert R001 is confirmed with TS G above reactant G and R002 reused R001.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :param cset: Active CandidateSet instance.
    :type cset: CandidateSet
    :return: None.
    :rtype: None
    """
    dir_path = tmp_path / "results" / "refine" / "test_sys" / "xtb"
    s_rows = jsonl_records(dir_path / "species.jsonl")
    r_rows = jsonl_records(dir_path / "reactions.jsonl")
    r001 = next(r for r in r_rows if r["reaction_id"] == "R001")
    r002 = next(r for r in r_rows if r["reaction_id"] == "R002")
    r_key = cset.candidates[0].reactant_keys[0]
    r_spec = next(s for s in s_rows if s["key"] == r_key)
    assert_reaction_record_fields(r001, r002, r_spec)
    ts_frames = ase.io.read(dir_path / "ts.extxyz", index=":", format="extxyz")
    assert len(ts_frames) == 1 and ts_frames[0].info["reaction_id"] == "R001"
    assert ts_frames[0].info["pair_index"] == r001["pair_index"]


def assert_calls_refine_dedup(tmp_path: Path) -> None:
    """
    Assert calls.csv contains reaction record for R001 but none for R002.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    base_dir = tmp_path / "results" / "refine" / "test_sys" / "xtb"
    calls_path = base_dir / "calls.csv"
    calls = read_calls(calls_path)

    rxn_calls = [row["item_id"] for row in calls if row["task"] == "reaction"]
    assert "R001" in rxn_calls
    assert "R002" not in rxn_calls


def assert_second_refine_noop(tmp_path: Path) -> None:
    """
    Assert re-running refine does not add duplicate rows to JSONL and CSV files.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    dir_path = tmp_path / "results" / "refine" / "test_sys" / "xtb"
    s_len = len((dir_path / "species.jsonl").read_text().splitlines())
    r_len = len((dir_path / "reactions.jsonl").read_text().splitlines())
    c_len = len(read_calls(dir_path / "calls.csv"))
    run_test_refine(tmp_path)
    assert len((dir_path / "species.jsonl").read_text().splitlines()) == s_len
    assert len((dir_path / "reactions.jsonl").read_text().splitlines()) == r_len
    assert len(read_calls(dir_path / "calls.csv")) == c_len


def test_refine_pipeline(tmp_path: Path) -> None:
    """
    Verify Stage 4 refinement pipeline with xTB, resume, and deduplication.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cset = setup_refine_test_system(tmp_path)
    run_test_refine(tmp_path)
    assert_species_refine_ok(tmp_path, cset)
    assert_reactions_refine_ok(tmp_path, cset)
    assert_calls_refine_dedup(tmp_path)
    assert_second_refine_noop(tmp_path)


def test_load_seeds_none(tmp_path: Path) -> None:
    """
    Verify load_seeds returns empty dictionary when seed_engine is None.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    assert load_seeds(tmp_path, "test_sys", None) == {}


def test_load_seeds_missing_engine(tmp_path: Path) -> None:
    """
    Verify load_seeds raises FileNotFoundError naming missing reactions file.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    with pytest.raises(FileNotFoundError) as exc_info:
        load_seeds(tmp_path, "test_sys", "missing_engine")
    assert "missing_engine" in str(exc_info.value)


def setup_fake_seed_files(seed_dir: Path) -> None:
    """
    Write fake reactions.jsonl and ts.extxyz with confirmed and failed entries.

    :param seed_dir: Seed engine results directory.
    :type seed_dir: Path
    :return: None.
    :rtype: None
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    confirmed = {"status": "confirmed", "pair_index": 3}
    r1 = {"reaction_id": "R001", "reused_from": None, **confirmed}
    r2 = {"reaction_id": "R002", "reused_from": "R001", **confirmed}
    r3 = {
        "reaction_id": "R003",
        "status": "saddle_not_converged",
        "reused_from": None,
        "pair_index": 0,
    }
    lines = [json.dumps(r) for r in (r1, r2, r3)]
    (seed_dir / "reactions.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    frame.info["reaction_id"] = "R001"
    ase.io.write(seed_dir / "ts.extxyz", frame, format="extxyz")


def test_load_seeds_fake_files(tmp_path: Path) -> None:
    """
    Verify load_seeds loads confirmed non-reused TS with correct pair_index.

    :param tmp_path: Temporary filesystem fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    seed_dir = tmp_path / "refine" / "test_sys" / "fake_eng"
    setup_fake_seed_files(seed_dir)
    seeds = load_seeds(tmp_path, "test_sys", "fake_eng")
    assert len(seeds) == 1
    assert "R001" in seeds
    assert seeds["R001"].pair_index == 3
    assert seeds["R001"].engine == "fake_eng"
    assert seeds["R001"].atoms.calc is None
