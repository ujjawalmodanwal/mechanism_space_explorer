"""
Unit tests for Kaggle refinement worker runner and partition execution.

Verifies parallel partition job creation, command formatting, output joining,
and system-level status aggregation across multi-part worker workspaces.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

RUN_REFINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "kaggle"
    / "run_refine.py"
)


def load_kaggle_module(module_name: str, path: Path) -> Any:
    """
    Import module dynamically from file location.

    :param module_name: Python module identifier to register.
    :type module_name: str
    :param path: Script file path.
    :type path: Path
    :return: Loaded module object.
    :rtype: Any
    :raises ImportError: If module specification or loader fails.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_REFINE = load_kaggle_module("run_refine", RUN_REFINE_PATH)


def setup_initial_main_results(main_engine_dir: Path) -> None:
    """
    Populate main workspace engine directory with initial calculation files.

    :param main_engine_dir: Target engine results directory in main workspace.
    :type main_engine_dir: Path
    """
    main_engine_dir.mkdir(parents=True, exist_ok=True)
    rxns = main_engine_dir / "reactions.jsonl"
    rxns.write_text(
        '{"id": "R000", "status": "confirmed"}\n', encoding="utf-8"
    )
    species_extxyz = main_engine_dir / "species.extxyz"
    species_extxyz.write_text(
        "2\nLattice=test energy=-10.0\nC 0 0 0\nO 0 0 1.2\n", encoding="utf-8"
    )
    calls = main_engine_dir / "calls.csv"
    calls.write_text(
        "timestamp_utc,engine,system,task,item_id,force_calls\n"
        "2026-11-20T00:00:00,uma,cpd_ma,reaction,R000,50\n",
        encoding="utf-8",
    )


def append_part_zero_results(p0_engine_dir: Path) -> None:
    """
    Simulate worker output appending for part zero.

    :param p0_engine_dir: Part zero engine directory path.
    :type p0_engine_dir: Path
    """
    with (p0_engine_dir / "reactions.jsonl").open("a", encoding="utf-8") as fp:
        fp.write('{"id": "R001", "status": "confirmed"}\n')
    with (p0_engine_dir / "species.extxyz").open("a", encoding="utf-8") as fp:
        fp.write("2\nLattice=test energy=-20.0\nC 0 0 0\nH 0 0 1.1\n")
    with (p0_engine_dir / "calls.csv").open("a", encoding="utf-8") as fp:
        fp.write("2026-11-20T00:01:00,uma,cpd_ma,reaction,R001,100\n")
    new_csv = p0_engine_dir / "new_table.csv"
    new_csv.write_text("col1,col2\nval1,val2\n", encoding="utf-8")


def append_part_one_results(p1_engine_dir: Path) -> None:
    """
    Simulate worker output appending for part one including incomplete row.

    :param p1_engine_dir: Part one engine directory path.
    :type p1_engine_dir: Path
    """
    with (p1_engine_dir / "reactions.jsonl").open("a", encoding="utf-8") as fp:
        fp.write('{"id": "R002", "status": "confirmed"}\n')
    with (p1_engine_dir / "species.extxyz").open("a", encoding="utf-8") as fp:
        fp.write("2\nLattice=test energy=-30.0\nO 0 0 0\nH 0 0 0.9\n")
    with (p1_engine_dir / "calls.csv").open("a", encoding="utf-8") as fp:
        fp.write(
            "2026-11-20T00:02:00,uma,cpd_ma,reaction,R002,150\n"
            "2026-11-20T00:03:00,uma,cpd_ma,reaction,incomplete"
        )
    new_csv = p1_engine_dir / "new_table.csv"
    new_csv.write_text("col1,col2\nval3,val4\n", encoding="utf-8")


def assert_joined_results_content(main_engine_dir: Path) -> None:
    """
    Verify correctness of joined content across jsonl, extxyz, and csv files.

    :param main_engine_dir: Joined engine results directory in main workspace.
    :type main_engine_dir: Path
    """
    rxns = (main_engine_dir / "reactions.jsonl").read_text(encoding="utf-8")
    assert rxns.splitlines() == [
        '{"id": "R000", "status": "confirmed"}',
        '{"id": "R001", "status": "confirmed"}',
        '{"id": "R002", "status": "confirmed"}',
    ]
    extxyz = (main_engine_dir / "species.extxyz").read_text(encoding="utf-8")
    assert extxyz.count("Lattice=test") == 3
    calls = (main_engine_dir / "calls.csv").read_text(encoding="utf-8")
    call_lines = calls.splitlines()
    expected_header = (
        "timestamp_utc,engine,system,task,item_id,force_calls"
    )
    assert call_lines[0] == expected_header
    assert calls.count("timestamp_utc") == 1
    assert "incomplete" not in calls
    assert len(call_lines) == 4
    new_csv = (main_engine_dir / "new_table.csv").read_text(encoding="utf-8")
    assert new_csv.splitlines() == ["col1,col2", "val1,val2", "val3,val4"]


def test_join_parts_on_temp_dirs(tmp_path: Path) -> None:
    """
    Verify joining two parts with offsets, trimmed line, and csv header.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    """
    main_dir = tmp_path / "work"
    main_engine = main_dir / "results" / "refine" / "cpd_ma" / "uma"
    setup_initial_main_results(main_engine)
    p0_dir = main_dir / "parts" / "cpd_ma-p0"
    p0_job = RUN_REFINE.RefineJob(
        job_id="cpd_ma-p0",
        system="cpd_ma",
        workdir=p0_dir,
        reaction_ids=("R001",),
        skip_species=False,
    )
    p1_dir = main_dir / "parts" / "cpd_ma-p1"
    p1_job = RUN_REFINE.RefineJob(
        job_id="cpd_ma-p1",
        system="cpd_ma",
        workdir=p1_dir,
        reaction_ids=("R002",),
        skip_species=True,
    )
    RUN_REFINE.create_part_workspace(main_dir, p0_job, "uma")
    RUN_REFINE.create_part_workspace(main_dir, p1_job, "uma")
    append_part_zero_results(p0_dir / "results" / "refine" / "cpd_ma" / "uma")
    append_part_one_results(p1_dir / "results" / "refine" / "cpd_ma" / "uma")
    RUN_REFINE.join_all_parts(main_dir, [p0_job, p1_job], "uma")
    assert_joined_results_content(main_engine)
    size_before = (main_engine / "reactions.jsonl").stat().st_size
    RUN_REFINE.join_all_parts(main_dir, [p0_job, p1_job], "uma")
    assert (main_engine / "reactions.jsonl").stat().st_size == size_before


def write_split_bundle_file(
    bundle_dir: Path,
    system: str,
    parts: list[list[str]],
) -> None:
    """
    Write mock split plan JSON inside bundle splits directory.

    :param bundle_dir: Mock bundle root directory.
    :type bundle_dir: Path
    :param system: Target system identifier.
    :type system: str
    :param parts: Partitioned lists of reaction ID strings.
    :type parts: list[list[str]]
    """
    splits_dir = bundle_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    plan_path = splits_dir / f"{system}.json"
    payload = {
        "system": system,
        "weights_from": "uma",
        "parts": parts,
        "part_weights": [100] * len(parts),
    }
    plan_path.write_text(json.dumps(payload), encoding="utf-8")


def test_job_list_for_two_part_plan(tmp_path: Path) -> None:
    """
    Verify job list creation, IDs, workdirs, skip-species, and GPU indices.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    """
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src").mkdir(parents=True)
    (bundle_dir / "BUNDLE.json").write_text("{}", encoding="utf-8")
    write_split_bundle_file(
        bundle_dir, "cpd_ma", [["R001", "R002"], ["R003", "R004"]]
    )
    work_dir = tmp_path / "work"
    jobs = RUN_REFINE.build_refine_jobs(bundle_dir, work_dir, ["cpd_ma"])
    assert len(jobs) == 2
    job0, job1 = jobs
    assert job0.job_id == "cpd_ma-p0"
    assert job0.workdir == work_dir / "parts" / "cpd_ma-p0"
    assert job0.reaction_ids == ("R001", "R002")
    assert not job0.skip_species
    assert job1.job_id == "cpd_ma-p1"
    assert job1.workdir == work_dir / "parts" / "cpd_ma-p1"
    assert job1.reaction_ids == ("R003", "R004")
    assert job1.skip_species
    gpus = 2
    assert 0 % gpus == 0
    assert 1 % gpus == 1
    cmd0 = RUN_REFINE.build_refine_command(
        job0.system, "mace", job0.reaction_ids, job0.skip_species
    )
    assert "--reactions" in cmd0 and "R001" in cmd0 and "R002" in cmd0
    assert "--skip-species" not in cmd0
    cmd1 = RUN_REFINE.build_refine_command(
        job1.system, "mace", job1.reaction_ids, job1.skip_species
    )
    assert "--reactions" in cmd1 and "R003" in cmd1 and "R004" in cmd1
    assert "--skip-species" in cmd1


def test_unsplit_system_keeps_today_command(tmp_path: Path) -> None:
    """
    Verify unsplit system produces single job with identical command tokens.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    """
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src").mkdir(parents=True)
    work_dir = tmp_path / "work"
    jobs = RUN_REFINE.build_refine_jobs(
        bundle_dir, work_dir, ["cpd_ma"], no_split=True
    )
    assert len(jobs) == 1
    job = jobs[0]
    assert job.job_id == "cpd_ma"
    assert job.workdir == work_dir
    assert job.reaction_ids is None
    assert not job.skip_species
    base_cmd = RUN_REFINE.build_refine_command("cpd_ma", "mace")
    job_cmd = RUN_REFINE.build_refine_command(
        job.system, "mace", job.reaction_ids, job.skip_species
    )
    assert job_cmd == base_cmd
    base_fmt = RUN_REFINE.format_worker_command(work_dir, "mace", "cpd_ma", 0)
    job_fmt = RUN_REFINE.format_job_command(job, "mace", 0)
    assert job_fmt == base_fmt


def test_system_with_one_failed_part_is_failed(tmp_path: Path) -> None:
    """
    Verify system outcome failure when at least one constituent part job fails.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    """
    work_dir = tmp_path / "work"
    job0 = RUN_REFINE.RefineJob(
        job_id="cpd_ma-p0",
        system="cpd_ma",
        workdir=work_dir / "parts" / "cpd_ma-p0",
    )
    job1 = RUN_REFINE.RefineJob(
        job_id="cpd_ma-p1",
        system="cpd_ma",
        workdir=work_dir / "parts" / "cpd_ma-p1",
    )
    exit_codes = {"cpd_ma-p0": 0, "cpd_ma-p1": 1}
    wall_times = {"cpd_ma-p0": 10.0, "cpd_ma-p1": 15.0}
    sys_exit, sys_wall = RUN_REFINE.aggregate_system_outcomes(
        ["cpd_ma"], [job0, job1], exit_codes, wall_times
    )
    assert sys_exit["cpd_ma"] == 1
    assert sys_wall["cpd_ma"] == 15.0
    assert RUN_REFINE.get_failed_systems(["cpd_ma"], sys_exit) == ["cpd_ma"]
    exit_codes_ok = {"cpd_ma-p0": 0, "cpd_ma-p1": 0}
    sys_exit_ok, _ = RUN_REFINE.aggregate_system_outcomes(
        ["cpd_ma"], [job0, job1], exit_codes_ok, wall_times
    )
    assert sys_exit_ok["cpd_ma"] == 0
    assert RUN_REFINE.get_failed_systems(["cpd_ma"], sys_exit_ok) == []
