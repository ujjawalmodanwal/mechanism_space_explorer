"""
Unit tests for computational engine call logging and execution records.

Verifies sequential CSV logging, exception recording and re-raising, status
and force-call counter updates, SHA-256 configuration hashing, and Git commit
detection with dirty working trees.
"""

import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nrnet.records import (
    RECORD_COLUMNS,
    STATUS_ENGINE_ERROR,
    STATUS_OK,
    CallRecord,
    RunContext,
    config_hash,
    git_commit,
    read_calls,
    record_call,
)


def create_test_context(log_path: Path) -> RunContext:
    """
    Construct a sample RunContext referencing a given log path.

    :param log_path: Path to target CSV log file.
    :type log_path: Path
    :return: Prepared RunContext instance.
    :rtype: RunContext
    """
    return RunContext(
        log_path=log_path,
        engine="xtb",
        system="hcho_nh3",
        device="cpu",
        config_hash="a1b2c3d4e5f6",
        git_commit="deadbeef",
    )


def make_test_commit(repo_dir: Path, message: str) -> None:
    """
    Execute a Git commit with test author parameters.

    :param repo_dir: Repository filesystem path.
    :type repo_dir: Path
    :param message: Commit message text.
    :type message: str
    :return: None.
    :rtype: None
    """
    cmd = [
        "git",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-m",
        message,
    ]
    subprocess.run(cmd, cwd=repo_dir, check=True)


def test_record_call_success_writes_header_and_rows(tmp_path: Path) -> None:
    """
    Verify two successful calls produce a single header and two rows in order.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    log_path = tmp_path / "calls.csv"
    context = create_test_context(log_path)
    with record_call(context, "opt", "mol_1"):
        pass
    with record_call(context, "opt", "mol_2"):
        pass
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == ",".join(RECORD_COLUMNS)
    rows = read_calls(log_path)
    assert len(rows) == 2
    assert [row["item_id"] for row in rows] == ["mol_1", "mol_2"]
    assert all(row["status"] == STATUS_OK for row in rows)


def test_record_call_exception_records_error_and_reraises(
    tmp_path: Path,
) -> None:
    """
    Verify an exception in the block records engine_error and is re-raised.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    log_path = tmp_path / "calls.csv"
    context = create_test_context(log_path)
    with pytest.raises(RuntimeError, match="calculation crashed"):
        with record_call(context, "ts_search", "pair_0"):
            raise RuntimeError("calculation crashed")
    rows = read_calls(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_ENGINE_ERROR
    assert rows[0]["note"] == "RuntimeError"
    assert rows[0]["task"] == "ts_search"
    assert rows[0]["item_id"] == "pair_0"


def test_record_call_records_status_and_force_calls(tmp_path: Path) -> None:
    """
    Verify status and force_calls modified within block are logged to CSV.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    log_path = tmp_path / "calls.csv"
    context = create_test_context(log_path)
    with record_call(context, "opt", "mol_42") as call_record:
        call_record.status = "converged"
        call_record.force_calls = 35
        call_record.note = "sella_step"
    rows = read_calls(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "converged"
    assert rows[0]["force_calls"] == "35"
    assert rows[0]["note"] == "sella_step"
    assert float(rows[0]["wall_s"]) >= 0.0


def test_config_hash_changes_on_file_change(tmp_path: Path) -> None:
    """
    Verify config_hash yields 12 hex characters and updates on modification.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    file_a = tmp_path / "engine.yaml"
    file_b = tmp_path / "system.yaml"
    file_a.write_text("model: mace\n", encoding="utf-8")
    file_b.write_text("temp: 300\n", encoding="utf-8")
    hash_1 = config_hash(file_a, file_b)
    assert len(hash_1) == 12
    file_b.write_text("temp: 500\n", encoding="utf-8")
    hash_2 = config_hash(file_a, file_b)
    assert len(hash_2) == 12
    assert hash_1 != hash_2


def test_git_commit_uncommitted_and_commit(tmp_path: Path) -> None:
    """
    Verify git_commit returns uncommitted in empty repo and hash after commit.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    git_commit.cache_clear()
    assert git_commit(tmp_path) == "uncommitted"
    (tmp_path / "input.txt").write_text("seed", encoding="utf-8")
    subprocess.run(["git", "add", "input.txt"], cwd=tmp_path, check=True)
    make_test_commit(tmp_path, "initial commit")
    git_commit.cache_clear()
    short_hash = git_commit(tmp_path)
    assert len(short_hash) >= 7
    assert short_hash != "uncommitted"


def test_git_commit_dirty_working_tree(tmp_path: Path) -> None:
    """
    Verify git_commit appends +dirty when modifications are present.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    make_test_commit(tmp_path, "add tracked")
    tracked.write_text("dirty", encoding="utf-8")
    git_commit.cache_clear()
    assert git_commit(tmp_path).endswith("+dirty")


def test_read_calls_empty_or_nonexistent(tmp_path: Path) -> None:
    """
    Verify read_calls returns empty list for non-existent file path.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    missing_file = tmp_path / "absent.csv"
    assert read_calls(missing_file) == []


def test_run_context_and_call_record_slots(tmp_path: Path) -> None:
    """
    Verify slots and immutability attributes on RunContext and CallRecord.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    context = create_test_context(tmp_path / "test.csv")
    assert hasattr(context, "__slots__")
    with pytest.raises(FrozenInstanceError):
        context.engine = "mace"
    record = CallRecord()
    assert hasattr(record, "__slots__")
    assert record.status == STATUS_OK
    assert record.force_calls == 0
    assert record.note == ""
