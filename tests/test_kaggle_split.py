"""
Unit tests for Kaggle reaction split planning and bundle packaging.

Verifies greedy partition balancing, deterministic ID assignments, median
weight imputation for missing reaction entries, and validation of split files
bundled into deployment packages.
"""

import csv
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest


def load_kaggle_module(module_name: str, script_name: str) -> Any:
    """
    Import module dynamically from scripts/kaggle directory.

    :param module_name: Python module identifier to register.
    :type module_name: str
    :param script_name: Script file basename.
    :type script_name: str
    :return: Loaded module object.
    :rtype: Any
    """
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "kaggle" / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN_SPLIT = load_kaggle_module("plan_split", "plan_split.py")
MAKE_BUNDLE = load_kaggle_module("make_bundle", "make_bundle.py")


def test_greedy_assignment_balanced_and_deterministic() -> None:
    """
    Verify greedy reaction partition balances weights and is deterministic.
    """
    weights = {
        "R01": 100,
        "R02": 80,
        "R03": 70,
        "R04": 50,
        "R05": 40,
        "R06": 20,
    }
    parts, part_weights = PLAN_SPLIT.assign_greedy(weights, n_parts=2)
    assert len(parts) == 2
    assert part_weights == [190, 170]
    all_assigned = [item for part in parts for item in part]
    assert sorted(all_assigned) == sorted(weights.keys())
    for part in parts:
        assert part == sorted(part)

    shuffled_weights = {
        "R06": 20,
        "R04": 50,
        "R01": 100,
        "R05": 40,
        "R02": 80,
        "R03": 70,
    }
    shuffled_parts, shuffled_weights_out = PLAN_SPLIT.assign_greedy(
        shuffled_weights, n_parts=2
    )
    assert parts == shuffled_parts
    assert part_weights == shuffled_weights_out


def test_greedy_assignment_tie_breaking() -> None:
    """
    Verify ties in part weights break toward the lowest part index.
    """
    weights = {"R01": 50, "R02": 50, "R03": 10}
    parts, part_weights = PLAN_SPLIT.assign_greedy(weights, n_parts=2)
    assert parts[0] == ["R01", "R03"]
    assert parts[1] == ["R02"]
    assert part_weights == [60, 50]


def test_missing_weight_ids_receive_median(tmp_path: Path) -> None:
    """
    Verify candidates missing from calls.csv are imputed with median weight.
    """
    calls_csv = tmp_path / "calls.csv"
    rows = [
        ["timestamp_utc", "engine", "system", "task", "item_id", "force_calls"],
        ["2026-11-20T00:00:00", "uma", "sys_x", "species", "sp1", "9999"],
        ["2026-11-20T00:01:00", "uma", "sys_x", "reaction", "R01", "100"],
        ["2026-11-20T00:02:00", "uma", "sys_x", "reaction", "R02", "200"],
        ["2026-11-20T00:03:00", "uma", "sys_x", "reaction", "R02", "300"],
        ["2026-11-20T00:04:00", "uma", "sys_x", "reaction", "R03", "500"],
    ]
    with open(calls_csv, mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    candidate_ids = ["R01", "R02", "R03", "R04"]
    weights = PLAN_SPLIT.read_weights(calls_csv, candidate_ids)
    assert weights["R01"] == 100
    assert weights["R02"] == 300
    assert weights["R03"] == 500
    assert weights["R04"] == 300


def test_make_bundle_rejects_unlisted_system_split(tmp_path: Path) -> None:
    """
    Verify make_bundle raises ValueError when split system is not listed.
    """
    split_path = tmp_path / "split_other.json"
    payload = {
        "system": "unknown_system",
        "weights_from": "uma",
        "parts": [["R01"]],
        "part_weights": [100],
    }
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown_system"):
        MAKE_BUNDLE.collect_split_files([split_path], systems=["cpd_ma"])


def test_make_bundle_includes_split_file(tmp_path: Path) -> None:
    """
    Verify valid split file is packaged at splits/<system>.json in bundle.
    """
    split_path = tmp_path / "split_cpd_ma.json"
    payload = {
        "system": "cpd_ma",
        "weights_from": "uma",
        "parts": [["R001"], ["R002"]],
        "part_weights": [100, 100],
    }
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    entries = MAKE_BUNDLE.collect_split_files([split_path], systems=["cpd_ma"])
    assert len(entries) == 1
    assert entries[0] == (split_path, "splits/cpd_ma.json")


def test_make_bundle_without_split_identical(tmp_path: Path) -> None:
    """
    Verify collect_split_files returns empty list when no split is supplied.
    """
    entries_none = MAKE_BUNDLE.collect_split_files(None, systems=["cpd_ma"])
    entries_empty = MAKE_BUNDLE.collect_split_files([], systems=["cpd_ma"])
    assert entries_none == []
    assert entries_empty == []


def test_build_bundle_rejects_unlisted_system_split(tmp_path: Path) -> None:
    """
    Verify build_bundle raises ValueError for split with unlisted system.
    """
    root = Path(__file__).resolve().parent.parent
    split_path = tmp_path / "split_other.json"
    payload = {
        "system": "unknown_system",
        "weights_from": "uma",
        "parts": [["R01"]],
        "part_weights": [100],
    }
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    out_zip = tmp_path / "bundle.zip"
    with pytest.raises(ValueError, match="unknown_system"):
        MAKE_BUNDLE.build_bundle(
            repo_root=root,
            engine="uma",
            systems=["cpd_ma"],
            out_path=out_zip,
            split_paths=[split_path],
        )


def test_format_enable_cueq_line() -> None:
    """
    Verify format_enable_cueq_line preserves indent and trailing comment.

    :return: None.
    :rtype: None
    """
    line_plain = "  enable_cueq: false\n"
    assert MAKE_BUNDLE.format_enable_cueq_line(line_plain) == (
        "  enable_cueq: true\n"
    )

    line_comment = "    enable_cueq: false  # gpu only\n"
    assert MAKE_BUNDLE.format_enable_cueq_line(line_comment) == (
        "    enable_cueq: true  # gpu only\n"
    )


def test_update_engine_cueq_mace_only() -> None:
    """
    Verify update_engine_cueq rewrites enable_cueq only for target engine.

    :return: None.
    :rtype: None
    """
    lines = [
        "xtb:\n",
        "  device: cpu\n",
        "mace:\n",
        "  device: cpu\n",
        "  enable_cueq: false  # comment\n",
        "uma:\n",
        "  device: cpu\n",
    ]
    updated = MAKE_BUNDLE.update_engine_cueq(lines, target_engine="mace")
    expected = (
        "xtb:\n"
        "  device: cpu\n"
        "mace:\n"
        "  device: cpu\n"
        "  enable_cueq: true  # comment\n"
        "uma:\n"
        "  device: cpu\n"
    )
    assert updated == expected


def test_update_engine_cueq_missing_raises() -> None:
    """
    Verify update_engine_cueq raises ValueError when engine or key is missing.

    :return: None.
    :rtype: None
    """
    lines = ["mace:\n", "  device: cpu\n"]
    with pytest.raises(ValueError, match="enable_cueq: setting not found"):
        MAKE_BUNDLE.update_engine_cueq(lines, target_engine="mace")

    with pytest.raises(ValueError, match="Engine 'unknown' not found"):
        MAKE_BUNDLE.update_engine_cueq(lines, target_engine="unknown")


def test_load_engines_config_enable_cueq_flag() -> None:
    """
    Verify load_engines_config rewrites enable_cueq only when flag is True.

    :return: None.
    :rtype: None
    """
    root = Path(__file__).resolve().parent.parent
    text_default, _ = MAKE_BUNDLE.load_engines_config(
        root, "mace", enable_cueq=False
    )
    assert "enable_cueq: false" in text_default
    assert "enable_cueq: true" not in text_default
    assert "device: cuda" in text_default

    text_flagged, _ = MAKE_BUNDLE.load_engines_config(
        root, "mace", enable_cueq=True
    )
    assert "enable_cueq: true" in text_flagged
    assert "enable_cueq: false" not in text_flagged
    assert "device: cuda" in text_flagged


def test_build_bundle_enable_cueq_flag(tmp_path: Path) -> None:
    """
    Verify build_bundle sets enable_cueq true in zip only with the flag.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    root = Path(__file__).resolve().parent.parent
    zip_default = tmp_path / "bundle_default.zip"
    zip_cueq = tmp_path / "bundle_cueq.zip"

    MAKE_BUNDLE.build_bundle(
        repo_root=root,
        engine="mace",
        systems=["cpd_ma"],
        out_path=zip_default,
        enable_cueq=False,
    )
    with zipfile.ZipFile(zip_default, "r") as arc:
        yaml_default = arc.read("configs/engines.yaml").decode("utf-8")
    assert "enable_cueq: false" in yaml_default
    assert "enable_cueq: true" not in yaml_default

    MAKE_BUNDLE.build_bundle(
        repo_root=root,
        engine="mace",
        systems=["cpd_ma"],
        out_path=zip_cueq,
        enable_cueq=True,
    )
    with zipfile.ZipFile(zip_cueq, "r") as arc:
        yaml_cueq = arc.read("configs/engines.yaml").decode("utf-8")
    assert "enable_cueq: true" in yaml_cueq
    assert "enable_cueq: false" not in yaml_cueq


def test_build_parser_enable_cueq_flag() -> None:
    """
    Verify --enable-cueq flag is recognized by make_bundle argument parser.

    :return: None.
    :rtype: None
    """
    parser = MAKE_BUNDLE.build_parser()
    args_default = parser.parse_args(
        ["--engine", "mace", "--systems", "cpd_ma", "--out", "bundle.zip"]
    )
    assert args_default.enable_cueq is False

    args_flag = parser.parse_args(
        [
            "--engine",
            "mace",
            "--systems",
            "cpd_ma",
            "--out",
            "bundle.zip",
            "--enable-cueq",
        ]
    )
    assert args_flag.enable_cueq is True
