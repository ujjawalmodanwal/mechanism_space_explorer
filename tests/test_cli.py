"""
Unit and integration tests for nrnet command-line interface.

Verifies CLI argument parsing for discover and freeze subcommands, module
default constants, trajectory sorting, and end-to-end nanoreactor execution.
"""

import argparse
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from nrnet.cli import (
    DEFAULT_CANDIDATES_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENGINES,
    DEFAULT_FIGURES_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEARCH,
    DEFAULT_SYSTEMS_DIR,
    DEFAULT_WORKERS,
    execute_evaluate,
    execute_figures,
    execute_rank,
    find_replicate_trajectories,
    get_parser,
    main,
    parse_float_list,
    print_rank_summary,
    replicate_index,
)
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.store import write_jsonl

CUSTOM_DISCOVER_ARGS: list[str] = [
    "discover",
    "--system",
    "custom_sys",
    "--workers",
    "4",
    "--systems-dir",
    "my_systems",
    "--runs-dir",
    "my_runs",
    "--candidates-dir",
    "my_candidates",
    "--engines",
    "my_engines.yaml",
]

CUSTOM_FREEZE_ARGS: list[str] = [
    "freeze",
    "--system",
    "custom_sys",
    "--systems-dir",
    "my_systems",
    "--runs-dir",
    "my_runs",
    "--candidates-dir",
    "my_candidates",
    "--engines",
    "my_engines.yaml",
]

CUSTOM_REFINE_ARGS: list[str] = [
    "refine",
    "--system",
    "custom_sys",
    "--engine",
    "xtb",
    "--systems-dir",
    "my_systems",
    "--runs-dir",
    "my_runs",
    "--candidates-dir",
    "my_candidates",
    "--engines",
    "my_engines.yaml",
    "--results-dir",
    "my_results",
    "--search",
    "my_search.yaml",
]

FIGURES_ARGS: list[str] = [
    "figures",
    "--system",
    "toy",
    "--reference",
    "xtb",
    "--compare",
    "aimnet",
]


def test_discover_parser_defaults() -> None:
    """
    Verify discover subcommand parser assigns expected module defaults.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    args = parser.parse_args(["discover", "--system", "hcho_nh3"])
    assert args.command == "discover"
    assert args.system == "hcho_nh3"
    assert args.systems_dir == DEFAULT_SYSTEMS_DIR
    assert args.runs_dir == DEFAULT_RUNS_DIR
    assert args.candidates_dir == DEFAULT_CANDIDATES_DIR
    assert args.engines == DEFAULT_ENGINES
    assert args.workers == DEFAULT_WORKERS


def assert_custom_discover_args(args: argparse.Namespace) -> None:
    """
    Assert parsed discover arguments match expected custom values.

    :param args: Parsed arguments to assert.
    :type args: argparse.Namespace
    :return: None.
    :rtype: None
    """
    assert args.command == "discover" and args.system == "custom_sys"
    assert args.workers == 4 and args.systems_dir == "my_systems"
    assert args.runs_dir == "my_runs"
    assert args.candidates_dir == "my_candidates"
    assert args.engines == "my_engines.yaml"


def test_discover_parser_custom_options() -> None:
    """
    Verify discover subcommand parser overrides options with user values.

    :return: None.
    :rtype: None
    """
    args = get_parser().parse_args(CUSTOM_DISCOVER_ARGS)
    assert_custom_discover_args(args)


def test_freeze_parser_defaults() -> None:
    """
    Verify freeze subcommand parser assigns expected module defaults.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    args = parser.parse_args(["freeze", "--system", "hcho_nh3"])
    assert args.command == "freeze"
    assert args.system == "hcho_nh3"
    assert args.systems_dir == DEFAULT_SYSTEMS_DIR
    assert args.runs_dir == DEFAULT_RUNS_DIR
    assert args.candidates_dir == DEFAULT_CANDIDATES_DIR
    assert args.engines == DEFAULT_ENGINES
    assert args.add_literature is False


def test_freeze_parser_add_literature_option() -> None:
    """
    Verify freeze subcommand parser sets add_literature flag when passed.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    args = parser.parse_args(
        ["freeze", "--system", "hcho_nh3", "--add-literature"]
    )
    assert args.command == "freeze"
    assert args.add_literature is True


def assert_custom_freeze_args(args: argparse.Namespace) -> None:
    """
    Assert parsed freeze arguments match expected custom values.

    :param args: Parsed arguments to assert.
    :type args: argparse.Namespace
    :return: None.
    :rtype: None
    """
    assert args.command == "freeze" and args.system == "custom_sys"
    assert args.systems_dir == "my_systems"
    assert args.runs_dir == "my_runs"
    assert args.candidates_dir == "my_candidates"
    assert args.engines == "my_engines.yaml"


def test_freeze_parser_custom_options() -> None:
    """
    Verify freeze subcommand parser overrides options with user values.

    :return: None.
    :rtype: None
    """
    args = get_parser().parse_args(CUSTOM_FREEZE_ARGS)
    assert_custom_freeze_args(args)


def test_sweep_parser_options() -> None:
    """
    Verify sweep subcommand parser correctly sets grid flags and defaults.

    :return: None.
    :rtype: None
    """
    cli_str = (
        "sweep --system hcho_nh3 --kpush-per-atom 0.015,0.02 "
        "--temperatures 300,600 --groups 2 --first-group 1"
    )
    args = get_parser().parse_args(cli_str.split())
    assert args.command == "sweep" and args.system == "hcho_nh3"
    assert args.kpush_per_atom == "0.015,0.02"
    assert args.temperatures == "300,600"
    assert args.groups == 2 and args.first_group == 1
    assert args.workers == DEFAULT_WORKERS
    assert args.data_dir == DEFAULT_DATA_DIR


def test_freeze_parser_from_runs_option() -> None:
    """
    Verify freeze subcommand parses optional --from-runs directory paths.

    :return: None.
    :rtype: None
    """
    cli_args = ["freeze", "--system", "hcho_nh3", "--from-runs", "d1", "d2"]
    args = get_parser().parse_args(cli_args)
    assert args.command == "freeze"
    assert args.from_runs == ["d1", "d2"]


def test_parse_float_list_helper() -> None:
    """
    Verify parsing of comma-separated numeric strings into float sequences.

    :return: None.
    :rtype: None
    """
    parsed = parse_float_list("0.015, 0.02, 0.025")
    assert parsed == [0.015, 0.02, 0.025]
    assert parse_float_list([1.0, 2.5]) == [1.0, 2.5]


def test_refine_parser_defaults() -> None:
    """
    Verify refine subcommand parser assigns expected module defaults.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    args = parser.parse_args(
        ["refine", "--system", "hcho_nh3", "--engine", "xtb"]
    )
    assert args.command == "refine"
    assert args.system == "hcho_nh3" and args.engine == "xtb"
    assert args.systems_dir == DEFAULT_SYSTEMS_DIR
    assert args.runs_dir == DEFAULT_RUNS_DIR
    assert args.candidates_dir == DEFAULT_CANDIDATES_DIR
    assert args.engines == DEFAULT_ENGINES
    assert args.results_dir == DEFAULT_RESULTS_DIR
    assert args.search == DEFAULT_SEARCH
    assert args.seed_from is None
    assert args.reactions is None
    assert args.skip_species is False


def assert_custom_refine_args(args: argparse.Namespace) -> None:
    """
    Assert parsed refine arguments match expected custom values.

    :param args: Parsed arguments to assert.
    :type args: argparse.Namespace
    :return: None.
    :rtype: None
    """
    assert args.command == "refine" and args.system == "custom_sys"
    assert args.engine == "xtb" and args.systems_dir == "my_systems"
    assert args.runs_dir == "my_runs" and args.candidates_dir == "my_candidates"
    assert args.engines == "my_engines.yaml"
    assert args.results_dir == "my_results" and args.search == "my_search.yaml"
    assert args.seed_from is None
    assert args.reactions is None
    assert args.skip_species is False


def test_refine_parser_reactions_and_skip_species() -> None:
    """
    Verify refine subcommand parses reactions and skip-species flags.

    :return: None.
    :rtype: None
    """
    args = get_parser().parse_args(
        [
            "refine",
            "--system",
            "toy",
            "--engine",
            "xtb",
            "--reactions",
            "R001",
            "R002",
            "--skip-species",
        ]
    )
    assert args.reactions == ["R001", "R002"]
    assert args.skip_species is True


def test_execute_refine_passes_reactions_and_skip_species(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify execute_refine passes reactions and skip_species through.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :return: None.
    :rtype: None
    """
    captured: dict[str, Any] = {}

    def fake_refine(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("nrnet.cli.refine", fake_refine)
    parser = get_parser()
    args = parser.parse_args(
        [
            "refine",
            "--system",
            "toy",
            "--engine",
            "xtb",
            "--reactions",
            "R001",
            "R002",
            "--skip-species",
        ]
    )
    assert args.handler(args) == 0
    assert captured["reactions"] == ["R001", "R002"]
    assert captured["skip_species"] is True


def test_refine_parser_seed_from() -> None:
    """
    Verify refine subcommand parses seed-from option correctly.

    :return: None.
    :rtype: None
    """
    args = get_parser().parse_args(
        [
            "refine",
            "--system",
            "toy",
            "--engine",
            "xtb",
            "--seed-from",
            "aimnet",
        ]
    )
    assert args.seed_from == "aimnet"


def test_refine_parser_custom_options() -> None:
    """
    Verify refine subcommand parser overrides options with user values.

    :return: None.
    :rtype: None
    """
    args = get_parser().parse_args(CUSTOM_REFINE_ARGS)
    assert_custom_refine_args(args)


def test_rank_parser_defaults() -> None:
    """
    Verify rank subcommand parser assigns expected module defaults.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    args = parser.parse_args(["rank", "--system", "toy", "--engine", "xtb"])
    assert args.command == "rank"
    assert args.system == "toy" and args.engine == "xtb"
    assert args.handler == execute_rank
    assert args.results_dir == DEFAULT_RESULTS_DIR
    assert args.search == DEFAULT_SEARCH


def test_evaluate_parser_defaults() -> None:
    """
    Verify evaluate subcommand parses expected arguments and sets handler.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--system",
            "toy",
            "--reference",
            "xtb",
            "--compare",
            "aimnet,ani1xnr",
        ]
    )
    assert args.command == "evaluate"
    assert args.system == "toy"
    assert args.reference == "xtb"
    assert args.compare == "aimnet,ani1xnr"
    assert args.handler == execute_evaluate
    assert args.results_dir == DEFAULT_RESULTS_DIR
    assert args.search == DEFAULT_SEARCH


def test_evaluate_parser_requires_arguments() -> None:
    """
    Verify evaluate parser fails when reference or compare is missing.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["evaluate", "--system", "toy", "--compare", "aimnet"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--system", "toy", "--reference", "xtb"])


def test_figures_parser_defaults() -> None:
    """
    Verify figures subcommand parses expected arguments and sets handler.

    :return: None.
    :rtype: None
    """
    args = get_parser().parse_args(FIGURES_ARGS)
    assert args.command == "figures"
    assert args.system == "toy" and args.reference == "xtb"
    assert args.compare == "aimnet"
    assert args.figures_dir == DEFAULT_FIGURES_DIR
    assert args.handler == execute_figures


def test_figures_parser_requires_arguments() -> None:
    """
    Verify figures parser fails when reference or compare is missing.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["figures", "--system", "toy", "--compare", "aimnet"])
    with pytest.raises(SystemExit):
        parser.parse_args(["figures", "--system", "toy", "--reference", "xtb"])


def make_summary_test_files(tmp_path: Path) -> tuple[Path, Path]:
    """
    Create temporary system and ranks files for summary output test.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: Tuple of system YAML and ranks JSONL paths.
    :rtype: tuple[Path, Path]
    """
    system_path = tmp_path / "system.yaml"
    with system_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"ranking": {"temperatures_k": [298.15]}}, handle)
    ranks_path = tmp_path / "ranks.jsonl"
    row = {
        "temperature_k": 298.15,
        "ranker": "R0",
        "position": 1,
        "state": "P1",
        "score": -1.5,
        "path": [],
    }
    write_jsonl(ranks_path, [row])
    return system_path, ranks_path


def test_print_rank_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify print_rank_summary formats top product state and score or none.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :param capsys: Pytest standard output capture fixture.
    :type capsys: pytest.CaptureFixture[str]
    :return: None.
    :rtype: None
    """
    system_path, ranks_path = make_summary_test_files(tmp_path)
    print_rank_summary(ranks_path, system_path)
    output = capsys.readouterr().out
    assert "298.15 R0 P1 -1.5" in output
    assert "298.15 R1 none" in output


def test_parser_requires_system() -> None:
    """
    Verify argument parser fails when required --system argument is missing.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["discover"])
    with pytest.raises(SystemExit):
        parser.parse_args(["freeze"])
    with pytest.raises(SystemExit):
        parser.parse_args(["refine", "--engine", "xtb"])


def test_refine_parser_requires_engine() -> None:
    """
    Verify refine argument parser fails when required --engine is missing.

    :return: None.
    :rtype: None
    """
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["refine", "--system", "hcho_nh3"])


def test_replicate_sorting(tmp_path: Path) -> None:
    """
    Verify trajectory replicate discovery sorts by numeric replicate index.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    for idx in (10, 2, 0, 1):
        rep_dir = tmp_path / f"rep_{idx}"
        rep_dir.mkdir()
        (rep_dir / "xtb.trj").touch()
    sorted_paths = find_replicate_trajectories(tmp_path)
    indices = [replicate_index(path) for path in sorted_paths]
    assert indices == [0, 1, 2, 10]


def setup_slow_test_system(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Set up system files with short duration for slow integration test.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: Tuple of systems, runs, and candidates directory paths.
    :rtype: tuple[Path, Path, Path]
    """
    sys_dir = tmp_path / "systems"
    runs_dir = tmp_path / "runs"
    cand_dir = tmp_path / "candidates"
    shutil.copytree("systems/hcho_nh3", sys_dir / "hcho_nh3")
    yaml_file = sys_dir / "hcho_nh3" / "system.yaml"
    with yaml_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["md"]["time_ps"] = 0.5
    data["md"]["replicates"] = 2
    data["md"]["equilibrate_ps"] = 0.0
    with yaml_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return sys_dir, runs_dir, cand_dir


def run_test_discover(sys_dir: Path, runs_dir: Path, cand_dir: Path) -> int:
    """
    Execute discover subcommand for end-to-end test.

    :param sys_dir: Systems directory path.
    :type sys_dir: Path
    :param runs_dir: Runs directory path.
    :type runs_dir: Path
    :param cand_dir: Candidates directory path.
    :type cand_dir: Path
    :return: Exit code.
    :rtype: int
    """
    cmd = [
        "discover",
        "--system",
        "hcho_nh3",
        "--systems-dir",
        str(sys_dir),
        "--runs-dir",
        str(runs_dir),
        "--candidates-dir",
        str(cand_dir),
        "--workers",
        "2",
    ]
    return main(cmd)


def run_test_freeze(sys_dir: Path, runs_dir: Path, cand_dir: Path) -> int:
    """
    Execute freeze subcommand for end-to-end test.

    :param sys_dir: Systems directory path.
    :type sys_dir: Path
    :param runs_dir: Runs directory path.
    :type runs_dir: Path
    :param cand_dir: Candidates directory path.
    :type cand_dir: Path
    :return: Exit code.
    :rtype: int
    """
    cmd = [
        "freeze",
        "--system",
        "hcho_nh3",
        "--systems-dir",
        str(sys_dir),
        "--runs-dir",
        str(runs_dir),
        "--candidates-dir",
        str(cand_dir),
    ]
    return main(cmd)


def assert_freeze_output(cand_dir: Path, freeze_out: str) -> None:
    """
    Assert manifest integrity and candidate loading from frozen outputs.

    :param cand_dir: Root candidates directory.
    :type cand_dir: Path
    :param freeze_out: Standard output captured from freeze command.
    :type freeze_out: str
    :return: None.
    :rtype: None
    """
    assert "events:" in freeze_out
    manifest = cand_dir / "hcho_nh3" / "manifest.json"
    assert manifest.is_file()
    cset = load_candidates(cand_dir / "hcho_nh3")
    assert isinstance(cset.stats, dict) and "events" in cset.stats


@pytest.mark.slow
def test_cli_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify end-to-end discovery and freeze subcommands on HCHO + NH3.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :param capsys: Pytest standard output capture fixture.
    :type capsys: pytest.CaptureFixture[str]
    :return: None.
    :rtype: None
    """
    sys_dir, runs_dir, cand_dir = setup_slow_test_system(tmp_path)
    assert run_test_discover(sys_dir, runs_dir, cand_dir) == 0
    assert "xtb.trj" in capsys.readouterr().out
    assert run_test_freeze(sys_dir, runs_dir, cand_dir) == 0
    assert_freeze_output(cand_dir, capsys.readouterr().out)
