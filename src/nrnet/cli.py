"""
Command-line interface for the nanoreactor network pipeline.

Provides subcommands for stochastic reaction discovery with xTB molecular
dynamics, candidate freezing, engine refinement, network ranking,
multi-engine evaluation, and publication figure and table generation.
"""

import argparse
import logging
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import ase.io
from ase import Atoms

from nrnet.evaluate import evaluate
from nrnet.figures import make_figures
from nrnet.nanoreactor.candidate_store import freeze
from nrnet.nanoreactor.candidates import CandidateSet
from nrnet.nanoreactor.md import (
    MDSettings,
    load_md_settings,
    read_trajectory,
    run_replicates,
)
from nrnet.rankers import RANKERS
from nrnet.ranking import RANKS_FILE, rank
from nrnet.refine import load_thermo_temperatures, refine
from nrnet.store import read_jsonl

__all__ = [
    "COMMON_ARGUMENTS",
    "DEFAULT_CANDIDATES_DIR",
    "DEFAULT_ENGINES",
    "DEFAULT_FIGURES_DIR",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_SEARCH",
    "DEFAULT_SYSTEMS_DIR",
    "DEFAULT_WORKERS",
    "ENGINE_ARGUMENTS",
    "EVALUATE_ARGUMENTS",
    "OUTPUT_ARGUMENTS",
    "add_common_arguments",
    "add_discover_subparser",
    "add_engine_arguments",
    "add_evaluate_subparser",
    "add_figures_subparser",
    "add_freeze_subparser",
    "add_rank_subparser",
    "add_refine_subparser",
    "execute_discover",
    "execute_evaluate",
    "execute_figures",
    "execute_freeze",
    "execute_rank",
    "execute_refine",
    "find_replicate_trajectories",
    "freeze_system_candidates",
    "get_parser",
    "load_all_trajectories",
    "load_system_inputs",
    "main",
    "print_freeze_stats",
    "print_rank_summary",
    "replicate_index",
]

DEFAULT_SYSTEMS_DIR: str = "systems"
DEFAULT_RUNS_DIR: str = "runs"
DEFAULT_CANDIDATES_DIR: str = "candidates"
DEFAULT_ENGINES: str = "configs/engines.yaml"
DEFAULT_RESULTS_DIR: str = "results"
DEFAULT_SEARCH: str = "configs/search.yaml"
DEFAULT_FIGURES_DIR: str = "figures"
DEFAULT_WORKERS: int = os.cpu_count() or 1

COMMON_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("--system", {"required": True, "help": "System identifier."}),
    (
        "--systems-dir",
        {
            "default": DEFAULT_SYSTEMS_DIR,
            "help": "Systems specification directory.",
        },
    ),
    (
        "--runs-dir",
        {"default": DEFAULT_RUNS_DIR, "help": "Calculation runs directory."},
    ),
    (
        "--candidates-dir",
        {
            "default": DEFAULT_CANDIDATES_DIR,
            "help": "Candidates output directory.",
        },
    ),
    (
        "--engines",
        {
            "default": DEFAULT_ENGINES,
            "help": "Engines configuration YAML file.",
        },
    ),
)

OUTPUT_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--results-dir",
        {
            "default": DEFAULT_RESULTS_DIR,
            "help": "Results output directory.",
        },
    ),
    (
        "--search",
        {
            "default": DEFAULT_SEARCH,
            "help": "Search and thermo configuration YAML file.",
        },
    ),
)

ENGINE_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--engine",
        {
            "required": True,
            "help": "Computational chemistry engine identifier.",
        },
    ),
    *OUTPUT_ARGUMENTS,
)

EVALUATE_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--reference",
        {"required": True, "help": "Reference engine identifier."},
    ),
    (
        "--compare",
        {
            "required": True,
            "help": "Comma-separated engine identifiers to compare.",
        },
    ),
    *OUTPUT_ARGUMENTS,
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Attach common system and directory flags to a subparser.

    :param parser: Subcommand argument parser instance.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    for flag, kwargs in COMMON_ARGUMENTS:
        parser.add_argument(flag, **kwargs)


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Attach engine execution flags to a subparser.

    :param parser: Subcommand argument parser instance.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    for flag, kwargs in ENGINE_ARGUMENTS:
        parser.add_argument(flag, **kwargs)


def add_discover_subparser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """
    Configure discover subcommand parser with arguments.

    :param subparsers: Subparsers action container.
    :type subparsers: Any
    :return: Configured discover argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = subparsers.add_parser("discover", help="Run nanoreactor MD.")
    add_common_arguments(parser)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes.",
    )
    parser.set_defaults(handler=execute_discover)
    return parser


def add_freeze_subparser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """
    Configure freeze subcommand parser with arguments.

    :param subparsers: Subparsers action container.
    :type subparsers: Any
    :return: Configured freeze argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = subparsers.add_parser("freeze", help="Freeze candidate set.")
    add_common_arguments(parser)
    parser.set_defaults(handler=execute_freeze)
    return parser


def add_refine_subparser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """
    Configure refine subcommand parser with arguments.

    :param subparsers: Subparsers action container.
    :type subparsers: Any
    :return: Configured refine argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = subparsers.add_parser("refine", help="Refine candidate set.")
    add_common_arguments(parser)
    add_engine_arguments(parser)
    parser.add_argument(
        "--seed-from",
        default=None,
        help=(
            "Engine whose confirmed TSs seed this search; "
            "its refine must have run first."
        ),
    )
    parser.set_defaults(handler=execute_refine)
    return parser


def add_rank_subparser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """
    Configure rank subcommand parser with arguments.

    :param subparsers: Subparsers action container.
    :type subparsers: Any
    :return: Configured rank argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = subparsers.add_parser("rank", help="Rank reaction products.")
    add_common_arguments(parser)
    add_engine_arguments(parser)
    parser.set_defaults(handler=execute_rank)
    return parser


def add_evaluate_subparser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """
    Configure evaluate subcommand parser with arguments.

    :param subparsers: Subparsers action container.
    :type subparsers: Any
    :return: Configured evaluate argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "evaluate", help="Evaluate reaction network agreement across engines."
    )
    add_common_arguments(parser)
    for flag, kwargs in EVALUATE_ARGUMENTS:
        parser.add_argument(flag, **kwargs)
    parser.set_defaults(handler=execute_evaluate)
    return parser


def add_figures_subparser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """
    Configure figures subcommand parser with arguments.

    :param subparsers: Subparsers action container.
    :type subparsers: Any
    :return: Configured figures argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "figures", help="Generate publication figures and summary tables."
    )
    add_common_arguments(parser)
    for flag, kwargs in EVALUATE_ARGUMENTS:
        parser.add_argument(flag, **kwargs)
    parser.add_argument(
        "--figures-dir",
        default=DEFAULT_FIGURES_DIR,
        help="Figures output directory.",
    )
    parser.set_defaults(handler=execute_figures)
    return parser


def get_parser() -> argparse.ArgumentParser:
    """
    Construct top-level CLI parser for nrnet commands.

    :return: Configured argument parser with subcommands.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog="nrnet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_discover_subparser(subparsers)
    add_freeze_subparser(subparsers)
    add_refine_subparser(subparsers)
    add_rank_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    add_figures_subparser(subparsers)
    return parser


def load_system_inputs(
    systems_dir: str | Path,
    system: str,
) -> tuple[Atoms, MDSettings]:
    """
    Load reactant structure and MD settings for a given system.

    :param systems_dir: Base directory containing system folders.
    :type systems_dir: str or Path
    :param system: System identifier name.
    :type system: str
    :return: Tuple of reactant geometry Atoms and MDSettings configuration.
    :rtype: tuple[Atoms, MDSettings]
    """
    sys_dir = Path(systems_dir) / system
    reactant = ase.io.read(sys_dir / "reactant.xyz")
    settings = load_md_settings(sys_dir / "system.yaml")
    return reactant, settings


def execute_discover(args: argparse.Namespace) -> int:
    """
    Execute nanoreactor discovery replicates via xTB molecular dynamics.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    reactant, settings = load_system_inputs(args.systems_dir, args.system)
    workdir = Path(args.runs_dir) / args.system / "md"
    trajectories = run_replicates(
        reactant, settings, workdir, workers=args.workers
    )
    for path in trajectories:
        print(path)
    return 0


def replicate_index(path: Path) -> int:
    """
    Extract integer replicate index from trajectory path.

    :param path: Path to replicate trajectory file.
    :type path: Path
    :return: Numerical index of replicate.
    :rtype: int
    """
    return int(path.parent.name.split("_")[-1])


def find_replicate_trajectories(md_dir: Path) -> list[Path]:
    """
    Locate all replicate trajectories sorted by replicate number.

    :param md_dir: Directory containing replicate folders.
    :type md_dir: Path
    :return: Sorted list of trajectory paths.
    :rtype: list[Path]
    """
    return sorted(md_dir.glob("rep_*/xtb.trj"), key=replicate_index)


def load_all_trajectories(md_dir: Path) -> list[list[Atoms]]:
    """
    Read all replicate trajectories sorted by replicate number.

    :param md_dir: Directory containing replicate folders.
    :type md_dir: Path
    :return: List of trajectory frame lists.
    :rtype: list[list[Atoms]]
    """
    paths = find_replicate_trajectories(md_dir)
    return [read_trajectory(path) for path in paths]


def print_freeze_stats(stats: Mapping[str, int]) -> None:
    """
    Print candidate set statistics formatted as key: value lines.

    :param stats: Candidate set statistics mapping.
    :type stats: Mapping[str, int]
    :return: None.
    :rtype: None
    """
    for key, value in stats.items():
        print(f"{key}: {value}")


def freeze_system_candidates(
    reactant: Atoms,
    settings: MDSettings,
    trajectories: list[list[Atoms]],
    engines: str,
    out_dir: Path,
) -> CandidateSet:
    """
    Freeze candidate reactions from trajectories using system settings.

    :param reactant: Initial reactant atomic structure.
    :type reactant: Atoms
    :param settings: Simulation configuration settings.
    :type settings: MDSettings
    :param trajectories: Replicate trajectory frame lists.
    :type trajectories: list[list[Atoms]]
    :param engines: Path to engines YAML configuration.
    :type engines: str
    :param out_dir: Destination candidate set directory.
    :type out_dir: Path
    :return: Frozen candidate set instance.
    :rtype: CandidateSet
    """
    min_frames = math.ceil(settings.min_lifetime_fs / settings.dump_fs)
    return freeze(
        reactant=reactant,
        trajectories=trajectories,
        min_frames=min_frames,
        dump_fs=settings.dump_fs,
        engine_settings=engines,
        out_dir=out_dir,
    )


def execute_freeze(args: argparse.Namespace) -> int:
    """
    Execute candidate extraction and freezing from discovery trajectories.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    reactant, settings = load_system_inputs(args.systems_dir, args.system)
    md_dir = Path(args.runs_dir) / args.system / "md"
    trajectories = load_all_trajectories(md_dir)
    out_dir = Path(args.candidates_dir) / args.system
    candidate_set = freeze_system_candidates(
        reactant, settings, trajectories, args.engines, out_dir
    )
    print_freeze_stats(candidate_set.stats)
    return 0


def execute_refine(args: argparse.Namespace) -> int:
    """
    Execute refinement of candidate reactions and species with an engine.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    refine(
        system=args.system,
        engine=args.engine,
        systems_dir=args.systems_dir,
        candidates_dir=args.candidates_dir,
        results_dir=args.results_dir,
        engines_path=args.engines,
        search_path=args.search,
        seed_engine=args.seed_from,
    )
    return 0


def print_rank_summary(ranks_path: Path, search_path: str | Path) -> None:
    """
    Print top ranked state and score per temperature and ranker strategy.

    :param ranks_path: Path to serialized ranks JSONL file.
    :type ranks_path: Path
    :param search_path: Path to search YAML configuration.
    :type search_path: str or Path
    :return: None.
    :rtype: None
    """
    top = {
        (float(row["temperature_k"]), row["ranker"]): row
        for row in read_jsonl(ranks_path)
        if row["position"] == 1
    }
    for temp in load_thermo_temperatures(search_path):
        for name in RANKERS:
            row = top.get((float(temp), name))
            if row is not None:
                print(f"{temp} {name} {row['state']} {row['score']}")
            else:
                print(f"{temp} {name} none")


def execute_rank(args: argparse.Namespace) -> int:
    """
    Execute ranking across temperatures for a refined system and engine.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    out_dir = rank(
        system=args.system,
        engine=args.engine,
        candidates_dir=args.candidates_dir,
        results_dir=args.results_dir,
        search_path=args.search,
    )
    print_rank_summary(out_dir / RANKS_FILE, args.search)
    return 0


def execute_evaluate(args: argparse.Namespace) -> int:
    """
    Execute multi-engine evaluation and summary table export.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    compare_engines = [
        item.strip() for item in args.compare.split(",") if item.strip()
    ]
    out_dir = evaluate(
        system=args.system,
        reference=args.reference,
        engines=compare_engines,
        candidates_dir=args.candidates_dir,
        results_dir=args.results_dir,
        search_path=args.search,
    )
    print(out_dir)
    return 0


def execute_figures(args: argparse.Namespace) -> int:
    """
    Execute publication figure and table generation for a chemical system.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    compare_engines = [
        item.strip() for item in args.compare.split(",") if item.strip()
    ]
    out_dir = make_figures(
        system=args.system,
        reference=args.reference,
        engines=compare_engines,
        candidates_dir=args.candidates_dir,
        results_dir=args.results_dir,
        search_path=args.search,
        figures_dir=args.figures_dir,
    )
    print(out_dir)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main entry point for nrnet command-line interface.

    :param argv: Optional sequence of argument strings.
    :type argv: Sequence[str] or None
    :return: Process return code.
    :rtype: int
    """
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
