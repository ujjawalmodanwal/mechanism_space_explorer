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

from ase import Atoms

from nrnet.evaluate import evaluate
from nrnet.figures import make_figures
from nrnet.literature import (
    add_literature_to_candidates,
    load_literature_reactions,
)
from nrnet.nanoreactor.candidate_store import freeze, write_candidate_files
from nrnet.nanoreactor.candidates import CandidateSet
from nrnet.nanoreactor.md import (
    MDSettings,
    find_replicate_trajectories,
    load_all_trajectories,
    load_md_settings,
    load_system_reactants,
    read_trajectory,
    replicate_index,
    run_replicates,
)
from nrnet.rankers import RANKERS
from nrnet.ranking import RANKS_FILE, rank
from nrnet.refine import load_thermo_temperatures, refine
from nrnet.store import read_jsonl
from nrnet.sweep import run_sweep

__all__ = [
    "COMMON_ARGUMENTS",
    "DEFAULT_CANDIDATES_DIR",
    "DEFAULT_DATA_DIR",
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
    "REFINE_ARGUMENTS",
    "SWEEP_ARGUMENTS",
    "add_common_arguments",
    "add_discover_subparser",
    "add_engine_arguments",
    "add_evaluate_subparser",
    "add_figures_subparser",
    "add_freeze_arguments",
    "add_freeze_subparser",
    "add_rank_subparser",
    "add_refine_arguments",
    "add_refine_subparser",
    "add_sweep_subparser",
    "apply_freeze_literature",
    "execute_discover",
    "execute_evaluate",
    "execute_figures",
    "execute_freeze",
    "execute_rank",
    "execute_refine",
    "execute_sweep",
    "find_replicate_trajectories",
    "freeze_from_cli_args",
    "freeze_system_candidates",
    "get_parser",
    "load_all_trajectories",
    "load_from_runs_trajectories",
    "load_system_inputs",
    "main",
    "parse_float_list",
    "print_freeze_stats",
    "print_rank_summary",
    "replicate_index",
    "resolve_freeze_trajectories",
]

DEFAULT_SYSTEMS_DIR: str = "systems"
DEFAULT_RUNS_DIR: str = "runs"
DEFAULT_CANDIDATES_DIR: str = "candidates"
DEFAULT_DATA_DIR: str = "data"
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

REFINE_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--seed-from",
        {
            "default": None,
            "help": (
                "Engine whose confirmed TSs seed this search; "
                "its refine must have run first."
            ),
        },
    ),
    (
        "--retry",
        {
            "nargs": "+",
            "default": (),
            "help": "Statuses to re-search instead of skipping.",
        },
    ),
    (
        "--reactions",
        {
            "nargs": "+",
            "metavar": "ID",
            "default": None,
            "help": "Refine only these reaction IDs.",
        },
    ),
    (
        "--skip-species",
        {
            "action": "store_true",
            "help": "Skip the species stage.",
        },
    ),
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

SWEEP_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--kpush-per-atom",
        {
            "required": True,
            "help": "Comma-separated push strengths per atom.",
        },
    ),
    (
        "--temperatures",
        {
            "required": True,
            "help": "Comma-separated target temperatures in Kelvin.",
        },
    ),
    (
        "--groups",
        {
            "type": int,
            "default": 1,
            "help": "Total number of replicate groups per setting.",
        },
    ),
    (
        "--first-group",
        {
            "type": int,
            "default": 1,
            "help": "First group index identifier.",
        },
    ),
    (
        "--workers",
        {
            "type": int,
            "default": DEFAULT_WORKERS,
            "help": "Number of parallel worker processes.",
        },
    ),
    (
        "--data-dir",
        {
            "default": DEFAULT_DATA_DIR,
            "help": "Data output directory.",
        },
    ),
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


def add_refine_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Attach refine search flags to a subparser.

    :param parser: Subcommand argument parser instance.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    for flag, kwargs in REFINE_ARGUMENTS:
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


def add_sweep_subparser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """
    Configure sweep subcommand parser with arguments.

    :param subparsers: Subparsers action container.
    :type subparsers: Any
    :return: Configured sweep argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = subparsers.add_parser(
        "sweep", help="Screen MD parameters across grid and write table."
    )
    add_common_arguments(parser)
    for flag, kwargs in SWEEP_ARGUMENTS:
        parser.add_argument(flag, **kwargs)
    parser.set_defaults(handler=execute_sweep)
    return parser


def add_freeze_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register arguments specific to the freeze subcommand.

    :param parser: Freeze subcommand parser.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    parser.add_argument(
        "--from-runs",
        nargs="+",
        default=None,
        help="Sweep run directories to freeze from.",
    )
    parser.add_argument(
        "--add-literature",
        action="store_true",
        default=False,
        help="Add literature reactions to the frozen candidate set.",
    )


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
    add_freeze_arguments(parser)
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
    add_refine_arguments(parser)
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
    add_sweep_subparser(subparsers)
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
    reactants = load_system_reactants(sys_dir)
    settings = load_md_settings(sys_dir / "system.yaml")
    return reactants[0][1], settings


def execute_discover(args: argparse.Namespace) -> int:
    """
    Execute nanoreactor discovery replicates via xTB molecular dynamics.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    sys_dir = Path(args.systems_dir) / args.system
    reactants = load_system_reactants(sys_dir)
    settings = load_md_settings(sys_dir / "system.yaml")
    workdir = Path(args.runs_dir) / args.system / "md"
    trajectories = run_replicates(
        reactants, settings, workdir, workers=args.workers
    )
    for path in trajectories:
        print(path)
    return 0


def parse_float_list(arg: str | Sequence[float]) -> list[float]:
    """
    Parse comma-separated string or sequence into list of float values.

    :param arg: Comma-separated string or sequence of numbers.
    :type arg: str or Sequence[float]
    :return: List of parsed float values.
    :rtype: list[float]
    """
    if isinstance(arg, str):
        return [float(x.strip()) for x in arg.split(",") if x.strip()]
    return [float(x) for x in arg]


def execute_sweep(args: argparse.Namespace) -> int:
    """
    Execute parameter sweep over MD settings and write metrics table.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    csv_path = run_sweep(
        system=args.system,
        kpush_per_atom=parse_float_list(args.kpush_per_atom),
        temperatures=parse_float_list(args.temperatures),
        groups=args.groups,
        first_group=args.first_group,
        workers=args.workers,
        systems_dir=args.systems_dir,
        runs_dir=args.runs_dir,
        data_dir=getattr(args, "data_dir", DEFAULT_DATA_DIR),
        engines=args.engines,
    )
    print(csv_path)
    return 0


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


def load_from_runs_trajectories(
    run_dirs: Sequence[str | Path],
) -> tuple[list[list[Atoms]], list[tuple[str, int]]]:
    """
    Load trajectories and build (folder, replicate) identities across runs.

    :param run_dirs: Sequence of run directory paths.
    :type run_dirs: Sequence[str or Path]
    :return: Tuple of trajectory lists and composite replicate identifiers.
    :rtype: tuple[list[list[Atoms]], list[tuple[str, int]]]
    """
    trajectories: list[list[Atoms]] = []
    replicate_ids: list[tuple[str, int]] = []
    for d in run_dirs:
        folder = Path(d)
        paths = find_replicate_trajectories(folder)
        for p in paths:
            trajectories.append(read_trajectory(p))
            replicate_ids.append((str(folder), replicate_index(p)))
    return trajectories, replicate_ids


def apply_freeze_literature(
    candidate_set: CandidateSet,
    system_dir: Path,
    engines: str,
    out_dir: Path,
) -> CandidateSet:
    """
    Load and add literature reactions to a frozen candidate set.

    :param candidate_set: Frozen CandidateSet instance.
    :type candidate_set: CandidateSet
    :param system_dir: Directory containing literature.yaml.
    :type system_dir: Path
    :param engines: Path to engines YAML configuration.
    :type engines: str
    :param out_dir: Destination candidate set directory.
    :type out_dir: Path
    :return: Updated CandidateSet instance.
    :rtype: CandidateSet
    """
    reactions = load_literature_reactions(system_dir / "literature.yaml")
    if not reactions:
        return candidate_set
    cset = add_literature_to_candidates(candidate_set, reactions, engines)
    write_candidate_files(cset, out_dir)
    return cset


def freeze_system_candidates(
    reactant: Atoms,
    settings: MDSettings,
    trajectories: list[list[Atoms]],
    engines: str,
    out_dir: Path,
    replicate_ids: Sequence[int | tuple[str, int]] | None = None,
    add_literature: bool = False,
    system_dir: Path | None = None,
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
    :param replicate_ids: Optional custom replicate identity sequence.
    :type replicate_ids: Sequence[int or tuple[str, int]] or None
    :param add_literature: Whether to add hand-built literature reactions.
    :type add_literature: bool
    :param system_dir: Optional path to system input directory.
    :type system_dir: Path or None
    :return: Frozen candidate set instance.
    :rtype: CandidateSet
    """
    min_frames = math.ceil(settings.min_lifetime_fs / settings.dump_fs)
    cset = freeze(
        reactant=reactant,
        trajectories=trajectories,
        min_frames=min_frames,
        dump_fs=settings.dump_fs,
        engine_settings=engines,
        out_dir=out_dir,
        replicate_ids=replicate_ids,
    )
    if add_literature and system_dir is not None:
        cset = apply_freeze_literature(cset, system_dir, engines, out_dir)
    return cset


def resolve_freeze_trajectories(
    args: argparse.Namespace,
) -> tuple[list[list[Atoms]], Sequence[int | tuple[str, int]] | None]:
    """
    Load trajectories and replicate IDs for freeze subcommand.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Tuple of trajectory lists and replicate identifiers.
    :rtype: tuple[list[list[Atoms]], Sequence[int or tuple[str, int]] or None]
    """
    if getattr(args, "from_runs", None):
        return load_from_runs_trajectories(args.from_runs)
    md_dir = Path(args.runs_dir) / args.system / "md"
    return load_all_trajectories(md_dir), None


def freeze_from_cli_args(args: argparse.Namespace) -> CandidateSet:
    """
    Execute freeze workflow using parsed command-line arguments.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Resulting CandidateSet instance.
    :rtype: CandidateSet
    """
    reactant, settings = load_system_inputs(args.systems_dir, args.system)
    trjs, rep_ids = resolve_freeze_trajectories(args)
    return freeze_system_candidates(
        reactant,
        settings,
        trjs,
        args.engines,
        Path(args.candidates_dir) / args.system,
        rep_ids,
        add_literature=getattr(args, "add_literature", False),
        system_dir=Path(args.systems_dir) / args.system,
    )


def execute_freeze(args: argparse.Namespace) -> int:
    """
    Execute candidate extraction and freezing from discovery trajectories.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Exit code 0 on success.
    :rtype: int
    """
    candidate_set = freeze_from_cli_args(args)
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
        retry=tuple(args.retry) if args.retry else (),
        reactions=args.reactions,
        skip_species=args.skip_species,
    )
    return 0


def print_rank_summary(
    ranks_path: Path,
    system_path: str | Path,
) -> None:
    """
    Print top ranked state and score per temperature and ranker strategy.

    :param ranks_path: Path to serialized ranks JSONL file.
    :type ranks_path: Path
    :param system_path: Path to system YAML or directory containing it.
    :type system_path: str or Path
    :return: None.
    :rtype: None
    """
    top = {
        (float(row["temperature_k"]), row["ranker"]): row
        for row in read_jsonl(ranks_path)
        if row["position"] == 1
    }
    for temp in load_thermo_temperatures(system_path):
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
    sys_yaml = Path(args.systems_dir) / args.system / "system.yaml"
    out_dir = rank(
        system=args.system,
        engine=args.engine,
        candidates_dir=args.candidates_dir,
        results_dir=args.results_dir,
        search_path=args.search,
        system_yaml=sys_yaml,
    )
    print_rank_summary(out_dir / RANKS_FILE, sys_yaml)
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
        systems_dir=args.systems_dir,
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
        systems_dir=args.systems_dir,
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
