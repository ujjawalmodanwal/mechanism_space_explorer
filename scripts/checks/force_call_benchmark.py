"""
Benchmark single force call execution time for computational engines.

Times repeated energy and gradient evaluations with coordinate jitter under
a uniform protocol to evaluate single-call latency and throughput across
molecular structures and MLIP/quantum engines. Appends structured benchmark
metrics to CSV.
"""

import argparse
import csv
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
from ase import Atoms
from orca_benchmark import load_saddle_frame, split_structure_spec

from nrnet.engines import EngineSettings, attach, load_engine_settings

__all__ = [
    "BENCHMARK_CSV_FIELDNAMES",
    "CONFIGS_ENGINES_PATH",
    "DEFAULT_CALLS",
    "DEFAULT_JITTER_ANGSTROM",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SEED",
    "DEFAULT_WARMUP",
    "PATH_ARGUMENTS",
    "PROTOCOL_ARGUMENTS",
    "STRUCTURE_ARGUMENTS",
    "add_paths_arguments",
    "add_protocol_arguments",
    "add_structure_arguments",
    "append_benchmark_row",
    "attach_arguments",
    "build_benchmark_row",
    "build_parser",
    "configure_threads",
    "format_benchmark_line",
    "main",
    "measure_force_calls",
    "record_benchmark_row",
    "resolve_structure",
    "run_benchmarks",
    "run_single_benchmark",
]

DEFAULT_WARMUP: int = 3
DEFAULT_CALLS: int = 30
DEFAULT_JITTER_ANGSTROM: float = 0.01
DEFAULT_SEED: int = 0
CONFIGS_ENGINES_PATH: Path = Path("configs/engines.yaml")
DEFAULT_RESULTS_DIR: Path = Path("results")
DEFAULT_OUTPUT_PATH: Path = Path("data/benchmarks/force_call_benchmark.csv")

BENCHMARK_CSV_FIELDNAMES: tuple[str, ...] = (
    "timestamp_utc",
    "engine",
    "model",
    "structure",
    "n_atoms",
    "threads",
    "warmup",
    "calls",
    "jitter_angstrom",
    "median_ms",
    "mean_ms",
    "min_ms",
    "max_ms",
)

STRUCTURE_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--engine",
        {"required": True, "help": "Engine name in configuration."},
    ),
    (
        "--structure",
        {
            "action": "append",
            "required": True,
            "help": "Structure path or SYSTEM:ENGINE:REACTION_ID spec.",
        },
    ),
)

PROTOCOL_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--warmup",
        {
            "type": int,
            "default": DEFAULT_WARMUP,
            "help": "Warmup force call count.",
        },
    ),
    (
        "--calls",
        {
            "type": int,
            "default": DEFAULT_CALLS,
            "help": "Timed force call count.",
        },
    ),
    (
        "--jitter",
        {
            "type": float,
            "default": DEFAULT_JITTER_ANGSTROM,
            "help": "Coordinate jitter standard deviation in Angstrom.",
        },
    ),
    (
        "--seed",
        {
            "type": int,
            "default": DEFAULT_SEED,
            "help": "RNG seed for jitter.",
        },
    ),
)

PATH_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "--engines-path",
        {
            "type": Path,
            "default": CONFIGS_ENGINES_PATH,
            "help": "Path to engines YAML file.",
        },
    ),
    (
        "--results-dir",
        {
            "type": Path,
            "default": DEFAULT_RESULTS_DIR,
            "help": "Root directory holding refined reaction outputs.",
        },
    ),
    (
        "--output",
        {
            "type": Path,
            "default": DEFAULT_OUTPUT_PATH,
            "help": "Output CSV benchmark path.",
        },
    ),
)


def configure_threads() -> int:
    """
    Read OMP_NUM_THREADS and configure Torch thread count if available.

    :return: Thread count used for calculations.
    :rtype: int
    """
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    return threads


def resolve_structure(spec: str, results_dir: Path) -> Atoms:
    """
    Load structure from file or SYSTEM:ENGINE:REACTION_ID spec.

    :param spec: Path to geometry file or SYSTEM:ENGINE:REACTION_ID spec.
    :type spec: str
    :param results_dir: Root results directory holding refined saddles.
    :type results_dir: Path
    :return: Loaded atomic structure.
    :rtype: Atoms
    """
    if Path(spec).is_file():
        return ase.io.read(spec)
    system, engine, reaction_id = split_structure_spec(spec)
    ts_path = results_dir / "refine" / system / engine / "ts.extxyz"
    return load_saddle_frame(ts_path, reaction_id)


def measure_force_calls(
    atoms: Atoms,
    warmup: int,
    calls: int,
    jitter: float,
    seed: int,
) -> list[float]:
    """
    Measure force call execution times in milliseconds.

    :param atoms: Atoms configuration with calculator attached.
    :type atoms: Atoms
    :param warmup: Number of warmup rounds before timing collection.
    :type warmup: int
    :param calls: Number of timed evaluation rounds.
    :type calls: int
    :param jitter: Coordinate jitter magnitude in Angstrom.
    :type jitter: float
    :param seed: Random seed for jitter generation.
    :type seed: int
    :return: Elapsed execution times in milliseconds.
    :rtype: list[float]
    """
    rng = np.random.default_rng(seed)
    base = atoms.get_positions()
    timings: list[float] = []
    for round_idx in range(warmup + calls):
        noise = rng.normal(0.0, jitter, base.shape)
        atoms.set_positions(base + noise)
        start = time.perf_counter()
        atoms.get_forces()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        if round_idx >= warmup:
            timings.append(elapsed_ms)
    return timings


def build_benchmark_row(
    settings: EngineSettings,
    spec: str,
    n_atoms: int,
    threads: int,
    warmup: int,
    calls: int,
    jitter: float,
    timings: Sequence[float],
) -> dict[str, Any]:
    """
    Assemble benchmark result row dictionary for CSV export.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param spec: Raw structure specification string.
    :type spec: str
    :param n_atoms: Total number of atoms in structure.
    :type n_atoms: int
    :param threads: Active thread budget count.
    :type threads: int
    :param warmup: Warmup iterations count.
    :type warmup: int
    :param calls: Timed iterations count.
    :type calls: int
    :param jitter: Coordinate jitter magnitude in Angstrom.
    :type jitter: float
    :param timings: Sequence of measured durations in milliseconds.
    :type timings: Sequence[float]
    :return: Dictionary row ready for CSV serialization.
    :rtype: dict[str, Any]
    """
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "engine": settings.name,
        "model": settings.model,
        "structure": spec,
        "n_atoms": n_atoms,
        "threads": threads,
        "warmup": warmup,
        "calls": calls,
        "jitter_angstrom": jitter,
        "median_ms": round(float(np.median(timings)), 3),
        "mean_ms": round(float(np.mean(timings)), 3),
        "min_ms": round(float(min(timings)), 3),
        "max_ms": round(float(max(timings)), 3),
    }


def append_benchmark_row(path: Path, row: Mapping[str, Any]) -> None:
    """
    Append one benchmark row to CSV file, writing header if file is new.

    :param path: Destination CSV output path.
    :type path: Path
    :param row: Benchmark metric row mapping.
    :type row: Mapping[str, Any]
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BENCHMARK_CSV_FIELDNAMES)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def format_benchmark_line(row: Mapping[str, Any]) -> str:
    """
    Format concise single-line structure benchmark summary for stdout.

    :param row: Benchmark metric row mapping.
    :type row: Mapping[str, Any]
    :return: Formatted output string.
    :rtype: str
    """
    return (
        f"{row['engine']} {row['structure']} {row['n_atoms']} "
        f"{row['median_ms']:.2f} ms"
    )


def attach_arguments(
    parser: argparse.ArgumentParser,
    specs: Sequence[tuple[str, dict[str, Any]]],
) -> None:
    """
    Attach CLI flag specifications to an argument parser.

    :param parser: CLI parser to configure.
    :type parser: argparse.ArgumentParser
    :param specs: Sequence of flag name and keyword arguments tuples.
    :type specs: Sequence[tuple[str, dict[str, Any]]]
    :return: None.
    :rtype: None
    """
    for flag, kwargs in specs:
        parser.add_argument(flag, **kwargs)


def add_structure_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register engine and structure target arguments on CLI parser.

    :param parser: CLI parser to configure.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    attach_arguments(parser, STRUCTURE_ARGUMENTS)


def add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register benchmark timing protocol arguments on CLI parser.

    :param parser: CLI parser to configure.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    attach_arguments(parser, PROTOCOL_ARGUMENTS)


def add_paths_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register engine, results, and output path arguments on CLI parser.

    :param parser: CLI parser to configure.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    attach_arguments(parser, PATH_ARGUMENTS)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct CLI parser for force call benchmarking.

    :return: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_structure_arguments(parser)
    add_protocol_arguments(parser)
    add_paths_arguments(parser)
    return parser


def record_benchmark_row(
    path: Path,
    settings: EngineSettings,
    spec: str,
    n_atoms: int,
    threads: int,
    args: argparse.Namespace,
    timings: Sequence[float],
) -> dict[str, Any]:
    """
    Build, append, and print a benchmark result row.

    :param path: Destination CSV output path.
    :type path: Path
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param spec: Raw structure specification string.
    :type spec: str
    :param n_atoms: Total number of atoms in structure.
    :type n_atoms: int
    :param threads: Active thread budget count.
    :type threads: int
    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :param timings: Sequence of measured durations in milliseconds.
    :type timings: Sequence[float]
    :return: Dictionary row ready for CSV serialization.
    :rtype: dict[str, Any]
    """
    row = build_benchmark_row(
        settings,
        spec,
        n_atoms,
        threads,
        args.warmup,
        args.calls,
        args.jitter,
        timings,
    )
    append_benchmark_row(path, row)
    print(format_benchmark_line(row), flush=True)
    return row


def run_single_benchmark(
    spec: str,
    settings: EngineSettings,
    threads: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Execute force call benchmark and record metrics for one structure.

    :param spec: Structure path or SYSTEM:ENGINE:REACTION_ID spec.
    :type spec: str
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param threads: Active thread budget count.
    :type threads: int
    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Benchmark result row dictionary.
    :rtype: dict[str, Any]
    """
    atoms = resolve_structure(spec, args.results_dir)
    calc_atoms = attach(atoms, settings)
    timings = measure_force_calls(
        calc_atoms, args.warmup, args.calls, args.jitter, args.seed
    )
    return record_benchmark_row(
        args.output, settings, spec, len(atoms), threads, args, timings
    )


def run_benchmarks(args: argparse.Namespace) -> None:
    """
    Execute force call benchmarks across requested structures.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: None.
    :rtype: None
    """
    threads = configure_threads()
    settings = load_engine_settings(args.engines_path, args.engine)
    for spec in args.structure:
        run_single_benchmark(spec, settings, threads, args)


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint for force call benchmarking.

    :param argv: Optional command-line argument strings.
    :type argv: Sequence[str] or None
    :return: Process exit status code.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    run_benchmarks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
