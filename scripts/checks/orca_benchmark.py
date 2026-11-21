"""
Time ORCA electronic structure calculations across structures and procs.

Runs benchmark jobs at the reference DFT level in configs/engines.yaml
to determine scaling and memory characteristics for transition state refinement.
Supports single-point energy (SP), gradient (EnGrad), and vibrational frequency
(Freq) tasks across varying CPU process counts.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.io
from ase import Atoms

from nrnet.chem import multiplicity
from nrnet.engines import load_engine_settings, orca_command

__all__ = [
    "BENCHMARK_CSV_FIELDNAMES",
    "CONFIGS_ENGINES_PATH",
    "DEFAULT_MEMORY_BUDGET_MB",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_SCRATCH_DIR",
    "DFT_ENGINE_NAME",
    "MAX_MAXCORE_MB",
    "STATUS_FAILED",
    "STATUS_OK",
    "BenchmarkTarget",
    "add_calculation_arguments",
    "add_config_arguments",
    "add_output_arguments",
    "add_resource_arguments",
    "append_benchmark_row",
    "benchmark_jobs",
    "build_benchmark_row",
    "build_parser",
    "compute_maxcore_mb",
    "dispatch_benchmark_run",
    "execute_cli_benchmarks",
    "execute_orca_run",
    "extract_last_error",
    "format_atom_lines",
    "format_job_summary",
    "format_orca_input",
    "load_benchmark_targets",
    "load_saddle_frame",
    "main",
    "parse_benchmark_metrics",
    "parse_final_energy",
    "parse_imaginary_frequencies",
    "parse_orca_run_seconds",
    "parse_orca_version",
    "parse_scf_cycles",
    "parse_structure_target",
    "resolve_benchmark_options",
    "run_benchmarks",
    "run_single_benchmark",
    "split_structure_spec",
]

DEFAULT_MEMORY_BUDGET_MB: int = 12000
MAX_MAXCORE_MB: int = 2500
DEFAULT_OUTPUT_PATH: Path = Path("data/benchmarks/orca_benchmark.csv")
DEFAULT_SCRATCH_DIR: Path = Path("runs/orca_bench")
CONFIGS_ENGINES_PATH: Path = Path("configs/engines.yaml")
DEFAULT_RESULTS_DIR: Path = Path("results")
DFT_ENGINE_NAME: str = "dft"
STATUS_OK: str = "ok"
STATUS_FAILED: str = "failed"

BENCHMARK_CSV_FIELDNAMES: tuple[str, ...] = (
    "system",
    "engine",
    "reaction_id",
    "atoms",
    "task",
    "nprocs",
    "maxcore_mb",
    "wall_s",
    "orca_run_s",
    "scf_cycles",
    "energy_eh",
    "n_imag",
    "orca_version",
    "status",
    "note",
)


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    """
    Target molecular saddle structure for benchmarking.

    :param system: Chemical system identifier.
    :type system: str
    :param engine: Optimization engine identifier.
    :type engine: str
    :param reaction_id: Reaction identifier string.
    :type reaction_id: str
    :param atoms: Atomic geometry frame of the saddle point.
    :type atoms: Atoms
    """

    system: str
    engine: str
    reaction_id: str
    atoms: Atoms


def compute_maxcore_mb(budget_mb: int, nprocs: int) -> int:
    """
    Calculate maximum memory in megabytes per core for given process count.

    :param budget_mb: Total memory budget in megabytes.
    :type budget_mb: int
    :param nprocs: Number of parallel CPU processes.
    :type nprocs: int
    :return: Memory allocation per process in megabytes.
    :rtype: int
    """
    return min(MAX_MAXCORE_MB, budget_mb // nprocs)


def split_structure_spec(spec: str) -> tuple[str, str, str]:
    """
    Parse a structure specification string into system, engine, and reaction.

    :param spec: Specification in SYSTEM:ENGINE:REACTION_ID format.
    :type spec: str
    :return: Tuple of (system, engine, reaction_id).
    :rtype: tuple[str, str, str]
    :raises ValueError: If specification does not have three colon tokens.
    """
    tokens = spec.split(":")
    if len(tokens) != 3:
        raise ValueError(
            f"Invalid structure {spec!r}; expected SYSTEM:ENGINE:REACTION_ID"
        )
    return tokens[0], tokens[1], tokens[2]


def load_saddle_frame(ts_path: Path, reaction_id: str) -> Atoms:
    """
    Load matching transition state frame from an extxyz trajectory file.

    :param ts_path: Path to the transition states extxyz file.
    :type ts_path: Path
    :param reaction_id: Reaction identifier to match in frame metadata.
    :type reaction_id: str
    :return: Matched atomic configuration frame.
    :rtype: Atoms
    :raises FileNotFoundError: If the extxyz file does not exist.
    :raises KeyError: If reaction_id is not found in the file.
    """
    if not ts_path.is_file():
        raise FileNotFoundError(f"TS trajectory file not found: {ts_path}")
    frames: list[Atoms] = ase.io.read(ts_path, index=":", format="extxyz")
    for frame in frames:
        if str(frame.info.get("reaction_id")) == reaction_id:
            return frame
    raise KeyError(f"Reaction ID {reaction_id!r} not found in {ts_path}")


def parse_structure_target(spec: str, results_dir: Path) -> BenchmarkTarget:
    """
    Resolve a structure specification into a BenchmarkTarget with atoms.

    :param spec: Specification in SYSTEM:ENGINE:REACTION_ID format.
    :type spec: str
    :param results_dir: Root results directory.
    :type results_dir: Path
    :return: BenchmarkTarget holding metadata and atomic structure.
    :rtype: BenchmarkTarget
    """
    system, engine, reaction_id = split_structure_spec(spec)
    ts_path = results_dir / "refine" / system / engine / "ts.extxyz"
    frame = load_saddle_frame(ts_path, reaction_id)
    return BenchmarkTarget(
        system=system,
        engine=engine,
        reaction_id=reaction_id,
        atoms=frame,
    )


def format_atom_lines(atoms: Atoms) -> list[str]:
    """
    Format Cartesian coordinates for ORCA geometry input block.

    :param atoms: Atomic configuration.
    :type atoms: Atoms
    :return: List of formatted atom position strings.
    :rtype: list[str]
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    return [
        f"{sym:<2} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}"
        for sym, pos in zip(symbols, positions, strict=True)
    ]


def format_orca_input(
    model: str,
    task: str,
    scf: str,
    nprocs: int,
    maxcore_mb: int,
    atoms: Atoms,
) -> str:
    """
    Format complete ORCA input file contents for a benchmark run.

    :param model: Density functional and basis set specification.
    :type model: str
    :param task: Calculation task keyword (SP, EnGrad, or Freq).
    :type task: str
    :param scf: SCF convergence keyword (e.g. TightSCF).
    :type scf: str
    :param nprocs: Number of parallel CPU processes.
    :type nprocs: int
    :param maxcore_mb: Memory budget per process in megabytes.
    :type maxcore_mb: int
    :param atoms: Atomic configuration to calculate.
    :type atoms: Atoms
    :return: Formatted ORCA input file text.
    :rtype: str
    """
    mult = multiplicity(atoms)
    header = [
        f"! {model} {task} {scf}",
        f"%pal nprocs {nprocs} end",
        f"%maxcore {maxcore_mb}",
        f"* xyz 0 {mult}",
    ]
    atom_lines = format_atom_lines(atoms)
    return "\n".join(header + atom_lines + ["*\n"])


def extract_last_error(stdout: str, stderr: str) -> str:
    """
    Extract the last error line from standard output and standard error.

    :param stdout: Standard output string from the process.
    :type stdout: str
    :param stderr: Standard error string from the process.
    :type stderr: str
    :return: The last error line found, or empty string.
    :rtype: str
    """
    lines = [
        line.strip()
        for text in (stdout, stderr)
        for line in text.splitlines()
        if line.strip()
    ]
    error_tokens = ("error", "abort", "fatal", "failed")
    error_lines = [
        line
        for line in lines
        if any(token in line.lower() for token in error_tokens)
    ]
    if error_lines:
        return error_lines[-1]
    return lines[-1] if lines else ""


def parse_orca_version(output: str) -> str:
    """
    Parse ORCA software version from standard output header.

    :param output: Standard output text from ORCA.
    :type output: str
    :return: ORCA version string, or empty string if not found.
    :rtype: str
    """
    match = re.search(r"Program Version\s+(\S+)", output)
    return match.group(1) if match else ""


def parse_orca_run_seconds(output: str) -> float | None:
    """
    Parse ORCA total run time in seconds from output text.

    :param output: Standard output text from ORCA.
    :type output: str
    :return: Total run time in seconds, or None if unparsed.
    :rtype: float or None
    """
    pattern = (
        r"TOTAL RUN TIME:\s*(\d+)\s*days\s*(\d+)\s*hours\s*"
        r"(\d+)\s*minutes\s*(\d+)\s*seconds(?:\s*(\d+)\s*msec)?"
    )
    match = re.search(pattern, output)
    if not match:
        return None
    days, hours = int(match.group(1)), int(match.group(2))
    mins, secs = int(match.group(3)), int(match.group(4))
    msec = int(match.group(5) or 0)
    total = days * 86400.0 + hours * 3600.0 + mins * 60.0 + secs + msec / 1e3
    return round(total, 3)


def parse_scf_cycles(output: str) -> int | None:
    """
    Parse the number of SCF cycles of the last SCF from ORCA output text.

    :param output: Standard output text from ORCA.
    :type output: str
    :return: Number of cycles in the last SCF, or None if unparsed.
    :rtype: int or None
    """
    matches = re.findall(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", output)
    if matches:
        return int(matches[-1])
    fallback = re.findall(r"SCF NOT CONVERGED AFTER\s+(\d+)\s+CYCLES", output)
    return int(fallback[-1]) if fallback else None


def parse_final_energy(output: str) -> float | None:
    """
    Parse the final single point energy in Hartree from ORCA output text.

    :param output: Standard output text from ORCA.
    :type output: str
    :return: Final single point energy in Hartree, or None if unparsed.
    :rtype: float or None
    """
    matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", output)
    return float(matches[-1]) if matches else None


def parse_imaginary_frequencies(output: str, task: str) -> int | None:
    """
    Count imaginary frequencies from ORCA output for frequency calculations.

    :param output: Standard output text from ORCA.
    :type output: str
    :param task: Benchmark task name ('SP', 'EnGrad', or 'Freq').
    :type task: str
    :return: Count of imaginary frequencies, or None if task is not Freq.
    :rtype: int or None
    """
    if task != "Freq" or "VIBRATIONAL FREQUENCIES" not in output:
        return None
    return len(re.findall(r"\*\*\*imaginary mode\*\*\*", output))


def execute_orca_run(
    orca_bin: str,
    workdir: Path,
) -> tuple[int, str, str, float]:
    """
    Execute ORCA calculation in scratch directory and measure wall time.

    :param orca_bin: Path to the ORCA executable.
    :type orca_bin: str
    :param workdir: Scratch execution directory containing orca.inp.
    :type workdir: Path
    :return: Tuple of return code, stdout, stderr, and wall time in seconds.
    :rtype: tuple[int, str, str, float]
    """
    start_time = time.perf_counter()
    proc = subprocess.run(
        [orca_bin, "orca.inp"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_s = round(time.perf_counter() - start_time, 3)
    return proc.returncode, proc.stdout, proc.stderr, wall_s


def parse_benchmark_metrics(
    stdout: str,
    stderr: str,
    task: str,
    returncode: int,
) -> dict[str, Any]:
    """
    Extract calculation outcome and quantum metrics from output text.

    :param stdout: Captured standard output from ORCA.
    :type stdout: str
    :param stderr: Captured standard error from ORCA.
    :type stderr: str
    :param task: Benchmark task name.
    :type task: str
    :param returncode: Process exit code.
    :type returncode: int
    :return: Dictionary of parsed metrics including status and note.
    :rtype: dict[str, Any]
    """
    ok = returncode == 0 and "****ORCA TERMINATED NORMALLY****" in stdout
    run_s = parse_orca_run_seconds(stdout)
    cycles = parse_scf_cycles(stdout)
    energy = parse_final_energy(stdout)
    n_imag = parse_imaginary_frequencies(stdout, task)
    return {
        "orca_run_s": "" if run_s is None else run_s,
        "scf_cycles": "" if cycles is None else cycles,
        "energy_eh": "" if energy is None else energy,
        "n_imag": "" if n_imag is None else n_imag,
        "orca_version": parse_orca_version(stdout),
        "status": STATUS_OK if ok else STATUS_FAILED,
        "note": "" if ok else extract_last_error(stdout, stderr),
    }


def build_benchmark_row(
    target: BenchmarkTarget,
    task: str,
    nprocs: int,
    maxcore_mb: int,
    wall_s: float,
    stdout: str,
    stderr: str,
    returncode: int,
) -> dict[str, Any]:
    """
    Assemble structured benchmark result dictionary from run metrics.

    :param target: Benchmark target holding structure and metadata.
    :type target: BenchmarkTarget
    :param task: Calculation task name.
    :type task: str
    :param nprocs: Process count.
    :type nprocs: int
    :param maxcore_mb: Memory per process in megabytes.
    :type maxcore_mb: int
    :param wall_s: Wall clock elapsed time in seconds.
    :type wall_s: float
    :param stdout: Captured standard output from ORCA.
    :type stdout: str
    :param stderr: Captured standard error from ORCA.
    :type stderr: str
    :param returncode: Process exit code.
    :type returncode: int
    :return: Dictionary row ready for CSV serialization.
    :rtype: dict[str, Any]
    """
    row: dict[str, Any] = {
        "system": target.system,
        "engine": target.engine,
        "reaction_id": target.reaction_id,
        "atoms": len(target.atoms),
        "task": task,
        "nprocs": nprocs,
        "maxcore_mb": maxcore_mb,
        "wall_s": wall_s,
    }
    metrics = parse_benchmark_metrics(stdout, stderr, task, returncode)
    row.update(metrics)
    return row


def append_benchmark_row(path: Path, row: Mapping[str, Any]) -> None:
    """
    Append one benchmark result row to CSV, writing headers if newly created.

    :param path: Path to the CSV output file.
    :type path: Path
    :param row: Row data mapping to write.
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


def format_job_summary(row: Mapping[str, Any]) -> str:
    """
    Format a concise single-line description of a finished job.

    :param row: Row mapping containing benchmark metrics.
    :type row: Mapping[str, Any]
    :return: Formatted summary string.
    :rtype: str
    """
    prefix = (
        f"[{row['status']}] {row['system']}:{row['engine']}:"
        f"{row['reaction_id']} {row['task']} np={row['nprocs']} "
        f"maxcore={row['maxcore_mb']}MB wall={row['wall_s']}s"
    )
    if row["status"] == STATUS_OK:
        return (
            f"{prefix} orca={row['orca_run_s']}s scf={row['scf_cycles']} "
            f"E={row['energy_eh']}"
        )
    return f"{prefix} note={row['note']!r}"


def run_single_benchmark(
    target: BenchmarkTarget,
    task: str,
    nprocs: int,
    maxcore_mb: int,
    inp_text: str,
    orca_bin: str,
    scratch_dir: Path,
    out_csv: Path,
) -> dict[str, Any]:
    """
    Run one benchmark combination in its scratch directory and record row.

    :param target: Benchmark target configuration.
    :type target: BenchmarkTarget
    :param task: Calculation task keyword.
    :type task: str
    :param nprocs: Number of parallel CPU processes.
    :type nprocs: int
    :param maxcore_mb: Memory per process in megabytes.
    :type maxcore_mb: int
    :param inp_text: Full formatted ORCA input file content.
    :type inp_text: str
    :param orca_bin: Absolute path to the ORCA executable.
    :type orca_bin: str
    :param scratch_dir: Base scratch directory for calculation.
    :type scratch_dir: Path
    :param out_csv: Path to output CSV file.
    :type out_csv: Path
    :return: Computed benchmark result row.
    :rtype: dict[str, Any]
    """
    job_name = (
        f"{target.system}_{target.engine}_{target.reaction_id}_"
        f"{task}_np{nprocs}"
    )
    workdir = scratch_dir / job_name
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "orca.inp").write_text(inp_text, encoding="utf-8")
    ret, stdout, stderr, wall_s = execute_orca_run(orca_bin, workdir)
    row = build_benchmark_row(
        target, task, nprocs, maxcore_mb, wall_s, stdout, stderr, ret
    )
    append_benchmark_row(out_csv, row)
    print(format_job_summary(row))
    return row


def benchmark_jobs(
    targets: Sequence[BenchmarkTarget],
    tasks: Sequence[str],
    nprocs_list: Sequence[int],
) -> list[tuple[BenchmarkTarget, str, int]]:
    """
    Generate combinations of benchmark targets, tasks, and process counts.

    :param targets: Sequence of benchmark target configurations.
    :type targets: Sequence[BenchmarkTarget]
    :param tasks: Sequence of task keywords (SP, EnGrad, Freq).
    :type tasks: Sequence[str]
    :param nprocs_list: Sequence of parallel process counts.
    :type nprocs_list: Sequence[int]
    :return: List of (target, task, nprocs) tuples.
    :rtype: list[tuple[BenchmarkTarget, str, int]]
    """
    return [
        (target, task, nprocs)
        for target in targets
        for task in tasks
        for nprocs in nprocs_list
    ]


def dispatch_benchmark_run(
    target: BenchmarkTarget,
    task: str,
    nprocs: int,
    budget_mb: int,
    model: str,
    scf: str,
    scratch_dir: Path,
    out_csv: Path,
    orca_bin: str,
    dry_run: bool,
) -> None:
    """
    Prepare input and execute or dry-run a single benchmark combination.

    :param target: Benchmark target configuration.
    :type target: BenchmarkTarget
    :param task: Calculation task keyword.
    :type task: str
    :param nprocs: Number of parallel CPU processes.
    :type nprocs: int
    :param budget_mb: Total memory budget in megabytes.
    :type budget_mb: int
    :param model: Density functional and basis set specification.
    :type model: str
    :param scf: SCF convergence criteria keyword.
    :type scf: str
    :param scratch_dir: Base scratch directory for calculation.
    :type scratch_dir: Path
    :param out_csv: Path to output CSV file.
    :type out_csv: Path
    :param orca_bin: Path to the ORCA executable.
    :type orca_bin: str
    :param dry_run: Whether to print inputs only without executing.
    :type dry_run: bool
    :return: None.
    :rtype: None
    """
    maxcore_mb = compute_maxcore_mb(budget_mb, nprocs)
    inp = format_orca_input(model, task, scf, nprocs, maxcore_mb, target.atoms)
    if dry_run:
        print(inp.strip())
        return
    run_single_benchmark(
        target,
        task,
        nprocs,
        maxcore_mb,
        inp,
        orca_bin,
        scratch_dir,
        out_csv,
    )


def run_benchmarks(
    targets: Sequence[BenchmarkTarget],
    tasks: Sequence[str],
    nprocs_list: Sequence[int],
    budget_mb: int,
    model: str,
    scf: str,
    scratch_dir: Path,
    out_csv: Path,
    dry_run: bool,
) -> None:
    """
    Execute or dry-run benchmark combinations across structures and tasks.

    :param targets: Sequence of benchmark target configurations.
    :type targets: Sequence[BenchmarkTarget]
    :param tasks: Sequence of task keywords (SP, EnGrad, Freq).
    :type tasks: Sequence[str]
    :param nprocs_list: Sequence of parallel process counts.
    :type nprocs_list: Sequence[int]
    :param budget_mb: Total memory budget in megabytes.
    :type budget_mb: int
    :param model: Density functional and basis set specification.
    :type model: str
    :param scf: SCF convergence criteria keyword.
    :type scf: str
    :param scratch_dir: Base scratch directory path.
    :type scratch_dir: Path
    :param out_csv: Destination CSV path.
    :type out_csv: Path
    :param dry_run: Whether to print inputs only without executing.
    :type dry_run: bool
    :return: None.
    :rtype: None
    """
    orca_bin = "" if dry_run else orca_command()
    for target, task, nprocs in benchmark_jobs(targets, tasks, nprocs_list):
        dispatch_benchmark_run(
            target,
            task,
            nprocs,
            budget_mb,
            model,
            scf,
            scratch_dir,
            out_csv,
            orca_bin,
            dry_run,
        )


def add_calculation_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register calculation target and task arguments on the CLI parser.

    :param parser: CLI argument parser to configure.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    parser.add_argument(
        "--structure",
        action="append",
        required=True,
        help="Saddle structure in SYSTEM:ENGINE:REACTION_ID format.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=["SP", "EnGrad", "Freq"],
        help="Task to benchmark (SP, EnGrad, or Freq).",
    )


def add_resource_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register resource allocation arguments on the CLI parser.

    :param parser: CLI argument parser to configure.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    parser.add_argument(
        "--nprocs",
        type=int,
        action="append",
        help="Number of parallel processes to benchmark.",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=int,
        default=DEFAULT_MEMORY_BUDGET_MB,
        help="Total memory budget in MB across all processes.",
    )


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register filesystem output and scratch path arguments on the CLI parser.

    :param parser: CLI argument parser to configure.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output CSV benchmark file.",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=DEFAULT_SCRATCH_DIR,
        help="Base scratch directory for job execution.",
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register engine and results path arguments on the CLI parser.

    :param parser: CLI argument parser to configure.
    :type parser: argparse.ArgumentParser
    :return: None.
    :rtype: None
    """
    parser.add_argument(
        "--engines-path",
        type=Path,
        default=CONFIGS_ENGINES_PATH,
        help="Path to engines YAML configuration file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory holding refined reaction outputs.",
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Construct command line argument parser for ORCA benchmarking.

    :return: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_calculation_arguments(parser)
    add_resource_arguments(parser)
    add_output_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated ORCA inputs only without running.",
    )
    return parser


def load_benchmark_targets(
    args: argparse.Namespace,
) -> list[BenchmarkTarget]:
    """
    Resolve and load benchmark target structures from parsed CLI arguments.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: List of loaded benchmark target configurations.
    :rtype: list[BenchmarkTarget]
    """
    return [
        parse_structure_target(spec, args.results_dir)
        for spec in args.structure
    ]


def resolve_benchmark_options(
    args: argparse.Namespace,
) -> tuple[Any, Sequence[str], Sequence[int]]:
    """
    Resolve engine settings, task keywords, and process counts from arguments.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Tuple of engine settings, task keywords, and process counts.
    :rtype: tuple[Any, Sequence[str], Sequence[int]]
    """
    settings = load_engine_settings(args.engines_path, DFT_ENGINE_NAME)
    tasks = args.task or ["SP"]
    nprocs_list = args.nprocs or [int(settings.extra["nprocs"])]
    return settings, tasks, nprocs_list


def execute_cli_benchmarks(
    args: argparse.Namespace,
    targets: Sequence[BenchmarkTarget],
) -> None:
    """
    Execute benchmark runs using loaded targets and parsed CLI arguments.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :param targets: Loaded benchmark target molecular structures.
    :type targets: Sequence[BenchmarkTarget]
    :return: None.
    :rtype: None
    """
    settings, tasks, nprocs_list = resolve_benchmark_options(args)
    run_benchmarks(
        targets=targets,
        tasks=tasks,
        nprocs_list=nprocs_list,
        budget_mb=args.memory_budget_mb,
        model=settings.model,
        scf=str(settings.extra["scf"]),
        scratch_dir=args.scratch,
        out_csv=args.out,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entrypoint for ORCA benchmarking script.

    :param argv: Optional command-line argument list.
    :type argv: Sequence[str] or None
    :return: Process exit status code.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    targets = load_benchmark_targets(args)
    execute_cli_benchmarks(args, targets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
