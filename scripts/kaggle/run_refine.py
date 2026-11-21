"""
Execute nanoreactor refinement tasks across multiple systems on Kaggle GPUs.

Handles bundle staging, parallel multi-GPU worker execution, deadline safety,
and packaging of refinement outputs using only standard Python libraries.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "RefineJob",
    "WorkerPool",
    "add_runner_engine_arguments",
    "add_runner_location_arguments",
    "add_runner_mode_arguments",
    "add_runner_recovery_arguments",
    "aggregate_system_outcomes",
    "archive_directory_tree",
    "archive_engine_results",
    "build_parser",
    "build_refine_command",
    "build_refine_jobs",
    "build_system_jobs",
    "build_warmup_code",
    "build_warmup_command",
    "build_worker_environment",
    "copy_bundle",
    "count_reactions",
    "create_part_workspace",
    "drop_csv_duplicate_header",
    "execute_optional_refinement",
    "extract_resume_members",
    "finalize_and_package_results",
    "find_bundle_candidate_subfolders",
    "format_command_env_prefix",
    "format_job_command",
    "format_single_system_summary",
    "format_system_summary_line",
    "format_warmup_command",
    "format_worker_command",
    "get_failed_systems",
    "ignore_workspace_parts_and_logs",
    "join_all_parts",
    "join_file_tail",
    "join_single_part",
    "launch_all_refine_workers",
    "launch_refine_worker",
    "load_part_offsets",
    "load_split_plan",
    "main",
    "package_results",
    "poll_completed_workers",
    "prepare_part_workspaces",
    "print_dry_run_commands",
    "print_execution_status",
    "print_process_tail",
    "print_systems_summary",
    "record_file_offsets",
    "record_system_completion",
    "resolve_bundle_dir",
    "restore_resume_zip",
    "run_parallel_refine",
    "run_refinement_workflow",
    "send_signal_to_process_group",
    "setup_workspace_and_resume",
    "supervise_parallel_workers",
    "terminate_remaining_workers",
    "terminate_worker_on_deadline",
    "trim_file_if_incomplete",
    "trim_refine_directory",
    "trim_refine_files",
    "warmup_engine_cache",
]


@dataclass(frozen=True, slots=True)
class RefineJob:
    """
    Specification for a refinement worker execution job.
    """

    job_id: str
    system: str
    workdir: Path
    reaction_ids: tuple[str, ...] | None = None
    skip_species: bool = False


@dataclass(frozen=True, slots=True)
class WorkerPool:
    """
    State container for active worker processes and execution tracking.
    """

    processes: dict[str, subprocess.Popen[bytes]]
    log_handles: dict[str, TextIO]
    start_times: dict[str, float]
    exit_codes: dict[str, int]
    wall_times: dict[str, float]


def format_command_env_prefix(
    work_dir: Path,
    gpu_id: int,
) -> str:
    """
    Format shell environment variables prefix for command string.

    :param work_dir: Working directory path.
    :type work_dir: Path
    :param gpu_id: GPU device ordinal.
    :type gpu_id: int
    :return: Environment prefix string.
    :rtype: str
    """
    return (
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"OMP_NUM_THREADS=1 "
        f"MKL_NUM_THREADS=1 "
        f"PYTHONPATH={work_dir}/src"
    )


def build_warmup_code(engine: str) -> str:
    """
    Construct Python code string to warm up engine model cache.

    :param engine: Target computational chemistry engine.
    :type engine: str
    :return: Python one-liner command string.
    :rtype: str
    """
    return (
        "import ase.build; "
        "from nrnet.engines import attach, load_engine_settings; "
        f"settings = load_engine_settings('configs/engines.yaml', '{engine}'); "
        "attach(ase.build.molecule('H2O'), settings).get_potential_energy()"
    )


def format_warmup_command(
    work_dir: Path,
    engine: str,
) -> str:
    """
    Format shell command string for dry-run cache warm-up.

    :param work_dir: Working directory path.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :return: Formatted shell command line string.
    :rtype: str
    """
    env_str = format_command_env_prefix(work_dir, 0)
    return f'{env_str} python -c "{build_warmup_code(engine)}"'


def format_worker_command(
    work_dir: Path,
    engine: str,
    system: str,
    gpu_id: int,
    reaction_ids: Sequence[str] | None = None,
    skip_species: bool = False,
) -> str:
    """
    Format shell command string for single worker in dry-run mode.

    :param work_dir: Working directory path.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param system: System identifier string.
    :type system: str
    :param gpu_id: GPU ordinal integer.
    :type gpu_id: int
    :param reaction_ids: Optional reaction identifiers to refine.
    :type reaction_ids: Sequence[str] | None
    :param skip_species: Whether to skip refining frozen species minimums.
    :type skip_species: bool
    :return: Formatted shell command line string.
    :rtype: str
    """
    env_str = format_command_env_prefix(work_dir, gpu_id)
    cmd = (
        f"{env_str} "
        'python -c "from nrnet.cli import main; main()" '
        f"refine --system {system} --engine {engine} --seed-from xtb"
    )
    if reaction_ids:
        cmd += f" --reactions {' '.join(reaction_ids)}"
    if skip_species:
        cmd += " --skip-species"
    return cmd


def format_job_command(
    job: RefineJob,
    engine: str,
    gpu_id: int,
) -> str:
    """
    Format shell command string for single refinement job in dry-run mode.

    :param job: Refinement job configuration.
    :type job: RefineJob
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param gpu_id: GPU device ordinal.
    :type gpu_id: int
    :return: Formatted shell command line string.
    :rtype: str
    """
    return format_worker_command(
        job.workdir,
        engine,
        job.system,
        gpu_id,
        job.reaction_ids,
        job.skip_species,
    )


def print_dry_run_commands(
    work_dir: Path,
    engine: str,
    jobs: Sequence[RefineJob],
    gpus: int,
) -> None:
    """
    Print executable refinement commands and workdirs for dry-run inspection.

    :param work_dir: Working directory where refinement runs.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param jobs: Sequence of refinement jobs to execute.
    :type jobs: Sequence[RefineJob]
    :param gpus: Number of available GPUs.
    :type gpus: int
    """
    print(format_warmup_command(work_dir, engine))
    for idx, job in enumerate(jobs):
        print(f"workdir: {job.workdir}")
        print(format_job_command(job, engine, idx % gpus))


def find_bundle_candidate_subfolders(bundle_dir: Path) -> list[Path]:
    """
    Search directory for subfolders containing both src/ and BUNDLE.json.

    :param bundle_dir: Directory path to scan for subfolders.
    :type bundle_dir: Path
    :return: List of valid candidate subfolder paths.
    :rtype: list[Path]
    """
    if not bundle_dir.is_dir():
        return []
    return [
        sub
        for sub in sorted(bundle_dir.iterdir())
        if sub.is_dir()
        and (sub / "src").is_dir()
        and (sub / "BUNDLE.json").is_file()
    ]


def resolve_bundle_dir(bundle_dir: Path) -> Path:
    """
    Resolve bundle root directly or locate its single qualifying subfolder.

    :param bundle_dir: Base directory path to resolve.
    :type bundle_dir: Path
    :return: Resolved bundle directory path containing src/ and BUNDLE.json.
    :rtype: Path
    :raises SystemExit: If directory cannot be resolved to a single bundle.
    """
    if not bundle_dir.exists():
        sys.exit(f"Bundle directory does not exist: {bundle_dir}")
    if (bundle_dir / "src").is_dir():
        return bundle_dir
    candidates = find_bundle_candidate_subfolders(bundle_dir)
    if len(candidates) == 1:
        return candidates[0]
    found = [p.name for p in sorted(bundle_dir.iterdir())]
    sys.exit(
        f"Cannot resolve bundle in '{bundle_dir}': expected 'src/' or a "
        f"single subfolder with src/ and BUNDLE.json, but found: {found}"
    )


def copy_bundle(
    bundle_dir: Path,
    work_dir: Path,
) -> None:
    """
    Copy bundle directory contents into working directory.

    :param bundle_dir: Directory containing unpacked bundle files.
    :type bundle_dir: Path
    :param work_dir: Target working directory.
    :type work_dir: Path
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    if bundle_dir.resolve() == work_dir.resolve():
        return
    for item in bundle_dir.iterdir():
        dest = work_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def extract_resume_members(
    archive: zipfile.ZipFile,
    work_dir: Path,
    engine: str,
    systems: Sequence[str],
) -> None:
    """
    Extract matching results files from resume zip archive into work directory.

    :param archive: Open zip archive instance.
    :type archive: zipfile.ZipFile
    :param work_dir: Target working directory.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers to restore.
    :type systems: Sequence[str]
    """
    for name in archive.namelist():
        for system in systems:
            prefix = f"results/refine/{system}/{engine}/"
            if name.startswith(prefix) and not name.endswith("/"):
                target = work_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def restore_resume_zip(
    resume_zip: Path,
    work_dir: Path,
    engine: str,
    systems: Sequence[str],
) -> None:
    """
    Restore previous refinement results from zip archive.

    :param resume_zip: Path to previous results zip archive.
    :type resume_zip: Path
    :param work_dir: Target working directory.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers to restore.
    :type systems: Sequence[str]
    :raises SystemExit: If resume_zip file does not exist.
    """
    if not resume_zip.exists():
        sys.exit(f"Resume zip does not exist: {resume_zip}")
    with zipfile.ZipFile(resume_zip, "r") as archive:
        extract_resume_members(archive, work_dir, engine, systems)


def trim_file_if_incomplete(path: Path) -> bool:
    """
    Drop trailing line if file does not end in a newline character.

    :param path: Path to text file to inspect and potentially trim.
    :type path: Path
    :return: True if incomplete line was removed, False otherwise.
    :rtype: bool
    """
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    last_nl = data.rfind(b"\n")
    path.write_bytes(data[: last_nl + 1] if last_nl != -1 else b"")
    print(f"Trimmed {path}")
    return True


def trim_refine_directory(target_dir: Path) -> list[Path]:
    """
    Trim incomplete trailing lines from files in an engine directory.

    :param target_dir: Directory containing results for an engine.
    :type target_dir: Path
    :return: List of all file paths trimmed.
    :rtype: list[Path]
    """
    trimmed: list[Path] = []
    for pattern in ("*.jsonl", "calls.csv"):
        for path in sorted(target_dir.glob(pattern)):
            if path.is_file() and trim_file_if_incomplete(path):
                trimmed.append(path)
    return trimmed


def trim_refine_files(
    work_dir: Path,
    engine: str,
) -> list[Path]:
    """
    Trim incomplete trailing lines from results under work_dir.

    :param work_dir: Working directory containing refinement results or
        target engine directory.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :return: List of all file paths trimmed across systems.
    :rtype: list[Path]
    """
    if work_dir.is_dir() and work_dir.name == engine:
        return trim_refine_directory(work_dir)
    trimmed: list[Path] = []
    refine_base = work_dir / "results" / "refine"
    if not refine_base.exists():
        return trimmed
    for system_dir in sorted(refine_base.iterdir()):
        target_dir = system_dir / engine
        if target_dir.is_dir():
            trimmed.extend(trim_refine_directory(target_dir))
    return trimmed


def build_worker_environment(
    work_dir: Path,
    gpu_id: int,
) -> dict[str, str]:
    """
    Build environment variables dictionary for refinement worker subprocess.

    :param work_dir: Working directory containing source tree.
    :type work_dir: Path
    :param gpu_id: Target GPU device ordinal string.
    :type gpu_id: int
    :return: Environment dictionary for worker subprocess.
    :rtype: dict[str, str]
    """
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(work_dir / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
        }
    )
    return env


def build_refine_command(
    system: str,
    engine: str,
    reaction_ids: Sequence[str] | None = None,
    skip_species: bool = False,
) -> list[str]:
    """
    Construct command-line arguments list for nrnet refine worker execution.

    :param system: Target chemical system identifier.
    :type system: str
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param reaction_ids: Optional reaction identifiers to refine.
    :type reaction_ids: Sequence[str] | None
    :param skip_species: Whether to skip refining frozen species minimums.
    :type skip_species: bool
    :return: Command tokens list for subprocess invocation.
    :rtype: list[str]
    """
    cmd = [
        sys.executable,
        "-c",
        "from nrnet.cli import main; main()",
        "refine",
        "--system",
        system,
        "--engine",
        engine,
        "--seed-from",
        "xtb",
    ]
    if reaction_ids:
        cmd.extend(["--reactions", *reaction_ids])
    if skip_species:
        cmd.append("--skip-species")
    return cmd


def build_warmup_command(engine: str) -> list[str]:
    """
    Construct command token list for engine cache warm-up.

    :param engine: Target computational chemistry engine.
    :type engine: str
    :return: Command argument tokens.
    :rtype: list[str]
    """
    return [sys.executable, "-c", build_warmup_code(engine)]


def print_process_tail(
    output: str,
    max_lines: int = 20,
) -> None:
    """
    Print the trailing lines of subprocess output.

    :param output: Combined standard output and error text.
    :type output: str
    :param max_lines: Maximum number of lines from the tail to print.
    :type max_lines: int
    """
    lines = output.strip().splitlines()
    if lines:
        print("\n".join(lines[-max_lines:]))
    else:
        print("Warm-up failed with no output recorded.")


def warmup_engine_cache(
    work_dir: Path,
    engine: str,
) -> None:
    """
    Warm up engine cache before launching parallel workers.

    :param work_dir: Working directory containing unpacked bundle.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :raises SystemExit: If warm-up execution fails.
    """
    cmd = build_warmup_command(engine)
    env = build_worker_environment(work_dir, 0)
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        print_process_tail(proc.stdout)
        sys.exit(proc.returncode or 1)


def launch_refine_worker(
    job: RefineJob,
    engine: str,
    gpu_id: int,
    logs_dir: Path,
) -> tuple[subprocess.Popen[bytes], TextIO]:
    """
    Launch asynchronous refinement subprocess for a single refinement job.

    :param job: Refinement job configuration.
    :type job: RefineJob
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param gpu_id: GPU device ordinal for CUDA_VISIBLE_DEVICES.
    :type gpu_id: int
    :param logs_dir: Directory where worker stdout log is written.
    :type logs_dir: Path
    :return: Tuple of spawned process instance and open log file handle.
    :rtype: tuple[subprocess.Popen[bytes], TextIO]
    """
    env = build_worker_environment(job.workdir, gpu_id)
    cmd = build_refine_command(
        job.system, engine, job.reaction_ids, job.skip_species
    )
    log_handle = (logs_dir / f"{engine}_{job.job_id}.log").open(
        "w", encoding="utf-8"
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(job.workdir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_handle


def launch_all_refine_workers(
    work_dir: Path,
    engine: str,
    jobs: Sequence[RefineJob],
    gpus: int,
) -> tuple[
    dict[str, subprocess.Popen[bytes]],
    dict[str, TextIO],
    dict[str, float],
]:
    """
    Spawn all worker processes across assigned GPU devices.

    :param work_dir: Working directory containing unpacked bundle.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param jobs: Sequence of refinement jobs to execute.
    :type jobs: Sequence[RefineJob]
    :param gpus: Number of available GPU devices.
    :type gpus: int
    :return: Tuple of mappings: processes, log handles, start timestamps.
    :rtype: tuple[dict, dict, dict]
    """
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs: dict[str, subprocess.Popen[bytes]] = {}
    handles: dict[str, TextIO] = {}
    starts: dict[str, float] = {}
    for idx, job in enumerate(jobs):
        proc, handle = launch_refine_worker(job, engine, idx % gpus, logs_dir)
        procs[job.job_id] = proc
        handles[job.job_id] = handle
        starts[job.job_id] = time.time()
    return procs, handles, starts


def record_system_completion(
    system: str,
    code: int,
    start_time: float,
    log_handle: TextIO,
    exit_codes: dict[str, int],
    wall_times: dict[str, float],
) -> None:
    """
    Record exit code and elapsed wall time and close log file handle.

    :param system: Target job identifier string.
    :type system: str
    :param code: Process exit status code.
    :type code: int
    :param start_time: Process start epoch timestamp.
    :type start_time: float
    :param log_handle: Log file handle to close.
    :type log_handle: TextIO
    :param exit_codes: Mapping of job to exit code.
    :type exit_codes: dict[str, int]
    :param wall_times: Mapping of job to elapsed wall seconds.
    :type wall_times: dict[str, float]
    """
    exit_codes[system] = code
    wall_times[system] = time.time() - start_time
    log_handle.close()


def poll_completed_workers(
    active: set[str],
    pool: WorkerPool,
) -> None:
    """
    Check active processes and record outcomes for those that have finished.

    :param active: Set of currently active job identifiers.
    :type active: set[str]
    :param pool: Worker pool tracking processes and runtimes.
    :type pool: WorkerPool
    """
    for job_id in list(active):
        code = pool.processes[job_id].poll()
        if code is not None:
            record_system_completion(
                job_id,
                code,
                pool.start_times[job_id],
                pool.log_handles[job_id],
                pool.exit_codes,
                pool.wall_times,
            )
            active.remove(job_id)


def send_signal_to_process_group(
    proc: subprocess.Popen[bytes],
    sig: signal.Signals,
) -> None:
    """
    Send termination signal to process group safely ignoring lookup errors.

    :param proc: Process whose group should receive signal.
    :type proc: subprocess.Popen[bytes]
    :param sig: Signal to deliver (e.g. SIGTERM or SIGKILL).
    :type sig: signal.Signals
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        pass


def terminate_worker_on_deadline(
    proc: subprocess.Popen[bytes],
) -> int:
    """
    Poll or force kill process on deadline timeout and return exit code.

    :param proc: Process to terminate.
    :type proc: subprocess.Popen[bytes]
    :return: Process exit code.
    :rtype: int
    """
    code = proc.poll()
    if code is not None:
        return code
    send_signal_to_process_group(proc, signal.SIGKILL)
    return proc.wait()


def terminate_remaining_workers(
    active: set[str],
    pool: WorkerPool,
) -> None:
    """
    Deliver SIGTERM to active workers, wait grace period, and force kill.

    :param active: Set of active job identifiers.
    :type active: set[str]
    :param pool: Worker pool tracking processes and runtimes.
    :type pool: WorkerPool
    """
    for job_id in list(active):
        send_signal_to_process_group(pool.processes[job_id], signal.SIGTERM)
    time.sleep(15.0)
    for job_id in list(active):
        code = terminate_worker_on_deadline(pool.processes[job_id])
        record_system_completion(
            job_id,
            code,
            pool.start_times[job_id],
            pool.log_handles[job_id],
            pool.exit_codes,
            pool.wall_times,
        )
        active.remove(job_id)


def supervise_parallel_workers(
    pool: WorkerPool,
    deadline_seconds: float,
) -> None:
    """
    Monitor running workers until completion or deadline expiration.

    :param pool: Worker pool tracking processes and runtimes.
    :type pool: WorkerPool
    :param deadline_seconds: Timeout threshold in seconds.
    :type deadline_seconds: float
    """
    active = set(pool.processes.keys())
    start_all = time.time()
    while active:
        if time.time() - start_all >= deadline_seconds:
            terminate_remaining_workers(active, pool)
            break
        poll_completed_workers(active, pool)
        if active:
            time.sleep(1.0)


def ignore_workspace_parts_and_logs(
    directory: str,
    names: Sequence[str],
) -> set[str]:
    """
    Ignore parts and logs folders when cloning workspace directory tree.

    :param directory: Current directory path being visited by copytree.
    :type directory: str
    :param names: Sequence of file and directory names in current directory.
    :type names: Sequence[str]
    :return: Set of names to exclude from copying.
    :rtype: set[str]
    """
    return {name for name in names if name in {"parts", "logs"}}


def record_file_offsets(target_dir: Path) -> dict[str, int]:
    """
    Record byte sizes of all files in target directory.

    :param target_dir: Directory containing files to inspect.
    :type target_dir: Path
    :return: Mapping of filename to byte size.
    :rtype: dict[str, int]
    """
    if not target_dir.is_dir():
        return {}
    offsets: dict[str, int] = {}
    for item in sorted(target_dir.iterdir()):
        if item.is_file():
            offsets[item.name] = item.stat().st_size
    return offsets


def create_part_workspace(
    main_work_dir: Path,
    job: RefineJob,
    engine: str,
) -> None:
    """
    Clone main workspace into part directory and record file offsets.

    :param main_work_dir: Main workspace directory.
    :type main_work_dir: Path
    :param job: Part refinement job configuration.
    :type job: RefineJob
    :param engine: Target computational chemistry engine.
    :type engine: str
    """
    if job.workdir == main_work_dir:
        return
    job.workdir.parent.mkdir(parents=True, exist_ok=True)
    if job.workdir.exists():
        shutil.rmtree(job.workdir)
    shutil.copytree(
        main_work_dir,
        job.workdir,
        ignore=ignore_workspace_parts_and_logs,
    )
    part_engine_dir = job.workdir / "results" / "refine" / job.system / engine
    offsets = record_file_offsets(part_engine_dir)
    (job.workdir / "part_offsets.json").write_text(
        json.dumps(offsets, indent=2), encoding="utf-8"
    )


def prepare_part_workspaces(
    main_work_dir: Path,
    jobs: Sequence[RefineJob],
    engine: str,
) -> None:
    """
    Prepare isolated workspace directories and record offsets for part jobs.

    :param main_work_dir: Main working directory.
    :type main_work_dir: Path
    :param jobs: Sequence of refinement jobs.
    :type jobs: Sequence[RefineJob]
    :param engine: Target computational chemistry engine.
    :type engine: str
    """
    for job in jobs:
        if job.workdir != main_work_dir:
            create_part_workspace(main_work_dir, job, engine)


def load_part_offsets(workdir: Path) -> dict[str, int]:
    """
    Load recorded file offsets mapping from part workspace directory.

    :param workdir: Part workspace directory path.
    :type workdir: Path
    :return: Mapping of filename to byte offset.
    :rtype: dict[str, int]
    """
    offsets_path = workdir / "part_offsets.json"
    if not offsets_path.is_file():
        return {}
    return json.loads(offsets_path.read_text(encoding="utf-8"))


def drop_csv_duplicate_header(
    tail: bytes,
    main_file: Path,
) -> bytes:
    """
    Drop first line of CSV tail if main file is non-empty with matching header.

    :param tail: Raw tail bytes to inspect.
    :type tail: bytes
    :param main_file: Path to destination file in main workspace.
    :type main_file: Path
    :return: Adjusted tail bytes without duplicate header.
    :rtype: bytes
    """
    if not (main_file.is_file() and main_file.stat().st_size > 0):
        return tail
    main_first_line = main_file.read_bytes().split(b"\n", 1)[0]
    tail_first_line = tail.split(b"\n", 1)[0]
    if tail_first_line == main_first_line:
        return tail.split(b"\n", 1)[1] if b"\n" in tail else b""
    return tail


def join_file_tail(
    part_file: Path,
    main_engine_dir: Path,
    offset: int,
) -> None:
    """
    Append byte tail from part file past offset into main engine results file.

    :param part_file: Source file path in part workspace.
    :type part_file: Path
    :param main_engine_dir: Destination engine directory in main workspace.
    :type main_engine_dir: Path
    :param offset: Recorded byte offset before execution.
    :type offset: int
    """
    data = part_file.read_bytes()
    if len(data) <= offset:
        return
    tail = data[offset:]
    main_file = main_engine_dir / part_file.name
    if part_file.suffix == ".csv":
        tail = drop_csv_duplicate_header(tail, main_file)
    if not tail:
        return
    main_engine_dir.mkdir(parents=True, exist_ok=True)
    with main_file.open("ab") as handle:
        handle.write(tail)


def join_single_part(
    main_work_dir: Path,
    part_job: RefineJob,
    engine: str,
) -> None:
    """
    Join new results from a part workspace into the main workspace.

    :param main_work_dir: Main working directory.
    :type main_work_dir: Path
    :param part_job: Part refinement job configuration.
    :type part_job: RefineJob
    :param engine: Target computational chemistry engine.
    :type engine: str
    """
    marker = part_job.workdir / "part_joined"
    if marker.exists():
        return
    part_engine_dir = (
        part_job.workdir / "results" / "refine" / part_job.system / engine
    )
    if not part_engine_dir.is_dir():
        marker.touch()
        return
    trim_refine_files(part_engine_dir, engine)
    offsets = load_part_offsets(part_job.workdir)
    main_engine_dir = (
        main_work_dir / "results" / "refine" / part_job.system / engine
    )
    for file_path in sorted(part_engine_dir.iterdir()):
        if file_path.is_file():
            offset = offsets.get(file_path.name, 0)
            join_file_tail(file_path, main_engine_dir, offset)
    marker.touch()


def join_all_parts(
    main_work_dir: Path,
    jobs: Sequence[RefineJob],
    engine: str,
) -> None:
    """
    Join results from all part jobs into the main workspace in part order.

    :param main_work_dir: Main working directory.
    :type main_work_dir: Path
    :param jobs: Sequence of refinement jobs.
    :type jobs: Sequence[RefineJob]
    :param engine: Target computational chemistry engine.
    :type engine: str
    """
    for job in jobs:
        if job.workdir != main_work_dir:
            join_single_part(main_work_dir, job, engine)


def aggregate_system_outcomes(
    systems: Sequence[str],
    jobs: Sequence[RefineJob],
    exit_codes: dict[str, int],
    wall_times: dict[str, float],
) -> tuple[dict[str, int], dict[str, float]]:
    """
    Aggregate per-job exit codes and wall times into per-system outcomes.

    :param systems: Sequence of system identifiers.
    :type systems: Sequence[str]
    :param jobs: Sequence of executed refinement jobs.
    :type jobs: Sequence[RefineJob]
    :param exit_codes: Mapping of job ID to process exit code.
    :type exit_codes: dict[str, int]
    :param wall_times: Mapping of job ID to elapsed runtime in seconds.
    :type wall_times: dict[str, float]
    :return: Tuple of system exit codes and system wall times.
    :rtype: tuple[dict[str, int], dict[str, float]]
    """
    sys_exit: dict[str, int] = {}
    sys_wall: dict[str, float] = {}
    for sys_id in systems:
        sys_jobs = [job for job in jobs if job.system == sys_id]
        if not sys_jobs:
            sys_exit[sys_id] = 1
            sys_wall[sys_id] = 0.0
            continue
        job_codes = [exit_codes.get(job.job_id, 1) for job in sys_jobs]
        first_fail = next((code for code in job_codes if code != 0), 0)
        sys_exit[sys_id] = first_fail
        job_times = [wall_times.get(job.job_id, 0.0) for job in sys_jobs]
        sys_wall[sys_id] = max(job_times) if job_times else 0.0
    return sys_exit, sys_wall


def run_parallel_refine(
    work_dir: Path,
    engine: str,
    jobs: Sequence[RefineJob],
    gpus: int,
    deadline_hours: float,
) -> tuple[dict[str, int], dict[str, float]]:
    """
    Launch and supervise parallel refinement worker processes.

    :param work_dir: Working directory containing unpacked bundle.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param jobs: Sequence of refinement jobs to execute.
    :type jobs: Sequence[RefineJob]
    :param gpus: Number of available GPU devices.
    :type gpus: int
    :param deadline_hours: Maximum allowable runtime in hours before SIGTERM.
    :type deadline_hours: float
    :return: Tuple of mappings: exit codes and wall times per job.
    :rtype: tuple[dict[str, int], dict[str, float]]
    """
    warmup_engine_cache(work_dir, engine)
    prepare_part_workspaces(work_dir, jobs, engine)
    procs, handles, starts = launch_all_refine_workers(
        work_dir, engine, jobs, gpus
    )
    exit_codes: dict[str, int] = {}
    wall_times: dict[str, float] = {}
    pool = WorkerPool(procs, handles, starts, exit_codes, wall_times)
    supervise_parallel_workers(pool, deadline_hours * 3600.0)
    return exit_codes, wall_times


def count_reactions(reactions_path: Path) -> int:
    """
    Count valid JSON lines in a reactions.jsonl file.

    :param reactions_path: Path to reactions.jsonl file.
    :type reactions_path: Path
    :return: Number of reaction records.
    :rtype: int
    """
    if not reactions_path.exists():
        return 0
    count = 0
    with reactions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def archive_directory_tree(
    archive: zipfile.ZipFile,
    source_dir: Path,
    base_dir: Path,
) -> None:
    """
    Recursively add files from source directory into open zip archive.

    :param archive: Open zip archive instance.
    :type archive: zipfile.ZipFile
    :param source_dir: Root directory whose contents are archived.
    :type source_dir: Path
    :param base_dir: Base directory for computing archive relative paths.
    :type base_dir: Path
    """
    if not source_dir.is_dir():
        return
    for file_path in sorted(source_dir.rglob("*")):
        if file_path.is_file():
            rel_name = file_path.relative_to(base_dir).as_posix()
            archive.write(file_path, arcname=rel_name)


def archive_engine_results(
    archive: zipfile.ZipFile,
    work_dir: Path,
    engine: str,
) -> None:
    """
    Archive refinement outputs for target engine across all systems.

    :param archive: Open zip archive instance.
    :type archive: zipfile.ZipFile
    :param work_dir: Working directory containing refinement results.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    """
    refine_base = work_dir / "results" / "refine"
    if not refine_base.exists():
        return
    for system_dir in sorted(refine_base.iterdir()):
        archive_directory_tree(archive, system_dir / engine, work_dir)


def package_results(
    work_dir: Path,
    engine: str,
    output_zip: Path,
) -> None:
    """
    Package refined calculation results, logs, and bundle metadata into zip.

    :param work_dir: Working directory containing results.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param output_zip: Destination zip archive path.
    :type output_zip: Path
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as arc:
        archive_engine_results(arc, work_dir, engine)
        archive_directory_tree(arc, work_dir / "logs", work_dir)
        bundle_json = work_dir / "BUNDLE.json"
        if bundle_json.is_file():
            arc.write(bundle_json, arcname="BUNDLE.json")


def format_system_summary_line(
    system: str,
    code: int,
    wall: float,
    rxns: int,
) -> str:
    """
    Format execution summary string for one system.

    :param system: System identifier.
    :type system: str
    :param code: Exit status code.
    :type code: int
    :param wall: Runtime in seconds.
    :type wall: float
    :param rxns: Count of reactions recorded.
    :type rxns: int
    :return: Formatted summary string.
    :rtype: str
    """
    return (
        f"system: {system} | exit: {code} | wall: {wall:.1f}s | "
        f"reactions: {rxns}"
    )


def format_single_system_summary(
    work_dir: Path,
    engine: str,
    sys_id: str,
    exit_code: int,
    wall_time: float,
) -> str:
    """
    Format execution summary for a single system using results on disk.

    :param work_dir: Working directory containing refinement results.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param sys_id: System identifier string.
    :type sys_id: str
    :param exit_code: System process exit status code.
    :type exit_code: int
    :param wall_time: Wall execution time in seconds.
    :type wall_time: float
    :return: Formatted summary line for system.
    :rtype: str
    """
    rxn_path = (
        work_dir / "results" / "refine" / sys_id / engine / "reactions.jsonl"
    )
    return format_system_summary_line(
        sys_id, exit_code, wall_time, count_reactions(rxn_path)
    )


def get_failed_systems(
    systems: Sequence[str],
    exit_codes: dict[str, int],
) -> list[str]:
    """
    Return list of systems with non-zero or missing exit codes.

    :param systems: Sequence of expected system identifiers.
    :type systems: Sequence[str]
    :param exit_codes: Mapping of system identifier to exit code.
    :type exit_codes: dict[str, int]
    :return: List of failed system identifiers.
    :rtype: list[str]
    """
    return [sys_id for sys_id in systems if exit_codes.get(sys_id) != 0]


def print_execution_status(failed_systems: Sequence[str]) -> None:
    """
    Print final execution status indicating failed or finished systems.

    :param failed_systems: Sequence of failed system identifiers.
    :type failed_systems: Sequence[str]
    """
    if failed_systems:
        print(f"FAILED systems: {', '.join(failed_systems)}")
    else:
        print("All systems finished")


def print_systems_summary(
    work_dir: Path,
    engine: str,
    systems: Sequence[str],
    exit_codes: dict[str, int],
    wall_times: dict[str, float],
) -> None:
    """
    Print one line per system summarizing execution status and reaction count.

    :param work_dir: Working directory containing refinement results.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers.
    :type systems: Sequence[str]
    :param exit_codes: Mapping from system to process exit code.
    :type exit_codes: dict[str, int]
    :param wall_times: Mapping from system to runtime in seconds.
    :type wall_times: dict[str, float]
    """
    for sys_id in systems:
        summary_line = format_single_system_summary(
            work_dir,
            engine,
            sys_id,
            exit_codes.get(sys_id, 1),
            wall_times.get(sys_id, 0.0),
        )
        print(summary_line)
    print_execution_status(get_failed_systems(systems, exit_codes))


def add_runner_location_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register directory path arguments on run_refine parser.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--bundle-dir", required=True, type=Path, help="Unpacked bundle path."
    )
    parser.add_argument(
        "--work-dir",
        default=Path("/kaggle/working/mse"),
        type=Path,
        help="Working directory path.",
    )


def add_runner_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register engine and system selection arguments on run_refine parser.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument("--engine", required=True, help="Engine identifier.")
    parser.add_argument(
        "--systems", nargs="+", required=True, help="System identifiers."
    )
    parser.add_argument(
        "--gpus", default=2, type=int, help="Available GPU count."
    )


def add_runner_recovery_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register resume checkpoint and timeout deadline arguments on parser.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--resume-zip", default=None, type=Path, help="Resume zip archive."
    )
    parser.add_argument(
        "--deadline-hours",
        default=11.5,
        type=float,
        help="Hours before SIGTERM.",
    )


def add_runner_mode_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register execution mode and packaging flags on run_refine parser.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--dry-run", action="store_true", help="Print worker commands only."
    )
    parser.add_argument(
        "--zip-only", action="store_true", help="Package results only."
    )
    parser.add_argument(
        "--results-zip", default=None, type=Path, help="Output zip override."
    )
    parser.add_argument(
        "--no-split", action="store_true", help="Ignore reaction split plans."
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for run_refine CLI.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Run nanoreactor refinement tasks across systems on GPU."
    )
    add_runner_location_arguments(parser)
    add_runner_engine_arguments(parser)
    add_runner_recovery_arguments(parser)
    add_runner_mode_arguments(parser)
    return parser


def load_split_plan(
    bundle_dir: Path,
    system: str,
) -> dict[str, Any] | None:
    """
    Load reaction split plan JSON from bundle splits directory if present.

    :param bundle_dir: Resolved bundle root directory.
    :type bundle_dir: Path
    :param system: Target chemical system identifier.
    :type system: str
    :return: Plan dictionary if file exists, else None.
    :rtype: dict[str, Any] | None
    """
    plan_path = bundle_dir / "splits" / f"{system}.json"
    if not plan_path.is_file():
        return None
    return json.loads(plan_path.read_text(encoding="utf-8"))


def build_system_jobs(
    bundle_dir: Path,
    work_dir: Path,
    system: str,
    no_split: bool = False,
) -> list[RefineJob]:
    """
    Construct refinement jobs for a single system based on split plan.

    :param bundle_dir: Resolved bundle root directory.
    :type bundle_dir: Path
    :param work_dir: Main working directory.
    :type work_dir: Path
    :param system: System identifier.
    :type system: str
    :param no_split: If True, ignore any split plans.
    :type no_split: bool
    :return: List of configured refinement jobs for system.
    :rtype: list[RefineJob]
    """
    if no_split:
        return [RefineJob(job_id=system, system=system, workdir=work_dir)]
    plan = load_split_plan(bundle_dir, system)
    parts = plan.get("parts", []) if plan else []
    if len(parts) <= 1:
        return [RefineJob(job_id=system, system=system, workdir=work_dir)]
    return [
        RefineJob(
            job_id=f"{system}-p{idx}",
            system=system,
            workdir=work_dir / "parts" / f"{system}-p{idx}",
            reaction_ids=tuple(part_ids),
            skip_species=(idx > 0),
        )
        for idx, part_ids in enumerate(parts)
    ]


def build_refine_jobs(
    bundle_dir: Path,
    work_dir: Path,
    systems: Sequence[str],
    no_split: bool = False,
) -> list[RefineJob]:
    """
    Construct all refinement jobs across systems respecting split plans.

    :param bundle_dir: Resolved bundle root directory.
    :type bundle_dir: Path
    :param work_dir: Main working directory.
    :type work_dir: Path
    :param systems: Sequence of system identifiers.
    :type systems: Sequence[str]
    :param no_split: If True, ignore any split plans.
    :type no_split: bool
    :return: Sequence of all configured refinement jobs.
    :rtype: list[RefineJob]
    """
    jobs: list[RefineJob] = []
    for system in systems:
        jobs.extend(build_system_jobs(bundle_dir, work_dir, system, no_split))
    return jobs


def setup_workspace_and_resume(
    bundle_dir: Path,
    work_dir: Path,
    engine: str,
    systems: Sequence[str],
    resume_zip: Path | None,
) -> None:
    """
    Copy bundle contents and restore previous results if resume zip is given.

    :param bundle_dir: Resolved bundle root directory.
    :type bundle_dir: Path
    :param work_dir: Target working directory.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers.
    :type systems: Sequence[str]
    :param resume_zip: Optional earlier results zip to restore.
    :type resume_zip: Path | None
    """
    copy_bundle(bundle_dir=bundle_dir, work_dir=work_dir)
    if resume_zip is not None:
        restore_resume_zip(resume_zip, work_dir, engine, systems)
        trim_refine_files(work_dir, engine)


def finalize_and_package_results(
    work_dir: Path,
    engine: str,
    systems: Sequence[str],
    results_zip: Path | None,
    exit_codes: dict[str, int],
    wall_times: dict[str, float],
) -> None:
    """
    Trim results, package archive, and print per-system summary.

    :param work_dir: Working directory containing results.
    :type work_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param systems: Sequence of system identifiers.
    :type systems: Sequence[str]
    :param results_zip: Optional explicit zip archive destination path.
    :type results_zip: Path | None
    :param exit_codes: Mapping of system to process exit code.
    :type exit_codes: dict[str, int]
    :param wall_times: Mapping of system to runtime in seconds.
    :type wall_times: dict[str, float]
    """
    trim_refine_files(work_dir, engine)
    out_zip = results_zip or (work_dir.parent / f"{engine}_results.zip")
    package_results(work_dir, engine, out_zip)
    print_systems_summary(work_dir, engine, systems, exit_codes, wall_times)


def execute_optional_refinement(
    args: argparse.Namespace,
    jobs: Sequence[RefineJob],
) -> tuple[dict[str, int], dict[str, float]]:
    """
    Execute refinement workers unless zip_only flag is specified.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :param jobs: Sequence of refinement jobs to execute.
    :type jobs: Sequence[RefineJob]
    :return: Tuple of exit codes and wall times mappings.
    :rtype: tuple[dict[str, int], dict[str, float]]
    """
    if args.zip_only:
        return {job.job_id: 0 for job in jobs}, {}
    return run_parallel_refine(
        args.work_dir,
        args.engine,
        jobs,
        args.gpus,
        args.deadline_hours,
    )


def run_refinement_workflow(
    args: argparse.Namespace,
    bundle_dir: Path,
    jobs: Sequence[RefineJob] | None = None,
) -> int:
    """
    Execute refinement workflow steps: restore, parallel execute, and package.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :param bundle_dir: Resolved bundle root path.
    :type bundle_dir: Path
    :param jobs: Optional pre-built sequence of refinement jobs.
    :type jobs: Sequence[RefineJob] | None
    :return: Process exit status code: 0 if all succeeded, 1 on any failure.
    :rtype: int
    """
    if jobs is None:
        jobs = build_refine_jobs(
            bundle_dir,
            args.work_dir,
            args.systems,
            getattr(args, "no_split", False),
        )
    setup_workspace_and_resume(
        bundle_dir, args.work_dir, args.engine, args.systems, args.resume_zip
    )
    exit_codes, wall_times = execute_optional_refinement(args, jobs)
    join_all_parts(args.work_dir, jobs, args.engine)
    sys_exit, sys_wall = aggregate_system_outcomes(
        args.systems, jobs, exit_codes, wall_times
    )
    finalize_and_package_results(
        args.work_dir,
        args.engine,
        args.systems,
        args.results_zip,
        sys_exit,
        sys_wall,
    )
    failed = get_failed_systems(args.systems, sys_exit)
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute run_refine command line interface.

    :param argv: Optional command-line arguments.
    :type argv: Sequence[str] | None
    :return: Process exit status code.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    bundle_dir = resolve_bundle_dir(args.bundle_dir)
    jobs = build_refine_jobs(
        bundle_dir, args.work_dir, args.systems, args.no_split
    )
    if args.dry_run:
        print_dry_run_commands(args.work_dir, args.engine, jobs, args.gpus)
        return 0
    return run_refinement_workflow(args, bundle_dir, jobs)


if __name__ == "__main__":
    sys.exit(main())
