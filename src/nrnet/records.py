"""
Execution recording and CSV call logging for computational engine tasks.

Provides RunContext and CallRecord dataclasses, SHA-256 configuration hashing,
Git commit resolution, context-managed timing and logging of engine calls,
and CSV record deserialization.
"""

import contextlib
import csv
import functools
import hashlib
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "RECORD_COLUMNS",
    "STATUS_ENGINE_ERROR",
    "STATUS_OK",
    "CallRecord",
    "RunContext",
    "append_record",
    "build_record_row",
    "config_hash",
    "get_git_head",
    "git_commit",
    "is_git_dirty",
    "read_calls",
    "record_call",
]

STATUS_OK: str = "ok"
STATUS_ENGINE_ERROR: str = "engine_error"

RECORD_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "engine",
    "system",
    "task",
    "item_id",
    "status",
    "wall_s",
    "force_calls",
    "device",
    "git_commit",
    "config_hash",
    "note",
)


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Execution context holding invariant metadata for an engine run.

    :param log_path: Filesystem path to destination CSV log.
    :type log_path: Path
    :param engine: Name of computational chemistry engine.
    :type engine: str
    :param system: Chemical system identifier.
    :type system: str
    :param device: Execution device identifier (e.g. 'cpu', 'cuda').
    :type device: str
    :param config_hash: 12-character SHA-256 digest of input configurations.
    :type config_hash: str
    :param git_commit: Repository Git commit hash with dirty tag if modified.
    :type git_commit: str
    """

    log_path: Path
    engine: str
    system: str
    device: str
    config_hash: str
    git_commit: str


@dataclass(slots=True)
class CallRecord:
    """
    Mutable record of an engine call outcome and performance counters.

    :param status: Execution status string ('ok', 'engine_error', etc.).
    :type status: str
    :param force_calls: Number of gradient evaluations.
    :type force_calls: int
    :param note: Additional notes or exception class name on failure.
    :type note: str
    """

    status: str = STATUS_OK
    force_calls: int = 0
    note: str = ""


def config_hash(*paths: Path | str) -> str:
    """
    Compute 12-character SHA-256 digest over file contents in order.

    :param paths: Sequence of file paths to hash.
    :type paths: Path | str
    :return: First 12 hex characters of combined SHA-256 digest.
    :rtype: str
    """
    hasher = hashlib.sha256()
    for file_path in paths:
        hasher.update(Path(file_path).read_bytes())
    return hasher.hexdigest()[:12]


def get_git_head(repo_dir: Path | str) -> str | None:
    """
    Retrieve short Git commit hash of HEAD if repository has commits.

    :param repo_dir: Path to repository root directory.
    :type repo_dir: Path | str
    :return: Short commit hash if commit exists, otherwise None.
    :rtype: str | None
    """
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    res = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True, check=False
    )
    return res.stdout.strip() if res.returncode == 0 else None


def is_git_dirty(repo_dir: Path | str) -> bool:
    """
    Check if repository has uncommitted modifications or untracked files.

    :param repo_dir: Path to repository root directory.
    :type repo_dir: Path | str
    :return: True if repository working tree is modified, False otherwise.
    :rtype: bool
    """
    cmd = ["git", "status", "--porcelain"]
    res = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True, check=False
    )
    return bool(res.stdout.strip())


@functools.cache
def git_commit(repo_dir: Path | str = Path(".")) -> str:
    """
    Retrieve short Git commit hash, appending '+dirty' if modified.

    :param repo_dir: Path to repository root directory.
    :type repo_dir: Path | str
    :return: Short commit hash, hash with '+dirty', or 'uncommitted'.
    :rtype: str
    """
    head = get_git_head(repo_dir)
    if head is None:
        return "uncommitted"
    if is_git_dirty(repo_dir):
        return f"{head}+dirty"
    return head


def build_record_row(
    context: RunContext,
    task: str,
    item_id: str,
    record: CallRecord,
    wall_s: float,
) -> list[str]:
    """
    Construct CSV row values matching RECORD_COLUMNS order.

    :param context: Active run context containing metadata.
    :type context: RunContext
    :param task: Name of computational task.
    :type task: str
    :param item_id: Identifier of item processed.
    :type item_id: str
    :param record: Execution record data.
    :type record: CallRecord
    :param wall_s: Elapsed wall-clock time in seconds.
    :type wall_s: float
    :return: List of formatted string field values.
    :rtype: list[str]
    """
    return [
        datetime.now(UTC).isoformat(),
        context.engine,
        context.system,
        str(task),
        str(item_id),
        record.status,
        f"{wall_s:.6f}",
        str(record.force_calls),
        context.device,
        context.git_commit,
        context.config_hash,
        record.note,
    ]


def append_record(
    context: RunContext,
    task: str,
    item_id: str,
    record: CallRecord,
    wall_s: float,
) -> None:
    """
    Append a single execution record row to the context log CSV file.

    :param context: Active run context containing metadata and log path.
    :type context: RunContext
    :param task: Name of computational task being recorded.
    :type task: str
    :param item_id: Unique identifier of item processed by task.
    :type item_id: str
    :param record: Execution outcome details and counters.
    :type record: CallRecord
    :param wall_s: Elapsed wall-clock time in seconds.
    :type wall_s: float
    :return: None.
    :rtype: None
    """
    path = Path(context.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    row = build_record_row(context, task, item_id, record, wall_s)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(RECORD_COLUMNS)
        writer.writerow(row)


@contextlib.contextmanager
def record_call(
    context: RunContext,
    task: str,
    item_id: str,
) -> Iterator[CallRecord]:
    """
    Context manager timing an engine call and logging outcome to CSV.

    :param context: Active run context with target log path and run metadata.
    :type context: RunContext
    :param task: Name of computational task being executed.
    :type task: str
    :param item_id: Identifier of molecular item processed.
    :type item_id: str
    :return: Yields mutable CallRecord for updating status and counters.
    :rtype: Iterator[CallRecord]
    :raises Exception: Re-raises any exception raised within the managed block.
    """
    record = CallRecord()
    start_time = time.perf_counter()
    try:
        yield record
    except BaseException as exc:
        # BaseException too: a Ctrl-C or a Slurm kill must not log as ok.
        record.status = STATUS_ENGINE_ERROR
        record.note = exc.__class__.__name__
        raise
    finally:
        wall_s = time.perf_counter() - start_time
        append_record(context, task, item_id, record, wall_s)


def read_calls(path: Path | str) -> list[dict[str, str]]:
    """
    Read call log CSV rows as dictionaries mapping column names to strings.

    :param path: Path to CSV call log file.
    :type path: Path | str
    :return: List of call records as column-to-value dictionaries.
    :rtype: list[dict[str, str]]
    """
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
