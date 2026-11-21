"""
Audit covalent bond detection scales and radical rejections on MD trajectories.

Applies covalent radius cutoffs (1.1, 1.2, and 1.3) to trajectory frames without
relaxation, detects reactive transitions via graph connectivity, identifies
events with multiple radical centers, and exports event counts to CSV.
"""

import argparse
import csv
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from nrnet.chem import radical_count
from nrnet.nanoreactor.candidates import reacting_atom_ids
from nrnet.nanoreactor.events import find_events
from nrnet.nanoreactor.md import (
    find_replicate_trajectories,
    read_trajectory,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")

__all__ = [
    "AUDIT_CSV_FIELDNAMES",
    "AUDIT_SCALES",
    "CLI_ARGUMENTS",
    "DEFAULT_DUMP_FS",
    "DEFAULT_MIN_FRAMES",
    "DEFAULT_WORKERS",
    "audit_all_trajectories",
    "audit_event_record",
    "audit_scale_events",
    "audit_trajectory",
    "build_audit_rows",
    "build_parser",
    "collect_trajectories",
    "count_event_radicals",
    "format_audit_row",
    "format_state_formula",
    "main",
    "print_all_scale_audits",
    "print_scale_audit",
    "write_audit_csv",
]

AUDIT_CSV_FIELDNAMES: tuple[str, ...] = (
    "scale",
    "state_before",
    "state_after",
    "multiple_radicals",
    "count",
)
AUDIT_SCALES: tuple[float, ...] = (1.1, 1.2, 1.3)
DEFAULT_WORKERS: int = 4
DEFAULT_MIN_FRAMES: int = 10
DEFAULT_DUMP_FS: float = 5.0

CLI_ARGUMENTS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("--system",),
        {
            "type": str,
            "required": True,
            "help": "System name (used for default CSV destination).",
        },
    ),
    (
        ("--runs",),
        {
            "nargs": "+",
            "required": True,
            "help": "One or more run directories or glob patterns.",
        },
    ),
    (
        ("--workers",),
        {
            "type": int,
            "default": DEFAULT_WORKERS,
            "help": "Number of parallel worker processes (default: 4).",
        },
    ),
    (
        ("--csv",),
        {
            "type": Path,
            "default": None,
            "help": (
                "Optional path to write audit CSV (default: data/<system>/...)."
            ),
        },
    ),
)


def format_state_formula(state: Sequence[str]) -> str:
    """
    Format state fragment keys into a sorted plus-separated Hill formula.

    :param state: Sequence of species keys.
    :type state: Sequence[str]
    :return: Sorted plus-separated formula string.
    :rtype: str
    """
    return " + ".join(sorted(key.split("-")[0] for key in state))


def count_event_radicals(event: Any) -> int:
    """
    Determine the maximum radical center count in reacting atoms for an event.

    :param event: Reactive transition event object.
    :type event: Any
    :return: Maximum radical center count between before and after frames.
    :rtype: int
    """
    atom_ids = reacting_atom_ids(event.frame_before, event.frame_after)
    if not atom_ids:
        return 0
    return max(
        radical_count(frame, atom_ids)
        for frame in (event.frame_before, event.frame_after)
    )


def audit_event_record(
    event: Any,
    scale: float,
) -> tuple[float, str, str, bool]:
    """
    Construct an audit record tuple for a single reactive event.

    :param event: Reactive transition event object.
    :type event: Any
    :param scale: Multiplicative bond scale cutoff.
    :type scale: float
    :return: Tuple of scale, state_before, state_after, and multiple_radicals.
    :rtype: tuple[float, str, str, bool]
    """
    rad_count = count_event_radicals(event)
    return (
        scale,
        format_state_formula(event.state_before),
        format_state_formula(event.state_after),
        rad_count > 1,
    )


def audit_scale_events(
    frames: Sequence[Any],
    scale: float,
) -> list[tuple[float, str, str, bool]]:
    """
    Detect and audit reactive events for a specific bond scale cutoff.

    :param frames: Trajectory frames to inspect.
    :type frames: Sequence[Any]
    :param scale: Multiplicative bond scale cutoff.
    :type scale: float
    :return: List of audited event tuples for the scale.
    :rtype: list[tuple[float, str, str, bool]]
    """
    events = find_events(
        frames,
        replicate=0,
        min_frames=DEFAULT_MIN_FRAMES,
        dump_fs=DEFAULT_DUMP_FS,
        scale=scale,
        settle_frames=DEFAULT_MIN_FRAMES,
    )
    return [audit_event_record(event, scale) for event in events]


def audit_trajectory(path: Path) -> list[tuple[float, str, str, bool]]:
    """
    Audit reaction events and radical counts across bond scale cutoffs.

    :param path: Path to trajectory file.
    :type path: Path
    :return: List of tuples with scale, before, after, and multiple_radicals.
    :rtype: list[tuple[float, str, str, bool]]
    """
    frames = read_trajectory(path)
    records: list[tuple[float, str, str, bool]] = []
    for scale in AUDIT_SCALES:
        records.extend(audit_scale_events(frames, scale))
    return records


def collect_trajectories(run_patterns: Sequence[str]) -> list[Path]:
    """
    Find all replicate trajectory paths from run-group directories or globs.

    :param run_patterns: Sequence of directory paths or glob expressions.
    :type run_patterns: Sequence[str]
    :return: Unique list of trajectory paths.
    :rtype: list[Path]
    """
    trajectories: list[Path] = []
    for pattern in run_patterns:
        for dir_path in sorted(Path().glob(pattern)):
            if dir_path.is_dir():
                trajectories.extend(find_replicate_trajectories(dir_path))
            elif dir_path.is_file() and dir_path.name.endswith(".trj"):
                trajectories.append(dir_path)
    return list(dict.fromkeys(trajectories))


def print_scale_audit(
    scale: float,
    scale_rows: Sequence[tuple[float, str, str, bool]],
) -> None:
    """
    Print summary statistics and commonest kept/rejected reactions for scale.

    :param scale: Multiplicative bond scale cutoff.
    :type scale: float
    :param scale_rows: Filtered audit rows for this scale.
    :type scale_rows: Sequence[tuple[float, str, str, bool]]
    :return: None
    :rtype: None
    """
    rad = [row for row in scale_rows if row[3]]
    pct = len(rad) / max(len(scale_rows), 1)
    print(
        f"\nscale {scale}: {len(scale_rows)} events, {len(rad)} with 2+ "
        f"radicals ({pct:.0%})"
    )
    print("  commonest kept:")
    kept = Counter((row[1], row[2]) for row in scale_rows if not row[3])
    for (before, after), count in kept.most_common(6):
        print(f"    {count:4d}  {before}  ->  {after}")
    print("  commonest radical rejects:")
    rejected = Counter((row[1], row[2]) for row in rad)
    for (before, after), count in rejected.most_common(6):
        print(f"    {count:4d}  {before}  ->  {after}")


def format_audit_row(
    key: tuple[float, str, str, bool],
    count: int,
) -> dict[str, Any]:
    """
    Convert an audit key and event count pair into a dictionary row for CSV.

    :param key: Tuple of (scale, state_before, state_after, multiple_radicals).
    :type key: tuple[float, str, str, bool]
    :param count: Aggregated frequency of the event.
    :type count: int
    :return: Formatted dictionary row with standard CSV keys.
    :rtype: dict[str, Any]
    """
    scale, before, after, rad_flag = key
    return {
        "scale": scale,
        "state_before": before,
        "state_after": after,
        "multiple_radicals": rad_flag,
        "count": count,
    }


def build_audit_rows(
    counts: Mapping[tuple[float, str, str, bool], int],
) -> list[dict[str, Any]]:
    """
    Sort and convert aggregated audit counts into CSV row records.

    :param counts: Map of (scale, state_before, state_after, rad) to count.
    :type counts: Mapping[tuple[float, str, str, bool], int]
    :return: List of sorted row dictionaries for CSV output.
    :rtype: list[dict[str, Any]]
    """
    sorted_items = sorted(
        counts.items(),
        key=lambda item: (item[0][0], -item[1], item[0][1], item[0][2]),
    )
    return [format_audit_row(key, count) for key, count in sorted_items]


def write_audit_csv(
    csv_path: Path,
    counts: Mapping[tuple[float, str, str, bool], int],
) -> None:
    """
    Export aggregated bond audit counts to CSV.

    :param csv_path: Target path for output CSV.
    :type csv_path: Path
    :param counts: Map of (scale, state_before, state_after, rad) to count.
    :type counts: Mapping[tuple[float, str, str, bool], int]
    :return: None
    :rtype: None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_audit_rows(counts)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def audit_all_trajectories(
    paths: Sequence[Path],
    workers: int,
) -> list[tuple[float, str, str, bool]]:
    """
    Run parallel bond audits across all located trajectory files.

    :param paths: Sequence of trajectory file paths.
    :type paths: Sequence[Path]
    :param workers: Number of parallel worker processes.
    :type workers: int
    :return: Flat list of audited event records.
    :rtype: list[tuple[float, str, str, bool]]
    """
    with Pool(workers) as pool:
        nested_rows = pool.map(audit_trajectory, paths)
    return [row for chunk in nested_rows for row in chunk]


def print_all_scale_audits(
    all_rows: Sequence[tuple[float, str, str, bool]],
) -> None:
    """
    Print audit statistics for all supported bond scale cutoffs.

    :param all_rows: Flat list of audited event records across trajectories.
    :type all_rows: Sequence[tuple[float, str, str, bool]]
    :return: None
    :rtype: None
    """
    for scale in AUDIT_SCALES:
        scale_rows = [row for row in all_rows if row[0] == scale]
        print_scale_audit(scale, scale_rows)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct command-line argument parser for bond scale audit.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Audit covalent bond detection scales and radical counts."
    )
    for flags, options in CLI_ARGUMENTS:
        parser.add_argument(*flags, **options)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute CLI entry point for bond detection audit.

    :param argv: Optional command-line arguments list.
    :type argv: Sequence[str] or None
    :return: Process exit status code.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = collect_trajectories(args.runs)
    all_rows = audit_all_trajectories(paths, args.workers)
    print(f"{args.system}: {len(paths)} trajectories")
    print_all_scale_audits(all_rows)
    csv_path = args.csv or (
        Path("data") / args.system / "checks" / "bond_audit.csv"
    )
    counts = Counter(all_rows)
    write_audit_csv(csv_path, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
