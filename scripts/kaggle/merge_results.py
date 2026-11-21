"""
Merge Kaggle refinement calculation results into repository results tree.

Extracts engine refinement directories, archives any pre-existing local results
when replacement is requested, audits execution devices in calls.csv, and
reports reactions confirmed or attempted across all merged systems.
"""

import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "add_merge_dest_arguments",
    "add_merge_source_arguments",
    "archive_existing_directory",
    "archive_system_if_present",
    "audit_and_report_system",
    "audit_calls_device",
    "build_parser",
    "count_reactions_by_status",
    "detect_systems_in_zip",
    "extract_system_results",
    "format_status_breakdown",
    "main",
    "merge_results",
    "merge_single_system",
    "report_system_reactions",
    "validate_target_collisions",
]


def detect_systems_in_zip(
    zip_path: Path,
    engine: str,
) -> list[str]:
    """
    Identify all unique system identifiers present in the results zip archive.

    :param zip_path: Path to Kaggle results zip archive.
    :type zip_path: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :return: Sorted list of system identifier strings.
    :rtype: list[str]
    """
    systems: set[str] = set()
    prefix = "results/refine/"
    needle = f"/{engine}/"
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name.startswith(prefix) and needle in name:
                remainder = name[len(prefix) :]
                system_name = remainder.split(needle)[0]
                if system_name:
                    systems.add(system_name)
    return sorted(systems)


def archive_existing_directory(
    target_dir: Path,
    engine: str,
    system: str,
) -> Path:
    """
    Move existing results folder into timestamped user archive directory.

    :param target_dir: Existing directory to archive.
    :type target_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param system: System identifier.
    :type system: str
    :return: Path to destination archive directory.
    :rtype: Path
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_base = Path.home() / "mse-archive-2026-11-18"
    archive_dest = archive_base / f"{engine}_{system}_replaced_{stamp}"
    archive_base.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target_dir), str(archive_dest))
    return archive_dest


def extract_system_results(
    zip_path: Path,
    repo_root: Path,
    engine: str,
    system: str,
) -> int:
    """
    Extract files for a given system and engine from zip into repo_root.

    :param zip_path: Path to results zip archive.
    :type zip_path: Path
    :param repo_root: Base repository root directory.
    :type repo_root: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param system: System identifier.
    :type system: str
    :return: Count of files extracted.
    :rtype: int
    """
    count = 0
    prefix = f"results/refine/{system}/{engine}/"
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name.startswith(prefix) and not name.endswith("/"):
                dest_path = repo_root / name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.open(name) as src_fp,
                    dest_path.open("wb") as dst_fp,
                ):
                    shutil.copyfileobj(src_fp, dst_fp)
                count += 1
    return count


def audit_calls_device(
    calls_path: Path,
    system: str,
    engine: str,
) -> None:
    """
    Inspect calls.csv and warn if any invocation row did not use cuda device.

    :param calls_path: Path to calls.csv file.
    :type calls_path: Path
    :param system: System identifier.
    :type system: str
    :param engine: Target computational chemistry engine.
    :type engine: str
    """
    if not calls_path.exists():
        return
    with calls_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            device = (row.get("device") or "").strip().lower()
            if device != "cuda":
                item_id = row.get("item_id", "unknown")
                print(
                    f"WARNING: Non-cuda device '{device}' in "
                    f"{system}/{engine} calls.csv (item: {item_id})",
                    file=sys.stderr,
                )


def count_reactions_by_status(
    reactions_path: Path,
) -> tuple[int, dict[str, int]]:
    """
    Read reactions.jsonl and count occurrences of each reaction status.

    :param reactions_path: Path to reactions.jsonl file.
    :type reactions_path: Path
    :return: Tuple of total count and status-to-frequency mapping.
    :rtype: tuple[int, dict[str, int]]
    """
    if not reactions_path.exists():
        return 0, {}
    statuses: dict[str, int] = {}
    total = 0
    with reactions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_str = line.strip()
            if line_str:
                total += 1
                status = json.loads(line_str).get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1
    return total, statuses


def format_status_breakdown(statuses: dict[str, int]) -> str:
    """
    Format dictionary of reaction statuses into comma-separated text.

    :param statuses: Mapping of status names to counts.
    :type statuses: dict[str, int]
    :return: Comma-delimited status string or 'none'.
    :rtype: str
    """
    if not statuses:
        return "none"
    return ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items()))


def report_system_reactions(
    reactions_path: Path,
    system: str,
) -> None:
    """
    Print total reaction count and per-status breakdown for a system.

    :param reactions_path: Path to reactions.jsonl file.
    :type reactions_path: Path
    :param system: System identifier.
    :type system: str
    """
    total, statuses = count_reactions_by_status(reactions_path)
    status_str = format_status_breakdown(statuses)
    print(f"system: {system} | reactions: {total} | statuses: {status_str}")


def validate_target_collisions(
    repo_root: Path,
    systems: Sequence[str],
    engine: str,
    replace: bool,
) -> None:
    """
    Verify that destination results directories do not collide without replace.

    :param repo_root: Repository root path.
    :type repo_root: Path
    :param systems: Sequence of system identifiers.
    :type systems: Sequence[str]
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param replace: Flag allowing replacement of existing directories.
    :type replace: bool
    """
    if replace:
        return
    for system in systems:
        target_dir = repo_root / "results" / "refine" / system / engine
        if target_dir.exists() and any(target_dir.iterdir()):
            sys.exit(
                f"Target folder '{target_dir}' exists and is not empty. "
                "Use --replace to archive and overwrite."
            )


def archive_system_if_present(
    target_dir: Path,
    engine: str,
    system: str,
) -> None:
    """
    Archive existing non-empty system directory to user archive location.

    :param target_dir: Existing results directory path.
    :type target_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param system: System identifier.
    :type system: str
    """
    if target_dir.exists() and any(target_dir.iterdir()):
        archive_dest = archive_existing_directory(target_dir, engine, system)
        print(f"Archived existing directory to {archive_dest}")


def audit_and_report_system(
    target_dir: Path,
    engine: str,
    system: str,
) -> None:
    """
    Audit calls.csv execution device and report reaction statuses.

    :param target_dir: System results directory.
    :type target_dir: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param system: System identifier.
    :type system: str
    """
    audit_calls_device(target_dir / "calls.csv", system, engine)
    report_system_reactions(target_dir / "reactions.jsonl", system)


def merge_single_system(
    zip_path: Path,
    repo_root: Path,
    engine: str,
    system: str,
) -> None:
    """
    Archive existing results, extract system files, audit calls, and report.

    :param zip_path: Path to results zip archive.
    :type zip_path: Path
    :param repo_root: Path to repository root.
    :type repo_root: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param system: System identifier string.
    :type system: str
    """
    target_dir = repo_root / "results" / "refine" / system / engine
    archive_system_if_present(target_dir, engine, system)
    extract_system_results(zip_path, repo_root, engine, system)
    audit_and_report_system(target_dir, engine, system)


def merge_results(
    zip_path: Path,
    engine: str,
    repo_root: Path,
    replace: bool,
) -> None:
    """
    Coordinate checking, archiving, extraction, audit, and reporting.

    :param zip_path: Path to results zip archive.
    :type zip_path: Path
    :param engine: Target computational chemistry engine.
    :type engine: str
    :param repo_root: Path to repository root.
    :type repo_root: Path
    :param replace: Flag allowing replacement of existing non-empty folders.
    :type replace: bool
    :raises ValueError: If no matching results are found in archive.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Results zip not found: {zip_path}")

    systems = detect_systems_in_zip(zip_path, engine)
    if not systems:
        raise ValueError(
            f"No results for engine '{engine}' found in {zip_path}"
        )

    validate_target_collisions(repo_root, systems, engine, replace)
    for system in systems:
        merge_single_system(zip_path, repo_root, engine, system)


def add_merge_source_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register source archive and engine arguments on merge parser.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--zip",
        required=True,
        type=Path,
        help="Path to results zip archive (e.g. mace_results.zip).",
    )
    parser.add_argument(
        "--engine",
        required=True,
        help="Target computational chemistry engine (e.g. mace).",
    )


def add_merge_dest_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register destination repository and replacement flags on merge parser.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--repo-root",
        default=Path("."),
        type=Path,
        help="Path to repository root (defaults to current directory).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Archive and replace existing target directory if present.",
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for merge_results CLI.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Merge Kaggle refinement results into repository tree."
    )
    add_merge_source_arguments(parser)
    add_merge_dest_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute merge_results command line interface.

    :param argv: Optional command-line arguments.
    :type argv: Sequence[str] | None
    :return: Process exit status code.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    merge_results(
        zip_path=args.zip.resolve(),
        engine=args.engine,
        repo_root=args.repo_root.resolve(),
        replace=args.replace,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
