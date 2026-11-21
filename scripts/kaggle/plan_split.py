"""
Plan balanced reaction splits for distributed Kaggle GPU refinement.

Partitions candidate reaction networks into N workload-balanced groups using
historical force call counters from engine evaluation logs.
"""

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nrnet.refine import DEFAULT_CANDIDATES_DIR, load_candidates

__all__ = [
    "DEFAULT_PARTS_COUNT",
    "DEFAULT_WEIGHTS_ENGINE",
    "add_plan_split_arguments",
    "assign_greedy",
    "assign_median_weights",
    "build_parser",
    "build_split_payload",
    "calculate_median_weight",
    "format_part_summary",
    "load_candidate_reaction_ids",
    "main",
    "read_raw_call_weights",
    "read_weights",
    "write_split_json",
]

DEFAULT_WEIGHTS_ENGINE: str = "uma"
DEFAULT_PARTS_COUNT: int = 2


def calculate_median_weight(weights: Sequence[int]) -> int:
    """
    Compute integer median weight from recorded call counts.

    :param weights: Sequence of integer call counts.
    :type weights: Sequence[int]
    :return: Rounded integer median weight or default fallback.
    :rtype: int
    """
    if not weights:
        return 1
    return round(statistics.median(weights))


def assign_median_weights(
    candidate_ids: Sequence[str],
    known_weights: Mapping[str, int],
) -> dict[str, int]:
    """
    Assign force call weights to candidates, imputing missing IDs with median.

    :param candidate_ids: Sequence of candidate reaction identifiers.
    :type candidate_ids: Sequence[str]
    :param known_weights: Mapping of known reaction IDs to force call counts.
    :type known_weights: Mapping[str, int]
    :return: Mapping of each candidate ID to its integer weight.
    :rtype: dict[str, int]
    """
    observed = [
        known_weights[cid] for cid in candidate_ids if cid in known_weights
    ]
    if not observed and known_weights:
        observed = list(known_weights.values())
    fallback = calculate_median_weight(observed)
    return {cid: known_weights.get(cid, fallback) for cid in candidate_ids}


def read_raw_call_weights(calls_path: Path) -> dict[str, int]:
    """
    Read force calls per reaction item from calls.csv.

    :param calls_path: Path to calls.csv file.
    :type calls_path: Path
    :return: Mapping of item ID to force call count from last matching row.
    :rtype: dict[str, int]
    """
    if not calls_path.exists():
        return {}
    weights: dict[str, int] = {}
    with open(calls_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("task") != "species":
                item_id = row.get("item_id")
                raw_calls = row.get("force_calls")
                if item_id and raw_calls is not None:
                    weights[item_id] = int(raw_calls)
    return weights


def read_weights(
    calls_path: Path,
    candidate_ids: Sequence[str],
) -> dict[str, int]:
    """
    Read reaction force-call weights from calls.csv, imputing missing IDs.

    :param calls_path: Path to calls.csv file.
    :type calls_path: Path
    :param candidate_ids: Sequence of candidate reaction identifiers.
    :type candidate_ids: Sequence[str]
    :return: Mapping from candidate reaction ID to integer weight.
    :rtype: dict[str, int]
    """
    known_weights = read_raw_call_weights(calls_path)
    return assign_median_weights(candidate_ids, known_weights)


def assign_greedy(
    weights: Mapping[str, int],
    n_parts: int,
) -> tuple[list[list[str]], list[int]]:
    """
    Greedily partition reaction IDs across N parts balancing total weight.

    :param weights: Mapping of reaction ID to integer weight.
    :type weights: Mapping[str, int]
    :param n_parts: Number of target partitions.
    :type n_parts: int
    :return: Tuple of sorted ID lists per part and corresponding part weights.
    :rtype: tuple[list[list[str]], list[int]]
    :raises ValueError: If n_parts is less than 1.
    """
    if n_parts < 1:
        raise ValueError(f"n_parts must be at least 1, got {n_parts}")
    parts: list[list[str]] = [[] for _ in range(n_parts)]
    part_weights: list[int] = [0] * n_parts
    sorted_items = sorted(
        weights.items(),
        key=lambda item: (-item[1], item[0]),
    )
    for item_id, weight in sorted_items:
        min_idx = min(range(n_parts), key=lambda idx: part_weights[idx])
        parts[min_idx].append(item_id)
        part_weights[min_idx] += weight
    sorted_parts = [sorted(part) for part in parts]
    return sorted_parts, part_weights


def build_split_payload(
    system: str,
    weights_from: str,
    parts: Sequence[Sequence[str]],
    part_weights: Sequence[int],
) -> dict[str, Any]:
    """
    Construct JSON payload mapping for partitioned reaction work.

    :param system: Target system identifier.
    :type system: str
    :param weights_from: Source engine name used for weights.
    :type weights_from: str
    :param parts: Sequence of reaction ID sequences per part.
    :type parts: Sequence[Sequence[str]]
    :param part_weights: Sequence of total integer weights per part.
    :type part_weights: Sequence[int]
    :return: Formatted dictionary ready for JSON serialization.
    :rtype: dict[str, Any]
    """
    return {
        "system": system,
        "weights_from": weights_from,
        "parts": [list(part) for part in parts],
        "part_weights": [int(weight) for weight in part_weights],
    }


def format_part_summary(part_idx: int, count: int, weight: int) -> str:
    """
    Format single-line summary for reaction partition.

    :param part_idx: Zero-based part index.
    :type part_idx: int
    :param count: Count of reactions in part.
    :type count: int
    :param weight: Total force-call weight for part.
    :type weight: int
    :return: Formatted summary line string.
    :rtype: str
    """
    return f"part {part_idx}: {count} reactions, weight {weight}"


def load_candidate_reaction_ids(repo_root: Path, system: str) -> list[str]:
    """
    Load candidate reaction identifiers from candidate store.

    :param repo_root: Base repository root path.
    :type repo_root: Path
    :param system: Target system identifier.
    :type system: str
    :return: List of candidate reaction IDs.
    :rtype: list[str]
    """
    cand_path = repo_root / DEFAULT_CANDIDATES_DIR / system
    cset = load_candidates(cand_path)
    return [cand.reaction_id for cand in cset.candidates]


def write_split_json(out_path: Path, payload: Mapping[str, Any]) -> None:
    """
    Write serialized split dictionary to JSON file destination.

    :param out_path: Destination JSON file path.
    :type out_path: Path
    :param payload: Data dictionary to write.
    :type payload: Mapping[str, Any]
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def add_plan_split_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register command-line arguments for plan_split CLI.

    :param parser: Argument parser to configure.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument(
        "--system", required=True, help="Target system identifier."
    )
    parser.add_argument(
        "--weights-from",
        default=DEFAULT_WEIGHTS_ENGINE,
        help="Engine identifier for force-call weights (default: uma).",
    )
    parser.add_argument(
        "--parts",
        default=DEFAULT_PARTS_COUNT,
        type=int,
        help="Number of parts to split reactions into (default: 2).",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output destination JSON path."
    )
    parser.add_argument(
        "--repo-root",
        default=Path("."),
        type=Path,
        help="Repository root path (default: .).",
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for plan_split CLI.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Plan balanced reaction partitions for Kaggle refinement."
    )
    add_plan_split_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute plan_split command line interface.

    :param argv: Optional command-line argument sequence.
    :type argv: Sequence[str] | None
    :return: Process exit status code.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    candidate_ids = load_candidate_reaction_ids(repo_root, args.system)
    calls_path = (
        repo_root
        / "results"
        / "refine"
        / args.system
        / args.weights_from
        / "calls.csv"
    )
    weights = read_weights(calls_path, candidate_ids)
    parts, part_weights = assign_greedy(weights, args.parts)
    payload = build_split_payload(
        args.system, args.weights_from, parts, part_weights
    )
    write_split_json(args.out.resolve(), payload)
    for idx, (part, weight) in enumerate(zip(parts, part_weights, strict=True)):
        print(format_part_summary(idx, len(part), weight))
    return 0


if __name__ == "__main__":
    sys.exit(main())
