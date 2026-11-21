"""
List candidate reactions in a frozen set with species SMILES.

Loads a frozen candidate set directory, displays reaction identifiers, event
counts, pair numbers, engine sources, literature annotations, and SMILES
formulas for reactant and product states, and optionally exports a CSV.
"""

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import Candidate

__all__ = [
    "build_parser",
    "candidate_to_row",
    "display_candidates",
    "format_reaction_line",
    "format_state_side",
    "main",
    "write_reactions_csv",
]


def format_state_side(
    state: Sequence[str],
    names: Mapping[str, str],
) -> str:
    """
    Format reaction state keys as a plus-separated formula or SMILES string.

    :param state: Sequence of species keys on one reaction side.
    :type state: Sequence[str]
    :param names: Mapping of species keys to SMILES or display names.
    :type names: Mapping[str, str]
    :return: Formatted state string joined with plus signs.
    :rtype: str
    """
    return " + ".join(names.get(key, key) for key in state)


def format_reaction_line(
    candidate: Candidate,
    names: Mapping[str, str],
) -> str:
    """
    Format a candidate reaction entry as a fixed-width table row string.

    :param candidate: Candidate reaction object to format.
    :type candidate: Candidate
    :param names: Mapping of species keys to display labels.
    :type names: Mapping[str, str]
    :return: Formatted fixed-width row line for stdout.
    :rtype: str
    """
    side_a = format_state_side(candidate.state_a, names)
    side_b = format_state_side(candidate.state_b, names)
    lit_name = candidate.literature_name or ""
    return (
        f"{candidate.reaction_id} {candidate.n_events:4d} "
        f"{len(candidate.pairs):3d}p {candidate.source:10s} "
        f"{lit_name:30s} {side_a}  <>  {side_b}"
    )


def candidate_to_row(
    candidate: Candidate,
    names: Mapping[str, str],
) -> dict[str, Any]:
    """
    Convert a candidate reaction into a flat dictionary row for CSV output.

    :param candidate: Candidate reaction object to convert.
    :type candidate: Candidate
    :param names: Mapping of species keys to display labels.
    :type names: Mapping[str, str]
    :return: Dictionary row with canonical CSV column keys.
    :rtype: dict[str, Any]
    """
    return {
        "reaction_id": candidate.reaction_id,
        "n_events": candidate.n_events,
        "pairs": len(candidate.pairs),
        "source": candidate.source,
        "literature_name": candidate.literature_name or "",
        "state_a": format_state_side(candidate.state_a, names),
        "state_b": format_state_side(candidate.state_b, names),
    }


def write_reactions_csv(
    rows: Sequence[Mapping[str, Any]],
    csv_path: Path,
) -> None:
    """
    Write candidate reaction records to a CSV file.

    :param rows: Sequence of reaction row dictionaries.
    :type rows: Sequence[Mapping[str, Any]]
    :param csv_path: Destination path of the CSV file.
    :type csv_path: Path
    :return: None
    :rtype: None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reaction_id",
        "n_events",
        "pairs",
        "source",
        "literature_name",
        "state_a",
        "state_b",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct command-line argument parser for candidate set inspection.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="List reactions in a frozen candidate set with SMILES."
    )
    parser.add_argument(
        "candidate_dir",
        type=Path,
        help="Path to candidate set directory containing manifest.json.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write reaction table as CSV.",
    )
    return parser


def display_candidates(
    candidates: Sequence[Candidate],
    names: Mapping[str, str],
) -> None:
    """
    Print formatted candidate reactions to standard output.

    :param candidates: Sequence of candidate reactions.
    :type candidates: Sequence[Candidate]
    :param names: Mapping of species keys to display labels.
    :type names: Mapping[str, str]
    :return: None
    :rtype: None
    """
    for cand in candidates:
        print(format_reaction_line(cand, names))


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute CLI entry point for listing candidate reactions.

    :param argv: Optional command-line arguments list.
    :type argv: Sequence[str] or None
    :return: Process exit status code.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    cset = load_candidates(args.candidate_dir)
    names = {
        key: (spec.smiles or key) for key, spec in cset.species.items()
    }
    sorted_cands = sorted(cset.candidates, key=lambda cand: -cand.n_events)
    display_candidates(sorted_cands, names)
    if args.csv is not None:
        rows = [candidate_to_row(cand, names) for cand in sorted_cands]
        write_reactions_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
