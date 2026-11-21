"""
Compare main versus check replicate groups for phase B simulation sweeps.

Loads frozen reaction candidates for specified push constants across main
replicate groups (e.g. g1-4) and check replicate groups (e.g. g5-6), evaluates
shared reaction coverage, and exports a comparison CSV.
"""

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.sweep import candidate_identity

__all__ = [
    "CLI_ARGUMENTS",
    "COMPARISON_CSV_FIELDNAMES",
    "DEFAULT_CHECK_GROUPS",
    "DEFAULT_KPUSHES",
    "DEFAULT_MAIN_GROUPS",
    "build_comparison_rows",
    "build_parser",
    "export_comparison_results",
    "format_reaction_label",
    "load_group_reactions",
    "main",
    "parse_group_range",
    "parse_kpush_list",
    "print_kpush_summary",
    "print_reaction_event_table",
    "print_union_summary",
    "run_kpush_sweeps",
    "write_comparison_csv",
]

DEFAULT_KPUSHES: str = "0.03,0.035"
DEFAULT_MAIN_GROUPS: str = "1-4"
DEFAULT_CHECK_GROUPS: str = "5-6"

COMPARISON_CSV_FIELDNAMES: tuple[str, ...] = (
    "reaction",
    "main_events",
    "check_events",
)

CLI_ARGUMENTS: tuple[tuple[tuple[str, ...], dict[str, Any]], ...] = (
    (
        ("--system",),
        {
            "type": str,
            "required": True,
            "help": "System name under runs/sweep/<system>.",
        },
    ),
    (
        ("--kpush",),
        {
            "type": str,
            "default": DEFAULT_KPUSHES,
            "help": (
                "Push constant list (comma-separated, default: 0.03,0.035)."
            ),
        },
    ),
    (
        ("--main",),
        {
            "type": str,
            "default": DEFAULT_MAIN_GROUPS,
            "help": "Main replicate groups range (default: 1-4).",
        },
    ),
    (
        ("--check",),
        {
            "type": str,
            "default": DEFAULT_CHECK_GROUPS,
            "help": "Check replicate groups range (default: 5-6).",
        },
    ),
    (
        ("--csv",),
        {
            "type": Path,
            "default": None,
            "help": (
                "Optional path to output CSV "
                "(default: data/<system>/checks/...)."
            ),
        },
    ),
)


def parse_group_range(range_expr: str) -> list[int]:
    """
    Parse a group range or comma-separated string into integer list.

    :param range_expr: Expression such as '1-4' or '1,2,3'.
    :type range_expr: str
    :return: List of individual integer group identifiers.
    :rtype: list[int]
    """
    if "-" in range_expr:
        start_str, end_str = range_expr.split("-", maxsplit=1)
        return list(range(int(start_str), int(end_str) + 1))
    return [int(token.strip()) for token in range_expr.split(",") if token]


def parse_kpush_list(kpush_expr: str) -> list[str]:
    """
    Parse comma-separated push constants into cleaned string tokens.

    :param kpush_expr: Comma-separated push values such as '0.03,0.035'.
    :type kpush_expr: str
    :return: List of individual push strength string identifiers.
    :rtype: list[str]
    """
    return [token.strip() for token in kpush_expr.split(",") if token.strip()]


def load_group_reactions(
    system_dir: Path,
    kpush: str,
    groups: Sequence[int],
) -> tuple[Counter[str], dict[str, str]]:
    """
    Aggregate reaction event counts and species SMILES across groups.

    :param system_dir: Directory containing simulation sweep data.
    :type system_dir: Path
    :param kpush: Push constant identifier string.
    :type kpush: str
    :param groups: Replicate group indices to inspect.
    :type groups: Sequence[int]
    :return: Pair of reaction event counts and species key to SMILES map.
    :rtype: tuple[Counter[str], dict[str, str]]
    """
    events: Counter[str] = Counter()
    smiles_map: dict[str, str] = {}
    for group_idx in groups:
        cand_dir = (
            system_dir / f"T300_k{kpush}" / f"g{group_idx}" / "candidates"
        )
        if not (cand_dir / "manifest.json").is_file():
            continue
        cset = load_candidates(cand_dir)
        for key, spec in cset.species.items():
            smiles_map[key] = spec.smiles or key
        for cand in cset.candidates:
            events[candidate_identity(cand)] += cand.n_events
    return events, smiles_map


def format_reaction_label(
    identity: str,
    smiles_map: Mapping[str, str],
) -> str:
    """
    Convert candidate reaction identity string into human-readable SMILES.

    :param identity: Canonical identity string separated by ' <> '.
    :type identity: str
    :param smiles_map: Mapping from species keys to SMILES formulas.
    :type smiles_map: Mapping[str, str]
    :return: Labeled reaction string with SMILES formulas on both sides.
    :rtype: str
    """
    sides = [
        " + ".join(smiles_map.get(key, key) for key in side.split("|"))
        for side in identity.split(" <> ")
    ]
    return "  <>  ".join(sides)


def print_kpush_summary(
    kpush: str,
    main_counts: Mapping[str, int],
    check_counts: Mapping[str, int],
) -> None:
    """
    Print overlap statistics between main and check groups for one push.

    :param kpush: Push strength string label.
    :type kpush: str
    :param main_counts: Reaction event counts in main replicate groups.
    :type main_counts: Mapping[str, int]
    :param check_counts: Reaction event counts in check replicate groups.
    :type check_counts: Mapping[str, int]
    :return: None
    :rtype: None
    """
    both = set(main_counts) & set(check_counts)
    print(
        f"k={kpush}: main {len(main_counts)} reactions, "
        f"check {len(check_counts)}, shared {len(both)}; "
        f"check reactions also in main {len(both)}/{len(check_counts)}"
    )


def print_reaction_event_table(
    all_main: Counter[str],
    all_check: Counter[str],
    all_smiles: Mapping[str, str],
) -> None:
    """
    Print formatted table of event counts across all observed reactions.

    :param all_main: Cumulative event counts across all main groups.
    :type all_main: Counter[str]
    :param all_check: Cumulative event counts across all check groups.
    :type all_check: Counter[str]
    :param all_smiles: Mapping of species keys to SMILES identifiers.
    :type all_smiles: Mapping[str, str]
    :return: None
    :rtype: None
    """
    print("\nreactions (main events / check events):")
    all_reactions = sorted(
        set(all_main) | set(all_check),
        key=lambda item: -(all_main[item] + all_check[item]),
    )
    for rxn in all_reactions:
        label = format_reaction_label(rxn, all_smiles)
        print(f"{all_main[rxn]:4d} {all_check[rxn]:4d}  {label}")


def print_union_summary(
    all_main: Counter[str],
    all_check: Counter[str],
    all_smiles: Mapping[str, str],
) -> None:
    """
    Print overall union overlap metrics and per-reaction event summary table.

    :param all_main: Cumulative event counts across all main groups.
    :type all_main: Counter[str]
    :param all_check: Cumulative event counts across all check groups.
    :type all_check: Counter[str]
    :param all_smiles: Mapping of species keys to SMILES identifiers.
    :type all_smiles: Mapping[str, str]
    :return: None
    :rtype: None
    """
    both = set(all_main) & set(all_check)
    print(
        f"union: main {len(all_main)}, check {len(all_check)}, "
        f"shared {len(both)}; check also in main "
        f"{len(both)}/{len(all_check)}"
    )
    main_2 = [rxn for rxn, count in all_main.items() if count >= 2]
    check_overlap = sum(rxn in all_check for rxn in main_2)
    print(
        f"main reactions seen >=2 times: {len(main_2)}, "
        f"of which in check {check_overlap}"
    )
    print_reaction_event_table(all_main, all_check, all_smiles)


def write_comparison_csv(
    csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """
    Write reaction comparison records to a CSV file.

    :param csv_path: Path to target CSV file.
    :type csv_path: Path
    :param rows: Sequence of reaction comparison row dictionaries.
    :type rows: Sequence[Mapping[str, Any]]
    :return: None
    :rtype: None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """
    Construct command-line parser for replicate group comparison.

    :return: Configured argument parser instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Compare reaction counts between main and check groups."
    )
    for flags, options in CLI_ARGUMENTS:
        parser.add_argument(*flags, **options)
    return parser


def run_kpush_sweeps(
    system_dir: Path,
    kpushes: Sequence[str],
    main_groups: Sequence[int],
    check_groups: Sequence[int],
) -> tuple[Counter[str], Counter[str], dict[str, str]]:
    """
    Execute sweep candidate loading and summaries across push strengths.

    :param system_dir: System sweep base directory path.
    :type system_dir: Path
    :param kpushes: List of push strength identifiers.
    :type kpushes: Sequence[str]
    :param main_groups: Replicate group indices for main set.
    :type main_groups: Sequence[int]
    :param check_groups: Replicate group indices for check set.
    :type check_groups: Sequence[int]
    :return: Cumulative main events, check events, and species SMILES map.
    :rtype: tuple[Counter[str], Counter[str], dict[str, str]]
    """
    all_main: Counter[str] = Counter()
    all_check: Counter[str] = Counter()
    all_smiles: dict[str, str] = {}
    for kpush in kpushes:
        main_cnt, sm1 = load_group_reactions(system_dir, kpush, main_groups)
        check_cnt, sm2 = load_group_reactions(system_dir, kpush, check_groups)
        all_main.update(main_cnt)
        all_check.update(check_cnt)
        all_smiles.update(sm1)
        all_smiles.update(sm2)
        print_kpush_summary(kpush, main_cnt, check_cnt)
    return all_main, all_check, all_smiles


def build_comparison_rows(
    all_main: Counter[str],
    all_check: Counter[str],
    all_smiles: Mapping[str, str],
) -> list[dict[str, Any]]:
    """
    Build structured comparison rows for CSV export.

    :param all_main: Cumulative event counts across all main groups.
    :type all_main: Counter[str]
    :param all_check: Cumulative event counts across all check groups.
    :type all_check: Counter[str]
    :param all_smiles: Mapping of species keys to SMILES identifiers.
    :type all_smiles: Mapping[str, str]
    :return: List of row dictionaries with reaction and event counts.
    :rtype: list[dict[str, Any]]
    """
    all_rxns = sorted(
        set(all_main) | set(all_check),
        key=lambda item: -(all_main[item] + all_check[item]),
    )
    return [
        {
            "reaction": format_reaction_label(rxn, all_smiles),
            "main_events": all_main[rxn],
            "check_events": all_check[rxn],
        }
        for rxn in all_rxns
    ]


def export_comparison_results(
    system: str,
    custom_path: Path | None,
    all_main: Counter[str],
    all_check: Counter[str],
    all_smiles: Mapping[str, str],
) -> None:
    """
    Build rows and export group comparison results to CSV.

    :param system: Chemical system name.
    :type system: str
    :param custom_path: Optional explicit CSV path.
    :type custom_path: Path or None
    :param all_main: Cumulative event counts across all main groups.
    :type all_main: Counter[str]
    :param all_check: Cumulative event counts across all check groups.
    :type all_check: Counter[str]
    :param all_smiles: Mapping of species keys to SMILES identifiers.
    :type all_smiles: Mapping[str, str]
    :return: None
    :rtype: None
    """
    csv_path = custom_path or (
        Path("data") / system / "checks" / "group_comparison.csv"
    )
    rows = build_comparison_rows(all_main, all_check, all_smiles)
    write_comparison_csv(csv_path, rows)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute CLI entry point for group comparison check.

    :param argv: Optional command-line arguments list.
    :type argv: Sequence[str] or None
    :return: Process exit status code.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    system_dir = Path("runs/sweep") / args.system
    kpushes = parse_kpush_list(args.kpush)
    main_groups = parse_group_range(args.main)
    check_groups = parse_group_range(args.check)
    all_main, all_check, all_smiles = run_kpush_sweeps(
        system_dir, kpushes, main_groups, check_groups
    )
    print_union_summary(all_main, all_check, all_smiles)
    export_comparison_results(
        args.system, args.csv, all_main, all_check, all_smiles
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
