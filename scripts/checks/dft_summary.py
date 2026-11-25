"""
Summarize DFT comparison metrics across engines and systems (PLAN D23).

Reads data/<system>/checks/dft_compare.csv for each specified system,
aggregates barrier errors, reaction errors, DFT forces, IRC path status,
and path peak heights above saddle per engine in standard order.
Writes data/<system>/checks/dft_summary.csv and prints one markdown
table per system.
"""

import argparse
import csv
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dft_single_points import DEFAULT_ENGINES, DEFAULT_SYSTEMS

__all__ = [
    "SUMMARY_CSV_FIELDNAMES",
    "TABLE_HEADERS",
    "barrier_metrics",
    "build_parser",
    "calculate_engine_summary",
    "dft_metrics",
    "filter_numbers",
    "format_table_cell",
    "main",
    "mean_absolute",
    "mean_signed",
    "median_value",
    "parse_optional_float",
    "read_compare_rows",
    "render_markdown_table",
    "summarize_system",
    "write_summary_csv",
]

SUMMARY_CSV_FIELDNAMES: tuple[str, ...] = (
    "engine",
    "rows",
    "barrier_error_n",
    "barrier_error_mae",
    "barrier_error_mean",
    "barrier_error_min",
    "barrier_error_max",
    "reaction_error_mae",
    "dft_fmax_median",
    "dft_fmax_max",
    "path_status_ok",
    "peak_above_saddle_median",
    "peak_above_saddle_max",
)

TABLE_HEADERS: tuple[str, ...] = (
    "engine",
    "rows",
    "barrier n",
    "mean abs",
    "mean signed",
    "min",
    "max",
    "reaction mae",
    "fmax median",
    "fmax max",
    "path ok",
    "peak median",
    "peak max",
)


def parse_optional_float(value: Any) -> float | None:
    """
    Parse a numeric string or return None for empty cells.

    :param value: Value to parse.
    :type value: Any
    :return: Float value or None.
    :rtype: float or None
    """
    if value is None:
        return None
    text = str(value).strip()
    return float(text) if text else None


def filter_numbers(
    rows: Sequence[Mapping[str, str]], key: str
) -> list[float]:
    """
    Extract non-blank float values for a given column.

    :param rows: Input CSV row dictionaries.
    :type rows: Sequence[Mapping[str, str]]
    :param key: Column name to extract.
    :type key: str
    :return: List of parsed float values.
    :rtype: list[float]
    """
    values = [parse_optional_float(row.get(key)) for row in rows]
    return [val for val in values if val is not None]


def mean_absolute(values: Sequence[float]) -> float | None:
    """
    Compute mean absolute value rounded to 2 decimals.

    :param values: Sequence of float values.
    :type values: Sequence[float]
    :return: Mean absolute value or None when empty.
    :rtype: float or None
    """
    if not values:
        return None
    return round(sum(abs(x) for x in values) / len(values), 2)


def mean_signed(values: Sequence[float]) -> float | None:
    """
    Compute mean signed value rounded to 2 decimals.

    :param values: Sequence of float values.
    :type values: Sequence[float]
    :return: Mean signed value or None when empty.
    :rtype: float or None
    """
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def median_value(values: Sequence[float]) -> float | None:
    """
    Compute median value rounded to 2 decimals.

    :param values: Sequence of float values.
    :type values: Sequence[float]
    :return: Median value or None when empty.
    :rtype: float or None
    """
    if not values:
        return None
    return round(statistics.median(values), 2)


def barrier_metrics(barriers: Sequence[float]) -> dict[str, Any]:
    """
    Compute aggregated barrier error statistics.

    :param barriers: Extracted barrier error numbers.
    :type barriers: Sequence[float]
    :return: Dictionary with count, MAE, mean, min, and max.
    :rtype: dict[str, Any]
    """
    return {
        "barrier_error_n": len(barriers),
        "barrier_error_mae": mean_absolute(barriers),
        "barrier_error_mean": mean_signed(barriers),
        "barrier_error_min": round(min(barriers), 2) if barriers else None,
        "barrier_error_max": round(max(barriers), 2) if barriers else None,
    }


def dft_metrics(
    fmaxs: Sequence[float], peaks: Sequence[float]
) -> dict[str, Any]:
    """
    Compute DFT force and peak statistics.

    :param fmaxs: Extracted DFT fmax values.
    :type fmaxs: Sequence[float]
    :param peaks: Extracted peak-above-saddle values.
    :type peaks: Sequence[float]
    :return: Dictionary with median and max metrics.
    :rtype: dict[str, Any]
    """
    return {
        "dft_fmax_median": median_value(fmaxs),
        "dft_fmax_max": round(max(fmaxs), 2) if fmaxs else None,
        "peak_above_saddle_median": median_value(peaks),
        "peak_above_saddle_max": round(max(peaks), 2) if peaks else None,
    }


def calculate_engine_summary(
    engine: str, rows: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    """
    Aggregate DFT comparison statistics for one engine.

    :param engine: Engine identifier.
    :type engine: str
    :param rows: Filtered rows for this engine.
    :type rows: Sequence[Mapping[str, str]]
    :return: Summary metrics dictionary.
    :rtype: dict[str, Any]
    """
    barriers = filter_numbers(rows, "barrier_error")
    reactions = filter_numbers(rows, "reaction_error")
    fmaxs = filter_numbers(rows, "dft_fmax_ev_a")
    peaks = filter_numbers(rows, "peak_above_saddle")
    path_ok = sum(1 for row in rows if row.get("path_status") == "ok")
    summary = {"engine": engine, "rows": len(rows)}
    summary.update(barrier_metrics(barriers))
    summary["reaction_error_mae"] = mean_absolute(reactions)
    summary.update(dft_metrics(fmaxs, peaks))
    summary["path_status_ok"] = path_ok
    return summary


def format_table_cell(value: Any) -> str:
    """
    Format a value for markdown table display.

    :param value: Value to format.
    :type value: Any
    :return: Formatted string representation.
    :rtype: str
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_markdown_table(
    system: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    """
    Render engine summary rows as a markdown table.

    :param system: System name.
    :type system: str
    :param rows: Summary row dictionaries.
    :type rows: Sequence[Mapping[str, Any]]
    :return: Formatted markdown table string.
    :rtype: str
    """
    keys = list(SUMMARY_CSV_FIELDNAMES)
    header = "| " + " | ".join(TABLE_HEADERS) + " |"
    divider = "| " + " | ".join(["---"] * len(TABLE_HEADERS)) + " |"
    body = [
        "| "
        + " | ".join(format_table_cell(row.get(key)) for key in keys)
        + " |"
        for row in rows
    ]
    return f"### {system}\n\n" + "\n".join([header, divider, *body])


def write_summary_csv(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """
    Write summary rows to a CSV file.

    :param path: Output CSV path.
    :type path: Path
    :param rows: Summary rows to write.
    :type rows: Sequence[Mapping[str, Any]]
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(SUMMARY_CSV_FIELDNAMES), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            formatted = {
                key: (
                    round(value, 2)
                    if isinstance(value, float)
                    else ("" if value is None else value)
                )
                for key, value in row.items()
            }
            writer.writerow(formatted)


def read_compare_rows(path: Path) -> list[dict[str, str]]:
    """
    Read comparison rows from a dft_compare.csv file.

    :param path: Path to dft_compare.csv.
    :type path: Path
    :return: Rows as a list of dictionaries.
    :rtype: list[dict[str, str]]
    """
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def summarize_system(
    compare_rows: Sequence[Mapping[str, str]],
    engines: Sequence[str],
) -> list[dict[str, Any]]:
    """
    Build summary rows for each engine in standard order.

    :param compare_rows: All comparison rows for the system.
    :type compare_rows: Sequence[Mapping[str, str]]
    :param engines: Engines in required order.
    :type engines: Sequence[str]
    :return: Summary row per engine.
    :rtype: list[dict[str, Any]]
    """
    summary: list[dict[str, Any]] = []
    for engine in engines:
        matching = [
            row for row in compare_rows if row.get("engine") == engine
        ]
        summary.append(calculate_engine_summary(engine, matching))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    :return: Configured argument parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="System name (can be repeated).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        dest="systems_list",
        help="One or more system names.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
        help="Engine name (can be repeated).",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        dest="engines_list",
        help="One or more engine names.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Summarize DFT comparison CSVs and write summary tables.

    :param argv: Optional command-line argument list.
    :type argv: Sequence[str] or None
    :return: Exit status code (0).
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    systems = args.systems_list or args.systems or list(DEFAULT_SYSTEMS)
    engines = args.engines_list or args.engines or list(DEFAULT_ENGINES)
    for system in systems:
        compare_path = args.data_dir / system / "checks" / "dft_compare.csv"
        rows = read_compare_rows(compare_path)
        if not rows:
            continue
        summary_rows = summarize_system(rows, engines)
        out_path = args.data_dir / system / "checks" / "dft_summary.csv"
        write_summary_csv(out_path, summary_rows)
        print(render_markdown_table(system, summary_rows))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
