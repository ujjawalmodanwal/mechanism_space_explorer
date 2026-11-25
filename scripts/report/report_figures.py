"""
Draw the figures and the number table for the present-study report.

Reads the frozen candidate sets, the refinement records and the
per-reaction summaries, then writes vector PDFs and one LaTeX table to
reports/present_study/. Run from the repository root:

    .venv/bin/python scripts/report/report_figures.py

Colour follows the model family (blue: semi-empirical, orange: OMol25,
aqua: other ML) and marker shape follows the model, because more than
three hues cannot be told apart when markers share a row.
"""

import csv
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
from matplotlib.path import Path as CurvePath
from matplotlib.ticker import ScalarFormatter

OUT = Path("reports/present_study")
FIGURES = OUT / "figures"
GENERATED = OUT / "generated"
WIDTH_IN = 6.5

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SYSTEMS = (
    ("hcooh", "HCOOH"),
    ("hcooh_h2o", "HCOOH + H$_2$O"),
    ("hcho_nh3", "HCHO + NH$_3$"),
    ("cpd_ma", "CPD + MA"),
)
ENGINES = (
    ("xtb", "GFN2-xTB", BLUE, "o"),
    ("uma", "UMA", ORANGE, "s"),
    ("mace", "MACE-OMol25", ORANGE, "^"),
    ("aimnet", "AIMNet2-NSE", AQUA, "P"),
    ("ani1xnr", "ANI-1xnr", AQUA, "X"),
)
OUTCOMES = (
    ("confirmed", "confirmed saddle", {"color": "#3a3a37"}),
    ("barrierless", "no barrier", {"color": "#9a9893"}),
    ("elsewhere", "saddle leads elsewhere", {"color": WASH, "hatch": "//////"}),
    ("failed", "search failed", {"color": WASH, "hatch": "\\\\\\\\\\\\"}),
    ("radical", "radical: not computable", {"color": "#ffffff"}),
)
STATUS_OUTCOME = {
    "confirmed": "confirmed",
    "barrierless": "barrierless",
    "endpoint_mismatch": "elsewhere",
    "unsupported": "radical",
}
REACTION_ERRORS = (
    ("hcooh", "R001", r"HCOOH $\to$ CO$_2$ + H$_2$"),
    ("hcooh", "R002", r"HCOOH $\to$ CO + H$_2$O"),
    ("hcho_nh3", "R006", r"HCHO $\to$ CO + H$_2$"),
    ("cpd_ma", "R001", r"CPD + MA $\to$ one-bond adduct"),
)
BARRIERS = (
    ("hcooh", "R001", "1000.0", "HCOOH → CO$_2$ + H$_2$",
     r"HCOOH $\to$ CO$_2$ + H$_2$", "1000", "dehydration faster"),
    ("hcooh", "R002", "1000.0", "HCOOH → CO + H$_2$O",
     r"HCOOH $\to$ CO + H$_2$O", "1000", "dehydration faster"),
    ("hcho_nh3", "R004", "298.15", "HCHO + NH$_3$ → NH$_2$CH$_2$OH",
     r"HCHO + NH$_3$ $\to$ NH$_2$CH$_2$OH", "298", "31.4; about 34"),
    ("hcho_nh3", "R013", "298.15", "NH$_2$CH$_2$OH → CH$_2$NH + H$_2$O",
     r"NH$_2$CH$_2$OH $\to$ CH$_2$NH + H$_2$O", "298", "55.0--55.9"),
    ("hcho_nh3", "R006", "298.15", "HCHO → CO + H$_2$",
     r"HCHO $\to$ CO + H$_2$", "298", "--"),
    ("cpd_ma", "R038", "298.15", "Diels\u2013Alder, endo",
     r"Diels--Alder, endo", "298", "endo faster"),
    ("cpd_ma", "R008", "298.15", "Diels\u2013Alder, exo",
     r"Diels--Alder, exo", "298", "exo $-$ endo = 2.2"),
)
LITERATURE_RANGES = {
    ("hcho_nh3", "R004"): (31.4, 34.0),
    ("hcho_nh3", "R013"): (55.0, 55.9),
}
NETWORK_SYSTEMS = (
    ("hcooh_h2o", "(a) HCOOH + H$_2$O"),
    ("hcho_nh3", "(b) HCHO + NH$_3$"),
)
SPECIES_LABELS = {
    "CH2O2-2a71d22e": "HCOOH", "H2O-2ebfe80f": "H$_2$O",
    "CO2-9d2ea55c": "CO$_2$", "H2-212475fb": "H$_2$",
    "CO-72c368ed": "CO", "CH2O3-e08c66ff": "H$_2$CO$_3$",
    "CH4O3-1b786c83": "HC(OH)$_3$", "CHO2-f92468da": "HOCO\u2022",
    "CHO2-063d45b9": "HCOO\u2022", "H-34ba85c0": "H\u2022",
    "CH3O3-f0971681": "C(OH)$_3$\u2022", "CH2O-021b6208": "HCHO",
    "H3N-1414a4ee": "NH$_3$", "CH2O-63452dc2": "HCOH",
    "CH3N-109a8232": "CH$_2$NH", "CH3NO-3bd83c2e": "HCONH$_2$",
    "CH3NO-e12e7f58": "HN=CHOH", "CH4NO-e9e98c33": "HOCH$_2$NH\u2022",
    "CH5NO-cfd9fd04": "NH$_2$CH$_2$OH", "CHNO-2ebc5e12": "HOCN",
    "CHO-5e7af12d": "HCO\u2022", "H2N-7076b15f": "NH$_2$\u2022",
}
COLUMN_DX, ROW_DY, NODE_R = 1.7, 0.62, 0.075


def apply_style() -> None:
    """
    Set one quiet, print-ready style for every figure.

    :return: None.
    :rtype: None
    """
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
        "mathtext.sf": "Arial",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.labelsize": 8,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "xtick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
    })


def tidy_axes(ax: plt.Axes) -> None:
    """
    Keep only the bottom axis line and a hairline vertical grid.

    :param ax: Axes to tidy.
    :type ax: plt.Axes
    :return: None.
    :rtype: None
    """
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def latest_rows(path: Path) -> dict[str, dict]:
    """
    Read a JSON-lines record file, keeping the last row per reaction.

    :param path: reactions.jsonl of one engine and system.
    :type path: Path
    :return: Rows keyed by reaction id.
    :rtype: dict[str, dict]
    """
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["reaction_id"]] = row
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    Read all rows from a CSV file, or nothing when the file is missing.

    :param path: CSV file path.
    :type path: Path
    :return: List of row dictionaries.
    :rtype: list[dict[str, str]]
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def optional_float(
    row: dict[str, str] | None, column: str
) -> float | None:
    """
    Return one numeric cell from a CSV row, or None when missing or empty.

    :param row: Row dictionary, or None.
    :type row: dict[str, str] or None
    :param column: Column name.
    :type column: str
    :return: Parsed float value, or None.
    :rtype: float or None
    """
    if row is None:
        return None
    cell = row.get(column)
    if cell is None or not cell.strip():
        return None
    return float(cell)


def summary_rows(system: str, engine: str) -> dict[tuple[str, str], dict]:
    """
    Read one engine's per-reaction summary, keyed by reaction and T.

    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Rows keyed by (reaction id, temperature label).
    :rtype: dict[tuple[str, str], dict]
    """
    path = Path("data") / system / "checks" / f"refine_{engine}.csv"
    return {
        (row["reaction_id"], row["temperature_k"]): row
        for row in read_csv_rows(path)
    }


def dft_rows(system: str) -> dict[tuple[str, str], dict]:
    """
    Read one system's DFT comparison table, keyed by engine and reaction.

    :param system: System name.
    :type system: str
    :return: Rows keyed by (engine, reaction id).
    :rtype: dict[tuple[str, str], dict]
    """
    path = Path("data") / system / "checks" / "dft_compare.csv"
    return {
        (row["engine"], row["reaction_id"]): row
        for row in read_csv_rows(path)
    }


def dft_value(
    system: str, engine: str, reaction_id: str, column: str
) -> float | None:
    """
    Return one DFT-comparison number, or None when missing or empty.

    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :param reaction_id: Reaction id.
    :type reaction_id: str
    :param column: CSV column name.
    :type column: str
    :return: The value, or None.
    :rtype: float or None
    """
    row = dft_rows(system).get((engine, reaction_id))
    return optional_float(row, column)


def dft_summary_rows(system: str) -> dict[str, dict]:
    """
    Read one system's per-engine DFT summary, keyed by engine.

    :param system: System name.
    :type system: str
    :return: Summary rows keyed by engine name.
    :rtype: dict[str, dict]
    """
    path = Path("data") / system / "checks" / "dft_summary.csv"
    return {row["engine"]: row for row in read_csv_rows(path)}


def summary_value(
    system: str, engine: str, reaction: str, temperature: str, field: str
) -> float | None:
    """
    Return one summary number, or None when the step has no saddle.

    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :param reaction: Reaction id.
    :type reaction: str
    :param temperature: Temperature label as written in the CSV.
    :type temperature: str
    :param field: CSV column.
    :type field: str
    :return: The value, or None.
    :rtype: float or None
    """
    row = summary_rows(system, engine).get((reaction, temperature))
    if row is None or not row[field]:
        return None
    if field.startswith("dg_act") and row["status"] != "confirmed":
        return None
    return float(row[field])


def outcome_counts(system: str, engine: str) -> Counter[str] | None:
    """
    Count refinement outcomes for one engine on one system.

    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Outcome counts, or None when the engine has no results.
    :rtype: Counter[str] or None
    """
    path = Path("results/refine") / system / engine / "reactions.jsonl"
    if not path.exists():
        return None
    statuses = (row["status"] for row in latest_rows(path).values())
    return Counter(STATUS_OUTCOME.get(status, "failed") for status in statuses)


def freeze_stats(system: str) -> dict[str, int]:
    """
    Read the freeze statistics and the number of frozen reactions.

    :param system: System name.
    :type system: str
    :return: Events, rejections, kept events and reactions.
    :rtype: dict[str, int]
    """
    path = Path("candidates") / system / "candidates.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data["stats"]
    radicals = stats.get("rejected_multiple_radicals", 0)
    changed = stats.get("rejected_state_mismatch", 0)
    by_hand = sum(1 for c in data["candidates"] if c["source"] != "md")
    return {
        "events": stats["events"],
        "radicals": radicals,
        "changed": changed,
        "kept": stats["events"] - radicals - changed,
        "reactions": len(data["candidates"]),
        "by_hand": by_hand,
    }


def save(fig: plt.Figure, name: str) -> None:
    """
    Write one figure as a vector PDF.

    :param fig: Figure to write.
    :type fig: plt.Figure
    :param name: File stem.
    :type name: str
    :return: None.
    :rtype: None
    """
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


def draw_segment(
    ax: plt.Axes, row: float, left: float, width: float, style: dict
) -> None:
    """
    Draw one stacked-bar segment with a surface gap around it.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param row: Bar position.
    :type row: float
    :param left: Segment start.
    :type left: float
    :param width: Segment length.
    :type width: float
    :param style: Face colour and optional hatch.
    :type style: dict
    :return: None.
    :rtype: None
    """
    edge = AXIS if style["color"] == "#ffffff" else "white"
    ax.barh(
        row, width, left=left, height=0.64, color=style["color"],
        edgecolor=edge, linewidth=0.8, hatch=style.get("hatch"),
        hatchcolor=INK2,
    )


def legend_patch(style: dict) -> Patch:
    """
    Build a legend swatch that keeps its outline and hatch.

    :param style: Face colour and optional hatch.
    :type style: dict
    :return: Legend patch.
    :rtype: Patch
    """
    return Patch(facecolor=style["color"], edgecolor=AXIS, linewidth=0.5,
                 hatch=style.get("hatch"), hatchcolor=INK2)


def figure_discovery() -> None:
    """
    Figure 2: MD events kept and rejected by freeze, per system.

    :return: None.
    :rtype: None
    """
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.85), layout="constrained")
    parts = (
        ("kept", "kept: becomes a reaction", {"color": "#3a3a37"}),
        ("radicals", "rejected: two radicals",
         {"color": WASH, "hatch": "//////"}),
        ("changed", "rejected: changed when relaxed",
         {"color": WASH, "hatch": "\\\\\\\\\\\\"}),
    )
    for row, (system, _) in enumerate(SYSTEMS):
        stats, left = freeze_stats(system), 0
        for key, _, style in parts:
            draw_segment(ax, row, left, stats[key], style)
            left += stats[key]
        hand = f", {stats['by_hand']} added by hand" if stats["by_hand"] else ""
        note = f"{stats['events']} events → {stats['reactions']} reactions"
        ax.text(left + 8, row, note + hand, va="center", color=INK2,
                fontsize=7.5)
    ax.set_yticks(range(len(SYSTEMS)), [label for _, label in SYSTEMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1000)
    ax.set_xticks(range(0, 701, 100))
    ax.set_xlabel("MD events (bond changes that lasted 50 fs)")
    tidy_axes(ax)
    handles = [legend_patch(style) for _, _, style in parts]
    fig.legend(handles, [name for _, name, _ in parts], ncol=3,
               loc="outside upper left", handlelength=1.6)
    save(fig, "fig2_discovery")


def figure_outcomes() -> None:
    """
    Figure 3: how each engine's transition-state searches ended.

    :return: None.
    :rtype: None
    """
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 3.35),
                             layout="constrained", sharey=True)
    for ax, (system, label) in zip(axes.flat, SYSTEMS, strict=True):
        total = freeze_stats(system)["reactions"]
        for row, (engine, _, _, _) in enumerate(ENGINES):
            counts, left = outcome_counts(system, engine), 0
            if counts is None:
                ax.text(0.2, row, "running", va="center", color=MUTED,
                        style="italic")
                continue
            for key, _, style in OUTCOMES:
                if counts[key]:
                    draw_segment(ax, row, left, counts[key], style)
                    left += counts[key]
            ax.text(total * 1.02, row, f"{counts['confirmed']}/{total}",
                    va="center", color=INK2, fontsize=7.5)
        ax.set_title(f"{label} ({total} reactions)", fontsize=8, loc="left",
                     color=INK)
        ax.set_xlim(0, total * 1.13)
        tidy_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("Reactions")
    axes[0, 0].set_yticks(range(len(ENGINES)), [n for _, n, _, _ in ENGINES])
    axes[0, 0].invert_yaxis()
    handles = [legend_patch(style) for _, _, style in OUTCOMES]
    fig.legend(handles, [name for _, name, _ in OUTCOMES], ncol=5,
               loc="outside upper left", handlelength=1.6)
    save(fig, "fig3_outcomes")


def engine_handles() -> list[Line2D]:
    """
    Build legend markers for the five engines.

    :return: One marker handle per engine.
    :rtype: list[Line2D]
    """
    return [
        Line2D([], [], linestyle="none", marker=marker, color=color,
               markeredgecolor="white", markeredgewidth=0.6, markersize=6,
               label=name)
        for _, name, color, marker in ENGINES
    ]


def dodge(index: int) -> float:
    """
    Vertical offset that gives each engine its own lane in a row.

    :param index: Engine position.
    :type index: int
    :return: Offset in row units.
    :rtype: float
    """
    return (index - (len(ENGINES) - 1) / 2) * 0.13


def plot_engine_points(
    ax: plt.Axes,
    row: int,
    values: Sequence[float | None],
    dft: Sequence[float | None] | None = None,
) -> None:
    """
    Plot one row of engine markers, one lane per engine.

    When both a model value and a DFT value are given for an engine, draw a
    connecting line and a hollow DFT marker on the same lane.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param row: Row index.
    :type row: int
    :param values: One value per engine, None when missing.
    :type values: Sequence[float or None]
    :param dft: Optional DFT values per engine, None when omitted.
    :type dft: Sequence[float or None] or None
    :return: None.
    :rtype: None
    """
    dft_values = [None] * len(ENGINES) if dft is None else dft
    for index, ((_, _, color, marker), value, dft_val) in enumerate(
        zip(ENGINES, values, dft_values, strict=True)
    ):
        lane = row + dodge(index)
        if value is not None:
            ax.plot(
                value,
                lane,
                linestyle="none",
                marker=marker,
                color=color,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
        if value is not None and dft_val is not None:
            ax.plot(
                [value, dft_val],
                [lane, lane],
                color=color,
                linewidth=0.8,
                alpha=0.6,
                zorder=2,
            )
            ax.plot(
                dft_val,
                lane,
                linestyle="none",
                marker=marker,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=5.5,
                zorder=3,
            )


def row_labels(ax: plt.Axes, labels: Sequence[str]) -> None:
    """
    Label rows top to bottom and draw a hairline between them.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param labels: Row labels.
    :type labels: Sequence[str]
    :return: None.
    :rtype: None
    """
    ax.set_yticks(range(len(labels)), labels)
    for edge in range(1, len(labels)):
        ax.axhline(edge - 0.5, color=GRID, linewidth=0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)


def dft_handle() -> Line2D:
    """
    Build the legend marker representing DFT single-point values.

    :return: Hollow circle legend handle.
    :rtype: Line2D
    """
    return Line2D(
        [],
        [],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=INK2,
        markeredgewidth=1.0,
        markersize=5.5,
        linestyle="none",
        label="open: DFT on the model's shapes",
    )


def draw_error_panel(
    ax: plt.Axes,
    tables: dict[str, dict[tuple[str, str], dict]],
    column: str,
    title: str,
) -> None:
    """
    Draw one panel of sorted, jittered model-minus-DFT errors per engine.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param tables: Preloaded DFT comparison tables keyed by system name.
    :type tables: dict[str, dict[tuple[str, str], dict]]
    :param column: Error column to plot.
    :type column: str
    :param title: Panel title.
    :type title: str
    :return: None.
    :rtype: None
    """
    ax.axvspan(-1, 1, color=WASH, zorder=0)
    ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
    for row, (engine, _, color, marker) in enumerate(ENGINES):
        raw = [
            val
            for system, _ in SYSTEMS
            for (row_engine, _), record in tables[system].items()
            if row_engine == engine
            and (val := optional_float(record, column)) is not None
        ]
        values = sorted(raw)
        if values:
            ys = [row + ((idx % 5) - 2) * 0.09 for idx in range(len(values))]
            ax.plot(
                values,
                ys,
                linestyle="none",
                marker=marker,
                color=color,
                markersize=4.5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )
        ax.text(
            1.01,
            row,
            f"n = {len(values)}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            color=MUTED,
            fontsize=6.5,
        )
    row_labels(ax, [name for _, name, _, _ in ENGINES])
    ticks = [-30, -3, -1, 0, 1, 3, 30]
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-80, 80)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Model minus DFT (kcal/mol)")
    ax.set_title(title, loc="left", fontsize=8)
    tidy_axes(ax)


def figure_dft_errors() -> None:
    """
    Figure 4: model minus DFT errors on barriers and reaction energies.

    :return: None.
    :rtype: None
    """
    tables = {system: dft_rows(system) for system, _ in SYSTEMS}
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(WIDTH_IN, 2.4),
        layout="constrained",
        sharey=True,
    )
    draw_error_panel(left, tables, "barrier_error", "(a) Barriers")
    draw_error_panel(right, tables, "reaction_error", "(b) Reaction energies")
    fig.legend(handles=engine_handles(), ncol=5, loc="outside upper left")
    save(fig, "fig4_dft")


def draw_literature(ax: plt.Axes, key: tuple[str, str], row: int) -> None:
    """
    Draw literature guide values for one barrier row.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param key: (system, reaction id).
    :type key: tuple[str, str]
    :param row: Row index.
    :type row: int
    :return: None.
    :rtype: None
    """
    if key in LITERATURE_RANGES:
        low, high = LITERATURE_RANGES[key]
        ax.plot([low, high], [row + 0.4, row + 0.4], color=INK,
                linewidth=2.2, zorder=4, solid_capstyle="butt")


def figure_barriers() -> None:
    """
    Figure 5: forward free-energy barriers per engine.

    :return: None.
    :rtype: None
    """
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0), layout="constrained")
    labels = []
    for row, (system, rid, temp, label, _, t_label, _) in enumerate(
        BARRIERS
    ):
        values = [
            summary_value(system, engine, rid, temp, "dg_act_fwd_kcal")
            for engine, _, _, _ in ENGINES
        ]
        dft_values = [
            dft_value(system, engine, rid, f"dg_barrier_dft_{temp}")
            for engine, _, _, _ in ENGINES
        ]
        plot_engine_points(ax, row, values, dft=dft_values)
        draw_literature(ax, (system, rid), row)
        labels.append(f"{label}, {t_label} K")
        if (system, rid) == ("cpd_ma", "R038"):
            ax.text(
                123,
                row + 0.5,
                "ANI-1xnr: no saddle",
                ha="right",
                va="center",
                color=MUTED,
                fontsize=7,
                style="italic",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 1.0,
                },
            )
    row_labels(ax, labels)
    ax.set_xlim(0, 125)
    ax.set_xlabel(
        "Forward free-energy barrier (kcal/mol), from the "
        "separated reacting molecules at 1 atm"
    )
    tidy_axes(ax)
    literature = Line2D(
        [],
        [],
        color=INK,
        linewidth=2.2,
        solid_capstyle="butt",
        label="literature values (guide)",
    )
    fig.legend(
        handles=[*engine_handles(), literature, dft_handle()],
        ncol=4,
        loc="outside upper left",
        columnspacing=1.2,
    )
    save(fig, "fig5_barriers")


def channel_differences(
    pairs: tuple[tuple[str, str, str], tuple[str, str, str]],
    lookup: Callable[[str, str, str, str], float | None],
) -> list[float | None]:
    """
    Compute first-minus-second barrier differences per engine.

    :param pairs: Two (system, reaction id, temperature) keys.
    :type pairs: tuple[tuple[str, str, str], tuple[str, str, str]]
    :param lookup: Function returning a barrier for (system, engine, id, T).
    :type lookup: Callable[[str, str, str, str], float or None]
    :return: One difference per engine, None when either step is missing.
    :rtype: list[float or None]
    """
    differences: list[float | None] = []
    for engine, _, _, _ in ENGINES:
        first, second = (
            lookup(system, engine, rid, temp) for system, rid, temp in pairs
        )
        differences.append(
            None if first is None or second is None else first - second
        )
    return differences


def selectivity(
    pairs: tuple[tuple[str, str, str], tuple[str, str, str]],
) -> list[float | None]:
    """
    Barrier of the first step minus the second, per engine.

    :param pairs: Two (system, reaction id, temperature) keys.
    :type pairs: tuple
    :return: One difference per engine, None when a saddle is missing.
    :rtype: list[float or None]
    """
    return channel_differences(
        pairs,
        lambda system, engine, rid, temp: summary_value(
            system, engine, rid, temp, "dg_act_fwd_kcal"
        ),
    )


def dft_selectivity(
    pairs: tuple[tuple[str, str, str], tuple[str, str, str]],
) -> list[float | None]:
    """
    DFT barrier of the first step minus the second, per engine.

    :param pairs: Two (system, reaction id, temperature) keys.
    :type pairs: tuple[tuple[str, str, str], tuple[str, str, str]]
    :return: One DFT difference per engine, None when missing.
    :rtype: list[float or None]
    """
    return channel_differences(
        pairs,
        lambda system, engine, rid, temp: dft_value(
            system, engine, rid, f"dg_barrier_dft_{temp}"
        ),
    )


def draw_selectivity(
    ax: plt.Axes,
    values: Sequence[float | None],
    missing: Sequence[str],
    dft: Sequence[float | None] | None = None,
) -> list[plt.Text]:
    """
    Draw one horizontal bar per engine with its value and optional DFT point.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param values: One difference per engine.
    :type values: Sequence[float or None]
    :param missing: Text shown when an engine has no value.
    :type missing: Sequence[str]
    :param dft: Optional DFT difference per engine.
    :type dft: Sequence[float or None] or None
    :return: Value annotations for checking whether limits need widening.
    :rtype: list[plt.Text]
    """
    dft_values = [None] * len(ENGINES) if dft is None else dft
    placed: list[plt.Text] = []
    for row, ((_, _, color, _), value, note, dft_val) in enumerate(
        zip(ENGINES, values, missing, dft_values, strict=True)
    ):
        if dft_val is not None:
            ax.plot(
                dft_val,
                row,
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markersize=5.5,
                zorder=4,
            )
        if value is None:
            ax.text(
                0.3,
                row,
                note,
                va="center",
                color=MUTED,
                fontsize=7,
                style="italic",
            )
            continue
        ax.barh(
            row,
            value,
            height=0.6,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        if dft_val is None:
            anchor = value
        elif value >= 0:
            anchor = max(value, dft_val)
        else:
            anchor = min(value, dft_val)
        pad = 0.4 if value >= 0 else -0.4
        near_marker = dft_val is not None and abs(anchor - dft_val) < 0.7
        shift = (4 if value >= 0 else -4) if near_marker else 0
        text_artist = ax.annotate(
            f"{value:+.1f}",
            xy=(anchor + pad, row),
            xytext=(shift, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            color=INK2,
            fontsize=7.5,
        )
        placed.append(text_artist)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(ENGINES)), [name for _, name, _, _ in ENGINES])
    ax.set_ylim(len(ENGINES) - 0.5, -0.5)
    tidy_axes(ax)
    return placed


def widen_selectivity_xlim(
    fig: plt.Figure,
    ax: plt.Axes,
    placed: Sequence[plt.Text],
    dft_values: Sequence[float | None],
) -> tuple[float, float]:
    """
    Widen x-limits just enough when any DFT marker or bar label overflows.

    :param fig: Parent figure.
    :type fig: plt.Figure
    :param ax: Panel axes with initial limits set.
    :type ax: plt.Axes
    :param placed: Value annotations for each drawn bar.
    :type placed: Sequence[plt.Text]
    :param dft_values: Plotted DFT values for the panel.
    :type dft_values: Sequence[float or None]
    :return: Final (xmin, xmax) limits applied to the axes.
    :rtype: tuple[float, float]
    """
    fig.canvas.draw()
    xmin, xmax = ax.get_xlim()
    inv = ax.transData.inverted()
    margin = (xmax - xmin) * 0.02
    for text_artist in placed:
        bbox = text_artist.get_window_extent()
        x0, _ = inv.transform((bbox.x0, bbox.y0))
        x1, _ = inv.transform((bbox.x1, bbox.y1))
        if x0 < xmin:
            xmin = min(xmin, x0 - margin)
        if x1 > xmax:
            xmax = max(xmax, x1 + margin)
    for dft_val in dft_values:
        if dft_val is not None:
            if dft_val < xmin + margin:
                xmin = dft_val - margin
            if dft_val > xmax - margin:
                xmax = dft_val + margin
    ax.set_xlim(xmin, xmax)
    return xmin, xmax


def figure_selectivity() -> None:
    """
    Figure 6: which channel each engine makes faster.

    :return: None.
    :rtype: None
    """
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(WIDTH_IN, 2.1), layout="constrained", sharey=True
    )
    hcooh_pairs = (
        ("hcooh", "R002", "1000.0"),
        ("hcooh", "R001", "1000.0"),
    )
    hcooh = selectivity(hcooh_pairs)
    hcooh_dft = dft_selectivity(hcooh_pairs)
    left_texts = draw_selectivity(
        left, hcooh, [""] * len(ENGINES), dft=hcooh_dft
    )
    left.set_xlim(-19, 14)
    left.set_xlabel("Dehydration minus decarboxylation barrier (kcal/mol)")
    left.set_title("(a) HCOOH at 1000 K", loc="left", fontsize=8)
    diels_pairs = (
        ("cpd_ma", "R008", "298.15"),
        ("cpd_ma", "R038", "298.15"),
    )
    diels = selectivity(diels_pairs)
    diels_dft = dft_selectivity(diels_pairs)
    right_texts = draw_selectivity(
        right, diels, ["", "", "", "", "no saddle found"], dft=diels_dft
    )
    right.axvline(2.2, color=INK, linewidth=0.9, linestyle=(0, (3, 2)))
    right.text(2.28, -0.42, "reported 2.2", fontsize=7, color=INK, va="bottom")
    right.set_xlim(-0.5, 3.4)
    right.set_xlabel("Exo minus endo barrier (kcal/mol)")
    right.set_title("(b) Diels\u2013Alder at 298 K", loc="left", fontsize=8)
    for ax in (left, right):
        ax.set_ylim(len(ENGINES) - 0.2, -0.6)
    fig.legend(
        handles=[dft_handle()], loc="outside upper right", frameon=False
    )
    left_min, left_max = widen_selectivity_xlim(
        fig, left, left_texts, hcooh_dft
    )
    _, right_max = widen_selectivity_xlim(
        fig, right, right_texts, diels_dft
    )
    left.text(
        left_min + 0.5,
        4.62,
        "← dehydration faster (literature)",
        fontsize=7,
        color=INK2,
        va="center",
    )
    left.text(
        left_max - 0.5,
        4.62,
        "decarboxylation faster →",
        fontsize=7,
        color=INK2,
        va="center",
        ha="right",
    )
    right.text(
        right_max - 0.05,
        4.62,
        "endo faster →",
        fontsize=7,
        color=INK2,
        va="center",
        ha="right",
    )
    save(fig, "fig6_selectivity")


def candidate_list(system: str) -> tuple[tuple[str, ...], list[dict]]:
    """
    Read the start state and the frozen reactions of one system.

    :param system: System name.
    :type system: str
    :return: Start state and candidate records.
    :rtype: tuple[tuple[str, ...], list[dict]]
    """
    path = Path("candidates") / system / "candidates.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(data["reactant_state"]), data["candidates"]


def species_ok(system: str, engine: str) -> set[str]:
    """
    Return the species keys that are true minima for one engine.

    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Keys whose latest status is ok.
    :rtype: set[str]
    """
    path = Path("results/refine") / system / engine / "species.jsonl"
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["key"]] = row["status"]
    return {key for key, status in rows.items() if status == "ok"}


def kept_reactions(system: str, engine: str) -> set[str]:
    """
    Reaction ids that survive into one engine's network.

    A step survives when its saddle is confirmed or it has no barrier, and
    every molecule it touches is a true minimum for that engine.

    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Surviving reaction ids.
    :rtype: set[str]
    """
    path = Path("results/refine") / system / engine / "reactions.jsonl"
    rows, ok = latest_rows(path), species_ok(system, engine)
    _, candidates = candidate_list(system)
    return {
        c["reaction_id"] for c in candidates
        if rows[c["reaction_id"]]["status"] in ("confirmed", "barrierless")
        and set(c["reactant_keys"]) | set(c["product_keys"]) <= ok
    }


def state_label(state: Sequence[str]) -> str:
    """
    Plain label for a whole-box state, e.g. "CO + 2H$_2$O".

    :param state: Species keys of the state.
    :type state: Sequence[str]
    :return: Label.
    :rtype: str
    """
    counts = Counter(state)
    parts = [
        (f"{n}" if n > 1 else "") + SPECIES_LABELS.get(key, key.split("-")[0])
        for key, n in counts.items()
    ]
    return " + ".join(parts)


def order_column(graph: nx.Graph, nodes: Sequence[tuple]) -> list[tuple]:
    """
    Order one column so that steps inside it join neighbouring nodes.

    :param graph: Undirected state graph.
    :type graph: nx.Graph
    :param nodes: Nodes of the column.
    :type nodes: Sequence[tuple]
    :return: Nodes from top to bottom.
    :rtype: list[tuple]
    """
    sub = graph.subgraph(nodes)
    parts = sorted(nx.connected_components(sub), key=lambda c: (-len(c),
                                                                min(c)))
    ordered: list[tuple] = []
    for part in parts:
        start = min(n for n in part if sub.degree(n) <= 1) if len(part) > 1 \
            else next(iter(part))
        ordered += list(nx.dfs_preorder_nodes(sub.subgraph(part), start))
    return ordered


def place_component(
    digraph: nx.DiGraph, nodes: set, top: float, layout: dict
) -> float:
    """
    Lay one detached component left to right, below the main one.

    Sources get their label on the left, the last level on the right and
    any middle level above, so neighbouring labels never meet.

    :param digraph: Directed state graph.
    :type digraph: nx.DiGraph
    :param nodes: Nodes of the component.
    :type nodes: set
    :param top: Highest y the component may use.
    :type top: float
    :param layout: Node to (x, y, label side), filled in place.
    :type layout: dict
    :return: Lowest y used.
    :rtype: float
    """
    levels = list(nx.topological_generations(digraph.subgraph(nodes)))
    rows = max(len(level) for level in levels)
    for depth, level in enumerate(levels):
        side = ("left" if depth == 0 else
                "right" if depth == len(levels) - 1 else "above")
        for index, node in enumerate(sorted(level)):
            offset = (rows - len(level)) / 2 + index
            layout[node] = (depth * COLUMN_DX, top - offset * 0.42, side)
    return top - (rows - 1) * 0.42


def network_layout(start: tuple, candidates: Sequence[dict]) -> dict:
    """
    Place the start on the left, its products in a column, and detached
    radical chains in rows below.

    :param start: Start state.
    :type start: tuple
    :param candidates: Frozen reactions.
    :type candidates: Sequence[dict]
    :return: Node to (x, y, label side).
    :rtype: dict
    """
    digraph = nx.DiGraph()
    for c in candidates:
        digraph.add_edge(tuple(c["state_a"]), tuple(c["state_b"]))
    graph = digraph.to_undirected()
    main = nx.node_connected_component(graph, start)
    depth = nx.single_source_shortest_path_length(graph.subgraph(main), start)
    layout = {start: (0.0, 0.0, "left")}
    for level in sorted(set(depth.values()) - {0}):
        column = order_column(graph, [n for n in main if depth[n] == level])
        for index, node in enumerate(column):
            layout[node] = (level * COLUMN_DX,
                            ((len(column) - 1) / 2 - index) * ROW_DY, "right")
    top = min(y for _, y, _ in layout.values()) - 0.95
    others = [c for c in nx.connected_components(graph) if start not in c]
    for part in sorted(others, key=min):
        top = place_component(digraph, part, top, layout) - 0.75
    return layout


def edge_curve(
    p0: np.ndarray, p2: np.ndarray, offset: float, bend: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Control points of a trimmed quadratic curve between two nodes.

    :param p0: Start node centre.
    :type p0: np.ndarray
    :param p2: End node centre.
    :type p2: np.ndarray
    :param offset: Sideways shift for parallel steps.
    :type offset: float
    :param bend: Sideways bulge as a share of the length.
    :type bend: float
    :return: Start, control and end points.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    d = p2 - p0
    normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    a, b = p0 + normal * offset, p2 + normal * offset
    ctrl = (a + b) / 2 + normal * bend * np.linalg.norm(d)
    a = a + (ctrl - a) / np.linalg.norm(ctrl - a) * NODE_R * 1.6
    b = b + (ctrl - b) / np.linalg.norm(ctrl - b) * NODE_R * 1.6
    return a, ctrl, b


def draw_badges(ax: plt.Axes, centre: np.ndarray, kept: Sequence[bool]) -> None:
    """
    Draw one small square per engine on a step: filled when kept.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param centre: Where the row of squares sits.
    :type centre: np.ndarray
    :param kept: One flag per engine, in ENGINES order.
    :type kept: Sequence[bool]
    :return: None.
    :rtype: None
    """
    side, pitch = 0.075, 0.095
    width = pitch * (len(ENGINES) - 1) + side
    ax.add_patch(Rectangle(centre - np.array([width / 2 + 0.02,
                                              side / 2 + 0.02]),
                           width + 0.04, side + 0.04, facecolor="white",
                           edgecolor="none", zorder=3.5))
    for index, ((_, _, color, _), flag) in enumerate(zip(ENGINES, kept,
                                                         strict=True)):
        corner = centre + np.array([(index - 2) * pitch - side / 2,
                                    -side / 2])
        ax.add_patch(Rectangle(corner, side, side, zorder=4, linewidth=0.5,
                               facecolor=color if flag else "white",
                               edgecolor=color if flag else AXIS))


def badge_spot(points: tuple, taken: list[np.ndarray]) -> np.ndarray:
    """
    Pick a point along a curve where a row of squares fits freely.

    :param points: Start, control and end points of the curve.
    :type points: tuple
    :param taken: Centres of rows already drawn; the new one is added.
    :type taken: list[np.ndarray]
    :return: Centre for the new row.
    :rtype: np.ndarray
    """
    a, ctrl, b = points
    half_w, half_h = 0.3, 0.07
    spot = None
    for s in (0.5, 0.38, 0.62, 0.28, 0.72, 0.2, 0.8):
        spot = (1 - s) ** 2 * a + 2 * (1 - s) * s * ctrl + s**2 * b
        if all(abs(spot[0] - c[0]) > 2 * half_w or
               abs(spot[1] - c[1]) > 2 * half_h for c in taken):
            break
    taken.append(spot)
    return spot


def draw_step(
    ax: plt.Axes, points: tuple, taken: list[np.ndarray], helper: bool,
    kept: Sequence[bool]
) -> None:
    """
    Draw one step as an arrow with its row of engine squares.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param points: Start, control and end points of the curve.
    :type points: tuple
    :param taken: Centres of square rows already drawn in this panel.
    :type taken: list[np.ndarray]
    :param helper: Whether a helper molecule takes part (dashed line).
    :type helper: bool
    :param kept: One flag per engine.
    :type kept: Sequence[bool]
    :return: None.
    :rtype: None
    """
    a, ctrl, b = points
    path = CurvePath([a, ctrl, b], [CurvePath.MOVETO, CurvePath.CURVE3,
                                    CurvePath.CURVE3])
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>,head_length=3.2,head_width=1.6",
        color=INK2, linewidth=0.7, zorder=2,
        linestyle=(0, (2.5, 1.5)) if helper else "-"))
    draw_badges(ax, badge_spot(points, taken), kept)


def draw_node(ax: plt.Axes, node: tuple, place: tuple, is_start: bool) -> None:
    """
    Draw one state and its label on the side the layout chose.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param node: State.
    :type node: tuple
    :param place: (x, y, label side).
    :type place: tuple
    :param is_start: Whether this is the start state.
    :type is_start: bool
    :return: None.
    :rtype: None
    """
    x, y, side = place
    dark = "#3a3a37"
    ax.add_patch(Circle((x, y), NODE_R, zorder=5, linewidth=0.8,
                        facecolor=dark if is_start else "white",
                        edgecolor=dark if is_start else INK2))
    spot = {"left": (x - 0.16, y, "right", "center"),
            "right": (x + 0.16, y, "left", "center"),
            "above": (x, y + 0.13, "center", "bottom")}[side]
    ax.text(spot[0], spot[1], state_label(node), ha=spot[2], va=spot[3],
            fontsize=7 if is_start else 6.5,
            fontweight="bold" if is_start else "normal")


def draw_network(ax: plt.Axes, system: str, title: str) -> None:
    """
    Draw one system's frozen reactions with each engine's verdict.

    :param ax: Target axes.
    :type ax: plt.Axes
    :param system: System name.
    :type system: str
    :param title: Panel title.
    :type title: str
    :return: None.
    :rtype: None
    """
    start, candidates = candidate_list(system)
    layout = network_layout(start, candidates)
    pos = {k: np.array(v[:2]) for k, v in layout.items()}
    kept = {e: kept_reactions(system, e) for e, _, _, _ in ENGINES}
    groups = Counter(frozenset((tuple(c["state_a"]), tuple(c["state_b"])))
                     for c in candidates)
    seen: Counter = Counter()
    taken: list[np.ndarray] = []
    for c in candidates:
        a, b = tuple(c["state_a"]), tuple(c["state_b"])
        pair, index = frozenset((a, b)), seen[frozenset((a, b))]
        seen[pair] += 1
        offset = (index - (groups[pair] - 1) / 2) * 0.17
        bend = 0.0
        if abs(pos[a][0] - pos[b][0]) < 1e-6 and a != start:
            bend = 0.5 if (pos[b] - pos[a])[1] > 0 else -0.5
        draw_step(ax, edge_curve(pos[a], pos[b], offset, bend), taken,
                  bool(set(c["reactant_keys"]) & set(c["product_keys"])),
                  [c["reaction_id"] in kept[e] for e, _, _, _ in ENGINES])
    for node, place in layout.items():
        draw_node(ax, node, place, node == start)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 2.0, max(xs) + 2.0)
    ax.set_ylim(min(ys) - 0.3, max(ys) + 0.35)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=8)


def figure_networks() -> None:
    """
    Figure: the frozen networks with each engine's verdict per step.

    :return: None.
    :rtype: None
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.75),
                             layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.12]})
    for ax, (system, title) in zip(axes, NETWORK_SYSTEMS, strict=True):
        draw_network(ax, system, title)
    squares = [Patch(facecolor=color, edgecolor=color, label=name)
               for _, name, color, _ in ENGINES]
    squares.append(Patch(facecolor="white", edgecolor=AXIS,
                         label="not kept"))
    lines = [
        Line2D([], [], color=INK2, linewidth=0.8, label="step"),
        Line2D([], [], color=INK2, linewidth=0.8, linestyle=(0, (2.5, 1.5)),
               label="a helper molecule takes part"),
    ]
    fig.legend(handles=squares, ncol=6, loc="outside upper left",
               handlelength=1.2, columnspacing=1.1)
    fig.legend(handles=lines, ncol=2, loc="outside lower left",
               handlelength=1.8)
    save(fig, "fig_networks")


def tex_value(value: float | None) -> str:
    """
    Format a number for the LaTeX table.

    :param value: Number or None.
    :type value: float or None
    :return: One decimal, a minus sign in math mode, or a dash.
    :rtype: str
    """
    if value is None:
        return "--"
    return f"${round(value, 1) + 0.0:.1f}$"


def format_table_row(
    label: str,
    temp_label: str,
    values: Sequence[float | None],
    ref_label: str,
) -> str:
    """
    Format one LaTeX table row for the numbers appendix table.

    :param label: Reaction description in LaTeX.
    :type label: str
    :param temp_label: Temperature cell string.
    :type temp_label: str
    :param values: One value per engine, or None when missing.
    :type values: Sequence[float or None]
    :param ref_label: Reference cell string.
    :type ref_label: str
    :return: Formatted LaTeX table row ending with ``\\\\``.
    :rtype: str
    """
    cells = [tex_value(val) for val in values]
    return " & ".join([label, temp_label, *cells, ref_label]) + r" \\"


def write_number_table() -> None:
    """
    Write the appendix table behind Figures 4 to 6.

    :return: None.
    :rtype: None
    """
    lines = [
        r"\begin{tabular}{@{}lrrrrrrl@{}}",
        r"\toprule",
        (
            r"Reaction & $T$ (K) & xTB & UMA & MACE & "
            r"AIMNet2 & ANI-1xnr & Ref. \\"
        ),
        r"\midrule",
        (
            r"\multicolumn{8}{l}"
            r"{\emph{Reaction energy, model minus DFT on its own shapes}}\\"
        ),
    ]
    for system, rid, tex in REACTION_ERRORS:
        values = [
            dft_value(system, engine, rid, "reaction_error")
            for engine, _, _, _ in ENGINES
        ]
        lines.append(format_table_row(tex, "--", values, "--"))
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{8}{l}{\emph{Forward barrier}}\\")
    for system, rid, temp, _, tex, t_label, literature in BARRIERS:
        values = [
            summary_value(system, engine, rid, temp, "dg_act_fwd_kcal")
            for engine, _, _, _ in ENGINES
        ]
        lines.append(format_table_row(tex, t_label, values, literature))
    lines.append(r"\addlinespace")
    lines.append(
        r"\multicolumn{8}{l}"
        r"{\emph{Forward barrier, DFT on each model's shapes}}\\"
    )
    for system, rid, temp, _, tex, t_label, literature in BARRIERS:
        values = [
            dft_value(system, engine, rid, f"dg_barrier_dft_{temp}")
            for engine, _, _, _ in ENGINES
        ]
        lines.append(format_table_row(tex, t_label, values, literature))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "numbers_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def format_dft_cell(row: dict[str, str] | None) -> str:
    """
    Format one system cell of the DFT summary table as ``MAE (n)``.

    :param row: One engine's row from dft_summary.csv, or None.
    :type row: dict[str, str] or None
    :return: Formatted ``0.35 (4)`` cell, or ``--`` when missing.
    :rtype: str
    """
    mae = optional_float(row, "barrier_error_mae")
    count = optional_float(row, "barrier_error_n")
    if mae is None or count is None:
        return "--"
    return f"{mae:.2f} ({int(count)})"


def write_dft_table() -> None:
    """
    Write the per-system DFT error and force summary LaTeX table.

    :return: None.
    :rtype: None
    """
    summaries = {system: dft_summary_rows(system) for system, _ in SYSTEMS}
    header = " & ".join(
        ["Model", *(label for _, label in SYSTEMS), r"DFT force (eV/\AA)"]
    )
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for engine, display_name, _, _ in ENGINES:
        cells = [
            format_dft_cell(summaries[system].get(engine))
            for system, _ in SYSTEMS
        ]
        fmaxs = [
            val
            for system, _ in SYSTEMS
            if (
                val := optional_float(
                    summaries[system].get(engine), "dft_fmax_median"
                )
            )
            is not None
        ]
        fmax_cell = (
            f"{min(fmaxs):.2f}--{max(fmaxs):.2f}" if fmaxs else "--"
        )
        lines.append(
            " & ".join([display_name, *cells, fmax_cell]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "dft_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    """
    Draw every figure and write the number tables.

    :return: None.
    :rtype: None
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    apply_style()
    figure_discovery()
    figure_outcomes()
    figure_dft_errors()
    figure_barriers()
    figure_selectivity()
    figure_networks()
    write_number_table()
    write_dft_table()


if __name__ == "__main__":
    main()
