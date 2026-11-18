"""
Publication figures and LaTeX summary tables for reaction network evaluations.

Generates reference reaction network topology diagrams with mechanism paths,
kinetic barrier parity comparison panels across engines, master equation
kinetic yield time-evolution curves, computational cost versus accuracy
trade-off charts, and formatted LaTeX summary tables.
"""

import csv
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from nrnet.assemble import STATE_SEPARATOR, build_engine_network
from nrnet.kinetics import rate_matrix, simulate
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import CandidateSet
from nrnet.ranking import RANKS_FILE, make_rank_dir
from nrnet.refine import load_thermo_temperatures
from nrnet.store import make_refine_paths, read_jsonl, write_csv

__all__ = [
    "AXIS",
    "COST_ACCURACY_TOP_FACTOR",
    "GRID",
    "INK",
    "INK_SECONDARY",
    "MIN_LABEL_YIELD",
    "MUTED",
    "NETWORK_MARGIN",
    "NODE_LABEL_OFFSET_Y",
    "NODE_SIZE",
    "OTHER",
    "PARITY_LABEL_FONTSIZE",
    "SERIES",
    "SURFACE",
    "TABLE1_COLUMNS",
    "TABLE2_COLUMNS",
    "TABLE3_COLUMNS",
    "annotate_parity_points",
    "build_all_networks",
    "collect_cost_accuracy_points",
    "compute_parity_limits",
    "configure_cost_axes",
    "configure_parity_axes",
    "configure_yield_axes",
    "cost_point",
    "draw_network_edges",
    "draw_network_elements",
    "draw_network_labels",
    "draw_parity_panel",
    "draw_parity_points",
    "draw_state_curve",
    "draw_yield_curves",
    "draw_yield_panel",
    "engine_barrier_map",
    "format_latex_cell",
    "generate_high_temp_figures",
    "generate_low_temp_figures",
    "latex_table_row",
    "load_top_path",
    "make_figures",
    "match_parity_points",
    "number",
    "plot_barrier_parity",
    "plot_cost_accuracy",
    "plot_reference_network",
    "plot_yield_curves",
    "reachable_states",
    "read_csv_rows",
    "render_yield_panels",
    "save_figure",
    "select_top_products",
    "simulate_all_networks",
    "smiles_map",
    "species_label",
    "state_color",
    "state_label",
    "style_axes",
    "to_latex_table",
    "write_table_files",
    "write_tables",
]

SURFACE: str = "#fcfcfb"
INK: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
MUTED: str = "#898781"
GRID: str = "#e1e0d9"
AXIS: str = "#c3c2b7"
SERIES: tuple[str, str, str] = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER: str = "#c3c2b7"
NETWORK_MARGIN: float = 0.25
NODE_SIZE: int = 300
NODE_LABEL_OFFSET_Y: float = 0.08
PARITY_LABEL_FONTSIZE: int = 7
MIN_LABEL_YIELD: float = 0.02
COST_ACCURACY_TOP_FACTOR: float = 1.15

TABLE1_COLUMNS: tuple[str, ...] = (
    "engine",
    "temperature_k",
    "de_act_kcal_mae",
    "dg_act_kcal_mae",
    "de_rxn_kcal_mae",
    "dg_rxn_kcal_mae",
    "ts_success",
)

TABLE2_COLUMNS: tuple[str, ...] = (
    "engine",
    "temperature_k",
    "jaccard",
    "kendall_tau",
    "same_major_product",
    "same_top_path",
)

TABLE3_COLUMNS: tuple[str, ...] = (
    "temperature_k",
    "ranker",
    "top1_agree",
    "top3_overlap",
)


def species_label(key: str) -> str:
    """
    Extract chemical formula prefix before the final hash dash.

    :param key: Unique species identifier key.
    :type key: str
    :return: Chemical formula prefix.
    :rtype: str
    """
    return key.rsplit("-", 1)[0]


def smiles_map(candidate_set: CandidateSet) -> dict[str, str | None]:
    """
    Extract species SMILES mapping from candidate set.

    :param candidate_set: Frozen candidate set container.
    :type candidate_set: CandidateSet
    :return: Mapping of species keys to SMILES strings or None.
    :rtype: dict[str, str or None]
    """
    return {k: sp.smiles for k, sp in candidate_set.species.items()}


def state_label(
    state_id: str,
    smiles_by_key: Mapping[str, str | None],
) -> str:
    """
    Format state identifier using SMILES or fallback to formulas.

    Formulas alone are ambiguous for isomers, so SMILES come first; a
    radical RDKit could not perceive has no SMILES and shows its formula.

    :param state_id: Composite state identifier string.
    :type state_id: str
    :param smiles_by_key: Mapping from species key to SMILES or None.
    :type smiles_by_key: Mapping[str, str or None]
    :return: Formatted state label with species SMILES or formulas.
    :rtype: str
    """
    labels = [
        smiles_by_key.get(part) or species_label(part)
        for part in state_id.split(STATE_SEPARATOR)
    ]
    return STATE_SEPARATOR.join(labels)


def number(cell: str | None) -> float | None:
    """
    Convert table cell string into floating point number or None.

    :param cell: Cell string representation.
    :type cell: str or None
    :return: Parsed float or None for empty or invalid strings.
    :rtype: float or None
    """
    if cell is None or not str(cell).strip():
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """
    Read CSV file into a list of row dictionaries.

    :param path: Filesystem path to the CSV file.
    :type path: str or Path
    :return: List of row dictionaries with string values.
    :rtype: list[dict[str, str]]
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axes(ax: plt.Axes) -> None:
    """
    Apply recessive hairline grid and axis styling rules.

    :param ax: Matplotlib axes to style.
    :type ax: plt.Axes
    :return: None.
    :rtype: None
    """
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY)


def save_figure(fig: plt.Figure, out_base: str | Path) -> plt.Figure:
    """
    Export figure to PNG and PDF at base destination and close figure.

    :param fig: Matplotlib figure to export.
    :type fig: plt.Figure
    :param out_base: Filesystem base path without extension.
    :type out_base: str or Path
    :return: The closed figure instance.
    :rtype: plt.Figure
    """
    target = Path(out_base)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        f"{target}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight"
    )
    fig.savefig(f"{target}.pdf", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return fig


def draw_network_edges(
    graph: nx.DiGraph,
    pos: Mapping[str, Any],
    path_edges: set[tuple[str, str]],
    ax: plt.Axes,
) -> None:
    """
    Render background and highlighted mechanism edges on network graph.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param pos: Node position layout mapping.
    :type pos: Mapping[str, Any]
    :param path_edges: Set of highlighted path edge tuples.
    :type path_edges: set[tuple[str, str]]
    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :return: None.
    :rtype: None
    """
    other = [e for e in graph.edges if e not in path_edges]
    nx.draw_networkx_edges(
        graph, pos, edgelist=other, edge_color=MUTED, width=1.5, ax=ax
    )
    high = [e for e in graph.edges if e in path_edges]
    if high:
        nx.draw_networkx_edges(
            graph, pos, edgelist=high, edge_color=SERIES[0], width=2.5, ax=ax
        )


def draw_network_labels(
    graph: nx.DiGraph,
    pos: Mapping[str, Any],
    smiles_by_key: Mapping[str, str | None],
    ax: plt.Axes,
) -> None:
    """
    Render offset text labels below nodes in ink styling.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param pos: Node position layout mapping.
    :type pos: Mapping[str, Any]
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :return: None.
    :rtype: None
    """
    labels = {n: state_label(n, smiles_by_key) for n in graph.nodes}
    label_pos = {
        n: (float(p[0]), float(p[1]) - NODE_LABEL_OFFSET_Y)
        for n, p in pos.items()
    }
    nx.draw_networkx_labels(
        graph,
        label_pos,
        labels=labels,
        font_color=INK,
        font_size=8,
        verticalalignment="top",
        # Edges run behind a surface-colored box, never through text.
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
        ax=ax,
    )


def draw_network_elements(
    graph: nx.DiGraph,
    pos: Mapping[str, Any],
    path_nodes: set[str],
    path_edges: set[tuple[str, str]],
    ax: plt.Axes,
    smiles_by_key: Mapping[str, str | None],
) -> None:
    """
    Render nodes, edges, and labels onto network diagram axes.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param pos: Node position layout mapping.
    :type pos: Mapping[str, Any]
    :param path_nodes: Set of highlighted path node identifiers.
    :type path_nodes: set[str]
    :param path_edges: Set of highlighted path edge tuples.
    :type path_edges: set[tuple[str, str]]
    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :return: None.
    :rtype: None
    """
    colors = [SERIES[0] if n in path_nodes else OTHER for n in graph.nodes]
    nx.draw_networkx_nodes(
        graph, pos, node_color=colors, node_size=NODE_SIZE, ax=ax
    )
    draw_network_edges(graph, pos, path_edges, ax)
    draw_network_labels(graph, pos, smiles_by_key, ax)


def plot_reference_network(
    graph: nx.DiGraph,
    top_path: Sequence[str],
    out_base: str | Path,
    smiles_by_key: Mapping[str, str | None],
) -> plt.Figure:
    """
    Plot reference reaction network topology with highlighted top mechanism.

    :param graph: Assembled reference reaction network graph.
    :type graph: nx.DiGraph
    :param top_path: Sequence of state identifiers in the top path.
    :type top_path: Sequence[str]
    :param out_base: Destination file base path without extension.
    :type out_base: str or Path
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :return: Closed matplotlib figure.
    :rtype: plt.Figure
    """
    pos = nx.spring_layout(graph, seed=0)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    nodes_set = set(top_path)
    edges_set = set(itertools.pairwise(top_path))
    draw_network_elements(
        graph, pos, nodes_set, edges_set, ax, smiles_by_key=smiles_by_key
    )
    # Labels hang below and beside nodes; widen the view so the outermost
    # ones (long SMILES) are not clipped at the axes edge.
    ax.margins(x=NETWORK_MARGIN, y=NETWORK_MARGIN)
    ax.set_title("Reference network, top mechanism highlighted", color=INK)
    return save_figure(fig, out_base)


def engine_barrier_map(
    rows: Sequence[Mapping[str, str]],
    engine: str,
    target_temp: float,
) -> dict[str, float | None]:
    """
    Map reaction identifier to activation free energy for an engine.

    :param rows: Evaluated step table records.
    :type rows: Sequence[Mapping[str, str]]
    :param engine: Target chemistry engine identifier.
    :type engine: str
    :param target_temp: Thermochemical temperature in Kelvin.
    :type target_temp: float
    :return: Mapping of reaction identifier to barrier in kcal/mol.
    :rtype: dict[str, float or None]
    """
    return {
        r["reaction_id"]: number(r.get("dg_act_kcal"))
        for r in rows
        if r.get("engine") == engine
        and number(r.get("temperature_k")) == target_temp
    }


def match_parity_points(
    steps_rows: Sequence[Mapping[str, str]],
    reference: str,
    engine: str,
    temperature_k: float,
) -> list[tuple[str, float, float]]:
    """
    Match reaction barrier free energies between reference and compared engine.

    :param steps_rows: Evaluated step table rows across engines.
    :type steps_rows: Sequence[Mapping[str, str]]
    :param reference: Reference engine identifier.
    :type reference: str
    :param engine: Compared engine identifier.
    :type engine: str
    :param temperature_k: Thermochemical temperature in Kelvin.
    :type temperature_k: float
    :return: List of tuples with reaction ID, reference, and engine barriers.
    :rtype: list[tuple[str, float, float]]
    """
    target_temp = float(temperature_k)
    ref = engine_barrier_map(steps_rows, reference, target_temp)
    eng = engine_barrier_map(steps_rows, engine, target_temp)
    return [
        (rid, float(ref[rid]), float(eng[rid]))
        for rid in eng
        if rid in ref and ref[rid] is not None and eng[rid] is not None
    ]


def compute_parity_limits(
    all_points: Sequence[Sequence[Any]],
) -> tuple[float, float]:
    """
    Compute equal coordinate bounds across all parity comparison panels.

    :param all_points: Collections of matched points per engine panel.
    :type all_points: Sequence[Sequence[Any]]
    :return: Common minimum and maximum axis limits.
    :rtype: tuple[float, float]
    """
    vals = [
        val
        for points in all_points
        for _, ref_kcal, eng_kcal in points
        for val in (ref_kcal, eng_kcal)
    ]
    if not vals:
        return 0.0, 50.0
    min_val, max_val = min(vals), max(vals)
    pad = max(1.0, 0.05 * (max_val - min_val))
    return min_val - pad, max_val + pad


def configure_parity_axes(
    ax: plt.Axes,
    engine: str,
    lims: tuple[float, float],
) -> None:
    """
    Configure equal scaling, limits, and axis labels for parity panel.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param engine: Compared engine identifier.
    :type engine: str
    :param lims: Coordinate boundaries for equal aspect ratio display.
    :type lims: tuple[float, float]
    :return: None.
    :rtype: None
    """
    style_axes(ax)
    ax.plot(lims, lims, color=MUTED, linestyle="--", linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference ΔG‡ (kcal/mol)", color=INK_SECONDARY)
    ax.set_ylabel(f"{engine} ΔG‡ (kcal/mol)", color=INK_SECONDARY)


def annotate_parity_points(
    ax: plt.Axes,
    points: Sequence[Any],
) -> None:
    """
    Render direct reaction identifier labels beside each parity point.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param points: Reaction identifier and activation barrier tuples.
    :type points: Sequence[Any]
    :return: None.
    :rtype: None
    """
    for rid, ref_kcal, eng_kcal in points:
        ax.annotate(
            f" {rid}",
            (ref_kcal, eng_kcal),
            color=INK_SECONDARY,
            va="center",
            fontsize=PARITY_LABEL_FONTSIZE,
        )


def draw_parity_points(
    ax: plt.Axes,
    engine: str,
    points: Sequence[Any],
) -> None:
    """
    Plot parity scatter points, direct reaction labels, and panel title.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param engine: Compared engine identifier.
    :type engine: str
    :param points: Reaction identifier and activation barrier tuples.
    :type points: Sequence[Any]
    :return: None.
    :rtype: None
    """
    if not points:
        ax.set_title(f"{engine}, no shared barriers", color=INK)
        return
    xs = [ref_kcal for _, ref_kcal, _ in points]
    ys = [eng_kcal for _, _, eng_kcal in points]
    ax.plot(xs, ys, "o", color=SERIES[0], markersize=7)
    annotate_parity_points(ax, points)
    mae = sum(abs(y - x) for x, y in zip(xs, ys, strict=False)) / len(points)
    ax.set_title(f"{engine}, MAE {mae:.2f} kcal/mol", color=INK)


def draw_parity_panel(
    ax: plt.Axes,
    engine: str,
    points: Sequence[Any],
    lims: tuple[float, float],
) -> None:
    """
    Render barrier parity points and diagonal line for one compared engine.

    :param ax: Matplotlib panel axes instance.
    :type ax: plt.Axes
    :param engine: Compared engine identifier.
    :type engine: str
    :param points: Paired reference and engine activation free energies.
    :type points: Sequence[Any]
    :param lims: Coordinate boundaries for equal aspect ratio display.
    :type lims: tuple[float, float]
    :return: None.
    :rtype: None
    """
    configure_parity_axes(ax, engine, lims)
    draw_parity_points(ax, engine, points)


def plot_barrier_parity(
    steps_rows: Sequence[Mapping[str, str]],
    reference: str,
    engines: Sequence[str],
    temperature_k: float,
    out_base: str | Path,
) -> plt.Figure:
    """
    Plot activation barrier parity panels for compared engines in one row.

    :param steps_rows: Evaluated step table records across engines.
    :type steps_rows: Sequence[Mapping[str, str]]
    :param reference: Reference engine identifier.
    :type reference: str
    :param engines: Sequence of compared engine identifiers.
    :type engines: Sequence[str]
    :param temperature_k: Thermochemical temperature in Kelvin.
    :type temperature_k: float
    :param out_base: Destination filesystem base path without extension.
    :type out_base: str or Path
    :return: Closed matplotlib figure.
    :rtype: plt.Figure
    """
    pts = [
        match_parity_points(steps_rows, reference, eng, temperature_k)
        for eng in engines
    ]
    lims = compute_parity_limits(pts)
    n_panels = max(len(engines), 1)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.0 * n_panels, 4.0), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for idx, eng in enumerate(engines):
        draw_parity_panel(axes[0][idx], eng, pts[idx], lims)
    fig.tight_layout()
    return save_figure(fig, out_base)


def select_top_products(
    sim_data: Mapping[str, tuple[np.ndarray, list[str], str]],
) -> tuple[str, ...]:
    """
    Identify up to three product states with highest final yield across engines.

    :param sim_data: Mapping of engine to simulated populations and metadata.
    :type sim_data: Mapping[str, tuple[np.ndarray, list[str], str]]
    :return: Tuple of up to three global top product state identifiers.
    :rtype: tuple[str, ...]
    """
    peak_yields: dict[str, float] = {}
    for pops, states, source in sim_data.values():
        for idx, state in enumerate(states):
            if state != source:
                final_val = float(pops[-1, idx])
                peak_yields[state] = max(peak_yields.get(state, 0.0), final_val)
    sorted_prods = sorted(
        peak_yields.keys(), key=lambda item: peak_yields[item], reverse=True
    )
    return tuple(sorted_prods[:3])


def state_color(
    state: str,
    source: str,
    top_products: Sequence[str],
) -> tuple[str, bool]:
    """
    Assign plotting color and determine if text label is required.

    :param state: Chemical state identifier.
    :type state: str
    :param source: Reactant source state identifier.
    :type source: str
    :param top_products: Globally prioritized top product states.
    :type top_products: Sequence[str]
    :return: Tuple of color hex code and label flag.
    :rtype: tuple[str, bool]
    """
    if state == source:
        return INK_SECONDARY, True
    if state in top_products:
        return SERIES[top_products.index(state)], True
    return OTHER, False


def configure_yield_axes(
    ax: plt.Axes,
    engine: str,
    is_first: bool,
) -> None:
    """
    Configure log scale, limits, and axis labels on a yield panel.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param is_first: Whether this panel is leftmost in the row.
    :type is_first: bool
    :return: None.
    :rtype: None
    """
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(engine, color=INK)
    ax.set_xlabel("Time (s)", color=INK_SECONDARY)
    if is_first:
        ax.set_ylabel("Yield", color=INK_SECONDARY)


def reachable_states(graph: nx.DiGraph, source: str) -> set[str]:
    """
    Find source state and all descendant states reachable in graph.

    :param graph: Reaction network graph.
    :type graph: nx.DiGraph
    :param source: Reactant source state identifier.
    :type source: str
    :return: Set of reachable state identifiers including source.
    :rtype: set[str]
    """
    return set(nx.descendants(graph, source)) | {source}


def draw_state_curve(
    ax: plt.Axes,
    times: np.ndarray,
    pop: np.ndarray,
    state: str,
    color: str,
    is_colored: bool,
    smiles_by_key: Mapping[str, str | None],
) -> None:
    """
    Render population trajectory line and optional endpoint label.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param times: Evaluation time points array in seconds.
    :type times: np.ndarray
    :param pop: Population values over time for one state.
    :type pop: np.ndarray
    :param state: State identifier string.
    :type state: str
    :param color: Matplotlib color code for the line.
    :type color: str
    :param is_colored: Whether line belongs to source or top product.
    :type is_colored: bool
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :return: None.
    :rtype: None
    """
    ax.plot(times, pop, color=color, linewidth=1.5, label=state)
    if is_colored and float(pop[-1]) >= MIN_LABEL_YIELD:
        label = f" {state_label(state, smiles_by_key)}"
        ax.text(times[-1], pop[-1], label, color=INK, va="center", fontsize=8)


def draw_yield_curves(
    ax: plt.Axes,
    pops: np.ndarray,
    states: Sequence[str],
    source: str,
    times: np.ndarray,
    top_products: Sequence[str],
    reachable: set[str],
    smiles_by_key: Mapping[str, str | None],
) -> None:
    """
    Render individual state curves and direct labels on yield panel.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param pops: State populations array of shape (n_times, n_states).
    :type pops: np.ndarray
    :param states: Ordered list of state identifiers.
    :type states: Sequence[str]
    :param source: Reactant source state identifier.
    :type source: str
    :param times: Evaluation time points array in seconds.
    :type times: np.ndarray
    :param top_products: Globally prioritized top product states.
    :type top_products: Sequence[str]
    :param reachable: The source and the states reachable from it; other
        states stay at zero and are not drawn.
    :type reachable: set[str]
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :return: None.
    :rtype: None
    """
    for idx, state in enumerate(states):
        if state not in reachable:
            continue
        color, is_colored = state_color(state, source, top_products)
        draw_state_curve(
            ax, times, pops[:, idx], state, color, is_colored, smiles_by_key
        )


def draw_yield_panel(
    ax: plt.Axes,
    engine: str,
    pops: np.ndarray,
    states: Sequence[str],
    source: str,
    times: np.ndarray,
    top_products: Sequence[str],
    is_first: bool,
    reachable: set[str],
    smiles_by_key: Mapping[str, str | None],
) -> None:
    """
    Render kinetic yield time curves and end labels on an engine panel.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param engine: Chemistry engine identifier.
    :type engine: str
    :param pops: State populations over time of shape (n_times, n_states).
    :type pops: np.ndarray
    :param states: Ordered list of state identifiers.
    :type states: Sequence[str]
    :param source: Reactant source state identifier.
    :type source: str
    :param times: Evaluation time points array in seconds.
    :type times: np.ndarray
    :param top_products: Globally prioritized top product states.
    :type top_products: Sequence[str]
    :param is_first: Whether this panel is the leftmost in the row.
    :type is_first: bool
    :param reachable: The source and the states reachable from it; other
        states stay at zero and are not drawn.
    :type reachable: set[str]
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :return: None.
    :rtype: None
    """
    configure_yield_axes(ax, engine, is_first)
    draw_yield_curves(
        ax,
        pops,
        states,
        source,
        times,
        top_products,
        reachable,
        smiles_by_key,
    )


def simulate_all_networks(
    networks: Mapping[str, nx.DiGraph],
    times: np.ndarray,
) -> dict[str, tuple[np.ndarray, list[str], str]]:
    """
    Run kinetic master equation simulations across multiple networks.

    :param networks: Mapping of engine identifier to assembled graph.
    :type networks: Mapping[str, nx.DiGraph]
    :param times: Evaluation time points array in seconds.
    :type times: np.ndarray
    :return: Simulation outputs mapping engine to populations and metadata.
    :rtype: dict[str, tuple[np.ndarray, list[str], str]]
    """
    sims: dict[str, tuple[np.ndarray, list[str], str]] = {}
    for eng, graph in networks.items():
        src = graph.graph["source"]
        _, states = rate_matrix(graph)
        sims[eng] = (simulate(graph, src, times), states, src)
    return sims


def render_yield_panels(
    axes: Any,
    sims: Mapping[str, tuple[np.ndarray, list[str], str]],
    networks: Mapping[str, nx.DiGraph],
    times: np.ndarray,
    top_prods: Sequence[str],
    smiles_by_key: Mapping[str, str | None],
) -> None:
    """
    Render individual yield curve panels for all simulated networks.

    :param axes: Array of panel axes.
    :type axes: Any
    :param sims: Mapping of engine to simulation results and metadata.
    :type sims: Mapping[str, tuple[np.ndarray, list[str], str]]
    :param networks: Mapping of engine to assembled graph.
    :type networks: Mapping[str, nx.DiGraph]
    :param times: Evaluation time points array in seconds.
    :type times: np.ndarray
    :param top_prods: Prioritized top product state identifiers.
    :type top_prods: Sequence[str]
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :return: None.
    :rtype: None
    """
    for idx, (eng, (pops, states, src)) in enumerate(sims.items()):
        draw_yield_panel(
            axes[0][idx],
            eng,
            pops,
            states,
            src,
            times,
            top_prods,
            idx == 0,
            reachable_states(networks[eng], src),
            smiles_by_key,
        )


def plot_yield_curves(
    networks: Mapping[str, nx.DiGraph],
    temperature_k: float,
    out_base: str | Path,
    smiles_by_key: Mapping[str, str | None],
) -> plt.Figure:
    """
    Plot kinetic simulation yield curves for chemical engines over log time.

    :param networks: Mapping of engine identifier to assembled reaction graph.
    :type networks: Mapping[str, nx.DiGraph]
    :param temperature_k: Thermochemical temperature in Kelvin.
    :type temperature_k: float
    :param out_base: Destination filesystem base path without extension.
    :type out_base: str or Path
    :param smiles_by_key: Mapping from species keys to SMILES.
    :type smiles_by_key: Mapping[str, str or None]
    :return: Closed matplotlib figure.
    :rtype: plt.Figure
    """
    times = np.logspace(-6, 6, 200)
    sims = simulate_all_networks(networks, times)
    top_prods = select_top_products(sims)
    n_panels = max(len(networks), 1)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.0 * n_panels, 4.0), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    render_yield_panels(axes, sims, networks, times, top_prods, smiles_by_key)
    fig.tight_layout()
    return save_figure(fig, out_base)


def cost_point(
    row: Mapping[str, str],
    costs: Mapping[str, float | None],
    target_temp: float,
) -> tuple[str, float, float] | None:
    """
    Extract wall time and barrier MAE point for one engine summary record.

    :param row: Single summary evaluation record.
    :type row: Mapping[str, str]
    :param costs: Mapping of engine identifier to wall time in hours.
    :type costs: Mapping[str, float or None]
    :param target_temp: Thermochemical temperature in Kelvin.
    :type target_temp: float
    :return: Tuple of engine name, wall hours, and MAE or None if invalid.
    :rtype: tuple[str, float, float] or None
    """
    if number(row.get("temperature_k")) != target_temp:
        return None
    eng = row.get("engine", "")
    mae = number(row.get("dg_act_kcal_mae"))
    wall = costs.get(eng)
    if mae is not None and wall is not None and wall > 0:
        return eng, wall, mae
    return None


def collect_cost_accuracy_points(
    summary_rows: Sequence[Mapping[str, str]],
    cost_rows: Sequence[Mapping[str, str]],
    temperature_k: float,
) -> list[tuple[str, float, float]]:
    """
    Match wall time and barrier MAE metrics across evaluated engines.

    :param summary_rows: Error summary table rows from e1_summary.
    :type summary_rows: Sequence[Mapping[str, str]]
    :param cost_rows: Cost metrics table rows from e4_cost.
    :type cost_rows: Sequence[Mapping[str, str]]
    :param temperature_k: Thermochemical temperature in Kelvin.
    :type temperature_k: float
    :return: List of tuples with engine name, wall hours, and barrier MAE.
    :rtype: list[tuple[str, float, float]]
    """
    costs = {
        r["engine"]: number(r.get("wall_h")) for r in cost_rows if "engine" in r
    }
    target_temp = float(temperature_k)
    pts = [cost_point(r, costs, target_temp) for r in summary_rows]
    return [pt for pt in pts if pt is not None]


def configure_cost_axes(
    ax: plt.Axes,
    points: Sequence[tuple[str, float, float]],
) -> None:
    """
    Configure log scale, labels, and zero-based barrier MAE limits.

    :param ax: Matplotlib axes instance.
    :type ax: plt.Axes
    :param points: Matched cost and accuracy data points.
    :type points: Sequence[tuple[str, float, float]]
    :return: None.
    :rtype: None
    """
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Wall time per network (h)", color=INK_SECONDARY)
    ax.set_ylabel("Barrier MAE (kcal/mol)", color=INK_SECONDARY)
    if points:
        max_mae = max(mae for eng, wall, mae in points)
        top = (
            COST_ACCURACY_TOP_FACTOR * max_mae
            if max_mae > 0.0
            else COST_ACCURACY_TOP_FACTOR
        )
        ax.set_ylim(bottom=0.0, top=top)
    else:
        ax.set_ylim(bottom=0.0)


def plot_cost_accuracy(
    summary_rows: Sequence[Mapping[str, str]],
    cost_rows: Sequence[Mapping[str, str]],
    temperature_k: float,
    out_base: str | Path,
) -> plt.Figure:
    """
    Plot computational wall time against barrier MAE trade-off.

    :param summary_rows: Error summary table records.
    :type summary_rows: Sequence[Mapping[str, str]]
    :param cost_rows: Computational cost table records.
    :type cost_rows: Sequence[Mapping[str, str]]
    :param temperature_k: Thermochemical temperature in Kelvin.
    :type temperature_k: float
    :param out_base: Destination filesystem base path without extension.
    :type out_base: str or Path
    :return: Closed matplotlib figure.
    :rtype: plt.Figure
    """
    points = collect_cost_accuracy_points(
        summary_rows, cost_rows, temperature_k
    )
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    fig.patch.set_facecolor(SURFACE)
    configure_cost_axes(ax, points)
    for eng, wall, mae in points:
        ax.plot(wall, mae, "o", color=SERIES[0], markersize=7)
        ax.annotate(
            eng,
            (wall, mae),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    return save_figure(fig, out_base)


def format_latex_cell(cell: Any) -> str:
    """
    Format a single table cell value for LaTeX tabular representation.

    :param cell: Raw cell value.
    :type cell: Any
    :return: Formatted cell string with escaped characters.
    :rtype: str
    """
    if cell is None:
        return "--"
    text = str(cell).strip()
    if not text or text == "None":
        return "--"
    num = number(text)
    if num is not None:
        return f"{num:.2f}"
    return text.replace("_", "\\_")


def latex_table_row(
    row: Mapping[str, Any],
    columns: Sequence[str],
) -> str:
    """
    Format a single row mapping as a LaTeX table row string.

    :param row: Row mapping record.
    :type row: Mapping[str, Any]
    :param columns: Sequence of column names to extract.
    :type columns: Sequence[str]
    :return: Formatted LaTeX table row.
    :rtype: str
    """
    cells = [format_latex_cell(row.get(col)) for col in columns]
    return " & ".join(cells) + " \\\\"


def to_latex_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> str:
    """
    Convert row mappings into a standalone LaTeX tabular string.

    :param rows: Sequence of row mappings.
    :type rows: Sequence[Mapping[str, Any]]
    :param columns: Sequence of column names to output.
    :type columns: Sequence[str]
    :return: Formatted LaTeX tabular environment text.
    :rtype: str
    """
    align = "l" * len(columns)
    header = " & ".join(col.replace("_", "\\_") for col in columns) + " \\\\"
    lines = [latex_table_row(row, columns) for row in rows]
    body = ("\n" + "\n".join(lines)) if lines else ""
    return (
        f"\\begin{{tabular}}{{{align}}}\n"
        f"\\hline\n"
        f"{header}\n"
        f"\\hline"
        f"{body}\n"
        f"\\hline\n"
        f"\\end{{tabular}}\n"
    )


def write_table_files(
    path_base: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    """
    Serialize table data to both CSV and LaTeX tabular files.

    :param path_base: Base filesystem destination without extension.
    :type path_base: Path
    :param rows: Sequence of row dictionary records.
    :type rows: Sequence[Mapping[str, Any]]
    :param columns: Sequence of column names to serialize.
    :type columns: Sequence[str]
    :return: None.
    :rtype: None
    """
    projected = [{c: r.get(c) for c in columns} for r in rows]
    write_csv(path_base.with_suffix(".csv"), projected)
    tex_str = to_latex_table(projected, columns)
    path_base.with_suffix(".tex").write_text(tex_str, encoding="utf-8")


def write_tables(eval_dir: str | Path, out_dir: str | Path) -> None:
    """
    Read evaluation summary CSVs and write tables 1, 2, and 3.

    :param eval_dir: Source evaluation results directory.
    :type eval_dir: str or Path
    :param out_dir: Destination tables directory.
    :type out_dir: str or Path
    :return: None.
    :rtype: None
    """
    src = Path(eval_dir)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    e1_rows = read_csv_rows(src / "e1_summary.csv")
    e2_rows = read_csv_rows(src / "e2_networks.csv")
    e3_rows = read_csv_rows(src / "e3_rankers.csv")
    write_table_files(dest / "table1", e1_rows, TABLE1_COLUMNS)
    write_table_files(dest / "table2", e2_rows, TABLE2_COLUMNS)
    write_table_files(dest / "table3", e3_rows, TABLE3_COLUMNS)


def load_top_path(
    ranks_path: Path,
    temperature_k: float,
) -> tuple[str, ...]:
    """
    Extract position-1 mechanism path for ranker R3 at target temperature.

    :param ranks_path: Filesystem path to serialized ranks JSONL.
    :type ranks_path: Path
    :param temperature_k: Thermochemical temperature in Kelvin.
    :type temperature_k: float
    :return: Sequence of state keys in the top mechanism path.
    :rtype: tuple[str, ...]
    """
    if not ranks_path.is_file():
        return ()
    target_temp = float(temperature_k)
    for row in read_jsonl(ranks_path):
        if (
            number(str(row.get("temperature_k"))) == target_temp
            and row.get("ranker") == "R3"
            and int(row.get("position", 0)) == 1
        ):
            return tuple(row.get("path", ()))
    return ()


def build_all_networks(
    candidate_set: CandidateSet,
    all_engines: Sequence[str],
    results_dir: str | Path,
    system: str,
    temperature_k: float,
) -> dict[str, nx.DiGraph]:
    """
    Construct reaction network graphs across multiple chemical engines.

    :param candidate_set: Frozen candidate reactions container.
    :type candidate_set: CandidateSet
    :param all_engines: Sequence of engine identifiers to process.
    :type all_engines: Sequence[str]
    :param results_dir: Base results directory path.
    :type results_dir: str or Path
    :param system: Chemical system identifier.
    :type system: str
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: Mapping of engine identifier to assembled DiGraph.
    :rtype: dict[str, nx.DiGraph]
    """
    networks: dict[str, nx.DiGraph] = {}
    for engine in all_engines:
        paths = make_refine_paths(results_dir, system, engine)
        graph, _ = build_engine_network(candidate_set, paths, temperature_k)
        networks[engine] = graph
    return networks


def generate_high_temp_figures(
    system: str,
    reference: str,
    engines: Sequence[str],
    candidate_set: CandidateSet,
    results_dir: str | Path,
    high_t: float,
    out_dir: Path,
) -> None:
    """
    Generate Figures 1 and 3 evaluated at the highest temperature.

    :param system: Chemical system identifier.
    :type system: str
    :param reference: Reference engine identifier.
    :type reference: str
    :param engines: Compared engine identifiers.
    :type engines: Sequence[str]
    :param candidate_set: Frozen candidate set instance.
    :type candidate_set: CandidateSet
    :param results_dir: Base results directory.
    :type results_dir: str or Path
    :param high_t: Highest thermochemical temperature in Kelvin.
    :type high_t: float
    :param out_dir: Figures output directory.
    :type out_dir: Path
    :return: None.
    :rtype: None
    """
    ref_paths = make_refine_paths(results_dir, system, reference)
    ref_graph, _ = build_engine_network(candidate_set, ref_paths, high_t)
    ranks_path = make_rank_dir(results_dir, system, reference) / RANKS_FILE
    top_path = load_top_path(ranks_path, high_t)
    smiles = smiles_map(candidate_set)
    plot_reference_network(
        ref_graph, top_path, out_dir / "fig1", smiles_by_key=smiles
    )
    all_engs = [reference] + [e for e in engines if e != reference]
    nets = build_all_networks(
        candidate_set, all_engs, results_dir, system, high_t
    )
    plot_yield_curves(nets, high_t, out_dir / "fig3", smiles_by_key=smiles)


def generate_low_temp_figures(
    eval_dir: Path,
    reference: str,
    engines: Sequence[str],
    low_t: float,
    out_dir: Path,
) -> None:
    """
    Generate Figures 2 and 4 evaluated at the lowest temperature.

    :param eval_dir: Source evaluation directory with summary CSVs.
    :type eval_dir: Path
    :param reference: Reference engine identifier.
    :type reference: str
    :param engines: Compared engine identifiers.
    :type engines: Sequence[str]
    :param low_t: Lowest thermochemical temperature in Kelvin.
    :type low_t: float
    :param out_dir: Figures output directory.
    :type out_dir: Path
    :return: None.
    :rtype: None
    """
    steps = read_csv_rows(eval_dir / "e1_steps.csv")
    plot_barrier_parity(steps, reference, engines, low_t, out_dir / "fig2")
    summary_rows = read_csv_rows(eval_dir / "e1_summary.csv")
    cost_rows = read_csv_rows(eval_dir / "e4_cost.csv")
    plot_cost_accuracy(summary_rows, cost_rows, low_t, out_dir / "fig4")


def make_figures(
    system: str,
    reference: str,
    engines: Sequence[str],
    candidates_dir: str | Path,
    results_dir: str | Path,
    search_path: str | Path,
    figures_dir: str | Path,
) -> Path:
    """
    Generate all publication figures and summary tables for a system.

    :param system: Chemical system identifier.
    :type system: str
    :param reference: Reference chemistry engine identifier.
    :type reference: str
    :param engines: Compared chemistry engine identifiers.
    :type engines: Sequence[str]
    :param candidates_dir: Root directory of frozen candidate sets.
    :type candidates_dir: str or Path
    :param results_dir: Root results directory.
    :type results_dir: str or Path
    :param search_path: Path to search YAML configuration file.
    :type search_path: str or Path
    :param figures_dir: Destination root directory for figures.
    :type figures_dir: str or Path
    :return: Destination output directory for the system figures.
    :rtype: Path
    """
    temps = load_thermo_temperatures(search_path)
    out_dir = Path(figures_dir) / system
    eval_dir = Path(results_dir) / "evaluate" / system
    cset = load_candidates(Path(candidates_dir) / system)
    generate_high_temp_figures(
        system, reference, engines, cset, results_dir, max(temps), out_dir
    )
    generate_low_temp_figures(eval_dir, reference, engines, min(temps), out_dir)
    write_tables(eval_dir, out_dir / "tables")
    return out_dir
