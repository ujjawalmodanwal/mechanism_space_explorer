"""
Unit and integration tests for publication figures and summary tables.

Verifies label formatting, numeric parsing, LaTeX tabular generation,
network topology plotting, barrier parity visualization, kinetic yield
curve comparison, and computational cost-accuracy trade-off charts.
"""

from pathlib import Path

import pytest
import yaml
from ase import Atoms

from nrnet.assemble import build_engine_network
from nrnet.figures import (
    make_figures,
    number,
    plot_barrier_parity,
    plot_cost_accuracy,
    plot_reference_network,
    plot_yield_curves,
    smiles_map,
    species_label,
    state_label,
    to_latex_table,
    write_tables,
)
from nrnet.nanoreactor.candidate_store import (
    CandidateSet,
    write_candidate_files,
)
from nrnet.nanoreactor.candidates import Species
from nrnet.ranking import RANKS_FILE, make_rank_dir
from nrnet.store import RefinePaths, write_csv, write_jsonl


def test_species_and_state_label() -> None:
    """
    Verify species and composite state label formatting helpers.

    :return: None.
    :rtype: None
    """
    assert species_label("CH2O-021b6208") == "CH2O"
    assert species_label("H2O") == "H2O"
    smiles_by_key = {"CH2O-021b6208": "C=O", "NH3-abcdef12": "N"}
    res = state_label("CH2O-021b6208 + NH3-abcdef12", smiles_by_key)
    assert res == "C=O + N"
    fallback_map = {"CH2O-021b6208": None}
    res_fallback = state_label("CH2O-021b6208 + NH3-abcdef12", fallback_map)
    assert res_fallback == "CH2O + NH3"
    assert state_label("CH2O-021b6208 + NH3-abcdef12", {}) == "CH2O + NH3"
    assert state_label("CH2O-021b6208 + NH3-abcdef12", {}) == "CH2O + NH3"
    assert state_label("A + B", {}) == "A + B"


def test_smiles_map() -> None:
    """
    Verify smiles_map extracts species keys and their SMILES strings.

    :return: None.
    :rtype: None
    """
    cset = CandidateSet(
        reactant_state=("A",),
        species={
            "A-1": Species("A-1", Atoms(), "C", "inchikey1", 1),
            "B-2": Species("B-2", Atoms(), None, None, 2),
        },
        candidates=(),
    )
    mapping = smiles_map(cset)
    assert mapping == {"A-1": "C", "B-2": None}


def test_number_parsing() -> None:
    """
    Verify numeric string parsing and empty or invalid handling.

    :return: None.
    :rtype: None
    """
    assert number(None) is None
    assert number("") is None
    assert number("   ") is None
    assert number("invalid") is None
    assert number("12.34") == 12.34
    assert number("-5.0") == -5.0
    assert number("0") == 0.0


def test_to_latex_table() -> None:
    """
    Verify LaTeX tabular formatting with escaped headers and missing markers.

    :return: None.
    :rtype: None
    """
    rows = [{"engine_name": "xtb", "barrier_mae": 1.234, "ts_val": None}]
    columns = ["engine_name", "barrier_mae", "ts_val"]
    latex = to_latex_table(rows, columns)
    assert "\\begin{tabular}{lll}" in latex
    assert "engine\\_name & barrier\\_mae & ts\\_val \\\\" in latex
    assert "xtb & 1.23 & -- \\\\" in latex
    assert "\\end{tabular}" in latex


def test_plot_reference_network(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
    tmp_path: Path,
) -> None:
    """
    Verify reference network plot creates outputs with offset node labels.

    :param fake_refine_results: Fixture providing candidate set and paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cset, paths = fake_refine_results
    graph, _ = build_engine_network(cset, paths, 298.15)
    out_base = tmp_path / "fig1_test"
    fig = plot_reference_network(graph, ("A + B + S", "C + S"), out_base, {})
    assert (tmp_path / "fig1_test.png").is_file()
    assert (tmp_path / "fig1_test.pdf").is_file()
    assert len(fig.axes[0].texts) == len(graph.nodes)


def sample_parity_rows() -> list[dict[str, str]]:
    """
    Construct synthetic reaction step records for parity testing.

    :return: List of step record dictionaries.
    :rtype: list[dict[str, str]]
    """
    row_ref = {
        "engine": "xtb",
        "temperature_k": "298.15",
        "reaction_id": "R1",
        "dg_act_kcal": "15.0",
    }
    row_eng = {
        "engine": "aimnet",
        "temperature_k": "298.15",
        "reaction_id": "R1",
        "dg_act_kcal": "16.5",
    }
    return [row_ref, row_eng]


def test_plot_barrier_parity(tmp_path: Path) -> None:
    """
    Verify barrier parity plot creates outputs and reaction ID labels.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    out_base = tmp_path / "fig2_test"
    fig = plot_barrier_parity(
        sample_parity_rows(), "xtb", ["aimnet"], 298.15, out_base
    )
    assert (tmp_path / "fig2_test.png").is_file()
    assert (tmp_path / "fig2_test.pdf").is_file()
    texts = [t.get_text().strip() for t in fig.axes[0].texts]
    assert "R1" in texts


def test_plot_yield_curves_same_color(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
    tmp_path: Path,
) -> None:
    """
    Verify kinetic yield plot creates outputs and preserves product colors.

    :param fake_refine_results: Fixture providing candidate set and paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cset, paths = fake_refine_results
    graph1, _ = build_engine_network(cset, paths, 298.15)
    graph2, _ = build_engine_network(cset, paths, 298.15)
    out_base = tmp_path / "fig3_test"
    nets = {"xtb": graph1, "aimnet": graph2}
    fig = plot_yield_curves(nets, 298.15, out_base, {})
    assert (tmp_path / "fig3_test.png").is_file()
    assert (tmp_path / "fig3_test.pdf").is_file()
    lines0 = {line.get_label(): line.get_color() for line in fig.axes[0].lines}
    lines1 = {line.get_label(): line.get_color() for line in fig.axes[1].lines}
    common = set(lines0.keys()) & set(lines1.keys())
    assert len(common) > 0
    for state in common:
        assert lines0[state] == lines1[state]


def test_plot_yield_curves_unreachable(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
    tmp_path: Path,
) -> None:
    """
    Verify states outside source component are excluded from yield curves.

    :param fake_refine_results: Fixture providing candidate set and paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cset, paths = fake_refine_results
    graph, inputs = build_engine_network(cset, paths, 298.15)
    assert inputs is not None
    graph.add_node("DISCONNECTED_STATE")
    out_base = tmp_path / "fig3_unreachable_test"
    fig = plot_yield_curves({"xtb": graph}, 298.15, out_base, {})
    labels = {line.get_label() for line in fig.axes[0].lines}
    assert "DISCONNECTED_STATE" not in labels
    assert graph.graph["source"] in labels


def test_plot_cost_accuracy(tmp_path: Path) -> None:
    """
    Verify cost accuracy trade-off plot creates output and starts y-axis at 0.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    summary = [
        {
            "engine": "aimnet",
            "temperature_k": "298.15",
            "dg_act_kcal_mae": "1.5",
        }
    ]
    cost = [{"engine": "aimnet", "wall_h": "0.2"}]
    out_base = tmp_path / "fig4_test"
    fig = plot_cost_accuracy(summary, cost, 298.15, out_base)
    assert (tmp_path / "fig4_test.png").is_file()
    assert (tmp_path / "fig4_test.pdf").is_file()
    bottom, top = fig.axes[0].get_ylim()
    assert bottom == 0.0
    assert top == pytest.approx(1.5 * 1.15)


def test_write_tables(tmp_path: Path) -> None:
    """
    Verify summary evaluation tables write both CSV and LaTeX formats.

    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    write_csv(eval_dir / "e1_summary.csv", [{"engine": "xtb"}])
    write_csv(eval_dir / "e2_networks.csv", [{"engine": "xtb"}])
    write_csv(eval_dir / "e3_rankers.csv", [{"ranker": "R0"}])
    out_dir = tmp_path / "tables"
    write_tables(eval_dir, out_dir)
    for name in ("table1", "table2", "table3"):
        assert (out_dir / f"{name}.csv").is_file()
        assert (out_dir / f"{name}.tex").is_file()


def write_empty_eval_tables(eval_dir: Path) -> None:
    """
    Create empty evaluation CSV files in evaluation directory.

    :param eval_dir: Evaluation results directory.
    :type eval_dir: Path
    :return: None.
    :rtype: None
    """
    names = ("e1_steps", "e1_summary", "e2_networks", "e3_rankers", "e4_cost")
    for name in names:
        write_csv(eval_dir / f"{name}.csv", [])


def setup_make_figures_system(
    base: Path,
    cset: CandidateSet,
) -> tuple[Path, Path]:
    """
    Populate evaluation files and candidates for make_figures test.

    :param base: Temporary root directory path.
    :type base: Path
    :param cset: Frozen CandidateSet instance.
    :type cset: CandidateSet
    :return: Tuple of search config path and candidates directory.
    :rtype: tuple[Path, Path]
    """
    search_path = base / "search.yaml"
    with search_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"thermo": {"temperatures_k": [298.15, 600.0]}}, handle)
    eval_dir = base / "evaluate" / "toy"
    eval_dir.mkdir(parents=True)
    write_empty_eval_tables(eval_dir)
    ranks_path = make_rank_dir(base, "toy", "xtb") / RANKS_FILE
    write_jsonl(ranks_path, [])
    cand_dir = base / "candidates" / "toy"
    cand_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_files(cset, cand_dir)
    return search_path, base / "candidates"


def assert_figure_outputs(fig_dir: Path) -> None:
    """
    Assert existence of four figure pairs and three table pairs.

    :param fig_dir: Output figures directory path.
    :type fig_dir: Path
    :return: None.
    :rtype: None
    """
    for idx in range(1, 5):
        assert (fig_dir / f"fig{idx}.png").is_file()
        assert (fig_dir / f"fig{idx}.pdf").is_file()
    for idx in range(1, 4):
        assert (fig_dir / "tables" / f"table{idx}.csv").is_file()
        assert (fig_dir / "tables" / f"table{idx}.tex").is_file()


def test_make_figures_pipeline(
    fake_refine_results: tuple[CandidateSet, RefinePaths],
    tmp_path: Path,
) -> None:
    """
    Verify make_figures produces all four figures and three tables.

    :param fake_refine_results: Fixture providing candidate set and paths.
    :type fake_refine_results: tuple[CandidateSet, RefinePaths]
    :param tmp_path: Temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    cset = fake_refine_results[0]
    search_path, cand_dir = setup_make_figures_system(tmp_path, cset)
    fig_dir = make_figures(
        system="toy",
        reference="xtb",
        engines=["xtb"],
        candidates_dir=cand_dir,
        results_dir=tmp_path,
        search_path=search_path,
        figures_dir=tmp_path / "figs",
    )
    assert_figure_outputs(fig_dir)
