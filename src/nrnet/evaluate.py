"""
Evaluation metrics, reaction energies, structural RMSD, and computational cost.

Computes reaction energies and forward activation barriers for candidate
reactions, evaluates mean absolute errors against reference calculations,
calculates transition state geometry root-mean-square deviations (RMSD),
summarizes engine call execution costs, assesses reaction network agreement,
and scores ranking heuristics against kinetic simulation yields.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import scipy.stats
import yaml
from ase import Atoms
from ase.build import minimize_rotation_and_translation

from nrnet.assemble import (
    confirmed_ts_energy,
    species_free_energies,
    state_free_energy,
)
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import Candidate, CandidateSet
from nrnet.network import KCAL_MOL_EV
from nrnet.ranking import RANKS_FILE, make_rank_dir
from nrnet.records import STATUS_ENGINE_ERROR, read_calls
from nrnet.refine import STATUS_OK, load_thermo_temperatures
from nrnet.store import (
    RefinePaths,
    latest_rows,
    make_refine_paths,
    read_jsonl,
    temperature_label,
    write_csv,
)
from nrnet.ts_search import TSStatus

__all__ = [
    "ENERGY_FIELDS",
    "NO_REACTION",
    "SECONDS_PER_H",
    "STATUS_MISSING",
    "TABLE_BUILDERS",
    "TOP_N",
    "EvaluationScope",
    "build_e1_steps_table",
    "build_e1_summary_table",
    "build_e2_networks_table",
    "build_e3_rankers_table",
    "build_e4_cost_table",
    "compute_jaccard",
    "compute_kendall_tau",
    "confirmed_ts_values",
    "cost_row",
    "energy_change_kcal",
    "engine_summary_rows",
    "evaluate",
    "field_error",
    "frame_rmsd",
    "load_engine_ranks",
    "load_yield_cutoff",
    "major_product",
    "make_scope",
    "network_agreement",
    "r4_yields",
    "ranker_agreement",
    "ranker_rows",
    "reaction_energies",
    "read_ts_frames",
    "single_ranker_agreement",
    "species_energies",
    "step_errors",
    "step_table",
    "top_position_match",
    "ts_rmsd",
    "ts_rmsd_summary",
]

ENERGY_FIELDS: tuple[str, ...] = (
    "de_rxn_kcal",
    "dg_rxn_kcal",
    "de_act_kcal",
    "dg_act_kcal",
)
# Status of a candidate that has no row in reactions.jsonl.
STATUS_MISSING: str = "missing"
SECONDS_PER_H: float = 3600.0
# Major outcome when no product clears the yield cutoff (E2).
NO_REACTION: str = "no_reaction"
# Rankers are compared on their three best products (E3).
TOP_N: int = 3


def species_energies(
    species_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """
    Extract electronic energies in eV for ok species.

    :param species_rows: Mapping of species keys to serialized record dicts.
    :type species_rows: Mapping[str, Mapping[str, Any]]
    :return: Mapping of species key to electronic energy in eV.
    :rtype: dict[str, float]
    """
    return {
        key: float(row["energy_ev"])
        for key, row in species_rows.items()
        if row["status"] == STATUS_OK
    }


def energy_change_kcal(
    start_ev: float | None,
    end_ev: float | None,
) -> float | None:
    """
    Return end minus start in kcal/mol, or None when either is missing.

    :param start_ev: Starting energy in eV, or None.
    :type start_ev: float or None
    :param end_ev: Final energy in eV, or None.
    :type end_ev: float or None
    :return: Energy change in kcal/mol, or None.
    :rtype: float or None
    """
    if start_ev is None or end_ev is None:
        return None
    return (end_ev - start_ev) / KCAL_MOL_EV


def confirmed_ts_values(
    row: Mapping[str, Any] | None,
    label: str,
) -> tuple[float | None, float | None]:
    """
    Return E_TS and G_TS of a confirmed reaction row, else (None, None).

    :param row: Reaction record, or None when the reaction has no row.
    :type row: Mapping[str, Any] or None
    :param label: Temperature label of the wanted G_TS.
    :type label: str
    :return: TS electronic energy and free energy in eV.
    :rtype: tuple[float or None, float or None]
    """
    g_ts = confirmed_ts_energy(row, label)
    if row is None or g_ts is None:
        return None, None
    return float(row["ts_energy_ev"]), g_ts


def reaction_energies(
    candidate: Candidate,
    row: Mapping[str, Any] | None,
    species_e: Mapping[str, float],
    species_g: Mapping[str, float],
    label: str,
) -> dict[str, Any]:
    """
    Compute reaction and activation energies of the reacting molecules.

    Bystanders cancel, so only reactant_keys and product_keys count.
    state_free_energy is a plain sum, so it serves E as well as G.

    :param candidate: Candidate reaction item.
    :type candidate: Candidate
    :param row: Transition state search record mapping or None.
    :type row: Mapping[str, Any] or None
    :param species_e: Mapping of species keys to electronic energy in eV.
    :type species_e: Mapping[str, float]
    :param species_g: Mapping of species keys to free energy in eV.
    :type species_g: Mapping[str, float]
    :param label: Temperature label of the G values.
    :type label: str
    :return: Reaction id, status, and the ENERGY_FIELDS in kcal/mol.
    :rtype: dict[str, Any]
    """
    e_react = state_free_energy(candidate.reactant_keys, species_e)
    g_react = state_free_energy(candidate.reactant_keys, species_g)
    e_prod = state_free_energy(candidate.product_keys, species_e)
    g_prod = state_free_energy(candidate.product_keys, species_g)
    e_ts, g_ts = confirmed_ts_values(row, label)
    return {
        "reaction_id": candidate.reaction_id,
        "status": STATUS_MISSING if row is None else row["status"],
        "de_rxn_kcal": energy_change_kcal(e_react, e_prod),
        "dg_rxn_kcal": energy_change_kcal(g_react, g_prod),
        "de_act_kcal": energy_change_kcal(e_react, e_ts),
        "dg_act_kcal": energy_change_kcal(g_react, g_ts),
    }


def step_table(
    candidate_set: CandidateSet,
    paths: RefinePaths,
    temperature_k: float,
) -> list[dict[str, Any]]:
    """
    Generate evaluation table of reaction energies across candidate set.

    :param candidate_set: Extracted candidate reaction set.
    :type candidate_set: CandidateSet
    :param paths: Refinement result filesystem paths.
    :type paths: RefinePaths
    :param temperature_k: Reaction temperature in Kelvin.
    :type temperature_k: float
    :return: List of evaluated reaction energy records with temperature.
    :rtype: list[dict[str, Any]]
    """
    species_rows = latest_rows(paths.species_jsonl, "key")
    reaction_rows = latest_rows(paths.reactions_jsonl, "reaction_id")
    species_e = species_energies(species_rows)
    species_g = species_free_energies(species_rows, temperature_k)
    label = temperature_label(temperature_k)
    return [
        {
            "temperature_k": temperature_k,
            **reaction_energies(
                cand,
                reaction_rows.get(cand.reaction_id),
                species_e,
                species_g,
                label,
            ),
        }
        for cand in candidate_set.candidates
    ]


def field_error(
    engine_rows: Sequence[Mapping[str, Any]],
    ref_by_id: Mapping[str, Mapping[str, Any]],
    field: str,
) -> tuple[float | None, int]:
    """
    Calculate mean absolute error and count for an energy field.

    :param engine_rows: Sequence of engine evaluation rows.
    :type engine_rows: Sequence[Mapping[str, Any]]
    :param ref_by_id: Mapping of reaction identifier to reference record.
    :type ref_by_id: Mapping[str, Mapping[str, Any]]
    :param field: Field name being compared.
    :type field: str
    :return: Tuple of mean absolute error (or None) and count.
    :rtype: tuple[float or None, int]
    """
    pairs = [
        (row[field], ref_by_id[row["reaction_id"]][field])
        for row in engine_rows
        if row["reaction_id"] in ref_by_id
    ]
    diffs = [
        abs(value - ref)
        for value, ref in pairs
        if value is not None and ref is not None
    ]
    if not diffs:
        return None, 0
    return sum(diffs) / len(diffs), len(diffs)


def step_errors(
    engine_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Compute mean absolute errors against reference and transition state success.

    :param engine_rows: Sequence of engine evaluation rows.
    :type engine_rows: Sequence[Mapping[str, Any]]
    :param reference_rows: Sequence of reference evaluation rows.
    :type reference_rows: Sequence[Mapping[str, Any]]
    :return: Summary metrics mapping containing MAE, sample counts, and success.
    :rtype: dict[str, Any]
    """
    ref_by_id = {row["reaction_id"]: row for row in reference_rows}
    out: dict[str, Any] = {}
    for field in ENERGY_FIELDS:
        mae, count = field_error(engine_rows, ref_by_id, field)
        out[f"{field}_mae"] = mae
        out[f"{field}_n"] = count
    confirmed = sum(
        row["status"] == TSStatus.confirmed.value for row in engine_rows
    )
    out["ts_success"] = confirmed / len(engine_rows) if engine_rows else None
    return out


def read_ts_frames(path: Path) -> dict[str, Atoms]:
    """
    Read transition state frames from extxyz file mapped by reaction id.

    :param path: Path to extxyz file containing transition state geometries.
    :type path: Path
    :return: Mapping of reaction identifier to Atoms object.
    :rtype: dict[str, Atoms]
    """
    raw_frames = ase.io.read(path, index=":")
    frames: list[Atoms] = raw_frames  # type: ignore[assignment]
    return {
        str(frame.info["reaction_id"]): frame
        for frame in frames
        if "reaction_id" in frame.info
    }


def frame_rmsd(target: Atoms, atoms: Atoms) -> float:
    """
    Compute RMSD in Angstrom between structures after optimal rigid alignment.

    :param target: Reference atomic configuration.
    :type target: Atoms
    :param atoms: Mobile atomic configuration to align to target.
    :type atoms: Atoms
    :return: Root-mean-square deviation of Cartesian coordinates in Angstrom.
    :rtype: float
    """
    mobile = atoms.copy()
    minimize_rotation_and_translation(target, mobile)
    delta = mobile.positions - target.positions
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))


def ts_rmsd(engine_ts: Path, reference_ts: Path) -> dict[str, float]:
    """
    Compute transition state RMSDs between engine and reference geometries.

    :param engine_ts: Path to engine transition states extxyz file.
    :type engine_ts: Path
    :param reference_ts: Path to reference transition states extxyz file.
    :type reference_ts: Path
    :return: Mapping of reaction identifier to aligned RMSD in Angstrom.
    :rtype: dict[str, float]
    """
    if not engine_ts.is_file() or not reference_ts.is_file():
        return {}
    eng_frames = read_ts_frames(engine_ts)
    ref_frames = read_ts_frames(reference_ts)
    return {
        rid: frame_rmsd(ref_frames[rid], eng_frames[rid])
        for rid in eng_frames
        if rid in ref_frames
    }


def cost_row(engine: str, calls_csv: Path) -> dict[str, Any]:
    """
    Summarize computational cost and error metrics from call log CSV.

    :param engine: Computational chemistry engine identifier.
    :type engine: str
    :param calls_csv: Filesystem path to engine call performance log.
    :type calls_csv: Path
    :return: Mapping of computational cost, call counts, and error frequency.
    :rtype: dict[str, Any]
    """
    rows = read_calls(calls_csv)
    tasks = Counter(row["task"] for row in rows)
    return {
        "engine": engine,
        "wall_h": sum(float(row["wall_s"]) for row in rows) / SECONDS_PER_H,
        "force_calls": sum(int(row["force_calls"]) for row in rows),
        "species_calls": tasks["species"],
        "reaction_calls": tasks["reaction"],
        "engine_errors": sum(
            row["status"] == STATUS_ENGINE_ERROR for row in rows
        ),
    }


def load_yield_cutoff(search_path: str | Path) -> float:
    """
    Load R4 product yield cutoff fraction from search configuration.

    :param search_path: Path to search YAML configuration file.
    :type search_path: str or Path
    :return: Yield cutoff threshold fraction.
    :rtype: float
    :raises KeyError: If evaluate section or yield_cutoff is missing.
    """
    with Path(search_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return float(data["evaluate"]["yield_cutoff"])


def ranker_rows(
    ranks: Sequence[Mapping[str, Any]],
    ranker: str,
    temperature_k: float,
) -> list[dict[str, Any]]:
    """
    Extract and sort ranker rows for a specified evaluation temperature.

    :param ranks: Sequence of ranked product records.
    :type ranks: Sequence[Mapping[str, Any]]
    :param ranker: Ranker strategy name (e.g. 'R0', 'R4').
    :type ranker: str
    :param temperature_k: Evaluation temperature in Kelvin.
    :type temperature_k: float
    :return: Filtered rows sorted by rank position.
    :rtype: list[dict[str, Any]]
    """
    filtered = [
        dict(row)
        for row in ranks
        if row["ranker"] == ranker
        and float(row["temperature_k"]) == float(temperature_k)
    ]
    return sorted(filtered, key=lambda row: int(row["position"]))


def r4_yields(
    ranks: Sequence[Mapping[str, Any]],
    temperature_k: float,
) -> dict[str, float]:
    """
    Map product states to their kinetic yield scores under ranker R4.

    :param ranks: Sequence of ranked product records.
    :type ranks: Sequence[Mapping[str, Any]]
    :param temperature_k: Evaluation temperature in Kelvin.
    :type temperature_k: float
    :return: Mapping of product state identifier to R4 yield score.
    :rtype: dict[str, float]
    """
    return {
        str(row["state"]): float(row["score"])
        for row in ranker_rows(ranks, "R4", temperature_k)
    }


def compute_jaccard(
    set_a: set[str],
    set_b: set[str],
) -> float | None:
    """
    Compute Jaccard similarity index between two sets of states.

    :param set_a: First set of string states.
    :type set_a: set[str]
    :param set_b: Second set of string states.
    :type set_b: set[str]
    :return: Jaccard similarity coefficient or None if union is empty.
    :rtype: float or None
    """
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def compute_kendall_tau(
    eng_yields: Mapping[str, float],
    ref_yields: Mapping[str, float],
    states: set[str],
) -> float | None:
    """
    Compute Kendall tau correlation of yields over state union.

    :param eng_yields: Mapping of engine product state to kinetic yield.
    :type eng_yields: Mapping[str, float]
    :param ref_yields: Mapping of reference product state to kinetic yield.
    :type ref_yields: Mapping[str, float]
    :param states: Union set of evaluated product states.
    :type states: set[str]
    :return: Kendall tau correlation coefficient or None if undefined.
    :rtype: float or None
    """
    if len(states) < 2:
        return None
    sorted_states = sorted(states)
    eng_vals = [eng_yields.get(s, 0.0) for s in sorted_states]
    ref_vals = [ref_yields.get(s, 0.0) for s in sorted_states]
    tau_res = scipy.stats.kendalltau(eng_vals, ref_vals)
    val = float(tau_res[0])
    return None if np.isnan(val) else val


def top_position_match(
    eng_rows: Sequence[Mapping[str, Any]],
    ref_rows: Sequence[Mapping[str, Any]],
    field: str,
) -> bool | None:
    """
    Check if position-1 records match between two ranker row sequences.

    :param eng_rows: Sequence of engine ranked rows.
    :type eng_rows: Sequence[Mapping[str, Any]]
    :param ref_rows: Sequence of reference ranked rows.
    :type ref_rows: Sequence[Mapping[str, Any]]
    :param field: Field identifier to compare for equality.
    :type field: str
    :return: True if position-1 fields match, False if different, None if empty.
    :rtype: bool or None
    """
    if not eng_rows or not ref_rows:
        return None
    # ranker_rows sorts by position, so the first row is position 1.
    return bool(eng_rows[0][field] == ref_rows[0][field])


def major_product(yields: Mapping[str, float], yield_cutoff: float) -> str:
    """
    Return the highest-yield product, or NO_REACTION when none is formed.

    R4 lists products only, never the start state, so a trace product
    would otherwise count as "major" while almost nothing reacts.

    :param yields: Product state -> R4 yield at the observation time.
    :type yields: Mapping[str, float]
    :param yield_cutoff: Yield a state needs to count as a product.
    :type yield_cutoff: float
    :return: The major product state id, or NO_REACTION.
    :rtype: str
    """
    formed = {
        state: val for state, val in yields.items() if val >= yield_cutoff
    }
    if not formed:
        return NO_REACTION
    return max(formed, key=formed.__getitem__)


def network_agreement(
    engine_ranks: Sequence[Mapping[str, Any]],
    reference_ranks: Sequence[Mapping[str, Any]],
    temperature_k: float,
    yield_cutoff: float,
) -> dict[str, Any]:
    """
    Evaluate network agreement metrics against reference ranking results.

    :param engine_ranks: Candidate engine ranked product records.
    :type engine_ranks: Sequence[Mapping[str, Any]]
    :param reference_ranks: Reference engine ranked product records.
    :type reference_ranks: Sequence[Mapping[str, Any]]
    :param temperature_k: Evaluation temperature in Kelvin.
    :type temperature_k: float
    :param yield_cutoff: Kinetic yield threshold defining product states.
    :type yield_cutoff: float
    :return: Dictionary containing Jaccard, Kendall tau, and top matches.
    :rtype: dict[str, Any]
    """
    eng_y = r4_yields(engine_ranks, temperature_k)
    ref_y = r4_yields(reference_ranks, temperature_k)
    prods_eng = {s for s, y in eng_y.items() if y >= yield_cutoff}
    prods_ref = {s for s, y in ref_y.items() if y >= yield_cutoff}
    union = prods_eng | prods_ref
    major = major_product(eng_y, yield_cutoff)
    ref_major = major_product(ref_y, yield_cutoff)
    r3_eng = ranker_rows(engine_ranks, "R3", temperature_k)
    r3_ref = ranker_rows(reference_ranks, "R3", temperature_k)
    return {
        "jaccard": compute_jaccard(prods_eng, prods_ref),
        "kendall_tau": compute_kendall_tau(eng_y, ref_y, union),
        "major_product": major,
        "reference_major_product": ref_major,
        "same_major_product": major == ref_major,
        "same_top_path": top_position_match(r3_eng, r3_ref, "path"),
    }


def single_ranker_agreement(
    ranker: str,
    ranker_rows_list: Sequence[Mapping[str, Any]],
    r4_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Score a single ranker strategy against R4 kinetic yield ranking.

    :param ranker: Ranker strategy name (e.g. 'R0').
    :type ranker: str
    :param ranker_rows_list: Ranked rows for this ranker.
    :type ranker_rows_list: Sequence[Mapping[str, Any]]
    :param r4_rows: Reference kinetic yield R4 ranked rows.
    :type r4_rows: Sequence[Mapping[str, Any]]
    :return: Agreement record with top1 match and top3 overlap.
    :rtype: dict[str, Any]
    """
    top1 = top_position_match(ranker_rows_list, r4_rows, "state")
    top3_r4 = {row["state"] for row in r4_rows[:TOP_N]}
    top3_r = {row["state"] for row in ranker_rows_list[:TOP_N]}
    overlap = len(top3_r & top3_r4) / len(top3_r4) if top3_r4 else None
    return {"ranker": ranker, "top1_agree": top1, "top3_overlap": overlap}


def ranker_agreement(
    reference_ranks: Sequence[Mapping[str, Any]],
    temperature_k: float,
) -> list[dict[str, Any]]:
    """
    Compare rankers R0 through R3 against R4 on reference network rankings.

    :param reference_ranks: Sequence of reference ranked product records.
    :type reference_ranks: Sequence[Mapping[str, Any]]
    :param temperature_k: Evaluation temperature in Kelvin.
    :type temperature_k: float
    :return: List of agreement records for rankers R0 through R3.
    :rtype: list[dict[str, Any]]
    """
    r4_rows = ranker_rows(reference_ranks, "R4", temperature_k)
    return [
        single_ranker_agreement(
            ranker,
            ranker_rows(reference_ranks, ranker, temperature_k),
            r4_rows,
        )
        for ranker in ("R0", "R1", "R2", "R3")
    ]


def ts_rmsd_summary(
    eng_ts: Path,
    ref_ts: Path,
) -> tuple[float | None, int]:
    """
    Compute mean transition state RMSD and valid geometry sample count.

    :param eng_ts: Path to evaluated engine TS extxyz file.
    :type eng_ts: Path
    :param ref_ts: Path to reference engine TS extxyz file.
    :type ref_ts: Path
    :return: Tuple of mean RMSD in Angstrom (or None) and matched count.
    :rtype: tuple[float or None, int]
    """
    rmsd_map = ts_rmsd(eng_ts, ref_ts)
    if not rmsd_map:
        return None, 0
    return float(np.mean(list(rmsd_map.values()))), len(rmsd_map)


def load_engine_ranks(
    results_dir: str | Path,
    system: str,
    engine: str,
) -> list[dict[str, Any]]:
    """
    Load serialized ranking records for a given chemical engine.

    :param results_dir: Base results directory.
    :type results_dir: str or Path
    :param system: Chemical system identifier.
    :type system: str
    :param engine: Chemical engine identifier.
    :type engine: str
    :return: List of ranked product record mappings.
    :rtype: list[dict[str, Any]]
    """
    path = make_rank_dir(results_dir, system, engine) / RANKS_FILE
    if not path.is_file():
        # An unranked engine would look like "no products" and skew E2.
        raise FileNotFoundError(
            f"{path} is missing; run nrnet rank --engine {engine} first"
        )
    return read_jsonl(path)


@dataclass(frozen=True, slots=True)
class EvaluationScope:
    """
    The inputs every evaluation table shares.

    :param candidate_set: Frozen candidate set of the system.
    :type candidate_set: CandidateSet
    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: Chemical system identifier.
    :type system: str
    :param reference: Reference engine identifier.
    :type reference: str
    :param engines: Compared engines, reference excluded.
    :type engines: tuple[str, ...]
    :param temperatures: Thermo temperatures in Kelvin.
    :type temperatures: tuple[float, ...]
    :param yield_cutoff: R4 yield at which a state counts as a product.
    :type yield_cutoff: float
    """

    candidate_set: CandidateSet
    results_dir: Path
    system: str
    reference: str
    engines: tuple[str, ...]
    temperatures: tuple[float, ...]
    yield_cutoff: float

    @property
    def all_engines(self) -> tuple[str, ...]:
        """
        Reference first, then the compared engines.

        :return: Every engine in the evaluation.
        :rtype: tuple[str, ...]
        """
        return (self.reference, *self.engines)

    def refine_paths(self, engine: str) -> RefinePaths:
        """
        Refine result paths of one engine.

        :param engine: Engine identifier.
        :type engine: str
        :return: That engine's refine paths.
        :rtype: RefinePaths
        """
        return make_refine_paths(self.results_dir, self.system, engine)

    def ranks(self, engine: str) -> list[dict[str, Any]]:
        """
        Ranked products of one engine.

        :param engine: Engine identifier.
        :type engine: str
        :return: Rows of that engine's ranks.jsonl.
        :rtype: list[dict[str, Any]]
        """
        return load_engine_ranks(self.results_dir, self.system, engine)


def make_scope(
    system: str,
    reference: str,
    engines: Sequence[str],
    candidates_dir: str | Path,
    results_dir: str | Path,
    search_path: str | Path,
) -> EvaluationScope:
    """
    Load the candidate set and settings into one evaluation scope.

    :param system: Chemical system identifier.
    :type system: str
    :param reference: Reference engine identifier.
    :type reference: str
    :param engines: Compared engines; the reference is dropped if listed.
    :type engines: Sequence[str]
    :param candidates_dir: Root directory of frozen candidate sets.
    :type candidates_dir: str or Path
    :param results_dir: Root results directory.
    :type results_dir: str or Path
    :param search_path: Path to search YAML configuration file.
    :type search_path: str or Path
    :return: The shared inputs of every table.
    :rtype: EvaluationScope
    """
    return EvaluationScope(
        candidate_set=load_candidates(Path(candidates_dir) / system),
        results_dir=Path(results_dir),
        system=system,
        reference=reference,
        engines=tuple(eng for eng in engines if eng != reference),
        temperatures=tuple(load_thermo_temperatures(search_path)),
        yield_cutoff=load_yield_cutoff(search_path),
    )


def build_e1_steps_table(scope: EvaluationScope) -> list[dict[str, Any]]:
    """
    Reaction energies of every engine, reference included (E1).

    :param scope: Shared evaluation inputs.
    :type scope: EvaluationScope
    :return: One row per engine, temperature and reaction.
    :rtype: list[dict[str, Any]]
    """
    return [
        {"engine": engine, **row}
        for engine in scope.all_engines
        for temp in scope.temperatures
        for row in step_table(
            scope.candidate_set, scope.refine_paths(engine), temp
        )
    ]


def engine_summary_rows(
    scope: EvaluationScope,
    engine: str,
) -> list[dict[str, Any]]:
    """
    Errors of one engine against the reference at every temperature.

    :param scope: Shared evaluation inputs.
    :type scope: EvaluationScope
    :param engine: Compared engine identifier.
    :type engine: str
    :return: One summary row per temperature.
    :rtype: list[dict[str, Any]]
    """
    eng_paths = scope.refine_paths(engine)
    ref_paths = scope.refine_paths(scope.reference)
    rmsd_mean, rmsd_n = ts_rmsd_summary(
        eng_paths.ts_extxyz, ref_paths.ts_extxyz
    )
    rows = []
    for temp in scope.temperatures:
        eng_steps = step_table(scope.candidate_set, eng_paths, temp)
        ref_steps = step_table(scope.candidate_set, ref_paths, temp)
        errors = step_errors(eng_steps, ref_steps)
        rows.append({"engine": engine, "temperature_k": temp, **errors})
        rows[-1].update(ts_rmsd_mean=rmsd_mean, ts_rmsd_n=rmsd_n)
    return rows


def build_e1_summary_table(scope: EvaluationScope) -> list[dict[str, Any]]:
    """
    Errors of every compared engine against the reference (E1).

    :param scope: Shared evaluation inputs.
    :type scope: EvaluationScope
    :return: One row per compared engine and temperature.
    :rtype: list[dict[str, Any]]
    """
    return [
        row
        for engine in scope.engines
        for row in engine_summary_rows(scope, engine)
    ]


def build_e2_networks_table(scope: EvaluationScope) -> list[dict[str, Any]]:
    """
    Product agreement of every compared network with the reference (E2).

    :param scope: Shared evaluation inputs.
    :type scope: EvaluationScope
    :return: One row per compared engine and temperature.
    :rtype: list[dict[str, Any]]
    """
    ref_ranks = scope.ranks(scope.reference)
    rows = []
    for engine in scope.engines:
        eng_ranks = scope.ranks(engine)
        for temp in scope.temperatures:
            agree = network_agreement(
                eng_ranks, ref_ranks, temp, scope.yield_cutoff
            )
            rows.append({"engine": engine, "temperature_k": temp, **agree})
    return rows


def build_e3_rankers_table(scope: EvaluationScope) -> list[dict[str, Any]]:
    """
    Agreement of rankers R0 to R3 with R4 on the reference network (E3).

    :param scope: Shared evaluation inputs.
    :type scope: EvaluationScope
    :return: One row per temperature and ranker.
    :rtype: list[dict[str, Any]]
    """
    ref_ranks = scope.ranks(scope.reference)
    return [
        {"temperature_k": temp, **row}
        for temp in scope.temperatures
        for row in ranker_agreement(ref_ranks, temp)
    ]


def build_e4_cost_table(scope: EvaluationScope) -> list[dict[str, Any]]:
    """
    Cost of every engine, reference included (E4).

    :param scope: Shared evaluation inputs.
    :type scope: EvaluationScope
    :return: One row per engine.
    :rtype: list[dict[str, Any]]
    """
    return [
        cost_row(engine, scope.refine_paths(engine).calls_csv)
        for engine in scope.all_engines
    ]


TABLE_BUILDERS: dict[str, Callable[[EvaluationScope], list[dict[str, Any]]]] = {
    "e1_steps.csv": build_e1_steps_table,
    "e1_summary.csv": build_e1_summary_table,
    "e2_networks.csv": build_e2_networks_table,
    "e3_rankers.csv": build_e3_rankers_table,
    "e4_cost.csv": build_e4_cost_table,
}


def evaluate(
    system: str,
    reference: str,
    engines: Sequence[str],
    candidates_dir: str | Path,
    results_dir: str | Path,
    search_path: str | Path,
) -> Path:
    """
    Execute full evaluation pipeline and write comparison tables as CSV.

    :param system: Chemical system identifier.
    :type system: str
    :param reference: Reference engine identifier.
    :type reference: str
    :param engines: Sequence of compared engine identifiers.
    :type engines: Sequence[str]
    :param candidates_dir: Root directory of frozen candidate sets.
    :type candidates_dir: str or Path
    :param results_dir: Root results directory.
    :type results_dir: str or Path
    :param search_path: Path to search YAML configuration file.
    :type search_path: str or Path
    :return: Destination evaluation output directory.
    :rtype: Path
    """
    scope = make_scope(
        system, reference, engines, candidates_dir, results_dir, search_path
    )
    out_dir = scope.results_dir / "evaluate" / system
    for name, builder in TABLE_BUILDERS.items():
        write_csv(out_dir / name, builder(scope))
    return out_dir
