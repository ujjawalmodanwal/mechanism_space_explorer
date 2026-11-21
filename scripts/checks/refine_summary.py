"""
Summarize one engine's refinement of a frozen candidate set, reaction by row.

Reads the frozen candidates and the engine's species and reaction records,
then writes one row per reaction and ranking temperature: search status,
route, imaginary frequency, and forward, reverse and reaction free energies
in kcal/mol. Barriers follow the network's rule: G_TS minus the G of the
reacting molecules only. Species statuses are printed as a count.
"""

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nrnet.assemble import state_free_energy
from nrnet.nanoreactor.candidate_store import load_candidates
from nrnet.nanoreactor.candidates import Candidate, CandidateSet
from nrnet.refine import load_thermo_temperatures
from nrnet.store import latest_rows, make_refine_paths, temperature_label
from nrnet.ts_search import TSStatus

__all__ = [
    "EV_TO_KCAL",
    "SUMMARY_CSV_FIELDNAMES",
    "build_parser",
    "free_energy_terms",
    "main",
    "species_smiles",
    "summary_row",
    "summary_rows",
    "write_summary_csv",
]

EV_TO_KCAL: float = 23.060548
SUMMARY_CSV_FIELDNAMES: tuple[str, ...] = (
    "reaction_id",
    "temperature_k",
    "source",
    "literature_name",
    "n_events",
    "reactants",
    "products",
    "status",
    "route",
    "seed_engine",
    "attempts",
    "imaginary_cm",
    "dg_act_fwd_kcal",
    "dg_act_rev_kcal",
    "dg_rxn_kcal",
    "note",
)


def species_smiles(keys: Sequence[str], candidate_set: CandidateSet) -> str:
    """
    Join the SMILES of a state's species, falling back to the species key.

    :param keys: Species keys of one side of a reaction.
    :type keys: Sequence[str]
    :param candidate_set: Frozen candidate set holding species labels.
    :type candidate_set: CandidateSet
    :return: SMILES joined by ' + '.
    :rtype: str
    """
    return " + ".join(candidate_set.species[key].smiles or key for key in keys)


def free_energy_terms(
    candidate: Candidate,
    row: Mapping[str, Any] | None,
    species_g: Mapping[str, float],
    label: str,
) -> tuple[float | None, float | None, float | None]:
    """
    Compute forward barrier, reverse barrier and reaction free energy.

    :param candidate: Frozen candidate reaction.
    :type candidate: Candidate
    :param row: Latest reaction record of the engine, or None.
    :type row: Mapping[str, Any] or None
    :param species_g: Species free energies in eV at one temperature.
    :type species_g: Mapping[str, float]
    :param label: Temperature label of the free energy maps.
    :type label: str
    :return: Forward and reverse barriers and dG_rxn in kcal/mol, or None.
    :rtype: tuple[float or None, float or None, float or None]
    """
    g_react = state_free_energy(candidate.reactant_keys, species_g)
    g_prod = state_free_energy(candidate.product_keys, species_g)
    dg_rxn = None
    if g_react is not None and g_prod is not None:
        dg_rxn = (g_prod - g_react) * EV_TO_KCAL
    confirmed = row is not None and row["status"] == TSStatus.confirmed
    if not confirmed or g_react is None or g_prod is None:
        return None, None, dg_rxn
    g_ts = float(row["g_ts_ev"][label])
    return (g_ts - g_react) * EV_TO_KCAL, (g_ts - g_prod) * EV_TO_KCAL, dg_rxn


def summary_row(
    candidate: Candidate,
    row: Mapping[str, Any] | None,
    species_g: Mapping[str, float],
    temperature_k: float,
    candidate_set: CandidateSet,
) -> dict[str, Any]:
    """
    Build one summary row for a candidate reaction at one temperature.

    :param candidate: Frozen candidate reaction.
    :type candidate: Candidate
    :param row: Latest reaction record of the engine, or None if not run.
    :type row: Mapping[str, Any] or None
    :param species_g: Species free energies in eV at this temperature.
    :type species_g: Mapping[str, float]
    :param temperature_k: Ranking temperature in kelvin.
    :type temperature_k: float
    :param candidate_set: Frozen candidate set holding species labels.
    :type candidate_set: CandidateSet
    :return: Summary row keyed by SUMMARY_CSV_FIELDNAMES.
    :rtype: dict[str, Any]
    """
    label = temperature_label(temperature_k)
    fwd, rev, rxn = free_energy_terms(candidate, row, species_g, label)
    record = row or {}
    return {
        "reaction_id": candidate.reaction_id,
        "temperature_k": temperature_k,
        "source": candidate.source,
        "literature_name": candidate.literature_name or "",
        "n_events": candidate.n_events,
        "reactants": species_smiles(candidate.reactant_keys, candidate_set),
        "products": species_smiles(candidate.product_keys, candidate_set),
        "status": record.get("status", "not_run"),
        "route": record.get("route") or "",
        "seed_engine": record.get("seed_engine") or "",
        "attempts": ";".join(record.get("attempts", ())),
        "imaginary_cm": record.get("imaginary_cm"),
        "dg_act_fwd_kcal": fwd,
        "dg_act_rev_kcal": rev,
        "dg_rxn_kcal": rxn,
        "note": record.get("note", ""),
    }


def summary_rows(
    system: str,
    engine: str,
    candidates_dir: Path,
    results_dir: Path,
    systems_dir: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """
    Build summary rows for every candidate and ranking temperature.

    :param system: System identifier.
    :type system: str
    :param engine: Engine identifier whose refinement is summarized.
    :type engine: str
    :param candidates_dir: Root directory of frozen candidate sets.
    :type candidates_dir: Path
    :param results_dir: Root directory of refinement results.
    :type results_dir: Path
    :param systems_dir: Root directory of system specifications.
    :type systems_dir: Path
    :return: Summary rows and the count of species statuses.
    :rtype: tuple[list[dict[str, Any]], Counter[str]]
    """
    candidate_set = load_candidates(candidates_dir / system)
    paths = make_refine_paths(results_dir, system, engine)
    species_rows = latest_rows(paths.species_jsonl, "key")
    reaction_rows = latest_rows(paths.reactions_jsonl, "reaction_id")
    temperatures = load_thermo_temperatures(systems_dir / system)
    rows: list[dict[str, Any]] = []
    for temperature_k in temperatures:
        label = temperature_label(temperature_k)
        species_g = {
            key: float(row["g_ev"][label])
            for key, row in species_rows.items()
            if row["status"] == "ok"
        }
        rows.extend(
            summary_row(
                candidate,
                reaction_rows.get(candidate.reaction_id),
                species_g,
                temperature_k,
                candidate_set,
            )
            for candidate in candidate_set.candidates
        )
    statuses = Counter(row["status"] for row in species_rows.values())
    return rows, statuses


def write_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """
    Write summary rows to CSV, rounding free energies to 0.01 kcal/mol.

    :param rows: Summary rows keyed by SUMMARY_CSV_FIELDNAMES.
    :type rows: Sequence[Mapping[str, Any]]
    :param path: Destination CSV path.
    :type path: Path
    :return: None.
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: round(value, 2) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def build_parser() -> argparse.ArgumentParser:
    """
    Configure the command-line parser.

    :return: Argument parser for the summary script.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, help="System identifier.")
    parser.add_argument("--engine", required=True, help="Engine identifier.")
    parser.add_argument("--candidates-dir", type=Path, default="candidates")
    parser.add_argument("--results-dir", type=Path, default="results")
    parser.add_argument("--systems-dir", type=Path, default="systems")
    parser.add_argument("--data-dir", type=Path, default="data")
    return parser


def main() -> None:
    """
    Write data/<system>/checks/refine_<engine>.csv and print a status count.

    :return: None.
    :rtype: None
    """
    args = build_parser().parse_args()
    rows, species_statuses = summary_rows(
        args.system,
        args.engine,
        args.candidates_dir,
        args.results_dir,
        args.systems_dir,
    )
    out = args.data_dir / args.system / "checks" / f"refine_{args.engine}.csv"
    write_summary_csv(rows, out)
    first_t = rows[0]["temperature_k"] if rows else None
    reactions = Counter(
        row["status"] for row in rows if row["temperature_k"] == first_t
    )
    print(f"{args.system} {args.engine}: species {dict(species_statuses)}")
    print(f"{args.system} {args.engine}: reactions {dict(reactions)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
