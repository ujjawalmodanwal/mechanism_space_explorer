"""
Compare two engines' kept saddles on the reactions both have confirmed.

Reads the two refine summaries (scripts/checks/refine_summary.py), the
reaction records and the saddle geometries, then writes one row per shared
confirmed reaction and ranking temperature: both forward free energy
barriers, the electronic barriers from the reactant and product complexes,
both imaginary frequencies, and the saddle RMSD after a Kabsch fit on the
same atom order. A small RMSD with a large barrier gap means the two
potentials disagree on the same saddle, not that they found different ones.
The RMSD does not try atom permutations, so a swap of equivalent atoms
(two H on one carbon) can make the same saddle look far apart. A confirmed
reaction without a barrier (a species that is not a minimum) gets blanks.
"""

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from nrnet.store import latest_rows, make_refine_paths
from nrnet.ts_search import TSStatus

__all__ = [
    "EV_TO_KCAL",
    "PAIR_CSV_FIELDNAMES",
    "build_parser",
    "kabsch_rmsd",
    "last_saddles",
    "main",
    "pair_rows",
]

EV_TO_KCAL = 23.060548

PAIR_CSV_FIELDNAMES = (
    "reaction_id",
    "temperature_k",
    "literature_name",
    "dg_act_a_kcal",
    "dg_act_b_kcal",
    "dg_act_diff_kcal",
    "de_ts_rcx_diff_kcal",
    "de_ts_pcx_diff_kcal",
    "imaginary_a_cm",
    "imaginary_b_cm",
    "rmsd_a",
)


def kabsch_rmsd(first: Atoms, second: Atoms) -> float:
    """
    RMSD of two geometries with the same atom order after the best rotation.

    :param first: First geometry.
    :type first: Atoms
    :param second: Second geometry, same atoms in the same order.
    :type second: Atoms
    :return: RMSD in angstrom.
    :rtype: float
    """
    p = first.positions - first.positions.mean(axis=0)
    q = second.positions - second.positions.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return float(np.sqrt(((p @ rotation - q) ** 2).sum(axis=1).mean()))


def last_saddles(path: Path) -> dict[str, Atoms]:
    """
    Map reaction identifier to its last saddle frame in an extxyz file.

    :param path: Path to ts.extxyz.
    :type path: Path
    :return: Mapping of reaction identifier to geometry.
    :rtype: dict[str, Atoms]
    """
    if not path.is_file():
        return {}
    return {atoms.info["reaction_id"]: atoms for atoms in read(path, ":")}


def summary_by_key(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """
    Map (reaction, temperature) to its row in a refine summary CSV.

    :param path: Path to data/<system>/checks/refine_<engine>.csv.
    :type path: Path
    :return: Mapping of (reaction_id, temperature_k) to row.
    :rtype: dict[tuple[str, str], dict[str, str]]
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["reaction_id"], row["temperature_k"]): row
            for row in csv.DictReader(handle)
        }


def complex_barriers(record: Mapping[str, Any]) -> tuple[float, float]:
    """
    Electronic barriers from the reactant and product complexes.

    :param record: Latest reaction record of one engine.
    :type record: Mapping[str, Any]
    :return: TS minus reactant complex, TS minus product complex, kcal/mol.
    :rtype: tuple[float, float]
    """
    ts = record["ts_energy_ev"]
    return (
        (ts - record["reactant_complex_ev"]) * EV_TO_KCAL,
        (ts - record["product_complex_ev"]) * EV_TO_KCAL,
    )


def pair_rows(
    system: str,
    engines: Sequence[str],
    results_dir: Path,
    data_dir: Path,
) -> list[dict[str, Any]]:
    """
    Build one row per shared confirmed reaction and ranking temperature.

    :param system: Chemical system identifier.
    :type system: str
    :param engines: The two engines, A then B; differences are B minus A.
    :type engines: Sequence[str]
    :param results_dir: Base refinement results directory.
    :type results_dir: Path
    :param data_dir: Base data directory holding the refine summaries.
    :type data_dir: Path
    :return: Rows keyed by PAIR_CSV_FIELDNAMES.
    :rtype: list[dict[str, Any]]
    """
    checks = data_dir / system / "checks"
    summaries = [summary_by_key(checks / f"refine_{e}.csv") for e in engines]
    paths = [make_refine_paths(results_dir, system, e) for e in engines]
    records = [latest_rows(p.reactions_jsonl, "reaction_id") for p in paths]
    saddles = [last_saddles(p.ts_extxyz) for p in paths]
    confirmed = TSStatus.confirmed.value
    rows: list[dict[str, Any]] = []
    for key, row_a in sorted(summaries[0].items()):
        row_b = summaries[1].get(key)
        if row_b is None or confirmed != row_a["status"] or (
            confirmed != row_b["status"]
        ):
            continue
        rid = key[0]
        rcx_a, pcx_a = complex_barriers(records[0][rid])
        rcx_b, pcx_b = complex_barriers(records[1][rid])
        geo_a, geo_b = saddles[0][rid], saddles[1][rid]
        same_atoms = list(geo_a.symbols) == list(geo_b.symbols)
        dg_a = row_a["dg_act_fwd_kcal"]
        dg_b = row_b["dg_act_fwd_kcal"]
        gap = round(float(dg_b) - float(dg_a), 2) if dg_a and dg_b else ""
        rows.append({
            "reaction_id": rid,
            "temperature_k": key[1],
            "literature_name": row_a["literature_name"],
            "dg_act_a_kcal": dg_a,
            "dg_act_b_kcal": dg_b,
            "dg_act_diff_kcal": gap,
            "de_ts_rcx_diff_kcal": round(rcx_b - rcx_a, 2),
            "de_ts_pcx_diff_kcal": round(pcx_b - pcx_a, 2),
            "imaginary_a_cm": row_a["imaginary_cm"],
            "imaginary_b_cm": row_b["imaginary_cm"],
            "rmsd_a": (
                round(kabsch_rmsd(geo_a, geo_b), 3) if same_atoms else ""
            ),
        })
    return rows


def build_parser() -> argparse.ArgumentParser:
    """
    Configure the command-line parser.

    :return: Argument parser for the pair comparison script.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, help="System identifier.")
    parser.add_argument(
        "--engines", nargs=2, required=True, metavar=("A", "B"),
        help="Two engine identifiers; differences are B minus A.",
    )
    parser.add_argument("--results-dir", type=Path, default="results")
    parser.add_argument("--data-dir", type=Path, default="data")
    return parser


def main() -> None:
    """
    Write data/<system>/checks/pair_<A>_<B>.csv and print the largest gap.

    :return: None.
    :rtype: None
    """
    args = build_parser().parse_args()
    rows = pair_rows(args.system, args.engines, args.results_dir,
                     args.data_dir)
    first, second = args.engines
    out = args.data_dir / args.system / "checks" / f"pair_{first}_{second}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    shared = sorted({row["reaction_id"] for row in rows})
    gaps = [abs(row["dg_act_diff_kcal"]) for row in rows
            if row["dg_act_diff_kcal"] != ""]
    largest = max(gaps, default=0)
    print(f"{args.system} {first} vs {second}: {len(shared)} shared "
          f"confirmed, largest |dG_act gap| {largest} kcal/mol")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
