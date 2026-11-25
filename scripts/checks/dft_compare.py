"""
Compare each engine's barriers with DFT on the same structures (PLAN D23).

One row per system, confirmed reaction and engine. Barriers are measured
from the separated molecules of each side of the reaction signature, in
kcal/mol: the engine's own numbers, the DFT numbers on the same engine's
geometries (results/dft_sp), and their difference. Free energy barriers
add the engine's own G - E to the DFT energies, at each ranking
temperature. The DFT force on the engine's saddle (largest atom force)
shows how far DFT would move it. From the IRC paths (results/irc and
results/dft_path) come the highest DFT energy on the engine's path, how
far it sits above the DFT energy at the engine's saddle, and how many
frames away. Per reaction, ``bound_best`` is the lowest path peak over
all engines, an upper limit to the DFT saddle energy, and
``saddle_sp_best`` the lowest DFT energy at any engine's saddle, an
estimate. Both are measured from the lowest DFT energy of each reactant
molecule over the engines' geometries; since molecules are not relaxed
with DFT, that reference sits slightly above the DFT minimum.
Writes data/<system>/checks/dft_compare.csv.
"""

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dft_single_points import DEFAULT_ENGINES, DEFAULT_SYSTEMS, points_path

from nrnet.store import latest_rows, make_refine_paths
from nrnet.ts_search import TSStatus

__all__ = [
    "EV_TO_KCAL",
    "EngineData",
    "build_parser",
    "compare_rows",
    "load_engine_data",
    "main",
]

EV_TO_KCAL = 23.060548


def jsonl_rows(path: Path, field: str) -> dict[str, dict[str, Any]]:
    """
    Latest row per id, or nothing when the file is missing.

    :param path: Path to a JSON Lines file.
    :type path: Path
    :param field: Key holding the id.
    :type field: str
    :return: Mapping of id to its latest row.
    :rtype: dict[str, dict[str, Any]]
    """
    return latest_rows(path, field) if path.is_file() else {}


@dataclass(frozen=True, slots=True)
class EngineData:
    """
    Everything stored for one system and engine.

    :param reactions: Latest refine row per reaction id.
    :param species: Latest ok refine row per species key.
    :param dft: Latest DFT row per uid (``ts:R001``, ``species:<key>``).
    :param paths: Latest IRC row per reaction id.
    :param path_dft: DFT energy (eV) per frame index, per reaction id.
    """

    reactions: dict[str, dict[str, Any]]
    species: dict[str, dict[str, Any]]
    dft: dict[str, dict[str, Any]]
    paths: dict[str, dict[str, Any]]
    path_dft: dict[str, dict[int, float]]


def load_engine_data(results_dir: Path, system: str, engine: str) -> EngineData:
    """
    Read the refine, DFT, IRC and path files of one system and engine.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: The engine's data.
    :rtype: EngineData
    """
    refine = make_refine_paths(results_dir, system, engine)
    path_rows = jsonl_rows(
        results_dir / "dft_path" / system / engine / "points.jsonl", "uid"
    )
    path_dft: dict[str, dict[int, float]] = {}
    for row in path_rows.values():
        if row["ok"]:
            rid, index = row["id"].rsplit("_", 1)
            path_dft.setdefault(rid, {})[int(index)] = row["energy_ev"]
    return EngineData(
        reactions=jsonl_rows(refine.reactions_jsonl, "reaction_id"),
        species={
            key: row
            for key, row in jsonl_rows(refine.species_jsonl, "key").items()
            if row["status"] == "ok"
        },
        dft=jsonl_rows(points_path(results_dir, system, engine), "uid"),
        paths=jsonl_rows(
            results_dir / "irc" / system / engine / "paths.jsonl",
            "reaction_id",
        ),
        path_dft=path_dft,
    )


def side_sum(
    values: Mapping[str, float | None], keys: Sequence[str]
) -> float | None:
    """
    Sum one value per molecule of a reaction side.

    :param values: Value per species key.
    :type values: Mapping[str, float or None]
    :param keys: Species keys of the side, repeats included.
    :type keys: Sequence[str]
    :return: The sum, or None if any molecule has no value.
    :rtype: float or None
    """
    parts = [values.get(key) for key in keys]
    return None if any(part is None for part in parts) else sum(parts)


def kcal(top: float | None, bottom: float | None) -> float | None:
    """
    Difference of two energies in eV, as kcal/mol rounded to 0.01.

    :param top: Upper energy in eV.
    :type top: float or None
    :param bottom: Lower energy in eV.
    :type bottom: float or None
    :return: The difference in kcal/mol, or None.
    :rtype: float or None
    """
    if top is None or bottom is None:
        return None
    return round((top - bottom) * EV_TO_KCAL, 2)


def minus(a: float | None, b: float | None) -> float | None:
    """
    Difference of two optional numbers, rounded to 0.01.

    :param a: First number.
    :type a: float or None
    :param b: Second number.
    :type b: float or None
    :return: a - b, or None.
    :rtype: float or None
    """
    return None if a is None or b is None else round(a - b, 2)


def dft_value(data: EngineData, uid: str, field: str) -> float | None:
    """
    One field of an ok DFT row.

    :param data: The engine's data.
    :type data: EngineData
    :param uid: Row uid.
    :type uid: str
    :param field: Field name.
    :type field: str
    :return: The value, or None when missing or failed.
    :rtype: float or None
    """
    row = data.dft.get(uid)
    return row[field] if row and row["ok"] else None


def path_status(path: Mapping[str, Any] | None) -> str:
    """
    Label an IRC row: blank, failed, ends_mismatch or ok.

    :param path: The IRC row, or None when no IRC was run.
    :type path: Mapping[str, Any] or None
    :return: The label; only ok paths give a peak.
    :rtype: str
    """
    if path is None:
        return ""
    if not path["ok"]:
        return "failed"
    return "ok" if path["ends_match"] else "ends_mismatch"


def engine_row(
    system: str, engine: str, rid: str, data: EngineData
) -> dict[str, Any]:
    """
    Build the comparison row of one reaction and engine.

    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :param rid: Reaction id.
    :type rid: str
    :param data: The engine's data.
    :type data: EngineData
    :return: Row with the per-engine columns filled.
    :rtype: dict[str, Any]
    """
    reaction = data.reactions[rid]
    reactants, products = reaction["signature"]
    e_engine = {k: row["energy_ev"] for k, row in data.species.items()}
    e_dft = {
        key: dft_value(data, f"species:{key}", "energy_ev")
        for key in data.species
    }
    ts_engine = reaction["ts_energy_ev"]
    ts_dft = dft_value(data, f"ts:{rid}", "energy_ev")
    r_engine, p_engine = (
        side_sum(e_engine, reactants),
        side_sum(e_engine, products),
    )
    r_dft, p_dft = side_sum(e_dft, reactants), side_sum(e_dft, products)
    row: dict[str, Any] = {
        "system": system,
        "reaction_id": rid,
        "literature_name": reaction.get("literature_name") or "",
        "engine": engine,
        "reactants": "+".join(reactants),
        "products": "+".join(products),
        "barrier_engine": kcal(ts_engine, r_engine),
        "barrier_dft": kcal(ts_dft, r_dft),
        "reverse_engine": kcal(ts_engine, p_engine),
        "reverse_dft": kcal(ts_dft, p_dft),
        "reaction_engine": kcal(p_engine, r_engine),
        "reaction_dft": kcal(p_dft, r_dft),
        "dft_fmax_ev_a": dft_value(data, f"ts:{rid}", "fmax_ev_a"),
    }
    for name in ("barrier", "reverse", "reaction"):
        row[f"{name}_error"] = minus(row[f"{name}_engine"], row[f"{name}_dft"])
    for label, g_ts in reaction["g_ts_ev"].items():
        g = {k: s.get("g_ev", {}).get(label) for k, s in data.species.items()}
        thermal = {
            key: None if g[key] is None else g[key] - e_engine[key] for key in g
        }
        r_thermal = side_sum(thermal, reactants)
        dg_engine = kcal(g_ts, side_sum(g, reactants))
        row[f"dg_barrier_engine_{label}"] = dg_engine
        row[f"dg_barrier_dft_{label}"] = (
            None
            if dg_engine is None
            or row["barrier_dft"] is None
            or r_thermal is None
            else round(
                row["barrier_dft"]
                + ((g_ts - ts_engine) - r_thermal) * EV_TO_KCAL,
                2,
            )
        )
    path = data.paths.get(rid)
    row["path_status"] = path_status(path)
    row["path_keep_going"] = bool(path and path.get("keep_going"))
    energies = dict(data.path_dft.get(rid, {}))
    usable = row["path_status"] == "ok"
    if usable and ts_dft is not None:
        energies[path["ts_index"]] = ts_dft
    row["path_dft_points"] = len(energies)
    if usable and len(energies) > 1:
        peak_index = max(energies, key=energies.__getitem__)
        row["path_peak_ev"] = energies[peak_index]
        row["path_peak_dft"] = kcal(energies[peak_index], r_dft)
        row["peak_above_saddle"] = kcal(energies[peak_index], ts_dft)
        row["peak_offset_frames"] = peak_index - path["ts_index"]
    return row


def add_best(
    rows: list[dict[str, Any]], data: Mapping[str, EngineData]
) -> None:
    """
    Add the per-reaction best limit and estimate over all engines.

    :param rows: Rows of one system, changed in place.
    :type rows: list[dict[str, Any]]
    :param data: Engine data by engine name.
    :type data: Mapping[str, EngineData]
    :return: None.
    :rtype: None
    """
    lowest: dict[str, float] = {}
    for item in data.values():
        for uid, row in item.dft.items():
            if uid.startswith("species:") and row["ok"]:
                key = uid.split(":", 1)[1]
                lowest[key] = min(
                    lowest.get(key, row["energy_ev"]), row["energy_ev"]
                )
    for rid in {row["reaction_id"] for row in rows}:
        group = [row for row in rows if row["reaction_id"] == rid]
        reference = side_sum(lowest, group[0]["reactants"].split("+"))
        peaks = [
            (row["path_peak_ev"], row["engine"])
            for row in group
            if row.get("path_peak_ev") is not None
        ]
        saddles = [
            (data[row["engine"]].dft[f"ts:{rid}"]["energy_ev"], row["engine"])
            for row in group
            if dft_value(data[row["engine"]], f"ts:{rid}", "energy_ev")
            is not None
        ]
        best = {}
        if peaks:
            energy, engine = min(peaks)
            best.update(bound_best=kcal(energy, reference), bound_engine=engine)
        if saddles:
            energy, engine = min(saddles)
            best.update(
                saddle_sp_best=kcal(energy, reference), saddle_sp_engine=engine
            )
        for row in group:
            row.update(best)


def compare_rows(
    results_dir: Path, system: str, engines: Sequence[str]
) -> list[dict[str, Any]]:
    """
    Build every comparison row of one system.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System name.
    :type system: str
    :param engines: Engines to include.
    :type engines: Sequence[str]
    :return: Rows sorted by reaction, then engine order.
    :rtype: list[dict[str, Any]]
    """
    data = {
        engine: load_engine_data(results_dir, system, engine)
        for engine in engines
        if make_refine_paths(
            results_dir, system, engine
        ).reactions_jsonl.is_file()
    }
    rows = [
        engine_row(system, engine, rid, item)
        for engine, item in data.items()
        for rid, reaction in item.reactions.items()
        if reaction["status"] == TSStatus.confirmed
    ]
    add_best(rows, data)
    order = {engine: index for index, engine in enumerate(engines)}
    rows.sort(key=lambda row: (row["reaction_id"], order[row["engine"]]))
    return rows


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.

    :return: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--engine", action="append", dest="engines")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Write one comparison CSV per system.

    :param argv: Optional argument list.
    :type argv: Sequence[str] or None
    :return: 0.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    engines = args.engines or list(DEFAULT_ENGINES)
    for system in args.systems or DEFAULT_SYSTEMS:
        rows = compare_rows(args.results_dir, system, engines)
        if not rows:
            continue
        fields = list(dict.fromkeys(key for row in rows for key in row))
        fields = [field for field in fields if field != "path_peak_ev"]
        out = args.data_dir / system / "checks" / "dft_compare.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"{out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
