"""
IRC paths from each engine's stored saddles, on that engine (PLAN D23).

For every confirmed saddle of one system and engine (latest row per
reaction), runs a Sella IRC forward and reverse from the stored geometry
and writes the whole path, reverse end first, to
results/irc/<system>/<engine>/<reaction_id>.extxyz with the engine energy
on each frame. One row per reaction goes to paths.jsonl: steps and
convergence per side, and whether the two ends carry the molecules of the
reaction signature. DFT single points along these paths
(dft_path_points.py) give an upper limit to the DFT saddle energy: the
saddle is the lowest possible high point of any path between the two
basins, so the highest DFT energy on this path cannot be below it.
When Sella's inner loop fails to converge, the IRC is run again with
keep_going: the path is then no longer an exact IRC but stays a
continuous path between the basins, which is all the upper limit needs;
such rows carry keep_going true and the count of inner-loop failures.
Run it with the engine's own environment, for example
.venvs/uma/bin/python. Reactions with a row are skipped on a rerun.
"""

import argparse
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from sella import IRC
from sella.optimize.irc import IRCInnerLoopConvergenceFailure

from nrnet.chem import state_key
from nrnet.engines import attach, load_engine_settings
from nrnet.store import append_jsonl, latest_rows, make_refine_paths
from nrnet.ts_search import TSStatus

__all__ = [
    "build_parser",
    "irc_dir",
    "main",
    "run_irc",
    "side_matches",
]


def irc_dir(results_dir: Path, system: str, engine: str) -> Path:
    """
    Directory holding one engine's IRC paths for one system.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Path to the directory.
    :rtype: Path
    """
    return results_dir / "irc" / system / engine


def side_matches(atoms: Atoms, side: Sequence[str]) -> bool:
    """
    Check whether a path end carries the molecules of one reaction side.

    :param atoms: Path end geometry.
    :type atoms: Atoms
    :param side: Species keys of that side of the signature.
    :type side: Sequence[str]
    :return: True when the molecules match, stereo included.
    :rtype: bool
    """
    return tuple(sorted(state_key(atoms, stereo=True))) == tuple(sorted(side))


def run_irc(
    saddle: Atoms,
    settings: Any,
    dx: float,
    fmax: float,
    steps: int,
    keep_going: bool = False,
) -> tuple[list[Atoms], dict[str, Any]]:
    """
    Run the IRC both ways and join the two halves into one path.

    :param saddle: Stored saddle geometry.
    :type saddle: Atoms
    :param settings: Engine settings.
    :type settings: EngineSettings
    :param dx: Sella IRC step in mass-weighted A amu^0.5.
    :type dx: float
    :param fmax: Force at which each half stops, eV/A.
    :type fmax: float
    :param steps: Step cap per half.
    :type steps: int
    :param keep_going: Continue past inner-loop failures.
    :type keep_going: bool
    :return: Path frames (reverse end first) and per-side statistics.
    :rtype: tuple[list[Atoms], dict[str, Any]]
    """
    target = attach(saddle, settings)
    irc = IRC(target, logfile=None, dx=dx, keep_going=keep_going)
    halves: dict[str, list[Atoms]] = {"forward": [], "reverse": []}
    stats: dict[str, Any] = {"keep_going": keep_going, "inner_failures": 0}
    current: list[Atoms] = []

    def record() -> None:
        frame = target.copy()
        frame.info = {"energy_ev": float(target.get_potential_energy())}
        current.append(frame)

    irc.attach(record)
    for direction in ("forward", "reverse"):
        current.clear()
        before = irc.nsteps
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            converged = bool(
                irc.run(fmax=fmax, steps=steps, direction=direction)
            )
        stats["inner_failures"] += sum(
            "inner loop" in str(item.message) for item in caught
        )
        halves[direction] = list(current)
        stats[f"{direction}_converged"] = converged
        stats[f"{direction}_steps"] = irc.nsteps - before
    reverse = halves["reverse"][1:][::-1]
    forward = halves["forward"][1:]
    ts_frame = saddle.copy()
    ts_frame.info = {"energy_ev": halves["forward"][0].info["energy_ev"]}
    path = [*reverse, ts_frame, *forward]
    for index, frame in enumerate(path):
        frame.info["index"] = index
        frame.info["ts_index"] = len(reverse)
    return path, stats


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.

    :return: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--reaction", action="append", dest="reactions")
    parser.add_argument(
        "--literature-only",
        action="store_true",
        help="Only reactions with a literature name.",
    )
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--engines-path", type=Path, default=Path("configs/engines.yaml")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the IRC for every pending saddle of one system and engine.

    :param argv: Optional argument list.
    :type argv: Sequence[str] or None
    :return: 0 when every IRC finished, 1 if any raised.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    paths = make_refine_paths(args.results_dir, args.system, args.engine)
    settings = load_engine_settings(args.engines_path, args.engine)
    reactions = latest_rows(paths.reactions_jsonl, "reaction_id")
    saddles = {
        str(frame.info["reaction_id"]): frame
        for frame in read(paths.ts_extxyz, index=":", format="extxyz")
    }
    out_dir = irc_dir(args.results_dir, args.system, args.engine)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "paths.jsonl"
    done = {
        rid
        for rid, row in (
            latest_rows(rows_path, "reaction_id") if rows_path.is_file() else {}
        ).items()
        if row["ok"]
    }
    failed = 0
    for rid, row in reactions.items():
        if (
            row["status"] != TSStatus.confirmed
            or rid not in saddles
            or rid in done
            or (args.reactions and rid not in args.reactions)
            or (args.literature_only and not row.get("literature_name"))
        ):
            continue
        start = time.perf_counter()
        result: dict[str, Any] = {"reaction_id": rid, "ok": False}
        irc_args = (saddles[rid], settings, args.dx, args.fmax, args.steps)
        try:
            try:
                path, stats = run_irc(*irc_args)
            except IRCInnerLoopConvergenceFailure:
                path, stats = run_irc(*irc_args, keep_going=True)
        except Exception as error:  # an engine or Sella failure ends one row
            result["note"] = f"{type(error).__name__}: {error}"
            failed += 1
        else:
            write(out_dir / f"{rid}.extxyz", path, format="extxyz")
            energies = np.array([frame.info["energy_ev"] for frame in path])
            reactant, product = row["signature"]
            ends = (path[0], path[-1])
            result.update(stats)
            result.update(
                ok=True,
                n_points=len(path),
                ts_index=path[0].info["ts_index"],
                peak_index=int(energies.argmax()),
                ends_match=(
                    side_matches(ends[0], reactant)
                    and side_matches(ends[1], product)
                )
                or (
                    side_matches(ends[0], product)
                    and side_matches(ends[1], reactant)
                ),
            )
        result["wall_s"] = round(time.perf_counter() - start, 1)
        append_jsonl(rows_path, result)
        print(args.system, args.engine, result, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
