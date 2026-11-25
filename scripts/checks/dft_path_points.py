"""
DFT single points along the engines' IRC paths (PLAN D23).

Reads the paths written by ml_irc.py and runs ORCA SP on chosen frames in
two passes. The coarse pass takes both ends and every few frames out from
the saddle on each side. Once a path's coarse pass is done, the fine pass
takes the frames between the two coarse neighbours of the highest DFT
energy, so the top of the path is sampled closely. The saddle frame
itself is the engine's stored saddle, whose DFT energy is already in
results/dft_sp; it is computed here only when that row is missing. The
highest DFT energy on a path is an upper limit to the DFT saddle energy
(see ml_irc.py). Rows go to results/dft_path/<system>/<engine>/
points.jsonl with uid ``path:<reaction_id>_<index>``; finished frames are
skipped on a rerun. Paths whose ends do not carry the reaction's
molecules are skipped: they do not join the two basins, so their peak
limits nothing.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ase import Atoms
from ase.io import read
from dft_single_points import (
    DEFAULT_ENGINES,
    DEFAULT_SYSTEMS,
    PointJob,
    points_path,
    run_point,
)

from nrnet.engines import load_engine_settings, orca_command
from nrnet.store import append_jsonl, latest_rows

__all__ = [
    "build_parser",
    "coarse_indices",
    "fine_indices",
    "main",
    "path_energies",
    "pending_jobs",
]


def coarse_indices(n_points: int, ts_index: int, per_side: int) -> list[int]:
    """
    Pick both ends and evenly spaced frames out from the saddle.

    :param n_points: Number of frames on the path.
    :type n_points: int
    :param ts_index: Index of the saddle frame.
    :type ts_index: int
    :param per_side: Frames to take on each side, the end included.
    :type per_side: int
    :return: Sorted frame indices, the saddle left out.
    :rtype: list[int]
    """
    picked = {0, n_points - 1}
    for length, sign in ((ts_index, -1), (n_points - 1 - ts_index, 1)):
        stride = max(1, math.ceil(length / per_side))
        picked.update(
            ts_index + sign * step for step in range(stride, length, stride)
        )
    picked.discard(ts_index)
    return sorted(picked)


def fine_indices(
    energies: dict[int, float], n_points: int, limit: int
) -> list[int]:
    """
    Pick the frames between the coarse neighbours of the highest energy.

    :param energies: DFT energy per computed frame index.
    :type energies: dict[int, float]
    :param n_points: Number of frames on the path.
    :type n_points: int
    :param limit: Most frames to pick.
    :type limit: int
    :return: Sorted frame indices not computed yet.
    :rtype: list[int]
    """
    done = sorted(energies)
    top = max(done, key=energies.__getitem__)
    position = done.index(top)
    low = done[position - 1] if position > 0 else 0
    high = done[position + 1] if position + 1 < len(done) else n_points - 1
    gap = [index for index in range(low + 1, high) if index not in energies]
    if len(gap) <= limit:
        return gap
    stride = len(gap) / limit
    return sorted({gap[int(step * stride)] for step in range(limit)})


def path_energies(
    results_dir: Path, system: str, engine: str, rid: str, ts_index: int
) -> tuple[dict[int, float], set[int]]:
    """
    Collect the DFT energies already computed on one path.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :param rid: Reaction id.
    :type rid: str
    :param ts_index: Index of the saddle frame.
    :type ts_index: int
    :return: DFT energy in eV per ok frame, and every frame tried.
    :rtype: tuple[dict[int, float], set[int]]
    """
    energies: dict[int, float] = {}
    tried: set[int] = set()
    sp_path = points_path(results_dir, system, engine)
    if sp_path.is_file():
        saddle = latest_rows(sp_path, "uid").get(f"ts:{rid}")
        if saddle and saddle["ok"]:
            energies[ts_index] = saddle["energy_ev"]
    own = results_dir / "dft_path" / system / engine / "points.jsonl"
    if own.is_file():
        for row in latest_rows(own, "uid").values():
            name, index = row["id"].rsplit("_", 1)
            if name != rid:
                continue
            tried.add(int(index))
            if row["ok"]:
                energies[int(index)] = row["energy_ev"]
    return energies, tried


def pending_jobs(
    results_dir: Path,
    systems: Sequence[str],
    engines: Sequence[str],
    per_side: int,
    fine_limit: int,
) -> list[PointJob]:
    """
    List the frames to compute next on every path.

    A path gets its coarse frames first and its fine frames only once the
    coarse ones are all done, so each call returns one pass.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param systems: Systems in run order.
    :type systems: Sequence[str]
    :param engines: Engines in run order.
    :type engines: Sequence[str]
    :param per_side: Coarse frames per side.
    :type per_side: int
    :param fine_limit: Most fine frames per path.
    :type fine_limit: int
    :return: Jobs for this pass.
    :rtype: list[PointJob]
    """
    jobs: list[PointJob] = []
    for system in systems:
        for engine in engines:
            irc = results_dir / "irc" / system / engine
            if not (irc / "paths.jsonl").is_file():
                continue
            rows = latest_rows(irc / "paths.jsonl", "reaction_id")
            for rid, row in rows.items():
                if not (row["ok"] and row["ends_match"]):
                    continue
                frames: list[Atoms] = read(irc / f"{rid}.extxyz", index=":")
                n_points, ts_index = len(frames), row["ts_index"]
                energies, tried = path_energies(
                    results_dir, system, engine, rid, ts_index
                )
                wanted = coarse_indices(n_points, ts_index, per_side)
                if ts_index not in energies:
                    wanted.append(ts_index)
                todo = [
                    index
                    for index in wanted
                    if index not in energies and index not in tried
                ]
                if not todo and energies:
                    todo = [
                        index
                        for index in fine_indices(
                            energies, n_points, fine_limit
                        )
                        if index not in tried
                    ]
                jobs.extend(
                    PointJob(
                        system,
                        engine,
                        "path",
                        f"{rid}_{index:03d}",
                        "SP",
                        frames[index],
                    )
                    for index in todo
                )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.

    :return: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--engine", action="append", dest="engines")
    parser.add_argument("--per-side", type=int, default=5)
    parser.add_argument("--fine-limit", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--maxcore-mb", type=int, default=1500)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--scratch", type=Path, default=Path("runs/dft_path"))
    parser.add_argument(
        "--engines-path", type=Path, default=Path("configs/engines.yaml")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run passes until no path has frames left to compute.

    :param argv: Optional argument list.
    :type argv: Sequence[str] or None
    :return: 0 when every frame ended ok, 1 otherwise.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    systems = args.systems or list(DEFAULT_SYSTEMS)
    engines = args.engines or list(DEFAULT_ENGINES)
    settings = load_engine_settings(args.engines_path, "dft")
    orca_bin = "" if args.dry_run else orca_command()
    failed, seen = 0, set()
    while True:
        jobs = [
            job
            for job in pending_jobs(
                args.results_dir,
                systems,
                engines,
                args.per_side,
                args.fine_limit,
            )
            if (job.system, job.engine, job.ident) not in seen
        ]
        print(f"pass: {len(jobs)} frames", flush=True)
        if not jobs or args.dry_run:
            break
        seen.update((job.system, job.engine, job.ident) for job in jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_point,
                    job,
                    settings.model,
                    str(settings.extra["scf"]),
                    args.nprocs,
                    args.maxcore_mb,
                    orca_bin,
                    args.scratch,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                job, row = futures[future], future.result()
                own = (
                    args.results_dir
                    / "dft_path"
                    / job.system
                    / job.engine
                    / "points.jsonl"
                )
                append_jsonl(own, row)
                failed += not row["ok"]
                print(
                    f"{job.system} {job.engine} {row['uid']} "
                    f"ok={row['ok']} wall={row['wall_s']}s {row['note']}",
                    flush=True,
                )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
