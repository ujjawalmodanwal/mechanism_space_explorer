"""
DFT energies and forces on every structure the engines stored (PLAN D23).

For each system and engine, reads the refined saddles (ts.extxyz, latest
confirmed row per reaction) and molecules (species.extxyz, latest row per
key), and runs ORCA at the study's DFT level on each geometry as stored,
without any optimization. Saddles get EnGrad (energy and force), molecules
get SP. The DFT force on an engine's saddle shows how far DFT would place
its own saddle from that geometry. Results go to
results/dft_sp/<system>/<engine>/points.jsonl, one row per structure;
a structure with an ok row is skipped on the next run, so the run can be
stopped and resumed. ORCA inputs and outputs stay in runs/dft_sp/.
Jobs run in a small thread pool, each one ORCA process group.
"""

import argparse
import shutil
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from ase.units import Bohr, Hartree
from orca_benchmark import (
    execute_orca_run,
    extract_last_error,
    format_orca_input,
    parse_final_energy,
    parse_orca_run_seconds,
    parse_orca_version,
    parse_scf_cycles,
)

from nrnet.chem import multiplicity
from nrnet.engines import load_engine_settings, orca_command
from nrnet.store import append_jsonl, latest_rows, make_refine_paths
from nrnet.ts_search import TSStatus

__all__ = [
    "DEFAULT_ENGINES",
    "DEFAULT_SYSTEMS",
    "KEPT_FILES",
    "PointJob",
    "build_parser",
    "main",
    "parse_engrad",
    "point_jobs",
    "points_path",
    "run_point",
]

DEFAULT_SYSTEMS: tuple[str, ...] = ("hcooh", "hcooh_h2o", "hcho_nh3", "cpd_ma")
DEFAULT_ENGINES: tuple[str, ...] = ("xtb", "aimnet", "uma", "mace", "ani1xnr")
KEPT_FILES: tuple[str, ...] = ("orca.inp", "orca.out", "orca.engrad")
SADDLE_TASK: str = "EnGrad"
SPECIES_TASK: str = "SP"


@dataclass(frozen=True, slots=True)
class PointJob:
    """
    One DFT calculation on one stored structure.

    :param system: System name.
    :type system: str
    :param engine: Engine that produced the geometry.
    :type engine: str
    :param kind: ``ts`` for a saddle, ``species`` for a molecule.
    :type kind: str
    :param ident: Reaction id for a saddle, species key for a molecule.
    :type ident: str
    :param task: ORCA task keyword (EnGrad or SP).
    :type task: str
    :param atoms: Geometry as the engine stored it.
    :type atoms: Atoms
    """

    system: str
    engine: str
    kind: str
    ident: str
    task: str
    atoms: Atoms


def points_path(results_dir: Path, system: str, engine: str) -> Path:
    """
    Path of the DFT points file for one system and engine.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Path to points.jsonl.
    :rtype: Path
    """
    return results_dir / "dft_sp" / system / engine / "points.jsonl"


def last_frames(path: Path, field: str) -> dict[str, Atoms]:
    """
    Map each id in an extxyz file to its last frame.

    :param path: Path to the extxyz file.
    :type path: Path
    :param field: Info key holding the id.
    :type field: str
    :return: Mapping of id to the last frame carrying it.
    :rtype: dict[str, Atoms]
    """
    if not path.is_file():
        return {}
    frames = read(path, index=":", format="extxyz")
    return {str(frame.info[field]): frame for frame in frames}


def engine_jobs(
    results_dir: Path, system: str, engine: str
) -> Iterator[PointJob]:
    """
    Yield the saddle jobs, then the molecule jobs, of one engine.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param system: System name.
    :type system: str
    :param engine: Engine name.
    :type engine: str
    :return: Iterator of jobs in run order.
    :rtype: Iterator[PointJob]
    """
    paths = make_refine_paths(results_dir, system, engine)
    if not paths.reactions_jsonl.is_file():
        return
    reactions = latest_rows(paths.reactions_jsonl, "reaction_id")
    saddles = last_frames(paths.ts_extxyz, "reaction_id")
    for rid, row in reactions.items():
        if row["status"] == TSStatus.confirmed and rid in saddles:
            yield PointJob(system, engine, "ts", rid, SADDLE_TASK, saddles[rid])
    species = latest_rows(paths.species_jsonl, "key")
    molecules = last_frames(paths.species_extxyz, "key")
    for key, row in species.items():
        if row["status"] == "ok" and key in molecules:
            yield PointJob(
                system, engine, "species", key, SPECIES_TASK, molecules[key]
            )


def done_keys(path: Path) -> set[tuple[str, str]]:
    """
    Collect the (kind, id) pairs that already have an ok DFT row.

    :param path: Path to points.jsonl.
    :type path: Path
    :return: Set of finished (kind, id) pairs.
    :rtype: set[tuple[str, str]]
    """
    if not path.is_file():
        return set()
    rows = latest_rows(path, "uid")
    return {(row["kind"], row["id"]) for row in rows.values() if row["ok"]}


def point_jobs(
    results_dir: Path,
    systems: Sequence[str],
    engines: Sequence[str],
) -> list[PointJob]:
    """
    List the jobs still to run, system by system.

    :param results_dir: Root results directory.
    :type results_dir: Path
    :param systems: Systems in run order.
    :type systems: Sequence[str]
    :param engines: Engines in run order.
    :type engines: Sequence[str]
    :return: Jobs without an ok row yet.
    :rtype: list[PointJob]
    """
    jobs: list[PointJob] = []
    for system in systems:
        for engine in engines:
            done = done_keys(points_path(results_dir, system, engine))
            jobs.extend(
                job
                for job in engine_jobs(results_dir, system, engine)
                if (job.kind, job.ident) not in done
            )
    return jobs


def parse_engrad(path: Path, n_atoms: int) -> np.ndarray | None:
    """
    Read ORCA's gradient file and return forces in eV/A.

    :param path: Path to orca.engrad.
    :type path: Path
    :param n_atoms: Number of atoms.
    :type n_atoms: int
    :return: Forces of shape (n_atoms, 3), or None if the file is missing.
    :rtype: numpy.ndarray or None
    """
    if not path.is_file():
        return None
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    gradient = np.array(values[2 : 2 + 3 * n_atoms], dtype=float)
    return -gradient.reshape(n_atoms, 3) * Hartree / Bohr


def keep_outputs(workdir: Path, keep_wavefunction: bool) -> None:
    """
    Delete ORCA's scratch files, keeping input, output and gradient.

    :param workdir: Job directory.
    :type workdir: Path
    :param keep_wavefunction: Keep orca.gbw as a guess for later points.
    :type keep_wavefunction: bool
    :return: None.
    :rtype: None
    """
    kept = set(KEPT_FILES) | ({"orca.gbw"} if keep_wavefunction else set())
    for item in workdir.iterdir():
        if item.name in kept:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def run_point(
    job: PointJob,
    model: str,
    scf: str,
    nprocs: int,
    maxcore_mb: int,
    orca_bin: str,
    scratch_dir: Path,
) -> dict[str, Any]:
    """
    Run ORCA on one structure and build its result row.

    :param job: The structure and task.
    :type job: PointJob
    :param model: Functional and basis set.
    :type model: str
    :param scf: SCF convergence keyword.
    :type scf: str
    :param nprocs: ORCA processes.
    :type nprocs: int
    :param maxcore_mb: Memory per process in MB.
    :type maxcore_mb: int
    :param orca_bin: ORCA executable.
    :type orca_bin: str
    :param scratch_dir: Base directory for job folders.
    :type scratch_dir: Path
    :return: Row for points.jsonl.
    :rtype: dict[str, Any]
    """
    workdir = scratch_dir / job.system / job.engine / f"{job.kind}_{job.ident}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    inp = format_orca_input(model, job.task, scf, nprocs, maxcore_mb, job.atoms)
    (workdir / "orca.inp").write_text(inp, encoding="utf-8")
    code, stdout, stderr, wall_s = execute_orca_run(orca_bin, workdir)
    (workdir / "orca.out").write_text(stdout, encoding="utf-8")
    energy_eh = parse_final_energy(stdout)
    forces = parse_engrad(workdir / "orca.engrad", len(job.atoms))
    ok = (
        code == 0
        and "****ORCA TERMINATED NORMALLY****" in stdout
        and energy_eh is not None
        and (job.task != SADDLE_TASK or forces is not None)
    )
    keep_outputs(workdir, keep_wavefunction=job.kind == "ts")
    row: dict[str, Any] = {
        "uid": f"{job.kind}:{job.ident}",
        "kind": job.kind,
        "id": job.ident,
        "task": job.task,
        "atoms": len(job.atoms),
        "multiplicity": multiplicity(job.atoms),
        "ok": ok,
        "energy_eh": energy_eh,
        "energy_ev": None if energy_eh is None else energy_eh * Hartree,
        "fmax_ev_a": None,
        "frms_ev_a": None,
        "forces_ev_a": None,
        "scf_cycles": parse_scf_cycles(stdout),
        "nprocs": nprocs,
        "maxcore_mb": maxcore_mb,
        "wall_s": wall_s,
        "orca_run_s": parse_orca_run_seconds(stdout),
        "orca_version": parse_orca_version(stdout),
        "note": "" if ok else extract_last_error(stdout, stderr),
    }
    if forces is not None:
        norms = np.linalg.norm(forces, axis=1)
        row["fmax_ev_a"] = float(norms.max())
        row["frms_ev_a"] = float(np.sqrt((forces**2).sum() / forces.size))
        row["forces_ev_a"] = forces.round(6).tolist()
    return row


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.

    :return: Configured parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--engine", action="append", dest="engines")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--maxcore-mb", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--scratch", type=Path, default=Path("runs/dft_sp"))
    parser.add_argument(
        "--engines-path", type=Path, default=Path("configs/engines.yaml")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the jobs still to run and stop.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run every pending DFT point and append each result as it finishes.

    :param argv: Optional argument list.
    :type argv: Sequence[str] or None
    :return: 0 when every job ended ok, 1 otherwise.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    systems = args.systems or list(DEFAULT_SYSTEMS)
    engines = args.engines or list(DEFAULT_ENGINES)
    jobs = point_jobs(args.results_dir, systems, engines)[: args.limit]
    print(f"{len(jobs)} jobs pending", flush=True)
    if args.dry_run:
        for job in jobs:
            print(job.system, job.engine, job.kind, job.ident, len(job.atoms))
        return 0
    settings = load_engine_settings(args.engines_path, "dft")
    orca_bin = orca_command()
    failed = 0
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
        for count, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            row = future.result()
            append_jsonl(
                points_path(args.results_dir, job.system, job.engine), row
            )
            failed += not row["ok"]
            print(
                f"[{count}/{len(jobs)}] {job.system} {job.engine} "
                f"{row['uid']} ok={row['ok']} wall={row['wall_s']}s "
                f"fmax={row['fmax_ev_a']} {row['note']}",
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
