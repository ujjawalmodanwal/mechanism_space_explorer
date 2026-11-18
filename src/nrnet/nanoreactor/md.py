"""
Molecular dynamics and metadynamics simulation drivers using the xTB engine.

Configures nanoreactor MD simulations confined within a spherical logfermi
potential wall, executes seeded stochastic replicates in parallel process
pools, and parses output trajectories.
"""

import contextlib
import multiprocessing
import os
import signal
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import ase.io
import numpy as np
import yaml
from ase import Atoms, units
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    thermalize_momenta,
)

__all__ = [
    "AU_VELOCITY",
    "XTB_THREAD_ENV",
    "MDSettings",
    "control_text",
    "ensure_empty_directory",
    "execute_replicate_task",
    "initial_velocities",
    "load_md_settings",
    "md_block",
    "mdrestart_text",
    "metadyn_block",
    "prepare_replicate",
    "rattle_reactant",
    "read_trajectory",
    "run_replicate",
    "run_replicates",
    "terminate_process_group",
    "wall_block",
    "xtb_process_session",
]

AU_VELOCITY: float = units.Bohr / (units._aut * units.second)
XTB_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


@dataclass(frozen=True, slots=True)
class MDSettings:
    """
    Configuration parameters for nanoreactor molecular dynamics simulations.

    :param temperature_k: Target simulation temperature in Kelvin.
    :type temperature_k: float
    :param time_ps: Total MD duration in picoseconds.
    :type time_ps: float
    :param step_fs: Integration time step in femtoseconds.
    :type step_fs: float
    :param dump_fs: Trajectory frame dump interval in femtoseconds.
    :type dump_fs: float
    :param kpush: Metadynamics push strength in Hartree.
    :type kpush: float
    :param alp: Metadynamics Gaussian width parameter alpha in Bohr^-2.
    :type alp: float
    :param save: Maximum number of past structures kept as bias references.
    :type save: int
    :param wall_radius_ang: Wall radius in Angstroms, or None for auto.
    :type wall_radius_ang: float or None
    :param replicates: Total number of independent replicates to execute.
    :type replicates: int
    :param base_seed: Base integer seed for replicate rattle geometry.
    :type base_seed: int
    :param rattle_ang: Standard deviation of atomic displacement in Angstroms.
    :type rattle_ang: float
    :param timeout_s: Maximum execution duration per replicate in seconds.
    :type timeout_s: float
    :param min_lifetime_fs: Minimum persistence lifetime for stable states.
    :type min_lifetime_fs: float
    """

    temperature_k: float
    time_ps: float
    step_fs: float
    dump_fs: float
    kpush: float
    alp: float
    save: int
    wall_radius_ang: float | None
    replicates: int
    base_seed: int
    rattle_ang: float
    timeout_s: float
    min_lifetime_fs: float


def load_md_settings(path: str | Path) -> MDSettings:
    """
    Read molecular dynamics simulation settings from a system YAML definition.

    :param path: Filesystem path to system YAML containing an 'md' section.
    :type path: str or Path
    :return: Validated MDSettings configuration instance.
    :rtype: MDSettings
    :raises KeyError: If the YAML file lacks an 'md' configuration block.
    """
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "md" not in data:
        raise KeyError(f"Missing 'md' section in configuration file: {path}")
    section = dict(data["md"])
    if section.get("wall_radius_ang") == "auto":
        section["wall_radius_ang"] = None
    return MDSettings(**section)


def md_block(settings: MDSettings) -> str:
    """
    Format xTB $md block with simulation parameters and restart flag.

    :param settings: Molecular dynamics simulation parameters.
    :type settings: MDSettings
    :return: Formatted $md block text including terminating $end.
    :rtype: str
    """
    return (
        "$md\n"
        f"   temp={settings.temperature_k}\n"
        f"   time={settings.time_ps}\n"
        f"   step={settings.step_fs}\n"
        f"   dump={settings.dump_fs}\n"
        "   shake=0\n"
        "   hmass=1\n"
        "   nvt=1\n"
        "   restart=true\n"
        "$end\n"
    )


def metadyn_block(settings: MDSettings) -> str:
    """
    Format xTB $metadyn block with history-dependent bias parameters.

    :param settings: Metadynamics simulation parameters.
    :type settings: MDSettings
    :return: Formatted $metadyn block text including terminating $end.
    :rtype: str
    """
    return (
        "$metadyn\n"
        f"   save={settings.save}\n"
        f"   kpush={settings.kpush}\n"
        f"   alp={settings.alp}\n"
        "$end\n"
    )


def wall_block(settings: MDSettings) -> str:
    """
    Format xTB $wall block with spherical logfermi confinement parameters.

    :param settings: Confinement wall simulation parameters.
    :type settings: MDSettings
    :return: Formatted $wall block text including terminating $end.
    :rtype: str
    """
    wall_spec = (
        "auto"
        if settings.wall_radius_ang is None
        else f"{settings.wall_radius_ang}"
    )
    return f"$wall\n   potential=logfermi\n   sphere: {wall_spec}, all\n$end\n"


def control_text(settings: MDSettings) -> str:
    """
    Format an xTB control deck with MD, metadynamics, and confinement walls.

    :param settings: Molecular dynamics simulation parameters.
    :type settings: MDSettings
    :return: Complete control input file text for xTB.
    :rtype: str
    """
    return md_block(settings) + metadyn_block(settings) + wall_block(settings)


def initial_velocities(
    atoms: Atoms,
    temperature_k: float,
    seed: int,
) -> None:
    """
    Sample Maxwell-Boltzmann velocities with zero net linear/angular momentum.

    :param atoms: Target atomic structure to receive velocities.
    :type atoms: Atoms
    :param temperature_k: Target temperature in Kelvin.
    :type temperature_k: float
    :param seed: Deterministic random seed for velocity sampling.
    :type seed: int
    :return: None.
    :rtype: None
    """
    rng = np.random.default_rng(seed)
    thermalize_momenta(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)


def mdrestart_text(atoms: Atoms) -> str:
    """
    Format an xTB mdrestart file with initial atomic positions and velocities.

    :param atoms: Atomic configuration with positions and velocities set.
    :type atoms: Atoms
    :return: Formatted mdrestart file content text.
    :rtype: str
    """
    velocities = atoms.get_velocities()
    if velocities is None:
        velocities = np.zeros_like(atoms.positions)
    pos = atoms.positions / units.Bohr
    vel = velocities / AU_VELOCITY
    rows = [" -1.0"]
    for atom_pos, atom_vel in zip(pos, vel, strict=True):
        coords = (*atom_pos, *atom_vel)
        rows.append(" ".join(f"{x: .14E}".replace("E", "D") for x in coords))
    return "\n".join(rows) + "\n"


def rattle_reactant(
    reactant: Atoms,
    settings: MDSettings,
    replicate: int,
) -> Atoms:
    """
    Perturb initial atomic coordinates with a deterministic seeded rattle.

    :param reactant: Base unperturbed molecular geometry.
    :type reactant: Atoms
    :param settings: Simulation settings containing base seed and rattle std.
    :type settings: MDSettings
    :param replicate: Integer replicate index offset added to base seed.
    :type replicate: int
    :return: Perturbed atomic geometry copy.
    :rtype: Atoms
    """
    perturbed = reactant.copy()
    seed = settings.base_seed + replicate
    perturbed.rattle(stdev=settings.rattle_ang, seed=seed)
    return perturbed


def ensure_empty_directory(directory: Path) -> None:
    """
    Ensure target directory exists and contains no prior calculation artifacts.

    :param directory: Target filesystem path to inspect or create.
    :type directory: Path
    :return: None.
    :rtype: None
    :raises FileExistsError: If directory exists and contains any files.
    """
    if directory.exists():
        if not directory.is_dir() or any(directory.iterdir()):
            raise FileExistsError(
                f"Replicate directory is not empty: {directory}"
            )
    else:
        directory.mkdir(parents=True, exist_ok=True)


def prepare_replicate(
    reactant: Atoms,
    settings: MDSettings,
    replicate: int,
    workdir: str | Path,
) -> Path:
    """
    Prepare directory, structure, and restart decks for an xTB replicate.

    :param reactant: Base unperturbed molecular geometry.
    :type reactant: Atoms
    :param settings: Simulation settings for MD, metadynamics, and rattle.
    :type settings: MDSettings
    :param replicate: Integer replicate index identifier.
    :type replicate: int
    :param workdir: Root execution directory; the replicate gets rep_<n>.
    :type workdir: str or Path
    :return: Path to initialized replicate directory.
    :rtype: Path
    :raises FileExistsError: If replicate directory exists and is not empty.
    """
    rep_dir = Path(workdir) / f"rep_{replicate}"
    ensure_empty_directory(rep_dir)
    seed = settings.base_seed + replicate
    perturbed = rattle_reactant(reactant, settings, replicate)
    initial_velocities(perturbed, settings.temperature_k, seed)
    ase.io.write(rep_dir / "start.xyz", perturbed)
    (rep_dir / "mdrestart").write_text(mdrestart_text(perturbed), "utf-8")
    (rep_dir / "control.inp").write_text(control_text(settings), "utf-8")
    return rep_dir


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    """
    Safely terminate a child process and its entire process group.

    :param process: Active subprocess instance.
    :type process: subprocess.Popen[str]
    :return: None.
    :rtype: None
    """
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@contextlib.contextmanager
def xtb_process_session(
    command: list[str],
    workdir: Path,
) -> Iterator[subprocess.Popen[str]]:
    """
    Context manager launching xTB in a new session, killing group on exit.

    :param command: Execution command argument vector.
    :type command: list[str]
    :param workdir: Execution directory for the subprocess.
    :type workdir: Path
    :return: Iterator yielding the running subprocess instance.
    :rtype: Iterator[subprocess.Popen[str]]
    """
    env = {**os.environ, **XTB_THREAD_ENV}
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        yield process
    finally:
        terminate_process_group(process)


def run_replicate(
    reactant: Atoms,
    settings: MDSettings,
    replicate: int,
    workdir: str | Path,
) -> Path:
    """
    Execute a single xTB MD replicate inside an isolated working directory.

    :param reactant: Base molecular geometry for the simulation.
    :type reactant: Atoms
    :param settings: Simulation parameters and timeout limits.
    :type settings: MDSettings
    :param replicate: Replicate index identifier.
    :type replicate: int
    :param workdir: Root execution directory.
    :type workdir: str or Path
    :return: Path to generated xtb.trj trajectory file.
    :rtype: Path
    :raises RuntimeError: If xTB exits with non-zero return code.
    :raises FileNotFoundError: If xtb.trj is missing after successful run.
    """
    rep_dir = prepare_replicate(reactant, settings, replicate, workdir)
    cmd = ["xtb", "start.xyz", "--md", "--input", "control.inp"]
    with xtb_process_session(cmd, rep_dir) as process:
        stdout, stderr = process.communicate(timeout=settings.timeout_s)
    if process.returncode != 0:
        tail = "\n".join((stdout + stderr).strip().splitlines()[-20:])
        raise RuntimeError(
            f"xTB replicate {replicate} failed (code {process.returncode}):\n"
            f"{tail}"
        )
    trj_path = rep_dir / "xtb.trj"
    if not trj_path.exists():
        raise FileNotFoundError(f"Trajectory not created at {trj_path}")
    return trj_path


def read_trajectory(path: str | Path) -> list[Atoms]:
    """
    Read an xTB trajectory file into a sequence of ASE Atoms configurations.

    :param path: Filesystem path to xtb.trj trajectory file.
    :type path: str or Path
    :return: List of molecular configuration frames.
    :rtype: list[Atoms]
    """
    return list(ase.io.read(str(path), index=":", format="xyz"))


def execute_replicate_task(
    task: tuple[Atoms, MDSettings, int, Path],
) -> Path:
    """
    Unpack worker arguments and execute a replicate in a process pool.

    :param task: Tuple of (reactant, settings, replicate, workdir).
    :type task: tuple[Atoms, MDSettings, int, Path]
    :return: Path to generated xtb.trj trajectory file.
    :rtype: Path
    """
    reactant, settings, replicate, workdir = task
    return run_replicate(reactant, settings, replicate, workdir)


def run_replicates(
    reactant: Atoms,
    settings: MDSettings,
    workdir: str | Path,
    workers: int = 1,
) -> list[Path]:
    """
    Dispatch multiple seeded xTB MD replicates across a multiprocessing pool.

    :param reactant: Base unperturbed molecular geometry.
    :type reactant: Atoms
    :param settings: Simulation settings specifying replicate count.
    :type settings: MDSettings
    :param workdir: Root execution directory.
    :type workdir: str or Path
    :param workers: Number of parallel worker processes.
    :type workers: int
    :return: Ordered list of paths to generated trajectory files.
    :rtype: list[Path]
    """
    root_dir = Path(workdir)
    tasks = [
        (reactant, settings, idx, root_dir)
        for idx in range(settings.replicates)
    ]
    with multiprocessing.Pool(processes=workers) as pool:
        return pool.map(execute_replicate_task, tasks)
