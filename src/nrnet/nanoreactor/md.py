"""
Molecular dynamics and metadynamics simulation drivers using the xTB engine.

Configures nanoreactor MD simulations confined within a spherical logfermi
potential wall, executes seeded stochastic replicates in parallel process
pools, and parses output trajectories.
"""

import contextlib
import json
import multiprocessing
import os
import random
import signal
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
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

from nrnet.chem import state_key

__all__ = [
    "AU_VELOCITY",
    "OPT_ENGINE_BLOCK",
    "XTB_FAILED_ENERGY_EH",
    "XTB_THREAD_ENV",
    "MDSettings",
    "build_all_equilibrated_tasks",
    "build_equilibrated_tasks_for_start",
    "build_equilibration_tasks",
    "build_optimization_tasks",
    "build_replicate_metadata",
    "build_replicate_tasks",
    "control_text",
    "ensure_empty_directory",
    "equilibration_control_text",
    "equilibration_md_block",
    "execute_equilibration_task",
    "execute_optimization_task",
    "execute_replicate_task",
    "execute_xtb_md",
    "execute_xtb_opt",
    "find_next_valid_frame",
    "find_replicate_trajectories",
    "first_failed_frame",
    "initial_velocities",
    "load_all_trajectories",
    "load_md_settings",
    "load_system_reactants",
    "md_block",
    "mdrestart_text",
    "metadyn_block",
    "normalize_replicate_starts",
    "prepare_equilibrated_run_tasks",
    "prepare_equilibration",
    "prepare_optimization",
    "prepare_replicate",
    "rattle_reactant",
    "read_reactant_filenames",
    "read_trajectory",
    "replicate_index",
    "run_equilibration",
    "run_optimization",
    "run_replicate",
    "run_replicates",
    "select_equilibration_frame_indices",
    "terminate_process_group",
    "trajectory_energies",
    "validate_reactant_consistency",
    "wall_block",
    "xtb_process_session",
]

AU_VELOCITY: float = units.Bohr / (units._aut * units.second)
XTB_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
# xtb 6.7.1's default ANC optimizer aborts on a Fortran format error in
# some builds (optimizer.f90 line 852); L-BFGS avoids that code path.
OPT_ENGINE_BLOCK: str = "$opt\n   engine=lbfgs\n$end\n"
# xtb 6.7.1 can fail an energy evaluation inside an MD step and still
# carry on, writing energy 0 and zero forces from then on: the atoms fly
# apart, which reads as the box atomizing. Frames from there are cut.
XTB_FAILED_ENERGY_EH: float = 1e-3


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
    :param equilibrate_ps: Duration of pre-equilibration MD in ps (0 off).
    :type equilibrate_ps: float
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
    equilibrate_ps: float = 0.0


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
    if "equilibrate_ps" in section:
        section["equilibrate_ps"] = float(section["equilibrate_ps"])
    return MDSettings(**section)


def read_reactant_filenames(yaml_path: Path) -> list[str]:
    """
    Read list of reactant filenames from system YAML configuration.

    :param yaml_path: Path to system YAML file.
    :type yaml_path: Path
    :return: List of reactant filenames.
    :rtype: list[str]
    """
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    filenames = (
        data.get("reactants", ["reactant.xyz"])
        if isinstance(data, dict)
        else ["reactant.xyz"]
    )
    return filenames or ["reactant.xyz"]


def validate_reactant_consistency(
    reactants: Sequence[tuple[str, Atoms]],
) -> None:
    """
    Verify all reactant geometries have matching symbols and state keys.

    :param reactants: Sequence of (filename, Atoms) tuples.
    :type reactants: Sequence[tuple[str, Atoms]]
    :return: None.
    :rtype: None
    :raises ValueError: If symbols or state keys mismatch first reactant.
    """
    first_name, first_atoms = reactants[0]
    first_symbols = first_atoms.get_chemical_symbols()
    first_key = state_key(first_atoms)
    for name, atoms in reactants[1:]:
        if atoms.get_chemical_symbols() != first_symbols:
            raise ValueError(
                f"Reactant {name} chemical symbols mismatch {first_name}"
            )
        if state_key(atoms) != first_key:
            raise ValueError(
                f"Reactant {name} state key {state_key(atoms)} != {first_key}"
            )


def load_system_reactants(
    system_dir: str | Path,
) -> list[tuple[str, Atoms]]:
    """
    Load and validate starting reactant geometries for a chemical system.

    :param system_dir: Directory containing system YAML and reactant files.
    :type system_dir: str or Path
    :return: List of tuples pairing reactant filename and Atoms structure.
    :rtype: list[tuple[str, Atoms]]
    :raises ValueError: If reactant structures have mismatched symbols or keys.
    :raises FileNotFoundError: If system directory or files do not exist.
    """
    sys_path = Path(system_dir)
    yaml_path = sys_path / "system.yaml" if sys_path.is_dir() else sys_path
    filenames = read_reactant_filenames(yaml_path)
    results = [
        (name, ase.io.read(yaml_path.parent / name)) for name in filenames
    ]
    validate_reactant_consistency(results)
    return results


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
    # xtb reads the sphere radius in bohr; the settings hold Angstrom.
    wall_spec = (
        "auto"
        if settings.wall_radius_ang is None
        else f"{settings.wall_radius_ang / units.Bohr:.4f}"
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


def equilibration_md_block(settings: MDSettings) -> str:
    """
    Format xTB $md block for unbiased pre-equilibration MD inside the wall.

    :param settings: Molecular dynamics simulation parameters.
    :type settings: MDSettings
    :return: Formatted $md block text for equilibration.
    :rtype: str
    """
    return (
        "$md\n"
        f"   temp={settings.temperature_k}\n"
        f"   time={settings.equilibrate_ps}\n"
        "   step=1.0\n"
        "   dump=10.0\n"
        "   shake=2\n"
        "   hmass=1\n"
        "   nvt=1\n"
        "   restart=true\n"
        "$end\n"
    )


def equilibration_control_text(settings: MDSettings) -> str:
    """
    Format an xTB control deck for unbiased pre-equilibration MD.

    :param settings: Molecular dynamics simulation parameters.
    :type settings: MDSettings
    :return: Complete control input file text for xTB equilibration.
    :rtype: str
    """
    return equilibration_md_block(settings) + wall_block(settings)


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


def prepare_equilibration(
    reactant: Atoms,
    settings: MDSettings,
    index: int,
    workdir: str | Path,
) -> Path:
    """
    Prepare directory and input decks for unbiased equilibration MD.

    :param reactant: Base unperturbed molecular geometry.
    :type reactant: Atoms
    :param settings: Molecular dynamics simulation parameters.
    :type settings: MDSettings
    :param index: Start index identifier for equilibration.
    :type index: int
    :param workdir: Root execution directory containing equil_<index>.
    :type workdir: str or Path
    :return: Path to initialized equilibration directory.
    :rtype: Path
    :raises FileExistsError: If target directory exists and is not empty.
    """
    equil_dir = Path(workdir) / f"equil_{index}"
    ensure_empty_directory(equil_dir)
    structure = reactant.copy()
    seed = settings.base_seed + index
    initial_velocities(structure, settings.temperature_k, seed)
    ase.io.write(equil_dir / "start.xyz", structure)
    (equil_dir / "mdrestart").write_text(mdrestart_text(structure), "utf-8")
    (equil_dir / "control.inp").write_text(
        equilibration_control_text(settings), "utf-8"
    )
    return equil_dir


def prepare_optimization(
    reactant: Atoms,
    settings: MDSettings,
    index: int,
    workdir: str | Path,
) -> Path:
    """
    Prepare directory and input decks for start geometry optimization.

    :param reactant: Base unperturbed molecular geometry.
    :type reactant: Atoms
    :param settings: Simulation settings containing wall parameters.
    :type settings: MDSettings
    :param index: Start index identifier for optimization.
    :type index: int
    :param workdir: Root execution directory containing opt_<index>.
    :type workdir: str or Path
    :return: Path to initialized optimization directory.
    :rtype: Path
    :raises FileExistsError: If target directory exists and is not empty.
    """
    opt_dir = Path(workdir) / f"opt_{index}"
    ensure_empty_directory(opt_dir)
    ase.io.write(opt_dir / "start.xyz", reactant)
    control = OPT_ENGINE_BLOCK + wall_block(settings)
    (opt_dir / "control.inp").write_text(control, "utf-8")
    return opt_dir


def build_replicate_metadata(
    start_filename: str,
    replicate: int,
    frame_index: int | None = None,
) -> dict[str, str | int]:
    """
    Construct metadata mapping for an execution replicate.

    :param start_filename: Starting geometry filename.
    :type start_filename: str
    :param replicate: Replicate index identifier.
    :type replicate: int
    :param frame_index: Equilibration frame index, or None.
    :type frame_index: int or None
    :return: Serialized metadata dictionary.
    :rtype: dict[str, str | int]
    """
    meta: dict[str, str | int] = {
        "start": start_filename,
        "start_file": start_filename,
        "replicate": replicate,
    }
    if frame_index is not None:
        meta["frame_index"] = frame_index
        meta["equil_frame"] = frame_index
    return meta


def prepare_replicate(
    reactant: Atoms,
    settings: MDSettings,
    replicate: int,
    workdir: str | Path,
    start_filename: str = "reactant.xyz",
    frame_index: int | None = None,
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
    :param start_filename: Starting geometry filename recorded in metadata.
    :type start_filename: str
    :param frame_index: Equilibration trajectory frame index if equilibrated.
    :type frame_index: int or None
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
    meta = build_replicate_metadata(start_filename, replicate, frame_index)
    (rep_dir / "metadata.json").write_text(json.dumps(meta), "utf-8")
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


def execute_xtb_md(
    rep_dir: Path,
    timeout_s: float,
    replicate: int,
) -> None:
    """
    Run xTB MD simulation process and verify successful exit code.

    :param rep_dir: Replicate working directory.
    :type rep_dir: Path
    :param timeout_s: Subprocess execution timeout in seconds.
    :type timeout_s: float
    :param replicate: Replicate index identifier.
    :type replicate: int
    :return: None.
    :rtype: None
    :raises RuntimeError: If xTB exits with non-zero return code.
    """
    cmd = ["xtb", "start.xyz", "--md", "--input", "control.inp"]
    with xtb_process_session(cmd, rep_dir) as process:
        stdout, stderr = process.communicate(timeout=timeout_s)
    if process.returncode != 0:
        tail = "\n".join((stdout + stderr).strip().splitlines()[-20:])
        raise RuntimeError(
            f"xTB replicate {replicate} failed (code {process.returncode}):\n"
            f"{tail}"
        )


def execute_xtb_opt(
    opt_dir: Path,
    timeout_s: float,
    index: int,
) -> None:
    """
    Run xTB geometry optimization process with wall constraint.

    :param opt_dir: Working directory containing start.xyz and control.inp.
    :type opt_dir: Path
    :param timeout_s: Subprocess execution timeout in seconds.
    :type timeout_s: float
    :param index: Optimization start index identifier.
    :type index: int
    :return: None.
    :rtype: None
    :raises RuntimeError: If xTB exits with non-zero return code.
    """
    cmd = ["xtb", "start.xyz", "--opt", "--input", "control.inp"]
    with xtb_process_session(cmd, opt_dir) as process:
        stdout, stderr = process.communicate(timeout=timeout_s)
    if process.returncode != 0:
        tail = "\n".join((stdout + stderr).strip().splitlines()[-20:])
        raise RuntimeError(
            f"xTB optimization {index} failed (code {process.returncode}):\n"
            f"{tail}"
        )


def run_optimization(
    reactant: Atoms,
    settings: MDSettings,
    index: int,
    workdir: str | Path,
) -> Atoms:
    """
    Execute constrained xTB geometry optimization for a starting structure.

    :param reactant: Base starting molecular geometry.
    :type reactant: Atoms
    :param settings: Simulation settings containing wall parameters.
    :type settings: MDSettings
    :param index: Start index identifier for optimization.
    :type index: int
    :param workdir: Root execution directory.
    :type workdir: str or Path
    :return: Validated optimized atomic structure.
    :rtype: Atoms
    :raises FileNotFoundError: If xtbopt.xyz is missing after run.
    :raises ValueError: If optimized state key differs from start.
    """
    opt_dir = prepare_optimization(reactant, settings, index, workdir)
    execute_xtb_opt(opt_dir, settings.timeout_s, index)
    opt_path = opt_dir / "xtbopt.xyz"
    if not opt_path.exists():
        raise FileNotFoundError(f"Optimized structure missing at {opt_path}")
    optimized = ase.io.read(str(opt_path), format="xyz")
    if state_key(optimized) != state_key(reactant):
        raise ValueError(
            f"Optimized structure state key {state_key(optimized)} "
            f"differs from start {state_key(reactant)}"
        )
    ase.io.write(opt_dir / "start_optimized.xyz", optimized)
    return optimized


def run_equilibration(
    reactant: Atoms,
    settings: MDSettings,
    index: int,
    workdir: str | Path,
) -> Path:
    """
    Execute unbiased xTB pre-equilibration MD inside confinement wall.

    :param reactant: Base unperturbed molecular geometry.
    :type reactant: Atoms
    :param settings: Simulation parameters and timeout limits.
    :type settings: MDSettings
    :param index: Start index identifier for equilibration.
    :type index: int
    :param workdir: Root execution directory.
    :type workdir: str or Path
    :return: Path to generated xtb.trj trajectory file.
    :rtype: Path
    :raises RuntimeError: If xTB exits with non-zero return code.
    :raises FileNotFoundError: If xtb.trj is missing after simulation.
    """
    equil_dir = prepare_equilibration(reactant, settings, index, workdir)
    execute_xtb_md(equil_dir, settings.timeout_s, index)
    trj_path = equil_dir / "xtb.trj"
    if not trj_path.exists():
        raise FileNotFoundError(f"Trajectory not created at {trj_path}")
    return trj_path


def run_replicate(
    reactant: Atoms,
    settings: MDSettings,
    replicate: int,
    workdir: str | Path,
    start_filename: str = "reactant.xyz",
    frame_index: int | None = None,
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
    :param start_filename: Starting geometry filename recorded in metadata.
    :type start_filename: str
    :param frame_index: Equilibration trajectory frame index if equilibrated.
    :type frame_index: int or None
    :return: Path to generated xtb.trj trajectory file.
    :rtype: Path
    :raises RuntimeError: If xTB exits with non-zero return code.
    :raises FileNotFoundError: If xtb.trj is missing after successful run.
    """
    rep_dir = prepare_replicate(
        reactant, settings, replicate, workdir, start_filename, frame_index
    )
    execute_xtb_md(rep_dir, settings.timeout_s, replicate)
    trj_path = rep_dir / "xtb.trj"
    if not trj_path.exists():
        raise FileNotFoundError(f"Trajectory not created at {trj_path}")
    return trj_path


def trajectory_energies(path: str | Path) -> list[float] | None:
    """
    Read the energy that xtb writes in each frame's comment line.

    :param path: Path to the xtb.trj file.
    :type path: str or Path
    :return: One energy per frame in Hartree, or None if a comment line
        is not in xtb's "energy: <value> ..." form.
    :rtype: list[float] or None
    """
    lines = Path(path).read_text().splitlines()
    step = int(lines[0]) + 2
    try:
        return [float(lines[i].split()[1]) for i in range(1, len(lines), step)]
    except (IndexError, ValueError):
        return None


def first_failed_frame(path: str | Path) -> int | None:
    """
    Find the first frame after xtb failed an energy evaluation.

    :param path: Path to the xtb.trj file.
    :type path: str or Path
    :return: Index of the first zero-energy frame, or None.
    :rtype: int or None
    """
    energies = trajectory_energies(path) or []
    return next(
        (i for i, e in enumerate(energies) if abs(e) < XTB_FAILED_ENERGY_EH),
        None,
    )


def read_trajectory(path: str | Path) -> list[Atoms]:
    """
    Read an xTB trajectory, stopping before any failed-energy frames.

    :param path: Path to the xtb.trj file.
    :type path: str or Path
    :return: Frames up to, not including, the first failed frame.
    :rtype: list[Atoms]
    """
    frames = list(ase.io.read(str(path), index=":", format="xyz"))
    return frames[: first_failed_frame(path)]


def replicate_index(path: Path) -> int:
    """
    Extract integer replicate index from trajectory path.

    :param path: Path to replicate trajectory file.
    :type path: Path
    :return: Numerical index of replicate.
    :rtype: int
    """
    return int(path.parent.name.split("_")[-1])


def find_replicate_trajectories(md_dir: Path) -> list[Path]:
    """
    Locate all replicate trajectories sorted by replicate number.

    :param md_dir: Directory containing replicate folders.
    :type md_dir: Path
    :return: Sorted list of trajectory paths.
    :rtype: list[Path]
    """
    return sorted(md_dir.glob("rep_*/xtb.trj"), key=replicate_index)


def load_all_trajectories(md_dir: Path) -> list[list[Atoms]]:
    """
    Read all replicate trajectories sorted by replicate number.

    :param md_dir: Directory containing replicate folders.
    :type md_dir: Path
    :return: List of trajectory frame lists.
    :rtype: list[list[Atoms]]
    """
    paths = find_replicate_trajectories(md_dir)
    return [read_trajectory(path) for path in paths]


def execute_equilibration_task(
    task: tuple[Atoms, MDSettings, int, Path],
) -> Path:
    """
    Unpack worker arguments and run equilibration MD in a process pool.

    :param task: Worker argument tuple (reactant, settings, index, workdir).
    :type task: tuple[Atoms, MDSettings, int, Path]
    :return: Path to generated xtb.trj trajectory file.
    :rtype: Path
    """
    reactant, settings, index, workdir = task
    return run_equilibration(reactant, settings, index, workdir)


def execute_optimization_task(
    task: tuple[Atoms, MDSettings, int, Path],
) -> Atoms:
    """
    Unpack worker arguments and run start geometry optimization in pool.

    :param task: Worker argument tuple (reactant, settings, index, workdir).
    :type task: tuple[Atoms, MDSettings, int, Path]
    :return: Optimized atomic geometry.
    :rtype: Atoms
    """
    reactant, settings, index, workdir = task
    return run_optimization(reactant, settings, index, workdir)


def execute_replicate_task(
    task: (
        tuple[Atoms, MDSettings, int, Path, str, int]
        | tuple[Atoms, MDSettings, int, Path, str]
        | tuple[Atoms, MDSettings, int, Path]
    ),
) -> Path:
    """
    Unpack worker arguments and execute a replicate in a process pool.

    :param task: Worker argument tuple.
    :type task: tuple
    :return: Path to generated xtb.trj trajectory file.
    :rtype: Path
    """
    if len(task) == 6:
        reactant, settings, replicate, workdir, start_name, frame_idx = task
    elif len(task) == 5:
        reactant, settings, replicate, workdir, start_name = task
        frame_idx = None
    else:
        reactant, settings, replicate, workdir = task
        start_name, frame_idx = "reactant.xyz", None
    return run_replicate(
        reactant, settings, replicate, workdir, start_name, frame_idx
    )


def normalize_replicate_starts(
    reactant: Atoms | Sequence[tuple[str, Atoms]] | Sequence[Atoms],
) -> list[tuple[str, Atoms]]:
    """
    Normalize reactant inputs into uniform list of named Atoms geometries.

    :param reactant: Base geometry, list of named starts, or Atoms sequence.
    :type reactant: Atoms or Sequence[tuple[str, Atoms]] or Sequence[Atoms]
    :return: List of (filename, Atoms) tuples.
    :rtype: list[tuple[str, Atoms]]
    """
    if isinstance(reactant, Atoms):
        return [("reactant.xyz", reactant)]
    starts: list[tuple[str, Atoms]] = []
    for item in reactant:
        if isinstance(item, tuple):
            starts.append((str(item[0]), item[1]))
        else:
            starts.append((f"reactant_{len(starts)}.xyz", item))
    return starts


def build_optimization_tasks(
    starts: Sequence[tuple[str, Atoms]],
    settings: MDSettings,
    workdir: Path,
) -> list[tuple[Atoms, MDSettings, int, Path]]:
    """
    Construct worker task tuples for parallel start geometry optimization.

    :param starts: Sequence of named starting reactant geometries.
    :type starts: Sequence[tuple[str, Atoms]]
    :param settings: Simulation settings containing wall parameters.
    :type settings: MDSettings
    :param workdir: Root execution directory.
    :type workdir: Path
    :return: List of optimization worker task tuples.
    :rtype: list[tuple[Atoms, MDSettings, int, Path]]
    """
    return [
        (atoms, settings, idx, workdir) for idx, (_, atoms) in enumerate(starts)
    ]


def build_equilibration_tasks(
    starts: Sequence[tuple[str, Atoms]],
    settings: MDSettings,
    workdir: Path,
) -> list[tuple[Atoms, MDSettings, int, Path]]:
    """
    Construct worker task tuples for parallel equilibration MD.

    :param starts: Sequence of named starting reactant geometries.
    :type starts: Sequence[tuple[str, Atoms]]
    :param settings: Simulation settings for equilibration.
    :type settings: MDSettings
    :param workdir: Root execution directory.
    :type workdir: Path
    :return: List of equilibration worker task tuples.
    :rtype: list[tuple[Atoms, MDSettings, int, Path]]
    """
    tasks: list[tuple[Atoms, MDSettings, int, Path]] = []
    for idx, (_, atoms) in enumerate(starts):
        tasks.append((atoms, settings, idx, workdir))
    return tasks


def find_next_valid_frame(
    frames: Sequence[Atoms],
    target: int,
    used: set[int],
    ref_key: str,
) -> int:
    """
    Find next available unreacted frame offset starting at target index.

    :param frames: Second-half configuration frames.
    :type frames: Sequence[Atoms]
    :param target: Preferred initial frame offset target.
    :type target: int
    :param used: Set of frame offsets already assigned to other replicates.
    :type used: set[int]
    :param ref_key: Reference reactant chemical state key.
    :type ref_key: str
    :return: Selected unreacted trajectory frame offset within frames.
    :rtype: int
    :raises ValueError: If no free unreacted frame exists.
    """
    total = len(frames)
    for step in range(total):
        cand = (target + step) % total
        if cand not in used and state_key(frames[cand]) == ref_key:
            used.add(cand)
            return cand
    raise ValueError("Too few unreacted frames in trajectory second half")


def select_equilibration_frame_indices(
    trajectory: Sequence[Atoms],
    count: int,
    reactant: Atoms,
    seed: int,
) -> list[int]:
    """
    Select distinct unreacted frame indices from trajectory second half.

    Frames are drawn at random with the given seed, so groups of runs with
    different seeds start from different frames (an independent check
    group), while one seed always gives the same frames.

    :param trajectory: Trajectory configuration frames from equilibration.
    :type trajectory: Sequence[Atoms]
    :param count: Number of replicate frames to sample.
    :type count: int
    :param reactant: Reference reactant geometry defining initial state.
    :type reactant: Atoms
    :param seed: Seed of the random frame draw.
    :type seed: int
    :return: Ordered list of trajectory frame indices.
    :rtype: list[int]
    :raises ValueError: If insufficient unreacted second-half frames exist.
    """
    if count <= 0:
        return []
    start_idx = len(trajectory) // 2
    frames = trajectory[start_idx:]
    if len(frames) < count:
        raise ValueError(f"Too few frames ({len(frames)}) for {count} reps")
    ref_key, used = state_key(reactant), set()
    targets = random.Random(seed).sample(range(len(frames)), count)
    offsets = [
        find_next_valid_frame(frames, target, used, ref_key)
        for target in targets
    ]
    return [start_idx + offset for offset in offsets]


def build_equilibrated_tasks_for_start(
    start: tuple[str, Atoms],
    trajectory: Sequence[Atoms],
    replicate_indices: Sequence[int],
    settings: MDSettings,
    workdir: Path,
) -> list[tuple[int, tuple[Atoms, MDSettings, int, Path, str, int]]]:
    """
    Build replicate task tuples for a single starting reactant geometry.

    :param start: Tuple of (filename, reference Atoms).
    :type start: tuple[str, Atoms]
    :param trajectory: Frames from equilibration MD trajectory.
    :type trajectory: Sequence[Atoms]
    :param replicate_indices: Replicate indices assigned to this start.
    :type replicate_indices: Sequence[int]
    :param settings: Molecular dynamics simulation settings.
    :type settings: MDSettings
    :param workdir: Root execution directory.
    :type workdir: Path
    :return: List of (replicate_index, task_tuple) pairs.
    :rtype: list[tuple[int, tuple[Atoms, MDSettings, int, Path, str, int]]]
    """
    name, reactant = start
    frame_indices = select_equilibration_frame_indices(
        trajectory, len(replicate_indices), reactant, settings.base_seed
    )
    tasks: list[tuple[int, tuple[Atoms, MDSettings, int, Path, str, int]]] = []
    for rep_idx, f_idx in zip(replicate_indices, frame_indices, strict=True):
        entry = (trajectory[f_idx], settings, rep_idx, workdir, name, f_idx)
        tasks.append((rep_idx, entry))
    return tasks


def build_all_equilibrated_tasks(
    starts: Sequence[tuple[str, Atoms]],
    trajectories: Sequence[Sequence[Atoms]],
    settings: MDSettings,
    workdir: Path,
) -> list[tuple[Atoms, MDSettings, int, Path, str, int]]:
    """
    Build ordered replicate tasks sampled from equilibration trajectories.

    :param starts: Sequence of named starting reactant geometries.
    :type starts: Sequence[tuple[str, Atoms]]
    :param trajectories: Trajectory frame sequences for each start.
    :type trajectories: Sequence[Sequence[Atoms]]
    :param settings: Molecular dynamics simulation settings.
    :type settings: MDSettings
    :param workdir: Root execution directory.
    :type workdir: Path
    :return: List of ordered replicate task argument tuples.
    :rtype: list[tuple[Atoms, MDSettings, int, Path, str, int]]
    """
    num_starts = len(starts)
    pairs: list[tuple[int, tuple[Atoms, MDSettings, int, Path, str, int]]] = []
    for idx, (start, trj) in enumerate(zip(starts, trajectories, strict=True)):
        rep_indices = [
            i for i in range(settings.replicates) if i % num_starts == idx
        ]
        pairs.extend(
            build_equilibrated_tasks_for_start(
                start, trj, rep_indices, settings, workdir
            )
        )
    pairs.sort(key=lambda item: item[0])
    return [task for _, task in pairs]


def build_replicate_tasks(
    starts: Sequence[tuple[str, Atoms]],
    settings: MDSettings,
    workdir: Path,
) -> list[tuple[Atoms, MDSettings, int, Path, str]]:
    """
    Construct parameter tuples for parallel replicate execution tasks.

    :param starts: Sequence of named starting geometries.
    :type starts: Sequence[tuple[str, Atoms]]
    :param settings: Simulation settings specifying replicate count.
    :type settings: MDSettings
    :param workdir: Root execution directory.
    :type workdir: Path
    :return: List of task argument tuples.
    :rtype: list[tuple[Atoms, MDSettings, int, Path, str]]
    """
    num_starts = len(starts)
    return [
        (
            starts[idx % num_starts][1],
            settings,
            idx,
            workdir,
            starts[idx % num_starts][0],
        )
        for idx in range(settings.replicates)
    ]


def prepare_equilibrated_run_tasks(
    starts: Sequence[tuple[str, Atoms]],
    settings: MDSettings,
    base_dir: Path,
    pool: Pool,
) -> list[tuple[Atoms, MDSettings, int, Path, str, int]]:
    """
    Optimize starts, run equilibration MD, and build replicate tasks.

    :param starts: Sequence of named starting reactant structures.
    :type starts: Sequence[tuple[str, Atoms]]
    :param settings: Molecular dynamics simulation parameters.
    :type settings: MDSettings
    :param base_dir: Root execution working directory.
    :type base_dir: Path
    :param pool: Active multiprocessing worker pool.
    :type pool: Pool
    :return: List of replicate tasks sampled from equilibration.
    :rtype: list[tuple[Atoms, MDSettings, int, Path, str, int]]
    """
    opt_tasks = build_optimization_tasks(starts, settings, base_dir)
    opt_atoms = pool.map(execute_optimization_task, opt_tasks)
    opt_starts = [(starts[i][0], opt_atoms[i]) for i in range(len(starts))]
    equil_tasks = build_equilibration_tasks(opt_starts, settings, base_dir)
    equil_paths = pool.map(execute_equilibration_task, equil_tasks)
    trajectories = [read_trajectory(path) for path in equil_paths]
    return build_all_equilibrated_tasks(
        opt_starts, trajectories, settings, base_dir
    )


def run_replicates(
    reactant: Atoms | Sequence[tuple[str, Atoms]] | Sequence[Atoms],
    settings: MDSettings,
    workdir: str | Path,
    workers: int = 1,
) -> list[Path]:
    """
    Dispatch multiple seeded xTB MD replicates across a multiprocessing pool.

    :param reactant: Base geometry, list of named starts, or Atoms sequence.
    :type reactant: Atoms or Sequence[tuple[str, Atoms]] or Sequence[Atoms]
    :param settings: Simulation settings specifying replicate count.
    :type settings: MDSettings
    :param workdir: Root execution directory.
    :type workdir: str or Path
    :param workers: Number of parallel worker processes.
    :type workers: int
    :return: Ordered list of paths to generated trajectory files.
    :rtype: list[Path]
    """
    starts = normalize_replicate_starts(reactant)
    base_dir = Path(workdir)
    with multiprocessing.Pool(processes=workers) as pool:
        if settings.equilibrate_ps > 0:
            tasks = prepare_equilibrated_run_tasks(
                starts, settings, base_dir, pool
            )
        else:
            tasks = build_replicate_tasks(starts, settings, base_dir)
        return pool.map(execute_replicate_task, tasks)
