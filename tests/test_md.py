"""
Unit tests for xTB nanoreactor molecular dynamics and replicate orchestration.

Tests cover control deck formatting, settings serialization round-trips,
replicate rattle determinism, and execution of live xTB MD runs.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import ase.io
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.build import molecule

from nrnet.nanoreactor.md import (
    OPT_ENGINE_BLOCK,
    MDSettings,
    build_all_equilibrated_tasks,
    build_replicate_tasks,
    control_text,
    equilibration_control_text,
    first_failed_frame,
    initial_velocities,
    load_md_settings,
    load_system_reactants,
    mdrestart_text,
    prepare_optimization,
    prepare_replicate,
    rattle_reactant,
    read_trajectory,
    run_replicate,
    run_replicates,
    select_equilibration_frame_indices,
    trajectory_energies,
)

GOLDEN_CONTROL_TEXT = """$md
   temp=1000.0
   time=20.0
   step=0.5
   dump=10.0
   shake=0
   hmass=1
   nvt=1
   restart=true
$end
$metadyn
   save=10
   kpush=0.05
   alp=1.0
$end
$wall
   potential=logfermi
   sphere: auto, all
$end
"""


def create_sample_settings(
    time_ps: float = 20.0,
    dump_fs: float = 10.0,
    wall_radius_ang: float | None = None,
    replicates: int = 8,
    equilibrate_ps: float = 0.0,
) -> MDSettings:
    """
    Construct a sample MDSettings instance for testing.

    :param time_ps: Simulation duration in picoseconds.
    :type time_ps: float
    :param dump_fs: Dump step interval in femtoseconds.
    :type dump_fs: float
    :param wall_radius_ang: Spherical wall radius in Angstroms, or None.
    :type wall_radius_ang: float or None
    :param replicates: Replicate count.
    :type replicates: int
    :param equilibrate_ps: Unbiased pre-equilibration MD duration in ps.
    :type equilibrate_ps: float
    :return: Configured MDSettings dataclass instance.
    :rtype: MDSettings
    """
    first = (1000.0, time_ps, 0.5, dump_fs, 0.05, 1.0, 10, wall_radius_ang)
    rest = (replicates, 17, 0.05, 3600.0, 50.0, equilibrate_ps)
    return MDSettings(*first, *rest)


def test_control_text_golden() -> None:
    """
    Verify control_text matches the golden formatted xTB control deck.

    :return: None.
    :rtype: None
    """
    settings = create_sample_settings()
    rendered = control_text(settings)
    assert rendered == GOLDEN_CONTROL_TEXT

    walled = create_sample_settings(wall_radius_ang=6.5)
    rendered_walled = control_text(walled)
    # 6.5 A written in bohr, the unit xtb reads.
    assert "sphere: 12.2832, all" in rendered_walled


def test_load_md_settings_round_trip(tmp_path: Path) -> None:
    """
    Verify MDSettings serialization and deserialization round-trip.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    original = create_sample_settings(wall_radius_ang=5.0)
    config_file = tmp_path / "system.yaml"
    with open(config_file, "w", encoding="utf-8") as handle:
        yaml.safe_dump({"md": asdict(original)}, handle)

    loaded = load_md_settings(config_file)
    assert loaded == original


def test_load_hcho_nh3_system_yaml() -> None:
    """
    Verify systems/hcho_nh3/system.yaml loads with consistent values.

    Values are tuned over time, so this checks relations, not numbers.

    :return: None.
    :rtype: None
    """
    settings = load_md_settings("systems/hcho_nh3/system.yaml")
    assert settings.dump_fs % settings.step_fs == 0
    assert settings.min_lifetime_fs >= 2 * settings.dump_fs
    assert settings.time_ps * 1000 > settings.min_lifetime_fs
    assert settings.replicates >= 1
    assert settings.equilibrate_ps == 100.0


def test_hcho_nh3_reactant_geometry() -> None:
    """
    Verify systems/hcho_nh3/reactant.xyz has valid geometry and C-N separation.

    :return: None.
    :rtype: None
    """
    reactant: Atoms = ase.io.read("systems/hcho_nh3/reactant.xyz")
    assert len(reactant) == 8
    c_indices = [idx for idx, sym in enumerate(reactant.symbols) if sym == "C"]
    n_indices = [idx for idx, sym in enumerate(reactant.symbols) if sym == "N"]
    assert len(c_indices) == 1
    assert len(n_indices) == 1
    distance_cn = reactant.get_distance(c_indices[0], n_indices[0])
    assert pytest.approx(3.5, abs=0.1) == distance_cn


def test_rattle_repeatability() -> None:
    """
    Verify rattle repeatability with same seed and difference with other seed.

    :return: None.
    :rtype: None
    """
    mol = molecule("H3CNH2")
    settings = create_sample_settings()
    rep0_a = rattle_reactant(mol, settings, replicate=0)
    rep0_b = rattle_reactant(mol, settings, replicate=0)
    rep1 = rattle_reactant(mol, settings, replicate=1)

    assert np.allclose(rep0_a.positions, rep0_b.positions)
    assert not np.allclose(rep0_a.positions, rep1.positions)


def test_mdrestart_text_lines() -> None:
    """
    Verify mdrestart_text formats exactly 1 header line plus n_atoms lines.

    :return: None.
    :rtype: None
    """
    reactant: Atoms = ase.io.read("systems/hcho_nh3/reactant.xyz")
    initial_velocities(reactant, temperature_k=1000.0, seed=42)
    rendered = mdrestart_text(reactant)
    lines = rendered.splitlines()
    assert len(lines) == 1 + len(reactant)
    assert lines[0] == " -1.0"
    for line in lines[1:]:
        assert "D" in line
        assert len(line.split()) == 6


def test_prepare_replicate_non_empty_folder_raises(tmp_path: Path) -> None:
    """
    Verify prepare_replicate raises FileExistsError on a non-empty directory.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    reactant: Atoms = ase.io.read("systems/hcho_nh3/reactant.xyz")
    settings = create_sample_settings(time_ps=0.1, replicates=1)
    rep_dir = tmp_path / "rep_0"
    rep_dir.mkdir(parents=True)
    (rep_dir / "stale.txt").write_text("stale artifact", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_replicate(
            reactant=reactant,
            settings=settings,
            replicate=0,
            workdir=tmp_path,
        )


def trajectory_coords(trj_path: Path) -> np.ndarray:
    """
    Extract atomic coordinate frames array from a trajectory path.

    :param trj_path: Filesystem path to xtb.trj trajectory file.
    :type trj_path: Path
    :return: Array of atomic coordinates per frame.
    :rtype: np.ndarray
    """
    return np.array([frame.positions for frame in read_trajectory(trj_path)])


@pytest.mark.slow
def test_run_replicate_xtb_smoke(tmp_path: Path) -> None:
    """
    Verify live xTB MD execution of a 0.2 ps replicate producing 8-atom frames.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    reactant: Atoms = ase.io.read("systems/hcho_nh3/reactant.xyz")
    settings = create_sample_settings(time_ps=0.2, dump_fs=10.0, replicates=1)
    trj_path = run_replicate(reactant, settings, 0, tmp_path)
    assert trj_path.exists()
    frames = read_trajectory(trj_path)
    expected_count = int(settings.time_ps * 1000.0 / settings.dump_fs)
    assert abs(len(frames) - expected_count) <= 2
    assert all(len(frame) == 8 for frame in frames)


@pytest.mark.slow
def test_replicate_seeded_reproducibility(tmp_path: Path) -> None:
    """
    Verify 0.2 ps replicates with same seed reproduce; different seeds differ.

    The start is exact, but xtb adds about 1e-7 A of run-to-run noise per
    frame, which chaos grows about tenfold per 0.1 ps. So the start is
    checked to 1e-8 and 0.2 ps only to 1e-5; long runs are reproducible
    in distribution, not frame by frame. Frozen candidates fix the input.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    reactant: Atoms = ase.io.read("systems/hcho_nh3/reactant.xyz")
    settings = create_sample_settings(time_ps=0.2, dump_fs=10.0, replicates=1)
    trj_a = run_replicate(reactant, settings, 0, tmp_path / "run_a")
    trj_b = run_replicate(reactant, settings, 0, tmp_path / "run_b")
    trj_c = run_replicate(reactant, settings, 1, tmp_path / "run_c")
    coords_a = trajectory_coords(trj_a)
    coords_b = trajectory_coords(trj_b)
    coords_c = trajectory_coords(trj_c)
    assert np.allclose(coords_a[0], coords_b[0], atol=1e-8)
    assert np.allclose(coords_a, coords_b, atol=1e-5)
    assert float(np.abs(coords_a - coords_c).max()) > 0.1


@pytest.mark.slow
def test_run_replicates_multiprocessing(tmp_path: Path) -> None:
    """
    Verify parallel replicate execution via multiprocessing pool.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    reactant = molecule("H3CNH2")
    settings = create_sample_settings(time_ps=0.1, dump_fs=10.0, replicates=2)

    paths = run_replicates(
        reactant=reactant,
        settings=settings,
        workdir=tmp_path,
        workers=2,
    )
    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        frames = read_trajectory(path)
        assert len(frames) > 0
        assert len(frames[0]) == 7


def test_run_replicate_nonzero_exit(tmp_path: Path) -> None:
    """
    Verify non-zero exit from xTB raises RuntimeError with process output.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    # Empty atoms structure triggers an immediate xTB failure.
    empty_atoms = Atoms()
    settings = create_sample_settings(time_ps=0.1, replicates=1)
    with pytest.raises(RuntimeError) as exc_info:
        run_replicate(
            reactant=empty_atoms,
            settings=settings,
            replicate=0,
            workdir=tmp_path,
        )
    assert "xTB replicate 0 failed" in str(exc_info.value)


def test_load_system_reactants_default(tmp_path: Path) -> None:
    """
    Verify load_system_reactants defaults to reactant.xyz when absent.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    sys_dir = tmp_path / "sys_default"
    sys_dir.mkdir()
    (sys_dir / "system.yaml").write_text("md:\n  time_ps: 1.0\n", "utf-8")
    mol = molecule("H2O")
    ase.io.write(sys_dir / "reactant.xyz", mol)
    starts = load_system_reactants(sys_dir)
    assert len(starts) == 1
    assert starts[0][0] == "reactant.xyz"
    assert starts[0][1].get_chemical_symbols() == mol.get_chemical_symbols()


@pytest.mark.parametrize(
    "idx,expected_name",
    [(0, "start_a.xyz"), (1, "start_b.xyz"), (2, "start_a.xyz")],
)
def test_replicates_alternation_with_multiple_starts(
    tmp_path: Path,
    idx: int,
    expected_name: str,
) -> None:
    """
    Verify alternation across replicates with multiple starts and metadata.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :param idx: Replicate index identifier.
    :type idx: int
    :param expected_name: Expected start filename in metadata.
    :type expected_name: str
    :return: None.
    :rtype: None
    """
    mol_a = molecule("H2O")
    mol_b = mol_a.copy()
    mol_b.positions[0] += 0.05
    starts = [("start_a.xyz", mol_a), ("start_b.xyz", mol_b)]
    settings = create_sample_settings(replicates=3)
    name, atoms = starts[idx % len(starts)]
    rep_dir = prepare_replicate(
        atoms, settings, idx, tmp_path, start_filename=name
    )
    meta = json.loads((rep_dir / "metadata.json").read_text("utf-8"))
    assert meta["start"] == expected_name
    assert meta["replicate"] == idx


def make_c2h4_positions(offset: float) -> list[list[float]]:
    """
    Generate planar C2H4 atomic coordinate positions with X offset.

    :param offset: Coordinate shift along X axis for second CH2 fragment.
    :type offset: float
    :return: Coordinate positions for C2H4.
    :rtype: list[list[float]]
    """
    return [
        [0.0, 0.0, 0.0],
        [offset, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [offset, 1.0, 0.0],
        [offset, -1.0, 0.0],
    ]


def make_c2h4_geometries() -> tuple[Atoms, Atoms]:
    """
    Construct bound and separated C2H4 geometries with different state keys.

    :return: Pair of bound and dissociated C2H4 Atoms instances.
    :rtype: tuple[Atoms, Atoms]
    """
    mol_near = Atoms("C2H4", positions=make_c2h4_positions(1.34))
    mol_far = Atoms("C2H4", positions=make_c2h4_positions(10.0))
    return mol_near, mol_far


def sample_mismatch_pairs() -> list[tuple[Atoms, Atoms]]:
    """
    Generate sample pairs of mismatched reactant geometries.

    :return: List of (first_atoms, second_atoms) pairs.
    :rtype: list[tuple[Atoms, Atoms]]
    """
    pair_c2h4 = make_c2h4_geometries()
    return [
        (molecule("H2O"), molecule("CH4")),
        (
            Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.1]]),
            Atoms("OC", positions=[[0, 0, 0], [0, 0, 1.1]]),
        ),
        pair_c2h4,
    ]


@pytest.mark.parametrize("atom_pair", sample_mismatch_pairs())
def test_load_system_reactants_mismatch_error(
    tmp_path: Path,
    atom_pair: tuple[Atoms, Atoms],
) -> None:
    """
    Verify load_system_reactants raises ValueError on symbol or key mismatch.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :param atom_pair: Pair of mismatched reactant structures.
    :type atom_pair: tuple[Atoms, Atoms]
    :return: None.
    :rtype: None
    """
    sys_dir = tmp_path / "sys_mismatch"
    sys_dir.mkdir(parents=True, exist_ok=True)
    yaml_text = "reactants: [start_1.xyz, start_2.xyz]\n"
    (sys_dir / "system.yaml").write_text(yaml_text, "utf-8")
    ase.io.write(sys_dir / "start_1.xyz", atom_pair[0])
    ase.io.write(sys_dir / "start_2.xyz", atom_pair[1])
    with pytest.raises(ValueError):
        load_system_reactants(sys_dir)


def test_equilibration_control_text_features() -> None:
    """
    Verify equilibration control text has no metadyn, has shake=2 and wall.

    :return: None.
    :rtype: None
    """
    settings = create_sample_settings(wall_radius_ang=5.0, equilibrate_ps=100.0)
    rendered = equilibration_control_text(settings)
    assert "$metadyn" not in rendered
    assert "shake=2" in rendered
    assert "$wall" in rendered
    assert "sphere: 9.4486, all" in rendered
    assert "temp=1000.0" in rendered
    assert "time=100.0" in rendered
    assert "step=1.0" in rendered
    assert "dump=10.0" in rendered


def test_frame_selection_second_half_and_seeded() -> None:
    """
    Verify frames are distinct, from the second half, and set by the seed.

    :return: None.
    :rtype: None
    """
    mol_a, _ = make_c2h4_geometries()
    frames = [mol_a.copy() for _ in range(200)]
    indices = select_equilibration_frame_indices(frames, 5, mol_a, 17)
    assert all(idx >= 100 for idx in indices)
    assert len(set(indices)) == 5
    again = select_equilibration_frame_indices(frames, 5, mol_a, 17)
    other = select_equilibration_frame_indices(frames, 5, mol_a, 10017)
    assert again == indices
    assert other != indices


def test_frame_selection_skips_reacted_state() -> None:
    """
    Verify frame selection skips frames with altered state key.

    :return: None.
    :rtype: None
    """
    mol_a, mol_b = make_c2h4_geometries()
    frames = [mol_a.copy() for _ in range(10)]
    frames += [mol_b.copy() for _ in range(9)] + [mol_a.copy()]
    indices = select_equilibration_frame_indices(frames, 1, mol_a, 3)
    assert indices == [19]


def test_frame_selection_too_few_frames_raises() -> None:
    """
    Verify select_equilibration_frame_indices raises on insufficient frames.

    :return: None.
    :rtype: None
    """
    mol_a, mol_b = make_c2h4_geometries()
    frames = [mol_a.copy() for _ in range(20)]
    with pytest.raises(ValueError, match="Too few frames"):
        select_equilibration_frame_indices(frames, 15, mol_a, 0)
    reacted = [mol_a.copy() for _ in range(10)] + [
        mol_b.copy() for _ in range(10)
    ]
    with pytest.raises(ValueError, match="Too few unreacted frames"):
        select_equilibration_frame_indices(reacted, 1, mol_a, 0)


def test_equilibrate_ps_zero_leaves_old_path_unchanged(
    tmp_path: Path,
) -> None:
    """
    Verify equilibrate_ps 0 leaves the metadata and task path unchanged.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    mol = molecule("H2O")
    settings = create_sample_settings(equilibrate_ps=0.0, replicates=2)
    rep_dir = prepare_replicate(mol, settings, 0, tmp_path)
    meta = json.loads((rep_dir / "metadata.json").read_text("utf-8"))
    assert "frame_index" not in meta
    assert "equil_frame" not in meta
    assert meta["start"] == "reactant.xyz"
    tasks = build_replicate_tasks([("r.xyz", mol)], settings, tmp_path)
    assert len(tasks) == 2
    assert len(tasks[0]) == 5


def test_build_all_equilibrated_tasks_metadata(tmp_path: Path) -> None:
    """
    Verify equilibrated task generation assigns frames and records metadata.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    mol = molecule("H2O")
    settings = create_sample_settings(equilibrate_ps=100.0, replicates=2)
    fake_trj = [mol.copy() for _ in range(10)]
    tasks = build_all_equilibrated_tasks(
        [("r.xyz", mol)], [fake_trj], settings, tmp_path
    )
    assert len(tasks) == 2
    rep_dir = prepare_replicate(
        tasks[0][0], settings, 0, tmp_path, tasks[0][4], tasks[0][5]
    )
    meta = json.loads((rep_dir / "metadata.json").read_text("utf-8"))
    assert meta["frame_index"] == tasks[0][5] >= 5
    assert meta["equil_frame"] == tasks[0][5]
    assert meta["start_file"] == "r.xyz"


def test_prepare_optimization_uses_lbfgs_and_wall(tmp_path: Path) -> None:
    """
    Verify the start optimization input selects L-BFGS and keeps the wall.

    :param tmp_path: Temporary directory provided by pytest.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    settings = create_sample_settings(wall_radius_ang=4.3)
    opt_dir = prepare_optimization(molecule("H2O"), settings, 0, tmp_path)
    control = (opt_dir / "control.inp").read_text("utf-8")
    assert control.startswith(OPT_ENGINE_BLOCK)
    assert "sphere: 8.1258, all" in control


def make_xtb_trajectory_text(comments: Sequence[str]) -> str:
    """
    Format a multi-frame 3-atom XYZ trajectory with specified comment lines.

    :param comments: Sequence of frame comment header lines.
    :type comments: Sequence[str]
    :return: Formatted XYZ trajectory file content.
    :rtype: str
    """
    atom_block = "H 0.0 0.0 0.0\nH 0.0 0.0 1.0\nO 0.0 1.0 0.0\n"
    return "".join(f"3\n{c}\n{atom_block}" for c in comments)


def format_xtb_comments(energies: Sequence[str]) -> list[str]:
    """
    Build xTB comment lines with energy annotations.

    :param energies: Sequence of energy string values.
    :type energies: Sequence[str]
    :return: List of formatted comment lines.
    :rtype: list[str]
    """
    return [
        f" energy: {e} gnorm: 0.1 xtb: 6.7.1 (unknown)" for e in energies
    ]


def test_read_trajectory_cuts_at_failed_frame(tmp_path: Path) -> None:
    """
    Verify read_trajectory cuts frames from first zero-energy failure.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    failed_trj = tmp_path / "failed.trj"
    failed_comments = format_xtb_comments(
        ("-11.5", "-11.4", "0.000000000000", "-11.3")
    )
    failed_trj.write_text(make_xtb_trajectory_text(failed_comments))
    assert first_failed_frame(failed_trj) == 2
    assert len(read_trajectory(failed_trj)) == 2

    clean_trj = tmp_path / "clean.trj"
    clean_comments = format_xtb_comments(("-11.5", "-11.4", "-11.2", "-11.3"))
    clean_trj.write_text(make_xtb_trajectory_text(clean_comments))
    assert len(read_trajectory(clean_trj)) == 4


def test_trajectory_without_energy_comments(tmp_path: Path) -> None:
    """
    Verify missing energy comment returns None and preserves all frames.

    :param tmp_path: Temporary test directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    raw_trj = tmp_path / "raw.trj"
    raw_trj.write_text(make_xtb_trajectory_text(["raw frame comment"] * 4))
    assert trajectory_energies(raw_trj) is None
    assert len(read_trajectory(raw_trj)) == 4
