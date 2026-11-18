"""
Unit tests for xTB nanoreactor molecular dynamics and replicate orchestration.

Tests cover control deck formatting, settings serialization round-trips,
replicate rattle determinism, and execution of live xTB MD runs.
"""

from dataclasses import asdict
from pathlib import Path

import ase.io
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.build import molecule

from nrnet.nanoreactor.md import (
    MDSettings,
    control_text,
    initial_velocities,
    load_md_settings,
    mdrestart_text,
    prepare_replicate,
    rattle_reactant,
    read_trajectory,
    run_replicate,
    run_replicates,
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
    :return: Configured MDSettings dataclass instance.
    :rtype: MDSettings
    """
    return MDSettings(
        temperature_k=1000.0,
        time_ps=time_ps,
        step_fs=0.5,
        dump_fs=dump_fs,
        kpush=0.05,
        alp=1.0,
        save=10,
        wall_radius_ang=wall_radius_ang,
        replicates=replicates,
        base_seed=17,
        rattle_ang=0.05,
        timeout_s=3600.0,
        min_lifetime_fs=50.0,
    )


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
    assert "sphere: 6.5, all" in rendered_walled


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
