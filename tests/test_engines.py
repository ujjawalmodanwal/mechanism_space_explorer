"""
Unit tests for computational chemistry engine settings and calculator loaders.

Tests cover engine settings deserialization, unknown engine handling,
tblite calculator attachment, spin multiplicity assignment, open-shell
validation guards, ML engine loaders, and the ORCA DFT calculator loader.
"""

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from nrnet.engines import (
    ORCA_COMMAND_ENV,
    EngineSettings,
    UnsupportedSpecies,
    attach,
    load_dft_calculator,
    load_engine_settings,
    orca_command,
)


def assert_energy_and_forces(
    atoms: Atoms,
    settings: EngineSettings,
    expected_spin: int,
) -> tuple[float, np.ndarray]:
    """
    Attach engine to structure and assert valid energy, forces, and spin.

    :param atoms: Target molecular structure.
    :type atoms: Atoms
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param expected_spin: Expected spin multiplicity.
    :type expected_spin: int
    :return: Computed energy in eV and forces array.
    :rtype: tuple[float, np.ndarray]
    """
    attached = attach(atoms, settings)
    assert attached.info["spin"] == expected_spin
    assert attached.info["charge"] == 0
    energy = attached.get_potential_energy()
    forces = attached.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
    return float(energy), forces


@pytest.mark.parametrize(
    ("name", "model", "closed_shell_only", "extra"),
    [
        ("xtb", "GFN2-xTB", False, {}),
        ("mace", "extra_large", False, {}),
        ("egret", "models/EGRET_1.model", True, {}),
        ("uma", "uma-s-1p1", False, {}),
        ("aimnet", "aimnet2", False, {}),
        (
            "dft",
            "wB97M-V def2-TZVPD",
            False,
            {
                "nprocs": 16,
                "maxcore_mb": 3000,
                "scf": "TightSCF",
                "scratch_dir": "runs/orca",
            },
        ),
    ],
)
def test_load_engine_settings_known(
    name: str,
    model: str,
    closed_shell_only: bool,
    extra: Mapping[str, str | int],
) -> None:
    """
    Verify engine settings load correctly from YAML config file.

    :param name: Engine identifier.
    :type name: str
    :param model: Expected model string.
    :type model: str
    :param closed_shell_only: Expected closed-shell restriction flag.
    :type closed_shell_only: bool
    :param extra: Expected extra parameters dictionary.
    :type extra: dict[str, int]
    :return: None.
    :rtype: None
    """
    settings = load_engine_settings("configs/engines.yaml", name)
    assert settings.name == name
    assert settings.model == model
    assert settings.device == "cpu"
    assert settings.closed_shell_only is closed_shell_only
    assert settings.extra == extra


def test_load_engine_settings_unknown() -> None:
    """
    Verify unknown engine name raises KeyError naming known options.

    :return: None.
    :rtype: None
    """
    with pytest.raises(KeyError) as exc_info:
        load_engine_settings("configs/engines.yaml", "unknown_engine")
    error_message = str(exc_info.value)
    assert "unknown_engine" in error_message
    assert "xtb" in error_message


def test_attach_hcho_xtb() -> None:
    """
    Verify attach on HCHO gives finite energy, forces, and detached input.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    attached = attach(hcho, settings)
    assert hcho.calc is None
    assert attached.info["spin"] == 1
    assert attached.info["charge"] == 0
    energy = attached.get_potential_energy()
    forces = attached.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (4, 3)
    assert np.all(np.isfinite(forces))


def test_attach_closed_shell_only_raises() -> None:
    """
    Verify open-shell species raises UnsupportedSpecies for closed-shell engine.

    :return: None.
    :rtype: None
    """
    ch3 = molecule("CH3")
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    closed_settings = replace(settings, closed_shell_only=True)
    with pytest.raises(UnsupportedSpecies):
        attach(ch3, closed_settings)


def test_attach_ch3_xtb_doublet() -> None:
    """
    Verify CH3 radical attaches and calculates successfully as a doublet.

    :return: None.
    :rtype: None
    """
    ch3 = molecule("CH3")
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    attached = attach(ch3, settings)
    assert attached.info["spin"] == 2
    assert attached.info["charge"] == 0
    energy = attached.get_potential_energy()
    forces = attached.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (4, 3)
    assert np.all(np.isfinite(forces))


def test_attach_unregistered_engine() -> None:
    """
    Verify unregistered engine name raises KeyError.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    settings = EngineSettings(name="unknown", model="test", device="cpu")
    with pytest.raises(KeyError) as exc_info:
        attach(hcho, settings)
    assert "unknown" in str(exc_info.value)


def test_orca_command_unset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify orca_command raises RuntimeError when NRNET_ORCA is unset or empty.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :return: None.
    :rtype: None
    """
    monkeypatch.delenv(ORCA_COMMAND_ENV, raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        orca_command()
    assert ORCA_COMMAND_ENV in str(exc_info.value)

    monkeypatch.setenv(ORCA_COMMAND_ENV, "")
    with pytest.raises(RuntimeError) as empty_exc_info:
        orca_command()
    assert ORCA_COMMAND_ENV in str(empty_exc_info.value)


def test_load_dft_calculator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Verify load_dft_calculator builds an ORCA calculator with valid parameters.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    monkeypatch.setenv(ORCA_COMMAND_ENV, "/mock/bin/orca")
    settings = load_engine_settings("configs/engines.yaml", "dft")
    scratch_dir = tmp_path / "scratch"
    extra = dict(settings.extra)
    extra["scratch_dir"] = str(scratch_dir)
    dft_settings = replace(settings, extra=extra)

    calc_singlet = load_dft_calculator(dft_settings, 1)
    calc_doublet = load_dft_calculator(dft_settings, 2)

    assert calc_singlet.parameters["orcasimpleinput"] == (
        "wB97M-V def2-TZVPD EnGrad TightSCF"
    )
    assert "%pal nprocs 16 end" in calc_singlet.parameters["orcablocks"]
    assert "%maxcore 3000" in calc_singlet.parameters["orcablocks"]

    assert calc_singlet.parameters["charge"] == 0
    assert calc_singlet.parameters["mult"] == 1
    assert calc_doublet.parameters["charge"] == 0
    assert calc_doublet.parameters["mult"] == 2

    singlet_path = Path(calc_singlet.directory)
    doublet_path = Path(calc_doublet.directory)
    assert singlet_path.is_relative_to(tmp_path)
    assert doublet_path.is_relative_to(tmp_path)
    assert singlet_path != doublet_path
    assert singlet_path.exists()
    assert doublet_path.exists()


@pytest.mark.engine
def test_mace_engine() -> None:
    """
    Verify MACE-OMol evaluation on HCHO singlet and CH3 doublet.

    :return: None.
    :rtype: None
    """
    pytest.importorskip("mace")
    settings = load_engine_settings("configs/engines.yaml", "mace")
    assert_energy_and_forces(molecule("H2CO"), settings, 1)
    assert_energy_and_forces(molecule("CH3"), settings, 2)


@pytest.mark.engine
def test_egret_engine() -> None:
    """
    Verify Egret-1 evaluation on HCHO and rejection of CH3 doublet.

    :return: None.
    :rtype: None
    """
    pytest.importorskip("mace")
    settings = load_engine_settings("configs/engines.yaml", "egret")
    assert_energy_and_forces(molecule("H2CO"), settings, 1)
    ch3 = molecule("CH3")
    with pytest.raises(UnsupportedSpecies):
        attach(ch3, settings)


@pytest.mark.engine
def test_uma_engine() -> None:
    """
    Verify UMA evaluation on HCHO and CH3 doublet with token check.

    :return: None.
    :rtype: None
    """
    pytest.importorskip("fairchem.core")
    import huggingface_hub

    token = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    if not token:
        pytest.skip("No Hugging Face token available for UMA")
    settings = load_engine_settings("configs/engines.yaml", "uma")
    assert_energy_and_forces(molecule("H2CO"), settings, 1)
    assert_energy_and_forces(molecule("CH3"), settings, 2)


@pytest.mark.engine
def test_aimnet_engine() -> None:
    """
    Verify AIMNet2 evaluation on HCHO singlet and CH3 doublet.

    :return: None.
    :rtype: None
    """
    pytest.importorskip("aimnet")
    settings = load_engine_settings("configs/engines.yaml", "aimnet")
    assert_energy_and_forces(molecule("H2CO"), settings, 1)
    assert_energy_and_forces(molecule("CH3"), settings, 2)


@pytest.mark.engine
def test_dft_engine_h2(tmp_path: Path) -> None:
    """
    Verify ORCA DFT single-point execution on H2 when NRNET_ORCA is set.

    :param tmp_path: Pytest temporary directory fixture.
    :type tmp_path: Path
    :return: None.
    :rtype: None
    """
    if not os.environ.get(ORCA_COMMAND_ENV):
        pytest.skip(f"Environment variable {ORCA_COMMAND_ENV} is not set")
    settings = load_engine_settings("configs/engines.yaml", "dft")
    scratch = tmp_path / "orca_h2"
    extra = dict(settings.extra)
    extra["scratch_dir"] = str(scratch)
    extra["nprocs"] = 1
    fast_settings = replace(settings, model="HF 3-21G", extra=extra)
    assert_energy_and_forces(molecule("H2"), fast_settings, 1)
