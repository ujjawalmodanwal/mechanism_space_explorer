"""
Engine settings, registry, and calculator attachment for molecular calculations.

Defines the configuration schema for computational chemistry engines, loads
engine settings from YAML definitions, and attaches ASE calculators to atomic
structures with proper charge and spin multiplicity state.
"""

import functools
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.orca import ORCA, OrcaProfile

from nrnet.chem import multiplicity

__all__ = [
    "ENGINE_LOADERS",
    "ORCA_COMMAND_ENV",
    "EngineLoader",
    "EngineSettings",
    "UnsupportedSpecies",
    "attach",
    "get_engine_loader",
    "load_aimnet_base_calculator",
    "load_aimnet_calculator",
    "load_dft_calculator",
    "load_egret_calculator",
    "load_egret_model",
    "load_engine_settings",
    "load_mace_calculator",
    "load_mace_model",
    "load_uma_calculator",
    "load_uma_predict_unit",
    "load_xtb_calculator",
    "orca_blocks",
    "orca_command",
    "orca_directory",
    "orca_simple_input",
    "parse_engine_settings",
]


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """
    Configuration parameters for a computational chemistry engine.

    :param name: Identifier of the engine (e.g. 'xtb', 'mace').
    :type name: str
    :param model: Specific model name or checkpoint path.
    :type model: str
    :param device: Execution device ('cpu' or 'cuda').
    :type device: str
    :param closed_shell_only: Whether the engine only supports singlets.
    :type closed_shell_only: bool
    :param extra: Additional engine-specific configuration parameters.
    :type extra: Mapping[str, str | int | float]
    """

    name: str
    model: str
    device: str
    closed_shell_only: bool = False
    extra: Mapping[str, str | int | float] = field(default_factory=dict)


class UnsupportedSpecies(ValueError):
    """
    Raised when an atomic species is unsupported by the requested engine.
    """


def load_xtb_calculator(
    settings: EngineSettings,
    multiplicity: int,
) -> Calculator:
    """
    Instantiate a tblite ASE calculator for xTB calculations.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Configured ASE calculator instance.
    :rtype: Calculator
    """
    # Each engine lives in its own environment (PLAN D5).
    from tblite.ase import TBLite

    return TBLite(
        method=settings.model,
        charge=0,
        multiplicity=multiplicity,
        verbosity=0,
    )


@functools.cache
def load_mace_model(model: str, device: str) -> Calculator:
    """
    Load and cache a MACE-OMol model instance.

    :param model: Model name or checkpoint path.
    :type model: str
    :param device: Execution device ('cpu' or 'cuda').
    :type device: str
    :return: MACE ASE calculator instance.
    :rtype: Calculator
    """
    # Each engine lives in its own environment (PLAN D5).
    from mace.calculators import mace_omol

    return mace_omol(model=model, device=device, default_dtype="float64")


def load_mace_calculator(
    settings: EngineSettings,
    multiplicity: int,
) -> Calculator:
    """
    Instantiate a MACE-OMol25 ASE calculator.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Configured MACE ASE calculator instance.
    :rtype: Calculator
    """
    return load_mace_model(settings.model, settings.device)


@functools.cache
def load_egret_model(model: str, device: str) -> Calculator:
    """
    Load and cache a Rowan Egret-1 model instance.

    :param model: Path to Egret checkpoint model file.
    :type model: str
    :param device: Execution device ('cpu' or 'cuda').
    :type device: str
    :return: Egret ASE calculator instance.
    :rtype: Calculator
    """
    # Each engine lives in its own environment (PLAN D5).
    from mace.calculators import mace_off

    return mace_off(model=model, device=device, default_dtype="float64")


def load_egret_calculator(
    settings: EngineSettings,
    multiplicity: int,
) -> Calculator:
    """
    Instantiate a Rowan Egret-1 ASE calculator.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Configured Egret ASE calculator instance.
    :rtype: Calculator
    """
    return load_egret_model(settings.model, settings.device)


@functools.cache
def load_uma_predict_unit(model: str, device: str) -> object:
    """
    Load and cache a FAIRChem UMA prediction unit.

    :param model: Model name or checkpoint path.
    :type model: str
    :param device: Execution device ('cpu' or 'cuda').
    :type device: str
    :return: Loaded MLIPPredictUnit object.
    :rtype: object
    """
    # Each engine lives in its own environment (PLAN D5).
    from fairchem.core.calculate import pretrained_mlip

    return pretrained_mlip.get_predict_unit(model, device=device)


def load_uma_calculator(
    settings: EngineSettings,
    multiplicity: int,
) -> Calculator:
    """
    Instantiate a FAIRChem UMA ASE calculator.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Configured FAIRChem ASE calculator instance.
    :rtype: Calculator
    """
    # Each engine lives in its own environment (PLAN D5).
    from fairchem.core import FAIRChemCalculator

    predict_unit = load_uma_predict_unit(settings.model, settings.device)
    return FAIRChemCalculator(predict_unit, task_name="omol")


@functools.cache
def load_aimnet_base_calculator(model: str, device: str) -> object:
    """
    Load and cache the base AIMNet2 potential model.

    :param model: Model name or checkpoint path.
    :type model: str
    :param device: Execution device ('cpu' or 'cuda').
    :type device: str
    :return: Base AIMNet2 calculator instance.
    :rtype: object
    """
    # Each engine lives in its own environment (PLAN D5).
    from aimnet.calculators import AIMNet2Calculator

    return AIMNet2Calculator(model=model, device=device)


def load_aimnet_calculator(
    settings: EngineSettings,
    multiplicity: int,
) -> Calculator:
    """
    Instantiate an AIMNet2 ASE calculator with spin multiplicity.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Configured AIMNet2 ASE calculator.
    :rtype: Calculator
    """
    # Each engine lives in its own environment (PLAN D5).
    from aimnet.calculators import AIMNet2ASE

    base_calc = load_aimnet_base_calculator(settings.model, settings.device)
    return AIMNet2ASE(base_calc, charge=0, mult=multiplicity)


ORCA_COMMAND_ENV = "NRNET_ORCA"


def orca_command() -> str:
    """
    Resolve the absolute path to the ORCA executable from the environment.

    :return: Executable path string for ORCA.
    :rtype: str
    :raises RuntimeError: If NRNET_ORCA is unset or empty.
    """
    command = os.environ.get(ORCA_COMMAND_ENV, "")
    if not command:
        raise RuntimeError(
            f"Environment variable {ORCA_COMMAND_ENV!r} is unset or empty; "
            f"ORCA requires an absolute executable path."
        )
    return command


def orca_simple_input(settings: EngineSettings) -> str:
    """
    Build the ORCA simple-input line for energy and gradient calculation.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Space-separated ORCA simple-input keyword string.
    :rtype: str
    """
    return f"{settings.model} EnGrad {settings.extra['scf']}"


def orca_blocks(settings: EngineSettings) -> str:
    """
    Format ORCA block options for parallel execution and memory allocation.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Multi-line string of ORCA block directives.
    :rtype: str
    """
    nprocs = settings.extra["nprocs"]
    maxcore = settings.extra["maxcore_mb"]
    return f"%pal nprocs {nprocs} end\n%maxcore {maxcore}"


def orca_directory(settings: EngineSettings) -> Path:
    """
    Create a unique scratch directory for an ORCA calculation instance.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Path to the newly created unique scratch directory.
    :rtype: Path
    """
    scratch_root = Path(str(settings.extra["scratch_dir"]))
    scratch_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="orca-", dir=scratch_root))


def load_dft_calculator(
    settings: EngineSettings,
    multiplicity: int,
) -> Calculator:
    """
    Instantiate an ORCA DFT ASE calculator.

    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :param multiplicity: Spin multiplicity (2S + 1).
    :type multiplicity: int
    :return: Configured ORCA ASE calculator instance.
    :rtype: Calculator
    """
    return ORCA(
        profile=OrcaProfile(command=orca_command()),
        directory=orca_directory(settings),
        charge=0,
        mult=multiplicity,
        orcasimpleinput=orca_simple_input(settings),
        orcablocks=orca_blocks(settings),
    )


EngineLoader = Callable[[EngineSettings, int], Calculator]

ENGINE_LOADERS: dict[str, EngineLoader] = {
    "xtb": load_xtb_calculator,
    "mace": load_mace_calculator,
    "egret": load_egret_calculator,
    "uma": load_uma_calculator,
    "aimnet": load_aimnet_calculator,
    "dft": load_dft_calculator,
}


def get_engine_loader(name: str) -> EngineLoader:
    """
    Look up the calculator loader for a registered engine.

    :param name: Engine name.
    :type name: str
    :return: Loader function for the engine.
    :rtype: EngineLoader
    :raises KeyError: If the engine is not registered.
    """
    if name not in ENGINE_LOADERS:
        known = ", ".join(repr(key) for key in sorted(ENGINE_LOADERS.keys()))
        raise KeyError(
            f"Unknown engine {name!r}; registered engines are: {known}"
        )
    return ENGINE_LOADERS[name]


def parse_engine_settings(
    name: str,
    entry: Mapping[str, object],
) -> EngineSettings:
    """
    Construct an EngineSettings instance from a configuration mapping.

    :param name: Engine identifier.
    :type name: str
    :param entry: Configuration mapping with model, device, and extra fields.
    :type entry: Mapping[str, object]
    :return: Validated EngineSettings dataclass.
    :rtype: EngineSettings
    """
    raw_extra = entry.get("extra")
    extra_data: dict[str, str | int | float] = (
        dict(raw_extra) if isinstance(raw_extra, Mapping) else {}
    )
    return EngineSettings(
        name=name,
        model=str(entry["model"]),
        device=str(entry.get("device", "cpu")),
        closed_shell_only=bool(entry.get("closed_shell_only", False)),
        extra=extra_data,
    )


def load_engine_settings(
    path: str | Path,
    name: str,
) -> EngineSettings:
    """
    Load configuration settings for an engine from a YAML file.

    :param path: Path to the YAML configuration file.
    :type path: str or Path
    :param name: Name of the engine configuration to load.
    :type name: str
    :return: Parsed engine settings dataclass.
    :rtype: EngineSettings
    :raises KeyError: If the requested engine name is not found in the file.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping) or name not in data:
        known = sorted(data.keys()) if isinstance(data, Mapping) else []
        known_str = ", ".join(repr(key) for key in known)
        raise KeyError(
            f"Unknown engine {name!r}; known engines are: {known_str}"
        )
    return parse_engine_settings(name, data[name])


def attach(
    atoms: Atoms,
    settings: EngineSettings,
) -> Atoms:
    """
    Attach an engine calculator to an atomic structure with spin and charge.

    :param atoms: Input atomic structure.
    :type atoms: Atoms
    :param settings: Engine configuration settings.
    :type settings: EngineSettings
    :return: Independent copy of atoms with calculator attached and metadata.
    :rtype: Atoms
    :raises UnsupportedSpecies: If engine only supports closed-shell singlets
        and the species is open-shell.
    :raises KeyError: If the engine loader is not registered.
    """
    target = atoms.copy()
    spin = multiplicity(target)
    if settings.closed_shell_only and spin != 1:
        raise UnsupportedSpecies(
            f"Engine {settings.name!r} requires closed-shell singlet, "
            f"got spin multiplicity {spin}."
        )
    target.info["charge"] = 0
    target.info["spin"] = spin
    loader = get_engine_loader(settings.name)
    target.calc = loader(settings, spin)
    return target
