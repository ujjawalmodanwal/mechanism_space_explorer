"""
Unit tests for atomic geometry relaxation driver.

Verifies convergence of stretched molecular geometries, termination on step
limits, calculator detachment, and proper exception handling.
"""

import numpy as np
import pytest
from ase.build import molecule

from nrnet.engines import attach, load_engine_settings
from nrnet.optimize import RelaxResult, relax


def test_relax_result_dataclass() -> None:
    """
    Verify RelaxResult dataclass initialization and field access.

    :return: None.
    :rtype: None
    """
    atoms = molecule("H2")
    res = RelaxResult(atoms=atoms, converged=True, steps=5)
    assert res.atoms is atoms
    assert res.converged is True
    assert res.steps == 5


def test_relax_without_calculator_raises() -> None:
    """
    Verify relax raises ValueError when passed structure without calculator.

    :return: None.
    :rtype: None
    """
    hcho = molecule("H2CO")
    with pytest.raises(
        ValueError, match="Atoms object must have an attached calculator"
    ):
        relax(hcho, fmax_ev_per_ang=0.05, max_steps=500)


def test_relax_converges_stretched_hcho() -> None:
    """
    Verify relax converges a stretched formaldehyde structure with tblite.

    :return: None.
    :rtype: None
    """
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    hcho = molecule("H2CO")
    pos = hcho.get_positions()
    pos[1][0] += 0.3
    hcho.set_positions(pos)
    attached = attach(hcho, settings)
    res = relax(attached, fmax_ev_per_ang=0.05, max_steps=500)
    assert res.converged is True
    assert res.steps > 0
    assert res.atoms.calc is None
    eval_atoms = attach(res.atoms, settings)
    forces = eval_atoms.get_forces()
    assert np.max(np.linalg.norm(forces, axis=1)) <= 0.05


def test_relax_unconverged_max_steps() -> None:
    """
    Verify relax reports unconverged status when max_steps is exceeded.

    :return: None.
    :rtype: None
    """
    settings = load_engine_settings("configs/engines.yaml", "xtb")
    hcho = molecule("H2CO")
    pos = hcho.get_positions()
    pos[1][0] += 0.5
    hcho.set_positions(pos)
    attached = attach(hcho, settings)
    res = relax(attached, fmax_ev_per_ang=0.001, max_steps=1)
    assert res.converged is False
    assert res.steps == 1
    assert res.atoms.calc is None
