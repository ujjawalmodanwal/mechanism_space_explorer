"""
Geometry relaxation using Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization.

Provides the relax driver and RelaxResult dataclass to find local energy
minima on potential energy surfaces using an already attached ASE calculator.
"""

from dataclasses import dataclass, field

from ase import Atoms
from ase.optimize import BFGS

__all__ = [
    "RelaxResult",
    "relax",
]


@dataclass(frozen=True, slots=True)
class RelaxResult:
    """
    Result of a geometry relaxation.

    :param atoms: Optimized atomic structure without attached calculator.
    :type atoms: Atoms
    :param converged: Whether geometry optimization reached convergence.
    :type converged: bool
    :param steps: Number of optimization steps taken.
    :type steps: int
    """

    atoms: Atoms = field(compare=False)
    converged: bool
    steps: int


def relax(
    atoms: Atoms,
    fmax_ev_per_ang: float = 0.05,
    max_steps: int = 500,
) -> RelaxResult:
    """
    Relax atomic positions to local minimum with BFGS.

    :param atoms: Atomic configuration with attached calculator.
    :type atoms: Atoms
    :param fmax_ev_per_ang: Maximum residual force threshold in eV/Å.
    :type fmax_ev_per_ang: float
    :param max_steps: Maximum allowable optimization steps.
    :type max_steps: int
    :return: Relaxation outcome with stripped calculator and convergence flag.
    :rtype: RelaxResult
    :raises ValueError: If atoms does not carry an attached calculator.
    """
    if atoms.calc is None:
        raise ValueError("Atoms object must have an attached calculator.")
    target = atoms.copy()
    target.calc = atoms.calc
    optimizer = BFGS(target, logfile=None)
    converged = optimizer.run(fmax=fmax_ev_per_ang, steps=max_steps)
    relaxed = target.copy()
    relaxed.calc = None
    return RelaxResult(
        atoms=relaxed,
        converged=bool(converged),
        steps=int(optimizer.get_number_of_steps()),
    )
