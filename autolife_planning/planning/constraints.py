"""User-defined manifold constraints, CasADi-backed.

Users write the constraint equation as a CasADi symbolic expression
in their own script.  The wrapper handles symbolic Jacobian via
autodiff, C codegen, compilation, caching, and hand-off to the
native C++ ``CompiledConstraint`` adapter.

No prebuilt constraint primitives are shipped.  Every constraint is
defined inline by the caller as a function of the planner's active
joint vector — typically built from :class:`SymbolicContext` (defined
in :mod:`autolife_planning.planning.symbolic`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca

from .symbolic import SymbolicContext, compile_function


@dataclass
class Constraint:
    """A user-defined holonomic constraint, JIT-compiled via CasADi.

    Constructing this class takes the symbolic Jacobian via
    ``ca.jacobian(residual, q_sym)``, generates C and compiles it to a
    cached shared library (see :func:`compile_function`).
    """

    residual: ca.SX
    q_sym: ca.SX
    name: str = "constraint"

    so_path: Path = field(init=False)
    ambient_dim: int = field(init=False)
    co_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.q_sym, ca.SX):
            raise TypeError("Constraint.q_sym must be a CasADi SX symbol")
        res = ca.reshape(self.residual, -1, 1)
        jac = ca.densify(ca.jacobian(res, self.q_sym))
        self.ambient_dim = int(self.q_sym.numel())
        self.co_dim = int(res.numel())
        self.so_path = compile_function(
            ca.Function(self.name, [self.q_sym], [res, jac]).expand(), "constraint"
        )

    @property
    def symbol_name(self) -> str:
        return self.name


__all__ = ["Constraint", "SymbolicContext"]
