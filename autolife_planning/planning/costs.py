"""User-defined soft path costs, CasADi-backed.

The asymptotically optimal planners (RRT*, BIT*, AIT*, …) minimise an
``ompl::base::OptimizationObjective``.  Out of the box OMPL only knows
about the geometric length of the path; this module lets users add
their own cost without touching C++.

The user writes a scalar CasADi expression in terms of the planner's
active joint symbol ``q`` — typically built from
:class:`autolife_planning.planning.symbolic.SymbolicContext`.  The
wrapper takes the gradient via CasADi autodiff, generates C, compiles
to a ``.so``, and caches the artefact so the next run is essentially
free.  At plan time the C++ planner ``dlopen``'s the library and wraps
it as a ``StateCostIntegralObjective`` — OMPL trapezoidally integrates
the per-state cost along each edge, which is the standard soft-cost
treatment for RRT*-family planners.

The design intentionally mirrors
:class:`autolife_planning.planning.constraints.Constraint`: same
CasADi SymPy-like authoring experience, same cache layout, same
ambient-dimension check at registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca

from .symbolic import compile_function


@dataclass
class Cost:
    """A user-defined soft path cost, JIT-compiled via CasADi.

    Pass a scalar CasADi expression in the planner's active joint
    vector.  Construction triggers (on cold cache):

        1. symbolic gradient via ``ca.gradient(expression, q_sym)``
        2. C code generation and compilation to a cached shared library
           (see :func:`~autolife_planning.planning.symbolic.compile_function`)

    The C++ planner wraps the loaded function in an OMPL
    ``StateCostIntegralObjective`` — so per-state values are
    trapezoidally integrated along every motion, which is what
    RRT*-family optimal planners expect.

    The expression must be non-negative (OMPL objectives accumulate
    with ``operator+`` and the optimal-planner tooling assumes the
    zero cost is the minimum).  We don't enforce this at runtime
    because that would require evaluating the symbolic expression,
    but violating it produces nonsensical RRT* solutions.
    """

    expression: ca.SX
    q_sym: ca.SX
    name: str = "cost"
    weight: float = 1.0

    so_path: Path = field(init=False)
    ambient_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.q_sym, ca.SX):
            raise TypeError("Cost.q_sym must be a CasADi SX symbol")
        expr = ca.SX(self.expression)
        if expr.numel() != 1:
            raise ValueError(f"Cost.expression must be scalar; got shape {expr.shape}")
        if self.weight < 0:
            raise ValueError("Cost.weight must be >= 0")
        # Ship the autodiff gradient alongside the value so the ABI matches
        # Constraint (1 input, 2 outputs); gradient-aware planners such as
        # TRRT can use it, RRT*/BIT* ignore it.
        grad = ca.densify(ca.gradient(expr, self.q_sym))
        self.ambient_dim = int(self.q_sym.numel())
        self.so_path = compile_function(
            ca.Function(self.name, [self.q_sym], [expr, grad]).expand(), "cost"
        )

    @property
    def symbol_name(self) -> str:
        return self.name


__all__ = ["Cost"]
