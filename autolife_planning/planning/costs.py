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

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca

from .symbolic import _cwd


def _cache_root() -> Path:
    """Return the cost cache directory.

    Honours ``AUTOLIFE_COST_CACHE_DIR`` if set.  Otherwise falls back
    to ``~/.cache/autolife_planning/costs``.
    """
    override = os.environ.get("AUTOLIFE_COST_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "autolife_planning" / "costs").resolve()


@dataclass
class Cost:
    """A user-defined soft path cost, JIT-compiled via CasADi.

    Pass a scalar CasADi expression in the planner's active joint
    vector.  Construction triggers (on cold cache):

        1. symbolic gradient via ``ca.gradient(expression, q_sym)``
        2. C code generation via CasADi
        3. compilation to a ``.so`` with ``c++ -O3 -shared -fPIC``
        4. caching under ``~/.cache/autolife_planning/costs/<sha>/``

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

    _so_path: Path = field(init=False)
    _ambient_dim: int = field(init=False)
    _symbol_name: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.q_sym, ca.SX):
            raise TypeError("Cost.q_sym must be a CasADi SX symbol")

        expr = ca.SX(self.expression)
        if expr.numel() != 1:
            raise ValueError(f"Cost.expression must be scalar; got shape {expr.shape}")
        if self.weight < 0:
            raise ValueError("Cost.weight must be >= 0")

        self._ambient_dim = int(self.q_sym.numel())

        # Gradient comes from CasADi autodiff — it's a (n, 1) column
        # vector with the same storage order Eigen expects.  Shipping
        # it alongside the scalar keeps the ABI uniform with Constraint
        # (both are: 1 input, 2 outputs).  Gradient-aware planners
        # such as TRRT can pick it up; RRT*/BIT* simply ignore it.
        grad = ca.densify(ca.gradient(expr, self.q_sym))

        f = ca.Function(self.name, [self.q_sym], [expr, grad]).expand()

        raw = f.serialize()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()

        cache_dir = _cache_root() / sha[:2] / sha[2:]
        cache_dir.mkdir(parents=True, exist_ok=True)

        c_path = cache_dir / "cost.c"
        so_path = cache_dir / "cost.so"

        if not so_path.exists():
            sys.stderr.write(f"[autolife] compiling cost {sha[:8]}... ")
            sys.stderr.flush()
            t0 = time.perf_counter()
            with _cwd(cache_dir):
                f.generate("cost.c")
            compiler = os.environ.get("AUTOLIFE_COST_CC", "c++")
            subprocess.run(
                [
                    compiler,
                    "-O3",
                    "-shared",
                    "-fPIC",
                    str(c_path),
                    "-o",
                    str(so_path),
                ],
                check=True,
            )
            dt = time.perf_counter() - t0
            sys.stderr.write(f"done ({dt * 1000:.0f} ms)\n")
            sys.stderr.flush()

        self._so_path = so_path
        self._symbol_name = self.name

    @property
    def so_path(self) -> Path:
        return self._so_path

    @property
    def ambient_dim(self) -> int:
        return self._ambient_dim

    @property
    def symbol_name(self) -> str:
        return self._symbol_name


__all__ = ["Cost"]
