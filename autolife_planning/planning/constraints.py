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

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca

from .symbolic import SymbolicContext, _cwd


def _cache_root() -> Path:
    """Return the constraint cache directory.

    Honours ``AUTOLIFE_CONSTRAINT_CACHE_DIR`` if set (useful for CI).
    Otherwise falls back to ``~/.cache/autolife_planning/constraints``.
    """
    override = os.environ.get("AUTOLIFE_CONSTRAINT_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "autolife_planning" / "constraints").resolve()


@dataclass
class Constraint:
    """A user-defined holonomic constraint, JIT-compiled via CasADi.

    Constructing this class triggers (on cold cache):
        1. symbolic Jacobian via ``ca.jacobian(residual, q_sym)``
        2. C code generation via CasADi
        3. compilation to a ``.so`` with ``c++ -O3 -shared -fPIC``
        4. caching under ``~/.cache/autolife_planning/constraints/<sha>/``

    On a cache hit the whole thing is a single ``stat`` + string compare.
    """

    residual: ca.SX
    q_sym: ca.SX
    name: str = "constraint"

    _so_path: Path = field(init=False)
    _ambient_dim: int = field(init=False)
    _co_dim: int = field(init=False)
    _symbol_name: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.q_sym, ca.SX):
            raise TypeError("Constraint.q_sym must be a CasADi SX symbol")

        res = ca.reshape(self.residual, -1, 1)

        self._ambient_dim = int(self.q_sym.numel())
        self._co_dim = int(res.numel())

        jac = ca.densify(ca.jacobian(res, self.q_sym))

        f = ca.Function(self.name, [self.q_sym], [res, jac]).expand()

        raw = f.serialize()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()

        cache_dir = _cache_root() / sha[:2] / sha[2:]
        cache_dir.mkdir(parents=True, exist_ok=True)

        c_path = cache_dir / "constraint.c"
        so_path = cache_dir / "constraint.so"

        if not so_path.exists():
            sys.stderr.write(f"[autolife] compiling constraint {sha[:8]}... ")
            sys.stderr.flush()
            t0 = time.perf_counter()
            with _cwd(cache_dir):
                f.generate("constraint.c")
            compiler = os.environ.get("AUTOLIFE_CONSTRAINT_CC", "c++")
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
    def co_dim(self) -> int:
        return self._co_dim

    @property
    def symbol_name(self) -> str:
        return self._symbol_name


__all__ = ["Constraint", "SymbolicContext"]
