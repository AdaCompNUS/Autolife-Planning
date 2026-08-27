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
import json
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


def cost_descriptor_path(name: str) -> Path:
    """Return the persistent descriptor path for a named compiled cost."""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not name or any(ch not in allowed for ch in name):
        raise ValueError(
            "Cost descriptor name may contain only letters, digits, '_' and '-'"
        )
    return _cache_root() / "descriptors" / f"{name}.json"


@dataclass(frozen=True)
class CompiledCost:
    """Metadata needed to load an already-compiled CasADi cost.

    Unlike :class:`Cost`, constructing this class performs no symbolic work or
    code generation.  A descriptor created by :meth:`Cost.save_descriptor` can
    therefore be restored cheaply in a new Python process.
    """

    so_path: Path
    symbol_name: str
    ambient_dim: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        so_path = Path(self.so_path).expanduser().resolve()
        if not so_path.is_file():
            raise FileNotFoundError(f"Compiled cost library not found: {so_path}")
        if not self.symbol_name:
            raise ValueError("CompiledCost.symbol_name must not be empty")
        if self.ambient_dim <= 0:
            raise ValueError("CompiledCost.ambient_dim must be > 0")
        if self.weight < 0:
            raise ValueError("CompiledCost.weight must be >= 0")
        object.__setattr__(self, "so_path", so_path)

    def save_descriptor(self, path: str | Path, *, cache_key: str) -> Path:
        """Atomically save this compiled cost's runtime metadata as JSON."""
        if not cache_key:
            raise ValueError("cache_key must not be empty")
        descriptor_path = Path(path).expanduser().resolve()
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "cache_key": cache_key,
            "so_path": str(self.so_path),
            "symbol_name": self.symbol_name,
            "ambient_dim": self.ambient_dim,
            "weight": self.weight,
        }
        temporary_path = descriptor_path.with_name(
            f".{descriptor_path.name}.{os.getpid()}.tmp"
        )
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, descriptor_path)
        return descriptor_path

    @classmethod
    def from_descriptor(
        cls,
        path: str | Path,
        *,
        expected_cache_key: str | None = None,
    ) -> "CompiledCost":
        """Load a compiled cost descriptor without constructing CasADi FK."""
        descriptor_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid cost descriptor JSON: {descriptor_path}") from exc

        if payload.get("format_version") != 1:
            raise ValueError(
                f"Unsupported cost descriptor format in {descriptor_path}: "
                f"{payload.get('format_version')!r}"
            )
        if (
            expected_cache_key is not None
            and payload.get("cache_key") != expected_cache_key
        ):
            raise ValueError(f"Stale cost descriptor: {descriptor_path}")

        try:
            return cls(
                so_path=Path(payload["so_path"]),
                symbol_name=str(payload["symbol_name"]),
                ambient_dim=int(payload["ambient_dim"]),
                weight=float(payload["weight"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing {exc.args[0]!r} in cost descriptor: {descriptor_path}"
            ) from exc


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

    def to_compiled(self) -> CompiledCost:
        """Return lightweight runtime metadata for this compiled cost."""
        return CompiledCost(
            so_path=self.so_path,
            symbol_name=self.symbol_name,
            ambient_dim=self.ambient_dim,
            weight=self.weight,
        )

    def save_descriptor(self, path: str | Path, *, cache_key: str) -> Path:
        """Persist metadata so future processes can skip symbolic rebuilding."""
        return self.to_compiled().save_descriptor(path, cache_key=cache_key)


__all__ = ["CompiledCost", "Cost", "cost_descriptor_path"]
