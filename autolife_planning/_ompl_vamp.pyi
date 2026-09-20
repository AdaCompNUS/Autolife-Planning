"""Type stubs for the ``_ompl_vamp`` C++ extension.

The actual implementation is in ``ext/ompl_vamp/ompl_vamp_ext.cpp`` and
ships as a compiled ``_ompl_vamp.cpython-*.so`` next to this file in the
installed package.  This stub mirrors the nanobind bindings exactly so
type checkers can resolve ``import autolife_planning._ompl_vamp``.
"""

from collections.abc import Sequence
from typing import overload

class PlanResult:
    """Result of a single ``OmplVampPlanner.plan`` call."""

    @property
    def solved(self) -> bool:
        """``True`` if OMPL returned a solution within the time limit."""
        ...

    @property
    def path(self) -> list[list[float]]:
        """Solution waypoints in the planner's active joint space.

        Each inner list is one configuration with length equal to
        :meth:`OmplVampPlanner.dimension`.  Empty when ``solved`` is
        false.
        """
        ...

    @property
    def planning_time_ns(self) -> int:
        """Wall-clock time spent inside ``ss.solve(...)``, in nanoseconds."""
        ...

    @property
    def path_cost(self) -> float:
        """Geometric length of the (possibly simplified) solution path.

        ``inf`` when ``solved`` is false.
        """
        ...

class OmplVampPlanner:
    """OMPL planner with VAMP SIMD-accelerated collision checking.

    Two construction modes:

    * ``OmplVampPlanner()`` — full body, 24 DOF (3 base + 21 joints).
    * ``OmplVampPlanner(active_indices, frozen_config)`` — subgroup
      planner over the joints listed in ``active_indices``; the C++
      collision checker injects ``frozen_config`` for every other slot
      in the 24-DOF body on every state and motion validity query.
    """

    @overload
    def __init__(self) -> None:
        """Create a full-body planner (24 DOF)."""
        ...

    @overload
    def __init__(
        self,
        active_indices: Sequence[int],
        frozen_config: Sequence[float],
    ) -> None:
        """Create a subgroup planner.

        Args:
            active_indices: Positions in the full 24-DOF body that this
                planner will plan over, in DOF order.
            frozen_config: 24-DOF stance to inject for every joint *not*
                in ``active_indices``.
        """
        ...

    def add_pointcloud(
        self,
        points: Sequence[Sequence[float]],
        r_min: float,
        r_max: float,
        point_radius: float,
    ) -> None:
        """Set the scene pointcloud.

        The planner holds at most one cloud; calling this replaces any
        previously-registered cloud.

        Args:
            points: ``(N, 3)`` array of obstacle positions in world frame.
            r_min: Minimum robot collision-sphere radius (from
                :meth:`min_max_radii`).
            r_max: Maximum robot collision-sphere radius (from
                :meth:`min_max_radii`).
            point_radius: Inflation radius applied to every cloud point.
        """
        ...

    def remove_pointcloud(self) -> bool:
        """Drop the currently-registered pointcloud.

        Returns ``False`` if there was no cloud to remove.
        """
        ...

    def has_pointcloud(self) -> bool:
        """``True`` if a pointcloud is currently registered."""
        ...

    def add_sphere(self, center: Sequence[float], radius: float) -> None:
        """Add a single sphere obstacle (centre + radius) to the environment."""
        ...

    def clear_environment(self) -> None:
        """Remove all obstacles from the collision environment."""
        ...

    def add_compiled_constraint(
        self,
        so_path: str,
        symbol_name: str,
        ambient_dim: int,
        co_dim: int,
    ) -> None:
        """Load a CasADi-generated shared library as an OMPL constraint.

        Args:
            so_path: Path to the compiled ``.so`` file.
            symbol_name: CasADi function symbol name inside the library.
            ambient_dim: Dimension of the joint space (must match
                :meth:`dimension`).
            co_dim: Number of constraint equations (rows of the residual).
        """
        ...

    def clear_constraints(self) -> None:
        """Drop every accumulated constraint."""
        ...

    def num_constraints(self) -> int:
        """Number of constraints currently registered."""
        ...

    def add_compiled_cost(
        self,
        so_path: str,
        symbol_name: str,
        ambient_dim: int,
        weight: float = 1.0,
    ) -> None:
        """Load a CasADi-generated shared library as a soft path cost.

        The cost is wrapped as an ``ompl::StateCostIntegralObjective``
        — trapezoidally integrated along every motion — and drives
        the search of asymptotically-optimal planners (RRT*, BIT*,
        AIT*, …).  Multiple costs are summed with their ``weight``.

        Args:
            so_path: Path to the compiled ``.so`` file.
            symbol_name: CasADi function symbol name inside the library.
            ambient_dim: Dimension of the joint space (must match
                :meth:`dimension`).
            weight: Positive scalar multiplier applied to the cost.
        """
        ...

    def clear_costs(self) -> None:
        """Drop every accumulated cost."""
        ...

    def num_costs(self) -> int:
        """Number of costs currently registered."""
        ...

    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        planner_name: str = "rrtc",
        time_limit: float = 10.0,
        simplify: bool = True,
        interpolate: bool = True,
        interpolate_count: int = 0,
        resolution: float = 64.0,
    ) -> PlanResult:
        """Plan a collision-free path from ``start`` to ``goal``.

        Args:
            start: Active-DOF start configuration (length :meth:`dimension`).
            goal: Active-DOF goal configuration (length :meth:`dimension`).
            planner_name: OMPL planner identifier (e.g. ``"rrtc"``,
                ``"rrtstar"``, ``"prm"``, ``"bitstar"``).
            time_limit: Solver time limit in seconds.
            simplify: If true, run ``SimpleSetup::simplifySolution`` on the
                returned path.
            interpolate: If true, densify the simplified path.  Density
                is picked by ``interpolate_count`` when it is ``> 0``,
                otherwise by ``resolution`` when it is ``> 0``, otherwise
                by OMPL's default longest-valid-segment fraction.
            interpolate_count: Target total waypoint count for the whole
                path.  OMPL distributes states across edges proportionally
                to their length.  ``0`` disables this knob.  Cannot be
                combined with ``resolution``.
            resolution: Waypoints per unit of state-space distance.  Each
                edge of length ``d`` is split into ``ceil(d * resolution)``
                equal segments, so higher values give denser paths.
                Default ``64.0``.  Set to ``0.0`` to fall back to OMPL's
                default interpolator.  Cannot be combined with
                ``interpolate_count``.
        """
        ...

    def simplify_path(
        self,
        path: Sequence[Sequence[float]],
        time_limit: float = 1.0,
    ) -> list[list[float]]:
        """Run OMPL's shortcut simplifier on a waypoint list.

        Reuses the current collision environment and constraints.
        Shortcuts only consult the motion validator, so custom soft
        costs are ignored.  Returns the simplified path as waypoints.

        Args:
            path: ``(N, dimension())`` waypoint list.
            time_limit: Simplifier wall-clock budget, seconds.
        """
        ...

    def interpolate_path(
        self,
        path: Sequence[Sequence[float]],
        count: int = 0,
        resolution: float = 64.0,
    ) -> list[list[float]]:
        """Densify a waypoint list along its existing edges.

        Pass at most one of ``count`` (exact total waypoints,
        distributed by edge length) or ``resolution`` (waypoints per
        unit of state-space distance).  Both zero falls back to OMPL's
        default longest-valid-segment fraction.  No collision check is
        performed — densification only calls ``StateSpace::interpolate``
        on the existing edges.

        Args:
            path: ``(N, dimension())`` waypoint list.
            count: Total waypoint count if > 0.
            resolution: Waypoints per unit distance if > 0.0.
        """
        ...

    def validate(self, config: Sequence[float]) -> bool:
        """Return ``True`` if ``config`` is collision-free.

        ``config`` must have length :meth:`dimension`.  Subgroup
        planners expand it to a full 24-DOF state with the stored
        ``frozen_config`` before checking.
        """
        ...

    def validate_batch(
        self,
        configs: Sequence[Sequence[float]],
    ) -> list[bool]:
        """Batched collision check — deep SIMD, per-config result.

        Packs up to ``rake`` distinct configurations into a single
        VAMP ``ConfigurationBlock<rake>`` so one ``Robot::fkcc<rake>``
        call sphere-FKs and collision-checks all of them
        simultaneously — the same SIMD primitive the motion-edge
        validator uses for interpolated samples, fed independent
        configs per lane instead.

        Returns one bool per input config in input order.  In the
        common case (most configs valid) costs ``ceil(N / rake)``
        SIMD calls; when a packed block fails, only that block falls
        back to per-lane single-state checks.

        Each ``configs[i]`` must have length :meth:`dimension`.
        Subgroup planners expand each to a full 24-DOF state with
        the stored ``frozen_config`` before packing.
        """
        ...

    def dimension(self) -> int:
        """Number of active joints — 24 for the full body, smaller for subgroups."""
        ...

    def lower_bounds(self) -> list[float]:
        """Per-joint lower bounds for the active DOFs."""
        ...

    def upper_bounds(self) -> list[float]:
        """Per-joint upper bounds for the active DOFs."""
        ...

    def min_max_radii(self) -> tuple[float, float]:
        """``(min_radius, max_radius)`` of the robot's collision spheres.

        Pass these to :meth:`add_pointcloud` so VAMP can index its
        broadphase correctly.
        """
        ...

    def set_subgroup(
        self,
        active_indices: Sequence[int],
        frozen_config: Sequence[float],
    ) -> None:
        """Switch to a different subgroup without rebuilding the environment.

        Clears all constraints.  The pointcloud and collision geometry
        are preserved.

        Args:
            active_indices: Positions in the full 24-DOF body that this
                planner will plan over, in DOF order.
            frozen_config: 24-DOF stance to inject for every joint *not*
                in ``active_indices``.
        """
        ...

    def set_full_body(self) -> None:
        """Switch back to full-body planning (24 DOF).

        Clears all constraints.  The pointcloud and collision geometry
        are preserved.
        """
        ...
