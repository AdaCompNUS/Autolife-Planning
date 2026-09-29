/**
 * Time-optimal trajectory parameterisation Python extension.
 *
 * Thin nanobind wrapper around the standalone Kunz-Stilman TOTG core in
 * totg.{hpp,cpp}.  Keeps the Path and the Trajectory that references it
 * alive in one owning handle so Python users do not need to manage their
 * lifetimes.
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "totg.hpp"

namespace nb = nanobind;
using autolife::trajectory::DEFAULT_PATH_TOLERANCE;
using autolife::trajectory::Path;
using autolife::trajectory::Trajectory;

namespace {

// Convert a row-major (N, ndof) matrix into a vector<VectorXd> so the
// TOTG core (which keeps an internal std::list of waypoints) has a
// stable, row-independent view regardless of the incoming memory layout.
std::vector<Eigen::VectorXd> waypoints_from_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& waypoints) {
  std::vector<Eigen::VectorXd> out;
  out.reserve(static_cast<std::size_t>(waypoints.rows()));
  for (Eigen::Index i = 0; i < waypoints.rows(); ++i) {
    out.emplace_back(waypoints.row(i).transpose());
  }
  return out;
}

struct TotgTrajectory {
  // Path stored by unique_ptr so the Trajectory-held reference is stable
  // across moves of the TotgTrajectory value (Trajectory holds a Path by
  // value internally, but keeping a heap-stable Path here avoids surprise
  // if we ever extend with alternative constructors).
  std::unique_ptr<Path> path;
  Trajectory trajectory;

  TotgTrajectory(Path p, Trajectory t)
      : path(std::make_unique<Path>(std::move(p))), trajectory(std::move(t)) {}

  double duration() const { return trajectory.getDuration(); }

  Eigen::VectorXd position(double t) const { return trajectory.getPosition(t); }
  Eigen::VectorXd velocity(double t) const { return trajectory.getVelocity(t); }
  Eigen::VectorXd acceleration(double t) const {
    return trajectory.getAcceleration(t);
  }

  // Sample the trajectory at a user-supplied time grid.  All three state
  // arrays are returned packed into (T, ndof) matrices so one C++/Python
  // round-trip produces a complete rollout.
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> sample(
      const Eigen::Ref<const Eigen::VectorXd>& times) const {
    const Eigen::Index T = times.size();
    const Eigen::Index ndof = trajectory.getPosition(0.0).size();
    Eigen::MatrixXd positions(T, ndof);
    Eigen::MatrixXd velocities(T, ndof);
    Eigen::MatrixXd accelerations(T, ndof);
    for (Eigen::Index i = 0; i < T; ++i) {
      positions.row(i) = trajectory.getPosition(times[i]).transpose();
      velocities.row(i) = trajectory.getVelocity(times[i]).transpose();
      accelerations.row(i) = trajectory.getAcceleration(times[i]).transpose();
    }
    return {std::move(positions), std::move(velocities),
            std::move(accelerations)};
  }

  // Uniformly-spaced rollout with step dt.  The last sample is clamped to
  // the trajectory duration so the returned grid always includes the
  // endpoint, matching what a streaming controller expects.
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
  sample_uniform(double dt) const {
    if (dt <= 0.0) {
      throw std::invalid_argument("dt must be > 0");
    }
    const double total = trajectory.getDuration();
    const Eigen::Index T =
        static_cast<Eigen::Index>(std::floor(total / dt)) + 1;
    Eigen::VectorXd times(T + 1);
    for (Eigen::Index i = 0; i < T; ++i) {
      times[i] = static_cast<double>(i) * dt;
    }
    times[T] = total;
    // De-duplicate the last point if the grid already lands exactly on the end.
    Eigen::Index samples = T + 1;
    if (std::abs(times[T] - times[T - 1]) < 1e-12) {
      samples = T;
    }
    Eigen::VectorXd out_times = times.head(samples);
    auto [positions, velocities, accelerations] = sample(out_times);
    return {std::move(out_times), std::move(positions), std::move(velocities),
            std::move(accelerations)};
  }
};

std::optional<TotgTrajectory> compute_trajectory(
    const Eigen::Ref<const Eigen::MatrixXd>& waypoints,
    const Eigen::Ref<const Eigen::VectorXd>& max_velocity,
    const Eigen::Ref<const Eigen::VectorXd>& max_acceleration,
    double max_deviation, double time_step) {
  if (waypoints.rows() < 2) {
    return std::nullopt;
  }
  if (waypoints.cols() != max_velocity.size() ||
      waypoints.cols() != max_acceleration.size()) {
    throw std::invalid_argument(
        "max_velocity and max_acceleration must match waypoints column count");
  }

  auto waypoint_list = waypoints_from_matrix(waypoints);
  auto path = Path::create(waypoint_list, max_deviation);
  if (!path) {
    return std::nullopt;
  }
  auto traj =
      Trajectory::create(*path, max_velocity, max_acceleration, time_step);
  if (!traj) {
    return std::nullopt;
  }
  return TotgTrajectory(std::move(*path), std::move(*traj));
}

}  // namespace

NB_MODULE(_time_parameterization, m) {
  m.doc() =
      "Time-optimal trajectory parameterization (TOTG / Kunz-Stilman, 2012).";

  m.attr("DEFAULT_PATH_TOLERANCE") = DEFAULT_PATH_TOLERANCE;

  nb::class_<TotgTrajectory>(m, "TotgTrajectory")
      .def_prop_ro("duration", &TotgTrajectory::duration,
                   "Total duration of the trajectory (seconds).")
      .def("position", &TotgTrajectory::position, nb::arg("t"),
           "Configuration at time ``t`` (seconds).")
      .def("velocity", &TotgTrajectory::velocity, nb::arg("t"),
           "Joint velocity at time ``t`` (seconds).")
      .def("acceleration", &TotgTrajectory::acceleration, nb::arg("t"),
           "Joint acceleration at time ``t`` (seconds).")
      .def(
          "sample",
          [](const TotgTrajectory& self,
             const Eigen::Ref<const Eigen::VectorXd>& times) {
            auto [p, v, a] = self.sample(times);
            return nb::make_tuple(std::move(p), std::move(v), std::move(a));
          },
          nb::arg("times"),
          "Sample (position, velocity, acceleration) at each time in "
          "``times``.\n"
          "Returns a 3-tuple of ``(T, ndof)`` matrices.")
      .def(
          "sample_uniform",
          [](const TotgTrajectory& self, double dt) {
            auto [t, p, v, a] = self.sample_uniform(dt);
            return nb::make_tuple(std::move(t), std::move(p), std::move(v),
                                  std::move(a));
          },
          nb::arg("dt"),
          "Uniformly-spaced rollout with step ``dt``.\n"
          "Returns ``(times, positions, velocities, accelerations)``.  "
          "The returned ``times`` always include ``t = 0`` and ``t = "
          "duration``.");

  m.def("compute_trajectory", &compute_trajectory, nb::arg("waypoints"),
        nb::arg("max_velocity"), nb::arg("max_acceleration"),
        nb::arg("max_deviation") = DEFAULT_PATH_TOLERANCE,
        nb::arg("time_step") = 1e-3,
        "Compute a time-optimal trajectory for a piecewise-linear path.\n\n"
        "Returns ``None`` if the path has fewer than two waypoints, the bounds "
        "are infeasible, or the integrator failed to converge.");
}
