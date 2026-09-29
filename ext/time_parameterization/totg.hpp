/*
 * Copyright (c) 2011-2012, Georgia Tech Research Corporation
 * All rights reserved.
 *
 * Author: Tobias Kunz <tobias@gatech.edu>
 * Date:   05/2012
 *
 * Humanoid Robotics Lab, Georgia Institute of Technology
 * Director: Mike Stilman  (http://www.golems.org)
 *
 * Algorithm details and publications:
 *   Kunz, Tobias, and Mike Stilman. "Time-optimal trajectory generation for
 *   path following with bounded acceleration and velocity."
 *   Robotics: Science and Systems VIII (2012).
 *
 * Standalone adaptation of the implementation vendored in MoveIt 2
 *   (moveit_core/trajectory_processing/time_optimal_trajectory_generation.{h,cpp})
 * — stripped of ROS/MoveIt dependencies so it depends only on <Eigen>
 * and the C++17 standard library.  Algorithmic logic is unchanged from
 * the upstream Kunz reference implementation and the MoveIt port.
 *
 * Provided under the original BSD-style license; see LICENSE.TOTG
 * alongside this file for the full text.
 */

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace autolife::trajectory {

// Default radial blend deviation applied at interior waypoints so that the
// resulting path is differentiable everywhere.  Units are the state-space
// distance used by the waypoints (radians for revolute joints, meters for
// prismatic joints).
inline constexpr double DEFAULT_PATH_TOLERANCE = 0.1;

// -----------------------------------------------------------------------------
// PathSegment — abstract base for linear and circular segments making up a
// differentiable path.  Users do not construct these directly; Path::create
// builds the segment list internally.
// -----------------------------------------------------------------------------
class PathSegment {
 public:
  explicit PathSegment(double length = 0.0) : length_(length) {}
  virtual ~PathSegment() = default;

  double getLength() const { return length_; }

  virtual Eigen::VectorXd getConfig(double s) const = 0;
  virtual Eigen::VectorXd getTangent(double s) const = 0;
  virtual Eigen::VectorXd getCurvature(double s) const = 0;
  virtual std::list<double> getSwitchingPoints() const = 0;
  virtual PathSegment* clone() const = 0;

  double position_ = 0.0;

 protected:
  double length_;
};

// -----------------------------------------------------------------------------
// Path — piecewise-linear waypoint list with circular blends at the corners.
// A valid Path is C¹ continuous along its entire arc length.
// -----------------------------------------------------------------------------
class Path {
 public:
  // Build a path from at least two waypoints.  The circular blend at every
  // interior waypoint is sized so it never deviates from the original
  // piecewise-linear polyline by more than `max_deviation`.
  //
  // Returns std::nullopt if:
  //   * fewer than 2 waypoints are supplied,
  //   * max_deviation is not strictly positive, or
  //   * three consecutive waypoints form a 180° turn (unsupported).
  static std::optional<Path> create(
      const std::vector<Eigen::VectorXd>& waypoints,
      double max_deviation = DEFAULT_PATH_TOLERANCE);

  Path(const Path& other);
  Path(Path&&) = default;
  Path& operator=(const Path& other);
  Path& operator=(Path&&) = default;

  double getLength() const;
  Eigen::VectorXd getConfig(double s) const;
  Eigen::VectorXd getTangent(double s) const;
  Eigen::VectorXd getCurvature(double s) const;

  // Arc length of the next switching point at s >= current.  `discontinuity`
  // is set to true if the tangent is discontinuous there (linear→circular
  // corners, end of path), false for curvature-only discontinuities.
  double getNextSwitchingPoint(double s, bool& discontinuity) const;

  std::list<std::pair<double, bool>> getSwitchingPoints() const;

 private:
  Path() = default;

  PathSegment* getPathSegment(double& s) const;

  double length_ = 0.0;
  std::list<std::pair<double, bool>> switching_points_;
  std::list<std::unique_ptr<PathSegment>> path_segments_;
};

// -----------------------------------------------------------------------------
// Trajectory — time-optimal parameterization s(t) along a Path, subject to
// per-joint velocity and acceleration bounds.
// -----------------------------------------------------------------------------
class Trajectory {
 public:
  // Build a time-optimal trajectory.  `time_step` controls the forward-
  // integration step along the path (smaller = more accurate, slower).
  // Returns std::nullopt if the integration failed (infeasible bounds).
  static std::optional<Trajectory> create(
      const Path& path, const Eigen::VectorXd& max_velocity,
      const Eigen::VectorXd& max_acceleration, double time_step = 1e-3);

  double getDuration() const;

  Eigen::VectorXd getPosition(double time) const;
  Eigen::VectorXd getVelocity(double time) const;
  Eigen::VectorXd getAcceleration(double time) const;

 private:
  Trajectory(const Path& path, const Eigen::VectorXd& max_velocity,
             const Eigen::VectorXd& max_acceleration, double time_step);

  struct TrajectoryStep {
    TrajectoryStep() = default;
    TrajectoryStep(double path_pos, double path_vel)
        : path_pos_(path_pos), path_vel_(path_vel) {}
    double path_pos_ = 0.0;
    double path_vel_ = 0.0;
    double time_ = 0.0;
  };

  bool getNextSwitchingPoint(double path_pos,
                             TrajectoryStep& next_switching_point,
                             double& before_acceleration,
                             double& after_acceleration) const;
  bool getNextAccelerationSwitchingPoint(double path_pos,
                                         TrajectoryStep& next_switching_point,
                                         double& before_acceleration,
                                         double& after_acceleration) const;
  bool getNextVelocitySwitchingPoint(double path_pos,
                                     TrajectoryStep& next_switching_point,
                                     double& before_acceleration,
                                     double& after_acceleration) const;
  bool integrateForward(std::list<TrajectoryStep>& trajectory,
                        double acceleration);
  void integrateBackward(std::list<TrajectoryStep>& start_trajectory,
                         double path_pos, double path_vel, double acceleration);
  double getMinMaxPathAcceleration(double path_position, double path_velocity,
                                   bool max) const;
  double getMinMaxPhaseSlope(double path_position, double path_velocity,
                             bool max) const;
  double getAccelerationMaxPathVelocity(double path_pos) const;
  double getVelocityMaxPathVelocity(double path_pos) const;
  double getAccelerationMaxPathVelocityDeriv(double path_pos) const;
  double getVelocityMaxPathVelocityDeriv(double path_pos) const;

  std::list<TrajectoryStep>::const_iterator getTrajectorySegment(
      double time) const;

  Path path_;
  Eigen::VectorXd max_velocity_;
  Eigen::VectorXd max_acceleration_;
  std::size_t joint_num_ = 0;
  bool valid_ = true;
  std::list<TrajectoryStep> trajectory_;
  std::list<TrajectoryStep> end_trajectory_;  // populated only on failure
  double time_step_ = 0.0;

  mutable double cached_time_ = std::numeric_limits<double>::max();
  mutable std::list<TrajectoryStep>::const_iterator cached_trajectory_segment_;
};

}  // namespace autolife::trajectory
