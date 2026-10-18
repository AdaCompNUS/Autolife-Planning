/**
 * OMPL ↔ VAMP state validity and motion validators.
 *
 * Two pairs of classes:
 *
 *  - ``AutolifeValidityChecker`` / ``AutolifeMotionValidator``
 *      Full-body planner: the OMPL state already has 24 DOFs, so we
 *      copy it straight into a VAMP ``Robot::Configuration``.
 *
 *  - ``SubgroupValidityChecker`` / ``SubgroupMotionValidator``
 *      Subgroup planner: the OMPL state is the reduced active subset.
 *      We expand it to a 24-DOF body via ``active_indices`` +
 *      ``frozen_config`` before calling VAMP.
 *
 * In both cases the actual collision checking is delegated to VAMP's
 * SIMD ``validate_motion`` (resolution 1 for single states, full robot
 * resolution for motion edges).  For a failed edge,
 * ``checkMotion(s1, s2, last_valid)`` also reports the last valid state
 * along it, which planners such as PDST, KPIECE and STRIDE grow from.
 */

#pragma once

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/WrapperStateSpace.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vamp/collision/environment.hh>
#include <vamp/planning/validate.hh>
// Our autolife FK/collision header lives in-tree so the vamp submodule
// stays a pure header-only third-party dep. ``ext/ompl_vamp`` is on the
// include path, which makes this resolve to
// ``ext/ompl_vamp/robot/autolife.hh``.
#include <vector>

#include "robot/autolife.hh"

namespace autolife {

namespace ob = ompl::base;

using Robot = vamp::robots::Autolife;
inline constexpr std::size_t kRake = vamp::FloatVectorWidth;
using VampEnv = vamp::collision::Environment<vamp::FloatVector<kRake>>;
using FloatEnv = vamp::collision::Environment<float>;

// Unwrap a state for either a plain RealVectorStateSpace or a
// ConstrainedStateSpace (which inherits from WrapperStateSpace and
// holds the wrapped real-vector state internally).  Lets the same
// validity checker run in both planning modes.
inline auto extract_real_state(const ob::State* state)
    -> const ob::RealVectorStateSpace::StateType* {
  if (auto* wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType*>(state)) {
    return wrapper->getState()->as<ob::RealVectorStateSpace::StateType>();
  }
  return state->as<ob::RealVectorStateSpace::StateType>();
}

// For a motion that failed its check: walks the states VAMP's
// ``validate_motion`` samples on the straight line (``t = k / n``,
// ``k = 1..n``) in order and reports the last valid one, as OMPL's
// ``checkMotion(s1, s2, last_valid)`` contract asks: its fraction in
// ``last_valid.second`` and, when ``last_valid.first`` is given, the state.
inline void find_last_valid(const ob::SpaceInformation* si, const ob::State* s1,
                            const ob::State* s2,
                            std::pair<ob::State*, double>& last_valid) {
  const auto& space = si->getStateSpace();
  const double steps =
      std::ceil(space->distance(s1, s2) / kRake * Robot::resolution);
  const int n = static_cast<int>(kRake * std::max(steps, 1.0));
  auto* s = space->allocState();
  int k = 1;
  for (; k <= n; ++k) {
    space->interpolate(s1, s2, static_cast<double>(k) / n, s);
    if (!si->isValid(s)) break;
  }
  space->freeState(s);
  last_valid.second = static_cast<double>(k - 1) / n;
  if (last_valid.first)
    space->interpolate(s1, s2, last_valid.second, last_valid.first);
}

// ─── Full-body checkers ─────────────────────────────────────────────

class AutolifeValidityChecker : public ob::StateValidityChecker {
 public:
  AutolifeValidityChecker(const ob::SpaceInformationPtr& si, const VampEnv& env)
      : ob::StateValidityChecker(si), env_(env) {}

  auto isValid(const ob::State* state) const -> bool override {
    auto config = ompl_to_vamp(state);
    return vamp::planning::validate_motion<Robot, kRake, 1>(config, config,
                                                            env_);
  }

 private:
  const VampEnv& env_;

  static auto ompl_to_vamp(const ob::State* state) -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    const auto* rv = extract_real_state(state);
    for (std::size_t i = 0; i < Robot::dimension; ++i)
      buf[i] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

class AutolifeMotionValidator : public ob::MotionValidator {
 public:
  AutolifeMotionValidator(const ob::SpaceInformationPtr& si, const VampEnv& env)
      : ob::MotionValidator(si), env_(env) {}

  auto checkMotion(const ob::State* s1, const ob::State* s2) const
      -> bool override {
    return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
        ompl_to_vamp(s1), ompl_to_vamp(s2), env_);
  }

  auto checkMotion(const ob::State* s1, const ob::State* s2,
                   std::pair<ob::State*, double>& last_valid) const
      -> bool override {
    if (checkMotion(s1, s2)) return true;
    find_last_valid(si_, s1, s2, last_valid);
    return false;
  }

 private:
  const VampEnv& env_;

  static auto ompl_to_vamp(const ob::State* state) -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    const auto* rv = extract_real_state(state);
    for (std::size_t i = 0; i < Robot::dimension; ++i)
      buf[i] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

// ─── Subgroup checkers (reduced dim → full config → VAMP) ───────────

class SubgroupValidityChecker : public ob::StateValidityChecker {
 public:
  SubgroupValidityChecker(const ob::SpaceInformationPtr& si, const VampEnv& env,
                          std::vector<int> active_indices,
                          std::vector<float> frozen_config)
      : ob::StateValidityChecker(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)) {}

  auto isValid(const ob::State* state) const -> bool override {
    auto config = expand(state);
    return vamp::planning::validate_motion<Robot, kRake, 1>(config, config,
                                                            env_);
  }

 private:
  const VampEnv& env_;
  std::vector<int> active_;
  std::vector<float> frozen_;

  auto expand(const ob::State* state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());
    const auto* rv = extract_real_state(state);
    for (std::size_t i = 0; i < active_.size(); ++i)
      buf[active_[i]] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

class SubgroupMotionValidator : public ob::MotionValidator {
 public:
  SubgroupMotionValidator(const ob::SpaceInformationPtr& si, const VampEnv& env,
                          std::vector<int> active_indices,
                          std::vector<float> frozen_config)
      : ob::MotionValidator(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)) {}

  auto checkMotion(const ob::State* s1, const ob::State* s2) const
      -> bool override {
    return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
        expand(s1), expand(s2), env_);
  }

  auto checkMotion(const ob::State* s1, const ob::State* s2,
                   std::pair<ob::State*, double>& last_valid) const
      -> bool override {
    if (checkMotion(s1, s2)) return true;
    find_last_valid(si_, s1, s2, last_valid);
    return false;
  }

 private:
  const VampEnv& env_;
  std::vector<int> active_;
  std::vector<float> frozen_;

  auto expand(const ob::State* state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());
    const auto* rv = extract_real_state(state);
    for (std::size_t i = 0; i < active_.size(); ++i)
      buf[active_[i]] = static_cast<float>(rv->values[i]);
    return Robot::Configuration(buf.data());
  }
};

}  // namespace autolife
