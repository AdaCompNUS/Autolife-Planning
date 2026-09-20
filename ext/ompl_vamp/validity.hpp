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
 * resolution for motion edges).
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
    last_valid.first = nullptr;
    last_valid.second = 0.0;
    return checkMotion(s1, s2);
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
    last_valid.first = nullptr;
    last_valid.second = 0.0;
    return checkMotion(s1, s2);
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
