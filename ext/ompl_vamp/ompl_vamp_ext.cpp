/**
 * OMPL + VAMP Python extension — nanobind bindings.
 *
 * The actual planner, validity checkers, constraint primitives, and
 * pinocchio robot loader live in self-contained internal headers
 * under this directory.  This file is intentionally kept thin: it
 * only imports those headers and exposes the C++ API to Python via
 * nanobind.  If you find yourself adding more than a few lines of
 * non-binding code here, that's a sign it belongs in one of the
 * internal headers instead.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "planner.hpp"

namespace nb = nanobind;
using autolife::OmplVampPlanner;
using autolife::PlanResult;

NB_MODULE(_ompl_vamp, m) {
  m.doc() = "OMPL + VAMP C++ planning extension for Autolife robot";

  nb::class_<PlanResult>(m, "PlanResult")
      .def_ro("solved", &PlanResult::solved)
      .def_ro("path", &PlanResult::path)
      .def_ro("planning_time_ns", &PlanResult::planning_time_ns)
      .def_ro("path_cost", &PlanResult::path_cost);

  nb::class_<OmplVampPlanner>(m, "OmplVampPlanner")
      .def(nb::init<>(), "Create a full-body planner (24 DOF).")
      .def(nb::init<std::vector<int>, std::vector<double>>(),
           "Create a subgroup planner.", nb::arg("active_indices"),
           nb::arg("frozen_config"))
      .def("add_pointcloud", &OmplVampPlanner::add_pointcloud,
           nb::arg("points"), nb::arg("r_min"), nb::arg("r_max"),
           nb::arg("point_radius"))
      .def("remove_pointcloud", &OmplVampPlanner::remove_pointcloud)
      .def("has_pointcloud", &OmplVampPlanner::has_pointcloud)
      .def("add_sphere", &OmplVampPlanner::add_sphere, nb::arg("center"),
           nb::arg("radius"))
      .def("clear_environment", &OmplVampPlanner::clear_environment)
      .def("add_compiled_constraint", &OmplVampPlanner::add_compiled_constraint,
           nb::arg("so_path"), nb::arg("symbol_name"), nb::arg("ambient_dim"),
           nb::arg("co_dim"))
      .def("clear_constraints", &OmplVampPlanner::clear_constraints)
      .def("num_constraints", &OmplVampPlanner::num_constraints)
      .def("add_compiled_cost", &OmplVampPlanner::add_compiled_cost,
           nb::arg("so_path"), nb::arg("symbol_name"), nb::arg("ambient_dim"),
           nb::arg("weight") = 1.0)
      .def("clear_costs", &OmplVampPlanner::clear_costs)
      .def("num_costs", &OmplVampPlanner::num_costs)
      .def("plan", &OmplVampPlanner::plan, nb::arg("start"), nb::arg("goal"),
           nb::arg("planner_name") = "rrtc", nb::arg("time_limit") = 10.0,
           nb::arg("simplify") = true, nb::arg("interpolate") = true,
           nb::arg("interpolate_count") = 0, nb::arg("resolution") = 64.0)
      .def("simplify_path", &OmplVampPlanner::simplify_path, nb::arg("path"),
           nb::arg("time_limit") = 1.0)
      .def("interpolate_path", &OmplVampPlanner::interpolate_path,
           nb::arg("path"), nb::arg("count") = 0, nb::arg("resolution") = 64.0)
      .def("validate", &OmplVampPlanner::validate, nb::arg("config"))
      .def("validate_batch", &OmplVampPlanner::validate_batch,
           nb::arg("configs"))
      .def("dimension", &OmplVampPlanner::dimension)
      .def("lower_bounds", &OmplVampPlanner::lower_bounds)
      .def("upper_bounds", &OmplVampPlanner::upper_bounds)
      .def("min_max_radii", &OmplVampPlanner::min_max_radii)
      .def("filter_pointcloud", &OmplVampPlanner::filter_pointcloud,
           nb::arg("points"), nb::arg("min_dist"), nb::arg("max_range"),
           nb::arg("origin"), nb::arg("workspace_min"),
           nb::arg("workspace_max"), nb::arg("cull") = true)
      .def("filter_self_from_pointcloud",
           &OmplVampPlanner::filter_self_from_pointcloud, nb::arg("points"),
           nb::arg("point_radius"), nb::arg("config"))
      .def("set_subgroup", &OmplVampPlanner::set_subgroup,
           nb::arg("active_indices"), nb::arg("frozen_config"))
      .def("set_full_body", &OmplVampPlanner::set_full_body);
}
