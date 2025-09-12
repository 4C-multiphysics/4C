// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::vector<Core::IO::InputSpec> Constraints::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> spec;

  /*----------------------------------------------------------------------*/
  /* parameters for constraint model */

  spec.push_back(group("CONSTRAINT",
      {parameter<EnforcementStrategy>(
           "CONSTRAINT_ENFORCEMENT", {.description = "Type of constraint enforcement"}),

          parameter<double>("PENALTY_PARAM",
              {.description = "Value of the penalty parameter", .default_value = 1e5})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for embedded mesh constraint submodel */

  spec.push_back(group("CONSTRAINT/EMBEDDED MESH COUPLING",
      {
          parameter<EmbeddedMesh::CouplingStrategy>("COUPLING_STRATEGY",
              {.description = "Strategy to couple background and overlapping mesh"}),

          parameter<EmbeddedMesh::SolidToSolidMortarShapefunctions>("MORTAR_SHAPE_FUNCTION",
              {.description = "Shape functions that should be use in case of coupling using the "
                              "Mortar/Lagrange  Multiplier method "}),
      },
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for multi point constraint submodel */

  spec.push_back(group("CONSTRAINT/MULTI POINT",
      {parameter<MultiPoint::RveReferenceDeformationDefinition>("RVE_REFERENCE_POINTS",
          {.description = "Method of definition of the reference points of an RVE",
              .default_value = MultiPoint::RveReferenceDeformationDefinition::automatic})},
      {.required = false}));

  return spec;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Constraints::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*----------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition rve_lineperiodic_condition(
      "DESIGN LINE PERIODIC RVE 2D BOUNDARY CONDITIONS", "LinePeriodicRve",
      "definition of edges forming 2D periodic boundary conditions",
      Core::Conditions::LineRvePeriodic, false, Core::Conditions::geometry_type_line);

  rve_lineperiodic_condition.add_component(
      deprecated_selection<std::string>("EDGE", {"x+", "x-", "y+", "y-", "undefined"},
          {.description = "edge line id", .default_value = "undefined"}));

  condlist.push_back(rve_lineperiodic_condition);

  /*----------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition rve_surfperiodic_condition(
      "DESIGN SURF PERIODIC RVE 3D BOUNDARY CONDITIONS", "SurfacePeriodicRve",
      "definition of surfaces forming 3D periodic boundary conditions",
      Core::Conditions::SurfaceRvePeriodic, false, Core::Conditions::geometry_type_surface);

  rve_surfperiodic_condition.add_component(
      deprecated_selection<std::string>("SURF", {"x+", "x-", "y+", "y-", "z+", "z-", "undefined"},
          {.description = "surface id", .default_value = "undefined"}));

  condlist.push_back(rve_surfperiodic_condition);

  /*----------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition rve_cornerpoint_condition(
      "DESIGN POINT PERIODIC RVE 2D BOUNDARY REFERENCE CONDITIONS", "PointPeriodicRveReferenceNode",
      "definition of reference points defining the reference vector of the periodic boundary"
      "condition -  only required if RVE_REFERENCE_POINTS = automatic",
      Core::Conditions::PointRvePeriodicReference, false, Core::Conditions::geometry_type_point);

  rve_cornerpoint_condition.add_component(deprecated_selection<std::string>("POSITION",
      {"N1L", "N1B", "N2", "N4", "N1", "N3", "undefined"},
      {.description = "position of reference node", .default_value = "undefined"}));

  condlist.push_back(rve_cornerpoint_condition);

  /*----------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition linear_ce("DESIGN POINT COUPLED DOF EQUATION CONDITIONS",
      "PointLinearCoupledEquation",
      "definition of the term of a linear couple equation coupling different degrees of "
      "freedom in "
      "2d",
      Core::Conditions::PointLinearCoupledEquation, false, Core::Conditions::geometry_type_point);

  linear_ce.add_component(parameter<int>("EQUATION", {.description = "EQUATION"}));
  linear_ce.add_component(deprecated_selection<std::string>("ADD", {"dispx", "dispy", "undefined"},
      {.description = "degrees of freedom", .default_value = "undefined"}));
  linear_ce.add_component(parameter<double>("COEFFICIENT"));

  condlist.push_back(linear_ce);
}


FOUR_C_NAMESPACE_CLOSE