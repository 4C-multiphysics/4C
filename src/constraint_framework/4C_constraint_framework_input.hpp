// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_INPUT_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  class ConditionDefinition;
}

namespace Constraints
{
  /// type of the submodel for constraintmodels
  enum class SubModelType
  {
    undefined,     ///< default
    pbc_rve,       ///< apply periodic displacement bcs on
    embeddedmesh,  ///< apply embedded mesh bcs
    nullspace      ///< apply nullspace constraint Bu=0 with nullspace B and solution vector u
  };

  /// type of employed constraint enforcement strategy
  enum class EnforcementStrategy
  {
    penalty,  ///< Penalty based constraint enforcement
    lagrange  ///< Lagrange Multiplier based constraint enforcement
  };

  namespace EmbeddedMesh
  {
    enum class CouplingStrategy
    {
      mortar,  ///< Mortar based coupling
      nitsche  ///< Nitsche based coupling
    };

    /**
     * \brief Shape function for the mortar Lagrange-multiplicators for solid to solid embedded
     * coupling
     */
    enum class SolidToSolidMortarShapefunctions
    {

      quad4,  ///< Linear Lagrange elements
      quad9,  ///< Quadratic Lagrange elements
      nurbs9  ///< Quadratic NURBS elements
    };
  }  // namespace EmbeddedMesh


  namespace MultiPoint
  {
    /// Definition Type of the MultiPoint Constraint
    enum ConstraintType
    {
      coupled_equation,  ///< Manually enter the dofs as coupled equation
      periodic_rve       ///< Automatically apply the MPCs that describe periodicity
    };

    /// Definition Type of Coupled Equation
    /// (this enum represents the input file parameter COUPLED_DOF_EQUATIONS)
    enum CeType
    {
      ce_none,
      ce_linear
    };

    /// Methods used to determine the reference points for an periodic RVE
    /// (this enum represents the input file parameter RVE_REFERENCE_POINTS)
    enum RveReferenceDeformationDefinition
    {
      automatic,  ///< Automatically use the corner nodes for reference
      manual      ///< provide the reference nodes as condition

    };

    enum RveDimension
    {
      rve2d,
      rve3d
    };

    enum RveEdgeIdentifiers
    {
      Gamma_xm,
      Gamma_ym,
    };
  }  // namespace MultiPoint

  //! \brief Set constraint parameters
  std::vector<Core::IO::InputSpec> valid_parameters();

  //! \brief Set constraint specific conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace Constraints

FOUR_C_NAMESPACE_CLOSE

#endif
