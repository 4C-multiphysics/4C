/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for multi point constraints used for periodic boundary conditions
 for representative volume elements (RVEs)
\level 3
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_INPAR_MPC_RVE_HPP
#define FOUR_C_INPAR_MPC_RVE_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_integration.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  class ConditionDefinition;
}

namespace Inpar::RveMpc
{
  /// Definition Type of the MultiPoint Constraint
  enum MultiPointConstraintType
  {
    none,
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

  /// Strategy used to enforce the MPCs
  /// (this enum represents the input file parameter ENFORCEMENT)
  enum EnforcementStrategy
  {
    penalty,            ///< Enforce the Multi-Point Constraint with the penalty method
    lagrangeMultiplier  ///< Enforce the Multi-Point Constraint with the Lagrange Multiplier
                        ///< Method
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
  /// set the multi point constraint parameters
  void set_valid_parameters(Teuchos::ParameterList& list);

  /// set multi point constraint specific conditions
  void set_valid_conditions(
      std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist);
}  // namespace Inpar::RveMpc

FOUR_C_NAMESPACE_CLOSE
#endif
