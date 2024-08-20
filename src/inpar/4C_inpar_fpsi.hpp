/*----------------------------------------------------------------------*/
/*! \file
\brief fpsi parameters
\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_INPAR_FPSI_HPP
#define FOUR_C_INPAR_FPSI_HPP

#include "4C_config.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration

/*----------------------------------------------------------------------*
 | Coupling Methods                                                                  |
 *----------------------------------------------------------------------*/
enum FpsiCoupling
{
  fpsi_monolithic_plain,
  partitioned
};

namespace Inpar
{
  namespace FPSI
  {
    // type of partitioned coupling for FPSI problems
    enum PartitionedCouplingMethod
    {
      RobinNeumann,
      monolithic,
      nocoupling
    };

    // type of norm to check for convergence
    enum ConvergenceNorm
    {
      absoluteconvergencenorm,            // compare absolute value with single tolerance
      absoluteconvergencenorm_sys_split,  // compare absolute value with correction of systemsize
                                          // with different tolerances for each field
      relativconvergencenorm_sys  // compare relative value with correction of systemsize with
                                  // single tolerance
    };

    // type of norm to check for convergence
    enum BinaryOp
    {
      bop_and,
      bop_or
    };

    enum FluidFieldHierachy
    {
      fluid,
      porofluid
    };

    /// set the fpsi parameters
    void set_valid_parameters(Teuchos::RCP<Teuchos::ParameterList> list);

    /// set specific fpsi conditions
    void set_valid_conditions(
        std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist);

  }  // namespace FPSI

}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
