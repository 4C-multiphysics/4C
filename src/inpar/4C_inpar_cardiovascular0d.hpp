/*----------------------------------------------------------------------*/
/*! \file

\brief Input parameters for 0d cardiovascular-structure coupling

\level 2

*/

/*----------------------------------------------------------------------*/

#ifndef FOUR_C_INPAR_CARDIOVASCULAR0D_HPP
#define FOUR_C_INPAR_CARDIOVASCULAR0D_HPP

#include "4C_config.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Inpar
{
  namespace Cardiovascular0D
  {
    /// possible 0D cardiovascular-structural solvers
    enum Cardvasc0DSolveAlgo
    {
      cardvasc0dsolve_direct,  ///< build monolithic 0D cardiovascular-structural system
      cardvasc0dsolve_simple,  ///< use simple preconditioner for iterative solve
      cardvasc0dsolve_AMGnxn
    };

    enum Cardvasc0DAtriumModel
    {
      atr_prescribed,
      atr_elastance_0d,
      atr_structure_3d
    };

    enum Cardvasc0DVentricleModel
    {
      ventr_prescribed,
      ventr_elastance_0d,
      ventr_structure_3d
    };

    enum Cardvasc0DRespiratoryModel
    {
      resp_none,
      resp_standard
    };

    /// set the 0Dcardiovascular parameters
    void set_valid_parameters(Teuchos::RCP<Teuchos::ParameterList> list);

    /// set specific 0Dcardiovascular conditions
    void set_valid_conditions(
        std::vector<Teuchos::RCP<Core::Conditions::ConditionDefinition>>& condlist);

  }  // namespace Cardiovascular0D
}  // namespace Inpar
/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
