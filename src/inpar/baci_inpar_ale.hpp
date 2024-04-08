/*----------------------------------------------------------------------*/
/*! \file

\brief Input parameters for ALE mesh motion

\level 2

*/

/*----------------------------------------------------------------------*/
#ifndef FOUR_C_INPAR_ALE_HPP
#define FOUR_C_INPAR_ALE_HPP

#include "baci_config.hpp"

#include "baci_utils_parameter_list.hpp"

BACI_NAMESPACE_OPEN

// forward declaration

namespace INPUT
{
  class ConditionDefinition;
}

/*----------------------------------------------------------------------*/
namespace INPAR
{
  namespace ALE
  {
    /// possible types of moving boundary simulation
    enum AleDynamic
    {
      none,              ///< default
      laplace_material,  ///< Laplacian smoothing based on material configuration
      laplace_spatial,   ///< Laplacian smoothing based on spatial configuration
      springs_material,  ///< use a spring analogy based on material configuration
      springs_spatial,   ///< use a spring analogy based on spatial configuration
      solid,             ///< nonlinear pseudo-structure approach
      solid_linear       ///< linear pseudo-structure approach
    };
    /// Handling of non-converged nonlinear solver
    enum DivContAct
    {
      divcont_stop,     ///< abort simulation
      divcont_continue  ///< continue nevertheless
    };

    /// mesh tying and mesh sliding algorithm
    enum MeshTying
    {
      no_meshtying,
      meshtying,
      meshsliding
    };

    /// initial displacement field
    enum InitialDisp
    {
      initdisp_zero_disp,        ///< zero initial displacement
      initdisp_disp_by_function  ///< initial displacement from function
    };

    /// Defines all valid parameters for ale problem
    void SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list);

    /// Defines ale specific conditions
    void SetValidConditions(std::vector<Teuchos::RCP<INPUT::ConditionDefinition>>& condlist);

  }  // namespace ALE

}  // namespace INPAR

/*----------------------------------------------------------------------*/
BACI_NAMESPACE_CLOSE

#endif  // INPAR_ALE_H
