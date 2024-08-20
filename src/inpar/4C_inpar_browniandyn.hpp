/*-----------------------------------------------------------*/
/*! \file
\brief input parameter for Brownian dynamics simulation


\level 2

*/
/*-----------------------------------------------------------*/
#ifndef FOUR_C_INPAR_BROWNIANDYN_HPP
#define FOUR_C_INPAR_BROWNIANDYN_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace BrownianDynamics
  {
    /// the way how damping coefficient values for beams are specified
    enum BeamDampingCoefficientSpecificationType
    {
      cylinder_geometry_approx,
      input_file,
      vague
    };

    /// set the brownian dynamic parameters
    void set_valid_parameters(Teuchos::RCP<Teuchos::ParameterList> list);

  }  // namespace BrownianDynamics

}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
