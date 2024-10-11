/*----------------------------------------------------------------------*/
/*! \file

\brief input parameters for VTK output of structural problem at runtime

\level 2

*/
/*----------------------------------------------------------------------*/
/* definitions */
#ifndef FOUR_C_INPAR_IO_RUNTIME_VTK_OUTPUT_STRUCTURE_HPP
#define FOUR_C_INPAR_IO_RUNTIME_VTK_OUTPUT_STRUCTURE_HPP


/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace Solid
    {
      /// set the valid parameters related to writing of VTK output at runtime
      void set_valid_parameters(Teuchos::ParameterList& list);

    }  // namespace Solid
  }    // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE

#endif
