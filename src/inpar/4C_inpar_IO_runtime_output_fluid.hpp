/*----------------------------------------------------------------------*/
/*! \file

\brief input parameters for output of a fluid field at runtime

\level 2

*/
/*----------------------------------------------------------------------*/
/* definitions */
#ifndef FOUR_C_INPAR_IO_RUNTIME_OUTPUT_FLUID_HPP
#define FOUR_C_INPAR_IO_RUNTIME_OUTPUT_FLUID_HPP


/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace FLUID
    {
      /// set the valid parameters related to writing of output at runtime
      void set_valid_parameters(Teuchos::RCP<Teuchos::ParameterList> list);

    }  // namespace FLUID
  }    // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE

#endif
