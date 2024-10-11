/*----------------------------------------------------------------------*/
/*! \file
\file inpar_io.H

\brief Input parameters for global IO section


\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_INPAR_IO_HPP
#define FOUR_C_INPAR_IO_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace IO
  {
    /*! \brief Define valid parameter for global IO control
     *
     * @param[in/out] list Parameter list to be filled with valid parameters and their defaults
     */
    void set_valid_parameters(Teuchos::ParameterList& list);

  }  // namespace IO
}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
