// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_IO_RUNTIME_OUTPUT_STRUCTURE_BEAMS_HPP
#define FOUR_C_INPAR_IO_RUNTIME_OUTPUT_STRUCTURE_BEAMS_HPP


/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace Beam
    {
      /// set the valid parameters related to writing of output at runtime
      void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

    }  // namespace Beam
  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE

#endif
