// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MONITOR_DBC_INPUT_HPP
#define FOUR_C_STRUCTURE_NEW_MONITOR_DBC_INPUT_HPP


/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_utils_exceptions.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN


namespace Solid::IOMonitorStructureDBC
{
  /// set the valid parameters related to writing of output at runtime
  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

}  // namespace Solid::IOMonitorStructureDBC

FOUR_C_NAMESPACE_CLOSE

#endif
