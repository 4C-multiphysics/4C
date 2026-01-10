// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_monitor_dbc_input.hpp"

#include "4C_io_input_spec_builders.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Solid::IOMonitorStructureDBC
{
  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  Core::IO::InputSpec valid_parameters()
  {
    using namespace Core::IO::InputSpecBuilders;

    // related sublist
    Core::IO::InputSpec spec = group("IO/MONITOR STRUCTURE DBC",
        {

            // output interval regarding steps: write output every INTERVAL_STEPS steps
            parameter<int>("INTERVAL_STEPS",
                {.description = "write reaction force output every INTERVAL_STEPS steps",
                    .default_value = -1}),

            // precision for screen
            parameter<int>("PRECISION_SCREEN",
                {.description = "precision for written screen output", .default_value = 5}),

            // type of written output file
            parameter<FileType>("FILE_TYPE",
                {.description = "type of written output file", .default_value = FileType::csv}),

            // whether to write information about monitored boundary condition to an output file
            parameter<bool>("WRITE_CONDITION_INFORMATION",
                {.description = "write information about monitored boundary condition to output "
                                "file. Only possible for yaml output.",
                    .default_value = false})},
        {.required = false});
    return spec;
  }
}  // namespace Solid::IOMonitorStructureDBC

FOUR_C_NAMESPACE_CLOSE