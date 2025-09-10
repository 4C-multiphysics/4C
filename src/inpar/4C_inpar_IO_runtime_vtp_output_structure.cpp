// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_IO_runtime_vtp_output_structure.hpp"

#include "4C_io_input_spec_builders.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace IORuntimeVTPStructure
  {
    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    Core::IO::InputSpec valid_parameters()
    {
      using namespace Core::IO::InputSpecBuilders;

      // related sublist
      Core::IO::InputSpec spec = group("IO/RUNTIME VTP OUTPUT STRUCTURE",
          {

              // output interval regarding steps: write output every INTERVAL_STEPS steps
              parameter<int>("INTERVAL_STEPS",
                  {.description = "write VTP output at runtime every INTERVAL_STEPS steps",
                      .default_value = -1}),
              parameter<int>("STEP_OFFSET",
                  {.description =
                          "An offset added to the current step to shift the steps to be written.",
                      .default_value = 0}),

              // whether to write output in every iteration of the nonlinear solver
              parameter<bool>("EVERY_ITERATION",
                  {.description = "write output in every iteration of the nonlinear solver",
                      .default_value = false}),

              // write owner at every visualization point
              parameter<bool>(
                  "OWNER", {.description = "write owner of every point", .default_value = false}),

              // write orientation at every visualization point
              parameter<bool>("ORIENTATIONANDLENGTH",
                  {.description = "write orientation at every point", .default_value = false}),

              // write number of bonds at every visualization point
              parameter<bool>("NUMBEROFBONDS",
                  {.description = "write number of bonds of every point", .default_value = false}),

              // write force actin in linker
              parameter<bool>("LINKINGFORCE",
                  {.description = "write force acting in linker", .default_value = false})},
          {.required = false});
      return spec;
    }


  }  // namespace IORuntimeVTPStructure
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE