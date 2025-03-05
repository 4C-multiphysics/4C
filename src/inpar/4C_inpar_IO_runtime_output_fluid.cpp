// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_IO_runtime_output_fluid.hpp"

#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace FLUID
    {
      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
      {
        using Teuchos::tuple;
        using namespace Core::IO::InputSpecBuilders;

        // related sublist
        Core::Utils::SectionSpecs sublist_IO{"IO"};
        Core::Utils::SectionSpecs sublist_IO_output{sublist_IO, "RUNTIME VTK OUTPUT"};
        Core::Utils::SectionSpecs sublist_IO_output_fluid{sublist_IO_output, "FLUID"};

        // whether to write output for fluid
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "OUTPUT_FLUID", {.description = "write fluid output", .default_value = false}));

        // whether to write velocity state
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "VELOCITY", {.description = "write velocity output", .default_value = false}));

        // whether to write pressure state
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "PRESSURE", {.description = "write pressure output", .default_value = false}));

        // whether to write acceleration state
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "ACCELERATION", {.description = "write acceleration output", .default_value = false}));

        // whether to write displacement state
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "DISPLACEMENT", {.description = "write displacement output", .default_value = false}));

        // whether to write displacement state
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "GRIDVELOCITY", {.description = "write grid velocity output", .default_value = false}));

        // whether to write element owner
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "ELEMENT_OWNER", {.description = "write element owner", .default_value = false}));

        // whether to write element GIDs
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>("ELEMENT_GID",
            {.description = "write 4C internal element GIDs", .default_value = false}));

        // whether to write node GIDs
        sublist_IO_output_fluid.specs.emplace_back(parameter<bool>(
            "NODE_GID", {.description = "write 4C internal node GIDs", .default_value = false}));

        sublist_IO_output_fluid.move_into_collection(list);
      }
    }  // namespace FLUID
  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE
