// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_IO_runtime_vtk_output_structure.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace Solid
    {
      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
      {
        using Teuchos::tuple;
        using namespace Core::IO::InputSpecBuilders;

        // related sublist
        Core::Utils::SectionSpecs sublist_IO{"IO"};
        Core::Utils::SectionSpecs sublist_IO_VTK{sublist_IO, "RUNTIME VTK OUTPUT"};
        Core::Utils::SectionSpecs sublist_IO_VTK_structure{sublist_IO_VTK, "STRUCTURE"};

        // whether to write output for structure
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>(
            "OUTPUT_STRUCTURE", {.description = "write structure output", .default_value = false}));

        // whether to write displacement state
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>(
            "DISPLACEMENT", {.description = "write displacement output", .default_value = false}));

        // whether to write velocity state
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>(
            "VELOCITY", {.description = "write velocity output", .default_value = false}));

        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>(
            "ACCELERATION", {.description = "write acceleration output", .default_value = false}));

        // whether to write element owner
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>(
            "ELEMENT_OWNER", {.description = "write element owner", .default_value = false}));

        // whether to write element GIDs
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>("ELEMENT_GID",
            {.description = "write 4C internal element GIDs", .default_value = false}));

        // write element ghosting information
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>("ELEMENT_GHOSTING",
            {.description = "write which processors ghost the elements", .default_value = false}));

        // whether to write node GIDs
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>(
            "NODE_GID", {.description = "write 4C internal node GIDs", .default_value = false}));

        // write element material IDs
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>("ELEMENT_MAT_ID",
            {.description = "Output of the material id of each element", .default_value = false}));

        // whether to write stress and / or strain data
        sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>("STRESS_STRAIN",
            {.description = "Write element stress and / or strain  data. The type of stress / "
                            "strain has to be selected in the --IO input section",
                .default_value = false}));

        // mode to write gauss point data
        Core::Utils::string_to_integral_parameter<Inpar::Solid::GaussPointDataOutputType>(
            "GAUSS_POINT_DATA_OUTPUT_TYPE", "none",
            "Where to write gauss point data. (none, projected to nodes, projected to element "
            "center, raw at gauss points)",
            tuple<std::string>("none", "nodes", "element_center", "gauss_points"),
            tuple<Inpar::Solid::GaussPointDataOutputType>(
                Inpar::Solid::GaussPointDataOutputType::none,
                Inpar::Solid::GaussPointDataOutputType::nodes,
                Inpar::Solid::GaussPointDataOutputType::element_center,
                Inpar::Solid::GaussPointDataOutputType::gauss_points),
            sublist_IO_VTK_structure);

        sublist_IO_VTK_structure.move_into_collection(list);
      }


    }  // namespace Solid
  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE
