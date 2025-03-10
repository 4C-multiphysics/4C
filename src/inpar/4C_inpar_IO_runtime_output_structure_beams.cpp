// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_IO_runtime_output_structure_beams.hpp"

#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


namespace Inpar
{
  namespace IORuntimeOutput
  {
    namespace Beam
    {
      /*----------------------------------------------------------------------*
       *----------------------------------------------------------------------*/
      void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
      {
        using Teuchos::tuple;
        using namespace Core::IO::InputSpecBuilders;

        // related sublist
        Core::Utils::SectionSpecs sublist_IO{"IO"};
        Core::Utils::SectionSpecs sublist_IO_VTK_structure{sublist_IO, "RUNTIME VTK OUTPUT"};
        Core::Utils::SectionSpecs sublist_IO_output_beams{sublist_IO_VTK_structure, "BEAMS"};

        // whether to write special output for beam elements
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("OUTPUT_BEAMS",
            {.description = "write special output for beam elements", .default_value = false}));

        // whether to write displacement state
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>(
            "DISPLACEMENT", {.description = "write displacement output", .default_value = false}));

        // use absolute positions or initial positions for vtu geometry (i.e. point coordinates)
        // 'absolute positions' requires writing geometry in every output step (default for now)
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>(
            "USE_ABSOLUTE_POSITIONS", {.description = "use absolute positions or initial positions "
                                                      "for vtu geometry (i.e. point coordinates)",
                                          .default_value = true}));

        // write internal (elastic) energy of element
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("INTERNAL_ENERGY_ELEMENT",
            {.description = "write internal (elastic) energy for each element",
                .default_value = false}));

        // write kinetic energy of element
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("KINETIC_ENERGY_ELEMENT",
            {.description = "write kinetic energy for each element", .default_value = false}));

        // write triads as three orthonormal base vectors at every visualization point
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("TRIAD_VISUALIZATIONPOINT",
            {.description = "write triads at every visualization point", .default_value = false}));

        // write material cross-section strains at the Gauss points:
        // axial & shear strains, twist & curvatures
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("STRAINS_GAUSSPOINT",
            {.description = "write material cross-section strains at the Gauss points",
                .default_value = false}));

        // write material cross-section strains at the visualization points:
        // axial & shear strains, twist & curvatures
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("STRAINS_CONTINUOUS",
            {.description = "write material cross-section strains at the visualization points",
                .default_value = false}));

        // write material cross-section stresses at the Gauss points:
        // axial and shear forces, torque and bending moments
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("MATERIAL_FORCES_GAUSSPOINT",
            {.description = "write material cross-section stresses at the Gauss points",
                .default_value = false}));

        // write material cross-section stresses at the visualization points:
        // axial and shear forces, torque and bending moments
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("MATERIAL_FORCES_CONTINUOUS",
            {.description = "write material cross-section stresses at the visualization points",
                .default_value = false}));

        // write spatial cross-section stresses at the Gauss points:
        // axial and shear forces, torque and bending moments
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("SPATIAL_FORCES_GAUSSPOINT",
            {.description = "write material cross-section stresses at the Gauss points",
                .default_value = false}));

        // write element filament numbers and type
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("BEAMFILAMENTCONDITION",
            {.description = "write element filament numbers", .default_value = false}));

        // write element and network orientation parameter
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("ORIENTATION_PARAMETER",
            {.description = "write element filament numbers", .default_value = false}));

        // write crossection forces of periodic RVE
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("RVE_CROSSSECTION_FORCES",
            {.description = " get sum of all internal forces of  ", .default_value = false}));

        // write reference length of beams
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("REF_LENGTH",
            {.description = "write reference length of all beams", .default_value = false}));

        // write element GIDs
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("ELEMENT_GID",
            {.description = "write the 4C internal element GIDs", .default_value = false}));

        // write element ghosting information
        sublist_IO_output_beams.specs.emplace_back(parameter<bool>("ELEMENT_GHOSTING",
            {.description = "write which processors ghost the elements", .default_value = false}));

        // number of subsegments along a single beam element for visualization
        sublist_IO_output_beams.specs.emplace_back(parameter<int>("NUMBER_SUBSEGMENTS",
            {.description = "Number of subsegments along a single beam element for visualization",
                .default_value = 5}));

        sublist_IO_output_beams.move_into_collection(list);
      }
    }  // namespace Beam
  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE
