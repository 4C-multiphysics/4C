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

        // related sublist
        Core::Utils::SectionSpecs sublist_IO{"IO"};
        Core::Utils::SectionSpecs sublist_IO_VTK_structure{sublist_IO, "RUNTIME VTK OUTPUT"};
        Core::Utils::SectionSpecs sublist_IO_output_beams{sublist_IO_VTK_structure, "BEAMS"};

        // whether to write special output for beam elements
        Core::Utils::bool_parameter("OUTPUT_BEAMS", "No", "write special output for beam elements",
            sublist_IO_output_beams);

        // whether to write displacement state
        Core::Utils::bool_parameter(
            "DISPLACEMENT", "No", "write displacement output", sublist_IO_output_beams);

        // use absolute positions or initial positions for vtu geometry (i.e. point coordinates)
        // 'absolute positions' requires writing geometry in every output step (default for now)
        Core::Utils::bool_parameter("USE_ABSOLUTE_POSITIONS", "Yes",
            "use absolute positions or initial positions for vtu geometry (i.e. point coordinates)",
            sublist_IO_output_beams);

        // write internal (elastic) energy of element
        Core::Utils::bool_parameter("INTERNAL_ENERGY_ELEMENT", "No",
            "write internal (elastic) energy for each element", sublist_IO_output_beams);

        // write kinetic energy of element
        Core::Utils::bool_parameter("KINETIC_ENERGY_ELEMENT", "No",
            "write kinetic energy for each element", sublist_IO_output_beams);

        // write triads as three orthonormal base vectors at every visualization point
        Core::Utils::bool_parameter("TRIAD_VISUALIZATIONPOINT", "No",
            "write triads at every visualization point", sublist_IO_output_beams);

        // write material cross-section strains at the Gauss points:
        // axial & shear strains, twist & curvatures
        Core::Utils::bool_parameter("STRAINS_GAUSSPOINT", "No",
            "write material cross-section strains at the Gauss points", sublist_IO_output_beams);

        // write material cross-section strains at the visualization points:
        // axial & shear strains, twist & curvatures
        Core::Utils::bool_parameter("STRAINS_CONTINUOUS", "No",
            "write material cross-section strains at the visualization points",
            sublist_IO_output_beams);

        // write material cross-section stresses at the Gauss points:
        // axial and shear forces, torque and bending moments
        Core::Utils::bool_parameter("MATERIAL_FORCES_GAUSSPOINT", "No",
            "write material cross-section stresses at the Gauss points", sublist_IO_output_beams);

        // write material cross-section stresses at the visualization points:
        // axial and shear forces, torque and bending moments
        Core::Utils::bool_parameter("MATERIAL_FORCES_CONTINUOUS", "No",
            "write material cross-section stresses at the visualization points",
            sublist_IO_output_beams);

        // write spatial cross-section stresses at the Gauss points:
        // axial and shear forces, torque and bending moments
        Core::Utils::bool_parameter("SPATIAL_FORCES_GAUSSPOINT", "No",
            "write material cross-section stresses at the Gauss points", sublist_IO_output_beams);

        // write element filament numbers and type
        Core::Utils::bool_parameter("BEAMFILAMENTCONDITION", "No", "write element filament numbers",
            sublist_IO_output_beams);

        // write element and network orientation parameter
        Core::Utils::bool_parameter("ORIENTATION_PARAMETER", "No", "write element filament numbers",
            sublist_IO_output_beams);

        // write crossection forces of periodic RVE
        Core::Utils::bool_parameter("RVE_CROSSSECTION_FORCES", "No",
            " get sum of all internal forces of  ", sublist_IO_output_beams);

        // write reference length of beams
        Core::Utils::bool_parameter(
            "REF_LENGTH", "No", "write reference length of all beams", sublist_IO_output_beams);

        // write element GIDs
        Core::Utils::bool_parameter(
            "ELEMENT_GID", "No", "write the 4C internal element GIDs", sublist_IO_output_beams);

        // write element ghosting information
        Core::Utils::bool_parameter("ELEMENT_GHOSTING", "No",
            "write which processors ghost the elements", sublist_IO_output_beams);

        // number of subsegments along a single beam element for visualization
        Core::Utils::int_parameter("NUMBER_SUBSEGMENTS", 5,
            "Number of subsegments along a single beam element for visualization",
            sublist_IO_output_beams);

        sublist_IO_output_beams.move_into_collection(list);
      }
    }  // namespace Beam
  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE
