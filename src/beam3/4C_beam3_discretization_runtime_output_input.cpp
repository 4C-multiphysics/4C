// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_discretization_runtime_output_input.hpp"

#include "4C_io_input_spec_builders.hpp"


FOUR_C_NAMESPACE_OPEN


namespace IORuntimeOutput
{
  namespace Beam
  {
    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
    {
      using namespace Core::IO::InputSpecBuilders;

      // related sublist
      list["IO/RUNTIME VTK OUTPUT/BEAMS"] = group("IO/RUNTIME VTK OUTPUT/BEAMS",
          {

              // whether to write special output for beam elements
              parameter<bool>(
                  "OUTPUT_BEAMS", {.description = "write special output for beam elements",
                                      .default_value = false}),

              // whether to write displacement state
              parameter<bool>("DISPLACEMENT",
                  {.description = "write displacement output", .default_value = false}),

              // use absolute positions or initial positions for vtu geometry (i.e. point
              // coordinates)
              // 'absolute positions' requires writing geometry in every output step (default for
              // now)
              parameter<bool>("USE_ABSOLUTE_POSITIONS",
                  {.description = "use absolute positions or initial positions "
                                  "for vtu geometry (i.e. point coordinates)",
                      .default_value = true}),

              // write internal (elastic) energy of element
              parameter<bool>("INTERNAL_ENERGY_ELEMENT",
                  {.description = "write internal (elastic) energy for each element",
                      .default_value = false}),

              // write kinetic energy of element
              parameter<bool>("KINETIC_ENERGY_ELEMENT",
                  {.description = "write kinetic energy for each element", .default_value = false}),

              // write triads as three orthonormal base vectors at every visualization point
              parameter<bool>("TRIAD_VISUALIZATIONPOINT",
                  {.description = "write triads at every visualization point",
                      .default_value = false}),

              // write material cross-section strains at the Gauss points:
              // axial & shear strains, twist & curvatures
              parameter<bool>("STRAINS_GAUSSPOINT",
                  {.description = "write material cross-section strains at the Gauss points",
                      .default_value = false}),

              // write material cross-section strains at the visualization points:
              // axial & shear strains, twist & curvatures
              parameter<bool>("STRAINS_CONTINUOUS",
                  {.description =
                          "write material cross-section strains at the visualization points",
                      .default_value = false}),

              // write material cross-section stresses at the Gauss points:
              // axial and shear forces, torque and bending moments
              parameter<bool>("MATERIAL_FORCES_GAUSSPOINT",
                  {.description = "write material cross-section stresses at the Gauss points",
                      .default_value = false}),

              // write material cross-section stresses at the visualization points:
              // axial and shear forces, torque and bending moments
              parameter<bool>("MATERIAL_FORCES_CONTINUOUS",
                  {.description =
                          "write material cross-section stresses at the visualization points",
                      .default_value = false}),

              // write spatial cross-section stresses at the Gauss points:
              // axial and shear forces, torque and bending moments
              parameter<bool>("SPATIAL_FORCES_GAUSSPOINT",
                  {.description = "write material cross-section stresses at the Gauss points",
                      .default_value = false}),

              // write element filament numbers and type
              parameter<bool>("BEAMFILAMENTCONDITION",
                  {.description = "write element filament numbers", .default_value = false}),

              // write element and network orientation parameter
              parameter<bool>("ORIENTATION_PARAMETER",
                  {.description = "write element filament numbers", .default_value = false}),

              // write crossection forces of periodic RVE
              parameter<bool>("RVE_CROSSSECTION_FORCES",
                  {.description = " get sum of all internal forces of  ", .default_value = false}),

              // write reference length of beams
              parameter<bool>("REF_LENGTH",
                  {.description = "write reference length of all beams", .default_value = false}),

              // write element GIDs
              parameter<bool>("ELEMENT_GID",
                  {.description = "write the 4C internal element GIDs", .default_value = false}),

              // write element ghosting information
              parameter<bool>(
                  "ELEMENT_GHOSTING", {.description = "write which processors ghost the elements",
                                          .default_value = false}),

              // number of subsegments along a single beam element for visualization
              parameter<int>("NUMBER_SUBSEGMENTS",
                  {.description =
                          "Number of subsegments along a single beam element for visualization",
                      .default_value = 5})},
          {.required = false});
    }
  }  // namespace Beam
}  // namespace IORuntimeOutput

FOUR_C_NAMESPACE_CLOSE