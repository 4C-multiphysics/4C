// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_input.hpp"

#include "4C_cut_enum.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN



void Cut::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Cut;
  using namespace Core::IO::InputSpecBuilders;

  list["CUT GENERAL"] = group("CUT GENERAL",
      {

          // intersection precision (double or cln)
          deprecated_selection<Cut::CutFloatType>("KERNEL_INTERSECTION_FLOATTYPE",
              {
                  {"cln", floattype_cln},
                  {"double", floattype_double},
              },
              {.description = "The floattype of the cut surface-edge intersection",
                  .default_value = floattype_double}),

          // Computing disctance surface to point precision (double or cln)
          deprecated_selection<Cut::CutFloatType>("KERNEL_DISTANCE_FLOATTYPE",
              {
                  {"cln", floattype_cln},
                  {"double", floattype_double},
              },
              {.description = "The floattype of the cut distance computation",
                  .default_value = floattype_double}),

          // A general floattype for Cut::Position for Embedded Elements (compute_distance)
          // If specified this floattype is used for all computations of Cut::Position with
          // embedded elements
          deprecated_selection<Cut::CutFloatType>("GENERAL_POSITION_DISTANCE_FLOATTYPE",
              {
                  {"none", floattype_none},
                  {"cln", floattype_cln},
                  {"double", floattype_double},
              },
              {.description = "A general floattype for Cut::Position for Embedded Elements "
                              "(compute_distance)",
                  .default_value = floattype_none}),

          // A general floattype for Cut::Position for Elements (ComputePosition)
          // If specified this floattype is used for all computations of Cut::Position
          deprecated_selection<Cut::CutFloatType>("GENERAL_POSITION_POSITION_FLOATTYPE",
              {
                  {"none", floattype_none},
                  {"cln", floattype_cln},
                  {"double", floattype_double},
              },
              {.description = "A general floattype for Cut::Position Elements (ComputePosition)",
                  .default_value = floattype_none}),

          // Specify which Referenceplanes are used in DirectDivergence
          deprecated_selection<Cut::CutDirectDivergenceRefplane>("DIRECT_DIVERGENCE_REFPLANE",
              {
                  {"all", DirDiv_refplane_all},
                  {"diagonal_side", DirDiv_refplane_diagonal_side},
                  {"facet", DirDiv_refplane_facet},
                  {"diagonal", DirDiv_refplane_diagonal},
                  {"side", DirDiv_refplane_side},
                  {"none", DirDiv_refplane_none},
              },
              {.description = "Specify which Referenceplanes are used in DirectDivergence",
                  .default_value = DirDiv_refplane_all}),

          // Specify is Cutsides are triangulated
          parameter<bool>(
              "SPLIT_CUTSIDES", {.description = "Split Quad4 CutSides into Tri3-Subtriangles?",
                                    .default_value = true}),

          // Do the Selfcut before standard CUT
          parameter<bool>("DO_SELFCUT", {.description = "Do the SelfCut?", .default_value = true}),

          // Do meshcorrection in Selfcut
          parameter<bool>("SELFCUT_DO_MESHCORRECTION",
              {.description = "Do meshcorrection in the SelfCut?", .default_value = true}),

          // Selfcut meshcorrection multiplicator
          parameter<int>("SELFCUT_MESHCORRECTION_MULTIPLICATOR",
              {.description =
                      "ISLANDS with maximal size of the bounding box of h*multiplacator will "
                      "be removed in "
                      "the meshcorrection",
                  .default_value = 30}),

          // Cubaturedegree utilized for the numerical integration on the CUT BoundaryCells.
          parameter<int>("BOUNDARYCELL_CUBATURDEGREE",
              {.description = "Cubaturedegree utilized for the numerical integration on the CUT "
                              "BoundaryCells.",
                  .default_value = 20}),

          // Integrate inside volume cells
          parameter<bool>("INTEGRATE_INSIDE_CELLS",
              {.description = "Should the integration be done on inside cells",
                  .default_value = true})},
      {.required = false});
}

FOUR_C_NAMESPACE_CLOSE