// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_input.hpp"

#include "4C_cut_enum.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Cut::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace FourC::Cut;
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs cut_general{"CUT GENERAL"};

  // intersection precision (double or cln)
  Core::Utils::string_to_integral_parameter<FourC::Cut::CutFloatType>(
      "KERNEL_INTERSECTION_FLOATTYPE", "double",
      "The floattype of the cut surface-edge intersection", tuple<std::string>("cln", "double"),
      tuple<FourC::Cut::CutFloatType>(floattype_cln, floattype_double), cut_general);

  // Computing disctance surface to point precision (double or cln)
  Core::Utils::string_to_integral_parameter<FourC::Cut::CutFloatType>("KERNEL_DISTANCE_FLOATTYPE",
      "double", "The floattype of the cut distance computation",
      tuple<std::string>("cln", "double"),
      tuple<FourC::Cut::CutFloatType>(floattype_cln, floattype_double), cut_general);

  // A general floattype for Cut::Position for Embedded Elements (compute_distance)
  // If specified this floattype is used for all computations of Cut::Position with
  // embedded elements
  Core::Utils::string_to_integral_parameter<FourC::Cut::CutFloatType>(
      "GENERAL_POSITION_DISTANCE_FLOATTYPE", "none",
      "A general floattype for Cut::Position for Embedded Elements (compute_distance)",
      tuple<std::string>("none", "cln", "double"),
      tuple<FourC::Cut::CutFloatType>(floattype_none, floattype_cln, floattype_double),
      cut_general);

  // A general floattype for Cut::Position for Elements (ComputePosition)
  // If specified this floattype is used for all computations of Cut::Position
  Core::Utils::string_to_integral_parameter<FourC::Cut::CutFloatType>(
      "GENERAL_POSITION_POSITION_FLOATTYPE", "none",
      "A general floattype for Cut::Position Elements (ComputePosition)",
      tuple<std::string>("none", "cln", "double"),
      tuple<FourC::Cut::CutFloatType>(floattype_none, floattype_cln, floattype_double),
      cut_general);

  // Specify which Referenceplanes are used in DirectDivergence
  Core::Utils::string_to_integral_parameter<FourC::Cut::CutDirectDivergenceRefplane>(
      "DIRECT_DIVERGENCE_REFPLANE", "all",
      "Specify which Referenceplanes are used in DirectDivergence",
      tuple<std::string>("all", "diagonal_side", "facet", "diagonal", "side", "none"),
      tuple<FourC::Cut::CutDirectDivergenceRefplane>(DirDiv_refplane_all,
          DirDiv_refplane_diagonal_side, DirDiv_refplane_facet, DirDiv_refplane_diagonal,
          DirDiv_refplane_side, DirDiv_refplane_none),
      cut_general);

  // Specify is Cutsides are triangulated
  cut_general.specs.emplace_back(parameter<bool>("SPLIT_CUTSIDES",
      {.description = "Split Quad4 CutSides into Tri3-Subtriangles?", .default_value = true}));

  // Do the Selfcut before standard CUT
  cut_general.specs.emplace_back(
      parameter<bool>("DO_SELFCUT", {.description = "Do the SelfCut?", .default_value = true}));

  // Do meshcorrection in Selfcut
  cut_general.specs.emplace_back(parameter<bool>("SELFCUT_DO_MESHCORRECTION",
      {.description = "Do meshcorrection in the SelfCut?", .default_value = true}));

  // Selfcut meshcorrection multiplicator
  Core::Utils::int_parameter("SELFCUT_MESHCORRECTION_MULTIPLICATOR", 30,
      "ISLANDS with maximal size of the bounding box of h*multiplacator will be removed in the "
      "meshcorrection",
      cut_general);

  // Cubaturedegree utilized for the numerical integration on the CUT BoundaryCells.
  Core::Utils::int_parameter("BOUNDARYCELL_CUBATURDEGREE", 20,
      "Cubaturedegree utilized for the numerical integration on the CUT BoundaryCells.",
      cut_general);

  // Integrate inside volume cells
  cut_general.specs.emplace_back(parameter<bool>("INTEGRATE_INSIDE_CELLS",
      {.description = "Should the integration be done on inside cells", .default_value = true}));

  cut_general.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
