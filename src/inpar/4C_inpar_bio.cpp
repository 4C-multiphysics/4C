// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_bio.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN


Core::IO::InputSpec Inpar::BioFilm::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  Core::IO::InputSpec spec = group("BIOFILM CONTROL",
      {

          parameter<bool>("BIOFILMGROWTH",
              {.description = "Scatra algorithm for biofilm growth", .default_value = false}),
          parameter<bool>("AVGROWTH",
              {.description = "The calculation of growth parameters is based on averaged values",
                  .default_value = false}),
          parameter<double>("FLUXCOEF",
              {.description = "Coefficient for growth due to scalar flux", .default_value = 0.0}),
          parameter<double>("NORMFORCEPOSCOEF",
              {.description = "Coefficient for erosion due to traction normal surface forces",
                  .default_value = 0.0}),
          parameter<double>("NORMFORCENEGCOEF",
              {.description = "Coefficient for erosion due to compression normal surface forces",
                  .default_value = 0.0}),
          parameter<double>("TANGONEFORCECOEF",
              {.description = "Coefficient for erosion due to the first tangential surface force",
                  .default_value = 0.0}),
          parameter<double>("TANGTWOFORCECOEF",
              {.description = "Coefficient for erosion due to the second tangential surface force",
                  .default_value = 0.0}),
          parameter<double>("BIOTIMESTEP",
              {.description = "Time step size for biofilm growth", .default_value = 0.05}),
          parameter<int>("BIONUMSTEP",
              {.description = "Maximum number of steps for biofilm growth", .default_value = 0}),
          parameter<bool>(
              "OUTPUT_GMSH", {.description = "Do you want to write Gmsh postprocessing files?",
                                 .default_value = false})},
      {.required = false});
  return spec;
}



void Inpar::BioFilm::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // Additional coupling for biofilm growth
  Core::Conditions::ConditionDefinition surfbiogr("DESIGN BIOFILM GROWTH COUPLING SURF CONDITIONS",
      "BioGrCoupling", "BioGrCoupling", Core::Conditions::BioGrCoupling, true,
      Core::Conditions::geometry_type_surface);

  surfbiogr.add_component(parameter<int>("coupling_id", {.description = "coupling_id"}));
  condlist.push_back(surfbiogr);
}

FOUR_C_NAMESPACE_CLOSE