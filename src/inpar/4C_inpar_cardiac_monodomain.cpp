// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_cardiac_monodomain.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::ElectroPhysiology::set_valid_parameters(
    std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs epcontrol{"CARDIAC MONODOMAIN CONTROL"};

  // Parameters for reaction-diffusion systems (for example cardiac electrophysiology)
  Core::Utils::int_parameter("WRITEMAXINTSTATE", 0,
      "number of maximal internal state variables to be postprocessed", epcontrol);
  Core::Utils::int_parameter("WRITEMAXIONICCURRENTS", 0,
      "number of maximal ionic currents to be postprocessed", epcontrol);

  epcontrol.specs.emplace_back(parameter<double>("ACTTHRES",
      {.description =
              "threshold for the potential for computing and postprocessing activation time ",
          .default_value = 1.0}));

  epcontrol.move_into_collection(list);
}


void Inpar::ElectroPhysiology::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
}

FOUR_C_NAMESPACE_CLOSE
