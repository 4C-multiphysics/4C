// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_input.hpp"

#include "4C_beaminteraction_contact_beam_to_beam_point_coupling_pair.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_input.hpp"
#include "4C_beaminteraction_contact_input.hpp"
#include "4C_beaminteraction_crosslinking_input.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_beaminteraction_spherebeamlinking_input.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_legacy_enum_definitions_conditions.hpp"

FOUR_C_NAMESPACE_OPEN


std::vector<Core::IO::InputSpec> BeamInteraction::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("BEAM INTERACTION",
      {

          deprecated_selection<BeamInteraction::RepartitionStrategy>("REPARTITIONSTRATEGY",
              {
                  {"Adaptive", repstr_adaptive},
                  {"adaptive", repstr_adaptive},
                  {"Everydt", repstr_everydt},
                  {"everydt", repstr_everydt},
              },
              {.description = "Type of employed repartitioning strategy",
                  .default_value = repstr_adaptive})},
      {.required = false}));

  // get beam contact parameters
  std::vector<Core::IO::InputSpec> contact_specs = BeamInteraction::valid_parameters_contact();
  specs.insert(specs.end(), contact_specs.begin(), contact_specs.end());

  // get beam crosslinking parameters
  std::vector<Core::IO::InputSpec> crosslinking_specs =
      BeamInteraction::valid_parameters_crosslinking();
  specs.insert(specs.end(), crosslinking_specs.begin(), crosslinking_specs.end());

  // get beam potential parameters
  Core::IO::InputSpec beam_potential_specs = BeamInteraction::Potential::valid_parameters();
  specs.push_back(beam_potential_specs);

  // get sphere beam linking parameters
  std::vector<Core::IO::InputSpec> spherebeamlinking_specs =
      BeamInteraction::valid_parameters_spherebeamlinking();
  specs.insert(specs.end(), spherebeamlinking_specs.begin(), spherebeamlinking_specs.end());

  return specs;
}

void BeamInteraction::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*-------------------------------------------------------------------*/
  // beam potential interaction: atom/charge density per unit length on LINE
  Core::Conditions::ConditionDefinition beam_filament_condition(
      "DESIGN LINE BEAM FILAMENT CONDITIONS", "BeamLineFilamentCondition",
      "Beam_Line_Filament_Condition", Core::Conditions::FilamentBeamLineCondition, false,
      Core::Conditions::geometry_type_line);

  beam_filament_condition.add_component(parameter<int>("ID", {.description = "filament id"}));
  beam_filament_condition.add_component(deprecated_selection<std::string>("TYPE",
      {"Arbitrary", "arbitrary", "Actin", "actin", "Collagen", "collagen"},
      {.description = "", .default_value = "Arbitrary"}));

  condlist.push_back(beam_filament_condition);

  /*-------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition penalty_coupling_condition_direct(
      "DESIGN POINT PENALTY COUPLING CONDITIONS", "PenaltyPointCouplingConditionDirect",
      "Couples beam nodes that lie on the same position",
      Core::Conditions::PenaltyPointCouplingConditionDirect, false,
      Core::Conditions::geometry_type_point);

  penalty_coupling_condition_direct.add_component(
      parameter<double>("POSITIONAL_PENALTY_PARAMETER"));
  penalty_coupling_condition_direct.add_component(
      parameter<double>("ROTATIONAL_PENALTY_PARAMETER"));

  condlist.push_back(penalty_coupling_condition_direct);

  // beam-to-solid interactions
  BeamToSolid::set_valid_conditions(condlist);

  // Beam-to-beam point couplings based on CPP conditions.
  {
    Core::Conditions::ConditionDefinition penalty_coupling_condition_indirect(
        "BEAM INTERACTION/BEAM TO BEAM POINT COUPLING CONDITIONS",
        "PenaltyPointCouplingConditionIndirect",
        "Coupling conditions between beams based on closest point projections.",
        Core::Conditions::PenaltyPointCouplingConditionIndirect, true,
        Core::Conditions::geometry_type_line);
    penalty_coupling_condition_indirect.add_component(parameter<int>("COUPLING_ID"));
    penalty_coupling_condition_indirect.add_component(group("PARAMETERS",
        {parameter<double>("POSITIONAL_PENALTY_PARAMETER", {.default_value = 0.0}),
            parameter<double>("ROTATIONAL_PENALTY_PARAMETER", {.default_value = 0.0}),
            parameter<double>("PROJECTION_VALID_FACTOR",
                {.description = "Factor multiplied with sum of cross section "
                                "radii to define valid projection distance",
                    .default_value = 2.0}),
            parameter<int>("MAX_NUMBER_OF_PAIRS_PER_ELEMENT",
                {.description = "How many Lagrange multipliers shall be allocated per beam element",
                    .default_value = 0}),
            parameter<
                BeamInteraction::BeamToBeamPointCouplingPairParameters::ConstraintEnforcement>(
                "CONSTRAINT_ENFORCEMENT",
                {.description = "Specifies the constraint enforcement technique for beam-to-beam "
                                "point couplings."})},
        {.required = false}));
    condlist.push_back(penalty_coupling_condition_indirect);
  }
}

FOUR_C_NAMESPACE_CLOSE