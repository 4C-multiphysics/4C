// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_edge_contact_params.hpp"

#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_beam_to_solid_conditions.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_spec_validators.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN



Core::IO::InputSpec BeamInteraction::valid_beam_to_edge_contact_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  using namespace Core::IO::InputSpecBuilders::Validators;

  // Beam to solid edge contact parameters.
  Core::IO::InputSpec spec = group<BeamToSolidEdgeContactParameters>(
      "BEAM INTERACTION/BEAM TO SOLID EDGE CONTACT",
      {

          parameter<Inpar::BeamToSolid::BeamToSolidConstraintEnforcement>("CONSTRAINT_ENFORCEMENT",
              {.description = "How the beam-to-edge constraints should be enforced.",
                  .default_value = Inpar::BeamToSolid::BeamToSolidConstraintEnforcement::none,
                  .store = in_struct(&BeamToSolidEdgeContactParameters::constraint_enforcement)}),

          group<PenaltyLawParameters>("PENALTY_LAW",
              {

                  parameter<Inpar::BeamToSolid::BeamToSolidSurfaceContactPenaltyLaw>(
                      "TYPE", {.description = "Type of penalty law",
                                  .default_value =
                                      Inpar::BeamToSolid::BeamToSolidSurfaceContactPenaltyLaw::none,
                                  .store = in_struct(&PenaltyLawParameters::type)}),

                  parameter<double>("PENALTY_PARAMETER",
                      {.description = "Penalty parameter for beam-to-solid surface contact",
                          .default_value = 0.0,
                          .validator = positive_or_zero<double>(),
                          .store = in_struct(&PenaltyLawParameters::penalty_parameter)}),

                  parameter<double>("PENALTY_PARAMETER_G0",
                      {.description = "First penalty regularization parameter G0 >=0: For gap<G0 "
                                      "contact is active",
                          .default_value = 0.0,
                          .store = in_struct(&PenaltyLawParameters::penalty_parameter_g0)})

              },
              {.description = "Parameters for penalty law",
                  .store = in_struct(&BeamToSolidEdgeContactParameters::penalty_law)}),
      },
      {.description = "Parameters for beam-to-solid edge contact.", .required = false});

  return spec;
};

void BeamInteraction::set_valid_beam_to_edge_contact_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  std::array<std::string, 2> condition_names;
  Inpar::BeamToSolid::beam_to_solid_interaction_get_string(
      Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_edge_contact,
      condition_names);

  Core::Conditions::ConditionDefinition beam_to_solid_edge_contact_condition(
      "BEAM INTERACTION/BEAM TO SOLID EDGE CONTACT BEAM", condition_names[0],
      "Beam-to-edge contact conditions - beam", Core::Conditions::BeamToSolidEdgeContactBeam, true,
      Core::Conditions::geometry_type_line);
  beam_to_solid_edge_contact_condition.add_component(parameter<int>("COUPLING_ID"));
  condlist.push_back(beam_to_solid_edge_contact_condition);

  beam_to_solid_edge_contact_condition = Core::Conditions::ConditionDefinition(
      "BEAM INTERACTION/BEAM TO SOLID EDGE CONTACT SOLID EDGE", condition_names[1],
      "Beam-to-edge contact conditions - solid edge", Core::Conditions::BeamToSolidEdgeContactEdge,
      true, Core::Conditions::geometry_type_line);
  beam_to_solid_edge_contact_condition.add_component(parameter<int>("COUPLING_ID"));
  condlist.push_back(beam_to_solid_edge_contact_condition);
}

std::unique_ptr<BeamInteraction::BeamToSolidEdgeContactParameters>
BeamInteraction::initialize_validate_beam_to_edge_contact_params()

{
  auto beam_to_edge_contact_params =
      std::make_unique<BeamInteraction::BeamToSolidEdgeContactParameters>(
          Global::Problem::instance()->parameters().get<BeamToSolidEdgeContactParameters>(
              "BEAM INTERACTION/BEAM TO SOLID EDGE CONTACT"));

  if (beam_to_edge_contact_params->penalty_law.type ==
      Inpar::BeamToSolid::BeamToSolidSurfaceContactPenaltyLaw::none)
    FOUR_C_THROW("Penalty law is required for Beam-to-solid edge contact");

  return beam_to_edge_contact_params;
};

FOUR_C_NAMESPACE_CLOSE