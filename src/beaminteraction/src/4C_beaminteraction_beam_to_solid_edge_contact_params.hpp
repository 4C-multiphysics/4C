// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_EDGE_CONTACT_PARAMS_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_SOLID_EDGE_CONTACT_PARAMS_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_io_input_spec.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  /// Parameters for beam-to-solid edge contact
  struct BeamToSolidEdgeContactParameters
  {
    Inpar::BeamToSolid::BeamToSolidConstraintEnforcement constraint_enforcement;
    PenaltyLawParameters penalty_law;
  };


  Core::IO::InputSpec valid_beam_to_edge_contact_parameters();

  void set_valid_beam_to_edge_contact_conditions(
      std::vector<Core::Conditions::ConditionDefinition>& condlist);

  std::unique_ptr<BeamInteraction::BeamToSolidEdgeContactParameters>
  initialize_validate_beam_to_edge_contact_params();

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
