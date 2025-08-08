// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_visualization_output_params.hpp"
#include "4C_geometry_pair_input.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToSolidVolumeMeshtyingParams::BeamToSolidVolumeMeshtyingParams()
    : BeamToSolidParamsBase(),
      integration_points_circumference_(0),
      n_fourier_modes_(-1),
      rotational_coupling_triad_construction_(
          Inpar::BeamToSolid::BeamToSolidRotationCoupling::none),
      rotational_coupling_penalty_parameter_(0.0),
      output_params_ptr_(nullptr),
      couple_restart_state_(false)
{
  // Empty Constructor.
}


/**
 *
 */
void BeamInteraction::BeamToSolidVolumeMeshtyingParams::init()
{
  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_to_solid_contact_params_list =
      Global::Problem::instance()->beam_interaction_params().sublist(
          "BEAM TO SOLID VOLUME MESHTYING");

  // Set the common beam-to-solid parameters.
  set_base_params(beam_to_solid_contact_params_list);

  // Get parameters form input file.
  {
    // Number of integrations points along the circumference of the cross section.
    integration_points_circumference_ =
        beam_to_solid_contact_params_list.get<int>("INTEGRATION_POINTS_CIRCUMFERENCE");

    // Number of Fourier modes.
    n_fourier_modes_ = beam_to_solid_contact_params_list.get<int>("MORTAR_FOURIER_MODES");

    // Type of rotational coupling.
    rotational_coupling_triad_construction_ =
        Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidRotationCoupling>(
            beam_to_solid_contact_params_list, "ROTATION_COUPLING");
    rotational_coupling_ = rotational_coupling_triad_construction_ !=
                           Inpar::BeamToSolid::BeamToSolidRotationCoupling::none;

    // Type of constraint enforcement
    constraint_enforcement_ =
        Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidConstraintEnforcement>(
            beam_to_solid_contact_params_list, "CONSTRAINT_STRATEGY");


    // Mortar contact discretization to be used.
    mortar_shape_function_rotation_ =
        Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidMortarShapefunctions>(
            beam_to_solid_contact_params_list, "ROTATION_COUPLING_MORTAR_SHAPE_FUNCTION");
    if (get_contact_discretization() ==
            Inpar::BeamToSolid::BeamToSolidContactDiscretization::mortar and
        rotational_coupling_ and
        mortar_shape_function_rotation_ ==
            Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::none)
      FOUR_C_THROW(
          "If mortar coupling and rotational coupling are activated, the shape function type for "
          "rotational coupling has to be explicitly given.");

    // Penalty parameter for rotational coupling.
    rotational_coupling_penalty_parameter_ =
        beam_to_solid_contact_params_list.get<double>("ROTATION_COUPLING_PENALTY_PARAMETER");

    // If the restart configuration should be coupled.
    couple_restart_state_ = beam_to_solid_contact_params_list.get<bool>("COUPLE_RESTART_STATE");
  }

  // Setup the output parameter object.
  {
    output_params_ptr_ = std::make_shared<BeamToSolidVolumeMeshtyingVisualizationOutputParams>();
    output_params_ptr_->init();
    output_params_ptr_->setup();
  }

  // Sanity checks.
  if (rotational_coupling_ and couple_restart_state_)
    FOUR_C_THROW(
        "Coupling restart state combined with rotational coupling is not yet implemented!");

  isinit_ = true;
}

/**
 *
 */
std::shared_ptr<BeamInteraction::BeamToSolidVolumeMeshtyingVisualizationOutputParams>
BeamInteraction::BeamToSolidVolumeMeshtyingParams::get_visualization_output_params_ptr()
{
  return output_params_ptr_;
};

FOUR_C_NAMESPACE_CLOSE
