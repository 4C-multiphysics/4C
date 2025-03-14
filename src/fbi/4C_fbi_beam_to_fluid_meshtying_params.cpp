// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"

#include "4C_fbi_beam_to_fluid_meshtying_output_params.hpp"
#include "4C_fbi_input.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_geometry_pair.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
FBI::BeamToFluidMeshtyingParams::BeamToFluidMeshtyingParams()
    : isinit_(false),
      issetup_(false),
      constraint_enforcement_(FBI::BeamToFluidConstraintEnforcement::none),
      meshtying_discretization_(FBI::BeamToFluidDiscretization::none),
      penalty_parameter_(-1.0),
      gauss_rule_(Core::FE::GaussRule1D::undefined),
      calcfluidweakdirichletforce_(false),
      mortar_shape_function_(FBI::BeamToFluidMeshtingMortarShapefunctions::none)
{
  // Empty Constructor.
}

/**
 *
 */
void FBI::BeamToFluidMeshtyingParams::init()
{
  // Teuchos parameter list for beam contact
  const Teuchos::ParameterList& beam_to_fluid_meshtying_params_list =
      Global::Problem::instance()->get_parameter_list()->sublist("FLUID BEAM INTERACTION");

  // Get parameters form input file.
  // Constraint enforcement.
  constraint_enforcement_ = Teuchos::getIntegralValue<FBI::BeamToFluidConstraintEnforcement>(
      beam_to_fluid_meshtying_params_list.sublist("BEAM TO FLUID MESHTYING"),
      "CONSTRAINT_STRATEGY");

  // Constraint enforcement.
  mortar_shape_function_ = Teuchos::getIntegralValue<FBI::BeamToFluidMeshtingMortarShapefunctions>(
      beam_to_fluid_meshtying_params_list.sublist("BEAM TO FLUID MESHTYING"),
      "MORTAR_SHAPE_FUNCTION");

  // Contact discretization to be used.
  meshtying_discretization_ = Teuchos::getIntegralValue<FBI::BeamToFluidDiscretization>(
      beam_to_fluid_meshtying_params_list.sublist("BEAM TO FLUID MESHTYING"),
      "MESHTYING_DISCRETIZATION");

  // Penalty parameter.
  penalty_parameter_ = (beam_to_fluid_meshtying_params_list.sublist("BEAM TO FLUID MESHTYING"))
                           .get<double>("PENALTY_PARAMETER");
  if (penalty_parameter_ < 0.0)
    FOUR_C_THROW("beam-to-volume-meshtying penalty parameter must not be negative!");

  // Gauss rule for integration along the beam (segments).
  gauss_rule_ = Inpar::GEOMETRYPAIR::int_to_gauss_rule1_d(
      beam_to_fluid_meshtying_params_list.sublist("BEAM TO FLUID MESHTYING")
          .get<int>("GAUSS_POINTS"));
  isinit_ = true;

  // Create and get visualization output parameter
  output_params_ = std::make_shared<FBI::BeamToFluidMeshtyingVtkOutputParams>();
  output_params_->init();
  output_params_->setup();
}


/**
 *
 */
void FBI::BeamToFluidMeshtyingParams::setup()
{
  check_init();

  // Empty for now.

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE
