// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_params_base.hpp"

#include "4C_geometry_pair_input.hpp"
#include "4C_global_data.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToSolidParamsBase::BeamToSolidParamsBase()
    : isinit_(false),
      issetup_(false),
      constraint_enforcement_(Inpar::BeamToSolid::BeamToSolidConstraintEnforcement::none),
      contact_discretization_(Inpar::BeamToSolid::BeamToSolidContactDiscretization::none),
      mortar_shape_function_(Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::none),
      penalty_parameter_(-1.0),
      gauss_rule_(Core::FE::GaussRule1D::undefined),
      rotational_coupling_(false),
      lagrange_formulation_(Inpar::BeamToSolid::BeamToSolidLagrangeFormulation::none)
{
  // Empty Constructor.
}


/**
 *
 */
void BeamInteraction::BeamToSolidParamsBase::set_base_params(
    const Teuchos::ParameterList& beam_to_solid_params_list)
{
  // Get parameters form input file.
  {
    // Constraint enforcement.
    constraint_enforcement_ =
        Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidConstraintEnforcement>(
            beam_to_solid_params_list, "CONSTRAINT_STRATEGY");

    //  Type of lagrange formulation used
    if (beam_to_solid_params_list.isParameter("LAGRANGE_FORMULATION"))
    {
      lagrange_formulation_ =
          Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidLagrangeFormulation>(
              beam_to_solid_params_list, "LAGRANGE_FORMULATION");
    }

    // Contact discretization to be used.
    contact_discretization_ =
        Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidContactDiscretization>(
            beam_to_solid_params_list, "CONTACT_DISCRETIZATION");

    // Contact discretization to be used.
    mortar_shape_function_ =
        Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidMortarShapefunctions>(
            beam_to_solid_params_list, "MORTAR_SHAPE_FUNCTION");

    // Penalty parameter.
    penalty_parameter_ = beam_to_solid_params_list.get<double>("PENALTY_PARAMETER");
    if (penalty_parameter_ < 0.0)
      FOUR_C_THROW("beam-to-volume-meshtying penalty parameter must not be negative!");

    // Gauss rule for integration along the beam (segments).
    gauss_rule_ =
        GeometryPair::int_to_gauss_rule1_d(beam_to_solid_params_list.get<int>("GAUSS_POINTS"));
  }

  isinit_ = true;
}


/**
 *
 */
void BeamInteraction::BeamToSolidParamsBase::setup()
{
  check_init();

  // Empty for now.

  issetup_ = true;
}

FOUR_C_NAMESPACE_CLOSE
