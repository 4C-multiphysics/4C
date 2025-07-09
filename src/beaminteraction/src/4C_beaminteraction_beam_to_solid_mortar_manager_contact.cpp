// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_mortar_manager_contact.hpp"

#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_fad.hpp"


FOUR_C_NAMESPACE_OPEN

/**
 *
 */
BeamInteraction::BeamToSolidMortarManagerContact::BeamToSolidMortarManagerContact(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<const BeamInteraction::BeamToSolidParamsBase>& params,
    int start_value_lambda_gid)
    : BeamToSolidMortarManager(discret, params, start_value_lambda_gid)
{
}

/**
 *
 */
std::tuple<std::shared_ptr<Core::LinAlg::Vector<double>>,
    std::shared_ptr<Core::LinAlg::Vector<double>>, std::shared_ptr<Core::LinAlg::Vector<double>>>
BeamInteraction::BeamToSolidMortarManagerContact::get_penalty_regularization(
    const bool compute_linearization) const
{
  using fad_type = fad_type_1st_order_2_variables;
  const auto beam_to_solid_contact_params =
      std::dynamic_pointer_cast<const BeamInteraction::BeamToSolidSurfaceContactParams>(
          beam_to_solid_params_);

  // Get the penalty regularized Lagrange multipliers and the derivative w.r.t. the constraint
  // vector (averaged gap) and the scaling vector (kappa)
  auto create_lambda_row_vector_with_zeros = [this]()
  {
    auto row_vector = std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);
    row_vector->put_scalar(0.0);
    return row_vector;
  };
  auto lambda = create_lambda_row_vector_with_zeros();
  auto lambda_lin_constraint = create_lambda_row_vector_with_zeros();
  auto lambda_lin_kappa = create_lambda_row_vector_with_zeros();
  for (int lid = 0; lid < lambda_dof_rowmap_->num_my_elements(); lid++)
  {
    if (lambda_active_->get_values()[lid] > 0.1)
    {
      const fad_type weighted_gap = Core::FADUtils::HigherOrderFadValue<fad_type>::apply(
          2, 0, constraint_->get_values()[lid]);
      const fad_type kappa =
          Core::FADUtils::HigherOrderFadValue<fad_type>::apply(2, 1, kappa_->get_values()[lid]);
      const fad_type scaled_gap = weighted_gap / kappa;

      // The -1 here is due to the way the lagrange multipliers are defined in the coupling
      // constraints.
      const fad_type local_lambda = -1.0 * penalty_force(scaled_gap, *beam_to_solid_contact_params);
      lambda->replace_local_value(lid, Core::FADUtils::cast_to_double(local_lambda));
      lambda_lin_constraint->replace_local_value(lid, local_lambda.dx(0));
      lambda_lin_kappa->replace_local_value(lid, local_lambda.dx(1));
    }
  }
  return {lambda, lambda_lin_constraint, lambda_lin_kappa};
}

FOUR_C_NAMESPACE_CLOSE