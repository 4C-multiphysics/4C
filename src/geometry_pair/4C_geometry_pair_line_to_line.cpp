// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometry_pair_line_to_line.hpp"

#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_utility_classes.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
template <typename ScalarType, typename LineA, typename LineB>
GeometryPair::ProjectionResult GeometryPair::line_to_line_closest_point_projection(
    const ElementData<LineA, ScalarType>& element_data_line_a,
    const ElementData<LineB, ScalarType>& element_data_line_b, ScalarType& eta_a, ScalarType& eta_b,
    const bool min_one_iteration)
{
  // Position and derivative vectors
  Core::LinAlg::Matrix<3, 1, ScalarType> diff;
  Core::LinAlg::Matrix<3, 1, ScalarType> r_a;
  Core::LinAlg::Matrix<3, 1, ScalarType> r_b;
  Core::LinAlg::Matrix<3, 1, ScalarType> dr_a;
  Core::LinAlg::Matrix<3, 1, ScalarType> dr_b;
  Core::LinAlg::Matrix<3, 1, ScalarType> ddr_a;
  Core::LinAlg::Matrix<3, 1, ScalarType> ddr_b;

  // Initialize the residual and increment (increment with a value that will not pass the first
  // convergence check)
  Core::LinAlg::Matrix<2, 1, ScalarType> residuum;
  Core::LinAlg::Matrix<2, 1, ScalarType> delta_eta;
  delta_eta.put_scalar(10 * Constants::projection_xi_eta_tol);

  // Jacobian / inverse.
  Core::LinAlg::Matrix<2, 2, ScalarType> J_J_inv;

  // Local Newton iteration.
  unsigned int counter = 0;
  while (counter < Constants::local_newton_iter_max)
  {
    evaluate_position(eta_a, element_data_line_a, r_a);
    evaluate_position(eta_b, element_data_line_b, r_b);
    evaluate_position_derivative1(eta_a, element_data_line_a, dr_a);
    evaluate_position_derivative1(eta_b, element_data_line_b, dr_b);
    evaluate_position_derivative2(eta_a, element_data_line_a, ddr_a);
    evaluate_position_derivative2(eta_b, element_data_line_b, ddr_b);

    // Calculate the residuum
    diff = r_a;
    diff -= r_b;
    residuum(0) = diff.dot(dr_a);
    residuum(1) = diff.dot(dr_b);

    // Calculate the jacobian
    J_J_inv(0, 0) = ddr_a.dot(diff) + dr_a.dot(dr_a);
    J_J_inv(0, 1) = -dr_a.dot(dr_b);
    J_J_inv(1, 0) = dr_b.dot(dr_a);
    J_J_inv(1, 1) = ddr_b.dot(diff) - dr_b.dot(dr_b);

    if (counter == 0 and min_one_iteration)
    {
      // if the min_one_iteration flag is set we run at least one iteration, so the dependency on
      // FAD variables is calculated correctly.
    }
    else if (Core::FADUtils::vector_norm(residuum) < Constants::local_newton_res_tol &&
             Core::FADUtils::vector_norm(delta_eta) < Constants::projection_xi_eta_tol)
    {
      if (valid_parameter_1d(eta_a) && valid_parameter_1d(eta_b))
        return ProjectionResult::projection_found_valid;
      else
        return ProjectionResult::projection_found_not_valid;
    }

    // Check if residuum is in a sensible range where we still expect to find a solution.
    if (Core::FADUtils::vector_norm(residuum) > Constants::local_newton_res_max) break;

    // Solve the linearized system.
    if (Core::LinAlg::solve_linear_system_do_not_throw_error_on_zero_determinant_scaled(
            J_J_inv, residuum, delta_eta, Constants::local_newton_det_tol))
    {
      // Set the new parameter coordinates.
      eta_a -= delta_eta(0);
      eta_b -= delta_eta(1);

      // Advance Newton iteration counter.
      counter++;
    }
    else
    {
      // System could not be solved (likely due to a singular Jacobian).
      return ProjectionResult::projection_not_found;
    }
  }

  // If we end up here, no convergence was achieved.
  return ProjectionResult::projection_not_found;
}


/**
 * Explicit template initialization of template class.
 */
namespace GeometryPair
{
  template ProjectionResult line_to_line_closest_point_projection<Sacado::ELRFad::DFad<double>,
      t_hermite, t_line2>(const ElementData<t_hermite, Sacado::ELRFad::DFad<double>>&,
      const ElementData<t_line2, Sacado::ELRFad::DFad<double>>&, Sacado::ELRFad::DFad<double>&,
      Sacado::ELRFad::DFad<double>&, const bool);
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE
