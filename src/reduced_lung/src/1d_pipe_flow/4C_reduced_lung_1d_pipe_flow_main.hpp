// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_MAIN_HPP
#define FOUR_C_REDUCED_LUNG_1D_PIPE_FLOW_MAIN_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_1d_pipe_flow_input.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung1dPipeFlow
{
  /**
   * Fills vectors created from nodal maps with data from input file.
   *
   * Input values are listed in @param parameters and accessed to fill the passed vectors.
   * The following vectors are filled with properties for each node, given that they are described
   * elementwise in the inputfile:
   * @param reference_area @param thickness @param Young @param beta @param radius
   * e.g. X = [X_0 X_1 ...]^T The solution
   * vector is set to the initial state with the two DOF/ node:
   *    @param solution = [A_0 u_0 A_1 u_1 ...]^T
   *  The @param discretization is required, as writing to the vectors is
   * performed across MPI ranks.
   */
  void fill_parameters(Parameters& parameters, Core::LinAlg::Vector<double>& solution,
      Core::LinAlg::Vector<double>& reference_area, Core::LinAlg::Vector<double>& thickness,
      Core::LinAlg::Vector<double>& Young, Core::LinAlg::Vector<double>& beta,
      Core::LinAlg::Vector<double>& radius, const Core::FE::Discretization& discretization);

  /**
   * Computes and returns element length.
   */
  auto compute_length(const Core::Elements::Element& element);

  /**
   * Computes the matrix of shape functions according to SUPG.
   * The SUPG shape functions are defined as Psi = N + delta * H^T * dNdxi * dxidx.
   * Since the problem has two unknowns, A and u, a matrix of shape functions needs to be defined.
   *
   * The result is stored in @param Psi_matrix .
   * The normal shape functions N are passed in matrix form in @param N_matrix .
   * The associated derivative matrix in @param dNdxi_matrix .
   * The @param flux_jacobian is defined by H = [u A, c^2/ A u] and @param delta is defined by L /
   * (2*lambda_max).
   * The derivative dxidx is defined by 1 / @param L
   */
  auto compute_psi_matrix(Core::LinAlg::Matrix<2, 4>& Psi_matrix,
      const Core::LinAlg::Matrix<2, 4>& N_matrix, const Core::LinAlg::Matrix<2, 4>& dNdxi_matrix,
      const Core::LinAlg::Matrix<2, 2>& flux_jacobian, const double L, const double delta);
  /**
   * Get conditions for A and u when flow Q is prescribed. Computed through Newton-Raphson with
   * f = (W_in - W_out)^4 /1024 (rho/beta)^2 /2 * (W_in + W_out) .
   * W_out is known from the domain, W_in needs to be determined from prescribed Q.
   *   * Constants needed for computation are passed in @param input and the prescribed flow in
   *   @param Q_condition .
   *   * Known parameters at the node are @param boundary_A0  @param characteristic_W_outgoing
   * @param beta
   * The computed conditions for A and u are written to @param A_condition and @param u_condition
   */
  void conditions_from_newton_raphson(const Parameters& input, const double& Q_condition,
      const double& boundary_A0, const double& characteristic_W_outgoing, const double& beta,
      double& A_condition, double& u_condition);

  void main();


}  // namespace ReducedLung1dPipeFlow

FOUR_C_NAMESPACE_CLOSE

#endif
