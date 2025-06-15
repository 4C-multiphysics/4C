// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_utlis_quaternion_interpolation.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <cassert>

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<4, 1> Core::LinAlg::spherical_linear_interpolation(
    const Core::LinAlg::Matrix<4, 1>& q1, const Core::LinAlg::Matrix<4, 1>& q2, double t)
{
  // Compute the dot product
  double dot = q1.dot(q2);

  // If the dot product is negative, slerp won't take the shorter path.
  // Fix by reversing one quaternion.
  Core::LinAlg::Matrix<4, 1> q2_mod = q2;
  if (dot < 0.0)
  {
    for (int i = 0; i < 4; ++i) q2_mod(i, 0) = -q2_mod(i, 0);
    dot = -dot;
  }

  const double linear_interpolation_threshold = 0.9995;
  if (dot > linear_interpolation_threshold)
  {
    // If the quaternions are close, use linear interpolation
    Core::LinAlg::Matrix<4, 1> result;
    for (int i = 0; i < 4; ++i) result(i, 0) = (1.0 - t) * q1(i, 0) + t * q2_mod(i, 0);
    // Normalize
    result.normalize();
    return result;
  }

  double theta_0 = std::acos(dot);  // angle between input quaternions
  double sin_theta_0 = std::sin(theta_0);

  double s0 = std::sin((1.0 - t) * theta_0) / sin_theta_0;
  double s1 = std::sin(t * theta_0) / sin_theta_0;

  Core::LinAlg::Matrix<4, 1> result;
  for (int i = 0; i < 4; ++i) result(i, 0) = s0 * q1(i, 0) + s1 * q2_mod(i, 0);

  // Normalize
  result.normalize();

  return result;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::
    GeneralizedSphericalLinearInterpolator(
        const std::vector<Core::LinAlg::Matrix<4, 1>>& unit_quaternions,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::ScalarInterpolation::WeightingFunction weight_func,
        const Core::LinAlg::ScalarInterpolation::InterpParams& interp_params)
    : unit_quaternions_(unit_quaternions),
      ref_locs_(ref_locs),
      weight_func_(weight_func),
      interp_params_(interp_params)
{
  // Check if reference locations same dimension
  FOUR_C_ASSERT_ALWAYS(!ref_locs.empty(),
      "Reference locations must have the same dimension as the interpolation location.");

  // Check if the number of reference locations matches the number of quaternions
  FOUR_C_ASSERT_ALWAYS(unit_quaternions_.size() == ref_locs.size(),
      "Number of quaternions must match number of reference locations.");

  // Check if all quaternions are unit quaternions
  FOUR_C_ASSERT_ALWAYS(std::all_of(unit_quaternions_.begin(), unit_quaternions_.end(),
                           [](const auto& quat) { return std::abs(quat.norm2() - 1.0) < 1e-6; }),
      "All quaternions must be normalized (norm should be close to 1).");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::
    GeneralizedSphericalLinearInterpolator(
        const std::vector<Core::LinAlg::Matrix<4, 1>>& unit_quaternions,
        const std::vector<double>& normalized_weights)
    : unit_quaternions_(unit_quaternions), normalized_weights_(normalized_weights)
{
  FOUR_C_ASSERT_ALWAYS(unit_quaternions_.size() == normalized_weights_.size(),
      "Number of quaternions and weights must match.");
  FOUR_C_ASSERT_ALWAYS(!unit_quaternions_.empty(), "No quaternions provided for interpolation.");
  FOUR_C_ASSERT_ALWAYS(!normalized_weights_.empty(), "No weights provided for interpolation.");

  // Check if all quaternions are normalized
  FOUR_C_ASSERT_ALWAYS(std::all_of(unit_quaternions_.begin(), unit_quaternions_.end(),
                           [](const auto& quat) { return std::abs(quat.norm2() - 1.0) < 1e-6; }),
      "All quaternions must be normalized (norm should be close to 1).");

  // Check if weights are non-negative and sum to 1
  double weight_sum = 0.0;
  for (const auto& weight : normalized_weights_)
  {
    if (weight < 0.0)
      FOUR_C_THROW("All weights must be non-negative.");
    else
      weight_sum += weight;
  }
  if (std::abs(weight_sum - 1.0) > 1e-6) FOUR_C_THROW("Weights must sum to 1.");
}

/*--------------------------------------------------------------------*
 * @brief Computes the spherical weighted average of points on the unit sphere $\mathbb{S}^3$ using
 * Algorithm A2 from Buss and Fillmore, 2001.
 *
 * @details
 * **Inputs:** Points $q_1, \ldots, q_n$ on $S^3$ and nonnegative weights $w_1, \dots, w_n$ with
 * $\sum_{i=1}^n w_i = 1$.
 *
 * **Initialization:**
 * Set $q$ to the normalized weighted sum: $q := \frac{\sum_{i=1}^n w_i q_i}{\left\| \sum_{i=1}^n
 * w_i q_i \right\|}$.
 *
 * **Main Iteration:**
 * - Construct a local Euclidean coordinate system $C$ at the current $q$.
 * - For each $i = 1, \ldots, n$:
 *   - Compute $q_i^* := \log_q(q_i)$ (the logarithmic map at $q$ of $q_i$).
 *   - Let $\rho_i := \| q_i^* - q \|$.
 *   - Define a local coordinate system $C_i$ at $q$ with the first axis pointing away from $q_i$.
 *   - Let $M_i$ be the transformation matrix from $C_i$ to $C$.
 *   - Let $H_i$ be a $d \times d$ diagonal matrix with first entry 1, others $\rho_i \cot(\rho_i)$.
 * - Compute:
 *   - $u := \sum_{i=1}^n w_i (q_i^* - q)$
 *   - $H := \sum_{i=1}^n M_i H_i M_i^T$
 *   - $v := H^{-1} u$
 *   - Update $q := \exp_q(q + v)$ (using the exponential map).
 * - Repeat until $\|v\|$ is sufficiently small, then return $q$.
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
Core::LinAlg::Matrix<4, 1>
Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::get_interpolated_quaternion(
    Core::LinAlg::Matrix<loc_dim, 1>* interp_loc)
{
  if (interp_loc != nullptr and normalized_weights_.empty())
  {
    normalized_weights_ = Core::LinAlg::ScalarInterpolation::calculate_normalized_weights(
        ref_locs_, *interp_loc, weight_func_, interp_params_);
  }

  FOUR_C_ASSERT_ALWAYS(
      !normalized_weights_.empty(), "No weights calculated for the given interpolation location.");
  FOUR_C_ASSERT_ALWAYS(normalized_weights_.size() == unit_quaternions_.size(),
      "Number of weights must match number of quaternions.");

  // compute initial estimate
  Core::LinAlg::Matrix<4, 1> q_interp(initial_estimate());

  if (q_interp(3) < 0.) q_interp.scale(-1.);
  q_interp.normalize();

  // Basis and local coordinate system
  Core::LinAlg::Matrix<4, 4> basis4x4(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> e1(Core::LinAlg::Initialization::zero),
      e2(Core::LinAlg::Initialization::zero), e3(Core::LinAlg::Initialization::zero);

  // Gradient and auxiliary vectors
  Core::LinAlg::Matrix<3, 1> grad_local(Core::LinAlg::Initialization::zero),
      grad(Core::LinAlg::Initialization::zero), clocgardy(Core::LinAlg::Initialization::zero),
      clocgardz(Core::LinAlg::Initialization::zero);

  // Rotation and Hessian matrices
  Core::LinAlg::Matrix<3, 3> rotmat(Core::LinAlg::Initialization::zero),
      hessian_local(Core::LinAlg::Initialization::zero),
      hessian(Core::LinAlg::Initialization::zero),
      inverse_hessian(Core::LinAlg::Initialization::zero);

  // Loop variables for quaternion operations
  Core::LinAlg::Matrix<4, 1> qi(Core::LinAlg::Initialization::zero),
      q_interptmp(Core::LinAlg::Initialization::zero), x_disp(Core::LinAlg::Initialization::zero),
      q_interpold(Core::LinAlg::Initialization::zero),
      dq_interp(Core::LinAlg::Initialization::zero), tmp4x1(Core::LinAlg::Initialization::zero);

  // Temporary vectors and matrices
  Core::LinAlg::Matrix<3, 1> tmp3x1(Core::LinAlg::Initialization::zero),
      x_disp_local(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> tmp3x3(Core::LinAlg::Initialization::zero);

  int iteration = 0;
  while (1)
  {
    basis4x4 = gram_schmidt_orthonormal_basis(q_interp);  //   4x4 Gram-Schmidt Orthonormalization

    for (int i = 0; i < 4; ++i)
    {
      e1(i) = basis4x4(i, 1);
      e2(i) = basis4x4(i, 2);
      e3(i) = basis4x4(i, 3);
    }
    e1.normalize();
    e2.normalize();
    e3.normalize();

    grad.put_scalar(0.);
    hessian.put_scalar(0.);

    double tt = 0.;
    // Next approximation
    for (size_t i = 0; i < unit_quaternions_.size(); ++i)  // Loop over all quaternion
    {
      double weight = normalized_weights_[i];
      qi.put_scalar(0.);
      qi.update(unit_quaternions_[i]);

      q_interptmp.put_scalar(0.);
      q_interptmp.update(project_quaternion(qi, q_interp));

      double sin_theta = q_interptmp.norm2();
      double cos_theta = q_interp.dot(qi);
      double theta = atan2(sin_theta, cos_theta);

      hessian_local.put_scalar(0.);
      grad_local.put_scalar(0.);

      if (sin_theta == 0.)
      {
        for (int i = 0; i < 3; ++i) hessian(i, i) = weight;
      }
      else
      {
        q_interptmp.scale(1. / sin_theta);

        grad_local(0) = q_interptmp.dot(e1);
        grad_local(1) = q_interptmp.dot(e2);
        grad_local(2) = q_interptmp.dot(e3);

        grad.update(weight * theta, grad_local, 1.);

        right_orthonormal_basis(grad_local, clocgardy, clocgardz);

        tmp3x1.put_scalar(0.);
        tmp3x1.update(grad_local);
        tmp3x1.normalize();

        rotmat.put_scalar(0.);
        for (int i = 0; i < 3; ++i)
        {
          rotmat(0, i) = tmp3x1(i);
          rotmat(1, i) = clocgardy(i);
          rotmat(2, i) = clocgardz(i);
        }

        hessian_local.update_t(1.0, rotmat, 0.);

        tt = weight * theta * cos_theta / sin_theta;

        hessian_local(0, 0) *= weight;
        hessian_local(1, 0) *= weight;
        hessian_local(2, 0) *= weight;

        hessian_local(0, 1) *= tt;
        hessian_local(1, 1) *= tt;
        hessian_local(2, 1) *= tt;

        hessian_local(0, 2) *= tt;
        hessian_local(1, 2) *= tt;
        hessian_local(2, 2) *= tt;

        tmp3x3.put_scalar(0.);
        tmp3x3.multiply(hessian_local, rotmat);

        hessian.update(1., tmp3x3, 1.);
      }
    }
    inverse_hessian.put_scalar(0.);
    inverse_hessian.invert(hessian);

    x_disp_local.put_scalar(0.);
    x_disp_local.multiply(inverse_hessian, grad);  // Solve

    x_disp.put_scalar(0.);
    x_disp.update(x_disp_local(0), e1, 0.);
    x_disp.update(x_disp_local(1), e2, 1.);
    x_disp.update(x_disp_local(2), e3, 1.);

    q_interpold.put_scalar(0.);
    q_interpold.update(q_interp);

    rotate_quaternion(q_interp, x_disp);
    q_interp.normalize();

    // Convergence check
    dq_interp.put_scalar(0.);
    dq_interp.put_scalar(0.);
    tmp4x1.update(q_interp);
    tmp4x1.update(q_interp);
    tmp4x1.update(-1., q_interpold, 1.);

    dq_interp.abs(tmp4x1);
    double error = dq_interp.max_value();

    if (error <= 1e-8) break;  // Convergence criterion

    if (iteration > 500)
      FOUR_C_THROW("GeneralizedSphericalLinearInterpolator: Unconverged after 500 iterations");

    ++iteration;
  }

  return q_interp;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
Core::LinAlg::Matrix<4, 1>
Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::weighted_sum(
    const std::vector<Core::LinAlg::Matrix<4, 1>>& quaternions, const std::vector<double>& weights)
{
  FOUR_C_ASSERT_ALWAYS(!quaternions.empty(), "No quaternions provided for weighted sum.");
  FOUR_C_ASSERT_ALWAYS(!weights.empty(), "No weights provided for weighted sum.");
  FOUR_C_ASSERT_ALWAYS(
      quaternions.size() == weights.size(), "Number of quaternions and weights must match.");

  Core::LinAlg::Matrix<4, 1> weighted_sum(Core::LinAlg::Initialization::zero);

  for (size_t i = 0; i < weights.size(); ++i)
  {
    if (weights[i] < 0) FOUR_C_THROW("Weights must be non-negative.");
    weighted_sum.update(weights[i], quaternions[i], 1.0);
  }
  if (weighted_sum.norm2() < 1e-8)
    FOUR_C_THROW("Weighted sum of quaternions is too close to zero, cannot normalize.");
  weighted_sum.normalize();
  return weighted_sum;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
Core::LinAlg::Matrix<4, 1>
Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::initial_estimate()
{
  // Find the coordinate with the max. total absolute value.
  double abs_sum_x = 0, abs_sum_y = 0, abs_sum_z = 0, abs_sum_w = 0;
  for (const auto& quaternion : unit_quaternions_)
  {
    abs_sum_x += std::abs(quaternion(0));
    abs_sum_y += std::abs(quaternion(1));
    abs_sum_z += std::abs(quaternion(2));
    abs_sum_w += std::abs(quaternion(3));
  }

  Core::LinAlg::Matrix<4, 1> q_initial(Core::LinAlg::Initialization::zero);
  if (abs_sum_x > abs_sum_y)
  {
    if (abs_sum_x > abs_sum_z)
    {
      if (abs_sum_x > abs_sum_w)
      {
        q_initial(0) = 1.0;
      }
      else
      {
        q_initial(3) = 1.0;
      }
    }
    else if (abs_sum_z > abs_sum_w)
    {
      q_initial(2) = 1.0;
    }
    else
    {
      q_initial(3) = 1.0;
    }
  }
  else if (abs_sum_y > abs_sum_z)
  {
    if (abs_sum_y > abs_sum_w)
    {
      q_initial(1) = 1.0;
    }
    else
    {
      q_initial(3) = 1.0;
    }
  }
  else if (abs_sum_z > abs_sum_w)
  {
    q_initial(2) = 1.0;
  }
  else
  {
    q_initial(3) = 1.0;
  }

  std::vector<Core::LinAlg::Matrix<4, 1>> unit_quaternions_tmp(unit_quaternions_);

  for (size_t i = 0; i < normalized_weights_.size(); i++)
    if (q_initial.dot(unit_quaternions_tmp[i]) < 0.0) unit_quaternions_tmp[i].scale(-1.0);

  return weighted_sum(unit_quaternions_tmp, normalized_weights_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
Core::LinAlg::Matrix<4, 1>
Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::project_quaternion(
    const Core::LinAlg::Matrix<4, 1>& quaternion_1, const Core::LinAlg::Matrix<4, 1>& quaternion_2)
{
  Core::LinAlg::Matrix<4, 1> projection(quaternion_1);
  projection.update(-1., quaternion_2, 1.);

  double dotproduct = projection.dot(quaternion_2);
  projection.update(-dotproduct, quaternion_2, 1.);  // (q1-q2) - [(q1-q2).q2] q2

  return projection;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
void Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::rotate_quaternion(
    Core::LinAlg::Matrix<4, 1>& quaternion, const Core::LinAlg::Matrix<4, 1>& direction)
{
  FOUR_C_ASSERT_ALWAYS(quaternion.norm2() < 1.0 + 1e-8 && quaternion.norm2() > 1.0 - 1e-8 &&
                           (direction.dot(quaternion)) < 1e-8 &&
                           (direction.dot(quaternion)) > -1e-8,
      "ERROR: check input quaternion and direction for validity.");

  double thetasq = direction.norm2() * direction.norm2();

  if (thetasq == 0.)
    return;
  else
  {
    double theta = direction.norm2();
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    Core::LinAlg::Matrix<4, 1> unitdir(direction);
    unitdir.scale(1.0 / theta);

    Core::LinAlg::Matrix<4, 1> tmp(quaternion);
    tmp.scale(cos_theta);
    tmp.update(sin_theta, unitdir, 1.);

    quaternion.clear();
    quaternion.update(tmp);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
Core::LinAlg::Matrix<4, 4>
Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::gram_schmidt_orthonormal_basis(
    const Core::LinAlg::Matrix<4, 1>& quaternion)
{
  Core::LinAlg::Matrix<4, 1> quaternion_copy(quaternion);

  Core::LinAlg::Matrix<4, 4> gs_basis(Core::LinAlg::Initialization::zero);
  quaternion_copy.normalize();

  // set first column
  for (int i = 0; i < 4; ++i) gs_basis(i, 0) = quaternion_copy(i);

  // set second column
  gs_basis(0, 1) = -1. * gs_basis(1, 0);
  gs_basis(1, 1) = gs_basis(0, 0);
  gs_basis(2, 1) = -1. * gs_basis(3, 0);
  gs_basis(3, 1) = gs_basis(2, 0);

  // set third column
  double d12 = gs_basis(0, 0) * gs_basis(1, 1) - gs_basis(0, 1) * gs_basis(1, 0);
  double d13 = gs_basis(0, 0) * gs_basis(2, 1) - gs_basis(0, 1) * gs_basis(2, 0);
  double d14 = gs_basis(0, 0) * gs_basis(3, 1) - gs_basis(0, 1) * gs_basis(3, 0);

  double d23 = gs_basis(1, 0) * gs_basis(2, 1) - gs_basis(1, 1) * gs_basis(2, 0);
  double d24 = gs_basis(1, 0) * gs_basis(3, 1) - gs_basis(1, 1) * gs_basis(3, 0);
  double d34 = gs_basis(2, 0) * gs_basis(3, 1) - gs_basis(2, 1) * gs_basis(3, 0);

  Core::LinAlg::Matrix<4, 1> basis_3(Core::LinAlg::Initialization::zero);

  if (d12 > 0.4 || d12 < -0.4 || d13 > 0.4 || d13 < -0.4 || d23 > 0.4 || d23 < -0.4)
  {
    basis_3(0) = d23;
    basis_3(1) = -d13;
    basis_3(2) = d12;
    basis_3(3) = 0.0;
  }
  else if (d24 > 0.4 || d24 < -0.4 || d14 > 0.4 || d14 < -0.4)
  {
    basis_3(0) = d24;
    basis_3(1) = -d14;
    basis_3(2) = 0.0;
    basis_3(3) = d12;
  }
  else
  {
    basis_3(0) = d34;
    basis_3(1) = 0.0;
    basis_3(2) = -d14;
    basis_3(3) = d13;
  }

  basis_3.normalize();

  for (int i = 0; i < 4; ++i) gs_basis(i, 2) = basis_3(i);

  // set 4th column
  gs_basis(0, 3) = -gs_basis(1, 2) * d34 + gs_basis(2, 2) * d24 - gs_basis(3, 2) * d23;
  gs_basis(1, 3) = gs_basis(0, 2) * d34 - gs_basis(2, 2) * d14 + gs_basis(3, 2) * d13;
  gs_basis(2, 3) = -gs_basis(0, 2) * d24 + gs_basis(1, 2) * d14 - gs_basis(3, 2) * d12;
  gs_basis(3, 3) = gs_basis(0, 2) * d23 - gs_basis(1, 2) * d13 + gs_basis(2, 2) * d12;

  assert(gs_basis.determinant() < 1.0 - 1e-8 && gs_basis.determinant() > 1.0 + 1e-8);

  return gs_basis;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
template <unsigned int loc_dim>
void Core::LinAlg::GeneralizedSphericalLinearInterpolator<loc_dim>::right_orthonormal_basis(
    Core::LinAlg::Matrix<3, 1>& u, Core::LinAlg::Matrix<3, 1>& v, Core::LinAlg::Matrix<3, 1>& w)
{
  v.clear();
  w.clear();
  v.put_scalar(0.);
  w.put_scalar(0.);

  if (u(0) > 0.5 or u(0) < -0.5 or u(1) > 0.5 or u(1) < -0.5)
  {
    v(0) = u(1);
    v(1) = -u(0);
    v(2) = 0.;
  }
  else
  {
    v(0) = 0.;
    v(1) = u(2);
    v(2) = -u(1);
  }
  v.normalize();
  w.cross_product(u, v);
  w.normalize();
}

// Explicit instantiations
template class Core::LinAlg::GeneralizedSphericalLinearInterpolator<1>;
template class Core::LinAlg::GeneralizedSphericalLinearInterpolator<2>;
template class Core::LinAlg::GeneralizedSphericalLinearInterpolator<3>;
FOUR_C_NAMESPACE_CLOSE