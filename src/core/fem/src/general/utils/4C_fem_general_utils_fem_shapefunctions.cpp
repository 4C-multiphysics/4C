// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_fem_shapefunctions.hpp"

#include "4C_linalg_utils_densematrix_inverse.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim(
    Core::LinAlg::Matrix<probdim, Core::FE::num_nodes(distype)>& deriv_xyz,
    const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes(distype)>& deriv,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(distype), probdim>& xyze,
    const Core::LinAlg::Matrix<probdim, 1>& normal)
{
  const int nen = Core::FE::num_nodes(distype);
  const int nsd_ele = Core::FE::dim<distype>;

  FOUR_C_ASSERT(nsd_ele != probdim,
      "This method is designed to be used if the dimension of the element is smaller than the "
      "problem dimension");

  Core::LinAlg::Matrix<probdim, nen> deriv_full;

  // transform the derivatives and Jacobians to the higher dimensional coordinates (problem
  // dimension)
  Core::LinAlg::Matrix<nsd_ele, probdim> dx_dr_red;
  Core::LinAlg::Matrix<probdim, probdim> dx_dr, dr_dx;
  dx_dr_red.multiply_nn(deriv, xyze);

  for (unsigned i = 0; i < probdim; ++i)
  {
    for (unsigned j = 0; j < nsd_ele; ++j) dx_dr(j, i) = dx_dr_red(j, i);
    dx_dr(nsd_ele, i) = normal(i);
  }

  for (unsigned i = 0; i < nen; ++i)
  {
    for (unsigned j = 0; j < nsd_ele; ++j) deriv_full(j, i) = deriv(j, i);
    deriv_full(nsd_ele, i) = 0.0;
  }

  // special case: 1D element embedded in 3D problem
  if (nsd_ele == 1 and probdim == 3)
  {
    // compute second unit normal
    const double normalvec2_0 = dx_dr_red(0, 1) * normal(2) - normal(1) * dx_dr_red(0, 2);
    const double normalvec2_1 = dx_dr_red(0, 2) * normal(0) - normal(2) * dx_dr_red(0, 0);
    const double normalvec2_2 = dx_dr_red(0, 0) * normal(1) - normal(0) * dx_dr_red(0, 1);

    // norm
    const double norm2 = std::sqrt(
        normalvec2_0 * normalvec2_0 + normalvec2_1 * normalvec2_1 + normalvec2_2 * normalvec2_2);

    dx_dr(2, 0) = normalvec2_0 / norm2;
    dx_dr(2, 1) = normalvec2_1 / norm2;
    dx_dr(2, 2) = normalvec2_2 / norm2;

    for (unsigned i = 0; i < nen; i++) deriv_full(2, i) = 0.0;
  }

  Core::LinAlg::invert(dr_dx, dx_dr);

  // compute global spatial derivatives
  deriv_xyz.multiply(dr_dx, deriv_full);
}

template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::line2, 2>(
    Core::LinAlg::Matrix<2, 2>& deriv_xyz, const Core::LinAlg::Matrix<1, 2>& deriv,
    const Core::LinAlg::Matrix<2, 2>& xyze, const Core::LinAlg::Matrix<2, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::line2, 3>(
    Core::LinAlg::Matrix<3, 2>& deriv_xyz, const Core::LinAlg::Matrix<1, 2>& deriv,
    const Core::LinAlg::Matrix<2, 3>& xyze, const Core::LinAlg::Matrix<3, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::line3, 2>(
    Core::LinAlg::Matrix<2, 3>& deriv_xyz, const Core::LinAlg::Matrix<1, 3>& deriv,
    const Core::LinAlg::Matrix<3, 2>& xyze, const Core::LinAlg::Matrix<2, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::nurbs3, 2>(
    Core::LinAlg::Matrix<2, 3>& deriv_xyz, const Core::LinAlg::Matrix<1, 3>& deriv,
    const Core::LinAlg::Matrix<3, 2>& xyze, const Core::LinAlg::Matrix<2, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::nurbs9, 3>(
    Core::LinAlg::Matrix<3, 9>& deriv_xyz, const Core::LinAlg::Matrix<2, 9>& deriv,
    const Core::LinAlg::Matrix<9, 3>& xyze, const Core::LinAlg::Matrix<3, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::quad4, 3>(
    Core::LinAlg::Matrix<3, 4>& deriv_xyz, const Core::LinAlg::Matrix<2, 4>& deriv,
    const Core::LinAlg::Matrix<4, 3>& xyze, const Core::LinAlg::Matrix<3, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::quad8, 3>(
    Core::LinAlg::Matrix<3, 8>& deriv_xyz, const Core::LinAlg::Matrix<2, 8>& deriv,
    const Core::LinAlg::Matrix<8, 3>& xyze, const Core::LinAlg::Matrix<3, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::quad9, 3>(
    Core::LinAlg::Matrix<3, 9>& deriv_xyz, const Core::LinAlg::Matrix<2, 9>& deriv,
    const Core::LinAlg::Matrix<9, 3>& xyze, const Core::LinAlg::Matrix<3, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::tri3, 3>(
    Core::LinAlg::Matrix<3, 3>& deriv_xyz, const Core::LinAlg::Matrix<2, 3>& deriv,
    const Core::LinAlg::Matrix<3, 3>& xyze, const Core::LinAlg::Matrix<3, 1>& normal);
template void
Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<Core::FE::CellType::tri6, 3>(
    Core::LinAlg::Matrix<3, 6>& deriv_xyz, const Core::LinAlg::Matrix<2, 6>& deriv,
    const Core::LinAlg::Matrix<6, 3>& xyze, const Core::LinAlg::Matrix<3, 1>& normal);
FOUR_C_NAMESPACE_CLOSE
