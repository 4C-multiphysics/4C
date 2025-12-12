// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL7P_ELE_CALC_EAS_UTILS_HPP
#define FOUR_C_SHELL7P_ELE_CALC_EAS_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_shell7p_ele_calc_lib.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements::Shell
{

  /*!
   *@brief A struct holding the iteration data used within the Enhanced Assumed Strain element
   * technology
   */
  struct EASIterationData
  {
    // EAS portion of internal forces Rtilde (dimension neas x 1)
    Core::LinAlg::SerialDenseMatrix RTilde_;
    // inverse EAS matrix Dtilde (dimension neas x neas)
    Core::LinAlg::SerialDenseMatrix invDTilde_;
    // transposed EAS matrix  L^T (dimension neas x numdofperelement)
    Core::LinAlg::SerialDenseMatrix transL_;
    // alpha (dimension neas x 1)
    Core::LinAlg::SerialDenseMatrix alpha_;
  };

  /*!
   * @brief A struct holding the enhanced EAS kinematic quantities.
   */
  struct EASKinematics
  {
    // Total (enhanced) gl strain tensor
    Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim> enhanced_gl_strain_;

    // Total (enhanced) deformation gradient for material evaluation
    Core::LinAlg::Tensor<double, Internal::num_dim, Internal::num_dim> enhanced_defgrd_;

    // EAS shape function matrix in material configuration (12 x neas)
    Core::LinAlg::SerialDenseMatrix M;
  };

  /*!
   * @brief Evaluates the transformation matrix T0^{-1} which maps the M-matrix from centroid point
   * (parameter space) to the gaussian point (material configuration)
   * @tparam distype
   * @param a_cov (in) : Jacobian mapping evaluated at the gaussina point
   * @param a_con_centroid (in) : Jacobian mapping evaluated at the element centroid
   *
   * @return Core::LinAlg::SerialDenseMatrix : Transformation matrix T0inv
   */
  template <Core::FE::CellType distype>
  Core::LinAlg::SerialDenseMatrix evaluate_t0inv(
      const Core::LinAlg::Matrix<Discret::Elements::Shell::Internal::num_dim,
          Discret::Elements::Shell::Internal::num_dim>& a_cov,
      const Core::LinAlg::Matrix<Discret::Elements::Shell::Internal::num_dim,
          Discret::Elements::Shell::Internal::num_dim>& a_con_centroid)
  {
    Core::LinAlg::SerialDenseMatrix T0inv(
        Discret::Elements::Shell::Internal::num_internal_variables,
        Discret::Elements::Shell::Internal::num_internal_variables);
    double t11, t12, t13, t21, t22, t23, t31, t32, t33;
    // components of the transformation matrix T0^{-1}
    t11 = 0.0;
    t12 = 0.0;
    t13 = 0.0;
    t21 = 0.0;
    t22 = 0.0;
    t23 = 0.0;
    t31 = 0.0;
    t32 = 0.0;
    t33 = 1.0;
    for (int i = 0; i < Discret::Elements::Shell::Internal::num_dim; ++i)
    {
      t11 += a_cov(0, i) * a_con_centroid(0, i);
      t12 += a_cov(0, i) * a_con_centroid(1, i);
      t21 += a_cov(1, i) * a_con_centroid(0, i);
      t22 += a_cov(1, i) * a_con_centroid(1, i);
    }

    T0inv(0, 0) = t11 * t11;
    T0inv(0, 1) = t12 * t12;
    T0inv(0, 2) = t13 * t13;
    T0inv(0, 3) = t11 * t12;
    T0inv(0, 4) = t12 * t13;
    T0inv(0, 5) = t11 * t13;

    T0inv(1, 0) = t21 * t21;
    T0inv(1, 1) = t22 * t22;
    T0inv(1, 2) = t23 * t23;
    T0inv(1, 3) = t21 * t22;
    T0inv(1, 4) = t22 * t23;
    T0inv(1, 5) = t21 * t23;

    T0inv(2, 0) = t31 * t31;
    T0inv(2, 1) = t32 * t32;
    T0inv(2, 2) = t33 * t33;
    T0inv(2, 3) = t31 * t32;
    T0inv(2, 4) = t32 * t33;
    T0inv(2, 5) = t31 * t33;

    T0inv(3, 0) = 2 * t11 * t21;
    T0inv(3, 1) = 2 * t12 * t22;
    T0inv(3, 2) = 2 * t13 * t23;
    T0inv(3, 3) = (t11 * t22 + t21 * t12);
    T0inv(3, 4) = (t12 * t23 + t22 * t13);
    T0inv(3, 5) = (t11 * t23 + t21 * t13);

    T0inv(4, 0) = 2 * t21 * t31;
    T0inv(4, 1) = 2 * t22 * t32;
    T0inv(4, 2) = 2 * t23 * t33;
    T0inv(4, 3) = (t21 * t32 + t31 * t22);
    T0inv(4, 4) = (t22 * t33 + t32 * t23);
    T0inv(4, 5) = (t21 * t33 + t31 * t23);

    T0inv(5, 0) = 2 * t11 * t31;
    T0inv(5, 1) = 2 * t12 * t32;
    T0inv(5, 2) = 2 * t13 * t33;
    T0inv(5, 3) = (t11 * t32 + t31 * t12);
    T0inv(5, 4) = (t12 * t33 + t32 * t13);
    T0inv(5, 5) = (t11 * t33 + t31 * t13);

    T0inv(6, 6) = t11 * t11;
    T0inv(6, 7) = t12 * t12;
    T0inv(6, 8) = t13 * t13;
    T0inv(6, 9) = t11 * t12;
    T0inv(6, 10) = t12 * t13;
    T0inv(6, 11) = t11 * t13;

    T0inv(7, 6) = t21 * t21;
    T0inv(7, 7) = t22 * t22;
    T0inv(7, 8) = t23 * t23;
    T0inv(7, 9) = t21 * t22;
    T0inv(7, 10) = t22 * t23;
    T0inv(7, 11) = t21 * t23;

    T0inv(8, 6) = t31 * t31;
    T0inv(8, 7) = t32 * t32;
    T0inv(8, 8) = t33 * t33;
    T0inv(8, 9) = t31 * t32;
    T0inv(8, 10) = t32 * t33;
    T0inv(8, 11) = t31 * t33;

    T0inv(9, 6) = 2 * t11 * t21;
    T0inv(9, 7) = 2 * t12 * t22;
    T0inv(9, 8) = 2 * t13 * t23;
    T0inv(9, 9) = (t11 * t22 + t21 * t12);
    T0inv(9, 10) = (t12 * t23 + t22 * t13);
    T0inv(9, 11) = (t11 * t23 + t21 * t13);

    T0inv(10, 6) = 2 * t21 * t31;
    T0inv(10, 7) = 2 * t22 * t32;
    T0inv(10, 8) = 2 * t23 * t33;
    T0inv(10, 9) = (t21 * t32 + t31 * t22);
    T0inv(10, 10) = (t22 * t33 + t32 * t23);
    T0inv(10, 11) = (t21 * t33 + t31 * t23);

    T0inv(11, 6) = 2 * t11 * t31;
    T0inv(11, 7) = 2 * t12 * t32;
    T0inv(11, 8) = 2 * t13 * t33;
    T0inv(11, 9) = (t11 * t32 + t31 * t12);
    T0inv(11, 10) = (t12 * t33 + t32 * t13);
    T0inv(11, 11) = (t11 * t33 + t31 * t13);

    for (int i = 0; i < Discret::Elements::Shell::Internal::node_dof; ++i)
    {
      for (int j = 0; j < Discret::Elements::Shell::Internal::node_dof; ++j)
      {
        T0inv(i, j + Discret::Elements::Shell::Internal::node_dof) = 0.0;
        T0inv(i + Discret::Elements::Shell::Internal::node_dof, j) = 0.0;
      }
    }
    return T0inv;
  }

  /*!
   * @brief Compute the matrix M that is the element-wise matrix of the shape functions for the
   * enhanced strains in the parameter space
   *
   * @tparam distype
   * @param xi_gp (in) : Coordinate in the parameter space
   * @param eas (in) : Number of eas parameters for each locking part
   * @param neas (in) : Total number of eas parameters
   * @return Core::LinAlg::SerialDenseMatrix : Enhanced strains shape function matrix M in parameter
   * space
   */
  template <Core::FE::CellType distype>
  Core::LinAlg::SerialDenseMatrix evaluate_eas_shape_functions_parameter_space(
      const std::array<double, 2>& xi_gp, const Solid::Elements::ShellLockingTypes& locking_types)
  {
    // evaluation of the shape function matrix to interpolate the enhanced strains alpha
    Core::LinAlg::SerialDenseMatrix M(
        Discret::Elements::Shell::Internal::num_internal_variables, locking_types.total);

    const double xi = xi_gp[0];
    const double eta = xi_gp[1];
    int M_index = 0;

    const double xieta = xi * eta;
    const double xixi = xi * xi;
    const double etaeta = eta * eta;
    const double xixieta = xi * xieta;
    const double xietaeta = xieta * eta;
    const double xixietaeta = xieta * xieta;

    const int num_node = Discret::Elements::Shell::Internal::num_node<distype>;
    if (num_node > 4)
    {
      // membrane locking: E_{11}, E_{12}, E_{22} constant
      switch (locking_types.membrane)
      {
        case 0:
          break;
        case 7:
          M(0, M_index) = eta - 3.0 * xixieta;
          M(0, M_index + 1) = etaeta - 3.0 * xixietaeta;
          M(1, M_index + 2) = xi - 3.0 * xietaeta;
          M(1, M_index + 3) = xixi - 3.0 * xixietaeta;
          M(3, M_index + 4) = eta - 3.0 * xixieta;
          M(3, M_index + 5) = xi - 3.0 * xietaeta;
          M(3, M_index + 6) = 1.0 - 3.0 * (xixi + etaeta) + 9.0 * xixietaeta;
          M_index += 7;
          break;
        case 9:
          M(0, M_index) = 1.0 - 3.0 * xixi;
          M(0, M_index + 1) = eta - 3.0 * xixieta;
          M(1, M_index + 2) = 1.0 - 3.0 * etaeta;
          M(1, M_index + 3) = xi - 3.0 * xietaeta;
          M(3, M_index + 4) = 1.0 - 3.0 * xixieta;
          M(3, M_index + 5) = 1.0 - 3.0 * xietaeta;
          M(3, M_index + 6) = eta - 3.0 * xixieta;
          M(3, M_index + 7) = xi - 3.0 * xietaeta;
          M(3, M_index + 8) = 1.0 - 3.0 * (xixi + etaeta) + 9.0 * xixietaeta;
          M_index += 9;
          break;
        case 11:
          M(0, M_index) = 1.0 - 3.0 * xixi;
          M(0, M_index + 1) = eta - 3.0 * xixieta;
          M(0, M_index + 2) = etaeta - 3.0 * xixietaeta;
          M(1, M_index + 3) = 1.0 - 3.0 * etaeta;
          M(1, M_index + 4) = xi - 3.0 * xietaeta;
          M(1, M_index + 5) = xixi - 3.0 * xixietaeta;
          M(3, M_index + 6) = 1.0 - 3.0 * xixi;
          M(3, M_index + 7) = 1.0 - 3.0 * etaeta;
          M(3, M_index + 8) = eta - 3.0 * xixieta;
          M(3, M_index + 9) = xi - 3.0 * xietaeta;
          M(3, M_index + 10) = 1.0 - 3.0 * (xixi + etaeta) + 9.0 * xixietaeta;
          M_index += 11;
          break;
        default:
          FOUR_C_THROW(
              "EAS Membrane locking: Only 0, 7, 9, 11 EAS modes are implemented. Given: "
              "{}",
              locking_types.membrane);
      }
      // bending locking: E_{11}, E_{12], E_{22} linear
      switch (locking_types.bending)
      {
        case 0:
          break;
        case 9:
          M(6, M_index) = 1.0 - 3.0 * xixi;
          M(6, M_index + 1) = eta - 3.0 * xixieta;
          M(7, M_index + 2) = 1.0 - 3.0 * etaeta;
          M(7, M_index + 3) = xi - 3.0 * xietaeta;
          M(9, M_index + 4) = 1.0 - 3.0 * xixi;
          M(9, M_index + 5) = 1.0 - 3.0 * etaeta;
          M(9, M_index + 6) = eta - 3.0 * xixieta;
          M(9, M_index + 7) = xi - 3.0 * xietaeta;
          M(9, M_index + 8) = 1.0 - 3.0 * (xixi + etaeta) + 9.0 * xixietaeta;
          M_index += 9;
          break;
        case 11:
          M(6, M_index) = 1.0 - 3.0 * xixi;
          M(6, M_index + 1) = eta - 3.0 * xixieta;
          M(6, M_index + 2) = etaeta - 3.0 * xixietaeta;
          M(7, M_index + 3) = 1.0 - 3.0 * etaeta;
          M(7, M_index + 4) = xi - 3.0 * xietaeta;
          M(7, M_index + 5) = xixi - 3.0 * xixietaeta;
          M(9, M_index + 6) = 1.0 - 3.0 * xixi;
          M(9, M_index + 7) = 1.0 - 3.0 * etaeta;
          M(9, M_index + 8) = eta - 3.0 * xixieta;
          M(9, M_index + 9) = xi - 3.0 * xietaeta;
          M(9, M_index + 10) = 1.0 - 3.0 * (xixi + etaeta) + 9.0 * xixietaeta;
          M_index += 11;
          break;
        default:
          FOUR_C_THROW("EAS bending part: Only 0, 9, 11 EAS modes are implemented. Given: {}",
              locking_types.bending);
      }
      // locking due to thickness changes E_{33} linear
      switch (locking_types.thickness)
      {
        case 0:
          break;
        case 1:
          M(8, M_index) = 1.0;
          M_index += 1;
          break;
        case 3:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M_index += 3;
          break;
        case 4:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M_index += 4;
          break;
        case 6:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M(8, M_index + 4) = xixi;
          M(8, M_index + 5) = etaeta;
          M_index += 6;
          break;
        case 8:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M(8, M_index + 4) = xixi;
          M(8, M_index + 5) = etaeta;
          M(8, M_index + 6) = xixieta;
          M(8, M_index + 7) = xietaeta;
          M_index += 8;
          break;
        case 9:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M(8, M_index + 4) = 1.0 - 3.0 * xixi;
          M(8, M_index + 5) = 1.0 - 3.0 * etaeta;
          M(8, M_index + 6) = xixieta;
          M(8, M_index + 7) = xietaeta;
          M(8, M_index + 8) = 1.0 - 9.0 * xixietaeta;
          M_index += 9;
          break;
        default:
          FOUR_C_THROW(
              "EAS thickness locking: Only 0, 1, 3, 4, 5, 8, 9 EAS modes are implemented. Given: "
              "{}",
              locking_types.thickness);
      }
      // transverse shear strain locking: E_{13}, E_{23} constant
      switch (locking_types.transverse_shear_strain_const)
      {
        case 0:
          break;
        case 2:
          M(4, M_index + 1) = xi - 3.0 * xietaeta;
          M(5, M_index) = eta - 3.0 * xixieta;
          M_index += 2;
          break;
        case 4:
          M(4, M_index + 2) = 1.0 - 3.0 * etaeta;
          M(4, M_index + 3) = xi - 3.0 * xietaeta;
          M(5, M_index) = 1.0 - 3.0 * xixi;
          M(5, M_index + 1) = eta - 3.0 * xixieta;
          M_index += 4;
          break;
        case 6:
          M(4, M_index + 3) = 1.0 - 3.0 * etaeta;
          M(4, M_index + 4) = xi - 3.0 * xietaeta;
          M(4, M_index + 5) = xixi - 3.0 * xixietaeta;
          M(5, M_index) = 1.0 - 3.0 * xixi;
          M(5, M_index + 1) = eta - 3.0 * xixieta;
          M(5, M_index + 2) = etaeta - 3.0 * xixietaeta;
          M_index += 6;
          break;
        default:
          FOUR_C_THROW(
              "EAS transverse shear strain locking: Only 0, 2, 4, 6 EAS modes are implemented. "
              "Given: {}",
              locking_types.transverse_shear_strain_const);
      }
      // transverse shear strain locking: E_{13}, E_{23} linear
      switch (locking_types.transverse_shear_strain_lin)
      {
        case 0:
          break;
        case 2:
          M(10, M_index + 1) = etaeta;
          M(11, M_index) = xixi;
          M_index += 2;
          break;
        case 4:
          M(10, M_index + 2) = etaeta;
          M(10, M_index + 3) = xixietaeta;
          M(11, M_index) = xixi;
          M(11, M_index + 1) = xixietaeta;
          M_index += 4;
          break;
        case 6:
          M(10, M_index + 3) = etaeta;
          M(10, M_index + 4) = xietaeta;
          M(10, M_index + 5) = xixietaeta;
          M(11, M_index) = xixi;
          M(11, M_index + 1) = xixieta;
          M(11, M_index + 2) = xixietaeta;
          M_index += 6;
          break;
        default:
          FOUR_C_THROW(
              "EAS transverse shear strain locking: Only 0, 2, 4, 6 EAS modes are implemented. "
              "Given: {}",
              locking_types.transverse_shear_strain_lin);
      }
    }
    // four node element
    else if (num_node == 4)
    {
      // membrane locking: E_{11}, E_{22} constant
      switch (locking_types.membrane)
      {
        case 0:
          break;
        case 1:
          M(1, M_index) = eta;
          M_index += 1;
          break;
        case 2:
          M(3, M_index) = xi;
          M(3, M_index + 1) = eta;
          M_index += 2;
          break;
        case 3:
          M(3, M_index) = xi;
          M(3, M_index + 1) = eta;
          M(3, M_index + 2) = xieta;
          M_index += 3;
          break;
        case 4:
          M(0, M_index) = xi;
          M(1, M_index + 1) = eta;
          M(3, M_index + 2) = xi;
          M(3, M_index + 3) = eta;
          M_index += 4;
          break;
        case 5:
          M(0, M_index) = xi;
          M(1, M_index + 1) = eta;
          M(3, M_index + 2) = xi;
          M(3, M_index + 3) = eta;
          M(3, M_index + 4) = xieta;
          M_index += 5;
          break;
        case 7:
          M(0, M_index) = xi;
          M(0, M_index + 4) = xieta;
          M(1, M_index + 1) = eta;
          M(1, M_index + 5) = xieta;
          M(3, M_index + 2) = xi;
          M(3, M_index + 3) = eta;
          M(3, M_index + 6) = xieta;
          M_index += 7;
          break;
        default:
          FOUR_C_THROW(
              "EAS Membrane locking: Only 0, 1, 2, 3, 4, 5, 7 EAS modes are implemented. Given: {}",
              locking_types.membrane);
      }
      // bending part: E_{11}, E_{12], E_{22} linear
      switch (locking_types.bending)
      {
        case 0:
          break;
        case 4:
          M(6, M_index) = xi;
          M(7, M_index + 1) = eta;
          M(9, M_index + 2) = xi;
          M(9, M_index + 3) = eta;
          M_index += 4;
          break;
        case 5:
          M(6, M_index) = xi;
          M(7, M_index + 1) = eta;
          M(9, M_index + 2) = xi;
          M(9, M_index + 3) = eta;
          M(9, M_index + 4) = xieta;
          M_index += 5;
          break;
        case 6:
          M(6, M_index) = xixi;
          M(6, M_index + 1) = xixietaeta;
          M(7, M_index + 2) = etaeta;
          M(7, M_index + 3) = xixietaeta;
          M(9, M_index + 4) = xixi;
          M(9, M_index + 5) = etaeta;
          M_index += 6;
          break;
        case 7:
          M(6, M_index) = xi;
          M(6, M_index + 4) = xieta;
          M(7, M_index + 1) = eta;
          M(7, M_index + 5) = xieta;
          M(9, M_index + 2) = xi;
          M(9, M_index + 3) = eta;
          M(9, M_index + 6) = xieta;
          M_index += 7;
          break;
        default:
          FOUR_C_THROW("EAS bending part: Only 0, 4, 5, 7, 8 EAS modes are implemented. Given: {}",
              locking_types.bending);
      }
      // locking due to thickness changes E_{33} linear
      switch (locking_types.thickness)
      {
        case 0:
          break;
        case 1:
          M(8, M_index) = 1.0;
          M_index += 1;
          break;
        case 3:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M_index += 3;
          break;
        case 4:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M_index += 4;
          break;
        case 6:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M(8, M_index + 4) = xixi;
          M(8, M_index + 5) = etaeta;
          M_index += 6;
          break;
        case 8:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M(8, M_index + 4) = xixi;
          M(8, M_index + 5) = etaeta;
          M(8, M_index + 6) = xixieta;
          M(8, M_index + 7) = xietaeta;
          M_index += 8;
          break;
        case 9:
          M(8, M_index) = 1.0;
          M(8, M_index + 1) = xi;
          M(8, M_index + 2) = eta;
          M(8, M_index + 3) = xieta;
          M(8, M_index + 4) = 1.0 - 3.0 * xixi;
          M(8, M_index + 5) = 1.0 - 3.0 * etaeta;
          M(8, M_index + 6) = xixieta;
          M(8, M_index + 7) = xietaeta;
          M(8, M_index + 8) = 1.0 - 9.0 * xixietaeta;
          M_index += 9;
          break;
        default:
          FOUR_C_THROW(
              "EAS thickness locking: Only 0, 3, 4, 6, 8, 9 EAS modes are implemented. Given: {}",
              locking_types.thickness);
      }
      // transverse shear strain locking: E_{13}, E_{23} const
      switch (locking_types.transverse_shear_strain_const)
      {
        case 0:
          break;
        case 2:
          M(4, M_index + 1) = eta;
          M(5, M_index) = xi;
          M_index += 2;
          break;
        case 4:
          M(4, M_index + 2) = eta;
          M(4, M_index + 3) = xieta;
          M(5, M_index) = xi;
          M(5, M_index + 1) = xieta;
          M_index += 4;
          break;
        default:
          FOUR_C_THROW(
              "EAS transverse shear strain locking: Only 0, 2, 4 EAS modes are implemented. Given: "
              "{}",
              locking_types.transverse_shear_strain_const);
      }
      // transverse shear strain locking: E_{13}, E_{23} linear
      switch (locking_types.transverse_shear_strain_lin)
      {
        case 0:
          break;
        case 2:
          M(10, M_index + 1) = eta;
          M(11, M_index) = xi;
          M_index += 2;
          break;
        case 4:
          M(10, M_index + 2) = eta;
          M(10, M_index + 3) = xieta;
          M(11, M_index) = xi;
          M(11, M_index + 1) = xieta;
          M_index += 4;
          break;
        default:
          FOUR_C_THROW(
              "EAS transverse shear strain locking: Only 0, 2, 4 EAS modes are implemented. Given: "
              "{}",
              locking_types.transverse_shear_strain_lin);
      }
    }  // else if (iel==4)
    else
      FOUR_C_THROW("EAS only implemented for 4, 8 and 9 node elements");

    FOUR_C_ASSERT(M_index == locking_types.total,
        "Wrong total number of EAS parameters. Something went wrong.");

    return M;
  }

  /*!
   * @brief Map the EAS shapefunction matrix M in the parameter space to M_gp in the global
   * configuration and return M_gp
   *
   *  M_gp = detJ0/detJ T0^{-T} M = J0^T M J0^{-T}
   *
   * @tparam distype : discretization type
   * @param neas (in) : Number of total EAS parameters
   * @param a_reference (in) : An object holding the current basis vectors and
   * metric tensors of the shell body
   * @param metrics_centroid_reference (in) : An object holding the current basis vectors and
   * metric tensors evaluated at the element center
   * @return Core::LinAlg::SerialDenseMatrix : EAS shapefunction matrix M
   */
  template <Core::FE::CellType distype>
  Core::LinAlg::SerialDenseMatrix map_eas_shape_functions_to_gauss_point(const int& neas,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_reference,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& metrics_centroid_reference,
      Core::LinAlg::SerialDenseMatrix& M)
  {
    // get transformation matrix T0^-1 to map M from midpoint to gaussian point
    Core::LinAlg::SerialDenseMatrix T0inv =
        evaluate_t0inv<distype>(a_reference.covariant_, metrics_centroid_reference.contravariant_);
    // transform basis of M-matrix to gaussian point: M_gp = detJ0/ detJ * T0^-T * M
    Core::LinAlg::SerialDenseMatrix M_gp(
        Discret::Elements::Shell::Internal::num_internal_variables, neas);
    Core::LinAlg::multiply(
        0.0, M_gp, metrics_centroid_reference.detJ_ / a_reference.detJ_, T0inv, M);
    return M_gp;
  }

  /*!
   * @brief Evaluate the EAS shape function matrix M at a given Gauss point in global configuration
   *
   * @tparam distype : discretization type
   * @param xi_gp (in) : Coordinate in the parameter space
   * @param locking_types (in) : Object holding the number of EAS parameters for each locking part
   * @param a_reference (in) : An object holding the current basis vectors and
   * metric tensors of the shell body
   * @param metrics_centroid_reference (in) : An object holding the current basis vectors and
   * metric tensors evaluated at the element center
   * @return Core::LinAlg::SerialDenseMatrix : EAS shapefunction matrix M at gauss point
   */
  template <Core::FE::CellType distype>
  Core::LinAlg::SerialDenseMatrix evaluate_eas_shape_functions(const std::array<double, 2>& xi_gp,
      const Solid::Elements::ShellLockingTypes& locking_types,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_reference,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& metrics_centroid_reference)
  {
    Core::LinAlg::SerialDenseMatrix M =
        evaluate_eas_shape_functions_parameter_space<distype>(xi_gp, locking_types);
    // Evaluates M_gp by setting up M and mapping M to M_gp via T0^{-T}
    Core::LinAlg::SerialDenseMatrix M_gp = map_eas_shape_functions_to_gauss_point(
        locking_types.total, a_reference, metrics_centroid_reference, M);
    return M_gp;
  }

  // Evaluate enhanced EAS strains and return the strain vector
  inline Core::LinAlg::SerialDenseVector evaluate_eas_strains(
      const Core::LinAlg::SerialDenseMatrix& alpha, const Core::LinAlg::SerialDenseMatrix& M)
  {
    Core::LinAlg::SerialDenseVector strain_enh(Internal::num_internal_variables);
    strain_enh.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, M.base(), alpha.base(), 0.0);
    return strain_enh;
  }

  /*!
   * @brief Build the enhanced Green-Lagrange strain tensor from EAS strain vector
   *
   * Constructs a symmetric GL tensor \f$ E_{ij}(\zeta) = \alpha_{ij} + \zeta \beta_{ij}\f$ from the
   * EAS strain enhancement vector scaled by the thickness coordinate zeta. The strain_enh vector
   * from M*alpha represents GL strain components in standard Voigt notation with
   * \f$\gamma_{ij} = 2*E_{ij}\f$.
   *
   * @param strain_enh (in) : Enhanced strain vector in standard Voigt notation
   * @param zeta (in) : Thickness coordinate (scaled by SDC)
   * @return SymmetricTensor<double, 3, 3> : Enhanced Green-Lagrange strain tensor
   */
  inline Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim>
  evaluate_enhanced_gl_tensor(const Core::LinAlg::SerialDenseVector& strain_enh, const double zeta)
  {
    Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim> gl{};

    gl(0, 0) = strain_enh(0) + zeta * strain_enh(6);
    gl(1, 1) = strain_enh(1) + zeta * strain_enh(7);
    gl(2, 2) = strain_enh(2) + zeta * strain_enh(8);
    gl(0, 1) = (strain_enh(3) + zeta * strain_enh(9)) * 0.5;
    gl(1, 2) = (strain_enh(4) + zeta * strain_enh(10)) * 0.5;
    gl(0, 2) = (strain_enh(5) + zeta * strain_enh(11)) * 0.5;

    return gl;
  }

  /*!
   * @brief Compute total enhanced GL strain in curvilinear coordinates
   *
   * This function:
   * 1. Evaluates EAS strain vector from alpha parameters: strain_enh = M * alpha
   * 2. Converts to GL strain tensor (accounting for zeta scaling)
   * 3. Adds displacement-based GL strain to get total enhanced strain
   *
   * @param alpha EAS parameters
   * @param M EAS shape functions
   * @param zeta Thickness coordinate (scaled by SDC)
   * @param gl_strain_disp Displacement-based GL strain in curvilinear coordinates
   * @return Total enhanced GL strain in curvilinear coordinates
   */
  inline Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim>
  compute_total_enhanced_gl_strain(const Core::LinAlg::SerialDenseMatrix& alpha,
      const Core::LinAlg::SerialDenseMatrix& M, const double zeta,
      const Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim>&
          gl_strain_disp)
  {
    // evaluate EAS strain vector: strain_enh = M * alpha
    auto eas_strains = evaluate_eas_strains(alpha, M);

    // convert to GL strain tensor
    auto gl_strain_eas = evaluate_enhanced_gl_tensor(eas_strains, zeta);

    // add displacement + EAS strains in curvilinear space
    return gl_strain_disp + gl_strain_eas;
  }

  /*!
   * @brief Compute displacement-based GL strain in curvilinear coordinates
   *
   * Evaluates E_ij = 0.5 * (g_ij - G_ij) where g is current and G is reference metric.
   *
   * @param g_reference Reference metric tensors
   * @param g_current Current metric tensors
   * @return Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim>: GL strain
   * in curvilinear coordinates
   */
  template <Core::FE::CellType distype>
  inline Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim>
  evaluate_gl_strain_curvilinear(const BasisVectorsAndMetrics<distype>& g_reference,
      const BasisVectorsAndMetrics<distype>& g_current)
  {
    return Core::LinAlg::assume_symmetry(
        0.5 * (Core::LinAlg::make_tensor_view(g_current.metric_covariant_) -
                  Core::LinAlg::make_tensor_view(g_reference.metric_covariant_)));
  }


  /*!
   * @brief Add EAS internal force contribution of one Gauss point
   *
   * @param LinvDTilde (in) : L*DTilde^{-1}
   * @param RTilde (in) : RTilde
   * @param force_vector (in/out) : internal force vector where the local contribution is added to
   */
  inline void add_eas_internal_force(const Core::LinAlg::SerialDenseMatrix& LinvDTilde,
      const Core::LinAlg::SerialDenseMatrix& RTilde, Core::LinAlg::SerialDenseVector& force_vector)
  {
    // EAS internal force vector is: R - L * DTilde^-1 * RTilde (R is the usual unenhanced
    // internal force vector
    force_vector.multiply(
        Teuchos::NO_TRANS, Teuchos::NO_TRANS, -1., LinvDTilde.base(), RTilde.base(), 1.0);
  }

  /*!
   * @brief Add EAS stiffness matrix contribution of one Gauss point
   *
   * @param LinvDTilde(in) : L*DTilde^{-1}
   * @param transL (in) : L^T
   * @param stiffness_matrix (in/out) : stiffness matrix where the local contribution is added
   * to
   */
  inline void add_eas_stiffness_matrix(const Core::LinAlg::SerialDenseMatrix& LinvDTilde,
      const Core::LinAlg::SerialDenseMatrix& transL,
      Core::LinAlg::SerialDenseMatrix& stiffness_matrix)
  {
    // EAS-stiffness matrix is: K- L * DTilde^-1 * L^T (K is the usual unenhanced stiffness
    // matrix)
    Core::LinAlg::multiply(1.0, stiffness_matrix, -1.0, LinvDTilde, transL);
  }

  /*!
   * @brief Evaluates the enhanced strains scalar increment alpha
   *
   * @tparam distype : discretization type
   * @param old_eas_data (in) : EAS iteration data
   * @param neas (in) : Number of EAS parameter
   * @param residual (in) : Residual displacement increment
   * @param alpha_inc (out) : Enhanced strains scalar increment
   */
  template <Core::FE::CellType distype>
  void evaluate_alpha_increment(EASIterationData& old_eas_data, const int& neas,
      const std::vector<double>& residual, Core::LinAlg::SerialDenseMatrix& delta_alpha)
  {
    // we need the (residual) displacement at the previous step
    Core::LinAlg::SerialDenseVector disp_inc(Internal::numdofperelement<distype>);
    for (int i = 0; i < Internal::numdofperelement<distype>; ++i) disp_inc(i) = residual[i];

    Core::LinAlg::SerialDenseMatrix eashelp(neas, 1);
    // make multiplication eashelp = - old L^T * disp_incr[kstep]
    Core::LinAlg::multiply(eashelp, old_eas_data.transL_, disp_inc);
    // add old RTilde to eashelp
    eashelp += old_eas_data.RTilde_;
    // make multiplication alpha_inc = - old invDTilde * eashelp
    Core::LinAlg::multiply(0.0, delta_alpha, -1.0, old_eas_data.invDTilde_, eashelp);
  }

  /*!
   * @brief Integrate the EAS data
   *
   *  Function needs to be called for every gaussian point
   *
   * @tparam distype : discretization type
   * @param stress_resultants (in) : An object holding the stress resultants
   * @param M (in) : EAS shapefunctions
   * @param Bop (in) : B-operator matrix
   * @param eas_data (in/out) : An object holding the EAS data
   * @param integration_factor (in) : Integration factor
   */
  template <Core::FE::CellType distype>
  void integrate_eas(const StressResultants& stress_resultants,
      const Core::LinAlg::SerialDenseMatrix& M, const Core::LinAlg::SerialDenseMatrix& Bop,
      EASIterationData& eas_data, const double& integration_factor, const int& neas)
  {
    // integrate D_Tilde += M^T * D * M  * detJ * w(gp)
    // IMPORTANT: here we save D_Tilde in invDTilde_, since after the loop over all Gaussian
    // points, we invert the matrix. At this point, this is still D_Tilde and NOT invD_Tilde
    Core::LinAlg::SerialDenseMatrix MTDmat(neas, Internal::num_internal_variables);
    Core::LinAlg::multiply_tn(MTDmat, M, stress_resultants.dmat_);

    Core::LinAlg::multiply(1.0, eas_data.invDTilde_, integration_factor, MTDmat, M);

    //  integrate transL (L^T) += M^T * D * B * detJ * w(gp)
    Core::LinAlg::multiply(1.0, eas_data.transL_, integration_factor, MTDmat, Bop);

    //  integrate Rtilde (R_Tilde) : Rtilde  += M^T * stress_r * detJ * w(gp)
    Core::LinAlg::multiply_tn(
        1.0, eas_data.RTilde_, integration_factor, M, stress_resultants.stress_);
  }
}  // namespace Discret::Elements::Shell

FOUR_C_NAMESPACE_CLOSE

#endif
