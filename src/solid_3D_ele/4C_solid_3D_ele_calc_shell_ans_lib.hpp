// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_SHELL_ANS_LIB_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_SHELL_ANS_LIB_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"

FOUR_C_NAMESPACE_OPEN
namespace Discret::Elements
{
  template <Core::FE::CellType celltype>
  struct SamplingPoints
  {
  };

  template <>
  struct SamplingPoints<Core::FE::CellType::hex8>
  {
    static constexpr std::array<std::array<double, 3>, 8> value{
        {{{0.0, -1.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{-1.0, 0.0, 0.0}},
            {{-1.0, -1.0, 0.0}}, {{1.0, -1.0, 0.0}}, {{1.0, 1.0, 0.0}}, {{-1.0, 1.0, 0.0}}}};
  };

  template <>
  struct SamplingPoints<Core::FE::CellType::wedge6>
  {
    static constexpr std::array<std::array<double, 3>, 5> value{{{{0.5, 0.0, 0.0}},
        {{0.0, 0.5, 0.0}}, {{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}}};
  };

  template <Core::FE::CellType celltype>
  struct SamplingPointData
  {
    static constexpr int num_ans = 3;
    ShapeFunctionsAndDerivatives<celltype> shape_functions{};
    Core::LinAlg::Matrix<3, 3> reference_jacobian{};
    Core::LinAlg::Matrix<3, 3> current_jacobian{};

    // modified B-operator at local parametric space
    Core::LinAlg::Matrix<num_ans, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
        bop_ans_local{};
  };

  inline Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::hex8>,
      Internal::num_dof_per_ele<Core::FE::CellType::hex8>>
  evaluate_local_b_operator(const ElementNodes<Core::FE::CellType::hex8>& element_nodes,
      const Core::LinAlg::Tensor<double, Core::FE::dim<Core::FE::CellType::hex8>>& xi,
      const ShapeFunctionsAndDerivatives<Core::FE::CellType::hex8>& shape_functions,
      const JacobianMapping<Core::FE::CellType::hex8>& jacobian_mapping,
      const std::array<SamplingPointData<Core::FE::CellType::hex8>, 8>& sampling_point_data)
  {
    Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::hex8>,
        Internal::num_dof_per_ele<Core::FE::CellType::hex8>>
        bop_loc{};
    Core::LinAlg::Matrix<3, 3> current_jacobian(jacobian_mapping.jacobian_.data());
    current_jacobian.multiply_nt(
        1.0, shape_functions.derivatives_, element_nodes.displacements, 1.0);

    for (int inode = 0; inode < Core::FE::num_nodes(Core::FE::CellType::hex8); ++inode)
    {
      for (int dim = 0; dim < Core::FE::dim<Core::FE::CellType::hex8>; ++dim)
      {
        // rr
        bop_loc(0, inode * 3 + dim) =
            shape_functions.derivatives_(0, inode) * current_jacobian(0, dim);

        // ss
        bop_loc(1, inode * 3 + dim) =
            shape_functions.derivatives_(1, inode) * current_jacobian(1, dim);

        // rs
        bop_loc(3, inode * 3 + dim) =
            shape_functions.derivatives_(0, inode) * current_jacobian(1, dim) +
            shape_functions.derivatives_(1, inode) * current_jacobian(0, dim);

        // interpolate along (r x s) of bop_ans_local(tt) (sampling points 4, 5, 6, 7)
        bop_loc(2, inode * 3 + dim) = 0.25 * (1 - xi(0)) * (1 - xi(1)) *
                                          sampling_point_data[4].bop_ans_local(0, inode * 3 + dim) +
                                      0.25 * (1 + xi(0)) * (1 - xi(1)) *
                                          sampling_point_data[5].bop_ans_local(0, inode * 3 + dim) +
                                      0.25 * (1 + xi(0)) * (1 + xi(1)) *
                                          sampling_point_data[6].bop_ans_local(0, inode * 3 + dim) +
                                      0.25 * (1 - xi(0)) * (1 + xi(1)) *
                                          sampling_point_data[7].bop_ans_local(0, inode * 3 + dim);

        // interpolate along (r x s) of bop_ans_local(st) (sampling points 1, 3)
        bop_loc(4, inode * 3 + dim) =
            0.5 * (1 + xi(0)) * sampling_point_data[1].bop_ans_local(1, inode * 3 + dim) +
            0.5 * (1 - xi(0)) * sampling_point_data[3].bop_ans_local(1, inode * 3 + dim);

        // interpolate along (r x s) of bop_ans_local(rt) (sampling points 0, 2)
        bop_loc(5, inode * 3 + dim) =
            0.5 * (1 - xi(1)) * sampling_point_data[0].bop_ans_local(2, inode * 3 + dim) +
            0.5 * (1 + xi(1)) * sampling_point_data[2].bop_ans_local(2, inode * 3 + dim);
        ;
      }
    }

    return bop_loc;
  }
  inline Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::wedge6>,
      Internal::num_dof_per_ele<Core::FE::CellType::wedge6>>
  evaluate_local_b_operator(const ElementNodes<Core::FE::CellType::wedge6>& element_nodes,
      const Core::LinAlg::Tensor<double, Core::FE::dim<Core::FE::CellType::wedge6>>& xi,
      const ShapeFunctionsAndDerivatives<Core::FE::CellType::wedge6>& shape_functions,
      const JacobianMapping<Core::FE::CellType::wedge6>& jacobian_mapping,
      const std::array<SamplingPointData<Core::FE::CellType::wedge6>, 5>& sampling_point_data)
  {
    Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::wedge6>,
        Internal::num_dof_per_ele<Core::FE::CellType::wedge6>>
        bop_loc{};
    Core::LinAlg::Matrix<3, 3> current_jacobian(jacobian_mapping.jacobian_.data());
    current_jacobian.multiply_nt(
        1.0, shape_functions.derivatives_, element_nodes.displacements, 1.0);

    for (int inode = 0; inode < Core::FE::num_nodes(Core::FE::CellType::wedge6); ++inode)
    {
      for (int dim = 0; dim < Core::FE::dim<Core::FE::CellType::wedge6>; ++dim)
      {
        // rr
        bop_loc(0, inode * 3 + dim) =
            shape_functions.derivatives_(0, inode) * current_jacobian(0, dim);

        // ss
        bop_loc(1, inode * 3 + dim) =
            shape_functions.derivatives_(1, inode) * current_jacobian(1, dim);

        // rs
        bop_loc(3, inode * 3 + dim) =
            shape_functions.derivatives_(0, inode) * current_jacobian(1, dim) +
            shape_functions.derivatives_(1, inode) * current_jacobian(0, dim);

        // interpolate along (r x s) of bop_ans_local(tt) (sampling points 2,3,4)
        bop_loc(2, inode * 3 + dim) =
            (1 - xi(0) - xi(1)) * sampling_point_data[2].bop_ans_local(0, inode * 3 + dim) +
            xi(0) * sampling_point_data[3].bop_ans_local(0, inode * 3 + dim) +
            xi(1) * sampling_point_data[4].bop_ans_local(0, inode * 3 + dim);

        // interpolate along (r x s) of bop_ans_local(st) (sampling point 1)
        bop_loc(4, inode * 3 + dim) =
            xi(0) * sampling_point_data[1].bop_ans_local(1, inode * 3 + dim);

        // interpolate along (r x s) of bop_ans_local(rt) (sampling point 0)
        bop_loc(5, inode * 3 + dim) =
            xi(1) * sampling_point_data[0].bop_ans_local(2, inode * 3 + dim);
        ;
      }
    }

    return bop_loc;
  }

  inline Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::hex8>, 1>
  evaluate_local_glstrain(const ElementNodes<Core::FE::CellType::hex8>& element_nodes,
      const Core::LinAlg::Tensor<double, Core::FE::dim<Core::FE::CellType::hex8>>& xi,
      const ShapeFunctionsAndDerivatives<Core::FE::CellType::hex8>& shape_functions,
      const JacobianMapping<Core::FE::CellType::hex8>& jacobian_mapping,
      const std::array<SamplingPointData<Core::FE::CellType::hex8>, 8>& sampling_point_data)
  {
    Core::LinAlg::Matrix<3, 3> current_jacobian(jacobian_mapping.jacobian_.data());
    current_jacobian.multiply_nt(
        1.0, shape_functions.derivatives_, element_nodes.displacements, 1.0);

    Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::hex8>, 1> glstrain;
    // evaluate glstrains in local(parameter) coords
    // Err = 0.5 * (dx/dr * dx/dr^T - dX/dr * dX/dr^T)
    glstrain(0) =
        0.5 * (+(current_jacobian(0, 0) * current_jacobian(0, 0) +
                   current_jacobian(0, 1) * current_jacobian(0, 1) +
                   current_jacobian(0, 2) * current_jacobian(0, 2)) -
                  (jacobian_mapping.jacobian_(0, 0) * jacobian_mapping.jacobian_(0, 0) +
                      jacobian_mapping.jacobian_(0, 1) * jacobian_mapping.jacobian_(0, 1) +
                      jacobian_mapping.jacobian_(0, 2) * jacobian_mapping.jacobian_(0, 2)));
    // Ess = 0.5 * (dy/ds * dy/ds^T - dY/ds * dY/ds^T)
    glstrain(1) =
        0.5 * (+(current_jacobian(1, 0) * current_jacobian(1, 0) +
                   current_jacobian(1, 1) * current_jacobian(1, 1) +
                   current_jacobian(1, 2) * current_jacobian(1, 2)) -
                  (jacobian_mapping.jacobian_(1, 0) * jacobian_mapping.jacobian_(1, 0) +
                      jacobian_mapping.jacobian_(1, 1) * jacobian_mapping.jacobian_(1, 1) +
                      jacobian_mapping.jacobian_(1, 2) * jacobian_mapping.jacobian_(1, 2)));
    // Ers = (dx/ds * dy/dr^T - dX/ds * dY/dr^T)
    glstrain(3) = (+(current_jacobian(0, 0) * current_jacobian(1, 0) +
                       current_jacobian(0, 1) * current_jacobian(1, 1) +
                       current_jacobian(0, 2) * current_jacobian(1, 2)) -
                   (jacobian_mapping.jacobian_(0, 0) * jacobian_mapping.jacobian_(1, 0) +
                       jacobian_mapping.jacobian_(0, 1) * jacobian_mapping.jacobian_(1, 1) +
                       jacobian_mapping.jacobian_(0, 2) * jacobian_mapping.jacobian_(1, 2)));

    // ANS modification of strains ************************************** ANS
    double dxdt_A = 0.0;
    double dXdt_A = 0.0;
    double dydt_B = 0.0;
    double dYdt_B = 0.0;
    double dxdt_C = 0.0;
    double dXdt_C = 0.0;
    double dydt_D = 0.0;
    double dYdt_D = 0.0;

    double dzdt_E = 0.0;
    double dZdt_E = 0.0;
    double dzdt_F = 0.0;
    double dZdt_F = 0.0;
    double dzdt_G = 0.0;
    double dZdt_G = 0.0;
    double dzdt_H = 0.0;
    double dZdt_H = 0.0;

    // vector product of rows of jacobians at corresponding sampling point
    for (int dim = 0; dim < Core::FE::dim<Core::FE::CellType::hex8>; ++dim)
    {
      dxdt_A += sampling_point_data[0].current_jacobian(0, dim) *
                sampling_point_data[0].current_jacobian(2, dim);  // g_13^A
      dXdt_A += sampling_point_data[0].reference_jacobian(0, dim) *
                sampling_point_data[0].reference_jacobian(2, dim);  // G_13^A
      dydt_B += sampling_point_data[1].current_jacobian(1, dim) *
                sampling_point_data[1].current_jacobian(2, dim);  // g_23^B
      dYdt_B += sampling_point_data[1].reference_jacobian(1, dim) *
                sampling_point_data[1].reference_jacobian(2, dim);  // G_23^B
      dxdt_C += sampling_point_data[2].current_jacobian(0, dim) *
                sampling_point_data[2].current_jacobian(2, dim);  // g_13^C
      dXdt_C += sampling_point_data[2].reference_jacobian(0, dim) *
                sampling_point_data[2].reference_jacobian(2, dim);  // G_13^C
      dydt_D += sampling_point_data[3].current_jacobian(1, dim) *
                sampling_point_data[3].current_jacobian(2, dim);  // g_23^D
      dYdt_D += sampling_point_data[3].reference_jacobian(1, dim) *
                sampling_point_data[3].reference_jacobian(2, dim);  // G_23^D

      dzdt_E += sampling_point_data[4].current_jacobian(2, dim) *
                sampling_point_data[4].current_jacobian(2, dim);
      dZdt_E += sampling_point_data[4].reference_jacobian(2, dim) *
                sampling_point_data[4].reference_jacobian(2, dim);
      dzdt_F += sampling_point_data[5].current_jacobian(2, dim) *
                sampling_point_data[5].current_jacobian(2, dim);
      dZdt_F += sampling_point_data[5].reference_jacobian(2, dim) *
                sampling_point_data[5].reference_jacobian(2, dim);
      dzdt_G += sampling_point_data[6].current_jacobian(2, dim) *
                sampling_point_data[6].current_jacobian(2, dim);
      dZdt_G += sampling_point_data[6].reference_jacobian(2, dim) *
                sampling_point_data[6].reference_jacobian(2, dim);
      dzdt_H += sampling_point_data[7].current_jacobian(2, dim) *
                sampling_point_data[7].current_jacobian(2, dim);
      dZdt_H += sampling_point_data[7].reference_jacobian(2, dim) *
                sampling_point_data[7].reference_jacobian(2, dim);
    }

    // E33: remedy of curvature thickness locking
    // Ett = 0.5* ( (1-r)(1-s)/4 * Ett(SP E) + ... + (1-r)(1+s)/4 * Ett(SP H) )
    glstrain(2) = 0.5 * (0.25 * (1 - xi(0)) * (1 - xi(1)) * (dzdt_E - dZdt_E) +
                            0.25 * (1 + xi(0)) * (1 - xi(1)) * (dzdt_F - dZdt_F) +
                            0.25 * (1 + xi(0)) * (1 + xi(1)) * (dzdt_G - dZdt_G) +
                            0.25 * (1 - xi(0)) * (1 + xi(1)) * (dzdt_H - dZdt_H));
    // E23: remedy of transverse shear locking
    // Est = (1+r)/2 * Est(SP B) + (1-r)/2 * Est(SP D)
    glstrain(4) = 0.5 * (1 + xi(0)) * (dydt_B - dYdt_B) + 0.5 * (1 - xi(0)) * (dydt_D - dYdt_D);
    // E13: remedy of transverse shear locking
    // Ert = (1-s)/2 * Ert(SP A) + (1+s)/2 * Ert(SP C)
    glstrain(5) = 0.5 * (1 - xi(1)) * (dxdt_A - dXdt_A) + 0.5 * (1 + xi(1)) * (dxdt_C - dXdt_C);

    return glstrain;
  }


  inline Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::wedge6>, 1>
  evaluate_local_glstrain(const ElementNodes<Core::FE::CellType::wedge6>& element_nodes,
      const Core::LinAlg::Tensor<double, Core::FE::dim<Core::FE::CellType::wedge6>>& xi,
      const ShapeFunctionsAndDerivatives<Core::FE::CellType::wedge6>& shape_functions,
      const JacobianMapping<Core::FE::CellType::wedge6>& jacobian_mapping,
      const std::array<SamplingPointData<Core::FE::CellType::wedge6>, 5>& sampling_point_data)
  {
    Core::LinAlg::Matrix<3, 3> current_jacobian(jacobian_mapping.jacobian_.data());
    current_jacobian.multiply_nt(
        1.0, shape_functions.derivatives_, element_nodes.displacements, 1.0);

    Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::wedge6>, 1> glstrain;
    // evaluate glstrains in local(parameter) coords
    // Err = 0.5 * (dx/dr * dx/dr^T - dX/dr * dX/dr^T)
    glstrain(0) =
        0.5 * (+(current_jacobian(0, 0) * current_jacobian(0, 0) +
                   current_jacobian(0, 1) * current_jacobian(0, 1) +
                   current_jacobian(0, 2) * current_jacobian(0, 2)) -
                  (jacobian_mapping.jacobian_(0, 0) * jacobian_mapping.jacobian_(0, 0) +
                      jacobian_mapping.jacobian_(0, 1) * jacobian_mapping.jacobian_(0, 1) +
                      jacobian_mapping.jacobian_(0, 2) * jacobian_mapping.jacobian_(0, 2)));
    // Ess = 0.5 * (dy/ds * dy/ds^T - dY/ds * dY/ds^T)
    glstrain(1) =
        0.5 * (+(current_jacobian(1, 0) * current_jacobian(1, 0) +
                   current_jacobian(1, 1) * current_jacobian(1, 1) +
                   current_jacobian(1, 2) * current_jacobian(1, 2)) -
                  (jacobian_mapping.jacobian_(1, 0) * jacobian_mapping.jacobian_(1, 0) +
                      jacobian_mapping.jacobian_(1, 1) * jacobian_mapping.jacobian_(1, 1) +
                      jacobian_mapping.jacobian_(1, 2) * jacobian_mapping.jacobian_(1, 2)));
    // Ers = (dx/ds * dy/dr^T - dX/ds * dY/dr^T)
    glstrain(3) = (+(current_jacobian(0, 0) * current_jacobian(1, 0) +
                       current_jacobian(0, 1) * current_jacobian(1, 1) +
                       current_jacobian(0, 2) * current_jacobian(1, 2)) -
                   (jacobian_mapping.jacobian_(0, 0) * jacobian_mapping.jacobian_(1, 0) +
                       jacobian_mapping.jacobian_(0, 1) * jacobian_mapping.jacobian_(1, 1) +
                       jacobian_mapping.jacobian_(0, 2) * jacobian_mapping.jacobian_(1, 2)));

    // ANS modification of strains ************************************** ANS
    double dydt_A = 0.0;
    double dYdt_A = 0.0;
    constexpr int spA = 0;
    double dxdt_B = 0.0;
    double dXdt_B = 0.0;
    constexpr int spB = 1;
    double dzdt_C = 0.0;
    double dZdt_C = 0.0;
    constexpr int spC = 2;
    double dzdt_D = 0.0;
    double dZdt_D = 0.0;
    constexpr int spD = 3;
    double dzdt_E = 0.0;
    double dZdt_E = 0.0;
    constexpr int spE = 4;

    constexpr int xdir = 0;  // index to matrix x-row, r-row respectively
    constexpr int ydir = 1;  // index to matrix y-row, s-row respectively
    constexpr int zdir = 2;  // index to matrix z-row, t-row respectively

    // vector product of rows of jacobians at corresponding sampling point
    for (int dim = 0; dim < Core::FE::dim<Core::FE::CellType::wedge6>; ++dim)
    {
      dydt_A += sampling_point_data[spA].current_jacobian(xdir, dim) *
                sampling_point_data[spA].current_jacobian(zdir, dim);
      dYdt_A += sampling_point_data[spA].reference_jacobian(xdir, dim) *
                sampling_point_data[spA].reference_jacobian(zdir, dim);
      dxdt_B += sampling_point_data[spB].current_jacobian(ydir, dim) *
                sampling_point_data[spB].current_jacobian(zdir, dim);
      dXdt_B += sampling_point_data[spB].reference_jacobian(ydir, dim) *
                sampling_point_data[spB].reference_jacobian(zdir, dim);

      dzdt_C += sampling_point_data[spC].current_jacobian(zdir, dim) *
                sampling_point_data[spC].current_jacobian(zdir, dim);
      dZdt_C += sampling_point_data[spC].reference_jacobian(zdir, dim) *
                sampling_point_data[spC].reference_jacobian(zdir, dim);
      dzdt_D += sampling_point_data[spD].current_jacobian(zdir, dim) *
                sampling_point_data[spD].current_jacobian(zdir, dim);
      dZdt_D += sampling_point_data[spD].reference_jacobian(zdir, dim) *
                sampling_point_data[spD].reference_jacobian(zdir, dim);
      dzdt_E += sampling_point_data[spE].current_jacobian(zdir, dim) *
                sampling_point_data[spE].current_jacobian(zdir, dim);
      dZdt_E += sampling_point_data[spE].reference_jacobian(zdir, dim) *
                sampling_point_data[spE].reference_jacobian(zdir, dim);
    }

    // E33: remedy of curvature thickness locking
    // Ett = 0.5* ( (1-r)(1-s)/4 * Ett(SP E) + ... + (1-r)(1+s)/4 * Ett(SP H) )
    glstrain(2) = 0.5 * ((1 - xi(0) - xi(1)) * (dzdt_C - dZdt_C) + xi(0) * (dzdt_D - dZdt_D) +
                            xi(1) * (dzdt_E - dZdt_E));
    // E23: remedy of transverse shear locking
    // Est = (1+r)/2 * Est(SP B) + (1-r)/2 * Est(SP D)
    glstrain(4) = xi(0) * (dxdt_B - dXdt_B);
    // E13: remedy of transverse shear locking
    // Ert = (1-s)/2 * Ert(SP A) + (1+s)/2 * Ert(SP C)
    glstrain(5) = xi(1) * (dydt_A - dYdt_A);

    return glstrain;
  }

  inline void add_ans_geometric_stiffness(const Core::LinAlg::Tensor<double, 3>& xi,
      const ShapeFunctionsAndDerivatives<Core::FE::CellType::hex8>& shape_functions,
      const Stress<Core::FE::CellType::hex8>& stress, const double integration_factor,
      const Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::hex8>,
          Internal::num_str<Core::FE::CellType::hex8>>& TinvT,
      const std::array<SamplingPointData<Core::FE::CellType::hex8>, 8>& sampling_point_data,
      Core::LinAlg::Matrix<Internal::num_dim<Core::FE::CellType::hex8> *
                               Internal::num_nodes<Core::FE::CellType::hex8>,
          Internal::num_dim<Core::FE::CellType::hex8> *
              Internal::num_nodes<Core::FE::CellType::hex8>>& stiffness_matrix)
  {
    for (int inod = 0; inod < Core::FE::num_nodes(Core::FE::CellType::hex8); ++inod)
    {
      for (int jnod = 0; jnod < Core::FE::num_nodes(Core::FE::CellType::hex8); ++jnod)
      {
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> G_ij;
        G_ij(0) = shape_functions.derivatives_(0, inod) *
                  shape_functions.derivatives_(0, jnod);  // rr-dir
        G_ij(1) = shape_functions.derivatives_(1, inod) *
                  shape_functions.derivatives_(1, jnod);  // ss-dir
        G_ij(3) = shape_functions.derivatives_(0, inod) * shape_functions.derivatives_(1, jnod) +
                  shape_functions.derivatives_(1, inod) *
                      shape_functions.derivatives_(0, jnod);  // rs-dir


        // ANS modification in tt-dir
        G_ij(2) = 0.25 * (1 - xi(0)) * (1 - xi(1)) *
                      sampling_point_data[4].shape_functions.derivatives_(2, inod) *
                      sampling_point_data[4].shape_functions.derivatives_(2, jnod) +
                  0.25 * (1 + xi(0)) * (1 - xi(1)) *
                      sampling_point_data[5].shape_functions.derivatives_(2, inod) *
                      sampling_point_data[5].shape_functions.derivatives_(2, jnod) +
                  0.25 * (1 + xi(0)) * (1 + xi(1)) *
                      sampling_point_data[6].shape_functions.derivatives_(2, inod) *
                      sampling_point_data[6].shape_functions.derivatives_(2, jnod) +
                  0.25 * (1 - xi(0)) * (1 + xi(1)) *
                      sampling_point_data[7].shape_functions.derivatives_(2, inod) *
                      sampling_point_data[7].shape_functions.derivatives_(2, jnod);
        // ANS modification in st-dir
        G_ij(4) =
            0.5 *
            ((1 + xi(0)) * (sampling_point_data[1].shape_functions.derivatives_(1, inod) *
                                   sampling_point_data[1].shape_functions.derivatives_(2, jnod) +
                               sampling_point_data[1].shape_functions.derivatives_(2, inod) *
                                   sampling_point_data[1].shape_functions.derivatives_(1, jnod)) +
                (1 - xi(0)) *
                    (sampling_point_data[3].shape_functions.derivatives_(1, inod) *
                            sampling_point_data[3].shape_functions.derivatives_(2, jnod) +
                        sampling_point_data[3].shape_functions.derivatives_(2, inod) *
                            sampling_point_data[3].shape_functions.derivatives_(1, jnod)));
        // ANS modification in rt-dir
        G_ij(5) =
            0.5 *
            ((1 - xi(1)) * (sampling_point_data[0].shape_functions.derivatives_(0, inod) *
                                   sampling_point_data[0].shape_functions.derivatives_(2, jnod) +
                               sampling_point_data[0].shape_functions.derivatives_(2, inod) *
                                   sampling_point_data[0].shape_functions.derivatives_(0, jnod)) +
                (1 + xi(1)) *
                    (sampling_point_data[2].shape_functions.derivatives_(0, inod) *
                            sampling_point_data[2].shape_functions.derivatives_(2, jnod) +
                        sampling_point_data[2].shape_functions.derivatives_(2, inod) *
                            sampling_point_data[2].shape_functions.derivatives_(0, jnod)));


        // transformation of local(parameter) space 'back' to global(material) space
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> G_ij_glob;
        G_ij_glob.multiply(TinvT, G_ij);

        // Scalar Gij results from product of G_ij with stress, scaled with detJ*weights
        const double Gij = integration_factor *
                           Core::LinAlg::make_stress_like_voigt_view(stress.pk2_).dot(G_ij_glob);

        // add "geometric part" Gij times detJ*weights to stiffness matrix
        stiffness_matrix(Core::FE::dim<Core::FE::CellType::hex8> * inod + 0,
            Core::FE::dim<Core::FE::CellType::hex8> * jnod + 0) += Gij;
        stiffness_matrix(Core::FE::dim<Core::FE::CellType::hex8> * inod + 1,
            Core::FE::dim<Core::FE::CellType::hex8> * jnod + 1) += Gij;
        stiffness_matrix(Core::FE::dim<Core::FE::CellType::hex8> * inod + 2,
            Core::FE::dim<Core::FE::CellType::hex8> * jnod + 2) += Gij;
      }
    }
  }

  inline void add_ans_geometric_stiffness(const Core::LinAlg::Tensor<double, 3>& xi,
      const ShapeFunctionsAndDerivatives<Core::FE::CellType::wedge6>& shape_functions,
      const Stress<Core::FE::CellType::wedge6>& stress, const double integration_factor,
      const Core::LinAlg::Matrix<Internal::num_str<Core::FE::CellType::wedge6>,
          Internal::num_str<Core::FE::CellType::wedge6>>& TinvT,
      const std::array<SamplingPointData<Core::FE::CellType::wedge6>, 5>& sampling_point_data,
      Core::LinAlg::Matrix<Internal::num_dim<Core::FE::CellType::wedge6> *
                               Internal::num_nodes<Core::FE::CellType::wedge6>,
          Internal::num_dim<Core::FE::CellType::wedge6> *
              Internal::num_nodes<Core::FE::CellType::wedge6>>& stiffness_matrix)
  {
    for (int inod = 0; inod < Core::FE::num_nodes(Core::FE::CellType::wedge6); ++inod)
    {
      for (int jnod = 0; jnod < Core::FE::num_nodes(Core::FE::CellType::wedge6); ++jnod)
      {
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> G_ij;
        G_ij(0) = shape_functions.derivatives_(0, inod) *
                  shape_functions.derivatives_(0, jnod);  // rr-dir
        G_ij(1) = shape_functions.derivatives_(1, inod) *
                  shape_functions.derivatives_(1, jnod);  // ss-dir
        G_ij(3) = shape_functions.derivatives_(0, inod) * shape_functions.derivatives_(1, jnod) +
                  shape_functions.derivatives_(1, inod) *
                      shape_functions.derivatives_(0, jnod);  // rs-dir

        // ANS modification in tt-dir
        G_ij(2) = (1 - xi(0) - xi(1)) *
                      sampling_point_data[2].shape_functions.derivatives_(2, inod) *
                      sampling_point_data[2].shape_functions.derivatives_(2, jnod) +
                  xi(0) * sampling_point_data[3].shape_functions.derivatives_(2, inod) *
                      sampling_point_data[3].shape_functions.derivatives_(2, jnod) +
                  xi(1) * sampling_point_data[4].shape_functions.derivatives_(2, inod) *
                      sampling_point_data[4].shape_functions.derivatives_(2, jnod);
        // ANS modification in st-dir
        G_ij(4) = xi(0) * (sampling_point_data[1].shape_functions.derivatives_(1, inod) *
                                  sampling_point_data[1].shape_functions.derivatives_(2, jnod) +
                              sampling_point_data[1].shape_functions.derivatives_(2, inod) *
                                  sampling_point_data[1].shape_functions.derivatives_(1, jnod));
        // ANS modification in rt-dir
        G_ij(5) = xi(1) * (sampling_point_data[0].shape_functions.derivatives_(0, inod) *
                                  sampling_point_data[0].shape_functions.derivatives_(2, jnod) +
                              sampling_point_data[0].shape_functions.derivatives_(2, inod) *
                                  sampling_point_data[0].shape_functions.derivatives_(0, jnod));


        // transformation of local(parameter) space 'back' to global(material) space
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> G_ij_glob;
        G_ij_glob.multiply(TinvT, G_ij);

        // Scalar Gij results from product of G_ij with stress, scaled with detJ*weights
        const double Gij = integration_factor *
                           Core::LinAlg::make_stress_like_voigt_view(stress.pk2_).dot(G_ij_glob);

        // add "geometric part" Gij times detJ*weights to stiffness matrix
        stiffness_matrix(Core::FE::dim<Core::FE::CellType::wedge6> * inod + 0,
            Core::FE::dim<Core::FE::CellType::wedge6> * jnod + 0) += Gij;
        stiffness_matrix(Core::FE::dim<Core::FE::CellType::wedge6> * inod + 1,
            Core::FE::dim<Core::FE::CellType::wedge6> * jnod + 1) += Gij;
        stiffness_matrix(Core::FE::dim<Core::FE::CellType::wedge6> * inod + 2,
            Core::FE::dim<Core::FE::CellType::wedge6> * jnod + 2) += Gij;
      }
    }
  }

  template <std::size_t num_sampling_points, Core::FE::CellType celltype>
  std::array<SamplingPointData<celltype>, num_sampling_points> evaluate_sampling_point_data(
      const Core::Elements::Element& ele, const ElementNodes<celltype>& nodal_coordinates,
      const std::array<std::array<double, 3>, num_sampling_points>& sampling_points)
  {
    std::array<SamplingPointData<celltype>, num_sampling_points> sampling_point_data{};
    std::size_t i = 0;
    for (const std::array<double, 3>& sampling_point : sampling_points)
    {
      const Core::LinAlg::Tensor<double, 3> xi = Core::LinAlg::make_tensor<3>(sampling_point);
      // evaluate derivative of the shape functions
      sampling_point_data[i].shape_functions =
          evaluate_shape_functions_and_derivs(xi, nodal_coordinates);

      const ShapeFunctionsAndDerivatives<celltype>& shape_functions =
          sampling_point_data[i].shape_functions;

      sampling_point_data[i].reference_jacobian.multiply_nt(
          shape_functions.derivatives_, nodal_coordinates.reference_coordinates);

      sampling_point_data[i].current_jacobian = sampling_point_data[i].reference_jacobian;
      sampling_point_data[i].current_jacobian.multiply_nt(
          1.0, shape_functions.derivatives_, nodal_coordinates.displacements, 1.0);


      const Core::LinAlg::Matrix<3, 3>& current_jacobian = sampling_point_data[i].current_jacobian;
      // build local ans b-operator
      for (int inode = 0; inode < Core::FE::num_nodes(celltype); ++inode)
      {
        for (int dim = 0; dim < Core::FE::dim<celltype>; ++dim)
        {
          // modified b-operator in tt
          sampling_point_data[i].bop_ans_local(0, inode * 3 + dim) =
              shape_functions.derivatives_(2, inode) * current_jacobian(2, dim);

          // modified b-operator in st
          sampling_point_data[i].bop_ans_local(1, inode * 3 + dim) =
              shape_functions.derivatives_(1, inode) * current_jacobian(2, dim) +
              shape_functions.derivatives_(2, inode) * current_jacobian(1, dim);

          // modified b-operator in rt
          sampling_point_data[i].bop_ans_local(2, inode * 3 + dim) =
              shape_functions.derivatives_(0, inode) * current_jacobian(2, dim) +
              shape_functions.derivatives_(2, inode) * current_jacobian(0, dim);
        }
      }


      i += 1;
    }

    return sampling_point_data;
  }

}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE
#endif