// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_tensor_interpolation.hpp"

#include "4C_unittest_utils_assertions_test.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN


namespace
{
  TEST(LinalgTensorInterpolationTest, 2Sym3x3MatrixInterp2D)
  {
    // see Satheesh et al., 2024, 10.1002/nme.7373, Section 3.1.1, R-LOG interpolation

    Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

    Core::LinAlg::SecondOrderTensorInterpolator<1> interp(1);

    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices;
    std::vector<Core::LinAlg::Matrix<1, 1>> ref_locs;

    Core::LinAlg::Matrix<3, 3> lambda_T_1(Core::LinAlg::Initialization::zero);
    lambda_T_1(0, 0) = 10.0;
    lambda_T_1(1, 1) = 1.0;
    lambda_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> Q_T_1(Core::LinAlg::Initialization::zero);
    double angle_T_1 = M_PI / 4.0;
    Q_T_1(0, 0) = std::cos(angle_T_1);
    Q_T_1(0, 1) = -std::sin(angle_T_1);
    Q_T_1(1, 0) = std::sin(angle_T_1);
    Q_T_1(1, 1) = std::cos(angle_T_1);
    Q_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> T_1(Core::LinAlg::Initialization::zero);
    temp3x3.multiply_tn(1.0, Q_T_1, lambda_T_1, 0.0);
    T_1.multiply_nn(1.0, temp3x3, Q_T_1, 0.0);
    Core::LinAlg::Matrix<1, 1> loc_T_1(Core::LinAlg::Initialization::zero);
    loc_T_1(0, 0) = -5.0;

    Core::LinAlg::Matrix<3, 3> lambda_T_2(Core::LinAlg::Initialization::zero);
    lambda_T_2(0, 0) = 20.0;
    lambda_T_2(1, 1) = 4.0;
    lambda_T_2(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> Q_T_2(Core::LinAlg::Initialization::zero);
    double angle_T_2 = -0.99 * M_PI / 4.0;
    Q_T_2(0, 0) = std::cos(angle_T_2);
    Q_T_2(0, 1) = -std::sin(angle_T_2);
    Q_T_2(1, 0) = std::sin(angle_T_2);
    Q_T_2(1, 1) = std::cos(angle_T_2);
    Q_T_2(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> T_2(Core::LinAlg::Initialization::zero);
    temp3x3.multiply_tn(1.0, Q_T_2, lambda_T_2, 0.0);
    T_2.multiply_nn(1.0, temp3x3, Q_T_2, 0.0);
    Core::LinAlg::Matrix<1, 1> loc_T_2(Core::LinAlg::Initialization::zero);
    loc_T_2(0, 0) = 5.0;


    ref_matrices.push_back(T_1);
    ref_matrices.push_back(T_2);
    ref_locs.push_back(loc_T_1);
    ref_locs.push_back(loc_T_2);

    Core::LinAlg::Matrix<1, 1> loc_interp(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<3, 3> T_interp =
        interp.get_interpolated_matrix(ref_matrices, ref_locs, loc_interp);

    // reference result
    Core::LinAlg::Matrix<3, 3> lambda_T_ref(Core::LinAlg::Initialization::zero);
    lambda_T_ref(0, 0) =
        std::pow(lambda_T_1(0, 0), 1.0 / 2.0) * std::pow(lambda_T_2(0, 0), 1.0 / 2.0);
    lambda_T_ref(1, 1) =
        std::pow(lambda_T_1(1, 1), 1.0 / 2.0) * std::pow(lambda_T_2(1, 1), 1.0 / 2.0);
    lambda_T_ref(2, 2) =
        std::pow(lambda_T_1(2, 2), 1.0 / 2.0) * std::pow(lambda_T_2(2, 2), 1.0 / 2.0);
    Core::LinAlg::Matrix<3, 3> Q_T_ref(Core::LinAlg::Initialization::zero);
    double angle_T_ref = (1 - 0.99) / 2.0 * M_PI / 4.0;
    Q_T_ref(0, 0) = std::cos(angle_T_ref);
    Q_T_ref(0, 1) = -std::sin(angle_T_ref);
    Q_T_ref(1, 0) = std::sin(angle_T_ref);
    Q_T_ref(1, 1) = std::cos(angle_T_ref);
    Q_T_ref(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> T_ref(Core::LinAlg::Initialization::zero);
    temp3x3.multiply_tn(1.0, Q_T_ref, lambda_T_ref, 0.0);
    T_ref.multiply_nn(1.0, temp3x3, Q_T_ref, 0.0);


    FOUR_C_EXPECT_NEAR(T_interp, T_ref, 1.0e-9);
  }


  TEST(LinalgTensorInterpolationTest, 2NonSym3x3MatrixInterp2D)
  {
    // see Satheesh et al., 2024, 10.1002/nme.7373, Section 3.2.1, R-LOG interpolation

    Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

    Core::LinAlg::SecondOrderTensorInterpolator<1> interp(1);

    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices;
    std::vector<Core::LinAlg::Matrix<1, 1>> ref_locs;

    Core::LinAlg::Matrix<3, 3> lambda_T_1(Core::LinAlg::Initialization::zero);
    lambda_T_1(0, 0) = 10.0;
    lambda_T_1(1, 1) = 1.0;
    lambda_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> Q_T_1(Core::LinAlg::Initialization::zero);
    double angle_T_1_U = M_PI / 4.0;
    Q_T_1(0, 0) = std::cos(angle_T_1_U);
    Q_T_1(0, 1) = -std::sin(angle_T_1_U);
    Q_T_1(1, 0) = std::sin(angle_T_1_U);
    Q_T_1(1, 1) = std::cos(angle_T_1_U);
    Q_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> R_T_1(Core::LinAlg::Initialization::zero);
    double angle_T_1_R = M_PI / 2.0;
    R_T_1(0, 0) = std::cos(angle_T_1_R);
    R_T_1(0, 1) = -std::sin(angle_T_1_R);
    R_T_1(1, 0) = std::sin(angle_T_1_R);
    R_T_1(1, 1) = std::cos(angle_T_1_R);
    R_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> T_1(Core::LinAlg::Initialization::zero);
    T_1.multiply_tn(1.0, Q_T_1, lambda_T_1, 0.0);
    temp3x3.multiply_nn(1.0, T_1, Q_T_1, 0.0);
    T_1.multiply_nn(1.0, R_T_1, temp3x3, 0.0);
    Core::LinAlg::Matrix<1, 1> loc_T_1(Core::LinAlg::Initialization::zero);
    loc_T_1(0, 0) = -5.0;

    Core::LinAlg::Matrix<3, 3> lambda_T_2(Core::LinAlg::Initialization::zero);
    lambda_T_2(0, 0) = 20.0;
    lambda_T_2(1, 1) = 4.0;
    lambda_T_2(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> Q_T_2(Core::LinAlg::Initialization::zero);
    double angle_T_2 = -0.99 * M_PI / 4.0;
    Q_T_2(0, 0) = std::cos(angle_T_2);
    Q_T_2(0, 1) = -std::sin(angle_T_2);
    Q_T_2(1, 0) = std::sin(angle_T_2);
    Q_T_2(1, 1) = std::cos(angle_T_2);
    Q_T_2(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> T_2(Core::LinAlg::Initialization::zero);
    temp3x3.multiply_tn(1.0, Q_T_2, lambda_T_2, 0.0);
    T_2.multiply_nn(1.0, temp3x3, Q_T_2, 0.0);
    Core::LinAlg::Matrix<1, 1> loc_T_2(Core::LinAlg::Initialization::zero);
    loc_T_2(0, 0) = 5.0;


    ref_matrices.push_back(T_1);
    ref_matrices.push_back(T_2);
    ref_locs.push_back(loc_T_1);
    ref_locs.push_back(loc_T_2);

    Core::LinAlg::Matrix<1, 1> loc_interp(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<3, 3> T_interp =
        interp.get_interpolated_matrix(ref_matrices, ref_locs, loc_interp);

    // reference result
    Core::LinAlg::Matrix<3, 3> lambda_T_ref(Core::LinAlg::Initialization::zero);
    lambda_T_ref(0, 0) =
        std::pow(lambda_T_1(0, 0), 1.0 / 2.0) * std::pow(lambda_T_2(0, 0), 1.0 / 2.0);
    lambda_T_ref(1, 1) =
        std::pow(lambda_T_1(1, 1), 1.0 / 2.0) * std::pow(lambda_T_2(1, 1), 1.0 / 2.0);
    lambda_T_ref(2, 2) =
        std::pow(lambda_T_1(2, 2), 1.0 / 2.0) * std::pow(lambda_T_2(2, 2), 1.0 / 2.0);
    Core::LinAlg::Matrix<3, 3> Q_T_ref(Core::LinAlg::Initialization::zero);
    double angle_T_ref_U = (1 - 0.99) / 2.0 * M_PI / 4.0;
    Q_T_ref(0, 0) = std::cos(angle_T_ref_U);
    Q_T_ref(0, 1) = -std::sin(angle_T_ref_U);
    Q_T_ref(1, 0) = std::sin(angle_T_ref_U);
    Q_T_ref(1, 1) = std::cos(angle_T_ref_U);
    Q_T_ref(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> R_T_ref(Core::LinAlg::Initialization::zero);
    double angle_T_ref_R = M_PI / 4.0;
    R_T_ref(0, 0) = std::cos(angle_T_ref_R);
    R_T_ref(0, 1) = -std::sin(angle_T_ref_R);
    R_T_ref(1, 0) = std::sin(angle_T_ref_R);
    R_T_ref(1, 1) = std::cos(angle_T_ref_R);
    R_T_ref(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> T_ref(Core::LinAlg::Initialization::zero);
    T_ref.multiply_tn(1.0, Q_T_ref, lambda_T_ref, 0.0);
    temp3x3.multiply_nn(1.0, T_ref, Q_T_ref, 0.0);
    T_ref.multiply_nn(1.0, R_T_ref, temp3x3, 0.0);

    FOUR_C_EXPECT_NEAR(T_interp, T_ref, 1.0e-9);
  }


}  // namespace

FOUR_C_NAMESPACE_CLOSE
