// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_tensor_interpolation.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <Sacado_tradvec.hpp>
#include <Teuchos_ParameterList.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <ios>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace
{
  TEST(LinalgTensorInterpolationTest, 2Sym3x3MatrixInterp2D)
  {
    // see Satheesh et al., 2024, 10.1002/nme.7373, Section 3.1.1, R-LOG interpolation

    Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

    // create interpolation parameter object
    Core::LinAlg::ScalarInterpolationParams interp_params;

    // create tensor interpolator
    Core::LinAlg::SecondOrderTensorInterpolator<1> interp{1,
        Core::LinAlg::RotationInterpolationType::RotationVector,
        Core::LinAlg::EigenvalInterpolationType::LOG, interp_params};

    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices;
    std::vector<Core::LinAlg::Matrix<1, 1>> ref_locs;

    Core::LinAlg::Matrix<3, 3> lambda_T_1(Core::LinAlg::Initialization::zero);
    lambda_T_1(0, 0) = 10.0;
    lambda_T_1(1, 1) = 1.0;
    lambda_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> Q_T_1(Core::LinAlg::Initialization::zero);
    double angle_T_1 = std::numbers::pi / 4.0;
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
    double angle_T_2 = -0.99 * std::numbers::pi / 4.0;
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

    Core::LinAlg::TensorInterpolationErrorType err_type =
        Core::LinAlg::TensorInterpolationErrorType::NoErrors;
    Core::LinAlg::Matrix<3, 3> T_interp =
        interp.get_interpolated_matrix(ref_matrices, ref_locs, loc_interp, err_type);
    FOUR_C_ASSERT_ALWAYS(err_type == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
        "Tensor interpolation failed with err: {}", Core::LinAlg::make_error_message(err_type));

    // reference result
    Core::LinAlg::Matrix<3, 3> lambda_T_ref(Core::LinAlg::Initialization::zero);
    lambda_T_ref(0, 0) =
        std::pow(lambda_T_1(0, 0), 1.0 / 2.0) * std::pow(lambda_T_2(0, 0), 1.0 / 2.0);
    lambda_T_ref(1, 1) =
        std::pow(lambda_T_1(1, 1), 1.0 / 2.0) * std::pow(lambda_T_2(1, 1), 1.0 / 2.0);
    lambda_T_ref(2, 2) =
        std::pow(lambda_T_1(2, 2), 1.0 / 2.0) * std::pow(lambda_T_2(2, 2), 1.0 / 2.0);
    Core::LinAlg::Matrix<3, 3> Q_T_ref(Core::LinAlg::Initialization::zero);
    double angle_T_ref = (1 - 0.99) / 2.0 * std::numbers::pi / 4.0;
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

    // create interpolation parameter object
    Core::LinAlg::ScalarInterpolationParams interp_params;

    // create tensor interpolator
    Core::LinAlg::SecondOrderTensorInterpolator<1> interp{1,
        Core::LinAlg::RotationInterpolationType::RotationVector,
        Core::LinAlg::EigenvalInterpolationType::LOG, interp_params};

    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices;
    std::vector<Core::LinAlg::Matrix<1, 1>> ref_locs;

    Core::LinAlg::Matrix<3, 3> lambda_T_1(Core::LinAlg::Initialization::zero);
    lambda_T_1(0, 0) = 10.0;
    lambda_T_1(1, 1) = 1.0;
    lambda_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> Q_T_1(Core::LinAlg::Initialization::zero);
    double angle_T_1_U = std::numbers::pi / 4.0;
    Q_T_1(0, 0) = std::cos(angle_T_1_U);
    Q_T_1(0, 1) = -std::sin(angle_T_1_U);
    Q_T_1(1, 0) = std::sin(angle_T_1_U);
    Q_T_1(1, 1) = std::cos(angle_T_1_U);
    Q_T_1(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> R_T_1(Core::LinAlg::Initialization::zero);
    double angle_T_1_R = std::numbers::pi / 2.0;
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
    double angle_T_2 = -0.99 * std::numbers::pi / 4.0;
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

    Core::LinAlg::TensorInterpolationErrorType err_type =
        Core::LinAlg::TensorInterpolationErrorType::NoErrors;
    Core::LinAlg::Matrix<3, 3> T_interp =
        interp.get_interpolated_matrix(ref_matrices, ref_locs, loc_interp, err_type);
    FOUR_C_ASSERT_ALWAYS(err_type == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
        "Tensor interpolation failed with err: {}", Core::LinAlg::make_error_message(err_type));

    // reference result
    Core::LinAlg::Matrix<3, 3> lambda_T_ref(Core::LinAlg::Initialization::zero);
    lambda_T_ref(0, 0) =
        std::pow(lambda_T_1(0, 0), 1.0 / 2.0) * std::pow(lambda_T_2(0, 0), 1.0 / 2.0);
    lambda_T_ref(1, 1) =
        std::pow(lambda_T_1(1, 1), 1.0 / 2.0) * std::pow(lambda_T_2(1, 1), 1.0 / 2.0);
    lambda_T_ref(2, 2) =
        std::pow(lambda_T_1(2, 2), 1.0 / 2.0) * std::pow(lambda_T_2(2, 2), 1.0 / 2.0);
    Core::LinAlg::Matrix<3, 3> Q_T_ref(Core::LinAlg::Initialization::zero);
    double angle_T_ref_U = (1 - 0.99) / 2.0 * std::numbers::pi / 4.0;
    Q_T_ref(0, 0) = std::cos(angle_T_ref_U);
    Q_T_ref(0, 1) = -std::sin(angle_T_ref_U);
    Q_T_ref(1, 0) = std::sin(angle_T_ref_U);
    Q_T_ref(1, 1) = std::cos(angle_T_ref_U);
    Q_T_ref(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> R_T_ref(Core::LinAlg::Initialization::zero);
    double angle_T_ref_R = std::numbers::pi / 4.0;
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

  TEST(LinalgTensorInterpolationTest, General1DInterpTest)
  {
    // set left matrix (loc = 0)
    Core::LinAlg::Matrix<3, 3> left_matrix{Core::LinAlg::Initialization::zero};
    left_matrix(0, 0) = 1.802440e-01;
    left_matrix(0, 1) = 7.461039e-01;
    left_matrix(0, 2) = -3.148670e-10;
    left_matrix(1, 0) = -5.701622e-01;
    left_matrix(1, 1) = 3.263754e+00;
    left_matrix(1, 2) = -8.927429e-10;
    left_matrix(2, 0) = 1.019112e-10;
    left_matrix(2, 1) = 1.152076e-10;
    left_matrix(2, 2) = 9.865120e-01;
    // perform polar decomposition of left matrix
    Core::LinAlg::Matrix<3, 3> left_matrix_R{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> left_matrix_U{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> left_matrix_lambda{Core::LinAlg::Initialization::zero};
    std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> left_matrix_spectral;
    Core::LinAlg::matrix_3x3_polar_decomposition(
        left_matrix, left_matrix_R, left_matrix_U, left_matrix_lambda, left_matrix_spectral);
    // build Q left matrix
    Core::LinAlg::Matrix<3, 3> left_matrix_Q{Core::LinAlg::Initialization::zero};
    left_matrix_Q(0, 0) = left_matrix_spectral[0].second(0);
    left_matrix_Q(0, 1) = left_matrix_spectral[0].second(1);
    left_matrix_Q(0, 2) = left_matrix_spectral[0].second(2);
    left_matrix_Q(1, 0) = left_matrix_spectral[1].second(0);
    left_matrix_Q(1, 1) = left_matrix_spectral[1].second(1);
    left_matrix_Q(1, 2) = left_matrix_spectral[1].second(2);
    left_matrix_Q(2, 0) = left_matrix_spectral[2].second(0);
    left_matrix_Q(2, 1) = left_matrix_spectral[2].second(1);
    left_matrix_Q(2, 2) = left_matrix_spectral[2].second(2);
    // get the rotation vectors
    Core::LinAlg::Matrix<3, 1> left_matrix_R_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(left_matrix_R);
    Core::LinAlg::Matrix<3, 1> left_matrix_Q_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(left_matrix_Q);

    // set right matrix (loc = 1)
    Core::LinAlg::Matrix<3, 3> right_matrix{Core::LinAlg::Initialization::zero};
    right_matrix(0, 0) = 4.337834e-01;
    right_matrix(0, 1) = 6.330147e-01;
    right_matrix(0, 2) = -9.099803e-11;
    right_matrix(1, 0) = 6.330147e-01;
    right_matrix(1, 1) = 3.260640e+00;
    right_matrix(1, 2) = -1.181847e-10;
    right_matrix(2, 0) = -9.099803e-11;
    right_matrix(2, 1) = -1.181847e-10;
    right_matrix(2, 2) = 9.864815e-01;
    // perform polar decomposition of right matrix
    Core::LinAlg::Matrix<3, 3> right_matrix_R{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> right_matrix_U{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> right_matrix_lambda{Core::LinAlg::Initialization::zero};
    std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> right_matrix_spectral;
    Core::LinAlg::matrix_3x3_polar_decomposition(
        right_matrix, right_matrix_R, right_matrix_U, right_matrix_lambda, right_matrix_spectral);
    // build Q matrix
    Core::LinAlg::Matrix<3, 3> right_matrix_Q{Core::LinAlg::Initialization::zero};
    right_matrix_Q(0, 0) = right_matrix_spectral[0].second(0);
    right_matrix_Q(0, 1) = right_matrix_spectral[0].second(1);
    right_matrix_Q(0, 2) = right_matrix_spectral[0].second(2);
    right_matrix_Q(1, 0) = right_matrix_spectral[1].second(0);
    right_matrix_Q(1, 1) = right_matrix_spectral[1].second(1);
    right_matrix_Q(1, 2) = right_matrix_spectral[1].second(2);
    right_matrix_Q(2, 0) = right_matrix_spectral[2].second(0);
    right_matrix_Q(2, 1) = right_matrix_spectral[2].second(1);
    right_matrix_Q(2, 2) = right_matrix_spectral[2].second(2);
    // get the rotation vectors
    Core::LinAlg::Matrix<3, 1> right_matrix_R_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(right_matrix_R);
    Core::LinAlg::Matrix<3, 1> right_matrix_Q_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(right_matrix_Q);

    // create interpolation parameter object
    Core::LinAlg::ScalarInterpolationParams interp_params;

    // create tensor interpolator
    Core::LinAlg::SecondOrderTensorInterpolator<1> interp{1,
        Core::LinAlg::RotationInterpolationType::RotationVector,
        Core::LinAlg::EigenvalInterpolationType::LOG, interp_params};

    // get interp matrix (loc = specified)
    double loc = 5.695328e-01;
    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices{left_matrix, right_matrix};
    std::vector<double> ref_locs{0.0, 1.0};
    Core::LinAlg::TensorInterpolationErrorType err_type =
        Core::LinAlg::TensorInterpolationErrorType::NoErrors;
    Core::LinAlg::Matrix<3, 3> interp_matrix =
        interp.get_interpolated_matrix(ref_matrices, ref_locs, loc, err_type);
    FOUR_C_ASSERT_ALWAYS(err_type == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
        "Tensor interpolation failed with err: {}", Core::LinAlg::make_error_message(err_type));

    // perform polar decomposition of interp matrix
    Core::LinAlg::Matrix<3, 3> interp_matrix_R{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> interp_matrix_U{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> interp_matrix_lambda{Core::LinAlg::Initialization::zero};
    std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> interp_matrix_spectral;
    Core::LinAlg::matrix_3x3_polar_decomposition(interp_matrix, interp_matrix_R, interp_matrix_U,
        interp_matrix_lambda, interp_matrix_spectral);
    // build Q matrix
    Core::LinAlg::Matrix<3, 3> interp_matrix_Q{Core::LinAlg::Initialization::zero};
    interp_matrix_Q(0, 0) = interp_matrix_spectral[0].second(0);
    interp_matrix_Q(0, 1) = interp_matrix_spectral[0].second(1);
    interp_matrix_Q(0, 2) = interp_matrix_spectral[0].second(2);
    interp_matrix_Q(1, 0) = interp_matrix_spectral[1].second(0);
    interp_matrix_Q(1, 1) = interp_matrix_spectral[1].second(1);
    interp_matrix_Q(1, 2) = interp_matrix_spectral[1].second(2);
    interp_matrix_Q(2, 0) = interp_matrix_spectral[2].second(0);
    interp_matrix_Q(2, 1) = interp_matrix_spectral[2].second(1);
    interp_matrix_Q(2, 2) = interp_matrix_spectral[2].second(2);
    // get the rotation vectors
    Core::LinAlg::Matrix<3, 1> interp_matrix_R_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(interp_matrix_R);
    Core::LinAlg::Matrix<3, 1> interp_matrix_Q_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(interp_matrix_Q);

    // define corresponding weighting during interpolation
    double left_unnorm = std::exp(-interp_params.exponential_decay_c * loc * loc);
    double right_unnorm = std::exp(-interp_params.exponential_decay_c * (1.0 - loc) * (1.0 - loc));
    double left_weight = left_unnorm / (left_unnorm + right_unnorm);
    double right_weight = right_unnorm / (left_unnorm + right_unnorm);

    // verify eigenvalues
    Core::LinAlg::Matrix<3, 3> lambda_ref{Core::LinAlg::Initialization::zero};
    lambda_ref(2, 2) = std::exp(left_weight * std::log(left_matrix_spectral[0].first) +
                                right_weight * std::log(right_matrix_spectral[0].first));
    lambda_ref(1, 1) = std::exp(left_weight * std::log(left_matrix_spectral[1].first) +
                                right_weight * std::log(right_matrix_spectral[1].first));
    lambda_ref(0, 0) = std::exp(left_weight * std::log(left_matrix_spectral[2].first) +
                                right_weight * std::log(right_matrix_spectral[2].first));

    FOUR_C_EXPECT_NEAR(interp_matrix_lambda, lambda_ref, 1.0e-9);

    // compute relative rotation matrices
    Core::LinAlg::Matrix<3, 3>& Q_master = right_matrix_Q;
    Core::LinAlg::Matrix<3, 3>& R_master = right_matrix_R;
    if (loc < 0.5)
    {
      Q_master = left_matrix_Q;
      R_master = left_matrix_R;
    }
    Core::LinAlg::Matrix<3, 3> left_matrix_Q_rel{Core::LinAlg::Initialization::zero};
    left_matrix_Q_rel.multiply_tn(1.0, Q_master, left_matrix_Q, 0.0);
    Core::LinAlg::Matrix<3, 3> right_matrix_Q_rel{Core::LinAlg::Initialization::zero};
    right_matrix_Q_rel.multiply_tn(1.0, Q_master, right_matrix_Q, 0.0);
    Core::LinAlg::Matrix<3, 3> interp_matrix_Q_rel{Core::LinAlg::Initialization::zero};
    interp_matrix_Q_rel.multiply_tn(1.0, Q_master, interp_matrix_Q, 0.0);
    Core::LinAlg::Matrix<3, 3> left_matrix_R_rel{Core::LinAlg::Initialization::zero};
    left_matrix_R_rel.multiply_tn(1.0, R_master, left_matrix_R, 0.0);
    Core::LinAlg::Matrix<3, 3> right_matrix_R_rel{Core::LinAlg::Initialization::zero};
    right_matrix_R_rel.multiply_tn(1.0, R_master, right_matrix_R, 0.0);
    Core::LinAlg::Matrix<3, 3> interp_matrix_R_rel{Core::LinAlg::Initialization::zero};
    interp_matrix_R_rel.multiply_tn(1.0, R_master, interp_matrix_R, 0.0);
    // compute relative rotation vectors
    Core::LinAlg::Matrix<3, 1> left_matrix_Q_rel_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(left_matrix_Q_rel);
    Core::LinAlg::Matrix<3, 1> right_matrix_Q_rel_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(right_matrix_Q_rel);
    Core::LinAlg::Matrix<3, 1> interp_matrix_Q_rel_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(interp_matrix_Q_rel);
    Core::LinAlg::Matrix<3, 1> left_matrix_R_rel_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(left_matrix_R_rel);
    Core::LinAlg::Matrix<3, 1> right_matrix_R_rel_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(right_matrix_R_rel);
    Core::LinAlg::Matrix<3, 1> interp_matrix_R_rel_vect =
        Core::LinAlg::calc_rot_vect_from_rot_matrix(interp_matrix_R_rel);


    // verify rotation vectors
    Core::LinAlg::Matrix<3, 1> Q_rel_vect_ref{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 1> R_rel_vect_ref{Core::LinAlg::Initialization::zero};
    // build P-matrix
    Core::LinAlg::Matrix<2, 2> P{Core::LinAlg::Initialization::zero};
    P(0, 0) = left_weight + right_weight;
    P(0, 1) = right_weight;
    P(1, 0) = right_weight;
    P(1, 1) = right_weight;
    // get inverse of the P-matrix
    Core::LinAlg::Matrix<2, 2> iP{Core::LinAlg::Initialization::zero};
    iP.invert(P);
    // declare coefficient <a_i> vector and right-hand side <b_i> vector
    Core::LinAlg::Matrix<2, 1> a_i{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<2, 1> b_i{Core::LinAlg::Initialization::zero};
    // build Q vector (reference)
    for (int i = 0; i < 3; ++i)
    {
      // build b_i vector
      b_i(0) = left_weight * left_matrix_Q_rel_vect(i) + right_weight * right_matrix_Q_rel_vect(i);
      b_i(1) = right_weight * right_matrix_Q_rel_vect(i);

      // get corresponding coefficients
      a_i.multiply(1.0, iP, b_i, 0.0);

      // compute corresponding rotation vector contribution (interpolation)
      Q_rel_vect_ref(i) = 1.0 * a_i(0) + loc * a_i(1);
    }
    // build R vector (reference)
    for (int i = 0; i < 3; ++i)
    {
      // build b_i vector
      b_i(0) = left_weight * left_matrix_R_rel_vect(i) + right_weight * right_matrix_R_rel_vect(i);
      b_i(1) = right_weight * right_matrix_R_rel_vect(i);

      // get corresponding coefficients
      a_i.multiply(1.0, iP, b_i, 0.0);

      // compute corresponding rotation vector contribution (interpolation)
      R_rel_vect_ref(i) = 1.0 * a_i(0) + loc * a_i(1);
    }

    FOUR_C_EXPECT_NEAR(interp_matrix_Q_rel_vect, Q_rel_vect_ref, 1.0e-9);
    FOUR_C_EXPECT_NEAR(interp_matrix_R_rel_vect, R_rel_vect_ref, 1.0e-9);
  }

  /// Compute the gradient \f$ \frac{\partial \lambda}{\partial
  /// \boldsymbol{x}} \f$. We assume that the interpolated eigenvalue is
  /// computed with the logarithmic weighted average method. We have two
  /// reference matrices, left and right, with corresponding eigenvalues
  /// and locations.
  double get_eigenval_gradient_log_weighted_average(const double c, const double left_eigenval,
      const double left_loc, const double right_eigenval, const double right_loc,
      const double interp_loc)
  {
    // compute the distances of the interpolation location with respect to
    // the reference locations
    double left_dist{interp_loc - left_loc};
    const double abs_left_dist = std::abs(left_dist);
    double right_dist{interp_loc - right_loc};
    const double abs_right_dist = std::abs(right_dist);

    // compute the unnormalized weights and their sum
    const double left_unnorm_weight = std::exp(-c * abs_left_dist * abs_left_dist);
    const double right_unnorm_weight = std::exp(-c * abs_right_dist * abs_right_dist);
    const double sum_unnorm_weight = left_unnorm_weight + right_unnorm_weight;

    // compute the normalized weights and their sum
    const double left_norm_weight = left_unnorm_weight / sum_unnorm_weight;
    const double right_norm_weight = right_unnorm_weight / sum_unnorm_weight;


    // compute the gradients of the unnormalized weights
    double left_unnorm_weight_gradient{-2.0 * c * left_unnorm_weight * left_dist};
    double right_unnorm_weight_gradient{-2.0 * c * right_unnorm_weight * right_dist};

    // compute the gradients of the normalized weights
    double left_norm_weight_gradient{
        (1.0 / sum_unnorm_weight - left_unnorm_weight / (sum_unnorm_weight * sum_unnorm_weight)) *
            left_unnorm_weight_gradient -
        left_unnorm_weight / (sum_unnorm_weight * sum_unnorm_weight) *
            right_unnorm_weight_gradient};
    double right_norm_weight_gradient{
        -right_unnorm_weight / (sum_unnorm_weight * sum_unnorm_weight) *
            left_unnorm_weight_gradient +
        (1.0 / sum_unnorm_weight - right_unnorm_weight / (sum_unnorm_weight * sum_unnorm_weight)) *
            right_unnorm_weight_gradient};

    // compute the interpolated eigenvalue
    double interp_eigenval = std::exp(
        left_norm_weight * std::log(left_eigenval) + right_norm_weight * std::log(right_eigenval));

    // compute the gradient of the interpolated eigenvalue
    double interp_eigenval_gradient{std::log(left_eigenval) * left_norm_weight_gradient +
                                    std::log(right_eigenval) * right_norm_weight_gradient};
    interp_eigenval_gradient *= interp_eigenval;

    return interp_eigenval_gradient;
  }

  TEST(LinalgTensorInterpolationTest, InterpGradientTest)
  {
    // set left matrix (loc = 0, 0, 0)
    Core::LinAlg::Matrix<3, 3> left_matrix{Core::LinAlg::Initialization::zero};
    left_matrix(0, 0) = 1.0;
    left_matrix(1, 1) = 1.0;
    left_matrix(2, 2) = 1.0;
    Core::LinAlg::Matrix<1, 1> left_matrix_loc{Core::LinAlg::Initialization::zero};


    // set right matrix (loc = 1, 1, 1)
    Core::LinAlg::Matrix<3, 3> right_matrix{Core::LinAlg::Initialization::zero};
    right_matrix(0, 0) = 2.0;
    right_matrix(1, 1) = 4.0;
    right_matrix(2, 2) = 6.0;
    Core::LinAlg::Matrix<1, 1> right_matrix_loc{Core::LinAlg::Initialization::zero};
    right_matrix_loc(0) = 1.0;

    // set interpolation location
    Core::LinAlg::Matrix<1, 1> interp_loc{Core::LinAlg::Initialization::zero};
    interp_loc(0) = 0.5;

    // create interpolation parameter object
    Core::LinAlg::ScalarInterpolationParams interp_params;
    interp_params.exponential_decay_c = 10.0;
    const double perturbation_factor = 1.0e-10;

    // create tensor interpolator
    Core::LinAlg::SecondOrderTensorInterpolator<1> interp{1,
        Core::LinAlg::RotationInterpolationType::RotationVector,
        Core::LinAlg::EigenvalInterpolationType::LOG, interp_params};

    // declare error type
    Core::LinAlg::TensorInterpolationErrorType err_type =
        Core::LinAlg::TensorInterpolationErrorType::NoErrors;

    // get interpolation gradient
    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices{left_matrix, right_matrix};
    std::vector<Core::LinAlg::Matrix<1, 1>> ref_locs{left_matrix_loc, right_matrix_loc};
    // ... using the standard method
    Core::LinAlg::Matrix<9, 1> interp_gradient_standard = interp.get_interpolation_gradient(
        ref_matrices, ref_locs, interp_loc, err_type, perturbation_factor);
    FOUR_C_ASSERT_ALWAYS(err_type == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
        "Tensor interpolation failed with err: {}", Core::LinAlg::make_error_message(err_type));
    // ... using the specialized method
    Core::LinAlg::Matrix<9, 1> interp_gradient_specialized = interp.get_interpolation_gradient(
        ref_matrices, std::vector<double>{left_matrix_loc(0, 0), right_matrix_loc(0, 0)},
        interp_loc(0, 0), err_type, perturbation_factor);
    FOUR_C_ASSERT_ALWAYS(err_type == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
        "Tensor interpolation failed with err: {}", Core::LinAlg::make_error_message(err_type));


    // declare the reference eigenvalue derivatives
    double ref_eigenval_gradient;
    // loop through eigenvalues
    for (unsigned int i = 0; i < 3; ++i)
    {
      // compute the reference solution
      ref_eigenval_gradient = get_eigenval_gradient_log_weighted_average(
          interp_params.exponential_decay_c, left_matrix(i, i), left_matrix_loc(0, 0),
          right_matrix(i, i), right_matrix_loc(0, 0), interp_loc(0, 0));


      // compare interpolation gradient with reference solution
      EXPECT_NEAR(interp_gradient_standard(i, 0), ref_eigenval_gradient, 1.0e-4);
      EXPECT_NEAR(interp_gradient_specialized(i, 0), ref_eigenval_gradient, 1.0e-4);
    }
  }


  TEST(LinalgTensorInterpolationTest, SymmetricSPDInterpolationDiagonal)
  {
    {
      const std::array<Core::LinAlg::SymmetricTensor<double, 3, 3>, 3> all_same_tensors = {
          Core::LinAlg::TensorGenerators::diagonal(std::array{10.0, 0.5, 2.0}),
          Core::LinAlg::TensorGenerators::diagonal(std::array{10.0, 0.5, 2.0}),
          Core::LinAlg::TensorGenerators::diagonal(std::array{10.0, 0.5, 2.0})};

      const std::array<double, 3> weights = {0.1, 0.5, 0.4};

      const Core::LinAlg::SymmetricTensor<double, 3, 3> interpolated_tensor =
          Core::LinAlg::interpolate_spd(weights, all_same_tensors);

      // The result should be the same as the input tensors
      FOUR_C_EXPECT_NEAR(interpolated_tensor, all_same_tensors[0], 1.0e-12);
    }

    {
      const std::array<Core::LinAlg::SymmetricTensor<double, 3, 3>, 3> ref_tensors = {
          Core::LinAlg::TensorGenerators::diagonal(std::array{10.0, 0.5, 2.0}),
          Core::LinAlg::TensorGenerators::diagonal(std::array{12.0, 0.4, 3.0}),
          Core::LinAlg::TensorGenerators::diagonal(std::array{13.0, 0.3, 4.0})};

      const std::array<double, 3> weights = {0.1, 0.5, 0.4};

      const Core::LinAlg::SymmetricTensor<double, 3, 3> interpolated_tensor =
          Core::LinAlg::interpolate_spd(weights, ref_tensors);

      // This should still be diagonal, with log-interpolated eigenvalues
      FOUR_C_EXPECT_NEAR(interpolated_tensor,
          Core::LinAlg::TensorGenerators::diagonal(std::array{12.16656453, 0.36456544, 3.23212109}),
          1.0e-7);
    }
  }



  TEST(LinalgTensorInterpolationTest, SymmetricSPDInterpolationPureRotation)
  {
    const std::array<Core::LinAlg::SymmetricTensor<double, 3, 3>, 3> tensors = {
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{
            {{0.7895765126664778, 1.013576418177698, 1.1044280206330133},
                {1.013576418177698, 1.4862383036989457, 1.564094529544559},
                {1.1044280206330133, 1.564094529544559, 1.7091488088039124}}}),
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{
            {{0.8431577941808065, 0.9106647034217905, 1.25714256472096},
                {0.9106647034217906, 1.1355705284667736, 1.465546721583386},
                {1.25714256472096, 1.4655467215833862, 2.0062353025217563}}}),
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{
            {{0.35134275260569925, 0.7419684187562707, 0.7016200682920156},
                {0.7419684187562707, 1.9807063716625941, 1.7714631959766316},
                {0.7016200682920154, 1.7714631959766318, 1.6529145009010444}}}),
    };

    const std::array<double, 3> weights = {0.2, 0.5, 0.3};

    const Core::LinAlg::SymmetricTensor<double, 3, 3> interpolated_tensor =
        Core::LinAlg::interpolate_spd(weights, tensors);

    FOUR_C_EXPECT_NEAR(interpolated_tensor,
        (Core::LinAlg::Tensor<double, 3, 3>{{
            {0.674, 0.916, 1.070},
            {0.916, 1.451, 1.605},
            {1.070, 1.605, 1.860},
        }}),
        1.0e-3);
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE
