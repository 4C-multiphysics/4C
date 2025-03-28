// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_derivatives.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_mat_service.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

namespace
{
  using namespace FourC;

  TEST(MaterialServiceTest, TestInvariantsPrincipal)
  {
    Core::LinAlg::Matrix<3, 3> sym_tensor(Core::LinAlg::Initialization::uninitialized);
    sym_tensor(0, 0) = 1.1;
    sym_tensor(1, 1) = 1.2;
    sym_tensor(2, 2) = 1.3;
    sym_tensor(0, 1) = sym_tensor(1, 0) = 0.01;
    sym_tensor(1, 2) = sym_tensor(2, 1) = 0.02;
    sym_tensor(0, 2) = sym_tensor(2, 0) = 0.03;

    Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::uninitialized);
    Mat::invariants_principal(prinv, sym_tensor);

    Core::LinAlg::Matrix<3, 1> prinv_reference(Core::LinAlg::Initialization::uninitialized);
    prinv_reference(0) = 3.5999999999999996;
    prinv_reference(1) = 4.3085999999999984;
    prinv_reference(2) = 1.7143620000000002;

    FOUR_C_EXPECT_NEAR(prinv, prinv_reference, 1.0e-10);
  }

  TEST(MaterialServiceTest, Testadd_derivative_of_inva_b_inva_product)
  {
    Core::LinAlg::Matrix<6, 1> A(Core::LinAlg::Initialization::uninitialized);
    A(0) = 0.5;
    A(1) = 0.3;
    A(2) = 0.6;
    A(3) = 0.2;
    A(4) = 0.4;
    A(5) = 0.9;

    Core::LinAlg::Matrix<6, 1> InvABInvB(Core::LinAlg::Initialization::uninitialized);
    InvABInvB(0) = 1.72;
    InvABInvB(1) = 1.65;
    InvABInvB(2) = 1.13;
    InvABInvB(3) = 1.27;
    InvABInvB(4) = 1.46;
    InvABInvB(5) = 1.23;

    Core::LinAlg::Matrix<6, 6> Result(Core::LinAlg::Initialization::zero);
    double scalar = 0.5;

    // result_ijkl = A_ik InvABInvB_jl +  A_il InvABInvB_jk + A_jk InvABInvB_il + A_jl InvABInvB_ik
    Core::LinAlg::Tensor::add_derivative_of_inva_b_inva_product(scalar, A, InvABInvB, Result);

    Core::LinAlg::Matrix<6, 6> Result_reference(Core::LinAlg::Initialization::uninitialized);
    Result_reference(0, 0) = -0.86;
    Result_reference(0, 1) = -0.254;
    Result_reference(0, 2) = -1.107;
    Result_reference(0, 3) = -0.4895;
    Result_reference(0, 4) = -0.6945;
    Result_reference(0, 5) = -1.0815;
    Result_reference(1, 0) = -0.254;
    Result_reference(1, 1) = -0.495;
    Result_reference(1, 2) = -0.584;
    Result_reference(1, 3) = -0.3555;
    Result_reference(1, 4) = -0.549;
    Result_reference(1, 5) = -0.40;
    Result_reference(2, 0) = -1.107;
    Result_reference(2, 1) = -0.584;
    Result_reference(2, 2) = -0.678;
    Result_reference(2, 3) = -0.903;
    Result_reference(2, 4) = -0.664;
    Result_reference(2, 5) = -0.8775;
    Result_reference(3, 0) = -0.4895;
    Result_reference(3, 1) = -0.3555;
    Result_reference(3, 2) = -0.903;
    Result_reference(3, 3) = -0.46225;
    Result_reference(3, 4) = -0.6635;
    Result_reference(3, 5) = -0.70175;
    Result_reference(4, 0) = -0.6945;
    Result_reference(4, 1) = -0.549;
    Result_reference(4, 2) = -0.664;
    Result_reference(4, 3) = -0.6635;
    Result_reference(4, 4) = -0.62425;
    Result_reference(4, 5) = -0.6985;
    Result_reference(5, 0) = -1.0815;
    Result_reference(5, 1) = -0.40;
    Result_reference(5, 2) = -0.8775;
    Result_reference(5, 3) = -0.70175;
    Result_reference(5, 4) = -0.6985;
    Result_reference(5, 5) = -0.95275;

    FOUR_C_EXPECT_NEAR(Result, Result_reference, 1.0e-10);
  }

  TEST(MaterialServiceTest, TestComputeJ2)
  {
    // test the calculation of J2 invariant
    Core::LinAlg::Matrix<3, 3> stress(Core::LinAlg::Initialization::uninitialized);
    stress(0, 0) = 1.0;
    stress(1, 1) = 2.0;
    stress(2, 2) = 3.0;
    stress(0, 1) = stress(1, 0) = 0.1;
    stress(1, 2) = stress(2, 1) = 0.0;
    stress(0, 2) = stress(2, 0) = 0.0;
    const double j2 = Mat::second_invariant_of_deviatoric_stress(stress);
    const double j2_ref = 0.5 * (2 + 2 * 0.1 * 0.1);
    EXPECT_NEAR(j2, j2_ref, 1.0e-10);
  }

  TEST(MaterialServiceTest, TestElasticTensor)
  {
    // test the calculation of fourth order elastic tensor and subroutine to transform 4th order
    // tensor to Matrix
    Core::LinAlg::FourTensor<3> Ce;
    const double E = 2.0;
    const double NU = 0.3;
    Mat::setup_linear_isotropic_elastic_tensor(Ce, E, NU);

    Core::LinAlg::Matrix<6, 6> De;
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(De, Ce);

    Core::LinAlg::Matrix<6, 6> De_ref;
    const double c1 = E / ((1.00 + NU) * (1 - 2 * NU));
    const double c2 = c1 * (1 - NU);
    const double c3 = c1 * NU;
    const double c4 = c1 * 0.5 * (1 - 2 * NU);

    De_ref(0, 0) = c2;
    De_ref(0, 1) = c3;
    De_ref(0, 2) = c3;
    De_ref(0, 3) = 0.;
    De_ref(0, 4) = 0.;
    De_ref(0, 5) = 0.;
    De_ref(1, 0) = c3;
    De_ref(1, 1) = c2;
    De_ref(1, 2) = c3;
    De_ref(1, 3) = 0.;
    De_ref(1, 4) = 0.;
    De_ref(1, 5) = 0.;
    De_ref(2, 0) = c3;
    De_ref(2, 1) = c3;
    De_ref(2, 2) = c2;
    De_ref(2, 3) = 0.;
    De_ref(2, 4) = 0.;
    De_ref(2, 5) = 0.;
    De_ref(3, 0) = 0.;
    De_ref(3, 1) = 0.;
    De_ref(3, 2) = 0.;
    De_ref(3, 3) = c4;
    De_ref(3, 4) = 0.;
    De_ref(3, 5) = 0.;
    De_ref(4, 0) = 0.;
    De_ref(4, 1) = 0.;
    De_ref(4, 2) = 0.;
    De_ref(4, 3) = 0.;
    De_ref(4, 4) = c4;
    De_ref(4, 5) = 0.;
    De_ref(5, 0) = 0.;
    De_ref(5, 1) = 0.;
    De_ref(5, 2) = 0.;
    De_ref(5, 3) = 0.;
    De_ref(5, 4) = 0.;
    De_ref(5, 5) = c4;

    FOUR_C_EXPECT_NEAR(De, De_ref, 1.0e-10);
  }

  TEST(MaterialServiceTest, TestDeviatoricTensor)
  {
    // test the calculation of fourth order deviatoric tensor and contraction of fourth order tensor
    // and Matrix
    Core::LinAlg::FourTensor<3> Id = Core::LinAlg::setup_deviatoric_projection_tensor<3>();

    Core::LinAlg::Matrix<3, 3> stress(Core::LinAlg::Initialization::uninitialized);
    stress(0, 0) = 1.0;
    stress(1, 1) = 2.0;
    stress(2, 2) = 3.0;
    stress(0, 1) = stress(1, 0) = 0.4;
    stress(1, 2) = stress(2, 1) = 0.5;
    stress(0, 2) = stress(2, 0) = 0.6;

    Core::LinAlg::Matrix<3, 3> s_ref(Core::LinAlg::Initialization::uninitialized);
    s_ref(0, 0) = -1.0;
    s_ref(1, 1) = 0.0;
    s_ref(2, 2) = 1.0;
    s_ref(0, 1) = s_ref(1, 0) = 0.4;
    s_ref(1, 2) = s_ref(2, 1) = 0.5;
    s_ref(0, 2) = s_ref(2, 0) = 0.6;

    Core::LinAlg::Matrix<3, 3> s;
    Core::LinAlg::Tensor::add_contraction_matrix_four_tensor(s, 1.0, Id, stress);

    FOUR_C_EXPECT_NEAR(s, s_ref, 1.0e-10);
  }



  TEST(MaterialServiceTest, TestADBCProduct)
  {
    // test the calculation of the fourth order ADBC product of two second-order tensors
    Core::LinAlg::Matrix<3, 3> x(Core::LinAlg::Initialization::uninitialized);
    x(0, 0) = 1.0000000000;
    x(0, 1) = 2.0000000000;
    x(0, 2) = 3.0000000000;
    x(1, 0) = 0.0000000000;
    x(1, 1) = 5.0000000000;
    x(1, 2) = 1.2000000000;
    x(2, 0) = 1.0000000000;
    x(2, 1) = 1.0000000000;
    x(2, 2) = 1.0000000000;

    Core::LinAlg::Matrix<3, 3> y(Core::LinAlg::Initialization::uninitialized);
    y(0, 0) = 1.0000000000;
    y(0, 1) = 0.0000000000;
    y(0, 2) = 5.0000000000;
    y(1, 0) = 0.0000000000;
    y(1, 1) = 0.0000000000;
    y(1, 2) = 7.0000000000;
    y(2, 0) = 1.3000000000;
    y(2, 1) = 1.2000000000;
    y(2, 2) = 1.0000000000;

    Core::LinAlg::Matrix<9, 9> x_adbc_y_ref(Core::LinAlg::Initialization::zero);
    x_adbc_y_ref(0, 0) = 1.0000000000;
    x_adbc_y_ref(0, 1) = 0.0000000000;
    x_adbc_y_ref(0, 2) = 15.0000000000;
    x_adbc_y_ref(0, 3) = 2.0000000000;
    x_adbc_y_ref(0, 4) = 0.0000000000;
    x_adbc_y_ref(0, 5) = 3.0000000000;
    x_adbc_y_ref(0, 6) = 0.0000000000;
    x_adbc_y_ref(0, 7) = 10.0000000000;
    x_adbc_y_ref(0, 8) = 5.0000000000;
    x_adbc_y_ref(1, 0) = 0.0000000000;
    x_adbc_y_ref(1, 1) = 0.0000000000;
    x_adbc_y_ref(1, 2) = 8.4000000000;
    x_adbc_y_ref(1, 3) = 0.0000000000;
    x_adbc_y_ref(1, 4) = 0.0000000000;
    x_adbc_y_ref(1, 5) = 0.0000000000;
    x_adbc_y_ref(1, 6) = 0.0000000000;
    x_adbc_y_ref(1, 7) = 35.0000000000;
    x_adbc_y_ref(1, 8) = 0.0000000000;
    x_adbc_y_ref(2, 0) = 1.3000000000;
    x_adbc_y_ref(2, 1) = 1.2000000000;
    x_adbc_y_ref(2, 2) = 1.0000000000;
    x_adbc_y_ref(2, 3) = 1.3000000000;
    x_adbc_y_ref(2, 4) = 1.2000000000;
    x_adbc_y_ref(2, 5) = 1.3000000000;
    x_adbc_y_ref(2, 6) = 1.2000000000;
    x_adbc_y_ref(2, 7) = 1.0000000000;
    x_adbc_y_ref(2, 8) = 1.0000000000;
    x_adbc_y_ref(3, 0) = 0.0000000000;
    x_adbc_y_ref(3, 1) = 0.0000000000;
    x_adbc_y_ref(3, 2) = 21.0000000000;
    x_adbc_y_ref(3, 3) = 0.0000000000;
    x_adbc_y_ref(3, 4) = 0.0000000000;
    x_adbc_y_ref(3, 5) = 0.0000000000;
    x_adbc_y_ref(3, 6) = 0.0000000000;
    x_adbc_y_ref(3, 7) = 14.0000000000;
    x_adbc_y_ref(3, 8) = 7.0000000000;
    x_adbc_y_ref(4, 0) = 0.0000000000;
    x_adbc_y_ref(4, 1) = 6.0000000000;
    x_adbc_y_ref(4, 2) = 1.2000000000;
    x_adbc_y_ref(4, 3) = 6.5000000000;
    x_adbc_y_ref(4, 4) = 1.4400000000;
    x_adbc_y_ref(4, 5) = 1.5600000000;
    x_adbc_y_ref(4, 6) = 0.0000000000;
    x_adbc_y_ref(4, 7) = 5.0000000000;
    x_adbc_y_ref(4, 8) = 0.0000000000;
    x_adbc_y_ref(5, 0) = 1.3000000000;
    x_adbc_y_ref(5, 1) = 2.4000000000;
    x_adbc_y_ref(5, 2) = 3.0000000000;
    x_adbc_y_ref(5, 3) = 2.6000000000;
    x_adbc_y_ref(5, 4) = 3.6000000000;
    x_adbc_y_ref(5, 5) = 3.9000000000;
    x_adbc_y_ref(5, 6) = 1.2000000000;
    x_adbc_y_ref(5, 7) = 2.0000000000;
    x_adbc_y_ref(5, 8) = 1.0000000000;
    x_adbc_y_ref(6, 0) = 0.0000000000;
    x_adbc_y_ref(6, 1) = 0.0000000000;
    x_adbc_y_ref(6, 2) = 6.0000000000;
    x_adbc_y_ref(6, 3) = 5.0000000000;
    x_adbc_y_ref(6, 4) = 0.0000000000;
    x_adbc_y_ref(6, 5) = 1.2000000000;
    x_adbc_y_ref(6, 6) = 0.0000000000;
    x_adbc_y_ref(6, 7) = 25.0000000000;
    x_adbc_y_ref(6, 8) = 0.0000000000;
    x_adbc_y_ref(7, 0) = 0.0000000000;
    x_adbc_y_ref(7, 1) = 0.0000000000;
    x_adbc_y_ref(7, 2) = 7.0000000000;
    x_adbc_y_ref(7, 3) = 0.0000000000;
    x_adbc_y_ref(7, 4) = 0.0000000000;
    x_adbc_y_ref(7, 5) = 0.0000000000;
    x_adbc_y_ref(7, 6) = 0.0000000000;
    x_adbc_y_ref(7, 7) = 7.0000000000;
    x_adbc_y_ref(7, 8) = 7.0000000000;
    x_adbc_y_ref(8, 0) = 1.0000000000;
    x_adbc_y_ref(8, 1) = 0.0000000000;
    x_adbc_y_ref(8, 2) = 5.0000000000;
    x_adbc_y_ref(8, 3) = 1.0000000000;
    x_adbc_y_ref(8, 4) = 0.0000000000;
    x_adbc_y_ref(8, 5) = 1.0000000000;
    x_adbc_y_ref(8, 6) = 0.0000000000;
    x_adbc_y_ref(8, 7) = 5.0000000000;
    x_adbc_y_ref(8, 8) = 5.0000000000;

    Core::LinAlg::Matrix<9, 9> x_adbc_y(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Tensor::add_adbc_tensor_product(1.0, x, y, x_adbc_y);

    FOUR_C_EXPECT_NEAR(x_adbc_y, x_adbc_y_ref, 1.0e-10);
  }

  TEST(MaterialServiceTest, TestDyadicProductMatrixMatrix)
  {
    // test the dyadic product of two Matrices
    Core::LinAlg::Matrix<3, 3> stress(Core::LinAlg::Initialization::uninitialized);
    stress(0, 0) = 1.0;
    stress(1, 1) = 2.0;
    stress(2, 2) = 3.0;
    stress(0, 1) = stress(1, 0) = 0.4;
    stress(1, 2) = stress(2, 1) = 0.5;
    stress(0, 2) = stress(2, 0) = 0.6;

    Core::LinAlg::FourTensor<3> T;
    Core::LinAlg::Tensor::add_dyadic_product_matrix_matrix(T, 1.0, stress, stress);

    Core::LinAlg::Matrix<3, 3> result;
    Core::LinAlg::Tensor::add_contraction_matrix_four_tensor(result, 1.0, T, stress);

    Core::LinAlg::Matrix<3, 3> result_ref = stress;
    result_ref.scale(std::pow(stress.norm2(), 2));

    FOUR_C_EXPECT_NEAR(result, result_ref, 1.0e-10);
  }

}  // namespace
