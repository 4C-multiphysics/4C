// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_utils_densematrix_inverse.hpp"

#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

/*
 * \note The values for the matrix used in tests below are generated with Mathematica:
 *       > SeedRandom[666];
 *       > A = Table[RandomReal[WorkingPrecision->50], {i, n}, {j, n}];
 *       where n needs do be replace by the dimension, e.g., n=2, n=3, or n=4
 */

namespace
{
  TEST(LinalgDenseMatrixInverseTest, 2x2InverseReorder)
  {
    Core::LinAlg::Matrix<2, 2, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.81862230026150939335;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.0052737129228371719370;

    Core::LinAlg::Matrix<2, 2, double> B(A);

    Core::LinAlg::inverse_reorder_matrix_entries(A);

    EXPECT_NEAR(A(0, 0), B(1, 1), 1e-14);
    EXPECT_NEAR(A(1, 1), B(0, 0), 1e-14);
    EXPECT_NEAR(A(0, 1), -B(0, 1), 1e-14);
    EXPECT_NEAR(A(1, 0), -B(1, 0), 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 2x2Inverse)
  {
    Core::LinAlg::Matrix<2, 2, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.81862230026150939335;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.0052737129228371719370;

    Core::LinAlg::inverse(A);

    EXPECT_NEAR(A(0, 0), -0.019983345434747187407, 1e-14);
    EXPECT_NEAR(A(1, 0), 3.1019534900872646712, 1e-14);
    EXPECT_NEAR(A(0, 1), 1.2393609438018440619, 1e-14);
    EXPECT_NEAR(A(1, 1), -2.7624762444413145470, 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 2x2InverseSingular)
  {
    Core::LinAlg::Matrix<2, 2, double> A_singular(Core::LinAlg::Initialization::zero);
    A_singular(0, 0) = 1.0;
    EXPECT_ANY_THROW(Core::LinAlg::inverse(A_singular));
  }

  TEST(LinalgDenseMatrixInverseTest, 2x2InverseDoNotThrowErrorOnZeroDeterminant)
  {
    Core::LinAlg::Matrix<2, 2, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.81862230026150939335;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.0052737129228371719370;

    EXPECT_TRUE(Core::LinAlg::inverse_do_not_throw_error_on_zero_determinant(A, 1e-12));

    EXPECT_NEAR(A(0, 0), -0.019983345434747187407, 1e-14);
    EXPECT_NEAR(A(1, 0), 3.1019534900872646712, 1e-14);
    EXPECT_NEAR(A(0, 1), 1.2393609438018440619, 1e-14);
    EXPECT_NEAR(A(1, 1), -2.7624762444413145470, 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 2x2InverseDoNotThrowErrorOnZeroDeterminantSingular)
  {
    Core::LinAlg::Matrix<2, 2, double> A_singular(Core::LinAlg::Initialization::zero);
    A_singular(0, 0) = 1.0;
    EXPECT_TRUE(
        not Core::LinAlg::inverse_do_not_throw_error_on_zero_determinant(A_singular, 1e-12));
  }

  TEST(LinalgDenseMatrixInverseTest, 3x3InverseReorder)
  {
    Core::LinAlg::Matrix<3, 3, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.0052737129228371719370;
    A(2, 0) = 0.36847164343389089096;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.87570663114228933311;
    A(2, 1) = 0.76895132151127114661;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.64019842179333806573;
    A(2, 2) = 0.69378923027976465858;

    Core::LinAlg::Matrix<3, 3, double> B(A);

    Core::LinAlg::inverse_reorder_matrix_entries(A);

    EXPECT_NEAR(A(0, 0), B(1, 1) * B(2, 2) - B(2, 1) * B(1, 2), 1e-14);
    EXPECT_NEAR(A(1, 0), -B(1, 0) * B(2, 2) + B(2, 0) * B(1, 2), 1e-14);
    EXPECT_NEAR(A(2, 0), B(1, 0) * B(2, 1) - B(2, 0) * B(1, 1), 1e-14);
    EXPECT_NEAR(A(0, 1), -B(0, 1) * B(2, 2) + B(2, 1) * B(0, 2), 1e-14);
    EXPECT_NEAR(A(1, 1), B(0, 0) * B(2, 2) - B(2, 0) * B(0, 2), 1e-14);
    EXPECT_NEAR(A(2, 1), -B(0, 0) * B(2, 1) + B(2, 0) * B(0, 1), 1e-14);
    EXPECT_NEAR(A(0, 2), B(0, 1) * B(1, 2) - B(1, 1) * B(0, 2), 1e-14);
    EXPECT_NEAR(A(1, 2), -B(0, 0) * B(1, 2) + B(1, 0) * B(0, 2), 1e-14);
    EXPECT_NEAR(A(2, 2), B(0, 0) * B(1, 1) - B(1, 0) * B(0, 1), 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 3x3Inverse)
  {
    Core::LinAlg::Matrix<3, 3, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.0052737129228371719370;
    A(2, 0) = 0.36847164343389089096;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.87570663114228933311;
    A(2, 1) = 0.76895132151127114661;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.64019842179333806573;
    A(2, 2) = 0.69378923027976465858;

    Core::LinAlg::inverse(A);

    EXPECT_NEAR(A(0, 0), -1.1432496777455423383, 1e-14);
    EXPECT_NEAR(A(1, 0), -2.3032334349351217883, 1e-14);
    EXPECT_NEAR(A(2, 0), 3.1599358789014842560, 1e-14);
    EXPECT_NEAR(A(0, 1), -3.9924461924239508629, 1e-14);
    EXPECT_NEAR(A(1, 1), -2.0247424048120138037, 1e-14);
    EXPECT_NEAR(A(2, 1), 4.3644833698599101987, 1e-14);
    EXPECT_NEAR(A(0, 2), 5.0330089889776210806, 1e-14);
    EXPECT_NEAR(A(1, 2), 4.5859967347165455115, 1e-14);
    EXPECT_NEAR(A(2, 2), -6.3144960342296159564, 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 3x3InverseSingular)
  {
    Core::LinAlg::Matrix<3, 3, double> A_singular(Core::LinAlg::Initialization::zero);
    A_singular(0, 0) = 1.0;
    A_singular(1, 1) = 1.0;
    EXPECT_ANY_THROW(Core::LinAlg::inverse(A_singular));
  }

  TEST(LinalgDenseMatrixInverseTest, 3x3InverseDoNotThrowErrorOnZeroDeterminant)
  {
    Core::LinAlg::Matrix<3, 3, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.0052737129228371719370;
    A(2, 0) = 0.36847164343389089096;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.87570663114228933311;
    A(2, 1) = 0.76895132151127114661;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.64019842179333806573;
    A(2, 2) = 0.69378923027976465858;

    EXPECT_TRUE(Core::LinAlg::inverse_do_not_throw_error_on_zero_determinant(A, 1e-12));

    EXPECT_NEAR(A(0, 0), -1.1432496777455423383, 1e-14);
    EXPECT_NEAR(A(1, 0), -2.3032334349351217883, 1e-14);
    EXPECT_NEAR(A(2, 0), 3.1599358789014842560, 1e-14);
    EXPECT_NEAR(A(0, 1), -3.9924461924239508629, 1e-14);
    EXPECT_NEAR(A(1, 1), -2.0247424048120138037, 1e-14);
    EXPECT_NEAR(A(2, 1), 4.3644833698599101987, 1e-14);
    EXPECT_NEAR(A(0, 2), 5.0330089889776210806, 1e-14);
    EXPECT_NEAR(A(1, 2), 4.5859967347165455115, 1e-14);
    EXPECT_NEAR(A(2, 2), -6.3144960342296159564, 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 3x3InverseDoNotThrowErrorOnZeroDeterminantSingular)
  {
    Core::LinAlg::Matrix<3, 3, double> A_singular(Core::LinAlg::Initialization::zero);
    A_singular(0, 0) = 1.0;
    A_singular(1, 1) = 1.0;
    EXPECT_TRUE(
        not Core::LinAlg::inverse_do_not_throw_error_on_zero_determinant(A_singular, 1e-12));
  }

  TEST(LinalgDenseMatrixInverseTest, 4x4InverseReorder)
  {
    Core::LinAlg::Matrix<4, 4, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.87570663114228933311;
    A(2, 0) = 0.69378923027976465858;
    A(3, 0) = 0.019637190415090362652;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.64019842179333806573;
    A(2, 1) = 0.15928293569477706215;
    A(3, 1) = 0.13119201434024140151;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.36847164343389089096;
    A(2, 2) = 0.12278929762839221138;
    A(3, 2) = 0.12028240083390837511;
    A(0, 3) = 0.0052737129228371719370;
    A(1, 3) = 0.76895132151127114661;
    A(2, 3) = 0.024003735765356129168;
    A(3, 3) = 0.27465069811053651449;

    Core::LinAlg::inverse_reorder_matrix_entries(A);

    EXPECT_NEAR(A(0, 0), 0.0071277916600247445689, 1e-14);
    EXPECT_NEAR(A(1, 0), -0.01928098922790522779, 1e-14);
    EXPECT_NEAR(A(2, 0), -0.018551379633640403821, 1e-14);
    EXPECT_NEAR(A(3, 0), 0.016824812484074490315, 1e-14);
    EXPECT_NEAR(A(0, 1), 0.023132438895313536809, 1e-14);
    EXPECT_NEAR(A(1, 1), -0.13269385313439016616, 1e-14);
    EXPECT_NEAR(A(2, 1), 0.032108864599905515003, 1e-14);
    EXPECT_NEAR(A(3, 1), 0.047667770531278727542, 1e-14);
    EXPECT_NEAR(A(0, 2), -0.058356413033299463222, 1e-14);
    EXPECT_NEAR(A(1, 2), 0.17766145788899323499, 1e-14);
    EXPECT_NEAR(A(2, 2), -0.018545829326959681338, 1e-14);
    EXPECT_NEAR(A(3, 2), -0.072568800285664356031, 1e-14);
    EXPECT_NEAR(A(0, 3), -0.059801550040274355224, 1e-14);
    EXPECT_NEAR(A(1, 3), 0.35635175191115986415, 1e-14);
    EXPECT_NEAR(A(2, 3), -0.087919492632716270131, 1e-14);
    EXPECT_NEAR(A(3, 3), -0.18645052203295775506, 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 4x4Inverse)
  {
    Core::LinAlg::Matrix<4, 4, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.87570663114228933311;
    A(2, 0) = 0.69378923027976465858;
    A(3, 0) = 0.019637190415090362652;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.64019842179333806573;
    A(2, 1) = 0.15928293569477706215;
    A(3, 1) = 0.13119201434024140151;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.36847164343389089096;
    A(2, 2) = 0.12278929762839221138;
    A(3, 2) = 0.12028240083390837511;
    A(0, 3) = 0.0052737129228371719370;
    A(1, 3) = 0.76895132151127114661;
    A(2, 3) = 0.024003735765356129168;
    A(3, 3) = 0.27465069811053651449;

    Core::LinAlg::inverse(A);

    EXPECT_NEAR(A(0, 0), -0.43977637337572104289, 1e-14);
    EXPECT_NEAR(A(1, 0), 1.1896143886050492292, 1e-14);
    EXPECT_NEAR(A(2, 0), 1.1445983336121054974, 1e-14);
    EXPECT_NEAR(A(3, 0), -1.0380711684475831990, 1e-14);
    EXPECT_NEAR(A(0, 1), -1.4272443093098390369, 1e-14);
    EXPECT_NEAR(A(1, 1), 8.1870548809629493769, 1e-14);
    EXPECT_NEAR(A(2, 1), -1.9810792318962757373, 1e-14);
    EXPECT_NEAR(A(3, 1), -2.9410454529305114157, 1e-14);
    EXPECT_NEAR(A(0, 2), 3.6005221408100166266, 1e-14);
    EXPECT_NEAR(A(1, 2), -10.961503276990213139, 1e-14);
    EXPECT_NEAR(A(2, 2), 1.1442558862090945693, 1e-14);
    EXPECT_NEAR(A(3, 2), 4.4774097409218651830, 1e-14);
    EXPECT_NEAR(A(0, 3), 3.6896853967343974819, 1e-14);
    EXPECT_NEAR(A(1, 3), -21.986484534963294880, 1e-14);
    EXPECT_NEAR(A(2, 3), 5.4245294283636667402, 1e-14);
    EXPECT_NEAR(A(3, 3), 11.503778211354084434, 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 4x4InverseSingular)
  {
    Core::LinAlg::Matrix<4, 4, double> A_singular(Core::LinAlg::Initialization::zero);
    A_singular(0, 0) = 1.0;
    A_singular(1, 1) = 1.0;
    A_singular(2, 2) = 1.0;
    EXPECT_ANY_THROW(Core::LinAlg::inverse(A_singular));
  }

  TEST(LinalgDenseMatrixInverseTest, 4x4InverseDoNotThrowErrorOnZeroDeterminant)
  {
    Core::LinAlg::Matrix<4, 4, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.87570663114228933311;
    A(2, 0) = 0.69378923027976465858;
    A(3, 0) = 0.019637190415090362652;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.64019842179333806573;
    A(2, 1) = 0.15928293569477706215;
    A(3, 1) = 0.13119201434024140151;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.36847164343389089096;
    A(2, 2) = 0.12278929762839221138;
    A(3, 2) = 0.12028240083390837511;
    A(0, 3) = 0.0052737129228371719370;
    A(1, 3) = 0.76895132151127114661;
    A(2, 3) = 0.024003735765356129168;
    A(3, 3) = 0.27465069811053651449;

    EXPECT_TRUE(Core::LinAlg::inverse_do_not_throw_error_on_zero_determinant(A, 1e-12));

    EXPECT_NEAR(A(0, 0), -0.43977637337572104289, 1e-14);
    EXPECT_NEAR(A(1, 0), 1.1896143886050492292, 1e-14);
    EXPECT_NEAR(A(2, 0), 1.1445983336121054974, 1e-14);
    EXPECT_NEAR(A(3, 0), -1.0380711684475831990, 1e-14);
    EXPECT_NEAR(A(0, 1), -1.4272443093098390369, 1e-14);
    EXPECT_NEAR(A(1, 1), 8.1870548809629493769, 1e-14);
    EXPECT_NEAR(A(2, 1), -1.9810792318962757373, 1e-14);
    EXPECT_NEAR(A(3, 1), -2.9410454529305114157, 1e-14);
    EXPECT_NEAR(A(0, 2), 3.6005221408100166266, 1e-14);
    EXPECT_NEAR(A(1, 2), -10.961503276990213139, 1e-14);
    EXPECT_NEAR(A(2, 2), 1.1442558862090945693, 1e-14);
    EXPECT_NEAR(A(3, 2), 4.4774097409218651830, 1e-14);
    EXPECT_NEAR(A(0, 3), 3.6896853967343974819, 1e-14);
    EXPECT_NEAR(A(1, 3), -21.986484534963294880, 1e-14);
    EXPECT_NEAR(A(2, 3), 5.4245294283636667402, 1e-14);
    EXPECT_NEAR(A(3, 3), 11.503778211354084434, 1e-14);
  }

  TEST(LinalgDenseMatrixInverseTest, 4x4InverseDoNotThrowErrorOnZeroDeterminantSingular)
  {
    Core::LinAlg::Matrix<4, 4, double> A_singular(Core::LinAlg::Initialization::zero);
    A_singular(0, 0) = 1.0;
    A_singular(1, 1) = 1.0;
    A_singular(2, 2) = 1.0;
    EXPECT_TRUE(
        not Core::LinAlg::inverse_do_not_throw_error_on_zero_determinant(A_singular, 1e-12));
  }

  TEST(LinalgDenseMatrixInverseTest, 4x4InverseDoNotThrowErrorOnZeroDeterminantScaled)
  {
    Core::LinAlg::Matrix<4, 4, double> A(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<4, 1, double> b(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<4, 1, double> x(Core::LinAlg::Initialization::zero);
    const double eps_det = 1e-10;
    const double eps_result = 1e-12;

    // Create an invertible diagonal matrix with an unscaled determinant of about 2e-11.
    for (unsigned int i = 0; i < 4; i++)
    {
      A(i, i) = 0.001 * (i + 1);
      b(i) = 0.1 * (i + 1);
    }

    // Solve the system by scaling the system matrix and compare the results.
    EXPECT_TRUE(Core::LinAlg::solve_linear_system_do_not_throw_error_on_zero_determinant_scaled(
        A, b, x, eps_det));
    EXPECT_NEAR(x(0), 100.0, eps_result);
    EXPECT_NEAR(x(1), 100.0, eps_result);
    EXPECT_NEAR(x(2), 100.0, eps_result);
    EXPECT_NEAR(x(3), 100.0, eps_result);
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE
