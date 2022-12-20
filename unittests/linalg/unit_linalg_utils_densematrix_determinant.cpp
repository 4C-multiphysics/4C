/*----------------------------------------------------------------------*/
/*! \file

\brief Unit test for linalg dense determinant calculation routines.

\level 0

*----------------------------------------------------------------------*/
#include <gtest/gtest.h>

#include "linalg_serialdensematrix.H"
#include "linalg_utils_densematrix_determinant.H"

/*
 * \note The values for the matrix used in tests below are generated with Mathematica:
 *       > SeedRandom[666];
 *       > A = Table[RandomReal[WorkingPrecision->50], {i, n}, {j, n}];
 *       where n needs do be replace by the dimension, e.g., n=2, n=3, or n=4
 */

namespace
{
  TEST(LinalgDenseMatrixDeterminantTest, 2x2Determinant)
  {
    LINALG::Matrix<2, 2, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.81862230026150939335;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.0052737129228371719370;

    EXPECT_NEAR(LINALG::Determinant(A), -0.2639054076334587, 1e-14);
  }

  TEST(LinalgDenseMatrixDeterminantTest, 3x3Determinant)
  {
    LINALG::Matrix<3, 3, double> A;
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.0052737129228371719370;
    A(2, 0) = 0.36847164343389089096;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.87570663114228933311;
    A(2, 1) = 0.76895132151127114661;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.64019842179333806573;
    A(2, 2) = 0.69378923027976465858;

    EXPECT_NEAR(LINALG::Determinant(A), -0.1008304741716571, 1e-14);
  }

  TEST(LinalgDenseMatrixDeterminantTest, 4x4Determinant)
  {
    LINALG::Matrix<4, 4, double> A;
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

    EXPECT_NEAR(LINALG::Determinant(A), -0.01620776397174742, 1e-14);
  }

  TEST(LinalgDenseMatrixDeterminantTest, 2x2DeterminantLU)
  {
    LINALG::SerialDenseMatrix A(2, 2, true);
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.81862230026150939335;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.0052737129228371719370;

    EXPECT_NEAR(LINALG::DeterminantLU(A), -0.2639054076334587, 1e-14);
  }

  TEST(LinalgDenseMatrixDeterminantTest, 3x3DeterminantLU)
  {
    LINALG::SerialDenseMatrix A(3, 3, true);
    A(0, 0) = 0.72903241936703114203;
    A(1, 0) = 0.0052737129228371719370;
    A(2, 0) = 0.36847164343389089096;
    A(0, 1) = 0.32707405507901372465;
    A(1, 1) = 0.87570663114228933311;
    A(2, 1) = 0.76895132151127114661;
    A(0, 2) = 0.81862230026150939335;
    A(1, 2) = 0.64019842179333806573;
    A(2, 2) = 0.69378923027976465858;

    EXPECT_NEAR(LINALG::DeterminantLU(A), -0.1008304741716571, 1e-14);
  }

  TEST(LinalgDenseMatrixDeterminantTest, 4x4DeterminantLU)
  {
    LINALG::SerialDenseMatrix A(4, 4, true);
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

    EXPECT_NEAR(LINALG::DeterminantLU(A), -0.01620776397174742, 1e-14);
  }
}  // namespace
