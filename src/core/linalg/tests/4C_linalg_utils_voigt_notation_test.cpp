// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  class VoigtNotationTest : public testing::Test
  {
   public:
    Core::LinAlg::Matrix<3, 3> tens;
    Core::LinAlg::Matrix<3, 3> items;
    Core::LinAlg::Matrix<6, 1> tens_strain;
    Core::LinAlg::Matrix<6, 1> items_strain;
    Core::LinAlg::Matrix<6, 1> tens_stress;
    Core::LinAlg::Matrix<6, 1> items_stress;

    VoigtNotationTest()
    {
      tens(0, 0) = 1.1;
      tens(1, 1) = 1.2;
      tens(2, 2) = 1.3;
      tens(0, 1) = tens(1, 0) = 0.01;
      tens(1, 2) = tens(2, 1) = 0.02;
      tens(0, 2) = tens(2, 0) = 0.03;

      items(0, 0) = 0.90972618;
      items(1, 1) = 0.83360457;
      items(2, 2) = 0.76990741;
      items(0, 1) = items(1, 0) = -0.00723301;
      items(1, 2) = items(2, 1) = -0.01265777;
      items(0, 2) = items(2, 0) = -0.0208824;

      tens_strain(0) = tens_stress(0) = 1.1;
      tens_strain(1) = tens_stress(1) = 1.2;
      tens_strain(2) = tens_stress(2) = 1.3;

      tens_stress(3) = 0.01;
      tens_stress(4) = 0.02;
      tens_stress(5) = 0.03;

      items_stress(0) = 0.90972618;
      items_stress(1) = 0.83360457;
      items_stress(2) = 0.76990741;
      items_stress(3) = -0.00723301;
      items_stress(4) = -0.01265777;
      items_stress(5) = -0.0208824;

      tens_strain(3) = 2 * 0.01;
      tens_strain(4) = 2 * 0.02;
      tens_strain(5) = 2 * 0.03;

      items_strain(0) = 0.90972618;
      items_strain(1) = 0.83360457;
      items_strain(2) = 0.76990741;
      items_strain(3) = 2 * -0.00723301;
      items_strain(4) = 2 * -0.01265777;
      items_strain(5) = 2 * -0.0208824;
    };
  };

  TEST_F(VoigtNotationTest, MatrixToVectorStressLike)
  {
    Core::LinAlg::Matrix<6, 1> cmp_stress(Core::LinAlg::Initialization::uninitialized);

    Core::LinAlg::Voigt::Stresses::matrix_to_vector(tens, cmp_stress);

    FOUR_C_EXPECT_NEAR(cmp_stress, tens_stress, 1e-10);
  }

  TEST_F(VoigtNotationTest, MatrixToVectorStrainLike)
  {
    Core::LinAlg::Matrix<6, 1> cmp_strain(Core::LinAlg::Initialization::uninitialized);

    Core::LinAlg::Voigt::Strains::matrix_to_vector(tens, cmp_strain);

    FOUR_C_EXPECT_NEAR(cmp_strain, tens_strain, 1e-10);
  }

  TEST_F(VoigtNotationTest, DeterminantStressLike)
  {
    EXPECT_NEAR(Core::LinAlg::Voigt::Stresses::determinant(tens_stress), 1.7143620000000002, 1e-10);
  }

  TEST_F(VoigtNotationTest, DeterminantStrainLike)
  {
    EXPECT_NEAR(Core::LinAlg::Voigt::Strains::determinant(tens_strain), 1.7143620000000002, 1e-10);
  }

  TEST_F(VoigtNotationTest, InvariantsPrincipalStressLike)
  {
    Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Stresses::invariants_principal(prinv, tens_stress);
    EXPECT_NEAR(prinv(0), 3.5999999999999996, 1e-10);
    EXPECT_NEAR(prinv(1), 4.3085999999999984, 1e-10);
    EXPECT_NEAR(prinv(2), 1.7143620000000002, 1e-10);
  }

  TEST_F(VoigtNotationTest, InvariantsPrincipalStrainLike)
  {
    Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Strains::invariants_principal(prinv, tens_strain);
    EXPECT_NEAR(prinv(0), 3.5999999999999996, 1e-10);
    EXPECT_NEAR(prinv(1), 4.3085999999999984, 1e-10);
    EXPECT_NEAR(prinv(2), 1.7143620000000002, 1e-10);
  }

  TEST_F(VoigtNotationTest, InverseStressLike)
  {
    Core::LinAlg::Matrix<6, 1> items_stress_result(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Stresses::inverse_tensor(tens_stress, items_stress_result);

    FOUR_C_EXPECT_NEAR(items_stress_result, items_stress, 1e-5);
  }

  TEST_F(VoigtNotationTest, InverseStrainLike)
  {
    Core::LinAlg::Matrix<6, 1> items_strain_result(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Strains::inverse_tensor(tens_strain, items_strain_result);

    FOUR_C_EXPECT_NEAR(items_strain_result, items_strain, 1e-5);
  }

  TEST_F(VoigtNotationTest, to_stress_like)
  {
    Core::LinAlg::Matrix<6, 1> strain_to_stress(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<6, 1> stress_to_stress(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Strains::to_stress_like(tens_strain, strain_to_stress);
    Core::LinAlg::Voigt::Stresses::to_stress_like(tens_stress, stress_to_stress);

    FOUR_C_EXPECT_NEAR(strain_to_stress, tens_stress, 1e-5);
    FOUR_C_EXPECT_NEAR(stress_to_stress, stress_to_stress, 1e-5);
  }

  TEST_F(VoigtNotationTest, to_strain_like)
  {
    Core::LinAlg::Matrix<6, 1> strain_to_strain(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<6, 1> stress_to_strain(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Strains::to_strain_like(tens_strain, strain_to_strain);
    Core::LinAlg::Voigt::Stresses::to_strain_like(tens_stress, stress_to_strain);


    FOUR_C_EXPECT_NEAR(strain_to_strain, tens_strain, 1e-5);
    FOUR_C_EXPECT_NEAR(stress_to_strain, tens_strain, 1e-5);
  }

  TEST_F(VoigtNotationTest, IdentityMatrix)
  {
    Core::LinAlg::Matrix<6, 1> id(Core::LinAlg::Initialization::uninitialized);

    Core::LinAlg::Voigt::identity_matrix(id);

    EXPECT_NEAR(id(0), 1.0, 1e-10);
    EXPECT_NEAR(id(1), 1.0, 1e-10);
    EXPECT_NEAR(id(2), 1.0, 1e-10);
    EXPECT_NEAR(id(3), 0.0, 1e-10);
    EXPECT_NEAR(id(4), 0.0, 1e-10);
    EXPECT_NEAR(id(5), 0.0, 1e-10);
  }

  TEST_F(VoigtNotationTest, StrainLikeVectorToMatrix)
  {
    Core::LinAlg::Matrix<3, 3> matrix(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Strains::vector_to_matrix(tens_strain, matrix);
    EXPECT_EQ(matrix, tens);
  }

  TEST_F(VoigtNotationTest, StressLikeVectorToMatrix)
  {
    Core::LinAlg::Matrix<3, 3> matrix(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Voigt::Stresses::vector_to_matrix(tens_stress, matrix);
    EXPECT_EQ(matrix, tens);
  }
}  // namespace
FOUR_C_NAMESPACE_CLOSE
