// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  TEST(FourTensorOperations, HolzapfelProductTest)
  {
    using T = double;

    // Initialize symmetric tensor
    Core::LinAlg::SymmetricTensor<T, 3, 3> Cinverse = Core::LinAlg::assume_symmetry(
        Core::LinAlg::Tensor<T, 3, 3>{{{2.0, 0.1, -0.05}, {0.1, 1.5, 0.2}, {-0.05, 0.2, 1.8}}});

    // Initialize matrix
    Core::LinAlg::Matrix<3, 3, T> Cinverse_m;
    Cinverse_m(0, 0) = 2.0;
    Cinverse_m(0, 1) = 0.1;
    Cinverse_m(0, 2) = -0.05;
    Cinverse_m(1, 0) = 0.1;
    Cinverse_m(1, 1) = 1.5;
    Cinverse_m(1, 2) = 0.2;
    Cinverse_m(2, 0) = -0.05;
    Cinverse_m(2, 1) = 0.2;
    Cinverse_m(2, 2) = 1.8;
    // Initialize matrix in voigt notation
    Core::LinAlg::Matrix<6, 1, T> Cinverse_v;
    Core::LinAlg::Voigt::Stresses::matrix_to_vector(Cinverse_m, Cinverse_v);

    const T scalar = 1.0;

    // Compute product matrix based
    Core::LinAlg::Matrix<6, 6, T> cmat_voigt_old(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FourTensorOperations::add_holzapfel_product(cmat_voigt_old, Cinverse_v, scalar);

    // Compute product tensor based
    Core::LinAlg::SymmetricTensor<T, 3, 3, 3, 3> cmat_sym{};
    cmat_sym += scalar * Core::LinAlg::FourTensorOperations::holzapfel_product(Cinverse);

    // Convert tensor to stress like voigt view for comparison
    Core::LinAlg::Matrix<6, 6, T> cmat_voigt_new =
        Core::LinAlg::make_stress_like_voigt_view(cmat_sym);

    // Comparison
    FOUR_C_EXPECT_NEAR(cmat_voigt_old, cmat_voigt_new, 1e-12);
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE