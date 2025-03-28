// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

FOUR_C_NAMESPACE_OPEN
namespace
{
  TEST(LinalgTensorProductsTest, TestKroneckerProduct)
  {
    // test the calculation of the fourth order kronecker product of two second-order tensors
    Core::LinAlg::Matrix<3, 3> a(Core::LinAlg::Initialization::uninitialized);
    a(0, 0) = 1.0000000000;
    a(0, 1) = 2.0000000000;
    a(0, 2) = 3.0000000000;
    a(1, 0) = 0.0000000000;
    a(1, 1) = 5.0000000000;
    a(1, 2) = 1.2000000000;
    a(2, 0) = 1.0000000000;
    a(2, 1) = 1.0000000000;
    a(2, 2) = 1.0000000000;

    Core::LinAlg::Matrix<3, 3> b(Core::LinAlg::Initialization::uninitialized);
    b(0, 0) = 1.0000000000;
    b(0, 1) = 0.0000000000;
    b(0, 2) = 5.0000000000;
    b(1, 0) = 0.0000000000;
    b(1, 1) = 0.0000000000;
    b(1, 2) = 7.0000000000;
    b(2, 0) = 1.3000000000;
    b(2, 1) = 1.2000000000;
    b(2, 2) = 1.0000000000;

    Core::LinAlg::Matrix<6, 6> a_kron_b_ref(Core::LinAlg::Initialization::zero);
    a_kron_b_ref(0, 0) = 1.0000000000;
    a_kron_b_ref(0, 1) = 0.0000000000;
    a_kron_b_ref(0, 2) = 15.0000000000;
    a_kron_b_ref(0, 3) = 1.0000000000;
    a_kron_b_ref(0, 4) = 5.0000000000;
    a_kron_b_ref(0, 5) = 4.0000000000;
    a_kron_b_ref(1, 0) = 0.0000000000;
    a_kron_b_ref(1, 1) = 0.0000000000;
    a_kron_b_ref(1, 2) = 8.4000000000;
    a_kron_b_ref(1, 3) = 0.0000000000;
    a_kron_b_ref(1, 4) = 17.5000000000;
    a_kron_b_ref(1, 5) = 0.0000000000;
    a_kron_b_ref(2, 0) = 1.3000000000;
    a_kron_b_ref(2, 1) = 1.2000000000;
    a_kron_b_ref(2, 2) = 1.0000000000;
    a_kron_b_ref(2, 3) = 1.2500000000;
    a_kron_b_ref(2, 4) = 1.1000000000;
    a_kron_b_ref(2, 5) = 1.1500000000;
    a_kron_b_ref(3, 0) = 0.0000000000;
    a_kron_b_ref(3, 1) = 0.0000000000;
    a_kron_b_ref(3, 2) = 21.0000000000;
    a_kron_b_ref(3, 3) = 0.0000000000;
    a_kron_b_ref(3, 4) = 7.0000000000;
    a_kron_b_ref(3, 5) = 3.5000000000;
    a_kron_b_ref(4, 0) = 0.0000000000;
    a_kron_b_ref(4, 1) = 6.0000000000;
    a_kron_b_ref(4, 2) = 1.2000000000;
    a_kron_b_ref(4, 3) = 3.2500000000;
    a_kron_b_ref(4, 4) = 3.2200000000;
    a_kron_b_ref(4, 5) = 0.7800000000;
    a_kron_b_ref(5, 0) = 1.3000000000;
    a_kron_b_ref(5, 1) = 2.4000000000;
    a_kron_b_ref(5, 2) = 3.0000000000;
    a_kron_b_ref(5, 3) = 1.9000000000;
    a_kron_b_ref(5, 4) = 2.8000000000;
    a_kron_b_ref(5, 5) = 2.4500000000;

    Core::LinAlg::Matrix<6, 6> a_kron_b(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Tensor::add_kronecker_tensor_product(a_kron_b, 1.0, a, b, 0.0);

    FOUR_C_EXPECT_NEAR(a_kron_b, a_kron_b_ref, 1.0e-10);
  }
}  // namespace
FOUR_C_NAMESPACE_CLOSE
