// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_linalg_tensor_eigen.hpp"

#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace
{
  TEST(TensorEigenTest, eig2x2_sym)
  {
    Core::LinAlg::Tensor<double, 2, 2> t = {
        {{0.9964456203546112, 0.490484665405466}, {0.490484665405466, 0.5611378979071144}}};

    const auto& [eigenvalues, eigenvectors] = Core::LinAlg::eig(t);

    // expecting real eigenvalues
    for (const auto& val : eigenvalues) EXPECT_EQ(val.imag(), 0.0);

    std::array<double, 2> real = {eigenvalues[0].real(), eigenvalues[1].real()};

    std::array eigenvalues_sorted = real;
    std::ranges::sort(eigenvalues_sorted);
    EXPECT_NEAR(eigenvalues_sorted[0], 0.242183512545406, 1e-10);
    EXPECT_NEAR(eigenvalues_sorted[1], 1.31540000571632, 1e-10);

    const auto original_matrix = eigenvectors *
                                 Core::LinAlg::TensorGenerators::diagonal(eigenvalues) *
                                 Core::LinAlg::inv(eigenvectors);

    FOUR_C_EXPECT_NEAR(original_matrix, t, 1e-10);
  }

  TEST(TensorEigenTest, eig3x3_sym)
  {
    Core::LinAlg::Tensor<double, 3, 3> t = {
        {{1.1948716876311152, 0.5399232848096387, 0.6905691418748001},
            {0.5399232848096387, 0.8273468489149256, 0.2538261251935583},
            {0.6905691418748001, 0.2538261251935583, 0.48064209250998646}}};

    const auto& [eigenvalues, eigenvectors] = Core::LinAlg::eig(t);

    // expecting real eigenvalues
    for (const auto& val : eigenvalues) EXPECT_EQ(val.imag(), 0.0);

    std::array<double, 3> eigenvalues_real_sorted = {
        eigenvalues[0].real(), eigenvalues[1].real(), eigenvalues[2].real()};
    std::ranges::sort(eigenvalues_real_sorted);

    // expecting real eigenvalues
    for (const auto& val : eigenvalues) EXPECT_EQ(val.imag(), 0.0);

    EXPECT_NEAR(eigenvalues_real_sorted[0], 0.052879897400611, 1e-10);
    EXPECT_NEAR(eigenvalues_real_sorted[1], 0.516153257193444, 1e-10);
    EXPECT_NEAR(eigenvalues_real_sorted[2], 1.933827474461972, 1e-10);

    const auto original_matrix = eigenvectors *
                                 Core::LinAlg::TensorGenerators::diagonal(eigenvalues) *
                                 Core::LinAlg::inv(eigenvectors);

    FOUR_C_EXPECT_NEAR(original_matrix, t, 1e-10);
  }

  TEST(TensorEigenTest, eig3x3_non_sym)
  {
    const Core::LinAlg::Tensor<double, 3, 3> t = {
        {{1.0, 1.0, 0.0}, {0.0, 2.0, 1.0}, {0.0, 0.0, 3.0}}};

    const auto& [eigenvalues, eigenvectors] = Core::LinAlg::eig(t);
    // expecting real eigenvalues
    for (const auto& val : eigenvalues) EXPECT_EQ(val.imag(), 0.0);

    std::array<double, 3> eigenvalues_real_sorted = {
        eigenvalues[0].real(), eigenvalues[1].real(), eigenvalues[2].real()};
    std::ranges::sort(eigenvalues_real_sorted);

    EXPECT_NEAR(eigenvalues_real_sorted[0], 1.0, 1e-10);
    EXPECT_NEAR(eigenvalues_real_sorted[1], 2.0, 1e-10);
    EXPECT_NEAR(eigenvalues_real_sorted[2], 3.0, 1e-10);

    const auto original_matrix = eigenvectors *
                                 Core::LinAlg::TensorGenerators::diagonal(eigenvalues) *
                                 Core::LinAlg::inv(eigenvectors);

    FOUR_C_EXPECT_NEAR(original_matrix, t, 1e-10);
  }

  TEST(TensorEigenTest, eig3x3_non_sym2)
  {
    Core::LinAlg::Tensor<double, 3, 3> t = {
        {{0.9362933635841993, -0.27509584731824377, 0.21835066314633444},
            {0.2896294776255156, 0.9564250858492325, -0.03695701352462507},
            {-0.19866933079506122, 0.0978433950072557, 0.975170327201816}}};

    const auto& [eigenvalues, eigenvectors] = Core::LinAlg::eig(t);

    const auto original_matrix = eigenvectors *
                                 Core::LinAlg::TensorGenerators::diagonal(eigenvalues) *
                                 Core::LinAlg::inv(eigenvectors);

    FOUR_C_EXPECT_NEAR(original_matrix, t, 1e-10);
  }

  TEST(TensorEigenTest, eig3x3_non_diagonalizable)
  {
    Core::LinAlg::Tensor<double, 3, 3> t = {{{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}}};

    // This matrix is not diagonalizable, so expect an exception
    EXPECT_ANY_THROW(Core::LinAlg::eig(t));
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE