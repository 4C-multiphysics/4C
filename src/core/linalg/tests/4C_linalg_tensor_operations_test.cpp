// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_linalg_tensor_operations.hpp"

#include "4C_linalg_tensor.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{

  TEST(TensorOperationsTest, Determinant)
  {
    Core::LinAlg::Tensor<double, 3, 3> t{};
    EXPECT_EQ(Core::LinAlg::det(t), 0.0);


    Core::LinAlg::Tensor<double, 2, 2> t2 = {{{1.0, 2.0}, {3.0, 4.0}}};
    EXPECT_EQ(Core::LinAlg::det(t2), -2.0);
  }

  TEST(TensorOperationsTest, Trace)
  {
    Core::LinAlg::Tensor<double, 3, 3> t{};
    EXPECT_EQ(Core::LinAlg::trace(t), 0.0);


    Core::LinAlg::Tensor<double, 2, 2> t2 = {{{1.0, 2.0}, {3.0, 4.0}}};
    EXPECT_EQ(Core::LinAlg::trace(t2), 5.0);
  }

  TEST(TensorOperationsTest, Inverse)
  {
    Core::LinAlg::Tensor<double, 2, 2> t2 = {{{1.0, 2.0}, {3.0, 4.0}}};
    Core::LinAlg::Tensor<double, 2, 2> t2_inv = Core::LinAlg::inv(t2);

    EXPECT_EQ(t2_inv(0, 0), -2.0);
    EXPECT_EQ(t2_inv(1, 1), -0.5);
    EXPECT_EQ(t2_inv(0, 1), 1.0);
    EXPECT_EQ(t2_inv(1, 0), 1.5);
  }

  TEST(TensorOperationsTest, Transpose)
  {
    Core::LinAlg::Tensor<double, 3, 2> t2 = {{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}};
    Core::LinAlg::Tensor<double, 2, 3> t2_t = Core::LinAlg::transpose(t2);

    EXPECT_EQ(t2_t.at(0, 0), 1.0);
    EXPECT_EQ(t2_t.at(0, 1), 3.0);
    EXPECT_EQ(t2_t.at(0, 2), 5.0);
    EXPECT_EQ(t2_t.at(1, 0), 2.0);
    EXPECT_EQ(t2_t.at(1, 1), 4.0);
    EXPECT_EQ(t2_t.at(1, 2), 6.0);
  }

  TEST(TensorOperationsTest, ContractionMatVec)
  {
    Core::LinAlg::Tensor<double, 3, 2> A = {{
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
    }};
    Core::LinAlg::Tensor<double, 2> b = {{1.0, 2.0}};
    Core::LinAlg::Tensor<double, 3> Axb = Core::LinAlg::contraction(A, b);

    EXPECT_EQ(Axb.at(0), 5.0);
    EXPECT_EQ(Axb.at(1), 11.0);
    EXPECT_EQ(Axb.at(2), 17.0);
  }

  TEST(TensorOperationsTest, ContractionMatMat)
  {
    Core::LinAlg::Tensor<double, 3, 2> A = {{
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
    }};

    Core::LinAlg::Tensor<double, 2, 3> B = {{
        {1.0, 2.0, 3.0},
        {3.0, 4.0, 5.0},
    }};
    Core::LinAlg::Tensor<double, 3, 3> AxB = Core::LinAlg::contraction(A, B);

    EXPECT_EQ(AxB.at(0, 0), 7.0);
    EXPECT_EQ(AxB.at(0, 1), 10.0);
    EXPECT_EQ(AxB.at(0, 2), 13.0);

    EXPECT_EQ(AxB.at(1, 0), 15.0);
    EXPECT_EQ(AxB.at(1, 1), 22.0);
    EXPECT_EQ(AxB.at(1, 2), 29.0);

    EXPECT_EQ(AxB.at(2, 0), 23.0);
    EXPECT_EQ(AxB.at(2, 1), 34.0);
    EXPECT_EQ(AxB.at(2, 2), 45.0);
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE