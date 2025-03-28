// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_linalg_tensor.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{

  TEST(TensorTest, DefaultConstruct1Tensor)
  {
    Core::LinAlg::Tensor<double, 3> t{};
    EXPECT_DOUBLE_EQ(t(0), 0.0);
    EXPECT_DOUBLE_EQ(t(0), 0.0);
    EXPECT_DOUBLE_EQ(t(0), 0.0);
  }

  TEST(TensorTest, DefaultConstruct2Tensor)
  {
    Core::LinAlg::Tensor<double, 2, 2> t{};
    EXPECT_DOUBLE_EQ(t(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(t(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(t(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(t(1, 1), 0.0);
  }

  TEST(TensorTest, InitializationAndIndexing1Tensor)
  {
    Core::LinAlg::Tensor<double, 3> t = {{0.0, 1.0, 2.0}};

    EXPECT_DOUBLE_EQ(t(0), 0.0);
    EXPECT_DOUBLE_EQ(t(1), 1.0);
    EXPECT_DOUBLE_EQ(t(2), 2.0);
  }

  TEST(TensorTest, InitializationAndIndexing2Tensor)
  {
    Core::LinAlg::Tensor<double, 3, 2> t = {{{0.0, 0.1}, {1.0, 1.1}, {2.0, 2.1}}};

    EXPECT_DOUBLE_EQ(t(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(t(0, 1), 0.1);
    EXPECT_DOUBLE_EQ(t(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(1, 1), 1.1);
    EXPECT_DOUBLE_EQ(t(2, 0), 2.0);
    EXPECT_DOUBLE_EQ(t(2, 1), 2.1);
  }

  TEST(TensorTest, InitializationAndIndexing3Tensor)
  {
    Core::LinAlg::Tensor<double, 3, 2, 2> t = {{
        {{0.00, 0.01}, {0.10, 0.11}},
        {{1.00, 1.01}, {1.10, 1.11}},
        {{2.00, 2.01}, {2.10, 2.11}},
    }};

    EXPECT_DOUBLE_EQ(t(0, 0, 0), 0.00);
    EXPECT_DOUBLE_EQ(t(0, 0, 1), 0.01);

    EXPECT_DOUBLE_EQ(t(0, 1, 0), 0.10);
    EXPECT_DOUBLE_EQ(t(0, 1, 1), 0.11);

    EXPECT_DOUBLE_EQ(t(1, 0, 0), 1.00);
    EXPECT_DOUBLE_EQ(t(1, 0, 1), 1.01);

    EXPECT_DOUBLE_EQ(t(1, 1, 0), 1.10);
    EXPECT_DOUBLE_EQ(t(1, 1, 1), 1.11);

    EXPECT_DOUBLE_EQ(t(2, 0, 0), 2.00);
    EXPECT_DOUBLE_EQ(t(2, 0, 1), 2.01);

    EXPECT_DOUBLE_EQ(t(2, 1, 0), 2.10);
    EXPECT_DOUBLE_EQ(t(2, 1, 1), 2.11);
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE