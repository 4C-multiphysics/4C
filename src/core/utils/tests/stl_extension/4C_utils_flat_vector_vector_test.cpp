// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_flat_vector_vector.hpp"

namespace
{
  using namespace FourC::Core::Utils;

  TEST(FlatVectorVectorTest, Default)
  {
    const FlatVectorVector<int> v;
    EXPECT_EQ(v.size(), 0);
    EXPECT_EQ(v.total_size(), 0);
  }

  TEST(FlatVectorVectorTest, Triangular)
  {
    const FlatVectorVector<int> v({{1}, {2, 3}, {4, 5, 6}});
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v.total_size(), 6);
    EXPECT_EQ(v[0][0], 1);
    EXPECT_EQ(v[1][0], 2);
    EXPECT_EQ(v[1][1], 3);
    EXPECT_EQ(v[2][0], 4);
    EXPECT_EQ(v[2][1], 5);
    EXPECT_EQ(v[2][2], 6);
  }

}  // namespace
