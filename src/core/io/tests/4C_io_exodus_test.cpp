// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_exodus.hpp"

#include "4C_unittest_utils_support_files_test.hpp"

namespace
{
  using namespace FourC;

  TEST(Exodus, MeshCubeHex)
  {
    Core::IO::Exodus::Mesh mesh(TESTING::get_support_file_path("test_files/exodus/cube.exo"));

    EXPECT_EQ(mesh.get_num_element_blocks(), 2);
    EXPECT_EQ(mesh.get_num_node_sets(), 1);
    EXPECT_EQ(mesh.get_num_side_sets(), 1);
    EXPECT_EQ(mesh.get_num_nodes(), 27);
    EXPECT_EQ(mesh.get_element_block(1).get_num_ele(), 4);
    EXPECT_EQ(mesh.get_element_block(2).get_num_ele(), 4);

    mesh.print(std::cout, true);
  }
}  // namespace
