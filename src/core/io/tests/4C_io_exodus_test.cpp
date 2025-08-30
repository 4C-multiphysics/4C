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
    Core::IO::MeshInput::Mesh mesh = Core::IO::Exodus::read_exodus_file(
        TESTING::get_support_file_path("test_files/exodus/cube.exo"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.side_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 27);
    EXPECT_EQ(mesh.cell_blocks[1].cell_connectivities.size(), 4);
    EXPECT_EQ(mesh.cell_blocks[2].cell_connectivities.size(), 4);

    Core::IO::MeshInput::print(mesh, std::cout, Core::IO::MeshInput::VerbosityLevel::none);
  }

  TEST(Exodus, NodeOffset)
  {
    Core::IO::MeshInput::Mesh mesh = Core::IO::Exodus::read_exodus_file(
        TESTING::get_support_file_path("test_files/exodus/cube.exo"),
        Core::IO::Exodus::MeshParameters{.node_start_id = 100});
    EXPECT_EQ(mesh.points[100], (std::vector<double>{-5.0, 0.0, 0.0}));
    EXPECT_EQ(mesh.cell_blocks[1].cell_connectivities.at(1),
        (std::vector<int>{108, 100, 103, 109, 110, 104, 107, 111}));
  }
}  // namespace
