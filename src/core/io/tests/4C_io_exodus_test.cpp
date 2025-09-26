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
    Core::IO::MeshInput::RawMesh mesh = Core::IO::Exodus::read_exodus_file(
        TESTING::get_support_file_path("test_files/exodus/cube.exo"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 27);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 4);
    EXPECT_EQ(mesh.cell_blocks.at(2).size(), 4);

    EXPECT_EQ(mesh.points[0], (std::array{-5.0, 0.0, 0.0}));
    ASSERT_TRUE(mesh.external_ids.has_value());
    EXPECT_EQ(mesh.external_ids->at(0), 23);

    std::vector<std::vector<int>> expected_cells = {{
        {0, 1, 2, 3, 4, 5, 6, 7},
        {8, 0, 3, 9, 10, 4, 7, 11},
        {12, 13, 1, 0, 14, 15, 5, 4},
        {16, 12, 0, 8, 17, 14, 4, 10},
    }};

    std::size_t i = 0;
    for (const auto& cell : mesh.cell_blocks.at(1).cells())
    {
      std::vector cell_vec(cell.begin(), cell.end());
      EXPECT_EQ(cell_vec, expected_cells[i]);
      ++i;
    }

    ASSERT_TRUE(mesh.cell_blocks.at(1).name.has_value());
    EXPECT_EQ(*mesh.cell_blocks.at(1).name, "left");

    ASSERT_TRUE(mesh.cell_blocks.at(2).name.has_value());
    EXPECT_EQ(*mesh.cell_blocks.at(2).name, "right");

    ASSERT_TRUE(mesh.point_sets.at(1).name.has_value());
    EXPECT_EQ(*mesh.point_sets.at(1).name, "node_set_top");

    std::stringstream ss;
    Core::IO::MeshInput::print(Core::IO::MeshInput::Mesh<3>::create_view(mesh), ss,
        Core::IO::MeshInput::VerbosityLevel::full);
  }

  TEST(Exodus, MeshFiltering)
  {
    Core::IO::MeshInput::RawMesh raw_mesh = Core::IO::Exodus::read_exodus_file(
        TESTING::get_support_file_path("test_files/exodus/cube.exo"));

    Core::IO::MeshInput::Mesh mesh(std::move(raw_mesh));
    EXPECT_EQ(mesh.cell_blocks().size(), 2);

    auto filtered_mesh = mesh.filter_by_cell_block_ids({1});

    EXPECT_EQ(filtered_mesh.cell_blocks().size(), 1);
  }
}  // namespace
