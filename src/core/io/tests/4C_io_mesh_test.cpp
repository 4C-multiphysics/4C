// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_io_mesh.hpp"

namespace
{
  using namespace FourC;
  using namespace Core::IO::MeshInput;


  void fill_test_mesh(RawMesh<3>& mesh)
  {
    // 27 points in a 3x3x3 grid from 0 to 1 in each direction
    mesh.points = {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.5, 0.0},
        {0.5, 0.5, 0.0}, {1.0, 0.5, 0.0}, {0.0, 1.0, 0.0}, {0.5, 1.0, 0.0}, {1.0, 1.0, 0.0},
        {0.0, 0.0, 0.5}, {0.5, 0.0, 0.5}, {1.0, 0.0, 0.5}, {0.0, 0.5, 0.5}, {0.5, 0.5, 0.5},
        {1.0, 0.5, 0.5}, {0.0, 1.0, 0.5}, {0.5, 1.0, 0.5}, {1.0, 1.0, 0.5}, {0.0, 0.0, 1.0},
        {0.5, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 0.5, 1}, {0.5, 0.5, 1}, {1.0, 0.5, 1},
        {0.0, 1.0, 1}, {0.5, 1.0, 1.0}, {1.0, 1.0, 1.0}};

    CellBlock<3> block1(Core::FE::CellType::hex8);
    block1.add_cell(std::array{0, 1, 4, 3, 9, 10, 13, 12});
    mesh.cell_blocks.emplace(1, block1);

    CellBlock<3> block2(Core::FE::CellType::hex8);
    block2.add_cell(std::array{1, 2, 5, 4, 10, 11, 14, 13});
    mesh.cell_blocks.emplace(2, block2);

    mesh.point_sets[10].point_ids = {0, 1, 3, 4};
    mesh.point_sets[20].point_ids = {22, 23, 25, 26};

    mesh.point_data["test_data"] = std::vector<double>(mesh.points.size(), 1.0);

    assert_valid(mesh);
  }

  TEST(Mesh, MeshFiltering)
  {
    Core::IO::MeshInput::RawMesh<3> raw_mesh;
    fill_test_mesh(raw_mesh);

    Core::IO::MeshInput::Mesh<3> mesh(std::move(raw_mesh));
    EXPECT_EQ(mesh.cell_blocks().size(), 2);

    auto filtered_mesh = mesh.filter_by_cell_block_ids({1});

    EXPECT_EQ(filtered_mesh.cell_blocks().size(), 1);
    EXPECT_EQ(filtered_mesh.points().size(), 8);
    EXPECT_EQ(filtered_mesh.point_sets().size(), 1);


    EXPECT_TRUE(filtered_mesh.has_point_data("test_data"));

    for (auto pd : filtered_mesh.points_with_data())
    {
      EXPECT_EQ(std::get<double>(pd.data("test_data")), 1.0);
      EXPECT_EQ(pd.data_as<double>("test_data"), 1.0);
    }
  }
}  // namespace
