// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_io_vtu_reader.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

namespace
{
  using namespace FourC;

  template <Core::FE::CellType celltype, unsigned dim>
  double evaluate_jacobian_determinant(Core::IO::MeshInput::Mesh<3>& mesh,
      std::span<const int> connectivities, const Core::LinAlg::Tensor<double, dim>& xi)
  {
    Core::Elements::ElementNodes<celltype, dim> element_nodes;
    for (std::size_t i = 0; i < connectivities.size(); ++i)
    {
      const std::size_t node_id = connectivities[i];
      for (std::size_t d = 0; d < dim; ++d)
        element_nodes.coordinates(i, d) = mesh.points[node_id].coords[d];
    }

    auto shape_functions =
        Core::Elements::evaluate_shape_functions_and_derivs<celltype>(xi, element_nodes);
    auto jacobian_mapping =
        Core::Elements::evaluate_jacobian_mapping<dim>(shape_functions, element_nodes);

    return jacobian_mapping.determinant();
  }

  TEST(VTU, MeshCubeHex8)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex8;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/hex8.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 16);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);
    EXPECT_EQ(mesh.cell_blocks.at(2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(VTU, MeshCubeHex20)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex20;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/hex20.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 44);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);
    EXPECT_EQ(mesh.cell_blocks.at(2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(VTU, MeshCubeHex27)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex27;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/hex27.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 63);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);
    EXPECT_EQ(mesh.cell_blocks.at(2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});

      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(VTU, MeshCubeTet4)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::tet4;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/tet4.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 15);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 28);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    std::size_t id = 0;
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      if (id < 20)
      {
        EXPECT_NEAR(detJ, 0.25, 1e-9);
      }
      else
      {
        EXPECT_NEAR(detJ, 0.125, 1e-9);
      }
      ++id;
    }
  }

  TEST(VTU, MeshCubeTet10)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::tet10;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/tet10.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 69);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 28);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    std::size_t id = 0;
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});

      if (id < 20)
      {
        EXPECT_NEAR(detJ, 0.25, 1e-9);
      }
      else
      {
        EXPECT_NEAR(detJ, 0.125, 1e-9);
      }
      ++id;
    }
  }

  TEST(VTU, MeshCubeWedge6)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::wedge6;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/wedge6.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 8);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.5, 1e-9);
    }
  }

  TEST(VTU, MeshCubeWedge15)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::wedge15;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/wedge15.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 22);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.5, 1e-9);
    }
  }

  TEST(VTU, MeshCubePyramid5)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::pyramid5;
    Core::IO::MeshInput::Mesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/pyramid5.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 9);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 6);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cell_connectivities())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }
}  // namespace
