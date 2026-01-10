// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_io_gmsh_reader.hpp"
#include "4C_io_mesh.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_unittest_utils_support_files_test.hpp"

#include <cstdio>

namespace
{
  using namespace FourC;

  template <Core::FE::CellType celltype, unsigned dim>
  double evaluate_jacobian_determinant(Core::IO::MeshInput::RawMesh<3>& mesh,
      std::span<const int> connectivities, const Core::LinAlg::Tensor<double, dim>& xi)
  {
    Core::Elements::ElementNodes<celltype, dim> element_nodes;
    for (std::size_t i = 0; i < connectivities.size(); ++i)
    {
      const std::size_t node_id = connectivities[i];
      for (std::size_t d = 0; d < dim; ++d)
        element_nodes.coordinates(i, d) = mesh.points[node_id][d];
    }

    auto shape_functions =
        Core::Elements::evaluate_shape_functions_and_derivs<celltype>(xi, element_nodes);
    auto jacobian_mapping =
        Core::Elements::evaluate_jacobian_mapping<dim>(shape_functions, element_nodes);

    return jacobian_mapping.determinant();
  }

  TEST(GMSH, MeshCubeHex8)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex8;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/hex8.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 16);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);
    EXPECT_EQ(mesh.cell_blocks.at(2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(GMSH, MeshCubeHex20)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex20;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/hex20.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 44);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);
    EXPECT_EQ(mesh.cell_blocks.at(2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(GMSH, MeshCubeHex27)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex27;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/hex27.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 63);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);
    EXPECT_EQ(mesh.cell_blocks.at(2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});

      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(GMSH, MeshCubeTet4)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::tet4;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/tet4.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 14);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 24);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.25, 1e-9);
    }
  }

  TEST(GMSH, MeshCubeTet10)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::tet10;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/tet10.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 63);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 24);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.25, 1e-9);
    }
  }

  TEST(GMSH, MeshCubeWedge6)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::wedge6;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/wedge6.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 10);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 4);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.25, 1e-9);
    }
  }

  TEST(GMSH, MeshCubeWedge15)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::wedge15;
    Core::IO::MeshInput::RawMesh<3> mesh = Core::IO::Gmsh::read_msh_file(
        TESTING::get_support_file_path("test_files/gmsh/wedge15.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 31);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 4);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.25, 1e-9);
    }
  }

  TEST(GMSH, MeshCubePyramid5)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::pyramid5;
    Core::IO::MeshInput::RawMesh<3> mesh = Core::IO::Gmsh::read_msh_file(
        TESTING::get_support_file_path("test_files/gmsh/pyramid5.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 5);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.112500125, 1e-9);
    }
  }

  TEST(GMSH, MeshRectangleQuad4)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::quad4;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/quad4.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 4);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 2>(mesh, connectivities, {{0.1, 0.2}});
      EXPECT_NEAR(detJ, 0.25, 1e-9);
    }
  }

  TEST(GMSH, MeshRectangleQuad8)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::quad8;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/quad8.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 8);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 2>(mesh, connectivities, {{0.1, 0.2}});
      EXPECT_NEAR(detJ, 0.25, 1e-9);
    }
  }

  TEST(GMSH, MeshRectangleQuad9)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::quad9;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/quad9.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 9);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 2>(mesh, connectivities, {{0.1, 0.2}});
      EXPECT_NEAR(detJ, 0.25, 1e-9);
    }
  }

  TEST(GMSH, MeshRectangleTri3)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::tri3;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/tri3.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 4);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 2>(mesh, connectivities, {{0.1, 0.2}});
      EXPECT_NEAR(detJ, 1.0, 1e-9);
    }
  }

  TEST(GMSH, MeshRectangleTri6)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::tri6;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/tri6.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 9);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 2);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 2>(mesh, connectivities, {{0.1, 0.2}});
      EXPECT_NEAR(detJ, 1.0, 1e-9);
    }
  }

  TEST(GMSH, MeshLine2)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::line2;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/line2.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 2);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 1>(mesh, connectivities, {{0.1}});
      EXPECT_NEAR(detJ, 0.5, 1e-9);
    }
  }
  TEST(GMSH, MeshLine3)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::line3;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/line3.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 3);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 1>(mesh, connectivities, {{0.1}});
      EXPECT_NEAR(detJ, 0.5, 1e-9);
    }
  }

  TEST(GMSH, MeshLine4)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::line4;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/line4.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 4);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 1>(mesh, connectivities, {{0.1}});
      EXPECT_NEAR(detJ, 0.5, 1e-9);
    }
  }

  TEST(GMSH, MeshLine5)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::line5;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/line5.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 5);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 1>(mesh, connectivities, {{0.1}});
      EXPECT_NEAR(detJ, 0.5, 1e-9);
    }
  }

  TEST(GMSH, MeshLine6)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::line6;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/line6.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 6);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ = evaluate_jacobian_determinant<celltype, 1>(mesh, connectivities, {{0.1}});
      EXPECT_NEAR(detJ, 0.5, 1e-9);
    }
  }

  TEST(GMSH, MeshPoint1)
  {
#ifndef FOUR_C_WITH_GMSH
    GTEST_SKIP() << "Skipping test: 4C msh-input requires Gmsh support";
#endif
    constexpr auto celltype = Core::FE::CellType::point1;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::Gmsh::read_msh_file(TESTING::get_support_file_path("test_files/gmsh/point1.msh"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 1);
    EXPECT_EQ(mesh.points.size(), 1);
    EXPECT_EQ(mesh.cell_blocks.at(1).size(), 1);

    EXPECT_EQ(mesh.cell_blocks.at(1).cell_type, celltype);
  }
}  // namespace
