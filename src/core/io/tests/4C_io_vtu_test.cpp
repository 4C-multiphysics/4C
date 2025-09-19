// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_io_mesh.hpp"
#include "4C_io_vtu_reader.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_unittest_utils_support_files_test.hpp"
#include "4C_utils_demangle.hpp"

#include <variant>

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
        element_nodes.coordinates(i, d) = mesh.points[node_id][d];
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(VTU, MeshCubeHex8PointDataInput)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    Core::IO::MeshInput::Mesh<3> mesh = Core::IO::VTU::read_vtu_file(
        TESTING::get_support_file_path("test_files/vtu/hex8_point_data.vtu"));

    ASSERT_EQ(mesh.points.size(), 16);
    EXPECT_EQ(mesh.point_data.size(), 10);

    // Check scalar double field
    ASSERT_TRUE((std::holds_alternative<std::vector<double>>(mesh.point_data["scalar_float"])));
    const auto& scalar_float = std::get<std::vector<double>>(mesh.point_data["scalar_float"]);
    EXPECT_NEAR(scalar_float[10], 10.0, 1e-10);

    // Check scalar int field
    ASSERT_TRUE((std::holds_alternative<std::vector<int>>(mesh.point_data["scalar_int"])));
    const auto& scalar_int = std::get<std::vector<int>>(mesh.point_data["scalar_int"]);
    EXPECT_EQ(scalar_int[10], 10);

    // Check reading a double rank-1-Tensor field
    ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<double, 3>>>(
        mesh.point_data["vector_float"])));
    const auto& vector_float =
        std::get<std::vector<Core::LinAlg::Tensor<double, 3>>>(mesh.point_data["vector_float"]);
    for (std::size_t node_id = 0; node_id < mesh.points.size(); ++node_id)
    {
      auto ref = Core::LinAlg::make_tensor_view<3>(mesh.points[node_id].data());
      Core::LinAlg::Tensor<double, 3> value = vector_float[node_id];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }

    // Check reading a int rank-1-Tensor field
    ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<int, 3>>>(
        mesh.point_data["vector_int"])));
    const auto& vector_int =
        std::get<std::vector<Core::LinAlg::Tensor<int, 3>>>(mesh.point_data["vector_int"]);
    for (std::size_t node_id = 0; node_id < mesh.points.size(); ++node_id)
    {
      Core::LinAlg::Tensor<int, 3> ref = {{
          static_cast<int>(mesh.points[node_id][0]),
          static_cast<int>(mesh.points[node_id][1]),
          static_cast<int>(mesh.points[node_id][2]),
      }};
      Core::LinAlg::Tensor<int, 3> value = vector_int[node_id];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }

    {
      // Check reading a double rank-2-SymmetricTensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>>(
          mesh.point_data["symmetric_tensor_float"])));
      const auto& tensor_float = std::get<std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>>(
          mesh.point_data["symmetric_tensor_float"]);
      const auto ref = Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{{

          {1.1, 4.1, 6.1},
          {4.1, 2.1, 5.1},
          {6.1, 5.1, 3.1},
      }});
      Core::LinAlg::SymmetricTensor<double, 3, 3> value = tensor_float[10];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }

    {
      // Check reading a int rank-2-SymmetricTensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::SymmetricTensor<int, 3, 3>>>(
          mesh.point_data["symmetric_tensor_int"])));
      const auto& tensor_int = std::get<std::vector<Core::LinAlg::SymmetricTensor<int, 3, 3>>>(
          mesh.point_data["symmetric_tensor_int"]);
      const auto ref = Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<int, 3, 3>{{

          {11, 14, 16},
          {14, 12, 15},
          {16, 15, 13},
      }});
      Core::LinAlg::SymmetricTensor<int, 3, 3> value = tensor_int[10];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }

    {
      // Check reading a double rank-2-Tensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>(
          mesh.point_data["tensor_float"])));
      const auto& tensor_float = std::get<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>(
          mesh.point_data["tensor_float"]);
      const Core::LinAlg::Tensor<double, 3, 3> ref = {{
          {1.1, 2.1, 3.1},
          {4.1, 5.1, 6.1},
          {7.1, 8.1, 9.1},
      }};
      Core::LinAlg::Tensor<double, 3, 3> value = tensor_float[10];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }


    {
      // Check reading a int rank-2-Tensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<int, 3, 3>>>(
          mesh.point_data["tensor_int"])));
      const auto& tensor_int =
          std::get<std::vector<Core::LinAlg::Tensor<int, 3, 3>>>(mesh.point_data["tensor_int"]);
      const Core::LinAlg::Tensor<double, 3, 3> ref = {{
          {11, 12, 13},
          {14, 15, 16},
          {17, 18, 19},
      }};
      Core::LinAlg::Tensor<int, 3, 3> value = tensor_int[10];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }
  }

  TEST(VTU, MeshCubeHex8CellDataInput)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    Core::IO::MeshInput::Mesh<3> mesh = Core::IO::VTU::read_vtu_file(
        TESTING::get_support_file_path("test_files/vtu/hex8_cell_data.vtu"));

    ASSERT_EQ(mesh.cell_blocks.size(), 2);
    ASSERT_EQ(mesh.cell_blocks.at(1).size(), 2);
    ASSERT_EQ(mesh.cell_blocks.at(2).size(), 1);

    EXPECT_EQ(mesh.cell_blocks.at(1).cell_data.size(), 10);

    const Core::IO::MeshInput::CellBlock<3>& cell_block = mesh.cell_blocks.at(1);

    // Check scalar double field
    ASSERT_TRUE(
        (std::holds_alternative<std::vector<double>>(cell_block.cell_data.at("scalar_float"))));
    const auto& scalar_float =
        std::get<std::vector<double>>(cell_block.cell_data.at("scalar_float"));
    EXPECT_NEAR(scalar_float[1], 1.2, 1e-10);

    // Check scalar int field
    ASSERT_TRUE((std::holds_alternative<std::vector<int>>(cell_block.cell_data.at("scalar_int"))));
    const auto& scalar_int = std::get<std::vector<int>>(cell_block.cell_data.at("scalar_int"));
    EXPECT_EQ(scalar_int[1], 1);

    {
      // Check reading a double rank-1-Tensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<double, 3>>>(
          cell_block.cell_data.at("vector_float"))));
      const auto& vector_float = std::get<std::vector<Core::LinAlg::Tensor<double, 3>>>(
          cell_block.cell_data.at("vector_float"));
      const Core::LinAlg::Tensor<double, 3> ref = {{0.2, 1.2, 0.2}};
      FOUR_C_EXPECT_NEAR(vector_float[1], ref, 1e-12);
    }

    {
      // Check reading a int rank-1-Tensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<int, 3>>>(
          cell_block.cell_data.at("vector_int"))));
      const auto& vector_int = std::get<std::vector<Core::LinAlg::Tensor<int, 3>>>(
          cell_block.cell_data.at("vector_int"));
      const Core::LinAlg::Tensor<int, 3> ref = {{0, 1, 0}};
      FOUR_C_EXPECT_NEAR(vector_int[1], ref, 1e-12);
    }

    {
      // Check reading a double rank-2-SymmetricTensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>>(
          cell_block.cell_data.at("symmetric_tensor_float"))));
      const auto& tensor_float = std::get<std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>>(
          cell_block.cell_data.at("symmetric_tensor_float"));
      const Core::LinAlg::Tensor<double, 3, 3> ref = {{
          {1.01, 4.01, 6.01},
          {4.01, 2.01, 5.01},
          {6.01, 5.01, 3.01},
      }};
      Core::LinAlg::SymmetricTensor<double, 3, 3> value = tensor_float[1];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }

    {
      // Check reading a int rank-2-SymmetricTensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::SymmetricTensor<int, 3, 3>>>(
          cell_block.cell_data.at("symmetric_tensor_int"))));
      const auto& tensor_int = std::get<std::vector<Core::LinAlg::SymmetricTensor<int, 3, 3>>>(
          cell_block.cell_data.at("symmetric_tensor_int"));
      const Core::LinAlg::Tensor<int, 3, 3> ref = {{
          {2, 5, 7},
          {5, 3, 6},
          {7, 6, 4},
      }};
      Core::LinAlg::SymmetricTensor<int, 3, 3> value = tensor_int[1];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }

    {
      // Check reading a int rank-2-Tensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>(
          cell_block.cell_data.at("tensor_float"))));
      const auto& tensor_float = std::get<std::vector<Core::LinAlg::Tensor<double, 3, 3>>>(
          cell_block.cell_data.at("tensor_float"));
      const Core::LinAlg::Tensor<double, 3, 3> ref = {{
          {1.01, 2.01, 3.01},
          {4.01, 5.01, 6.01},
          {7.01, 8.01, 9.01},
      }};
      Core::LinAlg::Tensor<double, 3, 3> value = tensor_float[1];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
    }

    {
      // Check reading a int rank-2-Tensor field
      ASSERT_TRUE((std::holds_alternative<std::vector<Core::LinAlg::Tensor<int, 3, 3>>>(
          cell_block.cell_data.at("tensor_int"))));
      const auto& tensor_int = std::get<std::vector<Core::LinAlg::Tensor<int, 3, 3>>>(
          cell_block.cell_data.at("tensor_int"));
      const Core::LinAlg::Tensor<int, 3, 3> ref = {{
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 10},
      }};
      Core::LinAlg::Tensor<int, 3, 3> value = tensor_int[1];
      FOUR_C_EXPECT_NEAR(value, ref, 1e-12);
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
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
    for (const auto& connectivities : mesh.cell_blocks.at(1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }
}  // namespace
