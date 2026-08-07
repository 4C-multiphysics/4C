// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_io_mesh.hpp"
#include "4C_io_mesh_test_utils_test.hpp"
#include "4C_io_vtu_reader.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_unittest_utils_support_files_test.hpp"
#include "4C_utils_demangle.hpp"
#include "4C_utils_flat_vector_vector.hpp"

#include <algorithm>
#include <set>
#include <variant>

namespace
{
  using namespace FourC;
  using Core::IO::MeshInput::Test::cell_blocks_by_group_id;
  using Core::IO::MeshInput::Test::unique_cell_block_by_group_id;
  using Core::IO::MeshInput::Test::unique_cell_block_by_group_id_and_cell_type;

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

  TEST(VTU, MeshCubeHex8)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex8;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/hex8.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 16);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 2);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }

  TEST(VTU, SplitsMixedCellTypesInLogicalBlock)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    Core::IO::MeshInput::RawMesh<3> raw_mesh = Core::IO::VTU::read_vtu_file(
        TESTING::get_support_file_path("test_files/vtu/mixed_hex8_tet4_same_block.vtu"));

    ASSERT_EQ(raw_mesh.cell_blocks.size(), 2);

    const auto logical_block_2 = cell_blocks_by_group_id(raw_mesh, 2);
    ASSERT_EQ(logical_block_2.size(), 2);
    const std::set cell_types_in_logical_block_2{
        logical_block_2[0].cell_type(), logical_block_2[1].cell_type()};
    EXPECT_EQ(cell_types_in_logical_block_2,
        (std::set{Core::FE::CellType::hex8, Core::FE::CellType::tet4}));

    const auto& split_tet4_block =
        unique_cell_block_by_group_id_and_cell_type(raw_mesh, 2, Core::FE::CellType::tet4);
    EXPECT_EQ(split_tet4_block.size(), 1);
  }

  TEST(VTU, MeshCubeHex8PointDataInput)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    Core::IO::MeshInput::Mesh<3> mesh = Core::IO::MeshInput::Mesh(Core::IO::VTU::read_vtu_file(
        TESTING::get_support_file_path("test_files/vtu/hex8_point_data.vtu")));

    ASSERT_EQ(mesh.points().size(), 16);
    EXPECT_EQ(mesh.points_with_data()[0].data_size(), 10);

    const auto point_ref = mesh.points_with_data()[10];

    // Check scalar double field
    ASSERT_TRUE((std::holds_alternative<std::span<const double>>(point_ref.data("scalar_float"))));
    const auto& scalar_float = point_ref.data_as<double>("scalar_float");
    EXPECT_NEAR(scalar_float, 10.0, 1e-10);

    // Check scalar int field
    ASSERT_TRUE((std::holds_alternative<std::span<const int>>(point_ref.data("scalar_int"))));
    const auto& scalar_int = point_ref.data_as<int>("scalar_int");
    EXPECT_EQ(scalar_int, 10);

    // Check reading a double rank-1-Tensor field
    ASSERT_TRUE((std::holds_alternative<std::span<const double>>(point_ref.data("vector_float"))));
    const auto& vector_float = point_ref.data_as<std::vector<double>>("vector_float");
    ASSERT_EQ(vector_float.size(), 3);
    auto ref = Core::LinAlg::make_tensor_view<3>(mesh.points()[10].data());
    FOUR_C_EXPECT_NEAR(
        (point_ref.data_as<Core::LinAlg::Tensor<double, 3>>("vector_float")), ref, 1e-12);

    // Check reading a int rank-1-Tensor field
    ASSERT_TRUE((std::holds_alternative<std::span<const int>>(point_ref.data("vector_int"))));
    const auto& vector_int = point_ref.data_as<std::vector<int>>("vector_int");
    ASSERT_EQ(vector_int.size(), 3);
    Core::LinAlg::Tensor<int, 3> ref_int = {{
        static_cast<int>(mesh.points()[10][0]),
        static_cast<int>(mesh.points()[10][1]),
        static_cast<int>(mesh.points()[10][2]),
    }};
    FOUR_C_EXPECT_NEAR(
        (point_ref.data_as<Core::LinAlg::Tensor<int, 3>>("vector_int")), ref_int, 1e-12);

    {
      // Check reading a double rank-2-SymmetricTensor field
      ASSERT_TRUE((std::holds_alternative<std::span<const double>>(
          point_ref.data("symmetric_tensor_float"))));
      const auto& vector_double = point_ref.data_as<std::vector<double>>("symmetric_tensor_float");
      ASSERT_EQ(vector_double.size(), 6);
      const auto ref = Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{{
          {1.1, 4.1, 6.1},
          {4.1, 2.1, 5.1},
          {6.1, 5.1, 3.1},
      }});
      FOUR_C_EXPECT_NEAR((point_ref.data_as<Core::LinAlg::SymmetricTensor<double, 3, 3>>(
                             "symmetric_tensor_float")),
          ref, 1e-12);
    }

    {
      // Check reading a double rank-2-SymmetricTensor field
      ASSERT_TRUE(
          (std::holds_alternative<std::span<const int>>(point_ref.data("symmetric_tensor_int"))));
      const auto& vector_int = point_ref.data_as<std::vector<int>>("symmetric_tensor_int");
      ASSERT_EQ(vector_int.size(), 6);
      const auto ref = Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<int, 3, 3>{{
          {11, 14, 16},
          {14, 12, 15},
          {16, 15, 13},
      }});
      FOUR_C_EXPECT_NEAR(
          (point_ref.data_as<Core::LinAlg::SymmetricTensor<int, 3, 3>>("symmetric_tensor_int")),
          ref, 1e-12);
    }

    {
      // Check reading a double rank-2-Tensor field
      ASSERT_TRUE(
          (std::holds_alternative<std::span<const double>>(point_ref.data("tensor_float"))));
      const auto& vector_double = point_ref.data_as<std::vector<double>>("tensor_float");
      ASSERT_EQ(vector_double.size(), 9);
      const auto ref = Core::LinAlg::Tensor<double, 3, 3>{{
          {1.1, 2.1, 3.1},
          {4.1, 5.1, 6.1},
          {7.1, 8.1, 9.1},
      }};
      FOUR_C_EXPECT_NEAR(
          (point_ref.data_as<Core::LinAlg::Tensor<double, 3, 3>>("tensor_float")), ref, 1e-12);
    }

    {
      // Check reading a int rank-2-Tensor field
      ASSERT_TRUE((std::holds_alternative<std::span<const int>>(point_ref.data("tensor_int"))));
      const auto& vector_int = point_ref.data_as<std::vector<int>>("tensor_int");
      ASSERT_EQ(vector_int.size(), 9);
      const auto ref = Core::LinAlg::Tensor<int, 3, 3>{{
          {11, 12, 13},
          {14, 15, 16},
          {17, 18, 19},
      }};
      FOUR_C_EXPECT_NEAR(
          (point_ref.data_as<Core::LinAlg::Tensor<int, 3, 3>>("tensor_int")), ref, 1e-12);
    }
  }

  TEST(VTU, MeshCubeHex8CellDataInput)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    Core::IO::MeshInput::Mesh<3> mesh = Core::IO::MeshInput::Mesh(Core::IO::VTU::read_vtu_file(
        TESTING::get_support_file_path("test_files/vtu/hex8_cell_data.vtu")));

    ASSERT_EQ(mesh.cell_blocks().size(), 2);
    ASSERT_EQ(mesh.cell_blocks()[0].size(), 2);
    ASSERT_EQ(mesh.cell_blocks()[1].size(), 1);

    EXPECT_EQ(mesh.cell_blocks()[0].cell_data().size(), 10);

    const auto& cell_block = mesh.cell_blocks()[0];
    const auto& cell_reference = cell_block.cells_with_data()[1];

    // Check scalar double field
    ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<double>>(
        cell_block.cell_data().at("scalar_float"))));
    const auto& scalar_float = cell_reference.data_as<double>("scalar_float");
    EXPECT_NEAR(scalar_float, 1.2, 1e-10);

    // I should also be able to read it as a vector of length 1
    const auto& scalar_float_vector = cell_reference.data_as<std::vector<double>>("scalar_float");
    EXPECT_EQ(scalar_float_vector.size(), 1);
    EXPECT_NEAR(scalar_float_vector[0], 1.2, 1e-10);

    // Check scalar int field
    ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<int>>(
        cell_block.cell_data().at("scalar_int"))));
    const auto& scalar_int = cell_reference.data_as<int>("scalar_int");
    EXPECT_EQ(scalar_int, 1);

    {
      // Check reading a double rank-1-Tensor field
      ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<double>>(
          cell_block.cell_data().at("vector_float"))));
      const Core::LinAlg::Tensor<double, 3> ref = {{0.2, 1.2, 0.2}};

      // Cannot read a vector as a double!
      EXPECT_ANY_THROW(
          [[maybe_unused]] const auto& _ = cell_reference.data_as<double>("vector_float"));

      // Can read a vector as a vector of double
      const auto& vector_float = cell_reference.data_as<std::vector<double>>("vector_float");
      ASSERT_THAT(vector_float, testing::SizeIs(3));
      ASSERT_THAT(vector_float[0], ref(0));
      ASSERT_THAT(vector_float[1], ref(1));
      ASSERT_THAT(vector_float[2], ref(2));

      // Can read a vector as a array of double with correct size
      const auto& array_float = cell_reference.data_as<std::array<double, 3>>("vector_float");
      ASSERT_THAT(array_float, testing::SizeIs(3));
      ASSERT_THAT(array_float[0], ref(0));
      ASSERT_THAT(array_float[1], ref(1));
      ASSERT_THAT(array_float[2], ref(2));

      // Cannot read a vector as a array of double with wrong size
      EXPECT_ANY_THROW(({
        [[maybe_unused]] const auto& _ =
            cell_reference.data_as<std::array<double, 2>>("vector_float");
      }));
      EXPECT_ANY_THROW(({
        [[maybe_unused]] const auto& _ =
            cell_reference.data_as<std::array<double, 4>>("vector_float");
      }));

      // Can read a vector as a Tensor of double with correct size
      const auto& tensor_float =
          cell_reference.data_as<Core::LinAlg::Tensor<double, 3>>("vector_float");
      FOUR_C_EXPECT_NEAR(tensor_float, ref, 1e-12);
    }

    {
      // Check reading a int rank-1-Tensor field
      ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<int>>(
          cell_block.cell_data().at("vector_int"))));
      const auto& vector_int = cell_reference.data_as<Core::LinAlg::Tensor<int, 3>>("vector_int");
      const Core::LinAlg::Tensor<int, 3> ref = {{0, 1, 0}};
      FOUR_C_EXPECT_NEAR(vector_int, ref, 1e-12);
    }

    {
      // Check reading a double rank-2-SymmetricTensor field
      ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<double>>(
          cell_block.cell_data().at("symmetric_tensor_float"))));
      const auto& tensor_float =
          cell_reference.data_as<Core::LinAlg::SymmetricTensor<double, 3, 3>>(
              "symmetric_tensor_float");
      const Core::LinAlg::Tensor<double, 3, 3> ref = {{
          {1.01, 4.01, 6.01},
          {4.01, 2.01, 5.01},
          {6.01, 5.01, 3.01},
      }};
      FOUR_C_EXPECT_NEAR(tensor_float, ref, 1e-12);
    }

    {
      // Check reading a int rank-2-SymmetricTensor field
      ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<int>>(
          cell_block.cell_data().at("symmetric_tensor_int"))));
      const auto& tensor_int =
          cell_reference.data_as<Core::LinAlg::SymmetricTensor<int, 3, 3>>("symmetric_tensor_int");
      const Core::LinAlg::Tensor<int, 3, 3> ref = {{
          {2, 5, 7},
          {5, 3, 6},
          {7, 6, 4},
      }};
      FOUR_C_EXPECT_NEAR(tensor_int, ref, 1e-12);
    }

    {
      // Check reading a int rank-2-Tensor field
      ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<double>>(
          cell_block.cell_data().at("tensor_float"))));
      const auto& tensor_float =
          cell_reference.data_as<Core::LinAlg::Tensor<double, 3, 3>>("tensor_float");
      const Core::LinAlg::Tensor<double, 3, 3> ref = {{
          {1.01, 2.01, 3.01},
          {4.01, 5.01, 6.01},
          {7.01, 8.01, 9.01},
      }};
      FOUR_C_EXPECT_NEAR(tensor_float, ref, 1e-12);
    }

    {
      // Check reading a int rank-2-Tensor field
      ASSERT_TRUE((std::holds_alternative<Core::Utils::Vector2D<int>>(
          cell_block.cell_data().at("tensor_int"))));
      const auto& tensor_int =
          cell_reference.data_as<Core::LinAlg::Tensor<int, 3, 3>>("tensor_int");
      const Core::LinAlg::Tensor<int, 3, 3> ref = {{
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 10},
      }};
      FOUR_C_EXPECT_NEAR(tensor_int, ref, 1e-12);
    }
  }

  TEST(VTU, MeshCubeHex20)
  {
#ifndef FOUR_C_WITH_VTK
    GTEST_SKIP() << "Skipping test: 4C vtu-input requires VTK support";
#endif
    constexpr auto celltype = Core::FE::CellType::hex20;
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/hex20.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 44);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 2);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
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
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/hex27.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 2);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 63);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 2);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 2).size(), 1);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
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
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/tet4.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 15);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 28);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    std::size_t id = 0;
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
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
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/tet10.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 69);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 28);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    std::size_t id = 0;
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
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
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/wedge6.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 8);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 2);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
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
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/wedge15.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 22);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 2);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
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
    Core::IO::MeshInput::RawMesh<3> mesh =
        Core::IO::VTU::read_vtu_file(TESTING::get_support_file_path("test_files/vtu/pyramid5.vtu"));

    EXPECT_EQ(mesh.cell_blocks.size(), 1);
    EXPECT_EQ(mesh.point_sets.size(), 2);
    EXPECT_EQ(mesh.points.size(), 9);
    EXPECT_EQ(unique_cell_block_by_group_id(mesh, 1).size(), 6);

    // check node ordering of the connectivity by evaluating the jacobian at xi=[0.1,0.2,0.3]
    for (const auto& connectivities : unique_cell_block_by_group_id(mesh, 1).cells())
    {
      double detJ =
          evaluate_jacobian_determinant<celltype, 3>(mesh, connectivities, {{0.1, 0.2, 0.3}});
      EXPECT_NEAR(detJ, 0.125, 1e-9);
    }
  }
}  // namespace
