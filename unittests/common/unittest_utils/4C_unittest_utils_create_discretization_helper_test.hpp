// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UNITTEST_UTILS_CREATE_DISCRETIZATION_HELPER_TEST_HPP
#define FOUR_C_UNITTEST_UTILS_CREATE_DISCRETIZATION_HELPER_TEST_HPP

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_builder.hpp"
#include "4C_rebalance.hpp"
#include "4C_rebalance_graph_based.hpp"

#include <mpi.h>

#include <array>

namespace TESTING
{
  using namespace FourC;

  /**
   * Fill the given @p discretization with a hypercube mesh. A total of `subdivisions^3` elements
   * are created and partitioned among all processes in @p comm.
   * */
  inline void fill_discretization_hyper_cube(
      Core::FE::Discretization& discretization, int subdivisions, MPI_Comm comm)
  {
    discretization.clear_discret();

    const int my_rank = Core::Communication::my_mpi_rank(comm);
    const int total_ranks = Core::Communication::num_mpi_ranks(comm);

    [[maybe_unused]] const int total_elements = subdivisions * subdivisions * subdivisions;
    // function to convert indices into a node lid
    const auto lid = [subdivisions](int i, int j, int k)
    { return i * (subdivisions + 1) * (subdivisions + 1) + j * (subdivisions + 1) + k; };

    const double imbalance_tol(1.1);

    Core::Rebalance::RebalanceParameters rebalance_parameters;
    rebalance_parameters.mesh_partitioning_parameters.min_ele_per_proc = total_ranks;
    rebalance_parameters.mesh_partitioning_parameters.imbalance_tol = imbalance_tol;
    rebalance_parameters.mesh_partitioning_parameters.rebalance_type =
        Core::Rebalance::RebalanceType::hypergraph;

    Core::FE::DiscretizationBuilder<3> builder(comm);

    if (my_rank == 0)
    {
      // Connect the nodes into elements on the owning ranks
      for (int i = 0; i < subdivisions; ++i)
      {
        for (int j = 0; j < subdivisions; ++j)
        {
          for (int k = 0; k < subdivisions; ++k)
          {
            const int ele_id = i * subdivisions * subdivisions + j * subdivisions + k;

            const std::array nodeids = {lid(i, j, k), lid(i + 1, j, k), lid(i + 1, j + 1, k),
                lid(i, j + 1, k), lid(i, j, k + 1), lid(i + 1, j, k + 1), lid(i + 1, j + 1, k + 1),
                lid(i, j + 1, k + 1)};

            builder.add_element(Core::FE::CellType::hex8, nodeids, ele_id,
                {.num_dof_per_node = 3, .num_dof_per_element = 0});
          }
        }
      }

      const double increment = 1.0 / subdivisions;
      // Add all nodes of the partitioned hypercube
      for (int i = 0; i < subdivisions + 1; ++i)
      {
        for (int j = 0; j < subdivisions + 1; ++j)
        {
          for (int k = 0; k < subdivisions + 1; ++k)
          {
            const std::array<double, 3> coords = {i * increment, j * increment, k * increment};
            builder.add_node(coords, lid(i, j, k), nullptr);
          }
        }
      }
    }
    builder.build(discretization, rebalance_parameters);
    discretization.fill_complete();

    FOUR_C_ASSERT(discretization.num_global_elements() == total_elements, "Internal error.");
  }


  /**
   * Fill the given @p discretization with the cells and points from the given @p mesh with @p
   * PureGeometryElements.
   */
  inline void fill_discretization_from_mesh(
      Core::FE::Discretization& dis, const FourC::Core::IO::MeshInput::Mesh<3>& mesh)
  {
    // Create discretization from mesh and redistribute
    Core::FE::DiscretizationBuilder<3> builder(dis.get_comm());

    for (const auto& point : mesh.points_with_data())
    {
      builder.add_node(point.coordinate(), point.id(), nullptr);
    }

    int cell_id = 0;
    for (const auto& [_, block] : mesh.cell_blocks())
    {
      for (const auto& cell : block.cells())
      {
        builder.add_element(
            block.cell_type, cell, cell_id, {.num_dof_per_node = 3, .num_dof_per_element = 0});

        cell_id++;
      }
    }
    Core::Rebalance::RebalanceParameters rebalance_parameters;

    builder.build(dis, rebalance_parameters);
  }


  /**
   * Create a tree of 1D line elements with a total of `2^levels - 1` elements.
   */
  inline void fill_tree_1d_lines(
      Core::FE::Discretization& discretization, unsigned levels, MPI_Comm comm)
  {
    FOUR_C_ASSERT(levels > 0, "Invalid number of levels.");
    discretization.clear_discret();

    const int my_rank = Core::Communication::my_mpi_rank(comm);

    const auto leaf_on_level = [&](unsigned level, unsigned ele_on_level) -> int
    { return std::pow(2, level) + ele_on_level; };

    Core::FE::DiscretizationBuilder<3> builder(comm);

    if (my_rank == 0)
    {
      // Add the root element
      {
        const std::array nodeids{0, 1};
        builder.add_element(Core::FE::CellType::line2, nodeids, 0,
            {.num_dof_per_node = 1, .num_dof_per_element = 0});
      }

      int ele_id = 1;
      for (unsigned level = 1; level < levels; ++level)
      {
        const unsigned n_elements = std::pow(2, level);
        for (unsigned ele_on_level = 0; ele_on_level < n_elements; ++ele_on_level)
        {
          const unsigned parent_ele = ele_on_level / 2;
          const std::array nodeids{
              leaf_on_level(level - 1, parent_ele), leaf_on_level(level, ele_on_level)};

          builder.add_element(Core::FE::CellType::line2, nodeids, ele_id,
              {.num_dof_per_node = 1, .num_dof_per_element = 0});

          ++ele_id;
        }
      }

      // Add the root node
      {
        const std::array coords = {0.0, 0.0, 0.0};
        builder.add_node(coords, 0, nullptr);
      }

      const double y_inc = 1.0;
      const double x_inc = 1.0;
      for (unsigned level = 0; level < levels; ++level)
      {
        const unsigned n_elements = std::pow(2, level);
        for (unsigned ele_on_level = 0; ele_on_level < n_elements; ++ele_on_level)
        {
          const int node_id = leaf_on_level(level, ele_on_level);
          const std::array coords = {ele_on_level * x_inc, level * y_inc, 0.0};
          builder.add_node(coords, node_id, nullptr);
        }
      }
    }

    const double imbalance_tol(1.1);
    Core::Rebalance::RebalanceParameters rebalance_parameters;
    rebalance_parameters.mesh_partitioning_parameters.imbalance_tol = imbalance_tol;
    rebalance_parameters.mesh_partitioning_parameters.rebalance_type =
        Core::Rebalance::RebalanceType::hypergraph;

    builder.build(discretization, rebalance_parameters);
    discretization.fill_complete();

    [[maybe_unused]] const int n_total_elements = std::pow(2, levels) - 1;
    FOUR_C_ASSERT(discretization.num_global_elements() == n_total_elements, "Internal error.");
  }

  inline void fill_single_tet(Core::FE::Discretization& discretization)
  {
    Core::FE::DiscretizationBuilder<3> builder(discretization.get_comm());

    if (Core::Communication::my_mpi_rank(discretization.get_comm()) == 0)
    {
      const std::array nodeids{0, 1, 2, 3};

      builder.add_element(
          Core::FE::CellType::tet4, nodeids, 0, {.num_dof_per_node = 3, .num_dof_per_element = 0});

      builder.add_node(std::array<double, 3>{0.0, 0.0, 0.0}, 0, nullptr);
      builder.add_node(std::array<double, 3>{1.0, 0.0, 0.0}, 1, nullptr);
      builder.add_node(std::array<double, 3>{0.0, 1.0, 0.0}, 2, nullptr);
      builder.add_node(std::array<double, 3>{0.0, 0.0, 1.0}, 3, nullptr);
    }

    Core::Rebalance::RebalanceParameters rebalance_parameters;
    builder.build(discretization, rebalance_parameters);
    discretization.fill_complete();
  }

  inline void fill_cylindrical_hex27(Core::FE::Discretization& discretization, MPI_Comm comm)
  {
    Core::FE::DiscretizationBuilder<3> builder(comm);

    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      std::array<int, 27> nodeids{};
      std::iota(nodeids.begin(), nodeids.end(), 0);

      const double inner = 1.0;
      const double outer = 1.1;
      const double depth = -0.1;
      const double half = (outer + inner) / 2;
      const double angle = 0.1;

      builder.add_element(
          Core::FE::CellType::hex27, nodeids, 0, {.num_dof_per_node = 3, .num_dof_per_element = 0});

      const auto add_node_polar = [&](int id, std::array<double, 3> coords_polar)
      {
        const std::array coords_cartesian{coords_polar[0] * std::cos(coords_polar[1]),
            coords_polar[0] * std::sin(coords_polar[1]), coords_polar[2]};
        builder.add_node(coords_cartesian, id, nullptr);
      };

      add_node_polar(0, {inner, 0.0, 0.0});
      add_node_polar(1, {outer, 0.0, 0.0});
      add_node_polar(2, {outer, 0.0, depth});
      add_node_polar(3, {inner, 0.0, depth});

      add_node_polar(4, {inner, angle, 0.0});
      add_node_polar(5, {outer, angle, 0.0});
      add_node_polar(6, {outer, angle, depth});
      add_node_polar(7, {inner, angle, depth});

      add_node_polar(8, {half, 0.0, 0.0});
      add_node_polar(10, {half, 0.0, depth});

      add_node_polar(9, {outer, 0.0, depth / 2});
      add_node_polar(11, {inner, 0.0, depth / 2});

      add_node_polar(16, {half, angle, 0.0});
      add_node_polar(18, {half, angle, depth});

      add_node_polar(12, {inner, (angle / 2), 0.0});
      add_node_polar(15, {inner, (angle / 2), depth});

      add_node_polar(13, {outer, (angle / 2), 0.0});
      add_node_polar(14, {outer, (angle / 2), depth});

      add_node_polar(19, {inner, angle, depth / 2});
      add_node_polar(17, {outer, angle, depth / 2});

      add_node_polar(20, {half, 0.0, depth / 2});
      add_node_polar(21, {half, (angle / 2), 0});
      add_node_polar(22, {outer, (angle / 2), depth / 2});
      add_node_polar(23, {half, (angle / 2), depth});
      add_node_polar(24, {inner, (angle / 2), depth / 2});
      add_node_polar(25, {half, (angle), depth / 2});
      add_node_polar(26, {half, (angle / 2), depth / 2});
    }

    Core::Rebalance::RebalanceParameters rebalance_parameters;
    builder.build(discretization, rebalance_parameters);
    discretization.fill_complete();
  }

  inline void fill_undeformed_hex27(Core::FE::Discretization& discretization, MPI_Comm comm)
  {
    Core::FE::DiscretizationBuilder<3> builder(comm);

    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      std::array<int, 27> nodeids{};
      std::iota(nodeids.begin(), nodeids.end(), 0);

      builder.add_element(
          Core::FE::CellType::hex27, nodeids, 0, {.num_dof_per_node = 3, .num_dof_per_element = 0});

      const std::vector<std::array<double, 3>> coords{{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0},
          {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0},
          {-1.0, 1.0, 1.0}, {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0},
          {-1.0, 0.0, -1.0}, {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0},
          {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {-1.0, 0.0, 1.0}, {0.0, 0.0, -1.0},
          {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
          {0.0, 0.0, 0.0}};

      int counter = 0;
      for (const auto& coord : coords) builder.add_node(coord, counter++, nullptr);
    }

    Core::Rebalance::RebalanceParameters rebalance_parameters;
    builder.build(discretization, rebalance_parameters);
    discretization.fill_complete();
  }

  inline void fill_deformed_hex27(Core::FE::Discretization& discretization, MPI_Comm comm)
  {
    Core::FE::DiscretizationBuilder<3> builder(comm);

    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      std::array<int, 27> nodeids{};
      std::iota(nodeids.begin(), nodeids.end(), 0);

      builder.add_element(
          Core::FE::CellType::hex27, nodeids, 0, {.num_dof_per_node = 3, .num_dof_per_element = 0});

      const std::vector<std::array<double, 3>> coords{{-0.9, -1.0, -1.0}, {1.0, -1.0, -1.0},
          {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.2}, {1.0, 1.0, 1.0},
          {-1.0, 1.0, 1.0}, {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0},
          {-1.0, 0.0, -1.0}, {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {.9, 1.0, 0.0}, {-1.0, 1.0, 0.0},
          {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {-1.0, 0.0, 1.0}, {0.0, 0.0, -1.0},
          {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
          {0.0, 0.0, 0.0}};

      int counter = 0;
      for (const auto& coord : coords) builder.add_node(coord, counter++, nullptr);
    }

    Core::Rebalance::RebalanceParameters rebalance_parameters;
    builder.build(discretization, rebalance_parameters);
    discretization.fill_complete();
  }

}  // namespace TESTING

#endif
