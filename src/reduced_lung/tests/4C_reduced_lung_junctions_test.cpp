// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_junctions.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_red_airways_elementbase.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <mpi.h>

#include <array>
#include <map>
#include <memory>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::Core::LinAlg;
  using namespace FourC::ReducedLung::Junctions;

  void expect_row_entries(Core::LinAlg::SparseMatrix& mat, int row,
      std::initializer_list<std::pair<int, double>> expected)
  {
    int n_entries = 0;
    double* values = nullptr;
    int* cols = nullptr;
    mat.extract_my_row_view(row, n_entries, values, cols);

    ASSERT_EQ(n_entries, static_cast<int>(expected.size()));
    for (const auto& [col, val] : expected)
    {
      bool found = false;
      for (int i = 0; i < n_entries; ++i)
      {
        if (cols[i] == col)
        {
          found = true;
          EXPECT_DOUBLE_EQ(values[i], val);
          break;
        }
      }
      EXPECT_TRUE(found) << "Missing column " << col << " in row " << row;
    }
  }

  std::unique_ptr<Core::FE::Discretization> make_airway_discretization(
      const std::vector<int>& node_ids, const std::vector<std::array<int, 2>>& element_nodes)
  {
    auto dis = std::make_unique<Core::FE::Discretization>("junctions_test", MPI_COMM_WORLD, 3);

    for (int node_id : node_ids)
    {
      std::array<double, 3> coords{static_cast<double>(node_id), 0.0, 0.0};
      dis->add_node(coords, node_id, nullptr);
    }

    for (size_t i = 0; i < element_nodes.size(); ++i)
    {
      auto ele = std::make_shared<Discret::Elements::RedAirway>(static_cast<int>(i), 0);
      ele->set_node_ids(2, element_nodes[i].data());
      dis->add_element(ele);
    }

    dis->fill_complete(Core::FE::OptionsFillComplete::none());
    return dis;
  }

  TEST(JunctionsTests, ConnectionResidualAssembly)
  {
    ConnectionData connections;
    BifurcationData bifurcations;

    connections.add_connection(0, 0, 1, {0, 1, 2, 3});
    connections.first_local_equation_id[0] = 0;
    connections.local_dof_ids[0] = {0, 1, 2, 3};

    Core::LinAlg::Map row_map(-1, 2, 0, MPI_COMM_WORLD);
    Core::LinAlg::Map col_map(-1, 4, 0, MPI_COMM_WORLD);
    Core::LinAlg::Vector<double> rhs(row_map, true);
    Core::LinAlg::Vector<double> locally_relevant_dofs(col_map, true);

    locally_relevant_dofs.get_values()[0] = 10.0;
    locally_relevant_dofs.get_values()[1] = 5.0;
    locally_relevant_dofs.get_values()[2] = 3.0;
    locally_relevant_dofs.get_values()[3] = 4.0;

    update_residual_vector(rhs, connections, bifurcations, locally_relevant_dofs);

    EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[0], 10.0 - 5.0);
    EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[1], 3.0 - 4.0);
  }

  TEST(JunctionsTests, BifurcationResidualAssembly)
  {
    ConnectionData connections;
    BifurcationData bifurcations;

    bifurcations.add_bifurcation(0, 0, 1, 2, {0, 1, 2, 3, 4, 5});
    bifurcations.first_local_equation_id[0] = 0;
    bifurcations.local_dof_ids[0] = {0, 1, 2, 3, 4, 5};

    Core::LinAlg::Map row_map(-1, 3, 0, MPI_COMM_WORLD);
    Core::LinAlg::Map col_map(-1, 6, 0, MPI_COMM_WORLD);
    Core::LinAlg::Vector<double> rhs(row_map, true);
    Core::LinAlg::Vector<double> locally_relevant_dofs(col_map, true);

    locally_relevant_dofs.get_values()[0] = 10.0;
    locally_relevant_dofs.get_values()[1] = 7.0;
    locally_relevant_dofs.get_values()[2] = 6.0;
    locally_relevant_dofs.get_values()[3] = 8.0;
    locally_relevant_dofs.get_values()[4] = 3.0;
    locally_relevant_dofs.get_values()[5] = 4.0;

    update_residual_vector(rhs, connections, bifurcations, locally_relevant_dofs);

    EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[0], 10.0 - 7.0);
    EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[1], 10.0 - 6.0);
    EXPECT_DOUBLE_EQ(rhs.local_values_as_span()[2], 8.0 - 3.0 - 4.0);
  }

  TEST(JunctionsTests, ConnectionJacobianAssembledOnce)
  {
    ConnectionData connections;
    BifurcationData bifurcations;

    connections.add_connection(0, 0, 1, {0, 1, 2, 3});
    connections.first_local_equation_id[0] = 0;
    connections.local_dof_ids[0] = {0, 1, 2, 3};

    Core::LinAlg::Map row_map(-1, 2, 0, MPI_COMM_WORLD);
    Core::LinAlg::Map col_map(-1, 4, 0, MPI_COMM_WORLD);
    Core::LinAlg::SparseMatrix jac(row_map, col_map, 2);

    update_jacobian(jac, connections, bifurcations);
    jac.complete();

    expect_row_entries(jac, 0, {{0, 1.0}, {1, -1.0}});
    expect_row_entries(jac, 1, {{2, 1.0}, {3, -1.0}});

    update_jacobian(jac, connections, bifurcations);

    expect_row_entries(jac, 0, {{0, 1.0}, {1, -1.0}});
    expect_row_entries(jac, 1, {{2, 1.0}, {3, -1.0}});
  }

  TEST(JunctionsTests, BifurcationJacobianAssembledOnce)
  {
    ConnectionData connections;
    BifurcationData bifurcations;

    bifurcations.add_bifurcation(0, 0, 1, 2, {0, 1, 2, 3, 4, 5});
    bifurcations.first_local_equation_id[0] = 0;
    bifurcations.local_dof_ids[0] = {0, 1, 2, 3, 4, 5};

    Core::LinAlg::Map row_map(-1, 3, 0, MPI_COMM_WORLD);
    Core::LinAlg::Map col_map(-1, 6, 0, MPI_COMM_WORLD);
    Core::LinAlg::SparseMatrix jac(row_map, col_map, 3);

    update_jacobian(jac, connections, bifurcations);
    jac.complete();

    expect_row_entries(jac, 0, {{0, 1.0}, {1, -1.0}});
    expect_row_entries(jac, 1, {{0, 1.0}, {2, -1.0}});
    expect_row_entries(jac, 2, {{3, 1.0}, {4, -1.0}, {5, -1.0}});

    update_jacobian(jac, connections, bifurcations);

    expect_row_entries(jac, 0, {{0, 1.0}, {1, -1.0}});
    expect_row_entries(jac, 1, {{0, 1.0}, {2, -1.0}});
    expect_row_entries(jac, 2, {{3, 1.0}, {4, -1.0}, {5, -1.0}});
  }

  TEST(JunctionsTests, AssignLocalEquationIds)
  {
    ConnectionData connections;
    BifurcationData bifurcations;

    connections.add_connection(0, 0, 1, {0, 1, 2, 3});
    connections.add_connection(1, 1, 2, {3, 4, 5, 6});
    bifurcations.add_bifurcation(0, 0, 1, 2, {0, 1, 2, 3, 4, 5});

    int n_local_equations = 5;
    assign_junction_local_equation_ids(connections, bifurcations, n_local_equations);

    EXPECT_EQ(connections.first_local_equation_id[0], 5);
    EXPECT_EQ(connections.first_local_equation_id[1], 7);
    EXPECT_EQ(bifurcations.first_local_equation_id[0], 9);
    EXPECT_EQ(n_local_equations, 12);
  }

  TEST(JunctionsTests, AssignLocalDofIds)
  {
    ConnectionData connections;
    BifurcationData bifurcations;

    connections.add_connection(0, 0, 1, {10, 11, 20, 21});
    bifurcations.add_bifurcation(0, 0, 1, 2, {10, 11, 30, 20, 21, 31});

    std::array<int, 6> global_dofs{10, 11, 20, 21, 30, 31};
    Core::LinAlg::Map col_map(-1, std::span<const int>(global_dofs), 0, MPI_COMM_WORLD);

    assign_junction_local_dof_ids(col_map, connections, bifurcations);

    EXPECT_EQ(connections.local_dof_ids[0], (std::array<int, 4>{0, 1, 2, 3}));
    EXPECT_EQ(bifurcations.local_dof_ids[0], (std::array<int, 6>{0, 1, 4, 2, 3, 5}));
  }

  TEST(JunctionsTests, CreateConnection)
  {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1)
    {
      GTEST_SKIP() << "Junction creation tests require a serial communicator.";
    }

    auto dis = make_airway_discretization({1, 2, 3}, {{1, 2}, {2, 3}});

    std::map<int, std::vector<int>> ele_ids_per_node{{1, {0}}, {2, {0, 1}}, {3, {1}}};
    std::map<int, int> global_dof_per_ele{{0, 3}, {1, 3}};
    std::map<int, int> first_global_dof_of_ele{{0, 0}, {1, 3}};

    ConnectionData connections;
    BifurcationData bifurcations;

    create_junctions(*dis, ele_ids_per_node, global_dof_per_ele, first_global_dof_of_ele,
        connections, bifurcations);

    ASSERT_EQ(connections.size(), 1u);
    ASSERT_EQ(bifurcations.size(), 0u);

    EXPECT_EQ(connections.local_connection_id[0], 0);
    EXPECT_EQ(connections.global_parent_element_id[0], 0);
    EXPECT_EQ(connections.global_child_element_id[0], 1);
    EXPECT_EQ(connections.global_dof_ids[0], (std::array<int, 4>{1, 3, 2, 5}));
  }

  TEST(JunctionsTests, CreateBifurcation)
  {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1)
    {
      GTEST_SKIP() << "Junction creation tests require a serial communicator.";
    }

    auto dis = make_airway_discretization({1, 2, 3, 4}, {{1, 2}, {2, 3}, {2, 4}});

    std::map<int, std::vector<int>> ele_ids_per_node{{1, {0}}, {2, {0, 1, 2}}, {3, {1}}, {4, {2}}};
    std::map<int, int> global_dof_per_ele{{0, 3}, {1, 3}, {2, 3}};
    std::map<int, int> first_global_dof_of_ele{{0, 0}, {1, 3}, {2, 6}};

    ConnectionData connections;
    BifurcationData bifurcations;

    create_junctions(*dis, ele_ids_per_node, global_dof_per_ele, first_global_dof_of_ele,
        connections, bifurcations);

    ASSERT_EQ(connections.size(), 0u);
    ASSERT_EQ(bifurcations.size(), 1u);

    EXPECT_EQ(bifurcations.local_bifurcation_id[0], 0);
    EXPECT_EQ(bifurcations.global_parent_element_id[0], 0);
    EXPECT_EQ(bifurcations.global_child_1_element_id[0], 1);
    EXPECT_EQ(bifurcations.global_child_2_element_id[0], 2);
    EXPECT_EQ(bifurcations.global_dof_ids[0], (std::array<int, 6>{1, 3, 6, 2, 5, 8}));
  }

  TEST(JunctionsTests, CreateJunctionsMissingAdjacencyThrows)
  {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1)
    {
      GTEST_SKIP() << "Junction creation tests require a serial communicator.";
    }

    auto dis = make_airway_discretization({1, 2, 3}, {{1, 2}, {2, 3}});

    std::map<int, std::vector<int>> ele_ids_per_node{{1, {0}}, {3, {1}}};
    std::map<int, int> global_dof_per_ele{{0, 3}, {1, 3}};
    std::map<int, int> first_global_dof_of_ele{{0, 0}, {1, 3}};

    ConnectionData connections;
    BifurcationData bifurcations;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(create_junctions(*dis, ele_ids_per_node, global_dof_per_ele,
                                         first_global_dof_of_ele, connections, bifurcations),
        Core::Exception, "missing from element adjacency map");
  }

  TEST(JunctionsTests, CreateJunctionsDuplicateConnectionThrows)
  {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1)
    {
      GTEST_SKIP() << "Junction creation tests require a serial communicator.";
    }

    auto dis = make_airway_discretization({1, 2, 3}, {{1, 2}, {3, 2}});

    std::map<int, std::vector<int>> ele_ids_per_node{{1, {0}}, {2, {0, 1}}, {3, {1}}};
    std::map<int, int> global_dof_per_ele{{0, 3}, {1, 3}};
    std::map<int, int> first_global_dof_of_ele{{0, 0}, {1, 3}};

    ConnectionData connections;
    BifurcationData bifurcations;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(create_junctions(*dis, ele_ids_per_node, global_dof_per_ele,
                                         first_global_dof_of_ele, connections, bifurcations),
        Core::Exception, "Second connection entity at parent element");
  }

  TEST(JunctionsTests, CreateJunctionsTooManyElementsThrows)
  {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1)
    {
      GTEST_SKIP() << "Junction creation tests require a serial communicator.";
    }

    auto dis = make_airway_discretization({1, 2, 3}, {{1, 2}, {2, 3}});

    std::map<int, std::vector<int>> ele_ids_per_node{{1, {0}}, {2, {0, 1, 2, 3}}, {3, {1}}};
    std::map<int, int> global_dof_per_ele{{0, 3}, {1, 3}};
    std::map<int, int> first_global_dof_of_ele{{0, 0}, {1, 3}};

    ConnectionData connections;
    BifurcationData bifurcations;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(create_junctions(*dis, ele_ids_per_node, global_dof_per_ele,
                                         first_global_dof_of_ele, connections, bifurcations),
        Core::Exception, "Too many elements at junction");
  }
}  // namespace
