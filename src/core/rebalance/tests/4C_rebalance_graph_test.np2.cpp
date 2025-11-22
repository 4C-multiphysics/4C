// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_rebalance_graph_based.hpp"

#include <Teuchos_ParameterList.hpp>

namespace
{
  using namespace FourC;

  class RebalanceGraph : public ::testing::Test
  {
   public:
    RebalanceGraph()
    {
      verbosity_ = Core::IO::minimal;

      testdis_ = std::make_shared<Core::FE::Discretization>("dummy", comm_, 3);
    }

   protected:
    std::shared_ptr<Core::FE::Discretization> testdis_;

    MPI_Comm comm_{MPI_COMM_WORLD};
    Core::IO::Verbositylevel verbosity_;
  };

  /**
   * Checks the rebalancing of an empty graph, which should throw.
   */
  TEST_F(RebalanceGraph, RebalanceEmptyGraph)
  {
    Core::LinAlg::Map map(20, 0, comm_);

    Core::LinAlg::Graph empty_graph(map, 3);

    EXPECT_EQ(false, empty_graph.filled());

    Teuchos::ParameterList rebalance_params;
    rebalance_params.set("algorithm", "phg");

    EXPECT_ANY_THROW(Core::Rebalance::rebalance_graph(empty_graph, rebalance_params));
  }

  /**
   * Checks parallel rebalancing, where the initial graph only exists on one processor.
   * After rebalancing the graph is equally split over two processors.
   */
  TEST_F(RebalanceGraph, RebalanceGraphEmptyProc)
  {
    const int num_global_elements = 20;
    int num_local_elements;

    if (Core::Communication::my_mpi_rank(comm_) == 0)
      num_local_elements = num_global_elements;
    else
      num_local_elements = 0;

    Core::LinAlg::Map map(num_global_elements, num_local_elements, 0, comm_);

    Core::LinAlg::Graph graph(map, 3);

    if (Core::Communication::my_mpi_rank(comm_) == 0)
    {
      for (int row = 0; row < map.num_my_elements(); row++)
      {
        if (row == 0)
        {
          std::array<int, 2> indices{row, row + 1};
          auto indices_view = std::span(indices.data(), 2);
          graph.insert_global_indices(row, indices_view);
        }
        else if (row == map.num_my_elements() - 1)
        {
          std::array<int, 2> indices{row - 1, row};
          auto indices_view = std::span(indices.data(), 2);
          graph.insert_global_indices(row, indices_view);
        }
        else
        {
          std::array<int, 3> indices{row - 1, row, row + 1};
          auto indices_view = std::span(indices.data(), 3);
          graph.insert_global_indices(row, indices_view);
        }
      }
    }

    graph.fill_complete();
    graph.optimize_storage();

    Teuchos::ParameterList rebalance_params;
    rebalance_params.set("algorithm", "phg");

    auto rebalanced_graph = Core::Rebalance::rebalance_graph(graph, rebalance_params);

    EXPECT_EQ(rebalanced_graph->row_map().num_global_elements(), 20);
    EXPECT_EQ(rebalanced_graph->row_map().num_my_elements(), 10);
    EXPECT_EQ(rebalanced_graph->num_local_nonzeros(), 29);
  }

  /**
   * Checks the construction of a monolithic graph of an empty discretization, which should throw.
   */
  TEST_F(RebalanceGraph, BuildMonolithicGraphEmptyDiscretization)
  {
    EXPECT_EQ(false, testdis_->filled());

    Teuchos::ParameterList geometric_search_params;
    geometric_search_params.set("BEAM_RADIUS_EXTENSION_FACTOR", 1.0);
    geometric_search_params.set("SPHERE_RADIUS_EXTENSION_FACTOR", 1.0);
    geometric_search_params.set("WRITE_GEOMETRIC_SEARCH_VISUALIZATION", false);
    Teuchos::ParameterList io_params;
    io_params.set("VERBOSITY", verbosity_);

    Core::GeometricSearch::GeometricSearchParams search_params(geometric_search_params, io_params);

    EXPECT_ANY_THROW(Core::Rebalance::build_monolithic_node_graph(*testdis_, search_params));
  }
}  // namespace
