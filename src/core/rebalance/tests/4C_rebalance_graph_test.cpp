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

      map_ = std::make_shared<Core::LinAlg::Map>(NumGlobalElements, 0, comm_);
    }

   protected:
    std::shared_ptr<Core::LinAlg::Map> map_;

    int NumGlobalElements = 20;
    MPI_Comm comm_{MPI_COMM_WORLD};
    Core::IO::Verbositylevel verbosity_;
  };

  /**
   * Checks serial rebalancing, where input and output graph should be the same.
   */
  TEST_F(RebalanceGraph, RebalanceGraphSerial)
  {
    Core::LinAlg::Graph graph(*map_, 3);

    for (int row = 0; row < map_->num_my_elements(); row++)
    {
      if (row == 0)
      {
        std::array<int, 2> indices{row, row + 1};
        auto indices_view = std::span(indices.data(), 2);
        graph.insert_global_indices(row, indices_view);
      }
      else if (row == map_->num_my_elements() - 1)
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

    graph.fill_complete();
    graph.optimize_storage();

    Teuchos::ParameterList rebalance_params;
    rebalance_params.set("algorithm", "phg");

    auto rebalanced_graph = Core::Rebalance::rebalance_graph(graph, rebalance_params);

    EXPECT_EQ(rebalanced_graph->num_global_nonzeros(), graph.num_global_nonzeros());
    EXPECT_EQ(rebalanced_graph->num_local_nonzeros(), graph.num_local_nonzeros());
    EXPECT_EQ(rebalanced_graph->num_global_nonzeros(), rebalanced_graph->num_local_nonzeros());
  }
}  // namespace
