// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_rebalance_weights.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_vector.hpp"

#include <algorithm>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::vector<int> build_local_node_ids(const Core::Elements::Element& ele)
  {
    const Core::Nodes::Node* const* nodes = ele.nodes();
    std::vector<int> node_ids(ele.num_node());
    for (int n = 0; n < ele.num_node(); ++n) node_ids[n] = nodes[n]->id();
    return node_ids;
  }

  std::vector<int> build_local_node_owners(const Core::Elements::Element& ele)
  {
    const Core::Nodes::Node* const* nodes = ele.nodes();
    std::vector<int> node_owners(ele.num_node());
    for (int n = 0; n < ele.num_node(); ++n) node_owners[n] = nodes[n]->owner();
    return node_owners;
  }
}  // namespace

Core::Rebalance::PartitionWeights Core::Rebalance::build_static_partition_weights(
    const Core::FE::Discretization& dis)
{
  const Core::LinAlg::Map* noderowmap = dis.node_row_map();

  PartitionWeights weights{
      .node_weights = std::make_shared<Core::LinAlg::Vector<double>>(*noderowmap, true),
      .edge_weights = std::make_shared<Core::LinAlg::SparseMatrix>(*noderowmap, 15)};

  for (int i = 0; i < dis.element_row_map()->num_my_elements(); ++i)
  {
    Core::Elements::Element* ele = dis.l_row_element(i);
    const std::vector<int> lm = build_local_node_ids(*ele);
    const std::vector<int> lmrowowner = build_local_node_owners(*ele);

    Core::LinAlg::SerialDenseMatrix edgeweights_ele;
    Core::LinAlg::SerialDenseVector nodeweights_ele;
    ele->nodal_connectivity(edgeweights_ele, nodeweights_ele);

    Core::LinAlg::assemble(*weights.edge_weights, edgeweights_ele, lm, lmrowowner, lm);
    Core::LinAlg::assemble(*weights.node_weights, nodeweights_ele, lm, lmrowowner);
  }

  weights.edge_weights->complete();
  return weights;
}

Core::Rebalance::PartitionWeights Core::Rebalance::build_eval_time_partition_weights(
    const Core::FE::Discretization& dis, const Core::LinAlg::Graph& graph,
    const double edge_weight_multiplier)
{
  const Core::LinAlg::Map& graph_row_map = graph.row_map();
  const Core::LinAlg::Map point_row_map(graph_row_map.num_global_elements(),
      graph_row_map.num_my_elements(), graph_row_map.my_global_elements(),
      graph_row_map.index_base(), graph.get_comm());

  const double local_eval_time_sum = [&dis]()
  {
    double sum = 0.0;
    for (int i = 0; i < dis.element_row_map()->num_my_elements(); ++i)
    {
      const Core::Elements::Element* ele = dis.l_row_element(i);
      sum += std::max(ele->eval_time(), 1.0e-12);
    }
    return sum;
  }();
  const double global_eval_time_sum =
      Core::Communication::sum_all(local_eval_time_sum, dis.get_comm());
  const double average_eval_time = std::max(
      global_eval_time_sum / static_cast<double>(dis.element_row_map()->num_global_elements()),
      1.0e-12);
  const double scaled_average_eval_time = edge_weight_multiplier * average_eval_time;

  PartitionWeights weights{
      .node_weights = std::make_shared<Core::LinAlg::Vector<double>>(point_row_map, true),
      .edge_weights = std::make_shared<Core::LinAlg::SparseMatrix>(point_row_map, 15)};

  std::vector<double> adjacent_eval_time_sum(point_row_map.num_my_elements(), 0.0);
  std::vector<int> adjacent_element_count(point_row_map.num_my_elements(), 0);

  for (int i = 0; i < dis.element_row_map()->num_my_elements(); ++i)
  {
    const Core::Elements::Element* ele = dis.l_row_element(i);
    const double element_eval_time = std::max(ele->eval_time(), 1.0e-12);
    const std::vector<int> node_ids = build_local_node_ids(*ele);
    for (const int node_id : node_ids)
    {
      const int local_node_id = point_row_map.lid(node_id);
      if (local_node_id == -1) continue;
      adjacent_eval_time_sum[local_node_id] += element_eval_time;
      adjacent_element_count[local_node_id] += 1;
    }
  }

  for (int local_node_id = 0; local_node_id < point_row_map.num_my_elements(); ++local_node_id)
  {
    const double average_adjacent_eval_time =
        adjacent_element_count[local_node_id] > 0
            ? adjacent_eval_time_sum[local_node_id] /
                  static_cast<double>(adjacent_element_count[local_node_id])
            : 1.0e-12;
    weights.node_weights->replace_local_value(local_node_id, average_adjacent_eval_time);
  }

  for (int local_row = 0; local_row < graph_row_map.num_my_elements(); ++local_row)
  {
    const int global_row = graph_row_map.gid(local_row);
    std::span<int> indices;
    graph.extract_local_row_view(local_row, indices);
    std::vector<double> values(indices.size(), scaled_average_eval_time);
    weights.edge_weights->insert_global_values(
        global_row, static_cast<int>(indices.size()), values.data(), indices.data());
  }

  weights.edge_weights->complete();
  return weights;
}

FOUR_C_NAMESPACE_CLOSE
