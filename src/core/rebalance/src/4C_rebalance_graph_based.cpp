// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_rebalance_graph_based.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_geometric_search_bounding_volume.hpp"
#include "4C_geometric_search_distributed_tree.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_rebalance.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_XpetraCrsGraphAdapter.hpp>
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>

#include <algorithm>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::pair<std::shared_ptr<Core::LinAlg::Map>, std::shared_ptr<Core::LinAlg::Map>>
Core::Rebalance::rebalance_node_maps(const Core::LinAlg::Graph& initialGraph,
    Teuchos::ParameterList& rebalanceParams,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& initialNodeWeights,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& initialEdgeWeights,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& initialNodeCoordinates)
{
  TEUCHOS_FUNC_TIME_MONITOR("Rebalance::rebalance_node_maps");

  // Compute rebalanced graph
  auto balanced_graph = Rebalance::rebalance_graph(initialGraph, rebalanceParams,
      initialNodeWeights, initialEdgeWeights, initialNodeCoordinates);

  // extract repartitioned maps
  std::shared_ptr<Core::LinAlg::Map> rownodes =
      std::make_shared<Core::LinAlg::Map>(-1, balanced_graph->row_map().num_my_elements(),
          balanced_graph->row_map().my_global_elements(), 0, initialGraph.get_comm());
  std::shared_ptr<Core::LinAlg::Map> colnodes =
      std::make_shared<Core::LinAlg::Map>(-1, balanced_graph->col_map().num_my_elements(),
          balanced_graph->col_map().my_global_elements(), 0, initialGraph.get_comm());

  return {rownodes, colnodes};
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Graph> Core::Rebalance::rebalance_graph(
    const Core::LinAlg::Graph& initialGraph, Teuchos::ParameterList& rebalanceParams,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& initialNodeWeights,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& initialEdgeWeights,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& initialNodeCoordinates)
{
  TEUCHOS_FUNC_TIME_MONITOR("Rebalance::rebalance_graph");

  using GraphAdapter = Zoltan2::XpetraCrsGraphAdapter<Epetra_CrsGraph, Epetra_MultiVector>;
  using VectorAdapter = Zoltan2::XpetraMultiVectorAdapter<Epetra_MultiVector>;

  if (!initialGraph.filled())
    FOUR_C_THROW("The graph needs to be fill completed before being able to be rebalanced.");

  GraphAdapter graphAdapter(Teuchos::rcpFromRef(initialGraph.get_epetra_crs_graph()),
      initialNodeWeights != nullptr ? 1 : 0, initialEdgeWeights != nullptr ? 1 : 0);

  if (initialNodeWeights != nullptr)
  {
    graphAdapter.setVertexWeights(initialNodeWeights->get_values(), 1, 0);
  }

  std::vector<double> edgeWeights;
  if (initialEdgeWeights != nullptr)
  {
    for (int local_row = 0; local_row < initialEdgeWeights->row_map().num_my_elements();
        local_row++)
    {
      int numEntries;
      double* entries;
      int* indices;
      initialEdgeWeights->extract_my_row_view(local_row, numEntries, entries, indices);
      for (int i = 0; i < numEntries; i++) edgeWeights.push_back(entries[i]);
    }
    graphAdapter.setEdgeWeights(edgeWeights.data(), 1, 0);
  }

  std::optional<VectorAdapter> vectorAdapter = std::nullopt;
  if (initialNodeCoordinates != nullptr)
  {
    std::vector<const double*> weights;
    std::vector<int> stride;

    vectorAdapter.emplace(
        Teuchos::rcpFromRef(initialNodeCoordinates->get_epetra_multi_vector()), weights, stride);
    graphAdapter.setCoordinateInput(&vectorAdapter.value());
  }

  Zoltan2::PartitioningProblem<GraphAdapter> problem(
      &graphAdapter, &rebalanceParams, initialGraph.get_comm());
  problem.solve();

  const auto solution = problem.getSolution();

  Teuchos::RCP<Epetra_CrsGraph> balancedGraph = Teuchos::null;
  graphAdapter.applyPartitioningSolution(
      initialGraph.get_epetra_crs_graph(), balancedGraph, solution);

  balancedGraph->FillComplete();
  balancedGraph->OptimizeStorage();

  return std::make_shared<Core::LinAlg::Graph>(*balancedGraph);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::pair<std::shared_ptr<Core::LinAlg::MultiVector<double>>,
    std::shared_ptr<Core::LinAlg::MultiVector<double>>>
Core::Rebalance::rebalance_coordinates(const Core::LinAlg::MultiVector<double>& initialCoordinates,
    Teuchos::ParameterList& rebalanceParams,
    const Core::LinAlg::MultiVector<double>& initialWeights)
{
  TEUCHOS_FUNC_TIME_MONITOR("Rebalance::rebalance_coordinates");

  using VectorAdapter = Zoltan2::XpetraMultiVectorAdapter<Epetra_MultiVector>;

  std::vector<const double*> weights(initialWeights.num_vectors());
  for (int weight_num = 0; weight_num < initialWeights.num_vectors(); weight_num++)
    weights[weight_num] = initialWeights.get_vector(weight_num).get_values();

  std::vector<int> stride(initialWeights.num_vectors(), initialWeights.num_vectors());

  VectorAdapter adapter(
      Teuchos::rcpFromRef(initialCoordinates.get_epetra_multi_vector()), weights, stride);
  Zoltan2::PartitioningProblem<VectorAdapter> problem(&adapter, &rebalanceParams);

  problem.solve();

  Teuchos::RCP<Epetra_MultiVector> balancedCoordinates;
  Teuchos::RCP<Epetra_MultiVector> balancedWeights;

  adapter.applyPartitioningSolution(
      initialCoordinates.get_epetra_multi_vector(), balancedCoordinates, problem.getSolution());
  adapter.applyPartitioningSolution(
      initialWeights.get_epetra_multi_vector(), balancedWeights, problem.getSolution());

  return {std::make_shared<Core::LinAlg::MultiVector<double>>(*balancedCoordinates),
      std::make_shared<Core::LinAlg::MultiVector<double>>(*balancedWeights)};
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::Rebalance::PartitionWeights Core::Rebalance::build_static_partition_weights(
    const Core::FE::Discretization& dis)
{
  const Core::LinAlg::Map* noderowmap = dis.node_row_map();

  auto crs_ge_weights = std::make_shared<Core::LinAlg::SparseMatrix>(*noderowmap, 15);
  std::shared_ptr<Core::LinAlg::Vector<double>> vweights =
      std::make_shared<Core::LinAlg::Vector<double>>(*noderowmap, true);

  // loop all row elements and get their cost of evaluation
  for (int i = 0; i < dis.element_row_map()->num_my_elements(); ++i)
  {
    Core::Elements::Element* ele = dis.l_row_element(i);
    Core::Nodes::Node** nodes = ele->nodes();
    const int numnode = ele->num_node();
    std::vector<int> lm(numnode);
    std::vector<int> lmrowowner(numnode);
    for (int n = 0; n < numnode; ++n)
    {
      lm[n] = nodes[n]->id();
      lmrowowner[n] = nodes[n]->owner();
    }

    // element vector and matrix for weights of nodes and edges
    Core::LinAlg::SerialDenseMatrix edgeweights_ele;
    Core::LinAlg::SerialDenseVector nodeweights_ele;
    // evaluate elements to get their evaluation cost
    ele->nodal_connectivity(edgeweights_ele, nodeweights_ele);

    Core::LinAlg::assemble(*crs_ge_weights, edgeweights_ele, lm, lmrowowner, lm);
    Core::LinAlg::assemble(*vweights, nodeweights_ele, lm, lmrowowner);
  }

  crs_ge_weights->complete();

  return {vweights, crs_ge_weights};
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
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
    const Core::Nodes::Node* const* nodes = ele->nodes();
    for (int n = 0; n < ele->num_node(); ++n)
    {
      const int local_node_id = point_row_map.lid(nodes[n]->id());
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

std::shared_ptr<const Core::LinAlg::Graph> Core::Rebalance::build_graph(
    Core::FE::Discretization& dis, const Core::LinAlg::Map& element_row_map)
{
  const int myrank = Core::Communication::my_mpi_rank(dis.get_comm());
  const int numproc = Core::Communication::num_mpi_ranks(dis.get_comm());

  // create a set of all nodes that I have
  std::set<int> mynodes;
  for (int lid = 0; lid < element_row_map.num_my_elements(); ++lid)
  {
    Core::Elements::Element* ele = dis.g_element(element_row_map.gid(lid));
    const int numnode = ele->num_node();
    const int* nodeids = ele->node_ids();
    copy(nodeids, nodeids + numnode, inserter(mynodes, mynodes.begin()));
  }

  // build a unique row map from the overlapping sets
  for (int proc = 0; proc < numproc; ++proc)
  {
    int size = 0;
    std::vector<int> recvnodes;
    if (proc == myrank)
    {
      recvnodes.clear();
      std::set<int>::iterator fool;
      for (fool = mynodes.begin(); fool != mynodes.end(); ++fool) recvnodes.push_back(*fool);
      size = (int)recvnodes.size();
    }
    Core::Communication::broadcast(&size, 1, proc, dis.get_comm());
    if (proc != myrank) recvnodes.resize(size);
    Core::Communication::broadcast(recvnodes.data(), size, proc, dis.get_comm());
    if (proc != myrank)
    {
      for (int i = 0; i < size; ++i)
      {
        std::set<int>::iterator fool = mynodes.find(recvnodes[i]);
        if (fool == mynodes.end())
          continue;
        else
          mynodes.erase(fool);
      }
    }
    Core::Communication::barrier(dis.get_comm());
  }

  std::shared_ptr<Core::LinAlg::Map> rownodes = nullptr;
  // copy the set to a vector
  {
    std::vector<int> nodes;
    std::set<int>::iterator fool;
    for (fool = mynodes.begin(); fool != mynodes.end(); ++fool) nodes.push_back(*fool);
    mynodes.clear();
    // create a non-overlapping row map
    rownodes =
        std::make_shared<Core::LinAlg::Map>(-1, (int)nodes.size(), nodes.data(), 0, dis.get_comm());
  }

  // start building the graph object
  std::map<int, std::set<int>> locals;
  std::map<int, std::set<int>> remotes;
  for (int lid = 0; lid < element_row_map.num_my_elements(); ++lid)
  {
    Core::Elements::Element* ele = dis.g_element(element_row_map.gid(lid));
    const int numnode = ele->num_node();
    const int* nodeids = ele->node_ids();
    for (int i = 0; i < numnode; ++i)
    {
      const int lid = rownodes->lid(nodeids[i]);  // am I owner of this gid?
      std::map<int, std::set<int>>* insertmap = nullptr;
      if (lid != -1)
        insertmap = &locals;
      else
        insertmap = &remotes;
      // see whether we already have an entry for nodeids[i]
      std::map<int, std::set<int>>::iterator fool = (*insertmap).find(nodeids[i]);
      if (fool == (*insertmap).end())  // no entry in that row yet
      {
        std::set<int> tmp;
        copy(nodeids, nodeids + numnode, inserter(tmp, tmp.begin()));
        (*insertmap)[nodeids[i]] = tmp;
      }
      else
      {
        std::set<int>& imap = fool->second;
        copy(nodeids, nodeids + numnode, inserter(imap, imap.begin()));
      }
    }
  }

  // run through locals and remotes to find the max bandwidth
  int maxband = 0;
  {
    int smaxband = 0;
    std::map<int, std::set<int>>::iterator fool;
    for (fool = locals.begin(); fool != locals.end(); ++fool)
      if (smaxband < (int)fool->second.size()) smaxband = (int)fool->second.size();
    for (fool = remotes.begin(); fool != remotes.end(); ++fool)
      if (smaxband < (int)fool->second.size()) smaxband = (int)fool->second.size();
    maxband = Core::Communication::max_all(smaxband, dis.get_comm());
  }

  auto graph = std::make_shared<Core::LinAlg::Graph>(*rownodes, maxband);
  Core::Communication::barrier(dis.get_comm());

  // fill all local entries into the graph
  {
    std::map<int, std::set<int>>::iterator fool = locals.begin();
    for (; fool != locals.end(); ++fool)
    {
      const int grid = fool->first;
      std::vector<int> cols;
      std::set<int>::iterator setfool = fool->second.begin();
      for (; setfool != fool->second.end(); ++setfool) cols.push_back(*setfool);
      auto indices = std::span(cols.data(), cols.size());
      graph->insert_global_indices(grid, indices);
    }
    locals.clear();
  }

  Core::Communication::barrier(dis.get_comm());

  // now we need to communicate and add the remote entries
  for (int proc = numproc - 1; proc >= 0; --proc)
  {
    std::vector<int> recvnodes;
    int size = 0;
    if (proc == myrank)
    {
      recvnodes.clear();
      std::map<int, std::set<int>>::iterator mapfool = remotes.begin();
      for (; mapfool != remotes.end(); ++mapfool)
      {
        recvnodes.push_back((int)mapfool->second.size() + 1);  // length of this entry
        recvnodes.push_back(mapfool->first);                   // global row id
        std::set<int>::iterator fool = mapfool->second.begin();
        for (; fool != mapfool->second.end(); ++fool)  // global col ids
          recvnodes.push_back(*fool);
      }
      size = (int)recvnodes.size();
    }
    Core::Communication::broadcast(&size, 1, proc, dis.get_comm());
    if (proc != myrank) recvnodes.resize(size);
    Core::Communication::broadcast(recvnodes.data(), size, proc, dis.get_comm());
    if (proc != myrank && size)
    {
      int* ptr = &recvnodes[0];
      while (ptr < &recvnodes[size - 1])
      {
        int num = *ptr;
        int grid = *(ptr + 1);
        // see whether I have grid in my row map
        if (rownodes->lid(grid) != -1)  // I have it, put stuff in my graph
        {
          auto index = std::span(ptr + 2, num - 1);
          graph->insert_global_indices(grid, index);
          ptr += (num + 1);
        }
        else  // I don't have it so I don't care for entries of this row, goto next row
          ptr += (num + 1);
      }
    }
    Core::Communication::barrier(dis.get_comm());
  }
  remotes.clear();

  Core::Communication::barrier(dis.get_comm());

  // finish graph
  graph->fill_complete();
  graph->optimize_storage();

  Core::Communication::barrier(dis.get_comm());

  return graph;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Graph> Core::Rebalance::build_monolithic_node_graph(
    const Core::FE::Discretization& dis, const Core::GeometricSearch::GeometricSearchParams& params,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement)
{
  if (!dis.filled())
    FOUR_C_THROW(
        "The discretization needs to be fill completed to be able to construct the monolithic "
        "graph.");

  // 1. Do a global geometric search
  std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> bounding_boxes;
  for (auto element : dis.my_row_element_range())
  {
    if (displacement == nullptr)
    {
      Core::LinAlg::Vector<double> zero_vector(*dis.dof_col_map(), true);

      bounding_boxes.emplace_back(std::make_pair(element.global_id(),
          element.user_element()->get_bounding_volume(dis, zero_vector, params)));
    }
    else
    {
      bounding_boxes.emplace_back(std::make_pair(element.global_id(),
          element.user_element()->get_bounding_volume(dis, *displacement, params)));
    }
  }

  auto result = Core::GeometricSearch::global_collision_search_print_results(
      bounding_boxes, bounding_boxes, dis.get_comm(), params.verbosity_);

  // 2. Get nodal connectivity of each element
  const int n_nodes_per_element_max = 27;  // element with highest node count is hex27
  Core::LinAlg::Graph element_connectivity(*dis.element_row_map(), n_nodes_per_element_max);
  for (int rowele_i = 0; rowele_i < dis.num_my_row_elements(); ++rowele_i)
  {
    const auto* element = dis.l_row_element(rowele_i);
    std::vector<int> element_node_ids(element->num_node());
    for (int i_node = 0; i_node < element->num_node(); ++i_node)
    {
      element_node_ids[i_node] = element->nodes()[i_node]->id();
    }
    auto indices = std::span(element_node_ids.data(), element_node_ids.size());
    element_connectivity.insert_global_indices(element->id(), indices);
  }
  element_connectivity.fill_complete(*dis.node_row_map(), *dis.element_row_map());

  // 3. Get the connectivity information of each element that collides with an element on this rank
  std::set<int> my_colliding_primitives;
  for (const auto& item : result)
  {
    my_colliding_primitives.insert(item.gid_primitive);
  }
  std::vector<int> my_colliding_primitives_vec(
      my_colliding_primitives.begin(), my_colliding_primitives.end());
  Core::LinAlg::Map my_colliding_primitives_map(-1, my_colliding_primitives_vec.size(),
      my_colliding_primitives_vec.data(), 0, dis.get_comm());
  Core::LinAlg::Import importer(my_colliding_primitives_map, *dis.element_row_map());
  Core::LinAlg::Graph my_colliding_primitives_connectivity(
      my_colliding_primitives_map, n_nodes_per_element_max);
  my_colliding_primitives_connectivity.import_from(
      element_connectivity, importer, Core::LinAlg::CombineMode::insert);

  // 4. Build and fill the graph with element internal connectivities
  auto my_graph = std::make_shared<Core::LinAlg::Graph>(
      *dis.node_row_map(), 40, Core::LinAlg::Graph::GraphType::FE_GRAPH);

  for (auto element : dis.my_row_element_range())
  {
    // Extract all global IDs of the nodes
    for (auto node_main : element.nodes())
    {
      int index_main = node_main.global_id();
      for (auto node_inner : element.nodes())
      {
        int node_inner_global_id = node_inner.global_id();
        auto index = std::span(&node_inner_global_id, 1);
        my_graph->insert_global_indices(index_main, index);
      }
    }
  }

  // 5. Fill the graph with the geometric close entries
  for (const auto& [predicate_lid, predicate_gid, primitive_gid, primitive_proc] : result)
  {
    int predicate_lid_discretization = dis.element_row_map()->lid(predicate_gid);
    if (predicate_lid_discretization < 0)
      FOUR_C_THROW("Could not find lid for predicate with gid {} on rank {}", predicate_gid,
          Core::Communication::my_mpi_rank(dis.get_comm()));
    if (predicate_lid != predicate_lid_discretization)
      FOUR_C_THROW("The ids dont match from arborx and the discretization");
    const auto* predicate = dis.g_element(predicate_gid);

    int primitive_lid_in_map = my_colliding_primitives_map.lid(primitive_gid);
    if (primitive_lid_in_map < 0) FOUR_C_THROW("Could not find lid for gid {}", primitive_gid);

    for (int i_node = 0; i_node < predicate->num_node(); ++i_node)
    {
      const auto* node_main = predicate->nodes()[i_node];
      int index_main = node_main->id();

      std::span<int> primitive_indices;
      my_colliding_primitives_connectivity.extract_global_row_view(
          primitive_gid, primitive_indices);

      my_graph->insert_global_indices(index_main, primitive_indices);
    }
  }

  my_graph->fill_complete();
  my_graph->optimize_storage();

  return my_graph;
}

FOUR_C_NAMESPACE_CLOSE
