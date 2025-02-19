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
#include "4C_fem_geometric_search_bounding_volume.hpp"
#include "4C_fem_geometric_search_distributed_tree.hpp"
#include "4C_fem_geometric_search_params.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"

#include <Epetra_FECrsGraph.h>
#include <Epetra_Import.h>
#include <Isorropia_Epetra.hpp>
#include <Isorropia_EpetraCostDescriber.hpp>
#include <Isorropia_EpetraPartitioner.hpp>
#include <Isorropia_EpetraRedistributor.hpp>
#include <Isorropia_Exception.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::pair<std::shared_ptr<Epetra_Map>, std::shared_ptr<Epetra_Map>>
Core::Rebalance::rebalance_node_maps(const Core::LinAlg::Graph& initialGraph,
    const Teuchos::ParameterList& rebalanceParams,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& initialNodeWeights,
    const std::shared_ptr<Epetra_CrsMatrix>& initialEdgeWeights,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& initialNodeCoordinates)
{
  TEUCHOS_FUNC_TIME_MONITOR("Rebalance::rebalance_node_maps");

  // Compute rebalanced graph
  auto balanced_graph = Rebalance::rebalance_graph(initialGraph, rebalanceParams,
      initialNodeWeights, initialEdgeWeights, initialNodeCoordinates);

  // extract repartitioned maps
  std::shared_ptr<Epetra_Map> rownodes =
      std::make_shared<Epetra_Map>(-1, balanced_graph->RowMap().NumMyElements(),
          balanced_graph->RowMap().MyGlobalElements(), 0, initialGraph.Comm());
  std::shared_ptr<Epetra_Map> colnodes =
      std::make_shared<Epetra_Map>(-1, balanced_graph->ColMap().NumMyElements(),
          balanced_graph->ColMap().MyGlobalElements(), 0, initialGraph.Comm());

  return {rownodes, colnodes};
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Graph> Core::Rebalance::rebalance_graph(
    const Core::LinAlg::Graph& initialGraph, const Teuchos::ParameterList& rebalanceParams,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& initialNodeWeights,
    const std::shared_ptr<Epetra_CrsMatrix>& initialEdgeWeights,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& initialNodeCoordinates)
{
  TEUCHOS_FUNC_TIME_MONITOR("Rebalance::RebalanceGraph");

  Isorropia::Epetra::CostDescriber costs = Isorropia::Epetra::CostDescriber();
  if (initialNodeWeights != nullptr)
    costs.setVertexWeights(Teuchos::rcpFromRef(*initialNodeWeights->get_ptr_of_Epetra_Vector()));
  if (initialEdgeWeights != nullptr)
    costs.setGraphEdgeWeights(Teuchos::rcpFromRef(*initialEdgeWeights));

  Teuchos::RCP<Isorropia::Epetra::Partitioner> partitioner;
  if (initialNodeCoordinates)
  {
    partitioner = Teuchos::make_rcp<Isorropia::Epetra::Partitioner>(
        &initialGraph.get_ref_of_Epetra_CrsGraph(), &costs,
        initialNodeCoordinates->get_ptr_of_Epetra_MultiVector().get(), nullptr, rebalanceParams);
  }
  else
  {
    partitioner = Teuchos::make_rcp<Isorropia::Epetra::Partitioner>(
        &initialGraph.get_ref_of_Epetra_CrsGraph(), &costs, rebalanceParams);
  }

  Isorropia::Epetra::Redistributor rd(partitioner);
  auto balancedGraph = std::make_shared<Core::LinAlg::Graph>(
      *rd.redistribute(initialGraph.get_ref_of_Epetra_CrsGraph(), true));

  balancedGraph->FillComplete();
  balancedGraph->OptimizeStorage();

  return balancedGraph;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::pair<std::shared_ptr<Core::LinAlg::MultiVector<double>>,
    std::shared_ptr<Core::LinAlg::MultiVector<double>>>
Core::Rebalance::rebalance_coordinates(const Core::LinAlg::MultiVector<double>& initialCoordinates,
    const Teuchos::ParameterList& rebalanceParams,
    const Core::LinAlg::MultiVector<double>& initialWeights)
{
  TEUCHOS_FUNC_TIME_MONITOR("Rebalance::RebalanceCoordinates");

  Teuchos::RCP<Isorropia::Epetra::Partitioner> part =
      Teuchos::make_rcp<Isorropia::Epetra::Partitioner>(
          Teuchos::rcpFromRef(*initialCoordinates.get_ptr_of_Epetra_MultiVector()),
          Teuchos::rcpFromRef(*initialWeights.get_ptr_of_Epetra_MultiVector()), rebalanceParams);

  Isorropia::Epetra::Redistributor rd(part);

  return {std::make_shared<Core::LinAlg::MultiVector<double>>(*rd.redistribute(initialCoordinates)),
      std::make_shared<Core::LinAlg::MultiVector<double>>(*rd.redistribute(initialWeights))};
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::pair<std::shared_ptr<Core::LinAlg::Vector<double>>, std::shared_ptr<Epetra_CrsMatrix>>
Core::Rebalance::build_weights(const Core::FE::Discretization& dis)
{
  const Epetra_Map* noderowmap = dis.node_row_map();

  std::shared_ptr<Epetra_CrsMatrix> crs_ge_weights =
      std::make_shared<Epetra_CrsMatrix>(Copy, *noderowmap, 15);
  std::shared_ptr<Core::LinAlg::Vector<double>> vweights =
      Core::LinAlg::create_vector(*noderowmap, true);

  // loop all row elements and get their cost of evaluation
  for (int i = 0; i < dis.element_row_map()->NumMyElements(); ++i)
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
    Core::LinAlg::SerialDenseMatrix edgeweigths_ele;
    Core::LinAlg::SerialDenseVector nodeweights_ele;

    // evaluate elements to get their evaluation cost
    ele->nodal_connectivity(edgeweigths_ele, nodeweights_ele);

    Core::LinAlg::assemble(*crs_ge_weights, edgeweigths_ele, lm, lmrowowner, lm);
    Core::LinAlg::assemble(*vweights, nodeweights_ele, lm, lmrowowner);
  }

  return {vweights, crs_ge_weights};
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Graph> Core::Rebalance::build_graph(
    Core::FE::Discretization& dis, const Epetra_Map& roweles)
{
  const int myrank = Core::Communication::my_mpi_rank(dis.get_comm());
  const int numproc = Core::Communication::num_mpi_ranks(dis.get_comm());

  // create a set of all nodes that I have
  std::set<int> mynodes;
  for (int lid = 0; lid < roweles.NumMyElements(); ++lid)
  {
    Core::Elements::Element* ele = dis.g_element(roweles.GID(lid));
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
    Core::Communication::broadcast(&recvnodes[0], size, proc, dis.get_comm());
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

  std::shared_ptr<Epetra_Map> rownodes = nullptr;
  // copy the set to a vector
  {
    std::vector<int> nodes;
    std::set<int>::iterator fool;
    for (fool = mynodes.begin(); fool != mynodes.end(); ++fool) nodes.push_back(*fool);
    mynodes.clear();
    // create a non-overlapping row map
    rownodes = std::make_shared<Epetra_Map>(
        -1, (int)nodes.size(), &nodes[0], 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  }

  // start building the graph object
  std::map<int, std::set<int>> locals;
  std::map<int, std::set<int>> remotes;
  for (int lid = 0; lid < roweles.NumMyElements(); ++lid)
  {
    Core::Elements::Element* ele = dis.g_element(roweles.GID(lid));
    const int numnode = ele->num_node();
    const int* nodeids = ele->node_ids();
    for (int i = 0; i < numnode; ++i)
    {
      const int lid = rownodes->LID(nodeids[i]);  // am I owner of this gid?
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
    Core::Communication::max_all(&smaxband, &maxband, 1, dis.get_comm());
  }

  std::shared_ptr<Core::LinAlg::Graph> graph =
      std::make_shared<Core::LinAlg::Graph>(Copy, *rownodes, maxband, false);
  Core::Communication::barrier(dis.get_comm());

  // fill all local entries into the graph
  {
    std::map<int, std::set<int>>::iterator fool = locals.begin();
    for (; fool != locals.end(); ++fool)
    {
      const int grid = fool->first;
      std::vector<int> cols(0, 0);
      std::set<int>::iterator setfool = fool->second.begin();
      for (; setfool != fool->second.end(); ++setfool) cols.push_back(*setfool);
      int err = graph->InsertGlobalIndices(grid, (int)cols.size(), &cols[0]);
      if (err < 0)
        FOUR_C_THROW(
            "Core::LinAlg::Graph::InsertGlobalIndices returned %d for global row %d", err, grid);
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
    Core::Communication::broadcast(&recvnodes[0], size, proc, dis.get_comm());
    if (proc != myrank && size)
    {
      int* ptr = &recvnodes[0];
      while (ptr < &recvnodes[size - 1])
      {
        int num = *ptr;
        int grid = *(ptr + 1);
        // see whether I have grid in my row map
        if (rownodes->LID(grid) != -1)  // I have it, put stuff in my graph
        {
          int err = graph->InsertGlobalIndices(grid, num - 1, (ptr + 2));
          if (err < 0) FOUR_C_THROW("Core::LinAlg::Graph::InsertGlobalIndices returned %d", err);
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
  graph->FillComplete();
  graph->OptimizeStorage();

  Core::Communication::barrier(dis.get_comm());

  return graph;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Graph> Core::Rebalance::build_monolithic_node_graph(
    const Core::FE::Discretization& dis, const Core::GeometricSearch::GeometricSearchParams& params)
{
  // 1. Do a global geometric search
  Core::LinAlg::Vector<double> zero_vector =
      Core::LinAlg::Vector<double>(*(dis.dof_col_map()), true);

  std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> bounding_boxes;
  for (const auto* element : dis.my_row_element_range())
  {
    bounding_boxes.emplace_back(
        std::make_pair(element->id(), element->get_bounding_volume(dis, zero_vector, params)));
  }

  auto result = Core::GeometricSearch::global_collision_search(
      bounding_boxes, bounding_boxes, dis.get_comm(), params.verbosity_);

  // 2. Get nodal connectivity of each element
  const int n_nodes_per_element_max = 27;  // element with highest node count is hex27
  int err;
  Core::LinAlg::Graph element_connectivity(
      Copy, *dis.element_row_map(), n_nodes_per_element_max, false);
  for (int rowele_i = 0; rowele_i < dis.num_my_row_elements(); ++rowele_i)
  {
    const auto* element = dis.l_row_element(rowele_i);
    std::vector<int> element_node_ids(element->num_node());
    for (int i_node = 0; i_node < element->num_node(); ++i_node)
    {
      element_node_ids[i_node] = element->nodes()[i_node]->id();
    }
    err = element_connectivity.InsertGlobalIndices(
        element->id(), element_node_ids.size(), element_node_ids.data());
    if (err != 0) FOUR_C_THROW("Core::LinAlg::Graph::InsertGlobalIndices returned %d", err);
  }
  element_connectivity.FillComplete();

  // 3. Get the connectivity information of each element that collides with an element on this rank
  std::set<int> my_colliding_primitives;
  for (const auto& item : result)
  {
    my_colliding_primitives.insert(item.gid_primitive);
  }
  std::vector<int> my_colliding_primitives_vec(
      my_colliding_primitives.begin(), my_colliding_primitives.end());
  Epetra_Map my_colliding_primitives_map(-1, my_colliding_primitives_vec.size(),
      my_colliding_primitives_vec.data(), 0, Core::Communication::as_epetra_comm(dis.get_comm()));
  Epetra_Import importer(my_colliding_primitives_map, *dis.element_row_map());
  Core::LinAlg::Graph my_colliding_primitives_connectivity(
      Copy, my_colliding_primitives_map, n_nodes_per_element_max, false);
  err = my_colliding_primitives_connectivity.Import(
      *element_connectivity.get_ptr_of_Epetra_CrsGraph(), importer, Insert);
  if (err != 0) FOUR_C_THROW("Core::LinAlg::Graph::Import returned %d", err);

  // 4. Build and fill the graph with element internal connectivities
  auto my_graph = std::make_shared<Epetra_FECrsGraph>(Copy, *(dis.node_row_map()), 40, false);

  for (const auto* element : dis.my_row_element_range())
  {
    for (int i_node = 0; i_node < element->num_node(); ++i_node)
    {
      const auto* node_main = element->nodes()[i_node];
      int index_main = node_main->id();
      for (int j_node = 0; j_node < element->num_node(); ++j_node)
      {
        const auto* node_inner = element->nodes()[j_node];
        int index = node_inner->id();

        int err = my_graph->InsertGlobalIndices(1, &index_main, 1, &index);
        if (err != 0)
          FOUR_C_THROW("Core::LinAlg::Graph::InsertGlobalIndices returned %d for global row %d",
              err, node_main->id());
      }
    }
  }

  // 5. Fill the graph with the geometric close entries
  for (const auto& [predicate_lid, predicate_gid, primitive_lid, primitive_gid, primitive_proc] :
      result)
  {
    int predicate_lid_discretization = dis.element_row_map()->LID(predicate_gid);
    if (predicate_lid_discretization < 0)
      FOUR_C_THROW("Could not find lid for predicate with gid %d on rank %d", predicate_gid,
          Core::Communication::my_mpi_rank(dis.get_comm()));
    if (predicate_lid != predicate_lid_discretization)
      FOUR_C_THROW("The ids dont match from arborx and the discretization");
    const auto* predicate = dis.g_element(predicate_gid);

    int primitive_lid_in_map = my_colliding_primitives_map.LID(primitive_gid);
    if (primitive_lid_in_map < 0) FOUR_C_THROW("Could not find lid for gid %d", primitive_gid);

    for (int i_node = 0; i_node < predicate->num_node(); ++i_node)
    {
      const auto* node_main = predicate->nodes()[i_node];
      int index_main = node_main->id();

      int primitive_num_nodes;
      int* primitive_node_indices;
      err = my_colliding_primitives_connectivity.ExtractGlobalRowView(
          primitive_gid, primitive_num_nodes, primitive_node_indices);
      if (err != 0) FOUR_C_THROW("Core::LinAlg::Graph::ExtractGlobalRowView returned %d", err);

      int err = my_graph->InsertGlobalIndices(
          1, &index_main, primitive_num_nodes, primitive_node_indices);
      if (err != 0)
        FOUR_C_THROW("Core::LinAlg::Graph::InsertGlobalIndices returned %d for global row %d", err,
            node_main->id());
    }
  }

  my_graph->GlobalAssemble(true);
  my_graph->OptimizeStorage();
  // cast graph
  return std::make_shared<const Core::LinAlg::Graph>(static_cast<Epetra_CrsGraph>(*my_graph));
}
FOUR_C_NAMESPACE_CLOSE
