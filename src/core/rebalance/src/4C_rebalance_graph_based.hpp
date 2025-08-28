// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REBALANCE_GRAPH_BASED_HPP
#define FOUR_C_REBALANCE_GRAPH_BASED_HPP

#include "4C_config.hpp"

#include "4C_geometric_search_params.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
}

namespace Core::Rebalance
{
  /*!
  \brief Compute rebalanced node maps for a given discretization considering weights to model
  costs

  Starting from a given discretization, a new node row and column map is computed, such that
  they both are better balanced across a given number partitions.

  \note This just computes the new node row/column maps, but does not perform any redistribution
  of data among ranks.

  @param[in] initialGraph Initial graph used for rebalancing
  @param[in] rebalanceParams Parameter list with rebalancing options
  @param[in] initialNodeWeights Initial weights of the graph nodes
  @param[in] initialEdgeWeights Initial weights of the graph edges
  @param[in] initialNodeCoordinates Coordinates of the discretization

  @return Node row map and node column map after rebalancing with weights
  */
  std::pair<std::shared_ptr<Core::LinAlg::Map>, std::shared_ptr<Core::LinAlg::Map>>
  rebalance_node_maps(const Core::LinAlg::Graph& initialGraph,
      Teuchos::ParameterList& rebalanceParams,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& initialNodeWeights = nullptr,
      const std::shared_ptr<Core::LinAlg::SparseMatrix>& initialEdgeWeights = nullptr,
      const std::shared_ptr<Core::LinAlg::MultiVector<double>>& initialNodeCoordinates = nullptr);

  /*!
  \brief Rebalance graph using node and edge weights based on the initial graph

  The partitioning will be done based on the method given in the parameter list. This method
  only makes sense with graph or hypergraph partitioning.

  @note Use Isorropia package to access Zoltan. By default, Isorropia will use Zoltan hypergraph
  partitioning, treating the graph columns as hyper-edges and the graph rows as vertices. The
  rebalanced graph will be fill_complete().

  \pre The initialGraph has to be filled()==true.

  @param[in] initialGraph Initial graph to be rebalanced
  @param[in] rebalanceParams Parameter list with rebalancing options
  @param[in] initialNodeWeights Initial weights of the graph nodes
  @param[in] initialEdgeWeights Initial weights of the graph edges
  @param[in] initialNodeCoordinates Coordinates of the discretization

  @return std::shared_ptr<Core::LinAlg::Graph>
  */
  std::shared_ptr<Core::LinAlg::Graph> rebalance_graph(const Core::LinAlg::Graph& initialGraph,
      Teuchos::ParameterList& rebalanceParams,
      const std::shared_ptr<Core::LinAlg::Vector<double>>& initialNodeWeights = nullptr,
      const std::shared_ptr<Core::LinAlg::SparseMatrix>& initialEdgeWeights = nullptr,
      const std::shared_ptr<Core::LinAlg::MultiVector<double>>& initialNodeCoordinates = nullptr);

  /*!
  \brief Rebalance coordinates using weights based on the initial coordinates

  This method only makes sense with geometric partitioning methods like RCB. The partitioner
  will figure things out in the background.

  @param[in] initialCoordinates Initial coordinates to be rebalanced
  @param[in] initialWeights Initial weights of the coordinates
  @param[in] rebalanceParams Parameter list with rebalancing options

  @return Rebalanced coordinates
  */
  std::pair<std::shared_ptr<Core::LinAlg::MultiVector<double>>,
      std::shared_ptr<Core::LinAlg::MultiVector<double>>>
  rebalance_coordinates(const Core::LinAlg::MultiVector<double>& initialCoordinates,
      Teuchos::ParameterList& rebalanceParams,
      const Core::LinAlg::MultiVector<double>& initialWeights);

  /*!
  \brief Create node and edge weights based on element connectivity

  @param[in] dis discretization used to build the weights

  @return Node and edge weights to be used for repartitioning
  */
  std::pair<std::shared_ptr<Core::LinAlg::Vector<double>>,
      std::shared_ptr<Core::LinAlg::SparseMatrix>>
  build_weights(const Core::FE::Discretization& dis);

  /*!
  \brief Build node graph of a given  discretization

  \pre discretization does NOT have to be fill_complete().

  @param[in] dis discretization whose node graph will be build
  @param[in] element_row_map Element row map of this discretization

  @return Uncompleted node graph of input discretization
  */
  std::shared_ptr<const Core::LinAlg::Graph> build_graph(
      Core::FE::Discretization& dis, const Core::LinAlg::Map& element_row_map);

  /*!
  \brief Build monolithic node graph of a given discretization

  The monolithic graph is build by using a global collision search on the reference configuration,
  to obtain information about close elements. Based on this information, additional edges are
  build into the graph.

  \pre The discretization has to be filled()==true.

  @param[in] dis discretization whose monolithic node graph will be build
  @param[in] displacement vector containing displacement values that shifts coordinate values for
                          the bounding volume construction

  @return Completed monolithic node graph of input discretization
  */
  std::shared_ptr<const Core::LinAlg::Graph> build_monolithic_node_graph(
      const Core::FE::Discretization& dis,
      const Core::GeometricSearch::GeometricSearchParams& params,
      const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement = nullptr);

}  // namespace Core::Rebalance

FOUR_C_NAMESPACE_CLOSE

#endif
