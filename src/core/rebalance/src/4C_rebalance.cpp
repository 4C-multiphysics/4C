// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_rebalance.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_rebalance_print.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

static std::pair<std::shared_ptr<Core::LinAlg::Map>, std::shared_ptr<Core::LinAlg::Map>>
do_rebalance_discretization(const Core::LinAlg::Graph& graph,
    Core::FE::Discretization& discretization, Core::Rebalance::RebalanceType rebalanceMethod,
    Teuchos::ParameterList& rebalanceParams, const Core::Rebalance::RebalanceParameters& parameters,
    MPI_Comm comm, const Core::Rebalance::PartitionWeights* partition_weights)
{
  std::shared_ptr<Core::LinAlg::Map> rowmap, colmap;
  const std::shared_ptr<Core::LinAlg::Vector<double>> node_weights =
      partition_weights != nullptr ? partition_weights->node_weights : nullptr;
  const std::shared_ptr<Core::LinAlg::SparseMatrix> edge_weights =
      partition_weights != nullptr ? partition_weights->edge_weights : nullptr;

  switch (rebalanceMethod)
  {
    case Core::Rebalance::RebalanceType::hypergraph:
    {
      if (!Core::Communication::my_mpi_rank(comm))
        std::cout << "Redistributing using hypergraph .........\n";

      rebalanceParams.set("algorithm", "phg");
      rebalanceParams.set("debug_level", "no_status");
      Teuchos::ParameterList& zparams = rebalanceParams.sublist("zoltan_parameters", false);
      zparams.set("DEBUG_LEVEL", "0");

      std::tie(rowmap, colmap) =
          Core::Rebalance::rebalance_node_maps(graph, rebalanceParams, node_weights, edge_weights);
      break;
    }
    case Core::Rebalance::RebalanceType::multijagged:
    {
      if (!Core::Communication::my_mpi_rank(comm))
        std::cout << "Redistributing using recursive coordinate bisection .........\n";

      rebalanceParams.set("algorithm", "multijagged");
      rebalanceParams.set("debug_level", "no_status");
      Teuchos::ParameterList& zparams = rebalanceParams.sublist("zoltan_parameters", false);
      zparams.set("DEBUG_LEVEL", "0");

      rowmap = std::make_shared<Core::LinAlg::Map>(
          -1, graph.row_map().num_my_elements(), graph.row_map().my_global_elements(), 0, comm);
      colmap = std::make_shared<Core::LinAlg::Map>(
          -1, graph.col_map().num_my_elements(), graph.col_map().my_global_elements(), 0, comm);

      discretization.redistribute(
          {
              *rowmap,
              *colmap,
          },
          {.fill_complete = Core::FE::OptionsFillComplete::none()});

      std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates =
          extract_node_coordinates(discretization);

      std::tie(rowmap, colmap) = Core::Rebalance::rebalance_node_maps(
          graph, rebalanceParams, node_weights, edge_weights, coordinates);
      break;
    }
    case Core::Rebalance::RebalanceType::monolithic:
    {
      if (!Core::Communication::my_mpi_rank(comm))
        std::cout << "Redistributing using monolithic hypergraph .........\n";

      rebalanceParams.set("algorithm", "phg");
      rebalanceParams.set("debug_level", "no_status");
      Teuchos::ParameterList& zparams = rebalanceParams.sublist("zoltan_parameters", false);
      zparams.set("DEBUG_LEVEL", "0");

      rowmap = std::make_shared<Core::LinAlg::Map>(
          -1, graph.row_map().num_my_elements(), graph.row_map().my_global_elements(), 0, comm);
      colmap = std::make_shared<Core::LinAlg::Map>(
          -1, graph.col_map().num_my_elements(), graph.col_map().my_global_elements(), 0, comm);

      discretization.redistribute(
          {
              *rowmap,
              *colmap,
          },
          {.fill_complete = Core::FE::OptionsFillComplete{.do_boundary_conditions = false}});

      std::shared_ptr<const Core::LinAlg::Graph> enriched_graph =
          Core::Rebalance::build_monolithic_node_graph(discretization,
              Core::GeometricSearch::GeometricSearchParams(
                  parameters.geometric_search_parameters, parameters.io_parameters));

      std::tie(rowmap, colmap) = Core::Rebalance::rebalance_node_maps(
          *enriched_graph, rebalanceParams, node_weights, edge_weights);
      break;
    }
    default:
      FOUR_C_THROW("Appropriate partitioning has to be set!");
  }

  return {rowmap, colmap};
}

void Core::Rebalance::rebalance_discretization(Core::FE::Discretization& discretization,
    const Core::LinAlg::Map& row_elements, const RebalanceParameters& parameters, MPI_Comm comm,
    bool use_eval_time_weights)
{
  std::shared_ptr<const Core::LinAlg::Graph> graph = nullptr;

  // Skip building the node graph if there are no elements
  if (row_elements.num_global_elements() > 0)
    graph = Core::Rebalance::build_graph(discretization, row_elements);

  // Create partitioning parameters
  const double imbalance_tol = parameters.mesh_partitioning_parameters.imbalance_tol;

  Teuchos::ParameterList rebalanceParams;
  rebalanceParams.set("imbalance_tolerance", imbalance_tol);

  const int minele_per_proc = parameters.mesh_partitioning_parameters.min_ele_per_proc;
  const int max_global_procs = Core::Communication::num_mpi_ranks(comm);
  int min_global_procs = max_global_procs;

  if (minele_per_proc > 0) min_global_procs = row_elements.num_global_elements() / minele_per_proc;
  const int num_procs = std::min(max_global_procs, min_global_procs);
  rebalanceParams.set("num_global_parts", num_procs);

  const auto rebalanceMethod = parameters.mesh_partitioning_parameters.rebalance_type;

  if (!Core::Communication::my_mpi_rank(comm))
    std::cout << "\nNumber of procs used for redistribution: " << num_procs << "\n";

  std::shared_ptr<Core::LinAlg::Map> rowmap, colmap;

  if (graph)
  {
    const std::optional<Core::Rebalance::PartitionWeights> effective_partition_weights =
        use_eval_time_weights
            ? std::make_optional(Core::Rebalance::build_eval_time_partition_weights(
                  discretization, *graph, parameters.edge_weight_multiplier))
            : std::nullopt;
    std::tie(rowmap, colmap) = do_rebalance_discretization(*graph, discretization, rebalanceMethod,
        rebalanceParams, parameters, comm,
        effective_partition_weights ? &effective_partition_weights.value() : nullptr);
  }
  else
  {
    rowmap = colmap = std::make_shared<Core::LinAlg::Map>(-1, 0, nullptr, 0, comm);
  }

  auto options_redistribution = Core::FE::OptionsRedistribution();
  if (rebalanceMethod == Core::Rebalance::RebalanceType::monolithic)
    options_redistribution.do_extended_ghosting = true;

  options_redistribution.fill_complete = FE::OptionsFillComplete::none();

  discretization.redistribute({*rowmap, *colmap}, options_redistribution);

  print_parallel_distribution(discretization);
}


FOUR_C_NAMESPACE_CLOSE
