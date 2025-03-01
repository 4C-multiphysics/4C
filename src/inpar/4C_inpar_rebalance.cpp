// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_rebalance.hpp"

#include "4C_rebalance.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::Rebalance::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs meshpartitioning{"MESH PARTITIONING"};

  Core::Utils::string_to_integral_parameter<Core::Rebalance::RebalanceType>("METHOD", "hypergraph",
      "Type of rebalance/partition algorithm to be used for decomposing the entire mesh into "
      "subdomains for parallel computing.",
      tuple<std::string>("none", "hypergraph", "recursive_coordinate_bisection", "monolithic"),
      tuple<Core::Rebalance::RebalanceType>(Core::Rebalance::RebalanceType::none,
          Core::Rebalance::RebalanceType::hypergraph,
          Core::Rebalance::RebalanceType::recursive_coordinate_bisection,
          Core::Rebalance::RebalanceType::monolithic),
      meshpartitioning);

  meshpartitioning.specs.emplace_back(parameter<double>("IMBALANCE_TOL",
      {.description = "Tolerance for relative imbalance of subdomain sizes for graph partitioning "
                      "of unstructured meshes read from input files.",
          .default_value = 1.1}));

  meshpartitioning.specs.emplace_back(parameter<int>("MIN_ELE_PER_PROC",
      {.description =
              "This parameter defines the minimum number of elements to be assigned to any "
              "MPI rank during redistribution. Use 0 to not interfere with the minimal size "
              "of a subdomain.",
          .default_value = 0}));

  meshpartitioning.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
