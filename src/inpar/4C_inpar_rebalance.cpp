// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_rebalance.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_rebalance.hpp"
FOUR_C_NAMESPACE_OPEN

void Inpar::Rebalance::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["MESH PARTITIONING"] = group("MESH PARTITIONING",
      {

          parameter<Core::Rebalance::RebalanceType>("METHOD",
              {.description = "Type of rebalance/partition algorithm to be used for decomposing "
                              "the entire mesh into subdomains for parallel computing.",
                  .default_value = Core::Rebalance::RebalanceType::hypergraph}),

          parameter<double>("IMBALANCE_TOL",
              {.description =
                      "Tolerance for relative imbalance of subdomain sizes for graph partitioning "
                      "of unstructured meshes read from input files.",
                  .default_value = 1.1}),

          parameter<int>("MIN_ELE_PER_PROC",
              {.description =
                      "This parameter defines the minimum number of elements to be assigned to any "
                      "MPI rank during redistribution. Use 0 to not interfere with the minimal "
                      "size "
                      "of a subdomain.",
                  .default_value = 0})},
      {.defaultable = true});
}

FOUR_C_NAMESPACE_CLOSE