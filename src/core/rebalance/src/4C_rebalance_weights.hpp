// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REBALANCE_WEIGHTS_HPP
#define FOUR_C_REBALANCE_WEIGHTS_HPP

#include "4C_config.hpp"

#include "4C_rebalance.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::LinAlg
{
  class Graph;
}

namespace Core::Rebalance
{
  /**
   * Build the default static repartitioning weights from the element connectivity.
   */
  PartitionWeights build_static_partition_weights(const Core::FE::Discretization& dis);

  /**
   * Build repartitioning weights from measured element evaluation time.
   *
   * The current model uses vertex weights only and leaves edge weights unset.
   */
  PartitionWeights build_eval_time_partition_weights(const Core::FE::Discretization& dis);

  /**
   * Build a sanity-check weight set with unit weight on every graph vertex and every graph edge.
   */
  PartitionWeights build_uniform_partition_weights(const Core::LinAlg::Graph& graph);
}  // namespace Core::Rebalance

FOUR_C_NAMESPACE_CLOSE

#endif
