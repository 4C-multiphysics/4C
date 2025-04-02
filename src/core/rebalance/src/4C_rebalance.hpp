// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REBALANCE_HPP
#define FOUR_C_REBALANCE_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Rebalance
{

  enum class RebalanceType
  {
    none,         //< no partitioning method
    hypergraph,   //< hypergraph based partitioning
    multijagged,  //< multijagged, geometric based partitioning
    monolithic    //< hypergraph based partitioning by using a global monolithic graph constructed
                  // via a global collision search
  };
}

FOUR_C_NAMESPACE_CLOSE

#endif
