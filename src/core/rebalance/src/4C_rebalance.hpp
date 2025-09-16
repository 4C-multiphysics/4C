// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REBALANCE_HPP
#define FOUR_C_REBALANCE_HPP

#include "4C_config.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::LinAlg
{
  class Map;
}

namespace Core::Rebalance
{

  enum class RebalanceType
  {
    none,                            //< no partitioning method
    hypergraph,                      //< hypergraph based partitioning
    recursive_coordinate_bisection,  //< recursive coordinate bisection, geometric based
                                     // partitioning
    monolithic  //< hypergraph based partitioning by using a global monolithic graph constructed
                // via a global collision search
  };


  /**
   * Additional parameters that govern the rebalancing process.
   */
  struct RebalanceParameters
  {
    /**
     * How to partition then mesh among processes.
     */
    Teuchos::ParameterList mesh_partitioning_parameters;

    /**
     * Geometric search parameters for certain partitioning methods.
     */
    Teuchos::ParameterList geometric_search_parameters;

    /**
     * General verbosity settings and I/O parameters.
     */
    Teuchos::ParameterList io_parameters;
  };


  /**
   * @brief Rebalance a discretization.
   *
   * The @p discretization is expected to have its elements distributed according to the @p
   * row_elements. This function will compute a new distribution of the elements and nodes of the
   * discretization according to the rebalancing parameters specified in @p parameters.
   * This is a collective call over all ranks in @p comm.
   */
  void rebalance_discretization(Core::FE::Discretization& discretization,
      const Core::LinAlg::Map& row_elements, const RebalanceParameters& parameters, MPI_Comm comm);
}  // namespace Core::Rebalance

FOUR_C_NAMESPACE_CLOSE

#endif
