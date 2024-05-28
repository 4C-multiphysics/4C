/*-----------------------------------------------------------*/
/*! \file

\brief Utility functions for the partitioning/rebalancing

\level 3

*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_REBALANCE_PRINT_HPP
#define FOUR_C_REBALANCE_PRINT_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace DRT
{
  class Discretization;
}

namespace CORE::REBALANCE::UTILS
{
  /*! \brief Prints details on the distribution/partitioning of the distribution
   */
  void print_parallel_distribution(const DRT::Discretization& dis);

}  // namespace CORE::REBALANCE::UTILS

FOUR_C_NAMESPACE_CLOSE

#endif