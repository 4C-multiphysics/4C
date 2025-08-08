// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRIC_SEARCH_DISTRIBUTED_TREE_HPP
#define FOUR_C_GEOMETRIC_SEARCH_DISTRIBUTED_TREE_HPP

#include "4C_config.hpp"

#include "4C_geometric_search_bounding_volume.hpp"
#include "4C_io_pstream.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  struct BoundingVolume;

  /*! \brief Structure to hold a resulting global/local ID pair found during a global collision
   * search
   */
  struct GlobalCollisionSearchResult
  {
    //! Local ID of the predicate (on this rank)
    int lid_predicate;
    //! Global ID of the predicate
    int gid_predicate;
    //! Global ID of the primitives
    int gid_primitive;
    //! Processor ID owning the primitive
    int pid_primitive;
  };

  /*! \brief Finds all primitives on different ranks meeting the locally owned predicates and
   * return results.
   *
   * Hereby the local/global index of the given predicate and the local/global index as well as
   * the MPI processor rank of the found primitive is returned.
   *
   * @param primitives Bounding volumes to search for intersections
   * @param predicates Bounding volumes to intersect with
   * @param comm Communicator object of the discretization
   * @param verbosity Enabling printout of the geometric search information
   * @return Collision pairs found with their global and local IDs
   *
   * D. Lebrun-Grandie, A. Prokopenko, B. Turcksin, and S. R. Slattery. 2020.
   * ArborX: A Performance Portable Geometric Search Library. ACM Trans. Math. Softw. 47, 1,
   * Article 2 (2021), https://doi.org/10.1145/3412558
   *
   */
  std::vector<GlobalCollisionSearchResult> global_collision_search(
      const std::vector<std::pair<int, BoundingVolume>>& primitives,
      const std::vector<std::pair<int, BoundingVolume>>& predicates, MPI_Comm comm,
      const Core::IO::Verbositylevel verbosity);

}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE

#endif
