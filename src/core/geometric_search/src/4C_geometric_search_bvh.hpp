// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRIC_SEARCH_BVH_HPP
#define FOUR_C_GEOMETRIC_SEARCH_BVH_HPP

#include "4C_config.hpp"

#include "4C_io_pstream.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  struct BoundingVolume;

  /*! \brief Structure to hold a pair found during a collision search
   */
  struct CollisionSearchResult
  {
    //! Global ID of the predicate
    int gid_predicate;
    //! Global ID of the primitive
    int gid_primitive;
  };

  /*! \brief Finds all primitives meeting the predicates and record results in {indices, offsets}.
   *
   * (From ArborX documentation)
   * Finds all primitives meeting the predicates and record results in {indices, offsets}. indices
   * stores the indices of the objects that satisfy the predicates. offsets stores the locations in
   * indices that start a predicate, that is, predicates(i) is satisfied by primitives(indices(j))
   * for offsets(i) <= j < offsets(i+1). Following the usual convention, offsets(n) ==
   * indices.size(), where n is the number of queries that were performed and indices.size() is the
   * total number of collisions. Be aware, that this function will return all in primitives vs. all
   * in predicates!
   *
   * @param primitives Bounding volumes to search for intersections
   * @param predicates Bounding volumes to intersect with
   * @param comm Communicator object of the discretization
   * @param verbosity Enabling printout of the geometric search information
   * @return pairs Vector of the found interaction pairs
   *
   * D. Lebrun-Grandie, A. Prokopenko, B. Turcksin, and S. R. Slattery. 2020.
   * ArborX: A Performance Portable Geometric Search Library. ACM Trans. Math. Softw. 47, 1,
   * Article 2 (2021), https://doi.org/10.1145/3412558
   *
   */
  std::vector<CollisionSearchResult> collision_search(
      const std::vector<std::pair<int, BoundingVolume>>& primitives,
      const std::vector<std::pair<int, BoundingVolume>>& predicates, MPI_Comm comm,
      const Core::IO::Verbositylevel verbosity);

}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE

#endif
