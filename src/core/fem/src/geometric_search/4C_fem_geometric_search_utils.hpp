// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRIC_SEARCH_UTILS_HPP
#define FOUR_C_FEM_GEOMETRIC_SEARCH_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_geometric_search_bounding_volume.hpp"

#include <mpi.h>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  /*! \brief Storing information on the geometric search
   */
  struct GeometricSearchInfo
  {
    int primitive_size;
    int predicate_size;
    int coupling_pair_size;
  };

  /*! \brief Prints details on the geometric search algorithm
   */
  void print_geometric_search_details(MPI_Comm comm, const GeometricSearchInfo info);

  /*! \brief Get the polyhedron representation of a k-DOP
   *
   * @param boundingVolume Bounding volume enclosing the respective element (as k-DOP)
   * @return Points of the polyhedron and connecting polygons
   */
  std::pair<std::vector<LinAlg::Matrix<3, 1>>, std::vector<std::vector<int>>>
  get_k_dop_polyhedron_representation(const BoundingVolume boundingVolume);

}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE

#endif
