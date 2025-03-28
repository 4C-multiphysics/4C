// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRIC_SEARCH_CREATE_BOUNDING_VOLUMES_TEST_HPP
#define FOUR_C_FEM_GEOMETRIC_SEARCH_CREATE_BOUNDING_VOLUMES_TEST_HPP

#include "4C_config.hpp"

#ifdef FOUR_C_WITH_ARBORX

#include "4C_fem_geometric_search_bounding_volume.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  /**
   * Create a set of three different bounding volumes that is used for some unit tests.
   * The geometry is based on https://github.com/arborx/ArborX/issues/867
   */
  std::vector<Core::GeometricSearch::BoundingVolume> create_kdop_bounding_volumes()
  {
    std::vector<Core::GeometricSearch::BoundingVolume> volumes(3);

    Core::LinAlg::Matrix<3, 1, double> point(true);

    // setting up bounding volume 1
    {
      point(0) = 0.0;
      point(1) = 0.0;
      point(2) = 0.0;
      volumes[0].add_point(point);

      point(0) = 1.0;
      point(1) = 1.0;
      point(2) = 0.0;
      volumes[0].add_point(point);

      point(0) = 0.5;
      point(1) = 0.0;
      point(2) = 0.0;
      volumes[0].add_point(point);
    }

    // setting up bounding volume 2
    {
      point(0) = 1.0;
      point(1) = -1.0;
      point(2) = 0.0;
      volumes[1].add_point(point);

      point(0) = 1.0;
      point(1) = 0.25;
      point(2) = 0.0;
      volumes[1].add_point(point);

      point(0) = 0.75;
      point(1) = 0.0;
      point(2) = 0.0;
      volumes[1].add_point(point);
    }

    // setting up bounding volume 3
    {
      point(0) = 0.5;
      point(1) = 0.25;
      point(2) = 0.0;
      volumes[2].add_point(point);

      point(0) = 0.8;
      point(1) = 0.25;
      point(2) = 0.0;
      volumes[2].add_point(point);

      point(0) = 0.75;
      point(1) = 0.125;
      point(2) = 0.0;
      volumes[2].add_point(point);
    }

    return volumes;
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

#endif

#endif
