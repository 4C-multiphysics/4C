// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_fem_geometric_search_bounding_volume.hpp"
#include "4C_fem_geometric_search_utils.hpp"


#ifdef FOUR_C_WITH_ARBORX

#include "4C_fem_geometric_search_bvh.hpp"
#include "4C_fem_geometric_search_create_bounding_volumes_test.hpp"
#include "4C_fem_geometric_search_utils.hpp"


namespace
{
  using namespace FourC;

  class GeometricSearch : public ::testing::Test
  {
   public:
    static void SetUpTestSuite() { Kokkos::initialize(); }

    static void TearDownTestSuite() { Kokkos::finalize(); }

    GeometricSearch() { verbosity_ = Core::IO::minimal; }

   protected:
    std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> primitives_, predicates_;
    MPI_Comm comm_{MPI_COMM_WORLD};
    Core::IO::Verbositylevel verbosity_;
  };

  /**
   * Test setup is based on https://github.com/arborx/ArborX/issues/867
   * Checking collision search of kdop primitives against kdop predicates
   */
  TEST_F(GeometricSearch, CollisionSearchKDOPVsKDOP)
  {
    const auto volumes = create_kdop_bounding_volumes();

    // Add the volumes to the primitives and predicates
    primitives_.emplace_back(std::pair{0, volumes[0]});
    primitives_.emplace_back(std::pair{1, volumes[1]});
    predicates_.emplace_back(std::pair{2, volumes[2]});

    EXPECT_EQ(primitives_.size(), 2);
    EXPECT_EQ(predicates_.size(), 1);

    const auto& [indices, offsets] =
        Core::GeometricSearch::collision_search(primitives_, predicates_, comm_, verbosity_);

    const auto pairs = Core::GeometricSearch::get_pairs(indices, offsets);

    EXPECT_EQ(pairs.size(), 1);
    EXPECT_EQ(pairs[0].first, 0);
    EXPECT_EQ(pairs[0].second, 0);
  }

  /**
   * Checking collision search with no primitives
   */
  TEST_F(GeometricSearch, CollisionSearchNoPrimitives)
  {
    const auto volumes = create_kdop_bounding_volumes();

    // Add the volumes to the primitives and predicates
    predicates_.emplace_back(std::pair{0, volumes[0]});
    predicates_.emplace_back(std::pair{1, volumes[1]});
    predicates_.emplace_back(std::pair{2, volumes[2]});

    EXPECT_EQ(primitives_.size(), 0);
    EXPECT_EQ(predicates_.size(), 3);

    const auto& [indices, offsets] =
        Core::GeometricSearch::collision_search(primitives_, predicates_, comm_, verbosity_);

    const auto pairs = Core::GeometricSearch::get_pairs(indices, offsets);

    EXPECT_EQ(pairs.size(), 0);
  }

  /**
   * Checking collision search with no predicates
   */
  TEST_F(GeometricSearch, CollisionSearchNoPredicates)
  {
    const auto volumes = create_kdop_bounding_volumes();

    // Add the volumes to the primitives and predicates
    primitives_.emplace_back(std::pair{0, volumes[0]});
    primitives_.emplace_back(std::pair{1, volumes[1]});
    primitives_.emplace_back(std::pair{2, volumes[2]});

    EXPECT_EQ(primitives_.size(), 3);
    EXPECT_EQ(predicates_.size(), 0);

    const auto& [indices, offsets] =
        Core::GeometricSearch::collision_search(primitives_, predicates_, comm_, verbosity_);

    const auto pairs = Core::GeometricSearch::get_pairs(indices, offsets);

    EXPECT_EQ(pairs.size(), 0);
  }

  /**
   * Checking collision search with no primitives and no predicates
   */
  TEST_F(GeometricSearch, CollisionSearchNoPrimitivesNoPredicates)
  {
    EXPECT_EQ(primitives_.size(), 0);
    EXPECT_EQ(predicates_.size(), 0);

    const auto& [indices, offsets] =
        Core::GeometricSearch::collision_search(primitives_, predicates_, comm_, verbosity_);

    const auto pairs = Core::GeometricSearch::get_pairs(indices, offsets);

    EXPECT_EQ(pairs.size(), 0);
  }

  /**
   * Check that the tolerance mechanism in the kDOP visualization works. The present values are
   * taken from a previously failing kDOP.
   */
  TEST_F(GeometricSearch, KdopVisualizationTolerances)
  {
    Core::GeometricSearch::BoundingVolume kdop;
    kdop.bounding_volume_._min_values[0] = -16.185886383056640625;
    kdop.bounding_volume_._max_values[0] = -15.878487586975097656;
    kdop.bounding_volume_._min_values[1] = -204.102142333984375;
    kdop.bounding_volume_._max_values[1] = -203.7178497314453125;
    kdop.bounding_volume_._min_values[2] = -178.448272705078125;
    kdop.bounding_volume_._max_values[2] = -178.121795654296875;
    kdop.bounding_volume_._min_values[3] = -220.28802490234375;
    kdop.bounding_volume_._max_values[3] = -219.78900146484375;
    kdop.bounding_volume_._min_values[4] = -194.6182403564453125;
    kdop.bounding_volume_._max_values[4] = -194.192138671875;
    kdop.bounding_volume_._min_values[5] = -382.412689208984375;
    kdop.bounding_volume_._max_values[5] = -381.9063720703125;
    kdop.bounding_volume_._min_values[6] = 187.547882080078125;
    kdop.bounding_volume_._max_values[6] = 188.1065521240234375;
    kdop.bounding_volume_._min_values[7] = 161.9359130859375;
    kdop.bounding_volume_._max_values[7] = 162.5491485595703125;
    kdop.bounding_volume_._min_values[8] = -25.9803466796875;
    kdop.bounding_volume_._max_values[8] = -25.2695770263671875;
    kdop.bounding_volume_._min_values[9] = -398.409820556640625;
    kdop.bounding_volume_._max_values[9] = -397.9437255859375;
    kdop.bounding_volume_._min_values[10] = 9.099609375;
    kdop.bounding_volume_._max_values[10] = 9.794464111328125;
    kdop.bounding_volume_._min_values[11] = -42.166229248046875;
    kdop.bounding_volume_._max_values[11] = -41.4358978271484375;
    kdop.bounding_volume_._min_values[12] = 365.86895751953125;
    kdop.bounding_volume_._max_values[12] = 366.5341796875;

    get_k_dop_polyhedron_representation(kdop);
  }
}  // namespace

#endif
