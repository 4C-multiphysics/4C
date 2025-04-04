// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometric_search_bvh.hpp"

#include "4C_fem_geometric_search_access_traits.hpp"
#include "4C_fem_geometric_search_bounding_volume.hpp"
#include "4C_fem_geometric_search_utils.hpp"
#include "4C_io_pstream.hpp"

#include <Teuchos_TimeMonitor.hpp>

#ifdef FOUR_C_WITH_ARBORX
#include <ArborX.hpp>
#endif

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  std::pair<std::vector<int>, std::vector<int>> collision_search(
      const std::vector<std::pair<int, BoundingVolume>>& primitives,
      const std::vector<std::pair<int, BoundingVolume>>& predicates, MPI_Comm comm,
      const Core::IO::Verbositylevel verbosity)
  {
#ifndef FOUR_C_WITH_ARBORX
    FOUR_C_THROW(
        "Core::GeometricSearch::CollisionSearch can only be used with ArborX."
        "To use it, enable ArborX during the configure process.");
    return {};
#else

    TEUCHOS_FUNC_TIME_MONITOR("Core::GeometricSearch::CollisionSearch");

    std::vector<int> indices_final;
    std::vector<int> offsets_final;

    if (primitives.size() == 0 or predicates.size() == 0)
    {
      // This is the special case, where we can a priori say that there are no collisions. ArborX
      // produces a floating point exception if primitives.size() == 0, therefore, we take care of
      // this special case here.

      // Indices stays empty, as there are no collisions.
      indices_final.resize(0);
      // Offsets have to be filled so put all indices to 0, to align with the expected ArborX
      // output.
      offsets_final.resize(predicates.size() + 1, 0);
    }
    else
    {
      using memory_space = Kokkos::HostSpace;
      Kokkos::DefaultExecutionSpace execution_space{};

      // Build tree structure containing all primitives.
      ArborX::BoundingVolumeHierarchy bounding_volume_hierarchy{
          execution_space, ArborX::Experimental::attach_indices(
                               BoundingVolumeVectorPlaceholder<PrimitivesTag>{primitives})};

      Kokkos::View<int*, memory_space> indices_full("indices_full", 0);
      Kokkos::View<int*, memory_space> offset_full("offset_full", 0);

      auto get_indices_callback =
          KOKKOS_LAMBDA(const auto predicate, const auto& value, const auto& out)->void
      {
        out(value.index);
      };

      // Perform the collision check.
      bounding_volume_hierarchy.query(execution_space,
          BoundingVolumeVectorPlaceholder<PredicatesTag>{predicates}, get_indices_callback,
          indices_full, offset_full);

      // Copy kokkos view to std::vector
      indices_final.insert(
          indices_final.begin(), indices_full.data(), indices_full.data() + indices_full.extent(0));
      offsets_final.insert(
          offsets_final.begin(), offset_full.data(), offset_full.data() + offset_full.extent(0));
    }

    if (verbosity == Core::IO::verbose)
    {
      Core::GeometricSearch::GeometricSearchInfo info = {static_cast<int>(primitives.size()),
          static_cast<int>(predicates.size()), static_cast<int>(indices_final.size())};
      Core::GeometricSearch::print_geometric_search_details(comm, info);
    }

    return {indices_final, offsets_final};
#endif
  }
}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE
