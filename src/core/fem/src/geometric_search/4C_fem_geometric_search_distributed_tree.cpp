// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometric_search_distributed_tree.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_geometric_search_access_traits.hpp"
#include "4C_fem_geometric_search_bounding_volume.hpp"
#include "4C_fem_geometric_search_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

#ifdef FOUR_C_WITH_ARBORX
#include <ArborX.hpp>
#endif

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  std::vector<GlobalCollisionSearchResult> global_collision_search(
      const std::vector<std::pair<int, BoundingVolume>>& primitives,
      const std::vector<std::pair<int, BoundingVolume>>& predicates, MPI_Comm comm,
      const Core::IO::Verbositylevel verbosity)
  {
#ifndef FOUR_C_WITH_ARBORX
    FOUR_C_THROW(
        "GeometricSearch::GlobalCollisionSearch can only be used with ArborX."
        "To use it, enable ArborX during the configure process.");
    return {};
#else

    TEUCHOS_FUNC_TIME_MONITOR("Core::GeometricSearch::GlobalCollisionSearch");

    int myrank = Core::Communication::my_mpi_rank(comm);

    using memory_space = Kokkos::HostSpace;
    using indexer_type = PrimitiveIndexableGetter;
    Kokkos::DefaultExecutionSpace execution_space{};

    std::vector<BoundingVolume> indexes(primitives.size());
    int id = 0;
    for (const auto& primitive : primitives)
    {
      indexes[id] = primitive.second;
      id++;
    }

    // Build tree structure containing all primitives.
    PrimitiveIndexableGetter indexable{indexes};
    ArborX::DistributedTree<memory_space, size_t, indexer_type> distributed_tree{
        comm, execution_space, indexes, indexable};

    // TODO: Check for better data structure in Kokkos (something like a tuple)
    Kokkos::View<Kokkos::pair<int, Kokkos::pair<int, int>>*, memory_space> indices_ranks_full(
        "indices_ranks_full", 0);
    Kokkos::View<int*, memory_space> offset_full("offset_full", 0);

    // The currently used ArborX version only supports tree structures for points and
    // axis-aligned-boundary-boxes (AABB). We convert the k-DOPs to AABB in the creation of the
    // tree. Therefore, the standard query would only give the intersections of the AABB of the
    // primitives with the predicates. With this callback we perform the query in the sense, that
    // we check each k-DOP of the primitives with each k-DOP of the predicates. This can lead to a
    // drastic reduction of the found pairs.
    auto IntersectActualVolumeType =
        KOKKOS_LAMBDA(const auto predicate, const int primitive_index, const auto& out)->void
    {
      const auto& primitive_vector_entry = primitives[primitive_index];
      const auto& primitive_geometry = primitive_vector_entry.second.bounding_volume_;
      if (predicate(primitive_geometry))
        out({primitive_index, {primitive_vector_entry.first, myrank}});
    };

    // Perform the collision check.
    distributed_tree.query(
        execution_space, predicates, IntersectActualVolumeType, indices_ranks_full, offset_full);

    // Create the vector with the pairs.
    std::vector<GlobalCollisionSearchResult> pairs;
    for (size_t i_offset = 0; i_offset < offset_full.size() - 1; i_offset++)
    {
      const int gid_predicate = predicates[i_offset].first;
      for (int j = offset_full[i_offset]; j < offset_full[i_offset + 1]; j++)
      {
        pairs.emplace_back(GlobalCollisionSearchResult{.lid_predicate = static_cast<int>(i_offset),
            .gid_predicate = gid_predicate,
            .lid_primitive = indices_ranks_full[j].first,
            .gid_primitive = indices_ranks_full[j].second.first,
            .pid_primitive = indices_ranks_full[j].second.second});
      }
    }

    if (verbosity == Core::IO::verbose)
    {
      Core::GeometricSearch::GeometricSearchInfo info = {static_cast<int>(primitives.size()),
          static_cast<int>(predicates.size()), static_cast<int>(indices_ranks_full.size())};
      Core::GeometricSearch::print_geometric_search_details(comm, info);
    }

    return pairs;
#endif
  }
}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE
