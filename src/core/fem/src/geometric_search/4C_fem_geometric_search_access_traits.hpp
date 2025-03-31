// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRIC_SEARCH_ACCESS_TRAITS_HPP
#define FOUR_C_FEM_GEOMETRIC_SEARCH_ACCESS_TRAITS_HPP

#include "4C_config.hpp"

#include "4C_fem_geometric_search_bounding_volume.hpp"

#ifdef FOUR_C_WITH_ARBORX
#include <ArborX.hpp>

struct PrimitiveIndexableGetter
{
  using memory_space = Kokkos::HostSpace;

  std::vector<FourC::Core::GeometricSearch::BoundingVolume> primitives;

  KOKKOS_FUNCTION auto size() const { return primitives.size(); }

  KOKKOS_FUNCTION auto operator()(int i) const
  {
    return ArborX::Box<3>{primitives[i].bounding_volume_};
  }
};

namespace ArborX
{
  using namespace FourC;

  template <>
  struct AccessTraits<std::vector<Core::GeometricSearch::BoundingVolume>>
  {
    using memory_space = Kokkos::HostSpace;
    using size_type = typename Kokkos::HostSpace::size_type;

    static KOKKOS_FUNCTION size_type size(
        const std::vector<Core::GeometricSearch::BoundingVolume>& vector)
    {
      return vector.size();
    }

    static KOKKOS_FUNCTION auto get(
        const std::vector<Core::GeometricSearch::BoundingVolume>& vector, size_type i)
    {
      return i;
    }
  };

  template <>
  struct AccessTraits<std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>>>
  {
    using memory_space = Kokkos::HostSpace;
    using size_type = typename Kokkos::HostSpace::size_type;

    static KOKKOS_FUNCTION size_type size(
        const std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>>& vector)
    {
      return vector.size();
    }

    static KOKKOS_FUNCTION auto get(
        const std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>>& vector,
        size_type i)
    {
      auto const& vector_entry = vector[i];
      return intersects(vector_entry.second.bounding_volume_);
    }
  };
}  // namespace ArborX

#endif

FOUR_C_NAMESPACE_OPEN
FOUR_C_NAMESPACE_CLOSE

#endif
