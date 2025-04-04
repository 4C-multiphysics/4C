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

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  struct PrimitivesTag
  {
  };

  struct PredicatesTag
  {
  };

  template <typename Tag>
  struct BoundingVolumeVectorPlaceholder
  {
    const std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>>& bounding_volumes_;
  };
}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE


namespace ArborX
{
  using namespace FourC::Core::GeometricSearch;

  template <typename Tag>
  struct AccessTraits<BoundingVolumeVectorPlaceholder<Tag>>
  {
    using memory_space = Kokkos::HostSpace;
    using size_type = typename Kokkos::HostSpace::size_type;

    static KOKKOS_FUNCTION size_type size(const BoundingVolumeVectorPlaceholder<Tag>& placeholder)
    {
      return placeholder.bounding_volumes_.size();
    }

    static KOKKOS_FUNCTION auto get(
        const BoundingVolumeVectorPlaceholder<PrimitivesTag>& placeholder, size_type i)
    {
      return placeholder.bounding_volumes_[i].second.bounding_volume_;
    }

    static KOKKOS_FUNCTION auto get(
        const BoundingVolumeVectorPlaceholder<PredicatesTag>& placeholder, size_type i)
    {
      return intersects(placeholder.bounding_volumes_[i].second.bounding_volume_);
    }
  };
}  // namespace ArborX

#endif

FOUR_C_NAMESPACE_OPEN
FOUR_C_NAMESPACE_CLOSE

#endif
