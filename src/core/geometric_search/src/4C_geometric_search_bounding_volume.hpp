// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRIC_SEARCH_BOUNDING_VOLUME_HPP
#define FOUR_C_GEOMETRIC_SEARCH_BOUNDING_VOLUME_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

#ifdef FOUR_C_WITH_ARBORX
#include <ArborX.hpp>
#include <ArborX_KDOP.hpp>
#endif

#ifdef FOUR_C_WITH_ARBORX

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  //! kdop_directions is the number of directions defining the possible faces of the k-DOP. In this
  //! case 13 means that we have the (3) Cartesian basis vectors, all vectors pointing to the edges
  //! of a unit cube (6) and all vectors pointing to corners of the unit cube (4). For more details
  //! have a look in ArborX_KDOP.hpp.
  constexpr int kdop_directions = 13;

  //! kdop_k is the number of bounding slabs for the k-DOP, i.e., the resulting polytope can have a
  //! maximum of kdop_k faces
  constexpr int kdop_k = 2 * kdop_directions;

  //! At the moment we consider all k-dops in R3
  constexpr int kdop_dim = 3;
}  // namespace Core::GeometricSearch
FOUR_C_NAMESPACE_CLOSE
#endif


#ifdef FOUR_C_WITH_ARBORX
namespace ArborX::Details
{
  /*! \brief  This struct helps to get access to the protected static member directions from ArborX.
   */
  template <int n_directions>
  struct GetKDOPDirections
      : private KDOP_Directions<FourC::Core::GeometricSearch::kdop_dim, 2 * n_directions, float>
  {
    static KOKKOS_FUNCTION auto const& directions()
    {
      return KDOP_Directions<FourC::Core::GeometricSearch::kdop_dim, 2 * n_directions,
          float>::directions();
    }
  };
}  // namespace ArborX::Details

#endif

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  /*!
   * \brief Class representing a bounding volume of a set of points.
   *
   * Internal this class uses k-DOPs implemented in ArborX. A k-DOP (discrete oriented polytope) is
   * a convex polytope containing the object.
   */
  struct BoundingVolume
  {
#ifndef FOUR_C_WITH_ARBORX
    /*! \brief This class can not be used without ArborX, add empty methods and a controlled error.
     */
    BoundingVolume()
    {
      FOUR_C_THROW(
          "The struct 'Core::GeometricSearch::BoundingVolume' can only be used with ArborX."
          "To use it, enable ArborX during the configure process.");
    }
    inline void extend_boundaries(const double offset) {}
    inline void add_point(const Core::LinAlg::Matrix<3, 1, double>& point) {}
#else
    /*! \brief Constructor initializing the bounding volume corners with numerical limit values.
     */
    BoundingVolume() : bounding_volume_{} {}

    /*! \brief Adds a point to the bounding volume.
     *
     * @param point Point to add to the bounding volume
     */
    inline void add_point(const Core::LinAlg::Matrix<3, 1, double>& point)
    {
      expand(bounding_volume_, ArborX::Point{static_cast<float>(point(0)),
                                   static_cast<float>(point(1)), static_cast<float>(point(2))});
    }

    /*! \brief Extends the bounding volume based on a scalar value.
     *
     * @param offset Value by which to expand the bounding volume
     */
    inline void extend_boundaries(const double offset)
    {
      // Loop over directions.
      for (int i_dir = 0; i_dir < kdop_directions; i_dir++)
      {
        const auto& direction =
            ArborX::Details::GetKDOPDirections<kdop_directions>::directions()[i_dir];
        const auto direction_scaling_factor =
            std::sqrt(direction._coords[0] * direction._coords[0] +
                      direction._coords[1] * direction._coords[1] +
                      direction._coords[2] * direction._coords[2]);
        const auto scaled_offset = offset * direction_scaling_factor;
        bounding_volume_._min_values[i_dir] -= static_cast<float>(scaled_offset);
        bounding_volume_._max_values[i_dir] += static_cast<float>(scaled_offset);
      }
    }

    /**
     * @brief Uniformly extends the bounding volume along all k-DOP directions.
     *
     * @param value The value by which the bounding volume is extended to in both the negative and
     * positive direction for each k-DOP axis.
     */
    inline void extend_boundary_to(const double value)
    {
      for (int i_dir = 0; i_dir < kdop_directions; ++i_dir)
      {
        bounding_volume_._min_values[i_dir] -= static_cast<float>(value);
        bounding_volume_._max_values[i_dir] += static_cast<float>(value);
      }
    }

    //! Use an ArborX geometry as internal storage.
    ArborX::Experimental::KDOP<kdop_dim, kdop_k> bounding_volume_;
#endif
  };
}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE

#endif
