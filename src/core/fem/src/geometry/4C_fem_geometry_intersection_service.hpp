// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_INTERSECTION_SERVICE_HPP
#define FOUR_C_FEM_GEOMETRY_INTERSECTION_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fem_geometry_geo_utils.hpp"
#include "4C_fem_geometry_intersection_math.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Core::Geo
{
  std::map<int, Core::LinAlg::Matrix<3, 2>> get_current_xaab_bs(const Core::FE::Discretization& dis,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions);

  /*!
  \brief checks if two 18 Dops are intersecting (note : for efficiency it only checks slabs
         which are not present for XAABBs)

  \param cutterDOP   (in)         : DOP of the cutting element
  \param xfemDOP     (in)         : DOP of the xfem element
  \return true if the DOP's intersect or false otherwise
   */
  bool intersection_of_kdo_ps(
      const Core::LinAlg::Matrix<9, 2>& cutterDOP, const Core::LinAlg::Matrix<9, 2>& xfemDOP);

  /*!
  \brief checks the intersection between two bounding volumes (AABB)
  \param currentBV   (in)         : AABB of the current element
  \param queryBV     (in)         : AABB of the query element
  \return true if the AABB's intersect or false otherwise
   */
  bool intersection_of_b_vs(
      const Core::LinAlg::Matrix<3, 2>& currentBV, const Core::LinAlg::Matrix<3, 2>& queryBV);

  /*!
  \brief checks the overlap of two intervals in one coordinate
  \param smin     (in)         : minimum value of the current interval
  \param smax     (in)         : maximum value of the current interval
  \param omin     (in)         : minimum value of the query interval
  \param omax     (in)         : maximum value of the query interval
  \return true if the intervals's overlap or false otherwise
   */
  bool overlap(double smin, double smax, double omin, double omax);

  /*!
  \brief checks if an element is Cartesian, linear or higherorder

  \param element        (in)         : element
  \param xyze_element   (in)         : coordinates of the element
  \param eleGeoType     (out)        : element geometric type CARTESIAN LINEAR or HIGHERORDER
   */
  void check_geo_type(const Core::Elements::Element* element,
      const Core::LinAlg::SerialDenseMatrix& xyze_element, EleGeoType& eleGeoType);

}  // namespace Core::Geo


FOUR_C_NAMESPACE_CLOSE

#endif
