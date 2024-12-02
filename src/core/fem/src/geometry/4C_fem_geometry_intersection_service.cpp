// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_intersection_service.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_geometry_element_coordtrafo.hpp"
#include "4C_fem_geometry_intersection_service.templates.hpp"
#include "4C_fem_geometry_position_array.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ICS:    checks if an element is CARTESIAN, LINEAR and    u.may 07/08|
 |          HIGHERORDER                                                 |
 *----------------------------------------------------------------------*/
void Core::Geo::check_geo_type(const Core::Elements::Element* element,
    const Core::LinAlg::SerialDenseMatrix& xyze_element, EleGeoType& eleGeoType)
{
  bool cartesian = true;
  int CartesianCount = 0;
  const int dimCoord = 3;
  const Core::FE::CellType distype = element->shape();
  const int eleDim = Core::FE::get_dimension(distype);

  if (Core::FE::get_order(distype) == 1)
    eleGeoType = LINEAR;
  else if (Core::FE::get_order(distype) == 2)
    eleGeoType = HIGHERORDER;
  else
    FOUR_C_THROW("order of element shapefuntion is not correct");

  // check if cartesian
  if (eleDim == 3)
  {
    const std::vector<std::vector<int>> eleNodeNumbering =
        Core::FE::get_ele_node_numbering_surfaces(distype);
    std::vector<std::shared_ptr<Core::Elements::Element>> surfaces =
        (const_cast<Core::Elements::Element*>(element))->surfaces();
    for (int i = 0; i < element->num_surface(); i++)
    {
      CartesianCount = 0;
      const Core::Elements::Element* surfaceP = surfaces[i].get();

      for (int k = 0; k < dimCoord; k++)
      {
        int nodeId = eleNodeNumbering[i][0];
        const double nodalcoord = xyze_element(k, nodeId);
        for (int j = 1; j < surfaceP->num_node(); j++)
        {
          nodeId = eleNodeNumbering[i][j];
          if (fabs(nodalcoord - xyze_element(k, nodeId)) > TOL7)
          {
            CartesianCount++;
            break;
          }
        }
      }
      if (CartesianCount > 2)
      {
        cartesian = false;
        break;
      }
    }  // for xfem surfaces
  }    // if eleDim == 3
  else if (eleDim == 2 || eleDim == 1)
  {
    CartesianCount = 0;
    for (int k = 0; k < dimCoord; k++)
    {
      const double nodalcoord = xyze_element(k, 0);
      for (int j = 1; j < element->num_node(); j++)
      {
        if (fabs(nodalcoord - xyze_element(k, j)) > TOL7)
        {
          CartesianCount++;
          break;
        }
      }
    }
    if (CartesianCount > 2) cartesian = false;
  }
  else
    FOUR_C_THROW("dimension of element is not correct");



  if (cartesian) eleGeoType = CARTESIAN;
}


/*----------------------------------------------------------------------*
 | delivers a axis-aligned bounding box for a given          u.may 12/08|
 | discretization                                                       |
 *----------------------------------------------------------------------*/
std::map<int, Core::LinAlg::Matrix<3, 2>> Core::Geo::get_current_xaab_bs(
    const Core::FE::Discretization& dis,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions)
{
  std::map<int, Core::LinAlg::Matrix<3, 2>> currentXAABBs;
  // loop over elements and merge XAABB with their eXtendedAxisAlignedBoundingBox
  for (int j = 0; j < dis.num_my_col_elements(); ++j)
  {
    const Core::Elements::Element* element = dis.l_col_element(j);
    const Core::LinAlg::SerialDenseMatrix xyze_element(
        Core::Geo::get_current_nodal_positions(element, currentpositions));
    Core::Geo::EleGeoType eleGeoType(Core::Geo::HIGHERORDER);
    Core::Geo::check_geo_type(element, xyze_element, eleGeoType);
    const Core::LinAlg::Matrix<3, 2> xaabbEle =
        Core::Geo::compute_fast_xaabb(element->shape(), xyze_element, eleGeoType);
    currentXAABBs[element->id()] = xaabbEle;
  }
  return currentXAABBs;
}


/*----------------------------------------------------------------------*
 |  ICS:    checks if two 18DOPs intersect                   u.may 12/08| |
 *----------------------------------------------------------------------*/
bool Core::Geo::intersection_of_kdo_ps(
    const Core::LinAlg::Matrix<9, 2>& cutterDOP, const Core::LinAlg::Matrix<9, 2>& xfemDOP)
{
  // check intersection of 18 kdops
  for (int i = 0; i < 9; i++)
    if (!(((cutterDOP(i, 0) > (xfemDOP(i, 0) - Core::Geo::TOL7)) &&
              (cutterDOP(i, 0) < (xfemDOP(i, 1) + Core::Geo::TOL7))) ||
            ((cutterDOP(i, 1) > (xfemDOP(i, 0) - Core::Geo::TOL7)) &&
                (cutterDOP(i, 1) < (xfemDOP(i, 1) + Core::Geo::TOL7))) ||
            ((xfemDOP(i, 0) > (cutterDOP(i, 0) - Core::Geo::TOL7)) &&
                (xfemDOP(i, 0) < (cutterDOP(i, 1) + Core::Geo::TOL7))) ||
            ((xfemDOP(i, 1) > (cutterDOP(i, 0) - Core::Geo::TOL7)) &&
                (xfemDOP(i, 1) < (cutterDOP(i, 1) + Core::Geo::TOL7)))))
      return false;

  return true;
}


/*----------------------------------------------------------------------*
 |  checks the intersection between two bounding volumes (AABB)         |
 |                                                          wirtz 08/14 |
 *----------------------------------------------------------------------*/
bool Core::Geo::intersection_of_b_vs(
    const Core::LinAlg::Matrix<3, 2>& currentBV, const Core::LinAlg::Matrix<3, 2>& queryBV)
{
  return (overlap(currentBV(0, 0), currentBV(0, 1), queryBV(0, 0), queryBV(0, 1)) and
          overlap(currentBV(1, 0), currentBV(1, 1), queryBV(1, 0), queryBV(1, 1)) and
          overlap(currentBV(2, 0), currentBV(2, 1), queryBV(2, 0), queryBV(2, 1)));

  return 0;
}


/*----------------------------------------------------------------------*
 |  checks the overlap of two intervals in one coordinate               |
 |                                                          wirtz 08/14 |
 *----------------------------------------------------------------------*/
bool Core::Geo::overlap(double smin, double smax, double omin, double omax)
{
  return ((omax > smin - Core::Geo::TOL7 and omin < smax + Core::Geo::TOL7) or
          (smax > omin - Core::Geo::TOL7 and smin < omax + Core::Geo::TOL7));
}

FOUR_C_NAMESPACE_CLOSE
