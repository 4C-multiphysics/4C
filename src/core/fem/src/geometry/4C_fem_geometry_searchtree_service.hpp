// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_SEARCHTREE_SERVICE_HPP
#define FOUR_C_FEM_GEOMETRY_SEARCHTREE_SERVICE_HPP

#include "4C_config.hpp"

#include "4C_fem_geometry_geo_utils.hpp"
#include "4C_fem_geometry_searchtree_nearestobject.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <memory>
#include <set>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Elements
{
  class Element;
}

namespace Core::LinAlg
{
  class SerialDenseMatrix;
}

namespace Core::Geo
{
  class NearestObject;

  /*!
   \brief Returns the eXtendedAxisAlignedBoundingBox of a discretization,
   also takes into account current displacements
   \param dis                  discretization
   \param currentpositions     current nodal positions in discretization
   \return  XAxisAlignedBoundingBox
   */
  Core::LinAlg::Matrix<3, 2> get_xaab_bof_dis(const Core::FE::Discretization& dis,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions);

  /*!
   \brief Returns the eXtendedAxisAlignedBoundingBox of a discretization,
   in reference configuration
   \param dis                  discretization
   \return  XAxisAlignedBoundingBox
   */
  Core::LinAlg::Matrix<3, 2> get_xaab_bof_dis(const Core::FE::Discretization& dis);

  /*!
   \brief Returns the eXtendedAxisAlignedBoundingBox of coords
   \param currentpositions     current nodal positions
   \return  XAxisAlignedBoundingBox
   */
  Core::LinAlg::Matrix<3, 2> get_xaab_bof_positions(
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions);

  /*!
   \brief Returns the eXtendedAxisAlignedBoundingBox of given elements
   \param elements           elements the box is around, enlarged a little bit for enclosing all
   source nodes \param currentpositions     current nodal positions of elements \return
   XAxisAlignedBoundingBox
   */
  Core::LinAlg::Matrix<3, 2> get_xaab_bof_eles(
      std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions);

  /*!
   \brief Returns a vector of eXtendedAxisAlignedBoundingBox for labeled structures
   \param dis                  discretization
   \param currentpositions     current nodal positions in discretization
   \param elementList          map of int = label and set of element ids
   \return  AxisAlignedBoundingBox as Core::LinAlg::Matrix<3,2>
   */
  std::vector<Core::LinAlg::Matrix<3, 2>> compute_xaabb_for_labeled_structures(
      const Core::FE::Discretization& dis,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const std::map<int, std::set<int>>& elementList);

  /*!
   \brief Returns a map of element ids lying in a circle of a given querypoint and a radius
   \param dis                  discretization
   \param currentpositions     current nodal positions in discretization
   \param querypoint           point to be examined
   \param radius               radius
   \param label                label
   \param elementList          map of elements label gid
   \return                     label
   */
  std::map<int, std::set<int>> get_elements_in_radius(const Core::FE::Discretization& dis,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const Core::LinAlg::Matrix<3, 1>& querypoint, const double radius, const int label,
      std::map<int, std::set<int>>& elementList);

  /*!
   \brief Returns a set of element gids. The bounding volumes (XAABB) of this elements is
   overlapping with the bounding volume (XAABB) of the query element. \param currentBVs bounding
   volumes (XAABB) of the considered elements \param queryBV            bounding volume (XAABB) of
   the query element \param label              ??? \param elementList        map of elements label
   gid \param collisions         set of gids of elements
   */
  void search_collisions(const std::map<int, Core::LinAlg::Matrix<3, 2>>& currentBVs,
      const Core::LinAlg::Matrix<3, 2>& queryBV, const int label,
      const std::map<int, std::set<int>>& elementList, std::set<int>& collisions);

  /*!
   \brief Returns a set of element gids. The bounding volumes (18-kdop) of this elements is
   overlapping with the bounding volume (18-kdop) of the query element. \param currentKDOPs bounding
   volumes (18-kdop) of the considered elements \param queryKDOP          bounding volume (18-kdop)
   of the query element \param label              ??? \param elementList        map of elements
   label gid \param collisions         set of gids of elements
   */
  void search_collisions(const std::map<int, Core::LinAlg::Matrix<9, 2>>& currentKDOPs,
      const Core::LinAlg::Matrix<9, 2>& queryKDOP, const int label,
      const std::map<int, std::set<int>>& elementList, std::set<int>& contactEleIds);

  /*!
   \brief Searches for the nearest point to the query point in elementList
   \param dis                  discretization
   \param elements            list of all elements
   \param currentpositions     current nodal positions in discretization
   \param elementList          element list of close elements, is looped
   \param point                point to be examined
   \param minDistCoords        Coords of the nearest point
   \return surface id of nearest object (node or line: a random adjacent surface is chosen)
   */
  int nearest_3d_object_in_node(const Core::FE::Discretization& dis,
      std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const std::map<int, std::set<int>>& elementList, const Core::LinAlg::Matrix<3, 1>& point,
      Core::LinAlg::Matrix<3, 1>& minDistCoords);

  /// returns the nearest coordinates on element and the corresponding object type
  Core::Geo::ObjectType nearest_3d_object_on_element(Core::Elements::Element* surfaceelement,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const Core::LinAlg::Matrix<3, 1>& point, Core::LinAlg::Matrix<3, 1>& minDistCoords);

  void nearest_2d_object_in_node(const Core::FE::Discretization& dis,
      std::map<int, std::shared_ptr<Core::Elements::Element>>& elements,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const std::map<int, std::set<int>>& elementList, const Core::LinAlg::Matrix<3, 1>& point,
      Core::LinAlg::Matrix<3, 1>& minDistCoords);

  /*!
   \brief Searches for the nearest surface element to a given point
   \param surfaceElement       surface element
   \param currentpositions     current nodal positions in discretization
   \param point                point to be examined
   \param x_surface_phys       physical coordinates
   \param distance             distance to nearest element
   \return true if nearest surface element found
   */
  bool get_distance_to_surface(const Core::Elements::Element* surfaceElement,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const Core::LinAlg::Matrix<3, 1>& point, Core::LinAlg::Matrix<3, 1>& x_surface_phys,
      double& distance);

  /*!
   \brief Searches for the nearest line element to a given point
   \param lineElement          lineElement
   \param currentpositions     current nodal positions in discretization
   \param point                point to be examined
   \param x_line_phys          physical coordinates
   \param distance             distance to nearest point
   \return true if nearest line element found
   */
  bool get_distance_to_line(const Core::Elements::Element* lineElement,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const Core::LinAlg::Matrix<3, 1>& point, Core::LinAlg::Matrix<3, 1>& x_line_phys,
      double& distance);

  /*!
   \brief Searches for the nearest node to a given point
   \param node                 node
   \param currentpositions     current nodal positions in discretization
   \param point                point to be examined
   \param distance             distance to nearest point
   */
  void get_distance_to_point(const Core::Nodes::Node* node,
      const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
      const Core::LinAlg::Matrix<3, 1>& point, double& distance);

  /*!
   \brief  Checks if a point lies with in a node box
   \param point         query point
   \param nodeBox       box of tree node
   \return true, if a point lies with in a node box, false otherwise
   */
  bool point_in_tree_node(
      const Core::LinAlg::Matrix<3, 1>& point, const Core::LinAlg::Matrix<3, 2>& nodeBox);

  /*!
   \brief Merges two axis-aligned bounding boxes
   \param AABB1          axis - aligned bounding box 1
   \param AABB2          axis - aligned bounding box 2
   \return merged axis-aligned bounding box
   */
  Core::LinAlg::Matrix<3, 2> merge_aabb(
      const Core::LinAlg::Matrix<3, 2>& AABB1, const Core::LinAlg::Matrix<3, 2>& AABB2);

  /*!
   \brief Check the type of geometry of an element,
   rough because only linear or nonlinear is checked
   \param element              element
   \param xyze_element         nodal coordinates
   \param eleGeoType           geometry type
   */
  void check_rough_geo_type(const Core::Elements::Element* element,
      const Core::LinAlg::SerialDenseMatrix xyze_element, Core::Geo::EleGeoType& eleGeoType);

  /*!
   \brief Check the type of geometry of an element,
   rough because only linear or nonlinear is checked
   \param element              element
   \param xyze_element         nodal coordinates
   \param eleGeoType           geometry type
   */
  void check_rough_geo_type(const Core::Elements::Element& element,
      Core::LinAlg::SerialDenseMatrix xyze_element, Core::Geo::EleGeoType& eleGeoType);

}  // namespace Core::Geo

FOUR_C_NAMESPACE_CLOSE

#endif
