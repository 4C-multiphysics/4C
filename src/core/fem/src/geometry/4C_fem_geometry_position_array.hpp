// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GEOMETRY_POSITION_ARRAY_HPP
#define FOUR_C_FEM_GEOMETRY_POSITION_ARRAY_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Geo
{
  /*!
   * a common task is to get an array of the positions of all nodes of this element
   *
   * template version
   *
   * \note xyze is defined as (3,numnode)
   *
   * \return Array with 3 dimensional position of all element nodes in the coordinate system of the
   * nodes
   */
  template <class M>
  void fill_initial_position_array(const Core::Elements::Element* const ele, M& xyze)
  {
    const int numnode = ele->num_node();

    const Core::Nodes::Node* const* nodes = ele->nodes();
    FOUR_C_ASSERT(nodes != nullptr,
        "element has no nodal pointers, so getting a position array doesn't make sense!");

    for (int inode = 0; inode < numnode; inode++)
    {
      const auto& x = nodes[inode]->x();
      xyze(0, inode) = x[0];
      xyze(1, inode) = x[1];
      xyze(2, inode) = x[2];
    }
  }


  /*!
   * a common task is to get an array of the positions of all nodes of this element
   *
   * template version
   *
   * \note array is defined as (3,numnode)
   *
   * \return Array with 3 dimensional position of all element nodes in the coordinate system of the
   * nodes
   */
  template <Core::FE::CellType distype, class M>
  void fill_initial_position_array(const Core::Elements::Element* const ele, M& xyze)
  {
    FOUR_C_ASSERT(distype == ele->shape(), "mismatch in distype");
    const int numnode = Core::FE::num_nodes(distype);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    FOUR_C_ASSERT(nodes != nullptr,
        "element has no nodal pointers, so getting a position array doesn't make sense!");

    for (int inode = 0; inode < numnode; inode++)
    {
      const auto& x = nodes[inode]->x();
      xyze(0, inode) = x[0];
      xyze(1, inode) = x[1];
      xyze(2, inode) = x[2];
    }
  }


  /*!
   * a common task is to get an array of the positions of all nodes of this element
   *
   * template version
   *
   * \note array is defined as (dim,numnode) with user-specified number of space dimensions
   *       that are of interest for the element application calling this method
   *
   * \return Array with 1, 2 or 3 dimensional position of all element nodes in the coordinate system
   * of the nodes
   */
  template <Core::FE::CellType distype, int dim, class M>
  void fill_initial_position_array(const Core::Elements::Element* const ele, M& xyze)
  {
    FOUR_C_ASSERT(distype == ele->shape(), "mismatch in distype");
    const int numnode = Core::FE::num_nodes(distype);

    const Core::Nodes::Node* const* nodes = ele->nodes();
    FOUR_C_ASSERT(nodes != nullptr,
        "element has no nodal pointers, so getting a position array doesn't make sense!");

    FOUR_C_ASSERT((dim > 0) && (dim < 4), "Illegal number of space dimensions");

    for (int inode = 0; inode < numnode; inode++)
    {
      const double* x = nodes[inode]->x().data();
      // copy the values in the current column
      std::copy(x, x + dim, &xyze(0, inode));
      // fill the remaining entries of the column with zeros, if the given matrix has
      // the wrong row dimension (this is primarily for safety reasons)
      std::fill(&xyze(0, inode) + dim, &xyze(0, inode) + xyze.num_rows(), 0.0);
    }
  }


  /*!
   * a common task is to get an array of the positions of all nodes of this element
   *
   * \note array is defined as (3,numnode)
   *
   * \return Array with 3 dimensional position of all element nodes in the coordinate system of the
   * nodes
   */
  void initial_position_array(
      Core::LinAlg::SerialDenseMatrix& xyze, const Core::Elements::Element* const ele);


  /*!
   * a common task is to get an array of the positions of all nodes of this element
   *
   * \note array is defined as (3,numnode)
   *
   * \return Array with 3 dimensional position of all element nodes in the coordinate system of the
   * nodes
   */
  Core::LinAlg::SerialDenseMatrix initial_position_array(const Core::Elements::Element* const
          ele  ///< pointer to element, whose nodes we evaluate for their position
  );


  /*!
   * get current nodal positions
   *
   * \note array is defined as (3,numnode)
   *
   * \return Array with 3 dimensional position of all element nodes in the coordinate system of the
   * nodes
   */
  Core::LinAlg::SerialDenseMatrix get_current_nodal_positions(
      const Core::Elements::Element* const ele,  ///< element with nodal pointers
      const std::map<int, Core::LinAlg::Matrix<3, 1>>&
          currentcutterpositions  ///< current positions of all cutter nodes
  );


  /*!
   * get current nodal positions
   *
   * \note array is defined as (3,numnode)
   *
   * \return Array with 3 dimensional position of all element nodes in the coordinate system of the
   * nodes
   */
  Core::LinAlg::SerialDenseMatrix get_current_nodal_positions(
      const Core::Elements::Element& ele,  ///< pointer on element
      const std::map<int, Core::LinAlg::Matrix<3, 1>>&
          currentpositions  ///< current positions of all cutter nodes
  );

}  // namespace Core::Geo

FOUR_C_NAMESPACE_CLOSE

#endif
