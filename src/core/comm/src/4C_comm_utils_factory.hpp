// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COMM_UTILS_FACTORY_HPP
#define FOUR_C_COMM_UTILS_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Core::Nodes
{
  class Node;
}
namespace Core::Elements
{
  class Element;
}

namespace Core::Communication
{
  class ParObject;
  class UnpackBuffer;

  /*!
  \brief Create an instance of a ParObject depending on the type stored in data

  An instance of ParObject is allocated and returned. The type of instance
  depends on the first entry in data and is an int (defined at the top of
  ParObject.H)

  \param data (in): A char vector used for communication or io

  \warning All instances of ParObject have to store the parobject id
           at the very beginning of the char vector

  \return Allocates and returns the correct instance of ParObject where data is
          already unpacked in the instance. The calling method is responsible for
          freeing this instance!
  */
  ParObject* factory(Core::Communication::UnpackBuffer& buffer);

  /*!
  \brief Create an instance of a finite element depending on the type of element

  A Core::Elements::Element derived class is allocated and returned. The type of element
  allocated depends on the input parameter eletype.

  \param eletype (in): A string containing the type of element
  \param distype (in): A string containing the distype of the element
  \param id      (in): id of the new element to be created
  \param owner   (in): owner of the new element

  */
  std::shared_ptr<Core::Elements::Element> factory(
      const std::string eletype, const std::string distype, const int id, const int owner);

  //! flag, whether surfaces or lines have to be created in the element_boundary_factory
  enum BoundaryBuildType
  {
    buildSurfaces,  ///< build surfaces
    buildLines,     ///< build lines
    buildNothing    ///< build nothing
  };

  /*!
   * create new instances of volume / surface / line elements for a given parent element
   *
   * template version
   *
   * \tparam BoundaryEle class name of desired volume/surface/line element, e.g. FluidSurface
   * \tparam ParentEle   class name of parent element, e.g. Fluid
   *
   * this templated function creates all volume / surface / line elements for a given element
   * and fills handed over vectors with std::shared_ptrs and raw pointers of these boundary elements
   * This method is a very powerful helper function for implementing the necessary
   * Surfaces() and Lines() methods for every element class (especially in 3D)
   * using the 4C conventions for element connectivity
   *
   * \return boundaryeles   vector filled with std::shared_ptrs of allocated boundary elements
   *

   */
  template <class BoundaryEle, class ParentEle>
  std::vector<std::shared_ptr<Core::Elements::Element>> element_boundary_factory(
      const BoundaryBuildType
          buildtype,  ///< flag, whether volumes, surfaces or lines have to be created
      ParentEle& ele  ///< pointer on the parent element
  )
  {
    // do we have to build volume, surface or line elements?
    // get node connectivity for specific distype of parent element
    unsigned int nele = 0;
    const Core::FE::CellType distype = ele.shape();
    std::vector<std::vector<int>> connectivity;
    switch (buildtype)
    {
      case buildSurfaces:
      {
        nele = ele.num_surface();
        connectivity = Core::FE::get_ele_node_numbering_surfaces(distype);
        break;
      }
      case buildLines:
      {
        nele = ele.num_line();
        connectivity = Core::FE::get_ele_node_numbering_lines(distype);
        break;
      }
      default:
        FOUR_C_THROW("buildNothing case not handled in element_boundary_factory");
    }
    // create vectors that will contain the volume, surface or line elements
    std::vector<std::shared_ptr<Core::Elements::Element>> boundaryeles(nele);

    // does Discret::UTILS convention match your implementation of NumSurface() or NumLine()?
    if (nele != connectivity.size()) FOUR_C_THROW("number of surfaces or lines does not match!");

    // now, build the new surface/line elements
    for (unsigned int iele = 0; iele < nele; iele++)
    {
      // allocate node vectors
      unsigned int nnode = connectivity[iele].size();  // this number changes for pyramids or wedges
      std::vector<int> nodeids(nnode);
      std::vector<Core::Nodes::Node*> nodes(nnode);

      // get connectivity infos
      for (unsigned int inode = 0; inode < nnode; inode++)
      {
        nodeids[inode] = ele.point_ids()[connectivity[iele][inode]];
        nodes[inode] = ele.points()[connectivity[iele][inode]];
      }

      // allocate a new boundary element
      boundaryeles[iele] = std::make_shared<BoundaryEle>(
          iele, ele.owner(), nodeids.size(), nodeids.data(), nodes.data(), &ele, iele);
    }

    return boundaryeles;
  }

  /*!
   * create new instances of volume / surface / line elements for a given parent element
   *
   * template version
   *
   * \tparam IntFaceEle  class name of desired volume/surface/line element, e.g. FluidSurface
   * \tparam ParentEle   class name of parent element, e.g. Fluid
   *
   * this templated function creates an internal face element for two given parent elements
   * and fills an std::shared_ptr and raw pointers of this internal faces elements
   * This method is a very powerful helper function for implementing the necessary
   * Surfaces() and Lines() methods for every element class (especially in 3D)
   * using the 4C conventions for element connectivity
   *
   * \return intface   std::shared_ptr of allocated internal face element
   *

   */
  template <class IntFaceEle, class ParentEle>
  std::shared_ptr<Core::Elements::Element> element_int_face_factory(int id,  ///< element id
      int owner,                  ///< owner (= owner of parent element with smallest gid)
      int nnode,                  ///< number of nodes
      const int* nodeids,         ///< node ids
      Core::Nodes::Node** nodes,  ///< nodes of surface
      ParentEle* master_ele,      ///< pointer on the master parent element
      ParentEle* slave_ele,       ///< pointer on the slave parent element
      const int lsurface_master,  ///< local surface index with respect to master parent element
      const int lsurface_slave,   ///< local surface index with respect to slave parent element
      const std::vector<int> localtrafomap  ///< local trafo map
  )
  {
    // create a new internal face element
    return std::make_shared<IntFaceEle>(id, owner, nnode, nodeids, nodes, master_ele, slave_ele,
        lsurface_master, lsurface_slave, localtrafomap);
  }

  template <class BoundaryEle, class ParentEle>
  std::vector<std::shared_ptr<Core::Elements::Element>> get_element_lines(ParentEle& ele)
  {
    // 1D boundary element and 2D/3D parent element
    if (Core::FE::get_dimension(ele.shape()) > 1)
    {
      return element_boundary_factory<BoundaryEle, ParentEle>(buildLines, ele);
    }
    else if (Core::FE::get_dimension(ele.shape()) == 1)
    {
      // 1D boundary element and 1D parent element
      //  -> we return the element itself
      return {Core::Utils::shared_ptr_from_ref(ele)};
    }
    FOUR_C_THROW("Lines  does not exist for points.");
  }

  template <class BoundaryEle, class ParentEle>
  std::vector<std::shared_ptr<Core::Elements::Element>> get_element_surfaces(ParentEle& ele)
  {
    if (Core::FE::get_dimension(ele.shape()) > 2)
    {
      // 2D boundary element and 3D parent element
      return element_boundary_factory<BoundaryEle, ParentEle>(buildSurfaces, ele);
    }
    else if (Core::FE::get_dimension(ele.shape()) == 2)
    {
      // 2D boundary element and 2D parent element
      // -> we return the element itself
      return {Core::Utils::shared_ptr_from_ref(ele)};
    }

    FOUR_C_THROW("Surfaces do not exist for 1D-elements.");
  }

}  // namespace Core::Communication



FOUR_C_NAMESPACE_CLOSE

#endif
