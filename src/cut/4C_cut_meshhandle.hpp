// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_MESHHANDLE_HPP
#define FOUR_C_CUT_MESHHANDLE_HPP

#include "4C_config.hpp"

#include "4C_cut_elementhandle.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_sidehandle.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class Element;
  class LinearElement;
  class QuadraticElement;
  class Mesh;
  class Options;

  class SideHandle;
  class LinearSideHandle;
  class QuadraticSideHandle;

  class ElementHandle;
  class LinearElementHandle;
  class QuadraticElementHandle;

  /*!
   \brief Outside world interface to the mesh
   */
  class MeshHandle
  {
   public:
    /// constructor
    MeshHandle(Options& options, double norm = 1, Teuchos::RCP<PointPool> pp = Teuchos::null,
        bool cutmesh = false, int myrank = -1)
        : mesh_(options, norm, pp, cutmesh, myrank)
    {
    }

    /*========================================================================*/
    //! @name Create-routines for cut sides and mesh elements
    /*========================================================================*/

    /// create a new side (sidehandle) of the cutter discretization and return the sidehandle,
    /// non-tri3 sides will be subdivided into tri3 subsides depending on the options
    SideHandle* create_side(
        int sid, const std::vector<int>& nids, Core::FE::CellType distype, Cut::Options& options);

    /// create a new element (elementhandle) of the background discretization and return the
    /// elementhandle, quadratic elements will create linear shadow elements
    ElementHandle* create_element(
        int eid, const std::vector<int>& nids, Core::FE::CellType distype);

    /// create a new data structure for face oriented stabilization; the sides of the linear
    /// element are included into a sidehandle
    void create_element_sides(Element& element);

    /// create a new data structure for face oriented stabilization; the sides of the quadratic
    /// element are included into a sidehandle
    void create_element_sides(const std::vector<int>& nids, Core::FE::CellType distype);

    /*========================================================================*/
    //! @name Get-routines for nodes, cutter sides, elements and element sides
    /*========================================================================*/

    /// get the node based on node id
    Node* get_node(int nid) const;

    /// get the side (handle) based on side id of the cut mesh
    SideHandle* get_side(int sid) const;

    /// get the mesh's element based on element id
    ElementHandle* get_element(int eid) const;

    /// get the element' side of the mesh's element based on node ids
    SideHandle* get_side(std::vector<int>& nodeids) const;

    /// Remove this side from the Sidehandle (Used by the SelfCut)
    void remove_sub_side(Cut::Side* side);

    /// Add this side into the corresponding Sidehandle (Used by the SelfCut)
    void add_sub_side(Cut::Side* side);

    /// Mark this side as unphysical (Used by SelfCut)
    void mark_sub_sideas_unphysical(Cut::Side* side);

    /*========================================================================*/
    //! @name Get method for private variables
    /*========================================================================*/

    /// get the linear mesh
    Mesh& linear_mesh() { return mesh_; }

   private:
    /*========================================================================*/
    //! @name private class variables
    /*========================================================================*/

    Mesh mesh_;  ///< the linear mesh
    std::map<int, LinearElementHandle>
        linearelements_;  ///< map of element id and linear element handles
    std::map<int, Teuchos::RCP<QuadraticElementHandle>>
        quadraticelements_;  ///< map of element id and quadratic element handles
    std::map<int, LinearSideHandle> linearsides_;  ///< map of cut side id and linear side handles
    std::map<int, Teuchos::RCP<QuadraticSideHandle>>
        quadraticsides_;  ///< map of cut side id and quadratic side handles
    std::map<plain_int_set, LinearSideHandle>
        elementlinearsides_;  ///< map of element side node ids and linear side handles
    std::map<plain_int_set, Teuchos::RCP<QuadraticSideHandle>>
        elementquadraticsides_;  ///< map of element side node ids and quadratic side handles
  };

}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
