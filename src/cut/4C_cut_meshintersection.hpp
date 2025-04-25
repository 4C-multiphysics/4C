// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_MESHINTERSECTION_HPP
#define FOUR_C_CUT_MESHINTERSECTION_HPP

#include "4C_config.hpp"

#include "4C_cut_parentintersection.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE


namespace Cut
{
  class Node;
  class Edge;
  class Side;
  class Element;
  class ElementHandle;

  /*!
  \brief Interface class for the surface mesh cut. The surface mesh is in general triangulated.
  */
  class MeshIntersection : public virtual ParentIntersection
  {
    using my = ParentIntersection;


   public:
    /// constructor for MeshIntersection class
    explicit MeshIntersection(int numcutmesh = 1, int myrank = -1) : ParentIntersection(myrank)
    {
      cut_mesh_.reserve(numcutmesh);
      for (int i = 0; i < numcutmesh; ++i)
      {
        cut_mesh_.push_back(std::make_shared<MeshHandle>(options_, 1, pp_, true, myrank));
      }
    }

    /*========================================================================*/
    //! @name Add functionality for elements and sides
    /*========================================================================*/

    /// add this background element if it falls within the bounding box of cut mesh
    ElementHandle* add_element(int eid, const std::vector<int>& nids,
        const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype,
        const double* lsv = nullptr);

    /// add a side of the cut mesh and return the sidehandle (e.g. quadratic sidehandle for
    /// quadratic sides)
    SideHandle* add_cut_side(
        int sid, const std::vector<int>& nids, Core::FE::CellType distype, int mi = 0);

    /// add a side of the cut mesh and return the sidehandle (e.g. quadratic sidehandle for
    /// quadratic sides)
    SideHandle* add_cut_side(int sid, const std::vector<int>& nids,
        const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype, int mi = 0);

    /// build the static search tree for the collision detection in the self cut
    void build_self_cut_tree();

    /// build the static search tree for the collision detection
    void build_static_search_tree();

    /*========================================================================*/
    //! @name Cut functionality, routines
    /*========================================================================*/

    /// standard cut routine for non-parallel frameworks and cuttest
    void cut_test_cut(bool include_inner, VCellGaussPts VCellgausstype = VCellGaussPts_Tessellation,
        BCellGaussPts BCellgausstype = BCellGaussPts_Tessellation, bool tetcellsonly = true,
        bool screenoutput = true,
        bool do_Cut_Positions_Dofsets = false);  // for cuttest with closed cutsides this option
                                                 // can be activated, otherwise this will fail!

    /// handles cut sides which cut each other
    void cut_self_cut(bool include_inner, bool screenoutput = true) override;

    /// detects if a side of the cut mesh possibly collides with an element of the background mesh
    void cut_collision_detection(bool include_inner, bool screenoutput = true) override;

    /// Routine for cutting the mesh. This creates lines, facets, volumecells and quadrature rules
    void cut_mesh_intersection(bool screenoutput = true) override;

    /// Routine for deciding the inside-outside position. This creates the dofset data, just
    /// serial
    void cut_positions_dofsets(bool include_inner, bool screenoutput = true);

    /*========================================================================*/
    //! @name get routines for nodes, elements, sides, mesh, meshhandles
    /*========================================================================*/

    /// get the cut mesh's side based on side id
    SideHandle* get_cut_side(int sid, int mi = 0) const;

    /// get the cut mesh based on mesh id
    Mesh& cut_mesh(int i = 0) { return cut_mesh_[i]->linear_mesh(); }

   private:
    /*========================================================================*/
    //! @name private class variables
    /*========================================================================*/

    std::vector<std::shared_ptr<MeshHandle>> cut_mesh_;  ///< a vector of cut_meshes
  };

}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
