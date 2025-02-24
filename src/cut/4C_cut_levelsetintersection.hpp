// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_LEVELSETINTERSECTION_HPP
#define FOUR_C_CUT_LEVELSETINTERSECTION_HPP

#include "4C_config.hpp"

#include "4C_cut_parentintersection.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Core::LinAlg
{
  class SerialDenseMatrix;
}


namespace Cut
{
  class Node;
  class Edge;
  class Side;
  class Element;

  /*!
  \brief Interface class for the level set cut.
  */
  class LevelSetIntersection : public virtual ParentIntersection
  {
    typedef ParentIntersection my;


   public:
    LevelSetIntersection(MPI_Comm comm, bool create_side = true);

    /** \brief add a side of the cut mesh and return the side-handle
     *
     * (e.g. quadratic side-handle for quadratic sides) */
    void add_cut_side(int levelset_sid);

    ///
    bool has_ls_cutting_side(int sid) { return true; /*return sid == side_->Id();*/ };

    /*========================================================================*/
    //! @name Cut functionality, routines
    /*========================================================================*/
    //! @{

    /*! \brief Performs the cut of the mesh with the level set
     *
     *  standard Cut routine for parallel Level Set Cut where dofsets and node
     *  positions have to be parallelized
     *
     *  \author winter
     *  \date 08/14  */
    void cut_mesh(bool screenoutput = false) override;

    /*! \brief Performs all the level set cut operations including find positions
     *  and triangulation. (Used for the test cases)
     *
     *  Standard Cut routine for two phase flow and combustion where dofsets and
     *  node positions have not to be computed, standard cut for cut_test (Only used
     *  for cut test)
     *
     *  \author winter
     *  \date 08/14  */
    void cut(bool include_inner = true, bool screenoutput = false,
        VCellGaussPts VCellGP = VCellGaussPts_Tessellation);

    //! @}
    /*========================================================================*/
    //! @name Add functionality for elements
    /*========================================================================*/
    //! @{

    /** \brief add this background element if it is cut. (determined by level set)
     *
     * Which implies that the level set function of the element has values which
     * are positive and negative. */
    Cut::ElementHandle* add_element(int eid, const std::vector<int>& nids,
        const Core::LinAlg::SerialDenseMatrix& xyz, Core::FE::CellType distype, const double* lsv,
        const bool lsv_only_plus_domain = false, const bool& check_lsv = false);

    //! @}

   private:
    MPI_Comm get_comm() const
    {
      if (not comm_) FOUR_C_THROW("Epetra communicator was not initialized!");

      return comm_;
    }

   protected:
    /*========================================================================*/
    //! @name private class variables
    /*========================================================================*/
    //! @{
    std::shared_ptr<Side> side_;

    MPI_Comm comm_;

    //! @}
  };

}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
