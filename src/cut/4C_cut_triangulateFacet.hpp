// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_TRIANGULATEFACET_HPP
#define FOUR_C_CUT_TRIANGULATEFACET_HPP

#include "4C_config.hpp"

#include "4C_cut_side.hpp"

#include <list>
#include <vector>

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class Point;
  class Facet;

  /*!
  \brief A class to split a facet into tri and quad cells
   */
  class TriangulateFacet
  {
   public:
    /*!
    \brief Constructor for a facet without holes
     */
    TriangulateFacet(std::vector<Point *> ptlist) : ptlist_(ptlist) {}

    /*!
    \brief Constructor for a facet with holes
     */
    TriangulateFacet(std::vector<Point *> ptlist, std::vector<std::vector<Point *>> inlists)
        : ptlist_(ptlist)
    {
      if (hasequal_ptlist_inlist(ptlist, inlists)) return;
      for (std::vector<std::vector<Point *>>::iterator i = inlists.begin(); i != inlists.end(); ++i)
      {
        std::vector<Point *> inlist = *i;
        inlists_.push_back(inlist);
      }
    }

    /*!
    \brief Split the facet into appropriate number of tri and quad
     */
    void split_facet();

    /*!
    \brief A general facet is triangulated with ear clipping method.
    When triOnly=true calls conventional Earclipping method. Otherwise it creates both Tri and
    Quad cells to reduce the number of Gaussian points
     */
    void ear_clipping(std::vector<int> ptConcavity,
        bool triOnly = false,           // create triangles only?
        bool DeleteInlinePts = false);  // how to deal with collinear points?

    /*!
    \brief Ear Clipping is a triangulation method for simple polygons (convex, concave, with
    holes). It is simple and robust but not very efficient (O(n^2)). As input parameter the outer
    polygon (ptlist_) and the inner polygons (inlists_) are required. Triangles will be generated
    as output, which are all combined in one vector (split_).
    */
    void ear_clipping_with_holes(Side *parentside);

    /*!
    \brief Returns Tri and Quad cells that are created by facet splitting
     */
    std::vector<std::vector<Point *>> get_split_cells() { return split_; }

   private:
    /*!
    \brief The cyles ptlist and inlists are equal
    */
    bool hasequal_ptlist_inlist(
        std::vector<Point *> ptlist, std::vector<std::vector<Point *>> inlists);

    /*!
    \brief Split a concave 4 noded facet into a 2 tri
    */
    void split4node_facet(std::vector<Point *> &poly, bool callFromSplitAnyFacet = false);

    /*!
    \brief Split a convex facet or a facet with only one concave point into 1 Tri and few Quad
    cells
     */
    void split_convex_1pt_concave_facet(std::vector<int> ptConcavity);

    /*!
    \brief A concave facet which has more than 2 concavity points are split into appropriate cells
    */
    void split_general_facet(std::vector<int> ptConcavity);

    /*!
    \brief check whether any two adjacent polygonal points are concave
     */
    bool has_two_continuous_concave_pts(std::vector<int> ptConcavity);

    //! Restores last ear that was deleted during triangulation
    void restore_last_ear(int ear_head_index, std::vector<int> &ptConcavity);

    //! Goes clockwise from the the only no on-line point on the triangle and generates thin
    //! triangles
    void split_triangle_with_points_on_line(unsigned int start_id);

    //! Find second best ear, from the ones we discarded during the first check on the first round
    //! of earclipping
    unsigned int find_second_best_ear(
        std::vector<std::pair<std::vector<Point *>, unsigned int>> &ears,
        const std::vector<int> &reflex);

    //! Corner points of the facet
    std::vector<Point *> ptlist_;

    //! Points describing holes in this facet
    std::list<std::vector<Point *>> inlists_;

    //! Holds the split Tri and Quad cells
    std::vector<std::vector<Point *>> split_;
  };
}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
