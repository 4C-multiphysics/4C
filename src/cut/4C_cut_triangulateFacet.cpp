// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_triangulateFacet.hpp"

#include "4C_cut_boundingbox.hpp"
#include "4C_cut_facet.hpp"
#include "4C_cut_kernel.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_side.hpp"

#include <math.h>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------------------*
              Split the facet into appropriate number of tri and quad Sudhakar 04/12 Work well
for both convex and concave facets
*------------------------------------------------------------------------------------------------------------*/
void Cut::TriangulateFacet::split_facet()
{
  // An edge should contain only 2 end points
  // delete all remaining points on the edge
  Kernel::delete_inline_pts(ptlist_);

  // Deal with zero point facet -- this should never occur, but happens in axel10 cut-test
  if (ptlist_.size() == 0) return;

  int numpts = ptlist_.size();

  if (numpts < 3)
  {
    FOUR_C_THROW("A facet must have atleast 3 corners");
  }
  else if (numpts == 3)  // form a Tri, and it is done
  {
    split_.push_back(ptlist_);
    return;
  }
  else
  {
    split_.clear();

    // get concave (reflex) points of polygon
    Cut::FacetShape geoType;
    std::vector<int> ptConcavity = Kernel::check_convexity(ptlist_, geoType);

    // a convex polygon or a polygon with only one concave point can be
    // very easily split
    if (geoType == Cut::Convex or geoType == Cut::SinglePtConcave)
    {
      split_convex_1pt_concave_facet(ptConcavity);
    }
    // facet with more than 1 concave point
    else
    {
      split_general_facet(ptConcavity);
    }
  }
}

/*-------------------------------------------------------------------------------------*
   If a 4 noded facet is convex, a quad is made. If it is concave,
   it is split into appropriate triangles.
   The Gauss integration rule is available for quad, only if it is convex

                                   /\
                                  / .\
                                 /A .B\                                   sudhakar 04/12
                                /  ++  \
                               / +    + \
                               +        +
*--------------------------------------------------------------------------------------*/
void Cut::TriangulateFacet::split4node_facet(std::vector<Point*>& poly, bool callFromSplitAnyFacet)
{
  if (poly.size() != 4) FOUR_C_THROW("This is not a 4 noded facet");

  Cut::FacetShape geoType;
  std::vector<int> ptConcavity = Kernel::check_convexity(poly, geoType);

  int indStart = 0;

  // convex quad can be directly added
  if (geoType == Cut::Convex)
  {
    split_.push_back(poly);
    return;
  }
  // concave quad is split into two tri cells
  else if (geoType == Cut::SinglePtConcave)
  {
    indStart = ptConcavity[0];

    std::vector<Point*> temp1(3), temp2(3);
    temp1[0] = poly[indStart];
    temp1[1] = poly[(indStart + 1) % 4];
    temp1[2] = poly[(indStart + 2) % 4];

    temp2[0] = poly[indStart];
    temp2[1] = poly[(indStart + 2) % 4];
    temp2[2] = poly[(indStart + 3) % 4];

    if (split_.size() == 0)
    {
      split_.resize(2);
      split_[0] = temp1;
      split_[1] = temp2;
    }
    else
    {
      split_.push_back(temp1);
      split_.push_back(temp2);
    }
    return;
  }
  else  // if there are two concave pts for 4 noded facet --> selfcut
  {
    // this means splitGeneralFacet is failed --> call earclipping
    /*if( callFromSplitAnyFacet )
    {
      std::cout<<"WARNING!!!! calling earclipping because splitanyfacet failed\n";
      ptConcavity = Kernel::CheckConvexity(  ptlist_, geoType );
      split_.clear();
      EarClipping( ptConcavity, false );
      return;
    }*/
    std::cout << "the points are\n";
    for (unsigned i = 0; i < poly.size(); i++)
    {
      Point* p1 = poly[i];
      double x1[3];
      p1->coordinates(x1);
      std::cout << x1[0] << "\t" << x1[1] << "\t" << x1[2] << "\n";
    }
    FOUR_C_THROW(
        "a 4 noded facet cannot have more than 1 concave point:"
        "This means that the facet is selfcut");
  }
}

/*---------------------------------------------------------------------------------------------------*
   A facet that is convex or having one concave point is split into 1 Tri and few Quad cells
   splitting starts with concave pt --- eliminate the need to check whether any other pt is
   inside the newCell formed                                                            Sudhakar
07/12
*----------------------------------------------------------------------------------------------------*/
void Cut::TriangulateFacet::split_convex_1pt_concave_facet(std::vector<int> ptConcavity)
{
  if (ptConcavity.size() > 1)
    FOUR_C_THROW("should be called only when the facet has one or no concave points");

  int num = ptlist_.size();

  if (ptlist_.size() == 3)
  {
    split_.push_back(ptlist_);
    return;
  }
  else if (ptlist_.size() == 4)
  {
    split4node_facet(ptlist_, false);
    return;
  }

  bool triDone = false, convex = true;
  int firstPt = 0, secondPt = 0, thirdPt = 0, lastPt = 1, endPt = num - 1;
  if (ptConcavity.size() == 1)
  {
    convex = false;
    firstPt = ptConcavity[0];
    lastPt = (firstPt + 1) % num;  // initialization
    if (firstPt != 0) endPt = firstPt - 1;
  }
  std::vector<Point*> newCell;

  while (1)
  {
    newCell.clear();

    secondPt = lastPt;  // properly initialized
    thirdPt = (secondPt + 1) % num;

    newCell.push_back(ptlist_[firstPt]);
    newCell.push_back(ptlist_[secondPt]);
    newCell.push_back(ptlist_[thirdPt]);  // tri cell is now formed

    if (thirdPt == endPt)
      triDone = true;
    else if (!triDone)  // check whether tri can be extended to quad cell
    {
      lastPt = (thirdPt + 1) % num;
      newCell.push_back(ptlist_[lastPt]);
      if (lastPt == endPt) triDone = true;
    }

    Kernel::delete_inline_pts(newCell);

    if (newCell.size() == 3 || convex)
      split_.push_back(newCell);
    else if (newCell.size() == 4)  // check the resulting quad is convex, else split into 2 tri
      split4node_facet(newCell, true);
    else
      FOUR_C_THROW("should have either 2 or 3 points");

    if (triDone) break;
  }
}

/*---------------------------------------------------------------------------------------------------*
 * Generalized facet splitting procedure which works for simple facets with any number  sudhakar
 *08/12 of concave points. Involves checking whether a reflex point is inside formed cell
 *---------------------------------------------------------------------------------------------------*/
void Cut::TriangulateFacet::split_general_facet(std::vector<int> ptConcavity)
{
  if (ptConcavity.size() < 2)
    FOUR_C_THROW("Call TriangulateFacet::split_convex_1pt_concave_facet in such cases");

  if (ptlist_.size() == 3)  // directly form a Tri cell
  {
    split_.push_back(ptlist_);
    return;
  }
  else if (ptlist_.size() == 4)  // can be a convex or concave quad
  {
    split4node_facet(ptlist_, false);
    return;
  }

  int num = ptlist_.size();
  int concsize = ptConcavity.size();

  int firstPt = 0, secondPt = 1, thirdPt = 2, fourthPt = 3;
  std::vector<Point*> newCell;

  while (1)
  {
    int start_size = ptlist_.size();
    if ((num - concsize) < 4)  // this means that no Quad cells can be formed for this geometry
    {
      ear_clipping(ptConcavity);
      return;
    }

    newCell.clear();
    int ncross = 0;
    for (int i = 0; i < num; ++i)
    {
      newCell.clear();
      ncross++;
      int concNo = 0;

      if (i == 0)
      {
        firstPt = ptConcavity[concNo];
        secondPt = (firstPt + 1) % num;
      }
      else
      {
        firstPt = (firstPt + 1) % num;
        secondPt = (firstPt + 1) % num;
      }
      if (std::find(ptConcavity.begin(), ptConcavity.end(), secondPt) != ptConcavity.end())
        continue;

      thirdPt = (secondPt + 1) % num;

      newCell.push_back(ptlist_[firstPt]);
      newCell.push_back(ptlist_[secondPt]);
      newCell.push_back(ptlist_[thirdPt]);  // tri cell is now formed

      // if 3rd point is not a concave point, then Quad can be formed
      if (std::find(ptConcavity.begin(), ptConcavity.end(), thirdPt) == ptConcavity.end())
      {
        fourthPt = (thirdPt + 1) % num;
        newCell.push_back(ptlist_[fourthPt]);
      }

      Kernel::delete_inline_pts(newCell);

      bool isEar = true;

      if (newCell.size() == 3)
      {
        for (unsigned j = 0; j < ptConcavity.size(); j++)
        {
          unsigned reflInd = ptConcavity[j];

          // considered pt is one of the corners of newCell --> no need to check
          if (std::find(newCell.begin(), newCell.end(), ptlist_[reflInd]) != newCell.end())
            continue;

          if (Kernel::pt_inside_triangle(newCell, ptlist_[reflInd]))
          {
            isEar = false;
            break;
          }
        }
        if (isEar)
        {
          ptlist_.erase(ptlist_.begin() + secondPt);  // erase a pt to form new polygon
          split_.push_back(newCell);
          break;
        }
      }
      else if (newCell.size() == 4)  // check the resulting quad is convex, else split into 2 tri
      {
        for (unsigned j = 0; j < ptConcavity.size(); j++)
        {
          unsigned reflInd = ptConcavity[j];
          // considered pt is one of the corners of newCell --> no need to check
          if (std::find(newCell.begin(), newCell.end(), ptlist_[reflInd]) != newCell.end())
            continue;

          if (Kernel::pt_inside_quad(newCell, ptlist_[reflInd]))
          {
            isEar = false;
            break;
          }
        }

        if (isEar)
        {
          split4node_facet(newCell, true);

          // erase internal points of cell to form new polygon
          // when a point is deleted, all other points are renumbered
          // if-else condition to make sure correct points are deleted
          if (thirdPt == 0)
          {
            ptlist_.erase(ptlist_.begin() + secondPt);
            ptlist_.erase(ptlist_.begin() + thirdPt);
          }
          else
          {
            ptlist_.erase(ptlist_.begin() + thirdPt);
            ptlist_.erase(ptlist_.begin() + secondPt);
          }
          break;
        }
      }
      else
      {
        std::cout << "number of points in the cell = " << newCell.size() << "\n";
        FOUR_C_THROW("neither tri nor quad: Something went wrong in SplitAnyFacet");
      }

      if (ncross == num) FOUR_C_THROW("cannot form cell even after making one cycle");
    }

    Kernel::delete_inline_pts(ptlist_);
    num = ptlist_.size();
    if (num == 3)
    {
      split_.push_back(ptlist_);
      return;
    }
    else if (num == 4)
    {
      split4node_facet(ptlist_, false);
      return;
    }
    else
    {
      ptConcavity.clear();
      Cut::FacetShape geoType;

      ptConcavity = Kernel::check_convexity(ptlist_, geoType);

      concsize = ptConcavity.size();
      if (concsize < 2)  // new ptlist_ forms a convex facet
      {
        split_convex_1pt_concave_facet(ptConcavity);
        return;
      }
      if (num == start_size)
      {
        // we can not split it with the normal strategy.
        // trianglulate the rest with earclipping
        // one may want more elaborate strategy here
        // if somethin goes wrong
        ear_clipping(ptConcavity, true, true);
        return;
        // FOUR_C_THROW("Starting infinite loop!");
      }
    }
  }
}

/*------------------------------------------------------------------------------------------------------*
            check whether the polygon has two continuous concave points.
            At the moment this is unused                                         sudhakar 04/12
*-------------------------------------------------------------------------------------------------------*/
bool Cut::TriangulateFacet::has_two_continuous_concave_pts(std::vector<int> ptConcavity)
{
  int size = ptConcavity.size();
  if (size < 2) return false;

  for (int i = 0; i < size; i++)
  {
    int firstPt = ptConcavity[i];
    int secondPt = ptConcavity[(i + 1) % size];
    if (firstPt != size - 1)
    {
      if ((secondPt - firstPt) == 1) return true;
    }
    else if ((secondPt - firstPt) == size - 1)
      return true;
  }
  return false;
}


void Cut::TriangulateFacet::restore_last_ear(int ear_head_index, std::vector<int>& ptConcavity)
{
  std::vector<Point*> last_added_ear = split_.back();
  split_.pop_back();
  ptlist_.insert(ptlist_.begin() + ear_head_index, last_added_ear[1]);

  Cut::FacetShape str1;
  ptConcavity.clear();
  ptConcavity =
      Kernel::check_convexity(ptlist_, str1, true, false);  // concave points for the new polygon
}


void Cut::TriangulateFacet::split_triangle_with_points_on_line(unsigned int start_id)
{
  unsigned int polPts = ptlist_.size();
  unsigned int split_start = (start_id + 1) % polPts;
  unsigned int split_end = (start_id == 0) ? (polPts - 1) : (start_id - 1) % polPts;

  for (unsigned int i = split_start; i != split_end; i = (i + 1) % polPts)
  {
    std::vector<Point*> tri(3);
    tri[0] = ptlist_[start_id];
    tri[1] = ptlist_[i];
    tri[2] = ptlist_[(i + 1) % polPts];
    split_.push_back(tri);
  }

  // we are done with everything
  ptlist_.clear();
}


unsigned int Cut::TriangulateFacet::find_second_best_ear(
    std::vector<std::pair<std::vector<Point*>, unsigned int>>& ears, const std::vector<int>& reflex)
{
  unsigned int polPts = ptlist_.size();
  std::vector<int> filtered_ears;
  unsigned int counter = 0;

  // first try to find ears, that do not contain other points,
  // using precise checks and ignoring 'inline' info
  for (auto& ear : ears)
  {
    std::vector<Point*> tri = ear.first;
    unsigned int i = ear.second;
    // recover the points
    unsigned ind0 = i - 1;
    if (i == 0) ind0 = polPts - 1;
    unsigned ind2 = (i + 1) % polPts;

    bool isEar = true;

    Core::LinAlg::Matrix<3, 3> tri_coord;
    tri[0]->coordinates(tri_coord.data());
    tri[1]->coordinates(tri_coord.data() + 3);
    tri[2]->coordinates(tri_coord.data() + 6);
    // check whether any point of polygon is inside
    for (unsigned j = 0; j < reflex.size(); j++)
    {
      unsigned reflInd = reflex[j];
      // skip points of the triangle itself
      if (reflInd == ind0 || reflInd == ind2) continue;

      Core::LinAlg::Matrix<3, 1> point_cord(Core::LinAlg::Initialization::zero);
      std::shared_ptr<Cut::Position> pos =
          Cut::Position::create(tri_coord, point_cord, Core::FE::CellType::tri3);
      // precise computation if it is inside
      bool is_inside = pos->compute(0.0);
      if (is_inside)
      {
        // another check if it is not the same point as triangle's vertices
        if (ptlist_[reflInd] != ptlist_[ind0] and ptlist_[reflInd] != ptlist_[i] and
            ptlist_[reflInd] != ptlist_[ind2])
        {
          isEar = false;
          break;
        }
      }
    }

    if (isEar)
    {
      filtered_ears.push_back(counter);
    }
    counter++;
  }


  if (filtered_ears.empty())
  {
    FOUR_C_THROW("Could find suitable ear for triangulation");
  }

  // now out of the filtered ears we try to find the "thinnest"
  return *std::min_element(filtered_ears.begin(), filtered_ears.end(),
      [&](const int& a, const int& b)
      {
        std::vector<Point*> tri_a = ears[a].first;
        const double tri_ear_head_proximity_a = Cut::distance_between_points(tri_a[1], tri_a[0]) +
                                                Cut::distance_between_points(tri_a[1], tri_a[2]);

        std::vector<Point*> tri_b = ears[b].first;
        const double tri_ear_head_proximity_b = Cut::distance_between_points(tri_b[1], tri_b[0]) +
                                                Cut::distance_between_points(tri_b[1], tri_b[2]);

        return tri_ear_head_proximity_a < tri_ear_head_proximity_b;
      });
}

/*-------------------------------------------------------------------------------------------------*
    Triangulation by ear clipping. Works for all cases, but costly.                 sudhakar 04/12
    Called when facets have two adjacent concave points
    During the process, if facet is free of adjacent concave points, splitanyfacet() is called
*--------------------------------------------------------------------------------------------------*/
void Cut::TriangulateFacet::ear_clipping(std::vector<int> ptConcavity,  // list of concave points
    bool triOnly,          // whether to create triangles only?
    bool DeleteInlinePts)  // how to deal with collinear points?
{
  std::vector<int> convex;

  // just form a Tri cell, and it is done
  if (ptlist_.size() == 3)
  {
    split_.push_back(ptlist_);
    return;
  }

  // creates only triangles --> do not call split_general_facet() even if possible
  // when ear clipping is called directly from other functions
  if (triOnly)
  {
    split_.clear();

    if (DeleteInlinePts)
    {
      Kernel::delete_inline_pts(ptlist_);
      if (ptlist_.size() == 3)  // after deleting the inline points, it may have only 3 points
      {
        split_.push_back(ptlist_);
        return;
      }
      if (ptlist_.size() < 3)
      {
        return;
      }
    }

    ptConcavity.clear();
    Cut::FacetShape geoType;

    ptConcavity = Kernel::check_convexity(ptlist_, geoType, true, DeleteInlinePts);
  }


  // to fix cases, with vertices on one line and no possibility of triangulation
  std::vector<Point*> last_added_ear;
  int last_added_ear_head = -1;

  while (true)
  {
    unsigned int split_size = split_.size();
    std::vector<int> reflex(ptConcavity);

    int polPts = ptlist_.size();
    convex.resize(polPts - reflex.size());

    // Find the convex points
    // They are non-reflex points of the polygon
    int conNum = 0;

    std::vector<std::pair<std::vector<Point*>, unsigned int>> discarded_ears;
    for (int i = 0; i < polPts; i++)
    {
      if (ptConcavity.size() > 0 && i == ptConcavity[0])
      {
        ptConcavity.erase(ptConcavity.begin());
        continue;
      }
      else
      {
        convex[conNum] = i;
        conNum++;
      }
    }

    bool haveinlinepts = false;
    if (not DeleteInlinePts)
    {
      haveinlinepts = Kernel::have_inline_pts(ptlist_);
    }

    // if (i) is an ear, the triangle formed by (i-1),i and (i+1) should be completely within the
    // polygon Find first ear point, and make the triangle and remove this ear from the polygon
    std::vector<Point*> tri(3);
    for (int i = 0; i < polPts; i++)
    {
      unsigned ind0 = i - 1;
      if (i == 0) ind0 = polPts - 1;
      unsigned ind2 = (i + 1) % polPts;

      if (not DeleteInlinePts and haveinlinepts)
      {
        // if we have inline points, one of the first or third point (ind0 or ind2) should be a
        // reflex point this should work as all inline points are handled like reflex points
        if (std::find(convex.begin(), convex.end(), ind0) != convex.end() and
            std::find(convex.begin(), convex.end(), ind2) != convex.end())
          continue;
      }

      // a reflex point cannot be a ear
      if (std::find(reflex.begin(), reflex.end(), i) != reflex.end()) continue;

      tri[0] = ptlist_[ind0];
      tri[1] = ptlist_[i];
      tri[2] = ptlist_[ind2];

      bool isEar = true;

      // check whether any point of polygon is inside this Tri
      // only reflex points are to be checked
      for (unsigned j = 0; j < reflex.size(); j++)
      {
        unsigned reflInd = reflex[j];
        if (reflInd == ind0 || reflInd == ind2) continue;

        if (Kernel::pt_inside_triangle(tri, ptlist_[reflInd], DeleteInlinePts))
        {
          if (ptlist_[reflInd] != ptlist_[ind0] and ptlist_[reflInd] != ptlist_[i] and
              ptlist_[reflInd] != ptlist_[ind2])
          {
            isEar = false;
            break;
          }
        }
      }

      // the only places where ear is discarded
      // if we  we discard all ears also, to process
      // them later on with the different algo
      if (!isEar)
      {
        discarded_ears.push_back(std::make_pair(tri, i));
        continue;
      }
      // last added ear is not needd
      last_added_ear = tri;
      last_added_ear_head = i;

      split_.push_back(tri);
      ptlist_.erase(
          ptlist_.begin() + i);  // the main pt of ear is removed, and new polygon is formed
      break;
    }

    if (ptlist_.size() < 3) FOUR_C_THROW("ear clipping produced 2 vertices polygon");

    if (DeleteInlinePts)
    {
      Kernel::delete_inline_pts(ptlist_);  // delete inline points in the new polygon
    }

    if (ptlist_.size() == 3)
    {
      split_.push_back(ptlist_);
      break;
    }

    else if (ptlist_.size() == 4 && !triOnly)
    {
      split4node_facet(ptlist_, false);
      break;
    }

    Cut::FacetShape str1;
    ptConcavity.clear();
    ptConcavity = Kernel::check_convexity(
        ptlist_, str1, true, DeleteInlinePts);  // concave points for the new polygon

    // one of the reason to fail is cases with triangles
    // with multiple points on the line,  we handle it
    // here
    if (split_size == split_.size())
    {
      if (discarded_ears.size() > 1)
      {
        unsigned int best_ear_index =
            (discarded_ears.size() == 1) ? 0 : find_second_best_ear(discarded_ears, reflex);

        std::vector<Point*> tri = discarded_ears[best_ear_index].first;
        unsigned int i = discarded_ears[best_ear_index].second;

        split_.push_back(tri);
        ptlist_.erase(ptlist_.begin() + i);

        if (ptlist_.size() == 3)
        {
          split_.push_back(ptlist_);
          break;
        }
        /// update triangle shape info
        ptConcavity.clear();
        Cut::FacetShape str1;
        ptConcavity = Kernel::check_convexity(
            ptlist_, str1, true, DeleteInlinePts);  // concave points for the new polygon
        last_added_ear = tri;
        last_added_ear_head = i;
        continue;
      }
      // either we wrongfully deleted an ear on iteration before
      // or haven't considered it in the current
      else
      {
        if (last_added_ear.size() != 0 or discarded_ears.size() != 0)
        {
          // recover last added ear
          if (discarded_ears.size() == 0) restore_last_ear(last_added_ear_head, ptConcavity);
          // use discarded one
          else
            last_added_ear_head = discarded_ears[0].second;

          // if last ear was convex, which it should be
          if (std::find(ptConcavity.begin(), ptConcavity.end(), last_added_ear_head) !=
              ptConcavity.end())
            FOUR_C_THROW("Unknown case!");

          const unsigned int split_start = last_added_ear_head;

          split_triangle_with_points_on_line(split_start);
          return;
        }
        else
        {
          if (convex.size() == 0)
            FOUR_C_THROW("Not sure how to split a triangle with all points on one line");
          else
            FOUR_C_THROW("Unknown case!");
        }
      }

      std::shared_ptr<BoundingBox> bb = std::shared_ptr<BoundingBox>(BoundingBox::create());
      std::cout << "The facet points are as follows\n";
      for (std::vector<Point*>::iterator it = ptlist_.begin(); it != ptlist_.end(); it++)
      {
        Point* pp = *it;
        const double* xp = pp->x();
        bb->add_point(xp);
        std::cout << xp[0] << " " << xp[1] << " " << xp[2] << "\n";
      }
      std::cout << "Bounding box details:\n";
      std::cout << "dx = " << fabs(bb->maxx() - bb->minx()) << "\n";
      std::cout << "dy = " << fabs(bb->maxy() - bb->miny()) << "\n";
      std::cout << "dz = " << fabs(bb->maxz() - bb->minz()) << "\n";
      FOUR_C_THROW("Ear clipping: no progress in the triangulation");
    }
  }
}


/*-----------------------------------------------------------------------------------------------------------*
    Ear Clipping is a triangulation method for simple polygons (convex, concave, with holes). wirtz
05/13 It is simple and robust but not very efficient (O(n^2)). As input parameter the outer polygon
(ptlist_) and the inner polygons (inlists_) are required. Triangles will be generated as output,
which are all combined in one vector (split_).
*------------------------------------------------------------------------------------------------------------*/
void Cut::TriangulateFacet::ear_clipping_with_holes(Side* parentside)
{
  while (inlists_.size() != 0)
  {
    // 1) Transformation in local coordinates
    std::map<int, Core::LinAlg::Matrix<3, 1>> localmaincyclepoints;
    std::map<std::vector<int>, Core::LinAlg::Matrix<3, 1>> localholecyclespoints;
    int j = 1;  // index for referencing a maincyclepoint
    std::vector<int> k(2);
    k[0] = 1;  // index for referencing a holecycle
    k[1] = 1;  // index for referencing a holecyclepoint
    for (std::vector<Point*>::iterator i = ptlist_.begin(); i != ptlist_.end(); ++i)
    {
      Point* maincyclepoint = *i;
      Core::LinAlg::Matrix<3, 1> maincyclepointcoordinates;
      Core::LinAlg::Matrix<3, 1> localmaincyclepointcoordinates;
      maincyclepoint->coordinates(maincyclepointcoordinates.data());
      parentside->local_coordinates(
          maincyclepointcoordinates, localmaincyclepointcoordinates, false);
      localmaincyclepoints[j] = localmaincyclepointcoordinates;
      j++;
    }
    for (std::list<std::vector<Point*>>::iterator i = inlists_.begin(); i != inlists_.end(); ++i)
    {
      std::vector<Point*> holecyclepoints = *i;
      for (std::vector<Point*>::iterator i = holecyclepoints.begin(); i != holecyclepoints.end();
          ++i)
      {
        Point* holecyclepoint = *i;
        Core::LinAlg::Matrix<3, 1> holecyclepointcoordinates;
        Core::LinAlg::Matrix<3, 1> localholecyclepointcoordinates;
        holecyclepoint->coordinates(holecyclepointcoordinates.data());
        parentside->local_coordinates(
            holecyclepointcoordinates, localholecyclepointcoordinates, false);
        localholecyclespoints[k] = localholecyclepointcoordinates;
        k[1] = k[1] + 1;
      }
      k[0] = k[0] + 1;
      k[1] = 1;
    }
    // 2) Holecyclepoint with the maximum r-value
    double maximumxvalue = -1;
    std::vector<int> maximumxvalueid(2);
    for (std::map<std::vector<int>, Core::LinAlg::Matrix<3, 1>>::iterator i =
             localholecyclespoints.begin();
        i != localholecyclespoints.end(); ++i)
    {
      Core::LinAlg::Matrix<3, 1> localholecyclespoint = i->second;
      if (localholecyclespoint(0, 0) > maximumxvalue)
      {
        maximumxvalue = localholecyclespoint(0, 0);
        maximumxvalueid = i->first;
      }
    }
    double correspondingyvalue = (localholecyclespoints[maximumxvalueid])(1, 0);
    Core::LinAlg::Matrix<3, 1> maximumpoint;
    maximumpoint(0, 0) = maximumxvalue;
    maximumpoint(1, 0) = correspondingyvalue;
    maximumpoint(2, 0) = 0;
    // 3) Closest visible point
    int maincyclesize = ptlist_.size();
    double closestedgexvalue = 2;
    std::vector<int> closestedgexvalueid(2);
    int intersectioncount = 0;  // counts the edges which are intersected by the ray to check
                                // whether the holecycle is inside or outside the maincycle
    bool intersectioninpoint =
        false;  // the counting with m is disturbed in case the ray hits a point of the maincycle
    for (int i = 1; i <= maincyclesize; ++i)
    {
      double edgepointxvalue1 = (localmaincyclepoints[i])(0, 0);
      double edgepointxvalue2 = (localmaincyclepoints[(i % maincyclesize) + 1])(0, 0);
      double edgepointyvalue1 = (localmaincyclepoints[i])(1, 0);
      double edgepointyvalue2 = (localmaincyclepoints[(i % maincyclesize) + 1])(1, 0);
      if (((edgepointyvalue1 <= correspondingyvalue) and
              (edgepointyvalue2 >= correspondingyvalue)) or
          ((edgepointyvalue2 <= correspondingyvalue) and (edgepointyvalue1 >= correspondingyvalue)))
      {
        double edgepointxvalue =
            ((edgepointxvalue2 - edgepointxvalue1) * correspondingyvalue +
                edgepointxvalue1 * edgepointyvalue2 - edgepointxvalue2 * edgepointyvalue1) /
            (edgepointyvalue2 - edgepointyvalue1);
        if (edgepointxvalue > maximumxvalue)
        {
          if (edgepointxvalue < closestedgexvalue)
          {
            closestedgexvalue = edgepointxvalue;
            closestedgexvalueid[0] = i;
            closestedgexvalueid[1] = (i % maincyclesize) + 1;
          }
          intersectioncount++;
          // need to scale POSITIONTOL here with InfNorm of Position (atm just works stable for
          // magnitued 1)
          if ((abs(edgepointxvalue - edgepointxvalue1) < POSITIONTOL and
                  abs(correspondingyvalue - edgepointyvalue1) < POSITIONTOL) or
              (abs(edgepointxvalue - edgepointxvalue2) < POSITIONTOL and
                  abs(correspondingyvalue - edgepointyvalue2) < POSITIONTOL))
          {
            intersectioninpoint = true;
          }
        }
      }
    }
    double epsilon = POSITIONTOL;  //??
    while (intersectioninpoint)
    {
      intersectioninpoint = false;
      for (int i = 1; i <= maincyclesize; ++i)
      {
        double edgepointxvalue1 = (localmaincyclepoints[i])(0, 0);
        double edgepointxvalue2 = (localmaincyclepoints[(i % maincyclesize) + 1])(0, 0);
        double edgepointyvalue1 = (localmaincyclepoints[i])(1, 0);
        double edgepointyvalue2 = (localmaincyclepoints[(i % maincyclesize) + 1])(1, 0);
        double A = edgepointxvalue1 - edgepointxvalue2;
        double B = edgepointyvalue1 - edgepointyvalue2;
        double C = edgepointxvalue1 + edgepointxvalue2 - 2 * maximumxvalue - 2;
        double D = edgepointyvalue1 + edgepointyvalue2 - 2 * correspondingyvalue - epsilon;
        double N = 2 * B - epsilon * A;
        if (abs(N) > POSITIONTOL)  // Tolerance scaling?
        {
          double eta = (B * C - A * D) / N;
          double xsi = (2 * D - epsilon * C) / N;
          if (eta < 1 and eta > -1 and xsi < 1 and xsi > -1)
          {
            intersectioncount++;
            double xlocalcoord = maximumxvalue + 1 + eta;
            double ylocalcoord = (2 * correspondingyvalue + epsilon + epsilon * eta) / 2;
            // need to scale POSITIONTOL here with InfNorm of Position (atm just works stable for
            // magnitued 1)
            if ((abs(xlocalcoord - edgepointxvalue1) < POSITIONTOL and
                    abs(ylocalcoord - edgepointyvalue1) < POSITIONTOL) or
                (abs(xlocalcoord - edgepointxvalue2) < POSITIONTOL and
                    abs(ylocalcoord - edgepointyvalue2) < POSITIONTOL))
            {
              intersectioninpoint = true;
              intersectioncount = 0;
              epsilon += POSITIONTOL;  //?
              break;
            }
          }
        }
      }
    }
    if (intersectioncount % 2 == 0)
    {
      int l = 1;  // index for searching the maximumxvalueid
      std::list<std::vector<Point*>>::iterator iter;
      for (std::list<std::vector<Point*>>::iterator i = inlists_.begin(); i != inlists_.end(); ++i)
      {
        if (l == maximumxvalueid[0])
        {
          iter = i;
        }
        l++;
      }
      inlists_.erase(iter);
      continue;
    }
    Core::LinAlg::Matrix<3, 1> closestpoint;
    closestpoint(0, 0) = closestedgexvalue;
    closestpoint(1, 0) = correspondingyvalue;
    closestpoint(2, 0) = 0;
    // 4) Closest visible point is node
    int mutuallyvisiblepointid = 0;
    // need to scale POSITIONTOL here with InfNorm of Position (atm just works stable for magnitued
    // 1)
    if ((abs(closestpoint(0, 0) - localmaincyclepoints[closestedgexvalueid[0]](0, 0)) <
            POSITIONTOL) and
        (abs(closestpoint(1, 0) - localmaincyclepoints[closestedgexvalueid[0]](1, 0)) <
            POSITIONTOL))
    {
      mutuallyvisiblepointid = closestedgexvalueid[0];
    }
    else if ((abs(closestpoint(0, 0) - localmaincyclepoints[closestedgexvalueid[1]](0, 0)) <
                 POSITIONTOL) and
             (abs(closestpoint(1, 0) - localmaincyclepoints[closestedgexvalueid[1]](1, 0)) <
                 POSITIONTOL))
    {
      mutuallyvisiblepointid = closestedgexvalueid[1];
    }
    else
    {
      // 5) Closest edge point with the maximum x value
      int potentuallyvisiblepointid;
      Core::LinAlg::Matrix<3, 1> potentuallyvisiblepoint;
      if (localmaincyclepoints[closestedgexvalueid[0]](0, 0) >
          localmaincyclepoints[closestedgexvalueid[1]](0, 0))
      {
        potentuallyvisiblepointid = closestedgexvalueid[0];
        potentuallyvisiblepoint(0, 0) = localmaincyclepoints[closestedgexvalueid[0]](0, 0);
        potentuallyvisiblepoint(1, 0) = localmaincyclepoints[closestedgexvalueid[0]](1, 0);
        potentuallyvisiblepoint(2, 0) = 0;
      }
      else
      {
        potentuallyvisiblepointid = closestedgexvalueid[1];
        potentuallyvisiblepoint(0, 0) = localmaincyclepoints[closestedgexvalueid[1]](0, 0);
        potentuallyvisiblepoint(1, 0) = localmaincyclepoints[closestedgexvalueid[1]](1, 0);
        potentuallyvisiblepoint(2, 0) = 0;
      }
      // 6) Reflex points in triangle
      Cut::FacetShape geoType;
      std::vector<int> reflexmaincyclepointids =
          Kernel::check_convexity(ptlist_, geoType, false, false);
      for (std::vector<int>::iterator i = reflexmaincyclepointids.begin();
          i != reflexmaincyclepointids.end(); ++i)
      {
        int* reflexmaincyclepointid = &*i;
        (*reflexmaincyclepointid)++;  // a little inconsistency here with Kernel::CheckConvexity
      }
      std::vector<int> insidemaincyclepointids;
      Core::LinAlg::Matrix<3, 3> triangle;
      for (int i = 0; i < 3; ++i)
      {
        triangle(i, 0) = maximumpoint(i, 0);
        triangle(i, 1) = closestpoint(i, 0);
        triangle(i, 2) = potentuallyvisiblepoint(i, 0);
      }
      for (std::vector<int>::iterator i = reflexmaincyclepointids.begin();
          i != reflexmaincyclepointids.end(); ++i)
      {
        int reflexmaincyclepointid = *i;
        Core::LinAlg::Matrix<3, 1> reflexmaincyclepoint =
            localmaincyclepoints[reflexmaincyclepointid];
        std::shared_ptr<Position> pos =
            Cut::Position::create(triangle, reflexmaincyclepoint, Core::FE::CellType::tri3);
        bool within = pos->is_given_point_within_element();
        if (within)
        {
          insidemaincyclepointids.push_back(reflexmaincyclepointid);
        }
      }
      if (insidemaincyclepointids.size() == 0)
      {
        mutuallyvisiblepointid = potentuallyvisiblepointid;
      }
      else
      {
        // 7) Closest angle
        double closestangle = 4;
        std::vector<int> closestanglemaincyclepointids;
        for (std::vector<int>::iterator i = insidemaincyclepointids.begin();
            i != insidemaincyclepointids.end(); ++i)
        {
          int insidemaincyclepointid = *i;
          Core::LinAlg::Matrix<3, 1> insidemaincyclepoint =
              localmaincyclepoints[insidemaincyclepointid];
          double distance1 = sqrt(pow((closestpoint(0, 0) - maximumpoint(0, 0)), 2) +
                                  pow((closestpoint(1, 0) - maximumpoint(1, 0)), 2));
          double distance2 = sqrt(pow((insidemaincyclepoint(0, 0) - maximumpoint(0, 0)), 2) +
                                  pow((insidemaincyclepoint(1, 0) - maximumpoint(1, 0)), 2));
          double distance3 = sqrt(pow((insidemaincyclepoint(0, 0) - closestpoint(0, 0)), 2) +
                                  pow((insidemaincyclepoint(1, 0) - closestpoint(1, 0)), 2));
          if ((distance1 > 1.0e-8) && (distance2 > 1.0e-8))
          {
            double alpha =
                acos((distance1 * distance1 + distance2 * distance2 - distance3 * distance3) /
                     (2 * distance1 * distance2));
            if (alpha - closestangle <= 1.0e-16)
            {
              closestangle = alpha;
              closestanglemaincyclepointids.push_back(insidemaincyclepointid);
            }
          }
        }
        if (closestanglemaincyclepointids.size() == 1)
        {
          mutuallyvisiblepointid = closestanglemaincyclepointids[0];
        }
        else
        {
          // 8) Mutually visible point
          double closestdistance = 5;
          for (std::vector<int>::iterator i = closestanglemaincyclepointids.begin();
              i != closestanglemaincyclepointids.end(); ++i)
          {
            int closestanglemaincyclepointid = *i;
            Core::LinAlg::Matrix<3, 1> closestanglemaincyclepoint =
                localmaincyclepoints[closestanglemaincyclepointid];
            double distance = sqrt(pow((closestanglemaincyclepoint(0, 0) - maximumpoint(0, 0)), 2) +
                                   pow((closestanglemaincyclepoint(1, 0) - maximumpoint(1, 0)), 2));
            if (distance < closestdistance)
            {
              mutuallyvisiblepointid = closestanglemaincyclepointid;
            }
          }
        }
      }
    }
    // 9) Make new pointlist
    std::vector<Point*> pttemp(ptlist_);
    ptlist_.clear();
    for (int i = 0; i < mutuallyvisiblepointid; ++i)
    {
      ptlist_.push_back(pttemp[i]);
    }
    std::vector<Point*> inlist;
    int l = 1;
    std::list<std::vector<Point*>>::iterator iter;
    for (std::list<std::vector<Point*>>::iterator i = inlists_.begin(); i != inlists_.end(); ++i)
    {
      if (l == maximumxvalueid[0])
      {
        inlist = *i;
        iter = i;
        break;
      }
      l++;
    }
    for (unsigned int j = (maximumxvalueid[1] - 1); j < inlist.size(); ++j)
    {
      ptlist_.push_back(inlist[j]);
    }
    for (int j = 0; j < (maximumxvalueid[1]); ++j)
    {
      ptlist_.push_back(inlist[j]);
    }
    inlists_.erase(iter);
    for (unsigned int i = mutuallyvisiblepointid - 1; i < pttemp.size(); ++i)
    {
      ptlist_.push_back(pttemp[i]);
    }
  }
  std::vector<int> ptConcavity;
  this->ear_clipping(ptConcavity, true, false);
}

bool Cut::TriangulateFacet::hasequal_ptlist_inlist(
    std::vector<Point*> ptlist, std::vector<std::vector<Point*>> inlists)
{
  if (inlists.size() != 1)
    return false;
  else if (ptlist.size() != (inlists[0]).size())
    return false;
  else
  {
    for (auto ptlist_it = ptlist.begin(); ptlist_it != ptlist.end(); ++ptlist_it)
    {
      bool found = false;
      for (auto inlist_it = (inlists[0]).begin(); inlist_it != (inlists[0]).end(); ++inlist_it)
      {
        if ((*ptlist_it)->id() == (*inlist_it)->id())
        {
          found = true;
          break;
        }
      }
      if (!found) return false;
    }
    return true;
  }
}

FOUR_C_NAMESPACE_CLOSE
