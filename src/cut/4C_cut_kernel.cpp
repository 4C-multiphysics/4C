// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_kernel.hpp"

#include "4C_cut_point.hpp"
#include "4C_cut_position.hpp"
#include "4C_io_pstream.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <iostream>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
unsigned Cut::Kernel::find_next_corner_point(const std::vector<Point*>& points,
    Core::LinAlg::Matrix<3, 1>& x1, Core::LinAlg::Matrix<3, 1>& x2, Core::LinAlg::Matrix<3, 1>& x3,
    Core::LinAlg::Matrix<3, 1>& b1, Core::LinAlg::Matrix<3, 1>& b2, Core::LinAlg::Matrix<3, 1>& b3,
    unsigned i)
{
  unsigned pointsize = points.size();
  unsigned j = (i + 1) % pointsize;
  if (pointsize < 3)
  {
    return j;
  }

  points[i]->coordinates(x1.data());
  points[j]->coordinates(x2.data());

  b1.update(1, x2, -1, x1, 0);

  double norm = b1.norm2();
  if (norm < std::numeric_limits<double>::min()) FOUR_C_THROW("same point in facet not supported");

  b1.scale(1. / norm);

  if (b1.norm2() < std::numeric_limits<double>::min())
    FOUR_C_THROW("same point in facet not supported");

  i = j;
  for (unsigned k = 2; k < pointsize; ++k)
  {
    i = (i + 1) % pointsize;
    Point* p = points[i];
    p->coordinates(x3.data());

    b2.update(1, x3, -1, x1, 0);

    norm = b2.norm2();
    if (norm < std::numeric_limits<double>::min())
      FOUR_C_THROW("same point in facet not supported");

    b2.scale(1. / norm);

    // cross product to get the normal at the point
    b3(0) = b1(1) * b2(2) - b1(2) * b2(1);
    b3(1) = b1(2) * b2(0) - b1(0) * b2(2);
    b3(2) = b1(0) * b2(1) - b1(1) * b2(0);

    if (b3.norm2() > PLANARTOL)
    {
      // Found. Return last node on this line.
      return (i + pointsize - 1) % pointsize;
    }
  }

  // All on one line. Return first and last point.
  if (j == 0)
  {
    return 0;
  }
  else
  {
    return pointsize - 1;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Kernel::is_valid_point1(const std::vector<Point*>& corner_points)
{
  return (corner_points.size() == 1);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Kernel::is_valid_line2(const std::vector<Point*>& corner_points)
{
  return (corner_points.size() == 2);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Kernel::is_valid_quad4(const std::vector<Point*>& points)
{
  if (points.size() == 4)
  {
    Core::LinAlg::Matrix<3, 3> xyze;
    Core::LinAlg::Matrix<3, 1> xyz;
    for (int i = 0; i < 4; ++i)
    {
      points[(i + 0) % 4]->coordinates(&xyze(0, 0));
      points[(i + 1) % 4]->coordinates(&xyze(0, 1));
      points[(i + 2) % 4]->coordinates(&xyze(0, 2));
      points[(i + 3) % 4]->coordinates(&xyz(0, 0));

      std::shared_ptr<Position> pos = Position::create(xyze, xyz, Core::FE::CellType::tri3);
      if (pos->compute())
      {
        return false;
      }
    }
    return true;
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Kernel::is_on_line(Point*& pt1, Point*& pt2, Point*& pt3, bool DeleteInlinePts)
{
  Core::LinAlg::Matrix<3, 1> x1, x2, x3;
  Core::LinAlg::Matrix<3, 1> pt1pt2, pt1pt3, cross;
  pt1->coordinates(x1.data());
  pt2->coordinates(x2.data());
  pt3->coordinates(x3.data());

  pt1pt2.update(1, x2, -1, x1, 0);
  pt1pt3.update(1, x3, -1, x1, 0);

  cross(0, 0) = pt1pt2(1, 0) * pt1pt3(2, 0) - pt1pt2(2, 0) * pt1pt3(1, 0);
  cross(1, 0) = pt1pt2(0, 0) * pt1pt3(2, 0) - pt1pt2(2, 0) * pt1pt3(0, 0);
  cross(2, 0) = pt1pt2(1, 0) * pt1pt3(0, 0) - pt1pt2(0, 0) * pt1pt3(1, 0);

  if (DeleteInlinePts)
  {
    // if the cross product is zero - on the same line
    // increasing this from 1e-10 to 1e-6 shown error in volume prediction
    if (cross.norm_inf() < TOL_POINTS_ON_LINE_FOR_DELETING) return true;
  }
  else
  {
    if (cross.norm_inf() < TOL_POINTS_ON_LINE) return true;
  }
  return false;
}

/*----------------------------------------------------------------------------*
  Check whether the list of points given forms a convex polygon
  If any 3 points fall along the line, this will delete the middle point and
  return new ptr. Initially the polygon is projected into the given plane
                                                                Sudhakar 04/12
 *----------------------------------------------------------------------------*/
std::vector<int> Cut::Kernel::check_convexity(const std::vector<Point*>& ptlist,
    Cut::FacetShape& geomType, bool InSplit, bool DeleteInlinePts)
{
  if (InSplit)  // if this function is called while performing facet splitting
  {
    if (ptlist.size() < 4)
    {
      std::cout << "ptlist.size(): " << ptlist.size() << "\n";
      FOUR_C_THROW("The number of points < 4. Is it called for appropriate facet?");
    }
  }

  std::string projPlane;

  if (DeleteInlinePts)
  {
    for (unsigned i = 0; i < ptlist.size(); i++)  // make sure there are no inline points
    {
      Point* pt2 = ptlist[i];
      Point* pt3 = ptlist[(i + 1) % ptlist.size()];
      unsigned ind = i - 1;
      if (i == 0) ind = ptlist.size() - 1;
      Point* pt1 = ptlist[ind];

      bool isline = is_on_line(pt1, pt2, pt3);
      if (isline)
      {
        Core::IO::cout << "the points are\n";
        for (unsigned i = 0; i < ptlist.size(); i++)
        {
          Point* ptx = ptlist[i];
          double coox[3];
          ptx->coordinates(coox);
          Core::IO::cout << coox[0] << "\t" << coox[1] << "\t" << coox[2] << "\n";
        }
        FOUR_C_THROW("Inline checking for facets not done before calling this");
      }
    }
  }

  bool isClockwise = is_clockwise_ordered_polygon(ptlist, projPlane);

  int ind1 = 0, ind2 = 0;
  if (projPlane == "x")
  {
    ind1 = 1;
    ind2 = 2;
  }
  else if (projPlane == "y")
  {
    ind1 = 2;
    ind2 = 0;
  }
  else if (projPlane == "z")
  {
    ind1 = 0;
    ind2 = 1;
  }
  else
    FOUR_C_THROW("unspecified projection type");

  Core::LinAlg::Matrix<3, 1> x1, x2, x3, xtemp;
  std::vector<int> leftind, rightind;
  std::vector<int> concPts;
  for (unsigned i = 0; i < ptlist.size(); i++)
  {
    Point* pt2 = ptlist[i];
    Point* pt3 = ptlist[(i + 1) % ptlist.size()];
    unsigned ind = i - 1;
    if (i == 0) ind = ptlist.size() - 1;
    Point* pt1 = ptlist[ind];

    pt1->coordinates(x1.data());
    pt2->coordinates(x2.data());
    pt3->coordinates(x3.data());

    xtemp.update(1.0, x2, -1.0, x1);

    double res = x3(ind1, 0) * xtemp(ind2, 0) - x3(ind2, 0) * xtemp(ind1, 0) +
                 xtemp(ind1, 0) * x1(ind2, 0) - xtemp(ind2, 0) * x1(ind1, 0);

    if (fabs(res) < TOL_POINTS_ON_LINE)  // this means small angled lines are just eliminated
    {
      if (not DeleteInlinePts)  // this means small angled lines are just concave
      {
        if (isClockwise)
        {
          leftind.push_back(i);
        }
        else
        {
          rightind.push_back(i);
        }
      }
      continue;
    }

    if (res < 0.0)
    {
      leftind.push_back(i);
    }
    else
    {
      rightind.push_back(i);
    }
  }

  if (leftind.size() == 0 or rightind.size() == 0)
    geomType = Cut::Convex;
  else if (leftind.size() == 1 or rightind.size() == 1)
    geomType = Cut::SinglePtConcave;
  else
    geomType = Cut::Concave;

  // if the points are ordered acw, right-turning points are concave points, and vice versa
  if (isClockwise) return leftind;
  return rightind;
}

/*-----------------------------------------------------------------------------------------------------*
            Find the equation of plane of the polygon defined by these facets
            Kernel::DeleteInlinePts() must be called before using this function
            This works only for simple polygons (not doubly connected, not self-intersecting)
                                                                                          Sudhakar
01/13
*------------------------------------------------------------------------------------------------------*/
/*std::vector<double> Cut::Kernel::EqnPlanePolygon( const std::vector<Point*>& ptlist,
bool DeleteInlinePts )
{
  std::vector<double> eqn_plane(4);
  if( ptlist.size() == 3 )
  {
    Point*p1 = ptlist[0];
    Point*p2 = ptlist[1];
    Point*p3 = ptlist[2];

    //eqn_plane = EqnPlane( ptlist[0], ptlist[1], ptlist[2] );
    eqn_plane = EqnPlane( p1, p2, p3 );
    return eqn_plane;
  }

  std::vector<int> concavePts;
  std::string geoType;
  concavePts = Kernel::CheckConvexity(  ptlist, geoType, false, DeleteInlinePts ); // find concave
points of the polygon

  // for finding equation of convex facet, any 3 points can be used
  if( concavePts.size() == 0 )
  {
    if ( DeleteInlinePts )
    {
      Point*p1 = ptlist[0];
      Point*p2 = ptlist[1];
      Point*p3 = ptlist[2];
      eqn_plane = EqnPlane( p1, p2, p3 );
    }
    else
    {
      std::vector<Point*> pttemp = ptlist;
      std::vector<Point*> preparedPoints = Get3NoncollinearPts( pttemp );
      eqn_plane = EqnPlane( preparedPoints[0], preparedPoints[1], preparedPoints[2] );
    }
    return eqn_plane;
  }

  // to find equation of plane for a concave facet we choose 3 adjacent points
  // if secondpt is a concave point, normal direction is not computed correctly
  unsigned ncross=0;
  unsigned npts = ptlist.size();
  bool eqndone=false;
  int firstPt=0,secondPt=0,thirdPt=0;

  for( unsigned i=0;i<npts;++i )
  {
    ncross++;

    int concNo = 0;

    if( i==0 )
    {
      firstPt = concavePts[concNo];
      secondPt = (firstPt+1)%npts;
    }
    else
    {
      firstPt = (firstPt+1)%npts;
      secondPt = (firstPt+1)%npts;
    }
    // check whether secondpt is a concave point
    if(std::find(concavePts.begin(), concavePts.end(), secondPt) != concavePts.end())
      continue;

    thirdPt = (secondPt+1)%npts;

    Point*p1 = ptlist[firstPt];
    Point*p2 = ptlist[secondPt];
    Point*p3 = ptlist[thirdPt];

    if ( not DeleteInlinePts )
    {
      if( IsOnLine( p1,p2,p3 ) )
      {
        continue;
      }
    }

    eqn_plane = EqnPlane( p1, p2, p3 );

    eqndone = true;
  }

  if( eqndone == false )
    FOUR_C_THROW("equation not computed");

  return eqn_plane;
}*/

/*-----------------------------------------------------------------------------------------------------*
            Find the equation of plane that contains these non-collinear points
            It must be noted while using this function to find equation of facets,
             none of these 3 points must be a reflex (concave) point
                                                                                          Sudhakar
04/12
*------------------------------------------------------------------------------------------------------*/
std::vector<double> Cut::Kernel::eqn_plane(Point*& pt1, Point*& pt2, Point*& pt3)
{
  bool collinear = is_on_line(pt1, pt2, pt3);
  if (collinear) FOUR_C_THROW(" 3 points lie on a line. Eqn of plane cannot be computed");

  std::vector<double> eqn_plane(4);
  double x1[3], x2[3], x3[3];

  pt1->coordinates(x1);
  pt2->coordinates(x2);
  pt3->coordinates(x3);

  eqn_plane[0] = x1[1] * (x2[2] - x3[2]) + x2[1] * (x3[2] - x1[2]) + x3[1] * (x1[2] - x2[2]);
  eqn_plane[1] = x1[2] * (x2[0] - x3[0]) + x2[2] * (x3[0] - x1[0]) + x3[2] * (x1[0] - x2[0]);
  eqn_plane[2] = x1[0] * (x2[1] - x3[1]) + x2[0] * (x3[1] - x1[1]) + x3[0] * (x1[1] - x2[1]);
  eqn_plane[3] = x1[0] * (x2[1] * x3[2] - x3[1] * x2[2]) + x2[0] * (x3[1] * x1[2] - x1[1] * x3[2]) +
                 x3[0] * (x1[1] * x2[2] - x2[1] * x1[2]);

  return eqn_plane;
}

/*------------------------------------------------------------------------------------------------------------*
           Newell's method of finding equation of plane of a polygon sudhakar 11/13 This method is
applicable for convex, concave, and polygons with in-line vertices Does not require deleting of any
vertices ---> so very general and robust
*-------------------------------------------------------------------------------------------------------------*/
std::vector<double> Cut::Kernel::eqn_plane_of_polygon(const std::vector<Point*>& ptlist)
{
  std::vector<std::vector<double>> vertices(ptlist.size());

  for (unsigned i = 0; i < ptlist.size(); i++)
  {
    Point* pt = ptlist[i];
    std::vector<double> coo(3);
    pt->coordinates(coo.data());
    vertices[i] = coo;
  }

  return eqn_plane_of_polygon(vertices);
}

/*------------------------------------------------------------------------------------------------------------*
           Newell's method of finding equation of plane of a polygon sudhakar 11/13 This method is
applicable for convex, concave, and polygons with in-line vertices Does not require deleting of any
vertices ---> so very general and robust
*-------------------------------------------------------------------------------------------------------------*/
std::vector<double> Cut::Kernel::eqn_plane_of_polygon(
    const std::vector<std::vector<double>>& vertices)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut::Kernel::EqnPlaneOfPolygon");

  // TODO: improvement by a factor of 4

  std::vector<double> eqn_plane(4, 0.0), middle(3, 0.0);

  unsigned num_vert = vertices.size();

  for (unsigned ii = 0; ii < num_vert; ii++)
  {
    const std::vector<double>& pt1 = vertices[ii];
    const std::vector<double>& pt2 = vertices[(ii + 1) % num_vert];

    eqn_plane[0] += (pt1[1] - pt2[1]) * (pt1[2] + pt2[2]);
    eqn_plane[1] += (pt1[2] - pt2[2]) * (pt1[0] + pt2[0]);
    eqn_plane[2] += (pt1[0] - pt2[0]) * (pt1[1] + pt2[1]);

    for (unsigned jj = 0; jj < 3; jj++) middle[jj] += pt1[jj];
  }

  for (unsigned jj = 0; jj < 3; jj++)
  {
    eqn_plane[3] += middle[jj] * eqn_plane[jj];
  }
  eqn_plane[3] /= num_vert;

  return eqn_plane;
}

/*------------------------------------------------------------------------------------------------------------*
           check whether the point "check" is inside the triangle formed by tri sudhakar 05/12 uses
barycentric coordinates as it is faster
*-------------------------------------------------------------------------------------------------------------*/
bool Cut::Kernel::pt_inside_triangle(std::vector<Point*> tri, Point* check, bool DeleteInlinePts)
{
  if (tri.size() != 3) FOUR_C_THROW("expecting a triangle");

  Core::LinAlg::Matrix<3, 1> t1;
  Core::LinAlg::Matrix<3, 1> t2;
  Core::LinAlg::Matrix<3, 1> t3;
  Core::LinAlg::Matrix<3, 1> pt;
  Core::LinAlg::Matrix<3, 1> v0(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<3, 1> v1(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<3, 1> v2(Core::LinAlg::Initialization::uninitialized);
  tri[0]->coordinates(t1.data());
  tri[1]->coordinates(t2.data());
  tri[2]->coordinates(t3.data());
  check->coordinates(pt.data());

  v0.update(1.0, t3, -1.0, t1);
  v1.update(1.0, t2, -1.0, t1);
  v2.update(1.0, pt, -1.0, t1);

  double dot00, dot01, dot02, dot11, dot12;
  dot00 = v0.dot(v0);
  dot01 = v0.dot(v1);
  dot02 = v0.dot(v2);
  dot11 = v1.dot(v1);
  dot12 = v1.dot(v2);

  double Det = dot00 * dot11 - dot01 * dot01;

  if (fabs(Det) < 1e-35)
  {
    if (DeleteInlinePts)
    {
      std::cout << "value of det = " << Det << "\n";
      std::cout << "triangle: "
                << "t1 " << t1 << "t2 " << t2 << "t3 " << t3 << std::endl;
      std::cout << "point " << pt << std::endl;
      FOUR_C_THROW("the triangle is actually on a line. Verify tolerances in cut_tolerance.H\n");
    }
    else
    {
      std::cout << "value of det = " << Det << "\n";
      std::cout
          << "WARNING: the triangle is actually on a line. Verify tolerances in cut_tolerance.H\n";
      return true;  // without deleting inlinepts, this can happen... it's not bad, but make sure
                    // that such a triangle is no ear
    }
  }

  double invDenom = 1.0 / Det;
  double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  // if u or v are very small, then set them to zero
  if (fabs(u) < 1e-35) u = 0.0;
  if (fabs(v) < 1e-35) v = 0.0;

  if ((u >= 0) && (v >= 0) && (u + v < 1)) return true;
  return false;
}

/*------------------------------------------------------------------------------------------------------------*
           Check whether the point "check" is inside the Quad formed by quad               sudhakar
07/12 Splits Quad into 2 Tri and perform the check on each
*-------------------------------------------------------------------------------------------------------------*/
bool Cut::Kernel::pt_inside_quad(std::vector<Point*> quad, Point* check)
{
  if (quad.size() != 4) FOUR_C_THROW("expecting a Quad");

  std::vector<int> concavePts;
  Cut::FacetShape str1;

  concavePts = check_convexity(quad, str1);

  int concsize = concavePts.size();
  if (concsize > 1)
  {
    Core::IO::cout << "The points of the failing Quad are\n";
    for (unsigned i = 0; i < 4; i++)
    {
      Point* pt = quad[i];
      double x[3];
      pt->coordinates(x);
      Core::IO::cout << x[0] << "\t" << x[1] << "\t" << x[2] << "\n";
    }
    FOUR_C_THROW("Quad has more than 1 concave pt --> Selfcut");
  }

  int indStart = 0;
  if (concsize == 1) indStart = concavePts[0];


  std::vector<Point*> tri1(3), tri2(3);

  tri1[0] = quad[indStart];
  tri1[1] = quad[(indStart + 1) % 4];
  tri1[2] = quad[(indStart + 2) % 4];

  bool insideTri1 = pt_inside_triangle(tri1, check);
  if (insideTri1) return true;

  tri2[0] = quad[indStart];
  tri2[1] = quad[(indStart + 2) % 4];
  tri2[2] = quad[(indStart + 3) % 4];

  bool insideTri2 = pt_inside_triangle(tri2, check);
  if (insideTri2) return true;

  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Kernel::is_clockwise_ordered_polygon(
    std::vector<Point*> polyPoints, std::string& projPlane)
{
  if (polyPoints.size() < 3) FOUR_C_THROW("polygon with less than 3 corner points");

  std::vector<double> eqn;

  eqn = eqn_plane_of_polygon(polyPoints);

  // projection on the plane which has max normal component - reduce round off error
  find_projection_plane(projPlane, eqn);

  int ind1 = 0, ind2 = 0;
  if (projPlane == "x")
  {
    ind1 = 1;
    ind2 = 2;
  }
  else if (projPlane == "y")
  {
    ind1 = 2;
    ind2 = 0;
  }
  else if (projPlane == "z")
  {
    ind1 = 0;
    ind2 = 1;
  }

  int numpts = polyPoints.size();
  double crossProd = 0.0;
  for (int i = 0; i < numpts; i++)
  {
    double pt1[3], pt2[3];

    polyPoints[i]->coordinates(pt1);
    polyPoints[(i + 1) % numpts]->coordinates(pt2);

    crossProd += (pt2[ind1] - pt1[ind1]) * (pt2[ind2] + pt1[ind2]);
  }

  if (crossProd > 0.0) return true;
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Kernel::find_projection_plane(std::string& projPlane, const std::vector<double>& eqn)
{
  if (fabs(eqn[0]) > fabs(eqn[1]) && fabs(eqn[0]) > fabs(eqn[2]))
    projPlane = "x";
  else if (fabs(eqn[1]) > fabs(eqn[2]) && fabs(eqn[1]) > fabs(eqn[0]))
    projPlane = "y";
  else
  {
    if (fabs(eqn[0]) == fabs(eqn[1]))
    {
      if (fabs(eqn[0]) > fabs(eqn[2]))
        projPlane = "x";
      else
        projPlane = "z";
    }
    else if (fabs(eqn[1]) == fabs(eqn[2]))
    {
      if (fabs(eqn[1]) > fabs(eqn[0]))
        projPlane = "y";
      else
        projPlane = "x";
    }
    else if (fabs(eqn[0]) == fabs(eqn[2]))
    {
      if (fabs(eqn[0]) > fabs(eqn[1]))
        projPlane = "z";
      else
        projPlane = "y";
    }
    else
    {
      projPlane = "z";
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Kernel::delete_inline_pts(std::vector<Point*>& poly)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::Kernel::DeleteInlinePts" );


  bool anyInLine = false;
  unsigned num = poly.size();

  for (unsigned i = 0; i < num; i++)
  {
    Point* pt1 = poly[i];
    Point* pt2 = poly[(i + 1) % num];  // next point
    unsigned ind = i - 1;
    if (i == 0) ind = num - 1;
    Point* pt3 = poly[ind];  // previous point

    anyInLine = is_on_line(pt3, pt1, pt2, true);

    if (anyInLine)
    {
      // iterator of the point to be deleted
      std::vector<Point*>::iterator delPt = poly.begin() + i;
      poly.erase(delPt);
      break;
    }
  }
  /* this makes sure the procedure is repeated until all the inline points of
   * the facet are deleted */
  if (anyInLine) delete_inline_pts(poly);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Kernel::have_inline_pts(std::vector<Point*>& poly)
{
  unsigned num = poly.size();
  for (unsigned i = 0; i < num; i++)
  {
    Point* pt1 = poly[i];
    Point* pt2 = poly[(i + 1) % num];  // next point
    unsigned ind = i - 1;
    if (i == 0) ind = num - 1;
    Point* pt3 = poly[ind];  // previous point
    if (is_on_line(pt3, pt1, pt2))
    {
      return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::vector<Cut::Point*> Cut::Kernel::get3_noncollinear_pts(std::vector<Point*>& polyPoints)
{
  std::vector<Point*> preparedPoints;
  for (unsigned i = 0; i < polyPoints.size(); i++)
  {
    Point* pt2 = polyPoints[i];
    Point* pt3 = polyPoints[(i + 1) % polyPoints.size()];
    unsigned ind = i - 1;
    if (i == 0) ind = polyPoints.size() - 1;
    Point* pt1 = polyPoints[ind];
    bool collinear = is_on_line(pt1, pt2, pt3);
    if (collinear)
    {
      continue;
    }
    preparedPoints.push_back(pt1);
    preparedPoints.push_back(pt2);
    preparedPoints.push_back(pt3);
    return preparedPoints;
  }
  FOUR_C_THROW("case with inline points: all points collinear");
  return preparedPoints;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Cut::Kernel::get_area_tri(
    const std::vector<Point*>& poly, Core::LinAlg::Matrix<3, 1>* normalvec)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::Kernel::getAreaTri" );

  if (poly.size() != 3) FOUR_C_THROW("expecting a triangle");

  double p0[3] = {0.0, 0.0, 0.0}, p1[3] = {0.0, 0.0, 0.0}, p2[3] = {0.0, 0.0, 0.0};
  poly[0]->coordinates(p0);
  poly[1]->coordinates(p1);
  poly[2]->coordinates(p2);

  return get_area_tri(p0, p1, p2, normalvec);
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
double Cut::Kernel::get_area_tri(const double* p0_ptr, const double* p1_ptr, const double* p2_ptr,
    Core::LinAlg::Matrix<3, 1>* normalvec)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::Kernel::getAreaTri" );

  // create planes consisting of 3 nodes each
  const Core::LinAlg::Matrix<3, 1> p0(p0_ptr, true);
  const Core::LinAlg::Matrix<3, 1> p1(p1_ptr, true);
  const Core::LinAlg::Matrix<3, 1> p2(p2_ptr, true);

  Core::LinAlg::Matrix<3, 1> v01;
  Core::LinAlg::Matrix<3, 1> v02;

  v01.update(1, p1, -1, p0, 0);
  v02.update(1, p2, -1, p0, 0);

  double doubleareacrossprod;
  // Cross product way:
  if (normalvec == nullptr)
  {
    Core::LinAlg::Matrix<3, 1> crossprod;
    crossprod.cross_product(v01, v02);  // Cross prod between two vectors of the triangle
    doubleareacrossprod = crossprod.norm2();
  }
  else
  {
    normalvec->cross_product(v01, v02);  // Cross prod between two vectors of the triangle
    doubleareacrossprod = normalvec->norm2();
    normalvec->scale(1.0 / doubleareacrossprod);  // Scale to unit normal
  }

#ifdef FOUR_C_ENABLE_ASSERTIONS
  Core::LinAlg::Matrix<3, 1> v12;
  v12.update(1, p1, -1, p2, 0);

  double areacrossprod = 0.5 * doubleareacrossprod;
  // Cross referencing!
  Core::LinAlg::Matrix<3, 1> crossprod2;
  crossprod2.cross_product(v01, v12);
  double areacrossprod2 = 0.5 * crossprod2.norm2();

  Core::LinAlg::Matrix<3, 1> crossprod3;
  crossprod3.cross_product(v02, v12);
  double areacrossprod3 = 0.5 * crossprod3.norm2();

  if (areacrossprod > REF_AREA_BCELL)
  {
    if (fabs(areacrossprod - areacrossprod2) > REF_AREA_BCELL or
        fabs(areacrossprod - areacrossprod3) > REF_AREA_BCELL)
    {
      std::cout << "     SIGNIFICANT DIFFERENCE IN AREA!!!!!!!       " << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
      std::cout << "    areacrossprod: " << areacrossprod << std::endl;
      std::cout << "    areacrossprod2: " << areacrossprod2 << std::endl;
      std::cout << "    areacrossprod3: " << areacrossprod3 << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      std::cout << "p0:" << std::endl;
      std::cout << "p0_0=mpf('" << std::setprecision(32) << p0(0, 0) << "')" << std::endl;
      std::cout << "p0_1=mpf('" << std::setprecision(32) << p0(1, 0) << "')" << std::endl;
      std::cout << "p0_2=mpf('" << std::setprecision(32) << p0(2, 0) << "')" << std::endl;

      std::cout << "p1:" << std::endl;
      std::cout << "p1_0=mpf('" << std::setprecision(32) << p1(0, 0) << "')" << std::endl;
      std::cout << "p1_1=mpf('" << std::setprecision(32) << p1(1, 0) << "')" << std::endl;
      std::cout << "p1_2=mpf('" << std::setprecision(32) << p1(2, 0) << "')" << std::endl;

      std::cout << "p2:" << std::endl;
      std::cout << "p2_0=mpf('" << std::setprecision(32) << p2(0, 0) << "')" << std::endl;
      std::cout << "p2_1=mpf('" << std::setprecision(32) << p2(1, 0) << "')" << std::endl;
      std::cout << "p2_2=mpf('" << std::setprecision(32) << p2(2, 0) << "')" << std::endl;

      FOUR_C_THROW(
          "The area is not the same for the different cross product implementations. Something "
          "might be wrong with this TRI.");
    }
  }

#endif

  return 0.5 * doubleareacrossprod;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Cut::Kernel::get_area_convex_quad(std::vector<Point*>& poly)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::Kernel::getAreaConvexQuad" );

  if (poly.size() != 4) FOUR_C_THROW("expecting a quad");

  std::vector<Point*> tri1, tri2;
  tri1.push_back(poly[0]);
  tri1.push_back(poly[1]);
  tri1.push_back(poly[2]);

  tri2.push_back(poly[0]);
  tri2.push_back(poly[2]);
  tri2.push_back(poly[3]);

  double area1 = get_area_tri(tri1);
  double area2 = get_area_tri(tri2);
  return (area1 + area2);
}



bool Cut::Kernel::close_to_zero(const double a) { return ((a < 1e-30) and (a > -1e-30)); };


bool Cut::Kernel::close_to_zero(const Core::CLN::ClnWrapper& a)
{
  cln::cl_F lpf = cln::least_positive_float(cln::float_format(a.Value()));
  cln::cl_F lnf = cln::least_negative_float(cln::float_format(a.Value()));
  return ((a < lpf) and (a > lnf));
};

// non templated static data from cut_kernel

#ifdef DEBUG_MEMORY_ALLOCATION

bool Cut::Kernel::AdaptiveKernelSharedData::custom_allocator_run_ = false;

bool Cut::Kernel::AdaptiveKernelSharedData::all_intersections_done_once_ = false;

bool Cut::Kernel::AdaptiveKernelSharedData::all_distance_done_once_ = false;

bool Cut::Kernel::AdaptiveKernelSharedData::all_position_done_once_ = true;

size_t Cut::Kernel::AdaptiveKernelSharedData::cln_byte_size_[] = {48, 56, 56, 64};

std::unordered_map<size_t, int> Cut::Kernel::AdaptiveKernelSharedData::memory_allocations_;

std::map<size_t, int> Cut::Kernel::AdaptiveKernelSharedData::memory_allocations_intersection_;

std::map<size_t, int> Cut::Kernel::AdaptiveKernelSharedData::memory_allocations_position_;

std::map<size_t, int> Cut::Kernel::AdaptiveKernelSharedData::memory_allocations_distance_;

#endif

FOUR_C_NAMESPACE_CLOSE
