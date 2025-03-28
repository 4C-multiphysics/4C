// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_boundarycell.hpp"
#include "4C_cut_kernel.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_triangulateFacet.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_linalg_gauss.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Facet::Facet(Mesh& mesh, const std::vector<Point*>& points, Side* side, bool cutsurface)
    : points_(points),
      parentside_(side),
      planar_(false),
      planar_known_(false),
      position_(cutsurface ? Point::oncutsurface : Point::undecided),
      is_planar_computed_(false)
{
  find_corner_points();

  if (cutsurface)
  {
    for (std::vector<Point*>::const_iterator i = points.begin(); i != points.end(); ++i)
    {
      Point* p = *i;
      p->position(Point::oncutsurface);
    }
  }
  else
  {
    // On multiple cuts there are facets on element sides that belong to an
    // old cut surface. Thus if all nodes are on a cut surface, the facet is
    // as well.
    bool allonsurface = true;
    for (std::vector<Point*>::const_iterator i = points.begin(); i != points.end(); ++i)
    {
      Point* p = *i;
      if (p->position() != Point::oncutsurface)
      {
        allonsurface = false;
        break;
      }
    }
    if (allonsurface)
    {
      // If my side has an id this facet is actually on a cut
      // surface. Otherwise it could be an inside or outside facet. The actual
      // decision does not matter much.
      if (on_cut_side())
      {
        position_ = Point::oncutsurface;
      }
      else
      {
        position_ = Point::undecided;
      }
    }
  }
  for (std::vector<Point*>::const_iterator i = points.begin(); i != points.end(); ++i)
  {
    Point* p = *i;
    p->register_entity(this);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::register_entity(VolumeCell* cell)
{
  cells_.insert(cell);
  if (cells_.size() > 2)
  {
    this->print();
    for (plain_volumecell_set::const_iterator ic = cells_.begin(); ic != cells_.end(); ++ic)
    {
      std::cout << "\n\nVolumeCell " << *ic << std::endl;
      (*ic)->print(std::cout);
    }

    // write details of volume cells
    Output::gmsh_volume_cells_only(cells_);

    std::ostringstream ostr;
    ostr << "Too many volume cells at facet! ( num cells = " << cells_.size() << " )";
    FOUR_C_THROW("{}", ostr.str());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::disconnect_volume(VolumeCell* cell) { cells_.erase(cell); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::on_cut_side() const { return parentside_->is_cut_side(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::on_boundary_cell_side() const { return parentside_->is_boundary_cell_side(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::on_marked_background_side() const
{
  return parentside_->is_marked_background_side();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Cut::Facet::side_id() const { return parentside_->id(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::coordinates(double* x)
{
  for (std::vector<Point*>::const_iterator i = points_.begin(); i != points_.end(); ++i)
  {
    Point* p = *i;
    p->coordinates(x);
    x += 3;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::corner_coordinates(double* x)
{
  find_corner_points();
  for (std::vector<Point*>::const_iterator i = corner_points_.begin(); i != corner_points_.end();
      ++i)
  {
    Point* p = *i;
    p->coordinates(x);
    x += 3;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::get_all_points(Mesh& mesh, PointSet& cut_points, bool dotriangulate)
{
  if (is_planar(mesh, dotriangulate))
  {
    std::copy(points_.begin(), points_.end(), std::inserter(cut_points, cut_points.begin()));
    for (plain_facet_set::iterator i = holes_.begin(); i != holes_.end(); ++i)
    {
      Facet* h = *i;
      h->get_all_points(mesh, cut_points);
    }
  }
  else
  {
    for (std::vector<std::vector<Point*>>::iterator i = triangulation_.begin();
        i != triangulation_.end(); ++i)
    {
      std::vector<Point*>& tri = *i;
      std::copy(tri.begin(), tri.end(), std::inserter(cut_points, cut_points.begin()));
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::add_hole(Facet* hole)
{
  double dot = 1.0;
  while (dot > 0)
  {
    std::vector<double> eqn_plane_hole = Kernel::eqn_plane_of_polygon(hole->points_);
    std::vector<double> eqn_plane_facet = Kernel::eqn_plane_of_polygon(corner_points_);
    dot = 0.0;
    for (unsigned int dim = 0; dim < 3; ++dim) dot += eqn_plane_hole[dim] * eqn_plane_facet[dim];
    if (dot >
        0)  // if this happens we get problems with the ear clipping algorithm for triangulation
    {
      std::reverse(hole->corner_points_.begin(), hole->corner_points_.end());
      std::reverse(hole->points_.begin(), hole->points_.end());
      std::cout << "WARNING:: Facet::AddHole: points_ and hole->points_ are not aligned correct "
                   "(other direction)! "
                << std::endl;
    }
  }

  for (std::vector<Point*>::iterator i = hole->points_.begin(); i != hole->points_.end(); ++i)
  {
    Point* p = *i;
    p->register_entity(this);
  }
  holes_.insert(hole);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_planar(Mesh& mesh, bool dotriangulate)
{
  if (dotriangulate)
  {
    // if ( not IsTriangulated() and points_.size() > 3 )
    if (not is_triangulated() and corner_points_.size() > 3)
    {
      create_triangulation(mesh, points_);
      planar_known_ = true;
      planar_ = false;
    }
  }

  if (not planar_known_)
  {
    planar_ = is_planar(mesh, points_);
    planar_known_ = true;
  }

  return planar_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_planar(Mesh& mesh, const std::vector<Point*>& points)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::Facet::is_planar" );

  if (is_planar_computed_) return is_planar_;


  Core::LinAlg::Matrix<3, 1> x1;
  Core::LinAlg::Matrix<3, 1> x2;
  Core::LinAlg::Matrix<3, 1> x3;

  Core::LinAlg::Matrix<3, 1> b1;
  Core::LinAlg::Matrix<3, 1> b2;
  Core::LinAlg::Matrix<3, 1> b3;

  unsigned i = normal(points, x1, x2, x3, b1, b2, b3);
  if (i == 0)  // all on one line is ok
  {
    is_planar_computed_ = true;
    is_planar_ = true;
    return true;
  }

  Core::LinAlg::Matrix<3, 3> A;
  std::copy(b1.data(), b1.data() + 3, A.data());
  std::copy(b2.data(), b2.data() + 3, A.data() + 3);
  std::copy(b3.data(), b3.data() + 3, A.data() + 6);

  // std::copy( points.begin(), points.end(), std::ostream_iterator<Point*>( std::cout, "; " ) );
  // std::cout << "\n";

  for (++i; i < points.size(); ++i)
  {
    Point* p = points[i];
    p->coordinates(x3.data());

    x3.update(-1, x1, 1);

    Core::LinAlg::Matrix<3, 3> B;
    B = A;
    x2 = 0;
    double det = Core::LinAlg::gauss_elimination<true, 3, double>(B, x3, x2);
    if (fabs(det) < LINSOLVETOL)
    {
      FOUR_C_THROW("failed to find point position");
    }

    if (fabs(x2(2)) > PLANARTOL)
    {
      // there is one point that is not within the plain

      create_triangulation(mesh, points);

      is_planar_computed_ = true;
      is_planar_ = false;
      return false;
    }
  }
  is_planar_computed_ = true;
  is_planar_ = true;
  return true;
}

/*----------------------------------------------------------------------------*
 * Only used in tetmesh debugging!!
 *----------------------------------------------------------------------------*/
bool Cut::Facet::share_same_cut_side(Facet* f) /* Does the facet share the same cut-side? */
{
  if (this->parentside_->id() == f->parent_side()->id())
  {
    return true;
  }
  return false;
}

/*----------------------------------------------------------------------------*
  Perform triangulation of the facet.                            sudhakar 06/15
  When convex, centre-point-triangulation is used that can better approximate
  the geometry. For concave facets, Ear-clipping procedure is utilized --
  slightly less accurate due to deleting the inline points but no other choice
*-----------------------------------------------------------------------------*/
void Cut::Facet::create_triangulation(Mesh& mesh, const std::vector<Point*>& points)
{
  // Perform centre-point triangulation
  // Find the middle point (M) and join them with the corners of the facet to create triangles
  // Works only for convex facets, because of the middle point calculation
  //                                                                  |
  //                                       4 _______ 3                |
  //                                        /.     .\                 |
  //                                       /  .   .  \                |
  //                                      /    . .    \               |
  //                                  5  /      . M    \ 2            |
  //                                     \     . .     /              |
  //                                      \   .   .   /               |
  //                                       \._______./                |
  //                                      0          1                |
  //

  // TODO: We force the facets created from levelset side to use centre-pt-tri
  // This is not completely correct but since we use a single levelset side for each element
  // (even though the levelset can break into two), we get some weird facet shapes.
  // This procedure seems to give better result even if the facet is concave in such cases
  // Also we assume that the levelset facet never contains a hole --> May be in future changes?
  Cut::FacetShape geoType;
  std::vector<int> concave_ids = Kernel::check_convexity(corner_points_, geoType, false, false);
  if (((geoType == Cut::Convex) or belongs_to_level_set_side()) and (not has_holes()))
  {
    std::vector<Point*> pts(points);
    // Find the middle point
    Core::LinAlg::Matrix<3, 1> cur;
    Core::LinAlg::Matrix<3, 1> avg(Core::LinAlg::Initialization::zero);  // fill with zeros
    for (std::vector<Point*>::iterator i = pts.begin(); i != pts.end(); i++)
    {
      Point* p1 = *i;
      p1->coordinates(cur.data());
      avg.update(1.0, cur, 1.0);
    }
    avg.scale(1.0 / pts.size());
    // One could create a better approx for the "midpoint" here parsing information from the
    // levelset
    //  and thus finding the zero level set easier. Might be worthwhile testing.
    Point* p_mid = mesh.new_point(avg.data(), nullptr, parent_side(),
        0.0);  // change tolerance here intelligently !!! - basically
               // there is no reason why I'd like to merge here!
    p_mid->position(position());
    p_mid->register_entity(this);

    // form triangles and store them in triangulation_
    triangulation_.clear();
    std::vector<Point*> newtri(3);
    newtri[0] = p_mid;
    for (int i = 0; i < (int)pts.size(); i++)
    {
      Point* pt1 = pts[i];
      Point* pt2 = pts[(i + 1) % pts.size()];

      // For LevelSet, make sure to orient a concave point on the 2nd entry of the triangulation.
      //  This assumption is necessary for Direct Divergence to work correctly.
      if (!belongs_to_level_set_side())
      {
        newtri[1] = pt1;
        newtri[2] = pt2;
      }
      else
      {
        bool reverse_order = false;
        for (std::vector<int>::iterator j = concave_ids.begin(); j != concave_ids.end(); ++j)
        {
          if ((*j) == i) reverse_order = true;
        }
        if (!reverse_order)
        {
          newtri[1] = pt1;
          newtri[2] = pt2;
        }
        else
        {
          newtri[2] = pt1;
          newtri[1] = pt2;
        }
      }

      triangulation_.push_back(newtri);
    }
  }
  // When the facet is concave, perform triangulation by applying Ear-clipping procedure
  // This is less accurate than the centre-point-triangulation, as it involves deleting in-line
  // points upto a tolerance But we have no choice in the case of concave polyhedra
  else
  {
    triangulation_.clear();
    std::vector<Point*> facetpts(points);

    if (not has_holes())
    {
      std::vector<int> ptc;
      Cut::TriangulateFacet tf(facetpts);
      tf.ear_clipping(ptc, true, true);
      triangulation_ = tf.get_split_cells();
    }
    else
    {
      std::vector<std::vector<Point*>> holepts;
      for (plain_facet_set::iterator ifacet = holes_.begin(); ifacet != holes_.end(); ++ifacet)
      {
        holepts.push_back((*ifacet)->corner_points_);
      }
      Cut::TriangulateFacet tf(facetpts, holepts);
      tf.ear_clipping_with_holes(parentside_);
      triangulation_ = tf.get_split_cells();
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::get_nodal_ids(
    Mesh& mesh, const std::vector<Point*>& points, std::vector<int>& nids)
{
  for (std::vector<Point*>::const_iterator i = points.begin(); i != points.end(); ++i)
  {
    Point* p = *i;
    Node* n = p->cut_node();
    if (n == nullptr)
    {
      plain_int_set point_id;
      point_id.insert(p->id());
      n = mesh.get_node(point_id, p->x());
    }
    nids.push_back(n->id());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::equals(
    const std::vector<Point*>& my_points, const std::vector<Point*>& facet_points)
{
  if (my_points.size() != facet_points.size() or holes_.size() > 0) return false;

  unsigned size = my_points.size();
  unsigned shift =
      std::find(my_points.begin(), my_points.end(), facet_points.front()) - my_points.begin();

  bool forward_match = true;
  for (unsigned i = 0; i < size; ++i)
  {
    unsigned j = (i + shift) % size;
    if (my_points[j] != facet_points[i])
    {
      forward_match = false;
      break;
    }
  }
  if (not forward_match)
  {
    for (unsigned i = 0; i < size; ++i)
    {
      unsigned j = (shift + size - i) % size;
      if (my_points[j] != facet_points[i])
      {
        return false;
      }
    }
  }
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_cut_side(Side* side)
{
  if (parentside_ == side) return false;

  int count = 0;
  for (std::vector<Point*>::iterator i = points_.begin(); i != points_.end(); ++i)
  {
    Point* p = *i;
    if (p->is_cut(side))
    {
      count += 1;
      if (count > 1)
      {
        return true;
      }
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::position(Point::PointPosition pos)
{
  FOUR_C_ASSERT(is_cut_position_unchanged(position_, pos),
      "Are you sure that you want to change the facet-position from inside to outside or vice "
      "versa?");

  if (position_ == Point::undecided)
  {
    if (position_ != pos)
    {
      position_ = pos;
      if (pos == Point::outside or pos == Point::inside)
      {
        for (std::vector<Point*>::iterator i = points_.begin(); i != points_.end(); ++i)
        {
          Point* p = *i;
          if (p->position() == Point::undecided)
          {
            p->position(pos);
          }
        }
        for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); ++i)
        {
          VolumeCell* c = *i;
          c->position(pos);
        }
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::get_lines(std::map<std::pair<Point*, Point*>, plain_facet_set>& lines)
{
  get_lines(points_, lines);

  // add hole lines but do not connect with parent facet
  for (plain_facet_set::iterator i = holes_.begin(); i != holes_.end(); ++i)
  {
    Facet* hole = *i;
    hole->get_lines(lines);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::get_lines(
    const std::vector<Point*>& points, std::map<std::pair<Point*, Point*>, plain_facet_set>& lines)
{
  unsigned length = points.size();
  for (unsigned i = 0; i < length; ++i)
  {
    unsigned j = (i + 1) % length;

    Point* p1 = points[i];
    Point* p2 = points[j];

    if (p1->id() < p2->id())
    {
      lines[std::make_pair(p1, p2)].insert(this);
    }
    else if (p1->id() > p2->id())
    {
      lines[std::make_pair(p2, p1)].insert(this);
    }
    else if (p1->id() == p2->id() and length == 1)
    {
      lines[std::make_pair(p1, p2)].insert(this);
    }
    else
      FOUR_C_THROW("line creation with identical begin and end points\n");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_line(Point* p1, Point* p2)
{
  if (is_triangulated())
  {
    for (std::vector<std::vector<Point*>>::iterator i = triangulation_.begin();
        i != triangulation_.end(); ++i)
    {
      std::vector<Point*>& points = *i;
      if (is_line(points, p1, p2)) return true;
    }
  }
  else
  {
    if (is_line(points_, p1, p2)) return true;
    for (plain_facet_set::iterator i = holes_.begin(); i != holes_.end(); ++i)
    {
      Facet* hole = *i;
      if (hole->is_line(p1, p2)) return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_line(const std::vector<Point*>& points, Point* p1, Point* p2)
{
  std::vector<Point*>::const_iterator i1 = std::find(points.begin(), points.end(), p1);
  if (i1 != points.end())
  {
    std::vector<Point*>::const_iterator i2 = i1 + 1;
    if (i2 != points.end())
    {
      if (*i2 == p2)
      {
        return true;
      }
    }
    else
    {
      i2 = points.begin();
      if (*i2 == p2)
      {
        return true;
      }
    }
    if (i1 != points.begin())
    {
      i2 = i1 - 1;
      if (*i2 == p2)
      {
        return true;
      }
    }
    else
    {
      i2 = points.end() - 1;
      if (*i2 == p2)
      {
        return true;
      }
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::contains(Point* p) const
{
  if (is_triangulated())
  {
    for (std::vector<std::vector<Point*>>::const_iterator i = triangulation_.begin();
        i != triangulation_.end(); ++i)
    {
      const std::vector<Point*>& points = *i;
      if (std::find(points.begin(), points.end(), p) != points.end()) return true;
    }
  }
  else
  {
    if (std::find(points_.begin(), points_.end(), p) != points_.end()) return true;
    for (plain_facet_set::const_iterator i = holes_.begin(); i != holes_.end(); ++i)
    {
      Facet* hole = *i;
      if (hole->contains(p)) return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::contains(const std::vector<Point*>& side) const
{
  for (std::vector<Point*>::const_iterator i = side.begin(); i != side.end(); ++i)
  {
    Point* p = *i;
    if (not contains(p))
    {
      return false;
    }
  }
  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::contains(const plain_facet_set& vcell) const
{
  for (plain_volumecell_set::const_iterator i_vc = cells_.begin(); i_vc != cells_.end(); ++i_vc)
  {
    const VolumeCell& vc = **i_vc;
    if (vc.is_equal(vcell)) return true;
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::contains_some(const std::vector<Point*>& side) const
{
  for (std::vector<Point*>::const_iterator i = side.begin(); i != side.end(); ++i)
  {
    Point* p = *i;
    if (contains(p))
    {
      return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::touches(Facet* f)
{
  for (std::vector<Point*>::iterator i = points_.begin(); i != points_.end(); ++i)
  {
    Point* p = *i;
    if (f->contains(p))
    {
      return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------------------------------*
 * If this Facet has a CommonEdge with another facet, based on this edge the point ordering is
 *checked ager 08/15
 *----------------------------------------------------------------------------------------------------*/
bool Cut::Facet::have_consistent_normal(Facet* f, bool& result)
{
  // 1// create map which stores references to all points cycles, which should be matched!
  //(this function uses the fact, that ordering of the points in holes is opposite to the ordering
  // of the outer points cycle!)
  std::map<std::vector<Point*>*, std::vector<Point*>*> matchingcycles;
  matchingcycles[&(corner_points_)] = &(f->corner_points_);  // outer points cycles
  if (has_holes())  // outer point cycle with inner hole point cycles
  {
    for (plain_facet_set::iterator hole = holes_.begin(); hole != holes_.end(); ++hole)
    {
      matchingcycles[&((*hole)->corner_points_)] = &(f->corner_points_);
    }
  }
  if (f->has_holes())
  {
    for (plain_facet_set::iterator hole = f->holes_.begin(); hole != f->holes_.end(); ++hole)
    {
      matchingcycles[&(corner_points_)] = &((*hole)->corner_points_);
    }
  }
  if (has_holes() && f->has_holes())  // inner hole point cycles
  {
    for (plain_facet_set::iterator hole = holes_.begin(); hole != holes_.end(); ++hole)
    {
      for (plain_facet_set::iterator fhole = f->holes_.begin(); fhole != f->holes_.end(); ++fhole)
      {
        matchingcycles[&((*hole)->corner_points_)] = &((*fhole)->corner_points_);
      }
    }
  }

  for (std::map<std::vector<Point*>*, std::vector<Point*>*>::iterator cyclepair =
           matchingcycles.begin();
      cyclepair != matchingcycles.end(); ++cyclepair)
  {
    std::vector<Point*>& points1 = *(cyclepair->first);
    std::vector<Point*>& points2 = *(cyclepair->second);

    // 2// find the corner points which are matching and store the local idx in a map<int,int>
    std::map<unsigned int, unsigned int> matchingppoints;

    for (unsigned int i = 0; i < points1.size(); i++)
    {
      for (unsigned int j = 0; j < points2.size(); j++)
      {
        if (points1[i] == points2[j])
        {
          matchingppoints[i] = j;
        }
      }
    }

    // 3// if there are more than 2 matching points, remove points at the end of the common edge(s)!
    if (matchingppoints.size() > 2)
    {
      std::cout << "WARNING: Got " << matchingppoints.size()
                << " common points of two facets! --> but just two points are used to check the "
                   "orientation!"
                << "-- if you see any problems, look into this!" << std::endl;

      std::map<unsigned int, unsigned int>::iterator bmatchingppoints1 = matchingppoints.begin();
      std::map<unsigned int, unsigned int>::iterator bmatchingppoints2 = matchingppoints.begin();
      unsigned int inlinecounter = 0;
      bmatchingppoints2++;
      while (matchingppoints.size() > 2)
      {
        if ((((bmatchingppoints1->first + 1) % (points1.size()) != bmatchingppoints2->first) &&
                ((bmatchingppoints2->first + 1) % (points1.size()) != bmatchingppoints1->first)) ||
            inlinecounter > matchingppoints.size())
        {
          matchingppoints.erase(bmatchingppoints1);
          bmatchingppoints1 = matchingppoints.begin();
          bmatchingppoints2 = matchingppoints.begin();
          bmatchingppoints2++;
          inlinecounter = 0;
          continue;
        }
        else  // in case all points are anyway on a line, it does not matter which point we delete!
        {
          inlinecounter++;
        }
        bmatchingppoints1++;
        bmatchingppoints2++;
        if (bmatchingppoints1 == matchingppoints.end()) bmatchingppoints1 = matchingppoints.begin();
        if (bmatchingppoints2 == matchingppoints.end()) bmatchingppoints2 = matchingppoints.begin();
      }
    }

    // 4// we have two common points left and check based on this information the orientation of the
    // facets!
    if (matchingppoints.size() == 2)
    {
      std::map<unsigned int, unsigned int>::iterator bmatchingppoints = matchingppoints.begin();
      bmatchingppoints++;


      bool firstordering = false;  // ordering of this facet
      if ((matchingppoints.begin()->first + 1) % (points1.size()) == bmatchingppoints->first)
        firstordering = true;
      else if ((matchingppoints.begin()->first) == (bmatchingppoints->first + 1) % (points1.size()))
        firstordering = false;
      else
        FOUR_C_THROW("Is there a point between the matched edges?");

      bool secondordering = false;  // ordering of the facet given as parameter
      if ((matchingppoints.begin()->second + 1) % (points2.size()) == bmatchingppoints->second)
        secondordering = true;
      else if ((matchingppoints.begin()->second) ==
               (bmatchingppoints->second + 1) % (points2.size()))
        secondordering = false;
      else
      {
        // it can be almost-hole facet, but touching at one point
        // in that case pointgraph produces two no-hole facets
        Side* parentside = f->parent_side();
        const auto& facets = parentside->facets();
        std::set<Point*> my_points(f->points_.begin(), f->points_.end());

        for (auto& facet : facets)
        {
          if (facet != f)
          {
            std::set<Point*> common;
            std::set<Point*> other_points(facet->points_.begin(), facet->points_.end());
            std::set_intersection(my_points.begin(), my_points.end(), other_points.begin(),
                other_points.end(), std::inserter(common, common.begin()));
            // if one is fully inside another
            if (common == other_points)
            {
              // we cannot find anything meaningful on this cycle -> it
              // includes all points of other facet, that is not a hole..
              // better try that facet instead
              return false;
            }
            else
            {
              FOUR_C_THROW("Check this case!");
            }
          }
        }
        FOUR_C_THROW("Is there a point between the matched edges?");
      }

      // ordering of the edge hast to be different, if the facets are ordered equal and vice versa!
      result = !(firstordering == secondordering);
      return true;
    }
  }
  // nowhere 2 common points in any cycle found --> facets have no common edge!
  return false;
}

Cut::VolumeCell* Cut::Facet::neighbor(VolumeCell* cell)
{
  if (cells_.size() > 2) FOUR_C_THROW("can only have two neighbors");

  if (cells_.count(cell) == 0) FOUR_C_THROW("not my neighbor");

  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* vc = *i;
    if (vc != cell)
    {
      return vc;
    }
  }
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::neighbors(Point* p, const plain_volumecell_set& cells,
    const plain_volumecell_set& done, plain_volumecell_set& connected, plain_element_set& elements)
{
  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* c = *i;
    if (cells.count(c) > 0)
    {
      if (done.count(c) == 0 and connected.count(c) == 0 and
          elements.count(c->parent_element()) == 0)
      {
        connected.insert(c);
        elements.insert(c->parent_element());
        c->neighbors(p, cells, done, connected, elements);
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::neighbors(Point* p, const plain_volumecell_set& cells,
    const plain_volumecell_set& done, plain_volumecell_set& connected)
{
  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* c = *i;
    if (cells.count(c) > 0)
    {
      if (done.count(c) == 0 and connected.count(c) == 0)
      {
        connected.insert(c);
        c->neighbors(p, cells, done, connected);
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::equals(Core::FE::CellType distype)
{
  if (holes_.size() == 0)
  {
    find_corner_points();
    switch (distype)
    {
      case Core::FE::CellType::point1:
        return Kernel::is_valid_point1(corner_points_);
        break;
      case Core::FE::CellType::line2:
        return Kernel::is_valid_line2(corner_points_);
        break;
      case Core::FE::CellType::quad4:
        return Kernel::is_valid_quad4(corner_points_);
        break;
      case Core::FE::CellType::tri3:
        return Kernel::is_valid_tri3(corner_points_);
        break;
      default:
        FOUR_C_THROW("unsupported distype requested");
        break;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
unsigned Cut::Facet::normal(const std::vector<Point*>& points, Core::LinAlg::Matrix<3, 1>& x1,
    Core::LinAlg::Matrix<3, 1>& x2, Core::LinAlg::Matrix<3, 1>& x3, Core::LinAlg::Matrix<3, 1>& b1,
    Core::LinAlg::Matrix<3, 1>& b2, Core::LinAlg::Matrix<3, 1>& b3)
{
  unsigned pointsize = points.size();
  if (pointsize < 3) return 0;

  points[0]->coordinates(x1.data());
  points[1]->coordinates(x2.data());

  b1.update(1, x2, -1, x1, 0);
  b1.scale(1. / b1.norm2());

  if (b1.norm2() < std::numeric_limits<double>::min())
    FOUR_C_THROW("same point in facet not supported");

  bool found = false;
  unsigned i = 2;
  for (; i < pointsize; ++i)
  {
    Point* p = points[i];
    p->coordinates(x3.data());

    b2.update(1, x3, -1, x1, 0);
    b2.scale(1. / b2.norm2());

    // cross product to get the normal at the point
    b3(0) = b1(1) * b2(2) - b1(2) * b2(1);
    b3(1) = b1(2) * b2(0) - b1(0) * b2(2);
    b3(2) = b1(0) * b2(1) - b1(1) * b2(0);

    if (b3.norm2() > PLANARTOL)
    {
      found = true;
      break;
    }
  }
  if (not found)
  {
    // all on one line, no normal
    return 0;
  }

  b3.scale(1. / b3.norm2());
  return i;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::triangulation_points(PointSet& points)
{
  for (std::vector<std::vector<Point*>>::const_iterator i = triangulation_.begin();
      i != triangulation_.end(); ++i)
  {
    const std::vector<Point*>& tri = *i;
    std::copy(tri.begin(), tri.end(), std::inserter(points, points.begin()));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::new_point1_cell(Mesh& mesh, VolumeCell* volume, const std::vector<Point*>& points,
    plain_boundarycell_set& bcells)
{
  BoundaryCell* bc = mesh.new_point1_cell(volume, this, points);
  bcells.insert(bc);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::new_line2_cell(Mesh& mesh, VolumeCell* volume, const std::vector<Point*>& points,
    plain_boundarycell_set& bcells)
{
  BoundaryCell* bc = mesh.new_line2_cell(volume, this, points);
  bcells.insert(bc);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::new_tri3_cell(Mesh& mesh, VolumeCell* volume, const std::vector<Point*>& points,
    plain_boundarycell_set& bcells)
{
  BoundaryCell* bc = mesh.new_tri3_cell(volume, this, points);
  bcells.insert(bc);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::new_quad4_cell(Mesh& mesh, VolumeCell* volume, const std::vector<Point*>& points,
    plain_boundarycell_set& bcells)
{
  if (mesh.create_options().gen_quad4())
  {
    BoundaryCell* bc = mesh.new_quad4_cell(volume, this, points);
    bcells.insert(bc);
  }
  else
  {  // split into two tri3 cells
    std::vector<Point*> tri3_points = points;
    tri3_points.pop_back();  // erase the last (fourth) point to obtain points (1,2,3)
    BoundaryCell* bc = mesh.new_tri3_cell(volume, this, tri3_points);
    bcells.insert(bc);
    tri3_points.erase(tri3_points.begin() + 1);  // erase the second point (1,3)
    tri3_points.push_back(points.back());        // add the last point again (1,3,4)
    bc = mesh.new_tri3_cell(volume, this, tri3_points);
    bcells.insert(bc);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::new_arbitrary_cell(Mesh& mesh, VolumeCell* volume,
    const std::vector<Point*>& points, plain_boundarycell_set& bcells,
    const Core::FE::GaussIntegration& gp, const Core::LinAlg::Matrix<3, 1>& normal)
{
  BoundaryCell* bc = mesh.new_arbitrary_cell(volume, this, points, gp, normal);
  bcells.insert(bc);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::get_boundary_cells(plain_boundarycell_set& bcells)
{
  for (plain_volumecell_set::iterator vit = cells_.begin(); vit != cells_.end(); ++vit)
  {
    VolumeCell* vc = *vit;
    const plain_boundarycell_set& vbcells = vc->boundary_cells();
    for (plain_boundarycell_set::const_iterator i = vbcells.begin(); i != vbcells.end(); ++i)
    {
      BoundaryCell* bc = *i;
      if (bc->get_facet() == this) bcells.insert(bc);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::test_facet_area(double tolerance, bool istetmeshintersection)
{
  if (on_cut_side() and cells_.size() > 1)
  {
    std::vector<double> area;
    area.reserve(2);
    // Does not necessarily have to be same on both sides of cutside.
    // Especially true for tetmeshintersection.
    for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); ++i)
    {
      VolumeCell* vc = *i;

      area.push_back(0.);
      double& a = area.back();

      const plain_boundarycell_set& vbcells = vc->boundary_cells();
      for (plain_boundarycell_set::const_iterator i = vbcells.begin(); i != vbcells.end(); ++i)
      {
        BoundaryCell* bc = *i;
        if (bc->get_facet() == this)
        {
          a += bc->area();
        }
      }
    }
    if (area.size() != 2)
    {
      FOUR_C_THROW("expect two volume cells at facet");
    }
    double diff = area[0] - area[1];
    if (fabs(diff) >= tolerance)
    {
      int eleId = (*cells_.begin())->parent_element()->id();
      std::stringstream str;
      if (istetmeshintersection)
      {
        str << "In eleId=" << eleId << ", in TetMeshIntersection,"
            << " area mismatch: a1=" << area[0] << " a2=" << area[1] << " diff=" << diff;
      }
      else
      {
        str << "In eleId=" << eleId << " area mismatch: a1=" << area[0] << " a2=" << area[1]
            << " diff=" << diff;
      }

      std::cout << "WARNING: " << str.str() << "\n";
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::find_corner_points()
{
  if (corner_points_.size() == 0)
  {
    corner_points_ = points_;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Facet::print(std::ostream& stream) const
{
  stream << "--- Facet ( address: " << this << " )\n";
  stream << "# Facet: "
         << "numpoints " << points_.size() << "\n# sideID: " << this->side_id()
         << "\n# OnCutSide: " << (on_cut_side() ? "TRUE" : "FALSE")
         << "\n# registered volume cells: ";
  std::copy(cells_.begin(), cells_.end(), std::ostream_iterator<VolumeCell*>(stream, " "));
  stream << "\n# points: ";
  std::copy(points_.begin(), points_.end(), std::ostream_iterator<Point*>(stream, " "));
  stream << "\n";
  for (std::vector<Point*>::const_iterator i = points_.begin(); i != points_.end(); ++i)
  {
    Point& p = **i;
    p.print(stream);
    stream << "\n";
  }
  stream << "\n";

  for (plain_facet_set::const_iterator i = holes_.begin(); i != holes_.end(); ++i)
  {
    Facet* hole = *i;
    hole->print(stream);
  }

  stream << "\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_triangle(const std::vector<Point*>& tri) const
{
  if (tri.size() != 3) FOUR_C_THROW("three points expected");

  return points_.size() == 3 and not is_triangulated() and not has_holes() and contains(tri);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_triangulated_side(const std::vector<Point*>& tri) const
{
  if (tri.size() != 3) FOUR_C_THROW("three points expected");

  for (std::vector<std::vector<Point*>>::const_iterator i = triangulation_.begin();
      i != triangulation_.end(); ++i)
  {
    const std::vector<Point*>& t = *i;
    bool found = true;
    for (std::vector<Point*>::const_iterator i = tri.begin(); i != tri.end(); ++i)
    {
      Point* p = *i;
      if (std::find(t.begin(), t.end(), p) == t.end())
      {
        found = false;
        break;
      }
    }
    if (found)
    {
      return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
unsigned Cut::Facet::num_points()
{
  unsigned numpoints = points_.size();
  if (is_triangulated()) return numpoints + 1;
  for (plain_facet_set::iterator i = holes_.begin(); i != holes_.end(); ++i)
  {
    Facet* hole = *i;
    numpoints += hole->num_points();
  }
  return numpoints;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Point* Cut::Facet::other_point(Point* p1, Point* p2)
{
  Point* result = nullptr;
  if (not has_holes() and not is_triangulated() and points().size() == 3)
  {
    for (std::vector<Point*>::iterator i = points_.begin(); i != points_.end(); ++i)
    {
      Point* p = *i;
      if (p != p1 and p != p2)
      {
        if (result == nullptr)
        {
          result = p;
        }
        else
        {
          FOUR_C_THROW("point not unique");
        }
      }
    }
  }
  else
  {
    FOUR_C_THROW("plain triangular facet required");
  }
  return result;
}

/*----------------------------------------------------------------------------*
  return the local coordinates of corner points with respect to the given element
  if shadow=true, then the mapping is w.r. to the parent quad element from which
  this element is derived
*-----------------------------------------------------------------------------*/
void Cut::Facet::corner_points_local(
    Element* elem1, std::vector<std::vector<double>>& cornersLocal, bool shadow)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::Facet::CornerPointsLocal" );

  // TODO: this routine is expensive
  cornersLocal.clear();

  const std::vector<Point*>& corners = corner_points();

  for (std::vector<Point*>::const_iterator k = corners.begin(); k != corners.end(); k++)
  {
    std::vector<double> pt_local;
    const Point* po = *k;
    const double* coords = po->x();
    Core::LinAlg::Matrix<3, 1> glo, loc;

    glo(0, 0) = coords[0];
    glo(1, 0) = coords[1];
    glo(2, 0) = coords[2];

    // TODO: do we really need call a Newton here?
    if (shadow and elem1->is_shadow())
      elem1->local_coordinates_quad(glo, loc);
    else
      elem1->local_coordinates(glo, loc);

    pt_local.push_back(loc(0, 0));
    pt_local.push_back(loc(1, 0));
    pt_local.push_back(loc(2, 0));

    cornersLocal.push_back(pt_local);
  }
}

/*----------------------------------------------------------------------------*
 * Return the global coordinates all of its corner points in order
 *                                                              sudhakar 05/15
 *----------------------------------------------------------------------------*/
std::vector<std::vector<double>> Cut::Facet::corner_points_global(Element* elem1, bool shadow)
{
  const std::vector<Point*>& corners = corner_points();
  std::vector<std::vector<double>> cornersLocal;
  for (std::vector<Point*>::const_iterator k = corners.begin(); k != corners.end(); k++)
  {
    std::vector<double> pt_local(3);
    const Point* po = *k;
    const double* coords = po->x();

    for (unsigned dim = 0; dim < 3; dim++) pt_local[dim] = coords[dim];

    cornersLocal.push_back(pt_local);
  }
  return cornersLocal;
}

/*------------------------------------------------------------------------*
 *        Split the facet into a number of tri and quad cells
 *------------------------------------------------------------------------*/
void Cut::Facet::split_facet(const std::vector<Point*>& facetpts)
{
  if (not this->has_holes())
  {
    TriangulateFacet tf(facetpts);
    tf.split_facet();
    split_cells_ = tf.get_split_cells();
  }
  else
  {
    std::vector<std::vector<Point*>> holepts;
    for (plain_facet_set::iterator ifacet = holes_.begin(); ifacet != holes_.end(); ++ifacet)
    {
      holepts.push_back((*ifacet)->corner_points_);
    }
    Cut::TriangulateFacet tf(facetpts, holepts);
    tf.ear_clipping_with_holes(parentside_);
    split_cells_ = tf.get_split_cells();
  }
}

/*----------------------------------------------------------------------------*
          Returns true if the facet is convex shaped
 *----------------------------------------------------------------------------*/
bool Cut::Facet::is_convex()
{
  if (this->has_holes()) return false;

  Cut::FacetShape geoType;
  std::vector<int> ptConcavity = Kernel::check_convexity(corner_points_, geoType, false, false);

  if (geoType == Cut::Convex) return true;

  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Facet::belongs_to_level_set_side() { return parentside_->is_level_set_side(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& stream, Cut::Facet& f)
{
  stream << "facet: {";
  if (f.is_triangulated())
  {
    const std::vector<std::vector<Cut::Point*>>& triangulation = f.triangulation();
    for (std::vector<std::vector<Cut::Point*>>::const_iterator i = triangulation.begin();
        i != triangulation.end(); ++i)
    {
      const std::vector<Cut::Point*>& tri = *i;
      stream << "{";
      for (std::vector<Cut::Point*>::const_iterator i = tri.begin(); i != tri.end(); ++i)
      {
        Cut::Point* p = *i;
        p->print(stream);
        stream << ",";
      }
      stream << "},";
    }
  }
  else
  {
    const std::vector<Cut::Point*>& points = f.points();
    for (std::vector<Cut::Point*>::const_iterator i = points.begin(); i != points.end(); ++i)
    {
      Cut::Point* p = *i;
      p->print(stream);
      stream << ",";
    }
    if (f.has_holes())
    {
      const Cut::plain_facet_set& holes = f.holes();
      for (Cut::plain_facet_set::const_iterator i = holes.begin(); i != holes.end(); ++i)
      {
        Cut::Facet& h = **i;
        stream << h;
      }
    }
  }
  stream << "}";
  return stream;
}

FOUR_C_NAMESPACE_CLOSE
