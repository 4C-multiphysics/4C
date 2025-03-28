// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_facetgraph.hpp"
#include "4C_cut_integrationcellcreator.hpp"
#include "4C_cut_intersection.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_tetmesh.hpp"
#include "4C_fem_geometry_element_volume.hpp"
#include "4C_global_data.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <stack>
#include <string>

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------*
 * struct for comparison of position of sides using ray-tracing techniques
 * shoot a ray starting from startpoint through the midpoint of one side
 * find the intersection point with the second side
 * dependent on the local coordinates along the ray, decide which
 * side lies in front of the other one
 *--------------------------------------------------------------------*/
struct NextSideAlongRay
{
  NextSideAlongRay(Cut::Point* startpoint, Cut::Point* cutpoint)
      : startpoint_(startpoint), cutpoint_(cutpoint)
  {
    startpoint_->coordinates(startpoint_xyz_.data());
    cutpoint_->coordinates(cutpoint_xyz_.data());
  };

  /*--------------------------------------------------------------------*
   * check if both sides have the same normal vector
   *--------------------------------------------------------------------*/
  bool same_normal(Cut::Side* s1, Cut::Side* s2, const Core::LinAlg::Matrix<3, 1>& cutpoint_xyz)
  {
    Core::LinAlg::Matrix<3, 1> rst(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<2, 1> rs(Core::LinAlg::Initialization::zero);

    //-------------
    // first side
    s1->local_coordinates(cutpoint_xyz, rst, false);

    rs(0) = rst(0);
    rs(1) = rst(1);

    Core::LinAlg::Matrix<3, 1> normal_1(Core::LinAlg::Initialization::zero);
    s1->normal(rs, normal_1);

    //-------------
    // second side
    s2->local_coordinates(cutpoint_xyz, rst, false);

    rs(0) = rst(0);
    rs(1) = rst(1);

    Core::LinAlg::Matrix<3, 1> normal_2(Core::LinAlg::Initialization::zero);
    s2->normal(rs, normal_2);

    //-------------
    if (normal_1.dot(normal_2) > 1 - REFERENCETOL) return true;

    return false;
  }

  /*--------------------------------------------------------------------*
   * comparator function to sort two sides, which side lies in front of the other along the ray
   *--------------------------------------------------------------------*/
  bool operator()(Cut::Side* s1, Cut::Side* s2)
  {
    // REMARK:
    // shoot a ray through the first side s1 starting from startpoint and find intersection with
    // side s2 if not successful shoot a second ray through side s2 and find intersection with s1 if
    // not successful check if the sides are parallel
    bool is_closer = false;

    if (s1->is_closer_side(startpoint_xyz_, s2, is_closer))
    {
      if (is_closer)
        return true;
      else
        return false;
    }
    else if (s2->is_closer_side(startpoint_xyz_, s1, is_closer))
    {
      if (!is_closer)
        return true;
      else
        return false;
    }
    else if (same_normal(s1, s2, cutpoint_xyz_))  // check if both sides are parallel to each other,
                                                  // then both sides lead the same position
    {
      return true;
    }
    else
    {
      // TODO: check if we can relax this case (Parallelogramm, both Positions would be the same!?)
      // try to return true or false, both sides should lead to the same position, sorting not
      // necessary

      // return true;
      std::cout << "side 1: " << *s1 << std::endl;
      std::cout << "side 2: " << *s2 << std::endl;
      std::cout << "startpoint: " << startpoint_xyz_ << std::endl;
      FOUR_C_THROW(
          "ray-tracing-based comparisons to find the nearest side along the ray failed for the "
          "first time!");
    }

    return false;
  }

  Cut::Point* startpoint_;
  Cut::Point* cutpoint_;

  Core::LinAlg::Matrix<3, 1> startpoint_xyz_;
  Core::LinAlg::Matrix<3, 1> cutpoint_xyz_;
};


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Element::Element(
    int eid, const std::vector<Side*>& sides, const std::vector<Node*>& nodes, bool active)
    : eid_(eid),
      active_(active),
      sides_(sides),
      nodes_(nodes),
      quadshape_(Core::FE::CellType::dis_none),
      eleinttype_(Cut::EleIntType_Undecided)
{
  for (std::vector<Side*>::const_iterator i = sides.begin(); i != sides.end(); ++i)
  {
    Side* s = *i;
    s->register_entity(this);
  }
  for (std::vector<Node*>::const_iterator i = nodes.begin(); i != nodes.end(); ++i)
  {
    Node* n = *i;
    n->register_entity(this);
    points_.push_back(n->point());
  }
  parent_id_ = -1;  // initialize with non-reasonable negative element Id

  // shadow elements are initialized separately
  is_shadow_ = false;

  boundingvolume_ = std::shared_ptr<BoundingBox>(BoundingBox::create(*this));
}

/*-----------------------------------------------------------------------------------*
 *  For this shadow element, set corner nodes of parent Quad element      sudhakar 11/13
 *-----------------------------------------------------------------------------------*/
void Cut::Element::set_quad_corners(Mesh& mesh, const std::vector<int>& nodeids)
{
  if (not is_shadow_) FOUR_C_THROW("You can't set Quad-corners for non-shadow element\n");

  for (unsigned i = 0; i < nodeids.size(); i++)
  {
    Node* n1 = mesh.get_node(nodeids[i]);
    quad_corners_.push_back(n1);
  }
}

/*----------------------------------------------------------------------------------------------------*
 *  Get corner nodes of parent Quad element from which this shadow element is derived       sudhakar
 *11/13
 *----------------------------------------------------------------------------------------------------*/
std::vector<Cut::Node*> Cut::Element::get_quad_corners()
{
  if ((not is_shadow_) or quad_corners_.size() == 0)
    FOUR_C_THROW("what?! you want Quadratic element corners for linear element?\n");
  return quad_corners_;
}

/*--------------------------------------------------------------------*
 *            cut this element with given cut_side ...
 *            Called by Tetmeshintersection and LS!!!!!!
 *            but not for normal meshintersection!!!
 *--------------------------------------------------------------------*/
bool Cut::Element::cut(Mesh& mesh, Side& cut_side)
{
  bool cut = false;

  // find nodal points inside the element (a level-set side does not have nodes)
  const std::vector<Node*>& side_nodes = cut_side.nodes();

  for (std::vector<Node*>::const_iterator i = side_nodes.begin(); i != side_nodes.end(); ++i)
  {
    Node* n = *i;
    Point* p = n->point();

    if (not p->is_cut(this))  // point does not know if this element is a cut_element_
    {
      if (point_inside(p))  // if point is inside the element
      {
        p->add_element(this);  // add element to cut_element_-list of this point
        cut = true;
      }
    }
    else  // point cuts this element, already determined by another side
    {
      cut = true;
    }
  }

  /* all the other cut points lie on sides of the element (s is an element side,
   * cut_side is the cutter side)
   * entry point for level-set cuts */
  const std::vector<Side*>& sides = Element::sides();
  for (std::vector<Side*>::const_iterator i = sides.begin(); i != sides.end(); ++i)
  {
    Side* s = *i;
    if (find_cut_points(mesh, *s, cut_side))
    {
      cut = true;
    }
  }

  // insert this side into cut_faces_
  if (cut)
  {
    cut_faces_.insert(&cut_side);
    return true;
  }
  else
  {
    return false;
  }
}

/*--------------------------------------------------------------------*
 * cut this element with its cut faces                    wirtz 08/14 *
 *--------------------------------------------------------------------*/
void Cut::Element::find_cut_points(Mesh& mesh)
{
  for (plain_side_set::iterator i = cut_faces_.begin(); i != cut_faces_.end();
      /* do not increment */)
  {
    Side& cut_side = **i;
    bool cut = find_cut_points(mesh, cut_side);

    /* insert this side into cut_faces_, also the case when a side just
     * touches the element at a single point, edge or the whole side */
    if (!cut)
    {
      set_erase(cut_faces_, i);
    }
    else
    {
      ++i;
    }
  }
}

/*--------------------------------------------------------------------*
 * cut this element with given cut_side                   wirtz 08/14 *
 *--------------------------------------------------------------------*/
bool Cut::Element::find_cut_points(Mesh& mesh, Side& cut_side)
{
  bool cut = false;

  // find nodal points inside the element
  const std::vector<Node*> side_nodes = cut_side.nodes();

  for (std::vector<Node*>::const_iterator i = side_nodes.begin(); i != side_nodes.end(); ++i)
  {
    Node* n = *i;
    Point* p = n->point();

    if (not p->is_cut(this))  // point does not know if this element is a cut_element_
    {
      if (point_inside(p))  // if point is inside the element
      {
        p->add_element(this);  // add element to cut_element_-list of this point
        cut = true;
      }
    }
    else  // point cuts this element, already determined by another side
    {
      cut = true;
    }
  }
  // all the other cut points lie on sides of the element (s is an element side, cut_side is the
  // cutter side)
  const std::vector<Side*>& sides = Element::sides();
  for (std::vector<Side*>::const_iterator i = sides.begin(); i != sides.end(); ++i)
  {
    Side* s = *i;
    if (find_cut_points(mesh, *s, cut_side))
    {
      cut = true;
    }
  }


  return cut;
}

/*---------------------------------------------------------------------------*
 * After all cut points are found, create cut lines for this element by
 * connecting appropriate cut points
 *---------------------------------------------------------------------------*/
void Cut::Element::make_cut_lines(Mesh& mesh)
{
  for (plain_side_set::iterator i = cut_faces_.begin(); i != cut_faces_.end(); ++i)
  {
    Side& cut_side = **i;

    const std::vector<Side*>& sides = Element::sides();
    // create cut lines over each side of background element
    for (std::vector<Side*>::const_iterator i = sides.begin(); i != sides.end(); ++i)
    {
      Side* s = *i;
      find_cut_lines(mesh, *s, cut_side);
    }
  }
}

/*-----------------------------------------------------------------------------*
 * Find cut points between a background element side and a cut side
 * Cut points are stored correspondingly
 *-----------------------------------------------------------------------------*/
bool Cut::Element::find_cut_points(Mesh& mesh, Side& ele_side, Side& cut_side)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 4/6 --- cut_mesh_intersection --- find_cut_points(ele)");

  // edges of element side cuts through cut side
  bool cut = ele_side.find_cut_points(mesh, this, cut_side);
  // edges of cut side cuts through element side
  // ( does nothing for the level-set case, since a level-set side has no edges! )
  bool reverse_cut = cut_side.find_cut_points(mesh, this, ele_side);
  return cut or reverse_cut;
}

/*----------------------------------------------------------------------------*
 *  Returns true if cut lines exist between the cut points produced by the two
 *  sides
 *----------------------------------------------------------------------------*/
bool Cut::Element::find_cut_lines(Mesh& mesh, Side& ele_side, Side& cut_side)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 4/6 --- cut_mesh_intersection --- find_cut_lines");

  return ele_side.find_cut_lines(mesh, this, cut_side);
}

/*----------------------------------------------------------------------------*
 * Create facets
 *----------------------------------------------------------------------------*/
void Cut::Element::make_facets(Mesh& mesh)
{
  if (facets_.size() == 0)
  {
    const std::vector<Side*>& sides = Element::sides();
    for (std::vector<Side*>::const_iterator i = sides.begin(); i != sides.end(); ++i)
    {
      Side& side = **i;
      side.make_owned_side_facets(mesh, this, facets_);
    }

    for (plain_side_set::iterator i = cut_faces_.begin(); i != cut_faces_.end(); ++i)
    {
      Side& cut_side = **i;
      cut_side.make_internal_facets(mesh, this, facets_);
    }
  }
}

/*------------------------------------------------------------------------------------------*
 *     Determine the inside/outside/oncutsurface position for the element's nodes
 *------------------------------------------------------------------------------------------*/
void Cut::Element::find_node_positions()
{
  // DEBUG flag for FindNodePositions
  // compute positions for nodes again, also if already set by other nodes, facets, vcs (safety
  // check)
  // #define check_for_all_nodes

  //----------------------------------------------------------------------------------------
  // new implementation based on cosine between normal vector on cut side and line-vec between point
  // and cut-point
  //----------------------------------------------------------------------------------------

  const std::vector<Node*>& nodes = Element::nodes();

  // determine positions for all the element's nodes
  for (std::vector<Node*>::const_iterator i = nodes.begin(); i != nodes.end(); ++i)
  {
    Node* n = *i;
    Point* p = n->point();
    Point::PointPosition pos = p->position();

#ifdef check_for_all_nodes
    std::cout << "Position for node " << n->Id() << std::endl;
    // do the computation in all cases
#else
    if (pos == Point::undecided)
#endif
    {
      bool done = false;

      // a) this line lies on the cut surface, then p has to be on cut surface for a least one
      // cut-side b) the line connects two points, both lying on different cut sides, then p has to
      // be on cut surface for a least one cut-side

      // check if the node lies on a cut-surface
      for (plain_side_set::const_iterator i = cut_faces_.begin(); i != cut_faces_.end(); ++i)
      {
        Side* s = *i;

        if (s->is_level_set_side()) continue;  // do not deal with level-set sides here!

        // check if the point lies on one of the element's cut sides
        if (p->is_cut(s))
        {
          p->position(Point::oncutsurface);

          done = true;
          break;
        }
      }

      if (done) continue;  // next node

      // c) search for a line connection between the point p and a cut side in this element
      //
      // is there a facet's (!) line between the point p and a cut-point on the side s ?
      // if there is a line, then no further point lies between p and the cut side
      // this line goes either through the outside or inside region

      const plain_facet_set& facets = p->facets();

      // loop all the facets sharing this node
      for (plain_facet_set::const_iterator j = facets.begin(); j != facets.end(); ++j)
      {
        Facet* f = *j;

        // loop all the cut-faces stored for this element
        // (includes cut-faces that only touch the element at points, edges, sides or parts of them)
        for (plain_side_set::const_iterator i = cut_faces_.begin(); i != cut_faces_.end(); ++i)
        {
          Side* s = *i;

          if (s->is_level_set_side()) continue;  // do not deal with level-set sides here!


          // is there a common point between facet and side?
          // and belongs the facet to the element (otherwise we enter the neighboring element via
          // the facet)?
          // - however we include cut-sides of neighboring elements that only touches the facet,
          // - the facet however has to an element's facet
          if (f->is_cut_side(s) and is_facet(f))
          {
            // for inside-outside decision there must be a direct line connection between the point
            // and the cut-side check for a common facet's line between a side's cut point and point
            // p

            std::map<std::pair<Point*, Point*>, plain_facet_set> lines;
            f->get_lines(lines);  // all facet's lines, each line sorted by P1->Id() < P2->Id()

            for (std::map<std::pair<Point*, Point*>, plain_facet_set>::iterator line_it =
                     lines.begin();
                line_it != lines.end(); line_it++)
            {
              std::pair<Point*, Point*> line = line_it->first;

              Point* cutpoint = nullptr;

              // find the right facet's line and the which endpoint is the cut-point
              if (line.first->id() == p->id() and line.second->is_cut(s))
              {
                cutpoint = line.second;
              }
              else if (line.second->id() == p->id() and line.first->is_cut(s))
              {
                cutpoint = line.first;
              }
              else
              {
                // this line is not a line between the point and the cut-side
                // continue with next line
                continue;
              }

              //---------------------------------------------------
              // call the main routine to compute the position based on the angle between
              // the line-vec (p-c) and an appropriate cut-side
              done = compute_position(p, cutpoint, f, s);
              //---------------------------------------------------

              if (done) break;
            }  // end lines

            if (done) break;
          }
        }  // end cutsides

        if (done) break;
      }  // loop facets
      if (p->position() == Point::undecided)
      {
        // Still undecided! No facets with cut side attached! Will be set in a
        // minute.
      }

      if (done) continue;
    }  // end if undecided

#ifdef check_for_all_nodes
    if (pos == Point::outside or pos == Point::inside)
#else
    else if (pos == Point::outside or pos == Point::inside)
#endif
    {
      // The nodal position is already known. Set it to my facets. If the
      // facets are already set, this will not have much effect anyway. But on
      // multiple cuts we avoid unset facets this way.
      const plain_facet_set& facets = p->facets();
      for (plain_facet_set::const_iterator k = facets.begin(); k != facets.end(); ++k)
      {
        Facet* f = *k;
        f->position(pos);
      }
    }  // end if outside or inside

  }  // loop nodes
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Element::compute_position(Point* p, Point* cutpoint, Facet* f, Side* s)
{
  //---------------------------
  // find the element's volume-cell the cut-side and the line has to be adjacent to
  const plain_volumecell_set& facet_cells = f->cells();
  plain_volumecell_set adjacent_cells;

  for (plain_volumecell_set::const_iterator f_cells_it = facet_cells.begin();
      f_cells_it != facet_cells.end(); f_cells_it++)
  {
    if (cells_.count(*f_cells_it)) adjacent_cells.insert(*f_cells_it);  // insert this cell
  }

  if (adjacent_cells.size() > 1)
  {
    std::cout << "Warning: there is not a unique element's volumecell, number="
              << adjacent_cells.size() << ", the line and facet is adjacent to-> Check this"
              << std::endl;
    FOUR_C_THROW("Warning: there is not a unique element's volumecell");
  }
  else if (adjacent_cells.size() == 0)
  {
    std::cout << "facet cells" << facet_cells.size() << std::endl;
    std::cout << "Warning: there is no adjacent volumecell, the line and facet is adjacent "
                 "to-> Check this"
              << std::endl;
    FOUR_C_THROW("Warning: there is no adjacent volumecell");
  }

  // that's the adjacent volume-cell
  VolumeCell* vc = *(adjacent_cells.begin());

  //---------------------------
  /* get the element's cut-sides adjacent to this cut-point and adjacent to
   * the same volume-cell */
  const plain_side_set& cut_sides = this->cut_sides();
  std::vector<Side*> point_cut_sides;

  //  plain_side_set e_cut_sides;
  for (plain_side_set::const_iterator side_it = cut_sides.begin(); side_it != cut_sides.end();
      side_it++)
  {
    /* Is the cut-point a cut-point of this element's cut_side?
     * Is this side a cut-side adjacent to this volume-cell ?
     * Remove sides, if normal vector is orthogonal to side and cut-point
     * lies on edge, since then the angle criterion does not work */
    if (cutpoint->is_cut(*side_it) and vc->is_cut(*side_it) and
        !is_orthogonal_side(*side_it, p, cutpoint))
    {
      // the angle-criterion has to be checked for this side
      point_cut_sides.push_back(*side_it);
    }
  }

  // std::cout << "how many cut_sides found? " << point_cut_sides.size() << std::endl;

  if (point_cut_sides.size() == 0)
  {
    /* no right cut_side found! -> Either another node can compute the position
     * or hope for distributed positions or hope for parallel communication */
    return false;
  }

  //------------------------------------------------------------------------
  /* Sort the sides and do the check for the first one!
   * The sorting is based on ray-tracing techniques:
   * Shoot a ray starting from point p through the midpoint of one of the
   * two sides and find another intersection point. The local coordinates
   * along this ray determines the order of the sides */
  //------------------------------------------------------------------------
  if (point_cut_sides.size() > 1)
    std::sort(point_cut_sides.begin(), point_cut_sides.end(), NextSideAlongRay(p, cutpoint));

  //------------------------------------------------------------------------
  /* determine the inside/outside position w.r.t the chosen cut-side
   * in case of the right side the "angle-criterion" leads to the right
   * decision (position) */
  Side* cut_side = *(point_cut_sides.begin());

  const bool success = position_by_angle(p, cutpoint, cut_side);
  //------------------------------------------------------------------------

  // if(success) std::cout << "set position to " << p->Position() << std::endl;
  // else std::cout << "not successful" << std::endl;

  return success;
}

/*------------------------------------------------------------------------------------------*
 *  determine the position of point p based on the angle between the line (p-c) and the
 *  side's normal vector, return if successful
 *------------------------------------------------------------------------------------------*/
bool Cut::Element::position_by_angle(Point* p, Point* cutpoint, Side* s)
{
  Core::LinAlg::Matrix<3, 1> xyz(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> cut_point_xyz(Core::LinAlg::Initialization::zero);

  p->coordinates(xyz.data());
  cutpoint->coordinates(cut_point_xyz.data());

  //------------------------------------------------------------------------
  // determine the inside/outside position w.r.t the chosen cut-side
  // in case of the right side the "angle-criterion" leads to the right decision (position)

  Core::LinAlg::Matrix<2, 1> rs(
      Core::LinAlg::Initialization::zero);  // local coordinates of the cut-point w.r.t side

  Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);
  s->normal(rs, normal);  // outward pointing normal at cut-point

  Core::LinAlg::Matrix<3, 1> line_vec(Core::LinAlg::Initialization::zero);
  line_vec.update(
      1.0, xyz, -1.0, cut_point_xyz);  // vector representing the line between p and the cut-point

  // check the cosine between normal and line_vec
  double n_norm = normal.norm2();
  double l_norm = line_vec.norm2();
  if (n_norm < MERGING_TOLERANCE or l_norm < MERGING_TOLERANCE)
  {
    double distance_between = Cut::distance_between_points(p, cutpoint);
    FOUR_C_THROW(
        " the norm of line_vec or n_norm is smaller than {}, should these "
        "points be one point in pointpool?, lnorm={}, nnorm={}, distance between them = {}",
        REFERENCETOL, l_norm, n_norm, distance_between);
  }

  // cosine between the line-vector and the normal vector
  double cosine = normal.dot(line_vec);
  cosine /= (n_norm * l_norm);

  if (cosine > 0.0 + 1e-3)
  {
    p->position(Point::outside);
    // std::cout << " set position to outside" << std::endl;
    return true;
  }
  else if (cosine < 0.0 - 1e-3)
  {
    p->position(Point::inside);
    // std::cout << " set position to inside" << std::endl;
    return true;
  }
  else
  {
    // Still undecided!
    // There must be another side with cosine != 0.0
    return false;
  }
  //------------------------------------------------------------------------
  return false;
}


/*------------------------------------------------------------------------------------------*
 *  check if the side's normal vector is orthogonal to the line between p and the cutpoint
 *------------------------------------------------------------------------------------------*/
bool Cut::Element::is_orthogonal_side(Side* s, Point* p, Point* cutpoint)
{
  if (s->on_edge(cutpoint))  // check if the point lies on at least one edge of the side, otherwise
                             // it cannot be orthogonal
  {
    Core::LinAlg::Matrix<3, 1> line(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> p_xyz(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> cut_point_xyz(Core::LinAlg::Initialization::zero);

    p->coordinates(p_xyz.data());
    cutpoint->coordinates(cut_point_xyz.data());
    line.update(1.0, p_xyz, -1.0, cut_point_xyz);

    double line_norm = line.norm2();

    if (line_norm > BASICTOL)
    {
      line.scale(1. / line_norm);
    }
    else
    {
      std::cout << "point: " << p_xyz << std::endl;
      std::cout << "cutpoint: " << cut_point_xyz << std::endl;
      FOUR_C_THROW("the line has nearly zero length: {}", line_norm);
    }

    if (s->shape() != Core::FE::CellType::tri3)
    {
      std::cout << "HERE !tri3 cutsides are used!!!" << std::endl;
      //      FOUR_C_THROW("expect only tri3 cutsides!");
    }

    // tri3/quad4 element center
    Core::LinAlg::Matrix<2, 1> rs(Core::LinAlg::Initialization::zero);

    if (s->shape() == Core::FE::CellType::tri3)
    {
      rs = Core::FE::get_local_center_position<2>(Core::FE::CellType::tri3);
    }
    else if (s->shape() == Core::FE::CellType::quad4)
    {
      rs = Core::FE::get_local_center_position<2>(Core::FE::CellType::quad4);
    }
    else
      FOUR_C_THROW("unsupported side-shape");

    Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);
    s->normal(rs, normal);

    // check for angle=+-90 between line and normal
    if (fabs(normal.dot(line)) < (0.015 + BASICTOL)) return true;
  }

  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Element::is_cut()
{
  /* count the number cut-sides for which the intersection of the side with the
   * element finds cut points note that also elements which are just touched by
   * a cut side at points, edges or on an element's side have the status
   * IsCut = true */
  if (cut_faces_.size() > 0)
  {
    return true;
  }

  // loop the element sides
  for (std::vector<Side*>::const_iterator i = sides().begin(); i != sides().end(); ++i)
  {
    Side& side = **i;
    /* side is cut if it has more than one facet, or when the unique facet is
     * created by a cut side (touched case) */
    if (side.is_cut())
    {
      return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Cut::Element::on_side(Facet* f)
{
  if (not f->has_holes())
  {
    return on_side(f->points());
  }
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Cut::Element::on_side(const std::vector<Point*>& facet_points)
{
  const std::vector<Node*>& nodes = Element::nodes();
  for (std::vector<Point*>::const_iterator i = facet_points.begin(); i != facet_points.end(); ++i)
  {
    Point* p = *i;
    if (not p->nodal_point(nodes))
    {
      return false;
    }
  }

  PointSet points;
  std::copy(facet_points.begin(), facet_points.end(), std::inserter(points, points.begin()));

  for (std::vector<Side*>::const_iterator i = sides().begin(); i != sides().end(); ++i)
  {
    Side& side = **i;
    if (side.on_side(points))
    {
      return true;
    }
  }

  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Cut::Element::get_integration_cells(plain_integrationcell_set& cells)
{
  FOUR_C_THROW("be aware of using this function! Read comment!");

  /* for non-Tessellation approaches there are no integration cells stored,
   * do you want to have all cells or sorted by position? */
  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* vc = *i;
    vc->get_integration_cells(cells);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Cut::Element::get_boundary_cells(plain_boundarycell_set& bcells)
{
  FOUR_C_THROW(
      "Be aware of using this function! \n\n"
      "When asking the element for boundary cells it is questionable which \n"
      "cells you want to have, for Tesselation boundary cells are stored for\n"
      "each volumecell (inside and outside) independently, the f->GetBoundaryCells\n"
      "then return the bcs just for the first vc stored. For DirectDivergence\n"
      "bcs are created just for outside vcs and therefore the return of\n"
      "f->GetBoundaryCells does not work properly as it can happen that the first\n"
      "vc of the facet is an inside vc which does not store the bcs. We have to\n"
      "restructure the storage of bcs. bcs should be stored unique! for each\n"
      "cut-facet and if necessary also for non-cut facets between elements. The\n"
      "storage of boundary-cells to the volume-cells is not right way to do this!");

  for (plain_facet_set::iterator i = facets_.begin(); i != facets_.end(); ++i)
  {
    Facet* f = *i;
    if (cut_faces_.count(f->parent_side()) != 0)
    {
      f->get_boundary_cells(bcells);
    }
  }
}

/*------------------------------------------------------------------------------------------*
 * Get cutpoints of this element, returns also all touch-points
 * (Remark: be aware of the fact, that you will just get cut_points, which lie on an edge of
 * this element!!!)
 *------------------------------------------------------------------------------------------*/
void Cut::Element::get_cut_points(PointSet& cut_points)
{
  for (std::vector<Side*>::const_iterator i = sides().begin(); i != sides().end(); ++i)
  {
    Side* side = *i;

    for (plain_side_set::iterator i = cut_faces_.begin(); i != cut_faces_.end(); ++i)
    {
      Side* other = *i;
      side->get_cut_points(this, *other, cut_points);
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Cut::Element::create_integration_cells(Mesh& mesh, int count, bool tetcellsonly)
{
  /* Is volume cell active? ( i.e. in recursive call, has this vc already
   * been removed in fix_broken_tets() ) */
  if (not active_) return;

  // check the options for 1-D
  if (n_dim() == 1)
  {
    if (tetcellsonly or (not mesh.create_options().simple_shapes()))
      FOUR_C_THROW(
          "You have to switch \"tetcellsonly\" OFF and "
          "\"SimpleShapes()\" ON for the 1-D integration cell creation!");
  }

  if (not tetcellsonly)
  {
    // try to create one single simple shaped integration cell if possible
    if (create_simple_shaped_integration_cells(mesh)) return;
    // return if this was possible
  }

  eleinttype_ = Cut::EleIntType_Tessellation;

  if (not tetcellsonly)
  {
    if (mesh.create_options().simple_shapes())  // try to create only simple-shaped integration
                                                // cells for all! volumecells
    {
      if (IntegrationCellCreator::create_cells(
              mesh, this, cells_))  // Does not help for cuts with a "tri"
      {
        calculate_volume_of_cells_tessellation();
        return;  // return if this was possible
      }
    }
  }

  PointSet cut_points;

  // There are never holes in a cut facet. Furthermore, cut facets are
  // always convex, as all elements and sides are convex. Thus, we are free
  // to triangulate all cut facets. This needs to be done, so repeated cuts
  // work in the right way.

  for (plain_facet_set::iterator i = facets_.begin(); i != facets_.end(); ++i)
  {
    Facet* f = *i;
    if (f->on_boundary_cell_side() and f->has_holes())
      FOUR_C_THROW("no holes in cut facet possible");
    f->get_all_points(
        mesh, cut_points, f->belongs_to_level_set_side() and f->on_boundary_cell_side());
  }

  std::vector<Point*> points;
  points.reserve(cut_points.size());
  points.assign(cut_points.begin(), cut_points.end());

  // sort points that go into qhull to obtain the same result independent of
  // pointer values (compiler flags, code structure, memory usage, ...)
  std::sort(points.begin(), points.end(), PointPidLess());

  // standard subtetrahedralization starts here, there also the boundary cells will be created
  TetMesh tetmesh(points, facets_, false);
  tetmesh.create_element_tets(mesh, this, cells_, cut_faces_, count, tetcellsonly);

  calculate_volume_of_cells_tessellation();
}

/* Can a simple shaped integration cells be formed for this element?
 * I.e. is the element un-cut???
 */
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Cut::Element::create_simple_shaped_integration_cells(Mesh& mesh)
{
  // TEUCHOS_FUNC_TIME_MONITOR( "Cut::Element::create_simple_shaped_integration_cells" );


  if (cells_.size() == 1)  // in case there is only one volumecell, check if a simple shaped
                           // integration cell is possible
  {
    VolumeCell* vc = *cells_.begin();
    if (IntegrationCellCreator::create_cell(mesh, shape(), vc))
    {
      calculate_volume_of_cells_tessellation();

      //      // check if the unique integration cell equals the whole (sub-)element
      //      plain_integrationcell_set intcells;
      //      vc->GetIntegrationCells( intcells );
      //      if(intcells.size() != 1) FOUR_C_THROW("there is not a unique integration cell");
      //      if(this->Shape() == intcells[0]->Shape())
      //      {
      //        Core::LinAlg::SerialDenseMatrix xyze(3, intcells[0]->Points().size());
      //        this->coordinates(xyze.data());
      //
      //        double vol_diff = vc->Volume() - Core::Geo::ElementVolume( this->Shape(), xyze );
      //
      //        if(fabs(vol_diff)<1e-14)
      //        {
      //          eleinttype_ = Cut::EleIntType_StandardUncut;
      //          return true;
      //        }
      //      }

      // simple integration cells could be created, however, does not equal the element itself
      eleinttype_ = Cut::EleIntType_Tessellation;
      return true;  // return if this was possible
    }
  }



  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Element::remove_empty_volume_cells()
{
  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end();)
  {
    VolumeCell* vc = *i;
    if (vc->empty())
    {
      vc->disconnect();
      set_erase(cells_, i);
    }
    else
    {
      ++i;
    }
  }
}

/*----------------------------------------------------------------------------*
 * Create volumecells
 *----------------------------------------------------------------------------*/
void Cut::Element::make_volume_cells(Mesh& mesh)
{
  std::shared_ptr<FacetGraph> fg = FacetGraph::create(sides_, facets_);
  fg->create_volume_cells(mesh, this, cells_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType elementtype, unsigned num_nodes_element,
    unsigned dim>
bool Cut::ConcreteElement<probdim, elementtype, num_nodes_element, dim>::point_inside(Point* p)
{
  std::shared_ptr<Position> pos = Position::create(*this, *p);
  return pos->compute();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType elementtype, unsigned num_nodes_element,
    unsigned dim>
bool Cut::ConcreteElement<probdim, elementtype, num_nodes_element, dim>::local_coordinates(
    const Core::LinAlg::Matrix<probdim, 1>& xyz, Core::LinAlg::Matrix<dim, 1>& rst)
{
  std::shared_ptr<Position> pos = PositionFactory::build_position<probdim, elementtype>(*this, xyz);
  bool success = pos->compute();
  pos->local_coordinates(rst);
  return success;
}


/*----------------------------------------------------------------------------*
 * Find local coordinates of given point w.r. to the parent Quad element
 *                                                      sudhakar 11/13
 *----------------------------------------------------------------------------*/
void Cut::Element::local_coordinates_quad(
    const Core::LinAlg::Matrix<3, 1>& xyz, Core::LinAlg::Matrix<3, 1>& rst)
{
  if (not is_shadow_) FOUR_C_THROW("This is not a shadow element\n");

  std::shared_ptr<Position> pos = Position::create(quad_corners_, xyz, get_quad_shape());

  bool success = pos->compute();
  if (success)
  {
  }

  pos->local_coordinates(rst);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Cut::Element::num_gauss_points(Core::FE::CellType shape)
{
  int numgp = 0;
  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* vc = *i;
    numgp += vc->num_gauss_points(shape);
  }
  return numgp;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Element::debug_dump()
{
  std::cout << "Problem in element " << id() << " of shape "
            << Core::FE::cell_type_to_string(shape()) << ":\n";
  bool haslevelsetside = false;
  const std::vector<Node*>& nodes = Element::nodes();
  for (std::vector<Node*>::const_iterator i = nodes.begin(); i != nodes.end(); ++i)
  {
    Node* n = *i;
    // std::cout << n->LSV();
    n->plot(std::cout);
  }
  std::cout << "\n";
  const plain_side_set& cutsides = cut_sides();
  for (plain_side_set::const_iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* s = *i;
    // s->print();
    if (s->is_level_set_side()) haslevelsetside = true;
    const std::vector<Node*>& side_nodes = s->nodes();
    for (std::vector<Node*>::const_iterator i = side_nodes.begin(); i != side_nodes.end(); ++i)
    {
      Node* n = *i;
      n->plot(std::cout);
    }
    std::cout << "\n";
  }

  gmsh_failure_element_dump();

  {
    // Write Element Cut Test!!!
    std::stringstream str;
    str << "cut_test_generated_" << id() << ".cpp";
    std::ofstream file(str.str().c_str());
    Cut::Output::gmsh_element_cut_test(file, this, haslevelsetside);
  }
}

/*----------------------------------------------------------------------*
 * When cut library is broken, write complete cut        sudhakar 06/14
 * configuration in to gmsh output file
 *----------------------------------------------------------------------*/
void Cut::Element::gmsh_failure_element_dump()
{
  std::stringstream str;
  str << ".cut_element" << id() << "_CUTFAIL.pos";
  std::string filename(Cut::Output::generate_gmsh_output_filename(str.str()));
  std::ofstream file(filename.c_str());

  Cut::Output::gmsh_complete_cut_element(file, this);
  file.close();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Element::gnuplot_dump()
{
  std::stringstream str;
  str << "element" << id() << ".plot";
  std::ofstream file(str.str().c_str());

  plain_edge_set all_edges;

  const std::vector<Side*>& sides = Element::sides();
  for (std::vector<Side*>::const_iterator i = sides.begin(); i != sides.end(); ++i)
  {
    Side* s = *i;
    const std::vector<Edge*>& edges = s->edges();

    std::copy(edges.begin(), edges.end(), std::inserter(all_edges, all_edges.begin()));
  }

  for (plain_edge_set::iterator i = all_edges.begin(); i != all_edges.end(); ++i)
  {
    Edge* e = *i;
    e->begin_node()->point()->plot(file);
    e->end_node()->point()->plot(file);
    file << "\n\n";
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Element::dump_facets()
{
  std::stringstream str;
  str << "facets" << id() << ".plot";
  std::string name = str.str();

  std::cout << "write '" << name << "'\n";
  std::ofstream file(name.c_str());

  for (plain_facet_set::iterator i = facets_.begin(); i != facets_.end(); ++i)
  {
    Facet* f = *i;
    f->print(file);
  }
}

/*----------------------------------------------------------------------*
 * Calculate volume of all volumecells when Tessellation is used
 *----------------------------------------------------------------------*/
void Cut::Element::calculate_volume_of_cells_tessellation()
{
  const plain_volumecell_set& volcells = volume_cells();
  for (plain_volumecell_set::const_iterator i = volcells.begin(); i != volcells.end(); i++)
  {
    VolumeCell* vc1 = *i;
    plain_integrationcell_set ics;
    vc1->get_integration_cells(ics);

    double volume = 0;
    for (plain_integrationcell_set::iterator j = ics.begin(); j != ics.end(); ++j)
    {
      IntegrationCell* ic = *j;
      volume += ic->volume();
    }

    vc1->set_volume(volume);
  }
}
/*----------------------------------------------------------------------------*
 * Integrate pre-defined functions over each volumecell created from this
 * element when using Tessellation
 *----------------------------------------------------------------------------*/
void Cut::Element::integrate_specific_functions_tessellation()
{
  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); i++)
  {
    VolumeCell* cell1 = *i;
    cell1->integrate_specific_functions_tessellation();
  }
}

/*----------------------------------------------------------------------------*
 * The Gauss rules for each cut element is constructed by performing moment
 * fitting for each volumecells. Unless specified moment fitting is performed
 * only for cells placed in the fluid region
 *----------------------------------------------------------------------------*/
void Cut::Element::moment_fit_gauss_weights(
    Mesh& mesh, bool include_inner, Cut::BCellGaussPts Bcellgausstype)
{
  if (not active_) return;

  // try to create one single simple shaped integration cell if possible
  if (create_simple_shaped_integration_cells(mesh)) return;
  // return if this was possible

  // When the cut side touches the element the shape of the element is retained
  /*if(cells_.size()==1)
   {
   VolumeCell * vc = *cells_.begin();
   if ( IntegrationCellCreator::CreateCell( mesh, Shape(), vc ) )
   {
   return;
   }
   }*/

  /* if ( mesh.CreateOptions().SimpleShapes() )
   {
   if ( IntegrationCellCreator::CreateCells( mesh, this, cells_ ) )
   {
   return;
   }
   }*/

  eleinttype_ = Cut::EleIntType_MomentFitting;

  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); i++)
  {
    VolumeCell* cell1 = *i;
    cell1->moment_fit_gauss_weights(this, mesh, include_inner, Bcellgausstype);
  }
}

/*----------------------------------------------------------------------------*
 * The Gauss rules for each cut element is constructed by triangulating the
 * facets and applying divergence theorem. Unless specified moment fitting is
 * performed only for cells placed in the fluid region
 *----------------------------------------------------------------------------*/
void Cut::Element::direct_divergence_gauss_rule(
    Mesh& mesh, bool include_inner, Cut::BCellGaussPts Bcellgausstype)
{
  if (not active_) return;

  // try to create one single simple shaped integration cell if possible
  if (create_simple_shaped_integration_cells(mesh)) return;
  // return if this was possible

  eleinttype_ = Cut::EleIntType_DirectDivergence;

  for (plain_volumecell_set::iterator i = cells_.begin(); i != cells_.end(); i++)
  {
    VolumeCell* cell1 = *i;
    cell1->direct_divergence_gauss_rule(this, mesh, include_inner, Bcellgausstype);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Element::has_level_set_side()
{
  const plain_facet_set facets = this->facets();
  for (plain_facet_set::const_iterator j = facets.begin(); j != facets.end(); j++)
  {
    Facet* facet = *j;
    if (facet->belongs_to_level_set_side())
    {
      return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Cut::Element> Cut::ElementFactory::create_element(Core::FE::CellType elementtype,
    int eid, const std::vector<Side*>& sides, const std::vector<Node*>& nodes, bool active) const
{
  std::shared_ptr<Element> e = nullptr;
  const int probdim = Global::Problem::instance()->n_dim();
  switch (elementtype)
  {
    case Core::FE::CellType::line2:
      e = std::shared_ptr<Element>(
          create_concrete_element<Core::FE::CellType::line2>(eid, sides, nodes, active, probdim));
      break;
    case Core::FE::CellType::tri3:
      e = std::shared_ptr<Element>(
          create_concrete_element<Core::FE::CellType::tri3>(eid, sides, nodes, active, probdim));
      break;
    case Core::FE::CellType::quad4:
      e = std::shared_ptr<Element>(
          create_concrete_element<Core::FE::CellType::quad4>(eid, sides, nodes, active, probdim));
      break;
    case Core::FE::CellType::tet4:
      e = std::shared_ptr<Element>(
          create_concrete_element<Core::FE::CellType::tet4>(eid, sides, nodes, active, probdim));
      break;
    case Core::FE::CellType::hex8:
      e = std::shared_ptr<Element>(
          create_concrete_element<Core::FE::CellType::hex8>(eid, sides, nodes, active, probdim));
      break;
    case Core::FE::CellType::pyramid5:
      e = std::shared_ptr<Element>(create_concrete_element<Core::FE::CellType::pyramid5>(
          eid, sides, nodes, active, probdim));
      break;
    case Core::FE::CellType::wedge6:
      e = std::shared_ptr<Element>(
          create_concrete_element<Core::FE::CellType::wedge6>(eid, sides, nodes, active, probdim));
      break;
    default:
    {
      FOUR_C_THROW("Unsupported element type! ( {} )", Core::FE::cell_type_to_string(elementtype));
      break;
    }
  }
  return e;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Cut::Element> Cut::Element::create(const Core::FE::CellType& elementtype,
    const int& eid, const std::vector<Side*>& sides, const std::vector<Node*>& nodes,
    const bool& active)
{
  ElementFactory factory;
  return factory.create_element(elementtype, eid, sides, nodes, active);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Cut::Element> Cut::Element::create(const unsigned& shardskey, const int& eid,
    const std::vector<Side*>& sides, const std::vector<Node*>& nodes, const bool& active)
{
  return create(Core::Elements::shards_key_to_dis_type(shardskey), eid, sides, nodes, active);
}

template class Cut::ConcreteElement<2, Core::FE::CellType::line2>;
template class Cut::ConcreteElement<3, Core::FE::CellType::line2>;
template class Cut::ConcreteElement<2, Core::FE::CellType::tri3>;
template class Cut::ConcreteElement<3, Core::FE::CellType::tri3>;
template class Cut::ConcreteElement<2, Core::FE::CellType::quad4>;
template class Cut::ConcreteElement<3, Core::FE::CellType::quad4>;
template class Cut::ConcreteElement<3, Core::FE::CellType::hex8>;
template class Cut::ConcreteElement<3, Core::FE::CellType::pyramid5>;
template class Cut::ConcreteElement<3, Core::FE::CellType::wedge6>;

FOUR_C_NAMESPACE_CLOSE
