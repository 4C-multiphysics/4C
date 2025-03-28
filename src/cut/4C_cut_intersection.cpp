// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_intersection.hpp"

#include "4C_cut_boundingbox.hpp"
#include "4C_cut_position.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_utils.hpp"
#include "4C_global_data.hpp"

FOUR_C_NAMESPACE_OPEN

// whether to perform triangulation during intersection
#define TRIANGULATED_INTERSECTION true

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
std::shared_ptr<Cut::IntersectionBase> Cut::IntersectionBase::create(
    const Core::FE::CellType& edge_type, const Core::FE::CellType& side_type)
{
  const IntersectionFactory factory;
  return factory.create_intersection(edge_type, side_type);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
void Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::set_coordinates()
{
  get_edge().coordinates(xyze_lineElement_);
  get_side().coordinates(xyze_surfaceElement_);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::compute_cut(Edge* sedge, Edge* eedge, Side* side, PointSet& ee_cut_points,
    double& tolerance)
{
  return eedge->compute_cut(get_mesh_ptr(), sedge, side, &ee_cut_points, tolerance);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
void Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::get_connectivity_info(const Core::LinAlg::Matrix<probdim, 1>& xreal,
    const std::vector<int>& touched_edges_ids, std::set<std::pair<Side*, Edge*>>& out)
{
  const std::vector<Edge*> side_edges = get_side().edges();
  for (std::vector<int>::const_iterator i = touched_edges_ids.begin(); i != touched_edges_ids.end();
      ++i)
  {
    // mapping indices to real edges
    Edge* e = side_edges[*i];

    // if this edge cuts other edge in this point, this edge cuts all sides of the edge there
    const plain_side_set& touched_edge_sides = e->sides();
    for (plain_side_set::const_iterator j = touched_edge_sides.begin();
        j != touched_edge_sides.end(); ++j)
    {
      Side* s = *j;
      out.insert(std::make_pair(s, &get_edge()));
    }

    // also add connection of < touched edge x sides of the cut_edge >
    const plain_side_set& cut_edge_sides = get_edge().sides();
    for (plain_side_set::const_iterator j = cut_edge_sides.begin(); j != cut_edge_sides.end(); ++j)
    {
      Side* s = *j;
      out.insert(std::make_pair(s, e));
    }
  }
}

template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
void Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::add_connectivity_info(Point* p, const Core::LinAlg::Matrix<probdim, 1>& xreal,
    const std::vector<int>& touched_edges_ids,
    const std::set<std::pair<Side*, Edge*>>& touched_cut_pairs)
{
  // original intersection pair
  std::pair<Side*, Edge*> or_int_pair = std::make_pair(&get_side(), &get_edge());
  const std::vector<Edge*>& side_edges = get_side().edges();
  std::set<Edge*> touched_edges;

  for (std::vector<int>::const_iterator i = touched_edges_ids.begin(); i != touched_edges_ids.end();
      ++i)
  {
    // mapping indices to real edges
    Edge* e = side_edges[*i];
    touched_edges.insert(e);

    // NOTE: This might cause some problems in the future, due to the way p->t is used
    if (probdim == 3)
    {
      Core::LinAlg::Matrix<3, 1> A;
      for (unsigned int cnt = 0; cnt < probdim; ++cnt) A(cnt, 0) = xreal(cnt, 0);
      p->t(e, A);
    }
    else
    {
      FOUR_C_THROW("Not supported");
    }
    p->add_edge(e);
  }

  for (std::set<std::pair<Side*, Edge*>>::const_iterator it = touched_cut_pairs.begin();
      it != touched_cut_pairs.end(); ++it)
  {
#if CUT_CREATION_INFO
    std::stringstream msg;
    if (touched_edges.find((*it).second) != touched_edges.end())
      msg << "Pair added because during intersection of side" << (or_int_pair.first)->Id()
          << " and edge " << (or_int_pair.second)->Id() << " edge " << (*it).second->Id()
          << " is a touching edge. Hence it cuts all sides of cut edge";
    else
      msg << "Pair added because during intersection of side" << (or_int_pair.first)->Id()
          << " and edge " << (or_int_pair.second)->Id() << " side " << (*it).first->Id()
          << " is a side of touching edge. Hence cut_edge cuts it";
    p->AddCreationInfo(*it, msg.str());
#endif
    //  p->AddEdge((*it).second);
    p->add_side((*it).first);
    p->add_pair((*it).first, (*it).second, or_int_pair);
  }
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::check_bounding_box_overlap()
{
  std::shared_ptr<BoundingBox> sbb(BoundingBox::create(get_side()));
  std::shared_ptr<BoundingBox> ebb(BoundingBox::create(get_edge()));

  return check_bounding_box_overlap(*sbb, *ebb);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::check_bounding_box_overlap(BoundingBox& ebb, BoundingBox& sbb) const
{
  return (not sbb.within(1.0, ebb));
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::check_parallelism(std::vector<Core::LinAlg::Matrix<dimside, 1>>&
                                           side_rs_intersect,
    std::vector<Core::LinAlg::Matrix<dimedge, 1>>& edge_r_intersect, double& tolerance)
{
  switch (dimside)
  {
    case 1:
    {
      // NOTE: Edge-Edge parallelism is directly handled during ComputeEdgeIntersection
      return false;
    }
    case 2:
    {
      return check_parallelism_between_side_and_edge(
          side_rs_intersect, edge_r_intersect, tolerance);
    }
    default:
    {
      FOUR_C_THROW(
          "The given side element type is currently unsupported! \n"
          "( dim = {} | sideType = {} ",
          dimside, Core::FE::cell_type_to_string(sidetype).c_str());
      exit(EXIT_FAILURE);
    }
  }
  // this cannot be reached
  exit(EXIT_FAILURE);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::
    check_parallelism_between_side_and_edge(
        std::vector<Core::LinAlg::Matrix<dimside, 1>>& side_rs_intersect,
        std::vector<Core::LinAlg::Matrix<dimedge, 1>>& edge_r_intersect, double& tolerance)
{
  side_rs_intersect.clear();
  edge_r_intersect.clear();
  tolerance = 0.0;

  //  if ( check_angle_criterion_between_side_normal_and_edge() )
  //    FOUR_C_THROW("This shouldn't happen at this point! -- hiermeier");

  return false;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::check_angle_criterion_between_side_normal_and_edge()
{
  // calculate the normal at the side element center
  const Core::LinAlg::Matrix<dimside, 1> rst_side_center(
      Core::FE::get_local_center_position<dimside>(sidetype));

  Core::LinAlg::Matrix<probdim, num_nodes_side> side_deriv1;
  Core::LinAlg::Matrix<probdim, probdim> xjm;
  Core::LinAlg::Matrix<probdim, 1> normal_center;
  Cut::eval_derivs_in_parameter_space<probdim, sidetype, double>(xyze_surfaceElement_,
      rst_side_center, side_deriv1, xjm, nullptr, &normal_center, nullptr, true);

  // calculate the direction vector of the edge element
  Core::LinAlg::Matrix<probdim, 1> dedge(&xyze_lineElement_(0, 0), false);
  const Core::LinAlg::Matrix<probdim, 1> e_endpoint(&xyze_lineElement_(0, 1), true);
  dedge.update(-1.0, e_endpoint, 1.0);
  const double e_nrm2 = dedge.norm2();
  if (e_nrm2 == 0.0)
    FOUR_C_THROW("The 1-st edge length is zero!");
  else
    dedge.scale(1.0 / e_nrm2);

  // calculate the inner product and check the angle between the normal and
  // the edge

  const double inner_product = dedge.dot(normal_center);
  /* If the angle between the normal and the edge is smaller than 89 degrees
   * ( cos( 89 degrees ) = 0.017472406... ), the vectors are definitely not parallel,
   * otherwise it's possible and we do another test. Note, that both vectors have
   * unit length! */
  {
    double a = std::acos(std::abs(inner_product)) * 180 / (2.0 * std::acos(0.0));
    std::cout << "angle between side normal and edge = " << a << "degrees" << std::endl;
  }
  return (std::abs(inner_product) <= 0.01747);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::check_collinearity(std::vector<Core::LinAlg::Matrix<dimside, 1>>&
                                            side_rs_corner_intersect,
    std::vector<Core::LinAlg::Matrix<dimedge, 1>>& edge_r_corner_intersect, double& tolerance)
{
  FOUR_C_THROW("Is this used?");
  if (num_nodes_edge != 2 or num_nodes_side != 2)
    FOUR_C_THROW(
        "Two line2 elements are expected, but instead a {} (edge) and {} (side) "
        "element were given.",
        Core::FE::cell_type_to_string(edgetype).c_str(),
        Core::FE::cell_type_to_string(sidetype).c_str());

  side_rs_corner_intersect.clear();
  edge_r_corner_intersect.clear();


  // quick check first -- check angle between the two edges, if the check fails,
  // the two edges are definitely not parallel.
  if (not check_angle_criterion_between_two_edges()) return false;

  /* If the lines seem to be parallel, we do the expensive calculation and
   * check if the end points of the edge are on the side edge ( and are collinear ). */
  bool is_collinear = true;

  std::vector<Core::LinAlg::Matrix<probdim, 1>> side_xyz_corner_intersect;
  side_xyz_corner_intersect.reserve(2);

  side_rs_corner_intersect.reserve(2);
  edge_r_corner_intersect.reserve(2);

  Core::LinAlg::Matrix<2, 1, Core::LinAlg::Matrix<2, 1>> e_corner_distance;
  for (unsigned i = 0; i < 2; ++i)
  {
    const Core::LinAlg::Matrix<probdim, 1> e_cornerpoint(&xyze_lineElement_(0, i), true);
    std::shared_ptr<Cut::Position> pos =
        Cut::Position::create(xyze_surfaceElement_, e_cornerpoint, sidetype);

    bool withinlimits = pos->compute(true);

    pos->distance(e_corner_distance(i));
    const double pos_tolerance = pos->newton_tolerance();

    // keep the largest up-coming tolerance
    if (i == 0 or tolerance < pos_tolerance) tolerance = pos_tolerance;

    if (e_corner_distance(i).norm2() < pos_tolerance)
    {
      // the edge end point seems to lie on the side edge, and the end-point is
      // also within the side limits
      if (withinlimits)
      {
        side_xyz_corner_intersect.push_back(e_cornerpoint);
        side_rs_corner_intersect.push_back(
            Core::LinAlg::Matrix<dimside, 1>(Core::LinAlg::Initialization::zero));
        pos->local_coordinates(*(side_rs_corner_intersect.end() - 1));
      }
    }
    // if one distance is larger than the given tolerance, the edge cannot be
    // collinear
    else
      is_collinear = false;
  }
  // set number of cut points
  num_cut_points_ = side_rs_corner_intersect.size();

  // The two edges are collinear and we found no end points of the edge on
  // the side.
  // --> No more intersections possible for two edges besides the already
  // existing end-points of the side edge.
  if (is_collinear and side_rs_corner_intersect.size() == 0)
  {
    return true;
  }

  // End points of the edge on the side edge. In this case
  // we are done, as well, and it doesn't matter if the two objects
  // are parallel or not.
  if (num_cut_points_ > 0)
  {
    for (unsigned i = 0; i < num_cut_points_; ++i)
    {
      xsi_side_.update(side_rs_corner_intersect[0]);
      if (not find_local_coordinate_of_edge_end_point(
              xsi_edge_(0), side_xyz_corner_intersect[i], tolerance))
        FOUR_C_THROW("We couldn't find the correct edge end-point!");
      if (num_cut_points_ > 1)
        edge_r_corner_intersect.push_back(
            Core::LinAlg::Matrix<dimedge, 1>(xsi_edge_.data(), false));

      // safety
      if (not(line_within_limits() and surface_within_limits()))
        FOUR_C_THROW("Something went wrong!");
    }

    return true;
  }

  if (not is_collinear)
  {
    double prod = 0.0;
    // check the signs of the distance values
    for (unsigned i = 0; i < (probdim - dimside); ++i)
    {
      prod = e_corner_distance(0)(i) * e_corner_distance(1)(i);
      // if the off-set direction is exactly in one of the normal directions, we
      // use the second criterion to detect numerical artifacts ( values close
      // to zero but different signs )
      if ((prod < 0.0) and std::abs(e_corner_distance(0)(i)) > tolerance and
          std::abs(e_corner_distance(1)(i)) > tolerance)
      {
        // there is a intersection possible, although the lines are almost
        // parallel
        return false;
      }
    }
    // all distance values have the same sign ( or are close to zero ) and the
    // edges are not collinear
    // --> no intersection possible
    return true;
  }

  return false;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::check_angle_criterion_between_two_edges()
{
  Core::LinAlg::Matrix<probdim, 1> deedge(&xyze_lineElement_(0, 0), false);
  Core::LinAlg::Matrix<probdim, 1> dsedge(&xyze_surfaceElement_(0, 0), false);

  const Core::LinAlg::Matrix<probdim, 1> e_endpoint(&xyze_lineElement_(0, 1), true);
  deedge.update(-1.0, e_endpoint, 1.0);
  const double e_nrm2 = deedge.norm2();
  if (e_nrm2 == 0.0) FOUR_C_THROW("The 1-st edge length is zero!");

  const Core::LinAlg::Matrix<probdim, 1> s_endpoint(&xyze_surfaceElement_(0, 1), true);
  dsedge.update(-1.0, s_endpoint, 1.0);
  const double s_nrm2 = dsedge.norm2();
  if (s_nrm2 == 0.0) FOUR_C_THROW("The 2-nd edge length is zero!");

  const double inner_product = deedge.dot(dsedge);
  /* If the angle between the two lines is larger than 1 degree ( cos( 1 degree ) = 0.99866695... ),
   * the two edges are definitely not parallel, otherwise it's possible and
   * we take a closer look. */
  if (debug)
  {
    std::cout << e_nrm2 << ", " << s_nrm2 << ", " << inner_product << std::endl;
    double a = std::abs(inner_product) / (e_nrm2 * s_nrm2);
    a = std::acos((a > 1.0 ? 1.0 : a)) * 180 / (2.0 * std::acos(0.0));
    std::cout << "angle between edge and edge = " << a << "degrees" << std::endl;
  }
  return (std::abs(inner_product) >= e_nrm2 * s_nrm2 * 0.9998);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::find_local_coordinate_of_edge_end_point(double& pos,
    const Core::LinAlg::Matrix<probdim, 1>& xyz, const double& tolerance) const
{
  const std::array<double, 2> r_endpoints = {-1.0, 1.0};
  Core::LinAlg::Matrix<probdim, 1> dist;
  Core::LinAlg::Matrix<num_nodes_edge, 1> lineFunct;

  for (unsigned i = 0; i < 2; ++i)
  {
    pos = r_endpoints[i];

    Core::FE::shape_function_1d(lineFunct, pos, edgetype);
    dist.multiply_nn(xyze_lineElement_, lineFunct);

    dist.update(1.0, xyz, -1.0);

    if (dist.norm2() < tolerance) return true;
  }
  return false;
}

template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
Cut::ParallelIntersectionStatus Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge,
    dimside, num_nodes_edge, num_nodes_side>::handle_parallel_intersection(PointSet& cuts,
    int skip_id, bool output)
{
  std::array<Node*, 2> id_to_node = {get_edge().begin_node(), get_edge().end_node()};
  int id_start = 0, id_end = 2;
  int count_inside = 0;  // number of points inside the surface

  bool found_intersections = false;

  bool zeroarea = false;
  // local coordinates are inside element
  Core::LinAlg::Matrix<2, 1, bool> lineendpoint_within_surfacelimits(
      Core::LinAlg::Initialization::zero);
  // point is really inside element (normal distance = 0)
  Core::LinAlg::Matrix<2, 1, bool> lineendpoint_in_surface(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<2, 1> lineendpoint_dist;
  Core::LinAlg::Matrix<2, 1> lineendpoint_tol;
  Core::LinAlg::Matrix<2, 1, bool> lineendpoint_conv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<dimside + dimedge, 2> lineendpoint_xsi(Core::LinAlg::Initialization::zero);
  std::vector<std::vector<int>> lineendpoint_touched_edges(2);
  std::vector<Kernel::PointOnSurfaceLoc> lineendpoint_location_kernel(2);

  // if one intersection point was found already for this cut_side & edge, and we are searching for
  // another one
  if (skip_id != -1)
  {
    if (skip_id == 0)
      id_start = 1;
    else if (skip_id == 1)
      id_end = 1;
    else
      FOUR_C_THROW(
          "Trying to skip id = {} of the edge nodes. You can only skip BeginNode(0) or "
          "EndNode(1)\n",
          skip_id);

    lineendpoint_conv(skip_id) = true;
    lineendpoint_location_kernel[skip_id] = Kernel::PointOnSurfaceLoc(true, true);
    count_inside++;
    found_intersections = true;
  }

  for (int lp = id_start; lp < id_end; ++lp)
  {
    Cut::Point* actpoint = id_to_node[lp]->point();

    switch (sidetype)
    {
      // --------------------------------------------------------------------
      /* handle parallel case for TRI3 elements in the 3-dimensional space
       * handle parallel case for 1-D elements in the 2-dimensional space */
      // --------------------------------------------------------------------
      case Core::FE::CellType::tri3:
      case Core::FE::CellType::line2:
      {
        lineendpoint_conv(lp) =
            compute_distance(actpoint, lineendpoint_dist(lp), lineendpoint_tol(lp), zeroarea,
                lineendpoint_location_kernel[lp], lineendpoint_touched_edges[lp], true);
        lineendpoint_within_surfacelimits(lp) = surface_within_limits();
        // copy xsi_ in the lp column of lineendpoint_xsi
        std::copy(xsi_.data(), xsi_.data() + (dimside + dimedge), &lineendpoint_xsi(0, lp));
        break;
      }  // end: case (Core::FE::CellType::tri3) and 1-D elements
         // --------------------------------------------------------------------
         // handle parallel case for QUAD4 elements
         // --------------------------------------------------------------------
      case Core::FE::CellType::quad4:
      {
        /* problem here is that compute_distance might change the normal direction if
         * the projected point lies outside the element, that why we prefer to compute
         * the distance onto tri3! */
        Core::LinAlg::Matrix<2, 1> tri_dist;
        Core::LinAlg::Matrix<2, 1> tri_tol;
        Core::LinAlg::Matrix<2, 1, bool> tri_conv;

        std::vector<Kernel::PointOnSurfaceLoc> tri_location_kernel(2);
        std::vector<std::vector<int>> tri_touched_edges(2);

        std::array<bool, 2> extended_tri_tolerance_loc_triangle_split = {false, false};
        std::array<bool, 2> on_side = {false, false};
        std::array<bool, 2> within_side = {false, false};

        unsigned tri = 0;
        // try to calculate by splitting quad4 intro 2 triangles
        for (tri = 0; tri < 2; ++tri)
        {
          tri_conv(tri) = compute_distance(actpoint, tri_dist(tri), tri_tol(tri), zeroarea,
              tri_location_kernel[tri], tri_touched_edges[tri], true, tri,
              extended_tri_tolerance_loc_triangle_split[tri]);

          // we prefer to get on_side data from the side the point is within
          if (tri_location_kernel[tri].within_side())
          {
            within_side[tri] = true;
            on_side[tri] = tri_location_kernel[tri].on_side();
          }

          if (tri_location_kernel[tri].on_side()) on_side[tri] = true;
        }

        // if it inside quad4
        if (within_side[0] or within_side[1])
        {
          // triangle in which point was inside will be served as reference
          tri = within_side[0] ? 0 : 1;

          // safety check to make sure obtained edges are different (which they should be, since
          // diagonal one is not added)
          std::set<int> touched_edges(tri_touched_edges[0].begin(), tri_touched_edges[0].end());
          std::copy(tri_touched_edges[1].begin(), tri_touched_edges[1].end(),
              std::inserter(touched_edges, touched_edges.begin()));
          if (touched_edges.size() != (tri_touched_edges[1].size() + tri_touched_edges[0].size()))
            FOUR_C_THROW("Got duplicate edges from triangle split");

          // set up all the topology data
          std::vector<int> tri_touched_edges_full(touched_edges.begin(), touched_edges.end());
          lineendpoint_touched_edges[lp] = tri_touched_edges_full;
          lineendpoint_location_kernel[lp] = tri_location_kernel[tri];
          lineendpoint_conv(lp) = tri_conv(tri);
          lineendpoint_dist(lp) = tri_dist(tri);
          lineendpoint_tol(lp) = tri_tol(tri);
          lineendpoint_within_surfacelimits(lp) = tri_location_kernel[tri].within_side();

          std::copy(xsi_.data(), xsi_.data() + (dimside + dimedge), &lineendpoint_xsi(0, lp));
        }
        // it is outside for sure
        else
        {
          // first check if it is not close to the common edge of two triangles and we missed it
          // if the point is parallel to both tri, we needle to handle extreme case of point being
          // close to shared edge of both tri (diagonal of the quad)
          if ((tri_location_kernel[0].on_side()) and (tri_location_kernel[1].on_side()))
          {
            if (extended_tri_tolerance_loc_triangle_split[0] and
                extended_tri_tolerance_loc_triangle_split[1])
            {
              generate_gmsh_dump();
              FOUR_C_THROW(
                  "Here we need to create point on diagonal line between two triangles in "
                  "triangulated quad4! This is not yet implemented");
            }
          }

          // basically we can take any triangle now as a reference, since point is outside of both

          // reset to previous successful value
          if (on_side[0])
            tri = 0;
          else
            tri = 1;

          lineendpoint_location_kernel[lp] = tri_location_kernel[tri];
          lineendpoint_touched_edges[lp] = tri_touched_edges[tri];
          lineendpoint_conv(lp) = tri_conv(tri);
          lineendpoint_dist(lp) = tri_dist(tri);
          lineendpoint_tol(lp) = tri_tol(tri);
        }

        break;
      }  // end: case (Core::FE::CellType::quad4)
      default:
      {
        FOUR_C_THROW(
            "intersection Error: Other surfaces than TRI3 and QUAD4 are not supported "
            "to avoid huge problems!!!");
        break;
      }
    }
  }


  if (!(lineendpoint_location_kernel[0].on_side()) and
      !(lineendpoint_location_kernel[1].on_side()) and
      lineendpoint_dist(0) * lineendpoint_dist(1) > 0 and not zeroarea)
  {
    if (debug)
    {
      std::cout << "No Cut Point, because we detected following normal distance "
                   "at the line end points: ... lineendpoint_dist[0]: "
                << lineendpoint_dist(0) << " lineendpoint_dist[1]: " << lineendpoint_dist(1)
                << "(Codeline:" << __LINE__ << ")" << std::endl;
    }
    return intersection_not_possible;
  }


  if (lineendpoint_conv(0) and lineendpoint_conv(1))
  {
    // ------------------------------------------------------------------------
    // starting- and end-point of the edge are inside/on the side
    // ------------------------------------------------------------------------

    if (lineendpoint_location_kernel[0].within_side() and lineendpoint_location_kernel[0].on_side())
      lineendpoint_in_surface(0) = true;


    if (lineendpoint_location_kernel[1].within_side() and lineendpoint_location_kernel[1].on_side())
      lineendpoint_in_surface(1) = true;

    for (int lp = id_start; lp < id_end; ++lp)
    {
      if (lineendpoint_in_surface(lp))
      {
        Core::LinAlg::Matrix<probdim, 1> x;
        id_to_node[lp]->point()->coordinates(x.data());
        insert_cut(id_to_node[lp], cuts);

#if CUT_CREATION_INFO
        std::stringstream msg;
        msg << "// Added because point" << id_to_node[lp]->point()->Id() << " of the edge "
            << get_edge().Id() << " lies on the side " << get_side().Id() << std::endl;
        msg << "// Point " << id_to_node[lp]->point()->Id() << " has "
            << lineendpoint_touched_edges[lp].size() << " touching edges of the side " << std::endl;
        msg << "// Computed from compute_distance" << std::endl;
        if (not lineendpoint_within_surfacelimits(lp))
          msg << "// Is NOT within surface limits" << std::endl;
        else
          msg << "// Within surface limits" << std::endl;
        id_to_node[lp]->point()->AddCreationInfo(
            std::make_pair(&get_side(), &get_edge()), msg.str());
#endif
        // Adding connection that are close to this nodal point
        std::set<std::pair<Side*, Edge*>> topological_cut_pairs;
        get_connectivity_info(x, lineendpoint_touched_edges[lp], topological_cut_pairs);
        add_connectivity_info(
            id_to_node[lp]->point(), x, lineendpoint_touched_edges[lp], topological_cut_pairs);

        ++count_inside;
        found_intersections = true;
      }
    }

    // reset for looping over edges later
    found_intersections = false;

    // if both points inside, no need to do anything
    if (count_inside == 2)
    {
      return intersection_found;
    }

    else
    {
      // ------------------------------------------------------------------------
      /* Try to find intersection of the current edge with all edges of the side   */
      // ------------------------------------------------------------------------
      std::shared_ptr<BoundingBox> edge_bb(BoundingBox::create(get_edge()));
      const std::vector<Edge*>& side_edges = get_side().edges();
      for (std::vector<Edge*>::const_iterator i = side_edges.begin(); i != side_edges.end(); ++i)
      {
        Edge* e = *i;
        std::shared_ptr<BoundingBox> side_edge_bb(BoundingBox::create(*e));
        // skip this edge if it is too far or intersection has been done already
        if (not side_edge_bb->within(1.0, *edge_bb)) continue;

        PointSet cut_points;
        get_edge().get_cut_points(e, cut_points);
        // cut point obtained e.g. from other neighboring edge (e.g. it is end point) or from the
        // other intersection

        std::stringstream msg;
#if CUT_CREATION_INFO
        msg << "// Edge-edge intersection was happening in intersection between edge "
            << get_edge().Id() << " and "
            << "edges of the side" << get_side().Id() << "( edge " << e->Id() << ")" << std::endl;
#endif
        if (cut_points.size() > 0)
        {
          if (cut_points.size() == 1)
          {
#if CUT_CREATION_INFO
            msg << "// Edge-edge intersection is added because point" << (*cut_points.begin())->Id()
                << " was already cut by connecting edge" << std::endl;
#endif
            /* Then in order for proper cut we need to add all the necessary connectivity
             * information */
            /* NOTE: Here it might be possible to better transfer original intersection from the
             * another pair */
            (*cut_points.begin())
                ->add_edge_intersection(&get_edge(), e, &get_side(), &get_edge(), msg.str());
            /* Nothing to be done. There are already cut points between these
             * edges. We cannot find new ones. */
            cuts.insert(*(cut_points.begin()));
          }
          else if (cut_points.size() == 2)
          {
            PointSet::iterator cp = cut_points.begin();
            cuts.insert(*cp);
#if CUT_CREATION_INFO
            msg << "Edge-edge intersection is added because point" << (*cp)->Id()
                << " was already cut by connecting edge. In fact two common points were found!"
                << std::endl;
#endif
            (*cp)->add_edge_intersection(&get_edge(), e, &get_side(), &get_edge(), msg.str());
            cp++;
            cuts.insert(*cp);
#if CUT_CREATION_INFO
            msg << "Edge-edge intersection is added because point" << (*cp)->Id()
                << " was already cut by connecting edge. In fact two common points were found!"
                << std::endl;
#endif
            (*cp)->add_edge_intersection(&get_edge(), e, &get_side(), &get_edge(), msg.str());
          }
          else
          {
            if (debug)
            {
              for (PointSet::iterator i = cut_points.begin(); i != cut_points.end(); ++i)
                (*i)->print(std::cout);
              get_edge().begin_node()->point()->print(std::cout);
              get_edge().end_node()->point()->print(std::cout);
              e->begin_node()->point()->print(std::cout);
              e->end_node()->point()->print(std::cout);
            }
            FOUR_C_THROW("Two Edges have more than two cutpoint, sounds strange!");
          }
          if (debug)
          {
            std::cout << "Cut points found, by intersection of edges!"
                      << " (Codeline:" << __LINE__ << ")" << std::endl;
          }
          found_intersections = true;
        }  // if ( cut_points.size() > 0 )
        // given edge has currently no cut points
        else
        {
          double tolerance = 0.0;
          // check if the given edge intersects with one of the side edges
          if (compute_cut(e, get_edge_ptr(), get_side_ptr(), cuts, tolerance))
          {
            if (debug)
            {
              std::cout << "Cut points found, by intersection of edges (edges of side "
                           "and given edge)!"
                        << " (Codeline:" << __LINE__ << ")" << std::endl;
            }
            found_intersections = true;
          }
        }
      }  // end: loop over side edges

      if (found_intersections)
      {
        return intersection_found;
      }

      // determining return status based compute distance data, that we calculated earlier
      else
      {
        // if both nodes of the edge are on the side and there was no intersection till now, it
        // means that they lie outside and parallel to the side --> Newton method for the
        // intersection of cut_side x cut_edge  will not converge so we don't want to continue from
        // here
        /* both points outside the plane on the same side --> no intersection
         * Note: distances of both end-points have the same sign. */

        if ((lineendpoint_location_kernel[0].on_side()) and
            (lineendpoint_location_kernel[1].on_side()))
        {
          if ((not(lineendpoint_location_kernel[0].within_side())) and
              (not(lineendpoint_location_kernel[1].within_side())))
          {
            return intersection_not_possible;
          }
          // if at least one point is inside, there are two options:
          // 1. Both are inside, and this should have been definitely handled before ( count_inside
          // == 2 )
          // 2. One is inside and parallel to one tri3 split, one is outside and parallel to another
          // tri3
          //    only in such case no edge-edge intersection would happen. The quad4 is need to be
          //    slightly distorted (e.g. even 1e-16) Otherwise there would be cut point from edge
          //    edge intersection
          else
          {
            std::pair<bool, bool> is_distorted = is_quad4_distorted();
            if (not(is_distorted.first or is_distorted.second))
            {
              generate_gmsh_dump();
              FOUR_C_THROW(
                  "This case should have been handled before with edge-edge intersection!");
            }
            else
              return intersection_found;
          }
        }
        // If one point is inside and there is no intersection with edges , we are sure that nothing
        // else can be done. Otherwise there is still possibility for a cut point
        else if (count_inside == 1)
        {
          return intersection_found;
        }
        // if one point lies on the extended side, but there was no edge-edge intersection, there is
        // no other possible intersection
        else if ((lineendpoint_location_kernel[0].on_side()) or
                 (lineendpoint_location_kernel[1].on_side()))
        {
          return intersection_not_possible;
        }

        else
          return intersection_not_found;
      }
    }
  }

  else if (zeroarea)
  {
    if (debug)
    {
      std::cout << "No Cut Point, because surface has no area and edges don't have "
                   "any intersection! (Codeline:"
                << __LINE__ << ")" << std::endl;
    }
    return intersection_not_found;
  }

  return intersection_not_found;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
void Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::generate_gmsh_dump()
{
  // just to create gmsh output of the failed intersection!!!
  try
  {
    // compute intersection with no cln, just to make sure we don't fail
    Kernel::ComputeIntersection<probdim, edgetype, sidetype, false> ci(xsi_);
    ci(xyze_surfaceElement_, xyze_lineElement_);
    std::string filename(Output::generate_gmsh_output_filename(".intersection_CUTFAIL_end.pos"));
    std::ofstream file(filename.c_str());
    ci.write_to_gmsh(file);
    file.close();
  }
  catch (Core::Exception& e)
  {
    FOUR_C_THROW(
        "Cautch error in the cut_intersection:  \n{} . Current tolerance must be increased",
        e.what_with_stacktrace().c_str());
  }
}

template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::handle_special_cases()
{
  Core::LinAlg::Matrix<dimside + dimedge, 1> i_xsi;
  i_xsi = xsi_;

  /* point is outside the element and is not part of the interpolation
   * space (just for QUAD4 try triangulation) */
  if (sidetype == Core::FE::CellType::quad4)
  {
    Core::LinAlg::Matrix<2, 1> tri_conv;
    Kernel::PointOnSurfaceLoc location_status[2];
    for (unsigned tri = 0; tri < 2; ++tri)
    {
      tri_conv(tri) = compute_edge_tri3_intersection(tri, location_status[tri]);
      /* we expect the QUAD4 to converge in case that the projected point is
       * inside the QUAD4 or TRI3! */
      if (location_status[tri].within_side())
      {
        generate_gmsh_dump();
        if (is_quad4_distorted().first) std::cout << "NOTICE: Element is distorted" << std::endl;

        const std::vector<Node*>& side_nodes = get_side().nodes();
        for (std::vector<Node*>::const_iterator it = side_nodes.begin(); it != side_nodes.end();
            ++it)
        {
          (*it)->point()->dump_connectivity_info();
        }
        FOUR_C_THROW(
            "compute_edge_side_intersection for Quad4 didn't converge, "
            "but compute_edge_tri3_intersection for triangulation (id={}) is inside the Element!",
            tri);
      }
    }
    // both converged and are outside!!! ( tested directly )
    if (tri_conv(0) and tri_conv(1))
    {
      if (debug)
      {
        std::cout << "No Cut Point found by intersection with triangulated quad4! "
                     "(Codeline:"
                  << __LINE__ << ")" << std::endl;
      }
      // point outside the interpolation space of QUAD4
      return false;
    }
  }
  // If we reach here, this is probably distorted  element
  FOUR_C_THROW(
      "Distorted element. To enable support for it switch TRIANGULATED_INTERSECTION to true");
  return false;
}


template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::triangulated_intersection(PointSet& cuts)
{
  double itol = 0.0;
  switch (sidetype)
  {
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::line2:
    {
      IntersectionStatus conv;
      try
      {
        conv = compute_edge_side_intersection(itol, true);
      }
      catch (Core::Exception& e)
      {
        generate_gmsh_dump();
        FOUR_C_THROW(
            "Cautch error in cut kernel. Current tolerance must be increased! Error is: \n {} ",
            e.what_with_stacktrace().c_str());
      }

      // if newton did not fail, we trust the result
      if (conv != intersect_newton_failed)
      {
        // if the point is inside ( or inside tolerance)
        if (conv == intersect_single_cut_point)
        {
          conv = intersect_single_cut_point;
          final_point();

          Point* p = Point::new_point(
              get_mesh(), x_.data(), xsi_(dimside, 0), get_edge_ptr(), get_side_ptr(), itol);

          cuts.insert(p);
          return true;
        }
        // no intersection
        else
          return false;
      }
      else
        FOUR_C_THROW(
            "Newton did not converge for edge-side(tri3) intersection and there is not cut point!");
      break;
    }
    case Core::FE::CellType::quad4:
    {
      std::vector<Core::LinAlg::Matrix<3, 1>> final_points;
      std::vector<Core::LinAlg::Matrix<dimedge, 1>> edge_coords;
      Core::LinAlg::Matrix<dimside + dimedge, 1> i_xsi;
      std::array<bool, 2> close_to_shared_edge = {false, false};
      i_xsi = xsi_;
      IntersectionStatus tri_status[2];
      // compute intersection for each of the triangles
      int cut_point_count = 0;
      bool first_close_to_shared = false;
      for (unsigned tri = 0; tri < 2; ++tri)
      {
        try
        {
          tri_status[tri] =
              compute_edge_tri3_intersection_quad4_split(tri, &close_to_shared_edge[tri]);
        }
        catch (Core::Exception& e)
        {
          generate_gmsh_dump();
          FOUR_C_THROW(
              "Cautch error in cut kernel. Current tolerance must be increased! Error is: \n {} ",
              e.what_with_stacktrace().c_str());
        }
        if (tri_status[tri] != intersect_newton_failed)
        {
          if (tri_status[tri] == intersect_single_cut_point)
          {
            istatus_ = intersect_single_cut_point;
            final_points.push_back(Core::LinAlg::Matrix<3, 1>(final_point()));
            edge_coords.push_back(Core::LinAlg::Matrix<dimedge, 1>(xsi_edge_.data()));
            // we found cut point
            cut_point_count++;
          }
          // if have not obtained a definite point already and reasonably close to shared edge
          else if ((cut_point_count == 0) and (close_to_shared_edge[tri]))
          {
            // if first already satisfy condition and we are on tri3, and can safely create point
            if (first_close_to_shared)
            {
              if (tri == 1)
              {
                // possible reference implementation
                istatus_ = intersect_single_cut_point;
                final_points.push_back(Core::LinAlg::Matrix<3, 1>(final_point()));
                edge_coords.push_back(Core::LinAlg::Matrix<dimedge, 1>(xsi_edge_.data()));
                // we found cut point
                cut_point_count++;
              }
              else
                FOUR_C_THROW("This should not happen");
            }
            // if tri equal to 1 we just let the check go to the next triangle
            if (tri == 0) first_close_to_shared = true;
          }
        }
      }

      switch (cut_point_count)
      {
        case 0:
        {
          // no intersection
          return false;
          break;
        }
        case 1:
        {
          // intersection
          Point* p = Point::new_point(get_mesh(), final_points[0].data(), edge_coords[0](0, 0),
              get_edge_ptr(), get_side_ptr(), itol);
          cuts.insert(p);
          // update status, in case of the other triangle returning different one
          istatus_ = intersect_single_cut_point;
          return true;
          break;
        }
        case 2:
        {
          // first check if this point is not on the split between triangles (diagonal)
          Core::LinAlg::Matrix<3, 1> diff = final_points[0];
          diff.update(-1, final_points[1], 1);
          // this case is unsupported
          if (diff.norm2() > TOPOLOGICAL_TOLERANCE)
          {
            std::stringstream err_msg;
            err_msg << " Go two intersection points during intersection of triangulated quad4.  "
                       "This case is not supported! Global coordinates are \n";
            for (std::vector<Core::LinAlg::Matrix<3, 1>>::iterator it = final_points.begin();
                it != final_points.end(); ++it)
            {
              err_msg << (*it);
            }
            err_msg << "Local coordinates on the edge are \n";
            for (typename std::vector<Core::LinAlg::Matrix<dimedge, 1>>::iterator it =
                     edge_coords.begin();
                it != edge_coords.end(); ++it)
            {
              err_msg << (*it);
            }
            generate_gmsh_dump();

            FOUR_C_THROW("{}", err_msg.str());
          }
          else
          {
            Point* p = Point::new_point(get_mesh(), final_points[0].data(), edge_coords[0](0, 0),
                get_edge_ptr(), get_side_ptr(), itol);
            cuts.insert(p);
            return true;
          }
          break;
        }
        default:
          FOUR_C_THROW("This should not be possible!");
      }
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "intersection Error: Other surfaces than TRI3 and QUAD4 are not supported "
          "to avoid huge problems!!!");
      break;
    }
  }
  return false;
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::intersect(PointSet& cuts)
{
  check_init();
  // ------------------------------------------------------------------------
  //  bounding box check
  // ------------------------------------------------------------------------
  bool debugbb = check_bounding_box_overlap();
  if (debugbb)
  {
    if (debug)
    {
      std::cout << "No Cut Point, as not even the Bounding Boxes are "
                   "Overlapping! ( Codeline:"
                << __LINE__ << " )" << std::endl;
    }

    if (use_bounding_box())
    {
      return false; /* not even the bounding boxes are overlapping, no need to
                     * search for an intersection point! */
    }
    else
      FOUR_C_THROW("We should use bounding box now!");
  }

  // try to find out if this intersection can be handled just by compute_distance
  // and edge-edge intersections
  ParallelIntersectionStatus parallel_res = handle_parallel_intersection(cuts);

  if (parallel_res == intersection_found)
    return true;

  else
  {
    // we don't want to continue then, because  newton will fail
    if (parallel_res == intersection_not_possible) return false;
    // ------------------------------------------------------------------------
    // Try to calculate the intersection point directly with Newton
    // ------------------------------------------------------------------------
#if (TRIANGULATED_INTERSECTION)
    return triangulated_intersection(cuts);
#else

    double itol = 0.0;
    IntersectionStatus conv;
    try
    {
      conv = compute_edge_side_intersection(itol, true);
    }
    catch (const Core::Exception& e)
    {
      generate_gmsh_dump();
      FOUR_C_THROW(
          "Cautch error in cut kernel. Current tolerance must be increased! Error is: \n {} ",
          e.what_with_stacktrace().c_str());
    }

    // if newton did not fail, we trust the result
    if (conv != intersect_newton_failed)
    {
      // if the point is inside ( or inside tolerance)
      if (conv == intersect_single_cut_point)
      {
        conv = intersect_single_cut_point;
        FinalPoint();

        Point* p = Point::NewPoint(
            GetMesh(), x_.data(), xsi_(dimside, 0), GetEdgePtr(), GetSidePtr(), itol);

#if CUT_CREATION_INFO
        std::stringstream msg;
        msg << "//Added because of the direct side " << get_side().Id() << " and "
            << get_edge().Id() << " intersection";
        p->AddCreationInfo(std::make_pair(&get_side(), &get_edge()), msg.str());
#endif

        cuts.insert(p);
        if (debug)
        {
          std::cout << "Cut points found by intersection, with local coords: xsi: " << xsi_(0, 0)
                    << " / eta: " << xsi_(1, 0) << " / alpha: " << xsi_(2, 0)
                    << " (Codeline:" << __LINE__ << ")" << std::endl;
        }
        if (debugbb)
        {
          std::cout
              << "\n ==== \n Bounding boxes said that there is no intersection point! \n ==== \n"
              << std::endl;
          generate_gmsh_dump();
          FOUR_C_THROW("Bounding Boxes said that there is no intersection point!");
        }
        return true;
      }
      else
      {
        if (debug)
        {
          std::cout << "No Cut Point found by intersection, with local coords: xsi: " << xsi_(0, 0)
                    << " / eta: " << xsi_(1, 0) << " / alpha: " << xsi_(2, 0)
                    << " and a tolerance of " << itol << " (Codeline:" << __LINE__ << ")"
                    << std::endl;
        }
        // there is no intersection between surface and line!!!
        return false;
      }
    }
    else
    {
      // check if point is not outside quad4 by splitting quad into 2 triangles and calculating
      // distances
      return HandleSpecialCases();
    }
#endif
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
bool Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside, num_nodes_edge,
    num_nodes_side>::refined_bb_overlap_check(int maxstep)
{
  if (sidetype != Core::FE::CellType::tri3)
    FOUR_C_THROW("refined_bb_overlap_check is made for distored tri3s!");

  std::vector<Core::LinAlg::Matrix<3, 1>> surfpoints;
  std::vector<Core::LinAlg::Matrix<3, 1>> linepoints;
  Core::LinAlg::Matrix<3, 1> actpoint;
  for (unsigned nid = 0; nid < num_nodes_side; nid++)
  {
    (get_side().nodes()[nid])->coordinates(actpoint.data());
    surfpoints.push_back(actpoint);
  }
  // find shortest edge on the surface!
  int smallestedge;
  {
    Core::LinAlg::Matrix<3, 1> d1, d2, d3;
    d1.update(1.0, surfpoints[1], -1.0, surfpoints[2]);
    d2.update(1.0, surfpoints[2], -1.0, surfpoints[0]);
    d3.update(1.0, surfpoints[0], -1.0, surfpoints[1]);
    std::vector<double> lengths;
    lengths.push_back(d1.norm2());
    lengths.push_back(d2.norm2());
    lengths.push_back(d3.norm2());
    if (lengths[0] < lengths[1] && lengths[0] < lengths[2])  // 0 is smallest
      smallestedge = 0;
    else if (lengths[1] < lengths[2] && lengths[1] < lengths[0])  // 1 is smallest
      smallestedge = 1;
    else  // 2 is smallest plus all the rest
      smallestedge = 2;
  }

  Core::LinAlg::Matrix<3, 1> v1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> v2(Core::LinAlg::Initialization::zero);
  v1.update(1.0, surfpoints[(smallestedge + 1) % 3], -1.0, surfpoints[smallestedge]);
  v2.update(1.0, surfpoints[(smallestedge + 2) % 3], -1.0, surfpoints[smallestedge]);

  for (unsigned nid = 0; nid < num_nodes_edge; nid++)
  {
    (get_edge().nodes()[nid])->coordinates(actpoint.data());
    linepoints.push_back(actpoint);
  }
  Core::LinAlg::Matrix<3, 1> v3(Core::LinAlg::Initialization::zero);
  v3.update(1.0, linepoints[1], -1.0, linepoints[0]);

  bool overlap = false;  // we have an overlap of the refined bounding boxes?

  int pow_fac = 4;
  int max_steps = pow(pow_fac, maxstep - 1);
  double dmax_steps = 1.0 / max_steps;
  int act_steps = 0;
  std::size_t act_boxidx = 0;
  std::vector<std::vector<int>> overlappingidx;
  std::vector<std::vector<int>> newoverlappingidx;
  std::vector<int> tmpoverlappingidx;
  tmpoverlappingidx.push_back(0);          // min idx with overlap for tri3
  tmpoverlappingidx.push_back(max_steps);  // max idx with overlap tri3
  tmpoverlappingidx.push_back(0);          // min idx with overlap for line2
  tmpoverlappingidx.push_back(max_steps);  // max idx with overlap for line2
  overlappingidx.push_back(tmpoverlappingidx);

  Core::LinAlg::Matrix<3, 1> p1, p2, p3, p4, lp1, lp2;

  for (int refinestep = 1; refinestep <= maxstep; refinestep++)
  {
    newoverlappingidx.clear();
    act_boxidx = 0;
    act_steps = pow(pow_fac, (maxstep - refinestep));

    overlap = false;
    for (int surfstep = 0; surfstep < max_steps; surfstep += act_steps)
    {
      tmpoverlappingidx.clear();
      while ((act_boxidx + 1) < overlappingidx.size() and
             (overlappingidx[act_boxidx + 1])[0] < surfstep)
        act_boxidx++;
      if (!(overlappingidx[act_boxidx][0] <= surfstep and
              overlappingidx[act_boxidx][1] >= surfstep + act_steps))
        continue;
      double alpha = surfstep * dmax_steps;
      double alphap = (surfstep + act_steps) * dmax_steps;
      p1.update(1.0, surfpoints[smallestedge], alpha, v1);
      p2.update(1.0, surfpoints[smallestedge], alphap, v1);
      p3.update(1.0, surfpoints[smallestedge], alpha, v2);
      p4.update(1.0, surfpoints[smallestedge], alphap, v2);
      std::shared_ptr<BoundingBox> sbb(BoundingBox::create());
      sbb->add_point(p1);
      sbb->add_point(p2);
      sbb->add_point(p3);
      sbb->add_point(p4);
      for (int linestep = 0; linestep < max_steps; linestep += act_steps)
      {
        if (!(overlappingidx[act_boxidx][2] <= linestep and
                overlappingidx[act_boxidx][3] >= linestep + act_steps))
          continue;
        double lalpha = linestep * dmax_steps;
        double lalphap = (linestep + act_steps) * dmax_steps;
        lp1.update(1.0, linepoints[0], lalpha, v3);
        lp2.update(1.0, linepoints[0], lalphap, v3);
        std::shared_ptr<BoundingBox> ebb(BoundingBox::create());
        ebb->add_point(lp1);
        ebb->add_point(lp2);

        if (sbb->within(POSITIONTOL / BOXOVERLAP, *ebb))
        {
          overlap = true;
          if (!tmpoverlappingidx.size())
          {
            tmpoverlappingidx.push_back(surfstep);              // min idx with overlap for tri3
            tmpoverlappingidx.push_back(surfstep + act_steps);  // max idx with overlap for tri3
            tmpoverlappingidx.push_back(linestep);              // min idx with overlap for line2
            tmpoverlappingidx.push_back(linestep + act_steps);  // max idx with overlap for line2
          }
          else
          {
            if (tmpoverlappingidx[2] > linestep) tmpoverlappingidx[2] = linestep;
            if (tmpoverlappingidx[3] < linestep + act_steps)
              tmpoverlappingidx[3] = linestep + act_steps;
          }
        }
      }
      if (tmpoverlappingidx.size()) newoverlappingidx.push_back(tmpoverlappingidx);
    }
    overlappingidx = newoverlappingidx;

    std::cout << "refined_bb_overlap_check: Refinement Level " << refinestep << " there is "
              << overlap << " overlap!" << std::endl;
    if (!overlap) break;
  }
  return overlap;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Cut::IntersectionBase> Cut::IntersectionFactory::create_intersection(
    Core::FE::CellType edge_type, Core::FE::CellType side_type) const
{
  const int probdim = Global::Problem::instance()->n_dim();
  switch (edge_type)
  {
    case Core::FE::CellType::line2:
      return std::shared_ptr<Cut::IntersectionBase>(
          create_intersection<Core::FE::CellType::line2>(side_type, probdim));
    default:
      FOUR_C_THROW(
          "Unsupported edgeType! If meaningful, add your edgeType here. \n"
          "Given edgeType = {}",
          Core::FE::cell_type_to_string(edge_type).c_str());
      break;
  }
  exit(EXIT_FAILURE);
}


template <unsigned probdim, Core::FE::CellType edgetype, Core::FE::CellType sidetype, bool debug,
    unsigned dimedge, unsigned dimside, unsigned num_nodes_edge, unsigned num_nodes_side>
std::pair<bool, bool> Cut::Intersection<probdim, edgetype, sidetype, debug, dimedge, dimside,
    num_nodes_edge, num_nodes_side>::is_quad4_distorted()
{
  // we assume problem dimension equal to 3 here
  Kernel::PointOnSurfaceLoc loc[2];
  Kernel::PointOnSurfaceLoc loc_strict[2];

  double distance[2];

  for (int tri_id = 0; tri_id < 2; ++tri_id)
  {
    bool signeddistance = true;
    int other_point_index = (tri_id == 0) ? 3 : 1;
    Core::LinAlg::Matrix<3, 1> point(xyze_surfaceElement_.data() + other_point_index * 3, true);
    Core::LinAlg::Matrix<3, 1> xsi;
    Core::LinAlg::Matrix<3, 3> xyze_triElement;
    get_triangle(xyze_triElement, tri_id);

    bool conv = false;
    switch (get_options_ptr()->geom_distance_floattype())
    {
      case Cut::floattype_double:
      {
        Kernel::ComputeDistance<3, Core::FE::CellType::tri3, false> cd(xsi);
        conv = cd(xyze_triElement, point, distance[tri_id], signeddistance);
        if (conv) loc[tri_id] = cd.get_side_location();
        break;
      }
      case Cut::floattype_cln:
      {
        Kernel::ComputeDistance<3, Core::FE::CellType::tri3, true> cd(xsi);
        conv = cd(xyze_triElement, point, distance[tri_id], signeddistance);
        if (conv) loc[tri_id] = cd.get_side_location();
        break;
      }
      default:
      {
        FOUR_C_THROW("Unexpected floattype for Kernel::compute_distance!");
      }
    }
    if (conv)
    {
      loc_strict[tri_id] = loc[tri_id];
    }
    else
    {
#if EXTENDED_CUT_DEBUG_OUTPUT
      std::cout << "NOTICE: Cut distance did not converge, when detecting distortion of the quad4"
                << std::endl;
#endif
    }
    if (std::abs(distance[tri_id]) >= 1e-30)
    {
      // it is outside of the plane of the side
      loc_strict[tri_id] = Kernel::PointOnSurfaceLoc(loc[tri_id].within_side(), false);
    }
  }

  bool result = false;
  bool result_strict = false;

  for (int tri_id = 0; tri_id < 2; ++tri_id)
  {
    if (not(loc[tri_id].on_side()))
    {
      result = true;
#if EXTENDED_CUT_DEBUG_OUTPUT

      int other_point_index = (tri_id == 0) ? 3 : 1;
      Point* out_of_plane = GetSidePtr()->Nodes()[other_point_index]->point();
      Core::LinAlg::Matrix<3, 1> coord;
      out_of_plane->coordinates(coord.data());

      for (std::vector<Node*>::const_iterator it = GetSidePtr()->Nodes().begin();
          it != GetSidePtr()->Nodes().end(); ++it)
      {
        Point* p = (*it)->point();
        Core::LinAlg::Matrix<3, 1> distc;
        p->coordinates(distc.data());
        distc.update(-1, coord, 1);
      }

      std::cout << "Point of the quad4 with Id = " << out_of_plane->Id()
                << " does not lie on the plane created by other 3 points" << std::endl;
      std::cout << "Distance is between out of plane point and side, created by 3 other points is  "
                << distance[tri_id] << std::endl;
#endif
    }
    if (not(loc_strict[tri_id].on_side())) result_strict = true;
  }
  return std::make_pair(result, result_strict);
}



template class Cut::Intersection<2, Core::FE::CellType::line2, Core::FE::CellType::line2>;
template class Cut::Intersection<3, Core::FE::CellType::line2, Core::FE::CellType::line2>;
// template class
// Cut::intersection<2,Core::FE::CellType::line2,Core::FE::CellType::quad4>;
template class Cut::Intersection<3, Core::FE::CellType::line2, Core::FE::CellType::quad4>;
// template class
// Cut::intersection<2,Core::FE::CellType::line2,Core::FE::CellType::quad8>;
template class Cut::Intersection<3, Core::FE::CellType::line2, Core::FE::CellType::quad8>;
// template class
// Cut::intersection<2,Core::FE::CellType::line2,Core::FE::CellType::quad9>;
template class Cut::Intersection<3, Core::FE::CellType::line2, Core::FE::CellType::quad9>;
// template class
// Cut::intersection<2,Core::FE::CellType::line2,Core::FE::CellType::tri3>;
template class Cut::Intersection<3, Core::FE::CellType::line2, Core::FE::CellType::tri3>;

FOUR_C_NAMESPACE_CLOSE
