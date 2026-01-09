// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_mesh.hpp"

#include "4C_cut_boundarycell.hpp"
#include "4C_cut_integrationcell.hpp"
#include "4C_cut_levelsetside.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_parallel.hpp"
#include "4C_cut_point_impl.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_element_volume.hpp"
#include "4C_fem_geometry_searchtree.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*
Functions to catch implementation errors in debug mode
*/
namespace
{
  [[maybe_unused]] bool is_unique_in_boundary_cell(const std::vector<Cut::Point*>& points)
  {
    Cut::plain_point_set pointtest;
    pointtest.insert(points.begin(), points.end());
    return points.size() == pointtest.size();
  }
}  // namespace


/*-------------------------------------------------------------------------------------*
 * constructor
 *-------------------------------------------------------------------------------------*/
Cut::Mesh::Mesh(
    Options& options, double norm, std::shared_ptr<PointPool> pp, bool cutmesh, int myrank)
    : options_(options),
      norm_(norm),
      pp_(pp),
      bb_(std::shared_ptr<BoundingBox>(BoundingBox::create())),
      cutmesh_(cutmesh),
      myrank_(myrank)
{
  if (pp_ == nullptr)
  {
    // create a new octTree based pointpool that stores all points
    pp_ = std::make_shared<PointPool>(norm);
  }
}


/*-------------------------------------------------------------------------------------*
 * creates a new element, dependent on distype
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_element(
    int eid, const std::vector<int>& nids, Core::FE::CellType distype)
{
  switch (distype)
  {
    case Core::FE::CellType::line2:
      return create_line2(eid, nids);
    case Core::FE::CellType::tri3:
      return create_tri3(eid, nids);
    case Core::FE::CellType::quad4:
      return create_quad4(eid, nids);
    case Core::FE::CellType::hex8:
      return create_hex8(eid, nids);
    case Core::FE::CellType::tet4:
      return create_tet4(eid, nids);
    case Core::FE::CellType::pyramid5:
      return create_pyramid5(eid, nids);
    case Core::FE::CellType::wedge6:
      return create_wedge6(eid, nids);
    default:
      FOUR_C_THROW("unsupported distype ( distype = {} )", Core::FE::cell_type_to_string(distype));
  }
  return nullptr;
}


/*-------------------------------------------------------------------------------------*
 * creates a new side, dependent on distype
 *-------------------------------------------------------------------------------------*/
Cut::Side* Cut::Mesh::create_side(int sid, const std::vector<int>& nids, Core::FE::CellType distype)
{
  switch (distype)
  {
    case Core::FE::CellType::quad4:
      return create_quad4_side(sid, nids);
    case Core::FE::CellType::tri3:
      return create_tri3_side(sid, nids);
    default:
      FOUR_C_THROW("unsupported distype ( distype = {} )", Core::FE::cell_type_to_string(distype));
  }
  return nullptr;
}


/*-------------------------------------------------------------------------------------*
 * creates a new line2 element based on element id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_line2(int eid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Line<2>>();
  Element* e = get_element(eid, nids, *top_data);

  return e;
}

/*-------------------------------------------------------------------------------------*
 * creates a new tri3 element based on element id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_tri3(int eid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Triangle<3>>();
  Element* e = get_element(eid, nids, *top_data);

  return e;
}

/*-------------------------------------------------------------------------------------*
 * creates a new quad4 element based on element id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_quad4(int eid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Quadrilateral<4>>();
  Element* e = get_element(eid, nids, *top_data);

  return e;
}

/*-------------------------------------------------------------------------------------*
 * creates a new tet4 element based on element id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_tet4(int eid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Tetrahedron<4>>();

  Element* e = get_element(eid, nids, *top_data);

  return e;
}


/*-------------------------------------------------------------------------------------*
 * creates a new pyramid5 element based on element id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_pyramid5(int eid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Pyramid<5>>();

  Element* e = get_element(eid, nids, *top_data);

  return e;
}


/*-------------------------------------------------------------------------------------*
 * creates a new wedge6 element based on element id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_wedge6(int eid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Wedge<6>>();

  Element* e = get_element(eid, nids, *top_data);

  return e;
}


/*-------------------------------------------------------------------------------------*
 * creates a new hex8 element based on element id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::create_hex8(int eid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Hexahedron<8>>();

  Element* e = get_element(eid, nids, *top_data);

  return e;
}


/*-------------------------------------------------------------------------------------*
 * creates a new tri3 side based on side id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Side* Cut::Mesh::create_tri3_side(int sid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Triangle<3>>();
  return get_side(sid, nids, top_data);
}


/*-------------------------------------------------------------------------------------*
 * creates a new quad4 side based on side id and node ids
 *-------------------------------------------------------------------------------------*/
Cut::Side* Cut::Mesh::create_quad4_side(int sid, const std::vector<int>& nids)
{
  const CellTopologyData* top_data = shards::getCellTopologyData<shards::Quadrilateral<4>>();
  return get_side(sid, nids, top_data);
}


/*-------------------------------------------------------------------------------------*
 * creates a new point, optional information about cut-edge and cut-side
 *-------------------------------------------------------------------------------------*/
Cut::Point* Cut::Mesh::new_point(
    const double* x, Edge* cut_edge, Side* cut_side, double tolerance, double tol_scale)
{
  bb_->add_point(x);  // add the point to the mesh's bounding box
  // Point* p = pp_->NewPoint( x, cut_edge, cut_side, setup_ ? SETUPNODECATCHTOL : MINIMALTOL );
  Point* p = pp_->new_point(x, cut_edge, cut_side, tolerance);  // add the point in the point pool
  return p;
}


/*-------------------------------------------------------------------------------------*
 * creates a new line
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::new_line(Point* p1, Point* p2, Side* cut_side1, Side* cut_side2,
    Element* cut_element, std::vector<Line*>* newlines)
{
  if (p1 == p2) FOUR_C_THROW("no line between same point");

  // If there is a line between those points already, return it. Otherwise
  // create a new one.
  Line* line = p1->common_line(p2);
  if (not line)
  {
    plain_edge_set edges;
    p1->common_edge(p2, edges);
    for (plain_edge_set::iterator i = edges.begin(); i != edges.end(); ++i)
    {
      Edge* e = *i;
      std::vector<Point*> line_points;
      try
      {
        e->cut_points_including(p1, p2, line_points);
      }
      catch (std::exception& err)
      {
        for (std::vector<Point*>::iterator it = line_points.begin(); it != line_points.end(); ++it)
        {
          if (not((*it)->is_cut(cut_side1, cut_side2)))
          {
            (*it)->dump_connectivity_info();
            FOUR_C_THROW("this point does not belong here!");
          }
        }
        FOUR_C_THROW("Point does lie on edge!");
        throw err;
      }
      new_lines_between(line_points, cut_side1, cut_side2, cut_element, newlines);
    }

    if (edges.size() == 0) new_line_internal(p1, p2, cut_side1, cut_side2, cut_element);
  }

  // return line;
}


/*-------------------------------------------------------------------------------------------------*
 * Create new line between the two given cut points that are in given two cut sides
 *-------------------------------------------------------------------------------------------------*/
Cut::Line* Cut::Mesh::new_line_internal(
    Point* p1, Point* p2, Side* cut_side1, Side* cut_side2, Element* cut_element)
{
  if (p1 == p2) FOUR_C_THROW("no line between same point");

  // If there is a line between those points already, return it. Otherwise
  // create a new one.
  Line* line = p1->common_line(p2);
  if (not line)
  {
    line = new Line(p1, p2, cut_side1, cut_side2, cut_element);
    lines_.push_back(std::shared_ptr<Line>(line));
  }
  else  // line already exists. just add cut side details to the line
  {
    if (cut_side1) line->add_side(cut_side1);
    if (cut_side2) line->add_side(cut_side2);
    if (cut_element) line->add_element(cut_element);
  }
  return line;
}


/*-------------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------------*/
bool Cut::Mesh::new_lines_between(const std::vector<Point*>& line, Side* cut_side1, Side* cut_side2,
    Element* cut_element, std::vector<Line*>* newlines)
{
  bool hasnewlines = false;
  std::vector<Point*>::const_iterator i = line.begin();
  if (i != line.end())
  {
    Point* bp = *i;
    for (++i; i != line.end(); ++i)
    {
      Point* ep = *i;
      Line* l = new_line_internal(bp, ep, cut_side1, cut_side2, cut_element);
      if (newlines != nullptr) newlines->push_back(l);
      bp = ep;
    }
    hasnewlines = true;
  }
  return hasnewlines;
}


/*-------------------------------------------------------------------------------------*
 * creates a new facet, consists of points, additional bool if it is a facet on a
 * cutsurface
 *-------------------------------------------------------------------------------------*/
Cut::Facet* Cut::Mesh::new_facet(const std::vector<Point*>& points, Side* side, bool cutsurface)
{
  if (points.size() == 0) FOUR_C_THROW("empty facet");

  std::vector<Point*>::const_iterator i = points.begin();
  plain_facet_set facets = (*i)->facets();
  for (++i; i != points.end(); ++i)
  {
    Point* p = *i;
    p->intersection(facets);
    if (facets.size() == 0)
    {
      break;
    }
  }

  for (plain_facet_set::iterator j = facets.begin(); j != facets.end(); ++j)
  {
    Facet* f = *j;
    if (f->equals(points))
    {
      if (side->is_cut_side())
      {
        f->exchange_side(side, true);
      }
      return f;
    }
  }

  Facet* f = new Facet(*this, points, side, cutsurface);
  facets_.push_back(std::shared_ptr<Facet>(f));

  return f;
}


/*-------------------------------------------------------------------------------------*
 * creates a new volumecell, consists of facets
 *-------------------------------------------------------------------------------------*/
Cut::VolumeCell* Cut::Mesh::new_volume_cell(const plain_facet_set& facets,
    const std::map<std::pair<Point*, Point*>, plain_facet_set>& volume_lines, Element* element)
{
  VolumeCell* c = new VolumeCell(facets, volume_lines, element);
  cells_.push_back(std::shared_ptr<VolumeCell>(c));  // store the pointer in mesh's cells_
  return c;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Point1BoundaryCell* Cut::Mesh::new_point1_cell(
    VolumeCell* volume, Facet* facet, const std::vector<Point*>& points)
{
  const unsigned num_nodes = Core::FE::num_nodes(Core::FE::CellType::point1);
  if (points.size() != num_nodes) FOUR_C_THROW("Mismatch of point and node number!");

  Core::LinAlg::SerialDenseMatrix xyz(3, 1);
  points[0]->coordinates(&xyz(0, 0));

  Point1BoundaryCell* bc = new Point1BoundaryCell(xyz, facet, points);
  boundarycells_.push_back(std::shared_ptr<Point1BoundaryCell>(bc));

  return bc;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Line2BoundaryCell* Cut::Mesh::new_line2_cell(
    VolumeCell* volume, Facet* facet, const std::vector<Point*>& points)
{
  const unsigned num_nodes = Core::FE::num_nodes(Core::FE::CellType::line2);
  if (points.size() != num_nodes) FOUR_C_THROW("Mismatch of point and node number!");

  Core::LinAlg::SerialDenseMatrix xyze(3, num_nodes);
  for (unsigned i = 0; i < static_cast<unsigned>(xyze.numCols()); ++i)
  {
    points[i]->coordinates(&xyze(0, i));
  }

  Line2BoundaryCell* bc = new Line2BoundaryCell(xyze, facet, points);
  boundarycells_.push_back(std::shared_ptr<Line2BoundaryCell>(bc));

  return bc;
}

/*-------------------------------------------------------------------------------------*
 * creates a new tri3 boundary cell
 *-------------------------------------------------------------------------------------*/
Cut::Tri3BoundaryCell* Cut::Mesh::new_tri3_cell(
    VolumeCell* volume, Facet* facet, const std::vector<Point*>& points)
{
  if (points.size() != 3) FOUR_C_THROW("expect 3 points");

  FOUR_C_ASSERT(is_unique_in_boundary_cell(points), "point used more than once in boundary cell");

  Core::LinAlg::SerialDenseMatrix xyz(3, 3);
  for (int i = 0; i < 3; ++i)
  {
    points[i]->coordinates(&xyz(0, i));
  }

  Tri3BoundaryCell* bc = new Tri3BoundaryCell(xyz, facet, points);
  bc->set_new_cubature_degree(options_.bc_cubaturedegree());
  boundarycells_.push_back(std::shared_ptr<Tri3BoundaryCell>(bc));

  return bc;
}


/*-------------------------------------------------------------------------------------*
 * creates a new quad4 boundary cell
 *-------------------------------------------------------------------------------------*/
Cut::Quad4BoundaryCell* Cut::Mesh::new_quad4_cell(
    VolumeCell* volume, Facet* facet, const std::vector<Point*>& points)
{
  if (points.size() != 4) FOUR_C_THROW("expect 4 points");

  FOUR_C_ASSERT(is_unique_in_boundary_cell(points), "point used more than once in boundary cell");

  Core::LinAlg::SerialDenseMatrix xyz(3, 4);
  for (int i = 0; i < 4; ++i)
  {
    points[i]->coordinates(&xyz(0, i));
  }

  Quad4BoundaryCell* bc = new Quad4BoundaryCell(xyz, facet, points);
  bc->set_new_cubature_degree(options_.bc_cubaturedegree());
  boundarycells_.push_back(std::shared_ptr<Quad4BoundaryCell>(bc));

  return bc;
}


/*-------------------------------------------------------------------------------------*
 * creates a new ??? boundary cell
 *-------------------------------------------------------------------------------------*/
Cut::ArbitraryBoundaryCell* Cut::Mesh::new_arbitrary_cell(VolumeCell* volume, Facet* facet,
    const std::vector<Point*>& points, const Core::FE::GaussIntegration& gaussRule,
    const Core::LinAlg::Matrix<3, 1>& normal)
{
  FOUR_C_ASSERT(is_unique_in_boundary_cell(points), "point used more than once in boundary cell");

  Core::LinAlg::SerialDenseMatrix xyz(3, points.size());
  for (unsigned i = 0; i < points.size(); ++i)
  {
    points[i]->coordinates(&xyz(0, i));
  }

  ArbitraryBoundaryCell* bc = new ArbitraryBoundaryCell(xyz, facet, points, gaussRule, normal);
  boundarycells_.push_back(std::shared_ptr<ArbitraryBoundaryCell>(bc));

  return bc;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Line2IntegrationCell* Cut::Mesh::new_line2_cell(
    Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell)
{
  const unsigned num_nodes = Core::FE::num_nodes(Core::FE::CellType::line2);
  if (points.size() != num_nodes) FOUR_C_THROW("Mismatch of point and node number!");

  Core::LinAlg::SerialDenseMatrix xyze(3, num_nodes);
  for (unsigned i = 0; i < static_cast<unsigned>(xyze.numCols()); ++i)
  {
    points[i]->coordinates(&xyze(0, i));
  }
  Line2IntegrationCell* c = new Line2IntegrationCell(position, xyze, points, cell);
  integrationcells_.push_back(std::shared_ptr<Line2IntegrationCell>(c));

  return c;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Tri3IntegrationCell* Cut::Mesh::new_tri3_cell(
    Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell)
{
  const unsigned num_nodes = Core::FE::num_nodes(Core::FE::CellType::tri3);
  if (points.size() != num_nodes) FOUR_C_THROW("Mismatch of point and node number!");

  Core::LinAlg::SerialDenseMatrix xyze(3, num_nodes);
  for (unsigned i = 0; i < static_cast<unsigned>(xyze.numCols()); ++i)
  {
    points[i]->coordinates(&xyze(0, i));
  }
  Tri3IntegrationCell* c = new Tri3IntegrationCell(position, xyze, points, cell);
  integrationcells_.push_back(std::shared_ptr<Tri3IntegrationCell>(c));

  return c;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Quad4IntegrationCell* Cut::Mesh::new_quad4_cell(
    Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell)
{
  const unsigned num_nodes = Core::FE::num_nodes(Core::FE::CellType::quad4);
  if (points.size() != num_nodes) FOUR_C_THROW("Mismatch of point and node number!");

  Core::LinAlg::SerialDenseMatrix xyze(3, num_nodes);
  for (unsigned i = 0; i < static_cast<unsigned>(xyze.numCols()); ++i)
  {
    points[i]->coordinates(&xyze(0, i));
  }
  Quad4IntegrationCell* c = new Quad4IntegrationCell(position, xyze, points, cell);
  integrationcells_.push_back(std::shared_ptr<Quad4IntegrationCell>(c));

  return c;
}

/*-------------------------------------------------------------------------------------*
 * creates a new hex8 integration cell
 *-------------------------------------------------------------------------------------*/
Cut::Hex8IntegrationCell* Cut::Mesh::new_hex8_cell(
    Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell)
{
  Core::LinAlg::SerialDenseMatrix xyz(3, points.size());
  for (unsigned i = 0; i < points.size(); ++i)
  {
    points[i]->coordinates(&xyz(0, i));
  }

  Hex8IntegrationCell* c = new Hex8IntegrationCell(position, xyz, points, cell);
  integrationcells_.push_back(std::shared_ptr<Hex8IntegrationCell>(c));

  return c;
}


/*-------------------------------------------------------------------------------------*
 * creates a new tet4 integration cell, based on points
 *-------------------------------------------------------------------------------------*/
Cut::Tet4IntegrationCell* Cut::Mesh::new_tet4_cell(
    Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell)
{
  if (points.size() != 4) FOUR_C_THROW("wrong number of cell points");
  Core::LinAlg::SerialDenseMatrix xyz(3, points.size());
  for (unsigned i = 0; i < points.size(); ++i)
  {
    points[i]->coordinates(&xyz(0, i));
  }

  Tet4IntegrationCell* c = new Tet4IntegrationCell(position, xyz, points, cell);
  integrationcells_.push_back(std::shared_ptr<Tet4IntegrationCell>(c));

  return c;
}


/*-------------------------------------------------------------------------------------*
 * creates a new tet4 integration cell, based on xyz coordinates
 *-------------------------------------------------------------------------------------*/
Cut::Tet4IntegrationCell* Cut::Mesh::new_tet4_cell(
    Point::PointPosition position, const Core::LinAlg::SerialDenseMatrix& xyz, VolumeCell* cell)
{
  std::vector<Point*> points;  // empty list of points
  Tet4IntegrationCell* c = new Tet4IntegrationCell(position, xyz, points, cell);
  integrationcells_.push_back(std::shared_ptr<Tet4IntegrationCell>(c));
  return c;
}


/*-------------------------------------------------------------------------------------*
 * creates a new wedge6 integration cell
 *-------------------------------------------------------------------------------------*/
Cut::Wedge6IntegrationCell* Cut::Mesh::new_wedge6_cell(
    Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell)
{
  Core::LinAlg::SerialDenseMatrix xyz(3, points.size());
  for (unsigned i = 0; i < points.size(); ++i)
  {
    points[i]->coordinates(&xyz(0, i));
  }

  Wedge6IntegrationCell* c = new Wedge6IntegrationCell(position, xyz, points, cell);
  integrationcells_.push_back(std::shared_ptr<Wedge6IntegrationCell>(c));

  return c;
}


/*-------------------------------------------------------------------------------------*
 * creates a new pyramid5 integration cell
 *-------------------------------------------------------------------------------------*/
Cut::Pyramid5IntegrationCell* Cut::Mesh::new_pyramid5_cell(
    Point::PointPosition position, const std::vector<Point*>& points, VolumeCell* cell)
{
  Core::LinAlg::SerialDenseMatrix xyz(3, points.size());
  for (unsigned i = 0; i < points.size(); ++i)
  {
    points[i]->coordinates(&xyz(0, i));
  }

  Pyramid5IntegrationCell* c = new Pyramid5IntegrationCell(position, xyz, points, cell);
  integrationcells_.push_back(std::shared_ptr<Pyramid5IntegrationCell>(c));

  return c;
}

/*------------------------------------------------------------------------------------------------*
 * build the static search tree for the collision detection in the self cut           wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void Cut::Mesh::build_self_cut_tree()
{
  // constructor for the search tree of the background mesh
  selfcuttree_ =
      std::make_shared<Core::Geo::SearchTree>(5);  // tree_depth_ 4 is reasonable, 5 is possible

  // extent the bounding volume of the root of the search tree to prevent symmetry issues
  Core::LinAlg::Matrix<3, 2> boundingvolume = bb_->get_bounding_volume();
  //  boundingvolume(0,1) += 1e-4;
  //  boundingvolume(1,1) += 1e-4;
  //  boundingvolume(2,1) += 1e-4;

  // initializes the search tree of the background mesh
  selfcuttree_->initialize_tree(boundingvolume, Core::Geo::TreeType(Core::Geo::OCTTREE));

  // inserts all linear elements into the search tree of the cutter mesh
  for (std::map<int, Side*>::iterator i = shadow_sides_.begin(); i != shadow_sides_.end(); ++i)
  {
    int sid = i->first;
    Side* s = i->second;
    selfcuttree_->insert_element(sid);
    selfcutbvs_[sid] = s->get_bounding_volume().get_bounding_volume();  // better kdop?
  }

  // builds the static search tree of the background mesh
  if (selfcutbvs_.size() !=
      0)  // *********************************************************** possible in case of
          // parallel computing and using only relevant elements
  {
    selfcuttree_->build_static_search_tree(selfcutbvs_);
  }
}

/*------------------------------------------------------------------------------------------------*
 * build the static search tree for the collision detection                           wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void Cut::Mesh::build_static_search_tree()
{
  // constructor for the search tree of the background mesh
  searchtree_ =
      std::make_shared<Core::Geo::SearchTree>(5);  // tree_depth_ 4 is reasonable, 5 is possible

  // extent the bounding volume of the root of the search tree to prevent symmetry issues
  Core::LinAlg::Matrix<3, 2> boundingvolume = bb_->get_bounding_volume();
  boundingvolume(0, 1) += 1e-4;
  boundingvolume(1, 1) += 1e-4;
  boundingvolume(2, 1) += 1e-4;

  // initializes the search tree of the background mesh
  searchtree_->initialize_tree(boundingvolume, Core::Geo::TreeType(Core::Geo::OCTTREE));

  // inserts all linear elements into the search tree of the background mesh
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    int eid = i->first;
    Element* e = &*i->second;
    searchtree_->insert_element(eid);
    boundingvolumes_[eid] = e->get_bounding_volume().get_bounding_volume();
  }

  // inserts all quadratic elements into the search tree of the background mesh
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    int eid = i->first;
    Element* e = &*i->second;
    searchtree_->insert_element(eid);
    boundingvolumes_[eid] = e->get_bounding_volume().get_bounding_volume();
  }

  // builds the static search tree of the background mesh
  if (boundingvolumes_.size() !=
      0)  // *********************************************************** possible in case of
          // parallel computing and using only relevant elements
  {
    searchtree_->build_static_search_tree(boundingvolumes_);
  }
}

/*---------------------------------------------------------------------*
 * Cuts the background elements of the mesh with all the cut sides     *
 * Called by Tetmeshintersection (but not for normal meshintersection) *
 *---------------------------------------------------------------------*/
void Cut::Mesh::cut(Mesh& mesh, plain_element_set& elements_done)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 6/6 --- Cut_Finalize --- CUT (incl. tetmesh-cut)");

  plain_element_set my_elements_done;

  // perform the cut for each side of the cut_mesh_
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = sides_.begin();
      i != sides_.end(); ++i)
  {
    Side& side = *i->second;
    mesh.cut(side, elements_done, my_elements_done);
  }

  std::copy(my_elements_done.begin(), my_elements_done.end(),
      std::inserter(elements_done, elements_done.begin()));
}


/*------------------------------------------------------------------------*
 * Cuts the background elements of the mesh with this considered cut side *
 * Called by Tetmeshintersection (but not for normal meshintersection)    *
 *-----------------------------------------------------------------------*/
void Cut::Mesh::cut(Side& side, const plain_element_set& done, plain_element_set& elements_done)
{
  // define a bounding box around the maybe twisted side to determine a
  // preselection of cutting sides and elements
  std::shared_ptr<BoundingBox> sidebox = std::shared_ptr<BoundingBox>(BoundingBox::create(side));
  plain_element_set elements;

  // use a brute force preselection
  //
  // is there an overlap between the side's bounding box
  // and the elements bounding box ?
  // if yes -> try to find a cut
  //
  // REMARK: if the brute force preselection is to slow, one could think about
  // an additional octtree for elements to get a logartithmic complexity
  //

  {
    TEUCHOS_FUNC_TIME_MONITOR(
        "Cut --- 6/6 --- Cut_Finalize --- preselection of possible cut between");

    // preselection of possible cut between linear elements and the current side
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
    {
      Element* e = &*(i->second);
      std::shared_ptr<BoundingBox> elementbox = std::shared_ptr<BoundingBox>(BoundingBox::create());
      elementbox->assign(*e);
      if (elementbox->within(1.0, *sidebox))
      {
        if (elements.count(e) == 0)
        {
          elements.insert(e);
        }
      }
    }
    // preselection of possible cut between shadow elements of quadratic elements and the current
    // side
    for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
        i != shadow_elements_.end(); i++)
    {
      Element* e = &*(i->second);
      std::shared_ptr<BoundingBox> elementbox = std::shared_ptr<BoundingBox>(BoundingBox::create());
      elementbox->assign(*e);
      if (elementbox->within(1.0, *sidebox))
      {
        if (elements.count(e) == 0)
        {
          elements.insert(e);
        }
      }
    }
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("Cut --- 6/6 --- Cut_Finalize --- cutting sides with elements");


    // perform the cut of this side for each involved element
    for (plain_element_set::iterator i = elements.begin(); i != elements.end(); ++i)
    {
      Element* e = *i;
      if (done.count(e) == 0)
      {
        if (e->cut(*this, side))
        {
          elements_done.insert(e);
        }
      }
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * Cuts the background elements with this levelset side
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::cut(Side& side)
{
  if (not side.is_level_set_side()) FOUR_C_THROW("This function expects a level set side!");

  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.cut(*this, side);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.cut(*this, side);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * ???
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::rectify_cut_numerics()
{
  for (std::map<plain_int_set, std::shared_ptr<Edge>>::iterator i = edges_.begin();
      i != edges_.end(); ++i)
  {
    Edge* e = &*i->second;
    e->rectify_cut_numerics();
  }
}

/*------------------------------------------------------------------------------------------------*
 * detects if a side of the cut mesh possibly collides with an element of the background mesh     *
 *                                                                                    wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void Cut::Mesh::search_collisions(Mesh& cutmesh)
{
  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = cutmesh.sides();
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    Core::LinAlg::Matrix<3, 2> cutsideBV = cutside->get_bounding_volume().get_bounding_volume();
    if (boundingvolumes_.size() !=
        0)  // ******************************************************* possible in case of parallel
            // computing and using only relevant elements
    {
      std::set<int> collisions;
      // search collision between cutside and elements in the static search tree
      searchtree_->search_collisions(boundingvolumes_, cutsideBV, 0, collisions);
      // add the cutside to all the elements which have been found
      for (std::set<int>::iterator ic = collisions.begin(); ic != collisions.end(); ++ic)
      {
        int collisionid = *ic;
        std::map<int, std::shared_ptr<Element>>::iterator ie = elements_.find(collisionid);
        if (ie != elements_.end())
        {
          Element& e = *ie->second;
          e.add_cut_face(cutside);
        }
        std::map<int, std::shared_ptr<Element>>::iterator ise = shadow_elements_.find(collisionid);
        if (ise != shadow_elements_.end())
        {
          Element& e = *ise->second;
          e.add_cut_face(cutside);
        }
      }
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * finds intersections between sides and edges                                         *
 *                                                                         wirtz 08/14 *
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::find_cut_points()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 4/6 --- cut_mesh_intersection --- find_cut_points");

  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.find_cut_points(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.find_cut_points(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * create cut lines based on the point cloud
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::make_cut_lines()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 4/6 --- cut_mesh_intersection --- MakeCutLines");

  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;

    // skip the cut line creation for 1-D elements
    if (e.n_dim() == 1) continue;

    try
    {
      e.make_cut_lines(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;

    // skip the cut line creation for 1-D elements
    if (e.n_dim() == 1) continue;

    try
    {
      e.make_cut_lines(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * create facets based on the cut lines
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::make_facets()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 4/6 --- cut_mesh_intersection --- MakeFacets");

  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.make_facets(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.make_facets(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * create volumecells based on created facets
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::make_volume_cells()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 4/6 --- cut_mesh_intersection --- MakeVolumeCells");

  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.make_volume_cells(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.make_volume_cells(*this);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * find node positions and propagate the positions to facets, points and volumecells
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::find_node_positions()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets --- FindNodePositions");


  // On multiple cuts former outside positions can become inside
  // positions. Thus reset all outside positions.
  pp_->reset_outside_points();

  // get nodal positions from elements
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.find_node_positions();
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.find_node_positions();
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  // find undecided nodes
  // * for serial simulations all node positions should be set
  // * for parallel simulations there can be some undecided nodes
  //  check_for_undecided_node_positions();
}


/*-------------------------------------------------------------------------------------*
 * Find node positions and propagate the positions to points.
 * Might need modification for more difficult cut situations with shadow elements.
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::find_ls_node_positions()
{
  for (std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
  {
    Node* n = &*i->second;
    Point* p = n->point();
    if (p->position() == Point::undecided)
    {
      // we have to take into account the tolerance for which a node lies on the levelset-side
      double lsv = n->lsv();
      if (lsv > 0.0)  // REFERENCETOL )
      {
        p->position(Point::outside);
      }
      else if (lsv < 0.0)  //-REFERENCETOL )
      {
        p->position(Point::inside);
      }
      else  //
      {
        // FOUR_C_THROW( "undecided nodal point on levelset
        // surface" );
        std::cout << "UNDECIDED CUT POSITION! SHOULD THIS HAPPEN?!"
                  << " lsv= " << std::setprecision(24) << lsv << std::endl;
        p->position(Point::oncutsurface);
      }
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * find facet positions for remaining facets, points, volumecells that have not been found
 * using FindNodePositions()
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::find_facet_positions()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets --- FindFacetPositions");


  plain_volumecell_set undecided;

  for (std::list<std::shared_ptr<VolumeCell>>::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* c = &**i;
    //     if ( c->Empty() )
    //       continue;
    if (c->position() == Point::undecided)
    {
      const plain_facet_set& facets = c->facets();

      //    bool haveundecided = false;

      Point::PointPosition position = Point::undecided;
      for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
      {
        Facet* f = *i;
        Point::PointPosition fp = f->position();
        switch (fp)
        {
          case Point::undecided:
            //        haveundecided = true;
            break;
          case Point::oncutsurface:
            break;
          case Point::inside:
          case Point::outside:
            if (position != Point::undecided and position != fp)
            {
              FOUR_C_THROW("mixed facet set");
            }
            position = fp;
            break;
        }
      }

      // set any undecided facets in a decided volume cell

      if (position != Point::undecided)
      {
        c->position(position);
      }
      else
      {
        undecided.insert(c);
      }
    }
  }

  // look to neighbouring volume cells to decide the position of this cell

  // this is a slow algorithm, but there should only be very few undecided
  // volume cells, so it should be fine.

  if (cells_.size() == undecided.size() and cells_.size() > 0)
    FOUR_C_THROW("all volume cells undecided and volume cells available");

  while (undecided.size() > 0)
  {
    unsigned size = undecided.size();
    for (plain_volumecell_set::iterator ui = undecided.begin(); ui != undecided.end();)
    {
      VolumeCell* c = *ui;
      bool done = false;
      const plain_facet_set& facets = c->facets();
      for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
      {
        Facet* f = *i;
        {
          VolumeCell* nc = f->neighbor(c);
          if (nc != nullptr)
          {
            Point::PointPosition np = nc->position();
            switch (np)
            {
              case Point::undecided:
                if (undecided.count(nc) == 0)
                  FOUR_C_THROW("incomplete set of undecided volume cells");
                break;
              case Point::oncutsurface:
                FOUR_C_THROW("illegal volume position");
                break;
              case Point::inside:
              case Point::outside:
                if (f->on_cut_side())
                  c->position(np == Point::inside ? Point::outside : Point::inside);
                else
                  c->position(np);
                done = true;
                break;
            }
            if (done) break;
          }
        }
      }
      if (done)
      {
        set_erase(undecided, ui);
      }
      else
      {
        ++ui;
      }
    }
    if (size == undecided.size()) FOUR_C_THROW("no progress in volume cell position");
  }

  // second pass

  //   for ( std::list<std::shared_ptr<VolumeCell> >::iterator i=cells_.begin(); i!=cells_.end();
  //   ++i )
  //   {
  //     VolumeCell * c = &**i;
  //     const plain_facet_set & facets = c->Facets();

  //     bool haveundecided = false;
  //     bool havecutsurface = false;
  //     Cut::Point::PointPosition position = Cut::Point::undecided;
  //     for ( plain_facet_set::const_iterator i=facets.begin(); i!=facets.end(); ++i )
  //     {
  //       Facet * f = *i;
  //       Cut::Point::PointPosition fp = f->Position();
  //       switch ( fp )
  //       {
  //       case Cut::Point::undecided:
  //         haveundecided = true;
  //         break;
  //       case Cut::Point::oncutsurface:
  //         havecutsurface = true;
  //         break;
  //       case Cut::Point::inside:
  //       case Cut::Point::outside:
  //         if ( position!=Cut::Point::undecided and position!=fp )
  //         {
  //           FOUR_C_THROW( "mixed facet set" );
  //         }
  //         position = fp;
  //       }
  //     }

  // this is a bold assumption.

  //     if ( position == Cut::Point::undecided )
  //     {
  //       if ( havecutsurface )
  //         position = Cut::Point::inside;
  //       else
  //         position = Cut::Point::outside;
  //     }

  // set any undecided facets in a decided volume cell

  //     if ( haveundecided and position != Cut::Point::undecided )
  //     {
  //       for ( plain_facet_set::const_iterator i=facets.begin(); i!=facets.end(); ++i )
  //       {
  //         Facet * f = *i;
  //         Cut::Point::PointPosition fp = f->Position();
  //         if ( fp==Cut::Point::undecided )
  //         {
  //           f->Position( position );
  //         }
  //       }
  //     }
  //   }
}


/*-------------------------------------------------------------------------------*
 | fill the vector with nids of nodes with undecided position,       schott 03/12 |
 *-------------------------------------------------------------------------------*/
bool Cut::Mesh::check_for_undecided_node_positions(
    std::map<int, int>& undecided_nodes, std::map<plain_int_set, int>& undecided_shadow_nodes)
{
  // return whether undecided node positions available
  bool undecided_node_positions = false;

  undecided_nodes.clear();

  // find nodes with undecided node positions
  for (std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
  {
    Node* n = &*i->second;
    Point* p = n->point();
    Point::PointPosition pos = p->position();

    // check for undecided position
    if (pos == Point::undecided)
    {
      if (n->id() < 0)
      {
        // continue for shadow nodes, shadow nodes will be handled afterwards
        continue;
      }
      // insert pair of nid and undecided PointPosition
      undecided_nodes.insert(std::pair<int, int>(n->id(), pos));

      // set undecided_node_positions to true
      undecided_node_positions = true;
    }
  }

  //----------------------------------
  // do the same for possible shadow nodes
  //----------------------------------

  // find nodes with undecided node positions for shadow nodes of e.g. hex20 elements
  for (std::map<plain_int_set, Node*>::iterator i = shadow_nodes_.begin(); i != shadow_nodes_.end();
      ++i)
  {
    Node* n = &*i->second;
    Point* p = n->point();
    Point::PointPosition pos = p->position();

    // check for undecided position
    if (pos == Point::undecided)
    {
      if (n->id() >= 0)
      {
        FOUR_C_THROW("this cannot be a shadow node, a shadow node should have negative nid");
      }

      // for hex20 elements, boundary shadow nodes are identified by the eight side-nids of the
      // quad8 side the inner shadow node is identified by the 20 nodes of the hex20 element

      // insert pair of sorted node-Ids of side or element to identify the unique shadow node and
      // undecided PointPosition
      undecided_shadow_nodes.insert(std::pair<plain_int_set, int>(i->first, pos));

      // set undecided_node_positions to true
      undecided_node_positions = true;
    }
  }


  return undecided_node_positions;
}


/*-------------------------------------------------------------------------------------*
 * still used???
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::find_nodal_dof_sets(bool include_inner)
{
  for (std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
  {
    Node* n = &*i->second;
    n->find_dof_sets(include_inner);
  }

  for (std::list<std::shared_ptr<VolumeCell>>::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* cell = &**i;
    cell->connect_nodal_dof_sets(include_inner);
  }
}


/*-------------------------------------------------------------------------------------*
 * Execute Tessellation with QHULL for each element to generate integrationcells
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::create_integration_cells(int count, bool tetcellsonly)
{
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.create_integration_cells(*this, count + 1, tetcellsonly);
    }
    catch (Core::Exception& err)
    {
      if (count > 0)
      {
        // FOUR_C_THROW("Error occurred in a recursive call i.e. in a call from
        // TetMeshIntersection.");
        std::cout << "Error occurred in a recursive call i.e. in a call from TetMeshIntersection."
                  << std::endl;
      }
      std::ostringstream error_msg;
      error_msg << "Caught error in a (recursive) call (count = " << count
                << ") "
                   "[i.e. if count > 0 in a call from TetMeshIntersection]:";

      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.create_integration_cells(*this, count + 1, tetcellsonly);
    }
    catch (Core::Exception& err)
    {
      if (count > 0)
        std::cout << "Error occurred in a recursive call i.e. in a call from TetMeshIntersection."
                  << std::endl;

      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * Call the moment fitting method for each element to generate the Gaussian integration rule
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::moment_fit_gauss_weights(bool include_inner, Cut::BCellGaussPts Bcellgausstype)
{
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.moment_fit_gauss_weights(*this, include_inner, Bcellgausstype);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.moment_fit_gauss_weights(*this, include_inner, Bcellgausstype);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * Call the DirectDivergence method for each element to generate the Gaussian integration rule
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::direct_divergence_gauss_rule(bool include_inner, Cut::BCellGaussPts Bcellgausstype)
{
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.direct_divergence_gauss_rule(*this, include_inner, Bcellgausstype);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.direct_divergence_gauss_rule(*this, include_inner, Bcellgausstype);
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * ?
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::remove_empty_volume_cells()
{
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.remove_empty_volume_cells();
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    try
    {
      e.remove_empty_volume_cells();
    }
    catch (Core::Exception& err)
    {
      debug_dump(&e, __FILE__, __LINE__);
      throw std::runtime_error(err.what());
    }
  }
}


/*-------------------------------------------------------------------------------------*
 * test if for all elements the element volume is equal to the volume of all integration cells
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::test_element_volume(bool fatal, VCellGaussPts VCellGP)
{
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    test_element_volume(e.shape(), e, fatal, VCellGP);
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    test_element_volume(e.shape(), e, fatal, VCellGP);
  }
}


/*-------------------------------------------------------------------------------------*
 * Find the difference between the volume of background element and the sum of volume
 * of all integration cells. There should be no difference between these two
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::test_element_volume(
    Core::FE::CellType shape, Element& e, bool fatal, VCellGaussPts VCellGP)
{
  if (e.is_cut())
  {
    const std::vector<Node*>& nodes = e.nodes();
    Core::LinAlg::SerialDenseMatrix xyze(3, nodes.size());
    double max_norm = 0.0;
    for (unsigned i = 0; i < nodes.size(); ++i)
    {
      nodes[i]->coordinates(&xyze(0, i));
      Core::LinAlg::Matrix<3, 1> vec;
      nodes[i]->coordinates(&vec(0, 0));
      if (vec.norm2() > max_norm) max_norm = vec.norm2();
    }

    double ev = Core::Geo::element_volume(e.shape(), xyze);

    [[maybe_unused]] int numgp = 0;
    [[maybe_unused]] int numic = 0;
    [[maybe_unused]] int numbc = 0;
    double cv = 0;
    [[maybe_unused]] double ba = 0;
    const plain_volumecell_set& cells = e.volume_cells();
    if (VCellGP == VCellGaussPts_Tessellation)
    {
      for (plain_volumecell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
      {
        VolumeCell* vc = *i;
        numic += vc->integration_cells().size();
        numgp += vc->num_gauss_points(shape);
        cv += vc->volume();

        const plain_boundarycell_set& bcells = vc->boundary_cells();
        numbc += bcells.size();
        for (plain_boundarycell_set::const_iterator i = bcells.begin(); i != bcells.end(); ++i)
        {
          BoundaryCell* bc = *i;
          ba += bc->area();
        }
      }
    }
    else if (VCellGP == VCellGaussPts_DirectDivergence)
    {
      for (plain_volumecell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
      {
        VolumeCell* vc = *i;
        // numic += vc->IntegrationCells().size();
        //        numgp += vc-> //vc->NumGaussPoints( shape );
        cv += vc->volume();

        const plain_boundarycell_set& bcells = vc->boundary_cells();
        numbc += bcells.size();
        for (plain_boundarycell_set::const_iterator i = bcells.begin(); i != bcells.end(); ++i)
        {
          BoundaryCell* bc = *i;
          ba += bc->area();
        }
      }
    }

    double volume_error = (ev - cv) / ev;
    if (fatal and fabs(ev - cv) / (max_norm) > LINSOLVETOL)
    {
      std::stringstream err;
      err << std::setprecision(10) << " !!!!! volume test failed: !!!!! "
          << "eleID=" << e.id() << "  "
          << "ve=" << ev << "  "
          << "vc=" << cv << "  "
          << "vd= " << ev - cv << "  "
          << "err=" << volume_error << "  "
          << "tol=" << LINSOLVETOL * max_norm;
      std::cout << err.str() << std::endl;

      // Cut test is written for level-set cases as well.
      debug_dump(&e, __FILE__, __LINE__);
      FOUR_C_THROW("{}", err.str());
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * ?
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::print_cell_stats()
{
  const int vectorlength = 21;
  unsigned cut = 0;
  std::vector<int> numvc(vectorlength, 0);
  std::map<Core::FE::CellType, std::vector<int>> numcells;

  // loop over elements
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element& e = *i->second;
    if (e.is_cut())
    {
      cut += 1;
      const plain_volumecell_set& volumecells = e.volume_cells();
      numvc[std::min(
          static_cast<int>(volumecells.size() - 1), static_cast<int>(numvc.size() - 1))] += 1;
      for (plain_volumecell_set::const_iterator i = volumecells.begin(); i != volumecells.end();
          ++i)
      {
        VolumeCell* vc = *i;
        std::map<Core::FE::CellType, int> cell_count;
        const plain_integrationcell_set& cells = vc->integration_cells();
        for (plain_integrationcell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
        {
          IntegrationCell* cell = *i;
          cell_count[cell->shape()] += 1;
        }
        const plain_boundarycell_set& bcells = vc->boundary_cells();
        for (plain_boundarycell_set::const_iterator i = bcells.begin(); i != bcells.end(); ++i)
        {
          BoundaryCell* bcell = *i;
          cell_count[bcell->shape()] += 1;
        }
        for (std::map<Core::FE::CellType, int>::iterator i = cell_count.begin();
            i != cell_count.end(); ++i)
        {
          Core::FE::CellType shape = i->first;
          int count = i->second;
          std::map<Core::FE::CellType, std::vector<int>>::iterator j = numcells.find(shape);
          if (j == numcells.end())
          {
            numcells[shape] = std::vector<int>(vectorlength, 0);
          }
          numcells[shape][std::min(static_cast<int>(numcells[shape].size() - 1), count - 1)] += 1;
        }
      }
    }
  }

  // loop over shadow elements
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element& e = *i->second;
    if (e.is_cut())
    {
      cut += 1;
      const plain_volumecell_set& volumecells = e.volume_cells();
      numvc[std::min(
          static_cast<int>(volumecells.size() - 1), static_cast<int>(numvc.size() - 1))] += 1;
      for (plain_volumecell_set::const_iterator i = volumecells.begin(); i != volumecells.end();
          ++i)
      {
        VolumeCell* vc = *i;
        std::map<Core::FE::CellType, int> cell_count;
        const plain_integrationcell_set& cells = vc->integration_cells();
        for (plain_integrationcell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
        {
          IntegrationCell* cell = *i;
          cell_count[cell->shape()] += 1;
        }
        const plain_boundarycell_set& bcells = vc->boundary_cells();
        for (plain_boundarycell_set::const_iterator i = bcells.begin(); i != bcells.end(); ++i)
        {
          BoundaryCell* bcell = *i;
          cell_count[bcell->shape()] += 1;
        }
        for (std::map<Core::FE::CellType, int>::iterator i = cell_count.begin();
            i != cell_count.end(); ++i)
        {
          Core::FE::CellType shape = i->first;
          int count = i->second;
          std::map<Core::FE::CellType, std::vector<int>>::iterator j = numcells.find(shape);
          if (j == numcells.end())
          {
            numcells[shape] = std::vector<int>(vectorlength, 0);
          }
          numcells[shape][std::min(static_cast<int>(numcells[shape].size() - 1), count - 1)] += 1;
        }
      }
    }
  }

  std::cout << "#elements = " << cut << " were cut of " << elements_.size() << " elements  and "
            << shadow_elements_.size() << " shadow elements in total\n";

  std::cout << "volume  cells: ";
  // std::streamsize w = std::cout.width( 4 );
  // std::copy( numvc.begin(), numvc.end(), std::ostream_iterator<int>( std::cout, " " ) );
  for (std::vector<int>::iterator i = numvc.begin(); i != numvc.end(); ++i)
  {
    int c = *i;
    if (c != 0)
    {
      std::cout << std::setw(4) << c << " ";
    }
    else
    {
      std::cout << "     ";
    }
  }
  std::cout << "\n";

  std::cout << "               ";
  for (int i = 1; i < vectorlength; ++i)
  {
    std::cout << std::setw(4) << i << " ";
  }
  std::cout << "   *\n";

  for (std::map<Core::FE::CellType, std::vector<int>>::iterator i = numcells.begin();
      i != numcells.end(); ++i)
  {
    Core::FE::CellType shape = i->first;
    std::vector<int>& nc = i->second;
    std::cout << Core::FE::cell_type_to_string(shape) << "\tcells: ";
    // std::copy( nc.begin(), nc.end(), std::ostream_iterator<int>( std::cout, " " ) );
    for (std::vector<int>::iterator i = nc.begin(); i != nc.end(); ++i)
    {
      int c = *i;
      if (c != 0)
      {
        std::cout << std::setw(4) << c << " ";
      }
      else
      {
        std::cout << "     ";
      }
    }
    std::cout << "\n";
  }
}

/*-------------------------------------------------------------------------------------*
 * print all facets
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::print_facets()
{
  //   for ( std::list<std::shared_ptr<Point> >::iterator i=points_.begin();
  //         i!=points_.end();
  //         ++i )
  //   {
  //     Point & p = **i;
  //     p.print();
  //     std::cout << "\n";
  //   }

  for (std::list<std::shared_ptr<Facet>>::iterator i = facets_.begin(); i != facets_.end(); ++i)
  {
    Facet& f = **i;
    f.print();
  }
}


/*-------------------------------------------------------------------------------------*
 * Write full Gmsh Output
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::dump_gmsh(std::string name)
{
  std::ofstream file(name.c_str());
  // file.precision(32); //higher precision!

  // ###############write all elements & shadow elements###############
  if (elements_.size() > 0 || shadow_elements_.size() > 0)
  {
    Cut::Output::gmsh_new_section(file, "Elements");
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); ++i)
      Cut::Output::gmsh_element_dump(file, &(*i->second));

    for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
        i != shadow_elements_.end(); ++i)
      Cut::Output::gmsh_element_dump(file, &(*i->second));
    Cut::Output::gmsh_end_section(file);
  }

  // ###############write all sides###############
  if (sides_.size() > 0)
  {
    Cut::Output::gmsh_new_section(file, "Sides");
    for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = sides_.begin();
        i != sides_.end(); ++i)
      Cut::Output::gmsh_side_dump(file, &(*i->second));
    Cut::Output::gmsh_end_section(file);
  }

  // ###############write all nodes###############
  if (sides_.size() > 0)
  {
    Cut::Output::gmsh_new_section(file, "Nodes");
    for (std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
      Cut::Output::gmsh_node_dump(file, &(*i->second));
    Cut::Output::gmsh_end_section(file);
  }

  // ###############write all points in pointpool###############
  if (pp_->get_points().size() > 0)
  {
    Cut::Output::gmsh_new_section(file, "PoolPoints");
    const RCPPointSet& points = pp_->get_points();
    for (RCPPointSet::const_iterator i = points.begin(); i != points.end(); ++i)
      Cut::Output::gmsh_point_dump(file, &(*(*i)), (*i)->id());
    Cut::Output::gmsh_end_section(file);
  }

  // ###############write all edges###############
  if (edges_.size() > 0)
  {
    Cut::Output::gmsh_new_section(file, "Edges");
    for (std::map<plain_int_set, std::shared_ptr<Edge>>::iterator i = edges_.begin();
        i != edges_.end(); ++i)
      Cut::Output::gmsh_edge_dump(file, &(*i->second));
    Cut::Output::gmsh_end_section(file);
  }

  // ###############write all lines###############
  if (lines_.size() > 0)
  {
    Cut::Output::gmsh_new_section(file, "Lines");
    for (std::list<std::shared_ptr<Line>>::iterator i = lines_.begin(); i != lines_.end(); ++i)
      Cut::Output::gmsh_line_dump(file, &(**i));
    Cut::Output::gmsh_end_section(file);
  }

  // ###############write all facets (or basically the facet points)###############
  if (facets_.size() > 0)
  {
    // ###############write all facets (or basically the facet lines)###############
    Cut::Output::gmsh_new_section(file, "Facet_Lines");
    for (std::list<std::shared_ptr<Facet>>::iterator i = facets_.begin(); i != facets_.end(); ++i)
      Cut::Output::gmsh_facet_dump(file, &(**i), "lines");
    Cut::Output::gmsh_end_section(file);

    // ###############write all triangulated facets ###############
    Cut::Output::gmsh_new_section(file, "Facets");
    for (std::list<std::shared_ptr<Facet>>::iterator i = facets_.begin(); i != facets_.end(); ++i)
      Cut::Output::gmsh_facet_dump(file, &(**i), "sides");
    Cut::Output::gmsh_end_section(file);

    // ###############write all cut facets all ###############
    Cut::Output::gmsh_new_section(file, "cut_Facets");
    for (std::list<std::shared_ptr<Facet>>::iterator i = facets_.begin(); i != facets_.end(); ++i)
    {
      if ((*i)->parent_side()->is_cut_side())
        Cut::Output::gmsh_facet_dump(file, &(**i), "sides", true);
    }
    Cut::Output::gmsh_end_section(file);

    // ###############write all triangulated facets all ###############
    Cut::Output::gmsh_new_section(file, "ele_Facets");
    for (std::list<std::shared_ptr<Facet>>::iterator i = facets_.begin(); i != facets_.end(); ++i)
    {
      if (!(*i)->parent_side()->is_cut_side())
        Cut::Output::gmsh_facet_dump(file, &(**i), "sides", true);
    }
    Cut::Output::gmsh_end_section(file);
  }

  // #############write level set information from cut if level set side exists ################
  bool haslevelsetside = false;
  // Does one element have a level set side?
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); i++)
  {
    Element* ele = &*i->second;
    haslevelsetside = ele->has_level_set_side();
    if (haslevelsetside) break;
  }

  if (haslevelsetside)
  {
    Cut::Output::gmsh_new_section(file, "LevelSetValues");
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
      Cut::Output::gmsh_level_set_value_dump(file, &(*i->second));
    Cut::Output::gmsh_end_section(file);

    Cut::Output::gmsh_new_section(file, "LevelSetGradient");
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
      Cut::Output::gmsh_level_set_gradient_dump(file, &(*i->second));
    Cut::Output::gmsh_end_section(file);

    Cut::Output::gmsh_new_section(file, "LevelSetOrientation");
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
      Cut::Output::gmsh_level_set_orientation_dump(file, &(*i->second));
    Cut::Output::gmsh_end_section(file);
  }
}

/*-------------------------------------------------------------------------------------*
 * ?
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::dump_gmsh_volume_cells(std::string name, bool include_inner)
{
  std::ofstream file(name.c_str());
  int count = 0;

  file.setf(std::ios::scientific, std::ios::floatfield);
  file.precision(16);

  file << "View \"VolumeCells\" {\n";
  for (std::list<std::shared_ptr<VolumeCell>>::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* vc = &**i;

    //    if ( true  ) // std::cout all volumecells - inside and outside
    if (include_inner or vc->position() != Point::inside)
    {
      const plain_integrationcell_set& integrationcells = vc->integration_cells();
      for (plain_integrationcell_set::const_iterator i = integrationcells.begin();
          i != integrationcells.end(); ++i)
      {
        IntegrationCell* ic = *i;
        ic->dump_gmsh(file, &count);
      }
    }

    count += 1;
  }
  file << "};\n";

  file << "View \"Elements, NumVcs\" {\n";
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); ++i)
  {
    Element* e = &*i->second;
    const plain_volumecell_set& volumes = e->volume_cells();
    count = volumes.size();
    for (plain_volumecell_set::const_iterator i = volumes.begin(); i != volumes.end(); ++i)
    {
      VolumeCell* vc = *i;
      if (include_inner or vc->position() != Point::inside)
      {
        const plain_integrationcell_set& integrationcells = vc->integration_cells();
        for (plain_integrationcell_set::const_iterator i = integrationcells.begin();
            i != integrationcells.end(); ++i)
        {
          IntegrationCell* ic = *i;
          ic->dump_gmsh(file, &count);
        }
      }
    }
  }
  for (std::map<int, std::shared_ptr<Element>>::iterator i = shadow_elements_.begin();
      i != shadow_elements_.end(); ++i)
  {
    Element* e = &*i->second;
    const plain_volumecell_set& volumes = e->volume_cells();
    count = volumes.size();
    for (plain_volumecell_set::const_iterator i = volumes.begin(); i != volumes.end(); ++i)
    {
      VolumeCell* vc = *i;
      if (include_inner or vc->position() != Point::inside)
      {
        const plain_integrationcell_set& integrationcells = vc->integration_cells();
        for (plain_integrationcell_set::const_iterator i = integrationcells.begin();
            i != integrationcells.end(); ++i)
        {
          IntegrationCell* ic = *i;
          ic->dump_gmsh(file, &count);
        }
      }
    }
  }
  file << "};\n";

  file << "View \"Node-ID\" {\n";
  for (std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
  {
    Node* n = &*i->second;
    Point* p = n->point();
    const double* x = p->x();
    file << "SP(" << x[0] << "," << x[1] << "," << x[2] << "){" << n->id() << "};\n";
  }
  file << "};\n";

  file << "View \"Node-Positions\" {\n";
  for (std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.begin(); i != nodes_.end(); ++i)
  {
    Node* n = &*i->second;
    Point* p = n->point();
    const double* x = p->x();
    file << "SP(" << x[0] << "," << x[1] << "," << x[2] << "){" << p->position() << "};\n";
  }
  file << "};\n";


  // Does there exist a Level Set cut side?
  bool haslevelsetside = false;
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); i++)
  {
    Element* ele = &*i->second;
    haslevelsetside = ele->has_level_set_side();

    if (haslevelsetside) break;
  }

  if (haslevelsetside)
  {
    file << "View \"LevelSetValues\" {\n";
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
    {
      Element* ele = &*i->second;
      Cut::Output::gmsh_level_set_value_dump(file, ele);
    }
    file << "};\n";

    file << "View \"LevelSetGradient\" {\n";
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
    {
      Element* ele = &*i->second;
      Cut::Output::gmsh_level_set_gradient_dump(file, ele);
    }
    file << "};\n";

    file << "View \"LevelSetOrientation\" {\n";
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
    {
      Element* ele = &*i->second;
      Cut::Output::gmsh_level_set_orientation_dump(file, ele);
    }
    file << "};\n";
  }
}


/*-------------------------------------------------------------------------------------*
 * Write all integration cells and boundarycells intro GMSH output file
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::dump_gmsh_integration_cells(std::string name)
{
  std::ofstream file(name.c_str());
  file << "View \"IntegrationCells\" {\n";
  for (std::list<std::shared_ptr<IntegrationCell>>::iterator i = integrationcells_.begin();
      i != integrationcells_.end(); ++i)
  {
    IntegrationCell* ic = &**i;
    ic->dump_gmsh(file);
  }
  file << "};\n";

  // Write BCs and their normals for outside and inside VolumeCells:
  dump_gmsh_boundary_cells(file, Point::outside);
  dump_gmsh_boundary_cells(file, Point::inside);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Mesh::dump_gmsh_boundary_cells(std::ofstream& file, Point::PointPosition pos)
{
  // Write BCs for "pos" VolumeCell:
  file << "View \"BoundaryCells " << Point::point_position_to_string(pos) << "\" {\n";
  for (std::list<std::shared_ptr<VolumeCell>>::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* volcell = &**i;
    if (volcell->position() == pos)
    {
      plain_boundarycell_set bc_cells = volcell->boundary_cells();
      for (plain_boundarycell_set::iterator j = bc_cells.begin(); j != bc_cells.end(); ++j)
      {
        BoundaryCell* bc = *j;
        if (bc->is_valid()) bc->dump_gmsh(file);
      }
    }
  }
  file << "};\n";

  // write normal for boundary cells
  file << "View \"BoundaryCellsNormal " << Point::point_position_to_string(pos) << "\" {\n";
  for (std::list<std::shared_ptr<VolumeCell>>::iterator i = cells_.begin(); i != cells_.end(); ++i)
  {
    VolumeCell* volcell = &**i;
    if (volcell->position() == pos)
    {
      plain_boundarycell_set bc_cells = volcell->boundary_cells();
      for (plain_boundarycell_set::iterator j = bc_cells.begin(); j != bc_cells.end(); ++j)
      {
        BoundaryCell* bc = *j;
        if (bc->is_valid()) bc->dump_gmsh_normal(file);
      }
    }
  }
  file << "};\n";
}

/*-------------------------------------------------------------------------------------*
 * ?
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::dump_gmsh_volume_cells(std::string name)
{
  std::ofstream file(name.c_str());
  file << "View \"VolumeCells\" {\n";
  for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
      i != elements_.end(); i++)
  {
    Element& ele = *i->second;
    const plain_volumecell_set cells = ele.volume_cells();
    for (plain_volumecell_set::const_iterator j = cells.begin(); j != cells.end(); j++)
    {
      VolumeCell* vcc = *j;
      vcc->dump_gmsh(file);
    }
  }
  file << "};\n";

  file << "View \"BoundaryCells\" {\n";
  bool haslevelsetside = false;
  for (std::list<std::shared_ptr<BoundaryCell>>::iterator i = boundarycells_.begin();
      i != boundarycells_.end(); ++i)
  {
    BoundaryCell* bc = &**i;

    if (bc->get_facet()->belongs_to_level_set_side()) haslevelsetside = true;

    if (bc->is_valid()) bc->dump_gmsh(file);
  }
  file << "};\n";

  if (haslevelsetside)
  {
    file << "View \"LevelSetInfoOnFacet\" {\n";
    for (std::map<int, std::shared_ptr<Element>>::iterator i = elements_.begin();
        i != elements_.end(); i++)
    {
      Element* ele = &*i->second;
      const plain_facet_set facets = ele->facets();
      for (plain_facet_set::const_iterator j = facets.begin(); j != facets.end(); j++)
      {
        Facet* facet = *j;

        if (facet->on_cut_side())
        {
          Core::LinAlg::Matrix<3, 1> facet_triang_midpoint_coord(
              Core::LinAlg::Initialization::zero);

          if (facet->is_triangulated())
          {
            std::vector<std::vector<Point*>> facet_triang = facet->triangulation();
            Point* facet_triang_midpoint = (facet_triang[0])[0];

            // Choose midpoint of facet?
            facet_triang_midpoint->coordinates(&facet_triang_midpoint_coord(0, 0));
          }
          else
          {
            Core::LinAlg::Matrix<3, 1> cur;
            std::vector<Point*> pts = facet->points();
            for (std::vector<Point*>::iterator i = pts.begin(); i != pts.end(); i++)
            {
              Point* p1 = *i;
              p1->coordinates(cur.data());
              facet_triang_midpoint_coord.update(1.0, cur, 1.0);
            }
            facet_triang_midpoint_coord.scale(1.0 / pts.size());
          }

          file << "VT(";
          file << facet_triang_midpoint_coord(0, 0) << "," << facet_triang_midpoint_coord(1, 0)
               << "," << facet_triang_midpoint_coord(2, 0);

          file << "){";

          std::vector<double> normal = ele->get_level_set_gradient(
              facet_triang_midpoint_coord);  // facet->GetLevelSetFacetNormal(ele);
          file << normal[0] << "," << normal[1] << "," << normal[2];

          file << "};\n";
        }
      }
    }
  }
  file << "};\n";
}

/*-------------------------------------------------------------------------------------*
 * DebugDump to call before runtime error                                     ager 04/15
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::debug_dump(Cut::Element* ele, std::string file, int line)
{
  if (file != "" || line != -1)
    std::cout << "Cut::Mesh::DebugDump called in " << file << " in line " << line << std::endl;

  //  std::cout << "Skip Cut::Mesh::DebugDump for now -- hiermeier" << std::endl;
  //  return;
  std::stringstream str;
  str << ".full_debug_cut_output." << myrank_ << "_CUTFAIL.pos";
  std::string filename(Cut::Output::generate_gmsh_output_filename(str.str()));
  dump_gmsh(filename);

  ele->debug_dump();
}

/*-------------------------------------------------------------------------------------*
 * ? -> used in cut_tetmeshintersection
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::new_nodes_from_points(std::map<Point*, Node*>& nodemap)
{
  int nid = nodes_.size();
  for (std::map<Point*, Node*>::iterator i = nodemap.begin(); i != nodemap.end(); ++i)
  {
    Point* p = i->first;
    while (nodes_.contains(nid)) nid += 1;
    Node* n = get_node(nid, p->x());
    i->second = n;
    nid += 1;
  }
}


/*-------------------------------------------------------------------------------------*
 * get a map of node id and the pointer to the node
 *-------------------------------------------------------------------------------------*/
void Cut::Mesh::get_node_map(std::map<int, Node*>& nodemap)
{
  for (std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.begin(); i != nodes_.end(); i++)
  {
    int nid = i->first;

    nodemap.insert(std::pair<int, Node*>(nid, &*(i->second)));
  }
}


/*-------------------------------------------------------------------------------------*
    Returns the node with given id
 *-------------------------------------------------------------------------------------*/
Cut::Node* Cut::Mesh::get_node(int nid) const
{
  std::map<int, std::shared_ptr<Node>>::const_iterator i = nodes_.find(nid);
  if (i != nodes_.end())
  {
    return &*i->second;
  }
  return nullptr;
}

/*-------------------------------------------------------------------------------------*
    Returns the unique shadow node
    identified by given nids of a quad8 boundary side or all nodes of hex20 element
    for the inner shadow node
    Remark: Do not identify a shadow node via its Id, since it is not unique over processors
 *-------------------------------------------------------------------------------------*/
Cut::Node* Cut::Mesh::get_node(const plain_int_set& nids) const
{
  std::map<plain_int_set, Node*>::const_iterator i = shadow_nodes_.find(nids);
  if (i != shadow_nodes_.end())
  {
    return &*i->second;
  }
  return nullptr;
}


/*-------------------------------------------------------------------------------------*
    If node with the given id exists return the node, else create a new node with
    given coordinates and levelset value
 *-------------------------------------------------------------------------------------*/
Cut::Node* Cut::Mesh::get_node(int nid, const double* xyz, double lsv, double tolerance)
{
  std::map<int, std::shared_ptr<Node>>::iterator i = nodes_.find(nid);
  if (i != nodes_.end())
  {
    return &*i->second;
  }
  if (xyz == nullptr) FOUR_C_THROW("cannot create node without coordinates");

  //   Point * p = pp_->GetPoint( xyz, nullptr, nullptr, MINIMALTOL );
  //   if ( p!=nullptr )
  //   {
  //     // We already have a point at this location. See if there is a node to
  //     // it. If so, that's the one.
  //     for ( std::map<int, std::shared_ptr<Node> >::iterator i=nodes_.begin();
  //           i!=nodes_.end();
  //           ++i )
  //     {
  //       std::shared_ptr<Node> n = i->second;
  //       if ( n->point()==p )
  //       {
  //         nodes_[nid] = n;
  //         return nullptr;
  //       }
  //     }
  //   }
  pp_->set_merge_strategy(PointpoolMergeStrategy::InitialLoad);
  Point* p = new_point(xyz, nullptr, nullptr, tolerance);
  pp_->set_merge_strategy(PointpoolMergeStrategy::NormalCutLoad);
  Node* n = new Node(nid, p, lsv);
  nodes_[nid] = std::shared_ptr<Node>(n);
#ifdef CUT_DUMPCREATION
  if (cutmesh_)
    std::cout << "GetCutNode( " << nid << ", ";
  else
    std::cout << "GetNode( " << nid << ", ";
  std::copy(xyz, xyz + 3, std::ostream_iterator<double>(std::cout, ", "));
  std::cout << lsv << " );\n";
#endif
  return n;
}


/*-------------------------------------------------------------------------------------*
 * GetNode routine for shadow nodes (e.g. center nodes of hex20 element),
 * find the unique shadow node (if existent) using the eight quad8 nodes of a quad8 side as key
 * in case of a boundary center node, and use the 20 nodes of the hex20 element to identify the
 * inner shadow/center node of the hex20 element
 *-------------------------------------------------------------------------------------*/
Cut::Node* Cut::Mesh::get_node(const plain_int_set& nids, const double* xyz, double lsv)
{
  std::map<plain_int_set, Node*>::iterator i = shadow_nodes_.find(nids);
  if (i != shadow_nodes_.end())
  {
    return &*i->second;
  }
  int nid = -shadow_nodes_.size() - 1;
  if (nodes_.find(nid) != nodes_.end())
  {
    FOUR_C_THROW("shadow node already exists");
  }

  // Remark: the nid of a shadow node is not unique over processors, consequently numbered with
  // negative integers on each proc
  Node* n = get_node(nid, xyz, lsv);
  shadow_nodes_[nids] = n;
  return n;
}


/*-------------------------------------------------------------------------------------*
 * get the edge with begin node and end node
 *-------------------------------------------------------------------------------------*/
Cut::Edge* Cut::Mesh::get_edge(Node* begin, Node* end)
{
  if (begin->point() == end->point()) FOUR_C_THROW("edge between same point");

  plain_int_set nids;
  nids.insert(begin->id());
  nids.insert(end->id());

  std::vector<Node*> nodes(2);
  nodes[0] = begin;
  nodes[1] = end;

  return get_edge(nids, nodes, *shards::getCellTopologyData<shards::Line<2>>());
}


/*-------------------------------------------------------------------------------------*
 * ?
 *-------------------------------------------------------------------------------------*/
Cut::Edge* Cut::Mesh::get_edge(const plain_int_set& nids, const std::vector<Node*>& nodes,
    const CellTopologyData& edge_topology)
{
  std::map<plain_int_set, std::shared_ptr<Edge>>::iterator i = edges_.find(nids);
  if (i != edges_.end())
  {
    return &*i->second;
  }

  // Search for edges between those points from an other mesh. This might
  // happen if mesh and cut mesh use the same nodes and thus the same
  // edges. This will happen with double cuts.

  Point* p1 = nodes[0]->point();
  Point* p2 = nodes[1]->point();

  plain_edge_set edges;
  p1->common_edge(p2, edges);
  for (plain_edge_set::iterator i = edges.begin(); i != edges.end(); ++i)
  {
    Edge* e = *i;
    Point* ep1 = e->begin_node()->point();
    Point* ep2 = e->end_node()->point();
    if (p1 == ep1 and p2 == ep2)
    {
      edges_[nids] = Core::Utils::shared_ptr_from_ref(*e);
      return e;
    }
    if (p1 == ep2 and p2 == ep1)
    {
      edges_[nids] = Core::Utils::shared_ptr_from_ref(*e);
      return e;
    }
  }

  //     if ( nodes[0]->point()==nodes[1]->point() )
  //       FOUR_C_THROW( "edge between same point" );
  edges_[nids] = Edge::create(edge_topology.key, nodes);

  //  edges_[nids] = Teuchos::rcp( e );
  return edges_[nids].get();
}

/*----------------------------------------------------------------------------*
 * get the side that contains the nodes with the following node ids
 *----------------------------------------------------------------------------*/
Cut::Side* Cut::Mesh::get_side(std::vector<int>& nids) const
{
  // create a sorted vector
  plain_int_set node_ids;

  for (unsigned int i = 0; i < nids.size(); i++)
  {
    node_ids.insert(nids[i]);
  }

  std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = sides_.find(node_ids);
  if (i != sides_.end())
  {
    return &*i->second;
  }
  return nullptr;
}


/*----------------------------------------------------------------------------*
 * ???
 *----------------------------------------------------------------------------*/
Cut::Side* Cut::Mesh::get_side(
    int sid, const std::vector<int>& nids, const CellTopologyData* top_data)
{
  unsigned nc = top_data->node_count;
  std::vector<Node*> nodes;
  nodes.reserve(nc);

  for (unsigned i = 0; i < nc; ++i)
  {
    nodes.push_back(get_node(nids[i], static_cast<double*>(nullptr)));
  }

  return get_side(sid, nodes, top_data);
}


/*----------------------------------------------------------------------------*
 * ???
 *----------------------------------------------------------------------------*/
Cut::Side* Cut::Mesh::get_side(
    int sid, const std::vector<Node*>& nodes, const CellTopologyData* top_data)
{
  unsigned ec = top_data->edge_count;
  std::vector<Edge*> edges;
  edges.reserve(ec);

  plain_int_set nidset;

  for (unsigned i = 0; i < ec; ++i)
  {
    const CellTopologyData_Subcell& edge = top_data->edge[i];
    const CellTopologyData& edge_topology = *edge.topology;

    std::vector<Node*> edge_nodes;
    plain_int_set edge_nids;
    edge_nodes.reserve(edge_topology.node_count);
    for (unsigned j = 0; j < edge_topology.node_count; ++j)
    {
      edge_nids.insert(nodes[edge.node[j]]->id());
      edge_nodes.push_back(nodes[edge.node[j]]);
    }
    edges.push_back(get_edge(edge_nids, edge_nodes, edge_topology));

    std::copy(edge_nids.begin(), edge_nids.end(), std::inserter(nidset, nidset.begin()));
  }

  Side* s = get_side(sid, nidset, nodes, edges, *top_data);

  return s;
}


/*----------------------------------------------------------------------------*
 * ?
 *----------------------------------------------------------------------------*/
Cut::Side* Cut::Mesh::get_side(int sid, const plain_int_set& nids, const std::vector<Node*>& nodes,
    const std::vector<Edge*>& edges, const CellTopologyData& side_topology)
{
  std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = sides_.find(nids);
  if (i != sides_.end())
  {
    return &*i->second;
  }
  // Facet * f = new Facet;
  // facets_.push_back( Teuchos::rcp( f ) );
  sides_[nids] = std::shared_ptr<Side>(Cut::Side::create(side_topology.key, sid, nodes, edges));

  int seid = -shadow_sides_.size() - 1;
  /* Remark: the seid of a shadow node is not unique over processors, consequently
   * numbered with negative integers on each proc */
  shadow_sides_[seid] = sides_[nids].get();
  return sides_[nids].get();
}


/*----------------------------------------------------------------------------*
 * Returns the element with given id
 *----------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::get_element(int eid)
{
  std::map<int, std::shared_ptr<Element>>::iterator ie = elements_.find(eid);
  if (ie != elements_.end())
  {
    return &*ie->second;
  }
  return nullptr;
}


/*----------------------------------------------------------------------------*
 * If element with the given id exists return the element, else create a new
 * element with given node ids. All details of the element are in cell
 * topology data
 *----------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::get_element(
    int eid, const std::vector<int>& nids, const CellTopologyData& top_data, bool active)
{
  std::map<int, std::shared_ptr<Element>>::iterator ie = elements_.find(eid);
  if (ie != elements_.end())
  {
    return &*ie->second;
  }

  unsigned nc = top_data.node_count;

  /*if( nc != nids.size() )
    FOUR_C_THROW("node details does not match with topology data");*/

  std::vector<Node*> nodes;
  nodes.reserve(nc);

  for (unsigned i = 0; i < nc; ++i)
  {
    nodes.push_back(get_node(nids[i], static_cast<double*>(nullptr)));
  }

  return get_element(eid, nodes, top_data, active);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Element* Cut::Mesh::get_element(
    int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active)
{
  switch (top_data.dimension)
  {
    case 1:
      return get_element<1>(eid, nodes, top_data, active);
    case 2:
      return get_element<2>(eid, nodes, top_data, active);
    case 3:
      return get_element<3>(eid, nodes, top_data, active);
    default:
      FOUR_C_THROW("Element dimension out of range! ( dim = {} )", top_data.dimension);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <>
Cut::Element* Cut::Mesh::get_element<1>(
    int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active)
{
  /* put the actual element into the sides vector and into the edge vector
   * --------------------------------------------------------------------------
   * see hint in CellTopologyData:
   * "Nodes, edges, and sides are subcells with a particular dimension.
   *  A cell has edges and sides only if its dimension is greater than one.
   *  - node has Dim == 0
   *  - edge has Dim == 1
   *  - side has Dim == dimension - 1. "
   *
   *  hiermeier 08/16*/

  // --------------------------------------------------------------------------
  // prepare side (copy element topology information into side)
  // --------------------------------------------------------------------------
  std::vector<Side*> sides(1, nullptr);
  const CellTopologyData& side_topology = top_data;

  std::vector<Node*> side_nodes;
  std::vector<int> side_nids;
  side_nodes.reserve(side_topology.node_count);
  side_nids.reserve(side_topology.node_count);
  for (unsigned j = 0; j < nodes.size(); ++j)
  {
    side_nids.push_back(nodes[j]->id());
    side_nodes.push_back(nodes[j]);
  }

  // --------------------------------------------------------------------------
  // create side edge (copy element topology information into edge)
  // --------------------------------------------------------------------------
  std::vector<Edge*> side_edges(1, nullptr);
  const CellTopologyData& edge_topology = top_data;

  std::vector<Node*> edge_nodes;
  plain_int_set edge_nids;
  edge_nodes.reserve(edge_topology.node_count);
  for (unsigned j = 0; j < edge_topology.node_count; ++j)
  {
    edge_nids.insert(side_nids[j]);
    edge_nodes.push_back(side_nodes[j]);
  }

  side_edges[0] = get_edge(edge_nids, edge_nodes, edge_topology);

  // --------------------------------------------------------------------------
  // create side
  // --------------------------------------------------------------------------
  plain_int_set side_nidset;
  std::copy(side_nids.begin(), side_nids.end(), std::inserter(side_nidset, side_nidset.begin()));

  sides[0] = get_side(-1, side_nidset, side_nodes, side_edges, side_topology);

  Element* e = nullptr;
  if (eid > -1)
  {
    elements_[eid] = Cut::Element::create(top_data.key, eid, sides, nodes, active);
    e = elements_[eid].get();
  }
  else
  {
    int seid = -shadow_elements_.size() - 1;
    /* Remark: the seid of a shadow node is not unique over processors,
     * consequently numbered with negative integers on each proc */
    shadow_elements_[seid] = Cut::Element::create(top_data.key, eid, sides, nodes, active);
    e = shadow_elements_[seid].get();
  }
  return e;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <>
Cut::Element* Cut::Mesh::get_element<2>(
    int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active)
{
  std::vector<Side*> side(1, nullptr);

  // --------------------------------------------------------------------------
  // prepare side (copy element topology information into side)
  // --------------------------------------------------------------------------
  const CellTopologyData& side_topology = top_data;  //*side.topology;

  std::vector<Node*> side_nodes;
  std::vector<int> side_nids;
  side_nodes.reserve(side_topology.node_count);
  side_nids.reserve(side_topology.node_count);
  for (unsigned j = 0; j < side_topology.node_count; ++j)
  {
    side_nids.push_back(nodes[j]->id());
    side_nodes.push_back(nodes[j]);
  }

  // --------------------------------------------------------------------------
  // create side edge (copy side topology information into edge)
  // --------------------------------------------------------------------------
  std::vector<Edge*> side_edges;
  side_edges.reserve(side_topology.side_count);
  for (unsigned j = 0; j < side_topology.side_count; ++j)
  {
    const CellTopologyData_Subcell& edge = side_topology.side[j];
    const CellTopologyData& edge_topology = *edge.topology;

    std::vector<Node*> edge_nodes;
    plain_int_set edge_nids;
    edge_nodes.reserve(edge_topology.node_count);
    for (unsigned j = 0; j < edge_topology.node_count; ++j)
    {
      edge_nids.insert(side_nids[edge.node[j]]);
      edge_nodes.push_back(side_nodes[edge.node[j]]);
    }

    side_edges.push_back(get_edge(edge_nids, edge_nodes, edge_topology));
  }

  // --------------------------------------------------------------------------
  // create side
  // --------------------------------------------------------------------------
  plain_int_set side_nidset;
  std::copy(side_nids.begin(), side_nids.end(), std::inserter(side_nidset, side_nidset.begin()));

  side[0] = get_side(-1, side_nidset, side_nodes, side_edges, side_topology);

  Element* e = nullptr;
  if (eid > -1)
  {
    elements_[eid] = Cut::Element::create(top_data.key, eid, side, nodes, active);
    e = elements_[eid].get();
  }
  else
  {
    int seid = -shadow_elements_.size() - 1;
    // Remark: the seid of a shadow node is not unique over processors, consequently numbered with
    // negative integers on each proc
    shadow_elements_[seid] = Cut::Element::create(top_data.key, eid, side, nodes, active);
    e = shadow_elements_[seid].get();
  }
  return e;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <>
Cut::Element* Cut::Mesh::get_element<3>(
    int eid, const std::vector<Node*>& nodes, const CellTopologyData& top_data, bool active)
{
  static int shards_to_fourc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 26, 20, 25, 24, 22, 21, 23, -1};

  unsigned sc = top_data.side_count;

  std::vector<Side*> sides;

  sides.reserve(sc);


  for (unsigned i = 0; i < sc; ++i)
  {
    const CellTopologyData_Subcell& side = top_data.side[i];
    const CellTopologyData& side_topology = *side.topology;

    std::vector<Node*> side_nodes;
    std::vector<int> side_nids;
    side_nodes.reserve(side_topology.node_count);
    side_nids.reserve(side_topology.node_count);
    for (unsigned j = 0; j < side_topology.node_count; ++j)
    {
      int nid = shards_to_fourc[side.node[j]];
      side_nids.push_back(nodes[nid]->id());
      side_nodes.push_back(nodes[nid]);
    }

    std::vector<Edge*> side_edges;
    side_edges.reserve(side_topology.edge_count);
    for (unsigned j = 0; j < side_topology.edge_count; ++j)
    {
      const CellTopologyData_Subcell& edge = side_topology.edge[j];
      const CellTopologyData& edge_topology = *edge.topology;

      std::vector<Node*> edge_nodes;
      plain_int_set edge_nids;
      edge_nodes.reserve(edge_topology.node_count);
      for (unsigned j = 0; j < edge_topology.node_count; ++j)
      {
        edge_nids.insert(side_nids[edge.node[j]]);
        edge_nodes.push_back(side_nodes[edge.node[j]]);
      }

      side_edges.push_back(get_edge(edge_nids, edge_nodes, edge_topology));
    }

    plain_int_set side_nidset;
    std::copy(side_nids.begin(), side_nids.end(), std::inserter(side_nidset, side_nidset.begin()));
    sides.push_back(get_side(-1, side_nidset, side_nodes, side_edges, side_topology));
  }

  Element* e = nullptr;
  if (eid > -1)
  {
    elements_[eid] = Cut::Element::create(top_data.key, eid, sides, nodes, active);
    e = elements_[eid].get();
  }
  else
  {
    int seid = -shadow_elements_.size() - 1;
    // Remark: the seid of a shadow node is not unique over processors, consequently numbered with
    // negative integers on each proc
    shadow_elements_[seid] = Cut::Element::create(top_data.key, eid, sides, nodes, active);
    e = shadow_elements_[seid].get();
  }
  return e;
}


/*----------------------------------------------------------------------------*
 * check if xyz-coordinates lie within the mesh's bounding box
 *----------------------------------------------------------------------------*/
bool Cut::Mesh::within_bb(const Core::LinAlg::SerialDenseMatrix& xyz)
{
  return bb_->within(norm_, xyz);
}


/*----------------------------------------------------------------------------*
 * check if the element lies within the bounding box
 *----------------------------------------------------------------------------*/
bool Cut::Mesh::within_bb(Element& element) { return bb_->within(norm_, element); }


/*----------------------------------------------------------------------------*
 * ?
 *----------------------------------------------------------------------------*/
void Cut::Mesh::create_side_ids_cut_test(int lastid)
{
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = sides_.begin();
      i != sides_.end(); ++i)
  {
    Side* s = &*i->second;
    if (!s->is_cut_side())
    {
      lastid += 1;
      s->set_id(lastid);
    }
  }

  cutmesh_ = true;
}


// used just for testing, in order to track down ids of the sides that are failing
int Cut::Mesh::create_side_ids_all_cut_test(int lastid)
{
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = sides_.begin();
      i != sides_.end(); ++i)
  {
    Side* s = &*i->second;
    if (!s->is_cut_side())
    {
      lastid -= 1;
      s->set_id(lastid);
    }
  }
  return lastid;
}

/*----------------------------------------------------------------------------*
 * ?
 *----------------------------------------------------------------------------*/
void Cut::Mesh::assign_other_volume_cells_cut_test(const Mesh& other)
{
  const std::list<std::shared_ptr<VolumeCell>>& other_cells = other.volume_cells();
  plain_volumecell_set cells;
  for (std::list<std::shared_ptr<VolumeCell>>::const_iterator i = other_cells.begin();
      i != other_cells.end(); ++i)
  {
    VolumeCell* vc = &**i;

    const plain_facet_set& facets = vc->facets();
    plain_facet_set cut_facets;
    std::remove_copy_if(facets.begin(), facets.end(), std::inserter(cut_facets, cut_facets.begin()),
        [](auto&& PH1) { return !PH1->on_cut_side(); });

    plain_side_set cut_sides;
    std::transform(cut_facets.begin(), cut_facets.end(),
        std::inserter(cut_sides, cut_sides.begin()),
        [](auto&& facet) { return facet->parent_side(); });

    if (cut_sides.size() > 1)
    {
      plain_side_set::iterator i = cut_sides.begin();
      Side* s1 = *i;
      ++i;
      Side* s2 = *i;
      Element* e = s1->common_element(s2);
      if (e == nullptr) FOUR_C_THROW("no common element on cut sides");
      e->assign_other_volume_cell(vc);
    }
    else
    {
      cells.insert(vc);
    }
  }

  if (cells.size() > 0)
  {
    std::stringstream str;
    str << cells.size() << " volume cells left. Need to handle those.";
    FOUR_C_THROW("{}", str.str());
  }
}

/*----------------------------------------------------------------------------*
 * Tests if the area of the facet created by boundary cells is equal on both
 * sides.
 *
 * WARNING: If set sharply (i.e. 10^-11) this will cause some test cases to
 *          fail for Combust, cut is done in local coordinates. Thus this makes
 *          it likely that the cut is sensitive to 10^-11? Why?
 *----------------------------------------------------------------------------*/
void Cut::Mesh::test_facet_area(bool istetmeshintersection)
{
  for (std::list<std::shared_ptr<Facet>>::iterator i = facets_.begin(); i != facets_.end(); ++i)
  {
    Facet* f = &**i;

    // This is a crude test. We do not demand so much here...
    double tolerance = 1e-7;
    // double tolerance = 1e-11; // sharper tolerance
    f->test_facet_area(tolerance, istetmeshintersection);
  }
}


FOUR_C_NAMESPACE_CLOSE
