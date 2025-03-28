// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_pointgraph_simple.hpp"

#include "4C_cut_facet.hpp"
#include "4C_cut_line.hpp"
#include "4C_cut_side.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Impl::SimplePointGraph1D::SimplePointGraph1D(Mesh& mesh, Element* element, Side* side,
    PointGraph::Location location, PointGraph::Strategy strategy)
    : PointGraph::PointGraph(1) /* explicit call of the protected base
                                   class constructor */
{
  if (element->n_dim() != 1) FOUR_C_THROW("This class is only meaningful for 1-D elements!");

  Cycle cycle;
  fill_graph(element, side, cycle, strategy);

  get_graph().find_cycles(element, side, cycle, location, strategy);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph1D::build_cycle(
    const std::vector<Point*>& edge_points, Cycle& cycle) const
{
  /* since a closed cycle is not possible for 1-D elements, we have to consider
   * also the starting edge point */
  for (std::vector<Point*>::const_iterator cit = edge_points.begin(); cit != edge_points.end();
      ++cit)
  {
    Point* p = *cit;
    cycle.push_back(p);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph1D::add_cut_lines_to_graph(
    Element* element, Side* side, Strategy strategy, Cycle& cycle)
{
  // this is completely unnecessary for the element_side
  if (not side->is_cut_side()) return;

  add_cut_points_to_cycle(element, side, cycle);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph1D::add_cut_points_to_cycle(
    Element* element, Side* side, Cycle& cycle)
{
  std::vector<Side*> ele_sides = element->sides();
  if (ele_sides.size() != 1) FOUR_C_THROW("There should be only one side in the 1-D case!");

  Side& ele_side = **ele_sides.begin();
  PointSet cuts;

  // get the cut point(s) (probably only one)
  ele_side.get_cut_points(element, *side, cuts);

  if (cuts.size() > 1)
    FOUR_C_THROW("There are a multiple cut points for the 2 line cut, check this case.");

  // add the found cut points and actual in the case of a cut_side, no other points
  // were added till now, to the cycle.
  for (PointSet::const_iterator cit = cuts.begin(); cit != cuts.end(); ++cit) cycle.push_back(*cit);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph1D::Graph::find_cycles(
    Element* element, Side* side, Cycle& cycle, Location location, Strategy strategy)
{
  // each point defines a own main cycle in 1D
  Edge* edge = element->sides()[0]->edges()[0];
  std::map<double, Cycle> sorted_cycles;

  for (std::vector<Point*>::const_iterator cit = cycle().begin(); cit != cycle().end(); ++cit)
  {
    Point* p = *cit;
    double t = p->t(edge);

    Cycle pseudo_1d_cycle(std::vector<Point*>(1, p));
    sorted_cycles[t] = pseudo_1d_cycle;
  }

  main_cycles_.reserve(sorted_cycles.size());
  for (std::map<double, Cycle>::const_iterator cit = sorted_cycles.begin();
      cit != sorted_cycles.end(); ++cit)
    main_cycles_.push_back(cit->second);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Impl::SimplePointGraph2D::SimplePointGraph2D(Mesh& mesh, Element* element, Side* side,
    PointGraph::Location location, PointGraph::Strategy strategy)
    : PointGraph::PointGraph(mesh, element, side, location, strategy),
      graph_2d_(std::dynamic_pointer_cast<SimplePointGraph2D::Graph>(graph_ptr()))
{
  if (element->n_dim() != 2) FOUR_C_THROW("This class is only meaningful for 2-D elements!");
  /* intentionally left blank, the work is done in the base class */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Impl::SimplePointGraph2D::Graph::has_single_points(Location location)
{
  // it is allowed for the 2-D case
  if (location == cut_side) return false;

  return PointGraph::Graph::has_single_points(location);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph2D::Graph::find_cycles(
    Element* element, Side* side, Cycle& cycle, Location location, Strategy strategy)
{
  if (location == element_side)
  {
    Cut::Impl::PointGraph::Graph::find_cycles(element, side, cycle, location, strategy);

    if (correct_rotation_direction_) correct_rotation_direction(element->sides()[0], main_cycles_);

    split_main_cycles_into_line_cycles();
  }
  else
  {
    if (all_points_.size() > 2)
    {
      for (auto i = all_points_.begin(); i != all_points_.end(); ++i) i->second->print();
      FOUR_C_THROW(
          "A cut_side point cycle with more than 2 points is currently not "
          "supported in the 2-D case, since it cannot happen for level-set "
          "cut cases ( always straight cuts ). Check this scenario if you run "
          "into this and extend the functionality.");
    }

    unsigned lcount = 0;
    std::vector<Point*> line(2, nullptr);
    for (std::map<int, Point*>::const_iterator cit = all_points_.begin(); cit != all_points_.end();
        ++cit)
    {
      line[lcount++] = cit->second;
    }
    main_cycles_ = std::vector<Cycle>(1, Cycle(line));
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph2D::Graph::split_main_cycles_into_line_cycles()
{
  std::vector<Cycle> line_main_cycles;
  for (std::vector<Cycle>::const_iterator ic = main_cycles_.begin(); ic != main_cycles_.end(); ++ic)
  {
    Cycle closed_cycle = *ic;
    const unsigned cycle_length = closed_cycle.size();

    std::vector<Point*> line(2, nullptr);
    // split the connected cycle into line segments
    for (unsigned i = 0; i < cycle_length; ++i)
    {
      line[0] = closed_cycle()[i % cycle_length];
      line[1] = closed_cycle()[(i + 1) % cycle_length];
      line_main_cycles.push_back(Cycle(line));
    }
  }
  // store the previously generated surface main cycles
  surface_main_cycles_ = main_cycles_;

  // replace the main cycles by the split line main cycles
  main_cycles_ = line_main_cycles;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Impl::SimplePointGraph2D::SimplePointGraph2D()
    : Cut::Impl::PointGraph(2),
      graph_2d_(std::dynamic_pointer_cast<SimplePointGraph2D::Graph>(graph_ptr()))
{ /* intentionally left blank */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph2D::find_line_facet_cycles(
    const plain_facet_set& line_facets, Element* parent_element)
{
  Cycle cycle;
  fill_graph_and_cycle_with_line_facets(line_facets, cycle);

  graph_2d_->set_correct_rotation_direction(true);

  find_cycles(parent_element, cycle);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph2D::fill_graph_and_cycle_with_line_facets(
    const plain_facet_set& line_facets, Cycle& cycle)
{
  plain_point_set point_set;
  for (plain_facet_set::const_iterator cit = line_facets.begin(); cit != line_facets.end(); ++cit)
  {
    Facet* f = *cit;

    if (not f->equals(Core::FE::CellType::line2))
      FOUR_C_THROW("This function works only for line facets!");

    const std::vector<Point*>& line_points = f->points();
    get_graph().add_edge(line_points[0], line_points[1]);

    point_set.insert(line_points.begin(), line_points.end());
  }

  cycle = Cycle(std::vector<Point*>(point_set.begin(), point_set.end()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph2D::find_cycles(Element* element, Cycle& cycle)
{
  Side* side = element->sides()[0];
  get_graph().find_cycles(element, side, cycle, PointGraph::element_side, PointGraph::all_lines);
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::SimplePointGraph2D::correct_rotation_direction(
    const Side* side, std::vector<Cycle>& cycles)
{
  Core::LinAlg::Matrix<2, 1> rs = Core::FE::get_local_center_position<2>(side->shape());
  Core::LinAlg::Matrix<3, 1> normal_side(Core::LinAlg::Initialization::uninitialized);

  Core::LinAlg::SerialDenseMatrix xyze_side(3, side->num_nodes());
  side->coordinates(xyze_side);

  // get the normal direction of the underlying side
  eval_normal_vectors<3>(side->shape(), xyze_side, rs, normal_side);

  std::vector<Point*> invert_cycle;
  for (std::vector<Cycle>::iterator it = cycles.begin(); it != cycles.end(); ++it)
  {
    Cycle& cycle = *it;

    const unsigned cycle_size = cycle().size();
    if (cycle_size < 3) FOUR_C_THROW("The cycle needs at least three points!");

    std::vector<Point*>::const_iterator it_begin = cycle().begin();
    std::vector<Point*>::const_iterator it_end = cycle().end() - 1;

    const Core::LinAlg::Matrix<3, 1> x1((*it_begin)->x(), true);
    const Core::LinAlg::Matrix<3, 1> x2((*(++it_begin))->x(), true);
    const Core::LinAlg::Matrix<3, 1> x3((*it_end)->x(), true);

    Core::LinAlg::Matrix<3, 1> x12;
    Core::LinAlg::Matrix<3, 1> x13;

    x12.update(1.0, x2, -1.0, x1);
    x13.update(1.0, x3, -1.0, x1);

    Core::LinAlg::Matrix<3, 1> normal_cycle;
    normal_cycle.cross_product(x12, x13);

    // not really necessary, but we do it anyway ...
    double nrm2 = normal_cycle.norm2();
    normal_cycle.scale(1.0 / nrm2);

    // check orientation and skip the remaining part, if the orientation is okay
    if (normal_cycle.dot(normal_side) > 0.0) continue;

    invert_cycle.reserve(cycle_size);

    it_begin = cycle().end() - 1;
    it_end = cycle().begin() - 1;
    for (std::vector<Point*>::const_iterator cit = it_begin; cit != it_end; --cit)
    {
      invert_cycle.push_back(*cit);
    }

    // replace the old the cycle by the inverted one
    cycle = Cycle(invert_cycle);
    invert_cycle.clear();
  }
}

FOUR_C_NAMESPACE_CLOSE
