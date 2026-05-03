// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_pointgraph.hpp"

#include "4C_cut_mesh.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_pointgraph_simple.hpp"
#include "4C_cut_side.hpp"

#ifdef FOUR_C_CLANGCUDA_HOST_ONLY
  #undef __noinline__
#endif
#include <boost/graph/graphviz.hpp>

#include <cmath>
#include <iostream>
#include <iterator>


#define DEBUG_POINTGRAPH false
// #define CLN_CALC_OUTSIDE_KERNEL
#ifdef CLN_CALC_OUTSIDE_KERNEL
#include "4C_cut_clnwrapper.hpp"
#endif

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 * Constructor for the selfcut                                     wirtz 05/13
 *----------------------------------------------------------------------------*/
Cut::Impl::PointGraph::PointGraph(Side* side) : graph_(create_graph(side->n_dim()))
{
  Cycle cycle;
  fill_graph(side, cycle);
  if (get_graph().has_single_points(element_side))  // if any edge in graph has single point
  {
    get_graph().fix_single_points(cycle);  // delete single point edges
  }
  get_graph().find_cycles(side, cycle);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Impl::PointGraph::PointGraph(
    Mesh& mesh, Element* element, Side* side, Location location, Strategy strategy)
    : graph_(create_graph(element->n_dim()))
{
  // here we create the facets...
  Cycle cycle;
  fill_graph(element, side, cycle, strategy);

  // if any edge in graph has single point
  if (get_graph().has_single_points(location))
  {
    // NOTE: levelset method does not have complicated check for the single point. Feel free to
    // exend it
    if (side->is_level_set_side() or get_graph().simplify_connections(element, side) or
        (get_graph().has_touching_edge(element, side)))
    {
      get_graph().fix_single_points(cycle);  // delete single point edges
    }
    else
    {
      std::ofstream f("graph0.txt");
      get_graph().print(f);
      FOUR_C_THROW(
          "Pointgraph has single point.This shouldn't happen or we should understand why!");
    }
  }

  // Simplified graph strategy for three points as there is anyway just one way of creating the
  // facets! ager 09/19: this is a very good idea to do that but changed results of the testcase
  // xfluid_moving_torus* Thus, should be activated in a separate commit!

  try
  {
    get_graph().find_cycles(element, side, cycle, location, strategy);
  }
  catch (Core::Exception& err)
  {
    std::ofstream file("failed_pointgraph.pos");
    Cut::Output::gmsh_side_dump(file, side, std::string("Side"));

    // add cut lines to graph
    const std::vector<Line*>& cut_lines = side->cut_lines();

    for (std::vector<Line*>::const_iterator i = cut_lines.begin(); i != cut_lines.end(); ++i)
    {
      int line_index = i - cut_lines.begin();
      std::stringstream section_name;
      section_name << "Cut_lines" << line_index;
      Line* l = *i;
      Cut::Output::gmsh_new_section(file, section_name.str());
      Cut::Output::gmsh_line_dump(file, l, false, nullptr);
      Cut::Output::gmsh_end_section(file, false);
      // output distance between points of the line
      file << "// Distance between points of the line is"
           << Cut::distance_between_points(l->begin_point(), l->end_point()) << std::endl;
    }
    file.close();
    FOUR_C_THROW("");
  }
}

/*-------------------------------------------------------------------------------------*
 * Graph is filled wihl all edges of the selfcut: uncut edges, selfcutedges
 * and new split edges; but no the cut edges                          wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::fill_graph(Side* side, Cycle& cycle)
{
  const std::vector<Node*>& nodes = side->nodes();
  const std::vector<Edge*>& edges = side->edges();
  int end_pos = 0;

  // loop over all edges of the parent side
  for (std::vector<Edge*>::const_iterator i = edges.begin(); i != edges.end(); ++i)
  {
    Edge* e = *i;

    // get start and end node numbers corresponding to this edge
    int begin_pos = end_pos;
    end_pos = (end_pos + 1) % nodes.size();
    std::vector<Point*> edge_points;

    // get all points on this edge including start and end points
    // points are already sorted
    e->cut_point(nodes[begin_pos], nodes[end_pos], edge_points);

    // when edge of a side has "n" cut points, the edge itself is split into (n+1) edges
    // store all (n+1) edges to graph
    for (unsigned i = 1; i < edge_points.size(); ++i)  // no of edges = no of points-1
    {
      Point* p1 = edge_points[i - 1];
      Point* p2 = edge_points[i];
      get_graph().add_edge(p1, p2);
    }
    for (std::vector<Point*>::iterator i = edge_points.begin() + 1; i != edge_points.end(); ++i)
    {
      Point* p = *i;
      cycle.push_back(p);
    }
  }
  const plain_edge_set& selfcutedges = side->self_cut_edges();
  for (plain_edge_set::const_iterator i = selfcutedges.begin(); i != selfcutedges.end(); ++i)
  {
    Edge* selfcutedge = *i;
    get_graph().add_edge(selfcutedge->begin_node()->point(), selfcutedge->end_node()->point());
  }
}

/*----------------------------------------------------------------------------*
 * Get all edges created on this side after cut, store cycle of points on this
 * side to create facet. Also add cut lines to the graph
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::fill_graph(
    Element* element, Side* side, Cycle& cycle, Strategy strategy)
{
  const std::vector<Node*>& nodes = side->nodes();
  const std::vector<Edge*>& edges = side->edges();
  int end_pos = 0;

#if DEBUG_POINTGRAPH
  std::cout << "Filling graph" << std::endl;
#endif

  // loop over all edges of the parent side
  for (std::vector<Edge*>::const_iterator i = edges.begin(); i != edges.end(); ++i)
  {
    Edge* e = *i;

#if DEBUG_POINTGRAPH
    int index = i - edges.begin();
    std::cout << "Processing edge with index " << index << " and Id=" << e->Id() << std::endl;
#endif

    // get start and end node numbers corresponding to this edge
    int begin_pos = end_pos;
    end_pos = (end_pos + 1) % nodes.size();

    std::vector<Point*> edge_points;

    // get all points on this edge including start and end points
    // points are already sorted
    e->cut_point(nodes[begin_pos], nodes[end_pos], edge_points);

#if DEBUG_POINTGRAPH
    std::cout << "Number of points on the current edge is " << edge_points.size() << std::endl;
#endif

    // when edge of a side has "n" cut points, the edge itself is split into (n+1) edges
    // store all (n+1) edges to graph
    for (unsigned i = 1; i < edge_points.size(); ++i)  // number of edges = number of points-1
    {
      Point* p1 = edge_points[i - 1];
      Point* p2 = edge_points[i];
#if DEBUG_POINTGRAPH
      std::cout << "Adding line between points with ids " << p1->Id() << " and " << p2->Id()
                << std::endl;
#endif
      get_graph().add_edge(p1, p2);
    }

    build_cycle(edge_points, cycle);
  }

  add_cut_lines_to_graph(element, side, strategy);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::build_cycle(const std::vector<Point*>& edge_points, Cycle& cycle) const
{
  for (std::vector<Point*>::const_iterator i = edge_points.begin() + 1; i != edge_points.end(); ++i)
  {
    Point* p = *i;
    cycle.push_back(p);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::add_cut_lines_to_graph(Element* element, Side* side, Strategy strategy)
{
  const std::vector<Line*>& cut_lines = side->cut_lines();

  // add cut lines to graph
#if DEBUG_POINTGRAPH
  std::cout << "Adding cut lines to the graph " << std::endl;
#endif
  for (std::vector<Line*>::const_iterator i = cut_lines.begin(); i != cut_lines.end(); ++i)
  {
    Line* l = *i;

    // GetGraph().AddEdge(l->BeginPoint(), l->EndPoint());
    bool element_cut = l->is_cut(element);
    if (strategy == all_lines or element_cut)
      get_graph().add_edge(l->begin_point(), l->end_point());
#if DEBUG_POINTGRAPH
    l->BeginPoint()->print();
    l->EndPoint()->print();
#endif
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::add_edge(int row, int col)
{
  graph_[row].insert(col);
  graph_[col].insert(row);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::add_edge(Point* p1, Point* p2)
{
  all_points_[p1->id()] = p1;
  all_points_[p2->id()] = p2;

  add_edge(p1->id(), p2->id());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::print(std::ostream& stream)
{
  stream << "--- PointGraph::Graph ---\n";
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int p = i->first;
    plain_int_set& row = i->second;
    stream << p << ": ";
    for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
    {
      int p = *i;
      stream << p << " ";
    }
    stream << "\n";
  }
  stream << "\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::plot_all_points(std::ostream& stream)
{
  for (std::map<int, Point*>::iterator i = all_points_.begin(); i != all_points_.end(); ++i)
  {
    i->second->plot(stream);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::plot_points(Element* element)
{
  for (std::map<int, Point*>::iterator i = all_points_.begin(); i != all_points_.end(); ++i)
  {
    Point* p = i->second;
    std::cout << p->id() << "(" << p->is_cut(element) << ") ";
  }
  std::cout << "\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Impl::find_cycles(graph_t& g, Cut::Cycle& cycle,
    std::map<vertex_t, Core::LinAlg::Matrix<3, 1>>& local,
    std::vector<Cut::Cycle>& cycles) /* non-member function */
{
  name_map_t name_map = boost::get(boost::vertex_name, g);

  // Initialize the interior edge index
  edge_index_map_t e_index = boost::get(boost::edge_index, g);
  boost::graph_traits<graph_t>::edges_size_type edge_count = 0;
  boost::graph_traits<graph_t>::edge_iterator ei, ei_end;
  // updating property map of edges with indexes
  for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
    boost::put(e_index, *ei, edge_count++);

  using vec_t = std::vector<edge_t>;
  std::vector<vec_t> embedding(boost::num_vertices(g));


  // Use geometry to build embedding. The only safe way to do it.

#ifdef CLN_CALC_OUTSIDE_KERNEL
  using floatType = Core::CLN::ClnWrapper;
  // NOTE: Cln can be used, if one get problem with double arc and there is no other way to fix it
  // However, if running cln with as custom memory manager, this should be changed ( mostly to free
  // objects in a container, similarly as in cut_kernel )
#else
  using floatType = double;
#endif

  vertex_iterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
  {
    const Core::LinAlg::Matrix<3, 1>& pos = local[*vi];
#if DEBUG_POINTGRAPH
    std::cout << "First coordinate before subtraction " << std::setprecision(16) << pos
              << std::endl;
#endif

#if DEBUG_POINTGRAPH
    std::cout << "First point is " << name_map[*vi]->Id() << std::endl;
#endif
    std::map<floatType, vertex_t> arcs;
    adjacency_iterator ai, ai_end;
    for (boost::tie(ai, ai_end) = boost::adjacent_vertices(*vi, g); ai != ai_end; ++ai)
    {
      Core::LinAlg::Matrix<3, 1> d = local[*ai];

#if DEBUG_POINTGRAPH
      std::cout << "Adjacent point is " << name_map[*ai]->Id() << std::endl;
      std::cout << "Second coordinate before substrication " << std::setprecision(16) << d
                << std::endl;
#endif
      d.update(-1, pos, 1);

#ifdef CLN_CALC_OUTSIDE_KERNEL
      //                      Order of arguments is  __x___  ___y___
      floatType arc =
          cln::atan(cln::cl_float(d(0), cln::float_format(Core::CLN::ClnWrapper::precision_)),
              cln::cl_float(d(1), cln::float_format(Core::CLN::ClnWrapper::precision_)));
#else
      //                    Order of arguments is  __y___  ___x___
      floatType arc = std::atan2(d(1), d(0));
#endif

#if DEBUG_POINTGRAPH
      std::cout << "Arc is equal to " << arc << std::endl;
#endif

      std::map<floatType, vertex_t>::iterator j = arcs.find(arc);

      if (j != arcs.end())
      {
        // this can occur once when more than one nodes of the background element
        // has same coordinates (sudhakar)
        // check input file for two nodes (in same domain) having same coordinates
        std::stringstream err_msg;
        Point* first = name_map[*vi];
        Point* second = name_map[*ai];
        Point* previous = name_map[j->second];

        std::ofstream file("double_arc.pos");

        Cut::Output::gmsh_new_section(file, "NewLine");
        Cut::Output::gmsh_line_dump(file, first, second, first->id(), second->id(), false, nullptr);
        Cut::Output::gmsh_end_section(file);
        Cut::Output::gmsh_new_section(file, "OldLine");
        Cut::Output::gmsh_line_dump(
            file, first, previous, first->id(), previous->id(), false, nullptr);
        Cut::Output::gmsh_end_section(file, true);

        err_msg
            << "Numerical error: double arc when trying to create arc with between points with Id="
            << first->id() << " and  " << second->id()
            << "!. Arc of the same length exists between Ids " << first->id() << " and   "
            << previous->id() << std::endl;
        file.close();

        FOUR_C_THROW("{}", err_msg.str());
      }

      arcs[arc] = *ai;
    }

    vec_t& em = embedding[*vi];

// NOTE: We want to have embedding with clockwise ordering of edges. Otherwise it will produce an
// error
#if DEBUG_POINTGRAPH
    std::cout << "For vertex " << name_map[*vi]->Id() << " planar graph is edges with indexes: ";
#endif

    for (std::map<floatType, vertex_t>::iterator i = arcs.begin(); i != arcs.end(); ++i)
    {
      out_edge_iterator oi, oi_end;
      for (boost::tie(oi, oi_end) = boost::out_edges(*vi, g); oi != oi_end; ++oi)
      {
        edge_t e = *oi;
        if (boost::target(e, g) == i->second)
        {
#if DEBUG_POINTGRAPH
          std::cout << e_index[e] << " ; ";
#endif
          em.push_back(e);
          break;
        }
      }
    }
#if DEBUG_POINTGRAPH
    std::cout << "\n";
#endif
  }

  FaceVisitor vis(name_map, cycles);
  boost::planar_face_traversal(g, embedding.data(), vis);

#if DEBUG_POINTGRAPH
  for (std::vector<Cycle>::iterator i = cycles.begin(); i != cycles.end(); ++i)
  {
    Cycle& c = *i;
    c.TestUnique();
  }
#endif

  // boost face traversal will produce two cycles, in case if there  is one planar face ( surface
  // with no cut lines ), hence we need to remove redundant one and the other will serve as a facet
  // for us in case of normal configuration (more then one planar face),  we need to remove all the
  // instances of the full cycle produced by boost face traversal (which should be one), since the
  // set of small cycles would be the one creating facets

  bool save_first = cycles.size() == 2;

  int erase_count = 0;
  for (std::vector<Cycle>::iterator i = cycles.begin(); i != cycles.end();)
  {
    Cycle& c = *i;
    if (cycle.equals(c))
    {
      if (save_first and erase_count == 0)
      {
        ++i;
      }
      else
      {
        cycles.erase(i);
      }
      erase_count += 1;
    }
    else
    {
      ++i;
    }
  }

  if (erase_count > (save_first ? 2 : 1))
  {
    FOUR_C_THROW("more than one back facet");
  }

#if DEBUG_POINTGRAPH
  if (erase_count == 0)
  {
    std::cout << "ERASED 0 cycles ( no main cycle in the pointgraph)" << std::endl;
    std::cout << "Number of cycles is" << cycles.size() << std::endl;
    int counter = 0;
    for (std::vector<Cycle>::iterator i = cycles.begin(); i != cycles.end(); ++i, ++counter)
    {
      Cycle& c = *i;
      std::stringstream s;
      s << "Cycle_" << counter << ".pos";
      std::ofstream file(s.str());
      c.GmshDump(file);
      file.close();
    }
    std::ofstream file("main_cycle.pos");
    cycle.GmshDump(file);
    file.close();
  }
#endif

  return erase_count != 0;
}

/*-------------------------------------------------------------------------------------*
 * Creates maincycles (outer polygons) and holecycles (inner polygons = holes)
 * of the selfcut graph                                                     wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::find_cycles(Side* side, Cycle& cycle)
{
  graph_t g;

  // create boost graph

  name_map_t name_map = boost::get(boost::vertex_name, g);
  edge_index_map_t edge_index_map = boost::get(boost::edge_index, g);

  std::map<int, vertex_t> vertex_map;

  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int n = i->first;

    Point* p = get_point(n);
    vertex_t u = add_vertex(g);
    name_map[u] = p;
    vertex_map[n] = u;
  }

  int counter = 0;

  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int u = i->first;

    //    Point * p1 = GetPoint( u );

    plain_int_set& row = i->second;

    for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
    {
      int v = *i;
      //      Point * p2 = GetPoint( v );

      if (u < v)
      {
        edge_t e;
        bool inserted;
        boost::tie(e, inserted) = boost::add_edge(vertex_map[u], vertex_map[v], g);
        if (inserted)
        {
          edge_index_map[e] = counter;
          counter += 1;
        }
      }
    }
  }

  // All vertices are connected. If there is no cycle, done.
  if (boost::num_vertices(g) > boost::num_edges(g))
  {
    return;
  }


  // Use geometry to find the right embedding and find the cycles.
  // find local coordinates

  std::map<vertex_t, Core::LinAlg::Matrix<3, 1>> local;

  vertex_iterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
  {
    // prepare vars
    Point* p = name_map[*vi];
    Core::LinAlg::Matrix<3, 1> xyz(p->x());
    Core::LinAlg::Matrix<3, 1> tmpmat;

    // get coords
    side->local_coordinates(xyz, tmpmat);

    // add to map
    std::pair<vertex_t, Core::LinAlg::Matrix<3, 1>> tmppair(*vi, tmpmat);
    local.insert(tmppair);
  }

  // find unconnected components (main facet(s) and holes)

  std::vector<int> component(boost::num_vertices(g));

  int num_comp = boost::connected_components(
      g, boost::make_iterator_property_map(component.begin(), boost::get(boost::vertex_index, g)));

  // find cycles on each component

  if (num_comp == 1)
  {
    Cut::Impl::find_cycles(g, cycle, local, main_cycles_);
  }
  else if (num_comp > 1)
  {
    for (int i = 0; i < num_comp; ++i)
    {
      using filtered_graph_t = boost::filtered_graph<graph_t, EdgeFilter>;
      EdgeFilter filter(g, component, i);
      filtered_graph_t fg(g, filter);

      std::vector<Cycle> filtered_cycles;

      graph_t cg;
      boost::copy_graph(fg, cg);

      bool main_cycle = Cut::Impl::find_cycles(cg, cycle, local, filtered_cycles);

      if (main_cycle)
      {
        if (main_cycles_.size() != 0)
        {
          FOUR_C_THROW("one set of main cycles only");
        }
        std::swap(main_cycles_, filtered_cycles);
      }
      else
      {
        hole_cycles_.push_back(std::vector<Cycle>());
        std::swap(hole_cycles_.back(), filtered_cycles);
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::find_cycles(
    Element* element, Side* side, Cycle& cycle, Location location, Strategy strategy)
{
  graph_t g;

  // create boost graph

  name_map_t name_map = boost::get(boost::vertex_name, g);
  edge_index_map_t edge_index_map = boost::get(boost::edge_index, g);

  std::map<int, vertex_t> vertex_map;

  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int n = i->first;

    Point* p = get_point(n);

    // this check if it is  not levelset add all points and remove later, otherwise use this
    // strategy
    if (!(strategy == own_lines) or ((location == element_side) or (p->is_cut(element))))
    {
      vertex_t u = add_vertex(g);
      name_map[u] = p;
      vertex_map[n] = u;
    }
  }

  int counter = 0;
#if DEBUG_POINTGRAPH
  std::cout << "\n";
#endif
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int u = i->first;

    Point* p1 = get_point(u);

    if (!(strategy == own_lines) or ((location == element_side) or (p1->is_cut(element))))
    {
      plain_int_set& row = i->second;

      for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
      {
        int v = *i;
        Point* p2 = get_point(v);

        if (!(strategy == own_lines) or ((location == element_side) or (p2->is_cut(element))))
        {
          if (u < v)
          {
            edge_t e;
            bool inserted;
            boost::tie(e, inserted) = boost::add_edge(vertex_map[u], vertex_map[v], g);
            if (inserted)
            {
#if DEBUG_POINTGRAPH
              std::cout << "Inserting edge with edge_index " << counter << " between points "
                        << p1->Id() << " and " << p2->Id() << std::endl;
#endif
              edge_index_map[e] = counter;
              counter += 1;
            }
          }
        }
      }
    }
  }

  // All vertices are connected. If there is no cycle, done.
  if (boost::num_vertices(g) > boost::num_edges(g))
  {
    return;
  }

  if (strategy == own_lines)
  {
    // If just the lines owned by the element are here, use a "simpler"
    // algorithm that does not depend on geometry. This is required for
    // levelset cut sides than do not posses geometrical information.

    plain_cycle_set base_cycles;
    Cut::Impl::find_cycles(g, base_cycles);

    main_cycles_.reserve(base_cycles.size());

    for (plain_cycle_set::iterator i = base_cycles.begin(); i != base_cycles.end(); ++i)
    {
      cycle_t* c = *i;

      main_cycles_.push_back(Cycle());
      Cycle& pc = main_cycles_.back();
      pc.reserve(c->size());

      for (cycle_t::iterator i = c->begin(); i != c->end(); ++i)
      {
        vertex_t u = *i;
        pc.push_back(name_map[u]);
      }

      delete c;
    }
  }
  else
  {
    // Use geometry to find the right embedding and find the cycles.

    // find local coordinates

    std::map<vertex_t, Core::LinAlg::Matrix<3, 1>> local;

    vertex_iterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
    {
      // prepare vars
      Point* p = name_map[*vi];
      const Core::LinAlg::Matrix<3, 1> xyz(p->x(), true);
      Core::LinAlg::Matrix<3, 1> rst;

      // get local coordinates from the side element
      side->local_coordinates(xyz, rst);
#if DEBUG_POINTGRAPH
      std::cout << "For point" << p->Id() << std::endl;
      std::cout << "Local coordinate on the side are" << rst << std::endl;
#endif

      std::pair<vertex_t, Core::LinAlg::Matrix<3, 1>> tmppair(*vi, rst);
      local.insert(tmppair);
    }

    // find unconnected components (main facet(s) and holes)

    std::vector<int> component(boost::num_vertices(g));

    int num_comp = boost::connected_components(g,
        boost::make_iterator_property_map(component.begin(), boost::get(boost::vertex_index, g)));

    // find cycles on each component

    if (num_comp == 1)
    {
      bool main_cycle = Cut::Impl::find_cycles(g, cycle, local, main_cycles_);
      if (location == element_side and not main_cycle)
      {
        gnuplot_dump_cycles("cycles", main_cycles_);
        boost::print_graph(g, boost::get(boost::vertex_name, g));

        FOUR_C_THROW("cycle needs to contain side edges");
      }
    }
    else if (num_comp > 1)
    {
      for (int i = 0; i < num_comp; ++i)
      {
        using filtered_graph_t = boost::filtered_graph<graph_t, EdgeFilter>;
        EdgeFilter filter(g, component, i);
        filtered_graph_t fg(g, filter);

        std::vector<Cycle> filtered_cycles;

        graph_t cg;
        boost::copy_graph(fg, cg);
        bool main_cycle = Cut::Impl::find_cycles(cg, cycle, local, filtered_cycles);

        if (main_cycle)
        {
          if (main_cycles_.size() != 0)
          {
            FOUR_C_THROW("one set of main cycles only");
          }
          std::swap(main_cycles_, filtered_cycles);
        }
        else
        {
          hole_cycles_.push_back(std::vector<Cycle>());
          std::swap(hole_cycles_.back(), filtered_cycles);
        }
      }

      if (location == element_side and main_cycles_.size() == 0)
      {
        FOUR_C_THROW("cycle needs to contain side edges");
      }
    }
    else
    {
      if (location == element_side) FOUR_C_THROW("empty graph discovered");
    }
  }


  // filtering hole cycles and maincycles to include only internal cut_facets when creating facets
  // on the cut side
  if ((location == cut_side) and (!(strategy == own_lines)))
  {
    std::vector<Cycle> erased_cycles;
    for (std::vector<Cycle>::iterator it = main_cycles_.begin(); it != main_cycles_.end();)
    {
      const std::vector<Point*> cycle_points = (*it)();
      bool to_erase = false;
      for (std::vector<Point*>::const_iterator ip = cycle_points.begin(); ip != cycle_points.end();
          ++ip)
      {
        if (not(*ip)->is_cut(element))
        {
          erased_cycles.push_back(*it);
          to_erase = true;
          break;
        }
      }
      if (to_erase)
        it = main_cycles_.erase(it);
      else
        ++it;
    }
  }
}

/*---------------------------------------------------------------------------------*
 * In graph, if any edge has a single point, it will be deleted
 *---------------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::fix_single_points(Cycle& cycle)
{
  for (;;)
  {
    bool found = false;
    for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
    {
      int p = i->first;
      plain_int_set& row = i->second;
      if (row.size() < 2)
      {
        found = true;
        for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
        {
          int p2 = *i;
          plain_int_set& row2 = graph_[p2];
          row2.erase(p);
          if (row2.size() == 0) graph_.erase(p2);
        }
        graph_.erase(p);

        // There are degenerated cases. A very sharp triangle with one and the
        // same cut point on two edges close to the sharp node. In this case
        // the node will be dropped. The cycle will contain the cut point
        // twice. This needs to be fixed.

        cycle.drop_point(get_point(p));

        break;
      }
    }
    if (not found)
    {
      return;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Cut::Impl::PointGraph::Graph::has_single_points(Cut::Impl::PointGraph::Location location)
{
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    plain_int_set& row = i->second;
    if (row.size() < 2)
    {
      return true;
    }
  }
  return false;
}


// Check if this side has single point in the pointgraph, because other
// side was touched by the "tip" at this point
bool Cut::Impl::PointGraph::Graph::has_touching_edge(Element* element, Side* side)
{
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    plain_int_set& row = i->second;
    if (row.size() < 2)
    {
      Core::LinAlg::Matrix<3, 1> cut_pointxyz;
      // if there is  point in the poingraph, that have no less than neighbors
      Point* cut_point = all_points_[i->first];
      cut_point->coordinates(cut_pointxyz.data());

      for (plain_edge_set::const_iterator e = cut_point->cut_edges().begin();
          e != cut_point->cut_edges().end(); ++e)
      {
        Core::LinAlg::Matrix<3, 1> edge_vector;
        Edge* ed = *e;

        // get the vector from opposite node of the edge to cutting point
        if (cut_point->nodal_point(ed->nodes()))
        {
          if (ed->nodes()[0]->point() == cut_point)
          {
            (ed->nodes()[1]->point())->coordinates(edge_vector.data());
          }

          else if (ed->nodes()[1]->point() == cut_point)
          {
            (ed->nodes()[0]->point())->coordinates(edge_vector.data());
          }
          else
            FOUR_C_THROW("This should not happen or not implemented");

          edge_vector.update(-1.0, cut_pointxyz, 1.0);
          for (plain_side_set::const_iterator s = ed->sides().begin(); s != ed->sides().end(); ++s)
          {
            // getting side normal with respect to resp(0,0) by default local coordinates
            Core::LinAlg::Matrix<2, 1> resp;
            Core::LinAlg::Matrix<3, 1> norm_vec;
            Side* sd = *s;
            sd->normal(resp, norm_vec);

            for (plain_element_set::const_iterator el = sd->elements().begin();
                el != sd->elements().end(); ++el)
            {
              // getting element center for this element
              Core::LinAlg::Matrix<3, 1> element_center;
              Element* elmnt = *el;
              if (elmnt->shape() != Core::FE::CellType::hex8)
              {
                std::cout << "==| WARNING: Element Type != hex8 not supported by check "
                             "Graph::HasTouchingEdge! |==\n"
                          << "==| WARNING: Therefore we skip this test, please implement if you "
                             "use another element type! |=="
                          << std::endl;
                continue;
              }
              elmnt->element_center(element_center);
              // getting vector pointing outward the element
              Core::LinAlg::Matrix<3, 1> out_vec;
              out_vec.update(1.0, cut_pointxyz, -1.0, element_center);
              // if normal is pointing inwards, reverse it to point outwards
              if (out_vec.dot(norm_vec) < 0) norm_vec.scale(-1.0);
              if (norm_vec.dot(edge_vector) < 0)
              {
                FOUR_C_THROW("Single point problem, one elements is going inside another");
              }
            }
          }
        }
        else
        {
#ifdef DEBUG_POINTGRAPH
          std::ofstream file("touchign_element_detectiong_failed.pos");

          Cut::Output::gmsh_new_section(file, "Element");
          Cut::Output::gmsh_element_dump(file, element, false);
          Cut::Output::gmsh_end_section(file, false);

          Cut::Output::gmsh_side_dump(file, side, std::string("Side"));
          Cut::Output::gmsh_edge_dump(file, ed, std::string("EdgeContainingPoint"));
          Cut::Output::gmsh_point_dump(
              file, cut_point, cut_point->id(), std::string("CutPoint"), false, nullptr);


          if (row.size() == 1)
          {
            Point* next_point = all_points_[row[0]];
            file << "//Next point has Id" << next_point->id() << std::endl;
            cut_point->dump_connectivity_info();
            next_point->dump_connectivity_info();
          }
          else
            file << "//This point is not connected to anything\n";

          /// dump all the points of the pointgraph
          std::ofstream file_pgraph("pointgraph_dump.pos");
          for (std::map<int, Point*>::iterator it = all_points_.begin(); it != all_points_.end();
              ++it)
          {
            std::stringstream point_section_name;
            point_section_name << "Point" << (it->second)->id();
            Cut::Output::gmsh_new_section(file_pgraph, point_section_name.str());
            Cut::Output::gmsh_point_dump(
                file_pgraph, (it->second), (it->second)->id(), false, nullptr);
            Cut::Output::gmsh_end_section(file_pgraph, false);
            (it->second)->dump_connectivity_info();
          }

          file_pgraph.close();
          file.close();
#endif

          std::stringstream err_msg;
          err_msg << "The single cut point in pointgraph(Id=" << cut_point->id() << ")"
                  << " is not a nodal point of any of the edges connected to it (Not Touching)\n\
            This can for instance happen if your cut surface is not closed, so check your geometry first!\n";
          FOUR_C_THROW("{}", err_msg.str());
        }
      }
    }
  }
  return true;
}

bool Cut::Impl::PointGraph::Graph::simplify_connections(Element* element, Side* side)
{
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    plain_int_set& row = i->second;
    if (row.size() < 2)
    {
      if (row.size() == 1)
      {
        Point* single = all_points_[i->first];
        Point* other = all_points_[row.front()];
        if (single->nodal_point(side->nodes()))
        {
          // get touching edges of nodal point
          const std::vector<Edge*> side_edges = side->edges();
          std::vector<Edge*> point_side_edges;
          for (std::vector<Edge*>::const_iterator it = side_edges.begin(); it != side_edges.end();
              ++it)
          {
            if (single->nodal_point((*it)->nodes())) point_side_edges.push_back(*it);
          }
          std::vector<Edge*>::iterator it = point_side_edges.begin();
          // we are fine if single point touches all touching edges (on this side) of the cut point
          for (; it != point_side_edges.end(); ++it)
          {
            if (not other->is_cut(*it)) break;
          }

          if (it != point_side_edges.end())
            return false;
          else
            return true;
        }
        else
          return false;
      }
      else
        FOUR_C_THROW("Point in pointgraph is not connected to anything. Look into it!");
    }
  }
  return false;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::Impl::PointGraph::Graph::gnuplot_dump_cycles(
    const std::string& filename, const std::vector<Cycle>& cycles)
{
  int counter = 0;
  for (std::vector<Cycle>::const_iterator i = cycles.begin(); i != cycles.end(); ++i)
  {
    const Cycle& points = *i;

    std::stringstream str;
    str << filename << counter << ".plot";
    std::cout << str.str() << "\n";
    std::ofstream file(str.str().c_str());
    points.gnuplot_dump(file);

    counter += 1;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Point* Cut::Impl::PointGraph::Graph::get_point(int i)
{
  std::map<int, Point*>::iterator j = all_points_.find(i);
  if (j != all_points_.end()) return j->second;
  return nullptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Cut::Impl::PointGraph* Cut::Impl::PointGraph::create(Mesh& mesh, Element* element, Side* side,
    PointGraph::Location location, PointGraph::Strategy strategy)
{
  PointGraph* pg = nullptr;
  const unsigned dim = element->n_dim();
  switch (dim)
  {
    case 1:
      pg = new SimplePointGraph1D(mesh, element, side, location, strategy);
      break;
    case 2:
      pg = new SimplePointGraph2D(mesh, element, side, location, strategy);
      break;
    case 3:
      pg = new PointGraph(mesh, element, side, location, strategy);
      break;
    default:
      FOUR_C_THROW("Unsupported element dimension! ( dim = {} )", dim);
      break;
  }
  return pg;
};

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Cut::Impl::PointGraph::Graph> Cut::Impl::PointGraph::create_graph(unsigned dim)
{
  switch (dim)
  {
    case 1:
      return std::make_shared<SimplePointGraph1D::Graph>();
    case 2:
      return std::make_shared<SimplePointGraph2D::Graph>();
    case 3:
      return std::make_shared<PointGraph::Graph>();
    default:
      FOUR_C_THROW("Unsupported element dimension!");
  }
}

FOUR_C_NAMESPACE_CLOSE
