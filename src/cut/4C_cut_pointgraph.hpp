// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_POINTGRAPH_HPP
#define FOUR_C_CUT_POINTGRAPH_HPP

// necessary due to usage of graph_t, vertex_t etc.
#include "4C_config.hpp"

#include "4C_cut_find_cycles.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations

namespace Cut
{
  class Point;
  class Line;
  class Side;
  class Element;
  class Mesh;

  namespace Impl
  {
    /// a planar graph that is used to create facets out of lines and points
    class PointGraph
    {
     public:
      /** specifies to which side the pointgraph and the related facets will
       *  belong. See the Side::MakeInternalFacets and Side::MakeOwnedSideFacets
       *  routines for more information. */
      enum Location
      {
        element_side,
        cut_side
      };

      enum Strategy
      {
        all_lines,
        own_lines  // used in levelset cut to create internal facets
      };

      /** \brief create a point graph object
       *
       *  Call this method to get the pointgraph which fits to your problem dimension.
       *
       *  \author hiermeier \date 11/16 */
      static PointGraph* create(Mesh& mesh, Element* element, Side* side,
          PointGraph::Location location, PointGraph::Strategy strategy);

     protected:
      class Graph
      {
       public:
        /// constructor
        Graph()
        {
          // empty
        }

        virtual ~Graph() = default;

        void add_edge(int row, int col);

        void add_edge(Point* p1, Point* p2);

        Point* get_point(int i);

        void print(std::ostream& stream = std::cout);

        void plot_all_points(std::ostream& stream = std::cout);

        void plot_points(Element* element);

        /** Creates maincycles (outer polygons) and holecycles (inner polygons = holes)
         *  of the selfcut graph */
        void find_cycles(Side* side, Cycle& cycle);

        virtual void find_cycles(
            Element* element, Side* side, Cycle& cycle, Location location, Strategy strategy);

        /*!
        \brief Any edge with single point in the graph is deleted
         */
        void fix_single_points(Cycle& cycle);

        virtual bool has_single_points(Location location);

        virtual bool has_touching_edge(Element* element, Side* side);

        // Simplify connection if the single point lies close to the nodal point
        // and touches the same edges as the nodal point
        // however because of the way pointgraph is constructed only connection from nodal
        // to this point is created
        // In this case we remove connection to nodal point, and treat cut point and "new nodal
        // point"
        virtual bool simplify_connections(Element* element, Side* side);

        void gnuplot_dump_cycles(const std::string& filename, const std::vector<Cycle>& cycles);

        std::map<int, plain_int_set> graph_;
        std::map<int, Point*> all_points_;
        std::vector<Cycle> main_cycles_;
        std::vector<std::vector<Cycle>> hole_cycles_;

      };  // struct Graph

      /** empty constructor (for derived classes only)
       *
       *  \author hiermeier \date 11/16 */
      PointGraph(unsigned dim) : graph_(create_graph(dim)) { /* intentionally left blank */ };

     public:
      typedef std::vector<Cycle>::iterator facet_iterator;
      typedef std::vector<std::vector<Cycle>>::iterator hole_iterator;

      PointGraph(Mesh& mesh, Element* element, Side* side, Location location, Strategy strategy);

      /// Constructor for the selfcut
      PointGraph(Side* side);

      /// destructor
      virtual ~PointGraph() = default;

      facet_iterator fbegin() { return get_graph().main_cycles_.begin(); }

      facet_iterator fend() { return get_graph().main_cycles_.end(); }

      hole_iterator hbegin() { return get_graph().hole_cycles_.begin(); }

      hole_iterator hend() { return get_graph().hole_cycles_.end(); }

      void print() { get_graph().print(); }

     protected:
      /*! \brief Graph is filled with all edges that are created due to additional
       *  cut points and cut lines */
      void fill_graph(Element* element, Side* side, Cycle& cycle, Strategy strategy);

      /** Graph is filled with all edges of the selfcut: uncutted edges,
       *  selfcutedges and new splitted edges; but no the cutted edges */
      void fill_graph(Side* side, Cycle& cycle);

      /** \brief add cut lines to graph
       *
       *  no need to add any more point to cycle because cut lines just join already
       *  existing points on the edge. making cut lines do not introduce additional
       *  points */
      virtual void add_cut_lines_to_graph(Element* element, Side* side, Strategy strategy);

      virtual void build_cycle(const std::vector<Point*>& edge_points, Cycle& cycle) const;

      /// access the graph of the most derived class
      virtual inline PointGraph::Graph& get_graph() { return *graph_; }

      virtual inline const std::shared_ptr<PointGraph::Graph>& graph_ptr() { return graph_; }

     private:
      std::shared_ptr<Cut::Impl::PointGraph::Graph> create_graph(unsigned dim);

      std::shared_ptr<Graph> graph_;
    };  // class PointGraph

    // non-member function
    bool find_cycles(graph_t& g, Cycle& cycle,
        std::map<vertex_t, Core::LinAlg::Matrix<3, 1>>& local, std::vector<Cycle>& cycles);

    std::shared_ptr<PointGraph> create_point_graph(Mesh& mesh, Element* element, Side* side,
        const PointGraph::Location& location, const PointGraph::Strategy& strategy);
  }  // namespace Impl
}  // namespace Cut



FOUR_C_NAMESPACE_CLOSE

#endif
