/*----------------------------------------------------------------------------*/
/*! \file

\brief Create a simple point graph for 1-D and 2-D elements ( embedded in a
       higher dimensional space )

\date Nov 12, 2016

\level 2

 *------------------------------------------------------------------------------------------------*/

#ifndef FOUR_C_CUT_POINTGRAPH_SIMPLE_HPP
#define FOUR_C_CUT_POINTGRAPH_SIMPLE_HPP

#include "4C_config.hpp"

#include "4C_cut_pointgraph.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  namespace Impl
  {
    /*--------------------------------------------------------------------------*/
    /** \brief Simplified version of the base class for 1-D elements
     *
     *  \author hiermeier \date 11/16 */
    class SimplePointGraph1D : public PointGraph
    {
     public:
      class Graph : public PointGraph::Graph
      {
       public:
        /// constructor
        Graph() : PointGraph::Graph()
        {
          // empty
        }

        /** \brief Simplified version of the base class implementation
         *
         *  \author hiermeier \date 11/16 */
        void find_cycles(Element* element, Side* side, Cycle& cycle, Location location,
            Strategy strategy) override;

      };  // struct Graph

      /// constructor
      SimplePointGraph1D(
          Mesh& mesh, Element* element, Side* side, Location location, Strategy strategy);

     protected:
      /** \brief Build the cycle of edge points [derived]
       *
       *  \param edge_points (in) : points of the current edge
       *  \param cycle       (out): created point cycle
       *
       *  \author hiermeier \date 11/16 */
      void build_cycle(const std::vector<Point*>& edge_points, Cycle& cycle) const override;

      /** \brief Add cut lines to the graph object [derived]
       *
       *  There are no cut lines in 1-D, thus we jump to add_cut_points_to_cycle,
       *  thus it is at least a little bit less confusing.
       *
       *  \author hiermeier \date 11/16 */
      virtual void add_cut_lines_to_graph(
          Element* element, Side* side, Strategy strategy, Cycle& cycle);

      /** \brief Add points instead of lines in 1-D
       *
       *  \param element (in) : background element
       *  \param side    (in) : element or cut side (see PointGraph::Location enumerator)
       *  \param cycle   (out): point cycle to be filled
       *
       *  This routine becomes for example important for the LevelSetCut, since a
       *  LevelSet side has no edges nor nodes. But we need the cut point for the boundary
       *  integration cell construction. */
      void add_cut_points_to_cycle(Element* element, Side* side, Cycle& cycle);

    };  // class SimplePointGraph_1D

    /*--------------------------------------------------------------------------*/
    /** \brief Modified version of the base class for 2-D elements
     *
     *  \author hiermeier \date 11/16 */
    class SimplePointGraph2D : public PointGraph
    {
     public:
      class Graph : public PointGraph::Graph
      {
       public:
        /// constructor
        Graph()
            : PointGraph::Graph(),
              correct_rotation_direction_(false){
                  // empty
              };

        /** \brief Overloaded version of the base class implementation
         *
         *  The base class implementation is called first. Afterwards
         *  a split of the created surface facet cycles into line cycles
         *  is performed.
         *
         *  \author hiermeier \date 11/16 */
        void find_cycles(Element* element, Side* side, Cycle& cycle, Location location,
            Strategy strategy) override;

        inline const std::vector<Cycle>& surface_main_cycles() const
        {
          return surface_main_cycles_;
        }

        bool has_single_points(Location location) override;

        void set_correct_rotation_direction(bool correct_rotation)
        {
          correct_rotation_direction_ = correct_rotation;
        }

       private:
        /**  \brief split the main cycles which are created by the base class
         *   implementation into line cycles
         *
         *   We store the generated surface facet main cycles as well, since we
         *   can reuse the implementation for the volume cell generation in 2-D.
         *
         *   \author hiermeier \date 01/17 */
        void split_main_cycles_into_line_cycles();

       private:
        std::vector<Cycle> surface_main_cycles_;

        bool correct_rotation_direction_;
      };  // struct Graph

      typedef std::vector<Cycle>::const_iterator surface_const_iterator;

      inline surface_const_iterator sbegin() const
      {
        return graph_2d_->surface_main_cycles().begin();
      }

      inline surface_const_iterator send() const { return graph_2d_->surface_main_cycles().end(); }

      inline unsigned num_surfaces() const { return graph_2d_->surface_main_cycles().size(); }

      /// constructor
      SimplePointGraph2D(
          Mesh& mesh, Element* element, Side* side, Location location, Strategy strategy);

      // empty constructor
      SimplePointGraph2D();

      void find_line_facet_cycles(const plain_facet_set& line_facets, Element* parent_element);

      /** \brief Test and correct the rotation direction of the given cycles such
       *  that the direction fits the rotation direction of the underlying side
       *
       *  The test is necessary, since the created cycles from the boost library seem
       *  to have no preferred rotation direction. First the normal on the parent side
       *  is calculated, afterwards the normal of the given cycle is calculated. If the
       *  inner-product of these normals is positive, the rotation direction is correct,
       *  otherwise the cycle direction is inverted.
       *
       *  \param side   (in)     : parent side ( for 2-D elements this is equivalent to the
       * element ) \param cycles (in/out) : cycles which have to be checked and inverted if
       * necessary
       *
       *  \author hiermeier \date 01/17 */
      static void correct_rotation_direction(const Side* side, std::vector<Cycle>& cycles);

     private:
      void fill_graph_and_cycle_with_line_facets(const plain_facet_set& line_facets, Cycle& cycle);

      void find_cycles(Element* element, Cycle& cycle);

     private:
      Teuchos::RCP<SimplePointGraph2D::Graph> graph_2d_;
    };  // class SimplePointGraph_2D

  }  // namespace Impl
}  // namespace Cut



FOUR_C_NAMESPACE_CLOSE

#endif
