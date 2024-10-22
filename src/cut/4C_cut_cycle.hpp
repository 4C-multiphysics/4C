// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CUT_CYCLE_HPP
#define FOUR_C_CUT_CYCLE_HPP

#include "4C_config.hpp"

#include "4C_cut_utils.hpp"


FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  class Cycle;
  class Point;
  class Edge;
  class Side;
  class Element;

  /*!
  \brief Contains closed cycle of points. A utility class for facet creation from this cycle of
  points
   */
  class Cycle
  {
   public:
    Cycle() {}

    Cycle(const std::vector<Point*>& points) : points_(points) {}

    /*!
    \brief Returns true if the cycle of points are suitable for creating a facet
     */
    bool is_valid() const;

    /*!
    \brief Returns true if all the points in the cylce falls within the element
     */
    bool is_cut(Element* element) const;

    /*!
    \brief Returns all the points in the cycle
     */
    const std::vector<Point*>& operator()() const { return points_; }
    /*!
    \brief add all lines from this cycle to the set of lines
    */
    void add(point_line_set& lines) const;

    /*!
    \brief Find the common edges among the cycle of points
     */
    void common_edges(plain_edge_set& edges) const;

    /*!
    \brief Get only the sides that are cut by any one of the points in the cycle
     */
    void intersection(plain_side_set& sides) const;

    /*!
    \brief Check whether these two cycles are one and the same
     */
    bool equals(const Cycle& other);

    /*!
    \brief Delete the specified point from the cycle, and make the resulting cycle valid
     */
    void drop_point(Point* p);

    void test_unique();

    /*!
    \brief Add the specified point to the cycle
      */
    void push_back(Point* p) { points_.push_back(p); }

    /*!
    \brief Reserve "s" number of points in the cycle
      */
    void reserve(unsigned s) { points_.reserve(s); }

    /*!
    \brief Delete all the points in the cycle
      */
    void clear() { points_.clear(); }

    /*!
    \brief Returns the number of points
      */
    unsigned size() const { return points_.size(); }

    /*!
    \brief Reverse the order of storing the points
     */
    void reverse();

    void swap(Cycle& other) { std::swap(points_, other.points_); }

    void gnuplot_dump(std::ostream& stream) const;

    // output cycle as a collection of lines into gmsh
    void gmsh_dump(std::ofstream& file) const;

    /// Print the stored points to the screen
    void print() const;

   private:
    std::vector<Point*> points_;
  };

}  // namespace Cut


FOUR_C_NAMESPACE_CLOSE

#endif
