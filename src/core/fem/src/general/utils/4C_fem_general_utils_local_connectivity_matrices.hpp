// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_LOCAL_CONNECTIVITY_MATRICES_HPP
#define FOUR_C_FEM_GENERAL_UTILS_LOCAL_CONNECTIVITY_MATRICES_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  //! 2 surfaces on a Hex16 element with 8 nodes per surface
  const int eleNodeNumbering_hex16_surfaces_q8[2][8] = {
      {0, 3, 2, 1, 7, 6, 5, 4}, {8, 9, 10, 11, 12, 13, 14, 15}};

  //! 4 surfaces on a Hex16 element with 6 nodes per surface
  const int eleNodeNumbering_hex16_surfaces_q6[4][6] = {
      {0, 1, 4, 8, 9, 12}, {1, 2, 5, 9, 10, 13}, {2, 3, 6, 10, 11, 14}, {3, 0, 7, 11, 8, 15}};

  //! 8 lines of a Hex16 element with 3 nodes per line
  const int eleNodeNumbering_hex16_lines_quad[8][3] = {{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {0, 3, 7},
      {8, 9, 12}, {9, 10, 13}, {10, 11, 14}, {8, 11, 15}};
  //! 4 lines in a Hex16 element with 2 nodes per line
  const int eleNodeNumbering_hex16_lines_lin[4][2] = {{0, 8}, {1, 9}, {2, 10}, {3, 11}};

  //! 2 surfaces on a Hex18 element with 9 nodes per surface
  const int eleNodeNumbering_hex18_surfaces_q9[2][9] = {
      {0, 3, 2, 1, 7, 6, 5, 4, 8}, {9, 10, 11, 12, 13, 14, 15, 16, 17}};
  //! 4 surfaces on a Hex18 element with 6 nodes per surface
  const int eleNodeNumbering_hex18_surfaces_q6[4][6] = {
      {0, 1, 4, 9, 10, 13}, {1, 2, 5, 10, 11, 14}, {2, 3, 6, 11, 12, 15}, {3, 0, 7, 12, 9, 16}};

  //! 8 lines of a Hex18 element with 3 nodes per line
  const int eleNodeNumbering_hex18_lines_quad[8][3] = {{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {0, 3, 7},
      {9, 10, 13}, {10, 11, 14}, {11, 12, 15}, {9, 12, 16}};
  //! 4 lines in a Hex18 element with 2 nodes per line
  const int eleNodeNumbering_hex18_lines_lin[4][2] = {{0, 9}, {1, 10}, {2, 11}, {3, 12}};

  //! 6 Surfaces of a Hex27 element with 9 nodes per surface
  const int eleNodeNumbering_hex27_surfaces[6][9] = {{0, 3, 2, 1, 11, 10, 9, 8, 20},
      {0, 1, 5, 4, 8, 13, 16, 12, 21}, {1, 2, 6, 5, 9, 14, 17, 13, 22},
      {2, 3, 7, 6, 10, 15, 18, 14, 23}, {0, 4, 7, 3, 12, 19, 15, 11, 24},
      {4, 5, 6, 7, 16, 17, 18, 19, 25}};

  //! 12 lines of a Hex27 element with 3 nodes per line
  const int eleNodeNumbering_hex27_lines[12][3] = {{0, 1, 8}, {1, 2, 9}, {2, 3, 10}, {0, 3, 11},
      {0, 4, 12}, {1, 5, 13}, {2, 6, 14}, {3, 7, 15}, {4, 5, 16}, {5, 6, 17}, {6, 7, 18},
      {4, 7, 19}};

  //! 4 Surfaces of a Tet10 element with 6 nodes per surface
  const int eleNodeNumbering_tet10_surfaces[4][6] = {
      {0, 1, 3, 4, 8, 7}, {1, 2, 3, 5, 9, 8}, {0, 3, 2, 7, 9, 6}, {0, 2, 1, 6, 5, 4}};

  //! 6 lines of a Tet10 element with 3 nodes per line
  const int eleNodeNumbering_tet10_lines[6][3] = {
      {0, 1, 4}, {1, 2, 5}, {0, 2, 6}, {0, 3, 7}, {1, 3, 8}, {2, 3, 9}};

  //! 2 triangular surfaces of wedge18 with 6 nodes per surface
  const int eleNodeNumbering_wedge18_trisurfaces[2][6] = {
      {2, 1, 0, 7, 6, 8}, {5, 3, 4, 14, 12, 13}};

  //! 3 rectangular surfaces of wedge18 with 9 nodes per surface
  const int eleNodeNumbering_wedge18_quadsurfaces[3][9] = {{0, 1, 4, 3, 6, 10, 12, 9, 15},
      {1, 2, 5, 4, 7, 11, 13, 10, 16}, {2, 0, 3, 5, 8, 9, 14, 11, 17}};

  //! 9 lines of a wedge18 element with 3 nodes per line
  const int eleNodeNumbering_wedge18_lines[9][3] = {{0, 1, 6}, {1, 2, 7}, {2, 0, 8}, {3, 4, 12},
      {4, 5, 13}, {5, 3, 14}, {0, 3, 9}, {1, 4, 10}, {2, 5, 11}};

  //! 1 rectangular surface of pyramid5
  const int eleNodeNumbering_pyramid5_quadsurfaces[1][4] = {{0, 3, 2, 1}};

  //! 4 triangular surfaces of pyramid5
  const int eleNodeNumbering_pyramid5_trisurfaces[4][3] = {
      {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {0, 4, 3}};

  //! 9 lines of a pyramid5 element with 2 nodes per line
  const int eleNodeNumbering_pyramid5_lines[8][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};

  //! 4 lines of a Quad9 element with 3 nodes per line
  const int eleNodeNumbering_quad9_lines[4][3] = {{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}};

  //! 2 lines of a Quad6 element with 2 nodes per line
  const int eleNodeNumbering_quad6_lines_lin[2][2] = {{1, 2}, {3, 0}};
  //! 2 lines of a Quad6 element with 2 nodes per line
  const int eleNodeNumbering_quad6_lines_quad[2][3] = {{0, 1, 4}, {2, 3, 5}};

  //! 4 lines of a nurbs4 element with 2 control points per line
  //! we apply the nurbs specific cartesian numbering
  //! \verbatim
  //!
  //!          2   (2)   3
  //!           +-------+
  //!           |       |
  //!           |       |(1)
  //!        (3)|       |
  //!           |       |
  //!           +-------+
  //!          0   (0)   1
  //!
  //! \endverbatim

  const int eleNodeNumbering_nurbs4_lines[4][2] = {{0, 1}, {1, 3}, {2, 3}, {2, 0}};


  //! 4 lines of a nurbs9 element with 3 control points per line
  //! we apply the nurbs specific cartesian numbering
  //! \verbatim
  //!
  //!            (2) -->
  //!          6    7    8
  //!           +---+---+
  //!           |       | (1)
  //!       (3) |       |
  //!           +   +   +  ^
  //!        ^ 3|   4   |5 |
  //!        |  |       |
  //!           +---+---+
  //!          0    1    2
  //!            (0) -->
  //!
  //! \endverbatim
  const int eleNodeNumbering_nurbs9_lines[4][3] = {{0, 1, 2}, {2, 5, 8}, {6, 7, 8}, {0, 3, 6}};

  //! 3 lines of a Tri6 element with 3 nodes per line
  const int eleNodeNumbering_tri6_lines[3][3] = {{0, 1, 3}, {1, 2, 4}, {2, 0, 5}};

  //! for each of the 12 lines, tell me the 2 neighbouring faces
  const int eleNodeNumbering_hex27_lines_surfaces[12][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 4},
      {1, 2}, {2, 3}, {3, 4}, {1, 5}, {2, 5}, {3, 5}, {4, 5}};

  //! for each of the 6 lines, tell me the 2 neighbouring faces
  const int eleNodeNumbering_tet10_lines_surfaces[6][2] = {
      {0, 3}, {1, 3}, {2, 3}, {2, 0}, {0, 1}, {1, 2}};

  //! for each of the 4 corner nodes, tell me the 3 neighbouring lines
  const int eleNodeNumbering_tet10_nodes_lines[4][3] = {{0, 2, 3}, {0, 1, 4}, {1, 2, 5}, {3, 4, 5}};

  //! for each node of the linear and quadratic quads, the reference coordinates are stored for
  //! quad9
  const double eleNodeNumbering_quad9_nodes_reference[9][2] = {{-1.0, -1.0}, {1.0, -1.0},
      {1.0, 1.0}, {-1.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, 0.0}};

  //! for each node the reference coordinates are stored for nurbs4
  const double eleNodeNumbering_nurbs4_nodes_reference[4][2] = {
      {-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}};

  //! for each node the reference coordinates are stored for nurbs9
  const double eleNodeNumbering_nurbs9_nodes_reference[9][2] = {{-1.0, -1.0}, {0.0, -1.0},
      {1.0, -1.0}, {-1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {-1.0, 1.0}, {0.0, 1.0}, {1.0, 1.0}};

  //! for each node the reference coordinates are stored for quad6
  const double eleNodeNumbering_quad6_nodes_reference[6][2] = {
      {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}, {0.0, -1.0}, {0.0, 1.0}};

  //! for each node the reference coordinates are stored for hex16
  const double eleNodeNumbering_hex16_nodes_reference[16][3] = {{-1.0, -1.0, -1.0},
      {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0},
      {0.0, 1.0, -1.0}, {-1.0, 0.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0},
      {-1.0, 1.0, 1.0}, {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {-1.0, 0.0, 1.0}};

  //! for each node the reference coordinates are stored for hex18
  const double eleNodeNumbering_hex18_nodes_reference[18][3] = {{-1.0, -1.0, -1.0},
      {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0},
      {0.0, 1.0, -1.0}, {-1.0, 0.0, -1.0}, {0.0, 0.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
      {1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}, {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0},
      {-1.0, 0.0, 1.0}, {0.0, 0.0, 1.0}};

  //! 6 surfaces of a nurbs8 element with 4 control points per surface
  //! we apply the nurbs specific cartesian numbering, only interpolated surfaces are valid!
  //! \verbatim
  //!                         v
  //!                        /
  //!                       /
  //!          X-------------------X
  //!         /|6         /       /|7
  //!  w     / |         /       / |
  //!  ^    /  |        /       /  |
  //!  |   /   |               /   |
  //!  |  /    |              /    |
  //!  | /     |             /     |
  //!   /      |            /      |
  //!  X-------------------X       |
  //!  |4      |           |5      |
  //!  |       |           |       |
  //!  |       |           |       |
  //!  |       X-----------|-------X
  //!  |      / 2          |      / 3
  //!  |     /             |     /
  //!  |    /              |    /
  //!  |   /               |   /
  //!  |  /                |  /
  //!  | /                 | /
  //!  |/                  |/
  //!  X-------------------X ----->u
  //!   0                   1
  //!
  //! \endverbatim

  const int eleNodeNumbering_nurbs8_surfaces[6][4] = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}, {2, 3, 6, 7}, {0, 2, 4, 6}, {1, 3, 5, 7}};
  //!
  const int eleNodeNumbering_nurbs8_lines[12][2] = {{0, 1}, {1, 3}, {3, 2}, {0, 2}, {0, 4}, {1, 5},
      {3, 7}, {2, 6}, {4, 5}, {5, 7}, {7, 6}, {4, 6}};

  //! 6 surfaces of a nurbs27 element with 9 control points per surface
  //! we apply the nurbs specific cartesian numbering, only interpolated surfaces are valid!
  //!
  //! \verbatim
  //!                          v
  //!                         /
  //!                        /
  //!          X---------X---------X
  //!         /|24      /|25      /|26
  //!  w     / |       / |       / |
  //!  ^    /  |      /  |      /  |
  //!  |   X---------X---------X   |
  //!  |  /|21 |    /|22 |    /|23 |
  //!  | / |   X---/-|---X---/-|---X
  //!   /  |  /|15/  |  /|16/  |  /|17
  //!  X---------X---------X   | / |
  //!  |18 |/  | |19 |/  | |20 |/  |
  //!  |   X-----|---X-----|---X   |
  //!  |  /|12 | |  /|13 | |  /|14 |
  //!  | / |   X-|-/-|---X-|-/-|---X
  //!  |/  |  /6 |/  |  /7 |/  |  /8
  //!  X---------X---------X   | /
  //!  |9  |/    |10 |/    |11 |/
  //!  |   X-----|---X-----|---X
  //!  |  /3     |  /4     |  /5
  //!  | /       | /       | /
  //!  |/        |/        |/
  //!  X---------X---------X ----->u
  //!   0         1         2
  //!
  //! \endverbatim

  const int eleNodeNumbering_nurbs27_surfaces[6][9] = {{0, 1, 2, 3, 4, 5, 6, 7, 8},
      {18, 19, 20, 21, 22, 23, 24, 25, 26}, {0, 1, 2, 9, 10, 11, 18, 19, 20},
      {6, 7, 8, 15, 16, 17, 24, 25, 26}, {2, 5, 8, 11, 14, 17, 20, 23, 26},
      {0, 3, 6, 9, 12, 15, 18, 21, 24}};

  //! 12 lines of a Nurbs8 / Nurbs27 element with 3 control points per line
  //!                          v
  //!                         /
  //!                   10   /
  //!          X---------X---------X
  //!         /|                  /|
  //!  w     / |                 / |
  //!  ^    /  |                /  |
  //!  | 11X   |               X9  |
  //!  |  /    |              /    |
  //!  | /     X7            /     X6
  //!   /      |            /      |
  //!  X---------X---------X       |
  //!  |       | 8         |       |
  //!  |       |           |       |
  //!  |       |         2 |       |
  //!  |       X---------X-|-------X
  //!  |      /            |      /
  //! 4X     /             X5    /
  //!  |    /              |    /
  //!  |  3X               |   X1
  //!  |  /                |  /
  //!  | /                 | /
  //!  |/                  |/
  //!  X---------X---------X ----->u
  //!            0
  //!
  const int eleNodeNumbering_nurbs27_lines[12][3] = {{0, 1, 2}, {2, 5, 8}, {6, 7, 8}, {0, 3, 6},
      {0, 9, 18}, {2, 11, 20}, {8, 17, 26}, {6, 15, 24}, {18, 19, 20}, {20, 23, 26}, {24, 25, 26},
      {18, 21, 24}};

  //! for each node the reference coordinates are stored for tri6
  const double eleNodeNumbering_tri6_nodes_reference[6][2] = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}};

  //! for each node the reference coordinates are stored for hex27
  const double eleNodeNumbering_hex27_nodes_reference[27][3] = {{-1.0, -1.0, -1.0},
      {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
      {1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}, {0.0, -1.0, -1.0}, {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0},
      {-1.0, 0.0, -1.0}, {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0},
      {0.0, -1.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {-1.0, 0.0, 1.0}, {0.0, 0.0, -1.0},
      {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 1.0},
      {0.0, 0.0, 0.0}};

  //! for each node the reference coordinates are stored for nurbs8
  const double eleNodeNumbering_nurbs8_nodes_reference[8][3] = {{-1.0, -1.0, -1.0},
      {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
      {-1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};

  //! for each node the reference coordinates are stored for nurbs27
  const double eleNodeNumbering_nurbs27_nodes_reference[27][3] = {{-1.0, -1.0, -1.0},
      {0.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {-1.0, 0.0, -1.0}, {0.0, 0, -1.0}, {1.0, 0, -1.0},
      {-1.0, 1.0, -1.0}, {0.0, 1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, -1.0, 0.0}, {0.0, -1.0, 0.0},
      {1.0, -1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0, 0.0}, {1.0, 0, 0.0}, {-1.0, 1.0, 0.0},
      {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {-1.0, -1.0, 1.0}, {0.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
      {-1.0, 0.0, 1.0}, {0.0, 0, 1.0}, {1.0, 0, 1.0}, {-1.0, 1.0, 1.0}, {0.0, 1.0, 1.0},
      {1.0, 1.0, 1.0}};

  //! reference parameter coordinates for a tet10
  const double eleNodeNumbering_tet10_nodes_reference[10][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.5, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.0, 0.5, 0.0},
      {0.0, 0.0, 0.5}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};

  //! reference parameter coordinates for a pyramid5
  const double eleNodeNumbering_pyramid5_nodes_reference[5][3] = {
      {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  //! reference parameter coordinates for a wedge18
  const double eleNodeNumbering_wedge18_nodes_reference[18][3] = {{0.0, 0.0, -1.0},
      {1.0, 0.0, -1.0}, {0.0, 1.0, -1.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0},
      {0.5, 0.0, -1.0}, {0.5, 0.5, -1.0}, {0.0, 0.5, -1.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0}, {0.5, 0.0, 1.0}, {0.5, 0.5, 1.0}, {0.0, 0.5, 1.0}, {0.5, 0.0, 0.0},
      {0.5, 0.5, 0.0}, {0.0, 0.5, 0.0}};

  //! reference parameter coordinates for a line3
  const double eleNodeNumbering_line3_nodes_reference[3][1] = {{-1.0}, {1.0}, {0.0}};

  //! reference parameter coordinates for a line4
  const double eleNodeNumbering_line4_nodes_reference[4][1] = {
      {-1.0}, {1.0}, {-1.0 / 3.0}, {1.0 / 3.0}};

  /*!
   * @brief Returns the number of nodes for the given discretization type
   *
   * @param[in] distype discretization type
   *
   * @return number of nodes
   */
  int get_number_of_element_nodes(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the number of corner nodes for an element of the given discretization type
   *
   * @param[in] distype discretization type
   * @return number of corner nodes
   */
  int get_number_of_element_corner_nodes(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the number of corner nodes for each face element for an element of the given
   * discretization type
   *
   * @param[in] distype discretization type
   * @return number of corner nodes for each face element
   */
  std::vector<int> get_number_of_face_element_corner_nodes(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the number of internal nodes for each face element for an element of the given
   * discretization type
   *
   * @param[in] distype discretization type
   * @return number of internal nodes for each face element
   */
  std::vector<int> get_number_of_face_element_internal_nodes(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the number of lines for the given discretization type
   *
   * @param[in] distype discretization type
   * @return number of lines
   */
  int get_number_of_element_lines(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the number of surface for the given discretization type
   *
   * @param[in] distype discretization type
   * @return number of surfaces
   */
  int get_number_of_element_surfaces(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the number of volumes for the given discretization type
   *
   * @param[in] distype discretization type
   * @return number of volumes
   */
  int get_number_of_element_volumes(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the number of faces for the given discretization type (as opposed to
   * lines/surfaces, this is always dim-1)
   *
   * @param[in] distype discretization type
   * @return number of faces
   */
  int get_number_of_element_faces(const Core::FE::CellType& distype);

  /*!
   * @brief Returns the shape type of a face of a given element

   * @param[in] distype discretization type
   * @param[in] face face index
   * @return discretization type of the element face
   */
  Core::FE::CellType get_ele_face_shape_type(
      const Core::FE::CellType& distype, const unsigned int face = 0);

  /*!
   * @brief Fills a vector<std::vector<int>> with all nodes for every face (surface in 3D, line in
   * 2D)
   *
   * @param[in] distype discretization type
   * @return map with all nodes for each face
   */
  std::vector<std::vector<int>> get_ele_node_numbering_faces(const Core::FE::CellType& distype);

  /*!
   * @brief Fills a vector<std::vector<int>> with all nodes for every surface
   *
   * @param[in] distype discretization type
   * @return map with all nodes for each surface
   */
  std::vector<std::vector<int>> get_ele_node_numbering_surfaces(const Core::FE::CellType& distype);

  /*!
   * @brief Fills a vector<std::vector<int>> with all nodes for every line
   *
   * @param[in] distype discretization type
   * @return map with all nodes for each line
   */
  std::vector<std::vector<int>> get_ele_node_numbering_lines(const Core::FE::CellType& distype);

  /*!
   * @brief Fills a vector<std::vector<int>> with all surfaces for every line
   *
   * @param[in] distype discretization type
   * @return map with surfaces adjacent to each line
   */
  std::vector<std::vector<int>> get_ele_node_numbering_lines_surfaces(
      const Core::FE::CellType& distype);

  /*!
   * @brief Fills a Core::LinAlg::SerialDenseMatrix with parameter space coordinates for each node
   * of the given discretization type
   *
   * @param[in] celltype discretization type
   * @return map of parameter space coordinates for all nodes
   */
  Core::LinAlg::SerialDenseMatrix get_ele_node_numbering_nodes_paramspace(
      const Core::FE::CellType celltype);

  template <Core::FE::CellType celltype>
  constexpr std::array<std::array<double, Core::FE::dim<celltype>>, Core::FE::num_nodes<celltype>>
  get_element_nodes_in_parameter_space()
  {
    std::array<std::array<double, Core::FE::dim<celltype>>, Core::FE::num_nodes<celltype>>
        nodal_parameter_coordinate{};
    switch (celltype)
    {
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad8:
      case Core::FE::CellType::quad9:
      case Core::FE::CellType::nurbs9:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
          {
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_quad9_nodes_reference[inode][isd];
          }
        }
        break;
      }
      case Core::FE::CellType::nurbs4:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_nurbs4_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::quad6:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_quad6_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::tri3:
      case Core::FE::CellType::tri6:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_tri6_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::hex20:
      case Core::FE::CellType::hex27:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_hex27_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::nurbs8:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_nurbs8_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::nurbs27:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_nurbs27_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::hex16:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_hex16_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::hex18:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_hex18_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::wedge6:
      case Core::FE::CellType::wedge15:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
          {
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_wedge18_nodes_reference[inode][isd];
          }
        }
        break;
      }
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::tet10:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_tet10_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::pyramid5:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_pyramid5_nodes_reference[inode][isd];
        }
        break;
      }
      case Core::FE::CellType::line3:
      case Core::FE::CellType::line2:
      {
        for (int inode = 0; inode < Core::FE::num_nodes<celltype>; inode++)
        {
          for (int isd = 0; isd < Core::FE::dim<celltype>; isd++)
            nodal_parameter_coordinate[inode][isd] =
                eleNodeNumbering_line3_nodes_reference[inode][isd];
        }
        break;
      }
      default:
      {
        FOUR_C_THROW("discretization type not yet implemented");
      }
    }

    return nodal_parameter_coordinate;
  }

  /*!
   * @brief Returns the coordinates in parameter space for a given node id and discretization type
   *
   * @tparam probdim     dimension of the problem
   * @param[in] nodeId   node Id
   * @param[in] distype  discretization type
   * @return coordinates in parameter space for a given node id and discretization type
   */
  template <int probdim = 3>
  Core::LinAlg::Matrix<probdim, 1> get_node_coordinates(
      const int nodeId, const Core::FE::CellType distype);

  /*!
   * @brief Computes the indices of the element corner nodes lying adjacent to a specified higher
   * order (edge) node index for each discretization type
   *
   * @param[out] index1  index of corner node
   * @param[out] index2  index of corner node
   * @param[in] hoindex  index of edge node
   * @param[in] distype  discretization type
   */
  void get_corner_node_indices(
      int& index1, int& index2, const int& hoindex, const Core::FE::CellType distype);

  /*!
   * @brief Returns the dimension of an element based on its discretization type
   *
   * @param[in] distype  discretization type
   */
  int get_dimension(const Core::FE::CellType distype);

  /*!
   * @brief Returns the degree of an element based on its discretization type
   *
   * @tparam DISTYPE discretization type
   */
  template <Core::FE::CellType distype>
  struct DisTypeToDegree
  {
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::point1>
  {
    static constexpr int degree = 0;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::line2>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::line3>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::line4>
  {
    static constexpr int degree = 3;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::line5>
  {
    static constexpr int degree = 4;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::line6>
  {
    static constexpr int degree = 5;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::nurbs2>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::nurbs3>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::quad4>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::quad8>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::quad9>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::nurbs4>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::nurbs9>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::tri3>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::tri6>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::hex8>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::hex20>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::hex27>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::tet4>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::tet10>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::wedge6>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::wedge15>
  {
    static constexpr int degree = 2;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::pyramid5>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::nurbs8>
  {
    static constexpr int degree = 1;
  };
  template <>
  struct DisTypeToDegree<Core::FE::CellType::nurbs27>
  {
    static constexpr int degree = 2;
  };

  /*!
   * @brief Returns the number of independent components for 2nd derivatives per discretization type
   * @tparam DISTYPE discretization type
   *
   * Number of components necessary to store second derivatives:
   *  in general: (n*(n+1))/2 components for nsd=n;
   *  in practice:
   *   1 component  for nsd=1:  (N,xx)
   *   3 components for nsd=2:  (N,xx ; N,yy ; N,xy)
   *   6 components for nsd=3:  (N,xx ; N,yy ; N,zz ; N,xy ; N,xz ; N,yz)
   */
  template <Core::FE::CellType distype>
  struct DisTypeToNumDeriv2
  {
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::point1>
  {
    static constexpr int numderiv2 = 0;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::line2>
  {
    static constexpr int numderiv2 = 1;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::line3>
  {
    static constexpr int numderiv2 = 1;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::nurbs2>
  {
    static constexpr int numderiv2 = 1;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::nurbs3>
  {
    static constexpr int numderiv2 = 1;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::quad4>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::quad6>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::quad8>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::quad9>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::nurbs4>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::nurbs9>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::tri3>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::tri6>
  {
    static constexpr int numderiv2 = 3;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::hex8>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::hex16>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::hex18>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::hex20>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::hex27>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::tet4>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::tet10>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::wedge6>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::wedge15>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::pyramid5>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::nurbs8>
  {
    static constexpr int numderiv2 = 6;
  };
  template <>
  struct DisTypeToNumDeriv2<Core::FE::CellType::nurbs27>
  {
    static constexpr int numderiv2 = 6;
  };


  /*!
   * @brief Returns the shape of face elements of a given element at compile time (invalid for wedge
   * and pyramid as it depends on the face number)
   *
   * @tparam DISTYPE discretization type
   */
  template <Core::FE::CellType distype>
  struct DisTypeToFaceShapeType
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::max_distype;
  };
  // template<> struct DisTypeToFaceShapeType<Core::FE::CellType::point1>   {}; //
  // invalid
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::line2>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::point1;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::line3>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::point1;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::nurbs2>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::point1;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::nurbs3>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::point1;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::quad4>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::line2;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::quad6>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::line3;
  };  // invalid
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::quad8>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::line3;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::quad9>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::line3;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::nurbs4>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::nurbs2;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::nurbs9>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::nurbs3;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::tri3>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::line2;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::tri6>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::line3;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::hex8>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::quad4;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::hex16>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::quad8;
  };  // invalid
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::hex18>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::quad9;
  };  // invalid
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::hex20>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::quad8;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::hex27>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::quad9;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::nurbs8>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::nurbs4;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::nurbs27>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::nurbs9;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::tet4>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::tri3;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::tet10>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::tri6;
  };
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::wedge6>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::tri3;
  };  // invalid
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::wedge15>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::tri6;
  };  // invalid
  template <>
  struct DisTypeToFaceShapeType<Core::FE::CellType::pyramid5>
  {
    static constexpr Core::FE::CellType shape = Core::FE::CellType::tri3;
  };  // invalid


  /*!
   * @brief Returns the number of nodes on face elements (invalid for wedge and pyramid as it
   * depends on the face number)
   *
   * @tparam DISTYPE discretization type
   */
  template <Core::FE::CellType distype>
  struct DisTypeToNumNodePerFace
  {
    static constexpr int numNodePerFace =
        Core::FE::num_nodes<DisTypeToFaceShapeType<distype>::shape>;
  };


  /*!
   * @brief Returns the order of an element edges based on its discretization type
   *
   * @tparam DISTYPE discretization type
   */
  template <Core::FE::CellType distype>
  struct DisTypeToEdgeOrder
  {
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::point1>
  {
    static constexpr int order = 0;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::line2>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::line3>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::line4>
  {
    static constexpr int order = 3;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::line5>
  {
    static constexpr int order = 4;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::line6>
  {
    static constexpr int order = 5;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::nurbs2>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::nurbs3>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::quad4>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::quad8>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::quad9>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::nurbs4>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::nurbs9>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::tri3>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::tri6>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::hex8>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::hex20>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::hex27>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::nurbs8>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::nurbs27>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::tet4>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::tet10>
  {
    static constexpr int order = 2;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::pyramid5>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::wedge6>
  {
    static constexpr int order = 1;
  };
  template <>
  struct DisTypeToEdgeOrder<Core::FE::CellType::wedge15>
  {
    static constexpr int order = 2;
  };


  /*!
   * @brief returns the order of the element-edges
   *
   * @param[in]  distype  shape of the element
   * @param[in, opt] default_order if the distype is not known, throw an error
   *                               or return the value for default_order
   */
  int get_order(const Core::FE::CellType distype, std::optional<int> default_order = std::nullopt);

  /*!
   * @brief returns the order of the element
   *
   * @param[in]  distype  shape of the element
   * @param[in, opt] default_degree if the distype is not known, throw an error
   *                                or return the value for default_degree
   */
  int get_degree(
      const Core::FE::CellType distype, std::optional<int> default_degree = std::nullopt);

  /*!
   * @brief Returns local node number in the parent element from a given face number and local node
   * number in the face element
   */
  int get_parent_node_number_from_face_node_number(
      const Core::FE::CellType parent_distype, const int faceId, const int faceNodeId);

  /*!
   * @brief Get the geometric center of the element in local coordinates
   *
   * @param[in]  dim  dimension of the element space (e.g. dim=2 for quad4)
   * @param[in]  distype  shape of the element
   * @param[out] pos  local center position
   */
  template <unsigned dim>
  Core::LinAlg::Matrix<dim, 1> get_local_center_position(
      const Core::FE::CellType distype, Core::LinAlg::Matrix<dim, 1>& pos)
  {
    switch (distype)
    {
      case Core::FE::CellType::line2:
      case Core::FE::CellType::line3:
      case Core::FE::CellType::nurbs2:
      case Core::FE::CellType::nurbs3:
      {
        pos = 0.0;
        break;
      }
      case Core::FE::CellType::quad4:
      case Core::FE::CellType::quad6:
      case Core::FE::CellType::quad8:
      case Core::FE::CellType::quad9:
      case Core::FE::CellType::nurbs4:
      case Core::FE::CellType::nurbs9:
      {
        pos = 0.0;
        break;
      }
      case Core::FE::CellType::tri3:
      case Core::FE::CellType::tri6:
      {
        pos = 1.0 / 3.0;
        break;
      }
      case Core::FE::CellType::hex8:
      case Core::FE::CellType::hex16:
      case Core::FE::CellType::hex18:
      case Core::FE::CellType::hex20:
      case Core::FE::CellType::hex27:
      {
        pos = 0.0;
        break;
      }
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::tet10:
      {
        pos = 1.0 / 4.0;
        break;
      }
      case Core::FE::CellType::wedge6:
      case Core::FE::CellType::wedge15:
      {
        pos = 1.0 / 3.0;
        pos(2, 0) = 0;
        break;
      }
      case Core::FE::CellType::pyramid5:
      {
        pos = 0.0;
        pos(2, 0) = 0.25;
        break;
      }
      default:
      {
        FOUR_C_THROW("discretization type {} not yet implemented",
            Core::FE::cell_type_to_string(distype).c_str());
        exit(EXIT_FAILURE);
      }
    }
    return pos;
  }

  /*!
   * @brief Returns the geometric center of the element in local coordinates
   *
   * @tparam dim  dimension of the element space (e.g. dim=2 for quad4)
   * @param[in] distype  discretization type
   * @return local center position.
   */
  template <unsigned dim>
  inline Core::LinAlg::Matrix<dim, 1> get_local_center_position(const Core::FE::CellType distype)
  {
    Core::LinAlg::Matrix<dim, 1> pos(Core::LinAlg::Initialization::uninitialized);
    get_local_center_position(distype, pos);
    return pos;
  }

  /*!
   * @brief Returns the shape of a given boundary element (is used in various element classes for
   * implementing the Shape() method of boundary elements)
   *
   * @param[in] nen number of nodes of the boundary element
   * @param[in] parentshape shape of parent element
   */
  Core::FE::CellType get_shape_of_boundary_element(
      const int nen, const Core::FE::CellType parentshape);

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
