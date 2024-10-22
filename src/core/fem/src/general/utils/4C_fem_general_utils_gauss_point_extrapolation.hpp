// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_GAUSS_POINT_EXTRAPOLATION_HPP
#define FOUR_C_FEM_GENERAL_UTILS_GAUSS_POINT_EXTRAPOLATION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /*!
   * @brief Evaluates the linear operator that extrapolates quantities from the Gauss points to
   * the nodes.
   *
   * The evaluation happens only once per discretization type and integration rule. It will be
   * calculated only during the first execution and will then be stored internally for a faster
   * access. The procedure is as follows:
   *
   * (1) Get the base discretization type based on the given discretization type and the
   * integration rule. It is the discretization type of the same shape with the same or less
   * number of nodes compared to the number of Gauss points. Example: The base discretization type
   * of a HEX8-element that is integrated with 8 Gauss points is HEX8. The base discretization
   * type of a TET10-element with 4 Gauss points is TET4.
   *
   * (2) Evaluates the shape functions at the gauss points which is a matrix M=[N^i(xi_j)]
   *
   * (3) Either invert the matrix M (if it is a square matrix) or solve a least square algorithm
   *
   * (4) Extend the matrix such that the quantities on the additional nodes of the discretization
   * type compared to the base discretization type are interpolated using the shape functions of
   * the base discretization type.
   *
   * @note This method is not thread-safe and must not be called in a multi-threaded environment.
   *
   * @tparam distype discretization type of the element
   * @param integration (in) : Gauss integration points
   * @return Core::LinAlg::SerialDenseMatrix
   */
  template <Core::FE::CellType distype, class GaussIntegration>
  Core::LinAlg::SerialDenseMatrix evaluate_gauss_points_to_nodes_extrapolation_matrix(
      const GaussIntegration& intpoints);

  /*!
   * @brief Evaluates the linear operator that extrapolates quantities from the Gauss points to
   * the knots for NURBS-based elements.
   *
   * This method follows the same steps as evaluate_gauss_points_to_nodes_extrapolation_matrix(...)
   * but is adapted for NURBS elements. The main difference is that for the evaluation of the
   * shape functions (step 2) of evaluate_gauss_points_to_nodes_extrapolation_matrix(...)), the
   * element knot span and the weights of its control points must be obtained, which are stored in
   * the discretization related to the element.
   *
   * @note This method is not thread-safe and must not be called in a multi-threaded environment.
   *
   * @tparam distype discretization type of the element
   * @param dis (in) : discretization
   * @param ele (in) : element
   * @param integration (in) : Gauss integration points
   * @return Core::LinAlg::SerialDenseMatrix
   */
  template <Core::FE::CellType distype, class GaussIntegration>
  Core::LinAlg::SerialDenseMatrix evaluate_gauss_points_to_nurbs_knots_extrapolation_matrix(
      const Core::FE::Discretization& dis, const Core::Elements::Element& ele,
      const GaussIntegration& intpoints);

  /*!
   * @brief Extrapolates Gauss point data to the nodes and assembles it into a global data vector
   *
   * @tparam distype discretization type
   * @tparam GaussIntegration type of the container that holds the gauss integration rule
   * @param ele (in) :  element
   * @param gp_data (in) : Data at the Gauss points
   * @param global_data (out) : global data
   * @param nodal_average (in) : Whether to compute nodal averages at the nodes which are shared
   * between elements. Only set this to true if every element contributes equally to the nodal
   * quantity.
   * @param integration (in) : Container that holds the integration points
   */
  template <Core::FE::CellType distype, class GaussIntegration>
  void extrapolate_gp_quantity_to_nodes_and_assemble(const Core::Elements::Element& ele,
      const Core::LinAlg::SerialDenseMatrix& gp_data,
      Core::LinAlg::MultiVector<double>& global_data, bool nodal_average,
      const GaussIntegration& integration);

  /*!
   * @brief Extrapolates Gauss point data to the knots of a NURBS element and assembles
   * it into a global data vector
   *
   * The discretization is necessary for the evaluation of the shape functions, as it
   * stores the knot span and weights of a NURBS-based element
   * @tparam distype discretization type
   * @tparam GaussIntegration type of the container that holds the gauss integration rule
   * @param dis (in) : discretization
   * @param ele (in) :  element
   * @param gp_data (in) : Data at the Gauss points
   * @param global_data (out) : global data
   * @param nodal_average (in) : Whether to compute nodal averages at the nodes which are shared
   * between elements. Only set this to true if every element contributes equally to the nodal
   * quantity.
   * @param integration (in) : Container that holds the integration points
   */
  template <Core::FE::CellType distype, class GaussIntegration>
  void extrapolate_gp_quantity_to_nurbs_knots_and_assemble(const Core::FE::Discretization& dis,
      const Core::Elements::Element& ele, const LinAlg::SerialDenseMatrix& gp_data,
      Core::LinAlg::MultiVector<double>& global_data, bool nodal_average,
      const GaussIntegration& integration);
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
