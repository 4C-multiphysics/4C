// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_ELE_NEUMANN_EVALUATOR_HPP
#define FOUR_C_SOLID_ELE_NEUMANN_EVALUATOR_HPP
#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

#include <array>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret::Elements
{
  template <Core::FE::CellType celltype>
    requires(Core::FE::dim<celltype> == 1)
  /*!
   * @brief Add pressure normal to a plane solid boundary.
   *
   * The pressure is multiplied by the parent solid's reference thickness. If requested, this
   * function also assembles the consistent negative load linearization used by the structural
   * residual.
   */
  void add_normal_pressure_load(const Core::Elements::ShapeFunctionsAndDerivatives<celltype>& shape,
      const Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), 2>& current_coordinates,
      const double pressure_times_weight, const double reference_thickness,
      const int num_dof_per_node, Core::LinAlg::SerialDenseVector& force,
      Core::LinAlg::SerialDenseMatrix* load_linearization)
  {
    double radial_derivative = 0.0;
    double axial_derivative = 0.0;
    for (int node = 0; node < Core::FE::num_nodes(celltype); ++node)
    {
      radial_derivative += shape.derivatives(0, node) * current_coordinates(node, 0);
      axial_derivative += shape.derivatives(0, node) * current_coordinates(node, 1);
    }

    const std::array<double, 2> normal_measure{
        reference_thickness * axial_derivative, -reference_thickness * radial_derivative};
    for (int node = 0; node < Core::FE::num_nodes(celltype); ++node)
      for (int component = 0; component < 2; ++component)
        force[node * num_dof_per_node + component] +=
            shape.values(node) * pressure_times_weight * normal_measure[component];

    if (load_linearization == nullptr) return;

    for (int force_node = 0; force_node < Core::FE::num_nodes(celltype); ++force_node)
      for (int coordinate_node = 0; coordinate_node < Core::FE::num_nodes(celltype);
          ++coordinate_node)
      {
        const double factor = shape.values(force_node) * pressure_times_weight *
                              reference_thickness * shape.derivatives(0, coordinate_node);
        (*load_linearization)(
            force_node* num_dof_per_node, coordinate_node* num_dof_per_node + 1) -= factor;
        (*load_linearization)(
            force_node* num_dof_per_node + 1, coordinate_node * num_dof_per_node) += factor;
      }
  }

  /*!
   * @brief Evaluate pseudo-orthopressure or follower orthopressure on a plane solid boundary line.
   */
  void evaluate_normal_pressure_by_element(Core::Elements::Element& element,
      const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
      const std::vector<int>& dof_index_array, Core::LinAlg::SerialDenseVector& force,
      Core::LinAlg::SerialDenseMatrix* load_linearization, double total_time,
      double reference_thickness);

  /*!
   * @brief Evaluates a Neumann condition @p condition for the element @p element.
   *
   * The element force vector is
   *
   * \f[
   * \boldsymbol{f}^{(e)} = \left[
   *    f_x^{1(e)}~f_y^{1(e)}~f_z^{1(e)}~\cdots~f_x^{n(e)}~f_y^{n(e)}~f_z^{n(e)}
   * \right]
   * @f]
   * with
   * @f[
   *   f_{x/y/z}^{i(e)} = \int_{\Omega^{(e)}} N^i \cdot \mathrm{value}_{x/y/z} \cdot
   *   \mathrm{funct}_{x/y/z} (t)  \mathrm{d} \Omega
   * \f],
   * where \f$n\f$ is the number of nodes of the element and \f$N^i\f$ is the \f$i\f$-th shape
   * function of the element.
   *
   * @note This function determines the shape of the element at runtime and calls the respective
   * templated version of @p evaluate_neumann. If you already know the Core::FE::CellType
   * of the element at compile-time, you could directly call @evaluate_neumann.
   *
   * @param element (in) : The element where we integrate
   * @param discretization (in) : discretization
   * @param condition (in) : The Neumann condition to be evaluated within the element.
   * @param dof_index_array (in) : The index array of the dofs of the element
   * @param element_force_vector (out) : The element force vector for the evaluated Neumann
   * condition
   * @param total_time (in) : The total time for time dependent Neumann conditions
   */
  template <int dim>
  void evaluate_neumann_by_element(Core::Elements::Element& element,
      const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
      Core::LinAlg::SerialDenseVector& element_force_vector, double total_time);

  /*!
   * @brief Evaluates a Neumann condition @p condition for the element @p element with the
   * cell type known at compile time.
   *
   * The element force vector is
   *
   * @f[
   * \boldsymbol{f}_{(e)} = \left[
   *    f_x^{1(e)}~f_y^{1(e)}~f_z^{1(e)}~\cdots~f_x^{n(e)}~f_y^{n(e)}~f_z^{n(e)}
   * \right]
   * @f]
   * with
   * @f[
   * f_{x/y/z}^{i(e)} = \int_{\Omega^{(e)}} N^i \cdot \mathrm{value}_{x/y/z} \cdot
   * \mathrm{funct}_{x/y/z} (t)  \mathrm{d} \Omega
   * @f],
   * where \f$n\f$ is the number of nodes of the element and \f$N^i\f$ is the \f$i\f$-th shape
   * function of the element.
   *
   * @tparam celltype Cell type known at compile time
   *
   * @param element (in) : The element where we integrate
   * @param discretization (in) : discretization
   * @param condition (in) : The Neumann condition to be evaluated within the element.
   * @param dof_index_array (in) : The index array of the dofs of the element
   * @param element_force_vector (out) : The element force vector for the evaluated Neumann
   * condition
   * @param total_time (in) : The total time for time dependent Neumann conditions
   */
  template <Core::FE::CellType celltype, int dim>
  void evaluate_neumann(Core::Elements::Element& element,
      const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
      Core::LinAlg::SerialDenseVector& element_force_vector, double total_time);

}  // namespace Discret::Elements


FOUR_C_NAMESPACE_CLOSE

#endif
