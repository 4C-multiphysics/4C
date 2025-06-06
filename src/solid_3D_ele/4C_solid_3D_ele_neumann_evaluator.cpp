// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_3D_ele_neumann_evaluator.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

template <int dim>
void Discret::Elements::evaluate_neumann_by_element(Core::Elements::Element& element,
    const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    Core::LinAlg::SerialDenseVector& element_force_vector, double total_time)
{
  using supported_celltypes = Core::FE::CelltypeSequence<Core::FE::CellType::hex8,
      Core::FE::CellType::hex18, Core::FE::CellType::hex20, Core::FE::CellType::hex27,
      Core::FE::CellType::nurbs27, Core::FE::CellType::pyramid5, Core::FE::CellType::wedge6,
      Core::FE::CellType::tet4, Core::FE::CellType::tet10, Core::FE::CellType::quad4,
      Core::FE::CellType::quad8, Core::FE::CellType::quad9, Core::FE::CellType::tri3,
      Core::FE::CellType::tri6, Core::FE::CellType::line2, Core::FE::CellType::line3>;
  return Core::FE::cell_type_switch<supported_celltypes>(element.shape(),
      [&](auto celltype_t)
      {
        return evaluate_neumann<celltype_t(), dim>(
            element, discretization, condition, element_force_vector, total_time);
      });
}

template <Core::FE::CellType celltype, int dim>
void Discret::Elements::evaluate_neumann(Core::Elements::Element& element,
    const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    Core::LinAlg::SerialDenseVector& element_force_vector, double total_time)
{
  constexpr auto numnod = Core::FE::num_nodes(celltype);
  Core::FE::GaussIntegration gauss_integration = Core::FE::create_gauss_integration<celltype>(
      Discret::Elements::get_gauss_rule_stiffness_matrix<celltype>());

  // get values and switches from the condition
  const auto& onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto& value = condition.parameters().get<std::vector<double>>("VAL");

  // ensure that at least as many curves/functs as dofs are available
  if (onoff.size() < dim)
    FOUR_C_THROW("Fewer functions or curves defined than the element's dimension.");

  for (std::size_t checkdof = dim; checkdof < onoff.size(); ++checkdof)
  {
    if (onoff[checkdof] != 0)
    {
      FOUR_C_THROW(
          "You have activated more than {} dofs in your Neumann boundary condition. This is higher "
          "than the dimension of the element.",
          dim);
    }
  }

  // get ids of functions of space and time
  const auto& function_ids = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");

  const Core::Elements::ElementNodes<celltype, dim> element_nodes =
      Core::Elements::evaluate_element_nodes<celltype, dim>(discretization, element);

  Core::Elements::for_each_gauss_point<celltype, dim>(element_nodes, gauss_integration,
      [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
          const Core::Elements::ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const Core::Elements::JacobianMapping<celltype, dim>& jacobian_mapping,
          double integration_factor, int gp)
      {
        // material/reference co-ordinates of Gauss point
        Core::LinAlg::Matrix<dim, 1> gauss_point_reference_coordinates;
        gauss_point_reference_coordinates.multiply_tn(
            element_nodes.coordinates, shape_functions.values);

        for (auto i = 0; i < dim; ++i)
        {
          if (onoff[i])
          {
            // function evaluation
            const double function_scale_factor =
                (function_ids[i].has_value() && function_ids[i].value() > 0)
                    ? Global::Problem::instance()
                          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                              function_ids[i].value())
                          .evaluate(gauss_point_reference_coordinates.data(), total_time, i)
                    : 1.0;

            const double value_times_integration_factor =
                value[i] * function_scale_factor * integration_factor;

            for (auto nodeid = 0; nodeid < numnod; ++nodeid)
            {
              // Evaluates the Neumann boundary condition: f_{x,y,z}^i=\sum_j N^i(xi^j) * value(t) *
              // integration_factor_j
              // assembles the element force vector [f_x^1, f_y^1, f_z^1, ..., f_x^n, f_y^n, f_z^n]
              element_force_vector[nodeid * dim + i] +=
                  shape_functions.values(nodeid) * value_times_integration_factor;
            }
          }
        }
      });
}


template void Discret::Elements::evaluate_neumann_by_element<3>(Core::Elements::Element&,
    const Core::FE::Discretization&, const Core::Conditions::Condition&,
    Core::LinAlg::SerialDenseVector&, double);
template void Discret::Elements::evaluate_neumann_by_element<2>(Core::Elements::Element&,
    const Core::FE::Discretization&, const Core::Conditions::Condition&,
    Core::LinAlg::SerialDenseVector&, double);


FOUR_C_NAMESPACE_CLOSE
