// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_ele_neumann_evaluator.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_integration.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_solid_ele_calc_lib_integration.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <Core::FE::CellType celltype>
  void evaluate_normal_pressure(Core::Elements::Element& element,
      const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
      const std::vector<int>& dof_index_array, Core::LinAlg::SerialDenseVector& force,
      Core::LinAlg::SerialDenseMatrix* load_linearization, const double total_time,
      const double reference_thickness, const std::string& displacement_state)
  {
    constexpr int num_nodes = Core::FE::num_nodes(celltype);
    const int num_dof_per_node = element.num_dof_per_node(*element.nodes()[0]);
    const auto& onoff = condition.parameters().get<std::vector<int>>("ONOFF");
    const auto& values = condition.parameters().get<std::vector<double>>("VAL");
    const auto& function_ids = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");
    FOUR_C_ASSERT_ALWAYS(onoff.size() >= 2 && values.size() >= 2 && function_ids.size() >= 2,
        "A two-dimensional pressure condition requires at least two entries in ONOFF, VAL, and "
        "FUNCT.");
    FOUR_C_ASSERT_ALWAYS(
        onoff[0] == 1 && onoff[1] == 0, "Normal pressure must be activated on the first dof only.");

    const auto displacement = discretization.get_state(displacement_state);
    FOUR_C_ASSERT_ALWAYS(
        displacement != nullptr, "Cannot get state vector '{}'.", displacement_state);
    const std::vector<double> local_displacement =
        Core::FE::extract_values(*displacement, dof_index_array);

    Core::LinAlg::Matrix<num_nodes, 2> current_coordinates;
    for (int node = 0; node < num_nodes; ++node)
      for (int component = 0; component < 2; ++component)
        current_coordinates(node, component) =
            element.nodes()[node]->x()[component] +
            local_displacement[node * num_dof_per_node + component];

    const auto integration = Core::FE::create_gauss_integration<celltype>(
        Discret::Elements::get_gauss_rule_stiffness_matrix<celltype>());
    for (int gp = 0; gp < integration.num_points(); ++gp)
    {
      const auto xi = Core::Elements::evaluate_parameter_coordinate<celltype>(integration, gp);
      Core::Elements::ElementNodes<celltype, 2> nodes{.coordinates = current_coordinates};
      const auto shape = Core::Elements::evaluate_shape_functions_and_derivs<celltype>(xi, nodes);

      Core::LinAlg::Matrix<2, 1> current_coordinate;
      current_coordinate.multiply_tn(current_coordinates, shape.values);
      const double function_factor =
          function_ids[0].has_value() && function_ids[0].value() > 0
              ? Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(function_ids[0].value())
                    .evaluate(current_coordinate.as_span(), total_time, 0)
              : 1.0;

      Discret::Elements::add_normal_pressure_load<celltype>(shape, current_coordinates,
          values[0] * function_factor * integration.weight(gp), reference_thickness,
          num_dof_per_node, force, load_linearization);
    }
  }
}  // namespace

void Discret::Elements::evaluate_normal_pressure_by_element(Core::Elements::Element& element,
    const Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    const std::vector<int>& dof_index_array, Core::LinAlg::SerialDenseVector& force,
    Core::LinAlg::SerialDenseMatrix* load_linearization, const double total_time,
    const double reference_thickness)
{
  const auto& type = condition.parameters().get<std::string>("TYPE");
  std::string displacement_state;
  if (type == "pseudo_orthopressure")
  {
    displacement_state = "displacement";
    load_linearization = nullptr;
  }
  else if (type == "orthopressure")
  {
    FOUR_C_ASSERT_ALWAYS(
        Global::Problem::instance()->structural_dynamic_params().get<bool>("LOADLIN"),
        "If you use NEUMANN CONDITIONS with TYPE: \"orthopressure\" you need to set "
        "'LOADLIN: true' in 'STRUCTURAL DYNAMIC'.");
    displacement_state = "displacement new";
  }
  else
  {
    FOUR_C_THROW("Expected a normal-pressure condition, got TYPE '{}'.", type);
  }

  using supported_celltypes =
      Core::FE::CelltypeSequence<Core::FE::CellType::line2, Core::FE::CellType::line3>;
  Core::FE::cell_type_switch<supported_celltypes>(element.shape(),
      [&](auto celltype_t)
      {
        evaluate_normal_pressure<celltype_t()>(element, discretization, condition, dof_index_array,
            force, load_linearization, total_time, reference_thickness, displacement_state);
      });
}

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
                          .evaluate(gauss_point_reference_coordinates.as_span(), total_time, i)
                    : 1.0;

            const double value_times_integration_factor =
                value[i] * function_scale_factor * integration_factor;

            for (auto nodeid = 0; nodeid < numnod; ++nodeid)
            {
              int num_dof_per_node = element.num_dof_per_node(*element.nodes()[nodeid]);
              // Evaluates the Neumann boundary condition: f_{x,y,z}^i=\sum_j N^i(xi^j) * value(t) *
              // integration_factor_j
              // assembles the element force vector [f_x^1, f_y^1, f_z^1, [potential_extra_dof_1^1,
              // potential_extra_dofs_2^1, ...],
              // ..., f_x^n, f_y^n, f_z^n, [potential_extra_dof_1^n,
              // potential_extra_dofs_2^n, ...]]
              // Note, we only assemble on the first dim entries per node. There might be some extra
              // dofs per node (e.g., for solid-poro-p1 elements).
              element_force_vector[nodeid * num_dof_per_node + i] +=
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
