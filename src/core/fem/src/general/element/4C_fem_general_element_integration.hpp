// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_ELEMENT_INTEGRATION_HPP
#define FOUR_C_FEM_GENERAL_ELEMENT_INTEGRATION_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  template <Core::FE::CellType celltype, unsigned dim>
  struct ElementNodes
  {
    /*!
     * @brief Position of nodes in the reference configuration
     */
    Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), dim> coordinates;
  };

  template <Core::FE::CellType celltype, unsigned dim>
    requires(Core::FE::is_nurbs<celltype>)
  struct ElementNodes<celltype, dim>
  {
    /*!
     * @brief Position of nodes in the reference configuration
     */
    Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), dim> coordinates;

    /*!
     * @brief Knot span of a NURBS element
     */
    std::vector<Core::LinAlg::SerialDenseVector> knots;

    /*!
     * @brief Weights of control points
     */
    Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), 1, double> weights;
  };

  /*!
   * @brief Evaluate element node information given the element and a displacement vector
   *
   * @tparam celltype
   * @param ele (in): Element
   * @param disp (in) : Vector of nodal displacements of the element
   * @return ElementNodes<celltype>
   */
  template <Core::FE::CellType celltype, unsigned dim>
  ElementNodes<celltype, dim> evaluate_element_nodes(
      const Core::FE::Discretization& discretization, const Core::Elements::Element& ele)
  {
    ElementNodes<celltype, dim> element_nodes;
    for (int i = 0; i < Core::FE::num_nodes(celltype); ++i)
    {
      const std::vector<double>& x = ele.nodes()[i]->x();
      for (unsigned d = 0; d < dim; ++d)
      {
        element_nodes.coordinates(i, d) = x[d];
      }
    }

    if constexpr (Core::FE::is_nurbs<celltype>)
    {
      // Obtain the information required for a NURBS element
      bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
          discretization, &ele, element_nodes.knots, element_nodes.weights);

      FOUR_C_ASSERT_ALWAYS(!zero_size,
          "get_my_nurbs_knots_and_weights has to return a non zero size NURBS element.");
    }

    return element_nodes;
  }

  /*!
   * @brief Evaluates the parameter coordinate of the Gauss point according the the Gauss rule
   *
   * @tparam celltype : Cell type
   * @param intpoints (in) : Gauss integration points
   * @param gp (in) : id of the Gauss point
   * @return Core::LinAlg::Matrix<num_dim<celltype>, 1> : Coordinates of the Gauss Point in the
   * parameter space
   */
  template <Core::FE::CellType celltype>
  Core::LinAlg::Tensor<double, Core::FE::dim<celltype>> evaluate_parameter_coordinate(
      const Core::FE::GaussIntegration& intpoints, const int gp)
  {
    Core::LinAlg::Tensor<double, Core::FE::dim<celltype>> xi;
    for (int d = 0; d < Core::FE::dim<celltype>; ++d) xi(d) = intpoints.point(gp)[d];

    return xi;
  }

  /*!
   * @brief A struct holding the shape functions and its derivatives evaluated at a point within the
   * element
   */
  template <Core::FE::CellType celltype>
  struct ShapeFunctionsAndDerivatives
  {
    Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), 1> values{};
    Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::num_nodes(celltype)> derivatives{};
  };

  /*!
   * @brief Evaluates the shape functions and their derivatives at the specified point in the
   * parameter space
   *
   * @tparam celltype : Discretizationt type
   * @param xi (in) : Coordinate in the parameter space
   * @return ShapeFunctionsAndDerivatives<celltype> : An object holding the shape functions and the
   * first derivatives evaluated at the respective point in the parameter space
   */
  template <Core::FE::CellType celltype, unsigned dim>
  ShapeFunctionsAndDerivatives<celltype> evaluate_shape_functions_and_derivs(
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
      const ElementNodes<celltype, dim>& nodal_coordinates)
    requires(Core::FE::use_lagrange_shapefnct<celltype>)
  {
    ShapeFunctionsAndDerivatives<celltype> shapefcns;
    Core::FE::shape_function<celltype>(xi, shapefcns.values);
    Core::FE::shape_function_deriv1<celltype>(xi, shapefcns.derivatives);

    return shapefcns;
  }

  template <Core::FE::CellType celltype, unsigned dim>
  ShapeFunctionsAndDerivatives<celltype> evaluate_shape_functions_and_derivs(
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
      const ElementNodes<celltype, dim>& nodal_coordinates)
    requires(Core::FE::is_nurbs<celltype>)
  {
    ShapeFunctionsAndDerivatives<celltype> shapefcns;
    Core::FE::Nurbs::nurbs_get_funct_deriv(shapefcns.values, shapefcns.derivatives, xi,
        nodal_coordinates.knots, nodal_coordinates.weights, celltype);

    return shapefcns;
  }

  /*!
   * @brief A matrix holding the jacobian mapping from the parameter space to the physical space
   *
   * @tparam celltype
   * @tparam dim
   */
  template <Core::FE::CellType celltype, unsigned dim>
  using JacobianMapping = Core::LinAlg::Matrix<Core::FE::dim<celltype>, dim>;

  /*!
   * @brief Evaluates the jacobian mapping of the element
   *
   * @tparam celltype : Cell type
   * @param shapefcns (in) : Shape functions and derivatives evaluated at the respective point in
   * the parameter space
   * @param element_nodes (in) : Reference and current coordinates of the nodes of the element
   * @param gp (in) : Id of the Gauss point
   * @return JacobianMapping<celltype> : An object holding quantities of the jacobian mapping
   * (inverse Jacobian, determinant, derivatives of the shape functions w.r.t. XYZ, integration
   * factor)
   */
  template <unsigned dim, Core::FE::CellType celltype>
  JacobianMapping<celltype, dim> evaluate_jacobian_mapping(
      const ShapeFunctionsAndDerivatives<celltype>& shapefcns,
      const ElementNodes<celltype, dim>& element_nodes)
  {
    JacobianMapping<celltype, dim> jacobian;
    jacobian.multiply(shapefcns.derivatives, element_nodes.coordinates);

    return jacobian;
  }

  /*!
   * @brief The concept for a callable object called at each Gauss point during integration
   *
   * @tparam dim : Dimension of the space where the element is embedded (default is the dimension of
   * the element)
   * @tparam celltype : Cell type of the element
   * @tparam T
   */
  template <unsigned dim, Core::FE::CellType celltype, typename T>
  concept GaussPointEvaluatable = requires(T gp_evaluator,
      Core::LinAlg::Tensor<double, Core::FE::dim<celltype>> xi,
      const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
      const JacobianMapping<celltype, dim>& jacobian_mapping, double integration_factor, int gp) {
    { gp_evaluator(xi, shape_functions, jacobian_mapping, integration_factor, gp) };
  };
  template <Core::FE::CellType celltype, unsigned dim>
  double evaluate_integral_transformation_factor(
      const JacobianMapping<celltype, dim> jacobian_mapping)
  {
    if constexpr (dim == Core::FE::dim<celltype>)
    {
      return jacobian_mapping.determinant();
    }
    else
    {
      Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>> metric_tensor;
      metric_tensor.multiply_nt(jacobian_mapping, jacobian_mapping);
      return std::sqrt(metric_tensor.determinant());
    }
  }

  /*!
   * @brief Calls the @p gp_evaluator for each Gauss point with evaluated jacobian mapping using the
   * integration rule defined by @p integration.
   *
   * @tparam celltype : Cell type of the element
   * @tparam dim : Dimension of the space where the element is embedded (default is the dimension of
   * the element)
   * @tparam GaussPointEvaluator
   * @param nodal_coordinates (in) : The nodal coordinates of the element
   * @param integration (in) : The integration rule to be used.
   * @param gp_evaluator (in) : A callable object (e.g. lambda-function) with that is @p
   * GaussPointEvaluatable that will be called for each integration point.
   */
  template <Core::FE::CellType celltype, unsigned dim, typename GaussPointEvaluator>
  void for_each_gauss_point(const ElementNodes<celltype, dim>& element_nodes,
      const Core::FE::GaussIntegration& integration, GaussPointEvaluator gp_evaluator)
    requires GaussPointEvaluatable<dim, celltype, GaussPointEvaluator>
  {
    for (int gp = 0; gp < integration.num_points(); ++gp)
    {
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>> xi =
          evaluate_parameter_coordinate<celltype>(integration, gp);

      const ShapeFunctionsAndDerivatives<celltype> shape_functions =
          evaluate_shape_functions_and_derivs<celltype>(xi, element_nodes);

      const JacobianMapping<celltype, dim> jacobian_mapping =
          evaluate_jacobian_mapping<dim>(shape_functions, element_nodes);

      const double integration_factor =
          evaluate_integral_transformation_factor<celltype>(jacobian_mapping) *
          integration.weight(gp);

      gp_evaluator(xi, shape_functions, jacobian_mapping, integration_factor, gp);
    }
  }
}  // namespace Core::Elements

FOUR_C_NAMESPACE_CLOSE

#endif
