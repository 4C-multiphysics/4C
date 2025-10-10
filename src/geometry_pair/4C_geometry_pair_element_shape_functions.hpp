// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_ELEMENT_SHAPE_FUNCTIONS_HPP
#define FOUR_C_GEOMETRY_PAIR_ELEMENT_SHAPE_FUNCTIONS_HPP


#include "4C_config.hpp"

#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_geometry_pair_element.hpp"

FOUR_C_NAMESPACE_OPEN

namespace GeometryPair
{
  /**
   * \brief Struct to evaluate the shape functions for an element
   */
  template <typename ElementType, typename Enable = void>
  struct EvaluateShapeFunction
  {
    // Empty per default
  };

  /**
   * \brief Specialization for Lagrange elements and elements with averaged nodal normals (face
   * elements which require averaged nodal normals are currently always based on Lagrange elements
   * in 4C)
   */
  template <typename ElementType>
  struct EvaluateShapeFunction<ElementType,
      typename std::enable_if<IsLagrangeElement<ElementType>::value_ ||
                              IsSurfaceAveragedNormalsElement<ElementType>::value_>::type>
  {
    /**
     * \brief Evaluate the shape functions of the element at xi.
     *
     * The elements considered here, i.e., Lagrangian elements don't have specific shape function
     * data, so we allow this as an optional argument
     *
     * @param N (out) shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T, typename... NotNeededArgumentType>
    static void evaluate(V& N, const T& xi, const NotNeededArgumentType&... not_needed_argument)
    {
      if constexpr (ElementType::element_dim_ == 1)
      {
        Core::FE::shape_function_1d(N, xi, ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 2)
      {
        Core::FE::shape_function_2d(N, xi(0), xi(1), ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 3)
      {
        Core::FE::shape_function_3d(N, xi(0), xi(1), xi(2), ElementType::discretization_);
      }
      else
      {
        FOUR_C_THROW("Got unexpected element dimension {}", ElementType::element_dim_);
      }
    }

    /**
     * \brief Evaluate the derivatives of the shape functions of the element at xi.
     *
     * The elements considered here, i.e., Lagrangian elements don't have specific shape function
     * data, so we allow this as an optional argument
     *
     * @param dN (out) derivatives of shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T, typename... NotNeededArgumentType>
    static void evaluate_deriv1(
        V& dN, const T& xi, const NotNeededArgumentType&... not_needed_argument)
    {
      if constexpr (ElementType::element_dim_ == 1)
      {
        Core::FE::shape_function_1d_deriv1(dN, xi, ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 2)
      {
        Core::FE::shape_function_2d_deriv1(dN, xi(0), xi(1), ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 3)
      {
        Core::FE::shape_function_3d_deriv1(dN, xi(0), xi(1), xi(2), ElementType::discretization_);
      }
      else
      {
        FOUR_C_THROW("Got unexpected element dimension {}", ElementType::element_dim_);
      }
    }

    /**
     * \brief Evaluate the 2nd derivatives of the shape functions of the element at xi.
     *
     * The elements considered here, i.e., Lagrangian elements don't have specific shape function
     * data, so we allow this as an optional argument
     *
     * @param ddN (out) 2nd derivatives of shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T, typename... NotNeededArgumentType>
    static void evaluate_deriv2(
        V& ddN, const T& xi, const NotNeededArgumentType&... not_needed_argument)
    {
      if constexpr (ElementType::element_dim_ == 1)
      {
        Core::FE::shape_function_1d_deriv2(ddN, xi, ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 2)
      {
        Core::FE::shape_function_2d_deriv2(ddN, xi(0), xi(1), ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 3)
      {
        Core::FE::shape_function_3d_deriv2(ddN, xi(0), xi(1), xi(2), ElementType::discretization_);
      }
      else
      {
        FOUR_C_THROW("Got unexpected element dimension {}", ElementType::element_dim_);
      }
    }
  };

  /**
   * \brief Specialization for Hermite elements
   */
  template <>
  struct EvaluateShapeFunction<t_hermite>
  {
    /**
     * \brief Evaluate the shape functions of the element at xi.
     *
     * @param N (out) shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T>
    static void evaluate(V& N, const T& xi, const ShapeFunctionData<t_hermite>& shape_function_data)
    {
      Core::FE::shape_function_hermite_1d(
          N, xi, shape_function_data.ref_length_, t_hermite::discretization_);
    }

    /**
     * \brief Evaluate the derivatives of the shape functions of the element at xi.
     *
     * @param dN (out) derivatives of shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T>
    static void evaluate_deriv1(
        V& dN, const T& xi, const ShapeFunctionData<t_hermite>& shape_function_data)
    {
      Core::FE::shape_function_hermite_1d_deriv1(
          dN, xi, shape_function_data.ref_length_, t_hermite::discretization_);
    }

    /**
     * \brief Evaluate the 2nd derivatives of the shape functions of the element at xi.
     *
     * @param ddN (out) 2nd derivatives of shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T>
    static void evaluate_deriv2(
        V& ddN, const T& xi, const ShapeFunctionData<t_hermite>& shape_function_data)
    {
      Core::FE::shape_function_hermite_1d_deriv2(
          ddN, xi, shape_function_data.ref_length_, t_hermite::discretization_);
    }
  };

  /**
   * \brief Specialization for NURBS elements
   */
  template <typename ElementType>
  struct EvaluateShapeFunction<ElementType,
      typename std::enable_if<IsNurbsElement<ElementType>::value_>::type>
  {
    /**
     * \brief Evaluate the shape functions of the element at xi.
     *
     * @param N (out) shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T>
    static void evaluate(
        V& N, const T& xi, const ShapeFunctionData<ElementType>& shape_function_data)
    {
      if (shape_function_data.myknots_.size() == 0)
        FOUR_C_THROW(
            "Got shape function data with knot size 0 - did you forget to initialize the element "
            "data with get_element_data::Get?");

      if constexpr (ElementType::element_dim_ == 2)
      {
        Core::FE::Nurbs::nurbs_get_2d_funct<typename V::scalar_type>(N, xi,
            shape_function_data.myknots_, shape_function_data.weights_,
            ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 3)
      {
        Core::FE::Nurbs::nurbs_get_3d_funct(N, xi, shape_function_data.myknots_,
            shape_function_data.weights_, ElementType::discretization_);
      }
      else
      {
        FOUR_C_THROW("Got unexpected element dimension {}", ElementType::element_dim_);
      }
    }

    /**
     * \brief Evaluate the derivatives of the shape functions of the element at xi.
     *
     * @param dN (out) derivatives of shape functions.
     * @param xi (in) Parameter coordinate on the element.
     * @param shape_function_data (in) Shape function data container.
     */
    template <typename V, typename T>
    static void evaluate_deriv1(
        V& dN, const T& xi, const ShapeFunctionData<ElementType>& shape_function_data)
    {
      if (shape_function_data.myknots_.size() == 0)
        FOUR_C_THROW(
            "Got shape function data with knot size 0 - did you forget to initialize the element "
            "data with get_element_data::Get?");

      using type_dummy = Core::LinAlg::Matrix<ElementType::n_nodes_, 1, typename V::scalar_type>;
      type_dummy N_dummy;

      if constexpr (ElementType::element_dim_ == 2)
      {
        Core::FE::Nurbs::nurbs_get_2d_funct_deriv<typename V::scalar_type>(N_dummy, dN, xi,
            shape_function_data.myknots_, shape_function_data.weights_,
            ElementType::discretization_);
      }
      else if constexpr (ElementType::element_dim_ == 3)
      {
        Core::FE::Nurbs::nurbs_get_3d_funct_deriv(N_dummy, dN, xi, shape_function_data.myknots_,
            shape_function_data.weights_, t_nurbs27::discretization_);
      }
      else
      {
        FOUR_C_THROW("Got unexpected element dimension {}", ElementType::element_dim_);
      }
    }
  };

  /**
   * \brief Get the shape function matrix of the element at xi.
   *
   * Multiplying this shape function matrix with the element DOF vector results in the field
   * variable evaluated at xi.
   *
   * @param xi (in) Parameter coordinate on the element.
   * @param N (out) shape function matrix.
   * @param shape_function_data (in) Shape function data container.
   */
  template <typename ElementType, typename T, typename ScalarTypeResult,
      typename... ShapeFunctionDataType>
  inline void evaluate_shape_function_matrix(
      Core::LinAlg::Matrix<ElementType::spatial_dim_, ElementType::n_dof_, ScalarTypeResult>& N,
      const T& xi, const ShapeFunctionDataType&... shape_function_data)
  {
    // Matrix for shape function values.
    Core::LinAlg::Matrix<1, ElementType::n_nodes_ * ElementType::n_val_, ScalarTypeResult> N_flat(
        Core::LinAlg::Initialization::zero);

    // Evaluate the shape function values.
    EvaluateShapeFunction<ElementType>::evaluate(N_flat, xi, shape_function_data...);

    // Fill up the full shape function matrix.
    N.clear();
    for (unsigned int node = 0; node < ElementType::n_nodes_; node++)
      for (unsigned int dim = 0; dim < ElementType::spatial_dim_; dim++)
        for (unsigned int val = 0; val < ElementType::n_val_; val++)
          N(dim, ElementType::spatial_dim_ * ElementType::n_val_ * node +
                     ElementType::spatial_dim_ * val + dim) =
              N_flat(ElementType::n_val_ * node + val);
  }

}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif
