// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_ELEMENT_EVALUATION_FUNCTIONS_HPP
#define FOUR_C_GEOMETRY_PAIR_ELEMENT_EVALUATION_FUNCTIONS_HPP


#include "4C_config.hpp"

#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_shape_functions.hpp"
#include "4C_geometry_pair_utility_functions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace GeometryPair
{
  /**
   * \brief Evaluate the field function in the element
   */
  template <typename ElementType, typename T, typename V, typename ScalarType>
  inline void evaluate_position(const T& xi, const ElementData<ElementType, V>& element_data,
      Core::LinAlg::Matrix<ElementType::spatial_dim_, 1, ScalarType>& r)
  {
    // Matrix for shape function values
    Core::LinAlg::Matrix<1, ElementType::n_nodes_ * ElementType::n_val_, ScalarType> N(
        Core::LinAlg::Initialization::zero);

    // Evaluate the shape function values
    EvaluateShapeFunction<ElementType>::evaluate(N, xi, element_data.shape_function_data_);

    // Calculate the field function
    r.clear();
    for (unsigned int node = 0; node < ElementType::n_nodes_; node++)
      for (unsigned int dim = 0; dim < ElementType::spatial_dim_; dim++)
        for (unsigned int val = 0; val < ElementType::n_val_; val++)
          r(dim) += element_data.element_position_(
                        ElementType::spatial_dim_ * ElementType::n_val_ * node +
                        ElementType::spatial_dim_ * val + dim) *
                    N(ElementType::n_val_ * node + val);
  }

  /**
   * \brief Evaluate the derivative of the field function w.r.t xi in the element
   */
  template <typename ElementType, typename T, typename V, typename ScalarType>
  inline void evaluate_position_derivative1(const T& xi,
      const ElementData<ElementType, V>& element_data,
      Core::LinAlg::Matrix<ElementType::spatial_dim_, ElementType::element_dim_, ScalarType>& dr)
  {
    // Matrix for shape function values
    Core::LinAlg::Matrix<ElementType::element_dim_, ElementType::n_nodes_ * ElementType::n_val_,
        ScalarType>
        dN(Core::LinAlg::Initialization::zero);

    // Evaluate the shape function values
    EvaluateShapeFunction<ElementType>::evaluate_deriv1(dN, xi, element_data.shape_function_data_);

    // Calculate the derivative of the field function
    dr.clear();
    for (unsigned int dim = 0; dim < ElementType::spatial_dim_; dim++)
      for (unsigned int direction = 0; direction < ElementType::element_dim_; direction++)
        for (unsigned int node = 0; node < ElementType::n_nodes_; node++)
          for (unsigned int val = 0; val < ElementType::n_val_; val++)
            dr(dim, direction) += element_data.element_position_(
                                      ElementType::spatial_dim_ * ElementType::n_val_ * node +
                                      ElementType::spatial_dim_ * val + dim) *
                                  dN(direction, ElementType::n_val_ * node + val);
  }

  /**
   * \brief Evaluate the 2nd derivative of the field function w.r.t xi in the element
   */
  template <typename ElementType, typename T, typename V, typename ScalarType>
  inline void evaluate_position_derivative2(const T& xi,
      const ElementData<ElementType, V>& element_data,
      Core::LinAlg::Matrix<ElementType::spatial_dim_, ElementType::element_dim_, ScalarType>& ddr)
  {
    static_assert(ElementType::element_dim_ == 1,
        "evaluate_position_derivative2 is currently only implemented for 1D elements!");

    // Matrix for shape function values
    Core::LinAlg::Matrix<ElementType::element_dim_, ElementType::n_nodes_ * ElementType::n_val_,
        ScalarType>
        dN(Core::LinAlg::Initialization::zero);

    // Evaluate the shape function values
    EvaluateShapeFunction<ElementType>::evaluate_deriv2(dN, xi, element_data.shape_function_data_);

    // Calculate the derivative of the field function
    ddr.clear();
    for (unsigned int dim = 0; dim < ElementType::spatial_dim_; dim++)
      for (unsigned int direction = 0; direction < ElementType::element_dim_; direction++)
        for (unsigned int node = 0; node < ElementType::n_nodes_; node++)
          for (unsigned int val = 0; val < ElementType::n_val_; val++)
            ddr(dim, direction) += element_data.element_position_(
                                       ElementType::spatial_dim_ * ElementType::n_val_ * node +
                                       ElementType::spatial_dim_ * val + dim) *
                                   dN(direction, ElementType::n_val_ * node + val);
  }

  /**
   * \brief Evaluate the geometrical normal (the one resulting from the FE approximation) on the
   * face element
   */
  template <typename Surface, typename T, typename ScalarTypeDof, typename ScalarTypeResult>
  void evaluate_face_normal(const T& xi,
      const ElementData<Surface, ScalarTypeDof>& element_data_surface,
      Core::LinAlg::Matrix<3, 1, ScalarTypeResult>& normal)
  {
    // Check at compile time if a surface (2D) element is given
    static_assert(
        Surface::element_dim_ == 2, "evaluate_face_normal can only be called for 2D elements!");

    Core::LinAlg::Matrix<3, 2, ScalarTypeResult> dr;
    Core::LinAlg::Matrix<3, 1, ScalarTypeResult> dr_0;
    Core::LinAlg::Matrix<3, 1, ScalarTypeResult> dr_1;
    GeometryPair::evaluate_position_derivative1<Surface>(xi, element_data_surface, dr);
    for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
    {
      dr_0(i_dir) = dr(i_dir, 0);
      dr_1(i_dir) = dr(i_dir, 1);
    }
    normal.cross_product(dr_0, dr_1);
    normal.scale(1.0 / Core::FADUtils::vector_norm(normal));
    if constexpr (std::is_same<Surface, t_nurbs9>::value)
    {
      // In the NURBS case some normals have to be flipped to point outward of the volume element
      normal.scale(element_data_surface.shape_function_data_.surface_normal_factor_);
    }
  }

  /**
   * \brief Evaluate the normal at a position on the surface, depending on the surface type, either
   * the averaged nodal normal field or the kinematic normal field
   */
  template <typename Surface, typename T, typename ScalarTypeDof, typename ScalarTypeResult>
  void evaluate_surface_normal(const T& xi,
      const ElementData<Surface, ScalarTypeDof>& element_data_surface,
      Core::LinAlg::Matrix<3, 1, ScalarTypeResult>& normal)
  {
    // Check at compile time if a surface (2D) element is given
    static_assert(
        Surface::element_dim_ == 2, "evaluate_surface_normal can only be called for 2D elements!");

    if constexpr (IsSurfaceAveragedNormalsElement<Surface>::value_)
    {
      static_assert(Surface::n_val_ == 1,
          "EvaluateSurfaceNormal with averaged nodal normals only makes sense for elements with "
          "one value per node and dimension");

      // Calculate the normal as a interpolation of nodal normals
      Core::LinAlg::Matrix<1, Surface::n_nodes_, typename T::scalar_type> N(
          Core::LinAlg::Initialization::zero);
      EvaluateShapeFunction<Surface>::evaluate(N, xi, element_data_surface.shape_function_data_);
      normal.clear();
      for (unsigned int node = 0; node < Surface::n_nodes_; node++)
        for (unsigned int dim = 0; dim < Surface::spatial_dim_; dim++)
          normal(dim) +=
              element_data_surface.nodal_normals_(Surface::spatial_dim_ * node + dim) * N(node);
      normal.scale(1.0 / Core::FADUtils::vector_norm(normal));
    }
    else
    {
      // No averaged normals, so evaluate the kinematic normals on the face element
      evaluate_face_normal(xi, element_data_surface, normal);
    }
  }

  /**
   * \brief Evaluate a position on the surface, where the third parameter coordinate is taken along
   * the surface normal
   */
  template <typename Surface, typename ScalarTypeXi, typename ScalarTypeDof,
      typename ScalarTypeResult>
  void evaluate_surface_position(const Core::LinAlg::Matrix<3, 1, ScalarTypeXi>& xi,
      const ElementData<Surface, ScalarTypeDof>& element_data_surface,
      Core::LinAlg::Matrix<3, 1, ScalarTypeResult>& r)
  {
    // Check at compile time if a surface (2D) element is given
    static_assert(Surface::element_dim_ == 2,
        "evaluate_surface_position can only be called for 2D elements!");

    // Evaluate the normal
    Core::LinAlg::Matrix<3, 1, ScalarTypeResult> normal;
    evaluate_surface_normal<Surface>(xi, element_data_surface, normal);

    // Evaluate the position on the surface
    GeometryPair::evaluate_position<Surface>(xi, element_data_surface, r);

    // Add the normal part to the position
    normal.scale(xi(2));
    r += normal;
  }

  /**
   * \brief Evaluate a triad on a plane curve
   *
   * The curve has to lie in the x-y plane. The first basis vector of the triad is the tangent along
   * the curve, the second basis vector is the tangent rotated clockwise around the z axis by pi/2.
   * The third basis vector is the Cartesian e_z basis vector.
   */
  template <typename Line, typename ScalarTypeXi, typename ScalarTypeDof, typename ScalarTypeResult>
  void evaluate_triad_at_plane_curve(const ScalarTypeXi xi,
      const ElementData<Line, ScalarTypeDof>& element_data_line,
      Core::LinAlg::Matrix<3, 3, ScalarTypeResult>& triad)
  {
    Core::LinAlg::Matrix<3, 1, ScalarTypeResult> tangent, cross_section_director_2,
        cross_section_director_3;
    evaluate_position_derivative1<Line>(xi, element_data_line, tangent);

    if (std::abs(tangent(2)) > Constants::pos_tol)
      FOUR_C_THROW(
          "EvaluateTriadAtPlaneCurve: The tangent vector can not have a component in z direction! "
          "The component is {}!",
          Core::FADUtils::cast_to_double(tangent(2)));

    // Create the director vectors in the cross-section
    // Director 2 is the one in the y-axis (reference configuration)
    // Director 3 is the one in the z-axis (reference configuration)
    tangent.scale(1. / Core::FADUtils::vector_norm(tangent));
    cross_section_director_2.clear();
    cross_section_director_2(0) = -tangent(1);
    cross_section_director_2(1) = tangent(0);
    cross_section_director_3.clear();
    cross_section_director_3(2) = 1.;

    // Set the triad
    for (unsigned int dir = 0; dir < 3; dir++)
    {
      triad(dir, 0) = tangent(dir);
      triad(dir, 1) = cross_section_director_2(dir);
      triad(dir, 2) = cross_section_director_3(dir);
    }
  }

  /**
   * \brief Evaluate the Jacobi matrix for a volume element
   */
  template <typename Volume, typename ScalarType>
  void evaluate_jacobian(const Core::LinAlg::Matrix<3, 1, ScalarType>& xi,
      const GeometryPair::ElementData<Volume, double>& X_volume,
      Core::LinAlg::Matrix<3, 3, ScalarType>& J)
  {
    // Check at compile time if a volume (3D) element is given
    static_assert(
        Volume::element_dim_ == 3, "EvaluateJacobian can only be called for 3D elements!");

    // Get the derivatives of the reference position w.r.t the parameter coordinates. This is the
    // transposed Jacobi matrix.
    Core::LinAlg::Matrix<3, 3, ScalarType> dXdxi(Core::LinAlg::Initialization::zero);
    evaluate_position_derivative1<Volume>(xi, X_volume, dXdxi);
    J.clear();
    J.update_t(dXdxi);
  }

  /**
   * \brief Evaluate the 3D deformation gradient F in a volume element
   */
  template <typename Volume, typename ScalarTypeXi, typename ScalarTypeDof,
      typename ScalarTypeResult>
  void evaluate_deformation_gradient(const Core::LinAlg::Matrix<3, 1, ScalarTypeXi>& xi,
      const GeometryPair::ElementData<Volume, double>& X_volume,
      const GeometryPair::ElementData<Volume, ScalarTypeDof>& q_volume,
      Core::LinAlg::Matrix<3, 3, ScalarTypeResult>& F)
  {
    // Check at compile time if a volume (3D) element is given
    static_assert(Volume::element_dim_ == 3,
        "EvaluateDeformationGradient can only be called for 3D elements!");

    // Get the inverse of the Jacobian
    Core::LinAlg::Matrix<3, 3, ScalarTypeXi> inv_J(Core::LinAlg::Initialization::zero);
    evaluate_jacobian<Volume>(xi, X_volume, inv_J);
    Core::LinAlg::inverse(inv_J);

    // Get the derivatives of the shape functions w.r.t to the parameter coordinates
    Core::LinAlg::Matrix<Volume::element_dim_, Volume::n_nodes_ * Volume::n_val_, ScalarTypeXi>
        dNdxi(Core::LinAlg::Initialization::zero);
    GeometryPair::EvaluateShapeFunction<Volume>::evaluate_deriv1(
        dNdxi, xi, q_volume.shape_function_data_);

    // Transform to derivatives w.r.t physical coordinates
    Core::LinAlg::Matrix<Volume::element_dim_, Volume::n_nodes_ * Volume::n_val_, ScalarTypeXi>
        dNdX(Core::LinAlg::Initialization::zero);
    for (unsigned int i_row = 0; i_row < 3; i_row++)
      for (unsigned int i_col = 0; i_col < Volume::n_nodes_ * Volume::n_val_; i_col++)
        for (unsigned int i_sum = 0; i_sum < 3; i_sum++)
          dNdX(i_row, i_col) += inv_J(i_row, i_sum) * dNdxi(i_sum, i_col);

    // Calculate F
    F.clear();
    for (unsigned int i_row = 0; i_row < 3; i_row++)
      for (unsigned int i_col = 0; i_col < 3; i_col++)
        for (unsigned int i_sum = 0; i_sum < Volume::n_nodes_ * Volume::n_val_; i_sum++)
          F(i_col, i_row) += dNdX(i_row, i_sum) * q_volume.element_position_(3 * i_sum + i_col);
  }

  /**
   * \brief Evaluate a position on the surface and its derivative w.r.t. xi.
   * @param element_data_surface (in) Degrees of freedom for the surface.
   * @param xi (in) Parameter coordinates on the surface (the first two are in the surface
   * parameter coordinates, the third one is in the normal direction).
   * @param r (out) Position on the surface.
   * @param dr (out) Derivative of the position on the surface, w.r.t xi.
   */
  template <typename ScalarType, typename Surface>
  void evaluate_surface_position_and_derivative(
      const ElementData<Surface, ScalarType>& element_data_surface,
      const Core::LinAlg::Matrix<3, 1, ScalarType>& xi, Core::LinAlg::Matrix<3, 1, ScalarType>& r,
      Core::LinAlg::Matrix<3, 3, ScalarType>& dr)
  {
    // Create a nested FAD type.
    using FAD_outer = Sacado::ELRFad::SLFad<ScalarType, 3>;

    // Set up the AD variables.
    Core::LinAlg::Matrix<3, 1, FAD_outer> xi_AD;
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++) xi_AD(i_dim) = FAD_outer(3, i_dim, xi(i_dim));
    Core::LinAlg::Matrix<3, 1, FAD_outer> r_AD;

    // Evaluate the position.
    evaluate_surface_position<Surface>(xi_AD, element_data_surface, r_AD);

    // Extract the return values from the AD types.
    for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
    {
      r(i_dir) = r_AD(i_dir).val();
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++) dr(i_dir, i_dim) = r_AD(i_dir).dx(i_dim);
    }
  }

  /**
   * \brief Check if the parameter coordinate xi is in the valid range for a 1D element
   */
  template <typename T>
  bool valid_parameter_1d(const T& xi)
  {
    const double xi_limit = 1.0 + Constants::projection_xi_eta_tol;
    if (fabs(xi) < xi_limit)
      return true;
    else
      return false;
  }

  /**
   * \brief Check if the parameter coordinate xi is in the valid range for a 2D element
   */
  template <typename ElementType, typename T>
  bool valid_parameter_2d(const T& xi)
  {
    const double xi_limit = 1.0 + Constants::projection_xi_eta_tol;
    if (ElementType::geometry_type_ == DiscretizationTypeGeometry::quad)
    {
      if (fabs(xi(0)) < xi_limit && fabs(xi(1)) < xi_limit) return true;
    }
    else if (ElementType::geometry_type_ == DiscretizationTypeGeometry::triangle)
    {
      if (xi(0) > -Constants::projection_xi_eta_tol && xi(1) > -Constants::projection_xi_eta_tol &&
          xi(0) + xi(1) < 1.0 + Constants::projection_xi_eta_tol)
        return true;
    }
    else
    {
      FOUR_C_THROW("Wrong DiscretizationTypeGeometry given!");
    }

    // Default value
    return false;
  }

  /**
   * \brief Check if the parameter coordinate xi is in the valid range for a 3D element
   */
  template <typename ElementType, typename T>
  bool valid_parameter_3d(const T& xi)
  {
    const double xi_limit = 1.0 + Constants::projection_xi_eta_tol;
    if (ElementType::geometry_type_ == DiscretizationTypeGeometry::hexahedron)
    {
      if (fabs(xi(0)) < xi_limit && fabs(xi(1)) < xi_limit && fabs(xi(2)) < xi_limit) return true;
    }
    else if (ElementType::geometry_type_ == DiscretizationTypeGeometry::tetraeder)
    {
      if (xi(0) > -Constants::projection_xi_eta_tol && xi(1) > -Constants::projection_xi_eta_tol &&
          xi(2) > -Constants::projection_xi_eta_tol &&
          xi(0) + xi(1) + xi(2) < 1.0 + Constants::projection_xi_eta_tol)
        return true;
    }
    else if (ElementType::geometry_type_ == DiscretizationTypeGeometry::wedge)
    {
      if (xi(0) > -Constants::projection_xi_eta_tol && xi(0) < xi_limit &&
          xi(1) > -Constants::projection_xi_eta_tol && xi(1) < xi_limit - xi(0) &&
          fabs(xi(2)) < xi_limit)
        return true;
    }
    else
    {
      FOUR_C_THROW("Wrong DiscretizationTypeGeometry given!");
    }

    // Default value
    return false;
  }

  /**
   * \brief This struct sets parameter coordinates start values for local Newton iterations on an
   * element
   */
  template <DiscretizationTypeGeometry geometry_type>
  struct StartValues
  {
   public:
    /**
     * \brief Set the start values for a local Newton iterations on the element.
     *
     * This method will be specialized for each geometry type. By doing so the values can be set at
     * compile time.
     */
    template <typename T>
    static void set(T& xi)
    {
      std::string error_string =
          "This is the default implementation of StartValues::Set and should never be called, "
          "since we only want to call the templated versions. You are calling it with the "
          "DiscretizationTypeGeometry ";
      error_string += discretization_type_geometry_to_string(geometry_type);
      FOUR_C_THROW("{}", error_string);
    }
  };

  /**
   * \brief Template specialization for line elements
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::line>::set(T& xi)
  {
    xi = 0.0;
  }

  /**
   * \brief Template specialization for triangle surface elements
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::triangle>::set(T& xi)
  {
    // We do not use xi.put_scalar(0.25) here, since this might be a surface element which has a
    // normal direction and we do not want to set an initial value in the normal direction
    xi.put_scalar(0.0);
    for (int i_dim = 0; i_dim < 2; i_dim++) xi(i_dim) = 0.25;
  }

  /**
   * \brief Template specialization for quad surface elements
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::quad>::set(T& xi)
  {
    xi.put_scalar(0.0);
  }

  /**
   * \brief Template specialization for tetrahedra volume elements
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::tetraeder>::set(T& xi)
  {
    xi.put_scalar(0.25);
  }

  /**
   * \brief Template specialization for hexahedron volume elements
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::hexahedron>::set(T& xi)
  {
    xi.put_scalar(0.0);
  }

}  // namespace GeometryPair


FOUR_C_NAMESPACE_CLOSE

#endif
