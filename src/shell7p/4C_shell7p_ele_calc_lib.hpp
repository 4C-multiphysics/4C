// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SHELL7P_ELE_CALC_LIB_HPP
#define FOUR_C_SHELL7P_ELE_CALC_LIB_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_tensor_symmetric_einstein.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_shell7p_ele.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <numeric>

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements::Shell
{
  /*!
   * @brief An object holding the nodal coordinates in reference and current configuration
   *
   * @tparam distype : The discretization type known at compile time
   */
  template <Core::FE::CellType distype>
  struct NodalCoordinates
  {
    Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim> x_refe_;
    Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim> x_curr_;
    Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim> a3_refe_;
    Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim> a3_curr_;
  };

  /*!
   * \brief Create matrix with spatial configuration
   *
   * @tparam distype :  The discretization type known at compile time
   * @param x       (out)  : Nodal coords in spatial frame
   * @param x_refe  (in)  : Nodal coords in reference frame
   * @param disp    (in)  : Displacements
   * @param index   (in)  : Integer to consider either displacements or director displacmenets
   */
  template <Core::FE::CellType distype>
  void spatial_configuration(
      Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim>& x,
      const Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim>& x_refe,
      const std::vector<double>& disp, const int index)
  {
    const int nodedof = Discret::Elements::Shell::Internal::node_dof;
    for (int i = 0; i < Internal::num_node<distype>; ++i)
    {
      x(i, 0) = x_refe(i, 0) + disp[i * nodedof + 0 + index];
      x(i, 1) = x_refe(i, 1) + disp[i * nodedof + 1 + index];
      x(i, 2) = x_refe(i, 2) + disp[i * nodedof + 2 + index];
    }
  }
  /*!
   * \brief Evaluates coordinates of element nodes at the shell surface and approximated nodal
   * points at the shell thickness. The thickness is represented by nodal directors a3_reference.
   *
   * @tparam distype :  The discretization type known at compile time
   * @param nodes (in)  : Nodal coords in reference frame
   * @param disp (in)  : Displacements
   * @param thickness (in)  : Nodal thickness values in reference frame
   * @param a3_reference (in)  : Nodal directors in reference frame
   * @param factor (in)  : Scaling factor due to SDC
   */
  template <Core::FE::CellType distype>
  Discret::Elements::Shell::NodalCoordinates<distype> evaluate_nodal_coordinates(
      Core::Nodes::Node** nodes, std::vector<double>& disp, const double& thickness,
      const Core::LinAlg::SerialDenseMatrix& a3_reference, const double factor)
  {
    Discret::Elements::Shell::NodalCoordinates<distype> coordinates;
    for (auto i = 0; i < Discret::Elements::Shell::Internal::num_node<distype>; ++i)
    {
      const double h2 = thickness * factor * 0.5;

      const auto& x = nodes[i]->x();
      coordinates.x_refe_(i, 0) = x[0];
      coordinates.x_refe_(i, 1) = x[1];
      coordinates.x_refe_(i, 2) = x[2];

      coordinates.a3_refe_(i, 0) = a3_reference(i, 0) * h2;
      coordinates.a3_refe_(i, 1) = a3_reference(i, 1) * h2;
      coordinates.a3_refe_(i, 2) = a3_reference(i, 2) * h2;

      const int nodedof = Internal::node_dof;
      coordinates.x_curr_(i, 0) = coordinates.x_refe_(i, 0) + disp[i * nodedof + 0];
      coordinates.x_curr_(i, 1) = coordinates.x_refe_(i, 1) + disp[i * nodedof + 1];
      coordinates.x_curr_(i, 2) = coordinates.x_refe_(i, 2) + disp[i * nodedof + 2];

      coordinates.a3_curr_(i, 0) = coordinates.a3_refe_(i, 0) + disp[i * nodedof + 3];
      coordinates.a3_curr_(i, 1) = coordinates.a3_refe_(i, 1) + disp[i * nodedof + 4];
      coordinates.a3_curr_(i, 2) = coordinates.a3_refe_(i, 2) + disp[i * nodedof + 5];
    }
    return coordinates;
  }

  /*!
   * @brief A struct holding the the strains measures (deformation gradient and Green-lagrange
   * strains)
   */
  struct Strains
  {
    Core::LinAlg::Tensor<double, Internal::num_dim, Internal::num_dim> defgrd_{};
    Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim> gl_strain_{};
  };

  /*!
   * @brief A struct holding the the stress measures (2-Piola-Kirchhoff stresses and elasticity
   * matrix (Linearization of the 2. Piola Kirchhoff stress tensor w.r.t. Green-Lagrange strain
   * tensor)
   */
  struct Stress
  {
    Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim> pk2_;
    Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim, Internal::num_dim,
        Internal::num_dim>
        cmat_;
  };

  /*!
   * @brief A struct holding the thickness-integrated stress resultants (membrane forces and
   * bending moments) and the thickness-integrated material tangent
   */
  struct StressResultants
  {
    Core::LinAlg::SerialDenseVector stress_;
    Core::LinAlg::SerialDenseMatrix dmat_;
  };


  /*!
   * @brief Evaluates the material stress in a global cartesian coordinate system (2. Piola
   * Kirchhoff stress tensor and the linearization w.r.t Green-Lagrange strain)
   *
   * @tparam dim : Dimension
   * @param material (in) : Reference to the material
   * @param strains (in) : Strain measures of the element
   * @param params (in) : List of additional parameter to pass quantities from the time integrator
   * to the material
   * @param context (in) : Material evaluation context
   * @param gp (in) : Gauss point
   * @param eleGID (in) : Global element id
   * @return Stress : Object holding the 2. Piola Kirchhoff stress tensor and
   * the linearization w.r.t. Green Lagrange strain tensor
   */
  template <int dim>
  Stress evaluate_material_stress_cartesian_system(Mat::So3Material& material,
      const Core::LinAlg::Tensor<double, 3, 3>& defgrd,
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl_strain, Teuchos::ParameterList& params,
      const Mat::EvaluationContext& context, int gp, int eleGID)
  {
    Stress stress{};

    material.evaluate(&defgrd, gl_strain, params, context, stress.pk2_, stress.cmat_, gp, eleGID);

    return stress;
  }

  /*!
   * @brief Returns the optimal gauss integration rule based on the element discretization type
   */
  template <Core::FE::CellType distype>
  constexpr auto get_gauss_rule()
  {
    return Discret::Elements::DisTypeToOptGaussRule<distype>::rule;
  }

  /*!
   * @brief Returns the gauss integration based on the optimal gauss integration rule
   *
   * @tparam distype : The discretization type known at compile time
   * @tparam GaussRuleType : Optimal gauss rule integration type
   * @param gaussrule (in) : Gauss rule
   * @return Core::FE::IntegrationPoints2D : Integration points
   */
  template <Core::FE::CellType distype, typename GaussRuleType>
  Core::FE::IntegrationPoints2D create_gauss_integration_points(GaussRuleType gaussrule)
  {
    return Core::FE::IntegrationPoints2D(gaussrule);
  }

  /*!
   * \brief Calculate the deformation gradient that is consistent to the EAS enhanced GL strains.
   *
   * @param defgrd_disp (in): displacement-based deformation gradient F^{u} (needed for the
   * rotational part)
   * @param enhanced_gl_strain (in): Green-Lagrange strains E^{enh} to compute the deformation
   * gradient from
   * @return Core::LinAlg::Tensor<double, 3, 3> : deformation gradient F^{enh} computed from
   * enhanced GL strains
   */
  static inline Core::LinAlg::Tensor<double, 3, 3> calc_consistent_defgrd(
      const Core::LinAlg::Tensor<double, 3, 3>& defgrd_disp,
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& enhanced_gl_strain)
  {
    // calculate modified right stretch tensor
    const Core::LinAlg::SymmetricTensor<double, 3, 3> cauchy_green_enh =
        2 * enhanced_gl_strain + Core::LinAlg::TensorGenerators::identity<double, 3, 3>;
    const auto compute_pure_stretch_tensor =
        [](const Core::LinAlg::SymmetricTensor<double, 3, 3>& C)
    {
      auto [eigenvalues, eigenvectors] = Core::LinAlg::eig(C);

      // compute the sqrt of the eigenvalues
      std::ranges::for_each(eigenvalues, [](double& val) { val = std::sqrt(val); });
      const auto eig = Core::LinAlg::TensorGenerators::diagonal(eigenvalues);
      return Core::LinAlg::assume_symmetry(
          eigenvectors * eig * Core::LinAlg::transpose(eigenvectors));
    };

    // compute rotation tensor from deformation gradient
    const Core::LinAlg::SymmetricTensor<double, 3, 3> cauchy_green_disp =
        Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(defgrd_disp) * defgrd_disp);
    const Core::LinAlg::Tensor<double, 3, 3> R =
        defgrd_disp * Core::LinAlg::inv(compute_pure_stretch_tensor(cauchy_green_disp));

    // Compute consistent deformation gradient
    return R * compute_pure_stretch_tensor(cauchy_green_enh);
  }

  /*!
   * @brief A struct holding the mass matrix integration factors
   */
  struct MassMatrixVariables
  {
    double factor_v_ = 0.;
    double factor_w_ = 0.;
    double factor_vw_ = 0.;
  };

  /*!
   * @brief An object holding the shape functions and the first derivatives evaluated at the
   * respective point in the parameter space
   *
   * @tparam distype :  The discretization type known at compile time
   */
  template <Core::FE::CellType distype>
  struct ShapefunctionsAndDerivatives
  {
    Core::LinAlg::Matrix<Internal::num_node<distype>, 1> shapefunctions_;
    Core::LinAlg::Matrix<Internal::num_dim, Internal::num_node<distype>> derivatives_;
  };

  /*!
   * @brief Evaluates the shape functions and the first derivatives w.r.t spatial coordinates at the
   * specified point in the parameter space
   *
   * @tparam distype :  The discretization type known at compile time
   * @param xi_gp (in) : Coordinate of the integration point in the parameter space
   * @return ShapeFunctionsAndDerivatives<distype> : An object holding the shape functions and the
   * first derivatives evaluated at the respective point in the parameter space
   */
  template <Core::FE::CellType distype>
  ShapefunctionsAndDerivatives<distype> evaluate_shapefunctions_and_derivs(
      const std::array<double, 2>& xi_gp)
  {
    ShapefunctionsAndDerivatives<distype> shapefunctions;
    Core::FE::shape_function_2d(shapefunctions.shapefunctions_, xi_gp[0], xi_gp[1], distype);
    Core::FE::shape_function_2d_deriv1(shapefunctions.derivatives_, xi_gp[0], xi_gp[1], distype);
    return shapefunctions;
  }

  /*!
   * @brief An object holding the basis vectors and metric tensors
   *
   * @tparam distype :  The discretization type known at compile time
   */
  template <Core::FE::CellType distype>
  struct BasisVectorsAndMetrics
  {
    Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim> covariant_;
    Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim> contravariant_;
    Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim> metric_covariant_;
    Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim> metric_contravariant_;
    Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim> partial_derivative_;
    double detJ_;
  };

  /*!
   * @brief Evaluates the strain gradient (B-Operator) of the specified element
   *
   * @tparam distype :  The discretization type known at compile time
   * @param a_cov (in) : covariant basis vectors
   * @param da3_cov (in) : Partial derivatives of the covariant basis vectors w.r.t spatial
   * coordinates
   * @param shapefunctions_derivatives (in) : An object holding the shape functions and the
   * first derivatives w.r.t spatial coordinates evaluated at the respective point in the parameter
   * space
   * @return Core::LinAlg::SerialDenseMatrix: B-operator Matrix
   */
  template <Core::FE::CellType distype>
  Core::LinAlg::SerialDenseMatrix calc_b_operator(
      const Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim>& a_cov,
      const Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim>& da3kov,
      const Discret::Elements::Shell::ShapefunctionsAndDerivatives<distype>&
          shapefunctions_derivatives)
  {
    const Core::LinAlg::Matrix<Internal::num_node<distype>, 1> shapefunctions =
        shapefunctions_derivatives.shapefunctions_;
    const Core::LinAlg::Matrix<Internal::num_dim, Internal::num_node<distype>>& derivs =
        shapefunctions_derivatives.derivatives_;
    Core::LinAlg::SerialDenseMatrix Bop(
        Internal::num_internal_variables, Internal::numdofperelement<distype>);
    const int nodedof = Internal::node_dof;
    for (int i = 0; i < Internal::num_node<distype>; ++i)
    {
      Bop(0, nodedof * i + 0) = derivs(0, i) * a_cov(0, 0);
      Bop(0, nodedof * i + 1) = derivs(0, i) * a_cov(0, 1);
      Bop(0, nodedof * i + 2) = derivs(0, i) * a_cov(0, 2);
      Bop(0, nodedof * i + 3) = 0.0;
      Bop(0, nodedof * i + 4) = 0.0;
      Bop(0, nodedof * i + 5) = 0.0;

      Bop(1, nodedof * i + 0) = derivs(1, i) * a_cov(1, 0);
      Bop(1, nodedof * i + 1) = derivs(1, i) * a_cov(1, 1);
      Bop(1, nodedof * i + 2) = derivs(1, i) * a_cov(1, 2);
      Bop(1, nodedof * i + 3) = 0.0;
      Bop(1, nodedof * i + 4) = 0.0;
      Bop(1, nodedof * i + 5) = 0.0;

      Bop(2, nodedof * i + 0) = 0.0;
      Bop(2, nodedof * i + 1) = 0.0;
      Bop(2, nodedof * i + 2) = 0.0;
      Bop(2, nodedof * i + 3) = shapefunctions(i) * a_cov(2, 0);
      Bop(2, nodedof * i + 4) = shapefunctions(i) * a_cov(2, 1);
      Bop(2, nodedof * i + 5) = shapefunctions(i) * a_cov(2, 2);

      Bop(3, nodedof * i + 0) = derivs(1, i) * a_cov(0, 0) + derivs(0, i) * a_cov(1, 0);
      Bop(3, nodedof * i + 1) = derivs(1, i) * a_cov(0, 1) + derivs(0, i) * a_cov(1, 1);
      Bop(3, nodedof * i + 2) = derivs(1, i) * a_cov(0, 2) + derivs(0, i) * a_cov(1, 2);
      Bop(3, nodedof * i + 3) = 0.0;
      Bop(3, nodedof * i + 4) = 0.0;
      Bop(3, nodedof * i + 5) = 0.0;

      Bop(4, nodedof * i + 0) = derivs(1, i) * a_cov(2, 0);
      Bop(4, nodedof * i + 1) = derivs(1, i) * a_cov(2, 1);
      Bop(4, nodedof * i + 2) = derivs(1, i) * a_cov(2, 2);
      Bop(4, nodedof * i + 3) = shapefunctions(i) * a_cov(1, 0);
      Bop(4, nodedof * i + 4) = shapefunctions(i) * a_cov(1, 1);
      Bop(4, nodedof * i + 5) = shapefunctions(i) * a_cov(1, 2);

      Bop(5, nodedof * i + 0) = derivs(0, i) * a_cov(2, 0);
      Bop(5, nodedof * i + 1) = derivs(0, i) * a_cov(2, 1);
      Bop(5, nodedof * i + 2) = derivs(0, i) * a_cov(2, 2);
      Bop(5, nodedof * i + 3) = shapefunctions(i) * a_cov(0, 0);
      Bop(5, nodedof * i + 4) = shapefunctions(i) * a_cov(0, 1);
      Bop(5, nodedof * i + 5) = shapefunctions(i) * a_cov(0, 2);

      Bop(6, nodedof * i + 0) = derivs(0, i) * da3kov(0, 0);
      Bop(6, nodedof * i + 1) = derivs(0, i) * da3kov(0, 1);
      Bop(6, nodedof * i + 2) = derivs(0, i) * da3kov(0, 2);
      Bop(6, nodedof * i + 3) = derivs(0, i) * a_cov(0, 0);
      Bop(6, nodedof * i + 4) = derivs(0, i) * a_cov(0, 1);
      Bop(6, nodedof * i + 5) = derivs(0, i) * a_cov(0, 2);

      Bop(7, nodedof * i + 0) = derivs(1, i) * da3kov(1, 0);
      Bop(7, nodedof * i + 1) = derivs(1, i) * da3kov(1, 1);
      Bop(7, nodedof * i + 2) = derivs(1, i) * da3kov(1, 2);
      Bop(7, nodedof * i + 3) = derivs(1, i) * a_cov(1, 0);
      Bop(7, nodedof * i + 4) = derivs(1, i) * a_cov(1, 1);
      Bop(7, nodedof * i + 5) = derivs(1, i) * a_cov(1, 2);

      Bop(8, nodedof * i + 0) = 0.0;
      Bop(8, nodedof * i + 1) = 0.0;
      Bop(8, nodedof * i + 2) = 0.0;
      Bop(8, nodedof * i + 3) = 0.0;
      Bop(8, nodedof * i + 4) = 0.0;
      Bop(8, nodedof * i + 5) = 0.0;

      Bop(9, nodedof * i + 0) = derivs(0, i) * da3kov(1, 0) + derivs(1, i) * da3kov(0, 0);
      Bop(9, nodedof * i + 1) = derivs(0, i) * da3kov(1, 1) + derivs(1, i) * da3kov(0, 1);
      Bop(9, nodedof * i + 2) = derivs(0, i) * da3kov(1, 2) + derivs(1, i) * da3kov(0, 2);
      Bop(9, nodedof * i + 3) = derivs(0, i) * a_cov(1, 0) + derivs(1, i) * a_cov(0, 0);
      Bop(9, nodedof * i + 4) = derivs(0, i) * a_cov(1, 1) + derivs(1, i) * a_cov(0, 1);
      Bop(9, nodedof * i + 5) = derivs(0, i) * a_cov(1, 2) + derivs(1, i) * a_cov(0, 2);

      Bop(10, nodedof * i + 0) = 0.0;
      Bop(10, nodedof * i + 1) = 0.0;
      Bop(10, nodedof * i + 2) = 0.0;
      Bop(10, nodedof * i + 3) = shapefunctions(i) * da3kov(1, 0) + derivs(1, i) * a_cov(2, 0);
      Bop(10, nodedof * i + 4) = shapefunctions(i) * da3kov(1, 1) + derivs(1, i) * a_cov(2, 1);
      Bop(10, nodedof * i + 5) = shapefunctions(i) * da3kov(1, 2) + derivs(1, i) * a_cov(2, 2);

      Bop(11, nodedof * i + 0) = 0.0;
      Bop(11, nodedof * i + 1) = 0.0;
      Bop(11, nodedof * i + 2) = 0.0;
      Bop(11, nodedof * i + 3) = shapefunctions(i) * da3kov(0, 0) + derivs(0, i) * a_cov(2, 0);
      Bop(11, nodedof * i + 4) = shapefunctions(i) * da3kov(0, 1) + derivs(0, i) * a_cov(2, 1);
      Bop(11, nodedof * i + 5) = shapefunctions(i) * da3kov(0, 2) + derivs(0, i) * a_cov(2, 2);
    }
    return Bop;
  }

  /*!
   * @brief Modify the strain gradient (B-Operator) of the specified element due to transverse
   * shear strain ANS  (B-Bar method)
   *
   * @tparam distype : The discretization type known at compile time
   * @param Bop (in) : B-Operator matrix
   * @param shapefunctions_ans (in) : Vector holding the ANS shapefunctions and its derivatves w.r.t
   * spatial coordinates
   * @param shapefunctions_q (in) : Vector holding the shapefunctions and the first derivatives
   * w.r.t spatial coordinates evaluated at collocation points
   * @param metric_currq (in) : Metric tensor in material configuration evaluated at collocation
   * points
   * @param numans (in) : Number of ANS collocation points
   */
  template <Core::FE::CellType distype>
  void modify_b_operator_ans(Core::LinAlg::SerialDenseMatrix& Bop,
      const std::vector<double>& shapefunctions_ans,
      const std::vector<ShapefunctionsAndDerivatives<distype>>& shapefunctions_q,
      const std::vector<BasisVectorsAndMetrics<distype>>& metric_currq, const int& numans)
  {
    const int nodedof = Internal::node_dof;
    for (int i = 0; i < Internal::num_node<distype>; ++i)
    {
      Bop(5, nodedof * i + 0) = 0.0;
      Bop(5, nodedof * i + 1) = 0.0;
      Bop(5, nodedof * i + 2) = 0.0;
      Bop(5, nodedof * i + 3) = 0.0;
      Bop(5, nodedof * i + 4) = 0.0;
      Bop(5, nodedof * i + 5) = 0.0;

      Bop(4, nodedof * i + 0) = 0.0;
      Bop(4, nodedof * i + 1) = 0.0;
      Bop(4, nodedof * i + 2) = 0.0;
      Bop(4, nodedof * i + 3) = 0.0;
      Bop(4, nodedof * i + 4) = 0.0;
      Bop(4, nodedof * i + 5) = 0.0;

      for (int j = 0; j < numans; ++j)
      {
        const double a1x1 = metric_currq[j].covariant_(0, 0);
        const double a1y1 = metric_currq[j].covariant_(0, 1);
        const double a1z1 = metric_currq[j].covariant_(0, 2);

        const double a3x1 = metric_currq[j].covariant_(2, 0);
        const double a3y1 = metric_currq[j].covariant_(2, 1);
        const double a3z1 = metric_currq[j].covariant_(2, 2);

        const double a2x2 = metric_currq[j + numans].covariant_(1, 0);
        const double a2y2 = metric_currq[j + numans].covariant_(1, 1);
        const double a2z2 = metric_currq[j + numans].covariant_(1, 2);

        const double a3x2 = metric_currq[j + numans].covariant_(2, 0);
        const double a3y2 = metric_currq[j + numans].covariant_(2, 1);
        const double a3z2 = metric_currq[j + numans].covariant_(2, 2);

        const double N1 = shapefunctions_q[j].shapefunctions_(i);
        const double N2 = shapefunctions_q[j + numans].shapefunctions_(i);

        const double dN1d1 = shapefunctions_q[j].derivatives_(0, i);
        const double dN2d2 = shapefunctions_q[j + numans].derivatives_(1, i);

        const double Nans1 = shapefunctions_ans[j];
        const double Nans2 = shapefunctions_ans[j + numans];

        Bop(5, nodedof * i + 0) += dN1d1 * a3x1 * Nans1;
        Bop(5, nodedof * i + 1) += dN1d1 * a3y1 * Nans1;
        Bop(5, nodedof * i + 2) += dN1d1 * a3z1 * Nans1;
        Bop(5, nodedof * i + 3) += N1 * a1x1 * Nans1;
        Bop(5, nodedof * i + 4) += N1 * a1y1 * Nans1;
        Bop(5, nodedof * i + 5) += N1 * a1z1 * Nans1;

        Bop(4, nodedof * i + 0) += dN2d2 * a3x2 * Nans2;
        Bop(4, nodedof * i + 1) += dN2d2 * a3y2 * Nans2;
        Bop(4, nodedof * i + 2) += dN2d2 * a3z2 * Nans2;
        Bop(4, nodedof * i + 3) += N2 * a2x2 * Nans2;
        Bop(4, nodedof * i + 4) += N2 * a2y2 * Nans2;
        Bop(4, nodedof * i + 5) += N2 * a2z2 * Nans2;
      }
    }
  }

  /*!
   * @brief Updates the current thickness of an element at a gaussian point
   *
   * @tparam distype : The discretization type known at compile time
   * qparam a3current (in) : Nodal directors in spatial configuration
   * @param shape_functions (in) : Shape functions and derivatives evaluated at the respective point
   * in the parameter space
   */
  template <Core::FE::CellType distype>
  double update_gauss_point_thickness(
      const Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim>& a3current,
      const Core::LinAlg::Matrix<Internal::num_node<distype>, 1>& shapefunctions)
  {
    double current_thickness = 0;
    for (int i = 0; i < Internal::num_node<distype>; ++i)
    {
      double nodal_thickness = 0;
      for (int d = 0; d < Internal::num_dim; ++d)
        nodal_thickness += a3current(i, d) * a3current(i, d);
      nodal_thickness = 2 * std::sqrt(nodal_thickness);
      current_thickness += shapefunctions(i) * nodal_thickness;
    }
    return std::abs(current_thickness);
  }

  /*!
   * @brief Evaluates the covariant basis vectors and metric tensors of the element
   *
   * @tparam distype :  The discretization type known at compile time
   * @param shape_functions (in) : Shape functions and derivatives evaluated at the respective point
   * in the parameter space
   * @param basis_and_metrics_reference (in/out) : An object holding the basis vectors and metric
   * tensors of the element in reference configuration
   * @param basis_and_metrics_current (in/out) : An object holding the basis vectors and metric
   * tensors of the element in current configuration
   * @param nodal_coordinates (in) : Coordinates of the nodes of the element
   * @param zeta (in) : Thickness coordinate of gaussian point (scaled via SDC)
   */
  template <Core::FE::CellType distype>
  void evaluate_metrics(
      const Discret::Elements::Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& basis_and_metrics_reference,
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& basis_and_metrics_current,
      const Discret::Elements::Shell::NodalCoordinates<distype>& nodal_coordinates,
      const double zeta)
  {
    evaluate_covariant_vectors_and_metrics(shape_functions, basis_and_metrics_reference,
        nodal_coordinates.x_refe_, nodal_coordinates.a3_refe_, zeta);
    evaluate_contravariant_vectors_and_metrics(basis_and_metrics_reference);
    evaluate_covariant_vectors_and_metrics(shape_functions, basis_and_metrics_current,
        nodal_coordinates.x_curr_, nodal_coordinates.a3_curr_, zeta);
    evaluate_contravariant_vectors_and_metrics(basis_and_metrics_current);
  }

  /*!
   * @brief Evaluates the covariant basis vectors and metric tensors of the element
   *
   * @tparam distype :  The discretization type known at compile time
   * @param shape_functions(in) : Shape functions and derivatives evaluated at the respective point
   * in the parameter space
   * @param basis_and_metrics (in/out) : An object holding the basis vectors and metric tensors of
   * the element
   * @param x (in) : Coordinates of the nodes of the element
   * @param a3 (in) : Coordinates of the nodal directors of the element
   * @param zeta (in) : Thickness coordinate of gaussian point (scaled via SDC)
   */
  template <Core::FE::CellType distype>
  void evaluate_covariant_vectors_and_metrics(
      const Discret::Elements::Shell::ShapefunctionsAndDerivatives<distype>& shape_functions,
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& basis_and_metrics,
      const Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim>& x,
      const Core::LinAlg::Matrix<Internal::num_node<distype>, Internal::num_dim>& a3,
      const double zeta)
  {
    // interpolation of covariant a1,a2
    basis_and_metrics.covariant_.multiply_nn(1.0, shape_functions.derivatives_, x, 0.0);

    // displacement field in the third dimension is approximated by assuming a linear variation of
    // the displacements across the thickness
    if (zeta)
    {
      Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim> tmp(
          Core::LinAlg::Initialization::zero);
      tmp.multiply_nn(zeta, shape_functions.derivatives_, a3, 0.0);
      basis_and_metrics.covariant_.update(1.0, tmp, 1.0);
    }

    // interpolation of a3
    for (int idim = 0; idim < Internal::num_dim; idim++)
    {
      basis_and_metrics.covariant_(2, idim) = 0.0;
      for (int inode = 0; inode < Internal::num_node<distype>; inode++)
        basis_and_metrics.covariant_(2, idim) +=
            shape_functions.shapefunctions_(inode) * a3(inode, idim);
    }
    // get covariant metric tensor
    basis_and_metrics.metric_covariant_.multiply_nt(
        1.0, basis_and_metrics.covariant_, basis_and_metrics.covariant_, 0.0);

    // get partial derivatives of the basis vector in thickness direction
    basis_and_metrics.partial_derivative_.multiply_nn(1.0, shape_functions.derivatives_, a3, 0.0);
  }

  /*!
   * @brief Evaluates the contravariant basis vectors and metric tensors of the element
   *
   * @tparam distype : The discretization type known at compile time
   *
   * @param metrics (in/out) : An object holding the basis vectors and metric tensors of the element
   */
  template <Core::FE::CellType distype>
  void evaluate_contravariant_vectors_and_metrics(
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& metrics)
  {
    // get contravariant basis vectors g1,g2,g3 (inverse transpose of cov)
    metrics.contravariant_ = metrics.covariant_;
    metrics.detJ_ = metrics.contravariant_.invert();

    // get the transpose
    for (int i = 0; i < Internal::num_dim; ++i)
    {
      for (int j = i + 1; j < Internal::num_dim; ++j)
      {
        const double tmp = metrics.contravariant_(j, i);
        metrics.contravariant_(j, i) = metrics.contravariant_(i, j);
        metrics.contravariant_(i, j) = tmp;
      }
    }
    metrics.metric_contravariant_ = metrics.metric_covariant_;
    // get contravariant metrictensor
    metrics.metric_contravariant_.invert();
  }

  /*!
   * @brief Modify the covariant metric tensors of the element to neglect the quadratic terms of the
   * normal strain in thickness direction.
   *
   * @Note : After the modification for the current metric tensor holds g_ij != g_i*g_j
   *
   * @tparam distype :  The discretization type known at compile time
   * @param g_reference (in/out) : An object holding the reference basis vectors and metric
   * tensors of the shell body
   * @param g_current (in/out) : An object holding the current basis vectors and metric
   * tensors of the shell body
   *  @param a_reference (in) : An object holding the reference basis vectors and metric
   * tensors of the midsurface
   * @param a_current (in) : An object holding the current basis vectors and metric tensors
   * tensors of the midsurface
   * @param zeta (in) : Thickness coordinate of gaussian point (scaled via SDC)
   */
  template <Core::FE::CellType distype>
  void modify_covariant_metrics_standard(
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_reference,
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_current,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_reference,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_current,
      const double& zeta)
  {
    double b11c = 0.0;
    double b12c = 0.0;
    double b21c = 0.0;
    double b22c = 0.0;
    double b31c = 0.0;
    double b32c = 0.0;
    double b11r = 0.0;
    double b12r = 0.0;
    double b21r = 0.0;
    double b22r = 0.0;
    double b31r = 0.0;
    double b32r = 0.0;

    for (int i = 0; i < Internal::num_dim; ++i)
    {
      b11c += a_current.covariant_(0, i) * a_current.partial_derivative_(0, i);
      b12c += a_current.covariant_(0, i) * a_current.partial_derivative_(1, i);
      b21c += a_current.covariant_(1, i) * a_current.partial_derivative_(0, i);
      b22c += a_current.covariant_(1, i) * a_current.partial_derivative_(1, i);
      b31c += a_current.covariant_(2, i) * a_current.partial_derivative_(0, i);
      b32c += a_current.covariant_(2, i) * a_current.partial_derivative_(1, i);

      b11r += a_reference.covariant_(0, i) * a_reference.partial_derivative_(0, i);
      b12r += a_reference.covariant_(0, i) * a_reference.partial_derivative_(1, i);
      b21r += a_reference.covariant_(1, i) * a_reference.partial_derivative_(0, i);
      b22r += a_reference.covariant_(1, i) * a_reference.partial_derivative_(1, i);
      b31r += a_reference.covariant_(2, i) * a_reference.partial_derivative_(0, i);
      b32r += a_reference.covariant_(2, i) * a_reference.partial_derivative_(1, i);
    }
    g_current.metric_covariant_(0, 0) =
        g_reference.metric_covariant_(0, 0) +
        (a_current.metric_covariant_(0, 0) - a_reference.metric_covariant_(0, 0)) +
        zeta * 2.0 * (b11c - b11r);
    g_current.metric_covariant_(1, 1) =
        g_reference.metric_covariant_(1, 1) +
        (a_current.metric_covariant_(1, 1) - a_reference.metric_covariant_(1, 1)) +
        zeta * 2.0 * (b22c - b22r);
    g_current.metric_covariant_(2, 2) =
        g_reference.metric_covariant_(2, 2) +
        (a_current.metric_covariant_(2, 2) - a_reference.metric_covariant_(2, 2));

    g_current.metric_covariant_(0, 1) =
        g_reference.metric_covariant_(0, 1) +
        (a_current.metric_covariant_(0, 1) - a_reference.metric_covariant_(0, 1)) +
        zeta * (b21c + b12c - b21r - b12r);
    g_current.metric_covariant_(0, 2) =
        g_reference.metric_covariant_(0, 2) +
        (a_current.metric_covariant_(0, 2) - a_reference.metric_covariant_(0, 2)) +
        zeta * (b31c - b31r);
    g_current.metric_covariant_(1, 2) =
        g_reference.metric_covariant_(1, 2) +
        (a_current.metric_covariant_(1, 2) - a_reference.metric_covariant_(1, 2)) +
        zeta * (b32c - b32r);

    g_current.metric_covariant_(2, 0) = g_current.metric_covariant_(0, 2);
    g_current.metric_covariant_(2, 1) = g_current.metric_covariant_(1, 2);
    g_current.metric_covariant_(1, 0) = g_current.metric_covariant_(0, 1);

    g_current.metric_contravariant_.update(g_current.metric_covariant_);
    double detJ = g_current.metric_contravariant_.invert();
    g_current.detJ_ = std::sqrt(detJ);
    g_reference.metric_contravariant_.update(g_reference.metric_covariant_);
    detJ = g_reference.metric_contravariant_.invert();
    g_reference.detJ_ = std::sqrt(detJ);
  }

  /*!
   * @brief Modify the covariant metric tensors of the element due to transverse shear strain ANS
   *
   * @Note : After the modification: g_ij != g_i*g_j
   *
   * @tparam distype :  The discretization type known at compile time
   * @param g_reference (in/out) : An object holding the reference basis vectors and metric
   * tensors of the shell body
   * @param g_current (in/out) : An object holding the current basis vectors and metric
   * tensors of the shell body
   * @param a_reference (in) : An object holding the reference basis vectors and metric
   * tensors of the midsurface
   * @param a_current (in) : An object holding the current basis vectors and metric tensors
   * tensors of the midsurface
   * @param zeta (in) : thickness coordinate of gaussian point (scaled via SDC)
   * @param shapefunctions_ans (in) : Vector holding the ANS shapefunctions and derivatives
   * @param metrics_collocation_reference (in) : vector holding the reference basis vectors and
   * metric tensors of the midsurface at collocation points
   * @param metrics_collocation_current (in) : Vector holding the current basis vectors and metric
   * tensors of the midsurface at collocation points
   * @param numansq (in) : Number of ANS collocation points
   */
  template <Core::FE::CellType distype>
  void modify_covariant_metrics_ans(
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_reference,
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_current,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_reference,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_current,
      const double& zeta, const std::vector<double>& shapefunctions_ans,
      const std::vector<Discret::Elements::Shell::BasisVectorsAndMetrics<distype>>&
          metrics_collocation_reference,
      const std::vector<Discret::Elements::Shell::BasisVectorsAndMetrics<distype>>&
          metrics_collocation_current,
      const int& numansq)
  {
    double b11c = 0.0;
    double b12c = 0.0;
    double b21c = 0.0;
    double b22c = 0.0;
    double b31c = 0.0;
    double b32c = 0.0;
    double b11r = 0.0;
    double b12r = 0.0;
    double b21r = 0.0;
    double b22r = 0.0;
    double b31r = 0.0;
    double b32r = 0.0;

    for (int i = 0; i < Internal::num_dim; ++i)
    {
      b11c += a_current.covariant_(0, i) * a_current.partial_derivative_(0, i);
      b12c += a_current.covariant_(0, i) * a_current.partial_derivative_(1, i);
      b21c += a_current.covariant_(1, i) * a_current.partial_derivative_(0, i);
      b22c += a_current.covariant_(1, i) * a_current.partial_derivative_(1, i);
      b31c += a_current.covariant_(2, i) * a_current.partial_derivative_(0, i);
      b32c += a_current.covariant_(2, i) * a_current.partial_derivative_(1, i);

      b11r += a_reference.covariant_(0, i) * a_reference.partial_derivative_(0, i);
      b12r += a_reference.covariant_(0, i) * a_reference.partial_derivative_(1, i);
      b21r += a_reference.covariant_(1, i) * a_reference.partial_derivative_(0, i);
      b22r += a_reference.covariant_(1, i) * a_reference.partial_derivative_(1, i);
      b31r += a_reference.covariant_(2, i) * a_reference.partial_derivative_(0, i);
      b32r += a_reference.covariant_(2, i) * a_reference.partial_derivative_(1, i);
    }
    g_current.metric_covariant_(0, 0) =
        g_reference.metric_covariant_(0, 0) +
        (a_current.metric_covariant_(0, 0) - a_reference.metric_covariant_(0, 0)) +
        zeta * 2.0 * (b11c - b11r);
    g_current.metric_covariant_(1, 1) =
        g_reference.metric_covariant_(1, 1) +
        (a_current.metric_covariant_(1, 1) - a_reference.metric_covariant_(1, 1)) +
        zeta * 2.0 * (b22c - b22r);
    g_current.metric_covariant_(2, 2) =
        g_reference.metric_covariant_(2, 2) +
        (a_current.metric_covariant_(2, 2) - a_reference.metric_covariant_(2, 2));

    g_current.metric_covariant_(0, 1) =
        g_reference.metric_covariant_(0, 1) +
        (a_current.metric_covariant_(0, 1) - a_reference.metric_covariant_(0, 1)) +
        zeta * (b21c + b12c - b21r - b12r);
    g_current.metric_covariant_(0, 2) = g_reference.metric_covariant_(0, 2) + zeta * (b31c - b31r);
    g_current.metric_covariant_(1, 2) = g_reference.metric_covariant_(1, 2) + zeta * (b32c - b32r);
    for (int qp = 0; qp < numansq; ++qp)
    {
      g_current.metric_covariant_(0, 2) +=
          (metrics_collocation_current[qp].metric_covariant_(0, 2) -
              metrics_collocation_reference[qp].metric_covariant_(0, 2)) *
          shapefunctions_ans[qp];
      g_current.metric_covariant_(1, 2) +=
          (metrics_collocation_current[qp + numansq].metric_covariant_(1, 2) -
              metrics_collocation_reference[qp + numansq].metric_covariant_(1, 2)) *
          shapefunctions_ans[qp + numansq];
    }
    g_current.metric_covariant_(2, 0) = g_current.metric_covariant_(0, 2);
    g_current.metric_covariant_(2, 1) = g_current.metric_covariant_(1, 2);
    g_current.metric_covariant_(1, 0) = g_current.metric_covariant_(0, 1);

    g_current.metric_contravariant_.update(g_current.metric_covariant_);
    double detJ = g_current.metric_contravariant_.invert();
    g_current.detJ_ = std::sqrt(detJ);
    g_reference.metric_contravariant_.update(g_reference.metric_covariant_);
    detJ = g_reference.metric_contravariant_.invert();
    g_reference.detJ_ = std::sqrt(detJ);
  }


  /*!
   * @brief Modify the covariant metric tensors of the element
   *
   *
   * @tparam distype :  The discretization type known at compile time
   * @param g_reference (in/out) : An object holding the reference basis vectors and metric
   * tensors of the shell body
   * @param g_current (in/out) : An object holding the current basis vectors and metric
   * tensors of the shell body
   * @param a_reference (in) : An object holding the reference basis vectors and metric
   * tensors of the midsurface
   * @param a_current (in) : An object holding the current basis vectors and metric tensors
   * tensors of the midsurface
   * @param zeta (in) : thickness coordinate of gaussian point (scaled via SDC)
   * @param shapefunctions_ans (in) : Vector holding the ANS shapefunctions and derivatives
   * @param metrics_collocation_reference (in) : vector holding the reference basis vectors and
   * metric tensors of the midsurface at collocation points
   * @param metrics_collocation_current (in) : Vector holding the current basis vectors and metric
   * tensors of the midsurface at collocation points
   * @param numansq (in) : Number of ANS collocation points
   */
  template <Core::FE::CellType distype>
  void modify_covariant_metrics(
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_reference,
      Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_current,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_reference,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& a_current,
      const double& zeta, const std::vector<double>& shapefunctions_ans,
      const std::vector<Discret::Elements::Shell::BasisVectorsAndMetrics<distype>>&
          metrics_collocation_reference,
      const std::vector<Discret::Elements::Shell::BasisVectorsAndMetrics<distype>>&
          metrics_collocation_current,
      const int& numansq)
  {
    std::invoke(
        [&]()
        {
          if (numansq == 0)
          {
            modify_covariant_metrics_standard(g_reference, g_current, a_reference, a_current, zeta);
          }
          else
          {
            // modify the current covariant metric tensor due to transverse shear strain ANS
            modify_covariant_metrics_ans(g_reference, g_current, a_reference, a_current, zeta,
                shapefunctions_ans, metrics_collocation_reference, metrics_collocation_current,
                numansq);
          }
        });
  }

  /*!
   * @brief Evaluates displacement based Deformation gradient tensor from metric tensors
   *
   * The deformamtion gradient tensor is defined as :
   * \f[
   *    \mathbf{F} = \mathbf{g}_i \otimes \mathbf{G}^i
   * \f]
   * This can be transformed to a cartesian system by \f$ F_{ij}=\mathbf{E}_i\cdot \mathbf{F}
   * \mathbf{E}_j$\f. As we used a global cartesian basis with the basis vectors  \f$ E_1=[1, 0,
   * 0]^T, E_2=[0, 1, 0]^T, E_3=[0, 0, 1]^T$\f, no transformation is necessary
   *
   * @tparam distype : The discretization type known at compile time
   * @param g_reference (in) : An object holding the reference basis vectors and metric
   * tensors of the shell body
   * @param g_current (in) : An object holding the current basis vectors and metric
   * tensors of the shell body
   * @return Core::LinAlg::Tensor<double,3,3> : Deformation gradient tensor
   */
  template <Core::FE::CellType distype>
  Core::LinAlg::Tensor<double, 3, 3> evaluate_deformation_gradient(
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_reference,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_current)
  {
    Core::LinAlg::Matrix<3, 3> deformation_gradient(Core::LinAlg::Initialization::zero);
    deformation_gradient.multiply_nt(1.0, g_current.covariant_, g_reference.contravariant_);
    return Core::LinAlg::make_tensor(deformation_gradient);
  }

  /*!
   * @brief Evaluates Green-Lagrange strains in curvilinear coordinate system
   *
   *  The Green-Lagrange strains in curvilinear coordinates are defined as:
   * \f[
   *    E_{ij} = \frac{1}{2} (g_{ij} - G_{ij})
   * \f]
   * where g_ij is the current metric and G_ij is the reference metric.
   *
   * @tparam distype : The discretization type known at compile time
   * @param g_reference (in) : Reference basis vectors and metric tensors
   * @param g_current (in) : Current basis vectors and metric tensors
   * @return Core::LinAlg::SymmetricTensor<double, 3, 3> : Green-Lagrange strain tensor in
   * curvilinear coordinates
   */
  template <Core::FE::CellType distype>
  Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_green_lagrange_strain(
      const BasisVectorsAndMetrics<distype>& g_reference,
      const BasisVectorsAndMetrics<distype>& g_current)
  {
    return Core::LinAlg::assume_symmetry(
        0.5 * (Core::LinAlg::make_tensor_view(g_current.metric_covariant_) -
                  Core::LinAlg::make_tensor_view(g_reference.metric_covariant_)));
  }

  /*!
   * @brief Transform Green-Lagrange strain from curvilinear to Cartesian coordinates
   *
   * Applies the transformation: E_cart = G^T * E_curv * G
   * where G is the contravariant basis tensor from reference configuration.
   *
   * @tparam distype : The discretization type known at compile time
   * @param gl_strain_curvilinear (in) : GL strain in curvilinear coordinates
   * @param g_reference (in) : Reference metric tensors (provides contravariant basis)
   * @return Green-Lagrange strain tensor in Cartesian coordinates
   */
  template <Core::FE::CellType distype>
  inline Core::LinAlg::SymmetricTensor<double, 3, 3> transform_green_lagrange_strain_to_cartesian(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl_strain_curvilinear,
      const BasisVectorsAndMetrics<distype>& g_reference)
  {
    Core::LinAlg::TensorView<const double, 3, 3> contravariant_tensor =
        Core::LinAlg::make_tensor_view(g_reference.contravariant_);
    Core::LinAlg::Tensor<double, 3, 3> gl_strain_tensor_cartesian =
        Core::LinAlg::transpose(contravariant_tensor) * gl_strain_curvilinear *
        contravariant_tensor;
    return Core::LinAlg::assume_symmetry(gl_strain_tensor_cartesian);
  }


  /*!
   * @brief Maps the stress quantities (2. Piola-Kirchoff stress tensor and the linearization
   * w.r.t Green-Lagrange strain) from the global cartesioan system
   * back to the curvilinear system and re-arranged the order for the specific shell formulation
   *
   * @Note: PK Stress are ordered here like
   * \f[
   *    \mathbf{S} = [S_{11},S_{12}, S_{13}, S_{22}, S_{23}, S_{33}]^T
   * \f]
   *
   * @tparam distype :  The discretization type known at compile time
   * @param stress (in/out) : Stress quantities of the element (2. Piola-Kirchoff stress tensor
   * and the linearization w.r.t Green-Lagrange strain)
   * @param g_reference (in) : An object holding the reference basis vectors and metric
   * tensors of the shell body
   */
  template <Core::FE::CellType distype>
  void map_material_stress_to_curvilinear_system(
      Stress& stress, const BasisVectorsAndMetrics<distype>& g_reference)
  {
    // transform Piola-Kirchhoff-stresses from global cartesian coordinate system back to local
    // curvilinear coordinate system
    Core::LinAlg::TensorView<const double, 3, 3> contravariant_tensor =
        Core::LinAlg::make_tensor_view(g_reference.contravariant_);
    Core::LinAlg::Tensor<double, 3, 3> pk2_curvilinear =
        contravariant_tensor * stress.pk2_ * Core::LinAlg::transpose(contravariant_tensor);
    stress.pk2_ = Core::LinAlg::assume_symmetry(pk2_curvilinear);

    //  transform elasticity tensor from global cartesian coordinate system back to local
    //  curvilinear coordinate system
    stress.cmat_ = Core::LinAlg::einsum_sym<"iI", "jJ", "kK", "lL", "IJKL">(contravariant_tensor,
        contravariant_tensor, contravariant_tensor, contravariant_tensor, stress.cmat_);
  }

  /*!
   * @brief Get the shapefunctions evaluated at the gaussian points to remedy transverse shear
   * strains within the ANS method
   *
   * @tparam distype : The discretization type known at compile time
   * @param xi_gp (in) : Coordinate of the integration point in the parameter space
   * @return shapefunctions for ANS
   */
  template <Core::FE::CellType distype>
  static std::vector<double> set_shapefunctions_for_ans(const std::array<double, 2>& xi_gp)
  {
    std::vector<double> shapefunctions;
    double r = xi_gp[0];
    double s = xi_gp[1];

    if (distype == Core::FE::CellType::quad4)
    {
      shapefunctions.resize(4);
      shapefunctions[0] = 0.5 * (1.0 - s);
      shapefunctions[1] = 0.5 * (1.0 + s);
      shapefunctions[2] = 0.5 * (1.0 - r);
      shapefunctions[3] = 0.5 * (1.0 + r);
    }
    else if (distype == Core::FE::CellType::quad9)
    {
      shapefunctions.resize(12);
      const double rthreei = 1.0 / (sqrt(3.0));
      std::array<double, 3> pr;
      std::array<double, 3> ps;
      std::array<double, 2> qr;
      std::array<double, 2> qs;

      pr[0] = -0.5 * s * (1.0 - s);
      pr[1] = (1.0 - s) * (1.0 + s);
      pr[2] = 0.5 * s * (1.0 + s);

      qr[0] = 0.5 * (1.0 - r / rthreei);
      qr[1] = 0.5 * (1.0 + r / rthreei);

      ps[0] = -0.5 * r * (1.0 - r);
      ps[1] = (1.0 - r) * (1.0 + r);
      ps[2] = 0.5 * r * (1.0 + r);

      qs[0] = 0.5 * (1.0 - s / rthreei);
      qs[1] = 0.5 * (1.0 + s / rthreei);

      shapefunctions[0] = pr[0] * qr[0];
      shapefunctions[1] = pr[1] * qr[0];
      shapefunctions[2] = pr[2] * qr[0];
      shapefunctions[3] = pr[0] * qr[1];
      shapefunctions[4] = pr[1] * qr[1];
      shapefunctions[5] = pr[2] * qr[1];

      shapefunctions[6] = ps[0] * qs[0];
      shapefunctions[7] = ps[1] * qs[0];
      shapefunctions[8] = ps[2] * qs[0];
      shapefunctions[9] = ps[0] * qs[1];
      shapefunctions[10] = ps[1] * qs[1];
      shapefunctions[11] = ps[2] * qs[1];
    }
    return shapefunctions;
  }

  /*!
   * @brief Get the shapefunctions evaluated at the gaussian points to remedy transverse shear
   * strains within the ANS method
   *
   * @tparam distype : The discretization type known at compile time
   * @param xi_gp (in) : Coordinate of the integration point in the parameter space
   * @param num_ans (in) : Number of ANS collocation points
   * @return shapefunctions for ANS
   */
  template <Core::FE::CellType distype>
  std::vector<double> get_shapefunctions_for_ans(
      const std::array<double, 2>& xi_gp, const int& num_ans)
  {
    const std::vector<double> shape_functions_ans = std::invoke(
        [&]()
        {
          if (num_ans == 0)
          {
            return std::vector<double>{};
          }
          else
          {
            return Shell::set_shapefunctions_for_ans<distype>(xi_gp);
          }
        });
    return shape_functions_ans;
  }

  /*!
   * @brief Transform the Green-Lagrange strains to Euler-Almansi strains
   *
   * @param gl (in) :  Green-Lagrange strains
   * @param defgrd (in) : Deformations gradient tensor
   * @return Core::LinAlg::SymmetricTensor<double, 3, 3> : Euler-Almansi strain tensor
   */
  inline Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange_to_euler_almansi(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl,
      const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
  {
    Core::LinAlg::Tensor<double, 3, 3> invdefgrd = Core::LinAlg::inv(defgrd);

    return Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(invdefgrd) * gl * invdefgrd);
  }

  /*!
   * @brief Converts the 2nd Piola-Kirchhoff stress tensor to the
   * Cauchy stress tensor
   *
   * @param pk2 (in) : 2nd Piola-Kirchhoff stress tensor
   * @param defgrd (in) : Deformation gradient
   * @return Core::LinAlg::SymmetricTensor<double, 3, 3> : Cauchy stress tensor
   */
  inline Core::LinAlg::SymmetricTensor<double, 3, 3> pk2_to_cauchy(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& pk2,
      const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
  {
    return Core::LinAlg::assume_symmetry(defgrd * pk2 * Core::LinAlg::transpose(defgrd)) /
           Core::LinAlg::det(defgrd);
  }

  /*!
   * @brief Assemble a symmetric tensor into a matrix row
   *
   * @tparam size : Size of the symmetric tensor
   * @param tensor (in) : Symmetric tensor to be assembled into matrix
   * @param data (in/out) : Matrix the tensor is assembled into
   * @param row (in) : Matrix row
   * @param thickness_weight (in) : Weighting factor to consider thickness integration
   */
  template <std::size_t size>
  void assemble_symmetric_tensor_to_matrix_row(
      const Core::LinAlg::SymmetricTensor<double, size, size>& tensor,
      Core::LinAlg::SerialDenseMatrix& data, const int row, double thickness_weight = 1.0)
  {
    for (unsigned i = 0; i < tensor.container().size(); ++i)
      data(row, static_cast<int>(i)) += thickness_weight * tensor.data()[i];
  }

  /*!
   * @brief Assembles Green-Lagrange strains to the desired strain type and assemble to a given
   * matrix row
   *
   * @param gl_strain (in) : Green-Lagrange strain tensor
   * @param defgrd (in) : Deformation gradient tensor
   * @param strain_type (in) : Strain type
   * @param data (in) : Data to which the vector will be assembled
   * @param row (in) : Row number
   * @param thickness_weight (in) : Weighting factor to consider thickness integration
   */
  inline void assemble_strain_type_to_matrix_row(
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& gl_strain,
      const Core::LinAlg::Tensor<double, 3, 3>& defgrd, Inpar::Solid::StrainType strain_type,
      Core::LinAlg::SerialDenseMatrix& data, int row, const double thickness_weight)
  {
    switch (strain_type)
    {
      case Inpar::Solid::strain_gl:
      {
        assemble_symmetric_tensor_to_matrix_row(gl_strain, data, row, thickness_weight);
        return;
      }
      case Inpar::Solid::strain_ea:
      {
        const Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim> ea =
            green_lagrange_to_euler_almansi(gl_strain, defgrd);
        assemble_symmetric_tensor_to_matrix_row(ea, data, row, thickness_weight);
        return;
      }
      case Inpar::Solid::strain_none:
        return;
      default:
        FOUR_C_THROW("strain type not supported");
    }
  }
  /*!
   * @brief Assembles 2nd Piola-Kirchhoff stresses to the desired stress type in matrix row
   *
   * @param defgrd (in) : Deformation gradient tensor
   * @param stress (in) : Stress data
   * @param stress_type (in) : Stress type
   * @param data (in) : Data to which the vector will be assembled
   * @param row (in) : Row number
   * @param thickness_weight (in) : Weighting factor to consider thickness integration
   */
  inline void assemble_stress_type_to_matrix_row(const Core::LinAlg::Tensor<double, 3, 3>& defgrd,
      const Stress stress, Inpar::Solid::StressType stress_type,
      Core::LinAlg::SerialDenseMatrix& data, int row, const double thickness_weight)
  {
    switch (stress_type)
    {
      case Inpar::Solid::stress_2pk:
      {
        assemble_symmetric_tensor_to_matrix_row(stress.pk2_, data, row, thickness_weight);
        return;
      }
      case Inpar::Solid::stress_cauchy:
      {
        Core::LinAlg::SymmetricTensor<double, Internal::num_dim, Internal::num_dim> cauchy =
            pk2_to_cauchy(stress.pk2_, defgrd);
        assemble_symmetric_tensor_to_matrix_row(cauchy, data, row, thickness_weight);
        return;
      }
      case Inpar::Solid::stress_none:
        return;
      default:
        FOUR_C_THROW("stress type not supported");
    }
  }

  inline void serialize(const Core::LinAlg::SerialDenseMatrix& matrix, std::vector<char>& data)
  {
    Core::Communication::PackBuffer packBuffer;
    add_to_pack(packBuffer, matrix);
    std::copy(packBuffer().begin(), packBuffer().end(), std::back_inserter(data));
  }


  /*!
   * @brief Returns the local coordinates of the collocation points for ANS.
   *
   * This functions returns the local coordinates of the collocation points for ANS to avoid
   * transverse shear locking. The choice of coordinates depends on the discretization types of the
   * element. Here, they are chosen such that the points lie on the middle of each edge.
   *
   * @tparam distype : discretization type
   * @param qp (in) :  Index of collocation point
   */
  template <Core::FE::CellType distype>
  void get_coordinates_of_ans_collocation_points(
      std::vector<std::array<double, 2>>& collocation_points)
  {
    if (distype == Core::FE::CellType::quad4)
    {
      collocation_points.resize(4);
      // for a_13
      collocation_points[0] = {0.0, -1.0};
      collocation_points[1] = {0.0, 1.0};
      // for a_23
      collocation_points[2] = {-1.0, 0.0};
      collocation_points[3] = {1.0, 0.0};
    }
    else if (distype == Core::FE::CellType::quad9)
    {
      const double sqrt3inv = 1.0 / (sqrt(3.0));
      collocation_points.resize(Internal::num_internal_variables);
      // for a_13
      collocation_points[0] = {-sqrt3inv, -1.0};
      collocation_points[1] = {-sqrt3inv, 0.0};
      collocation_points[2] = {-sqrt3inv, 1.0};
      collocation_points[3] = {sqrt3inv, -1.0};
      collocation_points[4] = {sqrt3inv, 0.0};
      collocation_points[5] = {sqrt3inv, 1.0};
      // for a_23
      collocation_points[6] = {-1.0, -sqrt3inv};
      collocation_points[7] = {0.0, -sqrt3inv};
      collocation_points[8] = {1.0, -sqrt3inv};
      collocation_points[9] = {-1.0, sqrt3inv};
      collocation_points[10] = {0.0, sqrt3inv};
      collocation_points[11] = {1.0, sqrt3inv};
    }
  }

  /*!
   * @brief Setup shapefunction, the first derivatives w.r.t spatial coordinates and the metrics
   * evaluated at collocation points for ANS
   *
   * @tparam distype : The discretization type known at compile time
   * @param shapefunctions_collocation (in/out) :  An object holding the shape functions and the
   * first derivatives w.r.t spatial coordinates
   * @param metrics_collocation_reference (in/out) :  An object holding the reference basis vectors
   * and metric tensors of the shell body
   * @param metrics_collocation_current (in/out) : An object holding the current basis vectors and
   * metric tensors of the shell body
   * @param nodal_coordinates (in) : Nodal coordinates in reference and current frame
   * @param num_collocation_points (in) : Number of collocation points
   */
  template <Core::FE::CellType distype>
  void setup_ans(
      std::vector<Shell::ShapefunctionsAndDerivatives<distype>>& shapefunctions_collocation,
      std::vector<Shell::BasisVectorsAndMetrics<distype>>& metrics_collocation_reference,
      std::vector<Shell::BasisVectorsAndMetrics<distype>>& metrics_collocation_current,
      const NodalCoordinates<distype> nodal_coordinates, const int& num_collocation_points)
  {
    // get coordinates of collocations points
    std::vector<std::array<double, 2>> collocation_points;
    Shell::get_coordinates_of_ans_collocation_points<distype>(collocation_points);
    for (int qp = 0; qp < num_collocation_points; ++qp)
    {
      // get shape functions and derivatives for ANS at each collocation point
      shapefunctions_collocation[qp] =
          Shell::evaluate_shapefunctions_and_derivs<distype>(collocation_points[qp]);
      //  get basis vectors and metric at collocation points
      Shell::evaluate_metrics(shapefunctions_collocation[qp], metrics_collocation_reference[qp],
          metrics_collocation_current[qp], nodal_coordinates, 0.0);
    }
  }

  /*!
   * @brief This function performs the preintegration of the the 2. Piola-Kirchhoff stresses and
   * the material tensor through the shell thickness.
   *
   * @tparam distype : The discretization type known at compile time
   * @param stress_resultants (in/out) : An object holding the stress resultants
   * @param stress (in) :  An object holding the stress measures
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant
   * of the jacobian)
   * @param zeta (in) : Thickness coordinate of gaussian point (scaled via SDC)
   * to
   */
  template <Core::FE::CellType distype>
  void thickness_integration(Discret::Elements::Shell::StressResultants& stress_resultants,
      const Discret::Elements::Shell::Stress& stress, const double& integration_factor,
      const double& zeta)
  {
    const auto pk2_view = Core::LinAlg::make_stress_like_voigt_view(stress.pk2_);
    const auto cmat_view = Core::LinAlg::make_stress_like_voigt_view(stress.cmat_);

    for (int i = 0; i < Internal::node_dof; ++i)
    {
      const double stress_fact = pk2_view(i) * integration_factor;
      stress_resultants.stress_(i) += stress_fact;
      stress_resultants.stress_(i + Internal::node_dof) += stress_fact * zeta;

      for (int j = 0; j < Internal::node_dof; ++j)
      {
        const double C_fact = cmat_view(i, j) * integration_factor;
        stress_resultants.dmat_(i, j) += C_fact;
        stress_resultants.dmat_(i + Internal::node_dof, j) += C_fact * zeta;
        stress_resultants.dmat_(i + Internal::node_dof, j + Internal::node_dof) +=
            C_fact * zeta * zeta;
      }
    }
    // symmetrize enhanced material tensor
    for (int i = 0; i < Internal::num_internal_variables; i++)
      for (int j = i + 1; j < Internal::num_internal_variables; j++)
        stress_resultants.dmat_(i, j) = stress_resultants.dmat_(j, i);
  }

  /*!
   * @brief Evaluates strain measures in curvilinear coordinate system
   *
   * @tparam distype : The discretization type known at compile time
   * @param g_reference (in) : An object holding the reference basis vectors and metric
   * tensors of the shell body
   * @param g_current (in) : An object holding the current basis vectors and metric
   * tensors of the shell body
   * @return Strains<distype> : Strain measures of the element (deformation gradient, Green-Lagrange
   * strain tensor)
   */
  template <Core::FE::CellType distype>
  Discret::Elements::Shell::Strains evaluate_strains(
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_metric_reference,
      const Discret::Elements::Shell::BasisVectorsAndMetrics<distype>& g_metric_current)
  {
    Strains strains{};
    strains.defgrd_ = evaluate_deformation_gradient(g_metric_reference, g_metric_current);
    strains.gl_strain_ = evaluate_green_lagrange_strain(g_metric_reference, g_metric_current);
    return strains;
  }

  /*!
   * @brief Adds elastic stiffness matrix contribution of one Gauss point
   *
   * @tparam distype : The discretization type known at compile time
   * @param Bop (in) : Strain gradient (B-Operator)
   * @param D (in) : Material tensor (equals integration of the material tensor
   * through the shell thickness)
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant
   * of the jacobian times scaled director factor)
   * @param stiffness_matrix (in/out) : Stiffness matrix where the local contribution is added
   * to
   */
  template <Core::FE::CellType distype>
  void add_elastic_stiffness_matrix(const Core::LinAlg::SerialDenseMatrix& Bop,
      const Core::LinAlg::SerialDenseMatrix& Dmat, const double& integration_fac,
      Core::LinAlg::SerialDenseMatrix& stiffness_matrix)
  {
    // calculate Ke = integration_factor * B^TDB
    Core::LinAlg::SerialDenseMatrix DB(
        Internal::num_internal_variables, Internal::numdofperelement<distype>);
    Core::LinAlg::multiply(DB, Dmat, Bop);
    Core::LinAlg::multiply_tn(1.0, stiffness_matrix, integration_fac, Bop, DB);
  }

  /*!
   * @brief Adds geometric stiffness matrix contribution of one Gauss point
   *
   * @tparam distype : The discretization type known at compile time
   *  @param shapefunctions_collocation (in) : Shapefunctions and the first derivatives w.r.t
   * spatial coordinates evaluated at collocation points
   * @param shapefunctions_ans (in) : Shapefunctions for transverse shear strain ANS
   * @param shapefunctions (in) : Shapefunctions and the first derivatives w.r.t spatial coordinates
   * (midsurface)
   * @param stress_resultant_vector (in) : Stress resultant vector (membrane forces and moments)
   * @param numansq (in) : Number of ANS collocation points
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant
   * of the jacobian)
   * @param stiffness_matrix (in/out) : Stiffness matrix where the local contribution is added
   * to
   */
  template <Core::FE::CellType distype>
  void add_geometric_stiffness_matrix(
      const std::vector<Discret::Elements::Shell::ShapefunctionsAndDerivatives<distype>>&
          shapefunctions_q,
      const std::vector<double>& shapefunctions_ans,
      const Discret::Elements::Shell::ShapefunctionsAndDerivatives<distype>& shapefunctions,
      const Core::LinAlg::SerialDenseVector& stress_resultant_vector, const int& numans,
      const double& integration_fac, Core::LinAlg::SerialDenseMatrix& stiffness_matrix)
  {
    const int nodedof = Internal::node_dof;
    const int num_dim = Internal::num_dim;
    for (int inod = 0; inod < Internal::num_node<distype>; ++inod)
    {
      for (int jnod = 0; jnod <= inod; ++jnod)
      {
        const double Ni = shapefunctions.shapefunctions_(inod);
        const double Nj = shapefunctions.shapefunctions_(jnod);
        const double dN11 =
            shapefunctions.derivatives_(0, inod) * shapefunctions.derivatives_(0, jnod);
        const double dN12 =
            shapefunctions.derivatives_(0, inod) * shapefunctions.derivatives_(1, jnod);
        const double dN21 =
            shapefunctions.derivatives_(1, inod) * shapefunctions.derivatives_(0, jnod);
        const double dN22 =
            shapefunctions.derivatives_(1, inod) * shapefunctions.derivatives_(1, jnod);

        const double tmp1 =
            (dN11 * stress_resultant_vector(0) + (dN12 + dN21) * stress_resultant_vector(3) +
                dN22 * stress_resultant_vector(1)) *
            integration_fac;
        const double tmp2 =
            (dN11 * stress_resultant_vector(6) + (dN12 + dN21) * stress_resultant_vector(9) +
                dN22 * stress_resultant_vector(7)) *
            integration_fac;

        double tmp3 = 0.0;
        double tmp4 = 0.0;
        const double dN1dij = shapefunctions.derivatives_(0, inod) * Nj;
        const double dN1dji = shapefunctions.derivatives_(0, jnod) * Ni;
        const double dN2dij = shapefunctions.derivatives_(1, inod) * Nj;
        const double dN2dji = shapefunctions.derivatives_(1, jnod) * Ni;

        if (!numans)
        {
          tmp3 = (dN1dji * stress_resultant_vector(5) + dN2dji * stress_resultant_vector(4)) *
                 integration_fac;
          tmp4 = (dN1dij * stress_resultant_vector(5) + dN2dij * stress_resultant_vector(4)) *
                 integration_fac;
        }
        // modification due to transverse shear strain ANS
        else
        {
          for (int q = 0; q < numans; ++q)
          {
            double dNq1dij = shapefunctions_q[q].derivatives_(0, inod) *
                             shapefunctions_q[q].shapefunctions_(jnod) * shapefunctions_ans[q];
            double dNq1dji = shapefunctions_q[q].derivatives_(0, jnod) *
                             shapefunctions_q[q].shapefunctions_(inod) * shapefunctions_ans[q];
            int index_q23 = q + numans;
            double dNq2dij = shapefunctions_q[index_q23].derivatives_(1, inod) *
                             shapefunctions_q[index_q23].shapefunctions_(jnod) *
                             shapefunctions_ans[index_q23];
            double dNq2dji = shapefunctions_q[index_q23].derivatives_(1, jnod) *
                             shapefunctions_q[index_q23].shapefunctions_(inod) *
                             shapefunctions_ans[index_q23];
            tmp3 += (dNq1dji * stress_resultant_vector(5) + dNq2dji * stress_resultant_vector(4)) *
                    integration_fac;
            tmp4 += (dNq1dij * stress_resultant_vector(5) + dNq2dij * stress_resultant_vector(4)) *
                    integration_fac;
          }
        }

        const double tmp5 = ((dN1dij + dN1dji) * stress_resultant_vector(11) +
                                (dN2dij + dN2dji) * stress_resultant_vector(10)) *
                            integration_fac;
        const double tmp6 = (Ni * Nj * stress_resultant_vector(2)) * integration_fac;

        for (int d = 0; d < num_dim; ++d)
        {
          stiffness_matrix(inod * nodedof + d, jnod * nodedof + d) += tmp1;
          stiffness_matrix(inod * nodedof + d + num_dim, jnod * nodedof + d) += (tmp2 + tmp3);
          stiffness_matrix(inod * nodedof + d, jnod * nodedof + d + num_dim) += (tmp2 + tmp4);
          stiffness_matrix(inod * nodedof + d + num_dim, jnod * nodedof + d + num_dim) +=
              (tmp5 + tmp6);
        }
        if (inod != jnod)
        {
          for (int d = 0; d < num_dim; ++d)
          {
            stiffness_matrix(jnod * nodedof + d, inod * nodedof + d) += tmp1;
            stiffness_matrix(jnod * nodedof + d, inod * nodedof + d + num_dim) += (tmp2 + tmp3);
            stiffness_matrix(jnod * nodedof + d + num_dim, inod * nodedof + d) += (tmp2 + tmp4);
            stiffness_matrix(jnod * nodedof + d + num_dim, inod * nodedof + d + num_dim) +=
                (tmp5 + tmp6);
          }
        }
      }
    }
  }

  /*!
   * @brief Adds the internal force vector contribution of one Gauss point
   *
   * @tparam distype : The discretization type known at compile time
   * @param Bop (in) : Strain gradient (B-Operator)
   * @param stress_resultant_vector (in) : Stress resultant vector (membrane forces and moments)
   * @param integration_fac (in) : Integration factor (Gauss point weight times the determinant
   * of the jacobian)
   * @param force_vector (in/out) : Force vector where the local contribution is added to
   */
  template <Core::FE::CellType distype>
  void add_internal_force_vector(const Core::LinAlg::SerialDenseMatrix& Bop,
      const Core::LinAlg::SerialDenseVector& stress_resultant_vector, const double integration_fac,
      Core::LinAlg::SerialDenseVector& force_vector)
  {
    Core::LinAlg::multiply_tn(1.0, force_vector, integration_fac, Bop, stress_resultant_vector);
  }
  /*!
   * @brief Adds mass matrix contribution of one Gauss point
   *
   * @tparam distype : The discretization type known at compile time
   * @param shapefunctions (in) : Shape functions and the first derivatives w.r.t spatial
   * coordinates evaluated at the respective point in the parameter space
   * @param mass_matrix_variables (in) : Thickness Integration factor for mass matrix
   * @param mass_matrix (in/out) : Mass matrix where the local contribution is added to
   */
  template <Core::FE::CellType distype>
  void add_mass_matrix(
      const Discret::Elements::Shell::ShapefunctionsAndDerivatives<distype>& shapefunctions,
      Discret::Elements::Shell::MassMatrixVariables& mass_matrix_variables, const double& thickness,
      Core::LinAlg::SerialDenseMatrix& massmatrix)
  {
    // half element thickness at gaussian point
    double half_thickness = 0.0;
    for (int i = 0; i < Internal::num_node<distype>; ++i)
      half_thickness += thickness * shapefunctions.shapefunctions_(i);
    half_thickness *= 0.5;

    const double squared_half_thickness = half_thickness * half_thickness;

    // integrate consistent mass matrix
    double massfactor;
    const int nodedof = Internal::node_dof;
    for (int inod = 0; inod < Internal::num_node<distype>; ++inod)
    {
      for (int jnod = 0; jnod < Internal::num_node<distype>; ++jnod)
      {
        massfactor = shapefunctions.shapefunctions_(inod) * shapefunctions.shapefunctions_(jnod);
        double massfactor_v = massfactor * mass_matrix_variables.factor_v_;

        massmatrix(nodedof * jnod + 0, nodedof * inod + 0) += massfactor_v;
        massmatrix(nodedof * jnod + 1, nodedof * inod + 1) += massfactor_v;
        massmatrix(nodedof * jnod + 2, nodedof * inod + 2) += massfactor_v;

        double massfactor_w = massfactor * mass_matrix_variables.factor_w_ * squared_half_thickness;
        massmatrix(nodedof * jnod + 3, nodedof * inod + 3) += massfactor_w;
        massmatrix(nodedof * jnod + 4, nodedof * inod + 4) += massfactor_w;
        massmatrix(nodedof * jnod + 5, nodedof * inod + 5) += massfactor_w;

        if (std::abs(mass_matrix_variables.factor_vw_) > 1.0e-14)
        {
          double massfactor_vw = massfactor * mass_matrix_variables.factor_vw_ * half_thickness;
          massmatrix(nodedof * jnod + 0, nodedof * inod + 3) += massfactor_vw;
          massmatrix(nodedof * jnod + 1, nodedof * inod + 4) += massfactor_vw;
          massmatrix(nodedof * jnod + 2, nodedof * inod + 5) += massfactor_vw;
          massmatrix(nodedof * jnod + 3, nodedof * inod + 0) += massfactor_vw;
          massmatrix(nodedof * jnod + 4, nodedof * inod + 1) += massfactor_vw;
          massmatrix(nodedof * jnod + 5, nodedof * inod + 2) += massfactor_vw;
        }
      }
    }
  }

  /*!
   * @brief Calls the @p gp_evaluator to evaluate the jacobian mapping at each integration point of
   * @p intpoints_ of the mid-surface of the shell.
   *
   * @tparam distype  : The discretization type known at compile time
   * @param nodal_coordinates (in) : The nodal coordinates of the element
   * @param intpoints_ (in) : Integratiopn points .
   * @param gp_evaluator (in) : A callable object (e.g. lambda-function) with signature void(const
   * std::vector<double>& xi_gp, const Shell::ShapefunctionsAndDerivatives<distype>&
   * shape_functions, Shell::BasisVectorsAndMetrics<distype>& a_current,
   * Shell::BasisVectorsAndMetrics<distype>& a_reference, double gpweight, double da, int gp) that
   * will be called for each integration point.
   */
  template <Core::FE::CellType distype, typename GaussPointEvaluator>
  inline void for_each_gauss_point(
      Discret::Elements::Shell::NodalCoordinates<distype>& nodal_coordinates,
      const Core::FE::IntegrationPoints2D& intpoints_, GaussPointEvaluator gp_evaluator)
  {
    for (int gp = 0; gp < intpoints_.num_points(); ++gp)
    {
      // get gauss points from integration rule
      std::array<double, 2> xi_gp;
      xi_gp[0] = intpoints_.qxg[gp][0];
      xi_gp[1] = intpoints_.qxg[gp][1];

      // get gauss weight at current gp
      double gpweight = intpoints_.qwgt[gp];

      // get shape functions and derivatives at gaussian points
      ShapefunctionsAndDerivatives<distype> shapefunctions =
          evaluate_shapefunctions_and_derivs<distype>(xi_gp);

      // basis vectors and metric on mid-surface
      BasisVectorsAndMetrics<distype> a_reference;
      BasisVectorsAndMetrics<distype> a_current;

      evaluate_metrics(shapefunctions, a_reference, a_current, nodal_coordinates, 0.0);

      // make h as cross product in ref configuration to get area da on shell mid-surface
      Core::LinAlg::Matrix<Internal::num_dim, 1> h(Core::LinAlg::Initialization::zero);
      {
        Core::LinAlg::Matrix<Internal::num_dim, Internal::num_dim> a_cov_refe =
            a_reference.covariant_;
        h(0) = a_cov_refe(0, 1) * a_cov_refe(1, 2) - a_cov_refe(0, 2) * a_cov_refe(1, 1);
        h(1) = a_cov_refe(0, 2) * a_cov_refe(1, 0) - a_cov_refe(0, 0) * a_cov_refe(1, 2);
        h(2) = a_cov_refe(0, 0) * a_cov_refe(1, 1) - a_cov_refe(0, 1) * a_cov_refe(1, 0);
      }
      // make director unit length and get mid-surface area da from it
      double da = h.norm2();

      gp_evaluator(xi_gp, shapefunctions, a_current, a_reference, gpweight, da, gp);
    }
  }

}  // namespace Discret::Elements::Shell

FOUR_C_NAMESPACE_CLOSE

#endif
