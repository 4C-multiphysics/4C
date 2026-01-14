// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_CALC_LIB_NITSCHE_HPP
#define FOUR_C_SOLID_3D_ELE_CALC_LIB_NITSCHE_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_solid_3D_ele_calc_lib_formulation.hpp"
#include "4C_utils_exceptions.hpp"

#include <optional>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class So3Material;
}

namespace Discret::Elements
{
  namespace Internal
  {
    template <unsigned a, unsigned b, unsigned c>
    static inline void multiply_tn(const Core::LinAlg::Matrix<a, b>& mat1,
        const Core::LinAlg::Matrix<a, c>& mat2, Core::LinAlg::SerialDenseMatrix& out)
    {
      out.reshape(b, c);
      Core::LinAlg::Matrix<b, c> out_mat(out.values(), true);

      out_mat.multiply_tn(mat1, mat2);
    }
  }  // namespace Internal

  /*!
   * @brief Evaluate the derivative of the Cauchy stress w.r.t. the nodal displacements
   *
   * @tparam celltype
   * @param d_F_dd (in) : derivative of the deformation gradient w.r.t. nodal displacements
   * @param d_cauchyndir_dF (in) : derivative of the cauchy stress w.r.t. deformation gradient
   * @param d_cauchyndir_dd (out) : result
   */
  template <Core::FE::CellType celltype>
  void evaluate_d_cauchy_n_dir_d_displacements(
      const Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>&
          d_F_dd,
      const Core::LinAlg::Matrix<9, 1>& d_cauchyndir_dF,
      Core::LinAlg::SerialDenseMatrix& d_cauchyndir_dd)
  {
    Internal::multiply_tn(d_F_dd, d_cauchyndir_dF, d_cauchyndir_dd);
  }

  /*!
   * @brief Evaluate the 2. derivative of the Cauchy stress w.r.t. the nodal displacements and the
   * normal
   *
   * @tparam celltype
   * @param d_F_dd (in) : derivative of the deformation gradient w.r.t. nodal displacements
   * @param d2_cauchyndir_dF_dn (in) : second derivative of the Cauchy stress w.r.t. the deformation
   * gradient and the normal
   * @param d2_cauchyndir_dd_dn (out) : result
   */
  template <Core::FE::CellType celltype>
  void evaluate_d2_cauchy_n_dir_d_displacements_d_normal(
      const Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>&
          d_F_dd,
      const Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>& d2_cauchyndir_dF_dn,
      Core::LinAlg::SerialDenseMatrix& d2_cauchyndir_dd_dn)
  {
    Internal::multiply_tn(d_F_dd, d2_cauchyndir_dF_dn, d2_cauchyndir_dd_dn);
  }

  /*!
   * @brief Evaluate the 2. derivative of the Cauchy stress w.r.t. the nodal displacements and the
   * direction
   *
   * @tparam celltype
   * @param d_F_dd (in) : derivative of the deformation gradient w.r.t. nodal displacements
   * @param d2_cauchyndir_dF_ddir (in) : second derivative of the Cauchy stress w.r.t. the
   * deformation gradient and the direction
   * @param d2_cauchyndir_dd_ddir (out) : result
   */
  template <Core::FE::CellType celltype>
  void evaluate_d2_cauchy_n_dir_d_displacements_d_dir(
      const Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>&
          d_F_dd,
      const Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>& d2_cauchyndir_dF_ddir,
      Core::LinAlg::SerialDenseMatrix& d2_cauchyndir_dd_ddir)
  {
    Internal::multiply_tn(d_F_dd, d2_cauchyndir_dF_ddir, d2_cauchyndir_dd_ddir);
  }

  /*!
   * @brief Evaluate the 2. derivative of the Cauchy stress w.r.t. the nodal displacements
   *
   * @tparam celltype
   * @param d_F_dd (in) : derivative of the deformation gradient w.r.t. nodal displacements
   * @param d2_cauchyndir_dF2 (in) : second derivative of the Cauchy stress w.r.t. the
   * deformation gradient
   * @param d2_cauchyndir_dd_dd (out) : result
   */
  template <Core::FE::CellType celltype>
  void evaluate_d2_cauchy_n_dir_d_displacements2(
      const Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>&
          d_F_dd,
      const Core::LinAlg::Matrix<9, 9>& d2_cauchyndir_dF2,
      Core::LinAlg::SerialDenseMatrix& d2_cauchyndir_dd_dd)
  {
    Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>
        d2_cauchyndir_dF_2d_F_dd(Core::LinAlg::Initialization::uninitialized);
    d2_cauchyndir_dF_2d_F_dd.multiply(d2_cauchyndir_dF2, d_F_dd);
    Internal::multiply_tn(d_F_dd, d2_cauchyndir_dF_2d_F_dd, d2_cauchyndir_dd_dd);
  }

  /*!
   * @brief Evaluate the derivateive of the Cauchy stress w.r.t. xi
   *
   * @tparam celltype
   * @param d_F_dxi  (in) : derivative of the deformation gradient w.r.t. xi
   * @param d_cauchyndir_dF (in) : derivative of the Cauchy stress w.r.t. deformation gradient
   * @param d_cauchyndir_dxi (out) : result
   */
  template <Core::FE::CellType celltype>
  void evaluate_d_cauchy_n_dir_d_xi(const Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>& d_F_dxi,
      const Core::LinAlg::Matrix<9, 1>& d_cauchyndir_dF,
      Core::LinAlg::Matrix<3, 1>& d_cauchyndir_dxi)
  {
    d_cauchyndir_dxi.multiply_tn(d_F_dxi, d_cauchyndir_dF);
  }

  /*!
   * @brief Evaluate the second derivative of the Cauchy stress w.r.t. the nodal displacements and
   * xi
   *
   * @tparam celltype
   * @param d2_F_dxi_dd (in) : second derivative of the deformation gradient w.r.t. xi and the nodal
   * displacements
   * @param d_cauchyndir_dF (in) : derivative of the Cauchy stress w.r.t. deformation gradient
   * @param d2_cauchyndir_dd_dxi (out) : result
   */
  template <Core::FE::CellType celltype>
  void evaluate_d2_cauchy_n_dir_d_displacements_d_xi(
      const Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype> *
                                        Core::FE::dim<celltype>>& d2_F_dxi_dd,
      const Core::LinAlg::Matrix<9, 1>& d_cauchyndir_dF,
      Core::LinAlg::SerialDenseMatrix& d2_cauchyndir_dd_dxi)
  {
    constexpr int num_dof = Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>;

    d2_cauchyndir_dd_dxi.reshape(num_dof, Core::FE::dim<celltype>);

    for (int i = 0; i < Core::FE::dim<celltype>; ++i)
    {
      for (int j = 0; j < Core::FE::dim<celltype>; ++j)
      {
        for (int k = 0; k < Core::FE::num_nodes(celltype); ++k)
        {
          for (int l = 0; l < Core::FE::dim<celltype>; ++l)
          {
            using VoigtMapping = Core::LinAlg::Voigt::IndexMappings;
            d2_cauchyndir_dd_dxi(k * 3 + i, l) +=
                d_cauchyndir_dF(VoigtMapping::non_symmetric_tensor_to_voigt9_index(i, j), 0) *
                d2_F_dxi_dd(VoigtMapping::non_symmetric_tensor_to_voigt9_index(i, j),
                    Core::FE::dim<celltype> * (Core::FE::dim<celltype> * k + i) + l);
          }
        }
      }
    }
  }

  /*!
   * @brief A struct holding the Cauchy stress at a point xi in a specific direction and the
   * derivatives w.r.t. all input quantities
   *
   * @tparam dim
   */
  template <int dim>
  struct CauchyNDirLinearizations
  {
    /// @brief first derivatives
    ///@{

    /// first derivative w.r.t. normal
    Core::LinAlg::Matrix<dim, 1>* d_cauchyndir_dn = nullptr;

    /// first derivative w.r.t. given direction
    Core::LinAlg::Matrix<dim, 1>* d_cauchyndir_ddir = nullptr;

    /// first derivative w.r.t. xi
    Core::LinAlg::Matrix<dim, 1>* d_cauchyndir_dxi = nullptr;

    /// first derivative w.r.t. displacements
    Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dd = nullptr;
    ///@}

    /// @brief second derivatives
    ///@{

    /// second derivative w.r.t. displacements
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd2 = nullptr;

    /// second derivative w.r.t. displacements, normal
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd_dn = nullptr;

    /// second derivative w.r.t. displacements, given direction
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd_ddir = nullptr;

    /// second derivative w.r.t. displacements, xi
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd_dxi = nullptr;
    ///@}
  };

  /*!
   * @brief Struct holding dependencies needed to evalaute the linearizations of the Cauchy stress
   * at xi
   *
   * @tparam celltype
   */
  template <Core::FE::CellType celltype>
  struct CauchyNDirLinearizationDependencies
  {
    std::optional<Core::LinAlg::Matrix<9, Core::FE::num_nodes(celltype) * Core::FE::dim<celltype>>>
        d_F_dd{};
    std::optional<Core::LinAlg::Matrix<9, Core::FE::dim<celltype>>> d_F_dxi{};
    std::optional<Core::LinAlg::Matrix<9, 1>> d_cauchyndir_dF{};
    std::optional<Core::LinAlg::Matrix<9, 9>> d2_cauchyndir_dF2{};
    std::optional<Core::LinAlg::Matrix<9, 3>> d2_cauchyndir_dF_ddir{};
    std::optional<Core::LinAlg::Matrix<9, 3>> d2_cauchyndir_dF_dn{};
    std::optional<Core::LinAlg::Matrix<9,
        Core::FE::num_nodes(celltype) * Core::FE::dim<celltype> * Core::FE::dim<celltype>>>
        d2_F_dxi_dd{};
  };


  namespace Internal
  {
    template <typename T, typename... Args>
    std::optional<T> make_optional_if(bool condition, Args&&... params)
    {
      if (condition) return std::optional<T>(std::forward<Args>(params)...);
      return std::nullopt;
    }
  }  // namespace Internal

  /*!
   * @brief Returns the tensor-dependencies with those values initialized that are needed for the
   * evaluation of the requested linearizations
   *
   * @tparam celltype
   * @tparam SolidFormulation
   * @param evaluator
   * @param linearizations (in) : requested linearizations
   * @return CauchyNDirLinearizationDependencies<celltype>
   */
  template <Core::FE::CellType celltype, typename SolidFormulation>
  CauchyNDirLinearizationDependencies<celltype>
  get_initialized_cauchy_n_dir_linearization_dependencies(
      const Discret::Elements::ElementFormulationDerivativeEvaluator<celltype, SolidFormulation>&
          evaluator,
      Discret::Elements::CauchyNDirLinearizations<3>& linearizations)
  {
    CauchyNDirLinearizationDependencies<celltype> linearization_dependencies{};
    linearization_dependencies.d_cauchyndir_dF =
        Internal::make_optional_if<Core::LinAlg::Matrix<9, 1>>(
            linearizations.d_cauchyndir_dd || linearizations.d_cauchyndir_dxi ||
                linearizations.d2_cauchyndir_dd_dxi,
            Core::LinAlg::Initialization::zero);

    linearization_dependencies.d2_cauchyndir_dF2 =
        Internal::make_optional_if<Core::LinAlg::Matrix<9, 9>>(
            linearizations.d2_cauchyndir_dd2, Core::LinAlg::Initialization::zero);

    linearization_dependencies.d2_cauchyndir_dF_dn =
        Internal::make_optional_if<Core::LinAlg::Matrix<9, 3>>(
            linearizations.d2_cauchyndir_dd_dn, Core::LinAlg::Initialization::zero);

    linearization_dependencies.d2_cauchyndir_dF_ddir =
        Internal::make_optional_if<Core::LinAlg::Matrix<9, 3>>(
            linearizations.d2_cauchyndir_dd_ddir, Core::LinAlg::Initialization::zero);


    if (linearizations.d_cauchyndir_dd || linearizations.d2_cauchyndir_dd_dn ||
        linearizations.d2_cauchyndir_dd_ddir || linearizations.d2_cauchyndir_dd2)
    {
      linearization_dependencies.d_F_dd.emplace(
          evaluator.evaluate_d_deformation_gradient_d_displacements());
    }

    if (linearizations.d_cauchyndir_dxi)
    {
      linearization_dependencies.d_F_dxi.emplace(evaluator.evaluate_d_deformation_gradient_d_xi());
    }

    if (linearizations.d2_cauchyndir_dd_dxi)
    {
      linearization_dependencies.d2_F_dxi_dd.emplace(
          evaluator.evaluate_d2_deformation_gradient_d_displacements_d_xi());
    }

    return linearization_dependencies;
  }

  /*!
   * @brief Evaluate the requested linearizations
   *
   * @tparam celltype
   * @param linearization_dependencies (in) : tensor-dependencies needed for the evaluation
   * @param linearizations (out) : Linearizations
   */
  template <Core::FE::CellType celltype>
  void evaluate_cauchy_n_dir_linearizations(
      const CauchyNDirLinearizationDependencies<celltype>& linearization_dependencies,
      Discret::Elements::CauchyNDirLinearizations<3>& linearizations)
  {
    // evaluate first derivative w.r.t. displacements
    if (linearizations.d_cauchyndir_dd)
    {
      FOUR_C_ASSERT(linearization_dependencies.d_F_dd && linearization_dependencies.d_cauchyndir_dF,
          "Not all tensors are computed!");
      Discret::Elements::evaluate_d_cauchy_n_dir_d_displacements<celltype>(
          *linearization_dependencies.d_F_dd, *linearization_dependencies.d_cauchyndir_dF,
          *linearizations.d_cauchyndir_dd);
    }


    // evaluate second derivative w.r.t. displacements, normal
    if (linearizations.d2_cauchyndir_dd_dn)
    {
      FOUR_C_ASSERT(
          linearization_dependencies.d_F_dd && linearization_dependencies.d2_cauchyndir_dF_dn,
          "Not all tensors are computed!");
      Discret::Elements::evaluate_d2_cauchy_n_dir_d_displacements_d_normal<celltype>(
          *linearization_dependencies.d_F_dd, *linearization_dependencies.d2_cauchyndir_dF_dn,
          *linearizations.d2_cauchyndir_dd_dn);
    }

    // evaluate second derivative w.r.t. displacements, direction
    if (linearizations.d2_cauchyndir_dd_ddir)
    {
      FOUR_C_ASSERT(
          linearization_dependencies.d_F_dd && linearization_dependencies.d2_cauchyndir_dF_ddir,
          "Not all tensors are computed!");
      Discret::Elements::evaluate_d2_cauchy_n_dir_d_displacements_d_dir<celltype>(
          *linearization_dependencies.d_F_dd, *linearization_dependencies.d2_cauchyndir_dF_ddir,
          *linearizations.d2_cauchyndir_dd_ddir);
    }

    // evaluate second derivative w.r.t. displacements, displacements
    if (linearizations.d2_cauchyndir_dd2)
    {
      FOUR_C_ASSERT(
          linearization_dependencies.d_F_dd && linearization_dependencies.d2_cauchyndir_dF2,
          "Not all tensors are computed!");
      Discret::Elements::evaluate_d2_cauchy_n_dir_d_displacements2<celltype>(
          *linearization_dependencies.d_F_dd, *linearization_dependencies.d2_cauchyndir_dF2,
          *linearizations.d2_cauchyndir_dd2);
    }

    // evaluate first derivative w.r.t. xi
    if (linearizations.d_cauchyndir_dxi)
    {
      FOUR_C_ASSERT(
          linearization_dependencies.d_F_dxi && linearization_dependencies.d_cauchyndir_dF,
          "Not all tensors are computed!");
      Discret::Elements::evaluate_d_cauchy_n_dir_d_xi<celltype>(*linearization_dependencies.d_F_dxi,
          *linearization_dependencies.d_cauchyndir_dF, *linearizations.d_cauchyndir_dxi);
    }

    // evaluate second derivative w.r.t. displacements, xi
    if (linearizations.d2_cauchyndir_dd_dxi)
    {
      FOUR_C_ASSERT(
          linearization_dependencies.d2_F_dxi_dd && linearization_dependencies.d_cauchyndir_dF,
          "Not all tensors are computed!");
      Discret::Elements::evaluate_d2_cauchy_n_dir_d_displacements_d_xi<celltype>(
          *linearization_dependencies.d2_F_dxi_dd, *linearization_dependencies.d_cauchyndir_dF,
          *linearizations.d2_cauchyndir_dd_dxi);
    }
  }

  /// Check whether a solid variant can evaluate the Cauchy stress at xi in a specific direction
  /// including the derivatives
  template <typename T, int dim, typename AlwaysVoid = void>
  constexpr bool can_evaluate_cauchy_n_dir = false;

  template <typename T, int dim>
  constexpr bool can_evaluate_cauchy_n_dir<T, dim,
      std::void_t<decltype(std::declval<T>()->get_normal_cauchy_stress_at_xi(
          std::declval<const Core::Elements::Element&>(), std::declval<Mat::So3Material&>(),
          std::declval<const std::vector<double>&>(),
          std::declval<const Core::LinAlg::Tensor<double, dim>&>(),
          std::declval<const Core::LinAlg::Tensor<double, dim>&>(),
          std::declval<const Core::LinAlg::Tensor<double, dim>&>(),
          std::declval<CauchyNDirLinearizations<dim>&>(),
          std::declval<const Core::FE::Discretization*>()))>> = true;

  namespace Internal
  {
    template <int dim>
    struct EvaluateCauchyNDirAction
    {
      EvaluateCauchyNDirAction(const Core::Elements::Element& e, Mat::So3Material& m,
          const std::vector<double>& d, const Core::LinAlg::Tensor<double, dim>& x,
          const Core::LinAlg::Tensor<double, dim>& normal,
          const Core::LinAlg::Tensor<double, dim>& direction, CauchyNDirLinearizations<dim>& lins,
          const Core::FE::Discretization* dscrt = nullptr)
          : element(e),
            mat(m),
            disp(d),
            xi(x),
            n(normal),
            dir(direction),
            linearizations(lins),
            discret(dscrt)
      {
      }

      template <typename T>
      double operator()(T& cauchy_n_dir_evaluatable)
        requires(can_evaluate_cauchy_n_dir<T&, dim>)
      {
        return cauchy_n_dir_evaluatable->get_normal_cauchy_stress_at_xi(
            element, mat, disp, xi, n, dir, linearizations, discret);
      }

      template <typename T>
      double operator()(T& other)
        requires(!can_evaluate_cauchy_n_dir<T&, dim>)
      {
        FOUR_C_THROW(
            "Your element evaluation {} does not allow to evaluate the Cauchy stress at a specific "
            "point in a specific direction in the dimension dim={}.",
            typeid(T).name(), dim);
      }

      const Core::Elements::Element& element;
      Mat::So3Material& mat;
      const std::vector<double>& disp;
      const Core::LinAlg::Tensor<double, dim>& xi;
      const Core::LinAlg::Tensor<double, dim>& n;
      const Core::LinAlg::Tensor<double, dim>& dir;
      CauchyNDirLinearizations<dim>& linearizations;
      const Core::FE::Discretization* discret;
    };
  }  // namespace Internal

  /*!
   * @brief Evaluate optional function @p get_normal_cauchy_stress_at_xi(....) on SolidVariant @p
   * variant. If the function does not exist, an error will be thrown.
   *
   * @return double
   */
  template <typename VariantType>
  double get_normal_cauchy_stress_at_xi(VariantType& variant,
      const Core::Elements::Element& element, Mat::So3Material& mat,
      const std::vector<double>& disp, const Core::LinAlg::Tensor<double, 3>& xi,
      const Core::LinAlg::Tensor<double, 3>& n, const Core::LinAlg::Tensor<double, 3>& dir,
      CauchyNDirLinearizations<3>& linearizations, const Core::FE::Discretization* discretization)
  {
    return std::visit(Internal::EvaluateCauchyNDirAction<3>(
                          element, mat, disp, xi, n, dir, linearizations, discretization),
        variant);
  }
}  // namespace Discret::Elements
FOUR_C_NAMESPACE_CLOSE

#endif