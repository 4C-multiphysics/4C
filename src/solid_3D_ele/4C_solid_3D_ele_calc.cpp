// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_3D_ele_calc.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_solid_3D_ele_calc_displacement_based.hpp"
#include "4C_solid_3D_ele_calc_displacement_based_linear_kinematics.hpp"
#include "4C_solid_3D_ele_calc_eas.hpp"
#include "4C_solid_3D_ele_calc_eas_helpers.hpp"
#include "4C_solid_3D_ele_calc_fbar.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_formulation.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"
#include "4C_solid_3D_ele_calc_lib_io.hpp"
#include "4C_solid_3D_ele_calc_lib_nitsche.hpp"
#include "4C_solid_3D_ele_calc_mulf.hpp"
#include "4C_solid_3D_ele_calc_mulf_fbar.hpp"
#include "4C_solid_3D_ele_calc_shell_ans.hpp"
#include "4C_solid_3D_ele_calc_shell_eas_ans.hpp"
#include "4C_solid_3D_ele_formulation.hpp"
#include "4C_solid_3D_ele_interface_serializable.hpp"
#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

#include <concepts>
#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <typename SolidFormulation>
  constexpr bool is_prestress_updatable = requires(SolidFormulation f) {
    {
      std::integral_constant<bool, SolidFormulation::is_prestress_updatable>()
    } -> std::same_as<std::integral_constant<bool, true>>;
  };

  template <typename T>
  T* get_ptr(std::optional<T>& opt)
  {
    return opt.has_value() ? &opt.value() : nullptr;
  }

  template <Core::FE::CellType celltype, typename SolidFormulation>
  double evaluate_cauchy_n_dir_at_xi(Mat::So3Material& mat,
      const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>&
          deformation_gradient,
      const Core::LinAlg::Tensor<double, 3>& n, const Core::LinAlg::Tensor<double, 3>& dir,
      int eleGID,
      const Discret::Elements::ElementFormulationDerivativeEvaluator<celltype, SolidFormulation>&
          evaluator,
      Discret::Elements::CauchyNDirLinearizations<3>& linearizations)
  {
    Discret::Elements::CauchyNDirLinearizationDependencies<celltype> linearization_dependencies =
        Discret::Elements::get_initialized_cauchy_n_dir_linearization_dependencies(
            evaluator, linearizations);

    double cauchy_n_dir = mat.evaluate_cauchy_n_dir_and_derivatives(deformation_gradient, n, dir,
        linearizations.d_cauchyndir_dn, linearizations.d_cauchyndir_ddir,
        get_ptr(linearization_dependencies.d_cauchyndir_dF),
        get_ptr(linearization_dependencies.d2_cauchyndir_dF2),
        get_ptr(linearization_dependencies.d2_cauchyndir_dF_dn),
        get_ptr(linearization_dependencies.d2_cauchyndir_dF_ddir), -1, eleGID, nullptr, nullptr,
        nullptr, nullptr);

    Discret::Elements::evaluate_cauchy_n_dir_linearizations<celltype>(
        linearization_dependencies, linearizations);

    return cauchy_n_dir;
  }
}  // namespace

template <Core::FE::CellType celltype, typename ElementFormulation>
Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::SolidEleCalc()
    : stiffness_matrix_integration_(Core::FE::create_gauss_integration<celltype>(
          get_gauss_rule_stiffness_matrix<celltype>())),
      mass_matrix_integration_(
          Core::FE::create_gauss_integration<celltype>(get_gauss_rule_mass_matrix<celltype>()))
{
  Discret::Elements::resize_gp_history(history_data_, stiffness_matrix_integration_.num_points());
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::pack(
    Core::Communication::PackBuffer& data) const
{
  Discret::Elements::pack<ElementFormulation>(data, history_data_);
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::unpack<ElementFormulation>(buffer, history_data_);
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype,
    ElementFormulation>::evaluate_nonlinear_force_stiffness_mass(const Core::Elements::Element& ele,
    Mat::So3Material& solid_material, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector* force_vector,
    Core::LinAlg::SerialDenseMatrix* stiffness_matrix, Core::LinAlg::SerialDenseMatrix* mass_matrix)
{
  // Create views to SerialDenseMatrices
  std::optional<Core::LinAlg::Matrix<num_dof_per_ele_, num_dof_per_ele_>> stiff{};
  std::optional<Core::LinAlg::Matrix<num_dof_per_ele_, num_dof_per_ele_>> mass{};
  std::optional<Core::LinAlg::Matrix<num_dof_per_ele_, 1>> force{};
  if (stiffness_matrix != nullptr) stiff.emplace(*stiffness_matrix, true);
  if (mass_matrix != nullptr) mass.emplace(*mass_matrix, true);
  if (force_vector != nullptr) force.emplace(*force_vector, true);

  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  bool equal_integration_mass_stiffness =
      compare_gauss_integration(mass_matrix_integration_, stiffness_matrix_integration_);

  evaluate_centroid_coordinates_and_add_to_parameter_list(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      prepare(ele, nodal_coordinates, history_data_);

  if constexpr (has_condensed_contribution<ElementFormulation>)
  {
    if (!ele.is_params_interface())
    {
      // This is noecessary only in the old time integration framework. In the new time integration
      // framework, condensed, internal variables are updated in recover()
      const double step_length = 1.0;
      update_condensed_variables(ele, nullptr, nodal_coordinates,
          get_displacement_increment<celltype>(discretization, lm), step_length, preparation_data,
          history_data_);
    }

    reset_condensed_variable_integration(ele, nodal_coordinates, preparation_data, history_data_);
  }

  // Check for negative Jacobian determinants
  ensure_positive_jacobian_determinant_at_element_nodes(nodal_coordinates);

  double element_mass = 0.0;
  double element_volume = 0.0;
  Discret::Elements::for_each_gauss_point(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        evaluate_gp_coordinates_and_add_to_parameter_list(
            nodal_coordinates, shape_functions, params);
        evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& deformation_gradient,
                const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& gl_strain,
                const auto& linearization)
            {
              const Stress<celltype> stress = evaluate_material_stress<celltype>(
                  solid_material, deformation_gradient, gl_strain, params, gp, ele.id());

              if constexpr (has_condensed_contribution<ElementFormulation>)
              {
                integrate_condensed_contribution(
                    linearization, stress, integration_factor, preparation_data, history_data_, gp);
              }

              if (force.has_value())
              {
                Discret::Elements::add_internal_force_vector<ElementFormulation, celltype>(
                    jacobian_mapping, deformation_gradient, linearization, stress,
                    integration_factor, preparation_data, history_data_, gp, *force);
              }

              if (stiff.has_value())
              {
                add_stiffness_matrix<ElementFormulation, celltype>(jacobian_mapping,
                    deformation_gradient, xi, shape_functions, linearization, stress,
                    integration_factor, preparation_data, history_data_, gp, *stiff);
              }

              if (mass.has_value())
              {
                if (equal_integration_mass_stiffness)
                {
                  add_mass_matrix(
                      shape_functions, integration_factor, solid_material.density(gp), *mass);
                }
                else
                {
                  element_mass += solid_material.density(gp) * integration_factor;
                  element_volume += integration_factor;
                }
              }
            });
      });

  if constexpr (has_condensed_contribution<ElementFormulation>)
  {
    const auto condensed_contribution_data =
        prepare_condensed_contribution(preparation_data, history_data_);

    if (force.has_value())
    {
      add_condensed_contribution_to_force_vector<celltype>(
          condensed_contribution_data, preparation_data, history_data_, *force);
    }

    if (stiff.has_value())
    {
      add_condensed_contribution_to_stiffness_matrix<celltype>(
          condensed_contribution_data, preparation_data, history_data_, *stiff);
    }
  }

  if (mass.has_value() && !equal_integration_mass_stiffness)
  {
    // integrate mass matrix
    FOUR_C_ASSERT(element_mass > 0, "It looks like the element mass is 0.0");
    Discret::Elements::for_each_gauss_point<celltype>(nodal_coordinates, mass_matrix_integration_,
        [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
            const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
            const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
        {
          add_mass_matrix(
              shape_functions, integration_factor, element_mass / element_volume, *mass);
        });
  }
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::recover(
    Core::Elements::Element& ele, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params)
{
  if constexpr (has_condensed_contribution<ElementFormulation>)
  {
    FourC::Solid::Elements::ParamsInterface& params_interface =
        *std::dynamic_pointer_cast<FourC::Solid::Elements::ParamsInterface>(
            ele.params_interface_ptr());

    const double step_length = params_interface.get_step_length();

    const ElementNodes<celltype> element_nodes =
        evaluate_element_nodes<celltype>(ele, discretization, lm);

    const PreparationData<ElementFormulation> preparation_data =
        prepare(ele, element_nodes, history_data_);

    if (params_interface.is_default_step())
    {
      update_condensed_variables(ele, &params_interface, element_nodes,
          get_displacement_increment<celltype>(discretization, lm), step_length, preparation_data,
          history_data_);
    }
    else
    {
      correct_condensed_variables_for_linesearch(
          ele, &params_interface, step_length, preparation_data, history_data_);
    }
  }
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::update(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  old_history_data_ = history_data_;

  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  evaluate_centroid_coordinates_and_add_to_parameter_list(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      prepare(ele, nodal_coordinates, history_data_);

  Discret::Elements::for_each_gauss_point(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Tensor<double, Internal::num_dim<celltype>>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        evaluate_gp_coordinates_and_add_to_parameter_list(
            nodal_coordinates, shape_functions, params);
        evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& deformation_gradient,
                const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& gl_strain,
                const auto& linearization)
            { solid_material.update(deformation_gradient, gp, params, ele.id()); });
      });

  solid_material.update();
}

template <Core::FE::CellType celltype, typename ElementFormulation>
double Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::calculate_internal_energy(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  evaluate_centroid_coordinates_and_add_to_parameter_list(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      prepare(ele, nodal_coordinates, history_data_);

  double intenergy = 0;
  Discret::Elements::for_each_gauss_point(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        evaluate_gp_coordinates_and_add_to_parameter_list(
            nodal_coordinates, shape_functions, params);
        evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& deformation_gradient,
                const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& gl_strain,
                const auto& linearization)
            {
              double psi = solid_material.strain_energy(gl_strain, gp, ele.id());
              intenergy += psi * integration_factor;
            });
      });

  return intenergy;
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::calculate_stress(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material, const StressIO& stressIO,
    const StrainIO& strainIO, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Teuchos::ParameterList& params)
{
  std::vector<char>& serialized_stress_data = stressIO.mutable_data;
  std::vector<char>& serialized_strain_data = strainIO.mutable_data;
  Core::LinAlg::SerialDenseMatrix stress_data(stiffness_matrix_integration_.num_points(), num_str_);
  Core::LinAlg::SerialDenseMatrix strain_data(stiffness_matrix_integration_.num_points(), num_str_);

  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  evaluate_centroid_coordinates_and_add_to_parameter_list(nodal_coordinates, params);

  const PreparationData<ElementFormulation> preparation_data =
      prepare(ele, nodal_coordinates, history_data_);

  Discret::Elements::for_each_gauss_point(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      {
        evaluate_gp_coordinates_and_add_to_parameter_list(
            nodal_coordinates, shape_functions, params);
        evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
            history_data_, gp,
            [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& deformation_gradient,
                const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
                    Core::FE::dim<celltype>>& gl_strain,
                const auto& linearization)
            {
              const Stress<celltype> stress = evaluate_material_stress<celltype>(
                  solid_material, deformation_gradient, gl_strain, params, gp, ele.id());

              assemble_strain_type_to_matrix_row<celltype>(
                  gl_strain, deformation_gradient, strainIO.type, strain_data, gp);
              assemble_stress_type_to_matrix_row(
                  deformation_gradient, stress, stressIO.type, stress_data, gp);
            });
      });

  serialize(stress_data, serialized_stress_data);
  serialize(strain_data, serialized_strain_data);
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::update_prestress(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params)
{
  if constexpr (is_prestress_updatable<ElementFormulation>)
  {
    const ElementNodes<celltype> nodal_coordinates =
        evaluate_element_nodes<celltype>(ele, discretization, lm);

    const PreparationData<ElementFormulation> preparation_data =
        prepare(ele, nodal_coordinates, history_data_);

    Discret::Elements::update_prestress<ElementFormulation, celltype>(
        ele, nodal_coordinates, preparation_data, history_data_);

    Discret::Elements::for_each_gauss_point(nodal_coordinates, stiffness_matrix_integration_,
        [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
            const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
            const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
        {
          evaluate_gp_coordinates_and_add_to_parameter_list(
              nodal_coordinates, shape_functions, params);
          evaluate(ele, nodal_coordinates, xi, shape_functions, jacobian_mapping, preparation_data,
              history_data_, gp,
              [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>,
                      Core::FE::dim<celltype>>& deformation_gradient,
                  const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
                      Core::FE::dim<celltype>>& gl_strain,
                  const auto& linearization)
              {
                Discret::Elements::update_prestress<ElementFormulation, celltype>(ele,
                    nodal_coordinates, xi, shape_functions, jacobian_mapping, deformation_gradient,
                    preparation_data, history_data_, gp);
              });
        });
  }
  else
  {
    FOUR_C_THROW("The solid formulation does not support to update the prestress!");
  }
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::setup(
    Mat::So3Material& solid_material, const Core::IO::InputParameterContainer& container)
{
  solid_material.setup(stiffness_matrix_integration_.num_points(), container);
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::material_post_setup(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  Teuchos::ParameterList params{};

  // Check if element has fiber nodes, if so interpolate fibers to Gauss Points and add to params
  interpolate_fibers_to_gauss_points_and_add_to_parameter_list<celltype>(
      stiffness_matrix_integration_, ele, params);

  // Call post_setup of material
  solid_material.post_setup(params, ele.id());
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype,
    ElementFormulation>::initialize_gauss_point_data_output(const Core::Elements::Element& ele,
    const Mat::So3Material& solid_material,
    FourC::Solid::ModelEvaluator::GaussPointDataOutputManager& gp_data_output_manager) const
{
  FOUR_C_ASSERT(ele.is_params_interface(),
      "This action type should only be called from the new time integration framework!");

  ask_and_add_quantities_to_gauss_point_data_output(
      stiffness_matrix_integration_.num_points(), solid_material, gp_data_output_manager);
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype,
    ElementFormulation>::evaluate_gauss_point_data_output(const Core::Elements::Element& ele,
    const Mat::So3Material& solid_material,
    FourC::Solid::ModelEvaluator::GaussPointDataOutputManager& gp_data_output_manager) const
{
  FOUR_C_ASSERT(ele.is_params_interface(),
      "This action type should only be called from the new time integration framework!");

  collect_and_assemble_gauss_point_data_output<celltype>(
      stiffness_matrix_integration_, solid_material, ele, gp_data_output_manager);
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::reset_to_last_converged(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material)
{
  history_data_ = old_history_data_;
  solid_material.reset_step();
}

template <Core::FE::CellType celltype, typename ElementFormulation>
double
Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::get_normal_cauchy_stress_at_xi(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const std::vector<double>& disp, const Core::LinAlg::Tensor<double, 3>& xi,
    const Core::LinAlg::Tensor<double, 3>& n, const Core::LinAlg::Tensor<double, 3>& dir,
    CauchyNDirLinearizations<3>& linearizations)
{
  if constexpr (has_gauss_point_history<ElementFormulation>)
  {
    FOUR_C_THROW(
        "Cannot evaluate the Cauchy stress at xi with an element formulation with Gauss point "
        "history. The element formulation is {}.",
        Core::Utils::get_type_name<ElementFormulation>().c_str());
  }
  else
  {
    ElementNodes<celltype> element_nodes = evaluate_element_nodes<celltype>(ele, disp);

    const ShapeFunctionsAndDerivatives<celltype> shape_functions =
        evaluate_shape_functions_and_derivs<celltype>(xi, element_nodes);

    const JacobianMapping<celltype> jacobian_mapping =
        evaluate_jacobian_mapping(shape_functions, element_nodes);

    const PreparationData<ElementFormulation> preparation_data =
        prepare(ele, element_nodes, history_data_);

    return evaluate(ele, element_nodes, xi, shape_functions, jacobian_mapping, preparation_data,
        history_data_,
        [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>&
                deformation_gradient,
            const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>,
                Core::FE::dim<celltype>>& gl_strain,
            const auto& linearization)
        {
          const ElementFormulationDerivativeEvaluator<celltype, ElementFormulation> evaluator(ele,
              element_nodes, xi, shape_functions, jacobian_mapping, deformation_gradient,
              preparation_data, history_data_);

          return evaluate_cauchy_n_dir_at_xi<celltype>(
              solid_material, deformation_gradient, n, dir, ele.id(), evaluator, linearizations);
        });
  }
}

template <Core::FE::CellType celltype, typename ElementFormulation>
void Discret::Elements::SolidEleCalc<celltype, ElementFormulation>::for_each_gauss_point(
    const Core::Elements::Element& ele, Mat::So3Material& solid_material,
    const Core::FE::Discretization& discretization, const std::vector<int>& lm,
    const std::function<void(Mat::So3Material&, double, int)>& integrator) const
{
  const ElementNodes<celltype> nodal_coordinates =
      evaluate_element_nodes<celltype>(ele, discretization, lm);

  Discret::Elements::for_each_gauss_point(nodal_coordinates, stiffness_matrix_integration_,
      [&](const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>>& xi,
          const ShapeFunctionsAndDerivatives<celltype>& shape_functions,
          const JacobianMapping<celltype>& jacobian_mapping, double integration_factor, int gp)
      { integrator(solid_material, integration_factor, gp); });
}

template <Core::FE::CellType... celltypes>
struct VerifyPackable
{
  static constexpr bool are_all_packable =
      (Core::Communication::Packable<Discret::Elements::SolidEleCalc<celltypes,
              Discret::Elements::DisplacementBasedFormulation<celltypes>>> &&
          ...);

  static constexpr bool are_all_unpackable =
      (Core::Communication::Unpackable<Discret::Elements::SolidEleCalc<celltypes,
              Discret::Elements::DisplacementBasedFormulation<celltypes>>> &&
          ...);

  void static_asserts() const
  {
    static_assert(are_all_packable);
    static_assert(are_all_unpackable);
  }
};

template struct VerifyPackable<Core::FE::CellType::hex8, Core::FE::CellType::hex18,
    Core::FE::CellType::hex20, Core::FE::CellType::hex27, Core::FE::CellType::nurbs27,
    Core::FE::CellType::tet4, Core::FE::CellType::tet10, Core::FE::CellType::pyramid5,
    Core::FE::CellType::wedge6, Core::FE::CellType::hex8, Core::FE::CellType::hex8,
    Core::FE::CellType::hex8, Core::FE::CellType::hex8, Core::FE::CellType::hex8>;

// explicit instantiations of template classes
// for displacement based formulation
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::hex8>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex18,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::hex18>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex20,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::hex20>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex27,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::hex27>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::nurbs27,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::nurbs27>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::tet4,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::tet4>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::tet10,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::tet10>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::pyramid5,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::pyramid5>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::wedge6,
    Discret::Elements::DisplacementBasedFormulation<Core::FE::CellType::wedge6>>;

// for displacement based formulation with linear kinematics
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::hex8>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex18,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::hex18>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex20,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::hex20>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex27,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::hex27>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::nurbs27,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::nurbs27>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::tet4,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::tet4>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::tet10,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::tet10>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::pyramid5,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::pyramid5>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::wedge6,
    Discret::Elements::DisplacementBasedLinearKinematicsFormulation<Core::FE::CellType::wedge6>>;

// Fbar element technology
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::FBarFormulation<Core::FE::CellType::hex8>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::pyramid5,
    Discret::Elements::FBarFormulation<Core::FE::CellType::pyramid5>>;

// explicit instantiations for MULF
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::MulfFormulation<Core::FE::CellType::hex8>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex18,
    Discret::Elements::MulfFormulation<Core::FE::CellType::hex18>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex20,
    Discret::Elements::MulfFormulation<Core::FE::CellType::hex20>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex27,
    Discret::Elements::MulfFormulation<Core::FE::CellType::hex27>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::nurbs27,
    Discret::Elements::MulfFormulation<Core::FE::CellType::nurbs27>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::tet4,
    Discret::Elements::MulfFormulation<Core::FE::CellType::tet4>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::tet10,
    Discret::Elements::MulfFormulation<Core::FE::CellType::tet10>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::pyramid5,
    Discret::Elements::MulfFormulation<Core::FE::CellType::pyramid5>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::wedge6,
    Discret::Elements::MulfFormulation<Core::FE::CellType::wedge6>>;

// explicit instaniations for FBAR+MULF
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::MulfFBarFormulation<Core::FE::CellType::hex8>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::pyramid5,
    Discret::Elements::MulfFBarFormulation<Core::FE::CellType::pyramid5>>;

// explicit instantiations for shell_ans
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::ShellANSFormulation<Core::FE::CellType::hex8>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::wedge6,
    Discret::Elements::ShellANSFormulation<Core::FE::CellType::wedge6>>;

// explicit instantiations for shell_eas_ans
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::ShellEASANSFormulation<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_sh8_7>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::wedge6,
    Discret::Elements::ShellEASANSFormulation<Core::FE::CellType::wedge6,
        Discret::Elements::EasType::eastype_sw6_1>>;

// explicit instantiations for hex8 with EAS
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::EASFormulation<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_9, Inpar::Solid::KinemType::nonlinearTotLag>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::EASFormulation<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_21, Inpar::Solid::KinemType::nonlinearTotLag>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::EASFormulation<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_sh8_7, Inpar::Solid::KinemType::nonlinearTotLag>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::EASFormulation<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_9, Inpar::Solid::KinemType::linear>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::EASFormulation<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_h8_21, Inpar::Solid::KinemType::linear>>;
template class Discret::Elements::SolidEleCalc<Core::FE::CellType::hex8,
    Discret::Elements::EASFormulation<Core::FE::CellType::hex8,
        Discret::Elements::EasType::eastype_sh8_7, Inpar::Solid::KinemType::linear>>;

FOUR_C_NAMESPACE_CLOSE