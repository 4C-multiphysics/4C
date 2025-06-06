// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_legacy_enum_definitions_element_actions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_solid_3D_ele_calc_interface.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_io.hpp"
#include "4C_solid_3D_ele_calc_lib_nitsche.hpp"
#include "4C_solid_3D_ele_calc_mulf.hpp"
#include "4C_solid_3D_ele_neumann_evaluator.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::vector<double> get_acceleration_vector(
      const Core::FE::Discretization& discretization, const std::vector<int>& lm)
  {
    const Core::LinAlg::Vector<double>& acceleration = *discretization.get_state("acceleration");
    return Core::FE::extract_values(acceleration, lm);
  }

  void evaluate_inertia_force(const Core::LinAlg::SerialDenseMatrix& mass_matrix,
      const Core::LinAlg::SerialDenseVector& acceleration,
      Core::LinAlg::SerialDenseVector& inertia_force)
  {
    inertia_force.putScalar();
    inertia_force.multiply(Teuchos::ETransp::NO_TRANS, Teuchos::ETransp::NO_TRANS, 1.0, mass_matrix,
        acceleration, 0.0);
  }

  void evaluate_inertia_force(const Core::FE::Discretization& discretization,
      const std::vector<int>& lm, const Core::LinAlg::SerialDenseMatrix& mass_matrix,
      Core::LinAlg::SerialDenseVector& inertia_force)
  {
    std::vector<double> my_acceleration = get_acceleration_vector(discretization, lm);

    Core::LinAlg::SerialDenseVector acceleration_vector(
        Teuchos::DataAccess::View, my_acceleration.data(), static_cast<int>(lm.size()));

    evaluate_inertia_force(mass_matrix, acceleration_vector, inertia_force);
  }
}  // namespace

int Discret::Elements::Solid::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  if (!material_post_setup_)
  {
    std::visit([&](auto& interface) { interface->material_post_setup(*this, *solid_material()); },
        solid_calc_variant_);
    material_post_setup_ = true;
  }

  // get ptr to interface to time integration
  set_params_interface_ptr(params);

  const Core::Elements::ActionType action = std::invoke(
      [&]()
      {
        if (is_params_interface())
          return params_interface().get_action_type();
        else
          return Core::Elements::string_to_action_type(params.get<std::string>("action", "none"));
      });

  switch (action)
  {
    case Core::Elements::struct_calc_nlnstiff:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(
                *this, *solid_material(), discretization, lm, params, &elevec1, &elemat1, nullptr);
          },
          solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_internalforce:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(
                *this, *solid_material(), discretization, lm, params, &elevec1, nullptr, nullptr);
          },
          solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_nlnstiffmass:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(
                *this, *solid_material(), discretization, lm, params, &elevec1, &elemat1, &elemat2);
          },
          solid_calc_variant_);

      evaluate_inertia_force(discretization, lm, elemat2, elevec2);
      return 0;
    }
    case Core::Elements::struct_calc_nlnstifflmass:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(
                *this, *solid_material(), discretization, lm, params, &elevec1, &elemat1, &elemat2);
          },
          solid_calc_variant_);

      lump_matrix(elemat2);

      evaluate_inertia_force(discretization, lm, elemat2, elevec2);
      return 0;
    }
    case Core::Elements::struct_calc_internalinertiaforce:
    {
      const int num_dof_per_ele = static_cast<int>(lm.size());
      Core::LinAlg::SerialDenseMatrix mass_matrix(num_dof_per_ele, num_dof_per_ele);

      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, *solid_material(),
                discretization, lm, params, &elevec1, nullptr, &mass_matrix);
          },
          solid_calc_variant_);

      evaluate_inertia_force(discretization, lm, mass_matrix, elevec2);
      return 0;
    }
    case Core::Elements::struct_calc_update_istep:
    {
      std::visit([&](auto& interface)
          { interface->update(*this, *solid_material(), discretization, lm, params); },
          solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_update_prestress:
    {
      update_prestress(solid_calc_variant_, *this, *solid_material(), discretization, lm, params);
      return 0;
    }
    case Core::Elements::struct_calc_recover:
    {
      std::visit([&](auto& interface) { interface->recover(*this, discretization, lm, params); },
          solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_stress:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->calculate_stress(*this, *solid_material(),
                StressIO{get_io_stress_type(*this, params), get_stress_data(*this, params)},
                StrainIO{get_io_strain_type(*this, params), get_strain_data(*this, params)},
                discretization, lm, params);
          },
          solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_energy:
    {
      double int_energy = std::visit(
          [&](auto& interface)
          {
            return interface->calculate_internal_energy(
                *this, *solid_material(), discretization, lm, params);
          },
          solid_calc_variant_);


      if (is_params_interface())
      {
        // new structural time integration
        params_interface().add_contribution_to_energy_type(
            int_energy, FourC::Solid::internal_energy);
      }
      else
      {
        // old structural time integration
        // check length of elevec1
        if (elevec1.length() < 1) FOUR_C_THROW("The given result vector is too short.");

        elevec1(0) = int_energy;
      }
      return 0;
    }
    case Core::Elements::struct_init_gauss_point_data_output:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->initialize_gauss_point_data_output(*this, *solid_material(),
                *params_interface().gauss_point_data_output_manager_ptr());
          },
          solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_gauss_point_data_output:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_gauss_point_data_output(*this, *solid_material(),
                *params_interface().gauss_point_data_output_manager_ptr());
          },
          solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_reset_istep:
    {
      std::visit([&](auto& interface)
          { interface->reset_to_last_converged(*this, *solid_material()); }, solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_predict:
    {
      // do nothing for now
      return 0;
    }
    case Core::Elements::struct_calc_analytical_error:
    {
      // Get the function ID for the analytical solution
      const int analytical_function_id = params.get<int>("analytical_function_id");
      const auto& analytical_displacements_function =
          Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
              analytical_function_id);

      Core::FE::cell_type_switch<Internal::ImplementedSolidCellTypes>(celltype_,
          [&](auto celltype_t)
          {
            const auto error_result = compute_analytical_displacement_error_integration<celltype_t>(
                *this, discretization, lm, analytical_displacements_function);

            elevec1(0) = error_result.integrated_squared_error;
            elevec1(1) = error_result.integrated_squared_displacements;
            elevec1(2) = error_result.integrated_volume;
          });


      return 0;
    }

    default:
      FOUR_C_THROW("The element action {} is not yet implemented for the new solid elements",
          action_type_to_string(action));
  }

  return 0;
}

void Discret::Elements::Solid::set_integration_rule(
    const Core::FE::GaussIntegration& integration_rule)
{
  std::visit([&](auto& interface) { interface->set_integration_rule(integration_rule); },
      solid_calc_variant_);
}

int Discret::Elements::Solid::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);

  const double time = std::invoke(
      [&]()
      {
        if (is_params_interface())
          return params_interface().get_total_time();
        else
          return params.get("total time", -1.0);
      });

  Discret::Elements::evaluate_neumann_by_element<3>(
      *this, discretization, condition, elevec1, time);
  return 0;
}

double Discret::Elements::Solid::get_normal_cauchy_stress_at_xi(const std::vector<double>& disp,
    const Core::LinAlg::Tensor<double, 3>& xi, const Core::LinAlg::Tensor<double, 3>& n,
    const Core::LinAlg::Tensor<double, 3>& dir, CauchyNDirLinearizations<3>& linearizations)
{
  return Discret::Elements::get_normal_cauchy_stress_at_xi(
      solid_calc_variant_, *this, *solid_material(), disp, xi, n, dir, linearizations);
}

FOUR_C_NAMESPACE_CLOSE
