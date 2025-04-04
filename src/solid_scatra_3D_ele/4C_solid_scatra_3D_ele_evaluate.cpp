// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_elements_paramsinterface.hpp"
#include "4C_legacy_enum_definitions_element_actions.hpp"
#include "4C_solid_3D_ele_neumann_evaluator.hpp"
#include "4C_solid_scatra_3D_ele.hpp"
#include "4C_solid_scatra_3D_ele_calc_lib_nitsche.hpp"

FOUR_C_NAMESPACE_OPEN

int Discret::Elements::SolidScatra::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  if (!material_post_setup_)
  {
    std::visit([&](auto& interface) { interface->material_post_setup(*this, solid_material()); },
        solid_scatra_calc_variant_);
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
    case Core::Elements::ActionType::struct_calc_stifftemp:  // Todo: use stiffscalar also in tsi
                                                             // algorithm
    case Core::Elements::ActionType::calc_struct_stiffscalar:
      // Evaluate off-diagonal block for SSI
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_d_stress_d_scalar(
                *this, solid_material(), discretization, la, params, elemat1);
          },
          solid_scatra_calc_variant_);
      return 0;
    case Core::Elements::struct_calc_nlnstiff:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(
                *this, solid_material(), discretization, la, params, &elevec1, &elemat1, nullptr);
          },
          solid_scatra_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_nlnstiffmass:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(
                *this, solid_material(), discretization, la, params, &elevec1, &elemat1, &elemat2);
          },
          solid_scatra_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_internalforce:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(
                *this, solid_material(), discretization, la, params, &elevec1, nullptr, nullptr);
          },
          solid_scatra_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_update_istep:
    {
      std::visit([&](auto& interface)
          { interface->update(*this, solid_material(), discretization, la, params); },
          solid_scatra_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_recover:
    {
      std::visit([&](auto& interface) { interface->recover(*this, discretization, la, params); },
          solid_scatra_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_stress:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->calculate_stress(*this, solid_material(),
                StressIO{get_io_stress_type(*this, params), get_stress_data(*this, params)},
                StrainIO{get_io_strain_type(*this, params), get_strain_data(*this, params)},
                discretization, la, params);
          },
          solid_scatra_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_reset_istep:
    {
      std::visit([&](auto& interface)
          { interface->reset_to_last_converged(*this, solid_material()); },
          solid_scatra_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_calc_predict:
      // do nothing for now
      return 0;
    default:
      FOUR_C_THROW("The element action {} is not yet implemented for the new solid-scatra elements",
          action_type_to_string(action));
  }

  return 0;
}

int Discret::Elements::SolidScatra::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
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

double Discret::Elements::SolidScatra::get_normal_cauchy_stress_at_xi(
    const std::vector<double>& disp, const std::vector<double>& scalars,
    const Core::LinAlg::Matrix<3, 1>& xi, const Core::LinAlg::Matrix<3, 1>& n,
    const Core::LinAlg::Matrix<3, 1>& dir,
    Discret::Elements::SolidScatraCauchyNDirLinearizations<3>& linearizations)
{
  return Discret::Elements::get_normal_cauchy_stress_at_xi(solid_scatra_calc_variant_, *this,
      solid_material(), disp, scalars, xi, n, dir, linearizations);
}

FOUR_C_NAMESPACE_CLOSE