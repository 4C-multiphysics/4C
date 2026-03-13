// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_structporo.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_calc_lib_io.hpp"
#include "4C_solid_poro_3D_ele_pressure_based.hpp"

FOUR_C_NAMESPACE_OPEN


int Discret::Elements::SolidPoroPressureBased::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  FOUR_C_ASSERT(solid_calc_variant_.has_value(),
      "The solid calculation interface is not initialized for element id {}.", id());
  if (!material_post_setup_)
  {
    std::visit([&](auto& interface)
        { interface->material_post_setup(*this, struct_poro_material()); }, *solid_calc_variant_);
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
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization, la[0].lm_, params, &elevec1, &elemat1, nullptr);
          },
          *solid_calc_variant_);

      if (la.size() > 2 and this->num_material() > 1)
      {
        if (discretization.has_state(1, "porofluid"))
        {
          std::visit(
              [&](auto& interface)
              {
                interface->add_solidpressure_contribution_to_nonlinear_force_stiffness(*this,
                    this->struct_poro_material(), this->fluid_poro_material(),
                    this->get_ele_kinematic_type(), discretization, la, params, &elevec1, &elemat1);
              },
              solidporo_press_based_calc_variant_);
          if (this->get_possible_bodyforce_contribution().has_value())
          {
            std::visit(
                [&](auto& interface)
                {
                  interface->add_bodyforce_contribution_to_nonlinear_force_stiffness(*this,
                      this->struct_poro_material(), this->fluid_poro_material(),
                      this->get_ele_kinematic_type(),
                      this->get_possible_bodyforce_contribution().value(), discretization, la,
                      params, &elevec1, &elemat1);
                },
                solidporo_press_based_calc_variant_);
          }
        }
      }
      return 0;
    }
    case Core::Elements::struct_calc_internalforce:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization, la[0].lm_, params, &elevec1, nullptr, nullptr);
          },
          *solid_calc_variant_);

      if (la.size() > 2 and this->num_material() > 1)
      {
        if (discretization.has_state(1, "porofluid"))
        {
          std::visit(
              [&](auto& interface)
              {
                interface->add_solidpressure_contribution_to_nonlinear_force_stiffness(*this,
                    this->struct_poro_material(), this->fluid_poro_material(),
                    this->get_ele_kinematic_type(), discretization, la, params, &elevec1, nullptr);
              },
              solidporo_press_based_calc_variant_);
          if (this->get_possible_bodyforce_contribution().has_value())
          {
            std::visit(
                [&](auto& interface)
                {
                  interface->add_bodyforce_contribution_to_nonlinear_force_stiffness(*this,
                      this->struct_poro_material(), this->fluid_poro_material(),
                      this->get_ele_kinematic_type(),
                      this->get_possible_bodyforce_contribution().value(), discretization, la,
                      params, &elevec1, &elemat1);
                },
                solidporo_press_based_calc_variant_);
          }
        }
      }
      return 0;
    }
    case Core::Elements::struct_calc_nlnstiffmass:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization, la[0].lm_, params, &elevec1, &elemat1, &elemat2);
          },
          *solid_calc_variant_);

      // we skip this evaluation if the coupling is not setup yet, i.e.
      // if the secondary dofset or the secondary material was not set
      // this can happen during setup of the time integrator or restart
      // there might be a better way. For instance do not evaluate
      // before the setup of the multiphysics problem is completed.
      if (la.size() > 2 and this->num_material() > 1)
      {
        if (discretization.has_state(1, "porofluid"))
        {
          std::visit(
              [&](auto& interface)
              {
                interface->add_solidpressure_contribution_to_nonlinear_force_stiffness(*this,
                    this->struct_poro_material(), this->fluid_poro_material(),
                    this->get_ele_kinematic_type(), discretization, la, params, &elevec1, &elemat1);
              },
              solidporo_press_based_calc_variant_);
          if (this->get_possible_bodyforce_contribution().has_value())
          {
            std::visit(
                [&](auto& interface)
                {
                  interface->add_bodyforce_contribution_to_nonlinear_force_stiffness(*this,
                      this->struct_poro_material(), this->fluid_poro_material(),
                      this->get_ele_kinematic_type(),
                      this->get_possible_bodyforce_contribution().value(), discretization, la,
                      params, &elevec1, &elemat1);
                },
                solidporo_press_based_calc_variant_);
          }
        }
      }
      return 0;
    }
    case Core::Elements::struct_calc_nlnstifflmass:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization, la[0].lm_, params, &elevec1, &elemat1, &elemat2);
          },
          *solid_calc_variant_);
      Discret::Elements::lump_matrix(elemat2);
      return 0;
    }
    case Core::Elements::struct_poro_calc_scatracoupling:
    {
      // no coupling -> return
      return 0;
    }
    case Core::Elements::struct_poro_calc_fluidcoupling:
    {
      if (la.size() > 2)
      {
        if (discretization.has_state(1, "porofluid"))
        {
          std::visit(
              [&](auto& interface)
              {
                interface->evaluate_nonlinear_force_stiffness_od(*this,
                    this->struct_poro_material(), this->fluid_poro_material(),
                    this->get_ele_kinematic_type(), discretization, la, params, elemat1);
              },
              solidporo_press_based_calc_variant_);
        }
      }
      return 0;
    }
    case Core::Elements::struct_calc_update_istep:
    {
      std::visit([&](auto& interface)
          { interface->update(*this, solid_poro_material(), discretization, la[0].lm_, params); },
          *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_recover:
    {
      std::visit([&](auto& interface)
          { interface->recover(*this, discretization, la[0].lm_, params); }, *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_stress:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->calculate_stress(*this, this->struct_poro_material(),
                StressIO{get_io_stress_type(*this, params), get_stress_data(*this, params)},
                StrainIO{get_io_strain_type(*this, params), get_strain_data(*this, params)},
                discretization, la[0].lm_, params);
          },
          *solid_calc_variant_);

      return 0;
    }
    case Core::Elements::struct_init_gauss_point_data_output:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->initialize_gauss_point_data_output(*this, solid_poro_material(),
                *get_solid_params_interface().gauss_point_data_output_manager_ptr());
          },
          *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_gauss_point_data_output:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_gauss_point_data_output(*this, solid_poro_material(),
                *get_solid_params_interface().gauss_point_data_output_manager_ptr());
          },
          *solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_predict:
    {
      // do nothing for now
      return 0;
    }
    default:
      FOUR_C_THROW("The element action {} is not yet implemented for the new solid elements",
          action_type_to_string(action));
      return 0;
  }
}

int Discret::Elements::SolidPoroPressureBased::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW("not implemented");
  return 1;
}
FOUR_C_NAMESPACE_CLOSE
