// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_interpolation.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_legacy_enum_definitions_element_actions.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_factory.hpp"
#include "4C_solid_poro_3D_ele_calc_lib.hpp"
#include "4C_solid_poro_3D_ele_calc_lib_p1.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based_p1.hpp"
#include "4C_solid_scatra_3D_ele_factory.hpp"
#include "4C_utils_exceptions.hpp"

#include <type_traits>
#include <variant>

FOUR_C_NAMESPACE_OPEN

int Discret::Elements::SolidPoroPressureVelocityBasedP1::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  if (!material_post_setup_)
  {
    std::visit([&](auto& interface)
        { interface->material_post_setup(*this, struct_poro_material()); }, solid_calc_variant_);
    material_post_setup_ = true;
  }

  if (discretization.has_state(0, "displacement") and (!initial_porosity_))
  {
    initial_porosity_ = Core::FE::extract_values(*discretization.get_state(0, "displacement"),
        get_reduced_porosity_location_array(
            la[0].lm_, Core::FE::get_number_of_element_volumes(shape()), 3));
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

  const int num_nodes = Core::FE::get_number_of_element_nodes(shape());
  constexpr int dim = 3;
  int num_displ_dofs = num_nodes * dim;

  switch (action)
  {
    case Core::Elements::struct_calc_nlnstiff:
    {
      // Create local matrices that we will assemble later on
      Core::LinAlg::SerialDenseVector force_displ(num_displ_dofs);
      Core::LinAlg::SerialDenseMatrix K_dd(num_displ_dofs, num_displ_dofs);

      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization, get_reduced_displacement_location_array(la[0].lm_, num_nodes, dim),
                params, &force_displ, &K_dd, nullptr);
          },
          solid_calc_variant_);

      if (la.size() > 1)
      {
        Core::LinAlg::SerialDenseVector force_p(num_nodes);
        Core::LinAlg::SerialDenseMatrix K_dp(num_displ_dofs, num_nodes);
        Core::LinAlg::SerialDenseMatrix K_pd(num_nodes, num_displ_dofs);
        Core::LinAlg::SerialDenseMatrix K_pp(num_nodes, num_nodes);

        Core::LinAlg::SerialDenseMatrix reaction_matrix(num_displ_dofs, num_displ_dofs);

        const SolidPoroPrimaryVariables primary_variables =
            extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);

        SolidPoroDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
            diagonal_block_matrices{
                .force_vector = &force_displ,
                .porosity_force_vector = &force_p,
                .K_displacement_displacement = &K_dd,
                .K_displacement_porosity = &K_dp,
                .K_porosity_displacement = &K_pd,
                .K_porosity_porosity = &K_pp,
            };

        std::visit(
            [&](auto& interface)
            {
              interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                  this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                  this->kinematic_type(), discretization, primary_variables, params,
                  diagonal_block_matrices, &reaction_matrix);
            },
            solidporo_press_vel_based_calc_variant_);


        assemble_mixed_displacement_porosity_vector<SolidPoroDofType::porosity, dim>(
            elevec1, force_p);

        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
            SolidPoroDofType::porosity, dim>(elemat1, K_dp);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::displacement, dim>(elemat1, K_pd);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::porosity, dim>(elemat1, K_pp);

        if (elemat2.num_cols() > 0)
          assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
              SolidPoroDofType::displacement, dim>(elemat2, reaction_matrix);
      }

      assemble_mixed_displacement_porosity_vector<SolidPoroDofType::displacement, dim>(
          elevec1, force_displ);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::displacement, dim>(elemat1, K_dd);
      return 0;
    }
    case Core::Elements::struct_calc_internalforce:
    {
      // Create local matrices that we will assemble later on
      Core::LinAlg::SerialDenseVector force_displ(num_displ_dofs);
      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization, get_reduced_displacement_location_array(la[0].lm_, num_nodes, dim),
                params, &force_displ, nullptr, nullptr);
          },
          solid_calc_variant_);

      if (la.size() > 1)
      {
        Core::LinAlg::SerialDenseVector force_p(num_nodes);
        const SolidPoroPrimaryVariables primary_variables =
            extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);


        SolidPoroDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
            diagonal_block_matrices{
                .force_vector = &force_displ,
                .porosity_force_vector = &force_p,
            };

        std::visit(
            [&](auto& interface)
            {
              interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                  this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                  this->kinematic_type(), discretization, primary_variables, params,
                  diagonal_block_matrices, nullptr);
            },
            solidporo_press_vel_based_calc_variant_);

        assemble_mixed_displacement_porosity_vector<SolidPoroDofType::porosity, dim>(
            elevec1, force_p);
      }

      assemble_mixed_displacement_porosity_vector<SolidPoroDofType::displacement, dim>(
          elevec1, force_displ);
      return 0;
    }
    case Core::Elements::struct_calc_nlnstiffmass:
    {
      // Create local matrices that we will assemble later on
      Core::LinAlg::SerialDenseVector force_displ(num_displ_dofs);
      Core::LinAlg::SerialDenseMatrix K_dd(num_displ_dofs, num_displ_dofs);
      Core::LinAlg::SerialDenseMatrix M_dd(num_displ_dofs, num_displ_dofs);

      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_mass(*this, this->struct_poro_material(),
                discretization, get_reduced_displacement_location_array(la[0].lm_, num_nodes, dim),
                params, &force_displ, &K_dd, &M_dd);
          },
          solid_calc_variant_);


      if (la.size() > 1 and this->num_material() > 1)
      {
        Core::LinAlg::SerialDenseVector force_p(num_nodes);
        Core::LinAlg::SerialDenseMatrix K_dp(num_displ_dofs, num_nodes);
        Core::LinAlg::SerialDenseMatrix K_pd(num_nodes, num_displ_dofs);
        Core::LinAlg::SerialDenseMatrix K_pp(num_nodes, num_nodes);

        const SolidPoroPrimaryVariables primary_variables =
            extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);


        SolidPoroDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
            diagonal_block_matrices{
                .force_vector = &force_displ,
                .porosity_force_vector = &force_p,
                .K_displacement_displacement = &K_dd,
                .K_displacement_porosity = &K_dp,
                .K_porosity_displacement = &K_pd,
                .K_porosity_porosity = &K_pp,
            };

        std::visit(
            [&](auto& interface)
            {
              interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                  this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                  this->kinematic_type(), discretization, primary_variables, params,
                  diagonal_block_matrices, nullptr);
            },
            solidporo_press_vel_based_calc_variant_);

        assemble_mixed_displacement_porosity_vector<SolidPoroDofType::porosity, dim>(
            elevec1, force_p);

        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
            SolidPoroDofType::porosity, dim>(elemat1, K_dp);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::displacement, dim>(elemat1, K_pd);
        assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
            SolidPoroDofType::porosity, dim>(elemat1, K_pp);
      }


      assemble_mixed_displacement_porosity_vector<SolidPoroDofType::displacement, dim>(
          elevec1, force_displ);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::displacement, dim>(elemat1, K_dd);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::displacement, dim>(elemat2, M_dd);

      return 0;
    }
    case Core::Elements::struct_poro_calc_scatracoupling:
    {
      // no coupling -> return
      return 0;
    }
    case Core::Elements::struct_poro_calc_fluidcoupling:
    {
      Core::LinAlg::SerialDenseMatrix K_displ_fluid(num_displ_dofs, (dim + 1) * num_nodes);
      Core::LinAlg::SerialDenseMatrix K_por_pres(num_nodes, num_nodes);

      const SolidPoroPrimaryVariables primary_variables =
          extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);


      SolidPoroOffDiagonalBlockMatrices<PorosityFormulation::as_primary_variable>
          off_diagonal_block_matrices{
              .K_displacement_fluid_dofs = &K_displ_fluid,
              .K_porosity_pressure = &K_por_pres,
          };

      std::visit(
          [&](auto& interface)
          {
            interface->evaluate_nonlinear_force_stiffness_od(*this, this->struct_poro_material(),
                this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                this->kinematic_type(), discretization, primary_variables, params,
                off_diagonal_block_matrices);
          },
          solidporo_press_vel_based_calc_variant_);

      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::displacement,
          SolidPoroDofType::none, dim>(elemat1, K_displ_fluid);
      assemble_mixed_displacement_porosity_matrix<SolidPoroDofType::porosity,
          SolidPoroDofType::porosity, dim>(
          elemat1, K_por_pres);  // pressure is equally arranged as porosity

      return 0;
    }
    case Core::Elements::struct_calc_update_istep:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->update(*this, solid_poro_material(), discretization,
                get_reduced_displacement_location_array(la[0].lm_, num_nodes, dim), params);
          },
          solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_recover:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->recover(*this, discretization,
                get_reduced_displacement_location_array(la[0].lm_, num_nodes, dim), params);
          },
          solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_stress:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->calculate_stress(*this, this->struct_poro_material(),
                StressIO{.type = get_io_stress_type(*this, params),
                    .mutable_data = get_stress_data(*this, params)},
                StrainIO{.type = get_io_strain_type(*this, params),
                    .mutable_data = get_strain_data(*this, params)},
                discretization, get_reduced_displacement_location_array(la[0].lm_, num_nodes, dim),
                params);
          },
          solid_calc_variant_);

      const SolidPoroPrimaryVariables primary_variables =
          extract_solid_poro_primary_variables(discretization, la, shape(), *initial_porosity_);

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
          solid_calc_variant_);
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
          solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_predict:
    case Core::Elements::none:
    {
      // do nothing for now
      return 0;
    }
    default:
      FOUR_C_THROW("The element action {} is not yet implemented for the new solid elements",
          action_type_to_string(action));
      // do nothing (no error because there are some actions the poro element is supposed to
      // ignore)
      return 0;
  }
}


int Discret::Elements::SolidPoroPressureVelocityBasedP1::evaluate_neumann(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const Core::Conditions::Condition& condition, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW(
      "Cannot yet evaluate volume neumann forces within the pressure-velocity based poro p1 "
      "implementation.");
}
FOUR_C_NAMESPACE_CLOSE
