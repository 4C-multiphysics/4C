// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_interpolation.hpp"
#include "4C_legacy_enum_definitions_element_actions.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_solid_3D_ele_factory.hpp"
#include "4C_solid_poro_3D_ele_calc_lib.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based.hpp"
#include "4C_solid_scatra_3D_ele_factory.hpp"
#include "4C_utils_exceptions.hpp"

#include <type_traits>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <typename T, typename Variant>
  struct IsMemberOfVariant;

  template <typename T, typename... Types>
  struct IsMemberOfVariant<T, std::variant<Types...>>
      : public std::disjunction<std::is_same<T, Types>...>
  {
  };

  template <typename T>
  concept IsSolidInterface =
      IsMemberOfVariant<std::remove_cvref_t<T>, Discret::Elements::SolidCalcVariant>::value;

  template <typename T>
  concept IsSolidScatraInterface =
      IsMemberOfVariant<std::remove_cvref_t<T>, Discret::Elements::SolidScatraCalcVariant>::value;

  template <IsSolidScatraInterface SolidInterface>
  const auto& get_location_array(const Core::Elements::LocationArray& la)
  {
    // The solid-scatra interface expects all location arrays!
    return la;
  }

  template <IsSolidInterface SolidInterface>
  const auto& get_location_array(const Core::Elements::LocationArray& la)
  {
    // The solid interface only expects the first location array!
    return la[0].lm_;
  }
}  // namespace

int Discret::Elements::SolidPoroPressureVelocityBased::evaluate(Teuchos::ParameterList& params,
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
                discretization, get_location_array<decltype(interface)>(la), params, &elevec1,
                &elemat1, nullptr);
          },
          solid_calc_variant_);

      if (la.size() > 1)
      {
        if (discretization.has_state(1, "fluidvel"))
        {
          const SolidPoroPrimaryVariables primary_variables =
              extract_solid_poro_primary_variables(discretization, la, shape());
          std::visit(
              [&](auto& interface)
              {
                SolidPoroDiagonalBlockMatrices<PorosityFormulation::from_material_law>
                    diagonal_block_matrices{
                        .force_vector = &elevec1, .K_displacement_displacement = &elemat1};

                interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                    this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                    this->kinematic_type(), discretization, primary_variables, params,
                    diagonal_block_matrices, &elemat2);
              },
              solidporo_press_vel_based_calc_variant_);
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
                discretization, get_location_array<decltype(interface)>(la), params, &elevec1,
                nullptr, nullptr);
          },
          solid_calc_variant_);

      if (la.size() > 1)
      {
        if (discretization.has_state(1, "fluidvel"))
        {
          const SolidPoroPrimaryVariables primary_variables =
              extract_solid_poro_primary_variables(discretization, la, shape());
          std::visit(
              [&](auto& interface)
              {
                SolidPoroDiagonalBlockMatrices<PorosityFormulation::from_material_law>
                    diagonal_block_matrices{.force_vector = &elevec1};

                interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                    this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                    this->kinematic_type(), discretization, primary_variables, params,
                    diagonal_block_matrices, nullptr);
              },
              solidporo_press_vel_based_calc_variant_);
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
                discretization, get_location_array<decltype(interface)>(la), params, &elevec1,
                &elemat1, &elemat2);
          },
          solid_calc_variant_);

      if (la.size() > 1 and this->num_material() > 1)
      {
        if (discretization.has_state(1, "fluidvel"))
        {
          const SolidPoroPrimaryVariables primary_variables =
              extract_solid_poro_primary_variables(discretization, la, shape());
          std::visit(
              [&](auto& interface)
              {
                SolidPoroDiagonalBlockMatrices<PorosityFormulation::from_material_law>
                    diagonal_block_matrices{
                        .force_vector = &elevec1, .K_displacement_displacement = &elemat1};

                interface->evaluate_nonlinear_force_stiffness(*this, this->struct_poro_material(),
                    this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                    this->kinematic_type(), discretization, primary_variables, params,
                    diagonal_block_matrices, nullptr);
              },
              solidporo_press_vel_based_calc_variant_);
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
                discretization, get_location_array<decltype(interface)>(la), params, &elevec1,
                &elemat1, &elemat2);
          },
          solid_calc_variant_);
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
      if (discretization.has_state(1, "fluidvel"))
      {
        const SolidPoroPrimaryVariables primary_variables =
            extract_solid_poro_primary_variables(discretization, la, shape());
        std::visit(
            [&](auto& interface)
            {
              SolidPoroOffDiagonalBlockMatrices<PorosityFormulation::from_material_law>
                  off_diagonal_block_matrices{.K_displacement_fluid_dofs = &elemat1};

              interface->evaluate_nonlinear_force_stiffness_od(*this, this->struct_poro_material(),
                  this->fluid_poro_material(), this->get_anisotropic_permeability_property(),
                  this->kinematic_type(), discretization, primary_variables, params,
                  off_diagonal_block_matrices);
            },
            solidporo_press_vel_based_calc_variant_);
      }
      return 0;
    }
    case Core::Elements::struct_calc_update_istep:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->update(*this, solid_poro_material(), discretization,
                get_location_array<decltype(interface)>(la), params);
          },
          solid_calc_variant_);
      return 0;
    }
    case Core::Elements::struct_calc_recover:
    {
      std::visit(
          [&](auto& interface)
          {
            interface->recover(
                *this, discretization, get_location_array<decltype(interface)>(la), params);
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
                discretization, get_location_array<decltype(interface)>(la), params);
          },
          solid_calc_variant_);

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
      // do nothing (no error because there are some actions the poro element is supposed to ignore)
      return 0;
  }
}



double Discret::Elements::SolidPoroPressureVelocityBased::get_normal_cauchy_stress_at_xi(
    const std::vector<double>& disp, const std::optional<std::vector<double>>& pressures,
    const Core::LinAlg::Tensor<double, 3>& xi, const Core::LinAlg::Tensor<double, 3>& n,
    const Core::LinAlg::Tensor<double, 3>& dir,
    SolidPoroCauchyNDirLinearizations<3>& linearizations)
{
  double cauchy_stress_n_dir = std::visit(
      [&]<typename Interface>(Interface& solid) -> double
      {
        if constexpr (IsSolidInterface<Interface>)
        {
          return solid->get_normal_cauchy_stress_at_xi(
              *this, this->struct_poro_material(), disp, xi, n, dir, linearizations.solid);
        }
        FOUR_C_THROW("Not yet implemented");
      },
      solid_calc_variant_);

  if (!pressures) return cauchy_stress_n_dir;

  const double n_dot_dir = n * dir;

  using supported_celltypes = Core::FE::CelltypeSequence<Core::FE::CellType::hex8>;
  Core::FE::cell_type_switch<supported_celltypes>(shape(),
      [&](auto celltype_t)
      {
        constexpr Core::FE::CellType celltype = celltype_t();

        ElementNodes<celltype> element_nodes = evaluate_element_nodes<celltype>(*this, disp);

        const ShapeFunctionsAndDerivatives<celltype> shape_functions =
            evaluate_shape_functions_and_derivs<celltype>(xi, element_nodes);

        const double pressure_at_xi = Core::FE::interpolate_to_xi<celltype>(
            Core::LinAlg::make_matrix_view<3, 1>(xi), *pressures)[0];

        cauchy_stress_n_dir += -pressure_at_xi * n_dot_dir;

        if (linearizations.d_cauchyndir_dp || linearizations.solid.d_cauchyndir_dn ||
            linearizations.solid.d_cauchyndir_ddir || linearizations.solid.d_cauchyndir_dxi)
        {
          linearizations.d_cauchyndir_dp->reshape(Core::FE::num_nodes(celltype), 1);
          for (unsigned node = 0; node < Core::FE::num_nodes(celltype); ++node)
          {
            if (linearizations.d_cauchyndir_dp)
              (*linearizations.d_cauchyndir_dp)(node, 0) =
                  -n_dot_dir * shape_functions.shapefunctions_(node);

            for (unsigned dim = 0; dim < 3; ++dim)
            {
              if (linearizations.solid.d_cauchyndir_dn)
                (*linearizations.solid.d_cauchyndir_dn)(dim, 0) -= pressure_at_xi * dir(dim);

              if (linearizations.solid.d_cauchyndir_ddir)
                (*linearizations.solid.d_cauchyndir_ddir)(dim, 0) -= pressure_at_xi * n(dim);

              if (linearizations.solid.d_cauchyndir_dxi)
                (*linearizations.solid.d_cauchyndir_dxi)(dim, 0) -=
                    (*pressures)[node] * shape_functions.derivatives_(dim, node) * n_dot_dir;
            }
          }
        }
      });


  return cauchy_stress_n_dir;
}

int Discret::Elements::SolidPoroPressureVelocityBased::evaluate_neumann(
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    const Core::Conditions::Condition& condition, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW(
      "Cannot yet evaluate volume neumann forces within the pressure-velocity based poro "
      "implementation.");
}
FOUR_C_NAMESPACE_CLOSE
