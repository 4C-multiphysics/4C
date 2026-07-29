// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_terminal_unit.hpp"

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_terminal_unit_elasticity.hpp"
#include "4C_reduced_lung_terminal_unit_model_registry.hpp"
#include "4C_reduced_lung_terminal_unit_rheology.hpp"
#include "4C_reduced_lung_test_utils_test.hpp"

#include <array>
#include <unordered_map>

namespace
{
  namespace Core = FourC::Core;
  using namespace FourC::ReducedLung;
  using namespace FourC::Core::LinAlg;
  using namespace FourC::ReducedLung::TestUtils;
  using namespace FourC::ReducedLung::TerminalUnits;

  using RheologicalModelType =
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType;
  using ElasticityModelType =
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType;

  struct TerminalUnitModelCase
  {
    const char* name;
    RheologicalModelType rheological_model_type;
    ElasticityModelType elasticity_model_type;
    std::array<double, 3> kelvin_voigt_eta;
    std::array<double, 3> maxwell_e_m;
    std::array<double, 3> maxwell_eta_m;
    std::array<double, 3> linear_elasticity_e;
    std::array<double, 3> ogden_kappa;
    std::array<double, 3> ogden_beta;
    bool check_completed_jacobian_update = false;
  };

  Core::IO::InputField<double> make_elementwise_double_field(const std::array<double, 3>& values)
  {
    return Core::IO::InputField<double>(
        std::unordered_map<int, double>{{1, values[0]}, {2, values[1]}, {3, values[2]}});
  }

  ReducedLungParameters make_terminal_unit_parameters(const TerminalUnitModelCase& model_case)
  {
    ReducedLungParameters params{};

    params.lung_tree.terminal_units.rheological_model.rheological_model_type =
        Core::IO::InputField<RheologicalModelType>(model_case.rheological_model_type);
    params.lung_tree.terminal_units.rheological_model.kelvin_voigt.viscosity_kelvin_voigt_eta =
        make_elementwise_double_field(model_case.kelvin_voigt_eta);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell
        .viscosity_kelvin_voigt_eta = make_elementwise_double_field(model_case.kelvin_voigt_eta);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.elasticity_maxwell_e_m =
        make_elementwise_double_field(model_case.maxwell_e_m);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.viscosity_maxwell_eta_m =
        make_elementwise_double_field(model_case.maxwell_eta_m);

    params.lung_tree.terminal_units.elasticity_model.elasticity_model_type =
        Core::IO::InputField<ElasticityModelType>(model_case.elasticity_model_type);
    params.lung_tree.terminal_units.elasticity_model.linear.elasticity_e =
        make_elementwise_double_field(model_case.linear_elasticity_e);
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_kappa =
        make_elementwise_double_field(model_case.ogden_kappa);
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_beta =
        make_elementwise_double_field(model_case.ogden_beta);

    return params;
  }

  class TerminalUnitRegistryAndJacobianTest : public testing::TestWithParam<TerminalUnitModelCase>
  {
  };

  // Tests model registration + analytic Jacobian by comparing against FD residual derivatives.
  TEST_P(TerminalUnitRegistryAndJacobianTest, JacobianVsFiniteDifference)
  {
    const auto& model_case = GetParam();
    SCOPED_TRACE(model_case.name);

    const auto params = make_terminal_unit_parameters(model_case);
    TerminalUnitContainer terminal_units;

    constexpr std::array element_lengths{1.0, 2.0, 3.0};
    for (int global_element_id = 0; global_element_id < 3; ++global_element_id)
    {
      TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(terminal_units,
          global_element_id, global_element_id, element_lengths[global_element_id], params,
          model_case.rheological_model_type, model_case.elasticity_model_type);
    }
    ASSERT_EQ(terminal_units.models.size(), 1u);

    auto& model = terminal_units.models.front();

    if (model_case.rheological_model_type == RheologicalModelType::KelvinVoigt)
    {
      EXPECT_TRUE(std::holds_alternative<KelvinVoigt>(model.rheological_model));
    }
    else
    {
      EXPECT_TRUE(std::holds_alternative<FourElementMaxwell>(model.rheological_model));
    }

    if (model_case.elasticity_model_type == ElasticityModelType::Linear)
    {
      EXPECT_TRUE(std::holds_alternative<LinearElasticity>(model.elasticity_model));
    }
    else
    {
      EXPECT_TRUE(std::holds_alternative<OgdenHyperelasticity>(model.elasticity_model));
    }

    Airways::AirwayContainer airways;
    const std::map<int, int> global_dof_per_ele = {{0, 3}, {1, 3}, {2, 3}};
    std::map<int, int> first_global_dof_of_ele = {{0, 0}, {1, 3}, {2, 6}};

    assign_global_dof_ids_to_models(first_global_dof_of_ele, airways, terminal_units);

    int n_local_equations = 0;
    TerminalUnits::assign_local_equation_ids(terminal_units, n_local_equations);

    const auto dof_map = create_domain_map(MPI_COMM_WORLD, airways, terminal_units);
    const auto row_map = create_row_map(MPI_COMM_WORLD, airways, terminal_units, {}, {}, {});
    const auto col_map = create_column_map(MPI_COMM_WORLD, airways, terminal_units,
        global_dof_per_ele, first_global_dof_of_ele, {}, {}, {});
    TerminalUnits::assign_local_dof_ids(col_map, terminal_units);

    // Artificial dof vector
    Vector<double> dofs(dof_map, true);
    Vector<double> locally_relevant_dofs(col_map, true);
    dofs.replace_local_values(9, std::array<double, 9>{1, 1, 1, 1, 1, 1, 1, 1, 1}.data(),
        std::array<int, 9>{0, 1, 2, 3, 4, 5, 6, 7, 8}.data());
    export_to(dofs, locally_relevant_dofs);

    double dt = 1e-1;         // Dummy time step size
    const double eps = 1e-6;  // Perturbation parameter for the FD approximation

    TerminalUnits::create_evaluators(terminal_units);

    SparseMatrix jac(row_map, col_map, 3);
    model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

    check_jacobian_column_against_fd(
        model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
    check_jacobian_column_against_fd(
        model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
    check_jacobian_column_against_fd(
        model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);

    if (model_case.check_completed_jacobian_update)
    {
      jac.complete();  // Sparsity pattern already filled the first time
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
  }

  TEST(TerminalUnitRegistryTest, ThrowsOnUnknownRheologyType)
  {
    const auto model_case = TerminalUnitModelCase{
        .name = "invalid_rheology",
        .rheological_model_type = RheologicalModelType::KelvinVoigt,
        .elasticity_model_type = ElasticityModelType::Linear,
        .kelvin_voigt_eta = {0.0, 1.0, 100.0},
        .maxwell_e_m = {10.0, 0.0, 20.0},
        .maxwell_eta_m = {2.5, 10.0, 0.0},
        .linear_elasticity_e = {1.0, 1.0, 0.0},
        .ogden_kappa = {1.0, 1.0, 1.0},
        .ogden_beta = {5.0, -0.4, -8.0},
    };
    const auto params = make_terminal_unit_parameters(model_case);

    TerminalUnitContainer terminal_units;
    EXPECT_THROW(
        TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(terminal_units, 0, 0,
            1.0, params, static_cast<RheologicalModelType>(-1), ElasticityModelType::Linear),
        Core::Exception);
  }

  INSTANTIATE_TEST_SUITE_P(TerminalUnitModelPairs, TerminalUnitRegistryAndJacobianTest,
      testing::Values(TerminalUnitModelCase{.name = "KelvinVoigt_Linear",
                          .rheological_model_type = RheologicalModelType::KelvinVoigt,
                          .elasticity_model_type = ElasticityModelType::Linear,
                          .kelvin_voigt_eta = {0.0, 1.0, 100.0},
                          .maxwell_e_m = {10.0, 0.0, 20.0},
                          .maxwell_eta_m = {2.5, 10.0, 0.0},
                          .linear_elasticity_e = {1.0, 1.0, 0.0},
                          .ogden_kappa = {1.0, 1.0, 1.0},
                          .ogden_beta = {5.0, -0.4, -8.0}},
          TerminalUnitModelCase{.name = "KelvinVoigt_Ogden",
              .rheological_model_type = RheologicalModelType::KelvinVoigt,
              .elasticity_model_type = ElasticityModelType::Ogden,
              .kelvin_voigt_eta = {0.0, 1.0, 100.0},
              .maxwell_e_m = {10.0, 0.0, 20.0},
              .maxwell_eta_m = {2.5, 10.0, 0.0},
              .linear_elasticity_e = {1.0, 1.0, 0.0},
              .ogden_kappa = {1.0, 1.0, 1.0},
              .ogden_beta = {5.0, -0.4, -8.0}},
          TerminalUnitModelCase{.name = "FourElementMaxwell_Linear",
              .rheological_model_type = RheologicalModelType::FourElementMaxwell,
              .elasticity_model_type = ElasticityModelType::Linear,
              .kelvin_voigt_eta = {0.0, 1.0, 100.0},
              .maxwell_e_m = {10.0, 0.0, 20.0},
              .maxwell_eta_m = {2.5, 10.0, 0.0},
              .linear_elasticity_e = {1.0, 1.0, 0.0},
              .ogden_kappa = {1.0, 1.0, 1.0},
              .ogden_beta = {5.0, -0.4, -8.0},
              .check_completed_jacobian_update = true},
          TerminalUnitModelCase{.name = "FourElementMaxwell_Ogden",
              .rheological_model_type = RheologicalModelType::FourElementMaxwell,
              .elasticity_model_type = ElasticityModelType::Ogden,
              .kelvin_voigt_eta = {10.5, 1.0, 100.0},
              .maxwell_e_m = {10.0, 0.0, 20.0},
              .maxwell_eta_m = {2.5, 10.0, 0.0},
              .linear_elasticity_e = {1.0, 1.0, 0.0},
              .ogden_kappa = {0.0, 1.0, 1.0},
              .ogden_beta = {1.0, 6.4, -3.0}}),
      [](const testing::TestParamInfo<TerminalUnitModelCase>& info) { return info.param.name; });
}  // namespace
