// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// Reimplemented airway tests to match the style of the terminal unit tests.
#include <gtest/gtest.h>

#include "4C_reduced_lung_airways.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_reduced_lung_airways_model_registry.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_reduced_lung_test_utils_test.hpp"
// needed for export_to
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <unordered_map>


namespace
{
  namespace Core = FourC::Core;
  using namespace FourC::ReducedLung;
  using namespace FourC::Core::LinAlg;
  using namespace FourC::ReducedLung::Airways;

  using FlowModelType = ReducedLungParameters::LungTree::Airways::FlowModel::ResistanceType;
  using WallModelType = ReducedLungParameters::LungTree::Airways::WallModelType;

  ReducedLungParameters make_airway_registry_parameters()
  {
    ReducedLungParameters params{};
    params.air_properties = {
        .density = 1.176e-06,
        .dynamic_viscosity = 1.79105e-05,
    };

    params.lung_tree.airways.radius = Core::IO::InputField<double>(
        std::unordered_map<int, double>{{1, 1.0}, {2, 0.9}, {3, 0.8}, {4, 0.7}});
    params.lung_tree.airways.flow_model.include_inertia = Core::IO::InputField<bool>(
        std::unordered_map<int, bool>{{1, false}, {2, false}, {3, false}, {4, false}});
    params.lung_tree.airways.flow_model.resistance_model.non_linear.turbulence_factor_gamma =
        Core::IO::InputField<double>(
            std::unordered_map<int, double>{{1, 0.5}, {2, 0.6}, {3, 0.7}, {4, 0.8}});

    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_poisson_ratio =
        Core::IO::InputField<double>(
            std::unordered_map<int, double>{{1, 0.3}, {2, 0.3}, {3, 0.3}, {4, 0.3}});
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_elasticity =
        Core::IO::InputField<double>(std::unordered_map<int, double>{
            {1, 50000.0}, {2, 51000.0}, {3, 52000.0}, {4, 53000.0}});
    params.lung_tree.airways.wall_model.kelvin_voigt.elasticity.wall_thickness =
        Core::IO::InputField<double>(
            std::unordered_map<int, double>{{1, 0.001}, {2, 0.001}, {3, 0.001}, {4, 0.001}});
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_time_constant =
        Core::IO::InputField<double>(
            std::unordered_map<int, double>{{1, 0.01}, {2, 0.01}, {3, 0.01}, {4, 0.01}});
    params.lung_tree.airways.wall_model.kelvin_voigt.viscosity.viscous_phase_shift =
        Core::IO::InputField<double>(
            std::unordered_map<int, double>{{1, 0.0}, {2, 0.0}, {3, 0.0}, {4, 0.0}});

    return params;
  }

  void assign_model_evaluators(AirwayModel& model)
  {
    model.internal_state_updater = WallMechanics::make_internal_state_updater(
        model.wall_model, FlowResistance::make_internal_state_updater(model.flow_model));
    model.residual_evaluator =
        WallMechanics::make_residual_evaluator(model.wall_model, model.flow_model);
    model.jacobian_evaluator =
        WallMechanics::make_jacobian_evaluator(model.wall_model, model.flow_model);
  }

  TEST(AirwayTests, JacobianVsFiniteDifferenceRigidWall)
  {
    AirwayContainer airways;
    AirwayModel model;

    // Create small airway data for testing (3 elements)
    model.data.global_element_id = {0, 1, 2};
    model.data.local_element_id = {0, 1, 2};
    model.data.local_row_id = {0, 1, 2};
    model.data.gid_p1 = {0, 1, 2};
    model.data.gid_p2 = {3, 4, 5};
    model.data.gid_q1 = {6, 7, 8};
    model.data.gid_q2 = {6, 7, 8};
    model.data.lid_p1 = {0, 1, 2};
    model.data.lid_p2 = {3, 4, 5};
    model.data.lid_q1 = {6, 7, 8};
    model.data.lid_q2 = {6, 7, 8};
    model.data.ref_length = {1.0, 1.0, 1.0};
    model.data.ref_area = {0.5, 0.5, 0.5};
    model.data.n_state_equations = 1;
    model.data.air_properties.dynamic_viscosity = 1.79105e-05;
    model.data.air_properties.density = 1.176e-06;
    model.data.q1_n = {1.0, 1.0, 1.0};
    model.data.q2_n = {0.0, 0.0, 0.0};
    airways.models.push_back(model);

    // We'll reuse the same model structure and swap evaluators per case below.
    // Prepare maps used by both cases
    const auto dof_map =
        create_domain_map(MPI_COMM_WORLD, airways, TerminalUnits::TerminalUnitContainer{});
    const auto row_map =
        create_row_map(MPI_COMM_WORLD, airways, TerminalUnits::TerminalUnitContainer{}, {}, {}, {});
    const auto col_map =
        create_column_map(MPI_COMM_WORLD, airways, TerminalUnits::TerminalUnitContainer{},
            {{0, 3}, {1, 3}, {2, 3}}, {{0, 0}, {1, 3}, {2, 6}}, {}, {}, {});

    Vector<double> dofs(dof_map, true);
    Vector<double> locally_relevant_dofs(col_map, true);
    dofs.replace_local_values(9, std::array<double, 9>{1, 1, 1, 1, 1, 1, 100, 50, 10}.data(),
        std::array<int, 9>{0, 1, 2, 3, 4, 5, 6, 7, 8}.data());
    export_to(dofs, locally_relevant_dofs);

    double dt = 1e-1;
    const double eps = 1e-6;
    {
      SCOPED_TRACE("Linear resistive (no inertia)");

      SparseMatrix jac(row_map, col_map, 3);
      model.flow_model = LinearResistive{.has_inertia = std::vector<bool>{false, false, false}};
      model.wall_model = RigidWall{};

      assign_model_evaluators(model);

      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
    {
      SCOPED_TRACE("Linear resistive (with inertia)");
      SparseMatrix jac(row_map, col_map, 3);

      model.flow_model = LinearResistive{.has_inertia = std::vector<bool>{true, true, true}};
      // Wall model: rigid
      model.wall_model = RigidWall{};

      assign_model_evaluators(model);

      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
    {
      SCOPED_TRACE("Nonlinear resistive (no inertia)");

      SparseMatrix jac(row_map, col_map, 3);

      model.flow_model =
          NonLinearResistive{.turbulence_factor_gamma = std::vector<double>{0.6, 0.4, 0.2},
              .has_inertia = std::vector<bool>{false, false, false},
              .k_turb = std::vector<double>{2.0, 1.0, 1.0}};
      // Wall model: rigid
      model.wall_model = RigidWall{};

      assign_model_evaluators(model);

      model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
    {
      SCOPED_TRACE("Nonlinear resistive (with inertia)");
      SparseMatrix jac(row_map, col_map, 3);

      model.flow_model =
          NonLinearResistive{.turbulence_factor_gamma = std::vector<double>{0.6, 0.4, 0.2},
              .has_inertia = std::vector<bool>{true, true, true},
              .k_turb = std::vector<double>{2.0, 1.0, 1.0}};
      // Wall model: rigid
      model.wall_model = RigidWall{};

      assign_model_evaluators(model);

      model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
  }

  TEST(AirwayTests, JacobianVsFiniteDifferenceKelvinVoigtWall)
  {
    AirwayContainer airways;
    AirwayModel model;

    // Create small airway data for testing (3 elements)
    model.data.global_element_id = {0, 1, 2};
    model.data.local_element_id = {0, 1, 2};
    model.data.local_row_id = {0, 2, 4};
    model.data.gid_p1 = {0, 1, 2};
    model.data.gid_p2 = {3, 4, 5};
    model.data.gid_q1 = {6, 7, 8};
    model.data.gid_q2 = {9, 10, 11};
    model.data.lid_p1 = {0, 1, 2};
    model.data.lid_p2 = {3, 4, 5};
    model.data.lid_q1 = {6, 7, 8};
    model.data.lid_q2 = {9, 10, 11};
    model.data.ref_length = {1.0, 1.0, 1.0};
    model.data.ref_area = {0.5, 0.5, 0.5};
    model.data.n_state_equations = 2;
    model.data.air_properties.dynamic_viscosity = 1.79105e-05;
    model.data.air_properties.density = 1.176e-06;
    model.data.q1_n = {1.0, 1.0, 1.0};
    model.data.q2_n = {0.9, 0.9, 0.9};
    model.data.p1_n = {1.0, 1.0, 1.0};
    model.data.p2_n = {1.0, 1.0, 1.0};
    airways.models.push_back(model);

    // We'll reuse the same model structure and swap evaluators per case below.
    // Prepare maps used by both cases
    const auto dof_map =
        create_domain_map(MPI_COMM_WORLD, airways, TerminalUnits::TerminalUnitContainer{});
    const auto row_map =
        create_row_map(MPI_COMM_WORLD, airways, TerminalUnits::TerminalUnitContainer{}, {}, {}, {});
    const auto col_map =
        create_column_map(MPI_COMM_WORLD, airways, TerminalUnits::TerminalUnitContainer{},
            {{0, 4}, {1, 4}, {2, 4}}, {{0, 0}, {1, 4}, {2, 8}}, {}, {}, {});

    Vector<double> dofs(dof_map, true);
    Vector<double> locally_relevant_dofs(col_map, true);
    dofs.replace_local_values(12,
        std::array<double, 12>{1, 1, 1, 1, 1, 1, 100, 50, 10, 101, 52, 9}.data(),
        std::array<int, 12>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}.data());
    export_to(dofs, locally_relevant_dofs);

    double dt = 1e-1;
    const double eps = 1e-6;
    model.wall_model = KelvinVoigtWall{.wall_poisson_ratio = std::vector<double>{0.3, 0.3, 0.3},
        .wall_elasticity = std::vector<double>{50000.0, 50000.0, 50000.0},
        .wall_thickness = std::vector<double>{0.001, 0.001, 0.001},
        .viscous_time_constant = std::vector<double>{0.01, 0.01, 0.01},
        .viscous_phase_shift = std::vector<double>{0.0, 0.0, 0.0},
        .area_n = std::vector<double>{0.5, 0.5, 0.5},
        .area = std::vector<double>(3, 1.0),
        .viscous_resistance_Rvisc = std::vector<double>(3, 1.0),
        .compliance_C = std::vector<double>(3, 1.0),
        .gamma_w = std::vector<double>(3, 1.0),
        .beta_w = std::vector<double>(3, 1.0)};

    {
      SCOPED_TRACE("Linear resistive (no inertia) - KV wall model");
      SparseMatrix jac(row_map, col_map, 4);

      model.flow_model = LinearResistive{
          .has_inertia = std::vector<bool>{false, false, false},
      };

      assign_model_evaluators(model);
      model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q2, 3, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
    {
      SCOPED_TRACE("Linear resistive (with inertia) - KV wall model");
      SparseMatrix jac(row_map, col_map, 4);

      model.flow_model = LinearResistive{
          .has_inertia = std::vector<bool>{true, true, true},
      };

      assign_model_evaluators(model);
      model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q2, 3, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
    {
      SCOPED_TRACE("Nonlinear resistive (no inertia) - KV wall model");
      SparseMatrix jac(row_map, col_map, 4);

      model.flow_model =
          NonLinearResistive{.turbulence_factor_gamma = std::vector<double>{0.6, 0.4, 0.2},
              .has_inertia = std::vector<bool>{false, false, false},
              .k_turb = std::vector<double>{2.0, 1.0, 1.0}};

      assign_model_evaluators(model);
      model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q2, 3, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
    {
      SCOPED_TRACE("Nonlinear resistive (with inertia) - KV wall model");
      SparseMatrix jac(row_map, col_map, 4);

      model.flow_model =
          NonLinearResistive{.turbulence_factor_gamma = std::vector<double>{0.6, 0.4, 0.2},
              .has_inertia = std::vector<bool>{true, true, true},
              .k_turb = std::vector<double>{2.0, 1.0, 1.0}};

      assign_model_evaluators(model);
      model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q1, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
      TestUtils::check_jacobian_column_against_fd(
          model.data.lid_q2, 3, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
  }

  TEST(AirwayModelRegistryTest, ThrowsOnUnknownFlowType)
  {
    AirwayContainer airways;
    ReducedLungParameters params{};
    EXPECT_THROW(Airways::ModelRegistry::add_airway_with_model_selection(airways, 0, 0, 1.0, params,
                     static_cast<FlowModelType>(-1), WallModelType::Rigid),
        Core::Exception);
  }

  TEST(AirwayModelRegistryTest, ThrowsOnUnknownWallType)
  {
    AirwayContainer airways;
    ReducedLungParameters params{};
    EXPECT_THROW(Airways::ModelRegistry::add_airway_with_model_selection(airways, 0, 0, 1.0, params,
                     FlowModelType::Linear, static_cast<WallModelType>(-1)),
        Core::Exception);
  }

  TEST(AirwayModelRegistryTest, CreatesAndReusesAllModelCombinations)
  {
    AirwayContainer airways;
    const auto params = make_airway_registry_parameters();

    EXPECT_EQ(Airways::ModelRegistry::add_airway_with_model_selection(
                  airways, 0, 0, 1.0, params, FlowModelType::Linear, WallModelType::Rigid),
        1);
    EXPECT_EQ(Airways::ModelRegistry::add_airway_with_model_selection(
                  airways, 1, 1, 1.0, params, FlowModelType::Linear, WallModelType::KelvinVoigt),
        2);
    EXPECT_EQ(Airways::ModelRegistry::add_airway_with_model_selection(
                  airways, 2, 2, 1.0, params, FlowModelType::NonLinear, WallModelType::Rigid),
        1);
    EXPECT_EQ(Airways::ModelRegistry::add_airway_with_model_selection(
                  airways, 3, 3, 1.0, params, FlowModelType::NonLinear, WallModelType::KelvinVoigt),
        2);

    ASSERT_EQ(airways.models.size(), 4u);

    bool saw_linear_rigid = false;
    bool saw_linear_kv = false;
    bool saw_nonlinear_rigid = false;
    bool saw_nonlinear_kv = false;
    for (const auto& model : airways.models)
    {
      if (std::holds_alternative<LinearResistive>(model.flow_model) &&
          std::holds_alternative<RigidWall>(model.wall_model))
      {
        saw_linear_rigid = true;
        EXPECT_EQ(model.data.n_state_equations, 1);
        EXPECT_EQ(model.data.number_of_elements(), 1u);
      }
      else if (std::holds_alternative<LinearResistive>(model.flow_model) &&
               std::holds_alternative<KelvinVoigtWall>(model.wall_model))
      {
        saw_linear_kv = true;
        EXPECT_EQ(model.data.n_state_equations, 2);
        EXPECT_EQ(model.data.number_of_elements(), 1u);
      }
      else if (std::holds_alternative<NonLinearResistive>(model.flow_model) &&
               std::holds_alternative<RigidWall>(model.wall_model))
      {
        saw_nonlinear_rigid = true;
        EXPECT_EQ(model.data.n_state_equations, 1);
        EXPECT_EQ(model.data.number_of_elements(), 1u);
      }
      else if (std::holds_alternative<NonLinearResistive>(model.flow_model) &&
               std::holds_alternative<KelvinVoigtWall>(model.wall_model))
      {
        saw_nonlinear_kv = true;
        EXPECT_EQ(model.data.n_state_equations, 2);
        EXPECT_EQ(model.data.number_of_elements(), 1u);
      }
    }
    EXPECT_TRUE(saw_linear_rigid);
    EXPECT_TRUE(saw_linear_kv);
    EXPECT_TRUE(saw_nonlinear_rigid);
    EXPECT_TRUE(saw_nonlinear_kv);

    EXPECT_EQ(Airways::ModelRegistry::add_airway_with_model_selection(
                  airways, 0, 4, 1.0, params, FlowModelType::Linear, WallModelType::Rigid),
        1);
    ASSERT_EQ(airways.models.size(), 4u);

    int linear_rigid_element_count = 0;
    for (const auto& model : airways.models)
    {
      if (std::holds_alternative<LinearResistive>(model.flow_model) &&
          std::holds_alternative<RigidWall>(model.wall_model))
      {
        linear_rigid_element_count = static_cast<int>(model.data.number_of_elements());
      }
    }
    EXPECT_EQ(linear_rigid_element_count, 2);
  }
}  // namespace
