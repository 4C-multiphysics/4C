// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// Reimplemented airway tests to match the style of the terminal unit tests.
// Tests only linear and nonlinear resistive flow without inertia.
#include <gtest/gtest.h>

#include "4C_reduced_lung_airways.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_reduced_lung_test_utils_test.hpp"
// needed for export_to
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"


namespace
{
  using namespace FourC::ReducedLung;
  using namespace FourC::Core::LinAlg;
  using namespace FourC::ReducedLung::Airways;


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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);

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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);

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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);

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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);

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
    model.data.local_row_id = {0, 1, 2};
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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);
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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);
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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);
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

      model.internal_state_updater =
          std::visit(Airways::MakeInternalStateUpdater{model.flow_model}, model.wall_model);
      model.negative_residual_evaluator =
          std::visit(Airways::MakeNegativeResidualEvaluator{model.flow_model}, model.wall_model);
      model.jacobian_evaluator =
          std::visit(Airways::MakeJacobianEvaluator{model.flow_model}, model.wall_model);
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
}  // namespace