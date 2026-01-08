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
#include "4C_reduced_lung_test_utils_test.hpp"

namespace
{
  using namespace FourC::ReducedLung;
  using namespace FourC::Core::LinAlg;
  using namespace FourC::ReducedLung::TestUtils;

  // Tests the implementation of different model combinations by comparing the analytic jacobian
  // with a finite difference approximation of the residual functions.
  TEST(TerminalUnitTests, JacobianVsFiniteDifference)
  {
    TerminalUnitContainer terminal_units;
    TerminalUnitModel model;

    // Artificial terminal unit data for evaluating the model equations
    model.data = TerminalUnitData{std::vector<int>{0, 1, 2}, std::vector<int>{0, 1, 2},
        std::vector<int>{0, 1, 2}, std::vector<int>{0, 3, 6}, std::vector<int>{1, 4, 7},
        std::vector<int>{2, 5, 8}, std::vector<int>{0, 3, 6}, std::vector<int>{1, 4, 7},
        std::vector<int>{2, 5, 8}, std::vector<double>{1.0, 5.0, 150.0},
        std::vector<double>{1.0, 10.0, 100.0}};
    terminal_units.models.push_back(model);

    // Maps for the system of equations
    const auto dof_map = create_domain_map(MPI_COMM_WORLD, {}, terminal_units);
    const auto row_map = create_row_map(MPI_COMM_WORLD, {}, terminal_units, {}, {}, {});
    const auto col_map = create_column_map(MPI_COMM_WORLD, {}, terminal_units,
        {{0, 3}, {1, 3}, {2, 3}}, {{0, 0}, {1, 3}, {2, 6}}, {}, {}, {});

    // Artificial dof vector
    Vector<double> dofs(dof_map, true);
    Vector<double> locally_relevant_dofs(col_map, true);
    dofs.replace_local_values(9, std::array<double, 9>{1, 1, 1, 1, 1, 1, 1, 1, 1}.data(),
        std::array<int, 9>{0, 1, 2, 3, 4, 5, 6, 7, 8}.data());
    export_to(dofs, locally_relevant_dofs);

    double dt = 1e-1;         // Dummy time step size
    const double eps = 1e-6;  // Perturbation parameter for the FD approximation

    {
      SCOPED_TRACE("Kelvin Voigt + Linear Elasticity");
      SparseMatrix jac(row_map, col_map, 3);
      model.rheological_model = KelvinVoigt{std::vector<double>{0.0, 1.0, 100.0}};
      model.elasticity_model = LinearElasticity{std::vector<double>{1.0, 1.0, 0.0},
          std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{0.0, 0.0, 0.0}};
      model.internal_state_updater =
          std::visit(TerminalUnits::MakeInternalStateUpdater{}, model.rheological_model);
      model.negative_residual_evaluator =
          std::visit(TerminalUnits::MakeNegativeResidualEvaluator{model.elasticity_model},
              model.rheological_model);
      model.jacobian_evaluator = std::visit(
          TerminalUnits::MakeJacobianEvaluator{model.elasticity_model}, model.rheological_model);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

      check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }

    {
      SCOPED_TRACE("Kelvin Voigt + Ogden Hyperelasticity");
      SparseMatrix jac(row_map, col_map, 3);
      model.rheological_model = KelvinVoigt{std::vector<double>{0.0, 1.0, 100.0}};
      model.elasticity_model = OgdenHyperelasticity{std::vector<double>{1.0, 1.0, 1.0},
          std::vector<double>{5.0, -0.4, -8.0}, std::vector<double>{0.0, 0.0, 0.0},
          std::vector<double>{0.0, 0.0, 0.0}};
      model.internal_state_updater =
          std::visit(TerminalUnits::MakeInternalStateUpdater{}, model.rheological_model);
      model.negative_residual_evaluator =
          std::visit(TerminalUnits::MakeNegativeResidualEvaluator{model.elasticity_model},
              model.rheological_model);
      model.jacobian_evaluator = std::visit(
          TerminalUnits::MakeJacobianEvaluator{model.elasticity_model}, model.rheological_model);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

      check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }

    {
      SCOPED_TRACE("Four-Element Maxwell + Linear Elasticity");
      SparseMatrix jac(row_map, col_map, 3);
      model.rheological_model = FourElementMaxwell{std::vector<double>{0.0, 1.0, 100.0},
          std::vector<double>{10.0, 0.0, 20.0}, std::vector<double>{2.5, 10.0, 0.0},
          std::vector<double>{0.0, 0.0, 0.0}};
      model.elasticity_model = LinearElasticity{std::vector<double>{1.0, 1.0, 0.0},
          std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{0.0, 0.0, 0.0}};
      model.internal_state_updater =
          std::visit(TerminalUnits::MakeInternalStateUpdater{}, model.rheological_model);
      model.negative_residual_evaluator =
          std::visit(TerminalUnits::MakeNegativeResidualEvaluator{model.elasticity_model},
              model.rheological_model);
      model.jacobian_evaluator = std::visit(
          TerminalUnits::MakeJacobianEvaluator{model.elasticity_model}, model.rheological_model);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

      check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);

      jac.complete();  // Sparsity pattern already filled the first time
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }

    {
      SCOPED_TRACE("Four-Element Maxwell + Ogden Hyperelasticity");
      SparseMatrix jac(row_map, col_map, 3);
      model.rheological_model = FourElementMaxwell{std::vector<double>{10.5, 1.0, 100.0},
          std::vector<double>{10.0, 0.0, 20.0}, std::vector<double>{2.5, 10.0, 0.0},
          std::vector<double>{0.0, 0.0, 0.0}};
      model.elasticity_model = OgdenHyperelasticity{std::vector<double>{0.0, 1.0, 1.0},
          std::vector<double>{1.0, 6.4, -3.0}, std::vector<double>{0.0, 0.0, 0.0},
          std::vector<double>{0.0, 0.0, 0.0}};
      model.internal_state_updater =
          std::visit(TerminalUnits::MakeInternalStateUpdater{}, model.rheological_model);
      model.negative_residual_evaluator =
          std::visit(TerminalUnits::MakeNegativeResidualEvaluator{model.elasticity_model},
              model.rheological_model);
      model.jacobian_evaluator = std::visit(
          TerminalUnits::MakeJacobianEvaluator{model.elasticity_model}, model.rheological_model);
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

      check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
  }
}  // namespace