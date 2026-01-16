// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

// Shared test utilities for reduced_lung tests
#ifndef FOUR_C_REDUCED_LUNG_TEST_UTILS_TEST_HPP
#define FOUR_C_REDUCED_LUNG_TEST_UTILS_TEST_HPP

#include <gtest/gtest.h>

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"

#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TestUtils
{
  // Generalized helper to check a Jacobian column against a central finite-difference
  // approximation of the residual. Works for both TerminalUnitModel and AirwayModel as long
  // as the model exposes `data`, `negative_residual_evaluator` and the data structure
  // provides `number_of_elements()` and per-element lid vectors passed in `dof_lids`.
  template <typename Model>
  void check_jacobian_column_against_fd(const std::vector<int>& dof_lids, int jac_col, Model& model,
      Core::LinAlg::SparseMatrix& jac, Core::LinAlg::Vector<double>& locally_relevant_dofs,
      double dt, double eps, const Core::LinAlg::Map& row_map)
  {
    SCOPED_TRACE(
        std::string("Comparing FD approximation with Jacobian column ") + std::to_string(jac_col));

    // Perturb in +epsilon direction
    for (int lid : dof_lids) locally_relevant_dofs.get_values()[lid] += eps;
    Core::LinAlg::Vector<double> res_plus(row_map, true);
    model.internal_state_updater(model.data, locally_relevant_dofs, dt);
    model.negative_residual_evaluator(model.data, res_plus, locally_relevant_dofs, dt);

    // Perturb in -epsilon direction
    for (int lid : dof_lids) locally_relevant_dofs.get_values()[lid] -= 2 * eps;
    Core::LinAlg::Vector<double> res_minus(row_map, true);
    model.internal_state_updater(model.data, locally_relevant_dofs, dt);
    model.negative_residual_evaluator(model.data, res_minus, locally_relevant_dofs, dt);

    // Restore original state
    for (int lid : dof_lids) locally_relevant_dofs.get_values()[lid] += eps;

    // Compute FD approximation
    Core::LinAlg::Vector<double> fd_derivative(row_map, true);
    fd_derivative.update(-1.0 / (2 * eps), res_plus, 1.0 / (2 * eps), res_minus, 0.0);

    // Compare with Jacobian column
    for (size_t i = 0; i < model.data.number_of_elements(); ++i)
    {
      int n_entries = 0;
      double* jac_vals = nullptr;
      int* col_indices = nullptr;
      jac.extract_my_row_view(i, n_entries, jac_vals, col_indices);

      ASSERT_EQ(col_indices[jac_col], dof_lids[i]);
      EXPECT_NEAR(jac_vals[jac_col], fd_derivative.local_values_as_span()[i], eps)
          << "Mismatch at row " << i << ", col " << jac_col;
    }
  }
}  // namespace ReducedLung::TestUtils

FOUR_C_NAMESPACE_CLOSE

#endif
