// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_SOLVER_PROFILES_HPP
#define FOUR_C_REDUCED_LUNG_SOLVER_PROFILES_HPP

#include "4C_config.hpp"

#include <cstdint>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  /**
   * @brief Optional timing counters for the custom reduced-lung Newton solver.
   */
  struct NewtonSolverProfile
  {
    double total_solve_time = 0.0;                  ///< Accumulated nonlinear solve time.
    double state_sync_time = 0.0;                   ///< Time spent synchronizing model state.
    double residual_assembly_time = 0.0;            ///< Total residual assembly time.
    double residual_clear_time = 0.0;               ///< Time spent clearing the residual vector.
    double residual_airway_time = 0.0;              ///< Airway residual assembly time.
    double residual_terminal_unit_time = 0.0;       ///< Terminal-unit residual assembly time.
    double residual_junction_time = 0.0;            ///< Junction residual assembly time.
    double residual_boundary_condition_time = 0.0;  ///< Boundary-condition residual time.
    double residual_other_time = 0.0;               ///< Residual assembly time without phase label.
    double residual_norm_time = 0.0;                ///< Time spent computing residual norms.
    double sparse_jacobian_assembly_time = 0.0;     ///< Sparse Jacobian assembly time.
    double sparse_jacobian_complete_time = 0.0;     ///< Sparse Jacobian completion time.
    double structured_tree_linearization_assembly_time = 0.0;  ///< Structured block assembly time.
    double tree_linearization_preparation_time =
        0.0;  ///< Time spent preparing reusable tree-linearization data.
    double tree_linearization_airway_time = 0.0;  ///< Airway tree-linearization assembly time.
    double tree_linearization_terminal_unit_time = 0.0;       ///< Terminal-unit tree assembly time.
    double tree_linearization_junction_time = 0.0;            ///< Junction tree-linearization time.
    double tree_linearization_boundary_condition_time = 0.0;  ///< Boundary-condition tree time.
    double tree_linearization_other_time = 0.0;  ///< Tree-linearization time without phase label.
    double tree_linearization_solver_update_time = 0.0;  ///< Time spent passing blocks to solver.
    double linear_solve_time = 0.0;                      ///< Accumulated correction solve time.
    unsigned int solve_count = 0;                        ///< Number of nonlinear solves.
    unsigned int residual_evaluation_count = 0;          ///< Number of residual evaluations.
    unsigned int last_nonlinear_iterations = 0;          ///< Iteration count from the last solve.
    std::vector<double> last_residual_norms;             ///< Residual norms from the last solve.
    std::vector<double> last_increment_norms;            ///< Increment norms from the last solve.
  };

  /**
   * @brief Optional timing counters for the serial tree Newton correction solver.
   */
  struct TreeNewtonLinearSolverProfile
  {
    double total_solve_time = 0.0;               ///< Accumulated tree correction solve time.
    double bottom_up_time = 0.0;                 ///< Time spent in bottom-up condensation.
    double top_down_time = 0.0;                  ///< Time spent in top-down correction recovery.
    double dense_solve_time = 0.0;               ///< Time spent solving local dense systems.
    double coefficient_lookup_time = 0.0;        ///< Time spent reading linearization coefficients.
    unsigned int solve_count = 0;                ///< Number of tree correction solves.
    std::uint64_t dense_solve_count = 0;         ///< Number of local dense systems solved.
    std::uint64_t coefficient_lookup_count = 0;  ///< Number of coefficient lookups.
    std::uint64_t simd_group_count = 0;          ///< Number of SIMD batches processed.
    std::uint64_t simd_lane_count = 0;           ///< Number of SIMD lanes processed.
    std::uint64_t scalar_group_count = 0;        ///< Number of scalar groups processed.
    std::uint64_t scalar_tail_lane_count = 0;    ///< Number of scalar tail lanes processed.
    std::uint64_t dense_fallback_count = 0;      ///< Number of dense fallback solves.
    std::uint64_t unsupported_block_fallback_count = 0;  ///< Unsupported block fallback count.
    std::uint64_t element_count = 0;           ///< Number of elements represented in solves.
    std::uint64_t total_local_block_dofs = 0;  ///< Total local dense block size over elements.
    int max_local_block_size = 0;              ///< Maximum local dense block size.
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
