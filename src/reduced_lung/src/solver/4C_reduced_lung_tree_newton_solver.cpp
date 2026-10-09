// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_tree_newton_solver.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_tree_linear_solver.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    /**
     * Compute the residual two-norm with a serial fast path.
     */
    double compute_residual_norm(const Core::LinAlg::Vector<double>& residual)
    {
      if (Core::Communication::num_mpi_ranks(residual.get_comm()) != 1)
      {
        double norm = 0.0;
        residual.norm_2(&norm);
        return norm;
      }

      double norm_square = 0.0;
      for (const double value : residual.local_values_as_span())
      {
        norm_square += value * value;
      }
      return std::sqrt(norm_square);
    }
  }  // namespace

  TreeNewtonSolver::TreeNewtonSolver(const TreeNewtonSolverContext& context, double initial_time)
      : x_solution_(context.x),
        dofs_(context.dofs),
        locally_relevant_dofs_(context.locally_relevant_dofs),
        assembly_pipeline_(context.assembly_pipeline),
        residual_(context.x.get_map(), true),
        delta_(context.x.get_map(), true),
        dt_(context.dynamics.time_increment),
        current_time_(initial_time),
        max_nonlinear_iterations_(
            static_cast<unsigned int>(context.dynamics.max_nonlinear_iterations)),
        nonlinear_residual_tolerance_(context.dynamics.nonlinear_residual_tolerance),
        nonlinear_increment_tolerance_(context.dynamics.nonlinear_increment_tolerance),
        tree_linear_solver_(context.tree_linear_solver)
  {
    if (context.dynamics.max_nonlinear_iterations <= 0)
    {
      FOUR_C_THROW(
          "ReducedLung::TreeNewtonSolver requires a positive max_nonlinear_iterations, got {}.",
          context.dynamics.max_nonlinear_iterations);
    }
    if (tree_linear_solver_ == nullptr)
    {
      FOUR_C_THROW("ReducedLung::TreeNewtonSolver requires a valid tree Newton linear solver.");
    }
    if (assembly_pipeline_.residual_assemblers.empty())
    {
      FOUR_C_THROW(
          "ReducedLung::TreeNewtonSolver requires at least one residual assembler callback.");
    }
    if (assembly_pipeline_.tree_linearization_static_assemblers.empty() &&
        assembly_pipeline_.tree_linearization_assemblers.empty())
    {
      FOUR_C_THROW(
          "ReducedLung::TreeNewtonSolver requires tree-linearization assemblers for structured "
          "tree "
          "linear solves.");
    }
  }

  unsigned int TreeNewtonSolver::solve(double time)
  {
    current_time_ = time;
    double increment_norm = 0.0;

    for (unsigned int iteration = 0; iteration <= max_nonlinear_iterations_; ++iteration)
    {
      sync_state_from_x(x_solution_);
      const double residual_norm = assemble_residual_for_current_state();
      last_residual_norm_ = residual_norm;
      const bool residual_converged = residual_norm <= nonlinear_residual_tolerance_;

      // The residual is the authoritative convergence check. A large first correction is expected
      // when advancing in time, even for linear systems.
      if (residual_converged)
      {
        return iteration;
      }

      if (iteration > 0 && increment_norm <= nonlinear_increment_tolerance_)
      {
        FOUR_C_THROW(
            "ReducedLung::TreeNewtonSolver stagnated at time {} after {} Newton corrections. "
            "Residual norm: {}, increment norm: {}.",
            current_time_, iteration, residual_norm, increment_norm);
      }

      if (iteration == max_nonlinear_iterations_)
      {
        FOUR_C_THROW(
            "ReducedLung::TreeNewtonSolver did not converge at time {} after {} Newton "
            "corrections. "
            "Final residual norm: {}, final increment norm: {}.",
            current_time_, max_nonlinear_iterations_, residual_norm, increment_norm);
      }

      assemble_tree_linearization_for_current_state();
      increment_norm = solve_linear_correction();
      x_solution_.update(1.0, delta_, 1.0);
    }

    FOUR_C_THROW("ReducedLung::TreeNewtonSolver reached an unreachable nonlinear-solver state.");
  }

  void TreeNewtonSolver::sync_state_from_x(const Core::LinAlg::Vector<double>& x)
  {
    if (Core::Communication::num_mpi_ranks(x.get_comm()) == 1)
    {
      dofs_.update(1.0, x, 0.0);
      locally_relevant_dofs_.update(1.0, x, 0.0);
    }
    else
    {
      Core::LinAlg::export_to(x, dofs_);
      Core::LinAlg::export_to(dofs_, locally_relevant_dofs_);
    }

    for (const auto& update_state : assembly_pipeline_.state_updaters)
    {
      update_state(locally_relevant_dofs_, dt_);
    }
  }

  double TreeNewtonSolver::assemble_residual_for_current_state()
  {
    residual_.put_scalar(0.0);

    for (const auto& assemble_residual : assembly_pipeline_.residual_assemblers)
    {
      assemble_residual(residual_, locally_relevant_dofs_, current_time_, dt_);
    }

    return compute_residual_norm(residual_);
  }

  void TreeNewtonSolver::assemble_tree_linearization_for_current_state()
  {
    if (!tree_linearization_static_initialized_)
    {
      for (const auto& tree_linearization_static_assembler :
          assembly_pipeline_.tree_linearization_static_assemblers)
      {
        tree_linearization_static_assembler(*tree_linear_solver_);
      }
      tree_linearization_static_initialized_ = true;
    }
    for (const auto& tree_linearization_assembler :
        assembly_pipeline_.tree_linearization_assemblers)
    {
      tree_linearization_assembler(
          *tree_linear_solver_, locally_relevant_dofs_, current_time_, dt_);
    }
  }

  double TreeNewtonSolver::solve_linear_correction()
  {
    tree_linear_solver_->solve(residual_, delta_);

    double increment_norm = 0.0;
    delta_.norm_2(&increment_norm);
    return increment_norm;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
