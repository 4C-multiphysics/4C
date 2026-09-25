// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_newton_solver.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_solver_profiles.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <chrono>
#include <cmath>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    using Clock = std::chrono::steady_clock;

    /**
     * Return the elapsed wall-clock time in seconds since @p start.
     */
    double elapsed_seconds(const Clock::time_point start)
    {
      return std::chrono::duration<double>(Clock::now() - start).count();
    }

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

    /**
     * Accumulate structured tree-linearization timing into the matching profile bucket.
     */
    void add_tree_linearization_phase_time(NewtonSolverProfile& profile,
        ReducedLungAssemblyPipeline::TreeLinearizationAssemblyPhase phase, double elapsed_time)
    {
      using Phase = ReducedLungAssemblyPipeline::TreeLinearizationAssemblyPhase;
      switch (phase)
      {
        case Phase::Airways:
          profile.tree_linearization_airway_time += elapsed_time;
          break;
        case Phase::TerminalUnits:
          profile.tree_linearization_terminal_unit_time += elapsed_time;
          break;
        case Phase::Junctions:
          profile.tree_linearization_junction_time += elapsed_time;
          break;
        case Phase::BoundaryConditions:
          profile.tree_linearization_boundary_condition_time += elapsed_time;
          break;
        case Phase::Other:
          profile.tree_linearization_other_time += elapsed_time;
          break;
      }
    }

    /**
     * Accumulate residual assembly timing into the matching profile bucket.
     */
    void add_residual_phase_time(NewtonSolverProfile& profile,
        ReducedLungAssemblyPipeline::TreeLinearizationAssemblyPhase phase, double elapsed_time)
    {
      using Phase = ReducedLungAssemblyPipeline::TreeLinearizationAssemblyPhase;
      switch (phase)
      {
        case Phase::Airways:
          profile.residual_airway_time += elapsed_time;
          break;
        case Phase::TerminalUnits:
          profile.residual_terminal_unit_time += elapsed_time;
          break;
        case Phase::Junctions:
          profile.residual_junction_time += elapsed_time;
          break;
        case Phase::BoundaryConditions:
          profile.residual_boundary_condition_time += elapsed_time;
          break;
        case Phase::Other:
          profile.residual_other_time += elapsed_time;
          break;
      }
    }
  }  // namespace

  NewtonSolver::NewtonSolver(const NewtonSolverContext& context, double initial_time)
      : x_solution_(context.x),
        dofs_(context.dofs),
        locally_relevant_dofs_(context.locally_relevant_dofs),
        assembly_pipeline_(context.assembly_pipeline),
        residual_(context.x.get_map(), true),
        delta_(context.x.get_map(), true),
        tree_linearization_(context.x.local_length(), context.locally_relevant_dofs.local_length()),
        dt_(context.dynamics.time_increment),
        current_time_(initial_time),
        max_nonlinear_iterations_(
            static_cast<unsigned int>(context.dynamics.max_nonlinear_iterations)),
        nonlinear_residual_tolerance_(context.dynamics.nonlinear_residual_tolerance),
        nonlinear_increment_tolerance_(context.dynamics.nonlinear_increment_tolerance),
        linear_solver_(context.linear_solver),
        profile_(context.profile)
  {
    if (context.dynamics.max_nonlinear_iterations <= 0)
    {
      FOUR_C_THROW(
          "ReducedLung::NewtonSolver requires a positive max_nonlinear_iterations, got {}.",
          context.dynamics.max_nonlinear_iterations);
    }
    if (linear_solver_ == nullptr)
    {
      FOUR_C_THROW("ReducedLung::NewtonSolver requires a valid Newton linear solver instance.");
    }
    sparse_jacobian_ = linear_solver_->sparse_jacobian_target();
    if (assembly_pipeline_.residual_assemblers.empty() &&
        assembly_pipeline_.named_residual_assemblers.empty())
    {
      FOUR_C_THROW("ReducedLung::NewtonSolver requires at least one residual assembler callback.");
    }
    if (linear_solver_->linearization_type() == NewtonLinearizationType::SparseJacobian &&
        sparse_jacobian_ == nullptr)
    {
      FOUR_C_THROW(
          "ReducedLung::NewtonSolver requires a sparse Jacobian target for sparse linear "
          "solves.");
    }
    if (linear_solver_->linearization_type() == NewtonLinearizationType::SparseJacobian &&
        assembly_pipeline_.jacobian_assemblers.empty())
    {
      FOUR_C_THROW("ReducedLung::NewtonSolver requires at least one Jacobian assembler callback.");
    }
    if (linear_solver_->linearization_type() == NewtonLinearizationType::StructuredTreeBlocks &&
        sparse_jacobian_ != nullptr)
    {
      FOUR_C_THROW(
          "ReducedLung::NewtonSolver structured tree solves must not provide a sparse "
          "Jacobian target.");
    }
    if (linear_solver_->linearization_type() == NewtonLinearizationType::StructuredTreeBlocks &&
        assembly_pipeline_.tree_linearization_static_assemblers.empty() &&
        assembly_pipeline_.tree_linearization_assemblers.empty())
    {
      FOUR_C_THROW(
          "ReducedLung::NewtonSolver requires tree-linearization assemblers for structured tree "
          "linear solves.");
    }
  }

  unsigned int NewtonSolver::solve(double time)
  {
    const auto solve_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    current_time_ = time;
    double increment_norm = 0.0;
    if (profile_ != nullptr)
    {
      profile_->last_residual_norms.clear();
      profile_->last_increment_norms.clear();
    }

    for (unsigned int iteration = 0; iteration <= max_nonlinear_iterations_; ++iteration)
    {
      const auto sync_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
      sync_state_from_x(x_solution_);
      if (profile_ != nullptr)
      {
        profile_->state_sync_time += elapsed_seconds(sync_start);
      }
      const double residual_norm = assemble_residual_for_current_state();
      last_residual_norm_ = residual_norm;
      if (profile_ != nullptr)
      {
        profile_->last_residual_norms.push_back(residual_norm);
      }
      const bool residual_converged = residual_norm <= nonlinear_residual_tolerance_;

      // The residual is the authoritative convergence check. A large first correction is expected
      // when advancing in time, even for linear systems.
      if (residual_converged)
      {
        if (profile_ != nullptr)
        {
          profile_->total_solve_time += elapsed_seconds(solve_start);
          profile_->last_nonlinear_iterations = iteration;
          ++profile_->solve_count;
        }
        return iteration;
      }

      if (iteration > 0 && increment_norm <= nonlinear_increment_tolerance_)
      {
        FOUR_C_THROW(
            "ReducedLung::NewtonSolver stagnated at time {} after {} Newton corrections. "
            "Residual norm: {}, increment norm: {}.",
            current_time_, iteration, residual_norm, increment_norm);
      }

      if (iteration == max_nonlinear_iterations_)
      {
        FOUR_C_THROW(
            "ReducedLung::NewtonSolver did not converge at time {} after {} Newton corrections. "
            "Final residual norm: {}, final increment norm: {}.",
            current_time_, max_nonlinear_iterations_, residual_norm, increment_norm);
      }

      if (linear_solver_->linearization_type() == NewtonLinearizationType::StructuredTreeBlocks)
      {
        assemble_tree_linearization_for_current_state();
      }
      else
      {
        assemble_jacobian_for_current_state();
      }
      increment_norm = solve_linear_correction(iteration);
      if (profile_ != nullptr)
      {
        profile_->last_increment_norms.push_back(increment_norm);
      }
      x_solution_.update(1.0, delta_, 1.0);
    }

    FOUR_C_THROW("ReducedLung::NewtonSolver reached an unreachable nonlinear-solver state.");
  }

  void NewtonSolver::sync_state_from_x(const Core::LinAlg::Vector<double>& x)
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

  double NewtonSolver::assemble_residual_for_current_state()
  {
    const auto clear_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    residual_.put_scalar(0.0);
    if (profile_ != nullptr)
    {
      profile_->residual_clear_time += elapsed_seconds(clear_start);
    }

    const auto assembly_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    if (!assembly_pipeline_.named_residual_assemblers.empty())
    {
      for (const auto& assemble_residual : assembly_pipeline_.named_residual_assemblers)
      {
        const auto phase_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        assemble_residual.callback(residual_, locally_relevant_dofs_, current_time_, dt_);
        if (profile_ != nullptr)
        {
          add_residual_phase_time(*profile_, assemble_residual.phase, elapsed_seconds(phase_start));
        }
      }
    }
    else
    {
      const auto phase_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
      for (const auto& assemble_residual : assembly_pipeline_.residual_assemblers)
      {
        assemble_residual(residual_, locally_relevant_dofs_, current_time_, dt_);
      }
      if (profile_ != nullptr)
      {
        profile_->residual_other_time += elapsed_seconds(phase_start);
      }
    }
    if (profile_ != nullptr)
    {
      profile_->residual_assembly_time += elapsed_seconds(assembly_start);
      ++profile_->residual_evaluation_count;
    }

    double residual_norm = 0.0;
    const auto norm_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    residual_norm = compute_residual_norm(residual_);
    if (profile_ != nullptr)
    {
      profile_->residual_norm_time += elapsed_seconds(norm_start);
    }
    return residual_norm;
  }

  void NewtonSolver::assemble_jacobian_for_current_state()
  {
    FOUR_C_ASSERT_ALWAYS(
        sparse_jacobian_ != nullptr, "Sparse Jacobian assembly requires a matrix target.");
    const auto assembly_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    for (const auto& assemble_jacobian : assembly_pipeline_.jacobian_assemblers)
    {
      assemble_jacobian(*sparse_jacobian_, locally_relevant_dofs_, current_time_, dt_);
    }
    if (profile_ != nullptr)
    {
      profile_->sparse_jacobian_assembly_time += elapsed_seconds(assembly_start);
    }

    if (!sparse_jacobian_->filled())
    {
      const auto complete_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
      sparse_jacobian_->complete();
      if (profile_ != nullptr)
      {
        profile_->sparse_jacobian_complete_time += elapsed_seconds(complete_start);
      }
    }
  }

  void NewtonSolver::assemble_tree_linearization_for_current_state()
  {
    TEUCHOS_FUNC_TIME_MONITOR("ReducedLung::NewtonTree:  1)   Assemble coefficients");
    const auto assembly_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    if (TreeCoefficientAssemblyTarget* direct_target =
            linear_solver_->direct_tree_coefficient_target())
    {
      // Some optimized tree solvers assemble coefficients directly into their own SoA storage. The
      // validation/reference fallback below keeps a reusable TreeLinearization object for solvers
      // without this hook.
      const auto preparation_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
      if (!tree_linearization_static_initialized_)
      {
        for (const auto& tree_linearization_static_assembler :
            assembly_pipeline_.tree_linearization_static_assemblers)
        {
          tree_linearization_static_assembler.callback(*direct_target);
        }
        tree_linearization_static_initialized_ = true;
      }
      if (profile_ != nullptr)
      {
        profile_->tree_linearization_preparation_time += elapsed_seconds(preparation_start);
      }
      for (const auto& tree_linearization_assembler :
          assembly_pipeline_.tree_linearization_assemblers)
      {
        const auto phase_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        tree_linearization_assembler.callback(
            *direct_target, locally_relevant_dofs_, current_time_, dt_);
        if (profile_ != nullptr)
        {
          add_tree_linearization_phase_time(
              *profile_, tree_linearization_assembler.phase, elapsed_seconds(phase_start));
        }
      }
      if (profile_ != nullptr)
      {
        profile_->structured_tree_linearization_assembly_time += elapsed_seconds(assembly_start);
      }
      return;
    }

    const int num_rows = residual_.local_length();
    const int num_dofs = locally_relevant_dofs_.local_length();
    const auto preparation_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    if (tree_linearization_.num_rows() == num_rows && tree_linearization_.num_dofs() == num_dofs)
    {
      if (!tree_linearization_static_initialized_)
      {
        // Static coefficients are retained after the first assembly. Clear only the dynamic values
        // while the reusable storage still contains no static entries.
        tree_linearization_.clear_values();
      }
    }
    else
    {
      tree_linearization_.reset(num_rows, num_dofs);
      tree_linearization_capacity_initialized_ = false;
      tree_linearization_static_initialized_ = false;
    }
    if (!tree_linearization_capacity_initialized_)
    {
      for (const auto& initialize_capacity :
          assembly_pipeline_.tree_linearization_capacity_initializers)
      {
        initialize_capacity(tree_linearization_);
      }
      tree_linearization_capacity_initialized_ = true;
    }
    if (!tree_linearization_static_initialized_)
    {
      for (const auto& tree_linearization_static_assembler :
          assembly_pipeline_.tree_linearization_static_assemblers)
      {
        tree_linearization_static_assembler.callback(tree_linearization_);
      }
      tree_linearization_static_initialized_ = true;
    }
    if (profile_ != nullptr)
    {
      profile_->tree_linearization_preparation_time += elapsed_seconds(preparation_start);
    }
    for (const auto& tree_linearization_assembler :
        assembly_pipeline_.tree_linearization_assemblers)
    {
      const auto phase_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
      tree_linearization_assembler.callback(
          tree_linearization_, locally_relevant_dofs_, current_time_, dt_);
      if (profile_ != nullptr)
      {
        add_tree_linearization_phase_time(
            *profile_, tree_linearization_assembler.phase, elapsed_seconds(phase_start));
      }
    }
    const auto solver_update_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    linear_solver_->set_tree_linearization(tree_linearization_);
    if (profile_ != nullptr)
    {
      profile_->tree_linearization_solver_update_time += elapsed_seconds(solver_update_start);
      profile_->structured_tree_linearization_assembly_time += elapsed_seconds(assembly_start);
    }
  }

  double NewtonSolver::solve_linear_correction(unsigned int iteration)
  {
    const NewtonLinearSystemMetadata metadata{
        .current_time = current_time_,
        .time_step_size_dt = dt_,
        .nonlinear_iteration = iteration,
    };
    const auto linear_solve_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
    linear_solver_->solve(residual_, x_solution_, metadata, delta_);
    if (profile_ != nullptr)
    {
      profile_->linear_solve_time += elapsed_seconds(linear_solve_start);
    }

    double increment_norm = 0.0;
    delta_.norm_2(&increment_norm);
    return increment_norm;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
