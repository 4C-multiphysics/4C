// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_NEWTON_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_NEWTON_SOLVER_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_newton_linear_solver.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  struct NewtonSolverProfile;

  /**
   * @brief Context bundling all objects required by @ref NewtonSolver.
   */
  struct NewtonSolverContext
  {
    const ReducedLungParameters::Dynamics& dynamics;    ///< Nonlinear/timestep solver parameters.
    std::shared_ptr<NewtonLinearSolver> linear_solver;  ///< Newton correction linear solver.
    const ReducedLungAssemblyPipeline& assembly_pipeline;  ///< Ordered model assembly callbacks.
    Core::LinAlg::Vector<double>& dofs;                    ///< Owned dof vector.
    Core::LinAlg::Vector<double>& locally_relevant_dofs;   ///< Ghosted dof vector.
    Core::LinAlg::Vector<double>& x;                       ///< Nonlinear solution vector.
    NewtonSolverProfile* profile = nullptr;                ///< Optional benchmark profile sink.
  };

  /**
   * @brief Full-step Newton solver for reduced-lung nonlinear systems.
   *
   * This solver is intentionally parallel to @ref NoxSolver. It reuses the reduced-lung assembly
   * pipeline, but owns the nonlinear iteration loop and delegates correction solves to a
   * @ref NewtonLinearSolver backend.
   */
  class NewtonSolver
  {
   public:
    /**
     * @brief Construct a custom Newton solver for reduced-lung systems.
     *
     * @param context Solver setup context including dynamics, Newton linear solver, assembly
     * pipeline callbacks, and all bound vectors/matrices.
     * @param initial_time Initial time for the simulation.
     */
    NewtonSolver(const NewtonSolverContext& context, double initial_time = 0.0);

    /**
     * @brief Disable copying for referenced external state.
     */
    NewtonSolver(const NewtonSolver&) = delete;

    /**
     * @brief Disable copy assignment for referenced external state.
     */
    NewtonSolver& operator=(const NewtonSolver&) = delete;

    /**
     * @brief Disable moving for referenced external state.
     */
    NewtonSolver(NewtonSolver&&) = delete;

    /**
     * @brief Disable move assignment for referenced external state.
     */
    NewtonSolver& operator=(NewtonSolver&&) = delete;

    /**
     * @brief Destroy the Newton solver.
     */
    ~NewtonSolver() = default;

    /**
     * @brief Solve the nonlinear system at the given physical time.
     *
     * @param time Current physical time for time-dependent boundary conditions.
     * @return Number of Newton corrections applied.
     */
    unsigned int solve(double time);

    /**
     * @brief Final residual norm from the most recent nonlinear solve.
     *
     * @return Euclidean norm of the last assembled residual.
     */
    [[nodiscard]] double last_residual_norm() const { return last_residual_norm_; }

   private:
    /**
     * @brief Export a trial solution to owned and locally relevant dof vectors and update models.
     *
     * @param x Current nonlinear trial solution on the Newton correction map.
     */
    void sync_state_from_x(const Core::LinAlg::Vector<double>& x);

    /**
     * @brief Assemble the residual vector for the currently synchronized reduced-lung state.
     *
     * @return Euclidean norm of the assembled residual.
     */
    double assemble_residual_for_current_state();

    /**
     * @brief Assemble and complete the sparse Jacobian for sparse Newton correction solves.
     */
    void assemble_jacobian_for_current_state();

    /**
     * @brief Assemble structured tree-linearization coefficients for tree Newton correction solves.
     */
    void assemble_tree_linearization_for_current_state();

    /**
     * @brief Solve one linear Newton correction and return the correction norm.
     *
     * @param iteration Current nonlinear Newton iteration index.
     * @return Euclidean norm of the computed correction vector.
     */
    double solve_linear_correction(unsigned int iteration);

    Core::LinAlg::Vector<double>& x_solution_;
    Core::LinAlg::Vector<double>& dofs_;
    Core::LinAlg::Vector<double>& locally_relevant_dofs_;
    ReducedLungAssemblyPipeline assembly_pipeline_;
    Core::LinAlg::SparseMatrix* sparse_jacobian_ = nullptr;

    Core::LinAlg::Vector<double> residual_;
    Core::LinAlg::Vector<double> delta_;
    TreeLinearization tree_linearization_;
    bool tree_linearization_capacity_initialized_ = false;
    bool tree_linearization_static_initialized_ = false;

    double dt_;
    double current_time_;
    double last_residual_norm_ = 0.0;
    unsigned int max_nonlinear_iterations_;
    double nonlinear_residual_tolerance_;
    double nonlinear_increment_tolerance_;

    std::shared_ptr<NewtonLinearSolver> linear_solver_;
    NewtonSolverProfile* profile_ = nullptr;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
