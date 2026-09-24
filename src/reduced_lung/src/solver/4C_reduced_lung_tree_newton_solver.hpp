// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TREE_NEWTON_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_TREE_NEWTON_SOLVER_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_helpers.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  class TreeNewtonLinearSolver;

  /**
   * @brief Context bundling all objects required by @ref TreeNewtonSolver.
   */
  struct TreeNewtonSolverContext
  {
    const ReducedLungParameters::Dynamics& dynamics;  ///< Nonlinear/timestep solver parameters.
    std::shared_ptr<TreeNewtonLinearSolver>
        tree_linear_solver;  ///< Tree Newton correction linear solver.
    const NonlinearSolverAssemblyPipeline&
        assembly_pipeline;                                ///< Ordered model assembly callbacks.
    Core::LinAlg::Vector<double>& dofs;                   ///< Owned dof vector.
    Core::LinAlg::Vector<double>& locally_relevant_dofs;  ///< Ghosted dof vector.
    Core::LinAlg::Vector<double>& x;                      ///< Nonlinear solution vector.
  };

  /**
   * @brief Full-step Newton solver for reduced-lung tree nonlinear systems.
   *
   * This solver is intentionally parallel to @ref NoxSolver. It reuses the reduced-lung assembly
   * pipeline, but owns the nonlinear iteration loop and delegates correction solves to
   * @ref TreeNewtonLinearSolver.
   */
  class TreeNewtonSolver
  {
   public:
    /**
     * @brief Construct a NewtonTree nonlinear solver for reduced-lung systems.
     *
     * @param context Solver setup context including dynamics, tree linear solver, assembly
     * pipeline callbacks, and all bound vectors.
     * @param initial_time Initial time for the simulation.
     */
    TreeNewtonSolver(const TreeNewtonSolverContext& context, double initial_time = 0.0);

    /**
     * @brief Disable copying for referenced external state.
     */
    TreeNewtonSolver(const TreeNewtonSolver&) = delete;

    /**
     * @brief Disable copy assignment for referenced external state.
     */
    TreeNewtonSolver& operator=(const TreeNewtonSolver&) = delete;

    /**
     * @brief Disable moving for referenced external state.
     */
    TreeNewtonSolver(TreeNewtonSolver&&) = delete;

    /**
     * @brief Disable move assignment for referenced external state.
     */
    TreeNewtonSolver& operator=(TreeNewtonSolver&&) = delete;

    /**
     * @brief Destroy the tree Newton solver.
     */
    ~TreeNewtonSolver() = default;

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
     * @brief Assemble structured tree-linearization coefficients for tree Newton correction solves.
     */
    void assemble_tree_linearization_for_current_state();

    /**
     * @brief Solve one linear Newton correction and return the correction norm.
     *
     * @return Euclidean norm of the computed correction vector.
     */
    double solve_linear_correction();

    Core::LinAlg::Vector<double>& x_solution_;
    Core::LinAlg::Vector<double>& dofs_;
    Core::LinAlg::Vector<double>& locally_relevant_dofs_;
    NonlinearSolverAssemblyPipeline assembly_pipeline_;

    Core::LinAlg::Vector<double> residual_;
    Core::LinAlg::Vector<double> delta_;
    bool tree_linearization_static_initialized_ = false;

    double dt_;
    double current_time_;
    double last_residual_norm_ = 0.0;
    unsigned int max_nonlinear_iterations_;
    double nonlinear_residual_tolerance_;
    double nonlinear_increment_tolerance_;

    std::shared_ptr<TreeNewtonLinearSolver> tree_linear_solver_;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
