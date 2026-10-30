// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_NONLINEAR_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_NONLINEAR_SOLVER_HPP

#include "4C_config.hpp"

#include <functional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  class TreeNewtonLinearSolver;

  /**
   * @brief Ordered callback registry for nonlinear-solver assembly.
   *
   * The pipeline decouples solver workflows from concrete reduced-lung model blocks. Additional
   * model families can register callbacks without changing solver internals.
   */
  struct NonlinearSolverAssemblyPipeline
  {
    using ResidualAssembler = std::function<void(Core::LinAlg::Vector<double>& residual,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double current_time,
        double time_step_size_dt)>;

    using JacobianAssembler = std::function<void(Core::LinAlg::SparseMatrix& jacobian,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double current_time,
        double time_step_size_dt)>;

    /**
     * @brief Assemble state-dependent coefficients into the tree solver.
     */
    using TreeLinearizationAssembler = std::function<void(TreeNewtonLinearSolver& target,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double current_time,
        double time_step_size_dt)>;

    /**
     * @brief Assemble state-independent coefficients into the tree solver.
     */
    using StaticTreeLinearizationAssembler = std::function<void(TreeNewtonLinearSolver& target)>;

    using StateUpdater = std::function<void(
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

    std::vector<ResidualAssembler> residual_assemblers;  ///< Residual callbacks for all workflows.
    std::vector<JacobianAssembler> jacobian_assemblers;  ///< Sparse Jacobian callbacks.
    std::vector<StaticTreeLinearizationAssembler>
        tree_linearization_static_assemblers;  ///< Static structured coefficient callbacks.
    std::vector<TreeLinearizationAssembler>
        tree_linearization_assemblers;  ///< State-dependent structured coefficient callbacks.
    std::vector<StateUpdater>
        state_updaters;  ///< End-of-solve model state synchronization callbacks.
  };

  /**
   * @brief Common interface of the reduced-lung nonlinear solvers.
   *
   * The time integration only interacts with the selected solver through this interface.
   */
  class NonlinearSolver
  {
   public:
    virtual ~NonlinearSolver() = default;

    /**
     * @brief Solve the nonlinear system at the given physical time.
     *
     * @param time Current physical time for time-dependent boundary conditions.
     * @return Number of nonlinear iterations performed.
     */
    virtual unsigned int solve(double time) = 0;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
