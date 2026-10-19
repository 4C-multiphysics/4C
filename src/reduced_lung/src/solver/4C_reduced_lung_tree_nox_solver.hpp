// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TREE_NOX_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_TREE_NOX_SOLVER_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_nonlinear_solver.hpp"

#include <mpi.h>

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
   * @brief Context bundling all objects required by @ref TreeNoxSolver.
   */
  struct TreeNoxSolverContext
  {
    MPI_Comm comm;                                    ///< MPI communicator used by NOX.
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
   * @brief NOX-driven Newton solver using the structured-tree linear solver for corrections.
   *
   * PROTOTYPE: drives the reduced-lung tree linearization with NOX's line-search based Newton
   * method instead of the custom loop in @ref TreeNewtonSolver. Like @ref TreeNewtonSolver it
   * converges on the residual norm alone. It fails on a non-finite residual or on exceeding the
   * maximum number of nonlinear iterations; the increment tolerance is not used.
   */
  class TreeNoxSolver final : public NonlinearSolver
  {
   public:
    /**
     * @brief Construct the NOX stack around the tree Newton linear solver.
     *
     * @param context Solver setup context.
     * @param initial_time Initial time for the simulation.
     */
    TreeNoxSolver(const TreeNoxSolverContext& context, double initial_time = 0.0);

    TreeNoxSolver(const TreeNoxSolver&) = delete;
    TreeNoxSolver& operator=(const TreeNoxSolver&) = delete;
    TreeNoxSolver(TreeNoxSolver&&) = delete;
    TreeNoxSolver& operator=(TreeNoxSolver&&) = delete;
    ~TreeNoxSolver() override;

    /**
     * @brief Solve the nonlinear system at the given physical time.
     *
     * @param time Current physical time for time-dependent boundary conditions.
     * @return Number of Newton corrections applied.
     */
    unsigned int solve(double time) override;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
