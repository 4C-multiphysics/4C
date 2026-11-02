// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_NOX_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_NOX_SOLVER_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_input.hpp"
#include "4C_reduced_lung_nonlinear_solver.hpp"
#include "4C_solver_nonlin_nox_adapter.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
  class SparseOperator;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  /**
   * @brief Context bundling all objects required to construct and run @ref NoxSolver.
   */
  struct NoxSolverContext
  {
    MPI_Comm comm;  ///< MPI communicator used by NOX and the linear solver.
    const ReducedLungParameters::Dynamics& dynamics;  ///< Nonlinear/timestep solver parameters.
    const Teuchos::ParameterList& linear_solver_parameters;  ///< Linear solver configuration.
    std::function<const Teuchos::ParameterList&(int)>
        solver_params_callback;  ///< Callback for nested/ID-based solver parameters.
    const NonlinearSolverAssemblyPipeline&
        assembly_pipeline;                                ///< Ordered model assembly callbacks.
    Core::LinAlg::Vector<double>& dofs;                   ///< Owned dof vector.
    Core::LinAlg::Vector<double>& locally_relevant_dofs;  ///< Ghosted dof vector.
    Core::LinAlg::Vector<double>& x;                      ///< NOX solution vector.
    Core::LinAlg::SparseOperator& jacobian;               ///< NOX Jacobian operator.
  };

  /**
   * @brief NOX solver for reduced lung simulations.
   *
   * Encapsulates all NOX setup, callbacks, and time management for reduced lung physics.
   * Provides a simple interface: construct once, then call solve(time) in each timestep.
   */
  class NoxSolver final : public NonlinearSolver
  {
   public:
    /**
     * @brief Construct NOX solver with all required model components.
     *
     * @param context NOX setup context including communicator, dynamics, linear solver setup,
     * assembly pipeline callbacks, and all bound vectors/operators.
     * @param initial_time Initial time for the simulation.
     */
    NoxSolver(const NoxSolverContext& context, double initial_time = 0.0);

    NoxSolver(const NoxSolver&) = delete;
    NoxSolver& operator=(const NoxSolver&) = delete;
    NoxSolver(NoxSolver&&) = delete;
    NoxSolver& operator=(NoxSolver&&) = delete;
    ~NoxSolver() override = default;

    /**
     * @brief Solve nonlinear system at given physical time.
     *
     * Updates current time for boundary conditions, performs Newton solve,
     * and synchronizes the converged solution to dofs, locally_relevant_dofs,
     * and registered internal state vectors via the assembly pipeline.
     *
     * @param time Current physical time for time-dependent BCs.
     * @return Number of nonlinear iterations performed.
     */
    unsigned int solve(double time) override;

   private:
    // Callback implementations for NOX
    bool residual(const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& residual,
        NOX::Nln::FillType fill_type);

    bool jacobian(const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac);

    // Helper to sync state from NOX vector to model containers
    void sync_state_from_x(const Core::LinAlg::Vector<double>& x);

    Core::LinAlg::Vector<double>& x_solution_;
    Core::LinAlg::Vector<double>& dofs_;
    Core::LinAlg::Vector<double>& locally_relevant_dofs_;
    NonlinearSolverAssemblyPipeline assembly_pipeline_;

    // Time integration parameters
    double dt_;
    double current_time_;

    std::shared_ptr<Core::LinAlg::Solver> linear_solver_;

    // NOX adapter
    std::optional<NOX::Nln::Adapter> adapter_;
  };

  /**
   * @brief Create a NOX parameter list for reduced-lung nonlinear solves.
   *
   * @param dynamics Reduced-lung dynamics parameters.
   * @return Teuchos parameter list configured for NOX::Nln::Adapter.
   */
  Teuchos::ParameterList create_nox_parameter_list(const ReducedLungParameters::Dynamics& dynamics);
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
