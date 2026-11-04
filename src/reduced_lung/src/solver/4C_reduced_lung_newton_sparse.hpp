// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_NEWTON_SPARSE_HPP
#define FOUR_C_REDUCED_LUNG_NEWTON_SPARSE_HPP

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
  class SparseMatrix;
  class SparseOperator;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  /**
   * @brief Context bundling all objects required to construct and run @ref NewtonSparse.
   */
  struct NewtonSparseContext
  {
    MPI_Comm comm;  ///< MPI communicator used by NOX and the linear solver.
    const ReducedLungParameters::Dynamics& dynamics;  ///< Nonlinear/timestep solver parameters.
    const Teuchos::ParameterList& linear_solver_parameters;  ///< Linear solver configuration.
    std::function<const Teuchos::ParameterList&(int)>
        solver_params_callback;  ///< Callback for nested/ID-based solver parameters.
    const NonlinearSolverAssemblyPipeline&
        assembly_pipeline;                                ///< Ordered model assembly callbacks.
    Core::LinAlg::Vector<double>& locally_relevant_dofs;  ///< Ghosted dof vector.
    Core::LinAlg::Vector<double>& x;                      ///< NOX solution vector.
    Core::LinAlg::SparseMatrix& jacobian;                 ///< Sparse Jacobian matrix.
  };

  /**
   * @brief NOX Newton solver with an assembled sparse Jacobian for reduced lung simulations.
   *
   * Encapsulates all NOX setup, callbacks, and time management for reduced lung physics. The
   * Jacobian is assembled into a sparse matrix and solved with the configured linear solver, so
   * this solver also runs in MPI-parallel.
   * Provides a simple interface: construct once, then call solve(time) in each timestep.
   */
  class NewtonSparse final : public NonlinearSolver
  {
   public:
    /**
     * @brief Construct the NewtonSparse solver with all required model components.
     *
     * @param context NOX setup context including communicator, dynamics, linear solver setup,
     * assembly pipeline callbacks, and all bound vectors/operators.
     * @param initial_time Initial time for the simulation.
     */
    NewtonSparse(const NewtonSparseContext& context, double initial_time = 0.0);

    NewtonSparse(const NewtonSparse&) = delete;
    NewtonSparse& operator=(const NewtonSparse&) = delete;
    NewtonSparse(NewtonSparse&&) = delete;
    NewtonSparse& operator=(NewtonSparse&&) = delete;
    ~NewtonSparse() override = default;

    /**
     * @brief Solve nonlinear system at given physical time.
     *
     * Updates current time for boundary conditions, performs Newton solve,
     * and synchronizes the converged solution to locally_relevant_dofs and the registered
     * internal state vectors via the assembly pipeline.
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
    Core::LinAlg::Vector<double>& locally_relevant_dofs_;
    Core::LinAlg::SparseMatrix& jacobian_;
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
