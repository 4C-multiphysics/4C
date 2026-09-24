// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_NEWTON_LINEAR_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_NEWTON_LINEAR_SOLVER_HPP

#include "4C_config.hpp"

#include <mpi.h>
#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Map;
  class Solver;
  class SparseMatrix;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  class TreeCoefficientAssemblyTarget;
  class TreeLinearization;

  /**
   * @brief Metadata passed to linear solvers for reduced-lung Newton correction systems.
   */
  struct NewtonLinearSystemMetadata
  {
    double current_time = 0.0;             ///< Current physical time.
    double time_step_size_dt = 0.0;        ///< Time-step size.
    unsigned int nonlinear_iteration = 0;  ///< Current Newton iteration index.
  };

  /**
   * @brief Linearization storage requested by a Newton correction solver.
   */
  enum class NewtonLinearizationType
  {
    SparseJacobian,        ///< Assemble and solve with a sparse Jacobian matrix.
    StructuredTreeBlocks,  ///< Assemble structured blocks for tree-based solvers.
  };

  /**
   * @brief Interface for reduced-lung Newton correction linear solvers.
   */
  class NewtonLinearSolver
  {
   public:
    /**
     * @brief Destroy the Newton linear solver interface.
     */
    virtual ~NewtonLinearSolver() = default;

    /**
     * @brief Return the linearization representation required by this solver.
     *
     * @return Requested Newton linearization type.
     */
    [[nodiscard]] virtual NewtonLinearizationType linearization_type() const
    {
      return NewtonLinearizationType::SparseJacobian;
    }

    /**
     * @brief Return the sparse Jacobian assembled by the Newton driver, if required.
     *
     * Structured solvers return nullptr and do not allocate or assemble a sparse matrix.
     */
    [[nodiscard]] virtual Core::LinAlg::SparseMatrix* sparse_jacobian_target() { return nullptr; }

    /**
     * @brief Provide structured tree-linearization data to solvers that consume it.
     *
     * This hook supports generic validation/reference storage. Optimized solvers may instead expose
     * a direct coefficient target through @ref direct_tree_coefficient_target.
     *
     * @param tree_linearization Structured tree coefficients assembled for the current Newton
     * state.
     */
    virtual void set_tree_linearization(const TreeLinearization& tree_linearization)
    {
      (void)tree_linearization;
    }

    /**
     * @brief Return a direct tree-coefficient target for solvers that own optimized storage.
     *
     * @return Target for direct coefficient assembly, or nullptr if not supported.
     */
    [[nodiscard]] virtual TreeCoefficientAssemblyTarget* direct_tree_coefficient_target()
    {
      return nullptr;
    }

    /**
     * @brief Solve one Newton correction system.
     *
     * Sign convention: solve `jacobian * delta = -residual`. The nonlinear solver then applies
     * `x = x + delta`.
     *
     * @param residual Residual vector for the current nonlinear state.
     * @param x Current nonlinear solution vector.
     * @param metadata Current time-step and Newton-iteration metadata.
     * @param delta Output Newton correction vector.
     */
    virtual void solve(const Core::LinAlg::Vector<double>& residual,
        const Core::LinAlg::Vector<double>& x, const NewtonLinearSystemMetadata& metadata,
        Core::LinAlg::Vector<double>& delta) = 0;
  };

  /**
   * @brief Context for constructing the baseline sparse Newton linear solver.
   */
  struct SparseNewtonLinearSolverContext
  {
    MPI_Comm comm;  ///< MPI communicator used by the sparse linear solver.
    const Teuchos::ParameterList& linear_solver_parameters;  ///< Linear solver configuration.
    std::function<const Teuchos::ParameterList&(int)>
        solver_params_callback;               ///< Callback for nested/ID-based solver parameters.
    const Core::LinAlg::Map& correction_map;  ///< Map for residual, right-hand side, and delta.
    Core::LinAlg::SparseMatrix& jacobian;     ///< Sparse Newton-system Jacobian.
  };

  /**
   * @brief Reduced-lung Newton linear solver backed by Core::LinAlg::Solver.
   */
  class SparseNewtonLinearSolver : public NewtonLinearSolver
  {
   public:
    /**
     * @brief Construct the sparse Newton linear solver.
     *
     * @param context Sparse linear solver setup and correction vector map.
     */
    explicit SparseNewtonLinearSolver(const SparseNewtonLinearSolverContext& context);

    /**
     * @brief Solve one sparse Newton correction system.
     *
     * @param residual Residual vector for the current nonlinear state.
     * @param x Current nonlinear solution vector.
     * @param metadata Current time-step and Newton-iteration metadata.
     * @param delta Output Newton correction vector.
     */
    [[nodiscard]] Core::LinAlg::SparseMatrix* sparse_jacobian_target() override
    {
      return &jacobian_;
    }

    void solve(const Core::LinAlg::Vector<double>& residual, const Core::LinAlg::Vector<double>& x,
        const NewtonLinearSystemMetadata& metadata, Core::LinAlg::Vector<double>& delta) override;

   private:
    std::shared_ptr<Core::LinAlg::Solver> linear_solver_;
    std::unique_ptr<Core::LinAlg::Vector<double>> rhs_;
    Core::LinAlg::SparseMatrix& jacobian_;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
