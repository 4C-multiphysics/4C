// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_ADAPTER_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_ADAPTER_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian_base.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"

#include <mpi.h>
#include <NOX_Abstract_Group.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Solver_Generic.H>
#include <NOX_StatusTest_Generic.H>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <functional>
#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class SparseOperator;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Abstract
  {
    class Group;
  }
  namespace Nln
  {
    class GlobalData;
    namespace Inner::StatusTest
    {
      class Generic;
    }
    class LinearSystemBase;
    class Problem;
    class Vector;
    namespace StatusTest
    {
      enum QuantityType : int;
    }  // namespace StatusTest

    /**
     * @brief Convenience adapter that wraps NOX behind callbacks.
     *
     * Users assign callbacks to @ref residual and @ref jacobian, then call
     * @ref solve. The adapter manages NOX internals (GlobalData, Problem,
     * LinearSystem, Group, StatusTests, Solver).
     *
     * Example:
     * @code
     *   NOX::Nln::Adapter::SolverMap solvers;
     *   solvers[NOX::Nln::sol_generic] = linear_solver;
     *
     *   NOX::Nln::Adapter adapter(comm, nox_params, std::move(solvers));
     *   adapter.residual = [&](const Core::LinAlg::Vector<double>& x,
     *                          Core::LinAlg::Vector<double>& f,
     *                          NOX::Nln::FillType) {
     *     // fill residual f(x)
     *     return true;
     *   };
     *   adapter.jacobian = [&](const Core::LinAlg::Vector<double>& x,
     *                          Core::LinAlg::SparseOperator& J) {
     *     // assemble Jacobian J(x)
     *     return true;
     *   };
     *
     *   const unsigned int iterations = adapter.solve(x, residual_vector, jacobian);
     * @endcode
     */
    class Adapter
    {
     public:
      /**
       * @brief Map from solution type to linear solver.
       */
      using SolverMap =
          std::map<::FourC::NOX::Nln::SolutionType, std::shared_ptr<Core::LinAlg::Solver>>;

      /**
       * @brief Optional callbacks to override default norm/scaling behavior.
       *
       * Any callback left empty causes the adapter to use its built-in
       * default implementation.
       */
      struct Options
      {
        /**
         * @brief Compute residual norm.
         *
         * @param residual Current residual vector.
         * @param quantity Quantity selector for norm evaluation.
         * @param norm_type Requested NOX norm type.
         * @param is_scaled Flag indicating whether scaling is enabled.
         */
        std::function<double(const Core::LinAlg::Vector<double>&, StatusTest::QuantityType,
            ::NOX::Abstract::Vector::NormType, bool)>
            residual_norm;

        /**
         * @brief Compute solution-update norm.
         *
         * @param x_new Current iterate.
         * @param x_old Previous iterate.
         * @param quantity Quantity selector for norm evaluation.
         * @param norm_type Requested NOX norm type.
         * @param is_scaled Flag indicating whether scaling is enabled.
         */
        std::function<double(const Core::LinAlg::Vector<double>&,
            const Core::LinAlg::Vector<double>&, StatusTest::QuantityType,
            ::NOX::Abstract::Vector::NormType, bool)>
            solution_update_norm;

        /**
         * @brief Compute RMS norm of solution update.
         *
         * @param x_new Current iterate.
         * @param x_old Previous iterate.
         * @param absolute_tolerance Absolute tolerance.
         * @param relative_tolerance Relative tolerance.
         * @param quantity Quantity selector for norm evaluation.
         * @param disable_implicit_weighting Disable implicit weighting flag.
         */
        std::function<double(const Core::LinAlg::Vector<double>&,
            const Core::LinAlg::Vector<double>&, double, double, StatusTest::QuantityType, bool)>
            solution_update_rms;

        /**
         * @brief Compute previous-solution norm.
         *
         * @param x_old Previous iterate.
         * @param quantity Quantity selector for norm evaluation.
         * @param norm_type Requested NOX norm type.
         * @param is_scaled Flag indicating whether scaling is enabled.
         */
        std::function<double(const Core::LinAlg::Vector<double>&, StatusTest::QuantityType,
            ::NOX::Abstract::Vector::NormType, bool)>
            previous_solution_norm;

        /**
         * @brief Compute reference force norm for relative checks.
         */
        std::function<double()> ref_norm_force;

        /**
         * @brief Provide element-level PTC scaling matrix.
         */
        std::function<Teuchos::RCP<Core::LinAlg::SparseMatrix>()> ptc_element_scaling;
      };

      /**
       * @brief Construct adapter with default callback behavior.
       *
       * @param comm MPI communicator.
       * @param nox_params NOX parameter list.
       * @param solvers Linear solvers keyed by solution type.
       */
      Adapter(MPI_Comm comm, const Teuchos::ParameterList& nox_params, SolverMap solvers);

      /**
       * @brief Construct adapter with callback overrides.
       *
       * @param comm MPI communicator.
       * @param nox_params NOX parameter list.
       * @param solvers Linear solvers keyed by solution type.
       * @param options Optional callback overrides.
       */
      Adapter(MPI_Comm comm, const Teuchos::ParameterList& nox_params, SolverMap solvers,
          const Options& options);

      /**
       * @brief Residual callback.
       *
       * Must be set before calling @ref solve.
       * @param x Current solution vector.
       * @param f Output residual vector.
       * @param fill_type Indicates whether the residual is being evaluated for an initial guess or
       * a new iterate.
       * @return True on success, false on failure.
       */
      std::function<bool(
          const Core::LinAlg::Vector<double>&, Core::LinAlg::Vector<double>&, NOX::Nln::FillType)>
          residual;

      /**
       * @brief Jacobian callback.
       *
       * Must be set before calling @ref solve.
       * @param x Current solution vector.
       * @param J Output Jacobian operator.
       * @return True on success, false on failure.
       */
      std::function<bool(const Core::LinAlg::Vector<double>&, Core::LinAlg::SparseOperator&)>
          jacobian;

      /**
       * @brief Run nonlinear solve.
       *
       * @param x In/out solution vector.
       * @param residual_vector Output final residual vector.
       * @param jacobian Jacobian operator used for assembly/solve.
       * @return Number of nonlinear iterations performed.
       */
      unsigned int solve(Core::LinAlg::Vector<double>& x,
          Core::LinAlg::Vector<double>& residual_vector, Core::LinAlg::SparseOperator& jacobian);

      /**
       * @brief Reset cached NOX objects and rebuild the solver stack.
       *
       * This requires at least one prior call to @ref solve so that the
       * adapter has cached vector/operator context.
       */
      void reset();

     private:
      /// (Re-)create the full NOX problem/group/solver stack for the given @p x and @p jacobian.
      /// Caches the pointers so that subsequent solve() calls can detect when a rebuild is needed.
      void build_solver(Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jacobian);

      MPI_Comm comm_;
      Teuchos::ParameterList nox_params_;
      SolverMap solvers_;

      Options options_;

      Core::LinAlg::Vector<double>* x_ptr_ = nullptr;
      Core::LinAlg::SparseOperator* jacobian_ptr_ = nullptr;

      std::shared_ptr<NOX::Nln::Interface::RequiredBase> interface_required_;
      std::shared_ptr<NOX::Nln::Interface::JacobianBase> interface_jacobian_;

      Teuchos::RCP<NOX::Nln::GlobalData> nox_global_data_;
      std::shared_ptr<NOX::Nln::Vector> x_nox_;
      Teuchos::RCP<NOX::Nln::Problem> nox_problem_;
      Teuchos::RCP<NOX::Nln::LinearSystemBase> linsys_;
      Teuchos::RCP<::NOX::Abstract::Group> group_;
      Teuchos::RCP<::NOX::StatusTest::Generic> ostatus_;
      Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic> istatus_;
      Teuchos::RCP<::NOX::Solver::Generic> nox_solver_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
