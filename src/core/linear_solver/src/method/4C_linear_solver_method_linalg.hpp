// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_METHOD_LINALG_HPP
#define FOUR_C_LINEAR_SOLVER_METHOD_LINALG_HPP

#include "4C_config.hpp"

#include "4C_io_pstream.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Operator.h>
#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;

  class SparseOperator;

  class KrylovProjector;
}  // namespace Core::LinAlg

namespace Core::LinearSolver
{
  template <class MatrixType, class VectorType>
  class SolverTypeBase;
}

namespace Core::LinAlg
{
  //! parameters to pass to the Solve() call
  struct SolverParams
  {
    //! system should be refactorized
    bool refactor = false;

    //! data from previous solves should be recalculated including preconditioners
    bool reset = false;

    //! Krylov space projector
    std::shared_ptr<Core::LinAlg::KrylovProjector> projector = nullptr;

    //! for adaptivity of the tolerance: tolerance of the nonlinear solver
    double nonlin_tolerance = {};

    //! for adaptivity of the tolerance: current residual of the nonlinear solver
    double nonlin_residual = {};

    //! for adaptivity of the tolerance: factor by which the tolerance of the linear solver should
    //! be stricter than the current residual of the nonlinear solver
    double lin_tol_better = -1.0;

    //! tolerance of linear solver
    double tolerance = -1.0;
  };

  /*!
  \brief A general solver interface to Trilinos solvers and spooles

  - The input of parameters needs to be completely reworked (hiwi job)

  - This class should implement Epetra_Operator in the future

  */
  class Solver
  {
   public:
    /*!
    \brief Destructor

    */
    virtual ~Solver();

    /*!
    \brief Constructor taking a validated input parameter list for Solver

    Creates a solver using the parameters provided by inparams. They are translated
    by #translate_solver_parameters to the format required by Belos, if @p translate_params is true.
    Otherwise, they need to be provided such that Belos understands them.

    \param inparams (in): input parameter list as provided by Global::Problem,
                          e.g. Global::Problem::SolverParams(num)
    \param comm     (in): a reference to a Epetra communicator object
    \param get_solver_params (in): function to get solver parameters based on ID used inside
                                   inparams
    \param verbosity (in): verbosity level for output

    \param translate_params_to_belos  (in): translate parameters to Belos

    */
    Solver(const Teuchos::ParameterList& inparams, MPI_Comm comm,
        const std::function<const Teuchos::ParameterList&(int)>& get_solver_params,
        Core::IO::Verbositylevel verbosity, bool translate_params_to_belos = true);

    //! @name Solve and ApplyInverse methods

    /*!
    \brief Setup system of equations

    \param matrix    (in/out): system of equations
    \param x         (in/out): initial guess on input, solution on output
    \param b         (in/out): right hand side vector
                               if project!=false it might be modified if not orthogonal
                               to matrix kernel.
    \param params  (in)    : parameters for the solver. See documentation of SolverParams
    */
    void setup(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> x,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const SolverParams& params);

    /*!
    \brief Solve system of equations in one go

    This includes setup. Reuse of preconditioners and factorized systems is
    provided.

    \param matrix    (in/out): system of equations
    \param x         (in/out): initial guess on input, solution on output
    \param b         (in/out): right hand side vector
                               if project!=false it might be modified if not orthogonal
                               to matrix kernel.
    \param params  (in)    : parameters for the solver. See documentation of SolverParams
    */
    int solve_with_multi_vector(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> x,
        std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const SolverParams& params);

    int solve(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
        std::shared_ptr<Core::LinAlg::Vector<double>> x,
        std::shared_ptr<Core::LinAlg::Vector<double>> b, const SolverParams& params);

    /*!
    \brief Reset the solver and clear data

    All data is destroyed except the parameter list
    */
    void reset();

    //! get tolerance from Belos solver
    double get_tolerance() const
    {
      return params().sublist("Belos Parameters").get<double>("Convergence Tolerance", 1.0e-8);
    }

    /*!
    \brief Adapt tolerance of iterative solver

    Reset the tolerance read from input file. Can only be used after a call to
    adapt_tolerance.

    \note This method works with iterative solvers only - it does nothing for all other
          solvers configured.

    \sa adapt_tolerance
    */
    void reset_tolerance();

    //@}
    //! @name Input of parameters

    /*!
    \brief Translate solver input parameters from input parameter list to
           internal solver parameters list style

    \param inparams (in): input parameter list as provided by Global::Problem,
                          e.g. Global::Problem::SolverParams(num) in case of solver for
                          structures and num according to STRUCTURAL DYNAMIC
    \param get_solver_params (in): function to get solver parameters based on ID used inside
                                   inparams
    \param verbosity (in): verbosity level for output
    \return             : internal parameter list ready to be associated
                          with #params_
    */
    static Teuchos::ParameterList translate_solver_parameters(
        const Teuchos::ParameterList& inparams,
        const std::function<const Teuchos::ParameterList&(int)>& get_solver_params,
        Core::IO::Verbositylevel verbosity);

    /*!
    \brief Add a validated input parameter list as sublist to internal
           parameters

    \param name     (in): name for sublist for #params_ to attach data to
    \param inparams (in): input parameter list as provided by Global::Problem,
                          e.g. Global::Problem::FluidPressureSolverParams in case
                          of additional solver for pressure preconditioner
    \param get_solver_params (in): function to get solver parameters based on ID used inside
                                   inparams
    \param verbosity (in): verbosity level for output

    */
    void put_solver_params_to_sub_params(const std::string name,
        const Teuchos::ParameterList& inparams,
        const std::function<const Teuchos::ParameterList&(int)>& get_solver_params,
        Core::IO::Verbositylevel verbosity)
    {
      (*params_).sublist(name) =
          translate_solver_parameters(inparams, get_solver_params, verbosity);
    }
    //@}

    //! @name Query methods

    /*!
    \brief Get communicator

    */
    inline MPI_Comm get_comm() const { return comm_; }

    /*!
    \brief Get solver parameters

    */
    inline Teuchos::ParameterList& params() const { return *params_; }

    //@}

    /*!
    \brief Return the solver name from the solver block in the input file

    \note This name is purely descriptive and does not affect any computations.
    */
    inline std::string name() const { return params_->get<std::string>("name"); }

    /*!
    \brief Return number of iterations performed by solver
    */
    int get_num_iters() const;

   private:
    /*!
   \brief Adapt tolerance of iterative solver

    This method allows to adapt the tolerance of the underlying iterative solver,
    if an iterative solver is used. It is meant to be used together with
    a relative convergence criteria AZ_r0 (decided from input file)
    and allows to adapt this relative convergence criteria depending on
    the current residual of the outer nonlinear solver

    It computes a new relative tolerance to be<br>
    <br>
    \code
    if (currentnlnres*tol < desirednlnres)
      tol = desirednlnres * better / currentnlnres
    \endcode

    \note This is a rule of thumb method - not a true adaptivity in the
          field of inexact Newton methods.

    \note This method works with iterative solvers only - it does nothing for all other
          solvers configured.

    \sa ResetTolerance

    \param desirednlnres (in): Desired residual in outer nonlinear solve
    \param currentnlnres (in): Current residual in outer nonlinear solve
    \param better        (in): The amount the linear solver shall be better than
                               currentnlnres

    */
    void adapt_tolerance(
        const double desirednlnres, const double currentnlnres, const double better);

    //! set tolerance to Belos solver
    void set_tolerance(double tolerance);

    //! a communicator
    MPI_Comm comm_;

    //! (internal) parameter list
    std::shared_ptr<Teuchos::ParameterList> params_;

    /// internal solver strategy
    std::shared_ptr<
        Core::LinearSolver::SolverTypeBase<Epetra_Operator, Core::LinAlg::MultiVector<double>>>
        solver_;

   private:
    //! don't want = operator
    Solver operator=(const Solver& old) = delete;

    //! don't want cctor
    Solver(const Solver& old) = delete;

  };  // class Solver

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
