// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_linearproblem.hpp"
#include "4C_solver_nonlin_nox_linearsystem_base.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

// Forward declaration
namespace Core::LinAlg
{
  class Solver;
  template <typename T>
  class Vector;
  struct SolverParams;
  class SparseOperator;
  class SparseMatrix;
  class SerialDenseMatrix;
  class SerialDenseVector;
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    namespace Solver
    {
      class PseudoTransient;
    }  // namespace Solver
    namespace LinSystem
    {
      class PrePostOperator;
    }  // namespace LinSystem
    class Scaling;
    class Vector;

    class LinearSystem : public NOX::Nln::LinearSystemBase
    {
     public:
      using SolverMap = std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>;

     public:
      //! Standard constructor with full functionality.
      LinearSystem(Teuchos::ParameterList& printParams, Teuchos::ParameterList& linearSolverParams,
          const SolverMap& solvers, const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
          const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& preconditioner,
          const NOX::Nln::Vector& cloneVector,
          const std::shared_ptr<NOX::Nln::Scaling> scalingObject);

      //! Constructor without scaling object
      LinearSystem(Teuchos::ParameterList& printParams, Teuchos::ParameterList& linearSolverParams,
          const SolverMap& solvers, const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
          const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& preconditioner,
          const NOX::Nln::Vector& cloneVector);

      //! Constructor without preconditioner
      LinearSystem(Teuchos::ParameterList& printParams, Teuchos::ParameterList& linearSolverParams,
          const SolverMap& solvers, const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
          const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& J, const NOX::Nln::Vector& cloneVector,
          const std::shared_ptr<NOX::Nln::Scaling> scalingObject);

      //! Constructor without preconditioner and scaling object
      LinearSystem(Teuchos::ParameterList& printParams, Teuchos::ParameterList& linearSolverParams,
          const SolverMap& solvers, const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
          const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& J, const NOX::Nln::Vector& cloneVector);

      //! reset the linear solver parameters
      void reset(Teuchos::ParameterList& p);

      //! reset PrePostOperator wrapper object
      void reset_pre_post_operator(Teuchos::ParameterList& p);

      //! Evaluate the Jacobian
      bool compute_jacobian(const NOX::Nln::Vector& x) override;

      //! Evaluate the Jacobian and the right hand side based on the solution vector x at once.
      virtual bool compute_f_and_jacobian(const NOX::Nln::Vector& x, NOX::Nln::Vector& rhs);

      bool compute_correction_system(const NOX::Nln::CorrectionType type,
          const ::NOX::Abstract::Group& grp, const NOX::Nln::Vector& x, NOX::Nln::Vector& rhs);

      bool apply_jacobian_block(const NOX::Nln::Vector& input,
          Teuchos::RCP<NOX::Nln::Vector>& result, unsigned rbid, unsigned cbid) const;

      bool apply_jacobian(const NOX::Nln::Vector& input, NOX::Nln::Vector& result) const override;

      bool apply_jacobian_transpose(
          const NOX::Nln::Vector& input, NOX::Nln::Vector& result) const override;

      bool apply_jacobian_inverse(Teuchos::ParameterList& linearSolverParams,
          const NOX::Nln::Vector& input, NOX::Nln::Vector& result) override;

      //! adjust the pseudo time step (using a least squares approximation)
      void adjust_pseudo_time_step(double& delta, const double& stepSize,
          const NOX::Nln::Vector& dir, const NOX::Nln::Vector& rhs,
          const NOX::Nln::Solver::PseudoTransient& ptcsolver);

      //! NOX::Nln::Interface::RequiredBase accessor
      std::shared_ptr<const NOX::Nln::Interface::RequiredBase> get_required_interface() const;

      //! ::NOX::Epetra::Interface::Jacobian accessor
      Teuchos::RCP<const ::NOX::Epetra::Interface::Jacobian> get_jacobian_interface() const;

      /** \brief return the Jacobian range map
       *
       *  \param rbid  row block id
       *  \param cbid  column block id */
      const Core::LinAlg::Map& get_jacobian_range_map(unsigned rbid, unsigned cbid) const;

      /** \brief access the Jacobian block
       *
       *  \param rbid  row block id
       *  \param cbid  column block id */
      const Core::LinAlg::SparseMatrix& get_jacobian_block(unsigned rbid, unsigned cbid) const;

      /** \brief get a copy of the block diagonal
       *
       *  \param diag_bid  diagonal block id */
      Teuchos::RCP<Core::LinAlg::Vector<double>> get_diagonal_of_jacobian(unsigned diag_bid) const;

      /** \brief replace the diagonal of the diagonal block in the Jacobian
       *
       *  \param diag_bid  diagonal block id */
      void replace_diagonal_of_jacobian(
          const Core::LinAlg::Vector<double>& new_diag, unsigned diag_bid);

      //! Returns Jacobian Epetra_Operator pointer
      Teuchos::RCP<const Epetra_Operator> get_jacobian_operator() const override;

      /// return jacobian operator
      Teuchos::RCP<Epetra_Operator> get_jacobian_operator() override;

      //! Returns the operator type of the jacobian
      const NOX::Nln::LinSystem::OperatorType& get_jacobian_operator_type() const;

      //! destroy the jacobian ptr
      bool destroy_jacobian();

      /// compute the respective condition number (only possible in serial mode)
      double compute_serial_condition_number_of_jacobian(
          const LinSystem::ConditionNumber condnum_type) const;

     protected:
      /// access the jacobian
      inline Core::LinAlg::SparseOperator& jacobian() const
      {
        if (jac_ptr_.is_null()) throw_error("JacPtr", "JacPtr is nullptr!");

        return *jac_ptr_;
      }

      /// access the jacobian (read-only)
      inline const Teuchos::RCP<Core::LinAlg::SparseOperator>& jacobian_ptr() const
      {
        if (jac_ptr_.is_null()) throw_error("JacPtr", "JacPtr is nullptr!");

        return jac_ptr_;
      }

      //! PURE VIRTUAL FUNCTIONS: These functions have to be defined in the derived
      //! problem specific subclasses.

      //! sets the options of the underlying solver
      virtual Core::LinAlg::SolverParams set_solver_options(Teuchos::ParameterList& p,
          Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
          const NOX::Nln::SolutionType& solverType) = 0;

      //! Returns a pointer to linear solver, which has to be used
      virtual NOX::Nln::SolutionType get_active_lin_solver(
          const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
          Teuchos::RCP<Core::LinAlg::Solver>& currSolver) = 0;

      //! Set-up the linear problem object
      virtual LinearProblem set_linear_problem_for_solve(Core::LinAlg::SparseOperator& jac,
          Core::LinAlg::Vector<double>& lhs, Core::LinAlg::Vector<double>& rhs) const;

      /*! \brief Complete the solution vector after a linear solver attempt
       *
       *  This method is especially meaningful, when a sub-part of the linear
       *  problem has been solved explicitly.
       *
       *  \param linProblem (in) : Solved linear problem
       *  \param lhs        (out): left-hand-side vector which can be extended
       *
       *  */
      virtual void complete_solution_after_solve(
          const NOX::Nln::LinearProblem& linProblem, Core::LinAlg::Vector<double>& lhs) const;

      /// convert jacobian matrix to dense matrix
      void convert_jacobian_to_dense_matrix(Core::LinAlg::SerialDenseMatrix& dense) const;

      /// convert sparse matrix to dense matrix
      void convert_sparse_to_dense_matrix(const Core::LinAlg::SparseMatrix& sparse,
          Core::LinAlg::SerialDenseMatrix& dense, const Core::LinAlg::Map& full_rangemap,
          const Core::LinAlg::Map& full_domainmap) const;

      /// prepare the dense matrix in case of a block sparse matrix
      void prepare_block_dense_matrix(const Core::LinAlg::BlockSparseMatrixBase& block_sparse,
          Core::LinAlg::SerialDenseMatrix& block_dense) const;

     private:
      //! throws an error
      void throw_error(const std::string& functionName, const std::string& errorMsg) const;

     protected:
      //! Printing Utilities object
      ::NOX::Utils utils_;

      //! Solver pointers
      const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers_;

      //! Reference to the user supplied required interface functions
      std::shared_ptr<NOX::Nln::Interface::RequiredBase> reqInterfacePtr_;

      //! Reference to the user supplied Jacobian interface functions
      Teuchos::RCP<::NOX::Epetra::Interface::Jacobian> jacInterfacePtr_;

      //! Type of operator for the Jacobian.
      NOX::Nln::LinSystem::OperatorType jacType_;

      //! Scaling object supplied by the user
      std::shared_ptr<NOX::Nln::Scaling> scaling_;

      double conditionNumberEstimate_;

      //! Teuchos::Time object
      Teuchos::Time timer_;

      //! Total time spent in apply_jacobian_inverse() (sec.).
      double timeApplyJacbianInverse_;

      //! residual 2-norm
      double resNorm2_;

      //! If set to true, solver information is printed to the "Output" sublist of the "Linear
      //! Solver" list.
      bool outputSolveDetails_;

      //! Zero out the initial guess for linear solves performed through apply_jacobian_inverse()
      //! calls (i.e. zero out the result vector before the linear solve).
      bool zeroInitialGuess_;

      //! Stores the parameter "Compute Scaling Manually".
      bool manualScaling_;

      //! Pointer to an user defined wrapped NOX::Nln::Abstract::PrePostOperator object.
      Teuchos::RCP<NOX::Nln::LinSystem::PrePostOperator> prePostOperatorPtr_;

     private:
      /*! \brief Pointer to the Jacobian operator.
       *
       *  Use the provided accessors to access this member. Direct access is prohibited
       *  due to the pointer management by changing states (e.g. XFEM). */
      Teuchos::RCP<Core::LinAlg::SparseOperator> jac_ptr_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
