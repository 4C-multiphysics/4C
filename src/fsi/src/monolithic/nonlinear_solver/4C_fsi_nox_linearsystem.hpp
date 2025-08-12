// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_NOX_LINEARSYSTEM_HPP
#define FOUR_C_FSI_NOX_LINEARSYSTEM_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <NOX.H>
#include <NOX_Common.H>
#include <NOX_Epetra_Group.H>
#include <NOX_Epetra_Interface_Jacobian.H>
#include <NOX_Epetra_Interface_Preconditioner.H>
#include <NOX_Epetra_Interface_Required.H>
#include <NOX_Epetra_LinearSystem.H>
#include <NOX_Epetra_Scaling.H>
#include <NOX_Epetra_Vector.H>
#include <NOX_Utils.H>
#include <Teuchos_Time.hpp>

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Solver;
}

namespace NOX::FSI
{
  class LinearSystem : public ::NOX::Epetra::LinearSystem
  {
   private:
    enum OperatorType
    {
      EpetraOperator,
      EpetraRowMatrix,
      EpetraVbrMatrix,
      EpetraCrsMatrix,
      SparseMatrix,
      BlockSparseMatrix
    };

   public:
    LinearSystem(Teuchos::ParameterList& printParams,  ///< printing parameters
        Teuchos::ParameterList& linearSolverParams,    ///< parameters for linear solution
        const std::shared_ptr<::NOX::Epetra::Interface::Jacobian>&
            iJac,  ///< NOX interface to Jacobian
        const std::shared_ptr<Core::LinAlg::SparseOperator>&
            J,                                     ///< the Jacobian or stiffness matrix
        const ::NOX::Epetra::Vector& cloneVector,  ///< initial guess of the solution process
        std::shared_ptr<Core::LinAlg::Solver>
            structure_solver,  ///< (used-defined) linear algebraic solver
        const Teuchos::RCP<::NOX::Epetra::Scaling> scalingObject =
            Teuchos::null);  ///< scaling of the linear system

    /// provide storage pattern of tangent matrix, i.e. the operator
    OperatorType get_operator_type(const Epetra_Operator& Op);

    ///
    void reset(Teuchos::ParameterList& linearSolverParams);

    /// Applies Jacobian to the given input vector and puts the answer in the result.
    bool applyJacobian(
        const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const override;

    /// Applies Jacobian-Transpose to the given input vector and puts the answer in the result.
    bool applyJacobianTranspose(
        const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const override;

    /// Applies the inverse of the Jacobian matrix to the given input vector and puts the answer
    /// in result.
    bool applyJacobianInverse(Teuchos::ParameterList& params, const ::NOX::Epetra::Vector& input,
        ::NOX::Epetra::Vector& result) override;

    /// Apply right preconditiong to the given input vector.
    bool applyRightPreconditioning(bool useTranspose, Teuchos::ParameterList& params,
        const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const override;

    /// Get the scaling object.
    Teuchos::RCP<::NOX::Epetra::Scaling> getScaling() override;

    /// Sets the diagonal scaling vector(s) used in scaling the linear system.
    void resetScaling(const Teuchos::RCP<::NOX::Epetra::Scaling>& s) override;

    /// Evaluates the Jacobian based on the solution vector x.
    bool computeJacobian(const ::NOX::Epetra::Vector& x) override;

    /// Explicitly constructs a preconditioner based on the solution vector x and the parameter
    /// list p.
    bool createPreconditioner(const ::NOX::Epetra::Vector& x, Teuchos::ParameterList& p,
        bool recomputeGraph) const override;

    /// Deletes the preconditioner.
    bool destroyPreconditioner() const override;

    /// Recalculates the preconditioner using an already allocated graph.
    bool recomputePreconditioner(
        const ::NOX::Epetra::Vector& x, Teuchos::ParameterList& linearSolverParams) const override;

    /// Evaluates the preconditioner policy at the current state.
    PreconditionerReusePolicyType getPreconditionerPolicy(bool advanceReuseCounter = true) override;

    /// Indicates whether a preconditioner has been constructed.
    bool isPreconditionerConstructed() const override;

    /// Indicates whether the linear system has a preconditioner.
    bool hasPreconditioner() const override;

    /// Return Jacobian operator.
    Teuchos::RCP<const Epetra_Operator> getJacobianOperator() const override;

    /// Return Jacobian operator.
    Teuchos::RCP<Epetra_Operator> getJacobianOperator() override;

    /// Return preconditioner operator.
    Teuchos::RCP<const Epetra_Operator> getGeneratedPrecOperator() const override;

    /// Return preconditioner operator.
    Teuchos::RCP<Epetra_Operator> getGeneratedPrecOperator() override;

    /// Set Jacobian operator for solve.
    void setJacobianOperatorForSolve(
        const Teuchos::RCP<const Epetra_Operator>& solveJacOp) override;

    /// Set preconditioner operator for solve.
    void setPrecOperatorForSolve(const Teuchos::RCP<const Epetra_Operator>& solvePrecOp) override;

   private:
    /// throw an error
    void throw_error(const std::string& functionName, const std::string& errorMsg) const;

    ::NOX::Utils utils_;

    std::shared_ptr<::NOX::Epetra::Interface::Jacobian> jac_interface_ptr_;
    std::shared_ptr<::NOX::Epetra::Interface::Preconditioner> prec_interface_ptr_;
    OperatorType jac_type_;
    mutable std::shared_ptr<Epetra_Operator> jac_ptr_;
    mutable std::shared_ptr<Epetra_Operator> prec_ptr_;
    mutable std::shared_ptr<Core::LinAlg::SparseOperator> operator_;
    Teuchos::RCP<::NOX::Epetra::Scaling> scaling_;
    mutable std::shared_ptr<::NOX::Epetra::Vector> tmp_vector_ptr_;

    bool output_solve_details_;
    bool zero_initial_guess_;
    bool manual_scaling_;

    /// index of Newton iteration
    int callcount_;

    /// linear algebraic solver
    std::shared_ptr<Core::LinAlg::Solver> solver_;

    Teuchos::Time timer_;
  };
}  // namespace NOX::FSI

FOUR_C_NAMESPACE_CLOSE

#endif
