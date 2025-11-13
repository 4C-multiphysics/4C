// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_NOX_JACOBIAN_HPP
#define FOUR_C_FSI_NOX_JACOBIAN_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_interface_required_base.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <Epetra_Operator.h>
#include <NOX_Abstract_Group.H>
#include <NOX_Epetra_Interface_Jacobian.H>
#include <NOX_Utils.H>

#include <memory>

// Forward Declarations
class Map;

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class Vector;
  }  // namespace Nln

  namespace FSI
  {
    /// Matrix Free Newton Krylov based on an approximation of the residuum derivatives
    class FSIMatrixFree : public Epetra_Operator, public virtual ::NOX::Epetra::Interface::Jacobian
    {
     public:
      /*! \brief Constructor

        The vector \c x is used to clone the solution vector.
      */
      FSIMatrixFree(Teuchos::ParameterList& printParams,
          const std::shared_ptr<NOX::Nln::Interface::RequiredBase> i, const NOX::Nln::Vector& x);


      //! If set true, transpose of this operator will be applied.
      /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this
        flag affects only the Apply() and ApplyInverse() methods.  If the implementation of this
        interface does not support transpose use, this method should return a value of -1. \param
        UseTranspose -If true, multiply by the transpose of operator, otherwise just use operator.

        \return Integer error code, set to 0 if successful.  Set to -1 if this implementation does
        not support transpose.
      */
      int SetUseTranspose(bool UseTranspose) override;

      //! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
      /*!
        \param     X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
        \param     Y - A Epetra_MultiVector of dimension NumVectors containing result.

        \return Integer error code, set to 0 if successful.
      */
      int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

      //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
      /*!
        \param     X - A Epetra_MultiVector of dimension NumVectors to solve for.
        \param     Y -A Epetra_MultiVector of dimension NumVectors containing result.

        \return Integer error code, set to 0 if successful.

        \warning In order to work with an iterative solver, any implementation of this method must
        support the case where X and Y are the same object.
      */
      int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

      //! Returns the infinity norm of the global matrix.
      /* Returns the quantity \f$ \| A \|_\infty\f$ such that
         \f[\| A \|_\infty = \max_{1\lei\lem} \sum_{j=1}^n |a_{ij}| \f].

         \warning This method must not be called unless HasNormInf() returns true.    */
      double NormInf() const override;

      //! Returns a character string describing the operator
      const char* Label() const override;

      //! Returns the current UseTranspose setting.
      bool UseTranspose() const override;

      //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
      bool HasNormInf() const override;

      //! Returns a reference to the Epetra_Comm communicator associated with this operator.
      const Epetra_Comm& Comm() const override;

      //! Returns the Core::LinAlg::Map object associated with the domain of this matrix operator.
      const Epetra_Map& OperatorDomainMap() const override;

      //! Returns the Core::LinAlg::Map object associated with the range of this matrix operator.
      const Epetra_Map& OperatorRangeMap() const override;

      //! Compute Jacobian given the specified input vector, x.  Returns true if computation was
      //! successful.
      bool computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac) override;

      //! Clone a ::NOX::Abstract::Group derived object and use the computeF() method of that group
      //! for the perturbation instead of the NOX::Nln::Interface::RequiredBase::computeF() method.
      //! This is required for LOCAL to get the operators correct during homotopy.
      void set_group_for_compute_f(const ::NOX::Abstract::Group& group);

     protected:
      //! Label for matrix
      std::string label;

      //! User provided interface function
      std::shared_ptr<NOX::Nln::Interface::RequiredBase> interface;

      //! The current solution vector
      NOX::Nln::Vector currentX;

      //! Perturbed solution vector
      mutable NOX::Nln::Vector perturbX;

      //! Perturbed solution vector
      mutable NOX::Nln::Vector perturbY;

      //! Core::LinAlg::Map object used in the returns of the Epetra_Operator derived methods.
      /*! If the user is using Core::LinAlg::Maps, then ::NOX::Epetra::MatrixFree must create an
       * equivalent Core::LinAlg::Map from the Core::LinAlg::Map that can be used as the return
       * object of the OperatorDomainMap() and OperatorRangeMap() methods.
       */
      std::shared_ptr<const Epetra_Map> epetraMap;

      //! Flag to enables the use of a group instead of the interface for the computeF() calls in
      //! the directional difference calculation.
      bool useGroupForComputeF;

      //! Pointer to the group for possible use in computeF() calls.
      std::shared_ptr<::NOX::Abstract::Group> groupPtr;

      //! Printing utilities.
      ::NOX::Utils utils;
    };

  }  // namespace FSI
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
