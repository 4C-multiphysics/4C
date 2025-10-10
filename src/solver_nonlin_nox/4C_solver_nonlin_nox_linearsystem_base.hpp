// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_BASE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_BASE_HPP

#include "4C_config.hpp"

#include <Epetra_Operator.h>
#include <NOX_Epetra_Vector.H>
#include <NOX_Utils.H>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    /**
     * \brief Base class for NOX linear systems.
     *
     * Currently, this class is a temporary base to proceed with a smooth step-by-step clean up of
     * the Epetra related interface methods. Most probably, it will be removed as soon as Epetra is
     * removed from 4C.
     */
    class LinearSystemBase
    {
     public:
      /**
       * \brief Virtual destructor.
       */
      virtual ~LinearSystemBase() {};

      /**
       * \brief Applies Jacobian to the given input vector and puts the answer in the result.
       */
      virtual bool apply_jacobian(
          const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const = 0;

      /**
       * \brief Applies Jacobian-Transpose to the given input vector and puts the answer in the
       *  result.
       */
      virtual bool apply_jacobian_transpose(
          const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) const = 0;

      /**
       * \brief Applies the inverse of the Jacobian matrix to the given input vector and puts the
       * answer in result.
       */
      virtual bool apply_jacobian_inverse(Teuchos::ParameterList& params,
          const ::NOX::Epetra::Vector& input, ::NOX::Epetra::Vector& result) = 0;

      /**
       * \brief Evaluates the Jacobian based on the solution vector x.
       */
      virtual bool compute_jacobian(const ::NOX::Epetra::Vector& x) = 0;

      /**
       * \brief Return Jacobian operator
       */
      virtual Teuchos::RCP<const Epetra_Operator> get_jacobian_operator() const = 0;

      /**
       * \brief Return Jacobian operator
       */
      virtual Teuchos::RCP<Epetra_Operator> get_jacobian_operator() = 0;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
