// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_PRECONDITIONER_TYPE_HPP
#define FOUR_C_LINEAR_SOLVER_PRECONDITIONER_TYPE_HPP

#include "4C_config.hpp"

#include "4C_linalg_graph.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_Map.h>
#include <Epetra_Operator.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver
{
  /// preconditioner base class
  /*!
     A KrylovSolver object needs one (or more) preconditioner objects. There
     are many possible preconditioners. A unified framework simplifies the
     solution process.
  */
  class PreconditionerTypeBase
  {
   public:
    /*!
       No setup is done upon construction, only the preconditioner object is
       created.
    */
    PreconditionerTypeBase() = default;

    /// virtual destruction
    virtual ~PreconditionerTypeBase() = default;

    /// Setup preconditioner with a given linear system.
    virtual void setup(bool create, Epetra_Operator* matrix, Core::LinAlg::MultiVector<double>* x,
        Core::LinAlg::MultiVector<double>* b) = 0;

    /// linear operator used for preconditioning
    virtual std::shared_ptr<Epetra_Operator> prec_operator() const = 0;
  };
}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
