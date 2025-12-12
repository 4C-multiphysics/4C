// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_ifpack.hpp"

#include "4C_comm_utils.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linear_solver_thyra_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Stratimikos_LinearSolverBuilder_decl.hpp>
#include <Teko_EpetraInverseOpWrapper.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

FOUR_C_NAMESPACE_OPEN

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Core::LinearSolver::IFPACKPreconditioner::IFPACKPreconditioner(Teuchos::ParameterList& ifpacklist)
    : ifpacklist_(ifpacklist)
{
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
void Core::LinearSolver::IFPACKPreconditioner::setup(
    Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b)
{
  auto A_crs = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
      Core::Utils::shared_ptr_from_ref(matrix));

  if (!A_crs)
    FOUR_C_THROW("The Ifpack based preconditioners are only available for plain sparse matrices!");

  auto comm = Core::Communication::unpack_epetra_comm(A_crs->Comm());

  pmatrix_ = Utils::create_thyra_linear_op(*A_crs, LinAlg::DataAccess::Copy);

  Teuchos::ParameterList ifpack_params;

  if (ifpacklist_.sublist("IFPACK Parameters").isParameter("IFPACK_XML_FILE"))
  {
    const std::string xmlFileName =
        ifpacklist_.sublist("IFPACK Parameters").get<std::string>("IFPACK_XML_FILE");

    Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr(&ifpack_params),
        *Core::Communication::to_teuchos_comm<int>(comm));
  }
  else
  {
    ifpack_params.set("Prec Type", "ILU");
  }


  // get preconditioner parameter list
  Teuchos::ParameterList stratimikos_params;
  Teuchos::ParameterList& ifpack_list =
      stratimikos_params.sublist("Preconditioner Types").sublist("Ifpack");
  ifpack_list.setParameters(ifpack_params);

  // setup preconditioner builder and enable relevant packages
  Stratimikos::LinearSolverBuilder<double> builder;
  builder.setParameterList(Teuchos::rcpFromRef(stratimikos_params));

  // construct preconditioning operator
  Teuchos::RCP<Thyra::PreconditionerFactoryBase<double>> precFactory =
      builder.createPreconditioningStrategy("Ifpack");
  Teuchos::RCP<Thyra::PreconditionerBase<double>> prec =
      Thyra::prec<double>(*precFactory, pmatrix_);
  auto inverseOp = prec->getUnspecifiedPrecOp();

  p_ = std::make_shared<Teko::Epetra::EpetraInverseOpWrapper>(inverseOp);
}

FOUR_C_NAMESPACE_CLOSE
