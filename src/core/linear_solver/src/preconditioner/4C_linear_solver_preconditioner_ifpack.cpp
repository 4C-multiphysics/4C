// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_ifpack.hpp"

#include "4C_comm_utils.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_XMLParameterListHelpers.hpp>

FOUR_C_NAMESPACE_OPEN

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Core::LinearSolver::IFPACKPreconditioner::IFPACKPreconditioner(
    Teuchos::ParameterList& ifpacklist, Teuchos::ParameterList& solverlist)
    : ifpacklist_(ifpacklist), solverlist_(solverlist)
{
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
void Core::LinearSolver::IFPACKPreconditioner::setup(Core::LinAlg::SparseOperator& matrix,
    const Core::LinAlg::MultiVector<double>& x, Core::LinAlg::MultiVector<double>& b)
{
  auto A_crs = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
      Core::Utils::shared_ptr_from_ref(matrix));

  if (A_crs == nullptr)
  {
    auto* A = dynamic_cast<Core::LinAlg::BlockSparseMatrixBase*>(&matrix);

    std::cout << "\n WARNING: IFPACK preconditioner is merging matrix, this is very expensive! \n";
    pmatrix_ = std::make_shared<Epetra_CrsMatrix>(A->merge()->epetra_matrix());
  }
  else
  {
    pmatrix_ = std::make_shared<Epetra_CrsMatrix>(A_crs->epetra_matrix());
  }

  std::string prectype;
  int overlap = 0;
  Teuchos::ParameterList ifpack_params;

  if (ifpacklist_.isParameter("IFPACK_XML_FILE"))
  {
    const std::string xmlFileName = ifpacklist_.get<std::string>("IFPACK_XML_FILE");

    auto comm = Core::Communication::to_teuchos_comm<int>(
        Core::Communication::unpack_epetra_comm(pmatrix_->Comm()));

    Teuchos::updateParametersFromXmlFileAndBroadcast(
        xmlFileName, Teuchos::Ptr(&ifpack_params), *comm);

    prectype = ifpack_params.get<std::string>("Preconditioner type");
    overlap = ifpack_params.get<int>("Overlap");
  }
  else
  {
    prectype = solverlist_.get("Preconditioner Type", "ILU");
    ifpack_params = ifpacklist_;
  }

  Ifpack Factory;
  prec_ = std::shared_ptr<Ifpack_Preconditioner>(Factory.Create(prectype, pmatrix_.get(), overlap));

  if (!prec_) FOUR_C_THROW("Creation of IFPACK preconditioner of type '{}' failed.", prectype);

  prec_->SetParameters(ifpack_params);
  prec_->Initialize();
  prec_->Compute();
}


FOUR_C_NAMESPACE_CLOSE
