// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_ifpack.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Core::LinearSolver::IFPACKPreconditioner::IFPACKPreconditioner(
    Teuchos::ParameterList& ifpacklist, Teuchos::ParameterList& solverlist)
    : ifpacklist_(ifpacklist), solverlist_(solverlist)
{
  return;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
void Core::LinearSolver::IFPACKPreconditioner::setup(bool create, Epetra_Operator* matrix,
    Core::LinAlg::MultiVector<double>* x, Core::LinAlg::MultiVector<double>* b)
{
  if (create)
  {
    std::shared_ptr<Epetra_CrsMatrix> A_crs =
        std::dynamic_pointer_cast<Epetra_CrsMatrix>(Core::Utils::shared_ptr_from_ref(*matrix));

    if (!A_crs)
    {
      std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> A =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
              Core::Utils::shared_ptr_from_ref(*matrix));

      std::cout
          << "\n WARNING: IFPACK preconditioner is merging matrix, this is very expensive! \n";
      A_crs = A->merge()->epetra_matrix();
    }

    pmatrix_ = std::make_shared<Epetra_CrsMatrix>(*A_crs);

    Teuchos::ParameterList& inverseList = ifpacklist_.sublist("IFPACK Parameters");

    std::string xmlFileName = inverseList.get<std::string>("IFPACK_XML_FILE");
    if (xmlFileName == "none") FOUR_C_THROW("IFPACK_XML_FILE parameter not set!");

    Teuchos::RCP<Teuchos::ParameterList> ifpack_params =
        Teuchos::make_rcp<Teuchos::ParameterList>();
    auto comm = pmatrix_->getRowMap()->getComm();
    Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, ifpack_params.ptr(), *comm);

    // get the type of ifpack preconditioner from solver parameter list
    /*std::string prectype = solverlist_.get("Preconditioner Type", "ILU");
    const int overlap = ifpacklist_.get("IFPACKOVERLAP", 0);*/

    // create the preconditioner
    Ifpack Factory;
    prec_ =
        std::shared_ptr<Ifpack_Preconditioner>(Factory.Create(prectype, pmatrix_.get(), overlap));

    if (!prec_)
      FOUR_C_THROW("Creation of IFPACK preconditioner of type '%s' failed.", prectype.c_str());

    // setup
    prec_->SetParameters(ifpacklist_);
    prec_->Initialize();
    prec_->Compute();

    return;
  }
}

FOUR_C_NAMESPACE_CLOSE
