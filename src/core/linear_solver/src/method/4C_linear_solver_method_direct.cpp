// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_method_direct.hpp"

#include "4C_linalg_krylov_projector.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
Core::LinearSolver::DirectSolver<MatrixType, VectorType>::DirectSolver(std::string solvertype)
    : solvertype_(solvertype), factored_(false), solver_(nullptr), projector_(nullptr)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
void Core::LinearSolver::DirectSolver<MatrixType, VectorType>::setup(
    std::shared_ptr<MatrixType> matrix, std::shared_ptr<VectorType> x,
    std::shared_ptr<VectorType> b, const bool refactor, const bool reset,
    std::shared_ptr<Core::LinAlg::KrylovProjector> projector)
{
  std::shared_ptr<Epetra_CrsMatrix> crsA = std::dynamic_pointer_cast<Epetra_CrsMatrix>(matrix);

  // 1. merge the block system matrix into a standard sparse matrix if necessary
  if (!crsA)
  {
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> Ablock =
        std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(matrix);

    int matrixDim = Ablock->full_range_map().NumGlobalElements();
    if (matrixDim > 50000)
      std::cout << "\n WARNING: Direct linear solver is merging matrix, this is very expensive! \n";

    std::shared_ptr<Core::LinAlg::SparseMatrix> Ablock_merged = Ablock->merge();
    crsA = Ablock_merged->epetra_matrix();
  }

  // 2. project the linear system if close to being singular and set the final matrix and vectors
  projector_ = projector;
  if (projector_ != nullptr)
  {
    Core::LinAlg::SparseMatrix A_view(crsA, Core::LinAlg::DataAccess::View);
    std::shared_ptr<Core::LinAlg::SparseMatrix> A2 = projector_->project(A_view);

    crsA = A2->epetra_matrix();
    projector_->apply_pt(*b);
  }

  x_ = x;
  b_ = b;
  a_ = crsA;

  // 3. create linear solver
  if (reset or refactor or not is_factored())
  {
    std::string solver_type;
    Teuchos::ParameterList params("Amesos2");

    if (solvertype_ == "umfpack")
    {
      solver_type = "Umfpack";
      auto umfpack_params = Teuchos::sublist(Teuchos::rcpFromRef(params), solver_type);
      umfpack_params->set("IsContiguous", false, "Are GIDs Contiguous");
    }
    else if (solvertype_ == "superlu")
    {
      solver_type = "SuperLU_DIST";
      auto superludist_params = Teuchos::sublist(Teuchos::rcpFromRef(params), solver_type);
      // superludist_params->set("Equal", true, "Whether to equilibrate the system before solve");
      // superludist_params->set("ColPerm", "NATURAL", "Column ordering");
      superludist_params->set("RowPerm", "LargeDiag_MC64", "Row ordering");
      superludist_params->set("ReplaceTinyPivot", true, "Replace tiny pivot");
      superludist_params->set("IsContiguous", false, "Are GIDs Contiguous");
    }
    else
    {
      solver_type = "KLU2";
      auto klu_params = Teuchos::sublist(Teuchos::rcpFromRef(params), solver_type);
      klu_params->set("IsContiguous", false, "Are GIDs Contiguous");
    }

    solver_ = Amesos2::create<Epetra_CrsMatrix, Epetra_MultiVector>(solver_type,
        Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(Teuchos::rcpFromRef(*a_)),
        Teuchos::rcpFromRef(*x_->get_ptr_of_Epetra_MultiVector()),
        Teuchos::rcpFromRef(*b_->get_ptr_of_Epetra_MultiVector()));

    solver_->setParameters(Teuchos::rcpFromRef(params));

    factored_ = false;
  }
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
template <class MatrixType, class VectorType>
int Core::LinearSolver::DirectSolver<MatrixType, VectorType>::solve()
{
  if (not is_factored())
  {
    solver_->symbolicFactorization();
    solver_->numericFactorization();
    factored_ = true;
  }

  solver_->solve();

  if (projector_ != nullptr) projector_->apply_p(*x_);

  return 0;
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
// explicit initialization
template class Core::LinearSolver::DirectSolver<Epetra_Operator, Core::LinAlg::MultiVector<double>>;

FOUR_C_NAMESPACE_CLOSE
