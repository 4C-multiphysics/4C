// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_method_direct.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_projector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::DirectSolver::DirectSolver(std::string solvertype)
    : solvertype_(solvertype), factored_(false), solver_(nullptr), projector_(nullptr)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::DirectSolver::setup(std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> x,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> b, const bool refactor, const bool reset,
    std::shared_ptr<Core::LinAlg::LinearSystemProjector> projector)
{
  auto crsA = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(matrix);

  // 1. merge the block system matrix into a standard sparse matrix if necessary
  if (!crsA)
  {
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> Ablock =
        std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(matrix);

    int matrixDim = Ablock->full_range_map().num_global_elements();
    if (matrixDim > 50000 and Communication::my_mpi_rank(matrix->domain_map().get_comm()) == 0)
      std::cout << "\n WARNING: Direct linear solver is merging matrix, this is very expensive! \n";

    crsA = Ablock->merge();
  }

  // 2. project the linear system if close to being singular and set the final matrix and vectors
  projector_ = projector;
  if (projector_ != nullptr)
  {
    Core::LinAlg::SparseMatrix A_view(*crsA);
    crsA = std::make_shared<Core::LinAlg::SparseMatrix>(projector_->to_reduced(A_view));


    FOUR_C_ASSERT_ALWAYS(b->num_vectors() == 1,
        "Expecting only one solution vector during projector call! Got {} vectors.",
        b->num_vectors());
    (*b)(0) = projector_->to_reduced((*b)(0));
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
      auto& umfpack_params = params.sublist(solver_type);
      umfpack_params.set("IsContiguous", false, "Are GIDs Contiguous");
    }
    else if (solvertype_ == "superlu")
    {
      solver_type = "SuperLU_DIST";
      auto& superludist_params = params.sublist(solver_type);
      superludist_params.set("Equil", true, "Whether to equilibrate the system before solve");
      superludist_params.set("RowPerm", "LargeDiag_MC64", "Row ordering");
      superludist_params.set("ReplaceTinyPivot", true, "Replace tiny pivot");
      superludist_params.set("IsContiguous", false, "Are GIDs Contiguous");
    }
    else
    {
      solver_type = "KLU2";
      auto& klu_params = params.sublist(solver_type);
      klu_params.set("IsContiguous", false, "Are GIDs Contiguous");
    }

    solver_ = Amesos2::create<Epetra_CrsMatrix, Epetra_MultiVector>(solver_type,
        Teuchos::rcpFromRef(a_->epetra_matrix()),
        Teuchos::rcpFromRef(x_->get_epetra_multi_vector()),
        Teuchos::rcpFromRef(b_->get_epetra_multi_vector()));

    solver_->setParameters(Teuchos::make_rcp<Teuchos::ParameterList>(std::move(params)));

    factored_ = false;
  }
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
int Core::LinearSolver::DirectSolver::solve()
{
  if (not is_factored())
  {
    solver_->symbolicFactorization();
    solver_->numericFactorization();

    factored_ = true;
  }

  solver_->solve();

  if (projector_ != nullptr)
  {
    FOUR_C_ASSERT_ALWAYS(x_->num_vectors() == 1,
        "Expecting only one solution vector during projector call! Got {} vectors.",
        x_->num_vectors());
    (*x_)(0) = projector_->to_full((*x_)(0));
  }

  return 0;
}

FOUR_C_NAMESPACE_CLOSE
