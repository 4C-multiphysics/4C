// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_Transpose_RowMatrix.h>
#include <Teuchos_SerialQRDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  namespace
  {
    /*----------------------------------------------------------------------*
     |  Internal optimized matrix addition                 kronbichler 11/15|
     |  B += A(transposed)*scalarA                                          |
     |  Return value: number of local rows in A added successfully          |
     |             (in case B must be uncompleted, this must be remembered) |
     *----------------------------------------------------------------------*/
    int do_add(const Core::LinAlg::SparseMatrix& A, const double scalarA,
        Core::LinAlg::SparseMatrix& B, const double scalarB, const int startRow = 0)
    {
      if (!A.filled()) FOUR_C_THROW("Internal error, matrix A must have called fill_complete()");

      const int NumMyRows = A.num_my_rows();

      // Case 1 where matrix B is filled. In that case, we can attempt to add in local indices,
      // much faster than the global indices... :-)
      if (B.filled())
      {
        if (startRow != 0) FOUR_C_THROW("Internal error. Not implemented.");

        // step 1: get the indexing from A to B in a random-access array
        std::vector<int> AcolToBcol(A.col_map().num_my_elements());
        for (int i = 0; i < A.col_map().num_my_elements(); ++i)
          AcolToBcol[i] = B.col_map().lid(A.col_map().gid(i));

        std::vector<int> indicesInB(A.max_num_entries());

        // step 2: loop over all local rows in matrix A and attempt the addition in local index
        // space
        for (int i = 0; i < NumMyRows; ++i)
        {
          const int myRowB = B.row_map().lid(A.row_map().gid(i));
          if (myRowB == -1)
            FOUR_C_THROW(
                "Core::LinAlg::Add: The row map of matrix B must be a superset of the row map of "
                "Matrix "
                "A.");

          // extract views of both the row in A and in B
          double *valuesA = nullptr, *valuesB = nullptr;
          int *indicesA = nullptr, *indicesB = nullptr;
          int NumEntriesA = -1, NumEntriesB = -1;
          A.extract_my_row_view(i, NumEntriesA, valuesA, indicesA);
          B.extract_my_row_view(myRowB, NumEntriesB, valuesB, indicesB);

          // check if we can identify all indices from matrix A in matrix B. quite a lot of
          // indirect addressing in here, but these are all local operations and thus pretty fast
          bool failed = false;
          indicesInB.clear();
          for (int jA = 0, jB = 0; jA < NumEntriesA; ++jA)
          {
            const int col = AcolToBcol[indicesA[jA]];
            if (col == -1)
            {
              failed = true;
              break;
            }
            while (jB < NumEntriesB && indicesB[jB] < col) ++jB;

            // did not find index in linear search (re-indexing from A.ColMap() to B.ColMap()
            // might pass through the indices differently), try binary search
            if (indicesB[jB] != col)
              jB = std::lower_bound(indicesB, indicesB + NumEntriesB, col) - indicesB;

            // not found, sparsity pattern of B does not contain the index from A -> terminate
            if (indicesB[jB] != col)
            {
              failed = true;
              break;
            }

            indicesInB.push_back(jB);
          }

          if (failed)
            return i;
          else
          {
            for (int j = 0; j < NumEntriesA; ++j) valuesB[indicesInB[j]] += scalarA * valuesA[j];
          }
        }

        return NumMyRows;
      }

      // case 2 where B is not filled -> good old slow addition in global indices

      // Loop over Aprime's rows and sum into
      std::vector<int> Indices(A.max_num_entries());
      std::vector<double> Values(A.max_num_entries());

      // Continue with i from the previous attempt
      for (int i = startRow; i < NumMyRows; ++i)
      {
        const int Row = A.global_row_index(i);
        int NumEntries = 0;
        int ierr = A.extract_global_row_copy(
            Row, Values.size(), NumEntries, Values.data(), Indices.data());
        if (ierr) FOUR_C_THROW("extract_global_row_copy() returned err={}", ierr);
        if (scalarA != 1.0)
          for (int j = 0; j < NumEntries; ++j) Values[j] *= scalarA;
        for (int j = 0; j < NumEntries; ++j)
        {
          int err = B.sum_into_global_values(Row, 1, &Values[j], &Indices[j]);
          if (err < 0 || err == 2) err = B.insert_global_values(Row, 1, &Values[j], &Indices[j]);
          if (err < 0) FOUR_C_THROW("insert_global_values() returned err={} at row {}", err, Row);
        }
      }

      return NumMyRows;
    }
  }  // namespace
}  // namespace Core::LinAlg

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::add(const Core::LinAlg::SparseMatrix& A, const bool transposeA,
    const double scalarA, Core::LinAlg::SparseMatrix& B, const double scalarB)
{
  if (!A.filled()) FOUR_C_THROW("fill_complete was not called on A");

  auto Aprime = std::make_shared<Core::LinAlg::SparseMatrix>(A);

  if (transposeA) Aprime = matrix_transpose(A);

  if (scalarB == 0.)
    B.put_scalar(0.0);
  else if (scalarB != 1.0)
    B.scale(scalarB);

  int rowsAdded = do_add(*Aprime, scalarA, B, scalarB);
  int localSuccess = rowsAdded == Aprime->row_map().num_my_elements();
  int globalSuccess = 0;
  Core::Communication::min_all(
      &localSuccess, &globalSuccess, 1, Core::Communication::unpack_epetra_comm(B.Comm()));
  if (!globalSuccess)
  {
    if (!B.filled()) FOUR_C_THROW("Unexpected state of B (expected: B not filled, got: B filled)");

    // not successful -> matrix structure must be un-completed to be able to add new indices.
    B.un_complete();
    do_add(*Aprime, scalarA, B, scalarB, rowsAdded);
    B.complete();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::matrix_put(const Core::LinAlg::SparseMatrix& A, const double scalarA,
    std::shared_ptr<const Core::LinAlg::Map> rowmap, Core::LinAlg::SparseMatrix& B)
{
  // put values onto sysmat
  if (A.get_matrixtype() != Core::LinAlg::SparseMatrix::CRS_MATRIX)
    FOUR_C_THROW("Please check code and see whether it is save to apply it to matrix type {}",
        A.get_matrixtype());
  Core::LinAlg::SparseMatrix* Aprime = const_cast<Core::LinAlg::SparseMatrix*>(&A);
  if (Aprime == nullptr) FOUR_C_THROW("Cast failed");

  // Loop over Aprime's rows, extract row content and replace respective row in sysmat
  const int MaxNumEntries = std::max(Aprime->max_num_entries(), B.max_num_entries());

  // define row map to tackle
  // if #rowmap (a subset of #RowMap()) is provided, a selective replacing is performed
  std::shared_ptr<const Map> tomap = nullptr;
  if (rowmap != nullptr)
    tomap = rowmap;
  else
    tomap = std::make_shared<const Map>(B.row_map());

  // working variables
  int NumEntries;
  std::vector<int> Indices(MaxNumEntries);
  std::vector<double> Values(MaxNumEntries);
  int err;

  // loop rows in #tomap and replace the rows of #this->sysmat_ with provided input matrix #A
  for (int lid = 0; lid < tomap->num_my_elements(); ++lid)
  {
    const int Row = tomap->gid(lid);
    if (Row < 0) FOUR_C_THROW("DOF not found on processor.");
    err = Aprime->extract_global_row_copy(
        Row, MaxNumEntries, NumEntries, Values.data(), Indices.data());
    if (err) FOUR_C_THROW("ExtractGlobalRowCopy returned err={}", err);
    if (scalarA != 1.0)
      for (int j = 0; j < NumEntries; ++j) Values[j] *= scalarA;
    err = B.replace_global_values(Row, NumEntries, Values.data(), Indices.data());
    if (err) FOUR_C_THROW("replace_global_values() failed with error code {}", err);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::unique_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_multiply(
    const SparseMatrix& A, bool transA, const SparseMatrix& B, bool transB, bool complete)
{
  // make sure fill_complete was called on the matrices
  if (!A.filled()) FOUR_C_THROW("A has to be fill_complete");
  if (!B.filled()) FOUR_C_THROW("B has to be fill_complete");

  // a first guess for the bandwidth of C leading to much less memory consumption
  const int nnz = std::max(A.max_num_entries(), B.max_num_entries());

  // now create resultmatrix C with correct rowmap
  auto map = transA ? A.domain_map() : A.range_map();
  auto C = std::make_unique<SparseMatrix>(map, nnz, A.explicit_dirichlet(), A.save_graph());
  auto A_trans = std::make_shared<Core::LinAlg::SparseMatrix>(A);
  auto B_trans = std::make_shared<Core::LinAlg::SparseMatrix>(B);

  if (transA)
  {
    A_trans = matrix_transpose(A);
    transA = false;
  }

  if (transB)
  {
    B_trans = matrix_transpose(B);
    transB = false;
  }

  int err = EpetraExt::MatrixMatrix::Multiply(A_trans->epetra_matrix(), transA,
      B_trans->epetra_matrix(), transB, C->epetra_matrix(), complete);
  if (err) FOUR_C_THROW("EpetraExt::MatrixMatrix::MatrixMultiply returned err = {}", err);

  return C;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::unique_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_multiply(const SparseMatrix& A,
    bool transA, const SparseMatrix& B, bool transB, bool explicitdirichlet, bool savegraph,
    bool complete)
{
  // make sure fill_complete was called on the matrices
  if (!A.filled()) FOUR_C_THROW("A has to be fill_complete");
  if (!B.filled()) FOUR_C_THROW("B has to be fill_complete");

  // a first guess for the bandwidth of C leading to much less memory consumption
  const int nnz = std::max(A.max_num_entries(), B.max_num_entries());

  // now create resultmatrix C with correct rowmap
  auto map = transA ? Map(A.domain_map()) : A.range_map();
  auto C = std::make_unique<SparseMatrix>(map, nnz, explicitdirichlet, savegraph);
  auto A_trans = std::make_shared<Core::LinAlg::SparseMatrix>(A);
  auto B_trans = std::make_shared<Core::LinAlg::SparseMatrix>(B);

  if (transA)
  {
    A_trans = matrix_transpose(A);
    transA = false;
  }

  if (transB)
  {
    B_trans = matrix_transpose(B);
    transB = false;
  }

  int err = EpetraExt::MatrixMatrix::Multiply(A_trans->epetra_matrix(), transA,
      B_trans->epetra_matrix(), transB, C->epetra_matrix(), complete);
  if (err) FOUR_C_THROW("EpetraExt::MatrixMatrix::MatrixMultiply returned err = {}", err);

  return C;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_transpose(const SparseMatrix& A)
{
  if (not A.filled()) FOUR_C_THROW("fill_complete was not called on matrix");

  EpetraExt::RowMatrix_Transpose transposer;
  std::shared_ptr<Core::LinAlg::SparseMatrix> matrix = nullptr;

  // Transposer does not modify the matrix but only provides a non-const interface
  Epetra_CrsMatrix* a_prime = &(dynamic_cast<Epetra_CrsMatrix&>(
      transposer(const_cast<Epetra_CrsMatrix&>(A.epetra_matrix()))));
  matrix = std::make_shared<SparseMatrix>(Core::Utils::shared_ptr_from_ref(*a_prime),
      Core::LinAlg::DataAccess::Copy, A.explicit_dirichlet(), A.save_graph());

  return matrix;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_sparse_inverse(
    const SparseMatrix& A, std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern)
{
  // construct the inverse matrix with the given sparsity pattern
  std::shared_ptr<Core::LinAlg::MultiMapExtractor> dbc_map = nullptr;
  std::shared_ptr<SparseMatrix> A_inverse =
      std::make_shared<SparseMatrix>(sparsity_pattern, dbc_map);

  // gather missing rows from other procs to generate an overlapping map
  Core::LinAlg::Import rowImport =
      Core::LinAlg::Import(sparsity_pattern->col_map(), sparsity_pattern->row_map());
  Epetra_CrsMatrix A_overlap = Epetra_CrsMatrix(A.epetra_matrix(), rowImport.get_epetra_import());

  // loop over all rows of the inverse sparsity pattern (this can be done in parallel)
  for (int k = 0; k < sparsity_pattern->num_local_rows(); k++)
  {
    // 1. get column indices Ik of local row k
    int* Ik;
    int Ik_size;
    sparsity_pattern->extract_local_row_view(k, Ik_size, Ik);

    // 2. get all local A(Ik,:) rows
    std::vector<int*> J(Ik_size);
    std::vector<int> J_size;
    std::vector<double*> Ak(Ik_size);
    std::vector<int> Jk;
    for (int i = 0; i < Ik_size; i++)
    {
      int Jk_size;
      A_overlap.ExtractMyRowView(Ik[i], Jk_size, Ak[i], J[i]);
      J_size.emplace_back(Jk_size);

      // store all local column indices
      for (int j = 0; j < J_size[i]; j++) Jk.emplace_back(J[i][j]);
    }

    // create set of unique column indices Jk
    std::sort(Jk.begin(), Jk.end());
    Jk.erase(std::unique(Jk.begin(), Jk.end()), Jk.end());
    // create map
    std::map<int, int> G;
    for (size_t i = 0; i < Jk.size(); i++) G.insert(std::pair<int, int>(Jk[i], i));

    // 3. merge local rows together
    Core::LinAlg::SerialDenseMatrix localA(Jk.size(), Ik_size, true);
    for (int i = 0; i < Ik_size; i++)
    {
      for (int j = 0; j < J_size[i]; j++)
      {
        localA(G.at(J[i][j]), i) = Ak[i][j];
      }
    }

    // 4. get direction-vector
    // diagonal needs an entry!
    Core::LinAlg::SerialDenseVector ek(Jk.size(), true);
    ek[std::distance(Jk.begin(), std::find(Jk.begin(), Jk.end(), k))] = 1.0;

    // 5. solve linear system for x
    Core::LinAlg::SerialDenseVector localX(Ik_size);
    Teuchos::SerialQRDenseSolver<int, double> qrSolver;
    qrSolver.setMatrix(Teuchos::rcpFromRef(localA.base()));
    qrSolver.setVectors(Teuchos::rcpFromRef(localX), Teuchos::rcpFromRef(ek));
    const int err = qrSolver.solve();
    if (err != 0) FOUR_C_THROW("Error in serial QR solve.");

    // 6. set calculated row into Ainv
    A_inverse->replace_my_values(k, localX.length(), localX.values(), Ik);
  }
  A_inverse->complete();

  return A_inverse;
}

void Core::LinAlg::multiply_multi_vectors(const Core::LinAlg::MultiVector<double>& v1,
    bool trans_v1, const Core::LinAlg::MultiVector<double>& v2, bool trans_v2,
    const Core::LinAlg::Map& red_map, const Core::LinAlg::Import& importer,
    Core::LinAlg::MultiVector<double>& result)
{
  // initialize temporary Core::LinAlg::MultiVector<double> (red_map: all procs hold all
  // elements/rows)
  Core::LinAlg::MultiVector<double> multivect_temp(red_map, v2.NumVectors(), true);

  // do the multiplication: (all procs hold the full result)
  int err = multivect_temp.Multiply(trans_v1 ? 'T' : 'N', trans_v2 ? 'T' : 'N', 1.0, v1, v2, 0.0);
  FOUR_C_ASSERT_ALWAYS(!err, "Multiplication failed.");

  // import the result to a Core::LinAlg::MultiVector<double> whose elements/rows are distributed
  // over all procs
  result.Import(multivect_temp, importer, Insert, nullptr);
}

FOUR_C_NAMESPACE_CLOSE
