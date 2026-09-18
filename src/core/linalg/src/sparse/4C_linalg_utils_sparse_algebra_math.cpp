// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
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
        A.extract_global_row_copy(Row, Values.size(), NumEntries, Values.data(), Indices.data());

        if (scalarA != 1.0)
          for (int j = 0; j < NumEntries; ++j) Values[j] *= scalarA;
        for (int j = 0; j < NumEntries; ++j)
        {
          B.assemble(Values[j], Row, Indices[j]);
        }
      }

      return NumMyRows;
    }
  }  // namespace
}  // namespace Core::LinAlg

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::matrix_add(const Core::LinAlg::SparseMatrix& A, const bool transposeA,
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
  globalSuccess = Core::Communication::min_all(localSuccess, B.get_comm());
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

  // loop rows in #tomap and replace the rows of #this->sysmat_ with provided input matrix #A
  for (int lid = 0; lid < tomap->num_my_elements(); ++lid)
  {
    const int Row = tomap->gid(lid);
    if (Row < 0) FOUR_C_THROW("DOF not found on processor.");
    Aprime->extract_global_row_copy(Row, MaxNumEntries, NumEntries, Values.data(), Indices.data());

    if (scalarA != 1.0)
      for (int j = 0; j < NumEntries; ++j) Values[j] *= scalarA;
    B.replace_global_values(Row, NumEntries, Values.data(), Indices.data());
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
  auto rangemap = transA ? A.domain_map() : A.range_map();
  auto domainmap = transB ? B.range_map() : B.domain_map();
  auto C = std::make_unique<SparseMatrix>(rangemap, nnz, A.explicit_dirichlet(), A.save_graph());

  FOUR_C_ASSERT(
      (transA ? A.range_map() : A.domain_map()).same_as(transB ? B.domain_map() : B.range_map()),
      "Matrix inner dimensions or distribution do not match for multiplication.");

  ASSERT_EPETRA_CALL(EpetraExt::MatrixMatrix::Multiply(
      A.epetra_matrix(), transA, B.epetra_matrix(), transB, C->epetra_matrix(), false));

  // manually calling complete to make sure the maps are set correctly (The epetra internal method
  // does not set the maps correctly for non-square sparse matrices)
  if (complete)
  {
    C->complete(domainmap, rangemap);
  }

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

  ASSERT_EPETRA_CALL(EpetraExt::MatrixMatrix::Multiply(A_trans->epetra_matrix(), transA,
      B_trans->epetra_matrix(), transB, C->epetra_matrix(), complete));

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
    const SparseMatrix& A, std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern,
    OptionsSparseMatrixInverse options)
{
  Core::LinAlg::Vector<double> diagonal(A.row_map());
  A.extract_diagonal_copy(diagonal);

  Core::LinAlg::Vector<double> sign(A.row_map());
  for (int row = 0; row < sign.local_length(); row++)
  {
    if (diagonal.get_values()[row] < 0)
      sign.get_values()[row] = -1.0;
    else
      sign.get_values()[row] = 1.0;
  }

  diagonal.update(options.alpha, sign, options.rho);

  Core::LinAlg::SparseMatrix A_perturbed(A);
  A_perturbed.replace_diagonal_values(diagonal);
  A_perturbed.complete();

  // construct the inverse matrix with the given sparsity pattern
  std::shared_ptr<Core::LinAlg::MultiMapExtractor> dbc_map = nullptr;
  std::shared_ptr<SparseMatrix> A_inverse =
      std::make_shared<SparseMatrix>(sparsity_pattern, dbc_map);

  // gather missing rows from other procs to generate an overlapping map
  Core::LinAlg::Import rowImport =
      Core::LinAlg::Import(sparsity_pattern->col_map(), sparsity_pattern->row_map());
  Epetra_CrsMatrix A_overlap =
      Epetra_CrsMatrix(A_perturbed.epetra_matrix(), rowImport.get_epetra_import());

  // loop over all rows of the inverse sparsity pattern (this can be done in parallel)
  for (int k = 0; k < sparsity_pattern->num_local_rows(); k++)
  {
    // 1. get column indices Ik of local row k
    std::span<int> Ik;
    sparsity_pattern->extract_local_row_view(k, Ik);

    // 2. get all local A(Ik,:) rows
    std::vector<int*> J(Ik.size());
    std::vector<int> J_size;
    std::vector<double*> Ak(Ik.size());
    std::vector<int> Jk;
    for (size_t i = 0; i < Ik.size(); i++)
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
    Core::LinAlg::SerialDenseMatrix localA(Jk.size(), Ik.size(), true);
    for (size_t i = 0; i < Ik.size(); i++)
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
    Core::LinAlg::SerialDenseVector localX(Ik.size());
    Teuchos::SerialQRDenseSolver<int, double> qrSolver;
    qrSolver.setMatrix(Teuchos::rcpFromRef(localA.base()));
    qrSolver.setVectors(Teuchos::rcpFromRef(localX.base()), Teuchos::rcpFromRef(ek.base()));
    qrSolver.factorWithEquilibration(true);
    const int err = qrSolver.solve();
    if (err != 0) FOUR_C_THROW("Error in serial QR solve.");

    // 6. set calculated row into Ainv
    A_inverse->replace_my_values(k, localX.length(), localX.values(), Ik.data());
  }
  A_inverse->complete();

  return A_inverse;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::unique_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_rank_correction(
    const SparseMatrix& A, const MultiVector<double>& basis,
    OptionsSparseMatrixRankCorrection options)
{
  const MultiVector<double>& orthonormal_basis =
      options.orthonormalize ? Core::LinAlg::orthonormalize_multi_vector(basis) : basis;

  Core::LinAlg::SparseMatrix basis_matrix(A.row_map(), basis.num_vectors());
  Core::LinAlg::Map coarse_map(basis.num_vectors(), 0, A.get_comm());
  Core::LinAlg::multi_vector_to_linalg_sparse_matrix(
      orthonormal_basis, A.row_map(), coarse_map, basis_matrix);

  // Build projection matrix
  auto projection = Core::LinAlg::matrix_multiply(basis_matrix, false, basis_matrix, true);

  auto projected_A = std::make_unique<Core::LinAlg::SparseMatrix>(A);
  Core::LinAlg::matrix_add(*projection, false, options.alpha, *projected_A, 1.0);

  return projected_A;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::MultiVector<double> Core::LinAlg::multiply_multi_vector_dense_matrix(
    const Core::LinAlg::MultiVector<double>& multi_vector,
    const Core::LinAlg::SerialDenseMatrix& matrix)
{
  const int columns = multi_vector.num_vectors();

  FOUR_C_ASSERT(matrix.num_rows() == columns,
      "The matrix row count ({}) must match the number of columns in the multivector ({})",
      matrix.num_rows(), columns);

  Core::LinAlg::MultiVector<double> result(multi_vector.get_map(), matrix.num_cols(), true);

  for (int matrix_col = 0; matrix_col < matrix.num_cols(); ++matrix_col)
  {
    auto& result_vec = result.get_vector(matrix_col);

    for (int vector_col = 0; vector_col < columns; ++vector_col)
      result_vec.update(matrix(vector_col, matrix_col), multi_vector.get_vector(vector_col), 1.0);
  }

  return result;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix Core::LinAlg::multiply_multi_vector_multi_vector(
    const Core::LinAlg::MultiVector<double>& mv1, const Core::LinAlg::MultiVector<double>& mv2,
    const int id, const bool fill)
{
  // compute information about density of P^T A P
  const Core::LinAlg::MultiVector<double>& get_multi_vector = [&]()
  {
    if (id == 1)
      return mv1;
    else if (id == 2)
      return mv2;
    else
      FOUR_C_THROW("id must be 1 or 2");
  }();

  Core::LinAlg::Vector<double> prod(get_multi_vector.get_vector(0));
  for (int i = 1; i < mv1.num_vectors(); ++i)
    prod.multiply(1.0, get_multi_vector.get_vector(i), prod, 1.0);
  int numnonzero = 0;
  for (int i = 0; i < prod.local_length(); ++i)
    if (prod.local_values_as_span()[i] != 0.0) numnonzero++;

  int glob_numnonzero = 0;
  glob_numnonzero = Core::Communication::sum_all(numnonzero, prod.get_comm());

  Core::LinAlg::Map mv1map(mv1.get_map().num_global_elements(),
      std::span<const int>(mv1.get_map().my_global_elements(), mv1.get_map().num_my_elements()), 0,
      mv1.get_map().get_comm());
  // initialization of mat with map of mv1
  Core::LinAlg::SparseMatrix mat(mv1map, glob_numnonzero, false);

  //-------------------------------
  // make mv2 redundant on all procs:
  //-------------------------------
  // auxiliary variables
  const int nummyrows = mv1.local_length();
  const int numvals = mv2.global_length();

  Core::LinAlg::Map mv2map(mv2.get_map().num_global_elements(),
      std::span<const int>(mv2.get_map().my_global_elements(), mv2.get_map().num_my_elements()), 0,
      mv2.get_map().get_comm());

  // fully redundant/overlapping map
  std::shared_ptr<Core::LinAlg::Map> redundant_map = Core::LinAlg::allreduce_e_map(mv2map);
  // initialize global mv2 without setting to 0
  Core::LinAlg::MultiVector<double> mv2glob(*redundant_map, mv2.num_vectors());
  // create importer with redundant target map and distributed source map
  Core::LinAlg::Import importer(*redundant_map, mv2.get_map());
  // import values to global mv2
  mv2glob.import(mv2, importer, Core::LinAlg::CombineMode::insert);

  //--------------------------------------------------------
  // compute mat by multiplying upright mv1 with lying mv2^T:
  //--------------------------------------------------------
  for (int rr = 0; rr < nummyrows; ++rr)
  {
    const int grid = mat.global_row_index(rr);

    std::vector<double> rowvals;
    rowvals.reserve(numvals);

    std::vector<int> indices;
    indices.reserve(numvals);

    for (int mm = 0; mm < numvals; ++mm)
    {
      double sum = 0.0;

      for (int vv = 0; vv < mv1.num_vectors(); ++vv)
      {
        sum += mv1.get_vector(vv).local_values_as_span()[rr] *
               mv2glob.get_vector(vv).local_values_as_span()[mm];
      }

      if (sum != 0.0)
      {
        rowvals.push_back(sum);
        indices.push_back(redundant_map->gid(mm));
      }
    }

    mat.insert_global_values(grid, indices.size(), rowvals.data(), indices.data());
  }

  // call fill complete
  if (fill) mat.complete();

  return mat;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::MultiVector<double> Core::LinAlg::orthonormalize_multi_vector(
    const Core::LinAlg::MultiVector<double>& multi_vector)
{
  auto local_map = Map(multi_vector.num_vectors(), 0, multi_vector.get_comm(),
      Core::LinAlg::LocalGlobal::locally_replicated);
  auto temp = Core::LinAlg::MultiVector<double>(local_map, multi_vector.num_vectors());

  temp.multiply('T', 'N', 1.0, multi_vector, multi_vector, 0.0);

  auto Q = Core::LinAlg::SerialDenseMatrix(Teuchos::DataAccess::Copy, temp.get_values(),
      temp.stride(), temp.num_vectors(), temp.num_vectors());

  Teuchos::LAPACK<int, double> lapack;
  int err = 0;

  lapack.POTRF('L', multi_vector.num_vectors(), Q.values(), Q.stride(), &err);
  FOUR_C_ASSERT(err == 0, "An error happened in the construction of the triangular factorization.");

  lapack.TRTRI('L', 'N', multi_vector.num_vectors(), Q.values(), Q.stride(), &err);
  FOUR_C_ASSERT(err == 0, "An error happened in the construction of the triangular inverse.");

  auto orthonormalized_multi_vector =
      Core::LinAlg::MultiVector<double>(multi_vector.get_map(), multi_vector.num_vectors());

  for (int i = 0; i < multi_vector.num_vectors(); i++)
    for (int j = 0; j <= i; j++)
      orthonormalized_multi_vector.get_vector(i).update(Q(i, j), multi_vector.get_vector(j), 1.0);

  return orthonormalized_multi_vector;
}

FOUR_C_NAMESPACE_CLOSE
