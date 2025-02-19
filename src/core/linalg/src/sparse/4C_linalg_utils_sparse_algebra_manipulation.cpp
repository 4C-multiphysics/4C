// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  export a Core::LinAlg::Vector<double>                                   mwgee 12/06|
 *----------------------------------------------------------------------*/
void Core::LinAlg::export_to(
    const Core::LinAlg::MultiVector<double>& source, Core::LinAlg::MultiVector<double>& target)
{
  try
  {
    const bool sourceunique = source.Map().UniqueGIDs();
    const bool targetunique = target.Map().UniqueGIDs();

    // both are unique, does not matter whether ex- or import
    if (sourceunique && targetunique && Core::Communication::num_mpi_ranks(source.Comm()) == 1 &&
        Core::Communication::num_mpi_ranks(target.Comm()) == 1)
    {
      if (source.NumVectors() != target.NumVectors())
        FOUR_C_THROW("number of vectors in source and target not the same!");
      for (int k = 0; k < source.NumVectors(); ++k)
        for (int i = 0; i < target.Map().NumMyElements(); ++i)
        {
          const int gid = target.Map().GID(i);
          if (gid < 0) FOUR_C_THROW("No gid for i");
          const int lid = source.Map().LID(gid);
          if (lid < 0) continue;
          // FOUR_C_THROW("No source for target");
          target(k)[i] = source(k)[lid];
        }
      return;
    }
    else if (sourceunique && targetunique)
    {
      Epetra_Export exporter(source.Map(), target.Map());
      int err = target.Export(source, exporter, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err=%d", err);
      return;
    }
    else if (sourceunique && !targetunique)
    {
      Epetra_Import importer(target.Map(), source.Map());
      int err = target.Import(source, importer, Insert);
      if (err) FOUR_C_THROW("Export using importer returned err=%d", err);
      return;
    }
    else if (!sourceunique && targetunique)
    {
      // copy locally data from source to target
      // do not allow for inter-processor communication to obtain source
      // as this may give a non-unique answer depending on the proc which is asked
      const Epetra_BlockMap& sourcemap = source.Map();
      const Epetra_BlockMap& targetmap = target.Map();
      for (int targetlid = 0; targetlid < targetmap.NumMyElements(); ++targetlid)
      {
        const int sourcelid = sourcemap.LID(targetmap.GID(targetlid));
        if (sourcelid < 0)
          FOUR_C_THROW(
              "Export of non-unique source failed. Source data not available on target proc");

        for (int k = 0; k < source.NumVectors(); ++k) target(k)[targetlid] = source(k)[sourcelid];
      }
      return;
    }
    else if (!sourceunique && !targetunique)
    {
      // Neither target nor source are unique - this is a problem.
      // We need a unique in between stage which we have to create artificially.
      // That's nasty.
      // As it is unclear whether this will ever be needed - do it later.
      FOUR_C_THROW("Neither target nor source maps are unique - cannot export");
    }
    else
      FOUR_C_THROW("VERY strange");
  }
  catch (int error)
  {
    FOUR_C_THROW("Caught an Epetra exception %d", error);
  }

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::export_to(
    const Core::LinAlg::Vector<int>& source, Core::LinAlg::Vector<int>& target)
{
  try
  {
    const bool sourceunique = source.Map().UniqueGIDs();
    const bool targetunique = target.Map().UniqueGIDs();

    // both are unique, does not matter whether ex- or import
    if (sourceunique && targetunique && Core::Communication::num_mpi_ranks(source.Comm()) == 1 &&
        Core::Communication::num_mpi_ranks(target.Comm()) == 1)
    {
      for (int i = 0; i < target.Map().NumMyElements(); ++i)
      {
        const int gid = target.Map().GID(i);
        if (gid < 0) FOUR_C_THROW("No gid for i");
        const int lid = source.Map().LID(gid);
        if (lid < 0) continue;
        target[i] = source[lid];
      }
      return;
    }
    else if (sourceunique && targetunique)
    {
      Epetra_Export exporter(source.Map(), target.Map());
      int err = target.Export(source, exporter, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err=%d", err);
      return;
    }
    else if (sourceunique && !targetunique)
    {
      Epetra_Import importer(target.Map(), source.Map());
      int err = target.Import(source, importer, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err=%d", err);
      return;
    }
    else if (!sourceunique && targetunique)
    {
      Epetra_Export exporter(source.Map(), target.Map());
      int err = target.Export(source, exporter, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err=%d", err);
      return;
    }
    else if (!sourceunique && !targetunique)
    {
      // Neither target nor source are unique - this is a problem.
      // We need a unique in between stage which we have to create artificially.
      // That's nasty.
      // As it is unclear whether this will ever be needed - do it later.
      FOUR_C_THROW("Neither target nor source maps are unique - cannot export");
    }
    else
      FOUR_C_THROW("VERY strange");
  }
  catch (int error)
  {
    FOUR_C_THROW("Caught an Epetra exception %d", error);
  }

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::unique_ptr<Core::LinAlg::Vector<double>> Core::LinAlg::extract_my_vector(
    const Core::LinAlg::Vector<double>& source, const Epetra_Map& target_map)
{
  std::unique_ptr<Core::LinAlg::Vector<double>> target =
      std::make_unique<Core::LinAlg::Vector<double>>(target_map);

  extract_my_vector(source, *target);

  return target;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::LinAlg::extract_my_vector(
    const Core::LinAlg::Vector<double>& source, Core::LinAlg::Vector<double>& target)
{
  const int my_num_target_gids = target.Map().NumMyElements();
  const int* my_target_gids = target.Map().MyGlobalElements();

  double* target_values = target.Values();

  const double* src_values = source.Values();

  for (int tar_lid = 0; tar_lid < my_num_target_gids; ++tar_lid)
  {
    const int target_gid = my_target_gids[tar_lid];

    const int src_lid = source.Map().LID(target_gid);
    // check if the target_map is a local sub-set of the source map on each proc
    if (src_lid == -1)
      FOUR_C_THROW("Couldn't find the target GID %d in the source map on proc %d.", target_gid,
          Core::Communication::my_mpi_rank(source.Comm()));

    target_values[tar_lid] = src_values[src_lid];
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::unique_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::threshold_matrix(
    const Core::LinAlg::SparseMatrix& A, const double threshold)
{
  std::unique_ptr<Core::LinAlg::SparseMatrix> A_thresh =
      std::make_unique<Core::LinAlg::SparseMatrix>(A.row_map(), A.max_num_entries());

  for (int row = 0; row < A.epetra_matrix()->NumMyRows(); row++)
  {
    int nnz_of_row;
    double* values;
    int* indices;
    A.epetra_matrix()->ExtractMyRowView(row, nnz_of_row, values, indices);

    std::vector<double> values_thresh(nnz_of_row);
    std::vector<int> indices_thresh(nnz_of_row);
    int nnz_thresh = 0;

    for (int i = 0; i < nnz_of_row; i++)
    {
      const int global_row = A.row_map().GID(row);
      const int col = A.col_map().LID(global_row);

      if (col == indices[i] || std::abs(values[i]) > std::abs(threshold))
      {
        indices_thresh[nnz_thresh] = A.col_map().GID(indices[i]);
        values_thresh[nnz_thresh] = values[i];
        nnz_thresh++;
      }
    }

    indices_thresh.resize(nnz_thresh);
    values_thresh.resize(nnz_thresh);

    A_thresh->epetra_matrix()->InsertGlobalValues(
        A.row_map().GID(row), nnz_thresh, values_thresh.data(), indices_thresh.data());
  }

  A_thresh->complete(A.domain_map(), A.range_map());

  return A_thresh;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Graph> Core::LinAlg::threshold_matrix_graph(
    const Core::LinAlg::SparseMatrix& A, const double threshold)
{
  std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern = std::make_shared<Core::LinAlg::Graph>(
      Epetra_DataAccess::Copy, A.row_map(), A.max_num_entries());

  Core::LinAlg::Vector<double> diagonal(A.row_map(), true);
  A.extract_diagonal_copy(diagonal);

  Core::LinAlg::Vector<double> ghosted_diagonal(A.col_map(), true);
  const Epetra_Import importer = Epetra_Import(A.col_map(), A.row_map());
  ghosted_diagonal.Import(
      diagonal.get_ref_of_Epetra_Vector(), importer, Epetra_CombineMode::Insert);

  double* D = ghosted_diagonal.Values();

  for (int row = 0; row < A.epetra_matrix()->NumMyRows(); row++)
  {
    int nnz_of_row;
    double* values;
    int* indices;
    A.epetra_matrix()->ExtractMyRowView(row, nnz_of_row, values, indices);

    const int global_row = A.row_map().GID(row);
    const int col = A.col_map().LID(global_row);

    const double Dk = D[col] > 0.0 ? D[col] : 1.0;
    std::vector<int> indices_new;

    for (int i = 0; i < nnz_of_row; i++)
    {
      if (col == indices[i] ||
          std::abs(std::sqrt(Dk) * values[i] * std::sqrt(Dk)) > std::abs(threshold))
        indices_new.emplace_back(A.col_map().GID(indices[i]));
    }

    sparsity_pattern->InsertGlobalIndices(global_row, indices_new.size(), indices_new.data());
  }

  sparsity_pattern->FillComplete();

  return sparsity_pattern;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Graph> Core::LinAlg::enrich_matrix_graph(
    const SparseMatrix& A, int power)
{
  SparseMatrix A_copy(A, Core::LinAlg::Copy);
  A_copy.complete();

  for (int pow = 0; pow < power - 1; pow++)
  {
    std::shared_ptr<SparseMatrix> A_power =
        Core::LinAlg::matrix_multiply(A_copy, false, A, false, true);
    A_power->complete();
    A_copy = *A_power;
  }

  return std::make_shared<Core::LinAlg::Graph>(A_copy.epetra_matrix()->Graph());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::split_matrix2x2(std::shared_ptr<Epetra_CrsMatrix> A,
    std::shared_ptr<BlockSparseMatrix<DefaultBlockMatrixStrategy>>& Ablock,
    std::shared_ptr<Epetra_Map>& A11rowmap, std::shared_ptr<Epetra_Map>& A22rowmap)
{
  if (A == nullptr) FOUR_C_THROW("A==null on entry");

  if (A11rowmap == nullptr && A22rowmap != nullptr)
    A11rowmap = Core::LinAlg::split_map(A->RowMap(), *A22rowmap);
  else if (A11rowmap != nullptr && A22rowmap == nullptr)
    A22rowmap = Core::LinAlg::split_map(A->RowMap(), *A11rowmap);
  else if (A11rowmap == nullptr && A22rowmap == nullptr)
    FOUR_C_THROW("Both A11rowmap and A22rowmap == null on entry");

  std::vector<std::shared_ptr<const Epetra_Map>> maps(2);
  maps[0] = std::make_shared<Epetra_Map>(*A11rowmap);
  maps[1] = std::make_shared<Epetra_Map>(*A22rowmap);
  Core::LinAlg::MultiMapExtractor extractor(A->RowMap(), maps);

  // create SparseMatrix view to input matrix A
  SparseMatrix a(A, View);

  // split matrix into pieces, where main diagonal blocks are square
  Ablock = Core::LinAlg::split_matrix<DefaultBlockMatrixStrategy>(a, extractor, extractor);
  Ablock->complete();

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::split_matrix2x2(std::shared_ptr<Core::LinAlg::SparseMatrix> A,
    std::shared_ptr<Epetra_Map>& A11rowmap, std::shared_ptr<Epetra_Map>& A22rowmap,
    std::shared_ptr<Epetra_Map>& A11domainmap, std::shared_ptr<Epetra_Map>& A22domainmap,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& A11,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& A12,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& A21,
    std::shared_ptr<Core::LinAlg::SparseMatrix>& A22)
{
  if (A == nullptr) FOUR_C_THROW("Core::LinAlg::split_matrix2x2: A==null on entry");

  // check and complete input row maps
  if (A11rowmap == nullptr && A22rowmap != nullptr)
    A11rowmap = Core::LinAlg::split_map(A->row_map(), *A22rowmap);
  else if (A11rowmap != nullptr && A22rowmap == nullptr)
    A22rowmap = Core::LinAlg::split_map(A->row_map(), *A11rowmap);
  else if (A11rowmap == nullptr && A22rowmap == nullptr)
    FOUR_C_THROW("Both A11rowmap and A22rowmap == null on entry");

  // check and complete input domain maps
  if (A11domainmap == nullptr && A22domainmap != nullptr)
    A11domainmap = Core::LinAlg::split_map(A->domain_map(), *A22domainmap);
  else if (A11domainmap != nullptr && A22domainmap == nullptr)
    A22domainmap = Core::LinAlg::split_map(A->domain_map(), *A11domainmap);
  else if (A11rowmap == nullptr && A22rowmap == nullptr)
    FOUR_C_THROW("Both A11domainmap and A22domainmap == null on entry");

  // local variables
  std::vector<std::shared_ptr<const Epetra_Map>> rangemaps(2);
  std::vector<std::shared_ptr<const Epetra_Map>> domainmaps(2);
  rangemaps[0] = std::make_shared<Epetra_Map>(*A11rowmap);
  rangemaps[1] = std::make_shared<Epetra_Map>(*A22rowmap);
  domainmaps[0] = std::make_shared<Epetra_Map>(*A11domainmap);
  domainmaps[1] = std::make_shared<Epetra_Map>(*A22domainmap);
  Core::LinAlg::MultiMapExtractor range(A->range_map(), rangemaps);
  Core::LinAlg::MultiMapExtractor domain(A->domain_map(), domainmaps);

  std::shared_ptr<BlockSparseMatrix<DefaultBlockMatrixStrategy>> Ablock =
      Core::LinAlg::split_matrix<DefaultBlockMatrixStrategy>(*A, domain, range);

  Ablock->complete();
  // extract internal data from Ablock in std::shared_ptr form and let Ablock die
  // (this way, internal data from Ablock will live)
  A11 = std::make_shared<SparseMatrix>((*Ablock)(0, 0), View);
  A12 = std::make_shared<SparseMatrix>((*Ablock)(0, 1), View);
  A21 = std::make_shared<SparseMatrix>((*Ablock)(1, 0), View);
  A22 = std::make_shared<SparseMatrix>((*Ablock)(1, 1), View);

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::split_matrix2x2(
    const Core::LinAlg::SparseMatrix& ASparse, Core::LinAlg::BlockSparseMatrixBase& ABlock)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::split2x2");

  if (ABlock.rows() != 2 || ABlock.cols() != 2) FOUR_C_THROW("Can only split in 2x2 system");
  if (!ASparse.filled()) FOUR_C_THROW("SparseMatrix must be filled");
  std::shared_ptr<Epetra_CrsMatrix> A = ASparse.epetra_matrix();
  std::shared_ptr<Epetra_CrsMatrix> A11 = ABlock(0, 0).epetra_matrix();
  std::shared_ptr<Epetra_CrsMatrix> A12 = ABlock(0, 1).epetra_matrix();
  std::shared_ptr<Epetra_CrsMatrix> A21 = ABlock(1, 0).epetra_matrix();
  std::shared_ptr<Epetra_CrsMatrix> A22 = ABlock(1, 1).epetra_matrix();
  if (A11->Filled() || A12->Filled() || A21->Filled() || A22->Filled())
    FOUR_C_THROW("Sub-matrices of the block operator are expected to be not filled");
  const Epetra_Map& A11rmap = ABlock.range_map(0);
  const Epetra_Map& A11dmap = ABlock.domain_map(0);
  const Epetra_Map& A22rmap = ABlock.range_map(1);
  const Epetra_Map& A22dmap = ABlock.domain_map(1);

  // find out about how the column map is linked to the individual processors.
  // this is done by filling the information about the rowmap into a vector that
  // is then exported to the column map
  Core::LinAlg::Vector<double> dselector(A->DomainMap());
  for (int i = 0; i < dselector.MyLength(); ++i)
  {
    const int gid = A->DomainMap().GID(i);
    if (A11dmap.MyGID(gid))
      dselector[i] = 0.;
    else if (A22dmap.MyGID(gid))
      dselector[i] = 1.;
    else
      dselector[i] = -1.;
  }
  Core::LinAlg::Vector<double> selector(A->ColMap());
  Core::LinAlg::export_to(dselector, selector);

  std::vector<int> gcindices1(A->MaxNumEntries());
  std::vector<double> gvalues1(A->MaxNumEntries());
  std::vector<int> gcindices2(A->MaxNumEntries());
  std::vector<double> gvalues2(A->MaxNumEntries());

  const int length = A->NumMyRows();
  for (int i = 0; i < length; ++i)
  {
    int err1 = 0;
    int err2 = 0;
    int count1 = 0;
    int count2 = 0;
    const int grid = A->GRID(i);
    if (!A11rmap.MyGID(grid) && !A22rmap.MyGID(grid)) continue;
    int numentries;
    double* values;
    int* cindices;
    int err = A->ExtractMyRowView(i, numentries, values, cindices);
    if (err) FOUR_C_THROW("ExtractMyRowView returned %d", err);
    for (int j = 0; j < numentries; ++j)
    {
      const int gcid = A->ColMap().GID(cindices[j]);
      FOUR_C_ASSERT(cindices[j] < selector.MyLength(), "Internal error");
      // column is in A*1
      if (selector[cindices[j]] == 0.)
      {
        gcindices1[count1] = gcid;
        gvalues1[count1++] = values[j];
      }
      // column is in A*2
      else if (selector[cindices[j]] == 1.)
      {
        gcindices2[count2] = gcid;
        gvalues2[count2++] = values[j];
      }
      else
        FOUR_C_THROW("Could not identify column index with block, internal error.");
    }
    if (A11rmap.MyGID(grid))
    {
      if (count1) err1 = A11->InsertGlobalValues(grid, count1, gvalues1.data(), gcindices1.data());
      if (count2) err2 = A12->InsertGlobalValues(grid, count2, gvalues2.data(), gcindices2.data());
    }
    else
    {
      if (count1) err1 = A21->InsertGlobalValues(grid, count1, gvalues1.data(), gcindices1.data());
      if (count2) err2 = A22->InsertGlobalValues(grid, count2, gvalues2.data(), gcindices2.data());
    }

    if (err1 < 0 || err2 < 0)
      FOUR_C_THROW("InsertGlobalValues returned err1=%d / err2=%d", err1, err2);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::LinAlg::split_matrixmxn(
    const Core::LinAlg::SparseMatrix& ASparse, Core::LinAlg::BlockSparseMatrixBase& ABlock)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::split_mxn");

  const int M = ABlock.rows();
  const int N = ABlock.cols();

  if (!ASparse.filled()) FOUR_C_THROW("SparseMatrix must be filled before splitting!");
  for (int m = 0; m < M; ++m)
  {
    for (int n = 0; n < N; ++n)
      if (ABlock(m, n).epetra_matrix()->Filled())
        FOUR_C_THROW("BlockSparseMatrixBase must not be filled before splitting!");
  }

  const Epetra_CrsMatrix& A = *ASparse.epetra_matrix();

  // associate each global column ID of SparseMatrix with corresponding block ID of
  // BlockSparseMatrixBase this is done via an Core::LinAlg::Vector<double> which is filled using
  // domain map information and then exported to column map
  Core::LinAlg::Vector<double> dselector(ASparse.domain_map());
  for (int collid = 0; collid < dselector.MyLength(); ++collid)
  {
    const int colgid = ASparse.domain_map().GID(collid);
    if (colgid < 0) FOUR_C_THROW("Couldn't find local column ID %d in domain map!", collid);

    int n(0);
    for (n = 0; n < N; ++n)
    {
      if (ABlock.domain_map(n).MyGID(colgid))
      {
        dselector[collid] = n;
        break;
      }
    }
    if (n == N) FOUR_C_THROW("Matrix column was not found in BlockSparseMatrixBase!");
  }
  Core::LinAlg::Vector<double> selector(A.ColMap());
  Core::LinAlg::export_to(dselector, selector);

  // allocate vectors storing global column indexes and values of matrix entries in a given row,
  // separated by blocks allocation is done outside loop over all rows for efficiency to be on the
  // safe side, we allocate more memory than we need for most rows
  std::vector<std::vector<int>> colgids(N, std::vector<int>(A.MaxNumEntries(), -1));
  std::vector<std::vector<double>> rowvalues(N, std::vector<double>(A.MaxNumEntries(), 0.));

  for (int rowlid = 0; rowlid < A.NumMyRows(); ++rowlid)
  {
    int numentries(0);
    double* values(nullptr);
    int* indices(nullptr);
    if (A.ExtractMyRowView(rowlid, numentries, values, indices))
      FOUR_C_THROW("Row of SparseMatrix couldn't be extracted during splitting!");

    std::vector<unsigned> counters(N, 0);

    for (int j = 0; j < numentries; ++j)
    {
      const int collid = indices[j];
      if (collid >= selector.MyLength()) FOUR_C_THROW("Invalid local column ID %d!", collid);

      const int blockid = static_cast<int>(selector[collid]);
      colgids[blockid][counters[blockid]] = A.ColMap().GID(collid);
      rowvalues[blockid][counters[blockid]++] = values[j];
    }

    const int rowgid = A.GRID(rowlid);
    int m(0);

    for (m = 0; m < M; ++m)
    {
      if (ABlock.range_map(m).MyGID(rowgid))
      {
        for (int n = 0; n < N; ++n)
        {
          if (counters[n])
          {
            if (ABlock(m, n).epetra_matrix()->InsertGlobalValues(
                    rowgid, counters[n], rowvalues[n].data(), colgids[n].data()))
              FOUR_C_THROW("Couldn't insert matrix entries into BlockSparseMatrixBase!");
          }
        }
        break;
      }
    }
    if (m == M) FOUR_C_THROW("Matrix row was not found in BlockSparseMatrixBase!");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Core::LinAlg::insert_my_row_diagonal_into_unfilled_matrix(
    Core::LinAlg::SparseMatrix& mat, const Core::LinAlg::Vector<double>& diag)
{
  if (mat.filled()) return -1;

  std::shared_ptr<Epetra_CrsMatrix> dst_mat_ptr = mat.epetra_matrix();
  Epetra_CrsMatrix& dst_mat = *dst_mat_ptr;

  const int my_num_entries = diag.Map().NumMyElements();
  const int* my_gids = diag.Map().MyGlobalElements();

  double* diag_values = diag.Values();

  for (int lid = 0; lid < my_num_entries; ++lid)
  {
    const int rgid = my_gids[lid];

    // skip rows which are not part of the matrix
    if (not dst_mat.RangeMap().MyGID(rgid))
      FOUR_C_THROW(
          "Could not find the row GID %d in the destination matrix RowMap"
          " on proc %d.",
          rgid,
          Core::Communication::my_mpi_rank(
              Core::Communication::unpack_epetra_comm(dst_mat.Comm())));

    if (dst_mat.NumAllocatedGlobalEntries(rgid))
    {
      // add all values, including zeros, as we need a proper matrix graph
      int err = dst_mat.SumIntoGlobalValues(rgid, 1, (diag_values + lid), &rgid);
      if (err > 0)
      {
        err = dst_mat.InsertGlobalValues(rgid, 1, (diag_values + lid), &rgid);
        if (err < 0) FOUR_C_THROW("InsertGlobalValues error: %d", err);
      }
      else if (err < 0)
        FOUR_C_THROW("SumIntoGlobalValues error: %d", err);
    }
    else
    {
      const int err = dst_mat.InsertGlobalValues(rgid, 1, (diag_values + lid), &rgid);
      if (err < 0) FOUR_C_THROW("InsertGlobalValues error: %d", err);
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 | split a map into 2 pieces with given Agiven                     06/06|
 *----------------------------------------------------------------------*/
std::shared_ptr<Epetra_Map> Core::LinAlg::split_map(
    const Epetra_Map& Amap, const Epetra_Map& Agiven)
{
  MPI_Comm Comm = Core::Communication::unpack_epetra_comm(Amap.Comm());
  const Epetra_Map& Ag = Agiven;

  int count = 0;
  std::vector<int> myaugids(Amap.NumMyElements());
  for (int i = 0; i < Amap.NumMyElements(); ++i)
  {
    const int gid = Amap.GID(i);
    if (Ag.MyGID(gid)) continue;
    myaugids[count] = gid;
    ++count;
  }
  myaugids.resize(count);
  int gcount;
  Core::Communication::sum_all(&count, &gcount, 1, Comm);
  std::shared_ptr<Epetra_Map> Aunknown = std::make_shared<Epetra_Map>(
      gcount, count, myaugids.data(), 0, Core::Communication::as_epetra_comm(Comm));

  return Aunknown;
}

/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
std::shared_ptr<Epetra_Map> Core::LinAlg::merge_map(
    const Epetra_Map& map1, const Epetra_Map& map2, bool overlap)
{
  // check for unique GIDs and for identity
  // if ((!map1.UniqueGIDs()) || (!map2.UniqueGIDs()))
  //  FOUR_C_THROW("Core::LinAlg::merge_map: One or both input maps are not unique");
  if (map1.SameAs(map2))
  {
    if ((overlap == false) && map1.NumGlobalElements() > 0)
      FOUR_C_THROW("Core::LinAlg::merge_map: Result map is overlapping");
    else
      return std::make_shared<Epetra_Map>(map1);
  }

  std::vector<int> mygids(map1.NumMyElements() + map2.NumMyElements());
  int count = map1.NumMyElements();

  // get GIDs of input map1
  for (int i = 0; i < count; ++i) mygids[i] = map1.GID(i);

  // add GIDs of input map2 (only new ones)
  for (int i = 0; i < map2.NumMyElements(); ++i)
  {
    // check for overlap
    if (map1.MyGID(map2.GID(i)))
    {
      if (overlap == false) FOUR_C_THROW("Core::LinAlg::merge_map: Result map is overlapping");
    }
    // add new GIDs to mygids
    else
    {
      mygids[count] = map2.GID(i);
      ++count;
    }
  }
  mygids.resize(count);

  // sort merged map
  sort(mygids.begin(), mygids.end());

  return std::make_shared<Epetra_Map>(-1, (int)mygids.size(), mygids.data(), 0, map1.Comm());
}

/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
std::shared_ptr<Epetra_Map> Core::LinAlg::merge_map(const std::shared_ptr<const Epetra_Map>& map1,
    const std::shared_ptr<const Epetra_Map>& map2, bool overlap)
{
  // check for cases with null std::shared_ptrs
  if (map1 == nullptr && map2 == nullptr)
    return nullptr;
  else if (map1 == nullptr)
    return std::make_shared<Epetra_Map>(*map2);
  else if (map2 == nullptr)
    return std::make_shared<Epetra_Map>(*map1);

  // wrapped call to non-std::shared_ptr version of MergeMap
  return Core::LinAlg::merge_map(*map1, *map2, overlap);
}

/*----------------------------------------------------------------------*
 | Find the intersection of two maps                     hiermeier 10/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<Epetra_Map> Core::LinAlg::intersect_map(
    const Epetra_Map& map1, const Epetra_Map& map2)
{
  // check if the maps are identical
  if (map1.SameAs(map2))
  {
    return std::make_shared<Epetra_Map>(map1);
  }

  std::vector<int> mygids(std::min(map1.NumMyElements(), map2.NumMyElements()), -1);
  int count = 0;

  for (int i = 0; i < map1.NumMyElements(); ++i)
  {
    // check for intersecting gids
    if (map2.MyGID(map1.GID(i)))
    {
      mygids[count] = map1.GID(i);
      ++count;
    }
  }
  mygids.resize(count);

  // sort merged map
  sort(mygids.begin(), mygids.end());

  return std::make_shared<Epetra_Map>(-1, (int)mygids.size(), mygids.data(), 0, map1.Comm());
}

/*----------------------------------------------------------------------*
 | split a vector into 2 pieces with given submaps            popp 02/08|
 *----------------------------------------------------------------------*/
bool Core::LinAlg::split_vector(const Epetra_Map& xmap, const Core::LinAlg::Vector<double>& x,
    std::shared_ptr<Epetra_Map>& x1map, std::shared_ptr<Core::LinAlg::Vector<double>>& x1,
    std::shared_ptr<Epetra_Map>& x2map, std::shared_ptr<Core::LinAlg::Vector<double>>& x2)
{
  // map extractor with fullmap(xmap) and two other maps (x1map and x2map)
  Core::LinAlg::MapExtractor extractor(xmap, x1map, x2map);

  // extract subvectors from fullvector
  x1 = extractor.extract_vector(x, 1);
  x2 = extractor.extract_vector(x, 0);

  return true;
}

/*----------------------------------------------------------------------*
 | split a vector into 2 pieces with given submaps           farah 02/16|
 *----------------------------------------------------------------------*/
bool Core::LinAlg::split_vector(const Epetra_Map& xmap, const Core::LinAlg::Vector<double>& x,
    std::shared_ptr<const Epetra_Map>& x1map, std::shared_ptr<Core::LinAlg::Vector<double>>& x1,
    std::shared_ptr<const Epetra_Map>& x2map, std::shared_ptr<Core::LinAlg::Vector<double>>& x2)
{
  // map extractor with fullmap(xmap) and two other maps (x1map and x2map)
  Core::LinAlg::MapExtractor extractor(xmap, x1map, x2map);

  // extract subvectors from fullvector
  x1 = extractor.extract_vector(x, 1);
  x2 = extractor.extract_vector(x, 0);

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::std_vector_to_epetra_multi_vector(const std::vector<double>& stdVector,
    Core::LinAlg::MultiVector<double>& epetraMultiVector, int blockSize)
{
  for (size_t dim = 0; dim < Teuchos::as<size_t>(blockSize); ++dim)
  {
    double** arrayOfPointers;
    epetraMultiVector.ExtractView(&arrayOfPointers);
    double* data = arrayOfPointers[dim];
    int localLength = epetraMultiVector.MyLength();

    Teuchos::ArrayRCP<double> dataVector(data, 0, localLength, false);

    const double myLength = epetraMultiVector.MyLength();
    for (double dofLID = 0; dofLID < myLength; ++dofLID)
    {
      dataVector[dofLID] = stdVector[dim * myLength + dofLID];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::epetra_multi_vector_to_std_vector(
    const Core::LinAlg::MultiVector<double>& epetraMultiVector, std::vector<double>& stdVector,
    int blockSize)
{
  for (size_t dim = 0; dim < Teuchos::as<size_t>(blockSize); ++dim)
  {
    double** arrayOfPointers;
    epetraMultiVector.ExtractView(&arrayOfPointers);
    double* data = arrayOfPointers[dim];
    int localLength = epetraMultiVector.MyLength();

    Teuchos::ArrayRCP<double> dataVector(data, 0, localLength, false);

    const double myLength = epetraMultiVector.MyLength();
    for (double dofLID = 0; dofLID < myLength; ++dofLID)
      stdVector[dim * myLength + dofLID] = dataVector[dofLID];
  }
}

FOUR_C_NAMESPACE_CLOSE
