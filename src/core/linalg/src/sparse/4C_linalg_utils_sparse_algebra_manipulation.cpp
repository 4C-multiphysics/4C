// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include "4C_comm_exporter.hpp"
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
    const bool sourceunique = source.get_map().unique_gids();
    const bool targetunique = target.get_map().unique_gids();

    // both are unique, does not matter whether ex- or import
    if (sourceunique && targetunique &&
        Core::Communication::num_mpi_ranks(source.get_comm()) == 1 &&
        Core::Communication::num_mpi_ranks(target.get_comm()) == 1)
    {
      if (source.num_vectors() != target.num_vectors())
        FOUR_C_THROW("number of vectors in source and target not the same!");
      for (int k = 0; k < source.num_vectors(); ++k)
        for (int i = 0; i < target.get_map().num_my_elements(); ++i)
        {
          const int gid = target.get_map().gid(i);
          if (gid < 0) FOUR_C_THROW("No gid for i");
          const int lid = source.get_map().lid(gid);
          if (lid < 0) continue;
          // FOUR_C_THROW("No source for target");
          target(k).get_values()[i] = source(k)[lid];
        }
      return;
    }
    else if (sourceunique && targetunique)
    {
      Core::LinAlg::Export exporter(source.get_map(), target.get_map());
      target.export_to(source, exporter, Insert);

      return;
    }
    else if (sourceunique && !targetunique)
    {
      Core::LinAlg::Import importer(target.get_map(), source.get_map());
      target.import(source, importer, Insert);

      return;
    }
    else if (!sourceunique && targetunique)
    {
      // copy locally data from source to target
      // do not allow for inter-processor communication to obtain source
      // as this may give a non-unique answer depending on the proc which is asked
      const Core::LinAlg::Map& sourcemap = source.get_map();
      const Core::LinAlg::Map& targetmap = target.get_map();
      for (int targetlid = 0; targetlid < targetmap.num_my_elements(); ++targetlid)
      {
        const int sourcelid = sourcemap.lid(targetmap.gid(targetlid));
        if (sourcelid < 0)
          FOUR_C_THROW(
              "Export of non-unique source failed. Source data not available on target proc");

        for (int k = 0; k < source.num_vectors(); ++k)
          target(k).get_values()[targetlid] = source(k)[sourcelid];
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
    FOUR_C_THROW("Caught an Epetra exception {}", error);
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
    const bool sourceunique = source.get_map().unique_gids();
    const bool targetunique = target.get_map().unique_gids();

    // both are unique, does not matter whether ex- or import
    if (sourceunique && targetunique &&
        Core::Communication::num_mpi_ranks(source.get_comm()) == 1 &&
        Core::Communication::num_mpi_ranks(target.get_comm()) == 1)
    {
      for (int i = 0; i < target.get_map().num_my_elements(); ++i)
      {
        const int gid = target.get_map().gid(i);
        if (gid < 0) FOUR_C_THROW("No gid for i");
        const int lid = source.get_map().lid(gid);
        if (lid < 0) continue;
        target.get_local_values()[i] = source.get_local_values()[lid];
      }
      return;
    }
    else if (sourceunique && targetunique)
    {
      Core::LinAlg::Export exporter(source.get_map(), target.get_map());
      int err = target.export_to(source, exporter, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err={}", err);
      return;
    }
    else if (sourceunique && !targetunique)
    {
      Core::LinAlg::Import importer(target.get_map(), source.get_map());
      int err = target.import(source, importer, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err={}", err);
      return;
    }
    else if (!sourceunique && targetunique)
    {
      Core::LinAlg::Export exporter(source.get_map(), target.get_map());
      int err = target.export_to(source, exporter, Insert);
      if (err) FOUR_C_THROW("Export using exporter returned err={}", err);
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
    FOUR_C_THROW("Caught an Epetra exception {}", error);
  }

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::unique_ptr<Core::LinAlg::Vector<double>> Core::LinAlg::extract_my_vector(
    const Core::LinAlg::Vector<double>& source, const Core::LinAlg::Map& target_map)
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
  const int my_num_target_gids = target.get_map().num_my_elements();
  const int* my_target_gids = target.get_map().my_global_elements();

  double* target_values = target.get_values();

  const double* src_values = source.get_values();

  for (int tar_lid = 0; tar_lid < my_num_target_gids; ++tar_lid)
  {
    const int target_gid = my_target_gids[tar_lid];

    const int src_lid = source.get_map().lid(target_gid);
    // check if the target_map is a local sub-set of the source map on each proc
    if (src_lid == -1)
      FOUR_C_THROW("Couldn't find the target GID {} in the source map on proc {}.", target_gid,
          Core::Communication::my_mpi_rank(source.get_comm()));

    target_values[tar_lid] = src_values[src_lid];
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::unique_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::threshold_matrix(
    const Core::LinAlg::SparseMatrix& A, const double threshold)
{
  auto A_thresh = std::make_unique<Core::LinAlg::SparseMatrix>(A.row_map(), A.max_num_entries());

  for (int row = 0; row < A.num_my_rows(); row++)
  {
    int nnz_of_row;
    double* values;
    int* indices;
    A.extract_my_row_view(row, nnz_of_row, values, indices);

    std::vector<double> values_thresh(nnz_of_row);
    std::vector<int> indices_thresh(nnz_of_row);
    int nnz_thresh = 0;

    for (int i = 0; i < nnz_of_row; i++)
    {
      const int global_row = A.row_map().gid(row);
      const int col = A.col_map().lid(global_row);

      if (col == indices[i] || std::abs(values[i]) > std::abs(threshold))
      {
        indices_thresh[nnz_thresh] = A.col_map().gid(indices[i]);
        values_thresh[nnz_thresh] = values[i];
        nnz_thresh++;
      }
    }

    indices_thresh.resize(nnz_thresh);
    values_thresh.resize(nnz_thresh);

    A_thresh->insert_global_values(
        A.row_map().gid(row), nnz_thresh, values_thresh.data(), indices_thresh.data());
  }

  A_thresh->complete(Map(A.domain_map()), A.range_map());

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
  const Core::LinAlg::Import importer = Core::LinAlg::Import(A.col_map(), A.row_map());
  ghosted_diagonal.import(
      diagonal.get_ref_of_epetra_vector(), importer, Epetra_CombineMode::Insert);

  double* D = ghosted_diagonal.get_values();

  for (int row = 0; row < A.num_my_rows(); row++)
  {
    int nnz_of_row;
    double* values;
    int* indices;
    A.extract_my_row_view(row, nnz_of_row, values, indices);

    const int global_row = A.row_map().gid(row);
    const int col = A.col_map().lid(global_row);

    const double Dk = D[col] > 0.0 ? D[col] : 1.0;
    std::vector<int> indices_new;

    for (int i = 0; i < nnz_of_row; i++)
    {
      if (col == indices[i] ||
          std::abs(std::sqrt(Dk) * values[i] * std::sqrt(Dk)) > std::abs(threshold))
        indices_new.emplace_back(A.col_map().gid(indices[i]));
    }

    sparsity_pattern->insert_global_indices(global_row, indices_new.size(), indices_new.data());
  }

  sparsity_pattern->fill_complete();

  return sparsity_pattern;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Graph> Core::LinAlg::enrich_matrix_graph(
    const SparseMatrix& A, int power)
{
  SparseMatrix A_copy(A, Core::LinAlg::DataAccess::Copy);
  A_copy.complete();

  for (int pow = 0; pow < power - 1; pow++)
  {
    std::shared_ptr<SparseMatrix> A_power =
        Core::LinAlg::matrix_multiply(A_copy, false, A, false, true);
    A_power->complete();
    A_copy = *A_power;
  }

  return std::make_shared<Core::LinAlg::Graph>(A_copy.epetra_matrix().Graph());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::split_matrix2x2(std::shared_ptr<Core::LinAlg::SparseMatrix> A,
    std::shared_ptr<Core::LinAlg::Map>& A11rowmap, std::shared_ptr<Core::LinAlg::Map>& A22rowmap,
    std::shared_ptr<Core::LinAlg::Map>& A11domainmap,
    std::shared_ptr<Core::LinAlg::Map>& A22domainmap,
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
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> rangemaps(2);
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> domainmaps(2);
  rangemaps[0] = std::make_shared<Core::LinAlg::Map>(*A11rowmap);
  rangemaps[1] = std::make_shared<Core::LinAlg::Map>(*A22rowmap);
  domainmaps[0] = std::make_shared<Core::LinAlg::Map>(*A11domainmap);
  domainmaps[1] = std::make_shared<Core::LinAlg::Map>(*A22domainmap);
  Core::LinAlg::MultiMapExtractor range(A->range_map(), rangemaps);
  Core::LinAlg::MultiMapExtractor domain(A->domain_map(), domainmaps);

  std::shared_ptr<BlockSparseMatrix<DefaultBlockMatrixStrategy>> Ablock =
      Core::LinAlg::split_matrix<DefaultBlockMatrixStrategy>(*A, domain, range);

  Ablock->complete();
  // extract internal data from Ablock in std::shared_ptr form and let Ablock die
  // (this way, internal data from Ablock will live)
  A11 = std::make_shared<SparseMatrix>((*Ablock)(0, 0), DataAccess::Share);
  A12 = std::make_shared<SparseMatrix>((*Ablock)(0, 1), DataAccess::Share);
  A21 = std::make_shared<SparseMatrix>((*Ablock)(1, 0), DataAccess::Share);
  A22 = std::make_shared<SparseMatrix>((*Ablock)(1, 1), DataAccess::Share);

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
  Core::LinAlg::SparseMatrix& A11 = ABlock(0, 0);
  Core::LinAlg::SparseMatrix& A12 = ABlock(0, 1);
  Core::LinAlg::SparseMatrix& A21 = ABlock(1, 0);
  Core::LinAlg::SparseMatrix& A22 = ABlock(1, 1);
  if (A11.filled() || A12.filled() || A21.filled() || A22.filled())
    FOUR_C_THROW("Sub-matrices of the block operator are expected to be not filled");
  const Core::LinAlg::Map& A11rmap = ABlock.range_map(0);
  const Core::LinAlg::Map& A11dmap = ABlock.domain_map(0);
  const Core::LinAlg::Map& A22rmap = ABlock.range_map(1);
  const Core::LinAlg::Map& A22dmap = ABlock.domain_map(1);

  // find out about how the column map is linked to the individual processors.
  // this is done by filling the information about the rowmap into a vector that
  // is then exported to the column map
  Core::LinAlg::Vector<double> dselector(Map(ASparse.domain_map()));
  for (int i = 0; i < dselector.local_length(); ++i)
  {
    const int gid = ASparse.domain_map().gid(i);
    if (A11dmap.my_gid(gid))
      dselector.get_values()[i] = 0.;
    else if (A22dmap.my_gid(gid))
      dselector.get_values()[i] = 1.;
    else
      dselector.get_values()[i] = -1.;
  }
  Core::LinAlg::Vector<double> selector(Map(ASparse.col_map()));
  Core::LinAlg::export_to(dselector, selector);

  std::vector<int> gcindices1(ASparse.max_num_entries());
  std::vector<double> gvalues1(ASparse.max_num_entries());
  std::vector<int> gcindices2(ASparse.max_num_entries());
  std::vector<double> gvalues2(ASparse.max_num_entries());

  const int length = ASparse.num_my_rows();
  for (int i = 0; i < length; ++i)
  {
    int err1 = 0;
    int err2 = 0;
    int count1 = 0;
    int count2 = 0;
    const int grid = ASparse.global_row_index(i);
    if (!A11rmap.my_gid(grid) && !A22rmap.my_gid(grid)) continue;
    int numentries;
    double* values;
    int* cindices;
    int err = ASparse.extract_my_row_view(i, numentries, values, cindices);
    if (err) FOUR_C_THROW("ExtractMyRowView returned {}", err);
    for (int j = 0; j < numentries; ++j)
    {
      const int gcid = ASparse.col_map().gid(cindices[j]);
      FOUR_C_ASSERT(cindices[j] < selector.local_length(), "Internal error");
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
    if (A11rmap.my_gid(grid))
    {
      if (count1) err1 = A11.insert_global_values(grid, count1, gvalues1.data(), gcindices1.data());
      if (count2) err2 = A12.insert_global_values(grid, count2, gvalues2.data(), gcindices2.data());
    }
    else
    {
      if (count1) err1 = A21.insert_global_values(grid, count1, gvalues1.data(), gcindices1.data());
      if (count2) err2 = A22.insert_global_values(grid, count2, gvalues2.data(), gcindices2.data());
    }

    if (err1 < 0 || err2 < 0)
      FOUR_C_THROW("InsertGlobalValues returned err1={} / err2={}", err1, err2);
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
      if (ABlock(m, n).filled())
        FOUR_C_THROW("BlockSparseMatrixBase must not be filled before splitting!");
  }

  // associate each global column ID of SparseMatrix with corresponding block ID of
  // BlockSparseMatrixBase this is done via an Core::LinAlg::Vector<double> which is filled using
  // domain map information and then exported to column map
  Core::LinAlg::Vector<double> dselector(Core::LinAlg::Map(ASparse.domain_map()));
  for (int collid = 0; collid < dselector.local_length(); ++collid)
  {
    const int colgid = ASparse.domain_map().gid(collid);
    if (colgid < 0) FOUR_C_THROW("Couldn't find local column ID {} in domain map!", collid);

    for (int n = 0; n < N; ++n)
    {
      if (ABlock.domain_map(n).my_gid(colgid))
      {
        dselector.get_values()[collid] = n;
        break;
      }
    }
  }
  Core::LinAlg::Vector<double> selector(Map(ASparse.col_map()));
  Core::LinAlg::export_to(dselector, selector);

  // allocate vectors storing global column indexes and values of matrix entries in a given row,
  // separated by blocks allocation is done outside loop over all rows for efficiency to be on the
  // safe side, we allocate more memory than we need for most rows
  std::vector<std::vector<int>> colgids(N, std::vector<int>(ASparse.max_num_entries(), -1));
  std::vector<std::vector<double>> rowvalues(N, std::vector<double>(ASparse.max_num_entries(), 0.));

  for (int rowlid = 0; rowlid < ASparse.num_my_rows(); ++rowlid)
  {
    int numentries(0);
    double* values(nullptr);
    int* indices(nullptr);
    if (ASparse.extract_my_row_view(rowlid, numentries, values, indices))
      FOUR_C_THROW("Row of SparseMatrix couldn't be extracted during splitting!");

    std::vector<unsigned> counters(N, 0);

    for (int j = 0; j < numentries; ++j)
    {
      const int collid = indices[j];
      if (collid >= selector.local_length()) FOUR_C_THROW("Invalid local column ID {}!", collid);

      const int blockid = static_cast<int>(selector[collid]);
      colgids[blockid][counters[blockid]] = ASparse.col_map().gid(collid);
      rowvalues[blockid][counters[blockid]++] = values[j];
    }

    const int rowgid = ASparse.global_row_index(rowlid);
    int m(0);

    for (m = 0; m < M; ++m)
    {
      if (ABlock.range_map(m).my_gid(rowgid))
      {
        for (int n = 0; n < N; ++n)
        {
          if (counters[n])
          {
            if (ABlock(m, n).insert_global_values(
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

  const int my_num_entries = diag.get_map().num_my_elements();
  const int* my_gids = diag.get_map().my_global_elements();

  const double* diag_values = diag.get_values();

  for (int lid = 0; lid < my_num_entries; ++lid)
  {
    const int rgid = my_gids[lid];

    // skip rows which are not part of the matrix
    if (not mat.range_map().my_gid(rgid))
      FOUR_C_THROW(
          "Could not find the row GID {} in the destination matrix RowMap"
          " on proc {}.",
          rgid,
          Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(mat.Comm())));

    if (mat.num_allocated_global_entries(rgid))
    {
      // add all values, including zeros, as we need a proper matrix graph
      int err = mat.sum_into_global_values(rgid, 1, (diag_values + lid), &rgid);
      if (err > 0)
      {
        err = mat.insert_global_values(rgid, 1, (diag_values + lid), &rgid);
        if (err < 0) FOUR_C_THROW("InsertGlobalValues error: {}", err);
      }
      else if (err < 0)
        FOUR_C_THROW("SumIntoGlobalValues error: {}", err);
    }
    else
    {
      const int err = mat.insert_global_values(rgid, 1, (diag_values + lid), &rgid);
      if (err < 0) FOUR_C_THROW("InsertGlobalValues error: {}", err);
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 | split a map into 2 pieces with given Agiven                     06/06|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::LinAlg::split_map(
    const Core::LinAlg::Map& Amap, const Core::LinAlg::Map& Agiven)
{
  MPI_Comm Comm = Amap.get_comm();
  const Core::LinAlg::Map& Ag = Agiven;

  int count = 0;
  std::vector<int> myaugids(Amap.num_my_elements());
  for (int i = 0; i < Amap.num_my_elements(); ++i)
  {
    const int gid = Amap.gid(i);
    if (Ag.my_gid(gid)) continue;
    myaugids[count] = gid;
    ++count;
  }
  myaugids.resize(count);
  int gcount;
  gcount = Core::Communication::sum_all(count, Comm);
  std::shared_ptr<Core::LinAlg::Map> Aunknown =
      std::make_shared<Core::LinAlg::Map>(gcount, count, myaugids.data(), 0, Comm);

  return Aunknown;
}

/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::LinAlg::merge_map(
    const Core::LinAlg::Map& map1, const Core::LinAlg::Map& map2, bool overlap)
{
  // check for unique GIDs and for identity
  // if ((!map1.UniqueGIDs()) || (!map2.UniqueGIDs()))
  //  FOUR_C_THROW("Core::LinAlg::merge_map: One or both input maps are not unique");
  if (map1.same_as(map2))
  {
    if ((overlap == false) && map1.num_global_elements() > 0)
      FOUR_C_THROW("Core::LinAlg::merge_map: Result map is overlapping");
    else
      return std::make_shared<Core::LinAlg::Map>(map1);
  }

  std::vector<int> mygids(map1.num_my_elements() + map2.num_my_elements());
  int count = map1.num_my_elements();

  // get GIDs of input map1
  for (int i = 0; i < count; ++i) mygids[i] = map1.gid(i);

  // add GIDs of input map2 (only new ones)
  for (int i = 0; i < map2.num_my_elements(); ++i)
  {
    // check for overlap
    if (map1.my_gid(map2.gid(i)))
    {
      if (overlap == false) FOUR_C_THROW("Core::LinAlg::merge_map: Result map is overlapping");
    }
    // add new GIDs to mygids
    else
    {
      mygids[count] = map2.gid(i);
      ++count;
    }
  }
  mygids.resize(count);

  // sort merged map
  sort(mygids.begin(), mygids.end());

  return std::make_shared<Core::LinAlg::Map>(
      -1, (int)mygids.size(), mygids.data(), 0, map1.get_comm());
}

/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::LinAlg::merge_map(
    const std::shared_ptr<const Core::LinAlg::Map>& map1,
    const std::shared_ptr<const Core::LinAlg::Map>& map2, bool overlap)
{
  // check for cases with null std::shared_ptrs
  if (map1 == nullptr && map2 == nullptr)
    return nullptr;
  else if (map1 == nullptr)
    return std::make_shared<Core::LinAlg::Map>(*map2);
  else if (map2 == nullptr)
    return std::make_shared<Core::LinAlg::Map>(*map1);

  // wrapped call to non-std::shared_ptr version of MergeMap
  return Core::LinAlg::merge_map(*map1, *map2, overlap);
}

/*----------------------------------------------------------------------*
 | Find the intersection of two maps                     hiermeier 10/14|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::LinAlg::intersect_map(
    const Core::LinAlg::Map& map1, const Core::LinAlg::Map& map2)
{
  // check if the maps are identical
  if (map1.same_as(map2))
  {
    return std::make_shared<Core::LinAlg::Map>(map1);
  }

  std::vector<int> mygids(std::min(map1.num_my_elements(), map2.num_my_elements()), -1);
  int count = 0;

  for (int i = 0; i < map1.num_my_elements(); ++i)
  {
    // check for intersecting gids
    if (map2.my_gid(map1.gid(i)))
    {
      mygids[count] = map1.gid(i);
      ++count;
    }
  }
  mygids.resize(count);

  // sort merged map
  sort(mygids.begin(), mygids.end());

  return std::make_shared<Core::LinAlg::Map>(
      -1, (int)mygids.size(), mygids.data(), 0, map1.get_comm());
}

/*----------------------------------------------------------------------*
 | split a vector into 2 pieces with given submaps            popp 02/08|
 *----------------------------------------------------------------------*/
bool Core::LinAlg::split_vector(const Core::LinAlg::Map& xmap,
    const Core::LinAlg::Vector<double>& x, std::shared_ptr<Core::LinAlg::Map>& x1map,
    std::shared_ptr<Core::LinAlg::Vector<double>>& x1, std::shared_ptr<Core::LinAlg::Map>& x2map,
    std::shared_ptr<Core::LinAlg::Vector<double>>& x2)
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
bool Core::LinAlg::split_vector(const Core::LinAlg::Map& xmap,
    const Core::LinAlg::Vector<double>& x, std::shared_ptr<const Core::LinAlg::Map>& x1map,
    std::shared_ptr<Core::LinAlg::Vector<double>>& x1,
    std::shared_ptr<const Core::LinAlg::Map>& x2map,
    std::shared_ptr<Core::LinAlg::Vector<double>>& x2)
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
void Core::LinAlg::std_vector_to_multi_vector(const std::vector<double>& std_vector,
    Core::LinAlg::MultiVector<double>& multi_vector, int block_size)
{
  for (size_t dim = 0; dim < Teuchos::as<size_t>(block_size); ++dim)
  {
    double** arrayOfPointers;
    multi_vector.extract_view(&arrayOfPointers);
    double* data = arrayOfPointers[dim];
    int localLength = multi_vector.local_length();

    Teuchos::ArrayRCP<double> dataVector(data, 0, localLength, false);

    const double myLength = multi_vector.local_length();
    for (double dofLID = 0; dofLID < myLength; ++dofLID)
    {
      dataVector[dofLID] = std_vector[dim * myLength + dofLID];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::multi_vector_to_std_vector(const Core::LinAlg::MultiVector<double>& multi_vector,
    std::vector<double>& std_vector, int block_size)
{
  for (size_t dim = 0; dim < Teuchos::as<size_t>(block_size); ++dim)
  {
    double** arrayOfPointers;
    multi_vector.extract_view(&arrayOfPointers);
    double* data = arrayOfPointers[dim];
    int localLength = multi_vector.local_length();

    Teuchos::ArrayRCP<double> dataVector(data, 0, localLength, false);

    const double myLength = multi_vector.local_length();
    for (double dofLID = 0; dofLID < myLength; ++dofLID)
      std_vector[dim * myLength + dofLID] = dataVector[dofLID];
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::multi_vector_to_linalg_sparse_matrix(
    const Core::LinAlg::MultiVector<double>& multivect, const Core::LinAlg::Map& rangemap,
    const Core::LinAlg::Map& domainmap, Core::LinAlg::SparseMatrix& sparsemat)
{
  const double* Values = multivect.get_values();

  double value;
  for (int i = 0; i < multivect.num_vectors(); i++)
  {
    for (int j = 0; j < multivect.local_length(); j++)
    {
      value = Values[i * multivect.local_length() + j];

      // if we have a zero value, just continue
      if (std::abs(value) < std::numeric_limits<double>::min()) continue;

      sparsemat.assemble(value, rangemap.gid(j), domainmap.min_all_gid() + i);
    }
  }

  sparsemat.complete(domainmap, rangemap);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_row_transform(
    const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap)
{
  return redistribute(inmat, newrowmap, inmat.domain_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_col_transform(
    const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newdomainmap)
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> outmat =
      std::make_shared<Core::LinAlg::SparseMatrix>(inmat);

  outmat->complete(newdomainmap, inmat.row_map(), {.enforce_complete = true});

  return outmat;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_row_col_transform(
    const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap,
    const Core::LinAlg::Map& newdomainmap)
{
  return redistribute(inmat, newrowmap, newdomainmap);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::redistribute(
    const Core::LinAlg::SparseMatrix& src, const Core::LinAlg::Map& permrowmap,
    const Core::LinAlg::Map& permdomainmap)
{
  Core::LinAlg::Export exporter(permrowmap, src.row_map());

  auto permsrc = std::make_shared<Core::LinAlg::SparseMatrix>(permrowmap, src.max_num_entries());
  int err = permsrc->import(src, exporter, Insert);
  if (err) FOUR_C_THROW("Import failed with err={}", err);

  permsrc->complete(permdomainmap, permrowmap);
  return permsrc;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_row_transform_gids(
    const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap)
{
  // initialize output matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> outmat =
      std::make_shared<Core::LinAlg::SparseMatrix>(newrowmap, 100, false, true);

  // transform input matrix to newrowmap
  for (int i = 0; i < inmat.num_my_rows(); ++i)
  {
    int NumEntries = 0;
    double* Values;
    int* Indices;
    int err = inmat.extract_my_row_view(i, NumEntries, Values, Indices);
    if (err != 0) FOUR_C_THROW("extract_my_row_view error: {}", err);

    // pull indices back to global
    std::vector<int> idx(NumEntries);
    for (int j = 0; j < NumEntries; ++j)
    {
      idx[j] = (inmat.col_map()).gid(Indices[j]);
    }

    err = outmat->insert_global_values(newrowmap.gid(i), NumEntries, Values, idx.data());
    if (err < 0) FOUR_C_THROW("insert_global_values error: {}", err);
  }

  // complete output matrix
  outmat->complete(inmat.domain_map(), newrowmap);

  return outmat;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_col_transform_gids(
    const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newdomainmap)
{
  // initialize output matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> outmat =
      std::make_shared<Core::LinAlg::SparseMatrix>(inmat.row_map(), 100, false, true);

  // mapping of column gids
  std::map<int, int> gidmap;
  Core::Communication::Exporter ex(
      inmat.domain_map(), inmat.col_map(), Core::Communication::unpack_epetra_comm(inmat.Comm()));
  for (int i = 0; i < inmat.domain_map().num_my_elements(); ++i)
    gidmap[inmat.domain_map().gid(i)] = newdomainmap.gid(i);
  ex.do_export(gidmap);

  // transform input matrix to newdomainmap
  for (int i = 0; i < inmat.num_my_rows(); ++i)
  {
    int NumEntries = 0;
    double* Values;
    int* Indices;
    int err = inmat.extract_my_row_view(i, NumEntries, Values, Indices);
    if (err != 0) FOUR_C_THROW("extract_my_row_view error: {}", err);
    std::vector<int> idx;
    std::vector<double> vals;
    idx.reserve(NumEntries);
    vals.reserve(NumEntries);

    for (int j = 0; j < NumEntries; ++j)
    {
      int gid = (inmat.col_map()).gid(Indices[j]);
      std::map<int, int>::const_iterator iter = gidmap.find(gid);
      if (iter != gidmap.end())
      {
        idx.push_back(iter->second);
        vals.push_back(Values[j]);
      }
      else
        FOUR_C_THROW("gid {} not found in map for lid {} at {}", gid, Indices[j], j);
    }

    Values = vals.data();
    NumEntries = vals.size();
    err = outmat->insert_global_values(inmat.row_map().gid(i), NumEntries, Values, idx.data());
    if (err < 0) FOUR_C_THROW("insert_global_values error: {}", err);
  }

  // complete output matrix
  outmat->complete(newdomainmap, inmat.row_map());

  return outmat;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::matrix_row_col_transform_gids(
    const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap,
    const Core::LinAlg::Map& newdomainmap)
{
  // initialize output matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> outmat =
      std::make_shared<Core::LinAlg::SparseMatrix>(newrowmap, 100, true, true);

  // mapping of column gids
  std::map<int, int> gidmap;
  Core::Communication::Exporter ex(
      inmat.domain_map(), inmat.col_map(), Core::Communication::unpack_epetra_comm(inmat.Comm()));
  for (int i = 0; i < inmat.domain_map().num_my_elements(); ++i)
    gidmap[inmat.domain_map().gid(i)] = newdomainmap.gid(i);
  ex.do_export(gidmap);

  // transform input matrix to newrowmap and newdomainmap
  for (int i = 0; i < inmat.num_my_rows(); ++i)
  {
    int NumEntries = 0;
    double* Values;
    int* Indices;
    int err = inmat.extract_my_row_view(i, NumEntries, Values, Indices);
    if (err != 0) FOUR_C_THROW("extract_my_row_view error: {}", err);
    std::vector<int> idx;
    std::vector<double> vals;
    idx.reserve(NumEntries);
    vals.reserve(NumEntries);

    for (int j = 0; j < NumEntries; ++j)
    {
      int gid = (inmat.col_map()).gid(Indices[j]);
      std::map<int, int>::const_iterator iter = gidmap.find(gid);
      if (iter != gidmap.end())
      {
        idx.push_back(iter->second);
        vals.push_back(Values[j]);
      }
      else
        FOUR_C_THROW("gid {} not found in map for lid {} at {}", gid, Indices[j], j);
    }

    Values = vals.data();
    NumEntries = vals.size();
    err = outmat->insert_global_values(newrowmap.gid(i), NumEntries, Values, idx.data());
    if (err < 0) FOUR_C_THROW("insert_global_values error: {}", err);
  }

  // complete output matrix
  outmat->complete(newdomainmap, newrowmap);

  return outmat;
}

FOUR_C_NAMESPACE_CLOSE
