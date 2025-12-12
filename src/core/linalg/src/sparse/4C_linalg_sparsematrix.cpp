// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_sparsematrix.hpp"

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(std::shared_ptr<Core::LinAlg::Graph> crsgraph,
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> dbcmaps)
    : explicitdirichlet_(true), savegraph_(true), matrixtype_(CRS_MATRIX)
{
  sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, crsgraph->get_epetra_crs_graph());
  graph_ = crsgraph;
  dbcmaps_ = dbcmaps;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const Core::LinAlg::Map& rowmap, const int npr,
    bool explicitdirichlet, bool savegraph, MatrixType matrixtype)
    : graph_(nullptr),
      dbcmaps_(nullptr),
      explicitdirichlet_(explicitdirichlet),
      savegraph_(savegraph),
      matrixtype_(matrixtype)
{
  if (!rowmap.unique_gids()) FOUR_C_THROW("Row map is not unique");

  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap.get_epetra_map(), npr, false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, rowmap.get_epetra_map(), npr, false);
  else
    FOUR_C_THROW("matrix type is not correct");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const Core::LinAlg::Map& rowmap,
    const Core::LinAlg::Map& colmap, const int npr, bool explicitdirichlet, bool savegraph,
    MatrixType matrixtype)
    : graph_(nullptr),
      dbcmaps_(nullptr),
      explicitdirichlet_(explicitdirichlet),
      savegraph_(savegraph),
      matrixtype_(matrixtype)
{
  if (!rowmap.unique_gids()) FOUR_C_THROW("Row map is not unique");

  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), colmap.get_epetra_map(), npr, false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), colmap.get_epetra_map(), npr, false);
  else
    FOUR_C_THROW("matrix type is not correct");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const Core::LinAlg::Map& rowmap,
    std::vector<int>& numentries, bool explicitdirichlet, bool savegraph, MatrixType matrixtype)
    : graph_(nullptr),
      dbcmaps_(nullptr),
      explicitdirichlet_(explicitdirichlet),
      savegraph_(savegraph),
      matrixtype_(matrixtype)
{
  if (!rowmap.unique_gids()) FOUR_C_THROW("Row map is not unique");

  if ((int)(numentries.size()) != rowmap.num_my_elements())
    FOUR_C_THROW("estimate for non zero entries per row does not match the size of row map");

  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), numentries.data(), false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), numentries.data(), false);
  else
    FOUR_C_THROW("matrix type is not correct");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(std::shared_ptr<Epetra_CrsMatrix> matrix,
    DataAccess access, bool explicitdirichlet, bool savegraph, MatrixType matrixtype)
    : graph_(nullptr),
      dbcmaps_(nullptr),
      explicitdirichlet_(explicitdirichlet),
      savegraph_(savegraph),
      matrixtype_(matrixtype)
{
  if (access == DataAccess::Copy)
  {
    if (matrixtype_ == CRS_MATRIX)
      sysmat_ = std::make_shared<Epetra_CrsMatrix>(*matrix);
    else if (matrixtype_ == FE_MATRIX)
    {
      sysmat_ = std::make_shared<Epetra_FECrsMatrix>(
          *(std::dynamic_pointer_cast<Epetra_FECrsMatrix>(matrix)));
    }
    else
      FOUR_C_THROW("matrix type is not correct");
  }
  else
  {
    if (matrixtype_ == CRS_MATRIX)
      sysmat_ = matrix;
    else if (matrixtype_ == FE_MATRIX)
      sysmat_ = std::dynamic_pointer_cast<Epetra_FECrsMatrix>(matrix);
    else
      FOUR_C_THROW("matrix type is not correct");
  }

  if (filled() and savegraph_)
  {
    graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const SparseMatrix& mat, DataAccess access)
    : explicitdirichlet_(mat.explicitdirichlet_),
      savegraph_(mat.savegraph_),
      matrixtype_(mat.matrixtype_)
{
  if (access == DataAccess::Copy)
  {
    // We do not care for exception proved code, so this is ok.
    *this = mat;
  }
  else
  {
    sysmat_ = mat.sysmat_;
    graph_ = mat.graph_;
    matrixtype_ = mat.matrixtype_;
    dbcmaps_ = mat.dbcmaps_;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const Core::LinAlg::Vector<double>& diag,
    bool explicitdirichlet, bool savegraph, MatrixType matrixtype)
    : graph_(nullptr),
      dbcmaps_(nullptr),
      explicitdirichlet_(explicitdirichlet),
      savegraph_(savegraph),
      matrixtype_(matrixtype)
{
  int length = diag.get_map().num_my_elements();
  Core::LinAlg::Map map(-1, length, diag.get_map().my_global_elements(),
      diag.get_map().index_base(), diag.get_comm());
  if (!map.unique_gids()) FOUR_C_THROW("Row map is not unique");

  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(
        ::Copy, map.get_epetra_map(), map.get_epetra_map(), 1, false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(
        ::Copy, map.get_epetra_map(), map.get_epetra_map(), 1, false);
  else
    FOUR_C_THROW("matrix type is not correct");

  for (int i = 0; i < length; ++i)
  {
    int gid = map.gid(i);
    assemble(diag.local_values_as_span()[i], gid, gid);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix& Core::LinAlg::SparseMatrix::operator=(const SparseMatrix& mat)
{
  explicitdirichlet_ = mat.explicitdirichlet_;
  savegraph_ = mat.savegraph_;
  matrixtype_ = mat.matrixtype_;
  dbcmaps_ = mat.dbcmaps_;

  if (not mat.filled())
  {
    // No communication. If just one processor fails, MPI will stop the other
    // ones as well.
    int nonzeros = mat.sysmat_->NumMyNonzeros();
    if (nonzeros > 0) FOUR_C_THROW("cannot copy non-filled matrix");
  }

  if (mat.filled())
  {
    if (matrixtype_ == CRS_MATRIX)
      sysmat_ = std::make_shared<Epetra_CrsMatrix>(*mat.sysmat_);
    else if (matrixtype_ == FE_MATRIX)
    {
      sysmat_ =
          std::make_shared<Epetra_FECrsMatrix>(dynamic_cast<Epetra_FECrsMatrix&>(*mat.sysmat_));
    }
    else
      FOUR_C_THROW("matrix type is not correct");
  }
  else
  {
    if (matrixtype_ == CRS_MATRIX)
      sysmat_ =
          std::make_shared<Epetra_CrsMatrix>(::Copy, mat.row_map().get_epetra_map(), 0, false);
    else if (matrixtype_ == FE_MATRIX)
      sysmat_ =
          std::make_shared<Epetra_FECrsMatrix>(::Copy, mat.row_map().get_epetra_map(), 0, false);
    else
      FOUR_C_THROW("matrix type is not correct");
  }

  if (mat.graph_ != nullptr)
    graph_ = std::make_shared<Core::LinAlg::Graph>(*mat.graph_);
  else
    graph_ = nullptr;

  return *this;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::assign(DataAccess access, const SparseMatrix& mat)
{
  if (access == DataAccess::Copy)
  {
    // We do not care for exception proved code, so this is ok.
    *this = mat;
  }
  else
  {
    sysmat_ = mat.sysmat_;
    graph_ = mat.graph_;
    explicitdirichlet_ = mat.explicitdirichlet_;
    savegraph_ = mat.savegraph_;
    matrixtype_ = mat.matrixtype_;
    dbcmaps_ = mat.dbcmaps_;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::zero()
{
  if (graph_ == nullptr)
  {
    if (filled() && !explicitdirichlet_)
      put_scalar(0.);
    else
      reset();
  }
  else
  {
    // Remove old matrix before creating a new one so we do not have old and
    // new matrix in memory at the same time!
    sysmat_ = nullptr;
    if (matrixtype_ == CRS_MATRIX)
      sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, graph_->get_epetra_crs_graph());
    else if (matrixtype_ == FE_MATRIX)
      sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, graph_->get_epetra_crs_graph());
    else
      FOUR_C_THROW("matrix type is not correct");

    complete(domain_map(), range_map());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::reset()
{
  const Core::LinAlg::Map rowmap = row_map();
  std::vector<int> numentries(rowmap.num_my_elements());

  Core::LinAlg::Graph graph(sysmat_->Graph());

  if (filled())
  {
    for (std::size_t i = 0; i < numentries.size(); ++i)
    {
      std::span<int> indices;
      graph.extract_local_row_view(i, indices);
      numentries[i] = indices.size();
    }
  }
  else
  {
    // use information about number of allocated entries not to fall back to matrix with zero size
    // otherwise assembly would be extremely expensive!
    for (std::size_t i = 0; i < numentries.size(); ++i)
    {
      numentries[i] = graph.num_allocated_local_indices(i);
    }
  }
  // Remove old matrix before creating a new one so we do not have old and
  // new matrix in memory at the same time!
  sysmat_ = nullptr;
  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), numentries.data(), false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), numentries.data(), false);
  else
    FOUR_C_THROW("matrix type is not correct");

  graph_ = nullptr;
  dbcmaps_ = nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::assemble(int eid, const std::vector<int>& lmstride,
    const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
    const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
{
  const int lrowdim = (int)lmrow.size();
  const int lcoldim = (int)lmcol.size();
  // allow Aele to provide entries past the end of lmrow and lmcol that are
  // not used here, therefore check only for ">" rather than "!="
  if (lrowdim != (int)lmrowowner.size() || lrowdim > Aele.numRows() || lcoldim > Aele.numCols())
    FOUR_C_THROW("Mismatch in dimensions");

  const int myrank =
      Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(sysmat_->Comm()));
  const Core::LinAlg::Map& rowmap = row_map();
  const Core::LinAlg::Map& colmap = col_map();

  auto& A = (Core::LinAlg::SerialDenseMatrix&)Aele;
  if (filled())  // assembly in local indices
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    // There is the case of nodes without dofs (XFEM).
    // If no row dofs are present on this proc, there is nothing to assemble.
    // However, the subsequent check for coldofs (in DEBUG mode) would incorrectly fail.
    bool doit = false;
    for (int lrow = 0; lrow < lrowdim; ++lrow)
      if (lmrowowner[lrow] == myrank)
      {
        doit = true;
        break;
      }
    if (!doit) return;
#endif

    std::vector<int> localcol(lcoldim);
    for (int lcol = 0; lcol < lcoldim; ++lcol)
    {
      const int cgid = lmcol[lcol];
      localcol[lcol] = colmap.lid(cgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (localcol[lcol] < 0) FOUR_C_THROW("Sparse matrix A does not have global column {}", cgid);
#endif
    }

    // loop rows of local matrix
    for (int lrow = 0; lrow < lrowdim; ++lrow)
    {
      // check ownership of row
      if (lmrowowner[lrow] != myrank) continue;

      // check whether I have that global row
      const int rgid = lmrow[lrow];

      // if we have a Dirichlet map check if this row is a Dirichlet row
      if (dbcmaps_ != nullptr and dbcmaps_->map(1)->my_gid(rgid)) continue;

      const int rlid = rowmap.lid(rgid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (rlid < 0) FOUR_C_THROW("Sparse matrix A does not have global row {}", rgid);
#endif
      int length;
      double* valview;
      int* indices;
      extract_my_row_view(rlid, length, valview, indices);

      const int numnode = (int)lmstride.size();
      int dofcount = 0;
      int pos = 0;
      for (int node = 0; node < numnode; ++node)
      {
        // check if 'pos' already points to the correct location before the binary search
        if (pos >= length || indices[pos] != localcol[dofcount])
        {
          int* loc = std::lower_bound(indices, indices + length, localcol[dofcount]);
#ifdef FOUR_C_ENABLE_ASSERTIONS
          if (*loc != localcol[dofcount])
            FOUR_C_THROW("Cannot find local column entry {}", localcol[dofcount]);
#endif
          pos = loc - indices;
        }
        const int stride = lmstride[node];
        // test continuity of data in sparsematrix
        bool reachedlength = false;
        bool continuous = true;
        if (stride + pos > length)
          continuous = false;
        else
        {
          for (int j = 1; j < stride; ++j)
          {
            if (indices[pos + j] == localcol[dofcount + j])
              continue;
            else
            {
              continuous = false;
              break;
            }
          }
        }

        if (continuous)
        {
          for (int j = 0; j < stride; ++j)
          {
            valview[pos++] += Aele(lrow, dofcount++);
            if (dofcount == lcoldim)
            {
              reachedlength = true;
              break;
            }
          }
        }
        else
        {
          for (int j = 0; j < stride; ++j)
          {
            sum_into_my_values(rlid, 1, &A(lrow, dofcount), &localcol[dofcount]);

            dofcount++;
            if (dofcount == lcoldim)
            {
              reachedlength = true;
              break;
            }
          }
        }
        if (reachedlength) break;
      }  // for (int node=0; node<numnode; ++node)
    }  // for (int lrow=0; lrow<ldim; ++lrow)
  }
  else  // assembly in global indices
  {
    // loop rows of local matrix
    for (int lrow = 0; lrow < lrowdim; ++lrow)
    {
      // check ownership of row
      if (lmrowowner[lrow] != myrank) continue;

      // check whether I have that global row
      const int rgid = lmrow[lrow];
      // #ifdef   FOUR_C_ENABLE_ASSERTIONS
      if (!rowmap.my_gid(rgid)) FOUR_C_THROW("Proc {} does not have global row {}", myrank, rgid);
      // #endif

      // if we have a Dirichlet map check if this row is a Dirichlet row
      if (dbcmaps_ != nullptr and dbcmaps_->map(1)->my_gid(rgid)) continue;

      for (int lcol = 0; lcol < lcoldim; ++lcol)
      {
        int cgid = lmcol[lcol];
        // Now that we do not rebuild the sparse mask in each step, we
        // are bound to assemble the whole thing. Zeros included.
        sum_or_insert_global_values(rgid, 1, &A(lrow, lcol), &cgid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::assemble(int eid, const Core::LinAlg::SerialDenseMatrix& Aele,
    const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
    const std::vector<int>& lmcol)
{
  const int lrowdim = (int)lmrow.size();
  const int lcoldim = (int)lmcol.size();
  // allow Aele to provide entries past the end of lmrow and lmcol that are
  // not used here, therefore check only for ">" rather than "!="
  if (lrowdim != (int)lmrowowner.size() || lrowdim > Aele.numRows() || lcoldim > Aele.numCols())
    FOUR_C_THROW("Mismatch in dimensions");

  const int myrank =
      Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(sysmat_->Comm()));
  const Core::LinAlg::Map& rowmap = row_map();
  const Core::LinAlg::Map& colmap = col_map();

  if (filled())
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    // There is the case of nodes without dofs (XFEM).
    // If no row dofs are present on this proc, their is nothing to assemble.
    // However, the subsequent check for coldofs (in DEBUG mode) would incorrectly fail.
    bool doit = false;
    for (int lrow = 0; lrow < lrowdim; ++lrow)
      if (lmrowowner[lrow] == myrank)
      {
        doit = true;
        break;
      }
    if (!doit) return;
#endif

    std::vector<double> values(lcoldim);
    std::vector<int> localcol(lcoldim);
    for (int lcol = 0; lcol < lcoldim; ++lcol)
    {
      const int cgid = lmcol[lcol];
      localcol[lcol] = colmap.lid(cgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (localcol[lcol] < 0) FOUR_C_THROW("Sparse matrix A does not have global column {}", cgid);
#endif
    }

    // loop rows of local matrix
    for (int lrow = 0; lrow < lrowdim; ++lrow)
    {
      // check ownership of row
      if (lmrowowner[lrow] != myrank) continue;

      // check whether I have that global row
      const int rgid = lmrow[lrow];

      // if we have a Dirichlet map check if this row is a Dirichlet row
      if (dbcmaps_ != nullptr and dbcmaps_->map(1)->my_gid(rgid)) continue;

      const int rlid = rowmap.lid(rgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (rlid < 0) FOUR_C_THROW("Sparse matrix A does not have global row {}", rgid);
#endif

      for (int lcol = 0; lcol < lcoldim; ++lcol)
      {
        values[lcol] = Aele(lrow, lcol);
      }
      sum_into_my_values(rlid, lcoldim, values.data(), localcol.data());
    }
  }
  else
  {
    // loop rows of local matrix
    for (int lrow = 0; lrow < lrowdim; ++lrow)
    {
      // check ownership of row
      if (lmrowowner[lrow] != myrank) continue;

      // check whether I have that global row
      const int rgid = lmrow[lrow];
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (!rowmap.my_gid(rgid)) FOUR_C_THROW("Proc {} does not have global row {}", myrank, rgid);
#endif

      // if we have a Dirichlet map check if this row is a Dirichlet row
      if (dbcmaps_ != nullptr and dbcmaps_->map(1)->my_gid(rgid)) continue;

      for (int lcol = 0; lcol < lcoldim; ++lcol)
      {
        double val = Aele(lrow, lcol);
        int cgid = lmcol[lcol];

        // Now that we do not rebuild the sparse mask in each step, we
        // are bound to assemble the whole thing. Zeros included.
        sum_or_insert_global_values(rgid, 1, &val, &cgid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::fe_assemble(const Core::LinAlg::SerialDenseMatrix& Aele,
    const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
    const std::vector<int>& lmcol)
{
  const int lrowdim = static_cast<int>(lmrow.size());
  const int lcoldim = static_cast<int>(lmcol.size());

  // allow Aele to provide entries past the end of lmrow and lmcol that are
  // not used here, therefore check only for ">" rather than "!="
  if (lrowdim != (int)lmrowowner.size() || lrowdim > Aele.numRows() || lcoldim > Aele.numCols())
    FOUR_C_THROW("Mismatch in dimensions");

  std::shared_ptr<Epetra_FECrsMatrix> fe_mat =
      std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_);
  const int myrank =
      Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(fe_mat->Comm()));

  // loop rows of local matrix
  for (int lrow = 0; lrow < lrowdim; ++lrow)
  {
    // check ownership of row
    if (lmrowowner[lrow] != myrank) continue;

    const int rgid = lmrow[lrow];

    for (int lcol = 0; lcol < lcoldim; ++lcol)
    {
      double val = Aele(lrow, lcol);
      const int cgid = lmcol[lcol];
      fe_assemble(val, rgid, cgid);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::fe_assemble(const Core::LinAlg::SerialDenseMatrix& Aele,
    const std::vector<int>& lmrow, const std::vector<int>& lmcol)
{
  const int lrowdim = static_cast<int>(lmrow.size());
  const int lcoldim = static_cast<int>(lmcol.size());
  // allow Aele to provide entries past the end of lmrow and lmcol that are
  // not used here, therefore check only for ">" rather than "!="
  if (lrowdim > Aele.numRows() || lcoldim > Aele.numCols()) FOUR_C_THROW("Mismatch in dimensions");

  std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_);

  // loop rows of local matrix
  for (int lrow = 0; lrow < lrowdim; ++lrow)
  {
    const int rgid = lmrow[lrow];

    for (int lcol = 0; lcol < lcoldim; ++lcol)
    {
      double val = Aele(lrow, lcol);
      const int cgid = lmcol[lcol];
      fe_assemble(val, rgid, cgid);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::assemble(double val, int rgid, int cgid)
{
  if (dbcmaps_ != nullptr and dbcmaps_->map(1)->my_gid(rgid))
    FOUR_C_THROW("no assembling to Dirichlet row");

  sum_or_insert_global_values(rgid, 1, &val, &cgid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::set_value(double val, int rgid, int cgid)
{
  if (dbcmaps_ != nullptr and dbcmaps_->map(1)->my_gid(rgid))
    FOUR_C_THROW("no assembling to Dirichlet row");

  replace_or_insert_global_values(rgid, 1, &val, &cgid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::fe_assemble(double val, int rgid, int cgid)
{
  // SumIntoGlobalValues works for filled matrices as well!
  int errone = (std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_))
                   ->SumIntoGlobalValues(1, &rgid, 1, &cgid, &val);
  // if value not already present , error > 0 then use insert method
  if (errone > 0 and not filled())
  {
    int errtwo = (std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_))
                     ->InsertGlobalValues(1, &rgid, 1, &cgid, &val);
    if (errtwo < 0)
      FOUR_C_THROW("Epetra_FECrsMatrix::InsertGlobalValues returned error code {}", errtwo);
  }
  else if (errone < 0)
    FOUR_C_THROW("Epetra_FECrsMatrix::SumIntoGlobalValues returned error code {}", errone);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::complete(OptionsMatrixComplete options_matrix_complete)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SparseMatrix::Complete");

  // for FE_Matrix we need to gather non-local entries, independent whether matrix is filled or not
  if (matrixtype_ == FE_MATRIX)
  {
    // false indicates here that fill_complete() is not called within GlobalAssemble()
    ASSERT_EPETRA_CALL(
        std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_)->GlobalAssemble(false));
  }

  if (filled() and not options_matrix_complete.enforce_complete) return;

  ASSERT_EPETRA_CALL(sysmat_->FillComplete(options_matrix_complete.optimize_data_storage));

  // keep mask for further use
  if (savegraph_ and graph_ == nullptr)
  {
    graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::complete(const Core::LinAlg::Map& domainmap,
    const Core::LinAlg::Map& rangemap, OptionsMatrixComplete options_matrix_complete)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SparseMatrix::Complete(domain,range)");

  // for FE_Matrix we need to gather non-local entries, independent whether matrix is filled or not
  if (matrixtype_ == FE_MATRIX)
  {
    // false indicates here that fill_complete() is not called within GlobalAssemble()
    ASSERT_EPETRA_CALL(std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_)->GlobalAssemble(
        domainmap.get_epetra_map(), rangemap.get_epetra_map(), false));
  }

  if (filled() and not options_matrix_complete.enforce_complete) return;

  if (options_matrix_complete.enforce_complete and filled())
  {
    ASSERT_EPETRA_CALL(
        sysmat_->ExpertStaticFillComplete(domainmap.get_epetra_map(), rangemap.get_epetra_map()));
  }
  else
  {
    ASSERT_EPETRA_CALL(sysmat_->FillComplete(domainmap.get_epetra_map(), rangemap.get_epetra_map(),
        options_matrix_complete.optimize_data_storage));
  }

  // keep mask for further use
  if (savegraph_ and graph_ == nullptr)
  {
    graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::un_complete()
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SparseMatrix::UnComplete");

  if (not filled()) return;

  Core::LinAlg::Graph graph(sysmat_->Graph());

  std::vector<int> nonzeros(graph.num_local_rows());
  for (std::size_t i = 0; i < nonzeros.size(); ++i)
  {
    nonzeros[i] = graph.num_local_indices(i);
  }

  const Core::LinAlg::Map& rowmap = row_map();
  const Core::LinAlg::Map& colmap = col_map();
  int elements = rowmap.num_my_elements();

  std::shared_ptr<Epetra_CrsMatrix> mat = nullptr;
  if (matrixtype_ == CRS_MATRIX)
    mat =
        std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap.get_epetra_map(), nonzeros.data(), false);
  else if (matrixtype_ == FE_MATRIX)
    mat = std::make_shared<Epetra_FECrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), nonzeros.data(), false);
  else
    FOUR_C_THROW("matrix type is not correct");

  nonzeros.clear();
  for (int i = 0; i < elements; ++i)
  {
    int NumEntries;
    double* Values;
    int* Indices;
    // if matrix is filled, global assembly was called already and all nonlocal values are
    // distributed
    extract_my_row_view(i, NumEntries, Values, Indices);

    std::vector<int> idx(NumEntries);
    for (int c = 0; c < NumEntries; ++c)
    {
      idx[c] = colmap.gid(Indices[c]);
      FOUR_C_ASSERT(idx[c] != -1, "illegal gid");
    }
    int rowgid = rowmap.gid(i);
    int err = mat->InsertGlobalValues(rowgid, NumEntries, Values, idx.data());
    if (err) FOUR_C_THROW("InsertGlobalValues err={}", err);
  }
  sysmat_ = mat;
  graph_ = nullptr;
}


/*----------------------------------------------------------------------*
 |  Apply dirichlet conditions  (public)                     mwgee 02/07|
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::apply_dirichlet(
    const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock)
{
  // if matrix is filled, global assembly was called already and all nonlocal values are
  // distributed
  if (not filled()) FOUR_C_THROW("expect filled matrix to apply dirichlet conditions");

  if (dbcmaps_ != nullptr)
  {
    FOUR_C_THROW("Dirichlet map and toggle vector cannot be combined");
  }

  if (explicitdirichlet_)
  {
    // Save graph of original matrix if not done already.
    // This will never happen as the matrix is guaranteed to be filled. But to
    // make the code more explicit...
    if (savegraph_ and graph_ == nullptr)
    {
      graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
      if (not graph_->filled()) FOUR_C_THROW("got unfilled graph from filled matrix");
    }

    // allocate a new matrix and copy all rows that are not dirichlet
    const Core::LinAlg::Map& rowmap = row_map();
    const int nummyrows = num_my_rows();
    const int maxnumentries = max_num_entries();

    std::shared_ptr<Epetra_CrsMatrix> Anew = nullptr;
    if (matrixtype_ == CRS_MATRIX)
      Anew =
          std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap.get_epetra_map(), maxnumentries, false);
    else if (matrixtype_ == FE_MATRIX)
      Anew = std::make_shared<Epetra_FECrsMatrix>(
          ::Copy, rowmap.get_epetra_map(), maxnumentries, false);
    else
      FOUR_C_THROW("matrix type is not correct");

    std::vector<int> indices(maxnumentries, 0);
    std::vector<double> values(maxnumentries, 0.0);
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = global_row_index(i);
      if (dbctoggle.local_values_as_span()[i] != 1.0)
      {
        int numentries;
        extract_global_row_copy(row, maxnumentries, numentries, values.data(), indices.data());

        // this is also ok for FE matrices, because fill complete was called on sysmat and the
        // globalAssemble method was called already
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err={}", err);
#else
        Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
#endif
      }
      else
      {
        double v;
        if (diagonalblock)
          v = 1.0;
        else
          v = 0.0;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = Anew->InsertGlobalValues(row, 1, &v, &row);
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err={}", err);
#else
        Anew->InsertGlobalValues(row, 1, &v, &row);
#endif
      }
    }
    sysmat_ = Anew;
    complete();
  }
  else
  {
    const int nummyrows = num_my_rows();
    for (int i = 0; i < nummyrows; ++i)
    {
      if (dbctoggle.local_values_as_span()[i] == 1.0)
      {
        int* indexOffset;
        int* indices;
        double* values;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractCrsDataPointers returned err={}", err);
#else
        sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
#endif
        // zero row
        memset(&values[indexOffset[i]], 0, (indexOffset[i + 1] - indexOffset[i]) * sizeof(double));

        if (diagonalblock)
        {
          double one = 1.0;
          sum_into_my_values(i, 1, &one, &i);
        }
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Apply dirichlet conditions  (public)                     mwgee 02/07|
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::apply_dirichlet(
    const Core::LinAlg::Map& dbctoggle, bool diagonalblock)
{
  if (not filled()) FOUR_C_THROW("expect filled matrix to apply dirichlet conditions");

  if (dbcmaps_ != nullptr)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (not dbctoggle.same_as(*dbcmaps_->map(1)))
    {
      FOUR_C_THROW("Dirichlet maps mismatch");
    }
#endif
    if (diagonalblock)
    {
      double v = 1.0;
      int numdbc = dbctoggle.num_my_elements();
      int* dbc = dbctoggle.my_global_elements();
      for (int i = 0; i < numdbc; ++i)
      {
        int row = dbc[i];
        replace_global_values(row, 1, &v, &row);
      }
    }
    return;
  }

  if (explicitdirichlet_)
  {
    // Save graph of original matrix if not done already.
    // This will never happen as the matrix is guaranteed to be filled. But to
    // make the code more explicit...
    if (savegraph_ and graph_ == nullptr)
    {
      graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
      if (not graph_->filled()) FOUR_C_THROW("got unfilled graph from filled matrix");
    }

    // allocate a new matrix and copy all rows that are not dirichlet
    const Core::LinAlg::Map& rowmap = row_map();
    const int nummyrows = num_my_rows();
    const int maxnumentries = max_num_entries();

    // std::shared_ptr<Epetra_CrsMatrix> Anew = Teuchos::rcp(new
    // Epetra_CrsMatrix(Copy,rowmap,maxnumentries,false));

    std::shared_ptr<Epetra_CrsMatrix> Anew = nullptr;
    if (matrixtype_ == CRS_MATRIX)
      Anew =
          std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap.get_epetra_map(), maxnumentries, false);
    else if (matrixtype_ == FE_MATRIX)
      Anew = std::make_shared<Epetra_FECrsMatrix>(
          ::Copy, rowmap.get_epetra_map(), maxnumentries, false);
    else
      FOUR_C_THROW("matrix type is not correct");

    std::vector<int> indices(maxnumentries, 0);
    std::vector<double> values(maxnumentries, 0.0);
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = global_row_index(i);
      if (not dbctoggle.my_gid(row))
      {
        int numentries;
        extract_global_row_copy(row, maxnumentries, numentries, values.data(), indices.data());

        // this is also ok for FE matrices, because fill complete was called on sysmat and the
        // globalAssemble method was called already
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err={}", err);
#else
        Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
#endif
      }
      else
      {
        if (diagonalblock)
        {
          double v = 1.0;
#ifdef FOUR_C_ENABLE_ASSERTIONS
          int err = Anew->InsertGlobalValues(row, 1, &v, &row);
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err={}", err);
#else
          Anew->InsertGlobalValues(row, 1, &v, &row);
#endif
        }
      }
    }
    Core::LinAlg::Map rangemap = range_map();
    Core::LinAlg::Map domainmap = domain_map();
    sysmat_ = Anew;
    complete(domainmap, rangemap);
  }
  else
  {
    const int nummyrows = num_my_rows();
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = global_row_index(i);
      if (dbctoggle.my_gid(row))
      {
        int* indexOffset;
        int* indices;
        double* values;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractCrsDataPointers returned err={}", err);
#else
        sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
#endif
        // zero row
        memset(&values[indexOffset[i]], 0, (indexOffset[i + 1] - indexOffset[i]) * sizeof(double));

        if (diagonalblock)
        {
          double one = 1.0;
          sum_into_my_values(i, 1, &one, &i);
        }
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Apply dirichlet conditions  (public)                     mwgee 02/07|
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::apply_dirichlet_with_trafo(const Core::LinAlg::SparseMatrix& trafo,
    const Core::LinAlg::Map& dbctoggle, bool diagonalblock, bool complete)
{
  if (not filled()) FOUR_C_THROW("expect filled matrix to apply dirichlet conditions");

  if (dbcmaps_ != nullptr)
  {
    FOUR_C_THROW("Dirichlet map and transformations cannot be combined");
  }

  if (explicitdirichlet_)
  {
    // Save graph of original matrix if not done already.
    // This will never happen as the matrix is guaranteed to be filled. But to
    // make the code more explicit...
    if (savegraph_ and graph_ == nullptr)
    {
      graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
      if (not graph_->filled()) FOUR_C_THROW("got unfilled graph from filled matrix");
    }

    // allocate a new matrix and copy all rows that are not dirichlet
    const Core::LinAlg::Map& rowmap = row_map();
    const Core::LinAlg::Map& colmap = col_map();
    const int nummyrows = num_my_rows();
    const int maxnumentries = max_num_entries();

    // prepare working arrays for extracting rows in trafo matrix
    const int trafomaxnumentries = trafo.max_num_entries();
    int trafonumentries = 0;
    std::vector<int> trafoindices(trafomaxnumentries, 0);
    std::vector<double> trafovalues(trafomaxnumentries, 0.0);

    // initialise matrix Anew with general size (rowmap x colmap)
    // in case of coupled problem (e.g. TSI) transform the rectangular off-diagonal block k_Td
    std::shared_ptr<Epetra_CrsMatrix> Anew = std::make_shared<Epetra_CrsMatrix>(
        ::Copy, rowmap.get_epetra_map(), colmap.get_epetra_map(), maxnumentries, false);
    std::vector<int> indices(maxnumentries, 0);
    std::vector<double> values(maxnumentries, 0.0);
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = global_row_index(i);
      if (not dbctoggle.my_gid(row))  // dof is not a Dirichlet dof
      {
        int numentries;
        extract_global_row_copy(row, maxnumentries, numentries, values.data(), indices.data());

#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err={}", err);
#else
        Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
#endif
      }
      else  // dof is an inclined Dirichlet dof
      {
        // diagonal block of dof with INCLINED Dirichlet boundary condition
        if (diagonalblock)
        {
          // extract values of trafo at the inclined dbc dof
          trafo.extract_global_row_copy(
              row, trafomaxnumentries, trafonumentries, trafovalues.data(), trafoindices.data());
        }
        // if entry of dof with inclined dbc is not a diagonal block, set zero
        // at this position
        else
        {
          trafonumentries = 1;
          trafovalues[0] = 0.0;
          trafoindices[0] = row;
        }
        // insert all these entries in transformed sysmat, i.e. in Anew
#ifdef FOUR_C_ENABLE_ASSERTIONS
        {
          int err = Anew->InsertGlobalValues(
              row, trafonumentries, trafovalues.data(), trafoindices.data());
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err={}", err);
        }
#else
        Anew->InsertGlobalValues(row, trafonumentries, trafovalues.data(), trafoindices.data());
#endif
      }
    }
    // Updated sysmat_
    // normal DBC dof: '1.0' at diagonal, rest of row is blanked --> row remains
    //                 the same
    // inclined DBC: (in) rotated matrix k^{~}, i.e. '1.0' at diagonal, rest of
    //                    row is blanked for n/t/b-direction
    //               (out) matrix in global system, i.e. k: for a node with 3
    //                     dofs in x/y/z-direction, trafo block is put at the
    //                     position of the dofs of this node, rest of row is blanked
    sysmat_ = Anew;
    if (complete) SparseMatrix::complete();
  }
  else
  {
    const int nummyrows = num_my_rows();

    // prepare working arrays for extracting rows in trafo matrix
    const int trafomaxnumentries = trafo.max_num_entries();
    int trafonumentries = 0;
    std::vector<int> trafoindices(trafomaxnumentries, 0);
    std::vector<double> trafovalues(trafomaxnumentries, 0.0);

    for (int i = 0; i < nummyrows; ++i)
    {
      int row = global_row_index(i);
      if (dbctoggle.my_gid(row))
      {
        int* indexOffset;
        int* indices;
        double* values;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
        if (err) FOUR_C_THROW("Epetra_CrsMatrix::ExtractCrsDataPointers returned err={}", err);
#else
        sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
#endif
        // zero row
        memset(&values[indexOffset[i]], 0, (indexOffset[i + 1] - indexOffset[i]) * sizeof(double));

        if (diagonalblock)
        {
          trafo.extract_my_row_copy(
              i, trafomaxnumentries, trafonumentries, trafovalues.data(), trafoindices.data());

          sum_into_my_values(i, trafonumentries, trafovalues.data(), trafoindices.data());
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::SparseMatrix::extract_dirichlet_rows(
    const Core::LinAlg::Vector<double>& dbctoggle)
{
  if (not filled()) FOUR_C_THROW("expect filled matrix to extract dirichlet lines");

  std::shared_ptr<SparseMatrix> dl = std::make_shared<SparseMatrix>(
      row_map(), max_num_entries(), explicit_dirichlet(), save_graph());

  const Core::LinAlg::Map& rowmap = row_map();
  const Core::LinAlg::Map& colmap = col_map();
  const int nummyrows = num_my_rows();

  const Core::LinAlg::Vector<double>& dbct = dbctoggle;

  std::vector<int> idx(max_num_entries());

  for (int i = 0; i < nummyrows; ++i)
  {
    if (dbct.local_values_as_span()[i] == 1.0)
    {
      int NumEntries;
      double* Values;
      int* Indices;

      extract_my_row_view(i, NumEntries, Values, Indices);
      for (int j = 0; j < NumEntries; ++j) idx[j] = colmap.gid(Indices[j]);

      dl->insert_global_values(rowmap.gid(i), NumEntries, Values, idx.data());
    }
  }

  dl->complete(domain_map(), range_map());
  return dl;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::SparseMatrix::extract_dirichlet_rows(
    const Core::LinAlg::Map& dbctoggle)
{
  if (not filled()) FOUR_C_THROW("expect filled matrix to extract dirichlet lines");
  if (not dbctoggle.unique_gids()) FOUR_C_THROW("unique map required");

  std::shared_ptr<SparseMatrix> dl = std::make_shared<SparseMatrix>(
      row_map(), max_num_entries(), explicit_dirichlet(), save_graph());

  const Core::LinAlg::Map& rowmap = row_map();
  const Core::LinAlg::Map& colmap = col_map();

  std::vector<int> idx(max_num_entries());

  const int mylength = dbctoggle.num_my_elements();
  const int* mygids = dbctoggle.my_global_elements();
  for (int i = 0; i < mylength; ++i)
  {
    int gid = mygids[i];
    int lid = rowmap.lid(gid);

    if (lid < 0) FOUR_C_THROW("illegal Dirichlet map");

    int NumEntries;
    double* Values;
    int* Indices;

    extract_my_row_view(lid, NumEntries, Values, Indices);
    for (int j = 0; j < NumEntries; ++j) idx[j] = colmap.gid(Indices[j]);

    dl->insert_global_values(gid, NumEntries, Values, idx.data());
  }

  dl->complete(domain_map(), range_map());
  return dl;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const char* Core::LinAlg::SparseMatrix::Label() const { return "Core::LinAlg::SparseMatrix"; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::SetUseTranspose(bool UseTranspose)
{
  return sysmat_->SetUseTranspose(UseTranspose);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  return sysmat_->Apply(X, Y);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::ApplyInverse(
    const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  return sysmat_->ApplyInverse(X, Y);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::SparseMatrix::UseTranspose() const { return sysmat_->UseTranspose(); }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::SparseMatrix::HasNormInf() const { return sysmat_->HasNormInf(); }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Comm& Core::LinAlg::SparseMatrix::Comm() const { return sysmat_->Comm(); }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map& Core::LinAlg::SparseMatrix::OperatorDomainMap() const
{
  return sysmat_->OperatorDomainMap();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map& Core::LinAlg::SparseMatrix::OperatorRangeMap() const
{
  return sysmat_->OperatorRangeMap();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::max_num_entries() const { return sysmat_->MaxNumEntries(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::global_max_num_entries() const
{
  return sysmat_->GlobalMaxNumEntries();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Core::LinAlg::SparseMatrix::NormInf() const { return sysmat_->NormInf(); }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Core::LinAlg::SparseMatrix::norm_one() const { return sysmat_->NormOne(); }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Core::LinAlg::SparseMatrix::norm_frobenius() const { return sysmat_->NormFrobenius(); }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::multiply(
    bool TransA, const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& y) const
{
  ASSERT_EPETRA_CALL(
      sysmat_->Multiply(TransA, x.get_ref_of_epetra_vector(), y.get_ref_of_epetra_vector()));
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::multiply(bool TransA, const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y) const
{
  return sysmat_->Multiply(TransA, X.get_epetra_multi_vector(), Y.get_epetra_multi_vector());
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::left_scale(const Core::LinAlg::Vector<double>& x)
{
  ASSERT_EPETRA_CALL(sysmat_->LeftScale(x));
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::right_scale(const Core::LinAlg::Vector<double>& x)
{
  ASSERT_EPETRA_CALL(sysmat_->RightScale(x));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::inv_row_sums(Core::LinAlg::Vector<double>& x) const
{
  ASSERT_EPETRA_CALL(sysmat_->InvRowSums(x));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::inv_col_sums(Core::LinAlg::Vector<double>& x) const
{
  ASSERT_EPETRA_CALL(sysmat_->InvColSums(x));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::put_scalar(double ScalarConstant)
{
  ASSERT_EPETRA_CALL(sysmat_->PutScalar(ScalarConstant));
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::scale(double ScalarConstant)
{
  return sysmat_->Scale(ScalarConstant);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::SparseMatrix::replace_diagonal_values(
    const Core::LinAlg::Vector<double>& Diagonal)
{
  // For now, we cant add ASSERT_EPETRA_CALL() here, as Epetra internally handles the error flag
  // inconsistently.
  return sysmat_->ReplaceDiagonalValues(Diagonal);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::extract_diagonal_copy(Core::LinAlg::Vector<double>& Diagonal) const
{
  ASSERT_EPETRA_CALL(sysmat_->ExtractDiagonalCopy(Diagonal.get_ref_of_epetra_vector()));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::extract_my_row_copy(
    int my_row, int length, int& num_entries, double* values, int* indices) const
{
  ASSERT_EPETRA_CALL(sysmat_->ExtractMyRowCopy(my_row, length, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::extract_global_row_copy(
    int global_row, int length, int& num_entries, double* values, int* indices) const
{
  ASSERT_EPETRA_CALL(
      sysmat_->ExtractGlobalRowCopy(global_row, length, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::extract_my_row_view(
    int my_row, int& num_entries, double*& values, int*& indices) const
{
  ASSERT_EPETRA_CALL(sysmat_->ExtractMyRowView(my_row, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::extract_global_row_view(
    int global_row, int& num_entries, double*& values, int*& indices) const
{
  ASSERT_EPETRA_CALL(sysmat_->ExtractGlobalRowView(global_row, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::insert_my_values(
    int my_row, int num_entries, const double* values, const int* indices)
{
  ASSERT_EPETRA_CALL(sysmat_->InsertMyValues(my_row, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::sum_into_my_values(
    int my_row, int num_entries, const double* values, const int* indices)
{
  ASSERT_EPETRA_CALL(sysmat_->SumIntoMyValues(my_row, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::replace_my_values(
    int my_row, int num_entries, const double* values, const int* indices)
{
  ASSERT_EPETRA_CALL(sysmat_->ReplaceMyValues(my_row, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::replace_global_values(
    int global_row, int num_entries, const double* values, const int* indices)
{
  ASSERT_EPETRA_CALL(sysmat_->ReplaceGlobalValues(global_row, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::insert_global_values(
    int global_row, int num_entries, const double* values, const int* indices)
{
  // For now, we cant add ASSERT_EPETRA_CALL() here, as we need to allow warnings.
  [[maybe_unused]] int err = sysmat_->InsertGlobalValues(global_row, num_entries, values, indices);
  FOUR_C_ASSERT(err >= 0, "Epetra error (code {}).", err);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::sum_into_global_values(
    int global_row, int num_entries, const double* values, const int* indices)
{
  ASSERT_EPETRA_CALL(sysmat_->SumIntoGlobalValues(global_row, num_entries, values, indices));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::sum_or_insert_global_values(
    int global_row, int num_entries, const double* values, const int* indices)
{
  const int errone = sysmat_->SumIntoGlobalValues(global_row, num_entries, values, indices);
  if (errone > 0)
  {
    [[maybe_unused]] int errtwo =
        sysmat_->InsertGlobalValues(global_row, num_entries, values, indices);
    FOUR_C_ASSERT(errtwo >= 0, "Epetra error (code {}).", errtwo);
  }
  else if (errone < 0)
  {
    FOUR_C_THROW("Epetra error (code {}).", errone);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::replace_or_insert_global_values(
    int global_row, int num_entries, const double* values, const int* indices)
{
  const int errone = sysmat_->ReplaceGlobalValues(global_row, num_entries, values, indices);
  if (errone > 0)
  {
    [[maybe_unused]] int errtwo =
        sysmat_->InsertGlobalValues(global_row, num_entries, values, indices);
    FOUR_C_ASSERT(errtwo >= 0, "Epetra error (code {}).", errtwo);
  }
  else if (errone < 0)
  {
    FOUR_C_THROW("Epetra error (code {}).", errone);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::add(const Core::LinAlg::SparseOperator& A, const bool transposeA,
    const double scalarA, const double scalarB)
{
  A.add_other(*this, transposeA, scalarA, scalarB);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::add_other(Core::LinAlg::SparseMatrix& B, const bool transposeA,
    const double scalarA, const double scalarB) const
{
  Core::LinAlg::matrix_add(*this, transposeA, scalarA, B, scalarB);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::add_other(Core::LinAlg::BlockSparseMatrixBase& B,
    const bool transposeA, const double scalarA, const double scalarB) const
{
  FOUR_C_THROW("BlockSparseMatrix and SparseMatrix cannot be added");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::cast_to_sparse_matrix_and_check_success(
    std::shared_ptr<Core::LinAlg::SparseOperator> input_matrix)
{
  auto sparse_matrix = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(input_matrix);
  FOUR_C_ASSERT(sparse_matrix != nullptr, "Matrix is not a sparse matrix!");

  return sparse_matrix;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::SparseMatrix>
Core::LinAlg::cast_to_const_sparse_matrix_and_check_success(
    std::shared_ptr<const Core::LinAlg::SparseOperator> input_matrix)
{
  auto sparse_matrix = std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(input_matrix);
  FOUR_C_ASSERT(sparse_matrix != nullptr, "Matrix is not a sparse matrix!");

  return sparse_matrix;
}

FOUR_C_NAMESPACE_CLOSE
