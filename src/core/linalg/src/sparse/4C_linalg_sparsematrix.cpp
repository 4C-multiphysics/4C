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
  sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, *crsgraph->get_ptr_of_Epetra_CrsGraph());
  graph_ = crsgraph;
  dbcmaps_ = dbcmaps;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const Epetra_Map& rowmap, const int npr,
    bool explicitdirichlet, bool savegraph, MatrixType matrixtype)
    : graph_(nullptr),
      dbcmaps_(nullptr),
      explicitdirichlet_(explicitdirichlet),
      savegraph_(savegraph),
      matrixtype_(matrixtype)
{
  if (!rowmap.UniqueGIDs()) FOUR_C_THROW("Row map is not unique");

  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, npr, false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, rowmap, npr, false);
  else
    FOUR_C_THROW("matrix type is not correct");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const Epetra_Map& rowmap, std::vector<int>& numentries,
    bool explicitdirichlet, bool savegraph, MatrixType matrixtype)
    : graph_(nullptr),
      dbcmaps_(nullptr),
      explicitdirichlet_(explicitdirichlet),
      savegraph_(savegraph),
      matrixtype_(matrixtype)
{
  if (!rowmap.UniqueGIDs()) FOUR_C_THROW("Row map is not unique");

  if ((int)(numentries.size()) != rowmap.NumMyElements())
    FOUR_C_THROW("estimate for non zero entries per row does not match the size of row map");

  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, numentries.data(), false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, rowmap, numentries.data(), false);
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
  if (access == Copy)
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

  if (sysmat_->Filled() and savegraph_)
  {
    graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SparseMatrix::SparseMatrix(const SparseMatrix& mat, DataAccess access)
    : Core::LinAlg::SparseMatrixBase(mat),
      explicitdirichlet_(mat.explicitdirichlet_),
      savegraph_(mat.savegraph_),
      matrixtype_(mat.matrixtype_)
{
  if (access == Copy)
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
  int length = diag.Map().NumMyElements();
  Epetra_Map map(-1, length, diag.Map().MyGlobalElements(), diag.Map().IndexBase(),
      Core::Communication::as_epetra_comm(diag.Comm()));
  if (!map.UniqueGIDs()) FOUR_C_THROW("Row map is not unique");

  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, map, map, 1, false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, map, map, 1, false);
  else
    FOUR_C_THROW("matrix type is not correct");

  // sysmat_->fill_complete();

  for (int i = 0; i < length; ++i)
  {
    int gid = map.GID(i);
    assemble(diag[i], gid, gid);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::SparseMatrix::destroy(bool throw_exception)
{
  // delete first the epetra matrix object
  if (throw_exception and sysmat_.use_count() > 1)
  {
    std::stringstream msg;
    msg << "Epetra_CrsMatrix cannot be finally deleted: The strong counter "
           "is still larger than 1. ( use_count() = "
        << sysmat_.use_count() << " )";
    FOUR_C_THROW(msg.str());
  }
  sysmat_ = nullptr;

  // delete now also the matrix' graph
  if (throw_exception and graph_.use_count() > 1)
  {
    std::stringstream msg;
    msg << "Graph cannot be finally deleted: The strong counter is "
           "still larger than 1. ( use_count() = "
        << graph_.use_count() << " )";
    FOUR_C_THROW(msg.str());
  }
  graph_ = nullptr;

  // delete now also the matrix' graph
  if (throw_exception and dbcmaps_.use_count() > 1)
  {
    std::stringstream msg;
    msg << "DBCMaps cannot be finally deleted: The strong counter is still "
           "larger than 1. ( use_count() = "
        << dbcmaps_.use_count() << " )";
    FOUR_C_THROW(msg.str());
  }
  dbcmaps_ = nullptr;

  return true;
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
      sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, mat.row_map(), 0, false);
    else if (matrixtype_ == FE_MATRIX)
      sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, mat.row_map(), 0, false);
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
  if (access == Copy)
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
      sysmat_->PutScalar(0.);
    else
      reset();
  }
  else
  {
    const Epetra_Map domainmap = sysmat_->DomainMap();
    const Epetra_Map rangemap = sysmat_->RangeMap();
    // Remove old matrix before creating a new one so we do not have old and
    // new matrix in memory at the same time!
    sysmat_ = nullptr;
    if (matrixtype_ == CRS_MATRIX)
      sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, *graph_->get_ptr_of_Epetra_CrsGraph());
    else if (matrixtype_ == FE_MATRIX)
      sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, *graph_->get_ptr_of_Epetra_CrsGraph());
    else
      FOUR_C_THROW("matrix type is not correct");

    sysmat_->FillComplete(domainmap, rangemap);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::reset()
{
  const Epetra_Map rowmap = sysmat_->RowMap();
  std::vector<int> numentries(rowmap.NumMyElements());

  auto graph = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());

  if (filled())
  {
    for (std::size_t i = 0; i < numentries.size(); ++i)
    {
      int* indices;
      int err = graph->ExtractMyRowView(i, numentries[i], indices);
      if (err != 0) FOUR_C_THROW("ExtractMyRowView failed");
    }
  }
  else
  {
    // use information about number of allocated entries not to fall back to matrix with zero size
    // otherwise assembly would be extremely expensive!
    for (std::size_t i = 0; i < numentries.size(); ++i)
    {
      numentries[i] = graph->NumAllocatedMyIndices(i);
    }
  }
  // Remove old matrix before creating a new one so we do not have old and
  // new matrix in memory at the same time!
  sysmat_ = nullptr;
  if (matrixtype_ == CRS_MATRIX)
    sysmat_ = std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, numentries.data(), false);
  else if (matrixtype_ == FE_MATRIX)
    sysmat_ = std::make_shared<Epetra_FECrsMatrix>(::Copy, rowmap, numentries.data(), false);
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
  const Epetra_Map& rowmap = sysmat_->RowMap();
  const Epetra_Map& colmap = sysmat_->ColMap();

  //-----------------------------------------------------------------------------------
  auto& A = (Core::LinAlg::SerialDenseMatrix&)Aele;
  if (sysmat_->Filled())  // assembly in local indices
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
      localcol[lcol] = colmap.LID(cgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (localcol[lcol] < 0) FOUR_C_THROW("Sparse matrix A does not have global column %d", cgid);
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
      if (dbcmaps_ != nullptr and dbcmaps_->Map(1)->MyGID(rgid)) continue;

      const int rlid = rowmap.LID(rgid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (rlid < 0) FOUR_C_THROW("Sparse matrix A does not have global row %d", rgid);
#endif
      int length;
      double* valview;
      int* indices;
#ifdef FOUR_C_ENABLE_ASSERTIONS
      int err =
#endif
          sysmat_->ExtractMyRowView(rlid, length, valview, indices);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (err) FOUR_C_THROW("Epetra_CrsMatrix::ExtractMyRowView returned error code %d", err);
#endif
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
            FOUR_C_THROW("Cannot find local column entry %d", localcol[dofcount]);
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
#ifdef FOUR_C_ENABLE_ASSERTIONS
            const int errone =
#endif
                sysmat_->SumIntoMyValues(rlid, 1, &A(lrow, dofcount), &localcol[dofcount]);
#ifdef FOUR_C_ENABLE_ASSERTIONS
            if (errone)
            {
              printf("Dimensions of A: %d x %d\n", A.numRows(), A.numCols());
              for (unsigned k = 0; k < lmstride.size(); ++k)
                printf("lmstride[%d] %d\n", k, lmstride[k]);
              FOUR_C_THROW(
                  "Epetra_CrsMatrix::SumIntoMyValues returned error code %d\nrlid %d localcol[%d] "
                  "%d dofcount %d length %d stride %d j %d node %d numnode %d",
                  errone, rlid, dofcount, localcol[dofcount], dofcount, length, stride, j, node,
                  numnode);
            }
#endif
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
  //-----------------------------------------------------------------------------------
  else  // assembly in global indices
  {
    // loop rows of local matrix
    for (int lrow = 0; lrow < lrowdim; ++lrow)
    {
      // check ownership of row
      if (lmrowowner[lrow] != myrank) continue;

      // check whether I have that global row
      const int rgid = lmrow[lrow];
      // #ifdef FOUR_C_ENABLE_ASSERTIONS
      if (!rowmap.MyGID(rgid)) FOUR_C_THROW("Proc %d does not have global row %d", myrank, rgid);
      // #endif

      // if we have a Dirichlet map check if this row is a Dirichlet row
      if (dbcmaps_ != nullptr and dbcmaps_->Map(1)->MyGID(rgid)) continue;

      for (int lcol = 0; lcol < lcoldim; ++lcol)
      {
        int cgid = lmcol[lcol];
        // Now that we do not rebuild the sparse mask in each step, we
        // are bound to assemble the whole thing. Zeros included.
        const int errone = sysmat_->SumIntoGlobalValues(rgid, 1, &A(lrow, lcol), &cgid);
        if (errone > 0)
        {
          const int errtwo = sysmat_->InsertGlobalValues(rgid, 1, &A(lrow, lcol), &cgid);
          if (errtwo < 0)
            FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned error code %d", errtwo);
        }
        else if (errone)
          FOUR_C_THROW("Epetra_CrsMatrix::SumIntoGlobalValues returned error code %d", errone);
      }  // for (int lcol=0; lcol<lcoldim; ++lcol)
    }  // for (int lrow=0; lrow<lrowdim; ++lrow)
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
  const Epetra_Map& rowmap = sysmat_->RowMap();
  const Epetra_Map& colmap = sysmat_->ColMap();

  if (sysmat_->Filled())
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
      localcol[lcol] = colmap.LID(cgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (localcol[lcol] < 0) FOUR_C_THROW("Sparse matrix A does not have global column %d", cgid);
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
      if (dbcmaps_ != nullptr and dbcmaps_->Map(1)->MyGID(rgid)) continue;

      const int rlid = rowmap.LID(rgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (rlid < 0) FOUR_C_THROW("Sparse matrix A does not have global row %d", rgid);
#endif

      for (int lcol = 0; lcol < lcoldim; ++lcol)
      {
        values[lcol] = Aele(lrow, lcol);
      }
      const int errone = sysmat_->SumIntoMyValues(rlid, lcoldim, values.data(), localcol.data());
      if (errone) FOUR_C_THROW("Epetra_CrsMatrix::SumIntoMyValues returned error code %d", errone);
    }  // for (int lrow=0; lrow<ldim; ++lrow)
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
      if (!rowmap.MyGID(rgid)) FOUR_C_THROW("Proc %d does not have global row %d", myrank, rgid);
#endif

      // if we have a Dirichlet map check if this row is a Dirichlet row
      if (dbcmaps_ != nullptr and dbcmaps_->Map(1)->MyGID(rgid)) continue;

      for (int lcol = 0; lcol < lcoldim; ++lcol)
      {
        double val = Aele(lrow, lcol);
        int cgid = lmcol[lcol];

        // Now that we do not rebuild the sparse mask in each step, we
        // are bound to assemble the whole thing. Zeros included.
        const int errone = sysmat_->SumIntoGlobalValues(rgid, 1, &val, &cgid);
        if (errone > 0)
        {
          const int errtwo = sysmat_->InsertGlobalValues(rgid, 1, &val, &cgid);
          if (errtwo < 0)
            FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned error code %d", errtwo);
        }
        else if (errone)
          FOUR_C_THROW("Epetra_CrsMatrix::SumIntoGlobalValues returned error code %d", errone);
      }  // for (int lcol=0; lcol<lcoldim; ++lcol)
    }  // for (int lrow=0; lrow<lrowdim; ++lrow)
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
  if (dbcmaps_ != nullptr and dbcmaps_->Map(1)->MyGID(rgid))
    FOUR_C_THROW("no assembling to Dirichlet row");

  // SumIntoGlobalValues works for filled matrices as well!
  int errone = sysmat_->SumIntoGlobalValues(rgid, 1, &val, &cgid);
  if (errone > 0)
  {
    int errtwo = sysmat_->InsertGlobalValues(rgid, 1, &val, &cgid);
    if (errtwo < 0)
      FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned error code %d", errtwo);
  }
  else if (errone)
    FOUR_C_THROW("Epetra_CrsMatrix::SumIntoGlobalValues returned error code %d", errone);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::set_value(double val, int rgid, int cgid)
{
  if (dbcmaps_ != nullptr and dbcmaps_->Map(1)->MyGID(rgid))
    FOUR_C_THROW("no assembling to Dirichlet row");

  int errone = sysmat_->ReplaceGlobalValues(rgid, 1, &val, &cgid);
  if (errone > 0)
  {
    int errtwo = sysmat_->InsertGlobalValues(rgid, 1, &val, &cgid);
    if (errtwo > 0)
      FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned error code %d", errtwo);
  }
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
      FOUR_C_THROW("Epetra_FECrsMatrix::InsertGlobalValues returned error code %d", errtwo);
  }
  else if (errone < 0)
    FOUR_C_THROW("Epetra_FECrsMatrix::SumIntoGlobalValues returned error code %d", errone);
}


/*----------------------------------------------------------------------*
 |  fill_complete a matrix  (public)                          mwgee 12/06|
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::complete(bool enforce_complete)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SparseMatrix::Complete");

  // for FE_Matrix we need to gather non-local entries, independent whether matrix is filled or not
  if (matrixtype_ == FE_MATRIX)
  {
    // false indicates here that fill_complete() is not called within GlobalAssemble()
    int err = (std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_))->GlobalAssemble(false);
    if (err) FOUR_C_THROW("Epetra_FECrsMatrix::GlobalAssemble() returned err=%d", err);
  }

  if (sysmat_->Filled() and not enforce_complete) return;

  int err = sysmat_->FillComplete(true);
  if (err) FOUR_C_THROW("Epetra_CrsMatrix::fill_complete(domain,range) returned err=%d", err);

  // keep mask for further use
  if (savegraph_ and graph_ == nullptr)
  {
    graph_ = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());
  }
}


/*----------------------------------------------------------------------*
 |  fill_complete a matrix  (public)                          mwgee 01/08|
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::complete(
    const Epetra_Map& domainmap, const Epetra_Map& rangemap, bool enforce_complete)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SparseMatrix::Complete(domain,range)");

  // for FE_Matrix we need to gather non-local entries, independent whether matrix is filled or not
  if (matrixtype_ == FE_MATRIX)
  {
    // false indicates here that fill_complete() is not called within GlobalAssemble()
    int err = (std::dynamic_pointer_cast<Epetra_FECrsMatrix>(sysmat_))
                  ->GlobalAssemble(domainmap, rangemap, false);
    if (err) FOUR_C_THROW("Epetra_FECrsMatrix::GlobalAssemble() returned err=%d", err);
  }

  if (sysmat_->Filled() and not enforce_complete) return;

  int err = 1;
  if (enforce_complete and sysmat_->Filled())
    err = sysmat_->ExpertStaticFillComplete(domainmap, rangemap);
  else
    err = sysmat_->FillComplete(domainmap, rangemap, true);

  if (err) FOUR_C_THROW("Epetra_CrsMatrix::fill_complete(domain,range) returned err=%d", err);

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

  auto graph = std::make_shared<Core::LinAlg::Graph>(sysmat_->Graph());

  std::vector<int> nonzeros(graph->NumMyRows());
  for (std::size_t i = 0; i < nonzeros.size(); ++i)
  {
    nonzeros[i] = graph->NumMyIndices(i);
  }

  const Epetra_Map& rowmap = sysmat_->RowMap();
  const Epetra_Map& colmap = sysmat_->ColMap();
  int elements = rowmap.NumMyElements();

  std::shared_ptr<Epetra_CrsMatrix> mat = nullptr;
  if (matrixtype_ == CRS_MATRIX)
    mat = std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, nonzeros.data(), false);
  else if (matrixtype_ == FE_MATRIX)
    mat = std::make_shared<Epetra_FECrsMatrix>(::Copy, rowmap, nonzeros.data(), false);
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
    int err = sysmat_->ExtractMyRowView(i, NumEntries, Values, Indices);
    if (err) FOUR_C_THROW("ExtractMyRowView err=%d", err);
    std::vector<int> idx(NumEntries);
    for (int c = 0; c < NumEntries; ++c)
    {
      idx[c] = colmap.GID(Indices[c]);
      FOUR_C_ASSERT(idx[c] != -1, "illegal gid");
    }
    int rowgid = rowmap.GID(i);
    err = mat->InsertGlobalValues(rowgid, NumEntries, Values, idx.data());
    if (err) FOUR_C_THROW("InsertGlobalValues err=%d", err);
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
      if (not graph_->Filled()) FOUR_C_THROW("got unfilled graph from filled matrix");
    }

    // allocate a new matrix and copy all rows that are not dirichlet
    const Epetra_Map& rowmap = sysmat_->RowMap();
    const int nummyrows = sysmat_->NumMyRows();
    const int maxnumentries = sysmat_->MaxNumEntries();

    std::shared_ptr<Epetra_CrsMatrix> Anew = nullptr;
    if (matrixtype_ == CRS_MATRIX)
      Anew = std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, maxnumentries, false);
    else if (matrixtype_ == FE_MATRIX)
      Anew = std::make_shared<Epetra_FECrsMatrix>(::Copy, rowmap, maxnumentries, false);
    else
      FOUR_C_THROW("matrix type is not correct");

    std::vector<int> indices(maxnumentries, 0);
    std::vector<double> values(maxnumentries, 0.0);
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = sysmat_->GRID(i);
      if (dbctoggle[i] != 1.0)
      {
        int numentries;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractGlobalRowCopy(
            row, maxnumentries, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractGlobalRowCopy returned err=%d", err);
#else
        sysmat_->ExtractGlobalRowCopy(
            row, maxnumentries, numentries, values.data(), indices.data());
#endif
        // this is also ok for FE matrices, because fill complete was called on sysmat and the
        // globalAssemble method was called already
#ifdef FOUR_C_ENABLE_ASSERTIONS
        err = Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err=%d", err);
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
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err=%d", err);
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
    const int nummyrows = sysmat_->NumMyRows();
    for (int i = 0; i < nummyrows; ++i)
    {
      if (dbctoggle[i] == 1.0)
      {
        int* indexOffset;
        int* indices;
        double* values;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractCrsDataPointers returned err=%d", err);
#else
        sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
#endif
        // zero row
        memset(&values[indexOffset[i]], 0, (indexOffset[i + 1] - indexOffset[i]) * sizeof(double));

        if (diagonalblock)
        {
          double one = 1.0;
#ifdef FOUR_C_ENABLE_ASSERTIONS
          int err = sysmat_->SumIntoMyValues(i, 1, &one, &i);
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::SumIntoMyValues returned err=%d", err);
#else
          sysmat_->SumIntoMyValues(i, 1, &one, &i);
#endif
        }
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Apply dirichlet conditions  (public)                     mwgee 02/07|
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::apply_dirichlet(const Epetra_Map& dbctoggle, bool diagonalblock)
{
  if (not filled()) FOUR_C_THROW("expect filled matrix to apply dirichlet conditions");

  if (dbcmaps_ != nullptr)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (not dbctoggle.SameAs(*dbcmaps_->Map(1)))
    {
      FOUR_C_THROW("Dirichlet maps mismatch");
    }
#endif
    if (diagonalblock)
    {
      double v = 1.0;
      int numdbc = dbctoggle.NumMyElements();
      int* dbc = dbctoggle.MyGlobalElements();
      for (int i = 0; i < numdbc; ++i)
      {
        int row = dbc[i];
        int err = sysmat_->ReplaceGlobalValues(row, 1, &v, &row);
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ReplaceGlobalValues returned err=%d", err);
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
      if (not graph_->Filled()) FOUR_C_THROW("got unfilled graph from filled matrix");
    }

    // allocate a new matrix and copy all rows that are not dirichlet
    const Epetra_Map& rowmap = sysmat_->RowMap();
    const int nummyrows = sysmat_->NumMyRows();
    const int maxnumentries = sysmat_->MaxNumEntries();

    // std::shared_ptr<Epetra_CrsMatrix> Anew = Teuchos::rcp(new
    // Epetra_CrsMatrix(Copy,rowmap,maxnumentries,false));

    std::shared_ptr<Epetra_CrsMatrix> Anew = nullptr;
    if (matrixtype_ == CRS_MATRIX)
      Anew = std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, maxnumentries, false);
    else if (matrixtype_ == FE_MATRIX)
      Anew = std::make_shared<Epetra_FECrsMatrix>(::Copy, rowmap, maxnumentries, false);
    else
      FOUR_C_THROW("matrix type is not correct");

    std::vector<int> indices(maxnumentries, 0);
    std::vector<double> values(maxnumentries, 0.0);
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = sysmat_->GRID(i);
      if (not dbctoggle.MyGID(row))
      {
        int numentries;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractGlobalRowCopy(
            row, maxnumentries, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractGlobalRowCopy returned err=%d", err);
#else
        sysmat_->ExtractGlobalRowCopy(
            row, maxnumentries, numentries, values.data(), indices.data());
#endif
        // this is also ok for FE matrices, because fill complete was called on sysmat and the
        // globalAssemble method was called already
#ifdef FOUR_C_ENABLE_ASSERTIONS
        err = Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err=%d", err);
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
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err=%d", err);
#else
          Anew->InsertGlobalValues(row, 1, &v, &row);
#endif
        }
      }
    }
    Epetra_Map rangemap = sysmat_->RangeMap();
    Epetra_Map domainmap = sysmat_->DomainMap();
    sysmat_ = Anew;
    complete(domainmap, rangemap);
  }
  else
  {
    const int nummyrows = sysmat_->NumMyRows();
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = sysmat_->GRID(i);
      if (dbctoggle.MyGID(row))
      {
        int* indexOffset;
        int* indices;
        double* values;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractCrsDataPointers returned err=%d", err);
#else
        sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
#endif
        // zero row
        memset(&values[indexOffset[i]], 0, (indexOffset[i + 1] - indexOffset[i]) * sizeof(double));

        if (diagonalblock)
        {
          double one = 1.0;
#ifdef FOUR_C_ENABLE_ASSERTIONS
          err = sysmat_->SumIntoMyValues(i, 1, &one, &i);
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::SumIntoMyValues returned err=%d", err);
#else
          sysmat_->SumIntoMyValues(i, 1, &one, &i);
#endif
        }
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Apply dirichlet conditions  (public)                     mwgee 02/07|
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::apply_dirichlet_with_trafo(const Core::LinAlg::SparseMatrix& trafo,
    const Epetra_Map& dbctoggle, bool diagonalblock, bool complete)
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
      if (not graph_->Filled()) FOUR_C_THROW("got unfilled graph from filled matrix");
    }

    // allocate a new matrix and copy all rows that are not dirichlet
    const Epetra_Map& rowmap = sysmat_->RowMap();
    const Epetra_Map& colmap = sysmat_->ColMap();
    const int nummyrows = sysmat_->NumMyRows();
    const int maxnumentries = sysmat_->MaxNumEntries();

    // prepare working arrays for extracting rows in trafo matrix
    const int trafomaxnumentries = trafo.max_num_entries();
    int trafonumentries = 0;
    std::vector<int> trafoindices(trafomaxnumentries, 0);
    std::vector<double> trafovalues(trafomaxnumentries, 0.0);

    // initialise matrix Anew with general size (rowmap x colmap)
    // in case of coupled problem (e.g. TSI) transform the rectangular off-diagonal block k_Td
    std::shared_ptr<Epetra_CrsMatrix> Anew =
        std::make_shared<Epetra_CrsMatrix>(::Copy, rowmap, colmap, maxnumentries, false);
    std::vector<int> indices(maxnumentries, 0);
    std::vector<double> values(maxnumentries, 0.0);
    for (int i = 0; i < nummyrows; ++i)
    {
      int row = sysmat_->GRID(i);
      if (not dbctoggle.MyGID(row))  // dof is not a Dirichlet dof
      {
        int numentries;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractGlobalRowCopy(
            row, maxnumentries, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractGlobalRowCopy returned err=%d", err);
#else
        sysmat_->ExtractGlobalRowCopy(
            row, maxnumentries, numentries, values.data(), indices.data());
#endif

#ifdef FOUR_C_ENABLE_ASSERTIONS
        err = Anew->InsertGlobalValues(row, numentries, values.data(), indices.data());
        if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err=%d", err);
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
#ifdef FOUR_C_ENABLE_ASSERTIONS
          int err = trafo.epetra_matrix()->ExtractGlobalRowCopy(
              row, trafomaxnumentries, trafonumentries, trafovalues.data(), trafoindices.data());
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractGlobalRowCopy returned err=%d", err);
#else
          trafo.epetra_matrix()->ExtractGlobalRowCopy(
              row, trafomaxnumentries, trafonumentries, trafovalues.data(), trafoindices.data());
#endif
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
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err=%d", err);
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
    const int nummyrows = sysmat_->NumMyRows();

    // prepare working arrays for extracting rows in trafo matrix
    const int trafomaxnumentries = trafo.max_num_entries();
    int trafonumentries = 0;
    std::vector<int> trafoindices(trafomaxnumentries, 0);
    std::vector<double> trafovalues(trafomaxnumentries, 0.0);

    for (int i = 0; i < nummyrows; ++i)
    {
      int row = sysmat_->GRID(i);
      if (dbctoggle.MyGID(row))
      {
        int* indexOffset;
        int* indices;
        double* values;
#ifdef FOUR_C_ENABLE_ASSERTIONS
        int err = sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
        if (err) FOUR_C_THROW("Epetra_CrsMatrix::ExtractCrsDataPointers returned err=%d", err);
#else
        sysmat_->ExtractCrsDataPointers(indexOffset, indices, values);
#endif
        // zero row
        memset(&values[indexOffset[i]], 0, (indexOffset[i + 1] - indexOffset[i]) * sizeof(double));

        if (diagonalblock)
        {
#ifdef FOUR_C_ENABLE_ASSERTIONS
          err = trafo.epetra_matrix()->ExtractMyRowCopy(
              i, trafomaxnumentries, trafonumentries, trafovalues.data(), trafoindices.data());
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::ExtractGlobalRowCopy returned err=%d", err);
#else
          trafo.epetra_matrix()->ExtractMyRowCopy(
              i, trafomaxnumentries, trafonumentries, trafovalues.data(), trafoindices.data());
#endif

#ifdef FOUR_C_ENABLE_ASSERTIONS
          err =
              sysmat_->SumIntoMyValues(i, trafonumentries, trafovalues.data(), trafoindices.data());
          if (err < 0) FOUR_C_THROW("Epetra_CrsMatrix::SumIntoMyValues returned err=%d", err);
#else
          sysmat_->SumIntoMyValues(i, trafonumentries, trafovalues.data(), trafoindices.data());
#endif
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

  const Epetra_Map& rowmap = sysmat_->RowMap();
  const Epetra_Map& colmap = sysmat_->ColMap();
  const int nummyrows = sysmat_->NumMyRows();

  const Core::LinAlg::Vector<double>& dbct = dbctoggle;

  std::vector<int> idx(max_num_entries());

  for (int i = 0; i < nummyrows; ++i)
  {
    if (dbct[i] == 1.0)
    {
      int NumEntries;
      double* Values;
      int* Indices;

      int err = sysmat_->ExtractMyRowView(i, NumEntries, Values, Indices);
      if (err) FOUR_C_THROW("ExtractMyRowView: err=%d", err);
      for (int j = 0; j < NumEntries; ++j) idx[j] = colmap.GID(Indices[j]);

      err = dl->sysmat_->InsertGlobalValues(rowmap.GID(i), NumEntries, Values, idx.data());
      if (err) FOUR_C_THROW("InsertGlobalValues: err=%d", err);
    }
  }

  dl->complete(sysmat_->DomainMap(), range_map());
  return dl;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::SparseMatrix::extract_dirichlet_rows(
    const Epetra_Map& dbctoggle)
{
  if (not filled()) FOUR_C_THROW("expect filled matrix to extract dirichlet lines");
  if (not dbctoggle.UniqueGIDs()) FOUR_C_THROW("unique map required");

  std::shared_ptr<SparseMatrix> dl = std::make_shared<SparseMatrix>(
      row_map(), max_num_entries(), explicit_dirichlet(), save_graph());

  const Epetra_Map& rowmap = sysmat_->RowMap();
  const Epetra_Map& colmap = sysmat_->ColMap();
  // const int nummyrows      = sysmat_->NumMyRows();

  std::vector<int> idx(max_num_entries());

  const int mylength = dbctoggle.NumMyElements();
  const int* mygids = dbctoggle.MyGlobalElements();
  for (int i = 0; i < mylength; ++i)
  {
    int gid = mygids[i];
    int lid = rowmap.LID(gid);

    if (lid < 0) FOUR_C_THROW("illegal Dirichlet map");

    int NumEntries;
    double* Values;
    int* Indices;

    int err = sysmat_->ExtractMyRowView(lid, NumEntries, Values, Indices);
    if (err) FOUR_C_THROW("ExtractMyRowView: err=%d", err);
    for (int j = 0; j < NumEntries; ++j) idx[j] = colmap.GID(Indices[j]);

    err = dl->sysmat_->InsertGlobalValues(gid, NumEntries, Values, idx.data());
    if (err) FOUR_C_THROW("InsertGlobalValues: err=%d", err);
  }

  dl->complete(sysmat_->DomainMap(), range_map());
  return dl;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const char* Core::LinAlg::SparseMatrix::Label() const { return "Core::LinAlg::SparseMatrix"; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::SparseMatrix::add(const Core::LinAlg::SparseMatrixBase& A, const bool transposeA,
    const double scalarA, const double scalarB)
{
  Core::LinAlg::add(*A.epetra_matrix(), transposeA, scalarA, *this, scalarB);
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
