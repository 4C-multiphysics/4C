// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_assemble.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::assemble(Core::LinAlg::SparseMatrix& A,
    const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
    const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
{
  const int lrowdim = (int)lmrow.size();
  const int lcoldim = (int)lmcol.size();
  // allow Aele to provide entries past the end of lmrow and lmcol that are
  // not used here, therefore check only for ">" rather than "!="
  if (lrowdim != (int)lmrowowner.size() || lrowdim > Aele.numRows() || lcoldim > Aele.numCols())
    FOUR_C_THROW("Mismatch in dimensions");

  const int myrank = Core::Communication::my_mpi_rank(A.get_comm());
  const Core::LinAlg::Map& rowmap = Map(A.row_map());

  // this 'Assemble' is not implemented for a Filled() matrix A
  if (A.filled())
    FOUR_C_THROW("Sparse matrix A is already Filled()");
  else
  {
    // loop rows of local matrix
    for (int lrow = 0; lrow < lrowdim; ++lrow)
    {
      // check ownership of row
      if (lmrowowner[lrow] != myrank) continue;

      // check whether I have that global row
      int rgid = lmrow[lrow];
      if (!(rowmap.my_gid(rgid))) FOUR_C_THROW("Sparse matrix A does not have global row {}", rgid);

      for (int lcol = 0; lcol < lcoldim; ++lcol)
      {
        double val = Aele(lrow, lcol);
        int cgid = lmcol[lcol];

        // Now that we do not rebuild the sparse mask in each step, we
        // are bound to assemble the whole thing. Zeros included.
        A.assemble(val, rgid, cgid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  assemble a vector                                        mwgee 12/06|
 *----------------------------------------------------------------------*/
void Core::LinAlg::assemble(Core::LinAlg::Vector<double>& V,
    const Core::LinAlg::SerialDenseVector& Vele, const std::vector<int>& lm,
    const std::vector<int>& lmowner)
{
  const int ldim = (int)lm.size();
  // allow Vele to provide entries past the end of lm that are not used here,
  // therefore check only for ">" rather than "!="
  if (ldim != (int)lmowner.size() || ldim > Vele.length()) FOUR_C_THROW("Mismatch in dimensions");

  const int myrank = Core::Communication::my_mpi_rank(V.get_comm());

  for (int lrow = 0; lrow < ldim; ++lrow)
  {
    if (lmowner[lrow] != myrank) continue;
    int rgid = lm[lrow];
    if (!V.get_map().my_gid(rgid))
      FOUR_C_THROW("Sparse vector V does not have global row {}", rgid);
    int rlid = V.get_map().lid(rgid);
    V.get_values()[rlid] += Vele[lrow];
  }  // for (int lrow=0; lrow<ldim; ++lrow)
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::LinAlg::assemble_my_vector(double scalar_target, Core::LinAlg::Vector<double>& target,
    double scalar_source, const Core::LinAlg::Vector<double>& source)
{
  for (int slid = 0; slid < source.get_map().num_my_elements(); ++slid)
  {
    const int sgid = source.get_map().gid(slid);
    const int tlid = target.get_map().lid(sgid);
    if (tlid == -1)
      FOUR_C_THROW(
          "The target vector has no global row {}"
          " on processor {}!",
          sgid, Core::Communication::my_mpi_rank(target.get_comm()));

    // update the vector row
    target.get_values()[tlid] = scalar_target * target.local_values_as_span()[tlid] +
                                scalar_source * source.local_values_as_span()[slid];
  }
}

/*----------------------------------------------------------------------*
 |  assemble a vector into MultiVector (public)              mwgee 01/08|
 *----------------------------------------------------------------------*/
void Core::LinAlg::assemble(Core::LinAlg::MultiVector<double>& V, const int n,
    const Core::LinAlg::SerialDenseVector& Vele, const std::vector<int>& lm,
    const std::vector<int>& lmowner)
{
  Core::LinAlg::assemble(V.get_vector(n), Vele, lm, lmowner);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::apply_dirichlet_to_system(Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& b, const Core::LinAlg::Vector<double>& dbcval,
    const Core::LinAlg::Vector<double>& dbctoggle)
{
  // set the prescribed value in x and b
  const int mylength = dbcval.local_length();
  for (int i = 0; i < mylength; ++i)
  {
    if (dbctoggle.local_values_as_span()[i] == 1.0)
    {
      x.get_values()[i] = dbcval.local_values_as_span()[i];
      b.get_values()[i] = dbcval.local_values_as_span()[i];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::apply_dirichlet_to_system(Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& b, const Core::LinAlg::Vector<double>& dbcval,
    const Core::LinAlg::Map& dbcmap)
{
  if (not dbcmap.unique_gids()) FOUR_C_THROW("unique map required");

  // We use two maps since we want to allow dbcv and X to be independent of
  // each other. So we are slow and flexible...
  const Core::LinAlg::Map& xmap = x.get_map();
  const Core::LinAlg::Map& dbcvmap = dbcval.get_map();

  const int mylength = dbcmap.num_my_elements();
  const int* mygids = dbcmap.my_global_elements();
  for (int i = 0; i < mylength; ++i)
  {
    int gid = mygids[i];

    int dbcvlid = dbcvmap.lid(gid);
    if (dbcvlid < 0) FOUR_C_THROW("illegal Dirichlet map");

    int xlid = xmap.lid(gid);
    if (xlid < 0) FOUR_C_THROW("illegal Dirichlet map");

    x.get_values()[xlid] = dbcval.local_values_as_span()[dbcvlid];
    b.get_values()[xlid] = dbcval.local_values_as_span()[dbcvlid];
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::apply_dirichlet_to_system(Core::LinAlg::Vector<double>& b,
    const Core::LinAlg::Vector<double>& dbcval, const Core::LinAlg::Map& dbcmap)
{
  if (not dbcmap.unique_gids()) FOUR_C_THROW("unique map required");

  const int mylength = dbcmap.num_my_elements();
  const int* mygids = dbcmap.my_global_elements();
  for (int i = 0; i < mylength; ++i)
  {
    const int gid = mygids[i];

    const int dbcvlid = dbcval.get_map().lid(gid);

    const int blid = b.get_map().lid(gid);
    // Note:
    // if gid is not found in vector b, just continue
    // b might only be a subset of a larger field vector
    if (blid >= 0)
    {
      if (dbcvlid < 0)
        FOUR_C_THROW("illegal Dirichlet map");
      else
        b.get_values()[blid] = dbcval.local_values_as_span()[dbcvlid];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::apply_dirichlet_to_system(Core::LinAlg::SparseOperator& A,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b,
    const Core::LinAlg::Vector<double>& dbcval, const Core::LinAlg::Vector<double>& dbctoggle)
{
  A.apply_dirichlet(dbctoggle);
  apply_dirichlet_to_system(x, b, dbcval, dbctoggle);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::apply_dirichlet_to_system(Core::LinAlg::SparseOperator& A,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b,
    const Core::LinAlg::Vector<double>& dbcval, const Core::LinAlg::Map& dbcmap)
{
  A.apply_dirichlet(dbcmap);
  apply_dirichlet_to_system(x, b, dbcval, dbcmap);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::apply_dirichlet_to_system(Core::LinAlg::SparseMatrix& A,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b,
    const Core::LinAlg::SparseMatrix& trafo, const Core::LinAlg::Vector<double>& dbcval,
    const Core::LinAlg::Map& dbcmap)
{
  A.apply_dirichlet_with_trafo(trafo, dbcmap);
  apply_dirichlet_to_system(x, b, dbcval, dbcmap);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MapExtractor> Core::LinAlg::convert_dirichlet_toggle_vector_to_maps(
    const Core::LinAlg::Vector<double>& dbctoggle)
{
  const Core::LinAlg::Map& fullblockmap = dbctoggle.get_map();
  // this copy is needed because the constructor of Core::LinAlg::MapExtractor
  // accepts only Core::LinAlg::Map and not Core::LinAlg::Map
  const Core::LinAlg::Map fullmap = Core::LinAlg::Map(fullblockmap.num_global_elements(),
      std::span<const int>(fullblockmap.my_global_elements(), fullblockmap.num_my_elements()),
      fullblockmap.index_base(), fullblockmap.get_comm());
  const int mylength = dbctoggle.local_length();
  const int* fullgids = fullmap.my_global_elements();
  // build sets containing the DBC or free global IDs, respectively
  std::vector<int> dbcgids;
  std::vector<int> freegids;
  for (int i = 0; i < mylength; ++i)
  {
    const int gid = fullgids[i];
    const int compo = (int)round((dbctoggle).local_values_as_span()[i]);
    if (compo == 0)
      freegids.push_back(gid);
    else if (compo == 1)
      dbcgids.push_back(gid);
    else
      FOUR_C_THROW("Unexpected component {}. It is neither 1.0 nor 0.0.",
          (dbctoggle).local_values_as_span()[i]);
  }
  // build map of Dirichlet DOFs
  std::shared_ptr<Core::LinAlg::Map> dbcmap = nullptr;
  {
    int nummyelements = 0;
    int* myglobalelements = nullptr;
    if (dbcgids.size() > 0)
    {
      nummyelements = dbcgids.size();
      myglobalelements = dbcgids.data();
    }
    dbcmap = std::make_shared<Core::LinAlg::Map>(-1,
        std::span<const int>(myglobalelements, nummyelements), fullmap.index_base(),
        fullmap.get_comm());
  }
  // build map of free DOFs
  std::shared_ptr<Core::LinAlg::Map> freemap = nullptr;
  {
    int nummyelements = 0;
    int* myglobalelements = nullptr;
    if (freegids.size() > 0)
    {
      nummyelements = freegids.size();
      myglobalelements = freegids.data();
    }
    freemap = std::make_shared<Core::LinAlg::Map>(-1,
        std::span<const int>(myglobalelements, nummyelements), fullmap.index_base(),
        fullmap.get_comm());
  }

  // build and return the map extractor of Dirichlet-conditioned and free DOFs
  return std::make_shared<Core::LinAlg::MapExtractor>(fullmap, dbcmap, freemap);
}

bool Core::LinAlg::is_dirichlet_boundary_condition_already_applied(
    const Core::LinAlg::SparseMatrix& A, const Core::LinAlg::Map& dbcmap, bool diagonalblock,
    const std::shared_ptr<const Core::LinAlg::SparseMatrix>& trafo)
{
  if (not A.filled()) FOUR_C_THROW("The matrix must be filled!");

  const int numdbcrows = dbcmap.num_my_elements();
  const int* dbcrows = dbcmap.my_global_elements();

  std::vector<int> gIndices(A.max_num_entries(), 0);
  std::vector<int> gtIndices((trafo ? trafo->max_num_entries() : 0), 0);

  bool isdbc = true;

  for (int i = 0; i < numdbcrows; ++i)
  {
    const int row = dbcrows[i];
    const int sys_rlid = A.row_map().lid(row);

    // this can happen for blocks of a BlockSparseMatrix
    if (sys_rlid == -1) continue;

    int NumEntries = 0;
    double* Values = nullptr;
    int* Indices = nullptr;
    A.extract_my_row_view(sys_rlid, NumEntries, Values, Indices);

    std::fill(gIndices.begin(), gIndices.end(), 0.0);
    for (int c = 0; c < NumEntries; ++c) gIndices[c] = A.col_map().gid(Indices[c]);

    // handle a diagonal block
    if (diagonalblock)
    {
      if (NumEntries == 0) FOUR_C_THROW("Row {} is empty and part of a diagonal block!", row);

      if (trafo)
      {
        if (not trafo->filled()) FOUR_C_THROW("The trafo matrix must be filled!");

        int tNumEntries = 0;
        double* tValues = nullptr;
        int* tIndices = nullptr;

        const int trafo_rlid = trafo->row_map().lid(row);
        trafo->epetra_matrix().ExtractMyRowView(trafo_rlid, tNumEntries, tValues, tIndices);

        // get the global indices corresponding to the extracted local indices
        std::fill(gtIndices.begin(), gtIndices.end(), 0.0);
        for (int c = 0; c < tNumEntries; ++c) gtIndices[c] = trafo->col_map().gid(tIndices[c]);

        for (int j = 0; j < tNumEntries; ++j)
        {
          int k = -1;
          while (++k < NumEntries)
            if (Indices[k] == tIndices[j]) break;

          if (k == NumEntries)
            FOUR_C_THROW("Couldn't find column index {} in row {}.", tIndices[j], row);

          if (std::abs(Values[k] - tValues[j]) > std::numeric_limits<double>::epsilon())
          {
            isdbc = false;
            break;
          }
        }
      }
      // handle standard diagonal blocks
      // --> 1.0 on the diagonal
      // --> 0.0 on all off-diagonals
      else
      {
        for (int j = 0; j < NumEntries; ++j)
        {
          if (gIndices[j] != row and Values[j] > std::numeric_limits<double>::epsilon())
          {
            isdbc = false;
            break;
          }
          else if (gIndices[j] == row)
            if (std::abs(1.0 - Values[j]) > std::numeric_limits<double>::epsilon())
            {
              isdbc = false;
              break;
            }
        }
      }
    }
    // we expect only zeros on the off-diagonal blocks
    else
    {
      for (int j = 0; j < NumEntries; ++j)
      {
        if (Values[j] > std::numeric_limits<double>::epsilon())
        {
          isdbc = false;
          break;
        }
      }
    }

    // stop as soon as the initial status changed once
    if (not isdbc) break;
  }

  int lisdbc = static_cast<int>(isdbc);
  int gisdbc = 0;
  gisdbc = Core::Communication::min_all(lisdbc, A.get_comm());

  return (gisdbc == 1);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::is_dirichlet_boundary_condition_already_applied(
    const Core::LinAlg::SparseOperator& A, const Core::LinAlg::Map& dbcmap, bool diagonalblock,
    const std::shared_ptr<const Core::LinAlg::SparseMatrix>& trafo)
{
  // Check if provided Operator corresponds to single sparse matrix
  auto const* sparse_matrix = dynamic_cast<const Core::LinAlg::SparseMatrix*>(&A);
  if (sparse_matrix != nullptr)
  {
    return Core::LinAlg::is_dirichlet_boundary_condition_already_applied(
        *sparse_matrix, dbcmap, diagonalblock, trafo);
  }
  // Or Sparse Block Matrix
  auto const* block = dynamic_cast<const Core::LinAlg::BlockSparseMatrixBase*>(&A);
  if (block != nullptr)
  {
    for (int rblock = 0; rblock < block->rows(); ++rblock)
    {
      for (int cblock = 0; cblock < block->cols(); ++cblock)
      {
        const Core::LinAlg::SparseMatrix& submat = block->matrix(rblock, cblock);

        if (not Core::LinAlg::is_dirichlet_boundary_condition_already_applied(
                submat, dbcmap, diagonalblock && (rblock == cblock), trafo))
        {
          return false;
        }
      }
    }
    return true;
  }
  else
  {
    FOUR_C_THROW(
        "Unsupported SparseOperator type in "
        "is_dirichlet_boundary_condition_already_applied().");
  }



  return false;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::has_dirichlet_boundary_condition(const Core::LinAlg::SparseMatrix& A)
{
  std::vector<int> dirichlet_rows;

  for (int row = 0; row < A.num_my_rows(); row++)
  {
    int num_entries;
    int* indices;
    double* values;

    A.extract_my_row_view(row, num_entries, values, indices);
    int number_of_nonzeros = 0;
    int nonzero_index = 0;

    for (int j = 0; j < num_entries; j++)
    {
      if (std::abs(values[j]) > std::numeric_limits<double>::epsilon())
      {
        nonzero_index = j;
        number_of_nonzeros++;
      }
    }

    if (number_of_nonzeros == 1)
    {
      const int col = indices[nonzero_index];
      const double val = values[nonzero_index];

      if (col == row && std::abs(val - 1.0) < std::numeric_limits<double>::epsilon())
      {
        dirichlet_rows.push_back(row);
      }
    }
  }

  const int number_of_dirichlet_rows =
      Core::Communication::sum_all(dirichlet_rows.size(), A.get_comm());

  return number_of_dirichlet_rows > 0;
}

FOUR_C_NAMESPACE_CLOSE
