// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_blocksparsematrix.hpp"

#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::BlockSparseMatrixBase::BlockSparseMatrixBase(const MultiMapExtractor& domainmaps,
    const MultiMapExtractor& rangemaps, int npr, bool explicitdirichlet, bool savegraph)
    : domainmaps_(domainmaps), rangemaps_(rangemaps), usetranspose_(false)
{
  blocks_.reserve(rows() * cols());

  // add sparse matrices in row,column order
  for (int r = 0; r < rows(); ++r)
  {
    for (int c = 0; c < cols(); ++c)
    {
      blocks_.emplace_back(range_map(r), npr, explicitdirichlet, savegraph);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::BlockSparseMatrixBase::merge(
    bool explicitdirichlet) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::BlockSparseMatrixBase::Merge");

  const SparseMatrix& m00 = matrix(0, 0);

  std::shared_ptr<SparseMatrix> sparse =
      std::make_shared<SparseMatrix>(*fullrowmap_, m00.max_num_entries(), explicitdirichlet);
  for (const auto& block : blocks_)
  {
    sparse->add(block, false, 1.0, 1.0);
  }
  if (filled())
  {
    sparse->complete(full_domain_map(), full_range_map());
  }
  return sparse;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::assign(
    int r, int c, DataAccess access, const SparseMatrix& mat)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not matrix(r, c).row_map().same_as(mat.row_map()))
    FOUR_C_THROW("cannot assign nonmatching matrices");
#endif
  matrix(r, c).assign(access, mat);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::zero()
{
  for (auto& block : blocks_) block.zero();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::reset()
{
  for (int i = 0; i < rows(); ++i)
  {
    for (int j = 0; j < cols(); ++j)
    {
      matrix(i, j).reset();
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::complete(OptionsMatrixComplete options_matrix_complete)
{
  for (int r = 0; r < rows(); ++r)
  {
    for (int c = 0; c < cols(); ++c)
    {
      matrix(r, c).complete(domain_map(c), range_map(r), options_matrix_complete);
    }
  }

  fullrowmap_ = std::make_shared<Core::LinAlg::Map>(*(rangemaps_.full_map()));

  if (fullcolmap_ == nullptr)
  {
    // build full col map
    std::vector<int> colmapentries;
    for (int c = 0; c < cols(); ++c)
    {
      for (int r = 0; r < rows(); ++r)
      {
        const Core::LinAlg::Map& colmap = matrix(r, c).col_map();
        colmapentries.insert(colmapentries.end(), colmap.my_global_elements(),
            colmap.my_global_elements() + colmap.num_my_elements());
      }
    }
    std::sort(colmapentries.begin(), colmapentries.end());
    colmapentries.erase(
        std::unique(colmapentries.begin(), colmapentries.end()), colmapentries.end());
    fullcolmap_ = std::make_shared<Core::LinAlg::Map>(-1, colmapentries.size(),
        colmapentries.data(), 0, Core::Communication::unpack_epetra_comm(Comm()));
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::complete(const Core::LinAlg::Map& domainmap,
    const Core::LinAlg::Map& rangemap, OptionsMatrixComplete options_matrix_complete)
{
  FOUR_C_THROW("Complete with arguments not supported for block matrices");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::BlockSparseMatrixBase::filled() const
{
  for (const auto& block : blocks_)
    if (not block.filled()) return false;
  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::un_complete()
{
  for (auto& block : blocks_) block.un_complete();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::apply_dirichlet(
    const Core::LinAlg::Vector<double>& dbctoggle, bool diagonalblock)
{
  for (int rblock = 0; rblock < rows(); ++rblock)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> rowtoggle =
        rangemaps_.extract_vector(dbctoggle, rblock);
    for (int cblock = 0; cblock < cols(); ++cblock)
    {
      Core::LinAlg::SparseMatrix& bmat = matrix(rblock, cblock);
      bmat.apply_dirichlet(*rowtoggle, diagonalblock and rblock == cblock);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::apply_dirichlet(
    const Core::LinAlg::Map& dbcmap, bool diagonalblock)
{
  for (int rblock = 0; rblock < rows(); ++rblock)
  {
    for (int cblock = 0; cblock < cols(); ++cblock)
    {
      Core::LinAlg::SparseMatrix& bmat = matrix(rblock, cblock);
      bmat.apply_dirichlet(dbcmap, diagonalblock and rblock == cblock);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::BlockSparseMatrixBase::is_dbc_applied(const Core::LinAlg::Map& dbcmap,
    bool diagonalblock, const Core::LinAlg::SparseMatrix* trafo) const
{
  for (int rblock = 0; rblock < rows(); ++rblock)
  {
    for (int cblock = 0; cblock < cols(); ++cblock)
    {
      if (not matrix(rblock, cblock)
              .is_dbc_applied(dbcmap, diagonalblock and (rblock == cblock), trafo))
        return false;
    }
  }
  return true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::BlockSparseMatrixBase::SetUseTranspose(bool UseTranspose)
{
  for (auto& block : blocks_) block.SetUseTranspose(UseTranspose);
  usetranspose_ = UseTranspose;
  return 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::BlockSparseMatrixBase::Apply(
    const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  Y.PutScalar(0.0);

  if (not UseTranspose())
  {
    for (int rblock = 0; rblock < rows(); ++rblock)
    {
      std::shared_ptr<Core::LinAlg::MultiVector<double>> rowresult =
          rangemaps_.vector(rblock, Y.NumVectors());
      std::shared_ptr<Core::LinAlg::MultiVector<double>> rowy =
          rangemaps_.vector(rblock, Y.NumVectors());
      for (int cblock = 0; cblock < cols(); ++cblock)
      {
        std::shared_ptr<Core::LinAlg::MultiVector<double>> colx =
            domainmaps_.extract_vector(Core::LinAlg::MultiVector<double>(X), cblock);
        const Core::LinAlg::SparseMatrix& bmat = matrix(rblock, cblock);
        int err = bmat.Apply(colx->get_epetra_multi_vector(), rowy->get_epetra_multi_vector());
        if (err != 0)
          FOUR_C_THROW(
              "failed to apply vector to matrix block ({},{}): err={}", rblock, cblock, err);
        rowresult->update(1.0, *rowy, 1.0);
      }
      View Y_view(Y);
      rangemaps_.insert_vector(*rowresult, rblock, Y_view);
    }
  }
  else
  {
    for (int rblock = 0; rblock < cols(); ++rblock)
    {
      std::shared_ptr<Core::LinAlg::MultiVector<double>> rowresult =
          rangemaps_.vector(rblock, Y.NumVectors());
      std::shared_ptr<Core::LinAlg::MultiVector<double>> rowy =
          rangemaps_.vector(rblock, Y.NumVectors());
      for (int cblock = 0; cblock < rows(); ++cblock)
      {
        std::shared_ptr<Core::LinAlg::MultiVector<double>> colx =
            domainmaps_.extract_vector(Core::LinAlg::MultiVector<double>(X), cblock);
        const Core::LinAlg::SparseMatrix& bmat = matrix(cblock, rblock);
        int err = bmat.Apply(colx->get_epetra_multi_vector(), rowy->get_epetra_multi_vector());
        if (err != 0) FOUR_C_THROW("failed to apply vector to matrix: err={}", err);
        rowresult->update(1.0, *rowy, 1.0);
      }
      View Y_view(Y);
      rangemaps_.insert_vector(*rowresult, rblock, Y_view);
    }
  }

  return 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::BlockSparseMatrixBase::ApplyInverse(
    const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  FOUR_C_THROW("Core::LinAlg::BlockSparseMatrixBase::ApplyInverse not implemented");
  return -1;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::add(const Core::LinAlg::SparseOperator& A,
    const bool transposeA, const double scalarA, const double scalarB)
{
  A.add_other(*this, transposeA, scalarA, scalarB);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::add_other(Core::LinAlg::BlockSparseMatrixBase& A,
    const bool transposeA, const double scalarA, const double scalarB) const
{
  A.add(*this, transposeA, scalarA, scalarB);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::add_other(Core::LinAlg::SparseMatrix& A,
    const bool transposeA, const double scalarA, const double scalarB) const
{
  FOUR_C_THROW("BlockSparseMatrix and SparseMatrix cannot be added");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::add(const Core::LinAlg::BlockSparseMatrixBase& A,
    const bool transposeA, const double scalarA, const double scalarB)
{
  for (int i = 0; i < rows(); i++)
  {
    for (int j = 0; j < cols(); j++)
    {
      if (transposeA)
        matrix(i, j).add(A.matrix(j, i), transposeA, scalarA, scalarB);
      else
        matrix(i, j).add(A.matrix(i, j), transposeA, scalarA, scalarB);
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::BlockSparseMatrixBase::scale(double ScalarConstant)
{
  for (int i = 0; i < rows(); i++)
  {
    for (int j = 0; j < cols(); j++)
    {
      int err = matrix(i, j).scale(ScalarConstant);
      if (err != 0) FOUR_C_THROW("Scaling of matrix block ({},{}) failed", i, j);
    }
  }
  return 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::LinAlg::BlockSparseMatrixBase::multiply(bool TransA,
    const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y) const
{
  if (TransA) FOUR_C_THROW("transpose multiply not implemented for BlockSparseMatrix");
  return Apply(X.get_epetra_multi_vector(), Y.get_epetra_multi_vector());
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Core::LinAlg::BlockSparseMatrixBase::NormInf() const { return -1; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const char* Core::LinAlg::BlockSparseMatrixBase::Label() const
{
  return "Core::LinAlg::BlockSparseMatrixBase";
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::BlockSparseMatrixBase::UseTranspose() const { return usetranspose_; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::LinAlg::BlockSparseMatrixBase::HasNormInf() const { return false; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Comm& Core::LinAlg::BlockSparseMatrixBase::Comm() const
{
  return full_domain_map().get_epetra_map().Comm();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map& Core::LinAlg::BlockSparseMatrixBase::OperatorDomainMap() const
{
  return full_domain_map().get_epetra_map();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Epetra_Map& Core::LinAlg::BlockSparseMatrixBase::OperatorRangeMap() const
{
  return full_range_map().get_epetra_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::BlockSparseMatrixBase::get_partial_extractor(
    const MultiMapExtractor& full_extractor, const std::vector<unsigned>& block_ids,
    MultiMapExtractor& partial_extractor) const
{
  const unsigned num_blocks = block_ids.size();

  std::shared_ptr<Core::LinAlg::Map> full_map = nullptr;

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> p_block_maps;
  p_block_maps.reserve(num_blocks);

  for (const int id : block_ids)
  {
    p_block_maps.push_back(full_extractor.map(id));

    full_map = merge_map(full_map, full_extractor.map(id), false);
  }

  partial_extractor.setup(*full_map, p_block_maps);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
Core::LinAlg::block_matrix2x2(Core::LinAlg::SparseMatrix& A00, Core::LinAlg::SparseMatrix& A01,
    Core::LinAlg::SparseMatrix& A10, Core::LinAlg::SparseMatrix& A11)
{
  if (!A00.range_map().same_as(A01.range_map()) || !A00.domain_map().same_as(A10.domain_map()) ||
      !A01.domain_map().same_as(A11.domain_map()) || !A10.range_map().same_as(A11.range_map()))
    FOUR_C_THROW("Core::LinAlg::BlockMatrix2x2: block maps are not compatible.");


  // generate range map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> range_maps;
  range_maps.reserve(2);

  range_maps.emplace_back(std::make_shared<Core::LinAlg::Map>(A00.range_map()));
  range_maps.emplace_back(std::make_shared<Core::LinAlg::Map>(A10.range_map()));
  std::shared_ptr<const Core::LinAlg::Map> range_map = MultiMapExtractor::merge_maps(range_maps);
  MultiMapExtractor rangeMMex(*range_map, range_maps);

  // generate domain map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> domain_maps;
  domain_maps.reserve(2);

  domain_maps.emplace_back(std::make_shared<Core::LinAlg::Map>(A00.domain_map()));
  domain_maps.emplace_back(std::make_shared<Core::LinAlg::Map>(A01.domain_map()));
  std::shared_ptr<const Core::LinAlg::Map> domain_map = MultiMapExtractor::merge_maps(domain_maps);
  MultiMapExtractor domainMMex(*domain_map, domain_maps);

  // generate result matrix
  std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>> C =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          domainMMex, rangeMMex);
  // std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> Cb =
  // std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(C);
  // assign matrices
  C->assign(0, 0, Core::LinAlg::DataAccess::Share, A00);
  C->assign(0, 1, Core::LinAlg::DataAccess::Share, A01);
  C->assign(1, 0, Core::LinAlg::DataAccess::Share, A10);
  C->assign(1, 1, Core::LinAlg::DataAccess::Share, A11);

  C->complete();

  return C;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::DefaultBlockMatrixStrategy::DefaultBlockMatrixStrategy(BlockSparseMatrixBase& mat)
    : mat_(mat), scratch_lcols_(mat_.rows())
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::LinAlg::DefaultBlockMatrixStrategy::complete(bool enforce_complete)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::DefaultBlockMatrixStrategy::Complete");

  if (mat_.filled() and not enforce_complete)
  {
    if (ghost_.size() != 0)
    {
      FOUR_C_THROW("no unresolved ghost entries in a filled block matrix allowed");
    }
    return;
  }

  // finish ghost entries

  int rows = mat_.rows();
  int cols = mat_.cols();

  std::set<int> cgids;

  // get the list of all ghost entries gids
  for (int rblock = 0; rblock < rows; ++rblock)
  {
    const Core::LinAlg::Map& rowmap = mat_.range_map(rblock);

    for (int rlid = 0; rlid < rowmap.num_my_elements(); ++rlid)
    {
      int rgid = rowmap.gid(rlid);
      std::transform(ghost_[rgid].begin(), ghost_[rgid].end(), std::inserter(cgids, cgids.begin()),
          [](const auto& pair) { return pair.first; });
    }
  }

  std::vector<int> cgidlist;
  cgidlist.reserve(cgids.size());
  cgidlist.assign(cgids.begin(), cgids.end());
  cgids.clear();

  // get to know the native processors of each ghost entry
  // this is expensive!

  std::vector<int> cpidlist(cgidlist.size());

  int err = mat_.full_domain_map().remote_id_list(
      cgidlist.size(), cgidlist.data(), cpidlist.data(), nullptr);
  if (err != 0) FOUR_C_THROW("RemoteIDList failed");

  MPI_Comm comm = mat_.full_range_map().get_comm();
  const int numproc = Core::Communication::num_mpi_ranks(comm);

  // Send the ghost gids to their respective processor to ask for the domain
  // map the gids belong to.

  std::vector<std::vector<int>> ghostgids(Core::Communication::num_mpi_ranks(comm));
  for (unsigned i = 0; i < cgidlist.size(); ++i)
  {
    ghostgids[cpidlist[i]].push_back(cgidlist[i]);
  }

  cpidlist.clear();
  cgidlist.clear();

  std::vector<std::vector<int>> requests;
  all_to_all_communication(comm, ghostgids, requests);

  // Now all gids are at the processors that own them. Lets find the owning
  // block for each of them.

  std::vector<std::vector<int>> block(Core::Communication::num_mpi_ranks(comm));

  for (int proc = 0; proc < numproc; ++proc)
  {
    for (unsigned i = 0; i < requests[proc].size(); ++i)
    {
      int gid = requests[proc][i];
      for (int cblock = 0; cblock < cols; ++cblock)
      {
        // assume row and range equal domain
        const Core::LinAlg::Map& domainmap = mat_.domain_map(cblock);
        if (domainmap.my_gid(gid))
        {
          block[proc].push_back(cblock);
          break;
        }
      }

      if (block[proc].size() != i + 1)
      {
        FOUR_C_THROW("gid {} not owned by any domain map", gid);
      }
    }
  }

  // communicate our findings back
  requests.clear();
  all_to_all_communication(comm, block, requests);
  block.clear();

  // store domain block number for each ghost gid

  std::map<int, int> ghostmap;
  for (int proc = 0; proc < numproc; ++proc)
  {
    if (requests[proc].size() != ghostgids[proc].size())
    {
      FOUR_C_THROW("size mismatch panic");
    }

    for (unsigned i = 0; i < requests[proc].size(); ++i)
    {
      int cblock = requests[proc][i];
      int cgid = ghostgids[proc][i];

      if (ghostmap.find(cgid) != ghostmap.end())
        FOUR_C_THROW("column gid {} defined more often that once", cgid);

      ghostmap[cgid] = cblock;
    }
  }

  requests.clear();
  ghostgids.clear();

  // and finally do the assembly of ghost entries

  for (auto& irow : ghost_)
  {
    // most stupid way to find the right row
    int rgid = irow.first;
    int rblock = row_block(rgid);
    if (rblock == -1) FOUR_C_THROW("row finding panic");

    for (auto& icol : irow.second)
    {
      int cgid = icol.first;
      if (ghostmap.find(cgid) == ghostmap.end()) FOUR_C_THROW("unknown ghost gid {}", cgid);

      int cblock = ghostmap[cgid];
      double val = icol.second;

      SparseMatrix& matrix = mat_.matrix(rblock, cblock);
      matrix.assemble(val, rgid, cgid);
    }
  }

  ghost_.clear();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
Core::LinAlg::cast_to_block_sparse_matrix_base_and_check_success(
    std::shared_ptr<Core::LinAlg::SparseOperator> input_matrix)
{
  auto block_matrix = std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(input_matrix);
  FOUR_C_ASSERT(block_matrix != nullptr, "Matrix is not a block matrix!");

  return block_matrix;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase>
Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(
    std::shared_ptr<const Core::LinAlg::SparseOperator> input_matrix)
{
  auto block_matrix =
      std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(input_matrix);
  FOUR_C_ASSERT(block_matrix != nullptr, "Matrix is not a block matrix!");

  return block_matrix;
}

FOUR_C_NAMESPACE_CLOSE
