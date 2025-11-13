// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_amgnxn_objects.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void Core::LinearSolver::AMGNxN::BlockedVector::update(
    double a_V, const BlockedVector& V, double a_this)
{
  for (int i = 0; i < get_num_blocks(); i++) get_vector(i)->update(a_V, *(V.get_vector(i)), a_this);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


Core::LinearSolver::AMGNxN::BlockedVector Core::LinearSolver::AMGNxN::BlockedVector::deep_copy()
    const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BlockedVector::DeepCopy");
  BlockedVector out(this->get_num_blocks());
  for (int i = 0; i < get_num_blocks(); i++)
    out.set_vector(Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*(this->get_vector(i))), i);
  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedVector>
Core::LinearSolver::AMGNxN::BlockedVector::deep_copy_rcp() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BlockedVector::DeepCopyRCP");
  Teuchos::RCP<BlockedVector> out = Teuchos::make_rcp<BlockedVector>(this->get_num_blocks());
  for (int i = 0; i < get_num_blocks(); i++)
    out->set_vector(
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*(this->get_vector(i))), i);
  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedVector>
Core::LinearSolver::AMGNxN::BlockedVector::new_rcp(bool ZeroIt) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BlockedVector::NewRCP");
  Teuchos::RCP<BlockedVector> out = Teuchos::make_rcp<BlockedVector>(this->get_num_blocks());
  for (int i = 0; i < get_num_blocks(); i++)
  {
    // int NV = this->GetNumBlocks();
    // out->SetVector(Teuchos::rcp( new Core::LinAlg::MultiVector<double>(Map,NV,ZeroIt)),i); //This
    // is buggy
    out->set_vector(Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*(this->get_vector(i))),
        i);  // This works, I don't know why...
  }
  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void Core::LinearSolver::AMGNxN::BlockedVector::put_scalar(double a)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BlockedVector::PutScalar");
  for (int i = 0; i < get_num_blocks(); i++) get_vector(i)->put_scalar(a);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Core::LinearSolver::AMGNxN::BlockedVector
Core::LinearSolver::AMGNxN::BlockedVector::get_blocked_vector(const std::vector<int>& blocks) const
{
  int NumSubBlocks = blocks.size();
  if (NumSubBlocks > get_num_blocks()) FOUR_C_THROW("Input error too long");

  BlockedVector out(NumSubBlocks);
  for (int i = 0; i < NumSubBlocks; i++)
  {
    int j = blocks[i];
    if (j >= get_num_blocks()) FOUR_C_THROW("The picked block id is too large");
    if (j < 0) FOUR_C_THROW("The picked block id is negative");
    out.set_vector(get_vector(j), i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedVector>
Core::LinearSolver::AMGNxN::BlockedVector::get_blocked_vector_rcp(
    const std::vector<int>& blocks) const
{
  int NumSubBlocks = blocks.size();
  if (NumSubBlocks > get_num_blocks()) FOUR_C_THROW("Input error too long");

  Teuchos::RCP<BlockedVector> out = Teuchos::make_rcp<BlockedVector>(NumSubBlocks);
  for (int i = 0; i < NumSubBlocks; i++)
  {
    int j = blocks[i];
    if (j >= get_num_blocks()) FOUR_C_THROW("The picked block id is too large");
    if (j < 0) FOUR_C_THROW("The picked block id is negative");
    out->set_vector(get_vector(j), i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void Core::LinearSolver::AMGNxN::BlockedMatrix::apply(
    const BlockedVector& in, BlockedVector& out) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BlockedMatrix::Apply");

  // We assume that the maps of the involved objects match!

  if (in.get_num_blocks() != get_num_cols()) FOUR_C_THROW("Bad number of blocks in input vector");

  if (out.get_num_blocks() != get_num_rows()) FOUR_C_THROW("Bad number of blocks in output vector");



  for (int i = 0; i < get_num_rows(); i++)
  {
    if (out.get_vector(i) == Teuchos::null) FOUR_C_THROW("Have you set your output error?");

    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Yi = out.get_vector(i);
    Yi->put_scalar(0.0);
    Core::LinAlg::MultiVector<double> Yitmp(Yi->get_map(), Yi->num_vectors(), true);

    for (int j = 0; j < get_num_cols(); j++)
    {
      if (in.get_vector(j) == Teuchos::null) FOUR_C_THROW("Have you set your input error?");

      Teuchos::RCP<Core::LinAlg::MultiVector<double>> Xj = in.get_vector(j);
      if (get_matrix(i, j) == Teuchos::null) FOUR_C_THROW("Have you set all the blocks?");

      get_matrix(i, j)->Apply(Xj->get_epetra_multi_vector(), Yitmp.get_epetra_multi_vector());
      Yi->update(1.0, Yitmp, 1.0);
    }
  }


  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase>
Core::LinearSolver::AMGNxN::BlockedMatrix::get_block_sparse_matrix(Core::LinAlg::DataAccess access)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BlockedMatrix::get_block_sparse_matrix");

  int npr = 0;
  int rows = get_num_rows();
  int cols = get_num_cols();

  for (int row = 0; row < rows; row++)
    for (int col = 1; col < cols; col++)
      if (!((matrices_[row * cols + col]->range_map())
                  .same_as(matrices_[row * cols + 0]->range_map())))
        FOUR_C_THROW("The range map must be the same for all matrices_ in the same row");

  for (int col = 0; col < cols; col++)
    for (int row = 1; row < rows; row++)
      if (!((matrices_[row * cols + col]->domain_map())
                  .same_as(matrices_[0 * cols + col]->domain_map())))
        FOUR_C_THROW("The domain map must be the same for all blocks in the same col");

  // build the partial and full domain maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> domain_maps(cols);
  for (int i = 0; i < cols; i++)
    domain_maps[i] = std::make_shared<Core::LinAlg::Map>(matrices_[0 * cols + i]->domain_map());
  auto fullmap_domain = Core::LinAlg::MultiMapExtractor::merge_maps(domain_maps);
  Core::LinAlg::MultiMapExtractor domainmaps(*fullmap_domain, domain_maps);

  // build the partial and full range maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> range_maps(rows);
  for (int i = 0; i < rows; i++)
    range_maps[i] = std::make_shared<Core::LinAlg::Map>(matrices_[i * cols + 0]->range_map());
  auto fullmap_range = Core::LinAlg::MultiMapExtractor::merge_maps(range_maps);
  Core::LinAlg::MultiMapExtractor rangemaps(*fullmap_range, range_maps);

  // Create the concrete matrix
  Teuchos::RCP<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
      the_matrix = Teuchos::make_rcp<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

          domainmaps, rangemaps, npr, true, false);

  // Assign the blocks
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Aij = matrices_[i * cols + j];
      if (Aij == Teuchos::null) FOUR_C_THROW("We need a SparseMatrix here!");
      the_matrix->assign(i, j, access, *Aij);
    }
  }

  // Call complete
  the_matrix->complete();

  // Return
  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> the_matrix_base =
      Teuchos::rcp_dynamic_cast<Core::LinAlg::BlockSparseMatrixBase>(the_matrix);

  return the_matrix_base;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Core::LinearSolver::AMGNxN::BlockedMatrix
Core::LinearSolver::AMGNxN::BlockedMatrix::get_blocked_matrix(
    const std::vector<int>& row_blocks, const std::vector<int>& col_blocks) const
{
  int NumSubRows = row_blocks.size();
  if (NumSubRows > get_num_rows()) FOUR_C_THROW("Row input too long");

  int NumSubCols = col_blocks.size();
  if (NumSubCols > get_num_cols()) FOUR_C_THROW("Col input too long");


  BlockedMatrix out(NumSubRows, NumSubCols);

  for (int i = 0; i < NumSubRows; i++)
  {
    int is = row_blocks[i];
    if (is >= get_num_rows()) FOUR_C_THROW("The picked row block id is too large");
    if (is < 0) FOUR_C_THROW("The picked row block id is negative");

    for (int j = 0; j < NumSubCols; j++)
    {
      int js = col_blocks[j];
      if (js >= get_num_cols()) FOUR_C_THROW("The picked col block id is too large");
      if (js < 0) FOUR_C_THROW("The picked col block id is negative");

      out.set_matrix(get_matrix(is, js), i, j);
    }
  }

  return out;
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void Core::LinearSolver::AMGNxN::DiagonalBlockedMatrix::apply(
    const BlockedVector& in, BlockedVector& out) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::DiagonalBlockedMatrix::Apply");
  // We assume that the maps of the involved objects match!

  if (in.get_num_blocks() != out.get_num_blocks())
    FOUR_C_THROW("Input and output vectors have different number of blocks");

  if (in.get_num_blocks() != get_num_cols()) FOUR_C_THROW("Bad number of blocks in input vector");

  if (out.get_num_blocks() != get_num_rows()) FOUR_C_THROW("Bad number of blocks in output vector");


  for (int i = 0; i < get_num_rows(); i++)
  {
    if (out.get_vector(i) == Teuchos::null) FOUR_C_THROW("Have you set your output error?");
    if (in.get_vector(i) == Teuchos::null) FOUR_C_THROW("Have you set your input error?");

    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Yi = out.get_vector(i);
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Xi = in.get_vector(i);
    get_matrix(i, i)->Apply(Xi->get_epetra_multi_vector(), Yi->get_epetra_multi_vector());
  }


  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::BlockedMatrix::parse_blocks(const std::string& block_string,
    const std::vector<int>& blocks, std::vector<std::vector<int>>& superblocks_to_blocks,
    std::vector<std::vector<int>>& superblocks_to_blocks_local)
{
  if (block_string == "none")
  {
    superblocks_to_blocks.resize(0);
    for (int i = 0; i < (int)blocks.size(); i++)
    {
      std::vector<int> vaux;
      vaux.push_back(blocks[i]);
      superblocks_to_blocks.push_back(vaux);
    }

    superblocks_to_blocks_local = superblocks_to_blocks;
    for (int i = 0; i < (int)blocks.size(); i++)
    {
      std::vector<int> vaux;
      vaux.push_back(i);
      superblocks_to_blocks_local[i] = vaux;
    }

    return;
  }

  // TODO add some checks

  // Parse the string
  superblocks_to_blocks.resize(0);
  bool brace_opened = false;
  std::vector<int> sb;
  std::string sint = "";
  for (int i = 0; i < (int)block_string.size(); i++)
  {
    std::string ch;
    ch.assign(1, block_string[i]);
    if (ch == "(")
    {
      if (brace_opened)
        FOUR_C_THROW(
            "Something wrong. Make sure you are setting correctly the blocks in your xml file");
      brace_opened = true;
      sb.resize(0);
    }
    else if (ch == ")")
    {
      if (not(brace_opened))
        FOUR_C_THROW(
            "Something wrong. Make sure you are setting correctly the blocks in your xml file");
      sb.push_back(atoi(sint.c_str()));
      sint = "";
      superblocks_to_blocks.push_back(sb);
      brace_opened = false;
    }
    else if (ch == ",")
    {
      if (brace_opened)
      {
        sb.push_back(atoi(sint.c_str()));
        sint = "";
      }
    }
    else
      sint += ch;
  }

  // Recover the local numeration and check whether each block was processed exactly once
  superblocks_to_blocks_local = superblocks_to_blocks;
  std::vector<bool> blocks_processed(blocks.size(), false);
  for (int i = 0; i < (int)superblocks_to_blocks.size(); i++)
  {
    for (int j = 0; j < (int)superblocks_to_blocks[i].size(); j++)
    {
      if (std::find(blocks.begin(), blocks.end(), superblocks_to_blocks[i][j]) == blocks.end())
        FOUR_C_THROW(
            "Something wrong. Make sure in your xml file you are counting the blocks starting with "
            "0 and you are not specifying too many blocks!");
      int pos =
          std::find(blocks.begin(), blocks.end(), superblocks_to_blocks[i][j]) - blocks.begin();
      if (blocks_processed[pos])
        FOUR_C_THROW(
            "Matrix block {} has been specified multiple times in the *.xml file!", blocks[pos]);
      else
      {
        blocks_processed[pos] = true;
        superblocks_to_blocks_local[i][j] = pos;
      }
    }
  }
  for (unsigned iblock = 0; iblock < blocks_processed.size(); ++iblock)
    if (not blocks_processed[iblock])
      FOUR_C_THROW("Matrix block {} has not been specified in the *.xml file!", blocks[iblock]);

  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedVector>
Core::LinearSolver::AMGNxN::BlockedMatrix::new_domain_blocked_vector(int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::make_rcp<BlockedVector>(get_num_cols());
  for (int i = 0; i < get_num_cols(); i++)
  {
    const Core::LinAlg::Map& Map = get_matrix(0, i)->domain_map();
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Vi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(
            Map, NV, ZeroIt);  // This constructor seems to be buggy inside function
                               // BlockedVector::NewRCP. I don't know why here works well.

    out->set_vector(Vi, i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedVector>
Core::LinearSolver::AMGNxN::BlockedMatrix::new_range_blocked_vector(int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::make_rcp<BlockedVector>(get_num_rows());
  for (int i = 0; i < get_num_rows(); i++)
  {
    const Core::LinAlg::Map& Map = get_matrix(i, 0)->range_map();
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Vi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(Map, NV, ZeroIt);

    out->set_vector(Vi, i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedVector>
Core::LinearSolver::AMGNxN::DiagonalBlockedMatrix::new_domain_blocked_vector(
    int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::make_rcp<BlockedVector>(get_num_cols());
  for (int i = 0; i < get_num_cols(); i++)
  {
    const Core::LinAlg::Map& Map = get_matrix(i, i)->domain_map();
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Vi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(Map, NV, ZeroIt);

    out->set_vector(Vi, i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedVector>
Core::LinearSolver::AMGNxN::DiagonalBlockedMatrix::new_range_blocked_vector(
    int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::make_rcp<BlockedVector>(get_num_rows());
  for (int i = 0; i < get_num_rows(); i++)
  {
    const Core::LinAlg::Map& Map = get_matrix(i, i)->range_map();
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Vi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(Map, NV, ZeroIt);

    out->set_vector(Vi, i);
  }

  return out;
}

FOUR_C_NAMESPACE_CLOSE
