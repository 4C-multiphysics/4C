/*----------------------------------------------------------------------*/
/*! \file

\brief Declaration

\level 1

*/
/*----------------------------------------------------------------------*/

#include "baci_linear_solver_amgnxn_objects.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void CORE::LINEAR_SOLVER::AMGNXN::BlockedVector::Update(
    double a_V, const BlockedVector& V, double a_this)
{
  for (int i = 0; i < GetNumBlocks(); i++) GetVector(i)->Update(a_V, *(V.GetVector(i)), a_this);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


CORE::LINEAR_SOLVER::AMGNXN::BlockedVector CORE::LINEAR_SOLVER::AMGNXN::BlockedVector::DeepCopy()
    const
{
  TEUCHOS_FUNC_TIME_MONITOR("CORE::LINALG::SOLVER::AMGNXN::BlockedVector::DeepCopy");
  BlockedVector out(this->GetNumBlocks());
  for (int i = 0; i < GetNumBlocks(); i++)
    out.SetVector(Teuchos::rcp(new Epetra_MultiVector(*(this->GetVector(i)))), i);
  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


Teuchos::RCP<CORE::LINEAR_SOLVER::AMGNXN::BlockedVector>
CORE::LINEAR_SOLVER::AMGNXN::BlockedVector::DeepCopyRCP() const
{
  TEUCHOS_FUNC_TIME_MONITOR("CORE::LINALG::SOLVER::AMGNXN::BlockedVector::DeepCopyRCP");
  Teuchos::RCP<BlockedVector> out = Teuchos::rcp(new BlockedVector(this->GetNumBlocks()));
  for (int i = 0; i < GetNumBlocks(); i++)
    out->SetVector(Teuchos::rcp(new Epetra_MultiVector(*(this->GetVector(i)))), i);
  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


Teuchos::RCP<CORE::LINEAR_SOLVER::AMGNXN::BlockedVector>
CORE::LINEAR_SOLVER::AMGNXN::BlockedVector::NewRCP(bool ZeroIt) const
{
  TEUCHOS_FUNC_TIME_MONITOR("CORE::LINALG::SOLVER::AMGNXN::BlockedVector::NewRCP");
  Teuchos::RCP<BlockedVector> out = Teuchos::rcp(new BlockedVector(this->GetNumBlocks()));
  for (int i = 0; i < GetNumBlocks(); i++)
  {
    // const Epetra_BlockMap&  Map = this->GetVector(i)->Map();
    // int NV = this->GetNumBlocks();
    // out->SetVector(Teuchos::rcp( new Epetra_MultiVector(Map,NV,ZeroIt)),i); //This is buggy
    out->SetVector(Teuchos::rcp(new Epetra_MultiVector(*(this->GetVector(i)))),
        i);  // This works, I don't know why...
  }
  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void CORE::LINEAR_SOLVER::AMGNXN::BlockedVector::PutScalar(double a)
{
  TEUCHOS_FUNC_TIME_MONITOR("CORE::LINALG::SOLVER::AMGNXN::BlockedVector::PutScalar");
  for (int i = 0; i < GetNumBlocks(); i++) GetVector(i)->PutScalar(a);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

CORE::LINEAR_SOLVER::AMGNXN::BlockedVector
CORE::LINEAR_SOLVER::AMGNXN::BlockedVector::GetBlockedVector(const std::vector<int>& blocks) const
{
  int NumSubBlocks = blocks.size();
  if (NumSubBlocks > GetNumBlocks()) FOUR_C_THROW("Input error too long");

  BlockedVector out(NumSubBlocks);
  for (int i = 0; i < NumSubBlocks; i++)
  {
    int j = blocks[i];
    if (j >= GetNumBlocks()) FOUR_C_THROW("The picked block id is too large");
    if (j < 0) FOUR_C_THROW("The picked block id is negative");
    out.SetVector(GetVector(j), i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<CORE::LINEAR_SOLVER::AMGNXN::BlockedVector>
CORE::LINEAR_SOLVER::AMGNXN::BlockedVector::GetBlockedVectorRCP(
    const std::vector<int>& blocks) const
{
  int NumSubBlocks = blocks.size();
  if (NumSubBlocks > GetNumBlocks()) FOUR_C_THROW("Input error too long");

  Teuchos::RCP<BlockedVector> out = Teuchos::rcp(new BlockedVector(NumSubBlocks));
  for (int i = 0; i < NumSubBlocks; i++)
  {
    int j = blocks[i];
    if (j >= GetNumBlocks()) FOUR_C_THROW("The picked block id is too large");
    if (j < 0) FOUR_C_THROW("The picked block id is negative");
    out->SetVector(GetVector(j), i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void CORE::LINEAR_SOLVER::AMGNXN::BlockedMatrix::Apply(
    const BlockedVector& in, BlockedVector& out) const
{
  TEUCHOS_FUNC_TIME_MONITOR("CORE::LINALG::SOLVER::AMGNXN::BlockedMatrix::Apply");

  // We assume that the maps of the involved objects match!

  if (in.GetNumBlocks() != GetNumCols()) FOUR_C_THROW("Bad number of blocks in input vector");

  if (out.GetNumBlocks() != GetNumRows()) FOUR_C_THROW("Bad number of blocks in output vector");



  for (int i = 0; i < GetNumRows(); i++)
  {
    if (out.GetVector(i) == Teuchos::null) FOUR_C_THROW("Have you set your output error?");

    Teuchos::RCP<Epetra_MultiVector> Yi = out.GetVector(i);
    Yi->PutScalar(0.0);
    Teuchos::RCP<Epetra_MultiVector> Yitmp =
        Teuchos::rcp(new Epetra_MultiVector(Yi->Map(), Yi->NumVectors(), true));

    for (int j = 0; j < GetNumCols(); j++)
    {
      if (in.GetVector(j) == Teuchos::null) FOUR_C_THROW("Have you set your input error?");

      Teuchos::RCP<Epetra_MultiVector> Xj = in.GetVector(j);
      if (GetMatrix(i, j) == Teuchos::null) FOUR_C_THROW("Have you set all the blocks?");

      GetMatrix(i, j)->Apply(*Xj, *Yitmp);
      Yi->Update(1.0, *Yitmp, 1.0);
    }
  }


  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase>
CORE::LINEAR_SOLVER::AMGNXN::BlockedMatrix::GetBlockSparseMatrix(CORE::LINALG::DataAccess access)
{
  TEUCHOS_FUNC_TIME_MONITOR("CORE::LINALG::SOLVER::AMGNXN::BlockedMatrix::GetBlockSparseMatrix");

  int npr = 0;
  int rows = GetNumRows();
  int cols = GetNumCols();

  for (int row = 0; row < rows; row++)
    for (int col = 1; col < cols; col++)
      if (!((matrices_[row * cols + col]->RangeMap())
                  .SameAs(matrices_[row * cols + 0]->RangeMap())))
        FOUR_C_THROW("The range map must be the same for all matrices_ in the same row");

  for (int col = 0; col < cols; col++)
    for (int row = 1; row < rows; row++)
      if (!((matrices_[row * cols + col]->DomainMap())
                  .SameAs(matrices_[0 * cols + col]->DomainMap())))
        FOUR_C_THROW("The domain map must be the same for all blocks in the same col");

  // build the partial and full domain maps
  std::vector<Teuchos::RCP<const Epetra_Map>> domain_maps(cols, Teuchos::null);
  for (int i = 0; i < cols; i++)
    domain_maps[i] = Teuchos::rcp(new Epetra_Map(matrices_[0 * cols + i]->DomainMap()));
  Teuchos::RCP<Epetra_Map> fullmap_domain = CORE::LINALG::MultiMapExtractor::MergeMaps(domain_maps);
  Teuchos::RCP<CORE::LINALG::MultiMapExtractor> domainmaps =
      Teuchos::rcp(new CORE::LINALG::MultiMapExtractor(*fullmap_domain, domain_maps));

  // build the partial and full range maps
  std::vector<Teuchos::RCP<const Epetra_Map>> range_maps(rows, Teuchos::null);
  for (int i = 0; i < rows; i++)
    range_maps[i] = Teuchos::rcp(new Epetra_Map(matrices_[i * cols + 0]->RangeMap()));
  Teuchos::RCP<Epetra_Map> fullmap_range = CORE::LINALG::MultiMapExtractor::MergeMaps(range_maps);
  Teuchos::RCP<CORE::LINALG::MultiMapExtractor> rangemaps =
      Teuchos::rcp(new CORE::LINALG::MultiMapExtractor(*fullmap_range, range_maps));

  // Create the concrete matrix
  Teuchos::RCP<CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>>
      the_matrix = Teuchos::rcp(
          new CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>(
              *domainmaps, *rangemaps, npr, true, false));

  // Assign the blocks
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      Teuchos::RCP<CORE::LINALG::SparseMatrix> Aij = matrices_[i * cols + j];
      if (Aij == Teuchos::null) FOUR_C_THROW("We need a SparseMatrix here!");
      the_matrix->Assign(i, j, access, *Aij);
    }
  }

  // Call complete
  the_matrix->Complete();

  // Return
  Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase> the_matrix_base =
      Teuchos::rcp_dynamic_cast<CORE::LINALG::BlockSparseMatrixBase>(the_matrix);

  return the_matrix_base;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

CORE::LINEAR_SOLVER::AMGNXN::BlockedMatrix
CORE::LINEAR_SOLVER::AMGNXN::BlockedMatrix::GetBlockedMatrix(
    const std::vector<int>& row_blocks, const std::vector<int>& col_blocks) const
{
  int NumSubRows = row_blocks.size();
  if (NumSubRows > GetNumRows()) FOUR_C_THROW("Row input too long");

  int NumSubCols = col_blocks.size();
  if (NumSubCols > GetNumCols()) FOUR_C_THROW("Col input too long");


  BlockedMatrix out(NumSubRows, NumSubCols);

  for (int i = 0; i < NumSubRows; i++)
  {
    int is = row_blocks[i];
    if (is >= GetNumRows()) FOUR_C_THROW("The picked row block id is too large");
    if (is < 0) FOUR_C_THROW("The picked row block id is negative");

    for (int j = 0; j < NumSubCols; j++)
    {
      int js = col_blocks[j];
      if (js >= GetNumCols()) FOUR_C_THROW("The picked col block id is too large");
      if (js < 0) FOUR_C_THROW("The picked col block id is negative");

      out.SetMatrix(GetMatrix(is, js), i, j);
    }
  }

  return out;
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


void CORE::LINEAR_SOLVER::AMGNXN::DiagonalBlockedMatrix::Apply(
    const BlockedVector& in, BlockedVector& out) const
{
  TEUCHOS_FUNC_TIME_MONITOR("CORE::LINALG::SOLVER::AMGNXN::DiagonalBlockedMatrix::Apply");
  // We assume that the maps of the involved objects match!

  if (in.GetNumBlocks() != out.GetNumBlocks())
    FOUR_C_THROW("Input and output vectors have different number of blocks");

  if (in.GetNumBlocks() != GetNumCols()) FOUR_C_THROW("Bad number of blocks in input vector");

  if (out.GetNumBlocks() != GetNumRows()) FOUR_C_THROW("Bad number of blocks in output vector");


  for (int i = 0; i < GetNumRows(); i++)
  {
    if (out.GetVector(i) == Teuchos::null) FOUR_C_THROW("Have you set your output error?");
    if (in.GetVector(i) == Teuchos::null) FOUR_C_THROW("Have you set your input error?");

    Teuchos::RCP<Epetra_MultiVector> Yi = out.GetVector(i);
    Teuchos::RCP<Epetra_MultiVector> Xi = in.GetVector(i);
    GetMatrix(i, i)->Apply(*Xi, *Yi);
  }


  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void CORE::LINEAR_SOLVER::AMGNXN::BlockedMatrix::ParseBlocks(const std::string& block_string,
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
            "Matrix block %d has been specified multiple times in the *.xml file!", blocks[pos]);
      else
      {
        blocks_processed[pos] = true;
        superblocks_to_blocks_local[i][j] = pos;
      }
    }
  }
  for (unsigned iblock = 0; iblock < blocks_processed.size(); ++iblock)
    if (not blocks_processed[iblock])
      FOUR_C_THROW("Matrix block %d has not been specified in the *.xml file!", blocks[iblock]);

  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<CORE::LINEAR_SOLVER::AMGNXN::BlockedVector>
CORE::LINEAR_SOLVER::AMGNXN::BlockedMatrix::NewDomainBlockedVector(int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::rcp(new BlockedVector(GetNumCols()));
  for (int i = 0; i < GetNumCols(); i++)
  {
    const Epetra_Map& Map = GetMatrix(0, i)->DomainMap();
    Teuchos::RCP<Epetra_MultiVector> Vi = Teuchos::rcp(new Epetra_MultiVector(
        Map, NV, ZeroIt));  // This constructor seems to be buggy inside function
                            // BlockedVector::NewRCP. I don't know why here works well.

    out->SetVector(Vi, i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<CORE::LINEAR_SOLVER::AMGNXN::BlockedVector>
CORE::LINEAR_SOLVER::AMGNXN::BlockedMatrix::NewRangeBlockedVector(int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::rcp(new BlockedVector(GetNumRows()));
  for (int i = 0; i < GetNumRows(); i++)
  {
    const Epetra_Map& Map = GetMatrix(i, 0)->RangeMap();
    Teuchos::RCP<Epetra_MultiVector> Vi = Teuchos::rcp(new Epetra_MultiVector(Map, NV, ZeroIt));

    out->SetVector(Vi, i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<CORE::LINEAR_SOLVER::AMGNXN::BlockedVector>
CORE::LINEAR_SOLVER::AMGNXN::DiagonalBlockedMatrix::NewDomainBlockedVector(
    int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::rcp(new BlockedVector(GetNumCols()));
  for (int i = 0; i < GetNumCols(); i++)
  {
    const Epetra_Map& Map = GetMatrix(i, i)->DomainMap();
    Teuchos::RCP<Epetra_MultiVector> Vi = Teuchos::rcp(new Epetra_MultiVector(Map, NV, ZeroIt));

    out->SetVector(Vi, i);
  }

  return out;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<CORE::LINEAR_SOLVER::AMGNXN::BlockedVector>
CORE::LINEAR_SOLVER::AMGNXN::DiagonalBlockedMatrix::NewRangeBlockedVector(int NV, bool ZeroIt) const
{
  Teuchos::RCP<BlockedVector> out = Teuchos::rcp(new BlockedVector(GetNumRows()));
  for (int i = 0; i < GetNumRows(); i++)
  {
    const Epetra_Map& Map = GetMatrix(i, i)->RangeMap();
    Teuchos::RCP<Epetra_MultiVector> Vi = Teuchos::rcp(new Epetra_MultiVector(Map, NV, ZeroIt));

    out->SetVector(Vi, i);
  }

  return out;
}

FOUR_C_NAMESPACE_CLOSE
