/*----------------------------------------------------------------------*/
/*! \file
\brief Implementation

\level 1

*----------------------------------------------------------------------*/

// TODO replace me by EpetraExt routines
// includes for MLMultiply...
#include "4C_linalg_multiply.hpp"

#include "4C_linalg_sparsematrix.hpp"

#include <EpetraExt_Transpose_RowMatrix.h>
#include <ml_MultiLevelPreconditioner.h>
#include <MLAPI_Operator_Utils.h>
#include <MLAPI_Workspace.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 02/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix> Core::LinAlg::ml_multiply(
    const Core::LinAlg::SparseMatrix& A, const Core::LinAlg::SparseMatrix& B, bool complete)
{
  return ml_multiply(
      *A.epetra_matrix(), *B.epetra_matrix(), A.explicitdirichlet_, A.savegraph_, complete);
}

/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 02/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix> Core::LinAlg::ml_multiply(
    const Core::LinAlg::SparseMatrix& A, const Core::LinAlg::SparseMatrix& B,
    bool explicitdirichlet, bool savegraph, bool complete)
{
  return ml_multiply(
      *A.epetra_matrix(), *B.epetra_matrix(), explicitdirichlet, savegraph, complete);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::SparseMatrix> Core::LinAlg::ml_multiply(
    const Core::LinAlg::SparseMatrix& A, bool transA, const Core::LinAlg::SparseMatrix& B,
    bool transB, bool explicitdirichlet, bool savegraph, bool completeoutput)
{
  // make sure fill_complete was called on the matrices
  if (!A.filled()) FOUR_C_THROW("A has to be fill_complete");
  if (!B.filled()) FOUR_C_THROW("B has to be fill_complete");

  EpetraExt::RowMatrix_Transpose transposera;
  EpetraExt::RowMatrix_Transpose transposerb;
  Epetra_CrsMatrix* Atrans = nullptr;
  Epetra_CrsMatrix* Btrans = nullptr;
  if (transA)
    Atrans = dynamic_cast<Epetra_CrsMatrix*>(&transposera(*A.epetra_matrix()));
  else
    Atrans = A.epetra_matrix().get();
  if (transB)
    Btrans = dynamic_cast<Epetra_CrsMatrix*>(&transposerb(*B.epetra_matrix()));
  else
    Btrans = B.epetra_matrix().get();

  Teuchos::RCP<Core::LinAlg::SparseMatrix> C;
  C = Core::LinAlg::ml_multiply(*Atrans, *Btrans, explicitdirichlet, savegraph, completeoutput);

  return C;
}

/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 02/08|
 *----------------------------------------------------------------------*/
// static void CopySortDeleteZeros(const Epetra_CrsMatrix& A, Epetra_CrsMatrix& As);
Teuchos::RCP<Core::LinAlg::SparseMatrix> Core::LinAlg::ml_multiply(const Epetra_CrsMatrix& Aorig,
    const Epetra_CrsMatrix& Borig, bool explicitdirichlet, bool savegraph, bool complete)
{
  EpetraExt::CrsMatrix_SolverMap Atransform;
  EpetraExt::CrsMatrix_SolverMap Btransform;
  const Epetra_CrsMatrix& A = Atransform(const_cast<Epetra_CrsMatrix&>(Aorig));
  const Epetra_CrsMatrix& B = Btransform(const_cast<Epetra_CrsMatrix&>(Borig));

  // make sure fill_complete was called on the matrices
  if (!A.Filled()) FOUR_C_THROW("A has to be fill_complete");
  if (!B.Filled()) FOUR_C_THROW("B has to be fill_complete");

  // For debugging, it might be helpful when all columns are
  // sorted and all zero values are wiped from the input:
  // Teuchos::RCP<Epetra_CrsMatrix> As = CreateMatrix(A.RowMap(),A.MaxNumEntries());
  // Teuchos::RCP<Epetra_CrsMatrix> Bs = CreateMatrix(B.RowMap(),B.MaxNumEntries());
  // CopySortDeleteZeros(A,*As);
  // CopySortDeleteZeros(B,*Bs);
  ML_Operator* ml_As = ML_Operator_Create(MLAPI::GetML_Comm());
  ML_Operator* ml_Bs = ML_Operator_Create(MLAPI::GetML_Comm());
  // ML_Operator_WrapEpetraMatrix(As.get(),ml_As);
  // ML_Operator_WrapEpetraMatrix(Bs.get(),ml_Bs);
  ML_Operator_WrapEpetraMatrix(const_cast<Epetra_CrsMatrix*>(&A), ml_As);
  ML_Operator_WrapEpetraMatrix(const_cast<Epetra_CrsMatrix*>(&B), ml_Bs);
  ML_Operator* ml_AtimesB = ML_Operator_Create(MLAPI::GetML_Comm());
  ML_2matmult(ml_As, ml_Bs, ml_AtimesB, ML_CSR_MATRIX);  // do NOT use ML_EpetraCRS_MATRIX !!
  ML_Operator_Destroy(&ml_As);
  ML_Operator_Destroy(&ml_Bs);
  // For ml_AtimesB we have to reconstruct the column map in global indexing,
  // The following is going down to the salt-mines of ML ...
  int N_local = ml_AtimesB->invec_leng;
  ML_CommInfoOP* getrow_comm = ml_AtimesB->getrow->pre_comm;
  if (!getrow_comm) FOUR_C_THROW("ML_Operator does not have CommInfo");
  ML_Comm* comm = ml_AtimesB->comm;
  if (N_local != B.DomainMap().NumMyElements())
    FOUR_C_THROW("Mismatch in local row dimension between ML and Epetra");
  int N_rcvd = 0;
  int N_send = 0;
  int flag = 0;
  for (int i = 0; i < getrow_comm->N_neighbors; i++)
  {
    N_rcvd += (getrow_comm->neighbors)[i].N_rcv;
    N_send += (getrow_comm->neighbors)[i].N_send;
    if (((getrow_comm->neighbors)[i].N_rcv != 0) &&
        ((getrow_comm->neighbors)[i].rcv_list != nullptr))
      flag = 1;
  }
  // For some unknown reason, ML likes to have stuff one larger than
  // neccessary...
  std::vector<double> dtemp(N_local + N_rcvd + 1);
  std::vector<int> cmap(N_local + N_rcvd + 1);
  for (int i = 0; i < N_local; ++i)
  {
    cmap[i] = B.DomainMap().GID(i);
    dtemp[i] = (double)cmap[i];
  }
  ML_cheap_exchange_bdry(dtemp.data(), getrow_comm, N_local, N_send, comm);
  if (flag)
  {
    int count = N_local;
    const int neighbors = getrow_comm->N_neighbors;
    for (int i = 0; i < neighbors; i++)
    {
      const int nrcv = getrow_comm->neighbors[i].N_rcv;
      for (int j = 0; j < nrcv; j++)
        cmap[getrow_comm->neighbors[i].rcv_list[j]] = (int)dtemp[count++];
    }
  }
  else
    for (int i = 0; i < N_local + N_rcvd; ++i) cmap[i] = (int)dtemp[i];
  dtemp.clear();

  // we can now determine a matching column map for the result
  Epetra_Map gcmap(-1, N_local + N_rcvd, cmap.data(), B.ColMap().IndexBase(), A.Comm());

  int allocated = 0;
  int rowlength;
  double* val = nullptr;
  int* bindx = nullptr;
  const int myrowlength = A.RowMap().NumMyElements();
  const Epetra_Map& rowmap = A.RowMap();

  // determine the maximum bandwith for the result matrix.
  int educatedguess = 0;
  for (int i = 0; i < myrowlength; ++i)
  {
    // get local row
    ML_get_matrix_row(ml_AtimesB, 1, &i, &allocated, &bindx, &val, &rowlength, 0);
    if (rowlength > educatedguess) educatedguess = rowlength;
  }

  // allocate our result matrix and fill it
  Teuchos::RCP<Epetra_CrsMatrix> result =
      Teuchos::rcp(new Epetra_CrsMatrix(::Copy, A.RangeMap(), gcmap, educatedguess, false));

  std::vector<int> gcid(educatedguess);
  for (int i = 0; i < myrowlength; ++i)
  {
    const int grid = rowmap.GID(i);
    // get local row
    ML_get_matrix_row(ml_AtimesB, 1, &i, &allocated, &bindx, &val, &rowlength, 0);
    if (!rowlength) continue;
    if ((int)gcid.size() < rowlength) gcid.resize(rowlength);
    for (int j = 0; j < rowlength; ++j)
    {
      gcid[j] = gcmap.GID(bindx[j]);
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (gcid[j] < 0) FOUR_C_THROW("This is really bad... cannot find gcid");
#endif
    }
#ifdef FOUR_C_ENABLE_ASSERTIONS
    int err = result->InsertGlobalValues(grid, rowlength, val, gcid.data());
    if (err != 0 && err != 1)
      FOUR_C_THROW("Epetra_CrsMatrix::InsertGlobalValues returned err=%d", err);
#else
    result->InsertGlobalValues(grid, rowlength, val, gcid.data());
#endif
  }
  if (bindx) ML_free(bindx);
  if (val) ML_free(val);
  ML_Operator_Destroy(&ml_AtimesB);
  if (complete)
  {
    int err = result->FillComplete(B.DomainMap(), A.RangeMap());
    if (err) FOUR_C_THROW("Epetra_CrsMatrix::FillComplete returned err=%d", err);
  }
  return Teuchos::rcp(new SparseMatrix(result, View, explicitdirichlet, savegraph));
}

FOUR_C_NAMESPACE_CLOSE
