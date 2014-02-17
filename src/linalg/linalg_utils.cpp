/*!----------------------------------------------------------------------
\file linalg_utils.cpp
\brief A collection of helper methods for namespace LINALG

<pre>
-------------------------------------------------------------------------
                 BACI finite element library subsystem
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library may solemnly used in conjunction with the BACI contact library
for purposes described in the above mentioned contract.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>
<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/

#include <algorithm>
#include <numeric>
#include <vector>

#include "../headers/compiler_definitions.h" /* access to fortran routines */
#include "linalg_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "EpetraExt_Transpose_RowMatrix.h"
#include "EpetraExt_MatrixMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_RowMatrixTransposer.h"
#include "Ifpack_AdditiveSchwarz.h"

/*----------------------------------------------------------------------*
 |  create a Epetra_CrsMatrix  (public)                      mwgee 12/06|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_CrsMatrix> LINALG::CreateMatrix(const Epetra_Map& rowmap, const int npr)
{
  if (!rowmap.UniqueGIDs()) dserror("Row map is not unique");
  return Teuchos::rcp(new Epetra_CrsMatrix(Copy,rowmap,npr,false));
}
/*----------------------------------------------------------------------*
 |  create a Epetra_Vector  (public)                         mwgee 12/06|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> LINALG::CreateVector(const Epetra_Map& rowmap, const bool init)
{
  return Teuchos::rcp(new Epetra_Vector(rowmap,init));
}
/*----------------------------------------------------------------------*
 |  export a Epetra_Vector  (public)                         mwgee 12/06|
 *----------------------------------------------------------------------*/
void LINALG::Export(const Epetra_MultiVector& source, Epetra_MultiVector& target)
{
  try
  {
    const bool sourceunique = source.Map().UniqueGIDs();
    const bool targetunique = target.Map().UniqueGIDs();

    // both are unique, does not matter whether ex- or import
    if (sourceunique && targetunique &&
        source.Comm().NumProc()==1   &&
        target.Comm().NumProc()==1 )
    {
      if (source.NumVectors() != target.NumVectors())
        dserror("number of vectors in source and target not the same!");
      for (int k=0; k<source.NumVectors(); ++k)
        for (int i=0; i<target.Map().NumMyElements(); ++i)
        {
          const int gid = target.Map().GID(i);
          if (gid<0) dserror("No gid for i");
          const int lid = source.Map().LID(gid);
          if (lid<0) continue;
            //dserror("No source for target");
          (*target(k))[i] = (*source(k))[lid];
        }
      return;
    }
    else if (sourceunique && targetunique)
    {
      Epetra_Export exporter(source.Map(),target.Map());
      int err = target.Export(source,exporter,Insert);
      if (err) dserror("Export using exporter returned err=%d",err);
      return;
    }
    else if (sourceunique && !targetunique)
    {
      Epetra_Import importer(target.Map(),source.Map());
      int err = target.Import(source,importer,Insert);
      if (err) dserror("Export using exporter returned err=%d",err);
      return;
    }
    else if (!sourceunique && targetunique)
    {
      Epetra_Export exporter(source.Map(),target.Map());
      int err = target.Export(source,exporter,Insert);
      if (err) dserror("Export using exporter returned err=%d",err);
      return;
    }
    else if (!sourceunique && !targetunique)
    {
      // Neither target nor source are unique - this is a problem.
      // We need a unique in between stage which we have to create artificially.
      // That's nasty.
      // As it is unclear whether this will ever be needed - do it later.
      dserror("Neither target nor source maps are unique - cannot export");
    }
    else dserror("VERY strange");
  }
  catch(int error)
  {
    dserror("Caught an Epetra exception %d",error);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  export a Epetra_IntVector  (public)                      mwgee 01/13|
 *----------------------------------------------------------------------*/
void LINALG::Export(const Epetra_IntVector& source, Epetra_IntVector& target)
{
  try
  {
    const bool sourceunique = source.Map().UniqueGIDs();
    const bool targetunique = target.Map().UniqueGIDs();

    // both are unique, does not matter whether ex- or import
    if (sourceunique && targetunique &&
        source.Comm().NumProc()==1   &&
        target.Comm().NumProc()==1 )
    {
        for (int i=0; i<target.Map().NumMyElements(); ++i)
        {
          const int gid = target.Map().GID(i);
          if (gid<0) dserror("No gid for i");
          const int lid = source.Map().LID(gid);
          if (lid<0) continue;
          target[i] = source[lid];
        }
      return;
    }
    else if (sourceunique && targetunique)
    {
      Epetra_Export exporter(source.Map(),target.Map());
      int err = target.Export(source,exporter,Insert);
      if (err) dserror("Export using exporter returned err=%d",err);
      return;
    }
    else if (sourceunique && !targetunique)
    {
      Epetra_Import importer(target.Map(),source.Map());
      int err = target.Import(source,importer,Insert);
      if (err) dserror("Export using exporter returned err=%d",err);
      return;
    }
    else if (!sourceunique && targetunique)
    {
      Epetra_Export exporter(source.Map(),target.Map());
      int err = target.Export(source,exporter,Insert);
      if (err) dserror("Export using exporter returned err=%d",err);
      return;
    }
    else if (!sourceunique && !targetunique)
    {
      // Neither target nor source are unique - this is a problem.
      // We need a unique in between stage which we have to create artificially.
      // That's nasty.
      // As it is unclear whether this will ever be needed - do it later.
      dserror("Neither target nor source maps are unique - cannot export");
    }
    else dserror("VERY strange");
  }
  catch(int error)
  {
    dserror("Caught an Epetra exception %d",error);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  assemble a matrix  (public)                               popp 01/08|
 *----------------------------------------------------------------------*/
void LINALG::Assemble(Epetra_CrsMatrix& A, const Epetra_SerialDenseMatrix& Aele,
                      const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
                      const std::vector<int>& lmcol)
{
  const int lrowdim = (int)lmrow.size();
  const int lcoldim = (int)lmcol.size();
  if (lrowdim!=(int)lmrowowner.size() || lrowdim!=Aele.M() || lcoldim!=Aele.N())
    dserror("Mismatch in dimensions");

  const int myrank = A.Comm().MyPID();
  const Epetra_Map& rowmap = A.RowMap();

  // this 'Assemble' is not implemented for a Filled() matrix A
  if (A.Filled()) dserror("Sparse matrix A is already Filled()");

  else
  {
    // loop rows of local matrix
    for (int lrow=0; lrow<lrowdim; ++lrow)
    {
      // check ownership of row
      if (lmrowowner[lrow] != myrank) continue;

      // check whether I have that global row
      int rgid = lmrow[lrow];
      if (!(rowmap.MyGID(rgid))) dserror("Sparse matrix A does not have global row %d",rgid);

      for (int lcol=0; lcol<lcoldim; ++lcol)
      {
        double val = Aele(lrow,lcol);
        int cgid = lmcol[lcol];

        // Now that we do not rebuild the sparse mask in each step, we
        // are bound to assemble the whole thing. Zeros included.
        int errone = A.SumIntoGlobalValues(rgid,1,&val,&cgid);
        if (errone>0)
        {
          int errtwo = A.InsertGlobalValues(rgid,1,&val,&cgid);
          if (errtwo<0) dserror("Epetra_CrsMatrix::InsertGlobalValues returned error code %d",errtwo);
        }
        else if (errone)
          dserror("Epetra_CrsMatrix::SumIntoGlobalValues returned error code %d",errone);
      } // for (int lcol=0; lcol<lcoldim; ++lcol)
    } // for (int lrow=0; lrow<lrowdim; ++lrow)
  }
  return;
}

/*----------------------------------------------------------------------*
 |  assemble a vector  (public)                              mwgee 12/06|
 *----------------------------------------------------------------------*/
void LINALG::Assemble(Epetra_Vector& V, const Epetra_SerialDenseVector& Vele,
                const std::vector<int>& lm, const std::vector<int>& lmowner)
{
  const int ldim = (int)lm.size();
  if (ldim!=(int)lmowner.size() || ldim!=Vele.Length())
    dserror("Mismatch in dimensions");

  const int myrank = V.Comm().MyPID();

  for (int lrow=0; lrow<ldim; ++lrow)
  {
    if (lmowner[lrow] != myrank) continue;
    int rgid = lm[lrow];
    if (!V.Map().MyGID(rgid)) dserror("Sparse vector V does not have global row %d",rgid);
    int rlid = V.Map().LID(rgid);
    V[rlid] += Vele[lrow];
  } // for (int lrow=0; lrow<ldim; ++lrow)

  return;
}

/*----------------------------------------------------------------------*
 |  assemble a vector into MultiVector (public)              mwgee 01/08|
 *----------------------------------------------------------------------*/
void LINALG::Assemble(Epetra_MultiVector& V, const int n, const Epetra_SerialDenseVector& Vele,
                const std::vector<int>& lm, const std::vector<int>& lmowner)
{
  LINALG::Assemble(*(V(n)),Vele,lm,lmowner);
  return;
}

/*----------------------------------------------------------------------*
 |  FillComplete a matrix  (public)                          mwgee 12/06|
 *----------------------------------------------------------------------*/
void LINALG::Complete(Epetra_CrsMatrix& A)
{
  if (A.Filled()) return;
  int err = A.FillComplete(A.OperatorDomainMap(),A.OperatorRangeMap(),true);
  if (err) dserror("Epetra_CrsMatrix::FillComplete(domain,range) returned err=%d",err);
  return;
}

/*----------------------------------------------------------------------*
 |  FillComplete a matrix  (public)                          mwgee 01/08|
 *----------------------------------------------------------------------*/
void  LINALG::Complete(Epetra_CrsMatrix& A, const Epetra_Map& domainmap, const Epetra_Map& rangemap)
{
  if (A.Filled()) return;
  int err = A.FillComplete(domainmap,rangemap,true);
  if (err) dserror("Epetra_CrsMatrix::FillComplete(domain,range) returned err=%d",err);
  return;
}

/*----------------------------------------------------------------------*
 |  Add a sparse matrix to another  (public)                 mwgee 12/06|
 |  B = B*scalarB + A(transposed)*scalarA                               |
 *----------------------------------------------------------------------*/
void LINALG::Add(const Epetra_CrsMatrix& A,
                 const bool transposeA,
                 const double scalarA,
                 Epetra_CrsMatrix& B,
                 const double scalarB)
{
  if (!A.Filled()) dserror("FillComplete was not called on A");

  Epetra_CrsMatrix* Aprime = NULL;
  Teuchos::RCP<EpetraExt::RowMatrix_Transpose> Atrans = Teuchos::null;
  if (transposeA)
  {
    //Atrans = Teuchos::rcp(new EpetraExt::RowMatrix_Transpose(false,NULL,false));
    Atrans = Teuchos::rcp(new EpetraExt::RowMatrix_Transpose());
    Aprime = &(dynamic_cast<Epetra_CrsMatrix&>(((*Atrans)(const_cast<Epetra_CrsMatrix&>(A)))));
  }
  else
  {
    Aprime = const_cast<Epetra_CrsMatrix*>(&A);
  }

  if (scalarB != 1.0) B.Scale(scalarB);
  if (scalarB == 0.0) B.PutScalar(0.0);

  //Loop over Aprime's rows and sum into
  int MaxNumEntries = EPETRA_MAX( Aprime->MaxNumEntries(), B.MaxNumEntries() );
  int NumEntries;
  std::vector<int>    Indices(MaxNumEntries);
  std::vector<double> Values(MaxNumEntries);

  const int NumMyRows = Aprime->NumMyRows();
  int Row, err;
  if (scalarA)
  {
    for( int i = 0; i < NumMyRows; ++i )
    {
      Row = Aprime->GRID(i);
      int ierr = Aprime->ExtractGlobalRowCopy(Row,MaxNumEntries,NumEntries,&Values[0],&Indices[0]);
      if (ierr) dserror("Epetra_CrsMatrix::ExtractGlobalRowCopy returned err=%d",ierr);
      if( scalarA != 1.0 )
        for( int j = 0; j < NumEntries; ++j ) Values[j] *= scalarA;
      for (int j=0; j<NumEntries; ++j)
      {
        err = B.SumIntoGlobalValues(Row,1,&Values[j],&Indices[j]);
        if (err<0 || err==2)
          err = B.InsertGlobalValues(Row,1,&Values[j],&Indices[j]);
        if (err < 0)
          dserror("Epetra_CrsMatrix::InsertGlobalValues returned err=%d",err);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 | Transpose matrix A                                         popp 02/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_CrsMatrix> LINALG::Transpose(const Epetra_CrsMatrix& A)
{
  if (!A.Filled()) dserror("FillComplete was not called on A");

  Teuchos::RCP<EpetraExt::RowMatrix_Transpose> Atrans =
      Teuchos::rcp(new EpetraExt::RowMatrix_Transpose(/*false,NULL,false*/));
  Epetra_CrsMatrix* Aprime =
      &(dynamic_cast<Epetra_CrsMatrix&>(((*Atrans)(const_cast<Epetra_CrsMatrix&>(A)))));


  return Teuchos::rcp(new Epetra_CrsMatrix(*Aprime));
}

/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 01/06|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_CrsMatrix> LINALG::Multiply(const Epetra_CrsMatrix& A, bool transA,
                                       const Epetra_CrsMatrix& B, bool transB,
                                       bool complete)
{
  /* ATTENTION (Q1/2013 and later)
   *
   * Be careful with LINALG::Multiply when using Trilinos Q1/2013.
   * The new EpetraExt MMM routines are very fast, but not well tested and
   * sometimes crash. To be on the safe side consider to use LINALG::MLMultiply
   * which is based on ML and well tested over several years.
   *
   * For improving speed of MM operations (even beyond MLMultiply) we probably
   * should rethink the design of our Multiply routines. As a starting point we
   * should have a look at the new variants of the EpetraExt routines. See also
   * the following personal communication with Chris Siefert
   *
   * "There are basically 3 different matrix-matrix-multiplies in the new EpetraExt MMM:
   *
   * 1) Re-use existing C.  I think this works, but it isn't well tested.
   * 2) Start with a clean C and FillComplete.  This is the one we're using in MueLu that works fine.
   * 3) Any other case.  This is the one you found a bug in.
   *
   * I'll try to track the bug in #3 down, but you really should be calling #2 (or #1) if at all possible.
   *
   * -Chris"
   *
   */

  // make sure FillComplete was called on the matrices
  if (!A.Filled()) dserror("A has to be FillComplete");
  if (!B.Filled()) dserror("B has to be FillComplete");

  // do a very coarse guess of nonzeros per row (horrible memory consumption!)
     // int guessnpr = A.MaxNumEntries()*B.MaxNumEntries();
  // a first guess for the bandwidth of C leading to much less memory allocation:
  const int guessnpr = std::max(A.MaxNumEntries(),B.MaxNumEntries());

  // create resultmatrix with correct rowmap
  Epetra_CrsMatrix* C = NULL;
  if (!transA)
    C = new Epetra_CrsMatrix(Copy,A.OperatorRangeMap(),guessnpr,false);
  else
    C = new Epetra_CrsMatrix(Copy,A.OperatorDomainMap(),guessnpr,false);

  int err = EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*C,complete);
  if (err) dserror("EpetraExt::MatrixMatrix::Multiply returned err = &d",err);

  return Teuchos::rcp(C);
}



/*----------------------------------------------------------------------*
 | Multiply matrices A*B*C                                   mwgee 02/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_CrsMatrix> LINALG::Multiply(const Epetra_CrsMatrix& A, bool transA,
                                       const Epetra_CrsMatrix& B, bool transB,
                                       const Epetra_CrsMatrix& C, bool transC,
                                       bool complete)
{
  Teuchos::RCP<Epetra_CrsMatrix> tmp = LINALG::Multiply(B,transB,C,transC,true);
  return LINALG::Multiply(A,transA,*tmp,false,complete);
}

/*----------------------------------------------------------------------*
 |  invert a dense matrix  (public)                          mwgee 04/08|
 *----------------------------------------------------------------------*/
double LINALG::NonsymInverse3x3(Epetra_SerialDenseMatrix& A)
{
#ifdef DEBUG
  if (A.M() != A.N()) dserror("Matrix is not square");
  if (A.M() != 3) dserror("Dimension supplied is not 3: dim=%d",A.M());
#endif

  const double b00 = A(0,0);
  const double b01 = A(0,1);
  const double b02 = A(0,2);
  const double b10 = A(1,0);
  const double b11 = A(1,1);
  const double b12 = A(1,2);
  const double b20 = A(2,0);
  const double b21 = A(2,1);
  const double b22 = A(2,2);
  A(0,0) =   b11*b22 - b21*b12;
  A(1,0) = - b10*b22 + b20*b12;
  A(2,0) =   b10*b21 - b20*b11;
  A(0,1) = - b01*b22 + b21*b02;
  A(1,1) =   b00*b22 - b20*b02;
  A(2,1) = - b00*b21 + b20*b01;
  A(0,2) =   b01*b12 - b11*b02;
  A(1,2) = - b00*b12 + b10*b02;
  A(2,2) =   b00*b11 - b10*b01;
  const double det = b00*A(0,0)+b01*A(1,0)+b02*A(2,0);
  if (det==0.0) dserror("Determinant of 3x3 matrix is exactly zero");
  A.Scale(1./det);
  return det;
}

/*----------------------------------------------------------------------*
 |  (public)                                                 mwgee 05/08|
 *----------------------------------------------------------------------*/
double LINALG::DeterminantSVD(const Epetra_SerialDenseMatrix& A)
{
#ifdef DEBUG
  if (A.M() != A.N()) dserror("Matrix is not square");
#endif
  Epetra_SerialDenseMatrix tmp(A);
  Epetra_LAPACK lapack;
  const int n = tmp.N();
  const int m = tmp.M();
  std::vector<double> s(std::min(n,m));
  int info;
  int lwork = std::max(3*std::min(m,n)+std::max(m,n),5*std::min(m,n));
  std::vector<double> work(lwork);
  lapack.GESVD('N','N',m,n,tmp.A(),tmp.LDA(),&s[0],
               NULL,tmp.LDA(),NULL,tmp.LDA(),&work[0],&lwork,&info);
  if (info) dserror("Lapack's dgesvd returned %d",info);
  double d=s[0];
  for (int i=1; i<n; ++i) d *= s[i];
  return d;
}

/*----------------------------------------------------------------------*
 |  (public)                                                 mwgee 05/08|
 *----------------------------------------------------------------------*/
double LINALG::DeterminantLU(const Epetra_SerialDenseMatrix& A)
{
#ifdef DEBUG
  if (A.M() != A.N()) dserror("Matrix is not square");
#endif
  Epetra_SerialDenseMatrix tmp(A);
  Epetra_LAPACK lapack;
  const int n = tmp.N();
  const int m = tmp.M();
  std::vector<int> ipiv(n);
  int info;
  lapack.GETRF(m,n,tmp.A(),tmp.LDA(),&ipiv[0],&info);
  if (info<0) dserror("Lapack's dgetrf returned %d",info);
  else if (info>0) return 0.0;
  double d = tmp(0,0);
  for (int i=1; i<n; ++i) d *= tmp(i,i);
  // swapping rows of A changes the sign of the determinant, so we have to
  // undo lapack's permutation w.r.t. the determinant
  // note the fortran indexing convention in ipiv
  for (int i=0; i<n; ++i)
    if (ipiv[i]!=i+1) d *= -1.0;
  return d;
}


/*----------------------------------------------------------------------*
 |  invert a dense symmetric matrix  (public)                mwgee 12/06|
 *----------------------------------------------------------------------*/
void LINALG::SymmetricInverse(Epetra_SerialDenseMatrix& A, const int dim)
{
  if (A.M() != A.N()) dserror("Matrix is not square");
  if (A.M() != dim) dserror("Dimension supplied does not match matrix");

  double* a = A.A();
  char uplo[5]; strncpy(uplo,"L ",2);
  std::vector<int> ipiv(dim);
  int lwork = 10*dim;
  std::vector<double> work(lwork);
  int info=0;
  int n = dim;
  int m = dim;

  dsytrf(uplo,&m,a,&n,&(ipiv[0]),&(work[0]),&lwork,&info);
  if (info) dserror("dsytrf returned info=%d",info);

  dsytri(uplo,&m,a,&n,&(ipiv[0]),&(work[0]),&info);
  if (info) dserror("dsytri returned info=%d",info);

  for (int i=0; i<dim; ++i)
    for (int j=0; j<i; ++j)
      A(j,i)=A(i,j);
  return;
}


/*----------------------------------------------------------------------*
 |                                             (public)        gee 06/07|
 *----------------------------------------------------------------------*/
void LINALG::SymmetriseMatrix(Epetra_SerialDenseMatrix& A)
{
  const int n = A.N();
  if (n != A.M()) dserror("Cannot symmetrize non-square matrix");
  // do not make deep copy of A, matrix addition and full scaling just to sym it
  for (int i=0; i<n; ++i)
    for (int j=i+1; j<n; ++j)
    {
      const double aver = 0.5*(A(i,j)+A(j,i));
      A(i,j) = A(j,i) = aver;
    }
  return;
}




/*----------------------------------------------------------------------*
| invert a dense nonsymmetric matrix (public)       g.bau 03/07|
*----------------------------------------------------------------------*/
void LINALG::NonSymmetricInverse(Epetra_SerialDenseMatrix& A, const int dim)
{
  if (A.M() != A.N()) dserror("Matrix is not square");
  if (A.M() != dim) dserror("Dimension supplied does not match matrix");

  Epetra_SerialDenseSolver solver;
  solver.SetMatrix(A);
  int err = solver.Invert();
  if (err!=0)
    dserror("Inversion of nonsymmetric matrix failed.");

 return;
}

/*----------------------------------------------------------------------*
 |  compute all eigenvalues of a real symmetric matrix A        lw 04/08|
 *----------------------------------------------------------------------*/
void LINALG::SymmetricEigenValues(Epetra_SerialDenseMatrix& A,
                                  Epetra_SerialDenseVector& L,
                                  const bool postproc)
{
  LINALG::SymmetricEigen(A, L, 'N', postproc);
}

/*----------------------------------------------------------------------*
 |  compute all eigenvalues and eigenvectors of a real symmetric        |
 |  matrix A (eigenvectors are stored in A, i.e. original matrix        |
 |  is destroyed!!!)                                            lw 04/08|
 *----------------------------------------------------------------------*/
void LINALG::SymmetricEigenProblem(Epetra_SerialDenseMatrix& A,
                                   Epetra_SerialDenseVector& L,
                                   const bool postproc)
{
  LINALG::SymmetricEigen(A, L, 'V', postproc);
}

/*----------------------------------------------------------------------*
 |  compute all eigenvalues and, optionally,                            |
 |  eigenvectors of a real symmetric matrix A  (public)        maf 06/07|
 *----------------------------------------------------------------------*/
void LINALG::SymmetricEigen(Epetra_SerialDenseMatrix& A,
                            Epetra_SerialDenseVector& L,
                            const char jobz,
                            const bool postproc)
{
  if (A.M() != A.N()) dserror("Matrix is not square");
  if (A.M() != L.Length()) dserror("Dimension of eigenvalues does not match");

  double* a = A.A();
  double* w = L.A();
  const char uplo = {'U'};
  const int lda = A.LDA();
  const int dim = A.M();

  int liwork=0;
  if (dim == 1) liwork = 1;
  else
  {
    if      (jobz == 'N') liwork = 1;
    else if (jobz == 'V') liwork = 3+5*dim;
  }
  std::vector<int> iwork(liwork);

  int lwork = 0;
  if (dim == 1) lwork = 1;
  else
  {
    if      (jobz == 'N') lwork = 2*dim+1;
    else if (jobz == 'V') lwork = 2*dim*dim+6*dim+1;
  }
  std::vector<double> work(lwork);
  int info=0;

  Epetra_LAPACK lapack;

  lapack.SYEVD(jobz,uplo,dim,a,lda,w,&(work[0]),lwork,&(iwork[0]),liwork,&info);

  if (!postproc)
  {
    if (info > 0) dserror("Lapack algorithm syevd failed");
    if (info < 0) dserror("Illegal value in Lapack syevd call");
  }
  // if we only calculate eigenvalues/eigenvectors for postprocessing,
  // a warning might be sufficient
  else
  {
    if (info > 0) cout << "Lapack algorithm syevd failed: " << info << " off-diagonal elements of intermediate tridiagonal form did not converge to zero" << endl;
    if (info < 0) cout << "Illegal value in Lapack syevd call" << endl;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  compute all eigenvalues for the generalized eigenvalue problem
 |  Ax =  lambda Bx via QZ-algorithm (B is singular) and returns the
 |  maximum eigenvalue                              shahmiri  05/13
 *----------------------------------------------------------------------*/
int LINALG::GeneralizedEigen(Epetra_SerialDenseMatrix& A,
                       		 Epetra_SerialDenseMatrix& B)
{
	Epetra_SerialDenseMatrix tmpA(A);
	Epetra_SerialDenseMatrix tmpB(B);

	//--------------------------------------------------------------------
	// STEP 1:
	// Transform the matrix B to upper triangular matrix with the help of
	// QR-factorization
	//--------------------------------------------------------------------

	int N = tmpA.M();
	double* a = tmpA.A();
	double* b = tmpB.A();
	int LDA = tmpA.LDA();
	int LDB = tmpB.LDA();

	// the order of permutation matrix
	int jpvt[N];
	// factor uses for calculating orthogonal matrix Q
	double tau[N];
    for(int i=0; i<N; ++i)
    {
    	jpvt[i] = 0;
    	tau[i] = 0.;
    }


	int lwork1 = 0;
	if (N == 1) lwork1 = 1;
	else
	{
	  lwork1 = N*3+1;
	}
	double work1[lwork1];
	int info;
	dgeqp3(&N,&N,b,&LDB,jpvt,tau,work1,&lwork1,&info);

    if (info < 0)
    	cout << "Lapack algorithm dgeqp3: The " << info  << "-th argument had an illegal value" << endl;


	// calculate the matrix Q from multiplying householder transformations
	// Q = H(1)*H(2) ... H(k)
	// H(i) = I - tau-v*v**T
	// v is a vector with v(1:i-1) = 0 and v(i+1:m) is stored on exit in B(i+1:m,i)

	// Q is initialized as an unit matrix
	Epetra_SerialDenseMatrix Q_new(true);
	Q_new.Shape(N,N);
	for(int i=0; i<N; ++i)
		Q_new(i,i)=1.0;

	for(int i=0; i<N; ++i)
	{
		Epetra_SerialDenseVector v;
		v.Shape(N,1);
		v(i,0) = 1.;
		for(int j=i+1; j<N; ++j)
			v(j,0) = tmpB(j,i);

	    Epetra_SerialDenseMatrix H;
	    H.Shape(N,N);

	    H.Multiply('N','T',tau[i],v,v,0.);
	    H.Scale(-1.);
	    for(int k=0; k<N; ++k)
	    	H(k,k) = 1.+H(k,k);

	    Epetra_SerialDenseMatrix Q_help;
	    Q_help.Shape(N,N);
	    Q_new.Apply(H,Q_help);
	    Q_new = Q_help;
	}

	// permutation matrix
	Epetra_SerialDenseMatrix P(true);
	P.Shape(N,N);
	for(int i=0; i<N; ++i)
	{
		int w = jpvt[i];
		P(w-1,i) = 1.;
	}

	// annul the under-diagonal elements of B
    // loop of columns
    for(int i=0; i<N; ++i)
    {
   	 // loop of rows
   	 for(int j=i+1; j<N; ++j)
   		 tmpB(j,i)=0.;
    }

	// the new A:= Q**T A P
	Epetra_SerialDenseMatrix A_tmp;
	A_tmp.Shape(N,N);
    //A_tt.Multiply('T','N',1.,Q_qr_tt,A,0.);
	A_tmp.Multiply('T','N',1.,Q_new,tmpA,0.);

    Epetra_SerialDenseMatrix A_new;
	A_new.Shape(N,N);
    A_new.Multiply('N','N',1.,A_tmp,P,0.);

    a = A_new.A();

	//--------------------------------------------------------
    // STEP 2
    // transform A to a upper hessenberg matrix and keep B as
    // an upper diagonal matrix
    //--------------------------------------------------------

	char job = {'S'};
	char COMPQ = {'V'};
	char COMPZ = {'V'};

	int ILO = 1;
	int IHI = N;

	int lwork = 0;
	if (N == 1) lwork = 1;
	else
	{
	  lwork = N;
	}
	double work[lwork];

	Epetra_SerialDenseMatrix A1(true);
	Epetra_SerialDenseMatrix A2(true);
	A1.Shape(N,N);
	A2.Shape(N,N);
	double* Q = A1.A();
	int LDQ = A1.LDA();
	double* Z = A2.A();
	int LDZ = A2.LDA();

	dgghrd(&COMPQ,&COMPZ,&N,&ILO,&IHI,a,&LDA,b,&LDB,Q,&LDQ,Z,&LDZ,&info);

    if (info < 0)
    	cout << "Lapack algorithm dgghrd: The " << info  << "-th argument had an illegal value" << endl;

	//--------------------------------------------------------
    // STEP 3
    // transform A which is an upper hessenberg matrix to an upper
	// diagonal matrix and keep B an upper diagonal matrix via a
	// QZ-transformation
    //--------------------------------------------------------

	// vectors which contain the eigenvalues of the problem
	Epetra_SerialDenseVector L1(true);
	Epetra_SerialDenseVector L2(true);
	Epetra_SerialDenseVector L3(true);
	L1.Shape(N,1);
	L2.Shape(N,1);
	L3.Shape(N,1);
	double* ALPHAR = L1.A();
	double* ALPHAI = L2.A();
	double* BETA = L3.A();

	int LDH = A_new.LDA();
	int LDT = B.LDA();

	char COMPQ2 = {'I'};
	char COMPZ2 = {'I'};

	Epetra_SerialDenseMatrix Q_2(true);
	Q_2.Shape(N,N);
	Epetra_SerialDenseMatrix Z_2(true);
	Z_2.Shape(N,N);
    double* q_2 = Q_2.A();
    double* z_2 = Z_2.A();

	dhgeqz(&job,&COMPQ2,&COMPZ2,&N,&ILO,&IHI,a,&LDH,b,&LDT,ALPHAR,ALPHAI,BETA,q_2,&LDQ,z_2,&LDZ,work,&lwork,&info);

    if (info < 0)
    	cout << "Lapack algorithm dhgeqz: The "<< info << "-th argument haa an illegal value!" << endl;
    else if(info > N)
    	cout << "Lapack algorithm dhgeqz: The QZ iteration did not converge. (H,T) is not in Schur Form, but the Eigenvalues should be correct!" << endl;

/*	cout << "--------Final----------" << endl;
	cout << std::setprecision(16) << "A 2" << A_new << endl;
	cout << std::setprecision(16) <<  "B 2" << tmpB << endl;
	cout << std::setprecision(16) << "Q 2 " << Q_2 << endl;
	cout << std::setprecision(16) << "Z 2 " << Z_2 << endl;*/


	double maxlambda = 0.;
	for (int i=0; i<N; ++i)
	{
		if (BETA[i] > 1e-13)
		{
			// Eigenvalues:
			// cout << "lambda " << i << ":  " <<  ALPHAR[i]/BETA[i] << endl;
			maxlambda = max(ALPHAR[i]/BETA[i],maxlambda);
		}
		if ( ALPHAI[i] > 1e-12 )
		{
			cout << " Warning: you have an imaginary EW " << ALPHAI[i] << endl;
		}
	}
	return maxlambda;
}


/*----------------------------------------------------------------------*
 |  singular value decomposition (SVD) of a real M-by-N matrix A.       |
 |  Wrapper for Lapack/Epetra_Lapack           (public)        maf 05/08|
 *----------------------------------------------------------------------*/
void LINALG::SVD(const Epetra_SerialDenseMatrix& A,
                 LINALG::SerialDenseMatrix& Q,
                 LINALG::SerialDenseMatrix& S,
                 LINALG::SerialDenseMatrix& VT)
{
  Epetra_SerialDenseMatrix tmp(A);  // copy, because content of A ist destroyed
  Epetra_LAPACK lapack;
  const char jobu = 'A';  // compute and return all M columns of U
  const char jobvt = 'A'; // compute and return all N rows of V^T
  const int n = tmp.N();
  const int m = tmp.M();
  std::vector<double> s(std::min(n,m));
  int info;
  int lwork = std::max(3*std::min(m,n)+std::max(m,n),5*std::min(m,n));
  std::vector<double> work(lwork);

  lapack.GESVD(jobu,jobvt,m,n,tmp.A(),tmp.LDA(),&s[0],
               Q.A(),Q.LDA(),VT.A(),VT.LDA(),&work[0],&lwork,&info);

  if (info) dserror("Lapack's dgesvd returned %d",info);

  for (int i = 0; i < std::min(n,m); ++i) {
    for (int j = 0; j < std::min(n,m); ++j) {
      S(i,j) = (i==j) * s[i];   // 0 for off-diagonal, otherwise s
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 |  Apply dirichlet conditions  (public)                     mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::ApplyDirichlettoSystem(Teuchos::RCP<Epetra_Vector>&            x,
                                    Teuchos::RCP<Epetra_Vector>&            b,
                                    const Teuchos::RCP<const Epetra_Vector> dbcval,
                                    const Teuchos::RCP<const Epetra_Vector> dbctoggle)
{
  const Epetra_Vector& dbct = *dbctoggle;
  if (x != Teuchos::null && b != Teuchos::null)
  {
    Epetra_Vector&       X    = *x;
    Epetra_Vector&       B    = *b;
    const Epetra_Vector& dbcv = *dbcval;
    // set the prescribed value in x and b
    const int mylength = dbcv.MyLength();
    for (int i=0; i<mylength; ++i)
      if (dbct[i]==1.0)
      {
        X[i] = dbcv[i];
        B[i] = dbcv[i];
      }
  }
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::ApplyDirichlettoSystem(Teuchos::RCP<Epetra_Vector>&            x,
                                    Teuchos::RCP<Epetra_Vector>&            b,
                                    const Teuchos::RCP<const Epetra_Vector> dbcval,
                                    const Epetra_Map&              dbcmap)
{
  if (not dbcmap.UniqueGIDs())
    dserror("unique map required");

  if (x != Teuchos::null and b != Teuchos::null)
  {
    Epetra_Vector&       X    = *x;
    Epetra_Vector&       B    = *b;
    const Epetra_Vector& dbcv = *dbcval;

    // We use two maps since we want to allow dbcv and X to be independent of
    // each other. So we are slow and flexible...
    const Epetra_BlockMap& xmap = X.Map();
    const Epetra_BlockMap& dbcvmap = dbcv.Map();

    const int mylength = dbcmap.NumMyElements();
    const int* mygids  = dbcmap.MyGlobalElements();
    for (int i=0; i<mylength; ++i)
    {
      int gid = mygids[i];

      int dbcvlid = dbcvmap.LID(gid);
      if (dbcvlid<0)
        dserror("illegal Dirichlet map");

      int xlid = xmap.LID(gid);
      if (xlid<0)
        dserror("illegal Dirichlet map");

      X[xlid] = dbcv[dbcvlid];
      B[xlid] = dbcv[dbcvlid];
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::ApplyDirichlettoSystem(Teuchos::RCP<Epetra_Vector>&            b,
                                    const Teuchos::RCP<const Epetra_Vector> dbcval,
                                    const Epetra_Map&              dbcmap)
{
  if (not dbcmap.UniqueGIDs())
    dserror("unique map required");

  if (b != Teuchos::null)
  {
    Epetra_Vector&       B    = *b;
    const Epetra_Vector& dbcv = *dbcval;

    // We use two maps since we want to allow dbcv and X to be independent of
    // each other. So we are slow and flexible...
    const Epetra_BlockMap& bmap = B.Map();
    const Epetra_BlockMap& dbcvmap = dbcv.Map();

    const int mylength = dbcmap.NumMyElements();
    const int* mygids  = dbcmap.MyGlobalElements();
    for (int i=0; i<mylength; ++i)
    {
      int gid = mygids[i];

      int dbcvlid = dbcvmap.LID(gid);

      int blid = bmap.LID(gid);
      // Note:
      // if gid is not found in vector b, just continue
      // b might only be a subset of a larger field vector
      if (blid>=0)
      {
        if (dbcvlid<0)
          dserror("illegal Dirichlet map");
        else
          B[blid] = dbcv[dbcvlid];
      }
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::ApplyDirichlettoSystem(Teuchos::RCP<LINALG::SparseOperator>       A,
                                    Teuchos::RCP<Epetra_Vector>&               x,
                                    Teuchos::RCP<Epetra_Vector>&               b,
                                    const Teuchos::RCP<const Epetra_Vector>    dbcval,
                                    const Teuchos::RCP<const Epetra_Vector>    dbctoggle)
{
  A->ApplyDirichlet(dbctoggle);
  ApplyDirichlettoSystem(x,b,dbcval,dbctoggle);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::ApplyDirichlettoSystem(Teuchos::RCP<LINALG::SparseOperator>       A,
                                    Teuchos::RCP<Epetra_Vector>&               x,
                                    Teuchos::RCP<Epetra_Vector>&               b,
                                    const Teuchos::RCP<const Epetra_Vector>&   dbcval,
                                    const Epetra_Map&                 dbcmap)
{
  A->ApplyDirichlet(dbcmap);
  ApplyDirichlettoSystem(x,b,dbcval,dbcmap);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void LINALG::ApplyDirichlettoSystem(Teuchos::RCP<LINALG::SparseOperator>       A,
                                    Teuchos::RCP<Epetra_Vector>&               x,
                                    Teuchos::RCP<Epetra_Vector>&               b,
                                    Teuchos::RCP<const LINALG::SparseMatrix>   trafo,
                                    const Teuchos::RCP<const Epetra_Vector>&   dbcval,
                                    const Epetra_Map&                 dbcmap)
{
  if (trafo != Teuchos::null)
    Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(A)->ApplyDirichletWithTrafo(trafo,dbcmap);
  else  // trafo==Teuchos::null
    A->ApplyDirichlet(dbcmap);
  ApplyDirichlettoSystem(x,b,dbcval,dbcmap);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::MapExtractor> LINALG::ConvertDirichletToggleVectorToMaps(
  const Teuchos::RCP<const Epetra_Vector>& dbctoggle)
{
  const Epetra_BlockMap& fullblockmap = dbctoggle->Map();
  // this copy is needed because the constructor of LINALG::MapExtractor
  // accepts only Epetra_Map and not Epetra_BlockMap
  const Epetra_Map fullmap = Epetra_Map(fullblockmap.NumGlobalElements(),
                                        fullblockmap.NumMyElements(),
                                        fullblockmap.MyGlobalElements(),
                                        fullblockmap.IndexBase(),
                                        fullblockmap.Comm());
  const int mylength = dbctoggle->MyLength();
  const int* fullgids = fullmap.MyGlobalElements();
  // build sets containing the DBC or free global IDs, respectively
  std::vector<int> dbcgids;
  std::vector<int> freegids;
  for (int i=0; i<mylength; ++i)
  {
    const int gid = fullgids[i];
    const int compo = (int) round((*dbctoggle)[i]);
    if (compo == 0)
      freegids.push_back(gid);
    else if (compo == 1)
      dbcgids.push_back(gid);
    else
      dserror("Unexpected component %f. It is neither 1.0 nor 0.0.", (*dbctoggle)[i]);
  }
  // build map of Dirichlet DOFs
  Teuchos::RCP<Epetra_Map> dbcmap = Teuchos::null;
  {
    int nummyelements = 0;
    int* myglobalelements = NULL;
    if (dbcgids.size() > 0)
    {
      nummyelements = dbcgids.size();
      myglobalelements = &(dbcgids[0]);
    }
    dbcmap = Teuchos::rcp(new Epetra_Map(-1, nummyelements, myglobalelements, fullmap.IndexBase(), fullmap.Comm()));
  }
  // build map of free DOFs
  Teuchos::RCP<Epetra_Map> freemap = Teuchos::null;
  {
    int nummyelements = 0;
    int* myglobalelements = NULL;
    if (freegids.size() > 0)
    {
      nummyelements = freegids.size();
      myglobalelements = &(freegids[0]);
    }
    freemap = Teuchos::rcp(new Epetra_Map(-1, nummyelements, myglobalelements, fullmap.IndexBase(), fullmap.Comm()));
  }

  // build and return the map extractor of Dirichlet-conditioned and free DOFs
  return Teuchos::rcp(new LINALG::MapExtractor(fullmap, dbcmap, freemap));
}

#if 0 // old version
/*----------------------------------------------------------------------*
 | split matrix into 2x2 block system                              06/06|
 | this version is to go away soon! mgee                                |
 *----------------------------------------------------------------------*/
bool LINALG::SplitMatrix2x2(Teuchos::RCP<Epetra_CrsMatrix> A,
                            Teuchos::RCP<Epetra_Map>& A11rowmap,
                            Teuchos::RCP<Epetra_Map>& A22rowmap,
                            Teuchos::RCP<Epetra_CrsMatrix>& A11,
                            Teuchos::RCP<Epetra_CrsMatrix>& A12,
                            Teuchos::RCP<Epetra_CrsMatrix>& A21,
                            Teuchos::RCP<Epetra_CrsMatrix>& A22)
{
  if (A==Teuchos::null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  if (A11rowmap==null && A22rowmap != Teuchos::null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != Teuchos::null && A22rowmap != Teuchos::null);
  else if (A11rowmap != Teuchos::null && A22rowmap == Teuchos::null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else
  	dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  const Epetra_Comm& Comm   = A->Comm();
  const Epetra_Map&  A22map = *(A22rowmap.get());
  const Epetra_Map&  A11map = *(A11rowmap.get());

  //----------------------------- create a parallel redundant map of A22map
  std::map<int,int> a22gmap;
  {
    std::vector<int> a22global(A22map.NumGlobalElements());
    int count=0;
    for (int proc=0; proc<Comm.NumProc(); ++proc)
    {
      int length = 0;
      if (proc==Comm.MyPID())
      {
        for (int i=0; i<A22map.NumMyElements(); ++i)
        {
          a22global[count+length] = A22map.GID(i);
          ++length;
        }
      }
      Comm.Broadcast(&length,1,proc);
      Comm.Broadcast(&a22global[count],length,proc);
      count += length;
    }
    if (count != A22map.NumGlobalElements())
    	dserror("LINALG::SplitMatrix2x2: mismatch in dimensions");

    // create the map
    for (int i=0; i<count; ++i)
      a22gmap[a22global[i]] = 1;
    a22global.clear();
  }

  //--------------------------------------------------- create matrix A22
  A22 = Teuchos::rcp(new Epetra_CrsMatrix(Copy,A22map,100));
  {
    std::vector<int>    a22gcindices(100);
    std::vector<double> a22values(100);
    for (int i=0; i<A->NumMyRows(); ++i)
    {
      const int grid = A->GRID(i);
      if (A22map.MyGID(grid)==false)
        continue;
      //cout << "Row " << grid << " in A22 Columns ";
      int     numentries;
      double* values;
      int*    cindices;
      int err = A->ExtractMyRowView(i,numentries,values,cindices);
      if (err)
      	dserror("LINALG::SplitMatrix2x2: A->ExtractMyRowView returned &i",err);

      if (numentries>(int)a22gcindices.size())
      {
        a22gcindices.resize(numentries);
        a22values.resize(numentries);
      }
      int count=0;
      for (int j=0; j<numentries; ++j)
      {
        const int gcid = A->ColMap().GID(cindices[j]);
        // see whether we have gcid in a22gmap
        std::map<int,int>::iterator curr = a22gmap.find(gcid);
        if (curr==a22gmap.end()) continue;
        //cout << gcid << " ";
        a22gcindices[count] = gcid;
        a22values[count]    = values[j];
        ++count;
      }
      //cout << endl; fflush(stdout);
      // add this filtered row to A22
      err = A22->InsertGlobalValues(grid,count,&a22values[0],&a22gcindices[0]);
      if (err<0)
      	dserror("LINALG::SplitMatrix2x2: A->InsertGlobalValues returned &i",err);

    } //for (int i=0; i<A->NumMyRows(); ++i)
    a22gcindices.clear();
    a22values.clear();
  }
  A22->FillComplete();
  A22->OptimizeStorage();

  //----------------------------------------------------- create matrix A11
  A11 = Teuchos::rcp(new Epetra_CrsMatrix(Copy,A11map,100));
  {
    std::vector<int>    a11gcindices(100);
    std::vector<double> a11values(100);
    for (int i=0; i<A->NumMyRows(); ++i)
    {
      const int grid = A->GRID(i);
      if (A11map.MyGID(grid)==false) continue;
      int     numentries;
      double* values;
      int*    cindices;
      int err = A->ExtractMyRowView(i,numentries,values,cindices);
      if (err)
      	dserror("LINALG::SplitMatrix2x2: A->ExtractMyRowView returned &i",err);

      if (numentries>(int)a11gcindices.size())
      {
        a11gcindices.resize(numentries);
        a11values.resize(numentries);
      }
      int count=0;
      for (int j=0; j<numentries; ++j)
      {
        const int gcid = A->ColMap().GID(cindices[j]);
        // see whether we have gcid as part of a22gmap
        std::map<int,int>::iterator curr = a22gmap.find(gcid);
        if (curr!=a22gmap.end()) continue;
        a11gcindices[count] = gcid;
        a11values[count] = values[j];
        ++count;
      }
      err = A11->InsertGlobalValues(grid,count,&a11values[0],&a11gcindices[0]);
      if (err<0)
      	dserror("LINALG::SplitMatrix2x2: A->InsertGlobalValues returned &i",err);

    } // for (int i=0; i<A->NumMyRows(); ++i)
    a11gcindices.clear();
    a11values.clear();
  }
  A11->FillComplete();
  A11->OptimizeStorage();

  //---------------------------------------------------- create matrix A12
  A12 = Teuchos::rcp(new Epetra_CrsMatrix(Copy,A11map,100));
  {
    std::vector<int>    a12gcindices(100);
    std::vector<double> a12values(100);
    for (int i=0; i<A->NumMyRows(); ++i)
    {
      const int grid = A->GRID(i);
      if (A11map.MyGID(grid)==false) continue;
      int     numentries;
      double* values;
      int*    cindices;
      int err = A->ExtractMyRowView(i,numentries,values,cindices);
      if (err)
      	dserror("LINALG::SplitMatrix2x2: A->ExtractMyRowView returned &i",err);

      if (numentries>(int)a12gcindices.size())
      {
        a12gcindices.resize(numentries);
        a12values.resize(numentries);
      }
      int count=0;
      for (int j=0; j<numentries; ++j)
      {
        const int gcid = A->ColMap().GID(cindices[j]);
        // see whether we have gcid as part of a22gmap
        std::map<int,int>::iterator curr = a22gmap.find(gcid);
        if (curr==a22gmap.end()) continue;
        a12gcindices[count] = gcid;
        a12values[count] = values[j];
        ++count;
      }
      err = A12->InsertGlobalValues(grid,count,&a12values[0],&a12gcindices[0]);
      if (err<0)
      	dserror("LINALG::SplitMatrix2x2: A->InsertGlobalValues returned &i",err);

    } // for (int i=0; i<A->NumMyRows(); ++i)
    a12values.clear();
    a12gcindices.clear();
  }
  A12->FillComplete(A22map,A11map);
  A12->OptimizeStorage();

  //----------------------------------------------------------- create A21
  A21 = Teuchos::rcp(new Epetra_CrsMatrix(Copy,A22map,100));
  {
    std::vector<int>    a21gcindices(100);
    std::vector<double> a21values(100);
    for (int i=0; i<A->NumMyRows(); ++i)
    {
      const int grid = A->GRID(i);
      if (A22map.MyGID(grid)==false) continue;
      int     numentries;
      double* values;
      int*    cindices;
      int err = A->ExtractMyRowView(i,numentries,values,cindices);
      if (err)
      	dserror("LINALG::SplitMatrix2x2: A->ExtractMyRowView returned &i",err);

      if (numentries>(int)a21gcindices.size())
      {
        a21gcindices.resize(numentries);
        a21values.resize(numentries);
      }
      int count=0;
      for (int j=0; j<numentries; ++j)
      {
        const int gcid = A->ColMap().GID(cindices[j]);
        // see whether we have gcid as part of a22gmap
        std::map<int,int>::iterator curr = a22gmap.find(gcid);
        if (curr!=a22gmap.end()) continue;
        a21gcindices[count] = gcid;
        a21values[count] = values[j];
        ++count;
      }
      err = A21->InsertGlobalValues(grid,count,&a21values[0],&a21gcindices[0]);
      if (err<0)
      	dserror("LINALG::SplitMatrix2x2: A->InsertGlobalValues returned &i",err);

    } // for (int i=0; i<A->NumMyRows(); ++i)
    a21values.clear();
    a21gcindices.clear();
  }
  A21->FillComplete(A11map,A22map);
  A21->OptimizeStorage();

  //-------------------------------------------------------------- tidy up
  a22gmap.clear();
  return true;
}
#else
/*----------------------------------------------------------------------*
 | split matrix into 2x2 block system                              06/06|
 | this version is to go away soon! mgee                                |
 *----------------------------------------------------------------------*/
bool LINALG::SplitMatrix2x2(Teuchos::RCP<Epetra_CrsMatrix> A,
                            Teuchos::RCP<Epetra_Map>& A11rowmap,
                            Teuchos::RCP<Epetra_Map>& A22rowmap,
                            Teuchos::RCP<Epetra_CrsMatrix>& A11,
                            Teuchos::RCP<Epetra_CrsMatrix>& A12,
                            Teuchos::RCP<Epetra_CrsMatrix>& A21,
                            Teuchos::RCP<Epetra_CrsMatrix>& A22)
{
  if (A==Teuchos::null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  if (A11rowmap==Teuchos::null && A22rowmap != Teuchos::null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != Teuchos::null && A22rowmap == Teuchos::null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else if (A11rowmap == Teuchos::null && A22rowmap == Teuchos::null)
    dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  std::vector<Teuchos::RCP<const Epetra_Map> > maps(2);
  maps[0] = Teuchos::rcp(new Epetra_Map(*A11rowmap));
  maps[1] = Teuchos::rcp(new Epetra_Map(*A22rowmap));
  LINALG::MultiMapExtractor extractor(A->RowMap(),maps);

  // create SparseMatrix view to input matrix A
  SparseMatrix a(A,View);

  // split matrix into pieces, where main diagonal blocks are square
  Teuchos::RCP<BlockSparseMatrix<DefaultBlockMatrixStrategy> > Ablock =
                       a.Split<DefaultBlockMatrixStrategy>(extractor,extractor);
  Ablock->Complete();

  // get Epetra objects out of the block matrix (prevents them from dying)
  A11 = (*Ablock)(0,0).EpetraMatrix();
  A12 = (*Ablock)(0,1).EpetraMatrix();
  A21 = (*Ablock)(1,0).EpetraMatrix();
  A22 = (*Ablock)(1,1).EpetraMatrix();

  return true;
}
#endif


#if 0 // old version
/*----------------------------------------------------------------------*
 | split matrix into 2x2 block system                         popp 02/08|
 *----------------------------------------------------------------------*/
bool LINALG::SplitMatrix2x2(Teuchos::RCP<LINALG::SparseMatrix> A,
                            Teuchos::RCP<Epetra_Map>& A11rowmap,
                            Teuchos::RCP<Epetra_Map>& A22rowmap,
                            Teuchos::RCP<Epetra_Map>& A11domainmap,
                            Teuchos::RCP<Epetra_Map>& A22domainmap,
                            Teuchos::RCP<LINALG::SparseMatrix>& A11,
                            Teuchos::RCP<LINALG::SparseMatrix>& A12,
                            Teuchos::RCP<LINALG::SparseMatrix>& A21,
                            Teuchos::RCP<LINALG::SparseMatrix>& A22)
{
  if (A==Teuchos::null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  // check and complete input row maps
  if (A11rowmap==null && A22rowmap != Teuchos::null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != Teuchos::null && A22rowmap != Teuchos::null);
  else if (A11rowmap != Teuchos::null && A22rowmap == Teuchos::null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else
    dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  // check and complete input domain maps
  if (A11domainmap==null && A22domainmap != Teuchos::null)
  	A11domainmap = LINALG::SplitMap(A->DomainMap(),*A22domainmap);
  else if (A11domainmap != Teuchos::null && A22domainmap != Teuchos::null);
  else if (A11domainmap != Teuchos::null && A22domainmap == Teuchos::null)
    A22domainmap = LINALG::SplitMap(A->DomainMap(),*A11domainmap);
  else
    dserror("LINALG::SplitMatrix2x2: Both A11domainmap and A22domainmap == null on entry");

  // local variables
  const Epetra_Comm& Comm   = A->Comm();
  const Epetra_Map&  A11rmap = *(A11rowmap.get());
  const Epetra_Map&  A11dmap = *(A11domainmap.get());
  const Epetra_Map&  A22rmap = *(A22rowmap.get());
  const Epetra_Map&  A22dmap = *(A22domainmap.get());

  //----------------------------- create a parallel redundant map of A11domainmap
  std::map<int,int> a11gmap;
  {
    std::vector<int> a11global(A11dmap.NumGlobalElements());
    int count=0;
    for (int proc=0; proc<Comm.NumProc(); ++proc)
    {
      int length = 0;
      if (proc==Comm.MyPID())
      {
        for (int i=0; i<A11dmap.NumMyElements(); ++i)
        {
          a11global[count+length] = A11dmap.GID(i);
          ++length;
        }
      }
      Comm.Broadcast(&length,1,proc);
      Comm.Broadcast(&a11global[count],length,proc);
      count += length;
    }
    if (count != A11dmap.NumGlobalElements())
    	dserror("LINALG::SplitMatrix2x2: mismatch in dimensions");

    // create the map
    for (int i=0; i<count; ++i)
      a11gmap[a11global[i]] = 1;
    a11global.clear();
  }

  //----------------------------------------------------- create matrix A11
  if (A11rmap.NumGlobalElements()>0 && A11dmap.NumGlobalElements()>0)
  {
    A11 = Teuchos::rcp(new LINALG::SparseMatrix(A11rmap,100));
    {
      std::vector<int>    a11gcindices(100);
      std::vector<double> a11values(100);
      for (int i=0; i<A->EpetraMatrix()->NumMyRows(); ++i)
      {
        const int grid = A->EpetraMatrix()->GRID(i);
        if (A11rmap.MyGID(grid)==false) continue;
        int     numentries;
        double* values;
        int*    cindices;
        int err = A->EpetraMatrix()->ExtractMyRowView(i,numentries,values,cindices);
        if (err)
          dserror("LINALG::Split2x2: A->ExtractMyRowView returned %i",err);

        if (numentries>(int)a11gcindices.size())
        {
          a11gcindices.resize(numentries);
          a11values.resize(numentries);
        }
        int count=0;
        for (int j=0; j<numentries; ++j)
        {
          const int gcid = A->ColMap().GID(cindices[j]);
          // see whether we have gcid as part of a11gmap
          std::map<int,int>::iterator curr = a11gmap.find(gcid);
          if (curr==a11gmap.end()) continue;
          a11gcindices[count] = gcid;
          a11values[count] = values[j];
          ++count;
        }
        err = A11->EpetraMatrix()->InsertGlobalValues(grid,count,&a11values[0],&a11gcindices[0]);
        if (err<0)
          dserror("LINALG::Split2x2: A->InsertGlobalValues returned %i",err);

      } // for (int i=0; i<A->NumMyRows(); ++i)
      a11gcindices.clear();
      a11values.clear();
    }
    A11->Complete(A11dmap,A11rmap);
  }

  //--------------------------------------------------- create matrix A22
  if (A22rmap.NumGlobalElements()>0 && A22dmap.NumGlobalElements()>0)
  {
    A22 = Teuchos::rcp(new LINALG::SparseMatrix(A22rmap,100));
    {
      std::vector<int>    a22gcindices(100);
      std::vector<double> a22values(100);
      for (int i=0; i<A->EpetraMatrix()->NumMyRows(); ++i)
      {
        const int grid = A->EpetraMatrix()->GRID(i);
        if (A22rmap.MyGID(grid)==false) continue;
        int     numentries;
        double* values;
        int*    cindices;
        int err = A->EpetraMatrix()->ExtractMyRowView(i,numentries,values,cindices);
        if (err)
          dserror("LINALG::Split2x2: A->ExtractMyRowView returned %i",err);

        if (numentries>(int)a22gcindices.size())
        {
          a22gcindices.resize(numentries);
          a22values.resize(numentries);
        }
        int count=0;
        for (int j=0; j<numentries; ++j)
        {
          const int gcid = A->ColMap().GID(cindices[j]);
          // see whether we have gcid as part of a11gmap
          std::map<int,int>::iterator curr = a11gmap.find(gcid);
          if (curr!=a11gmap.end()) continue;
          a22gcindices[count] = gcid;
          a22values[count]    = values[j];
          ++count;
        }
        err = A22->EpetraMatrix()->InsertGlobalValues(grid,count,&a22values[0],&a22gcindices[0]);
        if (err<0)
          dserror("LINALG::Split2x2: A->InsertGlobalValues returned %i",err);

      } //for (int i=0; i<A->NumMyRows(); ++i)
      a22gcindices.clear();
      a22values.clear();
    }
    A22->Complete(A22dmap,A22rmap);
  }

  //---------------------------------------------------- create matrix A12
  if (A11rmap.NumGlobalElements()>0 && A22dmap.NumGlobalElements()>0)
  {
    A12 = Teuchos::rcp(new LINALG::SparseMatrix(A11rmap,100));
    {
      std::vector<int>    a12gcindices(100);
      std::vector<double> a12values(100);
      for (int i=0; i<A->EpetraMatrix()->NumMyRows(); ++i)
      {
        const int grid = A->EpetraMatrix()->GRID(i);
        if (A11rmap.MyGID(grid)==false) continue;
        int     numentries;
        double* values;
        int*    cindices;
        int err = A->EpetraMatrix()->ExtractMyRowView(i,numentries,values,cindices);
        if (err)
          dserror("LINALG::Split2x2: A->ExtractMyRowView returned %i",err);

        if (numentries>(int)a12gcindices.size())
        {
          a12gcindices.resize(numentries);
          a12values.resize(numentries);
        }
        int count=0;
        for (int j=0; j<numentries; ++j)
        {
          const int gcid = A->ColMap().GID(cindices[j]);
          // see whether we have gcid as part of a11gmap
          std::map<int,int>::iterator curr = a11gmap.find(gcid);
          if (curr!=a11gmap.end()) continue;
          a12gcindices[count] = gcid;
          a12values[count] = values[j];
          ++count;
        }
        err = A12->EpetraMatrix()->InsertGlobalValues(grid,count,&a12values[0],&a12gcindices[0]);
        if (err<0)
          dserror("LINALG::Split2x2: A->InsertGlobalValues returned %i",err);

      } // for (int i=0; i<A->NumMyRows(); ++i)
      a12values.clear();
      a12gcindices.clear();
    }
    A12->Complete(A22dmap,A11rmap);
  }

  //---------------------------------------------------- create matrix A21
  if (A22rmap.NumGlobalElements()>0 && A11dmap.NumGlobalElements()>0)
  {
    A21 = Teuchos::rcp(new LINALG::SparseMatrix(A22rmap,100));
    {
      std::vector<int>    a21gcindices(100);
      std::vector<double> a21values(100);
      for (int i=0; i<A->EpetraMatrix()->NumMyRows(); ++i)
      {
        const int grid = A->EpetraMatrix()->GRID(i);
        if (A22rmap.MyGID(grid)==false) continue;
        int     numentries;
        double* values;
        int*    cindices;
        int err = A->EpetraMatrix()->ExtractMyRowView(i,numentries,values,cindices);
        if (err)
          dserror("LINALG::Split2x2: A->ExtractMyRowView returned %i",err);

        if (numentries>(int)a21gcindices.size())
        {
          a21gcindices.resize(numentries);
          a21values.resize(numentries);
        }
        int count=0;
        for (int j=0; j<numentries; ++j)
        {
          const int gcid = A->ColMap().GID(cindices[j]);
          // see whether we have gcid as part of a11gmap
          std::map<int,int>::iterator curr = a11gmap.find(gcid);
          if (curr==a11gmap.end()) continue;
          a21gcindices[count] = gcid;
          a21values[count] = values[j];
          ++count;
        }
        err = A21->EpetraMatrix()->InsertGlobalValues(grid,count,&a21values[0],&a21gcindices[0]);
        if (err<0)
          dserror("LINALG::Split2x2: A->InsertGlobalValues returned %i",err);

      } // for (int i=0; i<A->NumMyRows(); ++i)
      a21values.clear();
      a21gcindices.clear();
    }
    A21->Complete(A11dmap,A22rmap);
  }

  //-------------------------------------------------------------- tidy up
  a11gmap.clear();
  return true;
}
#else
/*----------------------------------------------------------------------*
 | split matrix into 2x2 block system                          gee 02/08|
 | new valid version                                                    |
 *----------------------------------------------------------------------*/
bool LINALG::SplitMatrix2x2(Teuchos::RCP<LINALG::SparseMatrix> A,
                            Teuchos::RCP<Epetra_Map>& A11rowmap,
                            Teuchos::RCP<Epetra_Map>& A22rowmap,
                            Teuchos::RCP<Epetra_Map>& A11domainmap,
                            Teuchos::RCP<Epetra_Map>& A22domainmap,
                            Teuchos::RCP<LINALG::SparseMatrix>& A11,
                            Teuchos::RCP<LINALG::SparseMatrix>& A12,
                            Teuchos::RCP<LINALG::SparseMatrix>& A21,
                            Teuchos::RCP<LINALG::SparseMatrix>& A22)
{
  if (A==Teuchos::null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  // check and complete input row maps
  if (A11rowmap==Teuchos::null && A22rowmap != Teuchos::null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != Teuchos::null && A22rowmap == Teuchos::null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else if (A11rowmap == Teuchos::null && A22rowmap == Teuchos::null)
    dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  // check and complete input domain maps
  if (A11domainmap==Teuchos::null && A22domainmap != Teuchos::null)
  	A11domainmap = LINALG::SplitMap(A->DomainMap(),*A22domainmap);
  else if (A11domainmap != Teuchos::null && A22domainmap == Teuchos::null)
    A22domainmap = LINALG::SplitMap(A->DomainMap(),*A11domainmap);
  else if (A11rowmap == Teuchos::null && A22rowmap == Teuchos::null)
    dserror("LINALG::SplitMatrix2x2: Both A11domainmap and A22domainmap == null on entry");

  // local variables
  std::vector<Teuchos::RCP<const Epetra_Map> > rangemaps(2);
  std::vector<Teuchos::RCP<const Epetra_Map> > domainmaps(2);
  rangemaps[0] = Teuchos::rcp(new Epetra_Map(*A11rowmap));
  rangemaps[1] = Teuchos::rcp(new Epetra_Map(*A22rowmap));
  domainmaps[0] = Teuchos::rcp(new Epetra_Map(*A11domainmap));
  domainmaps[1] = Teuchos::rcp(new Epetra_Map(*A22domainmap));
  LINALG::MultiMapExtractor range(A->RangeMap(),rangemaps);
  LINALG::MultiMapExtractor domain(A->DomainMap(),domainmaps);

  Teuchos::RCP<BlockSparseMatrix<DefaultBlockMatrixStrategy> > Ablock =
                       A->Split<DefaultBlockMatrixStrategy>(domain,range);

#if 0 // debugging
  cout << "A00\n" << (*Ablock)(0,0);
  cout << "A10\n" << (*Ablock)(1,0);
  cout << "A01\n" << (*Ablock)(0,1);
  cout << "A11\n" << (*Ablock)(1,1);
  cout << "A->Range\n" << A->RangeMap();
  cout << "A->Domain\n" << A->DomainMap();
  cout << "A11domainmap\n" << *A11domainmap;
  cout << "A22domainmap\n" << *A22domainmap;
#endif

  Ablock->Complete();
  // extract internal data from Ablock in Teuchos::RCP form and let Ablock die
  // (this way, internal data from Ablock will live)
  A11 = Teuchos::rcp(new SparseMatrix((*Ablock)(0,0),View));
  A12 = Teuchos::rcp(new SparseMatrix((*Ablock)(0,1),View));
  A21 = Teuchos::rcp(new SparseMatrix((*Ablock)(1,0),View));
  A22 = Teuchos::rcp(new SparseMatrix((*Ablock)(1,1),View));

  return true;
}
#endif

/*----------------------------------------------------------------------*
 | split a map into 2 pieces with given Agiven                     06/06|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> LINALG::SplitMap(const Epetra_Map& Amap,
                                          const Epetra_Map& Agiven)
{
  const Epetra_Comm& Comm = Amap.Comm();
  const Epetra_Map&  Ag = Agiven;

  int count=0;
  std::vector<int> myaugids(Amap.NumMyElements());
  for (int i=0; i<Amap.NumMyElements(); ++i)
  {
    const int gid = Amap.GID(i);
    if (Ag.MyGID(gid)) continue;
    myaugids[count] = gid;
    ++count;
  }
  myaugids.resize(count);
  int gcount;
  Comm.SumAll(&count,&gcount,1);
  Teuchos::RCP<Epetra_Map> Aunknown = Teuchos::rcp(new Epetra_Map(gcount,count,&myaugids[0],0,Comm));

  return Aunknown;
}


/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> LINALG::MergeMap(const Epetra_Map& map1,
                                 const Epetra_Map& map2,
                                 bool overlap)
{
  // check for unique GIDs and for identity
  //if ((!map1.UniqueGIDs()) || (!map2.UniqueGIDs()))
  //  dserror("LINALG::MergeMap: One or both input maps are not unique");
  if (map1.SameAs(map2))
  {
    if ((overlap==false) && map1.NumGlobalElements()>0)
      dserror("LINALG::MergeMap: Result map is overlapping");
    else
      return Teuchos::rcp(new Epetra_Map(map1));
  }

  std::vector<int> mygids(map1.NumMyElements()+map2.NumMyElements());
  int count = map1.NumMyElements();

  // get GIDs of input map1
  for (int i=0;i<count;++i)
    mygids[i] = map1.GID(i);

  // add GIDs of input map2 (only new ones)
  for (int i=0;i<map2.NumMyElements();++i)
  {
    // check for overlap
    if (map1.MyGID(map2.GID(i)))
    {
      if (overlap==false) dserror("LINALG::MergeMap: Result map is overlapping");
    }
    // add new GIDs to mygids
    else
    {
      mygids[count]=map2.GID(i);
      ++count;
    }
  }
  mygids.resize(count);

  // sort merged map
  sort(mygids.begin(),mygids.end());

  return Teuchos::rcp(new Epetra_Map(-1,(int)mygids.size(),&mygids[0],0,map1.Comm()));
}

/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> LINALG::MergeMap(const Teuchos::RCP<const Epetra_Map>& map1,
                                 const Teuchos::RCP<const Epetra_Map>& map2,
                                 bool overlap)
{
  // check for cases with null Teuchos::RCPs
  if (map1==Teuchos::null && map2==Teuchos::null)
    return Teuchos::null;
  else if (map1==Teuchos::null)
    return Teuchos::rcp(new Epetra_Map(*map2));
  else if (map2==Teuchos::null)
    return Teuchos::rcp(new Epetra_Map(*map1));

  // wrapped call to non-Teuchos::RCP version of MergeMap
  return LINALG::MergeMap(*map1,*map2,overlap);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> LINALG::CreateMap(const std::set<int>& gids, const Epetra_Comm& comm)
{
  std::vector<int> mapvec;
  mapvec.reserve(gids.size());
  mapvec.assign(gids.begin(), gids.end());
  Teuchos::RCP<Epetra_Map> map =
    Teuchos::rcp(new Epetra_Map(-1,
                                mapvec.size(),
                                &mapvec[0],
                                0,
                                comm));
  mapvec.clear();
  return map;
}


/*----------------------------------------------------------------------*
 | split a vector into 2 pieces with given submaps            popp 02/08|
 *----------------------------------------------------------------------*/
bool LINALG::SplitVector(const Epetra_Map& xmap,
                         const Epetra_Vector& x,
                         Teuchos::RCP<Epetra_Map>& x1map,
                         Teuchos::RCP<Epetra_Vector>&   x1,
                         Teuchos::RCP<Epetra_Map>& x2map,
                         Teuchos::RCP<Epetra_Vector>&   x2)
{
  // map extractor with fullmap(xmap) and two other maps (x1map and x2map)
  LINALG::MapExtractor extractor (xmap,x1map,x2map);

  // ectract subvectors from fullvector
  x1 = extractor.ExtractVector(x,1);
  x2 = extractor.ExtractVector(x,0);

  return true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::PrintSparsityToPostscript(const Epetra_RowMatrix& A)
{
  Ifpack_PrintSparsity(A);
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::PrintMatrixInMatlabFormat(std::string fname,
    const Epetra_CrsMatrix& A,
    const bool newfile)
{
  // The following source code has been adapted from the Print() method
  // of the Epetra_CrsMatrix class (see "Epetra_CrsMatrix.cpp").

  int MyPID = A.RowMap().Comm().MyPID();
  int NumProc = A.RowMap().Comm().NumProc();

  std::ofstream os;

  for (int iproc=0; iproc < NumProc; iproc++)
  {
    if (MyPID==iproc)
    {
      // open file for writing
      if ((iproc == 0) && (newfile))
        os.open(fname.c_str(),std::fstream::trunc);
      else
        os.open(fname.c_str(),std::fstream::ate | std::fstream::app);

      int NumMyRows1 = A.NumMyRows();
      int MaxNumIndices = A.MaxNumEntries();
      int * Indices  = new int[MaxNumIndices];
      double * Values  = new double[MaxNumIndices];
      int NumIndices;
      int i, j;

      for (i=0; i<NumMyRows1; i++)
      {
        int Row = A.GRID(i); // Get global row number
        A.ExtractGlobalRowCopy(Row, MaxNumIndices, NumIndices, Values, Indices);

        for (j = 0; j < NumIndices ; j++) {
          os << std::setw(10) << Row+1 ; // increase index by one for matlab
          os << std::setw(10) << Indices[j]+1;  // increase index by one for matlab
          os << std::setw(30) << std::setprecision(16) << std::scientific << Values[j];
          os << endl;
        }
      }

      delete [] Indices;
      delete [] Values;

      os << flush;

      // close file
      os.close();
    }
    // Do a few global ops to give I/O a chance to complete
    A.RowMap().Comm().Barrier();
    A.RowMap().Comm().Barrier();
    A.RowMap().Comm().Barrier();
  }

  // just to be sure
  if (os.is_open()) os.close();

  // have fun with your Matlab matrix
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::PrintSerialDenseMatrixInMatlabFormat(std::string fname,
    const Epetra_SerialDenseMatrix& A,
    const bool newfile)
{
  // The following source code has been adapted from the PrintMatrixInMatlabFormat
  // method in order to also print a Epetra_SerialDenseMatrix.


  std::ofstream os;

      // open file for writing
      if (newfile)
        os.open(fname.c_str(),std::fstream::trunc);
      else
        os.open(fname.c_str(),std::fstream::ate | std::fstream::app);

      int NumMyRows = A.RowDim();
      int NumMyColumns = A.ColDim();

      for (int i=0; i<NumMyRows; i++)
      {
        for (int j = 0; j < NumMyColumns ; j++)
        {
          os << std::setw(10) << i+1 ; // increase index by one for matlab
          os << std::setw(10) << j+1;  // increase index by one for matlab
          os << std::setw(30) << std::setprecision(16) << std::scientific << A(i,j);
          os << endl;
        }
      }

      os << flush;

      // close file
      os.close();

  // just to be sure
  if (os.is_open()) os.close();

  // have fun with your Matlab matrix
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::PrintBlockMatrixInMatlabFormat(
    std::string fname,
    const BlockSparseMatrixBase& A)
{
  // For each sub-matrix of A use the existing printing method
  for (int r = 0; r < A.Rows(); r++)
  {
    for (int c = 0; c < A.Cols(); c++)
    {
      const LINALG::SparseMatrix& M = A.Matrix(r, c);
      LINALG::PrintMatrixInMatlabFormat(fname,*(M.EpetraMatrix()), ((r==0) && (c==0)));
    }
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::PrintVectorInMatlabFormat(std::string fname,
    const Epetra_Vector& V,
    const bool newfile)
{
  // The following source code has been adapted from the Print() method
  // of the Epetra_CrsMatrix class (see "Epetra_CrsMatrix.cpp").

  int MyPID = V.Map().Comm().MyPID();
  int NumProc = V.Map().Comm().NumProc();

  std::ofstream os;

  for (int iproc=0; iproc <NumProc; iproc++)    // loop over all processors
  {
    if(MyPID==iproc)
    {
      // open file for writing
      if ((iproc == 0) && (newfile))
        os.open(fname.c_str(),std::fstream::trunc);
      else
        os.open(fname.c_str(),std::fstream::ate | std::fstream::app);

      int NumMyElements1 = V.Map().NumMyElements();
      int MaxElementSize1 = V.Map().MaxElementSize();
      int* MyGlobalElements1 = V.Map().MyGlobalElements();
      int* FirstPointInElementList1(NULL);
      if (MaxElementSize1!=1) FirstPointInElementList1 = V.Map().FirstPointInElementList();
      double ** A_Pointers = V.Pointers();

      for (int i=0; i<NumMyElements1; i++)
      {
        for(int ii=0; ii< V.Map().ElementSize(i); ii++)
        {
          int iii;
          if(MaxElementSize1==1)
          {
            os << std::setw(10)<< MyGlobalElements1[i] ;
            iii = i;
          }
          else
          {
            os << std::setw(10) << MyGlobalElements1[i]<< "/" << std::setw(10) << ii;
            iii = FirstPointInElementList1[i]+ii;
          }

          os << std::setw(30) << std::setprecision(16) <<  A_Pointers[0][iii];    // print out values of 1. vector (only Epetra_Vector supported, no Multi_Vector)
          os << endl;
        }
      }
      os << flush;
    }
    // close file
    os.close();

    // Do a few global ops to give I/O a chance to complete
    V.Map().Comm().Barrier();
    V.Map().Comm().Barrier();
    V.Map().Comm().Barrier();
  }

  // just to be sure
  if (os.is_open()) os.close();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::PrintMapInMatlabFormat(std::string fname,
    const Epetra_Map& map,
    const bool newfile)
{
  // The following source code has been adapted from the Print() method
  // of the Epetra_CrsMatrix class (see "Epetra_CrsMatrix.cpp").

  int MyPID = map.Comm().MyPID();
  int NumProc = map.Comm().NumProc();

  std::ofstream os;

  for (int iproc=0; iproc <NumProc; iproc++)    // loop over all processors
  {
    if(MyPID==iproc)
    {
      // open file for writing
      if ((iproc == 0) && (newfile))
        os.open(fname.c_str(),std::fstream::trunc);
      else
        os.open(fname.c_str(),std::fstream::ate | std::fstream::app);

      int NumMyElements1 = map.NumMyElements();
      int MaxElementSize1 = map.MaxElementSize();
      int* MyGlobalElements1 = map.MyGlobalElements();

      for (int i=0; i<NumMyElements1; i++)
      {
        for(int ii=0; ii< map.ElementSize(i); ii++)
        {
          if(MaxElementSize1==1)
          {
            os << std::setw(10)<< MyGlobalElements1[i]+1 ;
          }
          else
          {
            os << std::setw(10) << MyGlobalElements1[i]+1<< "/" << std::setw(10) << ii;
          }
          os << endl;
        }
      }
      os << flush;
    }
    // close file
    os.close();

    // Do a few global ops to give I/O a chance to complete
    map.Comm().Barrier();
    map.Comm().Barrier();
    map.Comm().Barrier();
  }

  // just to be sure
  if (os.is_open()) os.close();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int LINALG::FindMyPos(int nummyelements, const Epetra_Comm& comm)
{
  const int myrank  = comm.MyPID();
  const int numproc = comm.NumProc();

  std::vector<int> snum(numproc,0);
  std::vector<int> rnum(numproc);
  snum[myrank] = nummyelements;

  comm.SumAll(&snum[0],&rnum[0],numproc);

  return std::accumulate(&rnum[0], &rnum[myrank], 0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::AllreduceVector( const std::vector<int> & src, std::vector<int> & dest, const Epetra_Comm& comm )
{
  // communicate size
  int localsize = static_cast<int>( src.size() );
  int globalsize;
  comm.SumAll( &localsize, &globalsize, 1 );

  // communicate values
  int pos = FindMyPos( localsize, comm );
  std::vector<int> sendglobal( globalsize, 0 );
  dest.resize( globalsize );
  std::copy( src.begin(), src.end(), &sendglobal[pos] );
  comm.SumAll( &sendglobal[0], &dest[0], globalsize );

  // sort & unique
  std::sort( dest.begin(), dest.end() );
  std::vector<int>::iterator i = std::unique( dest.begin(), dest.end() );
  dest.erase( i, dest.end() );
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::AllreduceEMap(std::vector<int>& rredundant, const Epetra_Map& emap)
{
  const int mynodepos = FindMyPos(emap.NumMyElements(), emap.Comm());

  std::vector<int> sredundant(emap.NumGlobalElements(),0);

  int* gids = emap.MyGlobalElements();
  std::copy(gids, gids+emap.NumMyElements(), &sredundant[mynodepos]);

  rredundant.resize(emap.NumGlobalElements());
  emap.Comm().SumAll(&sredundant[0], &rredundant[0], emap.NumGlobalElements());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::AllreduceEMap(std::map<int,int>& idxmap, const Epetra_Map& emap)
{
#ifdef DEBUG
  if (not emap.UniqueGIDs())
    dserror("works only for unique Epetra_Maps");
#endif

  idxmap.clear();

  std::vector<int> rredundant;
  AllreduceEMap(rredundant, emap);

  for (std::size_t i=0; i<rredundant.size(); ++i)
  {
    idxmap[rredundant[i]] = i;
  }
}

/*----------------------------------------------------------------------*
 |  create an allreduced map on a distinct processor (public)  gjb 12/07|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> LINALG::AllreduceEMap(const Epetra_Map& emap, const int pid)
{
#ifdef DEBUG
  if (not emap.UniqueGIDs())
    dserror("works only for unique Epetra_Maps");
#endif
  std::vector<int> rv;
  AllreduceEMap(rv,emap);
  Teuchos::RCP<Epetra_Map> rmap;

  if (emap.Comm().MyPID()==pid)
  {
	  rmap = Teuchos::rcp(new Epetra_Map(-1,rv.size(),&rv[0],0,emap.Comm()));
	  // check the map
	  dsassert(rmap->NumMyElements() == rmap->NumGlobalElements(),
	  			  "Processor with pid does not get all map elements");
  }
  else
  {
	  rv.clear();
	  rmap = Teuchos::rcp(new Epetra_Map(-1,0,NULL,0,emap.Comm()));
	  // check the map
	  dsassert(rmap->NumMyElements() == 0,
	  			  "At least one proc will keep a map element");
  }
  return rmap;
}

/*----------------------------------------------------------------------*
 |  create an allreduced map on EVERY processor (public)        tk 12/07|
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> LINALG::AllreduceEMap(const Epetra_Map& emap)
{
#ifdef DEBUG
  if (not emap.UniqueGIDs())
    dserror("works only for unique Epetra_Maps");
#endif
  std::vector<int> rv;
  AllreduceEMap(rv,emap);
  Teuchos::RCP<Epetra_Map> rmap;

  rmap = Teuchos::rcp(new Epetra_Map(-1,rv.size(),&rv[0],0,emap.Comm()));
  // check the map

  return rmap;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Map> LINALG::AllreduceOverlappingEMap(const Epetra_Map& emap)
{
  std::vector<int> rv;
  AllreduceEMap(rv,emap);

  // remove duplicates
  std::set<int> rs(rv.begin(),rv.end());
  rv.assign(rs.begin(),rs.end());

  return Teuchos::rcp(new Epetra_Map(-1,rv.size(),&rv[0],0,emap.Comm()));
}


/*----------------------------------------------------------------------*
 |  Send and receive lists of ints.  (heiner 09/07)                     |
 *----------------------------------------------------------------------*/
void LINALG::AllToAllCommunication( const Epetra_Comm& comm,
                                    const std::vector< std::vector<int> >& send,
                                    std::vector< std::vector<int> >& recv )
{
#ifndef PARALLEL

  dsassert(send.size()==1, "there has to be just one entry for sending");

  // make a copy
  recv.clear();
  recv.push_back(send[0]);

#else

  if (comm.NumProc()==1)
  {
    dsassert(send.size()==1, "there has to be just one entry for sending");

    // make a copy
    recv.clear();
    recv.push_back(send[0]);
  }
  else
  {
    const Epetra_MpiComm& mpicomm = dynamic_cast<const Epetra_MpiComm&>(comm);

    std::vector<int> sendbuf;
    std::vector<int> sendcounts;
    sendcounts.reserve( comm.NumProc() );
    std::vector<int> sdispls;
    sdispls.reserve( comm.NumProc() );

    int displacement = 0;
    sdispls.push_back( 0 );
    for ( std::vector< std::vector<int> >::const_iterator iter = send.begin();
          iter != send.end(); ++iter )
    {
        sendbuf.insert( sendbuf.end(), iter->begin(), iter->end() );
        sendcounts.push_back( iter->size() );
        displacement += iter->size();
        sdispls.push_back( displacement );
    }

    std::vector<int> recvcounts( comm.NumProc() );

    // initial communication: Request. Send and receive the number of
    // ints we communicate with each process.

    int status = MPI_Alltoall( &sendcounts[0], 1, MPI_INT,
                               &recvcounts[0], 1, MPI_INT, mpicomm.GetMpiComm() );

    if ( status != MPI_SUCCESS )
        dserror( "MPI_Alltoall returned status=%d", status );

    std::vector<int> rdispls;
    rdispls.reserve( comm.NumProc() );

    displacement = 0;
    rdispls.push_back( 0 );
    for ( std::vector<int>::const_iterator iter = recvcounts.begin();
          iter != recvcounts.end(); ++iter )
    {
        displacement += *iter;
        rdispls.push_back( displacement );
    }

    std::vector<int> recvbuf( rdispls.back() );

    // transmit communication: Send and get the data.

    status = MPI_Alltoallv ( &sendbuf[0], &sendcounts[0], &sdispls[0], MPI_INT,
                             &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_INT,
                             mpicomm.GetMpiComm() );
    if ( status != MPI_SUCCESS )
        dserror( "MPI_Alltoallv returned status=%d", status );

    recv.clear();
    for ( int proc = 0; proc < comm.NumProc(); ++proc )
    {
        recv.push_back( std::vector<int>( &recvbuf[rdispls[proc]], &recvbuf[rdispls[proc+1]] ) );
    }
  }

#endif // PARALLEL
}







