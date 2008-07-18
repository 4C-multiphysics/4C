/*!----------------------------------------------------------------------
\file linalg_utils.cpp
\brief A collection of helper methods for namespace LINALG

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <algorithm>
#include <numeric>
#include <vector>

#include "linalg_utils.H"
#include "drt_dserror.H"
#include "EpetraExt_Transpose_RowMatrix.h"
#include "EpetraExt_MatrixMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_RowMatrixTransposer.h"

/*----------------------------------------------------------------------*
 |  create a Epetra_CrsMatrix  (public)                      mwgee 12/06|
 *----------------------------------------------------------------------*/
RefCountPtr<Epetra_CrsMatrix> LINALG::CreateMatrix(const Epetra_Map& rowmap, const int npr)
{
  if (!rowmap.UniqueGIDs()) dserror("Row map is not unique");
  return rcp(new Epetra_CrsMatrix(Copy,rowmap,npr,false));
}
/*----------------------------------------------------------------------*
 |  create a Epetra_Vector  (public)                         mwgee 12/06|
 *----------------------------------------------------------------------*/
RefCountPtr<Epetra_Vector> LINALG::CreateVector(const Epetra_Map& rowmap, const bool init)
{
  return rcp(new Epetra_Vector(rowmap,init));
}
/*----------------------------------------------------------------------*
 |  export a Epetra_Vector  (public)                         mwgee 12/06|
 *----------------------------------------------------------------------*/
void LINALG::Export(const Epetra_MultiVector& source, Epetra_MultiVector& target)
{
  bool sourceunique = false;
  bool targetunique = false;
  if (source.Map().UniqueGIDs()) sourceunique = true;
  if (target.Map().UniqueGIDs()) targetunique = true;

  // Note:
  // source map of an import must be unique
  // target map of an export must be unique

  // both are unique, does not matter whether ex- or import
  if (sourceunique && targetunique)
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
    // We need a unique in between stage which we have to create artifically.
    // That's nasty.
    // As it is unclear whether this will ever be needed - do it later.
    dserror("Neither target nor source maps are unique - cannot export");
  }
  else dserror("VERY strange");

  return;
}


/*----------------------------------------------------------------------*
 |  assemble a matrix  (public)                               popp 01/08|
 *----------------------------------------------------------------------*/
void LINALG::Assemble(Epetra_CrsMatrix& A, const Epetra_SerialDenseMatrix& Aele,
                      const vector<int>& lmrow, const vector<int>& lmrowowner,
                      const vector<int>& lmcol)
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
                const vector<int>& lm, const vector<int>& lmowner)
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
                const vector<int>& lm, const vector<int>& lmowner)
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
  if (B.Filled()) dserror("FillComplete was called on B before");

  Epetra_CrsMatrix* Aprime = NULL;
  RCP<EpetraExt::RowMatrix_Transpose> Atrans = null;
  if (transposeA)
  {
    Atrans = rcp(new EpetraExt::RowMatrix_Transpose(false,NULL,false));
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
  vector<int>    Indices(MaxNumEntries);
  vector<double> Values(MaxNumEntries);

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
  return;
}

/*----------------------------------------------------------------------*
 | Transpose matrix A                                         popp 02/08|
 *----------------------------------------------------------------------*/
RCP<Epetra_CrsMatrix> LINALG::Transpose(const Epetra_CrsMatrix& A)
{
  if (!A.Filled()) dserror("FillComplete was not called on A");

  if (!A.Filled()) dserror("FillComplete was not called on A");

  RCP<EpetraExt::RowMatrix_Transpose> Atrans =
  		rcp(new EpetraExt::RowMatrix_Transpose(false,NULL,false));
  Epetra_CrsMatrix* Aprime =
  		&(dynamic_cast<Epetra_CrsMatrix&>(((*Atrans)(const_cast<Epetra_CrsMatrix&>(A)))));


  return rcp(new Epetra_CrsMatrix(*Aprime));
}

/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 01/06|
 *----------------------------------------------------------------------*/
RCP<Epetra_CrsMatrix> LINALG::Multiply(const Epetra_CrsMatrix& A, bool transA,
                                       const Epetra_CrsMatrix& B, bool transB,
                                       bool complete)
{
  // make sure FillComplete was called on the matrices
  if (!A.Filled()) dserror("A has to be FillComplete");
  if (!B.Filled()) dserror("B has to be FillComplete");

  // do a very coarse guess of nonzeros per row
  int guessnpr = A.MaxNumEntries()*B.MaxNumEntries();

  // create resultmatrix with correct rowmap
  Epetra_CrsMatrix* C = NULL;
  if (!transA)
    C = new Epetra_CrsMatrix(Copy,A.OperatorRangeMap(),guessnpr,false);
  else
    C = new Epetra_CrsMatrix(Copy,A.OperatorDomainMap(),guessnpr,false);

  int err = EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*C,complete);
  if (err) dserror("EpetraExt::MatrixMatrix::Multiply returned err = &d",err);

  return rcp(C);
}



/*----------------------------------------------------------------------*
 | Multiply matrices A*B                                     mwgee 02/08|
 *----------------------------------------------------------------------*/
RCP<Epetra_CrsMatrix> LINALG::Multiply(const Epetra_CrsMatrix& A, bool transA,
                                       const Epetra_CrsMatrix& B, bool transB,
                                       const Epetra_CrsMatrix& C, bool transC,
                                       bool complete)
{
  RCP<Epetra_CrsMatrix> tmp = LINALG::Multiply(B,transB,C,transC,true);
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
  vector<double> s(min(n,m));
  int info;
  int lwork = max(3*min(m,n)+max(m,n),5*min(m,n));
  vector<double> work(lwork);
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
  vector<int> ipiv(n);
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

#ifdef LINUX_MUENCH
#define CCA_APPEND_U (1)
#endif
#ifdef CCA_APPEND_U
#define dsytrf dsytrf_
#define dsytri dsytri_
#define dgetrf dgetrf_
#define dgetri dgetri_
#endif
extern "C"
{
  void dsytrf(char *uplo, int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
  void dsytri(char *uplo, int *n, double *a, int *lda, int *ipiv, double *work, int *info);
  void dgetrf(int *m,int *n, double *a, int *lda, int *ipiv, int* info);
  void dgetri(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
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
  vector<int> ipiv(dim);
  int lwork = 10*dim;
  vector<double> work(lwork);
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
  vector<int> iwork(liwork);

  int lwork;
  if (dim == 1) lwork = 1;
  else
  {
    if      (jobz == 'N') lwork = 2*dim+1;
    else if (jobz == 'V') lwork = 2*dim*dim+6*dim+1;
  }
  vector<double> work(lwork);
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
    if (info > 0) cout << "Lapack algorithm syevd failed" << endl;
    if (info < 0) cout << "Illegal value in Lapack syevd call" << endl;
  }

  return;
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
  vector<double> s(min(n,m));
  int info;
  int lwork = max(3*min(m,n)+max(m,n),5*min(m,n));
  vector<double> work(lwork);

  lapack.GESVD(jobu,jobvt,m,n,tmp.A(),tmp.LDA(),&s[0],
               Q.A(),Q.LDA(),VT.A(),VT.LDA(),&work[0],&lwork,&info);

  if (info) dserror("Lapack's dgesvd returned %d",info);

  for (int i = 0; i < min(n,m); ++i) {
    for (int j = 0; j < min(n,m); ++j) {
      S(i,j) = (i==j) * s[i];   // 0 for off-diagonal, otherwise s
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 |  Apply dirichlet conditions  (public)                     mwgee 02/07|
 *----------------------------------------------------------------------*/
void LINALG::ApplyDirichlettoSystem(RCP<Epetra_Vector>&      x,
                                    RCP<Epetra_Vector>&      b,
                                    const RCP<Epetra_Vector> dbcval,
                                    const RCP<Epetra_Vector> dbctoggle)
{
  const Epetra_Vector& dbct = *dbctoggle;
  if (x != null && b != null)
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
void LINALG::ApplyDirichlettoSystem(RCP<LINALG::SparseOperator> A,
                                    RCP<Epetra_Vector>&         x,
                                    RCP<Epetra_Vector>&         b,
                                    const RCP<Epetra_Vector>    dbcval,
                                    const RCP<Epetra_Vector>    dbctoggle)
{
  A->ApplyDirichlet(dbctoggle);
  ApplyDirichlettoSystem(x,b,dbcval,dbctoggle);
}

#if 0 // old version
/*----------------------------------------------------------------------*
 | split matrix into 2x2 block system                              06/06|
 | this version is to go away soon! mgee                                |
 *----------------------------------------------------------------------*/
bool LINALG::SplitMatrix2x2(RCP<Epetra_CrsMatrix> A,
                            RCP<Epetra_Map>& A11rowmap,
                            RCP<Epetra_Map>& A22rowmap,
                            RCP<Epetra_CrsMatrix>& A11,
                            RCP<Epetra_CrsMatrix>& A12,
                            RCP<Epetra_CrsMatrix>& A21,
                            RCP<Epetra_CrsMatrix>& A22)
{
  if (A==null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  if (A11rowmap==null && A22rowmap != null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != null && A22rowmap != null);
  else if (A11rowmap != null && A22rowmap == null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else
  	dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  const Epetra_Comm& Comm   = A->Comm();
  const Epetra_Map&  A22map = *(A22rowmap.get());
  const Epetra_Map&  A11map = *(A11rowmap.get());

  //----------------------------- create a parallel redundant map of A22map
  map<int,int> a22gmap;
  {
    vector<int> a22global(A22map.NumGlobalElements());
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
  A22 = rcp(new Epetra_CrsMatrix(Copy,A22map,100));
  {
    vector<int>    a22gcindices(100);
    vector<double> a22values(100);
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
        map<int,int>::iterator curr = a22gmap.find(gcid);
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
  A11 = rcp(new Epetra_CrsMatrix(Copy,A11map,100));
  {
    vector<int>    a11gcindices(100);
    vector<double> a11values(100);
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
        map<int,int>::iterator curr = a22gmap.find(gcid);
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
  A12 = rcp(new Epetra_CrsMatrix(Copy,A11map,100));
  {
    vector<int>    a12gcindices(100);
    vector<double> a12values(100);
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
        map<int,int>::iterator curr = a22gmap.find(gcid);
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
  A21 = rcp(new Epetra_CrsMatrix(Copy,A22map,100));
  {
    vector<int>    a21gcindices(100);
    vector<double> a21values(100);
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
        map<int,int>::iterator curr = a22gmap.find(gcid);
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
bool LINALG::SplitMatrix2x2(RCP<Epetra_CrsMatrix> A,
                            RCP<Epetra_Map>& A11rowmap,
                            RCP<Epetra_Map>& A22rowmap,
                            RCP<Epetra_CrsMatrix>& A11,
                            RCP<Epetra_CrsMatrix>& A12,
                            RCP<Epetra_CrsMatrix>& A21,
                            RCP<Epetra_CrsMatrix>& A22)
{
  if (A==null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  if (A11rowmap==null && A22rowmap != null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != null && A22rowmap != null);
  else if (A11rowmap != null && A22rowmap == null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else
    dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  vector<RCP<const Epetra_Map> > maps(2);
  maps[0] = rcp(new Epetra_Map(*A11rowmap));
  maps[1] = rcp(new Epetra_Map(*A22rowmap));
  LINALG::MultiMapExtractor extractor(A->RowMap(),maps);

  // create SparseMatrix view to input matrix A
  SparseMatrix a(A,View);

  // split matrix into pieces, where main diagonal blocks are square
  RCP<BlockSparseMatrix<DefaultBlockMatrixStrategy> > Ablock =
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
bool LINALG::SplitMatrix2x2(RCP<LINALG::SparseMatrix> A,
                            RCP<Epetra_Map>& A11rowmap,
                            RCP<Epetra_Map>& A22rowmap,
                            RCP<Epetra_Map>& A11domainmap,
                            RCP<Epetra_Map>& A22domainmap,
                            RCP<LINALG::SparseMatrix>& A11,
                            RCP<LINALG::SparseMatrix>& A12,
                            RCP<LINALG::SparseMatrix>& A21,
                            RCP<LINALG::SparseMatrix>& A22)
{
  if (A==null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  // check and complete input row maps
  if (A11rowmap==null && A22rowmap != null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != null && A22rowmap != null);
  else if (A11rowmap != null && A22rowmap == null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else
    dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  // check and complete input domain maps
  if (A11domainmap==null && A22domainmap != null)
  	A11domainmap = LINALG::SplitMap(A->DomainMap(),*A22domainmap);
  else if (A11domainmap != null && A22domainmap != null);
  else if (A11domainmap != null && A22domainmap == null)
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
  map<int,int> a11gmap;
  {
    vector<int> a11global(A11dmap.NumGlobalElements());
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
    A11 = rcp(new LINALG::SparseMatrix(A11rmap,100));
    {
      vector<int>    a11gcindices(100);
      vector<double> a11values(100);
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
          map<int,int>::iterator curr = a11gmap.find(gcid);
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
    A22 = rcp(new LINALG::SparseMatrix(A22rmap,100));
    {
      vector<int>    a22gcindices(100);
      vector<double> a22values(100);
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
          map<int,int>::iterator curr = a11gmap.find(gcid);
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
    A12 = rcp(new LINALG::SparseMatrix(A11rmap,100));
    {
      vector<int>    a12gcindices(100);
      vector<double> a12values(100);
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
          map<int,int>::iterator curr = a11gmap.find(gcid);
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
    A21 = rcp(new LINALG::SparseMatrix(A22rmap,100));
    {
      vector<int>    a21gcindices(100);
      vector<double> a21values(100);
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
          map<int,int>::iterator curr = a11gmap.find(gcid);
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
bool LINALG::SplitMatrix2x2(RCP<LINALG::SparseMatrix> A,
                            RCP<Epetra_Map>& A11rowmap,
                            RCP<Epetra_Map>& A22rowmap,
                            RCP<Epetra_Map>& A11domainmap,
                            RCP<Epetra_Map>& A22domainmap,
                            RCP<LINALG::SparseMatrix>& A11,
                            RCP<LINALG::SparseMatrix>& A12,
                            RCP<LINALG::SparseMatrix>& A21,
                            RCP<LINALG::SparseMatrix>& A22)
{
  if (A==null)
    dserror("LINALG::SplitMatrix2x2: A==null on entry");

  // check and complete input row maps
  if (A11rowmap==null && A22rowmap != null)
    A11rowmap = LINALG::SplitMap(A->RowMap(),*A22rowmap);
  else if (A11rowmap != null && A22rowmap != null);
  else if (A11rowmap != null && A22rowmap == null)
    A22rowmap = LINALG::SplitMap(A->RowMap(),*A11rowmap);
  else
    dserror("LINALG::SplitMatrix2x2: Both A11rowmap and A22rowmap == null on entry");

  // check and complete input domain maps
  if (A11domainmap==null && A22domainmap != null)
  	A11domainmap = LINALG::SplitMap(A->DomainMap(),*A22domainmap);
  else if (A11domainmap != null && A22domainmap != null);
  else if (A11domainmap != null && A22domainmap == null)
    A22domainmap = LINALG::SplitMap(A->DomainMap(),*A11domainmap);
  else
    dserror("LINALG::SplitMatrix2x2: Both A11domainmap and A22domainmap == null on entry");

  // local variables
  vector<RCP<const Epetra_Map> > rangemaps(2);
  vector<RCP<const Epetra_Map> > domainmaps(2);
  rangemaps[0] = rcp(new Epetra_Map(*A11rowmap));
  rangemaps[1] = rcp(new Epetra_Map(*A22rowmap));
  domainmaps[0] = rcp(new Epetra_Map(*A11domainmap));
  domainmaps[1] = rcp(new Epetra_Map(*A22domainmap));
  LINALG::MultiMapExtractor range(A->RangeMap(),rangemaps);
  LINALG::MultiMapExtractor domain(A->DomainMap(),domainmaps);

  RCP<BlockSparseMatrix<DefaultBlockMatrixStrategy> > Ablock =
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
  // extract internal data from Ablock in RCP form and let Ablock die
  // (this way, internal data from Ablock will live)
  A11 = rcp(new SparseMatrix((*Ablock)(0,0),View));
  A12 = rcp(new SparseMatrix((*Ablock)(0,1),View));
  A21 = rcp(new SparseMatrix((*Ablock)(1,0),View));
  A22 = rcp(new SparseMatrix((*Ablock)(1,1),View));

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
  vector<int> myaugids(Amap.NumMyElements());
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
  myaugids.clear();
  return Aunknown;
}


/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
RCP<Epetra_Map> LINALG::MergeMap(const Epetra_Map& map1,
                                 const Epetra_Map& map2,
                                 bool overlap)
{
  // check for unique GIDs and for identity
  if ((!map1.UniqueGIDs()) || (!map2.UniqueGIDs()))
    dserror("LINALG::MergeMap: One or both input maps are not unique");
  if (map1.SameAs(map2))
  {
    if ((overlap==false) && map1.NumGlobalElements()>0)
      dserror("LINALG::MergeMap: Result map is overlapping");
    else
      return rcp(new Epetra_Map(map1));
  }

  vector<int> mygids(map1.NumMyElements()+map2.NumMyElements());
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

	return rcp(new Epetra_Map(-1,(int)mygids.size(),&mygids[0],0,map1.Comm()));
}

/*----------------------------------------------------------------------*
 | merge two given maps to one map                            popp 01/08|
 *----------------------------------------------------------------------*/
RCP<Epetra_Map> LINALG::MergeMap(const RCP<Epetra_Map>& map1,
                                 const RCP<Epetra_Map>& map2,
                                 bool overlap)
{
  // check for cases with null RCPs
  if (map1==null && map2==null)
    return null;
  else if (map1==null)
    return rcp(new Epetra_Map(*map2));
  else if (map2==null)
    return rcp(new Epetra_Map(*map1));

  // wrapped call to non-RCP version of MergeMap
  return LINALG::MergeMap(*map1,*map2,overlap);
}

/*----------------------------------------------------------------------*
 | split a vector into 2 pieces with given submaps            popp 02/08|
 *----------------------------------------------------------------------*/
bool LINALG::SplitVector(const Epetra_Vector& x,
                         const Epetra_Map& x1map,
                         RCP<Epetra_Vector>&   x1,
                         const Epetra_Map& x2map,
                         RCP<Epetra_Vector>&   x2)
{
  x1 = rcp(new Epetra_Vector(x1map,false));
  x2 = rcp(new Epetra_Vector(x2map,false));

  //use an exporter or importer object
  Epetra_Export exporter_x1(x.Map(),x1map);
  Epetra_Export exporter_x2(x.Map(),x2map);

  int err = x1->Export(x,exporter_x1,Insert);
  if (err) dserror("ERROR: SplitVector: Export returned error &i", err);

  err = x2->Export(x,exporter_x2,Insert);
  if (err) dserror("ERROR: SplitVector: Export returned error &i", err);

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
int LINALG::FindMyPos(int nummyelements, const Epetra_Comm& comm)
{
  const int myrank  = comm.MyPID();
  const int numproc = comm.NumProc();

  vector<int> snum(numproc);
  vector<int> rnum(numproc);
  fill(snum.begin(), snum.end(), 0);
  snum[myrank] = nummyelements;

  comm.SumAll(&snum[0],&rnum[0],numproc);

  return std::accumulate(&rnum[0], &rnum[myrank], 0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::AllreduceEMap(vector<int>& rredundant, const Epetra_Map& emap)
{
  int mynodepos = FindMyPos(emap.NumMyElements(), emap.Comm());

  vector<int> sredundant(emap.NumGlobalElements());
  fill(sredundant.begin(), sredundant.end(), 0);

  int* gids = emap.MyGlobalElements();
  copy(gids, gids+emap.NumMyElements(), &sredundant[mynodepos]);

  rredundant.resize(emap.NumGlobalElements());
  emap.Comm().SumAll(&sredundant[0], &rredundant[0], emap.NumGlobalElements());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void LINALG::AllreduceEMap(map<int,int>& idxmap, const Epetra_Map& emap)
{
  idxmap.clear();

  vector<int> rredundant(emap.NumGlobalElements());
  AllreduceEMap(rredundant, emap);

  for (unsigned i=0; i<rredundant.size(); ++i)
  {
    idxmap[rredundant[i]] = i;
  }
}

/*----------------------------------------------------------------------*
 |  create an allreduced map on a distinct processor (public)  gjb 12/07|
 *----------------------------------------------------------------------*/
RCP<Epetra_Map> LINALG::AllreduceEMap(const Epetra_Map& emap, const int pid)
{
  vector<int> rv;
  AllreduceEMap(rv,emap);
  RefCountPtr<Epetra_Map> rmap;

  if (emap.Comm().MyPID()==pid)
  {
	  rmap = rcp(new Epetra_Map(-1,rv.size(),&rv[0],0,emap.Comm()));
	  // check the map
	  dsassert(rmap->NumMyElements() == rmap->NumGlobalElements(),
	  			  "Processor with pid does not get all map elements");
  }
  else
  {
	  rv.clear();
	  rmap = rcp(new Epetra_Map(-1,0,NULL,0,emap.Comm()));
	  // check the map
	  dsassert(rmap->NumMyElements() == 0,
	  			  "At least one proc will keep a map element");
  }
  return rmap;
}

/*----------------------------------------------------------------------*
 |  create an allreduced map on EVERY processor (public)        tk 12/07|
 *----------------------------------------------------------------------*/
RCP<Epetra_Map> LINALG::AllreduceEMap(const Epetra_Map& emap)
{
  vector<int> rv;
  AllreduceEMap(rv,emap);
  RefCountPtr<Epetra_Map> rmap;

  rmap = rcp(new Epetra_Map(-1,rv.size(),&rv[0],0,emap.Comm()));
  // check the map

  return rmap;
}

/*----------------------------------------------------------------------*
 |  Send and receive lists of ints.  (heiner 09/07)                     |
 *----------------------------------------------------------------------*/
void LINALG::AllToAllCommunication( const Epetra_Comm& comm,
                                    const vector< vector<int> >& send,
                                    vector< vector<int> >& recv )
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

    vector<int> sendbuf;
    vector<int> sendcounts;
    sendcounts.reserve( comm.NumProc() );
    vector<int> sdispls;
    sdispls.reserve( comm.NumProc() );

    int displacement = 0;
    sdispls.push_back( 0 );
    for ( vector< vector<int> >::const_iterator iter = send.begin();
          iter != send.end(); ++iter )
    {
        sendbuf.insert( sendbuf.end(), iter->begin(), iter->end() );
        sendcounts.push_back( iter->size() );
        displacement += iter->size();
        sdispls.push_back( displacement );
    }

    vector<int> recvcounts( comm.NumProc() );

    // initial communication: Request. Send and receive the number of
    // ints we communicate with each process.

    int status = MPI_Alltoall( &sendcounts[0], 1, MPI_INT,
                               &recvcounts[0], 1, MPI_INT, mpicomm.GetMpiComm() );

    if ( status != MPI_SUCCESS )
        dserror( "MPI_Alltoall returned status=%d", status );

    vector<int> rdispls;
    rdispls.reserve( comm.NumProc() );

    displacement = 0;
    rdispls.push_back( 0 );
    for ( vector<int>::const_iterator iter = recvcounts.begin();
          iter != recvcounts.end(); ++iter )
    {
        displacement += *iter;
        rdispls.push_back( displacement );
    }

    vector<int> recvbuf( rdispls.back() );

    // transmit communication: Send and get the data.

    status = MPI_Alltoallv ( &sendbuf[0], &sendcounts[0], &sdispls[0], MPI_INT,
                             &recvbuf[0], &recvcounts[0], &rdispls[0], MPI_INT,
                             mpicomm.GetMpiComm() );
    if ( status != MPI_SUCCESS )
        dserror( "MPI_Alltoallv returned status=%d", status );

    recv.clear();
    for ( int proc = 0; proc < comm.NumProc(); ++proc )
    {
        recv.push_back( vector<int>( &recvbuf[rdispls[proc]], &recvbuf[rdispls[proc+1]] ) );
    }
  }

#endif // PARALLEL
}







#endif  // #ifdef CCADISCRET
