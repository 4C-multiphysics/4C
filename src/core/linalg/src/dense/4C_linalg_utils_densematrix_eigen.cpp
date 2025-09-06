// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_densematrix_eigen.hpp"

#include "4C_linalg_fortran_definitions.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  compute all eigenvalues of a real symmetric matrix A        lw 04/08|
 *----------------------------------------------------------------------*/
void Core::LinAlg::symmetric_eigen_values(
    Core::LinAlg::SerialDenseMatrix& A, Core::LinAlg::SerialDenseVector& L, const bool postproc)
{
  Core::LinAlg::symmetric_eigen(A, L, 'N', postproc);
}

/*----------------------------------------------------------------------*
 |  compute all eigenvalues and eigenvectors of a real symmetric        |
 |  matrix A (eigenvectors are stored in A, i.e. original matrix        |
 |  is destroyed!!!)                                            lw 04/08|
 *----------------------------------------------------------------------*/
void Core::LinAlg::symmetric_eigen_problem(
    Core::LinAlg::SerialDenseMatrix& A, Core::LinAlg::SerialDenseVector& L, const bool postproc)
{
  Core::LinAlg::symmetric_eigen(A, L, 'V', postproc);
}

/*----------------------------------------------------------------------*
 |  compute all eigenvalues and, optionally,                            |
 |  eigenvectors of a real symmetric matrix A                  maf 06/07|
 *----------------------------------------------------------------------*/
void Core::LinAlg::symmetric_eigen(Core::LinAlg::SerialDenseMatrix& A,
    Core::LinAlg::SerialDenseVector& L, const bool eval_eigenvectors, const bool postproc)
{
  if (A.numRows() != A.numCols()) FOUR_C_THROW("Matrix is not square");
  if (A.numRows() != L.length()) FOUR_C_THROW("Dimension of eigenvalues does not match");

  double* a = A.values();
  double* w = L.values();
  const char uplo = {'U'};
  const int lda = A.stride();
  const int dim = A.numRows();

  int lwork = 0;
  if (dim == 1)
    lwork = 1;
  else
  {
    if (eval_eigenvectors)
      lwork = 2 * dim * dim + 6 * dim + 1;
    else
      lwork = 3 * dim + 1;
  }
  std::vector<double> work(lwork);
  int info = 0;

  Teuchos::LAPACK<int, double> lapack;
  const char jobz = eval_eigenvectors ? 'V' : 'N';
  lapack.SYEV(jobz, uplo, dim, a, lda, w, work.data(), lwork, &info);

  if (!postproc)
  {
    if (info > 0) FOUR_C_THROW("Lapack algorithm syevd failed");
    if (info < 0) FOUR_C_THROW("Illegal value in Lapack syevd call");
  }
  // if we only calculate eigenvalues/eigenvectors for postprocessing,
  // a warning might be sufficient
  else
  {
    if (info > 0)
      std::cout
          << "Lapack algorithm syevd failed: " << info
          << " off-diagonal elements of intermediate tridiagonal form did not converge to zero"
          << std::endl;
    if (info < 0) std::cout << "Illegal value in Lapack syevd call" << std::endl;
  }
}

std::vector<std::complex<double>> Core::LinAlg::generalized_eigen(
    Core::LinAlg::SerialDenseMatrix& A, Core::LinAlg::SerialDenseMatrix& B)
{
  Core::LinAlg::SerialDenseMatrix tmpA(A);
  Core::LinAlg::SerialDenseMatrix tmpB(B);

  //--------------------------------------------------------------------
  // STEP 1:
  // Transform the matrix B to upper triangular matrix with the help of
  // QR-factorization
  //--------------------------------------------------------------------

  int N = tmpA.numRows();
  double* a = tmpA.values();
  double* b = tmpB.values();
  int LDA = tmpA.stride();
  int LDB = tmpB.stride();

  // the order of permutation matrix
  std::vector<int> jpvt(N);
  // factor uses for calculating orthogonal matrix Q
  std::vector<double> tau(N);
  for (int i = 0; i < N; ++i)
  {
    jpvt[i] = 0;
    tau[i] = 0.;
  }

  int lwork1 = 0;
  if (N == 1)
    lwork1 = 1;
  else
  {
    lwork1 = N * 3 + 1;
  }
  std::vector<double> work1(lwork1);
  int info;
  dgeqp3(&N, &N, b, &LDB, jpvt.data(), tau.data(), work1.data(), &lwork1, &info);

  if (info < 0)
    std::cout << "Lapack algorithm dgeqp3: The " << info << "-th argument had an illegal value"
              << std::endl;

  // calculate the matrix Q from multiplying householder transformations
  // Q = H(1)*H(2) ... H(k)
  // H(i) = I - tau-v*v**T
  // v is a vector with v(1:i-1) = 0 and v(i+1:m) is stored on exit in B(i+1:m,i)

  // Q is initialized as an unit matrix
  Core::LinAlg::SerialDenseMatrix Q_new;
  Q_new.shape(N, N);
  for (int i = 0; i < N; ++i) Q_new(i, i) = 1.0;

  for (int i = 0; i < N; ++i)
  {
    Core::LinAlg::SerialDenseVector v;
    v.size(N);
    v(i) = 1.;
    for (int j = i + 1; j < N; ++j) v(j) = tmpB(j, i);

    Core::LinAlg::SerialDenseMatrix H;
    H.shape(N, N);

    Core::LinAlg::multiply_nt(0., H, tau[i], v, v);
    H.scale(-1.);
    for (int k = 0; k < N; ++k) H(k, k) = 1. + H(k, k);

    Core::LinAlg::SerialDenseMatrix Q_help;
    Q_help.shape(N, N);
    Core::LinAlg::multiply(Q_help, Q_new, H);
    Q_new = Q_help;
  }

  // permutation matrix
  Core::LinAlg::SerialDenseMatrix P;
  P.shape(N, N);
  for (int i = 0; i < N; ++i)
  {
    int w = jpvt[i];
    P(w - 1, i) = 1.;
  }

  // annul the under-diagonal elements of B
  // loop of columns
  for (int i = 0; i < N; ++i)
  {
    // loop of rows
    for (int j = i + 1; j < N; ++j) tmpB(j, i) = 0.;
  }

  // the new A:= Q**T A P
  Core::LinAlg::SerialDenseMatrix A_tmp;
  A_tmp.shape(N, N);
  // A_tt.Multiply('T','N',1.,Q_qr_tt,A,0.);
  Core::LinAlg::multiply_tn(A_tmp, Q_new, tmpA);

  Core::LinAlg::SerialDenseMatrix A_new;
  A_new.shape(N, N);
  Core::LinAlg::multiply(A_new, A_tmp, P);

  a = A_new.values();

  //--------------------------------------------------------
  // STEP 2
  // transform A to a upper hessenberg matrix and keep B as
  // an upper diagonal matrix
  //--------------------------------------------------------

  // balance problem to obtain better accuracy
  char job = 'P';
  int ILO;
  int IHI;
  std::vector<double> lscale(N);
  std::vector<double> rscale(N);
  std::vector<double> work0(6 * N);
  dggbal(&job, &N, a, &N, b, &N, &ILO, &IHI, lscale.data(), rscale.data(), work0.data(), &info);
  if (info != 0) FOUR_C_THROW("error dggbal");

  job = 'E';
  char COMPQ = 'I';
  char COMPZ = 'I';

  int lwork = 0;
  if (N == 1)
    lwork = 1;
  else
  {
    lwork = N;
  }
  std::vector<double> work(lwork);

  Core::LinAlg::SerialDenseMatrix A1;
  Core::LinAlg::SerialDenseMatrix A2;
  A1.shape(N, N);
  A2.shape(N, N);
  double* Q = A1.values();
  int LDQ = A1.stride();
  double* Z = A2.values();
  int LDZ = A2.stride();

  dgghrd(&COMPQ, &COMPZ, &N, &ILO, &IHI, a, &LDA, b, &LDB, Q, &LDQ, Z, &LDZ, &info);

  if (info < 0)
    std::cout << "Lapack algorithm dgghrd: The " << info << "-th argument had an illegal value"
              << std::endl;

  //--------------------------------------------------------
  // STEP 3
  // transform A which is an upper hessenberg matrix to an upper
  // diagonal matrix and keep B an upper diagonal matrix via a
  // QZ-transformation
  //--------------------------------------------------------
  // vectors which contain the eigenvalues of the problem
  Core::LinAlg::SerialDenseVector::Base L1;
  Core::LinAlg::SerialDenseVector::Base L2;
  Core::LinAlg::SerialDenseVector::Base L3;
  L1.size(N);
  L2.size(N);
  L3.size(N);
  double* ALPHAR = L1.values();
  double* ALPHAI = L2.values();
  double* BETA = L3.values();

  int LDH = A_new.stride();
  int LDT = tmpB.stride();

  char COMPQ2 = 'V';
  char COMPZ2 = 'V';

  dhgeqz(&job, &COMPQ2, &COMPZ2, &N, &ILO, &IHI, a, &LDH, b, &LDT, ALPHAR, ALPHAI, BETA, Q, &LDQ, Z,
      &LDZ, work.data(), &lwork, &info);

  if (info < 0)
    std::cout << "Lapack algorithm dhgeqz: The " << info << "-th argument has an illegal value!"
              << std::endl;
  else if (info > N)
    std::cout << "Lapack algorithm dhgeqz: The QZ iteration did not converge. (H,T) is not in "
                 "Schur Form, but the Eigenvalues should be correct!"
              << std::endl;

  /*cout << "--------Final----------" << std::endl;
   std::cout << std::setprecision(16) << "A 2" << A_new << std::endl;
   std::cout << std::setprecision(16) <<  "B 2" << tmpB << std::endl;
   std::cout << std::setprecision(16) << "Q 2 " << Q_2 << std::endl;
   std::cout << std::setprecision(16) << "Z 2 " << Z_2 << std::endl;*/

  std::vector<std::complex<double>> eigenvalues;
  eigenvalues.reserve(N);
  int num_inf = 0;

  for (int i = 0; i < N; ++i)
  {
    if (std::abs(BETA[i]) < 1e-13)
    {
      ++num_inf;
      continue;  // skip infinite eigenvalue
    }
    if (ALPHAI[i] > 1e-12)
    {
      std::cout << " Warning: you have an imaginary EW " << ALPHAI[i] << '\n';
    }
    eigenvalues.emplace_back(ALPHAR[i] / BETA[i], ALPHAI[i] / BETA[i]);
  }
  if (num_inf > 0)
    std::cout << "Generalized eigenvalue computation: " << num_inf
              << " infinite eigenvalue(s) discarded." << '\n';

  return eigenvalues;
}

double Core::LinAlg::generalized_eigen_max_real_eigenvalue(
    Core::LinAlg::SerialDenseMatrix& A, Core::LinAlg::SerialDenseMatrix& B)
{
  const auto eigs = generalized_eigen(A, B);
  double max_lambda = 0.;
  for (const auto eig : eigs)
  {
    max_lambda = std::max(eig.real(), max_lambda);
  }
  return max_lambda;
}

FOUR_C_NAMESPACE_CLOSE
