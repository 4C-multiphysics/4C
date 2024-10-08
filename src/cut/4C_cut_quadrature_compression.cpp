/*----------------------------------------------------------------------*/
/*! \file
\brief see paper by Sudhakar


\level 2
 */
/*----------------------------------------------------------------------*/


#include "4C_cut_quadrature_compression.hpp"

#include "4C_cut_element.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialQRDenseSolver.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

// Use Chebyshev polynomials to form Vandermonde matrix.
// if commented use regular monomial basis for which the conditioning of the matrix may not be good
// Since we are using only 5th order
// Quadrature and the base quadrature rule has a very good point distribution (obtained from direct
// divergence), the problem due to matrix conditioning does not appear
// #define CHECBYSHEV

Cut::QuadratureCompression::QuadratureCompression() {}

/*---------------------------------------------------------------------------------------------------------------*
 * Perform all operations related to quadrature compression
 * Only computing Leja points (from LU-decomposition) is checked
 *---------------------------------------------------------------------------------------------------------------*/
bool Cut::QuadratureCompression::perform_compression_of_quadrature(
    Core::FE::GaussPointsComposite& gin, Cut::VolumeCell* vc)
{
  const double t_start = Teuchos::Time::wallTime();

  Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> vander =
      Teuchos::RCP(new Core::LinAlg::SerialDenseMatrix);
  Teuchos::RCP<Core::LinAlg::SerialDenseVector> x =
      Teuchos::RCP(new Core::LinAlg::SerialDenseVector);
  Teuchos::RCP<Core::LinAlg::SerialDenseVector> rhs =
      Teuchos::RCP(new Core::LinAlg::SerialDenseVector);

  form_matrix_system(gin, vander, rhs);

  bool success = compress_leja_points(gin, vander, rhs, x);

  const double t_end = Teuchos::Time::wallTime() - t_start;
  std::cout << "quadtime = " << t_end << "\n";

  return success;
}

/*---------------------------------------------------------------------------------------------------------------*
 * Compute the Vandermonde matrix and RHS of the matrix system sudhakar 08/15
 *---------------------------------------------------------------------------------------------------------------*/
void Cut::QuadratureCompression::form_matrix_system(Core::FE::GaussPointsComposite& gin,
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs)
{
  mat->shape(gin.num_points(), 56);

  int ma = mat->numRows();
  int na = mat->numCols();

  // std::cout<<"number of original quadrature points = "<<gin.num_points()<<"\n";

  rhs->shape(na, 1);

  for (int pt = 0; pt < ma; pt++)
  {
    const double* loc = gin.point(pt);
    const double wei = gin.weight(pt);

    double x = loc[0];
    double y = loc[1];
    double z = loc[2];


#ifdef CHECBYSHEV  // Chebyshev polynomial basis
    double x2 = cos(2.0 * acos(x));
    double y2 = cos(2.0 * acos(y));
    double z2 = cos(2.0 * acos(z));

    double x3 = cos(3.0 * acos(x));
    double y3 = cos(3.0 * acos(y));
    double z3 = cos(3.0 * acos(z));

    double x4 = cos(4.0 * acos(x));
    double y4 = cos(4.0 * acos(y));
    double z4 = cos(4.0 * acos(z));

    double x5 = cos(5.0 * acos(x));
    double y5 = cos(5.0 * acos(y));
    double z5 = cos(5.0 * acos(z));

#else  // monomial basis
    double x2 = x * x;
    double y2 = y * y;
    double z2 = z * z;

    double x3 = x2 * x;
    double y3 = y2 * y;
    double z3 = z2 * z;

    double x4 = x3 * x;
    double y4 = y3 * y;
    double z4 = z3 * z;

    double x5 = x4 * x;
    double y5 = y4 * y;
    double z5 = z4 * z;
#endif

    // zeroth order term
    (*mat)(pt, 0) = 1.0;

    // first order term
    (*mat)(pt, 1) = x;
    (*mat)(pt, 2) = y;
    (*mat)(pt, 3) = z;

    // second order terms
    if (na > 4)
    {
      (*mat)(pt, 4) = x2;     // x^2
      (*mat)(pt, 5) = x * y;  // xy
      (*mat)(pt, 6) = x * z;  // xz
      (*mat)(pt, 7) = y2;     // y^2
      (*mat)(pt, 8) = y * z;  // yz
      (*mat)(pt, 9) = z2;     // z^2
    }

    // third order terms
    if (na > 10)
    {
      (*mat)(pt, 10) = x3;         // x^3
      (*mat)(pt, 11) = x2 * y;     // x^2 y
      (*mat)(pt, 12) = x2 * z;     // x^2 z
      (*mat)(pt, 13) = x * y2;     // xy^2
      (*mat)(pt, 14) = x * y * z;  // xyz
      (*mat)(pt, 15) = x * z2;     // xz^2
      (*mat)(pt, 16) = y3;         // y^3
      (*mat)(pt, 17) = y2 * z;     // y^2 z
      (*mat)(pt, 18) = y * z2;     // yz^2
      (*mat)(pt, 19) = z3;         // z^3
    }

    // fourth order terms
    if (na > 20)
    {
      (*mat)(pt, 20) = x4;          // x^4
      (*mat)(pt, 21) = x3 * y;      // x^3 y
      (*mat)(pt, 22) = x3 * z;      // x^3 z
      (*mat)(pt, 23) = x2 * y2;     // x^2 y^2
      (*mat)(pt, 24) = x2 * y * z;  // x^2 yz
      (*mat)(pt, 25) = x2 * z2;     // x^2 z^2
      (*mat)(pt, 26) = x * y3;      // xy^3
      (*mat)(pt, 27) = x * y2 * z;  // xy^2 z
      (*mat)(pt, 28) = x * y * z2;  // xyz^2
      (*mat)(pt, 29) = x * z3;      // xz^3
      (*mat)(pt, 30) = y4;          // y^4
      (*mat)(pt, 31) = y3 * z;      // y^3 z
      (*mat)(pt, 32) = y2 * z2;     // y^2 z^2
      (*mat)(pt, 33) = y * z3;      // yz^3
      (*mat)(pt, 34) = z4;          // z^4
    }

    // fifth order terms
    if (na > 35)
    {
      (*mat)(pt, 35) = x5;           // x^5
      (*mat)(pt, 36) = x4 * y;       // x^4 y
      (*mat)(pt, 37) = x4 * z;       // x^4 z
      (*mat)(pt, 38) = x3 * y2;      // x^3 y^2
      (*mat)(pt, 39) = x3 * y * z;   // x^3 yz
      (*mat)(pt, 40) = x3 * z2;      // x^3 z^2
      (*mat)(pt, 41) = x2 * y3;      // x^2 y^3
      (*mat)(pt, 42) = x2 * y2 * z;  // x^2 y^2 z
      (*mat)(pt, 43) = x2 * y * z2;  // x^2 yz^2
      (*mat)(pt, 44) = x2 * z3;      // x^2 z^3
      (*mat)(pt, 45) = x * y4;       // xy^4
      (*mat)(pt, 46) = x * y3 * z;   // xy^3 z
      (*mat)(pt, 47) = x * y2 * z2;  // xy^2 z^2
      (*mat)(pt, 48) = x * y * z3;   // xyz^3
      (*mat)(pt, 49) = x * z4;       // xz^4
      (*mat)(pt, 50) = y5;           // y^5
      (*mat)(pt, 51) = y4 * z;       // y^4 z
      (*mat)(pt, 52) = y3 * z2;      // y^3 z^2
      (*mat)(pt, 53) = y2 * z3;      // y^2 z^3
      (*mat)(pt, 54) = y * z4;       // yz^4
      (*mat)(pt, 55) = z5;           // z^5
    }

    // form the RHS of the equations
    (*rhs)(0) += wei;
    for (int ind = 1; ind < na; ind++) (*rhs)(ind) += (*mat)(pt, ind) * wei;
  }
}

/*-----------------------------------------------------------------------------------------------------------------*
 * Perform quadrature compression by computing discrete Leja points.
 * Among the given quadrature points, Leja points maximize the determinant of the sudhakar 08/15
 * Vandermonde matrix.
 *
 * This involves two steps
 * 1. Find discrete Leja points by LU factorizing the Vandermonde matrix --> this will give us the
 *number of quadrature points equal to the number of base functions
 * 2. Solve the matrix system constructing Vandermonde matrix at these Leja points and obtain
 *quadrature weights
 *
 * Read the following paper for the details of the method:
 * L. Bos, S. De Marchi, A. Sommariva, and M. Vianello. Computing multivariate Fekete and Leja
 *points by numerical linear algebra. SIAM J Numer Anal. 48:1984--1999, 2010.
 *-----------------------------------------------------------------------------------------------------------------*/
bool Cut::QuadratureCompression::compress_leja_points(Core::FE::GaussPointsComposite& gin,
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol)
{
  int ma = mat->numRows();
  int na = mat->numCols();

  Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> matTemp =
      Teuchos::RCP(new Core::LinAlg::SerialDenseMatrix);
  matTemp->shape(ma, na);

  // copy this matrix to another
  // because after performing LU decomposition, the original matrix stores the components of L and U
  for (int ii = 0; ii < ma; ii++)
  {
    for (int jj = 0; jj < na; jj++) (*matTemp)(ii, jj) = (*mat)(ii, jj);
  }

  // printing for matlab checking
  /*std::cout<<"a = [";
  for( int i=0; i<ma; i++ )
  {
    for( int j=0; j<na; j++ )
    std::cout<<(*matTemp)(i,j)<<" ";
    std::cout<<";";
  }
  std::cout<<"]\n";*/


  Teuchos::RCP<Core::LinAlg::IntSerialDenseVector> work_temp =
      Teuchos::RCP(new Core::LinAlg::IntSerialDenseVector);
  work_temp->shape(na, 1);

  Teuchos::LAPACK<int, double> ll;

  int info = 0;
  ll.GETRF(ma, na, mat->values(), mat->stride(), work_temp->values(), &info);

  if (info != 0)
  {
    return false;
    // std::cout<<"info = "<<info<<"\n";
    // FOUR_C_THROW("Lu decomposition failed\n");
  }

  // Adapt the indices. GETRF uses Fortran implementation in the background
  // and hence return the vector with indices ranging from [1,n]
  // we convert it to [0,n-1] for further calculations
  for (int ii = 0; ii < na; ii++) (*work_temp)(ii) = (*work_temp)(ii)-1;

  std::vector<int> work(na, 0.0);

  get_pivotal_rows(work_temp, work);

  /*work_temp->print(std::cout);
  std::cout<<"work = ";
  for(int ii=0;ii<na;ii++)
    std::cout<<work[ii]<<" ";
  std::cout<<"\n";*/

  Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> sqrmat =
      Teuchos::RCP(new Core::LinAlg::SerialDenseMatrix);
  sqrmat->shape(na, na);

  for (int ptno = 0; ptno < na; ptno++)
  {
    int rowno = work[ptno];
    for (int ind = 0; ind < na; ind++)
    {
      (*sqrmat)(ind, ptno) = (*matTemp)(rowno, ind);
    }
  }

  sol->shape(na, 1);

  Teuchos::SerialDenseSolver<int, double> qr;
  qr.setMatrix(sqrmat);
  qr.setVectors(sol, rhs);

  int isSolved = qr.solve();
  if (isSolved != 0) FOUR_C_THROW("SerialDenseSolver failed\n");

  // rhs->print(std::cout);
  // sol->print(std::cout);

  gout_ = form_new_quadrature_rule(gin, sol, work, na);

  // compute_and_print_error( gin, rhs, sol, work, na );

  return true;
}

/*-------------------------------------------------------------------------------------------------------------------------*
 * Form new quadrature rule from the quadrature points and weights obtained from compression
 *sudhakar 08/15 (just creating the required data structure)
 *-------------------------------------------------------------------------------------------------------------------------*/
Teuchos::RCP<Core::FE::GaussPoints> Cut::QuadratureCompression::form_new_quadrature_rule(
    Core::FE::GaussPointsComposite& gin, Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol,
    std::vector<int>& work, int& na)
{
  Teuchos::RCP<Core::FE::CollectedGaussPoints> cgp =
      Teuchos::RCP(new Core::FE::CollectedGaussPoints(0));

  for (int pt = 0; pt < na; pt++)
  {
    double wei = (*sol)(pt);
    int quadNo = work[pt];
    const double* loc = gin.point(quadNo);

    cgp->append(loc[0], loc[1], loc[2], wei);
  }

  return cgp;
}

void Cut::QuadratureCompression::get_pivotal_rows(
    Teuchos::RCP<Core::LinAlg::IntSerialDenseVector>& work_temp, std::vector<int>& work)
{
  int na = (int)work.size();
  int index = 0;

  work[0] = (*work_temp)(0);

  for (int i = 1; i < na; i++)
  {
    int val = (*work_temp)(i);
    index = get_correct_index(val, work, i - 1);
    work[i] = index;
  }
}

bool Cut::QuadratureCompression::is_this_value_already_in_dense_vector(
    int& input, std::vector<int>& vec, int upper_range, int& index)
{
  index = 0;
  for (int i = upper_range; i >= 0; i--)
  {
    index = i;
    if (vec[i] == input) return true;
  }

  index = input;
  return false;
}

int Cut::QuadratureCompression::get_correct_index(
    int& input, std::vector<int>& vec, int upper_range)
{
  int index = 0;
  while (1)
  {
    bool isFound = is_this_value_already_in_dense_vector(input, vec, upper_range, index);
    if (not isFound)
      break;
    else
      input = index;
  }
  return index;
}

/*--------------------------------------------------------------------------------------------------------------------*
 * Compute error between original and compressed quadrature points to evalute defined      sudhakar
 *08/15 base functions (either Chebyshev or monomial) and print the maximum absolute error
 *--------------------------------------------------------------------------------------------------------------------*/
void Cut::QuadratureCompression::compute_and_print_error(Core::FE::GaussPointsComposite& gin,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol, std::vector<int>& work, int& na)
{
  /*-------------------------------------------------------------------*/
  // solution check
  std::vector<double> check(na);
  for (int pt = 0; pt < na; pt++) check[pt] = 0.0;

  for (int pt = 0; pt < na; pt++)
  {
    double wei = (*sol)(pt);
    int quadNo = work[pt];
    const double* loc = gin.point(quadNo);

    double x = loc[0];
    double y = loc[1];
    double z = loc[2];

#ifdef CHECBYSHEV  // Chebyshev polynomial basis
    double x2 = cos(2.0 * acos(x));
    double y2 = cos(2.0 * acos(y));
    double z2 = cos(2.0 * acos(z));

    double x3 = cos(3.0 * acos(x));
    double y3 = cos(3.0 * acos(y));
    double z3 = cos(3.0 * acos(z));

    double x4 = cos(4.0 * acos(x));
    double y4 = cos(4.0 * acos(y));
    double z4 = cos(4.0 * acos(z));

    double x5 = cos(5.0 * acos(x));
    double y5 = cos(5.0 * acos(y));
    double z5 = cos(5.0 * acos(z));
#else
    double x2 = x * x;
    double y2 = y * y;
    double z2 = z * z;

    double x3 = x2 * x;
    double y3 = y2 * y;
    double z3 = z2 * z;

    double x4 = x3 * x;
    double y4 = y3 * y;
    double z4 = z3 * z;

    double x5 = x4 * x;
    double y5 = y4 * y;
    double z5 = z4 * z;
#endif

    check[0] += wei;
    check[1] += x * wei;
    check[2] += y * wei;
    check[3] += z * wei;

    if (na > 4)
    {
      check[4] += x2 * wei;     // x^2
      check[5] += x * y * wei;  // xy
      check[6] += x * z * wei;  // xz
      check[7] += y2 * wei;     // y^2
      check[8] += y * z * wei;  // yz
      check[9] += z2 * wei;     // z^2
    }

    if (na > 10)
    {
      check[10] += x3 * wei;         // x^3
      check[11] += x2 * y * wei;     // x^2 y
      check[12] += x2 * z * wei;     // x^2 y
      check[13] += x * y2 * wei;     // xy^2
      check[14] += x * y * z * wei;  // xyz
      check[15] += x * z2 * wei;     // xz^2
      check[16] += y3 * wei;         // y^3
      check[17] += y2 * z * wei;     // y^2 z
      check[18] += y * z2 * wei;     // yz^2
      check[19] += z3 * wei;         // z^3
    }

    if (na > 20)
    {
      check[20] += x4 * wei;          // x^4
      check[21] += x3 * y * wei;      // x^3 y
      check[22] += x3 * z * wei;      // x^3 z
      check[23] += x2 * y2 * wei;     // x^2 y^2
      check[24] += x2 * y * z * wei;  // x^2 yz
      check[25] += x2 * z2 * wei;     // x^2 z^2
      check[26] += x * y3 * wei;      // xy^3
      check[27] += x * y2 * z * wei;  // xy^2 z
      check[28] += x * y * z2 * wei;  // xyz^2
      check[29] += x * z3 * wei;      // xz^3
      check[30] += y4 * wei;          // y^4
      check[31] += y3 * z * wei;      // y^3 z
      check[32] += y2 * z2 * wei;     // y^2 z^2
      check[33] += y * z3 * wei;      // yz^3
      check[34] += z4 * wei;          // z^4
    }

    if (na > 35)
    {
      check[35] += x5 * wei;           // x^5
      check[36] += x4 * y * wei;       // x^4 y
      check[37] += x4 * z * wei;       // x^4 z
      check[38] += x3 * y2 * wei;      // x^3 y^2
      check[39] += x3 * y * z * wei;   // x^3 yz
      check[40] += x3 * z2 * wei;      // x^3 z^2
      check[41] += x2 * y3 * wei;      // x^2 y^3
      check[42] += x2 * y2 * z * wei;  // x^2 y^2 z
      check[43] += x2 * y * z2 * wei;  // x^2 yz^2
      check[44] += x2 * z3 * wei;      // x^2 z^3
      check[45] += x * y4 * wei;       // xy^4
      check[46] += x * y3 * z * wei;   // xy^3 z
      check[47] += x * y2 * z2 * wei;  // x y^2 z^2
      check[48] += x * y * z3 * wei;   // xyz^3
      check[49] += x * z4 * wei;       // xz^4
      check[50] += y5 * wei;           // y^5
      check[51] += y4 * z * wei;       // y^4 z
      check[52] += y3 * z2 * wei;      // y^3 z^2
      check[53] += y2 * z3 * wei;      // y^2 z^3
      check[54] += y * z4 * wei;       // yz^4
      check[55] += z5 * wei;           // z^5
    }
  }
  double max_error = 0.0;
  std::vector<double> error(na);
  for (int pt = 0; pt < na; pt++)
  {
    error[pt] = fabs(check[pt] - (*rhs)(pt));
    if (error[pt] > max_error) max_error = error[pt];
  }

  if (max_error > 1e-10)
  {
    std::cout << "error in quadrature compression \n";
    for (int pt = 0; pt < na; pt++) std::cout << error[pt] << "\n";
  }
  std::cout << "===============Maximum error========" << max_error << "\n";
  /*-------------------------------------------------------------------*/
}

void Cut::QuadratureCompression::teuchos_gels(Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol)
{
  mat->shape(3, 4);
  rhs->shape(3, 1);
  sol->shape(4, 1);

  (*mat)(0, 0) = 5.0;
  (*mat)(0, 1) = 7.0;
  (*mat)(0, 2) = 4.0;
  (*mat)(0, 3) = 1.0;
  (*mat)(1, 0) = 7.0;
  (*mat)(1, 1) = 9.0;
  (*mat)(1, 2) = 8.0;
  (*mat)(2, 3) = 3.0;
  (*mat)(2, 0) = 2.0;
  (*mat)(2, 1) = 4.0;
  (*mat)(2, 2) = 10.0;
  (*mat)(2, 3) = 6.0;

  (*rhs)(0) = 17.0;
  (*rhs)(1) = 27.0;
  (*rhs)(2) = 22.0;
  (*sol)(0) = 0.0;
  (*sol)(1) = 0.0;
  (*sol)(2) = 0.0;
  (*sol)(3) = 0.0;

  int ma = mat->numRows();
  int na = mat->numCols();
  //  int mb = rhs->numRows();
  int nb = rhs->numCols();

  int lwork = 2 * na + (na + 1) * 64;

  Teuchos::RCP<Core::LinAlg::SerialDenseVector> work =
      Teuchos::RCP(new Core::LinAlg::SerialDenseVector);
  work->shape(na, 1);

  Teuchos::LAPACK<int, double> ll;

  int info = 0;
  ll.GELS('N', ma, na, nb, mat->values(), mat->stride(), rhs->values(), 1, work->values(), lwork,
      &info);

  rhs->print(std::cout);
  FOUR_C_THROW("done");
}

void Cut::QuadratureCompression::qr_decomposition_teuchos(
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol)
{
  Teuchos::SerialQRDenseSolver<int, double> qr;

  mat->shape(4, 3);
  rhs->shape(3, 1);
  sol->shape(4, 1);

  (*mat)(0, 0) = 5.0;
  (*mat)(0, 1) = 7.0;
  (*mat)(0, 2) = 4.0;
  (*mat)(0, 3) = 1.0;
  (*mat)(1, 0) = 7.0;
  (*mat)(1, 1) = 9.0;
  (*mat)(1, 2) = 8.0;
  (*mat)(2, 3) = 3.0;
  (*mat)(2, 0) = 2.0;
  (*mat)(2, 1) = 4.0;
  (*mat)(2, 2) = 10.0;
  (*mat)(2, 3) = 6.0;

  (*rhs)(0) = 17.0;
  (*rhs)(1) = 27.0;
  (*rhs)(2) = 22.0;

  qr.setMatrix(mat);
  qr.setVectors(sol, rhs);

  // qr.solveWithTranspose( true );
  qr.solveWithTransposeFlag(Teuchos::TRANS);
  int NotSolved = qr.solve();
  if (NotSolved != 0) FOUR_C_THROW("QR-factorization using Trilinos not successful\n");
}

void Cut::QuadratureCompression::qr_decomposition_lapack(
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs,
    Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol)
{
  int ma = mat->numRows();
  int na = mat->numCols();
  //  int mb = rhs->numRows();
  int nb = rhs->numCols();

  mat->shape(3, 4);
  rhs->shape(3, 1);
  sol->shape(4, 1);

  (*mat)(0, 0) = 5.0;
  (*mat)(0, 1) = 7.0;
  (*mat)(0, 2) = 4.0;
  (*mat)(0, 3) = 1.0;
  (*mat)(1, 0) = 7.0;
  (*mat)(1, 1) = 9.0;
  (*mat)(1, 2) = 8.0;
  (*mat)(2, 3) = 3.0;
  (*mat)(2, 0) = 2.0;
  (*mat)(2, 1) = 4.0;
  (*mat)(2, 2) = 10.0;
  (*mat)(2, 3) = 6.0;

  (*rhs)(0) = 17.0;
  (*rhs)(1) = 27.0;
  (*rhs)(2) = 22.0;
  (*sol)(0) = 0.0;
  (*sol)(1) = 0.0;
  (*sol)(2) = 0.0;
  (*sol)(3) = 0.0;

  Teuchos::RCP<Core::LinAlg::IntSerialDenseVector> jpvt =
      Teuchos::RCP(new Core::LinAlg::IntSerialDenseVector);
  Teuchos::RCP<Core::LinAlg::SerialDenseVector> tau =
      Teuchos::RCP(new Core::LinAlg::SerialDenseVector);
  Teuchos::RCP<Core::LinAlg::SerialDenseVector> work =
      Teuchos::RCP(new Core::LinAlg::SerialDenseVector);
  Teuchos::RCP<Core::LinAlg::SerialDenseVector> rwork =
      Teuchos::RCP(new Core::LinAlg::SerialDenseVector);
  jpvt->shape(na, 1);
  tau->shape(na, 1);
  work->shape(na, 1);
  rwork->shape(na, 1);

  int lwork = 2 * na + (na + 1) * 64;
  int info = 0;

  Teuchos::LAPACK<int, double> ll;
  ll.GEQP3(ma, na, mat->values(), mat->stride(), jpvt->values(), tau->values(), work->values(),
      lwork, rwork->values(), &info);
  if (info) FOUR_C_THROW("QR-decomposition with column pivoting failed\n");

  ll.ORMQR('L', 'T', ma, nb, na, mat->values(), mat->stride(), tau->values(), rhs->values(),
      rhs->stride(), work->values(), lwork, &info);

  int k = 0;
  for (k = 0; k < na; k++)
  {
    if ((*mat)(k, k) < 0.01 * (*mat)(0, 0)) break;
  }

  ll.TRTRS('U', 'N', 'N', na, nb, mat->values(), mat->stride(), rhs->values(), ma, &info);


  mat->print(std::cout);
  rhs->print(std::cout);

  /*DGEQPF_F77(ma, na, sqr->values(), &(sqr->stride()),
  jpvt->values(), tau->values(), work->values(), &info);*/

  FOUR_C_THROW("done");
}

/*------------------------------------------------------------------------------------------------------------------------------------*
 * Write volumecell geometry, original quadrature points and compressed points        sudhakar 08/15
 * into GMSH output
 *------------------------------------------------------------------------------------------------------------------------------------*/
void Cut::QuadratureCompression::write_compressed_quadrature_gmsh(
    Core::FE::GaussPointsComposite& gin, Cut::VolumeCell* vc)
{
  static int sideno = 0;
  sideno++;

  // when set true, the original and compressed quadrature rules are
  // written in separate text files
  bool outputQuadRule = true;

  // Write Geometry
  std::stringstream str;
  str << "compressedCells" << sideno << "_" << vc->parent_element()->id() << ".pos";
  std::ofstream file(str.str().c_str());
  vc->dump_gmsh(file);

  std::stringstream strc;
  strc << "compressedPts" << sideno << "_" << vc->parent_element()->id() << ".pos";
  std::ofstream filec(strc.str().c_str());

  std::stringstream stro;
  stro << "originalPts" << sideno << "_" << vc->parent_element()->id() << ".pos";
  std::ofstream fileo(stro.str().c_str());

  //-----------------------------------------------------------------------
  // write original Gauss points
  file << "Geometry.PointSize=8.0;\n";  // Increase the point size
  // file<<"Geometry.PointType=1;\n";
  file << "View \"Original points \" {\n";

  int numpts = gin.num_points();
  for (int npt = 0; npt < numpts; npt++)
  {
    const double* loc = gin.point(npt);
    file << "SP(" << loc[0] << "," << loc[1] << "," << loc[2] << ","
         << "1"
         << "){0.0};" << std::endl;

    if (outputQuadRule)
    {
      fileo << loc[0] << " " << loc[1] << " " << loc[2] << " " << gin.weight(npt);
    }
  }
  file << "};\n";
  file << "View[PostProcessing.NbViews-1].ColorTable = { {0,0,100} };\n";  // Changing color to red
  file << "View[PostProcessing.NbViews-1].Light=0;\n";                     // Disable the lighting
  file << "View[PostProcessing.NbViews-1].ShowScale=0;\n";                 // Disable legend
  file << "View[PostProcessing.NbViews-1].PointSize = 7.0;\n";             // increase point size
  //-----------------------------------------------------------------------

  //-----------------------------------------------------------------------
  // write compressed Gauss points
  file << "Geometry.PointSize=8.0;\n";  // Increase the point size
  // file<<"Geometry.PointType=1;\n";
  file << "View \"Compressed points \" {\n";

  int num = gout_->num_points();
  for (int npt = 0; npt < num; npt++)
  {
    const double* loc = gout_->point(npt);
    file << "SP(" << loc[0] << "," << loc[1] << "," << loc[2] << ","
         << "1"
         << "){0.0};" << std::endl;

    if (outputQuadRule)
    {
      filec << loc[0] << " " << loc[1] << " " << loc[2] << " " << gout_->weight(npt);
    }
  }
  file << "};\n";
  file << "View[PostProcessing.NbViews-1].ColorTable = { {100,0,0} };\n";  // Changing color to red
  file << "View[PostProcessing.NbViews-1].Light=0;\n";                     // Disable the lighting
  file << "View[PostProcessing.NbViews-1].ShowScale=0;\n";                 // Disable legend
  file << "View[PostProcessing.NbViews-1].PointSize = 6.0;\n";             // increase point size
  file << "View[PostProcessing.NbViews-1].PointType=1;\n";                 // show points as sphere
  file << "View[PostProcessing.NbViews-1].Light=1;\n";                     // enable lighting
  //-----------------------------------------------------------------------

  // Compute error of predefined polynomial functions
  integrate_predefined_polynomials(gin);
}

/*---------------------------------------------------------------------------------------------------------------------*
 * Integrate predefined polynomial functions using original and compressed quadrature and compute
 *error         sudhakar 08/15 between these values
 *---------------------------------------------------------------------------------------------------------------------*/
void Cut::QuadratureCompression::integrate_predefined_polynomials(
    Core::FE::GaussPointsComposite& gin)
{
  std::vector<double> intOri(6, 0.0), intCom(6, 0.0), err(6, 0.0);

  // Integrate using original quadrature
  int numpts = gin.num_points();
  for (int npt = 0; npt < numpts; npt++)
  {
    const double* loc = gin.point(npt);
    double wei = gin.weight(npt);

    double x = loc[0];
    double y = loc[1];
    double z = loc[2];

    // integrate constant 1 --> for volume comparison
    intOri[0] += wei;

    // integrate x+2y+3z
    intOri[1] += (x + 2 * y + 3 * z) * wei;

    // integrate x^2-2y^2+z^2
    intOri[2] += (x * x - 2 * y * y + z * z) * wei;

    // integrate -x^3+xyz+y^3+z^3
    intOri[3] += (-pow(x, 3) + x * y * z + pow(y, 3) + pow(z, 3)) * wei;

    // integrate x^4-4y^4+7xz^3+z^4
    intOri[4] += (pow(x, 4) - 4.0 * pow(y, 4) + 7.0 * x * pow(z, 3) + pow(z, 4)) * wei;

    // integrate x^5+5*xyz^3-10*xy^3z+5x^3yz+y^5+z^5
    intOri[5] += (pow(x, 5) + 5.0 * x * y * pow(z, 3) - 10.0 * x * pow(y, 3) * z +
                     5.0 * pow(x, 3) * y * z + pow(y, 5) + pow(z, 5)) *
                 wei;
  }

  // Integrate using compressed quadrature
  int num = gout_->num_points();
  for (int npt = 0; npt < num; npt++)
  {
    const double* loc = gout_->point(npt);
    double wei = gout_->weight(npt);

    double x = loc[0];
    double y = loc[1];
    double z = loc[2];

    // integrate constant 1 --> for volume comparison
    intCom[0] += wei;

    // integrate x+2y+3z
    intCom[1] += (x + 2 * y + 3 * z) * wei;

    // integrate x^2-2y^2+z^2
    intCom[2] += (x * x - 2 * y * y + z * z) * wei;

    // integrate -x^3+xyz+y^3+z^3
    intCom[3] += (-pow(x, 3) + x * y * z + pow(y, 3) + pow(z, 3)) * wei;

    // integrate x^4-4y^4+7xz^3+z^4
    intCom[4] += (pow(x, 4) - 4.0 * pow(y, 4) + 7.0 * x * pow(z, 3) + pow(z, 4)) * wei;

    // integrate x^5+5*xyz^3-10*xy^3z+5x^3yz+y^5+z^5
    intCom[5] += (pow(x, 5) + 5.0 * x * y * pow(z, 3) - 10.0 * x * pow(y, 3) * z +
                     5.0 * pow(x, 3) * y * z + pow(y, 5) + pow(z, 5)) *
                 wei;
  }

  // Compute error
  for (int ii = 0; ii < 6; ii++) err[ii] = fabs(intOri[ii] - intCom[ii]);

  // print error
  std::cout << "Errors in integration of predefined polynomials \n";
  for (int ii = 0; ii < 6; ii++) std::cout << err[ii] << "\n";
}

FOUR_C_NAMESPACE_CLOSE
