// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_DENSEMATRIX_EIGEN_HPP
#define FOUR_C_LINALG_UTILS_DENSEMATRIX_EIGEN_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

#include <Teuchos_LAPACK.hpp>

#include <complex>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   \brief Compute all eigenvalues of a real symmetric matrix A

   \param A (in):        Matrix to be analysed
   \param L (out):       Vector of eigenvalues in ascending order
   \param postproc (in): flag indicating whether we are using this
   routine for postprocessing only (in that
   case FOUR_C_THROW is replaced with a warning)
   */
  void symmetric_eigen_values(Core::LinAlg::SerialDenseMatrix& A,
      Core::LinAlg::SerialDenseVector& L, const bool postproc = false);

  /*!
   \brief Compute all eigenvalues and eigenvectors of a real symmetric matrix A

   \param A (in/out):    in: Matrix to be analysed, out: eigenvectors
   (i.e. original matrix is destroyed!!!)
   \param L (out):       Vector of eigenvalues in ascending order
   \param postproc (in): flag indicating whether we are using this
   routine for postprocessing only (in that
   case FOUR_C_THROW is replaced with a warning)
   */
  void symmetric_eigen_problem(Core::LinAlg::SerialDenseMatrix& A,
      Core::LinAlg::SerialDenseVector& L, const bool postproc = false);

  /*!
   \brief Compute all eigenvalues and, optionally, eigenvectors
   of a real symmetric matrix A

   \param A (in/out):    Matrix to be analysed, if eigv=true stores eigenvectors
   \param L (in/out):    Vector of eigenvalues in ascending order
   \param eval_eigenvectors (in):     flag to evaluate also eigenvectors
   \param postproc (in): flag indicating whether we are using this
   routine for postprocessing only (in that
   case FOUR_C_THROW is replaced with a warning)
   */
  void symmetric_eigen(Core::LinAlg::SerialDenseMatrix& A, Core::LinAlg::SerialDenseVector& L,
      bool eval_eigenvectors, bool postproc = false);

  /*!
   \brief Compute and return all eigenvalues of the generalized eigenvalue problem
   Ax = lambda Bx via QZ-algorithm (B is singular).

   \param A (in):    A Matrix
   \param B (in):    B Matrix
   */
  std::vector<std::complex<double>> generalized_eigen(
      Core::LinAlg::SerialDenseMatrix& A, Core::LinAlg::SerialDenseMatrix& B);

  /*!
   \brief Compute and return the maximum eigenvalue (only real part) of the generalized eigenvalue
   problem Ax = lambda Bx via QZ-algorithm (B is singular).

   \param A (in):    A Matrix
   \param B (in):    B Matrix
   */
  double generalized_eigen_max_real_eigenvalue(
      Core::LinAlg::SerialDenseMatrix& A, Core::LinAlg::SerialDenseMatrix& B);

  /*!
   \brief Compute all eigenvalues and eigenvectors of a real symmetric matrix A

   A = V * S * VT

   \param A (in):        M-by-M matrix to be decomposed
   \param S (out):       M-by-M matrix which is zero except for its diagonal entries holding the
   eigenvalues \param V (out):       M-by-M orthonormal matrix of eigenvectors
   */
  template <unsigned int dim>
  void syev(const Core::LinAlg::Matrix<dim, dim>& A, Core::LinAlg::Matrix<dim, dim>& S,
      Core::LinAlg::Matrix<dim, dim>& V)
  {
    // ----- settings for eigendecomposition ----- //

    // eigenvalues only or eigenvalues + eigenvectors?
    const char jobz = 'V';  // compute eigenvalues and eigenvectors

    // store upper triangle of A
    const char uplo = 'U';

    // order of the matrix A
    const int N = dim;

    // copy contents of the matrix A, since it will be destroyed
    Matrix<dim, dim> tmp(A.data(), false);

    // leading dimension of the array A
    const int lda = dim;

    // eigenvalues in ascending order
    std::array<double, dim> w;

    // further settings needed for the lapack routine
    const int lwork = 2 * dim * dim + 6 * dim + 1;
    std::array<double, lwork> work;
    int info;

    // ----- perform eigendecomposition ----- //
    Teuchos::LAPACK<int, double> lapack;
    lapack.SYEV(jobz, uplo, N, tmp.data(), lda, w.data(), work.data(), lwork, &info);

    FOUR_C_ASSERT_ALWAYS(info == 0, "Lapack's SYEV returned {}", info);

    // return eigenvectors
    V.update(tmp);

    // return eigenvalues
    S.clear();
    for (unsigned int i = 0; i < dim; ++i) S(i, i) = w[i];

    return;
  }

  /*!
   * \brief Compute all (generally complex) eigenvalues and eigenvectors of a real general, not
   * necessarily symmetric matrix A
   *
   * A = V * S * VT
   * \note the eigenvalues are not sorted!
   *
   * \param A (in):        M-by-M matrix to be decomposed
   * \param S (out):       M-by-M matrix which is zero except for its diagonal entries holding the
   * eigenvalues
   * \param V (out):       M-by-M orthonormal matrix of eigenvectors
   */
  template <unsigned int dim>
  void geev(const Core::LinAlg::Matrix<dim, dim, double>& A,
      Core::LinAlg::Matrix<dim, dim, std::complex<double>>& S,
      Core::LinAlg::Matrix<dim, dim, std::complex<double>>& V)
  {
    // ----- settings for eigendecomposition ----- //

    // set which eigenvectors to compute
    const char jobvl = 'N';  // do not compute left eigenvectors
    const char jobvr = 'V';  // compute only right eigenvectors

    // order of the matrix A
    const int N = dim;

    // copy contents of the matrix A, since it will be destroyed
    Matrix<dim, dim> tmp(A.data(), false);

    // leading dimension of the array A
    const int lda = dim;  //  LDA >=max(1,N)

    // real parts of eigenvalues
    std::array<double, dim> wr;

    // imaginary parts of eigenvalues
    std::array<double, dim> wi;

    // initialize eigenvectors (left and right)
    const int ldvl = dim;
    std::array<double, ldvl * N> vl;
    const int ldvr = dim;
    std::array<double, ldvr * N> vr;

    // further settings needed for the lapack routine
    const int lwork = 2 * dim * dim + 6 * dim + 1;
    std::array<double, lwork> work;
    int info;

    // ----- perform eigendecomposition ----- //
    Teuchos::LAPACK<int, double> lapack;
    lapack.GEEV(jobvl, jobvr, N, tmp.data(), lda, wr.data(), wi.data(), vl.data(), ldvl, vr.data(),
        ldvr, work.data(), lwork, &info);

    FOUR_C_ASSERT_ALWAYS(info == 0, "Lapack's GEEV returned {}", info);

    // save the temporary right eigenvectors, which are now in a "real" format instead of their
    // general complex form with complex conjugate pairs
    Matrix<dim, dim> temp_V(vr.data());

    // build the complex V matrix (complex eigenvector matrix) from the "real" eigenvector matrix
    unsigned int i = 0;
    while (i < dim)
    {
      if (std::abs(wi[i]) > 0.0)
      {
        // for complex eigenvalues: these come in complex conjugate eigenpairs, and geev sorts them
        // as to get the i-th complex eigenvector \f$ \boldsymbol{v}(i) \f$ from the computed real
        // eigenmatrix \f$ \boldsymbol{V}\f$ via
        //  \f$ \boldsymbol{v}(i) = \boldsymbol{V}(:, i) +  i \boldsymbol{V}(:, i + 1)  \f$ along
        //  with
        //  \f$ \boldsymbol{v}(i+1) = \boldsymbol{V}(:, i) -  i \boldsymbol{V}(:, i + 1)  \f$,
        //  whereby the i-th and (i+1)-th eigenvalues are complex conjugate
        for (unsigned int j = 0; j < dim; ++j)
        {
          V(j, i) = std::complex(temp_V(j, i), temp_V(j, i + 1));
          V(j, i + 1) = std::complex(temp_V(j, i), -temp_V(j, i + 1));
        }

        // increment column index by 2, as both conjugate eigenpairs were already considered
        i += 2;
      }
      else
      {
        // for real eigenvalues: the corresponding eigenvector in V is also real
        for (unsigned int j = 0; j < dim; ++j)
        {
          V(j, i) = std::complex(temp_V(j, i), 0.0);
        }

        // increment column index by 1
        i += 1;
      }
    }

    // return eigenvalues
    S.clear();
    for (unsigned int j = 0; j < dim; ++j) S(j, j) = std::complex<double>(wr[j], wi[j]);

    return;
  }


}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
