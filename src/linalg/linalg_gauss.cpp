/*----------------------------------------------------------------------*/
/*! \file

\brief Gauss elimination for small nxn systems

\level 1

*/
/*----------------------------------------------------------------------*/

#include "linalg_gauss.H"
#include <stdexcept>
#include <cmath>

#include "../drt_cut/cut_clnwrapper.H"

namespace LINALG
{
  /*!
    \brief computes a Gaussian elimination for a linear system of equations

    \tparam do_piv   (in)    : do_piv = true does pivoting, do_piv = false does not do pivoting
    \tparam dim      (in)    : dimension of the matrix
    \return determinant of system matrix
  */
  template <bool do_piv, unsigned dim, typename valtype>
  valtype gaussElimination(LINALG::Matrix<dim, dim, valtype>& A,  ///< (in)    : system matrix
      LINALG::Matrix<dim, 1, valtype>& b,                         ///< (in)    : right-hand-side
      LINALG::Matrix<dim, 1, valtype>& x                          ///< (out)   : solution vector
  )
  {
    if (dim > 1)
    {
      bool changesign = false;
      if (not do_piv)
      {
        for (unsigned k = 0; k < dim; ++k)
        {
          A(k, k) = 1. / A(k, k);

          for (unsigned i = k + 1; i < dim; ++i)
          {
            A(i, k) *= A(k, k);
            x(i) = A(i, k);

            for (unsigned j = k + 1; j < dim; ++j)
            {
              A(i, j) -= A(i, k) * A(k, j);
            }
          }

          for (unsigned i = k + 1; i < dim; ++i)
          {
            b(i) -= x(i) * b(k);
          }
        }
      }
      else
      {
        for (unsigned k = 0; k < dim; ++k)
        {
          unsigned pivot = k;

          // search for pivot element
          for (unsigned i = k + 1; i < dim; ++i)
          {
            pivot = (std::abs(A(pivot, k)) < std::abs(A(i, k))) ? i : pivot;
          }

          // exchange pivot row and current row
          if (pivot != k)
          {
            for (unsigned j = 0; j < dim; ++j)
            {
              std::swap(A(k, j), A(pivot, j));
            }
            std::swap(b(k, 0), b(pivot, 0));
            changesign = not changesign;
          }

          if (A(k, k) == 0.0)
          {
            return 0.0;
          }

          A(k, k) = 1. / A(k, k);

          for (unsigned i = k + 1; i < dim; ++i)
          {
            A(i, k) *= A(k, k);
            x(i, 0) = A(i, k);

            for (unsigned j = k + 1; j < dim; ++j)
            {
              A(i, j) -= A(i, k) * A(k, j);
            }
          }

          for (unsigned i = k + 1; i < dim; ++i)
          {
            b(i, 0) -= x(i, 0) * b(k, 0);
          }
        }
      }

      // back substitution
      x(dim - 1, 0) = b(dim - 1, 0) * A(dim - 1, dim - 1);

      for (int i = dim - 2; i >= 0; --i)
      {
        for (int j = dim - 1; j > i; --j)
        {
          b(i, 0) -= A(i, j) * x(j, 0);
        }
        x(i, 0) = b(i, 0) * A(i, i);
      }
      valtype det = 1.0;
      for (unsigned i = 0; i < dim; ++i) det *= 1.0 / A(i, i);

      if (changesign) det *= -1.0;

      return det;
    }
    else
    {
      x(0, 0) = b(0, 0) / A(0, 0);
      return x(0, 0);
    }
  }

  template GEO::CUT::ClnWrapper gaussElimination<true, 3, GEO::CUT::ClnWrapper>(
      LINALG::Matrix<3, 3, GEO::CUT::ClnWrapper>& A,  ///< (in)    : system matrix
      LINALG::Matrix<3, 1, GEO::CUT::ClnWrapper>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<3, 1, GEO::CUT::ClnWrapper>& x   ///< (out)   : solution vector
  );
  template GEO::CUT::ClnWrapper gaussElimination<true, 2, GEO::CUT::ClnWrapper>(
      LINALG::Matrix<2, 2, GEO::CUT::ClnWrapper>& A,  ///< (in)    : system matrix
      LINALG::Matrix<2, 1, GEO::CUT::ClnWrapper>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<2, 1, GEO::CUT::ClnWrapper>& x   ///< (out)   : solution vector
  );
  template GEO::CUT::ClnWrapper gaussElimination<true, 1, GEO::CUT::ClnWrapper>(
      LINALG::Matrix<1, 1, GEO::CUT::ClnWrapper>& A,  ///< (in)    : system matrix
      LINALG::Matrix<1, 1, GEO::CUT::ClnWrapper>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<1, 1, GEO::CUT::ClnWrapper>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<true, 1, double>(
      LINALG::Matrix<1, 1, double>& A,  ///< (in)    : system matrix
      LINALG::Matrix<1, 1, double>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<1, 1, double>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<false, 1, double>(
      LINALG::Matrix<1, 1>& A,  ///< (in)    : system matrix
      LINALG::Matrix<1, 1>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<1, 1>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<true, 2, double>(
      LINALG::Matrix<2, 2>& A,  ///< (in)    : system matrix
      LINALG::Matrix<2, 1>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<2, 1>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<false, 2, double>(
      LINALG::Matrix<2, 2>& A,  ///< (in)    : system matrix
      LINALG::Matrix<2, 1>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<2, 1>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<true, 3, double>(
      LINALG::Matrix<3, 3>& A,  ///< (in)    : system matrix
      LINALG::Matrix<3, 1>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<3, 1>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<false, 3, double>(
      LINALG::Matrix<3, 3>& A,  ///< (in)    : system matrix
      LINALG::Matrix<3, 1>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<3, 1>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<true, 4, double>(
      LINALG::Matrix<4, 4>& A,  ///< (in)    : system matrix
      LINALG::Matrix<4, 1>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<4, 1>& x   ///< (out)   : solution vector
  );
  template double gaussElimination<false, 4, double>(
      LINALG::Matrix<4, 4>& A,  ///< (in)    : system matrix
      LINALG::Matrix<4, 1>& b,  ///< (in)    : right-hand-side
      LINALG::Matrix<4, 1>& x   ///< (out)   : solution vector
  );



  /*!
    \brief computes a Gaussian elimination for a linear system of equations after infnorm scaling

    \tparam dim      (in)    : dimension of the matrix
    \return determinant of system matrix
  */
  template <unsigned dim>
  double scaledGaussElimination(LINALG::Matrix<dim, dim>& A,  ///< (in)    : system matrix
      LINALG::Matrix<dim, 1>& b,                              ///< (in)    : right-hand-side
      LINALG::Matrix<dim, 1>& x                               ///< (out)   : solution vector
  )
  {
    // infnorm scaling
    for (unsigned i = 0; i < dim; ++i)
    {
      // find norm of max entry in row
      double max = std::abs(A(i, 0));
      for (unsigned j = 1; j < dim; ++j)
      {
        const double norm = std::abs(A(i, j));
        if (norm > max) max = norm;
      }

      // close to zero row detected -> matrix does probably not have full rank
      if (max < 1.0e-14)
      {
        return LINALG::gaussElimination<true, dim>(A, b, x);
      }

      // scale row with inv of max entry
      const double scale = 1.0 / max;
      for (unsigned j = 0; j < dim; ++j)
      {
        A(i, j) *= scale;
      }
      b(i) *= scale;
    }

    // solve scaled system using pivoting
    return LINALG::gaussElimination<true, dim>(A, b, x);
  }


  template double scaledGaussElimination<2>(LINALG::Matrix<2, 2>& A,  ///< (in)    : system matrix
      LINALG::Matrix<2, 1>& b,                                        ///< (in)    : right-hand-side
      LINALG::Matrix<2, 1>& x                                         ///< (out)   : solution vector
  );
  template double scaledGaussElimination<3>(LINALG::Matrix<3, 3>& A,  ///< (in)    : system matrix
      LINALG::Matrix<3, 1>& b,                                        ///< (in)    : right-hand-side
      LINALG::Matrix<3, 1>& x                                         ///< (out)   : solution vector
  );
  template double scaledGaussElimination<4>(LINALG::Matrix<4, 4>& A,  ///< (in)    : system matrix
      LINALG::Matrix<4, 1>& b,                                        ///< (in)    : right-hand-side
      LINALG::Matrix<4, 1>& x                                         ///< (out)   : solution vector
  );

}  // namespace LINALG
