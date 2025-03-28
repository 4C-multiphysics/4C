// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FADMATRIX_UTILS_HPP
#define FOUR_C_LINALG_FADMATRIX_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg_utils_densematrix_inverse.hpp"

#include <Sacado.hpp>

typedef Sacado::Fad::DFad<double> FAD;

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /// Serial dense matrix with FAD (automatic differentiation data type) numbers and templated
  /// dimensions
  template <unsigned int rows, unsigned int cols>
  class FADMatrix : public Matrix<rows, cols, FAD>
  {
   public:
    /// Constructor
    explicit FADMatrix(Initialization init = Initialization::zero) : Matrix<rows, cols, FAD>(init)
    {
    }

    /// Constructor
    explicit FADMatrix(double* d, bool view = false) : Matrix<rows, cols, FAD>(d, view) {}

    /// Constructor
    explicit FADMatrix(const double* d, bool view = false) : Matrix<rows, cols, FAD>(d, view) {}

    /// Constructor
    explicit FADMatrix(Core::LinAlg::SerialDenseMatrix& d, bool view = false)
        : Matrix<rows, cols, FAD>(d, view)
    {
    }

    /// Constructor
    FADMatrix(FADMatrix<rows, cols>& source, bool view) : Matrix<rows, cols, FAD>(source, view) {}

    /// Copy constructor
    FADMatrix(const FADMatrix<rows, cols>& source) : Matrix<rows, cols, FAD>(source) {}

    /// = operator
    inline FADMatrix<rows, cols>& operator=(FADMatrix<rows, cols> const& other)
    {
      Matrix<rows, cols, FAD>::operator=(other);
      return *this;
    }

    /// = operator
    inline FADMatrix<rows, cols>& operator=(Matrix<rows, cols, FAD> const& other)
    {
      Matrix<rows, cols, FAD>::operator=(other);
      return *this;
    }

    /// = operator
    inline FADMatrix<rows, cols>& operator=(Matrix<rows, cols, double> const& other)
    {
      for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = 0; j < cols; ++j) (*this)(i, j) = other(i, j);

      return *this;
    };

    /// Set all components of the matrix as independent variable
    inline void diff(const int pos,  ///< appending array of derivatives starts from this position
        const int n)                 ///< total length of derivative array
    {
      if (rows != cols) FOUR_C_THROW("diff does only work for quadratic matrices");
      for (unsigned i = 0; i < rows; ++i) (*this)(i, i).diff(pos + i, n);
      (*this)(0, 1).diff(pos + 2 + 1, n);
      (*this)(1, 2).diff(pos + 2 + 2, n);
      (*this)(0, 2).diff(pos + 2 + 3, n);
      (*this)(1, 0).diff(pos + 2 + 4, n);
      (*this)(2, 1).diff(pos + 2 + 5, n);
      (*this)(2, 0).diff(pos + 2 + 6, n);
    };

    /// Convert FADMatrix to Matrix<rows,cols,double>
    inline Matrix<rows, cols> convertto_double() const
    {
      Core::LinAlg::Matrix<rows, cols> tmp(Core::LinAlg::Initialization::zero);
      for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = 0; j < cols; ++j) tmp(i, j) = (*this)(i, j).val();

      return tmp;
    };
  };
}  // namespace Core::LinAlg


/// Save first derivatives in a 3x3 double matrix
template <typename T>
inline void first_deriv_to_matrix(FAD const& r_fad,  ///< FAD function
    Core::LinAlg::Matrix<3, 3, T>& out)              ///< First derivatives
{
  for (int i = 0; i < 3; ++i) out(i, i) = r_fad.dx(i);
  out(0, 1) = r_fad.dx(3);
  out(1, 2) = r_fad.dx(4);
  out(0, 2) = r_fad.dx(5);
  out(1, 0) = r_fad.dx(6);
  out(2, 1) = r_fad.dx(7);
  out(2, 0) = r_fad.dx(8);
};

FOUR_C_NAMESPACE_CLOSE

#endif
