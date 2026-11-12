// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_densefunctions.hpp"

#include <Teuchos_LAPACK.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

double Core::LinAlg::DenseFunctions::determinant_large_matrix(
    unsigned int i, unsigned int j, const double* mat)
{
  // taken from src/linalg/linalg_utils_densematrix_eigen.cpp: Core::LinAlg::DeterminantLU,
  // only with minor changes.
  std::vector<double> tmp(i * j);
  std::copy(mat, mat + i * j, tmp.data());
  std::vector<int> ipiv(j);
  int info;

  Teuchos::LAPACK<int, double> lapack;
  lapack.GETRF(i, j, tmp.data(), i, ipiv.data(), &info);

  if (info < 0)
    FOUR_C_THROW("Lapack's dgetrf returned {}", info);
  else if (info > 0)
    return 0.0;
  double d = tmp[0];
  for (unsigned int c = 1; c < j; ++c) d *= tmp[c + i * c];
  // swapping rows of A changes the sign of the determinant, so we have to
  // undo lapack's permutation w.r.t. the determinant
  // note the fortran indexing convention in ipiv
  for (unsigned int c = 0; c < j; ++c)
    if (static_cast<unsigned>(ipiv[c]) != c + 1) d *= -1.0;
  return d;
}

FOUR_C_NAMESPACE_CLOSE