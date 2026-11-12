// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_fixedsizematrix.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

FOUR_C_NAMESPACE_OPEN

const double* Core::LinAlg::Internal::values(const Core::LinAlg::SerialDenseMatrix& matrix)
{
  return matrix.values();
}

double* Core::LinAlg::Internal::values(Core::LinAlg::SerialDenseMatrix& matrix)
{
  return matrix.values();
}


const double* Core::LinAlg::Internal::values(const Core::LinAlg::SerialDenseVector& vector)
{
  return vector.values();
}

double* Core::LinAlg::Internal::values(Core::LinAlg::SerialDenseVector& vector)
{
  return vector.values();
}

FOUR_C_NAMESPACE_CLOSE
