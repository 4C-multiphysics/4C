// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FIXEDSIZEMATRIX_GENERATORS_HPP
#define FOUR_C_LINALG_FIXEDSIZEMATRIX_GENERATORS_HPP


#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <unsigned int size, typename ValueType = double>
  LinAlg::Matrix<size, size, ValueType> diagonal_matrix(const ValueType diagonal_value)
  {
    LinAlg::Matrix<size, size, ValueType> diag_matrix(Initialization::zero);
    for (unsigned int i = 0; i < size; ++i) diag_matrix(i, i) = diagonal_value;
    return diag_matrix;
  }

  template <unsigned int size, typename ValueType = double>
  LinAlg::Matrix<size, size, ValueType> identity_matrix()
  {
    return diagonal_matrix<size, ValueType>(1);
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
