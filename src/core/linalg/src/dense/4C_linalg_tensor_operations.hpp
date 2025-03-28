// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_OPERATIONS_HPP
#define FOUR_C_LINALG_TENSOR_OPERATIONS_HPP

#include "4C_config.hpp"

#include "4C_linalg_tensor.hpp"

#include <cstddef>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include <Fastor/Fastor.h>
#pragma clang diagnostic pop

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  void matmul();

  template <typename Number, std::size_t n>
  Number det(Tensor<Number, n, n>& A)
  {
    return Fastor::_det<Number, n, n>(A.data());
  }

  template <typename Number, std::size_t n>
  Number trace(Tensor<Number, n, n>& A)
  {
    return Fastor::_trace<Number, n, n>(A.data());
  }

  template <typename Number, std::size_t n>
  Tensor<Number, n, n> inv(Tensor<Number, n, n>& A)
  {
    Tensor<Number, n, n> dest;
    Fastor::_inverse<Number, n>(A.data(), dest.data());
    return dest;
  }

  template <typename Number, std::size_t m, std::size_t n>
  Tensor<Number, n, m> transpose(Tensor<Number, m, n>& A)
  {
    Tensor<Number, n, m> dest;
    Fastor::_transpose<Number, m, n>(A.data(), dest.data());
    return dest;
  }

  template <typename Number, std::size_t m, std::size_t n>
  Tensor<Number, m> contraction(const Tensor<Number, m, n>& A, const Tensor<Number, n>& b)
  {
    Tensor<Number, m> dest;
    Fastor::_matmul<Number, m, n, 1>(A.data(), b.data(), dest.data());
    return dest;
  }

  template <typename Number, std::size_t m, std::size_t n>
  Tensor<Number, m, n> contraction(const Tensor<Number, m>& a, const Tensor<Number, m, n>& B)
  {
    Tensor<Number, n> dest;
    Fastor::_matmul<Number, 1, m, n>(a.data(), B.data(), dest.data());
    return dest;
  }

  template <typename Number, std::size_t m, std::size_t k, std::size_t n>
  Tensor<Number, m, n> contraction(const Tensor<Number, m, k>& A, const Tensor<Number, k, n>& B)
  {
    Tensor<Number, n, m> dest;
    Fastor::_matmul<Number, m, k, n>(A.data(), B.data(), dest.data());
    return dest;
  }

  template <typename Number, std::size_t n1, std::size_t n2>
  Number double_contraction(const Tensor<Number, n1, n2>& A, const Tensor<Number, n1, n2>& B)
  {
    return Fastor::_doublecontract<Number, n1, n2>(A.data(), B.data());
  }

  template <typename Number, std::size_t n1, std::size_t n2>
  Number double_contraction_nt(const Tensor<Number, n1, n2>& A, const Tensor<Number, n1, n2>& B)
  {
    return Fastor::_doublecontract_transpose<Number, n1, n2>(A.data(), B.data());
  }

  template <typename Number, std::size_t m_0, std::size_t n_0, std::size_t m_1, std::size_t n_1>
  Tensor<Number, m_0, n_0, m_1, n_1> outer(
      const Tensor<Number, m_0, n_0>& A, const Tensor<Number, m_1, n_1>& B)
  {
    Tensor<Number, m_0, n_0, m_1, n_1> dest;
    Fastor::_outer<Number, m_0, n_0, m_1, n_1>(A.data(), B.data(), dest.data());
    return dest;
  }

  template <typename Number, std::size_t n_0, std::size_t n_1>
  Tensor<Number, n_0, n_1> outer(const Tensor<Number, n_0>& a, const Tensor<Number, n_1>& b)
  {
    Tensor<Number, n_0, n_1> dest;
    Fastor::_outer<Number, n_0, n_1>(a.data(), b.data(), dest.data());
    return dest;
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif