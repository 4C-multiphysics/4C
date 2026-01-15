// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_GENERATORS_HPP
#define FOUR_C_LINALG_TENSOR_GENERATORS_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"

#include <type_traits>



FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg::TensorGenerators
{
  /*!
   * @brief Diagonal 2-tensor with the given value on the diagonal.
   */
  template <std::size_t... n>
    requires(sizeof...(n) == 2 && std::array{n...}[0] == std::array{n...}[1])
  constexpr auto diagonal(const auto& value)
  {
    SymmetricTensor<std::remove_cvref_t<decltype(value)>, n...> t{};
    for (std::size_t i = 0; i < std::array{n...}[0]; ++i)
    {
      t(i, i) = value;
    }
    return t;
  }

  /*!
   * @brief Diagonal 2-tensor with the given values on the diagonal.
   */
  template <std::size_t n, typename T>
  constexpr SymmetricTensor<T, n, n> diagonal(const std::array<T, n>& values)
  {
    SymmetricTensor<T, n, n> t{};
    for (std::size_t i = 0; i < n; ++i)
    {
      t(i, i) = values[i];
    }
    return t;
  }

  /*!
   * @brief Tensor with every value given by @p value.
   */
  template <std::size_t... n>
    requires(sizeof...(n) == 2 && std::array{n...}[0] == std::array{n...}[1])
  constexpr auto full(const auto& value)
  {
    SymmetricTensor<std::remove_cvref_t<decltype(value)>, n...> t;
    t.fill(value);
    return t;
  }

  /*!
   * @brief Tensor with every value given by @p value.
   */
  template <std::size_t... n>
    requires(sizeof...(n) != 2 || std::array{n...}[0] != std::array{n...}[1])
  constexpr auto full(const auto& value)
  {
    Tensor<std::remove_cvref_t<decltype(value)>, n...> t;
    t.fill(value);
    return t;
  }

  /*!
   * @brief Identity tensor with given dimensions. Might be a 2-tensor or a 4-tensor.
   */
  template <typename T, std::size_t... n>
    requires((sizeof...(n) == 2 && std::array{n...}[0] == std::array{n...}[1]) ||
                (sizeof...(n) == 4 && std::array{n...}[0] == std::array{n...}[1] &&
                    std::array{n...}[0] == std::array{n...}[2] &&
                    std::array{n...}[0] == std::array{n...}[3]))
  static constexpr auto identity = []() consteval
  {
    constexpr std::size_t rank = sizeof...(n);
    if constexpr (rank == 2)
    {
      return diagonal<n...>(static_cast<T>(1));
    }
    else if constexpr (rank == 4)
    {
      constexpr std::array n_arr = {n...};
      Tensor<double, n...> t{};
      for (std::size_t ik = 0; ik < n_arr[0]; ++ik)
      {
        for (std::size_t jl = 0; jl < n_arr[1]; ++jl)
        {
          t(ik, jl, ik, jl) = T(1);
        }
      }
      return t;
    }
  }();

  /*!
   * @brief Symmetric 4th-order identity tensor
   */
  template <typename T, std::size_t n1, std::size_t n2, std::size_t n3, std::size_t n4>
    requires(n1 == n2 && n1 == n3 && n1 == n4)
  static constexpr SymmetricTensor<T, n1, n2, n3, n4> symmetric_identity =
      Core::LinAlg::assume_symmetry(
          0.5 * (identity<T, n1, n2, n3, n4> +
                    Core::LinAlg::reorder_axis<0, 1, 3, 2>(identity<T, n1, n2, n3, n4>)));

  template <typename T, std::size_t... n>
  static constexpr SymmetricTensor<T, n...> ones = full<n...>(static_cast<T>(1));
}  // namespace Core::LinAlg::TensorGenerators

FOUR_C_NAMESPACE_CLOSE

#endif