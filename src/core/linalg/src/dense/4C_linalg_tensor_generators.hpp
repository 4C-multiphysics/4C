// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_GENERATORS_HPP
#define FOUR_C_LINALG_TENSOR_GENERATORS_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"



FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T, std::size_t... n>
    requires(sizeof...(n) == 2 && std::array{n...}[0] == std::array{n...}[1])
  constexpr SymmetricTensor<T, n...> diagonal(const T& value)
  {
    SymmetricTensor<T, n...> t{};
    for (std::size_t i = 0; i < std::array{n...}[0]; ++i)
    {
      t(i, i) = value;
    }
    return t;
  }

  template <typename T, std::size_t n>
  constexpr SymmetricTensor<T, n, n> diagonal(const std::array<T, n>& values)
  {
    SymmetricTensor<T, n, n> t{};
    for (std::size_t i = 0; i < n; ++i)
    {
      t(i, i) = values[i];
    }
    return t;
  }

  template <typename T, std::size_t... n>
    requires(sizeof...(n) == 2 && std::array{n...}[0] == std::array{n...}[1])
  constexpr SymmetricTensor<T, n...> full(const T& value)
  {
    SymmetricTensor<T, n...> t;
    t.fill(value);
    return t;
  }

  template <typename T, std::size_t... n>
    requires(sizeof...(n) != 2 || std::array{n...}[0] != std::array{n...}[1])
  constexpr Tensor<T, n...> full(const T& value)
  {
    Tensor<T, n...> t;
    t.fill(value);
    return t;
  }

  /*!
   * @brief Identity tensor with given dimensions.
   */
  template <typename T, std::size_t... n>
    requires(sizeof...(n) == 2 && std::array{n...}[0] == std::array{n...}[1])
  static constexpr SymmetricTensor<T, n...> identity = diagonal<T, n...>(T(1));

  template <typename T, std::size_t... n>
  static constexpr SymmetricTensor<T, n...> ones = full<T, n...>(1);

  auto from_matrix(const auto& matrix)
  {
    using MatrixType = std::remove_cvref_t<decltype(matrix)>;
    Tensor<typename MatrixType::scalar_type, MatrixType::m(), MatrixType::n()> tensor;
    std::copy_n(matrix.data(), MatrixType::m() * MatrixType::n(), tensor.data());
    return tensor;
  }


  auto make_matrix_view(auto& tensor)
    requires(std::remove_cvref_t<decltype(tensor)>::rank() == 2)
  {
    using ValueType = std::remove_cvref_t<decltype(tensor)>::value_type;
    constexpr std::size_t n1 = std::remove_cvref_t<decltype(tensor)>::template extent<0>();
    constexpr std::size_t n2 = std::remove_cvref_t<decltype(tensor)>::template extent<1>();
    return Core::LinAlg::Matrix<n1, n2, ValueType>{tensor.data(), true};
  }

  template <std::size_t n1, std::size_t n2>
  auto make_matrix_view(auto& tensor)
    requires(std::remove_cvref_t<decltype(tensor)>::rank() == 1 &&
             n1 * n2 == std::remove_cvref_t<decltype(tensor)>::template extent<0>())
  {
    using ValueType = std::remove_cvref_t<decltype(tensor)>::value_type;
    return Core::LinAlg::Matrix<n1, n2, ValueType>{tensor.data(), true};
  }

  template <std::size_t n1, std::size_t n2, typename T>
  Core::LinAlg::Matrix<n1, n2, T> make_matrix(const Core::LinAlg::Tensor<T, n1, n2>& tensor)
  {
    return Core::LinAlg::Matrix<n1, n2, T>{tensor.data(), false};
  }

  template <std::size_t n1, std::size_t n2, typename T, std::size_t... n>
    requires((n1 * n2) == (n * ...) && sizeof...(n) == 1)
  Core::LinAlg::Matrix<n1, n2, T> make_matrix(const Core::LinAlg::Tensor<T, n...>& tensor)
  {
    return Core::LinAlg::Matrix<n1, n2, T>{tensor.data(), false};
  }

  auto make_stress_like_voigt_view(auto& tensor)
    requires(Internal::is_symmetric_tensor<decltype(tensor)>)
  {
    using ValueType = std::remove_cvref_t<decltype(tensor)>::value_type;
    constexpr std::size_t rank = std::remove_cvref_t<decltype(tensor)>::rank();
    static_assert(rank == 2 || rank == 4,
        "Tensor must be a symmetric tensor of rank 2 or 4 for Voigt notation");


    if constexpr (rank == 2)
    {
      constexpr std::size_t compressed_size =
          std::remove_cvref_t<decltype(tensor)>::compressed_size;
      return Core::LinAlg::Matrix<compressed_size, 1, ValueType>(tensor.data(), true);
    }
    else if constexpr (rank == 4)
    {
      constexpr std::size_t size_left = std::remove_cvref_t<decltype(tensor)>::template extent<0>();
      constexpr std::size_t size_right =
          std::remove_cvref_t<decltype(tensor)>::template extent<2>();
      return Core::LinAlg::Matrix<size_left*(size_left + 1) / 2, size_right*(size_right + 1) / 2,
          ValueType>(tensor.data(), true);
    }
  }

  template <typename T, std::size_t size>
  Core::LinAlg::Matrix<size*(size + 1) / 2, 1, T> make_strain_like_voigt_matrix(
      const Core::LinAlg::SymmetricTensor<T, size, size>& tensor)
  {
    Core::LinAlg::Matrix<size*(size + 1) / 2, 1, T> matrix(tensor.data());
    Voigt::Stresses::to_strain_like(matrix, matrix);
    return matrix;
  }

  template <typename T, std::size_t size>
  Core::LinAlg::SymmetricTensor<T, size, size> from_stress_like_voigt_matrix(
      const Core::LinAlg::Matrix<size*(size + 1) / 2, 1, T>& vector)
  {
    Core::LinAlg::SymmetricTensor<T, size, size> arr;
    std::copy_n(vector.data(), size * (size + 1) / 2, arr.data());
    return arr;
  }

  template <typename T, std::size_t size>
  Core::LinAlg::SymmetricTensor<T, size, size> from_stress_like_voigt_matrix(
      const Core::LinAlg::Matrix<size*(size + 1) / 2, size*(size + 1) / 2, T>& matrix)
  {
    constexpr auto n = size * (size + 1) / 2;
    Core::LinAlg::SymmetricTensor<T, size, size, size, size> arr;
    std::copy_n(matrix.data(), n * n, arr.data());
    return arr;
  }

  template <std::size_t... n>
  auto reinterpret_matrix(const auto& matrix)
      -> Tensor<typename std::remove_cvref_t<decltype(matrix)>::scalar_type, n...>
    requires((n * ...) == std::remove_cvref_t<decltype(matrix)>::num_cols() *
                              std::remove_cvref_t<decltype(matrix)>::num_rows())
  {
    Tensor<typename std::remove_cvref_t<decltype(matrix)>::scalar_type, n...> tensor;
    std::copy_n(matrix.data(), (n * ...), tensor.data());
    return tensor;
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif