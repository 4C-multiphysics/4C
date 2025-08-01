// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_SYMMETRIC_EINSTEIN_HPP
#define FOUR_C_LINALG_TENSOR_SYMMETRIC_EINSTEIN_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_einstein.hpp"

#include <bits/utility.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  namespace EinsteinHelper
  {
    template <std::size_t i, std::size_t j>
    struct IntegerPair
    {
      static constexpr std::size_t first = i;
      static constexpr std::size_t second = j;
    };

    template <typename... IntegerPairs>
    struct IntegerPairSequence
    {
    };

    /*!
     * @brief Returns the list of indices for a symmetric 2-tensor.
     *
     * Example for a symmetric 2-tensor with size 2: [(0,0), (1,1), (0,1)]
     * Example for a symmetric 2-tensor with size 3: [(0,0), (1,1), (2,2), (0,1), (1,2), (0,2)]
     */
    template <std::size_t size>
    consteval std::array<std::pair<std::size_t, std::size_t>, size*(size + 1) / 2>
    make_symmetric_integer_pairs()
    {
      std::array<std::pair<std::size_t, std::size_t>, size*(size + 1) / 2> pairs{};

      std::size_t index = 0;
      for (std::size_t offset = 0; offset < size; ++offset)
      {
        for (std::size_t i = 0; i < size - offset; ++i)
        {
          // i,i+offset
          pairs[index].first = i;
          pairs[index].second = i + offset;
          ++index;
        }
      }

      return pairs;
    }

    template <std::array integer_pairs, typename IndexSequence>
    struct SymmetricIntegerPairSequenceHelper;

    template <std::array integer_pairs, std::size_t... i>
    struct SymmetricIntegerPairSequenceHelper<integer_pairs,
        std::integer_sequence<std::size_t, i...>>
    {
      using type =
          IntegerPairSequence<IntegerPair<integer_pairs[i].first, integer_pairs[i].second>...>;
    };

    template <std::size_t size>
    using SymmetricIntegerPairSequence =
        SymmetricIntegerPairSequenceHelper<make_symmetric_integer_pairs<size>(),
            std::make_index_sequence<size*(size + 1) / 2>>::type;

    template <typename IntegerSequence, typename... IntegerPairSequences>
    struct ConstExprSymmetricMultiFor;

    template <std::size_t... i, typename... IntegerPairs, typename... IntegerPairSequences>
    struct ConstExprSymmetricMultiFor<std::integer_sequence<std::size_t, i...>,
        IntegerPairSequence<IntegerPairs...>, IntegerPairSequences...>
    {
      template <typename Action>
      static constexpr void multi_for(Action action)
      {
        (ConstExprSymmetricMultiFor<
             std::integer_sequence<std::size_t, i..., IntegerPairs::first, IntegerPairs::second>,
             IntegerPairSequences...>::multi_for(action),
            ...);
      }
    };

    template <std::size_t... i, typename... IntegerPairs>
    struct ConstExprSymmetricMultiFor<std::integer_sequence<std::size_t, i...>,
        IntegerPairSequence<IntegerPairs...>>
    {
      template <typename Action>
      static constexpr void multi_for(Action action)
      {
        (action(std::integral_constant<std::size_t, i>{}...,
             std::integral_constant<std::size_t, IntegerPairs::first>{},
             std::integral_constant<std::size_t, IntegerPairs::second>{}),
            ...);
      }
    };

    template <std::array einstein_indices, typename Sequence, typename... IntegerPairSequences>
    struct ConstExprSymmetricMultiForMakerHelper;

    template <std::array einstein_indices, std::size_t i, std::size_t j, std::size_t... k,
        typename... IntegerPairSequences>
    struct ConstExprSymmetricMultiForMakerHelper<einstein_indices,
        std::integer_sequence<std::size_t, i, j, k...>, IntegerPairSequences...>
    {
      static_assert(einstein_indices[i].size == einstein_indices[j].size);
      using type = ConstExprSymmetricMultiForMakerHelper<einstein_indices,
          std::integer_sequence<std::size_t, k...>,
          SymmetricIntegerPairSequence<einstein_indices[i].size>, IntegerPairSequences...>::type;
    };

    template <std::array einstein_indices, typename... IntegerPairSequences>
    struct ConstExprSymmetricMultiForMakerHelper<einstein_indices,
        std::integer_sequence<std::size_t>, IntegerPairSequences...>
    {
      using type =
          ConstExprSymmetricMultiFor<std::integer_sequence<std::size_t>, IntegerPairSequences...>;
    };


    template <std::array einstein_indices>
    using ConstExprSymmetricMultiForMaker =
        typename ConstExprSymmetricMultiForMakerHelper<einstein_indices,
            std::make_index_sequence<std::size(einstein_indices)>>::type;

    template <std::array shape, typename T, typename IndexSequence>
    struct SymmetricTensorTypeDeducer;

    template <std::array shape, typename T, std::size_t... i>
    struct SymmetricTensorTypeDeducer<shape, T, std::integer_sequence<std::size_t, i...>>
    {
      using type = SymmetricTensor<T, shape[i]...>;
    };

    template <typename T, std::array shape>
    using SymmetricTensorTypeFromArray =
        typename SymmetricTensorTypeDeducer<shape, T, std::make_index_sequence<shape.size()>>::type;
  }  // namespace EinsteinHelper

  /**
   * @brief Performs a symmetric Einstein summation (einsum) over the provided tensor-like objects.
   *
   * @note This function assumes that the resulting tensor is symmetric (will not be checked)!
   *
   * This function implements a compile-time version of the symmetric Einstein summation convention,
   * allowing for flexible tensor contractions and dyadic products. The function supports
   * summing over repeated indices (contraction) and producing tensors with free indices
   * (dyadic indices). The implementation uses extensive constexpr metaprogramming to
   * validate index usage and sizes at compile time, ensuring correctness and efficiency.
   *
   * @code{.cpp}
   * auto tensor_result = einsum_sym<"ij", "ik", "kl">(F, C, F); // F^T C F
   * @endcode
   *
   * @return The result of the Einstein summation as a SymmetricTensor
   *
   * @note
   * - Each index must be used at most twice.
   * - All contraction indices must have the same size across all tensors.
   * - The resulting tensor must be symmetric (will not be checked!)
   * - Indices can be any character (typically i, j, k, ..., or a, b, c, ..., A, B, C, ... or 0, 1,
   * 2, ...).
   *
   * @throws Compilation error if index constraints are violated.
   */
  template <Utils::CompileTimeString... einstein_indices, typename... Tensor>
    requires(sizeof...(einstein_indices) == sizeof...(Tensor))
  auto einsum_sym(const Tensor&... t)
  {
    using TensorCompressionTypes = std::tuple<TensorCompressionType<Tensor>...>;

    constexpr auto all_einstein_indices =
        EinsteinHelper::extract_all_einstein_indices<einstein_indices...>(Tensor::shape()...);

    static_assert(EinsteinHelper::invalid_einstein_indices<all_einstein_indices>.size() == 0,
        "Invalid indices in Einstein summation. Each index must be used at most twice.");
    static_assert(EinsteinHelper::valid_contraction_einstein_indices<all_einstein_indices>,
        "All contraction indices must have the same size!");

    using value_type = FADUtils::ScalarOperationResultType<std::multiplies<>,
        typename std::remove_cvref_t<Tensor>::value_type...>;

    constexpr std::array contraction_einstein_indices =
        EinsteinHelper::contraction_einstein_indices<all_einstein_indices>;
    constexpr std::array dyadic_einstein_indices =
        EinsteinHelper::dyadic_einstein_indices<all_einstein_indices>;

    static_assert(
        std::size(dyadic_einstein_indices) == 2 || std::size(dyadic_einstein_indices) == 4,
        "The reslting tensor of a symmetric einstein summation must be of rank 2 or 4!");
    static_assert(dyadic_einstein_indices[0].size == dyadic_einstein_indices[1].size,
        "The first two indices of the resulting tensor must have the same size in a symmetric "
        "einstein summation (otherwise, it cannot be symmetric)!");
    static_assert(std::size(dyadic_einstein_indices) == 2 ||
                      dyadic_einstein_indices[2].size == dyadic_einstein_indices[3].size,
        "The last two dyadic indices of the tensor must have the same size in a symmetric "
        "einstein summation (otherwise, it cannot be symmetric)!");

    if constexpr (contraction_einstein_indices.size() == 0)
    {
      EinsteinHelper::SymmetricTensorTypeFromArray<value_type,
          EinsteinHelper::einstein_index_sizes<dyadic_einstein_indices>>
          tensor_result{};
      constexpr std::array<std::size_t, 0> contraction_index = {};

      EinsteinHelper::ConstExprSymmetricMultiForMaker<dyadic_einstein_indices>::multi_for(
          [&](auto... i)
          {
            constexpr std::array<std::size_t, sizeof...(i)> dyadic_index = {i()...};

            constexpr std::array tensor_flat_indices =
                EinsteinHelper::flat_indices<TensorCompressionTypes, contraction_einstein_indices,
                    dyadic_einstein_indices, contraction_index, dyadic_index,
                    einstein_indices.value...>;
            constexpr std::size_t flat_index =
                EinsteinHelper::flat_index<TensorCompressionType<decltype(tensor_result)>,
                    EinsteinHelper::get_tensor_component<
                        EinsteinHelper::einstein_index_names<dyadic_einstein_indices>,
                        contraction_einstein_indices, dyadic_einstein_indices>(
                        contraction_index, dyadic_index)>;

            *(tensor_result.data() + flat_index) +=
                EinsteinHelper::evaluate_contraction<tensor_flat_indices>(t...);
          });

      return tensor_result;
    }
    else
    {
      EinsteinHelper::SymmetricTensorTypeFromArray<value_type,
          EinsteinHelper::einstein_index_sizes<dyadic_einstein_indices>>
          tensor_result{};

      EinsteinHelper::ConstExprMultiForMaker<contraction_einstein_indices>::multi_for(
          [&](auto... j)
          {
            EinsteinHelper::ConstExprSymmetricMultiForMaker<dyadic_einstein_indices>::multi_for(
                [&](auto... i)
                {
                  constexpr std::array<std::size_t, sizeof...(j)> contraction_index = {j()...};
                  constexpr std::array<std::size_t, sizeof...(i)> dyadic_index = {i()...};

                  constexpr std::array tensor_flat_indices =
                      EinsteinHelper::flat_indices<TensorCompressionTypes,
                          contraction_einstein_indices, dyadic_einstein_indices, contraction_index,
                          dyadic_index, einstein_indices.value...>;
                  constexpr std::size_t flat_index =
                      EinsteinHelper::flat_index<TensorCompressionType<decltype(tensor_result)>,
                          EinsteinHelper::get_tensor_component<
                              EinsteinHelper::einstein_index_names<dyadic_einstein_indices>,
                              contraction_einstein_indices, dyadic_einstein_indices>(
                              contraction_index, dyadic_index)>;

                  *(tensor_result.data() + flat_index) +=
                      EinsteinHelper::evaluate_contraction<tensor_flat_indices>(t...);
                });
          });

      return tensor_result;
    }
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif