// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_EINSTEIN_HPP
#define FOUR_C_LINALG_TENSOR_EINSTEIN_HPP

#include "4C_config.hpp"

#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_internals.hpp"
#include "4C_linalg_tensor_meta_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad_meta.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  namespace EinsteinHelper
  {
    template <typename... IntegerSequences>
    struct ConstExprMultiFor;

    template <std::size_t... i, std::size_t... j, typename... IntegerSequences>
    struct ConstExprMultiFor<std::integer_sequence<std::size_t, i...>,
        std::integer_sequence<std::size_t, j...>, IntegerSequences...>
    {
      template <typename Action>
      static constexpr void multi_for(Action action)
      {
        (ConstExprMultiFor<std::integer_sequence<std::size_t, i..., j>,
             IntegerSequences...>::multi_for(action),
            ...);
      }
    };

    template <std::size_t... i, std::size_t... j>
    struct ConstExprMultiFor<std::integer_sequence<std::size_t, i...>,
        std::integer_sequence<std::size_t, j...>>
    {
      template <typename Action>
      static constexpr void multi_for(Action action)
      {
        (action(
             std::integral_constant<std::size_t, i>{}..., std::integral_constant<std::size_t, j>{}),
            ...);
      }
    };

    template <std::size_t n, std::array<std::size_t, n> arr, std::size_t... i>
    consteval auto make_integer_sequence_helper(std::integer_sequence<std::size_t, i...> index)
    {
      return std::integer_sequence<std::size_t, arr[i]...>{};
    }

    template <std::array arr>
    using IntegerSequenceFromArray = decltype(make_integer_sequence_helper<arr.size(), arr>(
        std::make_index_sequence<arr.size()>()));


    template <std::array index_sizes, std::array indexes, typename Sequence>
    struct ConstExprMultiForMakerHelper;

    template <std::array index_sizes, std::array indexes, std::size_t... i>
    struct ConstExprMultiForMakerHelper<index_sizes, indexes,
        std::integer_sequence<std::size_t, i...>>
    {
      using type = ConstExprMultiFor<std::integer_sequence<std::size_t>,
          std::make_index_sequence<index_sizes[i]>...>;
    };


    template <std::array index_sizes, std::array indexes>
    using ConstExprMultiForMaker = typename ConstExprMultiForMakerHelper<index_sizes, indexes,
        IntegerSequenceFromArray<indexes>>::type;

    template <typename Tuple>
    consteval auto make_array(Tuple&& tuple)
    {
      constexpr auto get_array = [](auto&&... x)
      { return std::array{std::forward<decltype(x)>(x)...}; };
      return std::apply(get_array, std::forward<Tuple>(tuple));
    }

    template <auto arr, typename Predicate>
    constexpr std::array filter_array = []() consteval
    {
      constexpr std::size_t num_entries = []() consteval
      {
        Predicate predicate{};
        std::size_t num_entries = 0;
        std::ranges::for_each(
            arr, [&](auto i) { num_entries += static_cast<std::size_t>(predicate(i)); });
        return num_entries;
      }();

      std::array<typename std::remove_cvref_t<decltype(arr)>::value_type, num_entries>
          filtered_array{};


      std::size_t index = 0;
      std::ranges::for_each(arr,
          [&](auto i)
          {
            Predicate predicate{};
            if (predicate(i))
            {
              filtered_array[index] = i;
              ++index;
            }
          });

      return filtered_array;
    }();

    template <auto arr, auto target>
    constexpr std::array apply_unique_to = []() consteval
    {
      constexpr std::size_t num_unique = [](auto array) consteval
      {
        std::size_t count = 0;

        for (std::size_t i = 0; i < std::size(array); ++i)
        {
          bool already_seen = false;
          for (std::size_t j = 0; j < i; ++j)
          {
            if (i != j && array[i] == array[j])
            {
              already_seen = true;
              break;
            }
          }
          if (!already_seen)
          {
            ++count;
          }
        }

        return count;
      }(arr);

      std::array<typename std::remove_cvref_t<decltype(target)>::value_type, num_unique>
          unique_array{};

      std::vector<typename std::remove_cvref_t<decltype(target)>::value_type> values{};


      std::size_t unique_index = 0;
      for (std::size_t index = 0; index < arr.size(); ++index)
      {
        if (std::ranges::find(values, arr[index]) == values.end())
        {
          values.push_back(arr[index]);
          unique_array[unique_index] = target[index];
          ++unique_index;
        }
      }

      return unique_array;
    }();

    template <auto arr>
    constexpr std::array unique_array = apply_unique_to<arr, arr>;


    template <auto arr>
    constexpr std::array invalid_indexes = unique_array<
        filter_array<arr, decltype([](auto i) { return std::ranges::count(arr, i) > 2; })>>;


    template <auto arr>
    constexpr std::array contraction_indexes = unique_array<
        filter_array<arr, decltype([](auto i) { return std::ranges::count(arr, i) == 2; })>>;


    template <auto indexes, auto index_sizes>
    constexpr bool valid_contraction_index_sizes = []() consteval
    {
      for (std::size_t i = 0; i < indexes.size(); ++i)
      {
        std::size_t index = std::ranges::find(indexes, indexes[i]) - indexes.begin();
        if (index_sizes[index] != index_sizes[i])
        {
          return false;
        }
      }
      return true;
    }();

    template <auto arr>
    constexpr std::array dyadic_indexes = []() consteval
    {
      std::array dyadic_indexes = unique_array<
          filter_array<arr, decltype([](auto i) { return std::ranges::count(arr, i) == 1; })>>;

      std::ranges::sort(dyadic_indexes);
      return dyadic_indexes;
    }();

    template <std::array shape, typename T, typename IndexSequence>
    struct TensorTypeDeducer;

    template <std::array shape, typename T, std::size_t... i>
    struct TensorTypeDeducer<shape, T, std::integer_sequence<std::size_t, i...>>
    {
      using type = Tensor<T, shape[i]...>;
    };

    template <typename T, std::array shape>
    using TensorTypeFromArray =
        typename TensorTypeDeducer<shape, T, std::make_index_sequence<shape.size()>>::type;


    template <typename TensorCompression, std::array index>
    constexpr std::size_t flat_index = []() consteval
    {
      return [&]<std::size_t... i>(std::integer_sequence<std::size_t, i...> seq)
      {
        return TensorCompression::template flatten_index<Internal::TensorBoundCheck::no_check>(
            i...);
      }(IntegerSequenceFromArray<index>());
    }();

    template <std::size_t rank, std::size_t num_contraction_indexes, std::size_t num_dyadic_indexes>
    consteval std::array<std::size_t, rank> get_tensor_index(std::array<std::size_t, rank> indexes,
        std::array<std::size_t, num_contraction_indexes> contraction_indexes,
        std::array<std::size_t, num_dyadic_indexes> dyadic_indexes,
        std::array<std::size_t, num_contraction_indexes> contraction_index,
        std::array<std::size_t, num_dyadic_indexes> dyadic_index)
    {
      std::array<std::size_t, rank> this_tensor_index{};

      for (std::size_t i = 0; i < rank; ++i)
      {
        bool found = false;
        for (std::size_t j = 0; j < contraction_indexes.size(); ++j)
        {
          if (contraction_indexes[j] == indexes[i])
          {
            this_tensor_index[i] = contraction_index[j];
            found = true;
            break;
          }
        }
        if (found) continue;

        for (std::size_t j = 0; j < dyadic_indexes.size(); ++j)
        {
          if (dyadic_indexes[j] == indexes[i])
          {
            this_tensor_index[i] = dyadic_index[j];
            found = true;
            break;
          }
        }

        FOUR_C_ASSERT_ALWAYS(
            found, "Internal error: Could not find index in contraction or dyadic indexes.");
      }


      return this_tensor_index;
    }

    template <typename TensorCompressionTypes, std::array contraction_indexes,
        std::array dyadic_indexes, std::array contraction_index, std::array dyadic_index,
        typename T, std::array... einstein_indexes>
    struct FlatTensorIndexHelper;

    template <typename TensorCompressionTypes, std::array contraction_indexes,
        std::array dyadic_indexes, std::array contraction_index, std::array dyadic_index,
        std::array... einstein_indexes, std::size_t... i>
    struct FlatTensorIndexHelper<TensorCompressionTypes, contraction_indexes, dyadic_indexes,
        contraction_index, dyadic_index, std::integer_sequence<std::size_t, i...>,
        einstein_indexes...>
    {
      static constexpr auto value = []()
      {
        constexpr std::tuple einstein_index_tuple = {einstein_indexes...};
        return std::array{flat_index<std::tuple_element_t<i, TensorCompressionTypes>,
            get_tensor_index(std::get<i>(einstein_index_tuple), contraction_indexes, dyadic_indexes,
                contraction_index, dyadic_index)>...};
      }();
    };

    template <typename TensorCompressionTypes, std::array contraction_indexes,
        std::array dyadic_indexes, std::array contraction_index, std::array dyadic_index,
        std::array... einstein_indexes>
    constexpr std::array flat_indexes = []() consteval
    {
      return FlatTensorIndexHelper<TensorCompressionTypes, contraction_indexes, dyadic_indexes,
          contraction_index, dyadic_index,
          std::make_index_sequence<std::tuple_size_v<TensorCompressionTypes>>,
          einstein_indexes...>::value;
    }();

    template <typename FlatIndexes, typename... Tensor>
    struct ContractionEvaluationHelper;

    template <std::size_t first_index, typename FirstTensor, std::size_t... other_indexes,
        typename... OtherTensor>
    struct ContractionEvaluationHelper<
        std::integer_sequence<std::size_t, first_index, other_indexes...>, FirstTensor,
        OtherTensor...>
    {
      static constexpr auto evaluate_contraction(
          const FirstTensor& first_tensor, const OtherTensor&... other_tensors)
      {
        return *(first_tensor.data() + first_index) *
               ContractionEvaluationHelper<std::integer_sequence<std::size_t, other_indexes...>,
                   OtherTensor...>::evaluate_contraction(other_tensors...);
      }
    };

    template <std::size_t first_index, typename FirstTensor>
    struct ContractionEvaluationHelper<std::integer_sequence<std::size_t, first_index>, FirstTensor>
    {
      static constexpr auto evaluate_contraction(const FirstTensor& first_tensor)
      {
        return *(first_tensor.data() + first_index);
      }
    };

    template <std::array flat_indexes, typename... Tensor>
    auto evaluate_contraction(const Tensor&... t)
    {
      return ContractionEvaluationHelper<IntegerSequenceFromArray<flat_indexes>,
          Tensor...>::evaluate_contraction(t...);
    }

    template <std::array all_indexes, std::array all_index_sizes>
    constexpr std::array unique_index_sizes = []() consteval
    {
      constexpr std::array unique_indexes = unique_array<all_indexes>;
      constexpr std::array unique_index_sizes = apply_unique_to<all_indexes, all_index_sizes>;
      std::array<std::size_t, unique_indexes.size()> index_sizes{};
      for (std::size_t i = 0; i < unique_indexes.size(); ++i)
      {
        index_sizes[unique_indexes[i]] = unique_index_sizes[i];
      }
      return index_sizes;
    }();

    template <std::array dyadic_indexes, std::array index_sizes>
    constexpr std::array dyadic_index_sizes = []() consteval
    {
      std::array<std::size_t, dyadic_indexes.size()> dyadic_index_sizes{};
      for (std::size_t i = 0; i < dyadic_indexes.size(); ++i)
      {
        dyadic_index_sizes[i] = index_sizes[dyadic_indexes[i]];
      }
      return dyadic_index_sizes;
    }();

    template <size_t size_with_null_terminator>
    struct FixedString
    {
      static constexpr size_t size = size_with_null_terminator - 1;
      std::array<char, size> value{};

      constexpr FixedString(const char (&str)[size_with_null_terminator])
      {
        for (size_t i = 0; i < size; ++i) value[i] = str[i];
      }

      constexpr operator std::string_view() const { return {value.data(), size}; }

      constexpr auto operator<=>(const FixedString&) const = default;

      constexpr std::array<std::size_t, size> to_index_array(char min_value) const
      {
        std::array<std::size_t, size> index_array;
        std::transform(value.begin(), value.end(), index_array.begin(),
            [min_value](char c) { return static_cast<std::size_t>(c - min_value); });
        return index_array;
      }
    };

    template <EinsteinHelper::FixedString... einstein_indexes>
    constexpr char smallest_used_index =
        std::ranges::min(EinsteinHelper::make_array(std::tuple_cat(einstein_indexes.value...)));

    template <EinsteinHelper::FixedString... einstein_indexes>
    constexpr std::array all_used_indexes = []() consteval
    {
      constexpr char min_index = smallest_used_index<einstein_indexes...>;
      constexpr auto all_raw_indexes = make_array(std::tuple_cat(einstein_indexes.value...));
      std::array<std::size_t, std::size(all_raw_indexes)> all_indexes{};
      std::transform(all_raw_indexes.begin(), all_raw_indexes.end(), all_indexes.begin(),
          [min_index](auto i) { return static_cast<std::size_t>(i - min_index); });
      return all_indexes;
    }();
  }  // namespace EinsteinHelper

  /**
   * @brief Performs Einstein summation (einsum) over the provided tensor-like objects.
   *
   * This function implements a compile-time version of the Einstein summation convention,
   * allowing for flexible tensor contractions and dyadic products. The function supports
   * summing over repeated indices (contraction) and producing tensors with free indices
   * (dyadic indices). The implementation uses extensive constexpr metaprogramming to
   * validate index usage and sizes at compile time, ensuring correctness and efficiency.
   *
   * @code{.cpp}
   * auto tensor_result = einsum<"ij", "j">(A, b); // matrix-vector product A*b
   * auto trace = einsum<"ii">(A); // trace of matrix A
   * @endcode
   *
   * @note Using the Einstein convention is a powerful way to express tensor operations, however,
   * it does not necessarily minimize the number of floating-point operations. The cost of
   * evaluating the Einstein summation is:
   * @f[
   *   \text{cost} = \text{dim}^{\text{num\_idx}} \times (\text{num\_tens} \times \text{additions}
   * +
   * \text{num\_tens} \times \text{multiplications})
   * @f]
   *
   * @return The result of the Einstein summation:
   *         - If all indices are contracted, returns a scalar of the appropriate value type.
   *         - If there are free (dyadic) indices, returns a tensor of the appropriate shape.
   *
   * @note
   * - Each index must be used at most twice.
   * - All contraction indices must have the same size across all tensors.
   * - Indices can be any character (typically i, j, k, ..., or a, b, c, ..., A, B, C, ... or 0,
   * 1, 2, ...).
   * - Indices must be contiguous; no indices may be omitted.
   * - The function uses static_asserts to enforce these constraints at compile time.
   *
   * @throws Compilation error if index constraints are violated.
   */
  template <EinsteinHelper::FixedString... einstein_indexes, typename... Tensor>
    requires(sizeof...(einstein_indexes) == sizeof...(Tensor))
  auto einsum(const Tensor&... t)
  {
    using TensorCompressionTypes = std::tuple<TensorCompressionType<Tensor>...>;

    constexpr auto min_index = EinsteinHelper::smallest_used_index<einstein_indexes...>;

    constexpr auto all_indexes = EinsteinHelper::all_used_indexes<einstein_indexes...>;
    constexpr auto all_index_sizes = EinsteinHelper::make_array(std::tuple_cat(Tensor::shape()...));

    static_assert(std::size(all_index_sizes) == std::size(all_indexes),
        "The indexes do not match the shapes of the given tensors!");

    static_assert(EinsteinHelper::invalid_indexes<all_indexes>.size() == 0,
        "Invalid indexes in Einstein summation. Each index must be used at most twice.");
    static_assert(
        std::ranges::max(all_indexes) == EinsteinHelper::unique_array<all_indexes>.size() - 1,
        "The used Einstein indexes must be contiguous without leaving out an index. GOOD: ijkl "
        "BAD: iklm (j is missing)");
    static_assert(EinsteinHelper::valid_contraction_index_sizes<all_indexes, all_index_sizes>,
        "All contraction indexes must have the same size!");

    using value_type = FADUtils::ScalarOperationResultType<std::multiplies<>,
        typename std::remove_cvref_t<Tensor>::value_type...>;

    constexpr std::array index_sizes =
        EinsteinHelper::unique_index_sizes<all_indexes, all_index_sizes>;

    constexpr std::array contraction_indexes = EinsteinHelper::contraction_indexes<all_indexes>;
    constexpr std::array dyadic_indexes = EinsteinHelper::dyadic_indexes<all_indexes>;

    [[maybe_unused]] constexpr std::array<std::size_t, dyadic_indexes.size()> dyadic_index_sizes =
        EinsteinHelper::dyadic_index_sizes<dyadic_indexes, index_sizes>;

    if constexpr (dyadic_indexes.size() == 0)
    {
      value_type value = 0.0;
      constexpr std::array<std::size_t, 0> dyadic_index = {};

      EinsteinHelper::ConstExprMultiForMaker<index_sizes, contraction_indexes>::multi_for(
          [&](auto... i)
          {
            constexpr std::array<std::size_t, sizeof...(i)> contraction_index = {i()...};

            constexpr std::array tensor_flat_indexes =
                EinsteinHelper::flat_indexes<TensorCompressionTypes, contraction_indexes,
                    dyadic_indexes, contraction_index, dyadic_index,
                    einstein_indexes.to_index_array(min_index)...>;

            value += EinsteinHelper::evaluate_contraction<tensor_flat_indexes>(t...);
          });

      return value;
    }
    else if constexpr (contraction_indexes.size() == 0)
    {
      EinsteinHelper::TensorTypeFromArray<value_type, dyadic_index_sizes> tensor_result{};

      constexpr std::array<std::size_t, 0> contraction_index = {};

      EinsteinHelper::ConstExprMultiForMaker<index_sizes, dyadic_indexes>::multi_for(
          [&](auto... i)
          {
            constexpr std::array<std::size_t, sizeof...(i)> dyadic_index = {i()...};

            constexpr std::array tensor_flat_indexes =
                EinsteinHelper::flat_indexes<TensorCompressionTypes, contraction_indexes,
                    dyadic_indexes, contraction_index, dyadic_index,
                    einstein_indexes.to_index_array(min_index)...>;
            constexpr std::size_t flat_index =
                EinsteinHelper::flat_index<TensorCompressionType<decltype(tensor_result)>,
                    EinsteinHelper::get_tensor_index(dyadic_indexes, contraction_indexes,
                        dyadic_indexes, contraction_index, dyadic_index)>;

            *(tensor_result.data() + flat_index) +=
                EinsteinHelper::evaluate_contraction<tensor_flat_indexes>(t...);
          });

      return tensor_result;
    }
    else
    {
      EinsteinHelper::TensorTypeFromArray<value_type, dyadic_index_sizes> tensor_result{};

      EinsteinHelper::ConstExprMultiForMaker<index_sizes, contraction_indexes>::multi_for(
          [&](auto... j)
          {
            constexpr std::array<std::size_t, sizeof...(j)> contraction_index = {j()...};

            EinsteinHelper::ConstExprMultiForMaker<index_sizes, dyadic_indexes>::multi_for(
                [&](auto... i)
                {
                  constexpr std::array<std::size_t, sizeof...(i)> dyadic_index = {i()...};

                  constexpr std::array tensor_flat_indexes =
                      EinsteinHelper::flat_indexes<TensorCompressionTypes, contraction_indexes,
                          dyadic_indexes, contraction_index, dyadic_index,
                          einstein_indexes.to_index_array(min_index)...>;
                  constexpr std::size_t flat_index =
                      EinsteinHelper::flat_index<TensorCompressionType<decltype(tensor_result)>,
                          EinsteinHelper::get_tensor_index(dyadic_indexes, contraction_indexes,
                              dyadic_indexes, contraction_index, dyadic_index)>;

                  *(tensor_result.data() + flat_index) +=
                      EinsteinHelper::evaluate_contraction<tensor_flat_indexes>(t...);
                });
          });

      return tensor_result;
    }
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif