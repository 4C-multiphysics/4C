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

#include <Teuchos_LAPACK.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{

  namespace Internal
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
    consteval auto get_array_from_tuple(Tuple&& tuple)
    {
      constexpr auto get_array = [](auto&&... x)
      { return std::array{std::forward<decltype(x)>(x)...}; };
      return std::apply(get_array, std::forward<Tuple>(tuple));
    }

    template <auto arr, typename Predicate>
    consteval auto filter_array()
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
    }

    template <auto arr, auto target>
    consteval auto make_unique_by()
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
    }

    template <auto arr>
    consteval auto filter_duplicates()
    {
      return make_unique_by<arr, arr>();
    }

    template <auto arr>
    consteval auto get_invalid_indexes()
    {
      return filter_duplicates<
          filter_array<arr, decltype([](auto i) { return std::ranges::count(arr, i) > 2; })>()>();
    }

    template <auto arr>
    consteval auto get_contraction_indexes()
    {
      return filter_duplicates<
          filter_array<arr, decltype([](auto i) { return std::ranges::count(arr, i) == 2; })>()>();
    }

    template <auto indexes, auto index_sizes>
    consteval bool validate_contraction_index_sizes()
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
    }

    template <auto arr>
    consteval auto get_dyadic_indexes()
    {
      std::array dyadic_indexes = filter_duplicates<
          filter_array<arr, decltype([](auto i) { return std::ranges::count(arr, i) == 1; })>()>();

      std::ranges::sort(dyadic_indexes);
      return dyadic_indexes;
    }

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


    template <typename Tensor, std::array index>
    consteval std::size_t get_flat_index()
    {
      return [&]<std::size_t... i>(std::integer_sequence<std::size_t, i...> seq)
      {
        return TensorCompressionType<Tensor>::template flatten_index<
            Internal::TensorBoundCheck::no_check>(i...);
      }(Internal::IntegerSequenceFromArray<index>());
    };

    template <typename T, std::size_t... n>
      requires(is_tensor<T> && std::remove_cvref_t<T>::rank() == sizeof...(n))
    struct EinsteinSummationItem
    {
      T& tensor;

      using value_type = std::remove_cvref_t<T>::value_type;

      static constexpr std::array indexes = {n...};
      static constexpr std::array index_sizes =
          Internal::get_array_from_tuple(std::remove_cvref_t<T>::shape());

      template <std::array contraction_indexes, std::array dyadic_indexes,
          std::array contraction_index, std::array dyadic_index>
      [[nodiscard]] constexpr auto* value() const
      {
        constexpr std::array this_tensor_index = []() consteval
        {
          std::array<std::size_t, std::remove_cvref_t<T>::rank()> this_tensor_index{};

          for (std::size_t i = 0; i < std::remove_cvref_t<T>::rank(); ++i)
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
        }();

        constexpr std::size_t flat_index = get_flat_index<T, this_tensor_index>();

        return tensor.data() + flat_index;
      }
    };


    template <typename TensorType, std::array indexes, typename IndexSequence>
    struct EinsteinSummationItemDeducer;

    template <typename TensorType, std::array indexes, std::size_t... i>
    struct EinsteinSummationItemDeducer<TensorType, indexes,
        std::integer_sequence<std::size_t, i...>>
    {
      using type = EinsteinSummationItem<TensorType, indexes[i]...>;
    };

    template <typename TensorType, std::array indexes>
    using EinsteinSummationItemTypeFromArray = typename EinsteinSummationItemDeducer<TensorType,
        indexes, std::make_index_sequence<indexes.size()>>::type;
  }  // namespace Internal

  /**
   * @brief Creates an Einstein summation item for a tensor with specified indexes.
   *
   * This function template constructs an `EinsteinSummationItem` from a given tensor to be used
   * with @p einsum, associating it with a compile-time list of indices.
   *
   * @tparam n... The compile-time indices specifying the Einstein summation dimensions.
   * @tparam T The tensor type, which must satisfy the `is_tensor` concept and have rank equal to
   * `sizeof...(n)`.
   * @param t The tensor to be wrapped, forwarded as an rvalue or lvalue reference.
   * @return An `Internal::EinsteinSummationItem` containing the tensor and its associated indices.
   */
  template <std::size_t... n, typename T>
    requires(is_tensor<T> && std::remove_cvref_t<T>::rank() == sizeof...(n))
  auto einidx(T&& t)
  {
    return Internal::EinsteinSummationItem<T, n...>{std::forward<T>(t)};
  }


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
   * auto tensor_result = einsum(einidx<0, 1>(A), einidx<1>(b)); // matrix-vector product A*b
   * auto trace = einsum(einidx<0, 0>(A)); // trace of matrix A
   * @endcode
   *
   * @note Using the Einstein convention is a powerful way to express tensor operations, however,
   * it does not necessarily minimize the number of floating-point operations. The cost of
   * evaluating the Einstein summation is:
   * @f[
   *   \text{cost} = \text{dim}^{\text{num\_idx}} \times (\text{num\_tens} \times \text{additions} +
   * \text{num\_tens} \times \text{multiplications})
   * @f]
   *
   * @tparam TensorIdx Variadic template parameter pack representing tensor index types.
   * @param t Variadic parameter pack of tensor-like objects to be contracted or multiplied.
   * @return The result of the Einstein summation:
   *         - If all indices are contracted, returns a scalar of the appropriate value type.
   *         - If there are free (dyadic) indices, returns a tensor of the appropriate shape.
   *
   * @note
   * - Each index must be used at most twice.
   * - All contraction indices must have the same size across all tensors.
   * - Indices must be contiguous; no indices may be omitted.
   * - The function uses static_asserts to enforce these constraints at compile time.
   *
   * @throws Compilation error if index constraints are violated.
   */
  template <typename... TensorIdx>
  auto einsum(const TensorIdx&... t)
  {
    constexpr auto all_indexes =
        Internal::get_array_from_tuple(std::tuple_cat(TensorIdx::indexes...));
    constexpr auto all_index_sizes =
        Internal::get_array_from_tuple(std::tuple_cat(TensorIdx::index_sizes...));

    static_assert(Internal::get_invalid_indexes<all_indexes>().size() == 0,
        "Invalid indexes in Einstein summation. Each index must be used at most twice.");
    static_assert(std::ranges::max(all_indexes) ==
                      Internal::make_unique_by<all_indexes, all_indexes>().size() - 1,
        "It is not allowed to leave out indexes in Einstein summation.");
    static_assert(Internal::validate_contraction_index_sizes<all_indexes, all_index_sizes>(),
        "All contraction indexes must have the same size!");

    using value_type =
        FADUtils::ScalarOperationResultType<std::multiplies<>, typename TensorIdx::value_type...>;


    constexpr std::array unique_indexes = Internal::make_unique_by<all_indexes, all_indexes>();
    constexpr std::array unique_index_sizes =
        Internal::make_unique_by<all_indexes, all_index_sizes>();
    constexpr std::array index_sizes = [&]() consteval
    {
      std::array<std::size_t, unique_indexes.size()> index_sizes{};
      for (std::size_t i = 0; i < unique_indexes.size(); ++i)
      {
        index_sizes[unique_indexes[i]] = unique_index_sizes[i];
      }
      return index_sizes;
    }();

    constexpr std::array contraction_indexes = Internal::get_contraction_indexes<all_indexes>();
    constexpr std::array dyadic_indexes = Internal::get_dyadic_indexes<all_indexes>();

    [[maybe_unused]] constexpr std::array<std::size_t, dyadic_indexes.size()> dyadic_index_sizes =
        [&]() consteval
    {
      std::array<std::size_t, dyadic_indexes.size()> dyadic_index_sizes{};
      for (std::size_t i = 0; i < dyadic_indexes.size(); ++i)
      {
        dyadic_index_sizes[i] = index_sizes[dyadic_indexes[i]];
      }
      return dyadic_index_sizes;
    }();

    if constexpr (dyadic_indexes.size() == 0)
    {
      value_type value = 0.0;

      Internal::ConstExprMultiForMaker<index_sizes, contraction_indexes>::multi_for(
          [&](auto... i)
          {
            constexpr std::array<std::size_t, sizeof...(i)> contraction_index = {i()...};

            constexpr std::array<std::size_t, sizeof...(i)> dyadic_index = {i()...};

            value += (*t.template value<contraction_indexes, dyadic_indexes, contraction_index,
                          dyadic_index>() *
                      ...);
          });

      return value;
    }
    else if constexpr (contraction_indexes.size() == 0)
    {
      Internal::TensorTypeFromArray<value_type, dyadic_index_sizes> tensor_result{};
      Internal::EinsteinSummationItemTypeFromArray<decltype(tensor_result), dyadic_indexes> result{
          tensor_result};

      Internal::ConstExprMultiForMaker<index_sizes, dyadic_indexes>::multi_for(
          [&](auto... i)
          {
            constexpr std::array<std::size_t, sizeof...(i)> contraction_index = {i()...};

            constexpr std::array<std::size_t, sizeof...(i)> dyadic_index = {i()...};

            *result.template value<contraction_indexes, dyadic_indexes, contraction_index,
                dyadic_index>() += (*t.template value<contraction_indexes, dyadic_indexes,
                                        contraction_index, dyadic_index>() *
                                    ...);
          });

      return tensor_result;
    }
    else
    {
      Internal::TensorTypeFromArray<value_type, dyadic_index_sizes> tensor_result{};
      Internal::EinsteinSummationItemTypeFromArray<decltype(tensor_result), dyadic_indexes> result{
          tensor_result};

      Internal::ConstExprMultiForMaker<index_sizes, contraction_indexes>::multi_for(
          [&](auto... j)
          {
            Internal::ConstExprMultiForMaker<index_sizes, dyadic_indexes>::multi_for(
                [&](auto... i)
                {
                  constexpr std::array<std::size_t, sizeof...(j)> contraction_index = {j()...};

                  constexpr std::array<std::size_t, sizeof...(i)> dyadic_index = {i()...};

                  *result.template value<contraction_indexes, dyadic_indexes, contraction_index,
                      dyadic_index>() += (*t.template value<contraction_indexes, dyadic_indexes,
                                              contraction_index, dyadic_index>() *
                                          ...);
                });
          });

      return tensor_result;
    }
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif