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
#include "4C_utils_compile_time_string.hpp"
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
  struct EinsteinIndex
  {
    char name;
    std::size_t size;
  };

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

    template <typename T, std::size_t n, std::array<T, n> arr, std::size_t... i>
    consteval auto make_integer_sequence_helper(std::integer_sequence<std::size_t, i...> index)
    {
      return std::integer_sequence<T, arr[i]...>{};
    }

    template <std::array arr>
    using IntegerSequenceFromArray =
        decltype(make_integer_sequence_helper<typename decltype(arr)::value_type, arr.size(), arr>(
            std::make_index_sequence<arr.size()>()));


    template <std::array einstein_indices, typename Sequence>
    struct ConstExprMultiForMakerHelper;

    template <std::array einstein_indices, std::size_t... i>
    struct ConstExprMultiForMakerHelper<einstein_indices, std::integer_sequence<std::size_t, i...>>
    {
      using type = ConstExprMultiFor<std::integer_sequence<std::size_t>,
          std::make_index_sequence<einstein_indices[i].size>...>;
    };

    template <std::array einstein_indices>
    using ConstExprMultiForMaker = typename ConstExprMultiForMakerHelper<einstein_indices,
        std::make_index_sequence<std::size(einstein_indices)>>::type;

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

    template <auto einstein_indices>
    constexpr std::array unique_einstein_indices = []() consteval
    {
      constexpr std::size_t num_unique = []() consteval
      {
        std::size_t count = 0;

        for (std::size_t i = 0; i < std::size(einstein_indices); ++i)
        {
          bool already_seen = false;
          for (std::size_t j = 0; j < i; ++j)
          {
            if (i != j && einstein_indices[i].name == einstein_indices[j].name)
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
      }();


      std::array<typename std::remove_cvref_t<decltype(einstein_indices)>::value_type, num_unique>
          unique_einstein_indices{};

      std::vector<char> values{};


      std::size_t unique_index = 0;
      for (std::size_t index = 0; index < einstein_indices.size(); ++index)
      {
        if (std::ranges::find(values, einstein_indices[index].name) == values.end())
        {
          values.push_back(einstein_indices[index].name);
          unique_einstein_indices[unique_index] = einstein_indices[index];
          ++unique_index;
        }
      }

      return unique_einstein_indices;
    }();


    template <auto arr>
    constexpr std::array invalid_einstein_indices = unique_einstein_indices<
        filter_array<arr, decltype([](auto einstein_index) { return std::ranges::count_if(arr, [einstein_index](auto this_einstein_index){
          return this_einstein_index.name == einstein_index.name;
        }) > 2; })>>;


    template <auto arr>
    constexpr std::array contraction_einstein_indices = unique_einstein_indices<
        filter_array<arr, decltype([](auto einstein_index) { return std::ranges::count_if(arr, [einstein_index](auto this_einstein_index){
          return this_einstein_index.name == einstein_index.name;
        }) == 2; })>>;

    template <std::array einstein_indices>
    constexpr bool valid_contraction_einstein_indices = []() consteval
    {
      for (std::size_t i = 0; i < einstein_indices.size(); ++i)
      {
        auto found_item = *std::ranges::find_if(
            einstein_indices, [i](auto item) { return item.name == einstein_indices[i].name; });
        if (found_item.size != einstein_indices[i].size)
        {
          return false;
        }
      }
      return true;
    }();

    template <auto arr>
    constexpr std::array dyadic_einstein_indices = []() consteval
    {
      std::array dyadic_einstein_indices = unique_einstein_indices<
          filter_array<arr, decltype([](auto einstein_index) { return std::ranges::count_if(arr, [einstein_index](auto this_einstein_index){
          return this_einstein_index.name == einstein_index.name;
        }) == 1; })>>;

      std::ranges::sort(dyadic_einstein_indices, [](auto a, auto b) { return a.name < b.name; });
      return dyadic_einstein_indices;
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


    /*!
     * @brief The flat index of a component for a tensor with a compression type @p
     * TensorCompression
     */
    template <typename TensorCompression, std::array index>
    constexpr std::size_t flat_index = []() consteval
    {
      return [&]<std::size_t... i>(std::integer_sequence<std::size_t, i...> seq)
      {
        return TensorCompression::template flatten_index<Internal::TensorBoundCheck::no_check>(
            i...);
      }(IntegerSequenceFromArray<index>());
    }();

    /*!
     * @brief Returns the tensor indices of a component for a tensor in an Einstein summation.
     *
     * Take the matrix-matrix product as an example:
     *
     * A_ij = B_ik C_kj.
     *
     * The contraction indices are [k], the dyadic indices are [i, j].
     *
     * If we not want to evaluate one iteration of the above contraction, we need to do
     *
     * A_ij += B_ik C_kj,
     *
     * for specific indices i, j, k. Let, for example, the indices be i=2, j=1, k=3. If we want to
     * evaluate this function for the tensor B, the parameters would be:
     *
     * my_tensor_einstein_index_names = [i, k]
     * contraction_einstein_indices = [k]
     * dyadic_einstein_indices = [i, j]
     * contraction_index = [3]
     * dyadic_index = [2, 1]
     *
     * This function will return [2, 3] for the component B_23 of B. When calling this function for
     * A, it would be [2, 1], and for C it would be [3, 1].
     *
     * @tparam my_tensor_einstein_index_names A char-array indicating the einstein index-names of
     * the respective tensor
     * @tparam contraction_einstein_indices An array of Einstein indices (type @p EinsteinIndex)
     * that are contracted over
     * @tparam dyadic_einstein_indices An array of Einstein indices (type @p EinsteinIndex) that are
     * dyadic
     * @param contraction_index Array of the contraction index
     * @param dyadic_index Array of the dyadic index
     * @return consteval
     */
    template <std::array my_tensor_einstein_index_names, std::array contraction_einstein_indices,
        std::array dyadic_einstein_indices, std::size_t num_contraction_index_names,
        std::size_t num_dyadic_index_names>
    consteval std::array<std::size_t, my_tensor_einstein_index_names.size()> get_tensor_component(
        std::array<std::size_t, num_contraction_index_names> contraction_index,
        std::array<std::size_t, num_dyadic_index_names> dyadic_index)
    {
      constexpr std::size_t rank = my_tensor_einstein_index_names.size();
      std::array<std::size_t, rank> this_tensor_index{};

      for (std::size_t i = 0; i < rank; ++i)
      {
        bool found = false;
        for (std::size_t j = 0; j < contraction_einstein_indices.size(); ++j)
        {
          if (contraction_einstein_indices[j].name == my_tensor_einstein_index_names[i])
          {
            this_tensor_index[i] = contraction_index[j];
            found = true;
            break;
          }
        }
        if (found) continue;

        for (std::size_t j = 0; j < dyadic_einstein_indices.size(); ++j)
        {
          if (dyadic_einstein_indices[j].name == my_tensor_einstein_index_names[i])
          {
            this_tensor_index[i] = dyadic_index[j];
            found = true;
            break;
          }
        }

        FOUR_C_ASSERT_ALWAYS(
            found, "Internal error: Could not find index in contraction or dyadic indices.");
      }


      return this_tensor_index;
    }

    template <typename TensorCompressionTypes, std::array contraction_einstein_indices,
        std::array dyadic_einstein_indices, std::array contraction_index, std::array dyadic_index,
        typename T, std::array... tensor_einstein_indices>
    struct FlatTensorIndexHelper;

    template <typename TensorCompressionTypes, std::array contraction_einstein_indices,
        std::array dyadic_einstein_indices, std::array contraction_components,
        std::array dyadic_components, std::array... einstein_index_names, std::size_t... i>
    struct FlatTensorIndexHelper<TensorCompressionTypes, contraction_einstein_indices,
        dyadic_einstein_indices, contraction_components, dyadic_components,
        std::integer_sequence<std::size_t, i...>, einstein_index_names...>
    {
      static constexpr auto value = []()
      {
        constexpr std::tuple einstein_index_tuple = {einstein_index_names...};
        return std::array{flat_index<std::tuple_element_t<i, TensorCompressionTypes>,
            get_tensor_component<std::get<i>(einstein_index_tuple), contraction_einstein_indices,
                dyadic_einstein_indices>(contraction_components, dyadic_components)>...};
      }();
    };

    /*!
     * @brief A std::array with the flat indices of the components of the tensors involved in a
     * Einstein summation.
     *
     * @tparam TensorCompressionTypes A tuple type holding the compression types of the involved
     * tensors
     * @tparam contraction_einstein_indices A std::array holding the einstein-indices of the
     * contraction dimensions
     * @tparam dyadic_einstein_indices A std::array holding the einstein-indices of the dyadic
     * dimensions
     * @tparam contraction_index A std::array holding the current indices of the contraction
     * dimensions of the current Einstein iteration
     * @tparam dyadic_index A std::array holding the current indices of the dyadic dimensions of the
     * current Einstein iteration
     * @tparam einstein_indices A std::array holding the indices of the Einstein summation
     */
    template <typename TensorCompressionTypes, std::array contraction_einstein_indices,
        std::array dyadic_einstein_indices, std::array contraction_index, std::array dyadic_index,
        std::array... tensor_einstein_index_names>
    constexpr std::array flat_indices = []() consteval
    {
      return FlatTensorIndexHelper<TensorCompressionTypes, contraction_einstein_indices,
          dyadic_einstein_indices, contraction_index, dyadic_index,
          std::make_index_sequence<std::tuple_size_v<TensorCompressionTypes>>,
          tensor_einstein_index_names...>::value;
    }();

    template <typename FlatIndices, typename... Tensor>
    struct ContractionEvaluationHelper;

    template <std::size_t first_index, typename FirstTensor, std::size_t... other_indices,
        typename... OtherTensor>
    struct ContractionEvaluationHelper<
        std::integer_sequence<std::size_t, first_index, other_indices...>, FirstTensor,
        OtherTensor...>
    {
      static constexpr auto evaluate_contraction(
          const FirstTensor& first_tensor, const OtherTensor&... other_tensors)
      {
        return *(first_tensor.data() + first_index) *
               ContractionEvaluationHelper<std::integer_sequence<std::size_t, other_indices...>,
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

    /*!
     * @brief Evaluates the Einstein summation contraction iteration for the given tensors @p t and
     * their flat-indices @p flat_indices.
     */
    template <std::array flat_indices, typename... Tensor>
    auto evaluate_contraction(const Tensor&... t)
    {
      return ContractionEvaluationHelper<IntegerSequenceFromArray<flat_indices>,
          Tensor...>::evaluate_contraction(t...);
    }

    template <Utils::CompileTimeString... einstein_indices>
    constexpr char smallest_used_index_name =
        std::ranges::min(EinsteinHelper::make_array(std::tuple_cat(einstein_indices.value...)));

    template <Utils::CompileTimeString... einstein_indices>
    constexpr std::array all_used_index_names = []() consteval
    {
      constexpr char min_index = smallest_used_index_name<einstein_indices...>;
      constexpr auto all_raw_indices = make_array(std::tuple_cat(einstein_indices.value...));
      std::array<char, std::size(all_raw_indices)> all_indices{};
      std::transform(all_raw_indices.begin(), all_raw_indices.end(), all_indices.begin(),
          [min_index](auto i) { return static_cast<char>(i - min_index); });
      return all_indices;
    }();

    template <Utils::CompileTimeString... einstein_indices>
    consteval auto extract_all_einstein_indices(auto... shapes)
    {
      std::array all_index_names = make_array(std::tuple_cat(einstein_indices.value...));
      std::array all_index_sizes = make_array(std::tuple_cat(shapes...));

      static_assert(std::size(all_index_sizes) == std::size(all_index_names),
          "The number of given indices do not match the shapes of the given tensors!");

      std::array<EinsteinIndex, std::size(all_index_names)> all_indices;
      for (std::size_t i = 0; i < all_index_names.size(); ++i)
      {
        all_indices[i] = {.name = all_index_names[i], .size = all_index_sizes[i]};
      }
      return all_indices;
    };

    template <std::array einstein_indices>
    constexpr std::array einstein_index_sizes = []()
    {
      std::array<std::size_t, std::size(einstein_indices)> sizes{};
      std::transform(einstein_indices.begin(), einstein_indices.end(), sizes.begin(),
          [](auto einstein_index) { return einstein_index.size; });
      return sizes;
    }();

    template <std::array einstein_indices>
    constexpr std::array einstein_index_names = []()
    {
      std::array<char, std::size(einstein_indices)> names{};
      std::transform(einstein_indices.begin(), einstein_indices.end(), names.begin(),
          [](auto einstein_index) { return einstein_index.name; });
      return names;
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
   * - The function uses static_asserts to enforce these constraints at compile time.
   *
   * @throws Compilation error if index constraints are violated.
   */
  template <Utils::CompileTimeString... einstein_indices, typename... Tensor>
    requires(sizeof...(einstein_indices) == sizeof...(Tensor))
  auto einsum(const Tensor&... t)
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

    if constexpr (dyadic_einstein_indices.size() == 0)
    {
      value_type value = 0.0;
      constexpr std::array<std::size_t, 0> dyadic_index = {};

      EinsteinHelper::ConstExprMultiForMaker<contraction_einstein_indices>::multi_for(
          [&](auto... i)
          {
            constexpr std::array<std::size_t, sizeof...(i)> contraction_index = {i()...};

            constexpr std::array tensor_flat_indices =
                EinsteinHelper::flat_indices<TensorCompressionTypes, contraction_einstein_indices,
                    dyadic_einstein_indices, contraction_index, dyadic_index,
                    einstein_indices.value...>;

            value += EinsteinHelper::evaluate_contraction<tensor_flat_indices>(t...);
          });

      return value;
    }
    else if constexpr (contraction_einstein_indices.size() == 0)
    {
      EinsteinHelper::TensorTypeFromArray<value_type,
          EinsteinHelper::einstein_index_sizes<dyadic_einstein_indices>>
          tensor_result{};

      constexpr std::array<std::size_t, 0> contraction_index = {};

      EinsteinHelper::ConstExprMultiForMaker<dyadic_einstein_indices>::multi_for(
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
      EinsteinHelper::TensorTypeFromArray<value_type,
          EinsteinHelper::einstein_index_sizes<dyadic_einstein_indices>>
          tensor_result{};

      EinsteinHelper::ConstExprMultiForMaker<contraction_einstein_indices>::multi_for(
          [&](auto... j)
          {
            constexpr std::array<std::size_t, sizeof...(j)> contraction_index = {j()...};

            EinsteinHelper::ConstExprMultiForMaker<dyadic_einstein_indices>::multi_for(
                [&](auto... i)
                {
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