// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_HPP
#define FOUR_C_LINALG_TENSOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor_internals.hpp"
#include "4C_linalg_tensor_meta_utils.hpp"
#include "4C_utils_demangle.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <functional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename Tensor>
  concept Rank1TensorConcept = is_tensor<Tensor> && Tensor::rank() == 1;

  template <typename Tensor>
  concept Rank2TensorConcept = is_tensor<Tensor> && Tensor::rank() == 2;

  template <typename Tensor>
  concept SquareTensor =
      Rank2TensorConcept<Tensor> && Tensor::template extent<0>() == Tensor::template extent<1>();


  template <std::size_t... n>
  struct NoCompression
  {
    static constexpr std::size_t compressed_size = (n * ...);

    template <Internal::TensorBoundCheck bound_check>
    static constexpr std::size_t flatten_index(decltype(n)... i)
    {
      return Internal::get_flat_index<Internal::OrderType::column_major, bound_check, n...>(i...);
    }
    // flatten index
  };
  /*!
   * @brief An owning, dense tensor of arbitrary rank
   *
   * @copydetails TensorInternal
   */
  template <typename T, std::size_t... n>
  using Tensor = TensorInternal<T, TensorStorageType::owning, NoCompression<n...>,
      n...>;  // owns the memory

  /*!
   * @brief A dense view to a Tensor of arbitrary rank
   *
   * @copydetails TensorInternal
   */
  template <typename T, std::size_t... n>
  using TensorView = TensorInternal<T, TensorStorageType::view, NoCompression<n...>,
      n...>;  // view onto's other
              // memory

  /*!
   * @brief Creates a TensorView from a pointer to data and the tensor shape
   *
   * This function creates a TensorView from a pointer to data and the tensor shape specified by
   * the template parameters. The data is expected to be stored using NoCompression, i.e., in a
   * contiguous memory block of size the same size as the underlying tensor (in column-major
   * layout).
   *
   * @tparam n The dimensions of the tensor.
   * @param data Pointer to the data.
   * @return A TensorView of type TensorView<ValueType, n...>.
   */
  template <std::size_t... n>
  constexpr auto make_tensor_view(auto* data)
  {
    using ValueType = std::remove_pointer_t<decltype(data)>;
    constexpr std::size_t size = (n * ...);
    std::span<ValueType, size> data_span(data, size);

    return TensorView<ValueType, n...>(std::move(data_span));
  }

  /*!
   * @brief Creates a Tensor from statically sized input data (std::array, std::span, etc.)
   *
   * The data is expected to be stored using NoCompression, i.e., in a contiguous memory block of
   * size the same size as the underlying tensor (in column-major layout). The data is copied into
   * the new Tensor that is returned.
   */
  template <std::size_t... n, typename T>
  constexpr auto make_tensor(const std::array<T, (n * ...)>& values)
  {
    constexpr std::size_t size = (n * ...);
    Tensor<T, n...> t;
    std::copy_n(values.begin(), size, t.data());

    return t;
  }

  namespace Internal
  {
    template <typename T>
    struct MakeTensorViewFromIntegerSequenceHelper
    {
    };


    template <std::size_t... n>
    struct MakeTensorViewFromIntegerSequenceHelper<std::integer_sequence<std::size_t, n...>>
    {
      static constexpr auto make_tensor_view_from_sequence(auto* data)
      {
        return make_tensor_view<n...>(data);
      }
    };
  }  // namespace Internal

  template <typename IntegerSequence>
  constexpr auto make_tensor_view(auto* data)
  {
    return Internal::MakeTensorViewFromIntegerSequenceHelper<
        IntegerSequence>::make_tensor_view_from_sequence(data);
  }

  // Implementation of tensor operations
  namespace Internal
  {
    template <typename T, typename S1, typename S2>
    struct DyadicProductTensorResult;

    template <typename T, std::size_t... s1, std::size_t... s2>
    struct DyadicProductTensorResult<T, std::integer_sequence<std::size_t, s1...>,
        std::integer_sequence<std::size_t, s2...>>
    {
      using type =
          TensorInternal<T, TensorStorageType::owning, NoCompression<s1..., s2...>, s1..., s2...>;
    };

    template <typename T, typename Shape, std::size_t... new_order>
    struct ReorderAxisTensorHelper;

    template <typename T, std::size_t... shape, std::size_t... new_order>
    struct ReorderAxisTensorHelper<T, std::integer_sequence<std::size_t, shape...>, new_order...>
    {
      static consteval std::size_t get_axis(std::size_t t)
      {
        constexpr std::array<std::size_t, sizeof...(shape)> tensor_shape = {shape...};
        return tensor_shape[t];
      }

      static consteval std::tuple<decltype(shape)...> get_new_index(
          std::tuple<decltype(shape)...> old_index)
      {
        auto make_array_from_tuple = [](auto&&... args) constexpr
        { return std::array<std::size_t, sizeof...(args)>{args...}; };

        std::array<std::size_t, sizeof...(shape)> old_index_array =
            std::apply(make_array_from_tuple, old_index);

        std::array<std::size_t, sizeof...(shape)> new_order_array = {new_order...};
        std::array<std::size_t, sizeof...(shape)> new_index_array;
        for (std::size_t i = 0; i < sizeof...(shape); ++i)
        {
          auto where_is_it = std::ranges::find(new_order_array, i);

          new_index_array[i] = old_index_array[std::distance(new_order_array.begin(), where_is_it)];
        }

        return std::tuple_cat(new_index_array);
      }

      static consteval std::array<std::size_t, (shape * ...)> get_index_mapping()
      {
        std::array<std::size_t, (shape * ...)> index_mapping = {0};

        for (std::size_t i = 0; i < (shape * ...); ++i)
        {
          auto old_index = get_md_index<OrderType::column_major, TensorBoundCheck::no_check,
              get_axis(new_order)...>(i);

          auto new_index = get_new_index(old_index);

          index_mapping[i] = std::apply(
              get_flat_index<OrderType::column_major, TensorBoundCheck::no_check, shape...>,
              new_index);
        }

        return index_mapping;
      }

      using result_type = TensorInternal<std::remove_cvref_t<T>, TensorStorageType::owning,
          NoCompression<get_axis(new_order)...>, get_axis(new_order)...>;
    };
  }  // namespace Internal

  /*!
   * @brief Computes and returns the determinant of a square tensor of rank 2
   *
   * @param A
   * @return auto
   */
  constexpr auto det(const SquareTensor auto& A)
    requires(!is_compressed_tensor<decltype(A)>);

  /*!
   * @brief Computes and returns the trace of a square tensor of rank 2
   *
   * @param A
   * @return auto
   */
  constexpr auto trace(const SquareTensor auto& A)
    requires(!is_compressed_tensor<decltype(A)>);

  /*!
   * @brief Computes and returns the L2-norm of a rank 1 tensor
   *
   * @param A
   * @return auto
   */
  constexpr auto norm2(const auto& A)
    requires(is_tensor<decltype(A)> && !is_compressed_tensor<decltype(A)> &&
             std::remove_cvref_t<decltype(A)>::rank() == 1);

  /*!
   * @brief Computes and returns the inverse of a square tensor of rank 2
   *
   * @param A
   * @return auto
   */
  auto inv(const SquareTensor auto& A)
    requires(!is_compressed_tensor<decltype(A)>);

  /*!
   * @brief Computes and returns the transpose of a rank 2 tensor
   *
   * @param A
   * @return auto
   */
  constexpr auto transpose(const Rank2TensorConcept auto& A)
    requires(!is_compressed_tensor<decltype(A)>);

  /*!
   * @brief Computes the dot product of two tensors
   *
   * This function performs the dot product of two tensors @p a and @p b with arbitrary rank. The
   * last dimension of @p a must match the first dimension of @p b. Returns a scalar value if @p a
   * and @p b are rank-1 tensors, otherwise returns a tensor.
   *
   * @tparam TensorLeft
   * @tparam TensorRight
   * @return scalar or tensur resulting from the dot product.
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && TensorLeft::rank() >= 1 &&
             TensorRight::rank() >= 1 &&
             TensorLeft::template extent<TensorLeft::rank() - 1>() ==
                 TensorRight::template extent<0>() &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto dot(const TensorLeft& a, const TensorRight& b);

  /*!
   * @brief Computes the double dot product of two tensors.
   *
   * This function computes the double dot product of two tensors @p A and @p B. If @p A and @p B
   * are rank-2 tensors (matrices), the result is a scalar value, otherwise, the result is a
   * tensor.
   *
   * @tparam TensorLeft The type of the first tensor.
   * @tparam TensorRight The type of the second tensor.
   * @param A The first tensor.
   * @param B The second tensor.
   * @return Scalar or tensor depending on the ranks of @p A and @p B.
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && TensorLeft::rank() >= 2 &&
             TensorRight::rank() >= 2 &&
             TensorLeft::template extent<TensorLeft::rank() - 2>() ==
                 TensorRight::template extent<0>() &&
             TensorLeft::template extent<TensorLeft::rank() - 1>() ==
                 TensorRight::template extent<1>() &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto ddot(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Adds two tensors element-wise and returns the resulting tensor.
   *
   * This function performs an element-wise addition of two tensors with the same shape.
   * The resulting tensor will have the same shape as the input tensors.
   *
   * @tparam TensorLeft The type of the first tensor.
   * @tparam TensorRight The type of the second tensor.
   * @param A The first tensor.
   * @param B The second tensor.
   * @return A tensor containing the element-wise sum of the input tensors.
   *
   * @note The input tensors must have the same shape, as enforced by the `requires` clause.
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto add(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Subtracts two tensors element-wise and returns the resulting tensor.
   *
   * This function performs an element-wise subtraction of two tensors with the same shape.
   * The resulting tensor will have the same shape as the input tensors.
   *
   * @tparam TensorLeft The type of the first tensor.
   * @tparam TensorRight The type of the second tensor.
   * @param A The first tensor.
   * @param B The second tensor.
   * @return A tensor containing the element-wise subtraction of the input tensors.
   *
   * @note The input tensors must have the same shape, as enforced by the `requires` clause.
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto subtract(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Return a tensor scaled by a scalar
   *
   * @tparam Tensor
   * @tparam Scalar
   */
  template <typename Tensor, typename Scalar>
    requires(is_scalar<Scalar> && is_tensor<Tensor> && !is_compressed_tensor<Tensor>)
  constexpr auto scale(const Tensor& tensor, const Scalar& b);

  /*!
   * @brief Returns the dyadic product of two tensors with arbitrary rank
   *
   * @tparam TensorLeft
   * @tparam TensorRight
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && !is_compressed_tensor<TensorLeft> &&
             !is_compressed_tensor<TensorRight>)
  constexpr auto dyadic(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Reorders the axes of a tensor (generalization of transpose)
   *
   * For example, given the tensor A of shape (2, 3, 4), @p reorder_axis<0, 2, 1>(A) will return a
   * tensor of the shape (2, 4, 3). The template arguments specify the new order of the axis, where
   * each value represents the index of the original axis, in this case, 0, 2, 1.
   *
   * @tparam new_order
   * @return A tensor with the axes reordered according to @p new_order
   */
  template <std::size_t... new_order>
  constexpr auto reorder_axis(auto tensor)
    requires(is_tensor<decltype(tensor)> && sizeof...(new_order) == decltype(tensor)::rank() &&
             !is_compressed_tensor<decltype(tensor)>);

  /*!
   * @brief Write tensor values to an output stream
   */
  void print_values(std::ostream& os, const auto& tensor)
    requires(is_tensor<decltype(tensor)> && !is_compressed_tensor<decltype(tensor)>);

  /*!
   * @brief Write tensor to an output stream
   */
  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  std::ostream& operator<<(std::ostream& os,
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor);


  /*!
   * @brief Add another tensor onto this tensor
   *
   * @tparam OtherTensor
   */
  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator+=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor,
      const OtherTensor& B);

  /*!
   * @brief Subtract another tensor from this tensor
   *
   * @tparam OtherTensor
   */
  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator-=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor,
      const OtherTensor& B);

  /*!
   * @brief Scale the tensor with a scalar value
   *
   * @tparam OtherTensor
   */
  template <typename Scalar, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_scalar<Scalar>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator*=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor, const Scalar b);

  /*!
   * @brief Scales the tensor with the inverse of the scalar value b
   *
   * @tparam Scalar
   * @tparam Number
   * @tparam storage_type
   * @tparam n
   */
  template <typename Scalar, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_scalar<Scalar>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator/=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor, const Scalar b);

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type>)
  constexpr auto operator+(const TensorLeft& A, const TensorRight& B)
  {
    return add(A, B);
  }
  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type>)
  constexpr auto operator-(const TensorLeft& A, const TensorRight& B)
  {
    return subtract(A, B);
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight>)
  constexpr auto operator*(const TensorLeft& A, const TensorRight& B)
  {
    return dot(A, B);
  }

  template <typename Scalar, typename Tensor>
    requires(is_scalar<Scalar> && is_tensor<Tensor>)
  constexpr auto operator*(const Tensor& tensor, const Scalar b)
  {
    return scale(tensor, b);
  }

  template <typename Scalar, typename Tensor>
    requires(is_scalar<Scalar> && is_tensor<Tensor>)
  constexpr auto operator*(const Scalar b, const Tensor& tensor)
  {
    return scale(tensor, b);
  }

  template <typename Scalar, typename Tensor>
    requires(is_scalar<Scalar> && is_tensor<Tensor>)
  constexpr auto operator/(const Tensor& tensor, const Scalar b)
  {
    return scale(tensor, Scalar(1) / b);
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type>)
  constexpr bool operator==(const TensorLeft& A, const TensorRight& B)
  {
    return std::ranges::equal(A.container(), B.container());
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type>)
  constexpr bool operator!=(const TensorLeft& A, const TensorRight& B)
  {
    return !std::ranges::equal(A.container(), B.container());
  }


  // Actual implementations
  constexpr auto det(const SquareTensor auto& A)
    requires(!is_compressed_tensor<decltype(A)>)
  {
    using Tensor = std::remove_cvref_t<decltype(A)>;
    using value_type = Tensor::value_type;
    constexpr std::size_t n = Tensor::template extent<0>();

    return DenseFunctions::determinant<value_type, n, n>(A.data());
  }

  constexpr auto trace(const SquareTensor auto& A)
    requires(!is_compressed_tensor<decltype(A)>)
  {
    using Tensor = std::remove_cvref_t<decltype(A)>;
    using value_type = Tensor::value_type;
    constexpr std::size_t n = Tensor::template extent<0>();

    return DenseFunctions::trace<value_type, n, n>(A.data());
  }

  constexpr auto norm2(const auto& A)
    requires(is_tensor<decltype(A)> && !is_compressed_tensor<decltype(A)> &&
             std::remove_cvref_t<decltype(A)>::rank() == 1)
  {
    using Tensor = std::remove_cvref_t<decltype(A)>;
    using value_type = Tensor::value_type;
    constexpr std::size_t n = Tensor::template extent<0>();

    return DenseFunctions::norm2<value_type, n, 1>(A.data());
  }

  auto inv(const SquareTensor auto& A)
    requires(!is_compressed_tensor<decltype(A)>)
  {
    using Tensor = std::remove_cvref_t<decltype(A)>;
    using value_type = Tensor::value_type;
    constexpr std::size_t n = Tensor::template extent<0>();

    Core::LinAlg::Tensor<value_type, n, n> dest;
    DenseFunctions::invert<value_type, n, n>(dest.data(), A.data());
    return dest;
  }

  constexpr auto transpose(const Rank2TensorConcept auto& A)
    requires(!is_compressed_tensor<decltype(A)>)
  {
    return reorder_axis<1, 0>(A);
  }

  namespace Internal
  {
    template <typename ValueType, typename Shape>
    struct TensorTypeFromIntegerSequence;

    template <typename ValueType, std::size_t... n>
    struct TensorTypeFromIntegerSequence<ValueType, std::integer_sequence<std::size_t, n...>>
    {
      using type = Tensor<ValueType, n...>;
    };

    template <typename Tuple>
    consteval auto make_array_from_tuple(Tuple&& tuple)
    {
      constexpr auto get_array = [](auto&&... x)
      { return std::array{std::forward<decltype(x)>(x)...}; };
      return std::apply(get_array, std::forward<Tuple>(tuple));
    }

    template <std::array array>
    consteval auto make_integer_sequence()
    {
      constexpr auto array_to_integer_sequence = []<std::size_t... n>(
                                                     std::index_sequence<n...>) consteval
      { return std::integer_sequence<std::size_t, array[n]...>{}; };

      return array_to_integer_sequence(std::make_index_sequence<array.size()>{});
    }

    template <typename TupleTypeLeft, typename TupleTypeRight>
    consteval auto get_dot_product_result_shape(
        const TupleTypeLeft& left_shape, const TupleTypeRight& right_shape)
    {
      const std::array left_shape_array = make_array_from_tuple(left_shape);
      const std::array right_shape_array = make_array_from_tuple(right_shape);
      constexpr std::size_t left_size = std::tuple_size_v<TupleTypeLeft>;
      constexpr std::size_t right_size = std::tuple_size_v<TupleTypeRight>;


      std::array<std::size_t, left_size + right_size - 2> result_shape{};

      std::copy(left_shape_array.begin(), left_shape_array.end() - 1, result_shape.begin());
      std::copy(right_shape_array.begin() + 1, right_shape_array.end(),
          result_shape.begin() + (left_size - 1));

      return result_shape;
    }

    template <typename TensorLeft, typename TensorRight>
    using DotProductResultType = TensorTypeFromIntegerSequence<
        FADUtils::ScalarOperationResultType<std::multiplies<>, typename TensorLeft::value_type,
            typename TensorRight::value_type>,
        decltype(make_integer_sequence<get_dot_product_result_shape(
                TensorLeft::shape(), TensorRight::shape())>())>::type;

    template <typename TupleType>
    consteval std::size_t get_dot_product_left_matrix_size(const TupleType& left_shape)
    {
      const std::array left_shape_array = make_array_from_tuple(left_shape);
      return std::accumulate(
          left_shape_array.begin(), left_shape_array.end() - 1, 1, std::multiplies<std::size_t>());
    }

    template <typename TupleType>
    consteval std::size_t get_dot_product_right_matrix_size(const TupleType& right_shape)
    {
      const std::array right_shape_array = make_array_from_tuple(right_shape);
      return std::accumulate(right_shape_array.begin() + 1, right_shape_array.end(), 1,
          std::multiplies<std::size_t>());
    }
  }  // namespace Internal

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && TensorLeft::rank() >= 1 &&
             TensorRight::rank() >= 1 &&
             TensorLeft::template extent<TensorLeft::rank() - 1>() ==
                 TensorRight::template extent<0>() &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto dot(const TensorLeft& a, const TensorRight& b)
  {
    using value_type = FADUtils::ScalarOperationResultType<std::multiplies<>,
        typename TensorLeft::value_type, typename TensorRight::value_type>;
    constexpr std::size_t k = TensorLeft::template extent<TensorLeft::rank() - 1>();

    if constexpr (TensorLeft::rank() == 1 && TensorRight::rank() == 1)
    {
      // Special case for rank-1 tensors: Result is a scalar
      return DenseFunctions::dot<value_type, k, 1>(a.data(), b.data());
    }
    else
    {
      constexpr std::size_t m = Internal::get_dot_product_left_matrix_size(TensorLeft::shape());
      constexpr std::size_t n = Internal::get_dot_product_right_matrix_size(TensorRight::shape());

      Internal::DotProductResultType<TensorLeft, TensorRight> dest;
      DenseFunctions::multiply<value_type, m, k, n>(dest.data(), a.data(), b.data());
      return dest;
    }
  }

  namespace Internal
  {
    template <typename T, T... n>
    consteval auto to_integer_sequence(std::integer_sequence<T, n...>)
    {
      return std::integer_sequence<T, n...>{};
    }

    template <typename Array>
    constexpr auto make_tuple_from_array(Array&& array)
    {
      return std::tuple_cat(array);
    }

    consteval auto ddot_product_right_tensor_reduced_shape(auto tuple)
    {
      return std::apply([](auto first, auto second, auto... args) constexpr
          { return std::array{first * second, args...}; }, tuple);
    }

    consteval auto ddot_product_left_tensor_reduced_shape(auto tuple)
    {
      std::array input_array = make_array_from_tuple(tuple);
      std::ranges::reverse(input_array);

      std::tuple reversed_resulting_shape =
          ddot_product_right_tensor_reduced_shape(make_tuple_from_array(input_array));

      std::array output_array = make_array_from_tuple(reversed_resulting_shape);
      std::ranges::reverse(output_array);
      return output_array;
    }
  }  // namespace Internal

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && TensorLeft::rank() >= 2 &&
             TensorRight::rank() >= 2 &&
             TensorLeft::template extent<TensorLeft::rank() - 2>() ==
                 TensorRight::template extent<0>() &&
             TensorLeft::template extent<TensorLeft::rank() - 1>() ==
                 TensorRight::template extent<1>() &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto ddot(const TensorLeft& A, const TensorRight& B)
  {
    using value_type = FADUtils::ScalarOperationResultType<std::multiplies<>,
        typename TensorLeft::value_type, typename TensorRight::value_type>;

    if constexpr (TensorLeft::rank() == 2 && TensorRight::rank() == 2)
    {
      // this is a special case since the result is a scalar
      constexpr std::size_t m = TensorLeft::template extent<0>();
      constexpr std::size_t n = TensorLeft::template extent<1>();

      return DenseFunctions::dot<value_type, m, n>(A.data(), B.data());
    }
    else
    {
      // in this case, result is again a tensor
      constexpr std::array reduced_shape_left =
          Internal::ddot_product_left_tensor_reduced_shape(TensorLeft::shape());
      constexpr std::array reduced_shape_right =
          Internal::ddot_product_right_tensor_reduced_shape(TensorRight::shape());

      // combine the two axis of the double-dot product and do a normal dot product
      return dot(
          make_tensor_view<
              decltype(Internal::template make_integer_sequence<reduced_shape_left>())>(A.data()),
          make_tensor_view<
              decltype(Internal::template make_integer_sequence<reduced_shape_right>())>(B.data()));
    }
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto add(const TensorLeft& A, const TensorRight& B)
  {
    using result_value_type = FADUtils::ScalarOperationResultType<std::plus<>,
        typename TensorLeft::value_type, typename TensorRight::value_type>;
    OwningTensorType<TensorLeft, result_value_type> tens_out{};
    DenseFunctions::update<result_value_type, TensorLeft::size(), 1>(
        tens_out.data(), A.data(), B.data());

    return tens_out;
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> &&
             std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             !is_compressed_tensor<TensorLeft> && !is_compressed_tensor<TensorRight>)
  constexpr auto subtract(const TensorLeft& A, const TensorRight& B)
  {
    using result_value_type = FADUtils::ScalarOperationResultType<std::minus<>,
        typename TensorLeft::value_type, typename TensorRight::value_type>;

    OwningTensorType<TensorLeft, result_value_type> tens_out{};
    DenseFunctions::update<result_value_type, TensorLeft::size(), 1>(
        tens_out.data(), 1.0, A.data(), -1.0, B.data());

    return tens_out;
  }

  template <typename Tensor, typename Scalar>
    requires(is_scalar<Scalar> && is_tensor<Tensor> && !is_compressed_tensor<Tensor>)
  constexpr auto scale(const Tensor& tensor, const Scalar& b)
  {
    using result_value_type =
        FADUtils::ScalarOperationResultType<std::multiplies<>, Scalar, typename Tensor::value_type>;

    OwningTensorType<Tensor, result_value_type> tens_out;

    std::transform(tensor.data(), tensor.data() + Tensor::size(), tens_out.data(),
        [&b](const auto& value) { return value * b; });

    return tens_out;
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && !is_compressed_tensor<TensorLeft> &&
             !is_compressed_tensor<TensorRight>)
  constexpr auto dyadic(const TensorLeft& A, const TensorRight& B)
  {
    using value_type = FADUtils::ScalarOperationResultType<std::multiplies<>,
        typename TensorLeft::value_type, typename TensorRight::value_type>;

    typename Internal::DyadicProductTensorResult<value_type, typename TensorLeft::shape_type,
        typename TensorRight::shape_type>::type dest{};

    DenseFunctions::multiply<value_type, TensorLeft::size(), 1, TensorRight::size()>(
        dest.data(), A.data(), B.data());

    return dest;
  }

  template <std::size_t... new_order>
  constexpr auto reorder_axis(auto tensor)
    requires(is_tensor<decltype(tensor)> && sizeof...(new_order) == decltype(tensor)::rank() &&
             !is_compressed_tensor<decltype(tensor)>)
  {
    constexpr std::array new_order_array = {new_order...};
    static_assert(std::ranges::max(new_order_array) < tensor.rank(),
        "Invalid tensor axis reordering. An axis index is out of bounds.");
    static_assert(
        [new_order_array]() constexpr
        {
          std::array sorted_indices = new_order_array;
          std::ranges::sort(sorted_indices);
          return std::ranges::adjacent_find(sorted_indices) == std::end(sorted_indices);
        }(),
        "All indices must be unique during reordering!");

    using Tensor = decltype(tensor);
    typename Internal::ReorderAxisTensorHelper<typename Tensor::value_type,
        typename Tensor::shape_type, new_order...>::result_type result;

    constexpr std::array<std::size_t, result.size()> index_reorder =
        Internal::ReorderAxisTensorHelper<typename Tensor::value_type, typename Tensor::shape_type,
            new_order...>::get_index_mapping();

    std::transform(index_reorder.begin(), index_reorder.end(), result.data(),
        [&tensor](const auto& i) { return tensor.container()[i]; });

    return result;
  }

  namespace Internal
  {
    void print_sub_values_helper(std::ostream& os, const auto& tensor, auto index_to_print)
      requires(!is_compressed_tensor<decltype(tensor)>)
    {
      auto make_array_from_tuple = [](auto&&... args) constexpr
      { return std::array<std::size_t, sizeof...(args)>{args...}; };

      constexpr std::array shape =
          std::apply(make_array_from_tuple, std::remove_cvref_t<decltype(tensor)>::shape());

      constexpr std::size_t num_indices_to_print = std::tuple_size_v<decltype(index_to_print)>;
      if constexpr (num_indices_to_print == std::size(shape) - 1)
      {
        os << std::string(num_indices_to_print, ' ') << "[";
        for (std::size_t i = 0; i < shape[num_indices_to_print]; ++i)
        {
          if (i > 0) os << ", ";
          std::apply([&os, &tensor](auto... idx) { os << tensor(idx...); },
              std::tuple_cat(index_to_print, std::make_tuple(i)));
        }
        os << "]";
      }
      else
      {
        os << std::string(num_indices_to_print, ' ') << "[\n";
        for (std::size_t i = 0; i < shape[num_indices_to_print]; ++i)
        {
          if (i > 0) os << ",\n";
          print_sub_values_helper(os, tensor, std::tuple_cat(index_to_print, std::make_tuple(i)));
        }
        os << "\n" << std::string(num_indices_to_print, ' ') << "]";
      }
    }
  }  // namespace Internal

  void print_values(std::ostream& os, const auto& tensor)
    requires(is_tensor<decltype(tensor)> && !is_compressed_tensor<decltype(tensor)>)
  {
    Internal::print_sub_values_helper(os, tensor, std::make_tuple());
  }

  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  std::ostream& operator<<(std::ostream& os,
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor)
  {
    constexpr auto tensor_type = []() consteval
    {
      if constexpr (storage_type == TensorStorageType::owning)
        return "Tensor";
      else if constexpr (storage_type == TensorStorageType::view)
        return "TensorView";
      else
        FOUR_C_THROW("Unknown tensor type!");
    }();

    auto make_array_from_tuple = [](auto&&... args) constexpr
    { return std::array<std::size_t, sizeof...(args)>{args...}; };

    constexpr std::array shape =
        std::apply(make_array_from_tuple, std::remove_cvref_t<decltype(tensor)>::shape());

    const std::string shape_str =
        std::accumulate(std::next(shape.begin()), shape.end(), std::to_string(shape[0]),
            [](const std::string& a, const std::size_t b) { return a + ", " + std::to_string(b); });

    os << tensor_type << "<" << Core::Utils::try_demangle(typeid(Number).name()) << ", "
       << shape_str << ">";
    print_values(os, tensor);

    return os;
  }

  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator+=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor, const OtherTensor& B)
  {
    DenseFunctions::update<
        typename TensorInternal<Number, storage_type, NoCompression<n...>, n...>::value_type,
        OtherTensor::size(), 1>(1.0, tensor.data(), 1.0, B.data());
    return tensor;
  }

  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator-=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor, const OtherTensor& B)
  {
    DenseFunctions::update<
        typename TensorInternal<Number, storage_type, NoCompression<n...>, n...>::value_type,
        OtherTensor::size(), 1>(1.0, tensor.data(), -1.0, B.data());
    return tensor;
  }

  template <typename Scalar, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_scalar<Scalar>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator*=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor, const Scalar b)
  {
    for (auto& value : tensor.container()) value *= b;
    return tensor;
  }

  template <typename Scalar, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_scalar<Scalar>)
  constexpr TensorInternal<Number, storage_type, NoCompression<n...>, n...>& operator/=(
      TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor, const Scalar b)
  {
    tensor *= Scalar(1.0) / b;
    return tensor;
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif