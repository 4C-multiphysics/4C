// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_SYMMETRIC_TENSOR_HPP
#define FOUR_C_LINALG_SYMMETRIC_TENSOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_internals.hpp"
#include "4C_linalg_tensor_meta_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <type_traits>
#include <utility>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{

  namespace Internal
  {
    template <std::size_t... n>
    consteval auto get_symmetry_index_mapping()
    {
      static_assert(
          sizeof...(n) == 2 || sizeof...(n) == 4, "Only rank 2 and rank 4 tensors are supported");

      constexpr std::array shape = {n...};
      if constexpr (sizeof...(n) == 2)
      {
        // i * (dim+1) for i = 0, 1, 2, ..., dim-1
        // i * (dim+1)+1 for i = 0, ..., dim-2 or (i+1) * (dim+1)-1 for i = 0, ..., dim-2
        // i * (dim+1)+2 for i = 0, ..., dim-3 or (i+2) * (dim+1)-2 for i = 0, ..., dim-3
        // ...

        static_assert(shape[0] == shape[1], "Symmetric tensor must have equal dimensions");
        constexpr std::size_t dim = shape[0];
        constexpr std::size_t full_size = dim * dim;
        constexpr std::size_t symmetric_size = dim * (dim + 1) / 2;
        std::array<std::size_t, full_size> indexes{};

        std::size_t offset = 0;
        std::size_t j = 0;
        for (std::size_t i = 0; i < symmetric_size; ++i)
        {
          if (j >= dim - offset)
          {
            offset++;
            j = 0;
          }
          if (j < dim - offset)
          {
            indexes[j * (dim + 1) + offset] = i;
            indexes[(j + offset) * (dim + 1) - offset] = i;
          }

          ++j;
        }

        return indexes;
      }
      else if constexpr (sizeof...(n) == 4)
      {
        static_assert(shape[0] == shape[1] && shape[2] == shape[3],
            "In symmetric tensor 4th order tensors, the first two and the last two dimensions must "
            "match (for the implemented type of symmetry)");

        // reuse the same logic pairwise for the first two and the last two dimensions
        std::array indices_i =
            get_symmetry_index_mapping<shape[0], shape[1]>();  // 1, 3, 5, 3, 2, ...
        std::array indices_j =
            get_symmetry_index_mapping<shape[2], shape[3]>();  // 1, 3, 5, 3, 2, ...


        constexpr std::size_t symmetric_size_i = shape[0] * (shape[0] + 1) / 2;
        constexpr std::size_t symmetric_size_j = shape[2] * (shape[2] + 1) / 2;
        constexpr std::size_t full_size = indices_i.size() * indices_j.size();
        std::array<std::size_t, full_size> indexes{};

        for (std::size_t j = 0; j < indices_j.size(); ++j)
        {
          for (std::size_t i = 0; i < indices_i.size(); ++i)
          {
            indexes[i + j * indices_j.size()] =
                Internal::get_flat_index<Internal::OrderType::column_major,
                    Internal::TensorBoundCheck::no_check, symmetric_size_i, symmetric_size_j>(
                    indices_i[i], indices_j[j]);
          }
        }

        return indexes;
      }
    }

    template <std::size_t... n>
    consteval auto get_index_reorder_from_symmetric_tensor()
    {
      return get_symmetry_index_mapping<n...>();
    }

    template <std::size_t... n>
    consteval auto get_index_reorder_to_symmetric_tensor()
    {
      constexpr std::array index_mapping = get_symmetry_index_mapping<n...>();
      constexpr std::size_t max = std::ranges::max(index_mapping);

      std::array<std::size_t, max + 1> reverse_mapping = {};
      for (std::size_t i = 0; i < index_mapping.size(); ++i)
      {
        reverse_mapping[index_mapping[i]] = i;
      }
      return reverse_mapping;
    }

    template <std::size_t... n>
    consteval auto get_enforced_equality_pairs()
    {
      auto evaluator = [](auto eval)
      {
        constexpr std::array index_reorder =
            Internal::get_index_reorder_from_symmetric_tensor<n...>();
        for (auto i = index_reorder.rbegin(); i != index_reorder.rend(); ++i)
        {
          auto j = std::find(index_reorder.begin(), index_reorder.end(), *i);

          std::size_t i_index = std::distance(i, index_reorder.rend()) - 1;
          std::size_t j_index = std::distance(index_reorder.begin(), j);


          if (i_index != j_index)
          {
            eval(i_index, j_index);
          }
        }
      };

      constexpr std::size_t size = std::invoke(
          [&]()
          {
            std::size_t size = 0;
            evaluator([&size](auto i, auto j) { size++; });
            return size;
          });

      std::array<std::pair<std::size_t, std::size_t>, size> pairs = {};
      std::size_t index = 0;

      evaluator(
          [&index, &pairs](auto i, auto j)
          {
            pairs[index] = {i, j};
            index++;
          });

      return pairs;
    }

    template <std::size_t... n>
    struct SymmetricCompression
    {
      static constexpr std::size_t compressed_size =
          get_index_reorder_to_symmetric_tensor<n...>().size();

      template <TensorBoundCheck bound_check>
      static constexpr std::size_t flatten_index(decltype(n)... i)
      {
        constexpr std::array index_reorder = get_index_reorder_from_symmetric_tensor<n...>();
        return index_reorder[get_flat_index<OrderType::column_major, bound_check, n...>(i...)];
      }
    };

    template <typename T>
    struct IsSymmetricCompressionTypeHelper : public std::false_type
    {
    };

    template <std::size_t... n>
    struct IsSymmetricCompressionTypeHelper<SymmetricCompression<n...>> : public std::true_type
    {
    };
  }  // namespace Internal

  template <typename T>
  constexpr bool is_symmetric_tensor =
      is_tensor<T> && Internal::IsSymmetricCompressionTypeHelper<
                          TensorCompressionType<std::remove_cvref_t<T>>>::value;

  /*!
   * @brief An owning, dense tensor of arbitrary rank with symmetry
   *
   * Symmetry for 2nd order tensors: A_ij = A_ji
   * Symmetry for 4th order tensors: A_ijkl = A_ijlk = A_jikl = A_jilk
   *
   * @note Only 2 and 4th order tensors are supported
   *
   * @note The underlying data storage of a symmetric 2-tensor is compatible with the stress-like
   * Voigt notation. The underlying data storage of a symmetric 4 tensor is compatible with the
   * linearization of a symmetric second order tensor in stress-like Voigt notation w.r.t. a
   * symmetric second order tensor in strain-like Voigt notation.
   *
   * @copydetails Internal::Tensor
   */
  template <typename T, std::size_t... n>
  using SymmetricTensor = TensorInternal<T, TensorStorageType::owning,
      Internal::SymmetricCompression<n...>, n...>;  // owns the memory

  /*!
   * @brief A dense view on a tensor of arbitrary rank with symmetry
   *
   * Symmetry for 2nd order tensors: A_ij = A_ji
   * Symmetry for 4th order tensors: A_ijkl = A_ijlk = A_jikl = A_jilk
   *
   * @note Only 2 and 4th order tensors are supported
   *
   * @note The underlying data storage is compatible with the stress-like Voigt notation. It can be
   * used to view a stress-like Voigt vector as a symmetric tensor.
   *
   * @copydetails Internal::Tensor
   */
  template <typename T, std::size_t... n>
  using SymmetricTensorView = TensorInternal<T, TensorStorageType::view,
      Internal::SymmetricCompression<n...>, n...>;  // view onto's other
                                                    // memory

  /*!
   * @brief Creates a SymmetricTensorView from a pointer to data and the tensor shape
   *
   * This function creates a SymmetricTensorView from a pointer to data and the tensor shape
   * specified by the template parameters. The data is expected to be stored in a contiguous memory
   * block of size @p (n*...) with voigt-like symmetric ordering.
   *
   * @tparam n The dimensions of the tensor.
   * @param data Pointer to the data.
   * @return A SymmetricTensorView of type TensorView<ValueType, n...>.
   */
  template <std::size_t... n>
  constexpr auto make_symmetric_tensor_view(auto* data)
  {
    constexpr std::size_t compressed_size = SymmetricTensorView<double, n...>::compressed_size;
    using ValueType = std::remove_pointer_t<decltype(data)>;

    std::span<ValueType, compressed_size> data_span(data, compressed_size);

    return SymmetricTensorView<ValueType, n...>(std::move(data_span));
  }

  /*!
   * @brief Returns true if the given tensor is symmetric according to the definition of the
   * SymmetricTensor
   */
  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  constexpr bool is_symmetric(
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor);

  /*!
   * @brief Returns a symmetric tensor from the given tensor.
   *
   * @note It is assumed that the input tensor is symmetric. If this is not the case, the result is
   * unspecified.
   */
  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  constexpr SymmetricTensor<Number, n...> assume_symmetry(
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor);

  /*!
   * @brief Convert a tensor to a symmetric tensor and throw if it is not symmetric.
   */
  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  SymmetricTensor<Number, n...> assert_symmetry(
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor);

  /*!
   * @brief Returns a full tensor from the given symmetric tensor.
   *
   * @note This function is typically not necessary since the interface to the outside is (in most
   * cases) identical to the normal tensor
   */
  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  constexpr auto get_full(
      const TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
          symmetric_tensor);

  /*!
   * @brief Print the tensor to the output stream
   */
  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  std::ostream& operator<<(std::ostream& os,
      const TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
          tensor);

  /*!
   * @brief Add another tensor onto this tensor
   *
   * @tparam OtherTensor
   */
  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator+=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const OtherTensor& B);

  /*!
   * @brief Subtract another tensor from this tensor
   *
   * @tparam OtherTensor
   */
  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator-=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const OtherTensor& B);

  /*!
   * @brief Scale the tensor with a scalar value
   *
   * @tparam OtherTensor
   */
  template <typename Scalar, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_scalar<Scalar>)
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator*=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const Scalar b);

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
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator/=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const Scalar b);

  /*!
   * @brief Compute the determinant of a symmetric tensor
   */
  constexpr auto det(const auto& A)
    requires(is_symmetric_tensor<decltype(A)>);

  /*!
   * @brief Compute the trace of a symmetric tensor
   */
  constexpr auto trace(const SquareTensor auto& A)
    requires(is_symmetric_tensor<decltype(A)>);

  /*!
   * @brief Compute the inverse of a symmetric tensor
   */
  auto inv(const SquareTensor auto& A)
    requires(is_symmetric_tensor<decltype(A)>);

  /*!
   * @brief Returns the tensor itself
   */
  constexpr auto transpose(const Rank2TensorConcept auto& A)
    requires(is_symmetric_tensor<decltype(A)>);

  /*!
   * @brief Computes the dot product of one tensor, where at least one of them is a symmetric
   * tensor.
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_symmetric_tensor<TensorLeft> || is_symmetric_tensor<TensorRight>)
  constexpr auto dot(const TensorLeft& a, const TensorRight& b);

  /*!
   * @brief Computes the double dot product of two tensors, where at least one of them is a
   * symmetric tensor.
   *
   * @tparam TensorLeft
   * @tparam TensorRight
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && TensorLeft::rank() >= 2 &&
             TensorRight::rank() >= 2 &&
             TensorLeft::template extent<TensorLeft::rank() - 2>() ==
                 TensorRight::template extent<0>() &&
             TensorLeft::template extent<TensorLeft::rank() - 1>() ==
                 TensorRight::template extent<1>() &&
             (is_symmetric_tensor<TensorLeft> || is_symmetric_tensor<TensorRight>))
  constexpr auto ddot(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Scales the tensor with a scalar value
   */
  template <typename Tensor, typename Scalar>
    requires(is_scalar<Scalar> && is_symmetric_tensor<Tensor>)
  constexpr auto scale(const Tensor& tensor, const Scalar& b);

  /*!
   * @brief Adds two symmetric tensors
   */
  template <typename TensorLeft, typename TensorRight>
    requires(std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight>)
  constexpr auto add(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Subtracts two symmetric tensors
   */
  template <typename TensorLeft, typename TensorRight>
    requires(std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight>)
  constexpr auto subtract(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Computes the dyadic product of two tensors, where both of them is a symmetric
   * tensor
   */
  template <typename TensorLeft, typename TensorRight>
    requires(is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight>)
  constexpr auto dyadic(const TensorLeft& A, const TensorRight& B);

  /*!
   * @brief Computes the dyadic product of @p a with itself.
   *
   * @tparam num_dyads The number of dyads to compute. Default is 2, which means the result is a
   * symmetric tensor of rank 2 (a \otimes a). If set to 4, the result is a symmetric tensor of rank
   * 4 (a \otimes a \otimes a \otimes a).
   */
  template <unsigned num_dyads = 2>
  constexpr auto self_dyadic(const auto& a)
    requires(is_tensor<decltype(a)> && std::remove_cvref_t<decltype(a)>::rank() == 1);

  // actual implementations
  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  constexpr bool is_symmetric(
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor)
  {
    for (const auto& [i, j] : Internal::get_enforced_equality_pairs<n...>())
    {
      if (tensor.container()[i] != tensor.container()[j]) return false;
    }
    return true;
  }

  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  constexpr SymmetricTensor<Number, n...> assume_symmetry(
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor)
  {
    constexpr std::array index_reorder = Internal::get_index_reorder_to_symmetric_tensor<n...>();
    SymmetricTensor<Number, n...> symmetric_tensor;
    std::transform(index_reorder.begin(), index_reorder.end(), symmetric_tensor.data(),
        [&tensor](const auto& i) { return tensor.container()[i]; });
    return symmetric_tensor;
  }

  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  SymmetricTensor<Number, n...> assert_symmetry(
      const TensorInternal<Number, storage_type, NoCompression<n...>, n...>& tensor)
  {
    FOUR_C_ASSERT_ALWAYS(is_symmetric(tensor), "The tensor is not symmetric.");

    return assume_symmetry(tensor);
  }

  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  constexpr auto get_full(
      const TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
          symmetric_tensor)
  {
    // returns a full tensor of the input tensor
    constexpr std::array index_reorder = Internal::get_index_reorder_from_symmetric_tensor<n...>();

    Tensor<std::remove_cvref_t<Number>, n...> tensor;
    std::transform(index_reorder.begin(), index_reorder.end(), tensor.data(),
        [&symmetric_tensor](const auto& i) { return symmetric_tensor.container()[i]; });
    return tensor;
  }

  template <typename Number, TensorStorageType storage_type, std::size_t... n>
  std::ostream& operator<<(std::ostream& os,
      const TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
          tensor)
  {
    constexpr auto tensor_type = []() consteval
    {
      if constexpr (storage_type == TensorStorageType::owning)
        return "SymmetricTensor";
      else if constexpr (storage_type == TensorStorageType::view)
        return "SymmetricTensorView";
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
    print_values(os, get_full(tensor));

    return os;
  }

  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator+=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const OtherTensor& B)
  {
    DenseFunctions::update<
        typename TensorInternal<Number, storage_type, NoCompression<n...>, n...>::value_type,
        Internal::SymmetricCompression<n...>::compressed_size, 1>(
        1.0, tensor.data(), 1.0, B.data());
    return tensor;
  }

  template <typename OtherTensor, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_tensor<OtherTensor> &&
             std::is_same_v<typename OtherTensor::shape_type, typename OtherTensor::shape_type>)
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator-=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const OtherTensor& B)
  {
    DenseFunctions::update<
        typename TensorInternal<Number, storage_type, NoCompression<n...>, n...>::value_type,
        Internal::SymmetricCompression<n...>::compressed_size, 1>(
        1.0, tensor.data(), -1.0, B.data());
    return tensor;
  }

  template <typename Scalar, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_scalar<Scalar>)
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator*=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const Scalar b)
  {
    for (auto& value : tensor.container()) value *= b;
    return tensor;
  }

  template <typename Scalar, typename Number, TensorStorageType storage_type, std::size_t... n>
    requires(is_scalar<Scalar>)
  constexpr TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>&
  operator/=(
      TensorInternal<Number, storage_type, Internal::SymmetricCompression<n...>, n...>& tensor,
      const Scalar b)
  {
    if constexpr (std::is_floating_point_v<Scalar>)
    {
      tensor *= Scalar(1) / b;
    }
    else
    {
      for (auto& value : tensor.container()) value /= b;
    }
    return tensor;
  }

  namespace Internal
  {
    auto get_reduced_dimensional_symmetric_tensor(
        const auto& tensor, const auto& off_diagonal_scale_factor)
      requires(std::remove_cvref_t<decltype(tensor)>::rank() == 2 &&
               is_symmetric_tensor<decltype(tensor)>)
    {
      using TensorType = std::remove_cvref_t<decltype(tensor)>;
      using ValueType = std::remove_cv_t<typename TensorType::value_type>;
      constexpr std::size_t size = TensorType::template extent<0>();

      std::array<ValueType, TensorType::compressed_size> container;
      std::ranges::copy(tensor.container(), container.begin());
      Tensor<ValueType, TensorType::compressed_size> reduced_tensor(std::move(container));

      // Scale off-diagonal elements of a symmetric 2-tensor
      std::for_each(reduced_tensor.container().begin() + size, reduced_tensor.container().end(),
          [&off_diagonal_scale_factor](auto& value) { value *= off_diagonal_scale_factor; });

      return reduced_tensor;
    }

    auto get_reduced_dimensional_symmetric_tensor(const auto& tensor)
      requires(std::remove_cvref_t<decltype(tensor)>::rank() == 4 &&
               is_symmetric_tensor<decltype(tensor)>)
    {
      using TensorType = std::remove_cvref_t<decltype(tensor)>;
      constexpr std::size_t size_left = TensorType::template extent<0>();
      constexpr std::size_t size_right = TensorType::template extent<2>();

      constexpr std::size_t new_size_left = size_left * (size_left + 1) / 2;
      constexpr std::size_t new_size_right = size_right * (size_right + 1) / 2;

      using ValueType = std::remove_cv_t<typename TensorType::value_type>;

      std::array<ValueType, new_size_left * new_size_right> container;
      std::ranges::copy(tensor.container(), container.begin());
      Tensor<ValueType, new_size_left, new_size_right> reduced_tensor(std::move(container));
      return reduced_tensor;
    }

    auto interpret_symmetric(const auto& tensor)
      requires(std::remove_cvref_t<decltype(tensor)>::rank() == 1)
    {
      using TensorType = std::remove_cvref_t<decltype(tensor)>;
      constexpr std::size_t size = TensorType::template extent<0>();

      constexpr std::size_t symmetric_size = [](auto size) consteval
      {
        switch (size)
        {
          case 3:
            return 2;
          case 6:
            return 3;
          case 10:
            return 4;
          case 15:
            return 5;
          case 21:
            return 6;
          default:
            FOUR_C_THROW("Unsupported symmetric tensor size: " + std::to_string(size));
        }
      }(size);

      using ValueType = std::remove_cv_t<typename TensorType::value_type>;

      auto container = tensor.container();
      SymmetricTensor<ValueType, symmetric_size, symmetric_size> symmetric_tensor(
          std::move(container));
      return symmetric_tensor;
    }
  }  // namespace Internal

  // define tensor operations for symmetric tensors
  constexpr auto det(const auto& A)
    requires(is_symmetric_tensor<decltype(A)>)
  {
    return det(get_full(A));
  }

  constexpr auto trace(const SquareTensor auto& A)
    requires(is_symmetric_tensor<decltype(A)>)
  {
    return trace(get_full(A));
  }


  auto inv(const SquareTensor auto& A)
    requires(is_symmetric_tensor<decltype(A)>)
  {
    return assume_symmetry(inv(get_full(A)));
  }


  constexpr auto transpose(const Rank2TensorConcept auto& A)
    requires(is_symmetric_tensor<decltype(A)>)
  {
    return A;
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_symmetric_tensor<TensorLeft> || is_symmetric_tensor<TensorRight>)
  constexpr auto dot(const TensorLeft& a, const TensorRight& b)
  {
    constexpr auto get_full_tensor = [](const auto& tensor) constexpr
    {
      if constexpr (is_symmetric_tensor<decltype(tensor)>)
      {
        return get_full(tensor);
      }
      else
      {
        return tensor;
      }
    };
    return dot(get_full_tensor(a), get_full_tensor(b));
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_tensor<TensorLeft> && is_tensor<TensorRight> && TensorLeft::rank() >= 2 &&
             TensorRight::rank() >= 2 &&
             TensorLeft::template extent<TensorLeft::rank() - 2>() ==
                 TensorRight::template extent<0>() &&
             TensorLeft::template extent<TensorLeft::rank() - 1>() ==
                 TensorRight::template extent<1>() &&
             (is_symmetric_tensor<TensorLeft> || is_symmetric_tensor<TensorRight>))
  constexpr auto ddot(const TensorLeft& A, const TensorRight& B)
  {
    // First handle common cases with an optimized implementation
    if constexpr (is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight> &&
                  TensorLeft::rank() == 4 && TensorRight::rank() == 2)
    {
      return Internal::interpret_symmetric(
          dot(Internal::get_reduced_dimensional_symmetric_tensor(A),
              Internal::get_reduced_dimensional_symmetric_tensor(B, 2)));
    }
    else if constexpr (is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight> &&
                       TensorLeft::rank() == 2 && TensorRight::rank() == 4)
    {
      return Internal::interpret_symmetric(
          dot(Internal::get_reduced_dimensional_symmetric_tensor(A, 2),
              Internal::get_reduced_dimensional_symmetric_tensor(B)));
    }

    // All remaining cases are handled by the generic implementation
    constexpr auto get_full_tensor = [](const auto& tensor) constexpr
    {
      if constexpr (is_symmetric_tensor<decltype(tensor)>)
      {
        return get_full(tensor);
      }
      else
      {
        return tensor;
      }
    };

    constexpr auto maybe_symmetrify = [](const auto& tensor) constexpr
    {
      if constexpr (((is_symmetric_tensor<TensorLeft> && TensorLeft::rank() == 4 &&
                         TensorRight::rank() == 2) ||
                        (is_symmetric_tensor<TensorRight> && TensorRight::rank() == 4 &&
                            TensorLeft::rank() == 2)))
      {
        // result is symmetric if one tensor is a symmetric 4th order tensor and the other one is a
        // second order tensor
        return assume_symmetry(tensor);
      }
      else
      {
        // Otherwise, the tensor is not guaranteed to be symmetric
        return tensor;
      }
    };

    return maybe_symmetrify(ddot(get_full_tensor(A), get_full_tensor(B)));
  }

  template <typename Tensor, typename Scalar>
    requires(is_scalar<Scalar> && is_symmetric_tensor<Tensor>)
  constexpr auto scale(const Tensor& tensor, const Scalar& b)
  {
    using result_value_type =
        FADUtils::ScalarOperationResultType<std::multiplies<>, Scalar, typename Tensor::value_type>;

    OwningTensorType<Tensor, result_value_type> tens_out;

    std::transform(tensor.data(), tensor.data() + Tensor::compressed_size, tens_out.data(),
        [&b](const auto& value) { return value * b; });

    return tens_out;
  }

  template <typename TensorLeft, typename TensorRight>
    requires(std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight>)
  constexpr auto add(const TensorLeft& A, const TensorRight& B)
  {
    using result_value_type = FADUtils::ScalarOperationResultType<std::plus<>,
        typename TensorLeft::value_type, typename TensorRight::value_type>;

    OwningTensorType<TensorLeft, result_value_type> tens_out;

    DenseFunctions::update<result_value_type, TensorLeft::compressed_size, 1>(
        tens_out.data(), A.data(), B.data());

    return tens_out;
  }

  template <typename TensorLeft, typename TensorRight>
    requires(std::is_same_v<typename TensorLeft::shape_type, typename TensorRight::shape_type> &&
             is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight>)
  constexpr auto subtract(const TensorLeft& A, const TensorRight& B)
  {
    using result_value_type = FADUtils::ScalarOperationResultType<std::minus<>,
        typename TensorLeft::value_type, typename TensorRight::value_type>;

    OwningTensorType<TensorLeft, result_value_type> tens_out;

    DenseFunctions::update<result_value_type, TensorLeft::compressed_size, 1>(
        tens_out.data(), 1.0, A.data(), -1.0, B.data());

    return tens_out;
  }

  template <typename TensorLeft, typename TensorRight>
    requires(is_symmetric_tensor<TensorLeft> && is_symmetric_tensor<TensorRight>)
  constexpr auto dyadic(const TensorLeft& A, const TensorRight& B)
  {
    return assume_symmetry(dyadic(get_full(A), get_full(B)));
  }

  template <unsigned num_dyads>
  constexpr auto self_dyadic(const auto& a)
    requires(is_tensor<decltype(a)> && std::remove_cvref_t<decltype(a)>::rank() == 1)
  {
    static_assert(
        num_dyads == 2 || num_dyads == 4, "Only self dyadic with two or four dyads are supported.");
    if constexpr (num_dyads == 2)
    {
      return assume_symmetry(dyadic(a, a));
    }
    else if constexpr (num_dyads == 4)
    {
      const auto a_dyadic_a = self_dyadic<2>(a);
      return dyadic(a_dyadic_a, a_dyadic_a);
    }
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif