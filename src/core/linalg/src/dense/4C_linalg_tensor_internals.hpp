// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_INTERNALS_HPP
#define FOUR_C_LINALG_TENSOR_INTERNALS_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_tensor_meta_utils.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  enum class TensorStorageType
  {
    owning,
    view
  };

  template <typename Tensor>
  constexpr bool is_compressed_tensor = std::remove_cvref_t<Tensor>::is_compressed;

  /*!
   * @brief General tensor class for dense tensors of arbitrary rank
   *
   * This class is designed for small tensors typically used in physics simulations, such as
   * strains, stresses, and linearizations at the Gauss point level. The tensor dimensions are
   * usually small (e.g., 2x2 or 3x3), with a rank of 1 or 2, and rarely rank 4. This
   * results in typically 9 or fewer entries (rarely 81 for 4-tensors in 3D). The data is
   * stored on the stack avoiding dynamic memory allocation and ensuring efficient memory access
   * via typically high locality.
   *
   * Key characteristics:
   * - **Stack Allocation**: The tensor's data (if it is owning) is stored on the stack, avoiding
   * the overhead of dynamic memory allocation.
   * - **Temporary Objects**: Operations on tensors may create temporary objects, but these do not
   *   involve dynamic memory allocation, ensuring efficient computation.
   * - **Use Case**: This class is ideal for small-scale tensor operations at the Gauss point
   * level in finite element simulations. It allows to write physical equations close to their
   * pendant on paper.
   *
   * Comparison with @p LinAlg::Matrix:
   * - **Tensor**: Designed for small, fixed-size data stored on the stack. Suitable for
   * operations at the Gauss point level (e.g., strains, stresses).
   * - **Matrix**: Designed for larger, dynamically allocated data. Suitable for element-level
   *   calculations (e.g., stiffness and mass matrices). For example, a stiffness matrix for a
   *   hex27 element may have dimensions 81x81 with 6561 entries, requiring dynamic memory
   * allocation.
   *
   * Shared Backend:
   * Both @p LinAlg::Tensor and @p LinAlg::Matrix share the same backend for matrix and vector
   * operations, ensuring consistency and avoiding duplication of functionality.
   *
   * @tparam Number Type of the tensor elements (e.g., `double`).
   * @tparam storage_type Specifies whether the tensor owns its data (`TensorStorageType::owning`)
   * or is a view (`TensorStorageType::view`).
   * @tparam Compression The compression of the tensor defines how the indices of the tensor (i, j,
   * ...) are mapped to the underlying data storage. With @p NoCompression, the underlying data is
   * stored in column-major layout. Compression can be used to efficiently store tensors with
   * internal constraints (like symmetry). Symmetric tensors can be stored in a compact form using
   * Voigt-like notations, see @p SymmetricCompression.
   * @tparam n Dimensions of the tensor, specified as a variadic template parameter.
   */
  template <typename Number, TensorStorageType storage_type, typename Compression, std::size_t... n>
  class TensorInternal
  {
   public:
    using value_type = Number;
    using shape_type = std::integer_sequence<std::size_t, n...>;

    static constexpr bool is_compressed = (n * ...) != Compression::compressed_size;


    static constexpr std::size_t rank_ = sizeof...(n);
    static constexpr std::size_t size_ = (n * ...);
    static constexpr std::size_t compressed_size = Compression::compressed_size;

    static_assert(storage_type == TensorStorageType::view or
                      std::is_same_v<std::remove_cv_t<value_type>, value_type>,
        "Owning Tensor must have a non-const, non-volatile value_type");
    static_assert(storage_type == TensorStorageType::owning or
                      std::is_same_v<std::remove_volatile_t<value_type>, value_type>,
        "TensorView must have a non-volatile value_type");
    static_assert(sizeof...(n) > 0, "Tensor must have at least rank one");
    static_assert(
        []() constexpr
        {
          std::array shape = {n...};
          return std::ranges::all_of(shape, [](auto dim) { return dim > 0; });
        }(),
        "The extents of all axes must be larger than zero");

    // Container is either a std::array for an owning tensor or a std::span for a view
    using ContainerType = std::conditional_t<storage_type == TensorStorageType::owning,
        std::array<Number, Compression::compressed_size>,
        std::span<Number, Compression::compressed_size>>;

   private:
    ContainerType data_;

   public:
    /*!
     * @brief Default constructor
     *
     * @note Only an owning tensor has the default constructor
     */
    TensorInternal()
      requires(std::is_default_constructible_v<ContainerType>)
    = default;

    TensorInternal(const TensorInternal&) = default;

    /*!
     * @brief Move constructor
     *
     * @note Copy operation of a view is cheap, so we don't need to have a move constructor
     */
    TensorInternal(TensorInternal&&) noexcept
      requires(storage_type == TensorStorageType::owning)
    = default;
    TensorInternal& operator=(const TensorInternal&) noexcept = default;

    /*!
     * @brief Move assignment operator
     *
     * @note Copy operation of a view is cheap, so we don't need to have a move assignment
     * operator
     */
    TensorInternal& operator=(TensorInternal&&) noexcept
      requires(storage_type == TensorStorageType::owning)
    = default;
    ~TensorInternal() = default;

    /*!
     * @brief Construct an owning tensor from given data
     *
     * You can assign the tensor via
     *
     * @code
     * Tensor<double, 3, 4> A = {{
     *     {1.0, 2.0, 3.0, 4.0},
     *     {5.0, 6.0, 7.0, 8.0},
     *     {9.0, 10.0, 11.0, 12.0}
     * }};
     * @endcode
     *
     */
    TensorInternal(const Internal::TensorInitializerList<Number, n...>::type& lst)
      requires(std::is_default_constructible_v<ContainerType> && !is_compressed);

    /*!
     * @brief Create a view on an owning tensor
     */
    TensorInternal(TensorInternal<Number, TensorStorageType::owning, Compression, n...>& view_on)
      requires(storage_type == TensorStorageType::view)
        : data_(view_on.container())
    {
    }

    /*!
     * @brief Create a constant view on a const owning tensor
     */
    TensorInternal(const TensorInternal<std::remove_cv_t<Number>, TensorStorageType::owning,
        Compression, n...>& view_on)
      requires(storage_type == TensorStorageType::view && std::is_const_v<Number>)
        : data_(view_on.container())
    {
    }

    /*!
     * @brief Create a copy of a view
     */
    TensorInternal(
        const TensorInternal<std::remove_cv_t<Number>, TensorStorageType::view, Compression, n...>&
            other)
      requires(storage_type == TensorStorageType::owning)
    {
      std::ranges::copy(other.container(), data_.begin());
    }

    TensorInternal(ContainerType&& data) : data_(std::move(data)) {}

    /*!
     * @brief Access to the underlying raw data of the tensor
     *
     * @return Number*
     */
    [[nodiscard]] Number* data() { return data_.data(); }

    /*!
     * @brief Access to the underlying raw data of the tensor for readonly access
     *
     * @return Number*
     */
    [[nodiscard]] const Number* data() const { return data_.data(); }

    /*!
     * @brief Access to the underlying container (std::array or std::span, depending on whether it
     * is owning or a view)
     *
     * @return Number*
     */
    [[nodiscard]] ContainerType& container() { return data_; }

    /*!
     * @brief Access to the underlying readonly container (std::array or std::span, depending on
     * whether it is owning or a view)
     *
     * @return Number*
     */
    [[nodiscard]] const ContainerType& container() const { return data_; }

    /*!
     * @brief Indexing operator to access individual values of the tensor (without bound checks)
     *
     * @param i
     * @return Number&
     */
    [[nodiscard]] Number& operator()(decltype(n)... i);

    /*!
     * @brief Indexing operator to access individual values of the tensor in readonly mode
     * (without bound checks)
     *
     * @param i
     * @return Number&
     */
    [[nodiscard]] const Number& operator()(decltype(n)... i) const;

    /*!
     * @brief Indexing operator to access individual values of the tensor (with bound checks)
     *
     * @param i
     * @return Number&
     */
    [[nodiscard]] Number& at(decltype(n)... i);

    /*!
     * @brief Indexing operator to access individual values of the tensor in readonly mode
     * (with bound checks)
     *
     * @param i
     * @return Number&
     */
    [[nodiscard]] const Number& at(decltype(n)... i) const;

    [[nodiscard]] static constexpr std::size_t rank() { return rank_; }
    [[nodiscard]] static constexpr std::size_t size() { return size_; }

    /*!
     * @brief Returns the dimension of the i-th axis of the tensor
     *
     * @tparam i
     */
    template <std::size_t i>
      requires(i < rank())
    [[nodiscard]] static constexpr auto extent()
    {
      return std::get<i>(std::make_tuple(n...));
    }

    /*!
     * @brief Returns a std::tuple of the shape of the tensor
     *
     * @return constexpr auto
     */
    [[nodiscard]] static constexpr auto shape() { return std::make_tuple(n...); }

    /*!
     * @brief Fills the tensor with the given value
     *
     * @param value
     */
    constexpr void fill(const Number& value) { std::ranges::fill(data_, value); }
  };

  template <std::size_t... n>
  constexpr auto array_of_tensor_indices = Internal::get_array_of_indices<n...>();

  namespace Internal
  {
    template <typename Tensor>
    struct CompressionTypeHelper;

    template <typename Number, TensorStorageType storage_type, typename Compression,
        std::size_t... n>
    struct CompressionTypeHelper<TensorInternal<Number, storage_type, Compression, n...>>
    {
      using type = Compression;
    };
  }  // namespace Internal

  template <typename Tensor>
  using TensorCompressionType =
      typename Internal::CompressionTypeHelper<std::remove_cvref_t<Tensor>>::type;

  // actual implementations

  template <typename Number, TensorStorageType storage_type, typename Compression, std::size_t... n>
  TensorInternal<Number, storage_type, Compression, n...>::TensorInternal(
      const Internal::TensorInitializerList<Number, n...>::type& lst)
    requires(std::is_default_constructible_v<ContainerType> && !is_compressed)
  {
    constexpr std::array<std::size_t, size()> index_mapping = Internal::order_type_mapping<n...>();
    for (std::size_t i = 0; i < size(); ++i)
    {
      data_[index_mapping[i]] = *(Internal::get_view_to_first_element<Number, n...>(lst) + i);
    }
  }

  template <typename Number, TensorStorageType storage_type, typename Compression, std::size_t... n>
  Number& TensorInternal<Number, storage_type, Compression, n...>::operator()(decltype(n)... i)
  {
    return data_[Compression::template flatten_index<Internal::TensorBoundCheck::no_check>(i...)];
  }

  template <typename Number, TensorStorageType storage_type, typename Compression, std::size_t... n>
  const Number& TensorInternal<Number, storage_type, Compression, n...>::operator()(
      decltype(n)... i) const
  {
    return data_[Compression::template flatten_index<Internal::TensorBoundCheck::no_check>(i...)];
  }

  template <typename Number, TensorStorageType storage_type, typename Compression, std::size_t... n>
  Number& TensorInternal<Number, storage_type, Compression, n...>::at(decltype(n)... i)
  {
    return data_[Compression::template flatten_index<Internal::TensorBoundCheck::check>(i...)];
  }

  template <typename Number, TensorStorageType storage_type, typename Compression, std::size_t... n>
  const Number& TensorInternal<Number, storage_type, Compression, n...>::at(decltype(n)... i) const
  {
    return data_[Compression::template flatten_index<Internal::TensorBoundCheck::check>(i...)];
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif