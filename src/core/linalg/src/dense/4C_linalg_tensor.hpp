// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_TENSOR_HPP
#define FOUR_C_LINALG_TENSOR_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{

  namespace Internal
  {
    enum class TensorBoundCheck
    {
      no_check,
      check_with_assertions,
      check
    };

    template <std::size_t... n>
    constexpr std::size_t prod()
    {
      if constexpr (sizeof...(n) == 0)
        return 1;
      else
        return (n * ...);
    }

    template <TensorBoundCheck bound_check, std::size_t... n>
    struct FlatIndexHelper;

    template <TensorBoundCheck bound_check>
    struct FlatIndexHelper<bound_check>
    {
      static constexpr std::size_t get_flat_index() { return 0; }
    };

    template <TensorBoundCheck bound_check, std::size_t n1, std::size_t... n>
    struct FlatIndexHelper<bound_check, n1, n...>
    {
      static constexpr std::size_t get_flat_index(std::size_t i1, decltype(n)... i)
      {
        if constexpr (bound_check == TensorBoundCheck::check)
        {
          FOUR_C_ASSERT_ALWAYS(
              i1 < n1, "An index exceeds the bounds. {} must be smaller than {}", i1, n1);
        }
        else if constexpr (bound_check == TensorBoundCheck::check_with_assertions)
        {
          FOUR_C_ASSERT(i1 < n1, "An index exceeds the bounds. {} must be smaller than {}", i1, n1);
        }

        return i1 * prod<n...>() + FlatIndexHelper<bound_check, n...>::get_flat_index(i...);
      }
    };

    template <typename Number, std::size_t... n>
    struct TensorInitializerList;

    template <typename Number, std::size_t n1>
    struct TensorInitializerList<Number, n1>
    {
      using type = Number[n1];
    };

    template <typename Number, std::size_t n1, std::size_t... n>
    struct TensorInitializerList<Number, n1, n...>
    {
      using type = typename TensorInitializerList<Number, n...>::type[n1];
    };

    template <typename Number, std::size_t... n>
    struct MultiFor;

    template <typename Number, std::size_t n1>
    struct MultiFor<Number, n1>
    {
      static constexpr void multi_for(
          const TensorInitializerList<Number, n1>::type& lst, auto function, std::size_t& i)
      {
        for (const auto& value : lst)
        {
          function(i, value);
          ++i;
        }
      }
    };

    template <typename Number, std::size_t n1, std::size_t... n>
    struct MultiFor<Number, n1, n...>
    {
      static constexpr void multi_for(
          const TensorInitializerList<Number, n1, n...>::type& lst, auto function, std::size_t& i)
      {
        for (const auto& value : lst)
        {
          MultiFor<Number, n...>::multi_for(value, function, i);
        }
      }
    };

    template <typename Number, std::size_t... n>
    void multi_for(const typename TensorInitializerList<Number, n...>::type& lst, auto function)
    {
      std::size_t i = 0;
      MultiFor<Number, n...>::multi_for(lst, function, i);
    }
  }  // namespace Internal


  template <std::size_t... n>
  constexpr std::size_t get_flat_index(decltype(n)... i)
  {
    return Internal::FlatIndexHelper<Internal::TensorBoundCheck::no_check, n...>::get_flat_index(
        i...);
  }

  template <std::size_t... n>
  constexpr std::size_t get_flat_index_with_bound_check(decltype(n)... i)
  {
    return Internal::FlatIndexHelper<Internal::TensorBoundCheck::check, n...>::get_flat_index(i...);
  }

  template <std::size_t... n>
  constexpr std::size_t get_flat_index_with_bound_assert(decltype(n)... i)
  {
    return Internal::FlatIndexHelper<Internal::TensorBoundCheck::check_with_assertions,
        n...>::get_flat_index(i...);
  }

  template <typename Number, std::size_t... n>
  class Tensor
  {
   public:
    using value_type = Number;

    static constexpr std::size_t rank_ = sizeof...(n);
    static constexpr std::size_t size_ = (n * ...);

   private:
    std::array<Number, size_> data_{0};

   public:
    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    Tensor(const Internal::TensorInitializerList<Number, n...>::type& lst)
    {
      Internal::multi_for<Number, n...>(
          lst, [&](std::size_t i, const auto& value) { data_[i] = value; });
    }

    [[nodiscard]] Number* data() { return data_.data(); }

    [[nodiscard]] const Number* data() const { return data_.data(); }

    [[nodiscard]] Number& operator()(decltype(n)... i) { return data_[get_flat_index<n...>(i...)]; }

    [[nodiscard]] const Number& operator()(decltype(n)... i) const
    {
      return data_[get_flat_index<n...>(i...)];
    }

    [[nodiscard]] Number& at(decltype(n)... i)
    {
      return data_[get_flat_index_with_bound_check<n...>(i...)];
    }

    [[nodiscard]] const Number& at(decltype(n)... i) const
    {
      return data_[get_flat_index_with_bound_check<n...>(i...)];
    }

    [[nodiscard]] constexpr std::size_t rank() const { return rank_; }
    [[nodiscard]] constexpr std::size_t size() const { return size_; }
    [[nodiscard]] constexpr auto shape() const { return std::make_tuple(n...); }

    void fill(const Number& value) { std::ranges::fill(data_, value); }
  };

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif