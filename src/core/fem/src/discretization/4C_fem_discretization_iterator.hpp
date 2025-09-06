// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DISCRETIZATION_ITERATOR_HPP
#define FOUR_C_FEM_DISCRETIZATION_ITERATOR_HPP

#include "4C_config.hpp"

#include <iterator>
#include <span>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;

  /**
   * @brief A proxy object that behaves like a reference for iterator dereferencing.
   */
  template <typename RefType>
  class IteratorProxy
  {
   public:
    explicit IteratorProxy(RefType&& value) : value_(std::move(value)) {}

    const RefType* operator->() const { return &value_; }
    RefType* operator->() { return &value_; }

    const RefType& operator*() const { return value_; }
    RefType& operator*() { return value_; }

   private:
    RefType value_;
  };

  /**
   * @brief An iterator over entities in a Discretization.
   *
   * This iterator supports iteration over a contiguous index array. Dereferencing the iterator
   * returns an object of type RefType, referring to the entity at the current index.
   * The iterator supports iteration over const and non-const Discretization objects.
   */
  template <typename RefType>
  class DiscretizationIterator
  {
   public:
    //! The type of internal index that is used to construct RefType objects
    using IndexType = int;
    using DiscretizationType = typename RefType::DiscretizationType;

    using iterator_category = std::random_access_iterator_tag;
    using value_type = RefType;
    using difference_type = std::ptrdiff_t;
    using reference = value_type;
    using pointer = IteratorProxy<value_type>;

    constexpr DiscretizationIterator() noexcept : discretization_(nullptr), index_ptr_(nullptr) {}

    constexpr DiscretizationIterator(
        DiscretizationType* discretization, const IndexType* index_ptr) noexcept
        : discretization_(discretization), index_ptr_(index_ptr)
    {
    }

    constexpr reference operator*() const { return RefType(discretization_, *index_ptr_); }
    constexpr pointer operator->() const { return pointer(RefType(discretization_, *index_ptr_)); }

    constexpr DiscretizationIterator& operator++() noexcept
    {
      ++index_ptr_;
      return *this;
    }
    constexpr DiscretizationIterator operator++(int) noexcept
    {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    constexpr DiscretizationIterator& operator--() noexcept
    {
      --index_ptr_;
      return *this;
    }
    constexpr DiscretizationIterator operator--(int) noexcept
    {
      auto tmp = *this;
      --(*this);
      return tmp;
    }

    constexpr DiscretizationIterator& operator+=(difference_type n) noexcept
    {
      index_ptr_ += n;
      return *this;
    }
    constexpr DiscretizationIterator& operator-=(difference_type n) noexcept
    {
      index_ptr_ -= n;
      return *this;
    }

    constexpr DiscretizationIterator operator+(difference_type n) const noexcept
    {
      return DiscretizationIterator(discretization_, index_ptr_ + n);
    }
    friend constexpr DiscretizationIterator operator+(
        difference_type n, const DiscretizationIterator& it) noexcept
    {
      return it + n;
    }

    constexpr DiscretizationIterator operator-(difference_type n) const noexcept
    {
      return DiscretizationIterator(discretization_, index_ptr_ - n);
    }
    constexpr difference_type operator-(const DiscretizationIterator& other) const noexcept
    {
      return index_ptr_ - other.index_ptr_;
    }

    constexpr reference operator[](difference_type n) const
    {
      return RefType(discretization_, *(index_ptr_ + n));
    }

    friend constexpr bool operator==(
        const DiscretizationIterator& a, const DiscretizationIterator& b) noexcept
    {
      return a.discretization_ == b.discretization_ && a.index_ptr_ == b.index_ptr_;
    }
    friend constexpr bool operator!=(
        const DiscretizationIterator& a, const DiscretizationIterator& b) noexcept
    {
      return !(a == b);
    }
    friend constexpr bool operator<(
        const DiscretizationIterator& a, const DiscretizationIterator& b) noexcept
    {
      return a.index_ptr_ < b.index_ptr_;
    }
    friend constexpr bool operator>(
        const DiscretizationIterator& a, const DiscretizationIterator& b) noexcept
    {
      return b < a;
    }
    friend constexpr bool operator<=(
        const DiscretizationIterator& a, const DiscretizationIterator& b) noexcept
    {
      return !(b < a);
    }
    friend constexpr bool operator>=(
        const DiscretizationIterator& a, const DiscretizationIterator& b) noexcept
    {
      return !(a < b);
    }

   private:
    //! Discretization containing the entities being iterated over
    DiscretizationType* discretization_;
    //! Pointer into the index array that we iterate over
    const IndexType* index_ptr_;
  };

  /**
   * A range of Iterator objects for use in range-based expressions.
   */
  template <typename Iterator>
  class IteratorRange : public std::ranges::view_interface<IteratorRange<Iterator>>
  {
   public:
    IteratorRange(Iterator begin, Iterator end) : begin_(begin), end_(end) {}

    constexpr Iterator begin() const { return begin_; }
    constexpr Iterator end() const { return end_; }

    [[nodiscard]] constexpr size_t size() const { return end_ - begin_; }

   private:
    Iterator begin_;
    Iterator end_;
  };

  namespace Internal
  {
    /**
     * @brief Create an iterator range over @p RefType entities in a @p discretization.
     *
     * The range iterates over the entities specified by the given @p indices. The iterators
     * are only valid as long as the @p discretization and the @p indices are valid.
     */
    template <typename RefType, typename DiscretizationType = typename RefType::DiscretizationType>
    auto make_iterator_range(DiscretizationType* discretization, std::span<const int> indices)
    {
      return IteratorRange(DiscretizationIterator<RefType>(discretization, indices.data()),
          DiscretizationIterator<RefType>(discretization, indices.data() + indices.size()));
    }
  }  // namespace Internal
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
