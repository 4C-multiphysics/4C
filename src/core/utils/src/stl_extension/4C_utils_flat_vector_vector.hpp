// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_FLAT_VECTOR_VECTOR_HPP
#define FOUR_C_UTILS_FLAT_VECTOR_VECTOR_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <span>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{

  /**
   * @brief A memory-efficient implementation of a vector-of-vectors.
   *
   * This class stores a vector-of-vectors in a flattened format, using a single data vector
   * and an offset vector to mark the start of each inner vector. The inner vectors can be
   * of variable length. This is useful for scenarios where you need a 2D structure but want to
   * avoid the overhead of multiple allocations and indirections associated with
   * std::vector<std::vector<T>>.
   */
  template <class T>
  class FlatVectorVector
  {
   public:
    using size_type = std::size_t;

    //! Default constructor. Creates an empty FlatVectorVector.
    FlatVectorVector() = default;

    //! @brief Construct from a vector-of-vectors
    explicit FlatVectorVector(const std::vector<std::vector<T>>& nested) { from_nested(nested); }

    //! @brief Replace contents from a vector-of-vectors
    void from_nested(const std::vector<std::vector<T>>& nested)
    {
      offsets_.resize(nested.size() + 1);
      size_type total = 0;
      for (size_type i = 0; i < nested.size(); ++i)
      {
        offsets_[i] = total;
        total += nested[i].size();
      }
      offsets_[nested.size()] = total;

      data_.resize(total);
      size_type pos = 0;
      for (const auto& inner : nested)
      {
        std::copy(inner.begin(), inner.end(), data_.begin() + pos);
        pos += inner.size();
      }
    }

    /**
     * @brief Number of outer vectors
     *
     * Since this class mimics a vector-of-vectors, this method returns the number of _outer_
     * elements. Use total_size() to get the total number of stored elements.
     */
    [[nodiscard]] size_type size() const noexcept
    {
      return offsets_.empty() ? 0 : offsets_.size() - 1;
    }

    /**
     * @brief The total number of stored elements across all inner vectors.
     */
    [[nodiscard]] size_type total_size() const noexcept { return data_.size(); }

    /**
     * !brief Access an inner vector by its @p outer_index.
     */
    [[nodiscard]] std::span<T> operator[](size_type outer_index)
    {
      FOUR_C_ASSERT(outer_index < size(), "outer_index is out of range");
      return std::span<T>(
          data_.data() + offsets_[outer_index], offsets_[outer_index + 1] - offsets_[outer_index]);
    }

    /**
     * !brief Access an inner vector by its @p outer_index.
     */
    [[nodiscard]] std::span<const T> operator[](size_type outer_index) const
    {
      FOUR_C_ASSERT(outer_index < size(), "outer_index is out of range");
      return std::span<const T>(
          data_.data() + offsets_[outer_index], offsets_[outer_index + 1] - offsets_[outer_index]);
    }

   private:
    //! Flattened data storage
    std::vector<T> data_;
    //! Offsets into data_ for each outer element. The i-th inner vector is given by
    //! the data_ elements in range [offsets[i],offsets[i+1])
    std::vector<size_type> offsets_;
  };
}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
