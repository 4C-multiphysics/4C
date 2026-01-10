// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_PAIREDVECTOR_HPP
#define FOUR_C_UTILS_PAIREDVECTOR_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <iomanip>
#include <map>
#include <ostream>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Gen
{
  /**
   * @brief A substitute for std::maps, that has different storage and access
   * characteristics.
   *
   * @tparam Key Type of key
   * @tparam T   Type of element
   *
   * \note This class is no longer a pure drop-in solution for std::map. Actually
   * it can still be used as one but with restricted functionality.
   *
   * The memory is allocated beforehand to eliminate the overhead of repeated
   * memory allocation. This requires the knowledge of an upper bound on the
   * number of entries and the size is not meant to be changed after
   * initialization. There are however some instances where that is inevitable.
   * The access characteristics are equivalent to those of a vector, which is
   * the container it is based on. Note especially that the elements are not
   * sorted.
   *
   * @note This data structure should be replaced with C++23 flat_map
   * (https://en.cppreference.com/w/cpp/container/flat_map.html).
   */
  template <typename Key, typename T>
  class Pairedvector
  {
   private:
    using pair_type = std::pair<Key, T>;
    using pairedvector_type = std::vector<pair_type>;

   public:
    using iterator = typename pairedvector_type::iterator;
    using const_iterator = typename pairedvector_type::const_iterator;

    /**
     *  @brief  constructor creates no elements, but reserves the maximum
     *          number of entries.
     *  @param reserve The number of elements that are preallocated
     */
    Pairedvector(size_t reserve) : m_(reserve, pair_type()), entries_(0) {}

    /**
     *  @brief  empty constructor creates no elements and does not reserve any
     *          number of entries. Use resize as soon as you know the necessary
     *          number of elements.
     */
    Pairedvector() : entries_(0) {}

    /**
     *  @brief  constructor creates no elements, but reserves the maximum
     *          number of entries.
     *  @param reserve The number of elements that are preallocated
     *  @param default_key default value for the key within the pair
     *  @param default_T   default value for the data within the pair
     */
    Pairedvector(size_t reserve, Key default_key, T default_T)
        : m_(reserve, pair_type(default_key, default_T)), entries_(0)
    {
    }

    /**
     *  @brief  copy constructor
     *
     *  @param[in] source %pairedmatrix object we want to copy.
     *  @param[in] type   Apply this copy type.
     *
     *  */
    Pairedvector(const Pairedvector& source) : m_(0, pair_type()), entries_(0) { clone(source); }

    /**
     *  Returns a read/write iterator that points to the first
     *  element in the %Pairedvector.  Iteration is done in ordinary
     *  element order.
     */
    iterator begin() { return m_.begin(); }

    /**
     *  Returns a read-only (constant) iterator that points to the first pair
     *  in the %Pairedvector.  Iteration is done in ordinary
     *  element order.
     */
    const_iterator begin() const { return m_.begin(); }

    /**
     *  Returns a read/write iterator that points one past the last
     *  pair in the %Pairedvector.  Iteration is done in ordinary
     *  element order.
     */
    iterator end() { return m_.begin() + entries_; }

    /**
     *  Returns a read-only (constant) iterator that points one past the last
     *  pair in the %Pairedvector.  Iteration is done in ordinary
     *  element order.
     */
    const_iterator end() const { return m_.begin() + entries_; }

    /**
     *  @brief  Tries to locate an element in a %Pairedvector.
     *  @param  k  Key of (key, value) %pair to be located.
     *  @return Iterator pointing to sought-after element, or end() if not
     *          found.
     *
     *  This function takes a key and tries to locate the element with which
     *  the key matches.  If successful the function returns an iterator
     *  pointing to the sought after %pair.  If unsuccessful it returns the
     *  past-the-end ( @c end() ) iterator.
     */
    iterator find(const Key k)
    {
      iterator last = m_.begin() + entries_;
      for (iterator it = m_.begin(); it != last; ++it)
      {
        if (it->first == k) return it;
      }
      return last;
    }

    /**
     *  @brief  Tries to locate an element in a %Pairedvector.
     *  @param  k  Key of (key, value) %pair to be located.
     *  @return Read-only (constant) iterator pointing to sought-after
     *          element, or end() if not found.
     *
     *  This function takes a key and tries to locate the element with which
     *  the key matches.  If successful the function returns a constant
     *  iterator pointing to the sought after %pair. If unsuccessful it
     *  returns the past-the-end ( @c end() ) iterator.
     */
    const_iterator find(const Key k) const
    {
      const_iterator last = m_.begin() + entries_;
      for (const_iterator it = m_.begin(); it != last; ++it)
      {
        if (it->first == k) return it;
      }
      return last;
    }

    /**
     * @param  x  Data with which old elements are overwritten.
     *
     *  Erases all elements in a %Pairedvector.  Note that this function only
     *  erases the elements, and that if the elements themselves are
     *  pointers, the pointed-to memory is not touched in any way.
     *  Managing the pointer is the user's responsibility.
     */
    void clear(const pair_type& x = pair_type())
    {
      if (empty()) return;

      entries_ = 0;

      // The vector must be overwritten explicitly, to avoid holding pointers
      // to smart- or reference counting pointers.
      for (typename pairedvector_type::iterator it = m_.begin(); it != m_.end(); ++it) *it = x;
    }

    /**
     *  @brief  Resizes the %Pairedvector to the specified number of elements.
     *  @param  new_size  Number of elements the %vector should contain.
     *  @param  x  Data with which new elements should be populated.
     *
     *  This function will %resize the %Pairedvector to the specified
     *  number of elements.  If the number is smaller than the
     *  %Pairedvector's current size the %Pairedvector is truncated, otherwise
     *  the %Pairedvector is extended and new elements are populated with
     *  given data.
     */
    void resize(size_t new_size, pair_type x = pair_type())
    {
      // adapt sentinel value thresholds
      if (m_.size() >= 0) std::fill(m_.end(), m_.end(), x);

      m_.resize(new_size, x);

      // If vector is truncated to new_size, adapt number of entries.
      if (new_size < entries_) entries_ = new_size;
    }

    /**
     *  @brief  Subscript ( @c [] ) access to %Pairedvector data.
     *  @param  k  The key for which data should be retrieved.
     *  @return A reference to the data of the (key,data) %pair.
     *
     *  Allows for easy lookup with the subscript ( @c [] )
     *  operator.  Returns data associated with the key specified in
     *  subscript.  If the key does not exist, a pair with that key
     *  is created using default values, which is then returned.
     */
    T& operator[](const Key k)
    {
      iterator last = m_.begin() + entries_;
      iterator it = find(k);
      if (it != last) return it->second;

      if (entries_ >= m_.size()) throw std::length_error("Pairedvector::operator[]");

      ++entries_;
      last->first = k;
      return last->second;
    }

    /** @brief Same behavior as %operator[] */
    T& operator()(const Key k) { return operator[](k); }

    /**
     *  @brief  Access to %Pairedvector data.
     *  @param  k  The key for which data should be retrieved.
     *  @return A reference to the data whose key is equivalent to @a k, if
     *          such a data is present in the %Pairedvector.
     *  @throw  Core::Exception("invalid key")  If no such data is present.
     */
    T& at(const Key k)
    {
      auto last = m_.begin() + entries_;
      auto it = find(k);
      if (it == last) FOUR_C_THROW("Pairedvector::at(): invalid key");

      return it->second;
    }

    /**
     *  @brief  Access to %Pairedvector data.
     *  @param  k  The key for which data should be retrieved.
     *  @return A reference to the data whose key is equivalent to @a k, if
     *          such a data is present in the %Pairedvector.
     *  @throw  Core::Exception("invalid key")  If no such data is present.
     */
    const T& at(const Key k) const
    {
      auto last = m_.begin() + entries_;
      auto it = find(k);
      if (it == last) FOUR_C_THROW("Pairedvector::at(): invalid key");

      return it->second;
    }

    /** Returns the current capacity of the %Pairedvector.  */
    size_t capacity() const { return (m_.size() > 0 ? m_.size() : 0); }

    /**  Returns the number of elements in the %Pairedvector.  */
    size_t size() const { return entries_; }

    /** Returns true if the %Pairedvector is empty.  (Thus begin() would equal
     *  end().)
     */
    bool empty() const { return (entries_ <= 0); }

    /**
     *  @brief  Swaps data with another %Pairedvector.
     *  @param  x  A %Pairedvector of identical allocator type.
     *
     *  This exchanges the elements between two vectors in constant time.
     *  (Three pointers and the entries information, so it should be quite fast.)
     *
     *  */
    void swap(Pairedvector& x)
    {
      // swap internal data structure
      m_.swap(x.m_);

      // swap entry number
      const size_t my_entries = entries_;
      entries_ = x.entries_;
      x.entries_ = my_entries;
    }

    /**
     *  @brief  assign a source %Pairedvector to this %Pairedvector
     *  @param  a %Pairedvector of identical allocator type.
     *
     *  If necessary the capacity of this %Pairedvector will be modified. All
     *  previous elements are lost and will be overwritten by the source values.
     */
    Pairedvector& operator=(const Pairedvector& source)
    {
      clone(source);
      return *this;
    }

    /**
     *  @brief  assign a source %std::map to this %Pairedvector
     *  @param  a %std_::map of identical allocator type.
     *
     *  If necessary the capacity of this %Pairedvector will be modified. All
     *  previous elements are lost and will be overwritten by the source values.
     */
    Pairedvector& operator=(const std::map<Key, T>& source)
    {
      clear();
      clone(source);

      return *this;
    }

    /** @brief clone the source object
     *
     *  If necessary the capacity of this %Pairedvector will be modified. All
     *  previous elements are lost and will be overwritten by the source values.
     *
     *  @param[in] source  Clone the given object.
     *  @param[in] type    type for the clone procedure. If ShapeCopy is chosen,
     *                     only the key but not the values are copied.
     *
     */
    void clone(const Pairedvector& source)
    {
      clear();
      clone_impl(source);
    }

    /** @brief access the raw internally stored data (read-only)
     *
     *  @note Do NOT modify this object, otherwise the paired vector may be
     *  compromised.
     */
    const pairedvector_type& data() const { return m_; }

   protected:
    /// access the internally stored data
    pairedvector_type& data() { return m_; }

    /** @brief internal clone method
     *
     *  @param[in] source  copy the source object into this.
     *
     */
    void clone_impl(const Pairedvector& source)
    {
      const size_t src_capacity = source.capacity();
      if (capacity() < src_capacity) resize(src_capacity);

      entries_ = source.size();

      iterator it = m_.begin();
      for (const auto& src_i : source) *(it++) = src_i;
    }

   private:
    static bool pair_comp(const Key& i, const Key& j) { return i.first < j.first; }

    /// raw data vector
    pairedvector_type m_;

    /// number of entries (without the pre-allocated entries, _entries != _m.size())
    size_t entries_;

  };  // class Pairedvector

}  // namespace Core::Gen

FOUR_C_NAMESPACE_CLOSE

#endif
