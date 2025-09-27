// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COMM_MPI_UTILS_HPP
#define FOUR_C_COMM_MPI_UTILS_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Comm.h>
#include <mpi.h>
#include <Teuchos_Array.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseVector.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Communication
{
  namespace Internal
  {
    template <typename T, typename... Others>
    constexpr bool is_same_as_any_of()
    {
      return (std::is_same_v<T, Others> || ...);
    }

    template <typename T>
    constexpr bool is_mpi_type()
    {
      return is_same_as_any_of<T, int, unsigned, long, unsigned long, long long, unsigned long long,
          float, double, long double, char, unsigned char, short, unsigned short>();
    }
  }  // namespace Internal

  /**
   * Concept to check whether a type is a natively supported MPI type.
   */
  template <typename T>
  concept IsNativeMpiType = Internal::is_mpi_type<T>();

  /**
   * Concept to check whether we know of a way to communicate a type over MPI. This is obviously
   * true for all native MPI types. It is also true for enums, as they can be communicated as
   * their underlying type. For user-defined types, we require that they can be packed and unpacked.
   */
  template <typename T>
  concept IsCommunicatable =
      IsNativeMpiType<T> || std::is_enum_v<T> || (Packable<T> && Unpackable<T>);


  /**
   * Helper function during migration away from Epetra_Comm. Returns the MPI communicator from an
   * Epetra_Comm object.
   */
  MPI_Comm unpack_epetra_comm(const Epetra_Comm& comm);

  /**
   * Helper function during migration away from Epetra_Comm. Returns the MPI_Comm @p comm wrapped in
   * an Epetra_Comm object.
   *
   * @note Epetra_Comm is a virtual interface and it is not trivial to create an instance on the fly
   * and return it here due to the lack of value semantics. Trilinos functions often expect a
   * `const Epetra_Comm&`. Therefore, this function has to maintain a global cache of Epetra_Comm
   * objects for the respective MPI communicator. They are kept alive for the duration of the
   * program. Don't try to create this object yourself, as you will likely run into issues with
   * dangling references.
   */
  const Epetra_Comm& as_epetra_comm(MPI_Comm comm);

  /**
   * Get the MPI rank of the calling process in the communicator @p comm.
   */
  int my_mpi_rank(MPI_Comm comm);

  /**
   * Get the total number of MPI ranks in the communicator @p comm.
   */
  int num_mpi_ranks(MPI_Comm comm);

  /**
   * Wait until all ranks in the communicator @p comm have reached this point.
   *
   * @note You should rarely need to call this function. All MPI communication functions perform any
   * necessary synchronization automatically.
   */
  void barrier(MPI_Comm comm);

  /**
   * Broadcast the value @p value from the MPI rank @p root to all other ranks in the communicator
   * @p comm. The array @p value is assumed to have @p count elements.
   *
   * @note Prefer the overload of this function that takes a reference to a single value.
   */
  template <IsNativeMpiType T>
  void broadcast(T* value, int count, int root, MPI_Comm comm);

  /**
   * Broadcast the value @p value from the MPI rank @p root to all other ranks in the communicator.
   * On rank @p root, the value must be initialized. On all other ranks in @p comm, the value is
   * overwritten with the broadcast value from rank @p root.
   */
  template <IsCommunicatable T>
  void broadcast(T& value, int root, MPI_Comm comm);

  /**
   * Return the sum of all @p partial values across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T>
  T sum_all(const T& partial, MPI_Comm comm);

  /**
   * Return the element-wise sum of all @p partial values across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T, unsigned long size>
  std::array<T, size> sum_all(const std::array<T, size>& partial, MPI_Comm comm);

  /**
   * Return the element-wise sum of all @p partial values across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T>
  std::vector<T> sum_all(const std::vector<T>& partial, MPI_Comm comm);

  /**
   * Return the element-wise sum of all @p partial values across all MPI ranks.
   * The result is distributed to all ranks.
   */
  inline Core::LinAlg::SerialDenseVector sum_all(
      const Core::LinAlg::SerialDenseVector& partial, MPI_Comm comm);

  /**
   * Return the element-wise sum of all @p partial values across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <unsigned int rows, unsigned int columns>
  LinAlg::Matrix<rows, columns> sum_all(
      const LinAlg::Matrix<rows, columns>& partial, MPI_Comm comm);

  /**
   * Return the element-wise sum of all @p partial values across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T>
  Teuchos::Array<T> sum_all(const Teuchos::Array<T>& partial, MPI_Comm comm);

  /**
   * Return the maximum of value of @p partial across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T>
  T max_all(const T& partial, MPI_Comm comm);

  /**
   * Return the element-wise maximum value of the vector @p partial across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T, unsigned long size>
  std::array<T, size> max_all(const std::array<T, size>& partial, MPI_Comm comm);

  /**
   * Return the minimum of value of @p partial across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T>
  T min_all(const T& partial, MPI_Comm comm);

  /**
   * Return the element-wise minimum value of the vector @p partial across all MPI ranks.
   * The result is distributed to all ranks.
   */
  template <IsNativeMpiType T, unsigned long size>
  std::array<T, size> min_all(const std::array<T, size>& partial, MPI_Comm comm);

  /**
   * Gather values @p my_values from all procs to @p all_values. The array @p my_values must have
   * @p count elements. The array @p all_values must have @p count * num_mpi_ranks(comm) elements.
   * The result is distributed to all procs.
   */
  template <IsNativeMpiType T>
  void gather_all(T* my_values, T* all_values, int count, MPI_Comm comm);

  //! Merge map @p map_in (key of type @p T and value of type @p U) from all procs to a merged
  //! map (key of type @p T and value of type @p U). It is distributed to all procs.
  template <IsCommunicatable T, IsCommunicatable U>
  std::map<T, U> all_reduce(const std::map<T, U>& map_in, MPI_Comm comm);

  //! Merge map @p unordered map_in (key of type @p T and value of type @p U) from all procs to a
  //! merged unordered map (key of type @p T and value of type @p U). It is distributed to all
  //! procs.
  template <IsCommunicatable T, IsCommunicatable U>
  std::unordered_map<T, U> all_reduce(const std::unordered_map<T, U>& map_in, MPI_Comm comm);

  //! Merge unordered multimap @p map_in (key of type @p T and value of type @p U) from all procs
  //! to a merged unordered multimap (key of type @p T and value of type @p U). It is distributed
  //! to all procs.
  template <IsCommunicatable T, IsCommunicatable U>
  std::unordered_multimap<T, U> all_reduce(
      const std::unordered_multimap<T, U>& map_in, MPI_Comm comm);

  //! Merge vector of pairs @p pairs_in (items of type @p T and @p U) from all procs to a merged
  //! vector (items of type @p T and @p U). The merged items are in an unspecified order. It is
  //! distributed to all procs.
  template <IsCommunicatable T, IsCommunicatable U>
  std::vector<std::pair<T, U>> all_reduce(
      const std::vector<std::pair<T, U>>& pairs_in, MPI_Comm comm);

  //! Merge @p set_in (items of type @p T) from all procs to a merged set (items of type @p T). It
  //! is distributed to all procs.
  template <IsCommunicatable T>
  std::set<T> all_reduce(const std::set<T>& set_in, MPI_Comm comm);

  //! Merge vector @p vec_in (items of type @p T) from all procs to a merged vector (items of type
  //! @p T). The items of are in an unspecified order. The result is distributed to all procs.
  template <IsCommunicatable T>
  std::vector<T> all_reduce(const std::vector<T>& vec_in, MPI_Comm comm);

  /**
   * Perform an all-reduce operation on a value @p value using the reduction operation @p
   * reduction_op. The result is distributed to all procs.
   */
  template <IsCommunicatable T, typename ReductionOp>
    requires(std::invocable<ReductionOp, const T&, const T&> &&
             std::convertible_to<std::invoke_result_t<ReductionOp, const T&, const T&>, T>)
  T all_reduce(const T& value, ReductionOp reduction_op, MPI_Comm comm);

  /**
   * Gather a value @p value from all procs to a vector of values. The result is distributed to all
   * procs.
   */
  template <IsCommunicatable T>
  std::vector<T> all_gather(const T& value, MPI_Comm comm);

}  // namespace Core::Communication

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
namespace Core::Communication::Internal
{
  template <IsCommunicatable T, IsCommunicatable U, typename M>
  void all_reduce_map_like_to_vectors(
      const M& map_in, std::vector<T>& vec_out1, std::vector<U>& vec_out2, MPI_Comm comm)
  {
    // split map or std::vector<std::pair> into two vectors
    std::vector<T> my_gid_vec1;
    std::vector<U> my_gid_vec2;
    for (const auto& pair : map_in)
    {
      my_gid_vec1.emplace_back(pair.first);
      my_gid_vec2.emplace_back(pair.second);
    }
    std::vector<T> vec1 = all_reduce(my_gid_vec1, comm);
    std::vector<U> vec2 = all_reduce(my_gid_vec2, comm);

    FOUR_C_ASSERT(vec1.size() == vec2.size(), "Vectors must have the same length.");

    // reconstruct map-like object
    for (unsigned i = 0; i < vec1.size(); ++i)
    {
      vec_out1.emplace_back(vec1[i]);
      vec_out2.emplace_back(vec2[i]);
    }
  }

  //! MPI datatype for type T
  template <IsNativeMpiType T>
  constexpr MPI_Datatype to_mpi_type()
  {
    if constexpr (std::is_same_v<T, int>)
      return MPI_INT;
    else if constexpr (std::is_same_v<T, unsigned>)
      return MPI_UNSIGNED;
    else if constexpr (std::is_same_v<T, long>)
      return MPI_LONG;
    else if constexpr (std::is_same_v<T, unsigned long>)
      return MPI_UNSIGNED_LONG;
    else if constexpr (std::is_same_v<T, long long>)
      return MPI_LONG_LONG;
    else if constexpr (std::is_same_v<T, unsigned long long>)
      return MPI_UNSIGNED_LONG_LONG;
    else if constexpr (std::is_same_v<T, float>)
      return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, double>)
      return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, long double>)
      return MPI_LONG_DOUBLE;
    else if constexpr (std::is_same_v<T, char>)
      return MPI_CHAR;
    else if constexpr (std::is_same_v<T, unsigned char>)
      return MPI_UNSIGNED_CHAR;
    else if constexpr (std::is_same_v<T, short>)
      return MPI_SHORT;
    else if constexpr (std::is_same_v<T, unsigned short>)
      return MPI_UNSIGNED_SHORT;

    return MPI_DATATYPE_NULL;
  }
}  // namespace Core::Communication::Internal

template <Core::Communication::IsNativeMpiType T>
void Core::Communication::broadcast(T* value, int count, int root, MPI_Comm comm)
{
  MPI_Bcast(value, count, Internal::to_mpi_type<T>(), root, comm);
}

template <Core::Communication::IsCommunicatable T>
void Core::Communication::broadcast(T& value, int root, MPI_Comm comm)
{
  // If T is an enum, we can broadcast a value of the underlying type.
  if constexpr (std::is_enum_v<T>)
  {
    std::underlying_type_t<T> underlying = static_cast<std::underlying_type_t<T>>(value);
    broadcast(&underlying, 1, root, comm);
    value = static_cast<T>(underlying);
  }
  // Shortcut when T is natively supported by MPI.
  else if constexpr (Internal::is_mpi_type<T>())
  {
    broadcast(&value, 1, root, comm);
  }
  // Otherwise, we need to pack and unpack the data.
  else
  {
    if (Core::Communication::my_mpi_rank(comm) == root)
    {
      PackBuffer buffer;
      Core::Communication::add_to_pack(buffer, value);

      // Send the size.
      auto size = buffer().size();
      Core::Communication::broadcast(&size, 1, root, comm);

      // Send the data.
      Core::Communication::broadcast(buffer().data(), static_cast<int>(size), root, comm);
    }
    else
    {
      // Receive the size.
      std::size_t size;
      Core::Communication::broadcast(&size, 1, root, comm);

      // Receive the data.
      std::vector<char> buffer(size);
      Core::Communication::broadcast(buffer.data(), static_cast<int>(size), root, comm);

      UnpackBuffer unpack_buffer(buffer);
      Core::Communication::extract_from_pack(unpack_buffer, value);
    }
  }
}

template <Core::Communication::IsNativeMpiType T>
T Core::Communication::sum_all(const T& partial, MPI_Comm comm)
{
  T global{};
  MPI_Allreduce(&partial, &global, 1, Internal::to_mpi_type<T>(), MPI_SUM, comm);
  return global;
}

template <Core::Communication::IsNativeMpiType T, long unsigned size>
std::array<T, size> Core::Communication::sum_all(const std::array<T, size>& partial, MPI_Comm comm)
{
  std::array<T, size> global{};
  MPI_Allreduce(
      partial.data(), global.data(), partial.size(), Internal::to_mpi_type<T>(), MPI_SUM, comm);
  return global;
}

template <Core::Communication::IsNativeMpiType T>
std::vector<T> Core::Communication::sum_all(const std::vector<T>& partial, MPI_Comm comm)
{
  std::vector<T> global(partial.size());
  MPI_Allreduce(
      partial.data(), global.data(), partial.size(), Internal::to_mpi_type<T>(), MPI_SUM, comm);
  return global;
}

Core::LinAlg::SerialDenseVector Core::Communication::sum_all(
    const Core::LinAlg::SerialDenseVector& partial, MPI_Comm comm)
{
  Core::LinAlg::SerialDenseVector global(partial.length());
  MPI_Allreduce(partial.values(), global.values(), partial.length(),
      Internal::to_mpi_type<double>(), MPI_SUM, comm);
  return global;
}

template <unsigned int rows, unsigned int columns>
Core::LinAlg::Matrix<rows, columns> Core::Communication::sum_all(
    const Core::LinAlg::Matrix<rows, columns>& partial, MPI_Comm comm)
{
  Core::LinAlg::Matrix<rows, columns> global;
  // We can treat the matrix as a contiguous array with size numRows()*numCols().
  MPI_Allreduce(partial.values(), global.values(), rows * columns, Internal::to_mpi_type<double>(),
      MPI_SUM, comm);
  return global;
}

template <Core::Communication::IsNativeMpiType T>
Teuchos::Array<T> Core::Communication::sum_all(const Teuchos::Array<T>& partial, MPI_Comm comm)
{
  Teuchos::Array<T> global(partial.length());
  MPI_Allreduce(
      partial.data(), global.data(), partial.length(), Internal::to_mpi_type<T>(), MPI_SUM, comm);
  return global;
}


template <Core::Communication::IsNativeMpiType T>
T Core::Communication::max_all(const T& partial, MPI_Comm comm)
{
  T global{};
  MPI_Allreduce(&partial, &global, 1, Internal::to_mpi_type<T>(), MPI_MAX, comm);
  return global;
}

template <Core::Communication::IsNativeMpiType T, unsigned long size>
std::array<T, size> Core::Communication::max_all(const std::array<T, size>& partial, MPI_Comm comm)
{
  std::array<T, size> global{};
  MPI_Allreduce(partial.data(), global.data(), size, Internal::to_mpi_type<T>(), MPI_MAX, comm);
  return global;
}

template <Core::Communication::IsNativeMpiType T>
T Core::Communication::min_all(const T& partial, MPI_Comm comm)
{
  T global{};
  MPI_Allreduce(&partial, &global, 1, Internal::to_mpi_type<T>(), MPI_MIN, comm);
  return global;
}

template <Core::Communication::IsNativeMpiType T, unsigned long size>
std::array<T, size> Core::Communication::min_all(const std::array<T, size>& partial, MPI_Comm comm)
{
  std::array<T, size> global{};
  MPI_Allreduce(partial.data(), global.data(), size, Internal::to_mpi_type<T>(), MPI_MIN, comm);
  return global;
}

template <Core::Communication::IsNativeMpiType T>
void Core::Communication::gather_all(T* my_values, T* all_values, int count, MPI_Comm comm)
{
  MPI_Allgather(my_values, count, Internal::to_mpi_type<T>(), all_values, count,
      Internal::to_mpi_type<T>(), comm);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <Core::Communication::IsCommunicatable T, Core::Communication::IsCommunicatable U>
std::map<T, U> Core::Communication::all_reduce(const std::map<T, U>& map_in, MPI_Comm comm)
{
  std::vector<T> vec1;
  std::vector<U> vec2;
  Internal::all_reduce_map_like_to_vectors<T, U>(map_in, vec1, vec2, comm);
  std::map<T, U> map_out;
  for (unsigned i = 0; i < vec1.size(); ++i) map_out.insert(std::make_pair(vec1[i], vec2[i]));
  return map_out;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <Core::Communication::IsCommunicatable T, Core::Communication::IsCommunicatable U>
std::unordered_map<T, U> Core::Communication::all_reduce(
    const std::unordered_map<T, U>& map_in, MPI_Comm comm)
{
  std::vector<T> vec1;
  std::vector<U> vec2;
  Internal::all_reduce_map_like_to_vectors<T, U>(map_in, vec1, vec2, comm);
  std::unordered_map<T, U> map_out;
  for (unsigned i = 0; i < vec1.size(); ++i) map_out.insert(std::make_pair(vec1[i], vec2[i]));
  return map_out;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <Core::Communication::IsCommunicatable T, Core::Communication::IsCommunicatable U>
std::unordered_multimap<T, U> Core::Communication::all_reduce(
    const std::unordered_multimap<T, U>& map_in, MPI_Comm comm)
{
  std::vector<T> vec1;
  std::vector<U> vec2;
  Internal::all_reduce_map_like_to_vectors<T, U>(map_in, vec1, vec2, comm);
  std::unordered_multimap<T, U> map_out;
  for (unsigned i = 0; i < vec1.size(); ++i) map_out.insert(std::make_pair(vec1[i], vec2[i]));
  return map_out;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <Core::Communication::IsCommunicatable T, Core::Communication::IsCommunicatable U>
std::vector<std::pair<T, U>> Core::Communication::all_reduce(
    const std::vector<std::pair<T, U>>& pairs_in, MPI_Comm comm)
{
  std::vector<T> vec1;
  std::vector<U> vec2;
  Internal::all_reduce_map_like_to_vectors<T, U>(pairs_in, vec1, vec2, comm);
  std::vector<std::pair<T, U>> pairs_out;
  for (unsigned i = 0; i < vec1.size(); ++i)
    pairs_out.emplace_back(std::make_pair(vec1[i], vec2[i]));
  return pairs_out;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <Core::Communication::IsCommunicatable T>
std::set<T> Core::Communication::all_reduce(const std::set<T>& set_in, MPI_Comm comm)
{
  std::vector<T> vec_in, vec_out;
  for (const auto& val : set_in) vec_in.emplace_back(val);
  vec_out = all_reduce(vec_in, comm);
  std::set<T> set_out;
  for (const auto& val : vec_out) set_out.insert(val);
  return set_out;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
template <Core::Communication::IsCommunicatable T>
std::vector<T> Core::Communication::all_reduce(const std::vector<T>& vec_in, MPI_Comm comm)
{
  std::vector<std::vector<T>> gathered = all_gather(vec_in, comm);

  // Merge the data from all processors into a single vector
  std::vector<T> all_vectors;
  for (auto& received : gathered)
  {
    all_vectors.insert(all_vectors.end(), received.begin(), received.end());
  }

  return all_vectors;
}


template <Core::Communication::IsCommunicatable T, typename ReductionOp>
  requires(std::invocable<ReductionOp, const T&, const T&> &&
           std::convertible_to<std::invoke_result_t<ReductionOp, const T&, const T&>, T>)
T Core::Communication::all_reduce(const T& value, ReductionOp reduction_op, MPI_Comm comm)
{
  std::vector<T> all_values = all_gather(value, comm);
  return std::accumulate(all_values.cbegin() + 1, all_values.cend(), all_values[0], reduction_op);
}


template <Core::Communication::IsCommunicatable T>
std::vector<T> Core::Communication::all_gather(const T& value, MPI_Comm comm)
{
  const int n_procs = num_mpi_ranks(comm);

  // Use PackBuffer to serialize the data into a vector<char>
  PackBuffer buffer;
  add_to_pack(buffer, value);

  int local_size = buffer().size();
  std::vector<int> size_all_data(n_procs, 0);
  [[maybe_unused]] int err =
      MPI_Allgather(&local_size, 1, MPI_INT, size_all_data.data(), 1, MPI_INT, comm);
  FOUR_C_ASSERT(err == MPI_SUCCESS, "MPI_Allgather failed.");

  // Compute offset of the data from the i-th rank
  std::vector<int> accumulated_offset(n_procs);
  std::partial_sum(size_all_data.begin(), size_all_data.end() - 1, accumulated_offset.begin() + 1);

  std::vector<char> combined_buffer(accumulated_offset.back() + size_all_data.back());

  err = MPI_Allgatherv(buffer().data(), local_size, MPI_CHAR,
      /*full received buffer*/ combined_buffer.data(),
      /*sizes of processor contributions*/ size_all_data.data(),
      /*displacement of processor contributions in full buffer*/ accumulated_offset.data(),
      MPI_CHAR, comm);
  FOUR_C_ASSERT(err == MPI_SUCCESS, "MPI_Allgatherv failed.");

  // Merge the data from all processors into a single vector
  std::vector<T> all_data(n_procs);
  for (int i = 0; i < n_procs; ++i)
  {
    // Since UnpackBuffer cannot view data, we need to make a copy of the data.
    std::vector<char> local_buffer(combined_buffer.begin() + accumulated_offset[i],
        combined_buffer.begin() + accumulated_offset[i] + size_all_data[i]);

    UnpackBuffer unpack_buffer(local_buffer);
    if constexpr (std::is_same_v<T, bool>)
    {
      bool tmp;
      extract_from_pack(unpack_buffer, tmp);
      all_data[i] = tmp;
    }
    else
    {
      extract_from_pack(unpack_buffer, all_data[i]);
    }
  }

  return all_data;
}

FOUR_C_NAMESPACE_CLOSE

#endif