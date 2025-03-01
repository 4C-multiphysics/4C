// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_EXTRACT_VALUES_HPP
#define FOUR_C_FEM_GENERAL_EXTRACT_VALUES_HPP

#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <iterator>
#include <ranges>


FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  namespace Internal
  {
    template <typename T>
    inline auto value_extractor(const Core::LinAlg::Vector<T>& global)
    {
      return [&](int global_id)
      {
        const int local_id = global.Map().LID(global_id);
        FOUR_C_ASSERT_ALWAYS(local_id >= 0,
            "Proc %d: Cannot find gid=%d in Core::LinAlg::Vector<double>",
            Core::Communication::my_mpi_rank(global.Comm()), global_id);
        return global[local_id];
      };
    }
  }  // namespace Internal

  /*!
   * @brief Extract a subset of local values from Core::LinAlg::Vector<T>
   *
   * @tparam num_entries Number of locally owned entries
   * @tparam T datatype
   * @param global (in) : Global values
   * @param lm (in) : Global ids of the values to extract
   * @return std::array<T, num_entries> Extracted values from the global vector
   */
  template <unsigned num_entries, typename T, std::ranges::range Range>
  std::array<T, num_entries> extract_values_as_array(
      const Core::LinAlg::Vector<T>& global, Range global_ids)
  {
    std::array<T, num_entries> local;
    std::transform(
        global_ids.begin(), global_ids.end(), local.begin(), Internal::value_extractor(global));

    return local;
  }

  /*!
   * @brief Extract a subset of local values from Core::LinAlg::Vector<T>
   *
   * @tparam T datatype
   * @param global (in) : Global values
   * @param lm (in) : Global ids of the values to extract
   * @param expected_size (in): (Expected) size of the local vector
   * @return std::vector<T> Extracted values from the global vector
   */
  template <typename T, std::ranges::range Range>
  std::vector<T> extract_values(
      const Core::LinAlg::Vector<T>& global, Range global_ids, std::size_t expected_size)
  {
    std::vector<T> local;
    local.reserve(expected_size);
    std::ranges::copy(global_ids | std::views::transform(Internal::value_extractor(global)),
        std::back_inserter(local));

    return local;
  }

  /*!
   * @brief Extract a subset of local values from Core::LinAlg::Vector<T>
   *
   * @tparam T datatype
   * @param global (in) : Global values
   * @param lm (in) : Global ids of the values to extract
   * @return std::vector<T> Extracted values from the global vector
   */
  template <typename T, std::ranges::sized_range Range>
  std::vector<T> extract_values(const Core::LinAlg::Vector<T>& global, const Range& global_ids)
  {
    return extract_values(global, global_ids, std::size(global_ids));
  }

  template <typename T, std::ranges::sized_range Range>
  std::vector<T> extract_values(const Core::LinAlg::MultiVector<T>& global, const Range& global_ids)
  {
    const int numcol = global.NumVectors();
    const size_t ldim = std::size(global_ids);

    std::vector<T> local(ldim * numcol);
    for (size_t i = 0; i < ldim; ++i)
    {
      const int lid = global.Map().LID(global_ids[i]);
      FOUR_C_ASSERT_ALWAYS(lid >= 0,
          "Proc %d: Cannot find gid=%d in Core::LinAlg::MultiVector<double>",
          Core::Communication::my_mpi_rank(global.Comm()), global_ids[i]);

      // loop over multi vector columns (numcol=1 for Core::LinAlg::Vector<double>)
      for (int col = 0; col < numcol; col++)
      {
        local[col + (numcol * i)] = global(col)[lid];
      }
    }

    return local;
  }

  void extract_my_values(const Core::LinAlg::Vector<double>& global,
      Core::LinAlg::SerialDenseVector& local, const std::vector<int>& lm);

  template <class Matrix>
  void extract_my_values(const Core::LinAlg::Vector<double>& global, std::vector<Matrix>& local,
      const std::vector<int>& lm)
  {
    // safety check
    if (local[0].n() != 1 or local.size() * (unsigned)local[0].m() != lm.size())
      FOUR_C_THROW("Received matrix vector of wrong size!");

    // loop over all nodes of current element
    for (unsigned inode = 0; inode < local[0].m(); ++inode)
    {
      // loop over all dofs of current node
      for (unsigned idof = 0; idof < local.size(); ++idof)
      {
        // extract local ID of current dof
        const int lid = global.Map().LID(lm[inode * local.size() + idof]);

        // safety check
        if (lid < 0)
          FOUR_C_THROW("Proc %d: Cannot find gid=%d in Core::LinAlg::Vector<double>",
              Core::Communication::my_mpi_rank(global.Comm()), lm[inode * local.size() + idof]);

        // store current dof in local matrix vector consisting of ndof matrices of size nnode x 1,
        // where nnode denotes the number of element nodes and ndof denotes the number of degrees
        // of freedom per element node.
        local[idof](inode, 0) = global[lid];
      }
    }
  }

  template <class Matrix>
  void extract_my_values(
      const Core::LinAlg::Vector<double>& global, Matrix& local, const std::vector<int>& lm)
  {
    // safety check
    if ((unsigned)(local.num_rows() * local.num_cols()) != lm.size())
      FOUR_C_THROW("Received matrix of wrong size!");

    // loop over all columns of cal matrix
    for (unsigned icol = 0; icol < local.num_cols(); ++icol)
    {
      // loop over all rows of local matrix
      for (unsigned irow = 0; irow < local.num_rows(); ++irow)
      {
        // extract local ID of current dof
        const unsigned index = icol * local.num_rows() + irow;
        const int lid = global.Map().LID(lm[index]);

        // safety check
        if (lid < 0)
          FOUR_C_THROW("Proc %d: Cannot find gid=%d in Core::LinAlg::Vector<double>",
              Core::Communication::my_mpi_rank(global.Comm()), lm[index]);

        // store current dof in local matrix, which is filled column-wise with the dofs listed in
        // the lm vector
        local(irow, icol) = global[lid];
      }
    }
  }

  /// Locally extract a subset of values from a (column)-nodemap-based
  /// Core::LinAlg::MultiVector<double>
  void extract_my_node_based_values(
      const Core::Elements::Element* ele,              ///< pointer to current element
      std::vector<double>& local,                      ///< local vector on element-level
      const Core::LinAlg::MultiVector<double>& global  ///< global (multi) vector
  );


  /// Locally extract a subset of values from a (column)-nodemap-based
  /// Core::LinAlg::MultiVector<double>
  void extract_my_node_based_values(
      const Core::Elements::Element* ele,         ///< pointer to current element
      Core::LinAlg::SerialDenseVector& local,     ///< local vector on element-level
      Core::LinAlg::MultiVector<double>& global,  ///< global vector
      const int nsd                               ///< number of space dimensions
  );


  /// Locally extract a subset of values from a (column)-nodemap-based
  /// Core::LinAlg::MultiVector<double> and fill a local matrix that has implemented the (.,.)
  /// operator
  template <class M>
  void extract_my_node_based_values(
      const Core::Elements::Element* ele,         ///< pointer to current element
      M& localmatrix,                             ///< local matrix on element-level
      Core::LinAlg::MultiVector<double>& global,  ///< global vector
      const int nsd                               ///< number of space dimensions
  )
  {
    if (nsd > global.NumVectors())
      FOUR_C_THROW("Requested %d of %d available columns", nsd, global.NumVectors());
    const int iel = ele->num_node();  // number of nodes
    if (((int)localmatrix.num_cols()) != iel)
      FOUR_C_THROW("local matrix has wrong number of columns");
    if (((int)localmatrix.num_rows()) != nsd) FOUR_C_THROW("local matrix has wrong number of rows");

    for (int i = 0; i < nsd; i++)
    {
      // loop over the element nodes
      for (int j = 0; j < iel; j++)
      {
        const int nodegid = (ele->nodes()[j])->id();
        const int lid = global.Map().LID(nodegid);
        if (lid < 0)
          FOUR_C_THROW("Proc %d: Cannot find gid=%d in Core::LinAlg::Vector<double>",
              Core::Communication::my_mpi_rank((global).Comm()), nodegid);
        localmatrix(i, j) = global(i)[lid];
      }
    }
  }

  /*!
  \brief extract local values from global node-based (multi) vector

  This function returns a column vector!

 */
  template <class M>
  void extract_my_node_based_values(
      const Core::Elements::Element* ele, M& local, const Core::LinAlg::MultiVector<double>& global)
  {
    const int numnode = ele->num_node();
    const int numcol = global.NumVectors();
    if (((int)local.n()) != 1) FOUR_C_THROW("local matrix must have one column");
    if (((int)local.m()) != numnode * numcol) FOUR_C_THROW("local matrix has wrong number of rows");

    // loop over element nodes
    for (int i = 0; i < numnode; ++i)
    {
      const int nodegid = (ele->nodes()[i])->id();
      const int lid = global.Map().LID(nodegid);
      if (lid < 0)
        FOUR_C_THROW("Proc %d: Cannot find gid=%d in Core::LinAlg::Vector<double>",
            Core::Communication::my_mpi_rank(global.Comm()), nodegid);

      // loop over multi vector columns (numcol=1 for Core::LinAlg::Vector<double>)
      for (int col = 0; col < numcol; col++)
      {
        local((col + (numcol * i)), 0) = global(col)[lid];
      }
    }
  }
}  // namespace Core::FE


FOUR_C_NAMESPACE_CLOSE

#endif
