// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_PRINT_HPP
#define FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_PRINT_HPP

#include "4C_config.hpp"

#include "4C_linalg_graph.hpp"
#include "4C_linalg_vector.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>


FOUR_C_NAMESPACE_OPEN

// Forward declarations
namespace Core::LinAlg
{
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg

namespace Core::LinAlg
{
  //! Print content of @p sparsematrix in Matlab format to file @p filename. Create new file or
  //! overwrite existing one if @p newfile is true
  void print_matrix_in_matlab_format(
      const std::string& filename, const Epetra_CrsMatrix& sparsematrix, const bool newfile = true);

  //! Print content of @p blockmatrix in Matlab format to file @p filename
  void print_block_matrix_in_matlab_format(
      const std::string& filename, const BlockSparseMatrixBase& blockmatrix);

  //! Print content of @p vector in Matlab format to file @p filename. Create new file or overwrite
  //! existing one if @p newfile is true
  void print_vector_in_matlab_format(const std::string& filename,
      const Core::LinAlg::Vector<double>& vector, const bool newfile = true);

  //! Print content of @p map in Matlab format to file @p filename. Create new file or overwrite
  //! existing one if @p newfile is true
  void print_map_in_matlab_format(
      const std::string& filename, const Epetra_Map& map, const bool newfile = true);

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
