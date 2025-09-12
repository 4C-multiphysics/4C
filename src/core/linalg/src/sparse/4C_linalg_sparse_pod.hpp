// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_SPARSE_POD_HPP
#define FOUR_C_LINALG_SPARSE_POD_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"

#include <filesystem>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::LinAlg
{
  /*!
   * \brief Implements Proper Orthogonal Decomposition (POD) for model order reduction (MOR) for
   * coupled block systems.
   *
   * The POD basis vectors are read from a binary file on construction from a Matrix Market file.
   *
   * This class provides methods to perform POD-based model order reduction on linear systems.
   * It enables projection of matrices and vectors onto a reduced basis, as well as extension
   * of reduced solutions back to the full space.
   */
  class ProperOrthogonalDecomposition
  {
   public:
    /*!
        \brief Constructs the POD object by reading the projection matrix from a Matrix market file.
     */
    ProperOrthogonalDecomposition(std::shared_ptr<const Core::LinAlg::Map> full_model_dof_row_map_,
        const std::filesystem::path& pod_matrix_file_name);

    //! M_red = V^T * M * V
    Core::LinAlg::SparseMatrix reduce_diagonal(const Core::LinAlg::SparseMatrix& M);

    //! M_red = V^T * M
    Core::LinAlg::SparseMatrix reduce_off_diagonal(const Core::LinAlg::SparseMatrix& M);

    //! v_red = V^T * v
    Core::LinAlg::MultiVector<double> reduce(const Core::LinAlg::MultiVector<double>& v);

    //! v_red = V^T * v
    Core::LinAlg::Vector<double> reduce(const Core::LinAlg::Vector<double>& v);

    //! v = V * v_red
    Core::LinAlg::Vector<double> extend(const Core::LinAlg::Vector<double>& v);

    int get_red_dim() { return projmatrix_.NumVectors(); };

   private:
    /// DOF row map of the full model, i.e. map of POD basis vectors
    std::shared_ptr<const Core::LinAlg::Map> full_model_dof_row_map_;

    //! Projection matrix for POD
    Core::LinAlg::MultiVector<double> projmatrix_;

    //! Unique map of structure dofs after POD-MOR
    Core::LinAlg::Map structmapr_;

    //! Full redundant map of structure dofs after POD-MOR
    Core::LinAlg::Map redstructmapr_;

    //! Importer for fully redundant map of structure dofs after POD-MOR into distributed one
    Core::LinAlg::Import structrimpo_;

    //! Importer for distributed map of structure dofs after POD-MOR into fully redundant one
    Core::LinAlg::Import structrinvimpo_;
  };
}  // namespace Core::LinAlg
FOUR_C_NAMESPACE_CLOSE

#endif
