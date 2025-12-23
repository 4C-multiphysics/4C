// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CARDIOVASCULAR0D_MOR_POD_HPP
#define FOUR_C_CARDIOVASCULAR0D_MOR_POD_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <filesystem>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg
namespace Cardiovascular0D
{
  class ProperOrthogonalDecomposition
  {
   public:
    /*!
        \brief Constructor
     */
    ProperOrthogonalDecomposition(std::shared_ptr<const Core::LinAlg::Map> full_model_dof_row_map_,
        const std::optional<std::filesystem::path>& pod_matrix_file_name,
        const std::string& absolute_path_to_input_file);

    //! M_red = V^T * M * V
    std::shared_ptr<Core::LinAlg::SparseMatrix> reduce_diagonal(Core::LinAlg::SparseMatrix& M);

    //! M_red = V^T * M
    std::shared_ptr<Core::LinAlg::SparseMatrix> reduce_off_diagonal(Core::LinAlg::SparseMatrix& M);

    //! v_red = V^T * v
    std::shared_ptr<Core::LinAlg::MultiVector<double>> reduce_rhs(
        Core::LinAlg::MultiVector<double>& v);

    //! v_red = V^T * v
    std::shared_ptr<Core::LinAlg::Vector<double>> reduce_residual(Core::LinAlg::Vector<double>& v);

    //! v = V * v_red
    std::shared_ptr<Core::LinAlg::Vector<double>> extend_solution(Core::LinAlg::Vector<double>& v);

    bool have_mor() { return havemor_; };

    int get_red_dim() { return projmatrix_->num_vectors(); };

   private:
    /*! \brief Read POD basis vectors from file
     *
     * Read matrix from specified binary file
     * The binary file has to be formatted like this:
     * Number of Rows: int
     * Number of Columns: int
     * Values (row-wise): float
     */
    void read_pod_basis_vectors_from_file(const std::filesystem::path& absolute_path_to_pod_file,
        std::shared_ptr<Core::LinAlg::MultiVector<double>>& projmatrix);

    //! Check orthogonality of POD basis vectors with M^T * M - I == 0
    bool is_pod_basis_orthogonal(const Core::LinAlg::MultiVector<double>& M);

    /// DOF row map of the full model, i.e. map of POD basis vectors
    std::shared_ptr<const Core::LinAlg::Map> full_model_dof_row_map_;

    //! Flag to indicate usage of model order reduction
    bool havemor_ = false;

    //! Projection matrix for POD
    std::shared_ptr<Core::LinAlg::MultiVector<double>> projmatrix_;

    //! Unique map of structure dofs after POD-MOR
    std::shared_ptr<Core::LinAlg::Map> structmapr_;

    //! Full redundant map of structure dofs after POD-MOR
    std::shared_ptr<Core::LinAlg::Map> redstructmapr_;

    //! Importer for fully redundant map of structure dofs after POD-MOR into distributed one
    std::shared_ptr<Core::LinAlg::Import> structrimpo_;

    //! Importer for distributed map of structure dofs after POD-MOR into fully redundant one
    std::shared_ptr<Core::LinAlg::Import> structrinvimpo_;

  };  // class
}  // namespace Cardiovascular0D
FOUR_C_NAMESPACE_CLOSE

#endif
