// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_sparse_pod.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_exceptions.hpp"

#include <EpetraExt_MultiVectorIn.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*! \brief Read POD basis vectors from a Matrix Market format (ASCII)
   *
   * ### Matrix Market format (ASCII)
   * The file can also be created from Python using `scipy.io.mmwrite`, for example:
   * @code{.py}
   * from scipy.io import mmwrite
   * mmwrite("pod_projection.mtx", projmatrix, precision=10)
   * @endcode
   *
   * @param filename  Path to the input file containing the POD basis.
   * @param full_mode_dof_row_map Map describing the full model dof layout.
   * @return A MultiVector (or equivalent matrix) containing the POD basis vectors.
   */
  Core::LinAlg::MultiVector<double> read_pod_basis_vectors_from_file(
      const std::filesystem::path& pod_matrix_file_name,
      const Core::LinAlg::Map& full_mode_dof_row_map)
  {
    Epetra_MultiVector* multi_vector = nullptr;
    EpetraExt::MatrixMarketFileToMultiVector(
        pod_matrix_file_name.c_str(), full_mode_dof_row_map.get_epetra_block_map(), multi_vector);
    std::unique_ptr<Epetra_MultiVector> auto_deleter(multi_vector);

    FOUR_C_ASSERT_ALWAYS(auto_deleter != nullptr, "Failed to read POD basis vectors from file {}",
        pod_matrix_file_name.relative_path().string());

    return Core::LinAlg::MultiVector<double>{*multi_vector};
  }

  bool is_pod_basis_orthogonal(const Core::LinAlg::MultiVector<double>& M, const double tol = 1e-7)
  {
    const int n = M.NumVectors();

    // calculate V^T * V (should be an nxn identity matrix)
    Core::LinAlg::Map map = Core::LinAlg::Map(n, n, 0, M.get_map().get_comm());
    Core::LinAlg::MultiVector<double> identity = Core::LinAlg::MultiVector<double>(map, n, true);
    identity.Multiply('T', 'N', 1.0, M, M, 0.0);

    // subtract one from diagonal
    for (int i = 0; i < n; ++i) identity.SumIntoGlobalValue(i, i, -1.0);

    // inf norm of columns
    std::vector<double> norms(n, 0.0);
    identity.NormInf(norms.data());

    for (int i = 0; i < n; ++i)
      if (norms[i] > tol) return false;

    return true;
  }

  Core::LinAlg::MultiVector<double> make_projection_matrix(
      const std::filesystem::path& pod_matrix_file_name,
      const Core::LinAlg::Map& full_model_dof_row_map)
  {
    Core::LinAlg::MultiVector<double> reduced_basis =
        read_pod_basis_vectors_from_file(pod_matrix_file_name, full_model_dof_row_map);

    Core::LinAlg::MultiVector<double> projection_matrix(
        full_model_dof_row_map, reduced_basis.NumVectors(), true);

    Core::LinAlg::Import dofrowimporter(full_model_dof_row_map, reduced_basis.get_map());
    int err = projection_matrix.Import(reduced_basis, dofrowimporter, Insert, nullptr);
    FOUR_C_ASSERT_ALWAYS(!err, "POD projection matrix could not be mapped onto the dof map");

    // check row dimension
    FOUR_C_ASSERT_ALWAYS(
        projection_matrix.GlobalLength() == full_model_dof_row_map.num_global_elements(),
        "Projection matrix does not match discretization.");

    // check orthogonality
    FOUR_C_ASSERT_ALWAYS(
        is_pod_basis_orthogonal(projection_matrix), "Projection matrix is not orthogonal.");

    return projection_matrix;
  }
}  // namespace

Core::LinAlg::ProperOrthogonalDecomposition::ProperOrthogonalDecomposition(
    std::shared_ptr<const Core::LinAlg::Map> full_model_dof_row_map,
    const std::filesystem::path& pod_matrix_file_name)
    : full_model_dof_row_map_(std::move(full_model_dof_row_map)),
      projmatrix_(make_projection_matrix(pod_matrix_file_name, *full_model_dof_row_map_)),
      structmapr_(projmatrix_.NumVectors(), 0, full_model_dof_row_map_->get_comm()),
      redstructmapr_(projmatrix_.NumVectors(), projmatrix_.NumVectors(), 0,
          full_model_dof_row_map_->get_comm()),
      structrimpo_(structmapr_, redstructmapr_),
      structrinvimpo_(redstructmapr_, structmapr_)
{
}

Core::LinAlg::SparseMatrix Core::LinAlg::ProperOrthogonalDecomposition::reduce_diagonal(
    const Core::LinAlg::SparseMatrix& M)
{
  // right multiply M * V
  Core::LinAlg::MultiVector<double> M_tmp(M.row_map(), projmatrix_.NumVectors(), true);
  int err = M.multiply(false, projmatrix_, M_tmp);
  if (err) FOUR_C_THROW("Multiplication M * V failed.");

  // left multiply V^T * (M * V)
  Core::LinAlg::MultiVector<double> M_red_mvec(structmapr_, M_tmp.NumVectors(), true);
  Core::LinAlg::multiply_multi_vectors(
      projmatrix_, 'T', M_tmp, 'N', redstructmapr_, structrimpo_, M_red_mvec);

  return Core::LinAlg::make_sparse_matrix(M_red_mvec, structmapr_, std::nullopt);
}

Core::LinAlg::SparseMatrix Core::LinAlg::ProperOrthogonalDecomposition::reduce_off_diagonal(
    const Core::LinAlg::SparseMatrix& M)
{
  // right multiply M * V
  Core::LinAlg::MultiVector<double> M_tmp(M.domain_map(), projmatrix_.NumVectors(), true);
  int err = M.multiply(true, projmatrix_, M_tmp);
  FOUR_C_ASSERT_ALWAYS(!err, "Multiplication V^T * M failed.");

  return Core::LinAlg::make_sparse_matrix(M_tmp, M.domain_map(), structmapr_);
}

Core::LinAlg::MultiVector<double> Core::LinAlg::ProperOrthogonalDecomposition::reduce(
    const Core::LinAlg::MultiVector<double>& v)
{
  Core::LinAlg::MultiVector<double> v_red(structmapr_, 1, true);
  Core::LinAlg::multiply_multi_vectors(
      projmatrix_, 'T', v, 'N', redstructmapr_, structrimpo_, v_red);

  return v_red;
}

Core::LinAlg::Vector<double> Core::LinAlg::ProperOrthogonalDecomposition::reduce(
    const Core::LinAlg::Vector<double>& v)
{
  Core::LinAlg::Vector<double> v_tmp(redstructmapr_);
  int err = v_tmp.multiply('T', 'N', 1.0, projmatrix_, v, 0.0);
  if (err) FOUR_C_THROW("Multiplication V^T * v failed.");

  Core::LinAlg::Vector<double> v_red(structmapr_);
  v_red.import(v_tmp, structrimpo_, Insert, nullptr);

  return v_red;
}

Core::LinAlg::Vector<double> Core::LinAlg::ProperOrthogonalDecomposition::extend(
    const Core::LinAlg::Vector<double>& v_red)
{
  Core::LinAlg::Vector<double> v_tmp(redstructmapr_);
  v_tmp.import(v_red, structrinvimpo_, Insert, nullptr);

  Core::LinAlg::Vector<double> v(*full_model_dof_row_map_);
  int err = v.multiply('N', 'N', 1.0, projmatrix_, v_tmp, 0.0);
  FOUR_C_ASSERT_ALWAYS(!err, "Multiplication V * v_red failed.");

  return v;
}

FOUR_C_NAMESPACE_CLOSE
