// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_THYRA_UTILS_HPP
#define FOUR_C_LINEAR_SOLVER_THYRA_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <Thyra_DefaultBlockedLinearOp_decl.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinearSolver::Utils
{
    // Create Thyra vector space from Map
  Teuchos::RCP<const Thyra::VectorSpaceBase<double>> create_thyra_map(const LinAlg::Map& map);


    // Create Thyra linear operator from SparseOperator, SparseMatrix or BlockSparseMatrixBase
    Teuchos::RCP<const Thyra::LinearOpBase<double>> create_thyra_linear_op(
        LinAlg::SparseOperator& matrix, Core::LinAlg::DataAccess access);

  Teuchos::RCP<const Thyra::LinearOpBase<double>> create_thyra_linear_op(
      const LinAlg::SparseMatrix& matrix, LinAlg::DataAccess access);

  Teuchos::RCP<const Thyra::LinearOpBase<double>> create_thyra_linear_op(
      const LinAlg::BlockSparseMatrixBase& matrix, LinAlg::DataAccess access);


    // Create Thyra multi vector from MultiVector and Map or Thyra vector space
    Teuchos::RCP<Thyra::MultiVectorBase<double>> create_thyra_multi_vector(
    const LinAlg::MultiVector<double>& multi_vector, const LinAlg::Map& map);

    Teuchos::RCP<Thyra::MultiVectorBase<double>> create_thyra_multi_vector(
        const LinAlg::MultiVector<double>& multi_vector,
        Teuchos::RCP<const Thyra::VectorSpaceBase<double>> map);


// Get Epetra vector from Thyra
  Teuchos::RCP<Epetra_Vector> get_epetra_vector_from_thyra(
      const Core::LinAlg::Map& map, const Teuchos::RCP<::Thyra::VectorBase<double>>& thyra_vector);

  Teuchos::RCP<const Epetra_Vector> get_epetra_vector_from_thyra(const Core::LinAlg::Map& map,
      const Teuchos::RCP<const ::Thyra::VectorBase<double>>& thyra_vector);


    // Get MultiVector from Thyra
  Teuchos::RCP<Core::LinAlg::MultiVector<double>> get_linalg_multi_vector_from_thyra(const Core::LinAlg::Map& map,
    const Teuchos::RCP<const Thyra::MultiVectorBase<double>>& thyra_vector);

}  // namespace Core::LinearSolver::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
