// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_thyra_utils.hpp"

#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teko_EpetraInverseOpWrapper.hpp>
#include <Thyra_EpetraLinearOp.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>
#include <Thyra_LinearOpDefaultBase.hpp>
#include <Thyra_MultiVectorBase.hpp>
#include <Thyra_PhysicallyBlockedLinearOpBase.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  class SparseOperatorThyraWrapper final : public ::Thyra::LinearOpDefaultBase<double>
  {
   public:
    explicit SparseOperatorThyraWrapper(std::shared_ptr<const Core::LinAlg::SparseOperator> matrix)
        : matrix_(std::move(matrix))
    {
      FOUR_C_ASSERT_ALWAYS(
          matrix_ != nullptr, "Cannot create a Thyra operator from a null SparseOperator.");

      domain_space_ = Core::LinearSolver::Utils::create_thyra_map(matrix_->domain_map());

      range_space_ = Core::LinearSolver::Utils::create_thyra_map(matrix_->range_map());
    }

    Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> domain() const override
    {
      return domain_space_;
    }

    Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> range() const override
    {
      return range_space_;
    }

   protected:
    bool opSupportedImpl(const ::Thyra::EOpTransp transpose) const override
    {
      return transpose == ::Thyra::NOTRANS;
    }

    void applyImpl(const ::Thyra::EOpTransp transpose, const ::Thyra::MultiVectorBase<double>& x,
        const Teuchos::Ptr<::Thyra::MultiVectorBase<double>>& y, const double alpha,
        const double beta) const override
    {
      FOUR_C_ASSERT_ALWAYS(transpose == ::Thyra::NOTRANS,
          "SparseOperatorThyraWrapper currently supports only "
          "non-transposed application.");

      FOUR_C_ASSERT_ALWAYS(
          !y.is_null(), "Cannot apply a Thyra operator to a null target multivector.");

      /*
       * get_Epetra_MultiVector() returns an Epetra view of any compatible
       * Thyra multivector. No values are copied here.
       */
      auto epetra_x = ::Thyra::get_Epetra_MultiVector(
          matrix_->domain_map().get_epetra_map(), Teuchos::rcpFromRef(x));

      auto epetra_y = ::Thyra::get_Epetra_MultiVector(
          matrix_->range_map().get_epetra_map(), Teuchos::rcpFromRef(*y));

      auto x_view = Core::LinAlg::MultiVector<double>::create_view(*epetra_x);

      auto y_view = Core::LinAlg::MultiVector<double>::create_view(*epetra_y);

      Core::LinAlg::MultiVector<double> result(matrix_->range_map(), epetra_x->NumVectors(), true);

      matrix_->multiply(false, *x_view, result);

      // y = beta*y + alpha*A*x
      y_view->update(alpha, result, beta);
    }

   private:
    std::shared_ptr<const Core::LinAlg::SparseOperator> matrix_;

    Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> domain_space_;

    Teuchos::RCP<const ::Thyra::VectorSpaceBase<double>> range_space_;
  };
}  // namespace

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> Core::LinearSolver::Utils::create_thyra_map(
    const Core::LinAlg::Map& map)
{
  Teuchos::RCP<Epetra_Map> epetra_map;

  try
  {
    epetra_map = Teuchos::make_rcp<Epetra_Map>(map.get_epetra_map());
  }
  catch (const Core::Exception&)
  {
    /*
     * A point map can only be constructed directly this way if every
     * BlockMap element represents exactly one point.
     */
    FOUR_C_ASSERT_ALWAYS(map.num_global_points() == map.num_global_elements() &&
                             map.num_my_points() == map.num_my_elements(),
        "Cannot create a flat Thyra vector space from an "
        "Epetra_BlockMap with element sizes greater than one.");

    epetra_map = Teuchos::make_rcp<Epetra_Map>(map.num_global_points(), map.num_my_points(),
        map.my_global_elements(), map.index_base(), map.get_epetra_block_map().Comm());
  }

  return Thyra::create_VectorSpace(epetra_map);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::MultiVectorBase<double>>
Core::LinearSolver::Utils::create_thyra_multi_vector(
    const Core::LinAlg::MultiVector<double>& multi_vector, const Core::LinAlg::Map& map)
{
  return Thyra::create_MultiVector(
      Teuchos::rcpFromRef(multi_vector.get_epetra_multi_vector()), create_thyra_map(map));
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<Thyra::MultiVectorBase<double>> Core::LinearSolver::Utils::create_thyra_multi_vector(
    Core::LinAlg::MultiVector<double>& multi_vector, const Core::LinAlg::Map& map)
{
  return Thyra::create_MultiVector(
      Teuchos::rcpFromRef(multi_vector.get_epetra_multi_vector()), create_thyra_map(map));
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::LinearOpBase<double>> Core::LinearSolver::Utils::create_thyra_linear_op(
    std::shared_ptr<const LinAlg::SparseOperator> matrix)
{
  FOUR_C_ASSERT_ALWAYS(
      matrix != nullptr, "Cannot create a Thyra operator from a null SparseOperator.");

  return Teuchos::make_rcp<SparseOperatorThyraWrapper>(std::move(matrix));
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::LinearOpBase<double>> Core::LinearSolver::Utils::create_thyra_linear_op(
    const Core::LinAlg::SparseMatrix& matrix, Core::LinAlg::DataAccess access)
{
  Teuchos::RCP<const Epetra_CrsMatrix> A_crs;

  if (access == Core::LinAlg::DataAccess::Copy)
  {
    A_crs = Teuchos::make_rcp<Epetra_CrsMatrix>(matrix.epetra_matrix());
  }
  else
  {
    A_crs = Teuchos::rcpFromRef(matrix.epetra_matrix());
  }

  return Thyra::epetraLinearOp(A_crs);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::LinearOpBase<double>> Core::LinearSolver::Utils::create_thyra_linear_op(
    const Core::LinAlg::BlockSparseMatrixBase& matrix, Core::LinAlg::DataAccess access)
{
  auto block_matrix = Thyra::defaultBlockedLinearOp<double>();

  block_matrix->beginBlockFill(matrix.rows(), matrix.cols());
  for (int row = 0; row < matrix.rows(); ++row)
  {
    for (int col = 0; col < matrix.cols(); ++col)
    {
      block_matrix->setBlock(row, col, create_thyra_linear_op(matrix(row, col), access));
    }
  }
  block_matrix->endBlockFill();

  return block_matrix;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
std::shared_ptr<Epetra_Operator> Core::LinearSolver::Utils::get_epetra_inverse_operator_from_thyra(
    const Teuchos::RCP<const Thyra::LinearOpBase<double>> thyra_linear_op)
{
  return std::make_shared<Teko::Epetra::EpetraInverseOpWrapper>(thyra_linear_op);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<Epetra_Vector> Core::LinearSolver::Utils::get_epetra_vector_from_thyra(
    const Core::LinAlg::Map& map, const Teuchos::RCP<::Thyra::VectorBase<double>>& thyra_vector)
{
  return ::Thyra::get_Epetra_Vector(map.get_epetra_map(), thyra_vector);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Epetra_Vector> Core::LinearSolver::Utils::get_epetra_vector_from_thyra(
    const Core::LinAlg::Map& map,
    const Teuchos::RCP<const ::Thyra::VectorBase<double>>& thyra_vector)
{
  return ::Thyra::get_Epetra_Vector(map.get_epetra_map(), thyra_vector);
}

FOUR_C_NAMESPACE_CLOSE
