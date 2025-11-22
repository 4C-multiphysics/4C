// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_thyra_utils.hpp"

#include <Thyra_DefaultProductMultiVector.hpp>
#include <Thyra_DefaultProductVectorSpace.hpp>
#include <Thyra_EpetraLinearOp.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>
#include <Thyra_PhysicallyBlockedLinearOpBase.hpp>
#include <Thyra_ProductMultiVectorBase.hpp>
#include <Thyra_SpmdMultiVectorBase.hpp>

FOUR_C_NAMESPACE_OPEN

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> Core::LinearSolver::Utils::create_thyra_map(
    const Core::LinAlg::Map& map)
{
  return Thyra::create_VectorSpace(Teuchos::rcpFromRef(map.get_epetra_map()));
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<Thyra::MultiVectorBase<double>> Core::LinearSolver::Utils::create_thyra_multi_vector(
    const Core::LinAlg::MultiVector<double>& multi_vector, const Core::LinAlg::Map& map)
{
  auto const_thyra_vector = Thyra::create_MultiVector(
      Teuchos::rcpFromRef(multi_vector.get_epetra_multi_vector()), create_thyra_map(map));

  return Teuchos::rcp_const_cast<Thyra::MultiVectorBase<double>>(const_thyra_vector);
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
void blockEpetraToThyra(int numVectors, double* epetraData, int leadingDim,
    const Teuchos::Ptr<Thyra::MultiVectorBase<double>>& mv, int& localDim)
{
  localDim = 0;

  // check the base case
  const Teuchos::Ptr<Thyra::ProductMultiVectorBase<double>> prodMV =
      ptr_dynamic_cast<Thyra::ProductMultiVectorBase<double>>(mv);
  if (prodMV == Teuchos::null)
  {
    // VS object must be a SpmdMultiVector object
    const Teuchos::Ptr<Thyra::SpmdMultiVectorBase<double>> spmdX =
        ptr_dynamic_cast<Thyra::SpmdMultiVectorBase<double>>(mv, true);
    const Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<double>> spmdVS = spmdX->spmdSpace();

    int localSubDim = spmdVS->localSubDim();

    Thyra::Ordinal thyraLeadingDim = 0;

    Teuchos::ArrayRCP<double> thyraData_arcp;
    Teuchos::ArrayView<double> thyraData;
    spmdX->getNonconstLocalData(Teuchos::outArg(thyraData_arcp), Teuchos::outArg(thyraLeadingDim));
    thyraData = thyraData_arcp();  // build array view

    for (int i = 0; i < localSubDim; i++)
    {
      // copy each vector
      for (int v = 0; v < numVectors; v++)
      {
        thyraData[i + thyraLeadingDim * v] = epetraData[i + leadingDim * v];
      }
    }

    localDim = localSubDim;

    return;
  }

  // this keeps track of current location in the epetraData vector
  double* localData = epetraData;

  // loop over all the blocks in the vector space
  for (int blkIndex = 0; blkIndex < prodMV->productSpace()->numBlocks(); blkIndex++)
  {
    int subDim = 0;
    const Teuchos::RCP<Thyra::MultiVectorBase<double>> blockVec =
        prodMV->getNonconstMultiVectorBlock(blkIndex);

    // perorm the recursive copy
    blockEpetraToThyra(numVectors, localData, leadingDim, blockVec.ptr(), subDim);

    // shift to the next block
    localData += subDim;

    // account for the size of this subblock
    localDim += subDim;
  }
}

Teuchos::RCP<Thyra::MultiVectorBase<double>> Core::LinearSolver::Utils::create_thyra_multi_vector(
    const Core::LinAlg::MultiVector<double>& multi_vector,
    Teuchos::RCP<const Thyra::VectorSpaceBase<double>> map)
{
  // TODO: We might need to cast if we have a product vector space and handle things differently ...
  // If we have a product space, we will also need to get a product vector.
  auto product_map = Teuchos::rcp_dynamic_cast<const Thyra::DefaultProductVectorSpace<double>>(map);

  if (product_map.is_null())
  {
    auto const_thyra_vector =
        Thyra::create_MultiVector(Teuchos::rcpFromRef(multi_vector.get_epetra_multi_vector()), map);

    return Teuchos::rcp_const_cast<Thyra::MultiVectorBase<double>>(const_thyra_vector);
  }
  else
  {
    const int num_vectors = multi_vector.num_vectors();

    auto thyra_multi_vector = Thyra::defaultProductMultiVector<double>(product_map, num_vectors);

    // extract local information from the Epetra_MultiVector
    int leadingDim = 0, localDim = 0;
    double* epetraData = 0;
    multi_vector.get_epetra_multi_vector().ExtractView(&epetraData, &leadingDim);

    blockEpetraToThyra(num_vectors, epetraData, leadingDim, thyra_multi_vector.ptr(), localDim);

    return thyra_multi_vector;
  }
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Teuchos::RCP<const Thyra::LinearOpBase<double>> Core::LinearSolver::Utils::create_thyra_linear_op(
    Core::LinAlg::SparseOperator& matrix, Core::LinAlg::DataAccess access)
{
  auto block_matrix = std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
      Core::Utils::shared_ptr_from_ref(matrix));

  if (block_matrix == nullptr)
  {
    auto sparse_matrix = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
        Core::Utils::shared_ptr_from_ref(matrix));
    return create_thyra_linear_op(*sparse_matrix, access);
  }
  else
  {
    return create_thyra_linear_op(*block_matrix, access);
  }
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
  for (int row = 0; row < matrix.rows(); row++)
  {
    for (int col = 0; col < matrix.cols(); col++)
    {
      Teuchos::RCP<const Epetra_CrsMatrix> A_crs;

      if (access == Core::LinAlg::DataAccess::Copy)
      {
        A_crs = Teuchos::make_rcp<Epetra_CrsMatrix>(matrix(row, col).epetra_matrix());
      }
      else
      {
        A_crs = Teuchos::rcpFromRef(matrix(row, col).epetra_matrix());
      }

      block_matrix->setBlock(row, col, Thyra::epetraLinearOp(A_crs));
    }
  }
  block_matrix->endBlockFill();

  return block_matrix;
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


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
void blockThyraToEpetra(int numVectors, double* epetraData, int leadingDim,
    const Teuchos::RCP<const Thyra::MultiVectorBase<double>>& tX, int& localDim)
{
  localDim = 0;

  // check the base case
  const Teuchos::RCP<const Thyra::ProductMultiVectorBase<double>> prodX =
      rcp_dynamic_cast<const Thyra::ProductMultiVectorBase<double>>(tX);
  if (prodX == Teuchos::null)
  {
    // the base case

    // VS object must be a SpmdMultiVector object
    Teuchos::RCP<const Thyra::SpmdMultiVectorBase<double>> spmdX =
        rcp_dynamic_cast<const Thyra::SpmdMultiVectorBase<double>>(tX, true);
    Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<double>> spmdVS = spmdX->spmdSpace();

    int localSubDim = spmdVS->localSubDim();

    Thyra::Ordinal thyraLeadingDim = 0;

    Teuchos::ArrayView<const double> thyraData;
    Teuchos::ArrayRCP<const double> thyraData_arcp;
    spmdX->getLocalData(Teuchos::outArg(thyraData_arcp), Teuchos::outArg(thyraLeadingDim));
    thyraData = thyraData_arcp();  // grab the array view

    for (int i = 0; i < localSubDim; i++)
    {
      // copy each vector
      for (int v = 0; v < numVectors; v++)
        epetraData[i + leadingDim * v] = thyraData[i + thyraLeadingDim * v];
    }

    localDim = localSubDim;

    return;
  }

  const Teuchos::RCP<const Thyra::ProductVectorSpaceBase<double>> prodVS = prodX->productSpace();

  // this keeps track of current location in the epetraData vector
  double* localData = epetraData;

  // loop over all the blocks in the vector space
  for (int blkIndex = 0; blkIndex < prodVS->numBlocks(); blkIndex++)
  {
    int subDim = 0;

    // construct the block vector
    blockThyraToEpetra(
        numVectors, localData, leadingDim, prodX->getMultiVectorBlock(blkIndex), subDim);

    // shift to the next block
    localData += subDim;

    // account for the size of this subblock
    localDim += subDim;
  }
}

Teuchos::RCP<Core::LinAlg::MultiVector<double>> Core::LinearSolver::Utils::get_linalg_multi_vector_from_thyra(
    const Core::LinAlg::Map& map, const Teuchos::RCP<const Thyra::MultiVectorBase<double>>& thyra_vector)
{
  // build an Epetra_MultiVector object
  int numVectors = thyra_vector->domain()->dim();

  auto epetra_multi_vector = Epetra_MultiVector(map.get_epetra_map(), numVectors);

  // extract local information from the Epetra_MultiVector
  int leadingDim = 0, localDim = 0;
  double* epetraData = 0;
  epetra_multi_vector.ExtractView(&epetraData, &leadingDim);

  // perform recursive copy
  blockThyraToEpetra(numVectors, epetraData, leadingDim, thyra_vector, localDim);

  return Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(epetra_multi_vector);
}

FOUR_C_NAMESPACE_CLOSE
