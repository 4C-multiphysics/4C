// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_utils_infnormscaling.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <stdio.h>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FLD::Utils::FluidInfNormScaling::FluidInfNormScaling(Core::LinAlg::MapExtractor& mapextractor)
    : myrank_(Core::Communication::my_mpi_rank(mapextractor.map(0)->get_comm())),
      velpressplitter_(mapextractor),
      leftscale_momentum_(true),
      leftscale_continuity_(false)
{
  return;
};

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::Utils::FluidInfNormScaling::scale_system(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix, Core::LinAlg::Vector<double>& b)
{
  if (myrank_ == 0) std::cout << "Performing scaling of linear system" << std::endl;

  // The matrices are modified here. Do we have to revert the change later on?
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> matrcp =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(matrix);

  if (matrcp != nullptr)  // yes, we have a block sparse matrix
  {
    Core::LinAlg::BlockSparseMatrixBase& mat = *matrcp;

    Core::LinAlg::SparseMatrix& A00 = mat.matrix(0, 0);
    srowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A00.row_map(), false);
    scolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A00.row_map(), false);

    if (leftscale_momentum_)
    {
      A00.inv_row_sums(*srowsum_);
      if (myrank_ == 0) std::cout << "do left scaling momentum" << std::endl;

      // we want to have the infnorm of the whole(!) row including
      // the off-diagonal block matrix M_(0,1)
      Core::LinAlg::Vector<double> temp1(mat.matrix(0, 0).row_map(), false);
      srowsum_->reciprocal(*srowsum_);
      mat.matrix(0, 1).inv_row_sums(temp1);
      temp1.reciprocal(temp1);
      srowsum_->update(1.0, temp1, 1.0);
      srowsum_->reciprocal(*srowsum_);
    }
    else
    {
      // no scaling
      srowsum_->put_scalar(1.0);
    }

    scolsum_->put_scalar(1.0);

    if (A00.left_scale(*srowsum_) or A00.right_scale(*scolsum_) or
        mat.matrix(0, 1).left_scale(*srowsum_) or mat.matrix(1, 0).right_scale(*scolsum_))
      FOUR_C_THROW("fluid scaling failed");

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = velpressplitter_.extract_vector(b, 0);

    sx->multiply(1.0, *srowsum_, *sx, 0.0);

    velpressplitter_.insert_vector(*sx, 0, b);

    // continuity equation
    Core::LinAlg::SparseMatrix& A11 = mat.matrix(1, 1);
    prowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A11.row_map(), false);
    pcolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A11.row_map(), false);

    Core::LinAlg::Vector<double> temp(A11.row_map(), false);
    if (leftscale_continuity_)
    {
      A11.inv_row_sums(*prowsum_);
      if (myrank_ == 0) std::cout << "do left scaling continuity" << std::endl;

      // we want to have the infnorm of the whole(!) row including
      // the off-diagonal block matrix M_(1,0)
      prowsum_->reciprocal(*prowsum_);
      mat.matrix(1, 0).inv_row_sums(temp);
      temp.reciprocal(temp);
      prowsum_->update(1.0, temp, 1.0);
      prowsum_->reciprocal(*prowsum_);
    }
    else
    {
      prowsum_->put_scalar(1.0);
    }

    pcolsum_->put_scalar(1.0);

    if (A11.left_scale(*prowsum_) or mat.matrix(1, 0).left_scale(*prowsum_))
      FOUR_C_THROW("fluid scaling failed");

    std::shared_ptr<Core::LinAlg::Vector<double>> px = velpressplitter_.extract_vector(b, 1);

    px->multiply(1.0, *prowsum_, *px, 0.0);

    velpressplitter_.insert_vector(*px, 1, b);


  }  // BlockSparseMatrix

  else  // we have a normal SparseMatrix
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> smat =
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(matrix);
    if (smat == nullptr) FOUR_C_THROW("Something went wrong.");

    srowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(smat->row_map(), false);
    scolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(smat->row_map(), false);
    prowsum_ = nullptr;
    pcolsum_ = nullptr;

    smat->inv_row_sums(*srowsum_);
    if (myrank_ == 0) std::cout << "do left scaling of SparseMatrix" << std::endl;

    // leave continuity equation unscaled! -> scaling factors are one
    std::shared_ptr<Core::LinAlg::Vector<double>> px =
        velpressplitter_.extract_vector(*srowsum_, 1);
    px->put_scalar(1.0);
    velpressplitter_.insert_vector(*px, 1, *srowsum_);

    if (smat->left_scale(*srowsum_)) FOUR_C_THROW("fluid scaling failed");
    b.multiply(1.0, *srowsum_, b, 0.0);

    smat->inv_col_sums(*scolsum_);
    if (myrank_ == 0) std::cout << "do right scaling pressure" << std::endl;

    // leave velocity columns equation unscaled!
    std::shared_ptr<Core::LinAlg::Vector<double>> ux =
        velpressplitter_.extract_vector(*scolsum_, 0);
    ux->put_scalar(1.0);
    velpressplitter_.insert_vector(*ux, 0, *scolsum_);

    if (smat->right_scale(*scolsum_)) FOUR_C_THROW("fluid scaling failed");
  }  // SparseMatrix


  // do output
  double srownorm, scolnorm, prownorm = 0.0;
  srowsum_->mean_value(&srownorm);
  scolsum_->mean_value(&scolnorm);
  if (prowsum_ != nullptr) prowsum_->mean_value(&prownorm);

  if (myrank_ == 0)
    std::cout << "MEAN: leftscalemom: " << srownorm << "  rightscale: " << scolnorm
              << "  leftscaleconti: " << prownorm << std::endl;

  srowsum_->min_value(&srownorm);
  scolsum_->min_value(&scolnorm);
  if (prowsum_ != nullptr) prowsum_->min_value(&prownorm);
  if (myrank_ == 0)
    std::cout << "MIN: leftscalemom: " << srownorm << "  rightscale: " << scolnorm
              << "  leftscaleconti: " << prownorm << std::endl;

  srowsum_->max_value(&srownorm);
  scolsum_->max_value(&scolnorm);
  if (prowsum_ != nullptr) prowsum_->max_value(&prownorm);
  if (myrank_ == 0)
    std::cout << "MAX: leftscalemom: " << srownorm << "  rightscale: " << scolnorm
              << "  leftscaleconti: " << prownorm << std::endl;

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FLD::Utils::FluidInfNormScaling::unscale_solution(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix, Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& b)
{
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> matrcp =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(matrix);

  if (matrcp != nullptr)  // yes, we have a block sparse matrix
  {
    Core::LinAlg::BlockSparseMatrixBase& mat = *matrcp;

    std::shared_ptr<Core::LinAlg::Vector<double>> sy = velpressplitter_.extract_vector(x, 0);

    sy->multiply(1.0, *scolsum_, *sy, 0.0);

    velpressplitter_.insert_vector(*sy, 0, x);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = velpressplitter_.extract_vector(b, 0);

    sx->reciprocal_multiply(1.0, *srowsum_, *sx, 0.0);

    velpressplitter_.insert_vector(*sx, 0, b);

    Core::LinAlg::SparseMatrix& A00 = mat.matrix(0, 0);
    srowsum_->reciprocal(*srowsum_);
    scolsum_->reciprocal(*scolsum_);
    if (A00.left_scale(*srowsum_) or A00.right_scale(*scolsum_) or
        mat.matrix(0, 1).left_scale(*srowsum_) or mat.matrix(1, 0).right_scale(*scolsum_))
      FOUR_C_THROW("fluid scaling failed");

    // undo left scaling of continuity equation
    Core::LinAlg::SparseMatrix& A11 = mat.matrix(1, 1);
    prowsum_->reciprocal(*prowsum_);
    if (A11.left_scale(*prowsum_) or mat.matrix(1, 0).left_scale(*prowsum_))
      FOUR_C_THROW("fluid scaling failed");
  }
  else
  {
    x.multiply(1.0, *scolsum_, x, 0.0);

    srowsum_->reciprocal(*srowsum_);
    scolsum_->reciprocal(*scolsum_);

    // revert matrix and rhs here
    if (myrank_ == 0)
      std::cout << "Only unscaling for solution vector!!! Matrix untouched. " << std::endl;
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
