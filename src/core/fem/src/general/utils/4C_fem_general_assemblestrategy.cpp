// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_assemblestrategy.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"

FOUR_C_NAMESPACE_OPEN

Core::FE::AssembleStrategy::AssembleStrategy(int firstdofset, int seconddofset,
    std::shared_ptr<LinAlg::SparseOperator> systemmatrix1,
    std::shared_ptr<LinAlg::SparseOperator> systemmatrix2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector1,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector2,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector3)
    : firstdofset_(firstdofset),
      seconddofset_(seconddofset),
      systemmatrix1_(systemmatrix1),
      systemmatrix2_(systemmatrix2),
      systemvector1_(systemvector1),
      systemvector2_(systemvector2),
      systemvector3_(systemvector3)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::zero()
{
  if (assemblemat1())
  {
    systemmatrix1_->zero();
  }
  if (assemblemat2())
  {
    systemmatrix1_->zero();
  }
  if (assemblevec1())
  {
    systemvector1_->PutScalar(0.0);
  }
  if (assemblevec2())
  {
    systemvector2_->PutScalar(0.0);
  }
  if (assemblevec3())
  {
    systemvector3_->PutScalar(0.0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::complete()
{
  if (assemblemat1())
  {
    systemmatrix1_->complete();
  }
  if (assemblemat2())
  {
    systemmatrix2_->complete();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::clear_element_storage(int rdim, int cdim)
{
  if (assemblemat1())
  {
    if (elematrix1_.numRows() != rdim or elematrix1_.numCols() != cdim)
      elematrix1_.shape(rdim, cdim);
    else
      elematrix1_.putScalar(0.0);
  }
  if (assemblemat2())
  {
    if (elematrix2_.numRows() != rdim or elematrix2_.numCols() != cdim)
      elematrix2_.shape(rdim, cdim);
    else
      elematrix2_.putScalar(0.0);
  }
  if (assemblevec1())
  {
    if (elevector1_.length() != rdim)
      elevector1_.size(rdim);
    else
      elevector1_.putScalar(0.0);
  }
  if (assemblevec2())
  {
    if (elevector2_.length() != rdim)
      elevector2_.size(rdim);
    else
      elevector2_.putScalar(0.0);
  }
  if (assemblevec3())
  {
    if (elevector3_.length() != rdim)
      elevector3_.size(rdim);
    else
      elevector3_.putScalar(0.0);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::assemble(LinAlg::SparseOperator& sysmat, int eid,
    const std::vector<int>& lmstride, const LinAlg::SerialDenseMatrix& Aele,
    const std::vector<int>& lm, const std::vector<int>& lmowner)
{
  sysmat.assemble(lmstride, Aele, lm, lmowner);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::assemble(LinAlg::SparseOperator& sysmat, int eid,
    const std::vector<int>& lmstride, const LinAlg::SerialDenseMatrix& Aele,
    const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
    const std::vector<int>& lmcol)
{
  sysmat.assemble(lmstride, Aele, lmrow, lmrowowner, lmcol);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::assemble(
    LinAlg::SparseOperator& sysmat, double val, int rgid, int cgid)
{
  sysmat.assemble(val, rgid, cgid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::assemble(Core::LinAlg::Vector<double>& V,
    const LinAlg::SerialDenseVector& Vele, const std::vector<int>& lm,
    const std::vector<int>& lmowner)
{
  LinAlg::assemble(V, Vele, lm, lmowner);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::AssembleStrategy::assemble(Core::LinAlg::MultiVector<double>& V, const int n,
    const LinAlg::SerialDenseVector& Vele, const std::vector<int>& lm,
    const std::vector<int>& lmowner)
{
  LinAlg::assemble(V, n, Vele, lm, lmowner);
}

FOUR_C_NAMESPACE_CLOSE
