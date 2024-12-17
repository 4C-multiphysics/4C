// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ALE_UTILS_HPP
#define FOUR_C_ALE_UTILS_HPP


#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace ALE
{
  namespace Utils
  {
    /// (FSI) interface block matrix split strategy
    class InterfaceSplitStrategy : public Core::LinAlg::DefaultBlockMatrixStrategy
    {
     public:
      explicit InterfaceSplitStrategy(Core::LinAlg::BlockSparseMatrixBase& mat)
          : Core::LinAlg::DefaultBlockMatrixStrategy(mat)
      {
      }

      /// assemble into the given block
      void assemble(int myrank, const std::vector<int>& lmstride,
          const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
          const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
      {
        if (condelements_->size() > 0)
        {
          // if we have an element with conditioned nodes, we have to do the
          // default assembling
          Core::LinAlg::DefaultBlockMatrixStrategy::assemble(
              myrank, lmstride, Aele, lmrow, lmrowowner, lmcol);
        }
        else
        {
          // if there are no conditioned nodes we can simply assemble to the
          // internal matrix
          Core::LinAlg::SparseMatrix& matrix = mat().matrix(0, 0);
          matrix.assemble(lmstride, Aele, lmrow, lmrowowner, lmcol);
        }
      }

      void assemble(double val, int rgid, int cgid)
      {
        // forward single value assembling
        Core::LinAlg::DefaultBlockMatrixStrategy::assemble(val, rgid, cgid);
      }

      void set_cond_elements(std::shared_ptr<std::set<int>> condelements)
      {
        condelements_ = condelements;
      }

     private:
      std::shared_ptr<std::set<int>> condelements_;
    };
  }  // namespace Utils
}  // namespace ALE


FOUR_C_NAMESPACE_CLOSE

#endif
