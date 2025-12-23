// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_BLOCK_POD_PROJECTOR_HPP
#define FOUR_C_LINALG_BLOCK_POD_PROJECTOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linear_solver_method_projector.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  struct PODProjectionBlock
  {
    SparseMatrix projection_matrix;
  };

  namespace Internal
  {
    struct PODProjectionBlock
    {
      SparseMatrix projection_matrix;
      SparseMatrix projection_matrix_transpose;
    };
  }  // namespace Internal
  struct NoProjectionBlock
  {
    Map range_map;
    Map domain_map;
  };

  /*!
   * @brief A block-wise POD projector.
   *
   * Each block can be projected with a different POD basis or not projected at all.
   *
   * @note Currently, only 2x2 block matrices are supported.
   */
  class BlockPODProjector : public LinearSystemProjector
  {
   public:
    BlockPODProjector(
        std::vector<std::variant<PODProjectionBlock, NoProjectionBlock>> block_projections);

    [[nodiscard]] Core::LinAlg::Vector<double> to_full(
        const Core::LinAlg::Vector<double>& x) const override;

    [[nodiscard]] Core::LinAlg::Vector<double> to_reduced(
        const Core::LinAlg::Vector<double>& x) const override;

    [[nodiscard]] std::unique_ptr<Core::LinAlg::SparseOperator> to_reduced(
        const Core::LinAlg::SparseOperator& A) const override;

   private:
    std::vector<std::variant<Internal::PODProjectionBlock, NoProjectionBlock>> block_projections_;

    MultiMapExtractor reduced_extractor_;
    MultiMapExtractor full_range_extractor_;
  };

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
