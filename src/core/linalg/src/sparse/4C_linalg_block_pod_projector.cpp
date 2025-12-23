// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later


#include "4C_config.hpp"

#include "4C_linalg_block_pod_projector.hpp"

#include "4C_linalg.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  Core::LinAlg::MultiMapExtractor make_multi_map_extractor(
      std::vector<std::shared_ptr<const Core::LinAlg::Map>>&& maps)
  {
    FOUR_C_ASSERT(
        maps.size() == 2, "Currently only two blocks in POD block projections are supported.");
    return {*Core::LinAlg::merge_map(*maps[0], *maps[1], false), maps};
  }

  Core::LinAlg::MultiMapExtractor make_full_map_extractor(
      const std::vector<std::variant<Core::LinAlg::Internal::PODProjectionBlock,
          Core::LinAlg::NoProjectionBlock>>& block_projections)
  {
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> range_maps;
    for (const auto& block : block_projections)
    {
      if (std::holds_alternative<Core::LinAlg::Internal::PODProjectionBlock>(block))
      {
        const auto& projection =
            std::get<Core::LinAlg::Internal::PODProjectionBlock>(block).projection_matrix;
        range_maps.push_back(std::make_shared<const Core::LinAlg::Map>(projection.range_map()));
      }
      else if (std::holds_alternative<Core::LinAlg::NoProjectionBlock>(block))
      {
        const auto& noproj = std::get<Core::LinAlg::NoProjectionBlock>(block);
        range_maps.push_back(std::make_shared<const Core::LinAlg::Map>(noproj.range_map));
      }
    }

    return make_multi_map_extractor(std::move(range_maps));
  }

  Core::LinAlg::MultiMapExtractor make_reduced_map_extractor(
      const std::vector<std::variant<Core::LinAlg::Internal::PODProjectionBlock,
          Core::LinAlg::NoProjectionBlock>>& block_projections)
  {
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> domain_maps;
    for (const auto& block : block_projections)
    {
      if (std::holds_alternative<Core::LinAlg::Internal::PODProjectionBlock>(block))
      {
        const auto& projection =
            std::get<Core::LinAlg::Internal::PODProjectionBlock>(block).projection_matrix;
        domain_maps.push_back(std::make_shared<const Core::LinAlg::Map>(projection.domain_map()));
      }
      else if (std::holds_alternative<Core::LinAlg::NoProjectionBlock>(block))
      {
        const auto& noproj = std::get<Core::LinAlg::NoProjectionBlock>(block);
        domain_maps.push_back(std::make_shared<const Core::LinAlg::Map>(noproj.domain_map));
      }
    }

    return make_multi_map_extractor(std::move(domain_maps));
  }

  std::vector<
      std::variant<Core::LinAlg::Internal::PODProjectionBlock, Core::LinAlg::NoProjectionBlock>>
  initialize_block_projections(const std::vector<std::variant<Core::LinAlg::PODProjectionBlock,
          Core::LinAlg::NoProjectionBlock>>& block_projections)
  {
    std::vector<
        std::variant<Core::LinAlg::Internal::PODProjectionBlock, Core::LinAlg::NoProjectionBlock>>
        internal_block_projections;
    internal_block_projections.reserve(block_projections.size());

    for (const auto& block : block_projections)
    {
      if (std::holds_alternative<Core::LinAlg::PODProjectionBlock>(block))
      {
        const auto& proj_block = std::get<Core::LinAlg::PODProjectionBlock>(block);

        // We need the projection matrix and its transpose in the internal representation
        Core::LinAlg::Internal::PODProjectionBlock internal_proj_block{
            .projection_matrix = {proj_block.projection_matrix, Core::LinAlg::DataAccess::Share},
            .projection_matrix_transpose =
                *Core::LinAlg::matrix_transpose(proj_block.projection_matrix)};

        internal_block_projections.emplace_back(std::move(internal_proj_block));
      }
      else if (std::holds_alternative<Core::LinAlg::NoProjectionBlock>(block))
      {
        const auto& noproj_block = std::get<Core::LinAlg::NoProjectionBlock>(block);
        internal_block_projections.emplace_back(noproj_block);
      }
    }

    return internal_block_projections;
  }
}  // namespace

Core::LinAlg::BlockPODProjector::BlockPODProjector(
    std::vector<std::variant<PODProjectionBlock, NoProjectionBlock>> block_projections)
    : block_projections_(initialize_block_projections(block_projections)),
      reduced_extractor_(make_reduced_map_extractor(block_projections_)),
      full_range_extractor_(make_full_map_extractor(block_projections_))
{
}

Core::LinAlg::Vector<double> Core::LinAlg::BlockPODProjector::to_full(
    const Core::LinAlg::Vector<double>& x) const
{
  Core::LinAlg::Vector<double> y{*full_range_extractor_.full_map(), true};

  for (int i = 0; i < static_cast<int>(block_projections_.size()); ++i)
  {
    if (std::holds_alternative<Internal::PODProjectionBlock>(block_projections_[i]))
    {
      auto& projection =
          std::get<Internal::PODProjectionBlock>(block_projections_[i]).projection_matrix;

      std::shared_ptr<Core::LinAlg::Vector<double>> xi = reduced_extractor_.extract_vector(x, i);

      Core::LinAlg::Vector<double> yi{projection.range_map(), false};
      projection.Apply(*xi, yi);
      full_range_extractor_.add_vector(yi, i, y);
    }
    else if (std::holds_alternative<NoProjectionBlock>(block_projections_[i]))
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> xi = reduced_extractor_.extract_vector(x, i);
      full_range_extractor_.add_vector(*xi, i, y);
    }
  }

  return y;
}

Core::LinAlg::Vector<double> Core::LinAlg::BlockPODProjector::to_reduced(
    const Core::LinAlg::Vector<double>& x) const
{
  Core::LinAlg::Vector<double> y{*reduced_extractor_.full_map(), true};

  for (int i = 0; i < static_cast<int>(block_projections_.size()); ++i)
  {
    if (std::holds_alternative<Internal::PODProjectionBlock>(block_projections_[i]))
    {
      auto& projection =
          std::get<Internal::PODProjectionBlock>(block_projections_[i]).projection_matrix_transpose;

      std::shared_ptr<Core::LinAlg::Vector<double>> xi = full_range_extractor_.extract_vector(x, i);

      Core::LinAlg::Vector<double> yi{projection.range_map(), true};
      projection.Apply(*xi, yi);
      reduced_extractor_.add_vector(yi, i, y);
    }
    else if (std::holds_alternative<NoProjectionBlock>(block_projections_[i]))
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> xi = full_range_extractor_.extract_vector(x, i);
      reduced_extractor_.add_vector(*xi, i, y);
    }
  }

  return y;
}

std::unique_ptr<Core::LinAlg::SparseOperator> Core::LinAlg::BlockPODProjector::to_reduced(
    const Core::LinAlg::SparseOperator& A) const
{
  // The sparse operator needs to be a BlockSparseMatrix
  const auto& block_A = dynamic_cast<const Core::LinAlg::BlockSparseMatrixBase&>(A);

  auto reduced_block_A =
      std::make_unique<Core::LinAlg::BlockSparseMatrix<DefaultBlockMatrixStrategy>>(
          reduced_extractor_, reduced_extractor_);

  FOUR_C_ASSERT(std::cmp_equal(block_A.rows(), block_projections_.size()),
      "The number of block rows is not equal the number of projections. {} != {}", block_A.rows(),
      block_projections_.size());
  FOUR_C_ASSERT(std::cmp_equal(block_A.cols(), block_projections_.size()),
      "The number of block cols is not equal the number of projections. {} != {}", block_A.cols(),
      block_projections_.size());

  // reduce each block
  for (int i = 0; i < block_A.rows(); ++i)
  {
    for (int j = 0; j < block_A.cols(); ++j)
    {
      const auto& block = block_A.matrix(i, j);

      std::unique_ptr<Core::LinAlg::SparseMatrix> reduced_block = nullptr;

      if (std::holds_alternative<Internal::PODProjectionBlock>(block_projections_[i]))
      {
        const auto& row_projection = std::get<Internal::PODProjectionBlock>(block_projections_[i])
                                         .projection_matrix_transpose;

        reduced_block = Core::LinAlg::matrix_multiply(row_projection, false, block, false, true);
      }

      if (std::holds_alternative<Internal::PODProjectionBlock>(block_projections_[j]))
      {
        const auto& col_projection =
            std::get<Internal::PODProjectionBlock>(block_projections_[j]).projection_matrix;
        reduced_block = Core::LinAlg::matrix_multiply(
            reduced_block ? *reduced_block : block, false, col_projection, false, true);
      }

      reduced_block_A->matrix(i, j).assign(
          DataAccess::Share, reduced_block ? *reduced_block : block);
    }
  }
  reduced_block_A->complete();

  return reduced_block_A;
}

FOUR_C_NAMESPACE_CLOSE
