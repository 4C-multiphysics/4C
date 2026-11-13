// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_anisotropy_utils.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

template <typename T, unsigned int numfib>
void Mat::compute_structural_tensors(
    std::vector<std::array<Core::LinAlg::Tensor<double, 3>, numfib>>& fibers,
    std::vector<std::array<T, numfib>>& structural_tensor,
    const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& strategy)
{
  // Need to compute the stuctural tensors
  if (!strategy)
  {
    FOUR_C_THROW("Structural tensor strategy is null!");
  }

  structural_tensor.resize(fibers.size());
  for (std::size_t gp = 0; gp < fibers.size(); ++gp)
  {
    for (std::size_t i = 0; i < numfib; ++i)
    {
      T A{};
      strategy->setup_structural_tensor(fibers[gp][i], A);

      structural_tensor[gp].at(i) = A;
    }
  }
}

/*----------------------------------------------------------------------------*/
// explicit instantiation of template functions
template void Mat::compute_structural_tensors<Core::LinAlg::SymmetricTensor<double, 3, 3>, 1u>(
    std::vector<std::array<Core::LinAlg::Tensor<double, 3>, 1>>& fibers,
    std::vector<std::array<Core::LinAlg::SymmetricTensor<double, 3, 3>, 1>>& structural_tensor,
    const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& strategy);
template void Mat::compute_structural_tensors<Core::LinAlg::SymmetricTensor<double, 3, 3>, 2u>(
    std::vector<std::array<Core::LinAlg::Tensor<double, 3>, 2>>& fibers,
    std::vector<std::array<Core::LinAlg::SymmetricTensor<double, 3, 3>, 2>>& structural_tensor,
    const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& strategy);

FOUR_C_NAMESPACE_CLOSE
