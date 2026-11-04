// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FIBER_INTERPOLATION_HPP
#define FOUR_C_MAT_FIBER_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_tensor.hpp"

#include <numeric>
#include <ranges>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /**
   * A simple interpolation strategy for unit-vector fields (e.g., fibers)
   */
  struct FiberInterpolation
  {
    Core::LinAlg::Tensor<double, 3> operator()(const std::ranges::sized_range auto& weights,
        const std::ranges::sized_range auto& fibers) const
    {
      // Do default interpolation
      Core::LinAlg::Tensor<double, 3> interpolated_fiber = std::inner_product(weights.begin(),
          weights.end(), fibers.begin(), Core::LinAlg::Tensor<double, 3>{{0.0, 0.0, 0.0}});

      // normalize vector
      return interpolated_fiber / Core::LinAlg::norm2(interpolated_fiber);
    }
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
