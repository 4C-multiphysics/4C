// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_NONLINEAR_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_NONLINEAR_SOLVER_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  /**
   * @brief Common interface of the reduced-lung nonlinear solvers.
   *
   * The time integration only interacts with the selected solver through this interface.
   */
  class NonlinearSolver
  {
   public:
    virtual ~NonlinearSolver() = default;

    /**
     * @brief Solve the nonlinear system at the given physical time.
     *
     * @param time Current physical time for time-dependent boundary conditions.
     * @return Number of nonlinear iterations performed.
     */
    virtual unsigned int solve(double time) = 0;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
