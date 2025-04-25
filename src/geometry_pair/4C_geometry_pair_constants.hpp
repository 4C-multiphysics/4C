// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_CONSTANTS_HPP
#define FOUR_C_GEOMETRY_PAIR_CONSTANTS_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace GeometryPair
{
  namespace Constants
  {
    //! Parameter coordinates are considered equal if they are within this tolerance.
    const double projection_xi_eta_tol = 1e-8;

    //! Maximum number of iterations in the local Newton.
    const unsigned int local_newton_iter_max = 10;

    //! Residuum is considered converged if its absolute value is smaller than this tolerance.
    const double local_newton_res_tol = 1e-10;

    //! Newton iteration is considered non solvable if the residuum is larger than this value.
    const double local_newton_res_max = 1e+10;

    //! Jacobian is considered non invertible if it is smaller than this value.
    const double local_newton_det_tol = 1e-8;

    //! Tolerance for positions to be equal.
    const double pos_tol = 1e-10;
  }  // namespace Constants
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif
