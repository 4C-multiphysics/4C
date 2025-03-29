// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LUBRICATION_ELE_ACTION_HPP
#define FOUR_C_LUBRICATION_ELE_ACTION_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Lubrication
{
  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible lubrication actions
   *///                                                            wirtz 11/15
  /*--------------------------------------------------------------------------*/
  enum Action
  {
    // domain action
    calc_mat_and_rhs,           // calc_condif_systemmat_and_residual,
    calc_mean_pressures,        // calc_mean_pressures,
    calc_error,                 // calc_error
    calc_lubrication_coupltang  // calculate off-diagonal tangent matrix term
  };  // enum Action

  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible lubrication actions on a boundary
   *///                                                            wirtz 11/15
  /*--------------------------------------------------------------------------*/
  enum BoundaryAction
  {
    // new action
    bd_calc_weak_Dirichlet,  // weak_dirichlet,
    bd_calc_Neumann,         // n/a
  };  // enum Lubrication::BoundaryAction
}  // namespace Lubrication

FOUR_C_NAMESPACE_CLOSE

#endif
