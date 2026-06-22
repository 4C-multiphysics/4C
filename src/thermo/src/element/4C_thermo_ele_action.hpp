// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_ELE_ACTION_HPP
#define FOUR_C_THERMO_ELE_ACTION_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <iostream>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Thermo
{
  /*--------------------------------------------------------------------------
   | enum that provides all possible thermo actions
   *--------------------------------------------------------------------------*/
  enum Action
  {
    none,
    calc_thermo_fint,
    calc_thermo_fintcapa,
    calc_thermo_finttang,
    calc_thermo_heatflux,
    integrate_shape_functions,
    calc_thermo_update_istep,
    calc_thermo_reset_istep,
    calc_thermo_energy,
    calc_thermo_coupltang,
    calc_thermo_fintcond,
    calc_thermo_fext,
    calc_thermo_error,
  };  // enum Action

  /*--------------------------------------------------------------------------
   | enum that provides all possible thermo actions on a boundary
   *--------------------------------------------------------------------------*/
  enum BoundaryAction
  {
    boundary_none,
    calc_thermo_fextconvection,
    calc_thermo_fextconvection_coupltang,
  };

}  // namespace Thermo

FOUR_C_NAMESPACE_CLOSE

#endif
