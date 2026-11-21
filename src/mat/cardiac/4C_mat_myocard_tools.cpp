// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_myocard_tools.hpp"

#include "4C_comm_pack_helpers.hpp"

#include <math.h> /* tanh, log */

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardTools::MyocardTools() {}

/*----------------------------------------------------------------------*
 |                                                           cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardTools::gating_function(const double Gate1, const double Gate2, const double p,
    const double var, const double thresh) const
{
  double Erg = Gate1 + (Gate2 - Gate1) * (1.0 + tanh(p * (var - thresh))) / 2;
  return Erg;
}


/*----------------------------------------------------------------------*
 |                                                           cbert 10/14 |
 *----------------------------------------------------------------------*/
double MyocardTools::gating_var_calc(
    const double dt, double y_0, const double y_inf, const double y_tau) const
{
  // Solve dy/dt = (1/a)*(y_inf-y)
  double y_1 = 1.0 / (1.0 + dt / y_tau) * (y_0 + dt * y_inf / y_tau);
  return y_1;
}

FOUR_C_NAMESPACE_CLOSE
