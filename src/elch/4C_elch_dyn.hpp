// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ELCH_DYN_HPP
#define FOUR_C_ELCH_DYN_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Global
{
  class Problem;
}  // namespace Global

/*!
 * \brief Entry point for the solution of electrochemistry problems
 *
 * \param problem  global problem providing parameters, discretizations and solvers
 * \param restart  restart step (0 if no restart is performed)
 */
void elch_dyn(Global::Problem& problem, int restart);

/*! prints the 4C electrochemistry-module logo on the screen */
void printlogo();


FOUR_C_NAMESPACE_CLOSE

#endif
