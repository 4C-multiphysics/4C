// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PASI_DYN_HPP
#define FOUR_C_PASI_DYN_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Global
{
  class Problem;
}  // namespace Global

/*!
 * \brief control routine for particle structure interaction
 *
 * \param problem  global problem providing parameters, discretizations and solvers
 */
void pasi_dyn(Global::Problem& problem);

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
