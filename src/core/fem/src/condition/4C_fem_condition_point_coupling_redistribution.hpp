// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_CONDITION_POINT_COUPLING_REDISTRIBUTION_HPP
#define FOUR_C_FEM_CONDITION_POINT_COUPLING_REDISTRIBUTION_HPP

#include "4C_config.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Conditions
{
  /*!
   * The procedure to assign degrees of freedom relies for point coupling conditions on a
   * layout in which the master and connected slave nodes must be owned by the same processor. This
   * function redistributes the row nodes accordingly.
   */
  void redistribute_for_point_coupling_conditions(Core::FE::Discretization& discret);

}  // namespace Core::Conditions

FOUR_C_NAMESPACE_CLOSE

#endif
