// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_rigidsphere.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::Elements::Rigidsphere::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // currently only rotationally symmetric profiles for beam --> Iyy = Izz
  radius_ = container.get<double>("RADIUS");
  rho_ = container.get<double>("DENSITY");

  return (true);
}

FOUR_C_NAMESPACE_CLOSE
