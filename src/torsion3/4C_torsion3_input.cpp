// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_material_factory.hpp"
#include "4C_torsion3.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*/
bool Discret::Elements::Torsion3::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read type of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  // read type of bending potential
  auto buffer = container.get<std::string>("BENDINGPOTENTIAL");

  // bending potential E_bend = 0.5*SPRING*\theta^2
  if (buffer == "quadratic") bendingpotential_ = quadratic;

  // bending potential E_bend = SPRING*(1 - \cos(\theta^2) )
  else if (buffer == "cosine")
    bendingpotential_ = cosine;

  else
    FOUR_C_THROW("Reading of Torsion3 element failed because of unknown potential type!");

  return true;
}

FOUR_C_NAMESPACE_CLOSE
