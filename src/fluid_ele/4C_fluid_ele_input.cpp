// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele.hpp"
#include "4C_mat_material_factory.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::Elements::Fluid::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  // set discretization type (setOptimalgaussrule is pushed into element
  // routine)
  set_dis_type(Core::FE::string_to_cell_type(distype));

  std::string na = container.get<std::string>("NA");
  if (na == "ale" or na == "ALE" or na == "Ale")
  {
    is_ale_ = true;
  }
  else if (na == "euler" or na == "EULER" or na == "Euler")
    is_ale_ = false;
  else
    FOUR_C_THROW("Reading of fluid element failed: Euler/Ale");

  return true;
}

FOUR_C_NAMESPACE_CLOSE
