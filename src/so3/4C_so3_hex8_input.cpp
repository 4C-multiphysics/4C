// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_robinson.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_so3_hex8.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::Elements::SoHex8::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");

  set_material(0, Mat::factory(material_id));

  // set up of materials with GP data (e.g., history variables)
  solid_material()->setup(NUMGPT_SOH8, container);

  // read kinematic flag
  std::string kinem = container.get<std::string>("KINEM");
  if (kinem == "linear")
  {
    kintype_ = Inpar::Solid::KinemType::linear;
  }
  else if (kinem == "nonlinear")
  {
    kintype_ = Inpar::Solid::KinemType::nonlinearTotLag;
  }
  else
    FOUR_C_THROW("Reading SO_HEX8 element failed");

  // check if material kinematics is compatible to element kinematics
  solid_material()->valid_kinematics(kintype_);

  // read EAS technology flag
  std::string eas = container.get<std::string>("EAS");

  // full EAS technology
  if (eas == "full")
  {
    eastype_ = soh8_easfull;
    neas_ = 21;  // number of eas parameters for full EAS
    soh8_easinit();
  }
  // mild EAS technology
  else if (eas == "mild")
  {
    eastype_ = soh8_easmild;
    neas_ = 9;  // number of eas parameters for mild EAS
    soh8_easinit();
  }
  // no EAS technology
  else if (eas == "none")
  {
    eastype_ = soh8_easnone;
    neas_ = 0;
  }
  else
    FOUR_C_THROW("Reading of SO_HEX8 EAS technology failed");

  return true;
}

FOUR_C_NAMESPACE_CLOSE
