// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_so3_material.hpp"
#include "4C_so3_sh8.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::Elements::SoSh8::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  solid_material()->setup(NUMGPT_SOH8, container);

  // read kinematic flag
  std::string kinem = container.get<std::string>("KINEM");

  if (kinem == "linear")
  {
    // kintype_ = soh8_linear;
    FOUR_C_THROW("Only nonlinear kinematics for SO_SH8 implemented!");
  }
  else if (kinem == "nonlinear")
  {
    kintype_ = Inpar::Solid::KinemType::nonlinearTotLag;
  }
  else
    FOUR_C_THROW("Reading SO_SH8 element failed KINEM unknown");

  // check if material kinematics is compatible to element kinematics
  solid_material()->valid_kinematics(kintype_);

  // read EAS technology flag
  std::string eas = container.get<std::string>("EAS");

  // full EAS technology
  if (eas == "sosh8")
  {
    eastype_ = soh8_eassosh8;
    neas_ = 7;  // number of eas parameters for EAS_SOSH8
    soh8_easinit();
  }
  // no EAS technology
  else if (eas == "none")
  {
    eastype_ = soh8_easnone;
    neas_ = 0;  // number of eas parameters for EAS_SOSH8
  }
  else
    FOUR_C_THROW("Reading of SO_SH8 EAS technology failed");

  // read ANS technology flag
  std::string ans = container.get<std::string>("ANS");

  if (ans == "sosh8")
  {
    anstype_ = anssosh8;
  }
  // no ANS technology
  else if (ans == "none")
  {
    anstype_ = ansnone;
  }
  else
    FOUR_C_THROW("Reading of SO_SH8 ANS technology failed");


  // read THICKDIR flag
  std::string thickdir = container.get<std::string>("THICKDIR");
  nodes_rearranged_ = false;

  // global X
  if (thickdir == "xdir") thickdir_ = globx;
  // global Y
  else if (thickdir == "ydir")
    thickdir_ = globy;
  // global Z
  else if (thickdir == "zdir")
    thickdir_ = globz;
  // find automatically through Jacobian of Xrefe
  else if (thickdir == "auto")
    thickdir_ = autoj;
  // local r
  else if (thickdir == "rdir")
    thickdir_ = enfor;
  // local s
  else if (thickdir == "sdir")
    thickdir_ = enfos;
  // local t
  else if (thickdir == "tdir")
    thickdir_ = enfot;
  // no noderearrangement
  else if (thickdir == "none")
  {
    thickdir_ = none;
    nodes_rearranged_ = true;
  }
  else
    FOUR_C_THROW("Reading of SO_SH8 thickness direction failed");

  return true;
}

FOUR_C_NAMESPACE_CLOSE
