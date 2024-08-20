/*----------------------------------------------------------------------*/
/*! \file
\brief tri-quadratic displacement based solid element
\level 1

*----------------------------------------------------------------------*/

#include "4C_io_linedefinition.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_so3_hex27.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::ELEMENTS::SoHex27::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  Teuchos::RCP<Core::Mat::Material> mat = material();

  solid_material()->setup(NUMGPT_SOH27, container);

  std::string buffer = container.get<std::string>("KINEM");

  if (buffer == "linear")
  {
    kintype_ = Inpar::Solid::KinemType::linear;
  }
  else if (buffer == "nonlinear")
  {
    kintype_ = Inpar::Solid::KinemType::nonlinearTotLag;
  }
  else
    FOUR_C_THROW("Reading SO_HEX27 element failed KINEM unknown");

  // check if material kinematics is compatible to element kinematics
  solid_material()->valid_kinematics(kintype_);

  // Validate that materials doesn't use extended update call.
  if (solid_material()->uses_extended_update())
    FOUR_C_THROW("This element currently does not support the extended update call.");

  return true;
}

FOUR_C_NAMESPACE_CLOSE
