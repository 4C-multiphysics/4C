/*----------------------------------------------------------------------*/
/*! \file
\brief Solid Tet10 Element
\level 2
*----------------------------------------------------------------------*/

#include "4C_io_linedefinition.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_so3_tet10.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::ELEMENTS::SoTet10::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  solid_material()->setup(NUMGPT_SOTET10, container);

  std::string buffer = container.get<std::string>("KINEM");

  // geometrically linear
  if (buffer == "linear")
  {
    kintype_ = Inpar::Solid::KinemType::linear;
    FOUR_C_THROW("Reading of SO_TET10 element failed only nonlinear kinematics implemented");
  }
  // geometrically non-linear with Total Lagrangean approach
  else if (buffer == "nonlinear")
    kintype_ = Inpar::Solid::KinemType::nonlinearTotLag;
  // geometrically non-linear with Updated Lagrangean approach
  else
    FOUR_C_THROW("Reading of SO_TET10 element failed KINEM unknown");

  // check if material kinematics is compatible to element kinematics
  solid_material()->valid_kinematics(kintype_);

  return true;
}

FOUR_C_NAMESPACE_CLOSE
