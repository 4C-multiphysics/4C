// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_ale3.hpp"
#include "4C_mat_so3_material.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool Discret::Elements::Ale3::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  Core::FE::CellType shape = Core::FE::string_to_cell_type(distype);

  std::cout << " distype " << distype << std::endl;

  Core::FE::GaussRule3D gaussrule;

  switch (shape)
  {
    case Core::FE::CellType::hex8:
    {
      gaussrule = Core::FE::GaussRule3D::hex_8point;
      break;
    }
    case Core::FE::CellType::hex20:
    case Core::FE::CellType::hex27:
    {
      gaussrule = Core::FE::GaussRule3D::hex_27point;
      break;
    }
    case Core::FE::CellType::pyramid5:
    {
      gaussrule = Core::FE::GaussRule3D::pyramid_8point;
      break;
    }
    case Core::FE::CellType::tet4:
    {
      gaussrule = Core::FE::GaussRule3D::tet_1point;
      break;
    }
    case Core::FE::CellType::tet10:
    {
      gaussrule = Core::FE::GaussRule3D::tet_4point;
      break;
    }
    default:
      FOUR_C_THROW("Unknown distype %s for ALE3 element", distype.c_str());
      // just set to something to shutup compiler
      gaussrule = Core::FE::GaussRule3D::undefined;
      break;
  }  // end switch distype

  // set up of materials with GP data (e.g., history variables)
  std::shared_ptr<Mat::So3Material> so3mat =
      std::dynamic_pointer_cast<Mat::So3Material>(material());

  const Core::FE::IntegrationPoints3D intpoints(gaussrule);
  const int numgp = intpoints.nquad;
  so3mat->setup(numgp, container);

  return true;
}

FOUR_C_NAMESPACE_CLOSE
