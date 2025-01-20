// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_artery.hpp"
#include "4C_mat_cnst_1d_art.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Discret::Elements::Artery::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  int ngp = container.get<int>("GP");

  switch (ngp)
  {
    case 1:
      gaussrule_ = Core::FE::GaussRule1D::line_1point;
      break;
    case 2:
      gaussrule_ = Core::FE::GaussRule1D::line_2point;
      break;
    case 3:
      gaussrule_ = Core::FE::GaussRule1D::line_3point;
      break;
    case 4:
      gaussrule_ = Core::FE::GaussRule1D::line_4point;
      break;
    case 5:
      gaussrule_ = Core::FE::GaussRule1D::line_5point;
      break;
    case 6:
      gaussrule_ = Core::FE::GaussRule1D::line_6point;
      break;
    case 7:
      gaussrule_ = Core::FE::GaussRule1D::line_7point;
      break;
    case 8:
      gaussrule_ = Core::FE::GaussRule1D::line_8point;
      break;
    case 9:
      gaussrule_ = Core::FE::GaussRule1D::line_9point;
      break;
    case 10:
      gaussrule_ = Core::FE::GaussRule1D::line_10point;
      break;
    default:
      FOUR_C_THROW("Reading of ART element failed: Gaussrule for line not supported!\n");
  }

  // read artery implementation type
  std::string impltype = container.get<std::string>("TYPE");

  if (impltype == "Undefined")
    impltype_ = Inpar::ArtDyn::impltype_undefined;
  else if (impltype == "LinExp")
    impltype_ = Inpar::ArtDyn::impltype_lin_exp;
  else if (impltype == "PressureBased")
    impltype_ = Inpar::ArtDyn::impltype_pressure_based;
  else
    FOUR_C_THROW("Invalid implementation type for ARTERY elements!");

  // extract diameter
  double diam = container.get<double>("DIAM");

  // set diameter in material
  set_diam_in_material(diam);

  return true;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Discret::Elements::Artery::set_diam_in_material(const double diam)
{
  // now the element knows its material, and we can use it to set the diameter
  std::shared_ptr<Core::Mat::Material> mat = material();
  if (mat->material_type() == Core::Materials::m_cnst_art)
  {
    Mat::Cnst1dArt* arterymat = dynamic_cast<Mat::Cnst1dArt*>(mat.get());
    arterymat->set_diam(diam);
    arterymat->set_diam_initial(diam);
  }
  else
    FOUR_C_THROW("Artery element got unsupported material type %d", mat->material_type());
  return;
}

FOUR_C_NAMESPACE_CLOSE
