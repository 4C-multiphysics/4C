// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_volmortar_utils.hpp"

#include "4C_coupling_volmortar.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element_center.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  assign material to discretization A                       vuong 09/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::Utils::DefaultMaterialStrategy::assign_material2_to1(
    const Coupling::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele1,
    const std::vector<int>& ids_2, std::shared_ptr<Core::FE::Discretization> dis1,
    std::shared_ptr<Core::FE::Discretization> dis2)
{
  if (ele1 == nullptr) FOUR_C_THROW("ERROR: Got nullptr pointer for AssignMaterial for element!");

  // if no corresponding element found -> leave
  if (ids_2.empty()) return;

  // default strategy: take material of element with closest center in reference coordinates
  Core::Elements::Element* ele2 = nullptr;
  double mindistance = 1e10;
  {
    std::vector<double> centercoords1 = Core::FE::element_center_refe_coords(*ele1);

    for (unsigned i = 0; i < ids_2.size(); ++i)
    {
      Core::Elements::Element* actele2 = dis2->g_element(ids_2[i]);
      std::vector<double> centercoords2 = Core::FE::element_center_refe_coords(*actele2);

      Core::LinAlg::Matrix<3, 1> diffcoords(Core::LinAlg::Initialization::zero);

      for (int j = 0; j < 3; ++j) diffcoords(j, 0) = centercoords1[j] - centercoords2[j];

      if (diffcoords.norm2() - mindistance < 1e-16)
      {
        mindistance = diffcoords.norm2();
        ele2 = actele2;
      }
    }
  }

  // assign additional material to element A
  ele1->add_material(ele2->material());

  // done
  return;
};

/*----------------------------------------------------------------------*
 |  assign material to discretization B                       vuong 09/14|
 *----------------------------------------------------------------------*/
void Coupling::VolMortar::Utils::DefaultMaterialStrategy::assign_material1_to2(
    const Coupling::VolMortar::VolMortarCoupl* volmortar, Core::Elements::Element* ele2,
    const std::vector<int>& ids_1, std::shared_ptr<Core::FE::Discretization> dis1,
    std::shared_ptr<Core::FE::Discretization> dis2)
{
  if (ele2 == nullptr) FOUR_C_THROW("ERROR: Got nullptr pointer for AssignMaterial for element!");

  // if no corresponding element found -> leave
  if (ids_1.empty()) return;

  // default strategy: take material of element with closest center in reference coordinates
  Core::Elements::Element* ele1 = nullptr;
  double mindistance = 1e10;
  {
    std::vector<double> centercoords2 = Core::FE::element_center_refe_coords(*ele2);

    for (unsigned i = 0; i < ids_1.size(); ++i)
    {
      Core::Elements::Element* actele1 = dis1->g_element(ids_1[i]);
      std::vector<double> centercoords1 = Core::FE::element_center_refe_coords(*actele1);

      Core::LinAlg::Matrix<3, 1> diffcoords(Core::LinAlg::Initialization::zero);

      for (int j = 0; j < 3; ++j) diffcoords(j, 0) = centercoords1[j] - centercoords2[j];

      if (diffcoords.norm2() - mindistance < 1e-16)
      {
        mindistance = diffcoords.norm2();
        ele1 = actele1;
      }
    }
  }

  // assign additional material to element B
  ele2->add_material(ele1->material());

  // done
  return;
};

FOUR_C_NAMESPACE_CLOSE
