// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_clonestrategy.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_porofluid_pressure_based_ele.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::function<void(int)>& material_validation_callback()
  {
    static std::function<void(int)> callback;
    return callback;
  }
}  // namespace


/*----------------------------------------------------------------------*
 | define conditions to copy to the cloned discretization    vuong 08/16 |
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> PoroPressureBased::PorofluidCloneStrategy::conditions_to_copy()
{
  return {{"PoroDirichlet", "Dirichlet"}, {"PoroPointNeumann", "PointNeumann"},
      {"PoroLineNeumann", "LineNeumann"}, {"PoroSurfaceNeumann", "SurfaceNeumann"},
      {"PoroVolumeNeumann", "VolumeNeumann"}, {"Initfield", "Initfield"},
      {"ArtPorofluidCouplConNodebased", "ArtPorofluidCouplConNodebased"},
      {"ArtPorofluidCouplConNodeToPoint", "ArtPorofluidCouplConNodeToPoint"}};
}


void PoroPressureBased::PorofluidCloneStrategy::set_material_validation_callback(
    std::function<void(int)> callback)
{
  material_validation_callback() = std::move(callback);
}


/*----------------------------------------------------------------------*
 | check for correct material                               vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidCloneStrategy::check_material_type(const int matid)
{
  FOUR_C_ASSERT_ALWAYS(
      material_validation_callback(), "Material validation callback is not initialized.");
  material_validation_callback()(matid);
}


/*----------------------------------------------------------------------*
 | set element-specific data (material etc.)                 vuong 08/16 |
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again ugly as we have to extract the actual element type to access the material
  // property.

  // note: set_material() was reimplemented by the PoroFluidMultiPhase element!
  auto* porofluidele = dynamic_cast<Discret::Elements::PoroFluidMultiPhase*>(newele.get());
  if (porofluidele != nullptr)
  {
    porofluidele->set_material(0, Mat::factory(matid));
    porofluidele->set_dis_type(oldele->shape());  // set distype as well!
  }
  else
  {
    FOUR_C_THROW("unsupported element type '{}'", Core::Utils::get_dynamic_type_name(*newele));
  }
}


/*----------------------------------------------------------------------*
 | determine whether element is copied or not               vuong 08/16 |
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PorofluidCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // note: ismyele, actele remain unused here! Used only for ALE creation

  // we only support POROFLUIDMULTIPHASE elements here
  eletype.emplace_back("POROFLUIDMULTIPHASE");

  // all elements are copied
  return true;
}

FOUR_C_NAMESPACE_CLOSE
