// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_scatra_utils_clonestrategy.hpp"

#include "4C_adapter_structure_scatra_ele.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_poroelast_scatra_utils.hpp"
#include "4C_poroelast_utils_clonestrategy.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_solid_poro_ele_pressure_based.hpp"
#include "4C_solid_poro_ele_pressure_velocity_based.hpp"
#include "4C_solid_poro_ele_pressure_velocity_based_p1.hpp"

FOUR_C_NAMESPACE_OPEN


ScaTra::ImplType PoroElastScaTra::Utils::PoroScatraCloneStrategy::get_impl_type(
    Core::Elements::Element* ele  //! element whose ScaTra::ImplType shall be determined
)
{
  // the element type name, needed to cast correctly in the following
  const std::string& eletypename = ele->element_type().name();

  // Solidporo
  if (eletypename == "SolidPoroPressureBasedType<2>")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureBased<2>*>(ele))->get_impl_type();
  }
  else if (eletypename == "SolidPoroPressureBasedType<3>")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureBased<3>*>(ele))->get_impl_type();
  }
  else if (eletypename == "SolidPoroPressureVelocityBasedType<2>")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBased<2>*>(ele))
        ->get_impl_type();
  }
  else if (eletypename == "SolidPoroPressureVelocityBasedType<3>")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBased<3>*>(ele))
        ->get_impl_type();
  }
  else if (eletypename == "SolidPoroPressureVelocityBasedP1Type<2>")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBasedP1<2>*>(ele))
        ->get_impl_type();
  }
  else if (eletypename == "SolidPoroPressureVelocityBasedP1Type<3>")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBasedP1<3>*>(ele))
        ->get_impl_type();
  }
  // call base class routine
  else
    return Adapter::get_sca_tra_impl_type(ele);
}

bool PoroElastScaTra::Utils::PoroScatraCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // clone elements based on poro or scatra elements
  if (PoroElastScaTra::Utils::is_poro_scatra_element(actele) or
      PoroElast::Utils::is_poro_element(actele))
  {
    // we only support fluid elements here
    eletype.emplace_back("TRANSP");
    return true;
  }

  return false;
}


void PoroElastScaTra::Utils::PoroScatraCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // note: set_material() was reimplemented by the transport element!
  auto* trans = dynamic_cast<Discret::Elements::Transport*>(newele.get());
  if (trans == nullptr)
    FOUR_C_THROW("unsupported element type '{}'", Core::Utils::get_dynamic_type_name(*newele));


  // set material
  trans->set_material(matid, oldele);
  // set distype as well!
  trans->set_dis_type(oldele->shape());

  // now check whether ImplType is reasonable and if set the ImplType
  ScaTra::ImplType impltype =
      PoroElastScaTra::Utils::PoroScatraCloneStrategy::get_impl_type(oldele);

  if (impltype == ScaTra::impltype_undefined)
    FOUR_C_THROW(
        "PoroScatraCloneStrategy copies scatra discretization from structure discretization, but "
        "the STRUCTURE elements that are defined in the input file are either not meant to be "
        "copied "
        "to scatra elements or the ImplType is set 'Undefined' which is not meaningful for the "
        "created scatra discretization! "
        "Use SOLIDSCATRA, SHELLSCATRA, SOLIDPOROSCATRA, SOLIDPOROP1SCATRA, "
        "SOLIDPORO_PRESSURE_BASED, WALLPOROSCATRA or WALLPOROP1SCATRA Elements with meaningful "
        "ImplType instead!");

  trans->set_impl_type(impltype);
}

std::map<std::string, std::string>
PoroElastScaTra::Utils::PoroScatraCloneStrategy::conditions_to_copy() const
{
  return {{"TransportDirichlet", "Dirichlet"}, {"TransportPointNeumann", "PointNeumann"},
      {"TransportLineNeumann", "LineNeumann"}, {"TransportSurfaceNeumann", "SurfaceNeumann"},
      {"TransportVolumeNeumann", "VolumeNeumann"},
      {"TransportNeumannInflow", "TransportNeumannInflow"}, {"FSICoupling", "FSICoupling"},
      {"ScaTraCoupling", "ScaTraCoupling"}, {"ScaTraFluxCalc", "ScaTraFluxCalc"},
      {"Initfield", "Initfield"}, {"PoroCoupling", "PoroCoupling"}, {"Initfield", "Initfield"},
      {"ArtScatraCouplConNodebased", "ArtScatraCouplConNodebased"},
      {"PoroMultiphaseScatraOxyPartPressCalcCond", "PoroMultiphaseScatraOxyPartPressCalcCond"},
      {"TransportRobin", "TransportRobin"},
      {"ArtScatraCouplConNodeToPoint", "ArtScatraCouplConNodeToPoint"}};
}

void PoroElastScaTra::Utils::PoroScatraCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  Core::Materials::MaterialType mtype =
      Global::Problem::instance()->materials()->parameter_by_id(matid)->type();
  if ((mtype != Core::Materials::m_scatra) && (mtype != Core::Materials::m_elchmat) &&
      (mtype != Core::Materials::m_electrode) && (mtype != Core::Materials::m_matlist) &&
      (mtype != Core::Materials::m_matlist_reactions) && (mtype != Core::Materials::m_myocard) &&
      (mtype != Core::Materials::m_thermo_fourier))
    FOUR_C_THROW("Material with ID {} is not admissible for scalar transport elements", matid);
}

bool PoroElastScaTra::Utils::PoroelastCloneStrategyforScatraElements::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // clone the element only if it is a poro or scatra element (we support submeshes here)
  if (PoroElastScaTra::Utils::is_poro_scatra_element(actele) or
      PoroElast::Utils::is_poro_element(actele))
  {
    // we only support fluid elements here
    eletype.emplace_back("FLUIDPORO");
    return true;
  }

  return false;
}
FOUR_C_NAMESPACE_CLOSE
