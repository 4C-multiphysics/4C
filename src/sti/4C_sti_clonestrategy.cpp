// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_sti_clonestrategy.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_scatra_ele.hpp"

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::ScatraThermoCloneStrategy::check_material_type(const int matid)
{
  // check whether material with specified ID is compatible with cloned element or not
  switch (Global::Problem::instance()->materials()->parameter_by_id(matid)->type())
  {
    case Core::Materials::m_soret:
    case Core::Materials::m_thermo_fourier:
      // do nothing in case of compatible material
      break;

    default:
    {
      // throw error in case of incompatible material
      FOUR_C_THROW("Material with ID %d is not compatible with cloned transport element!", matid);
      break;
    }
  }
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
std::map<std::string, std::string> STI::ScatraThermoCloneStrategy::conditions_to_copy() const
{
  return {{"PointThermoCoupling", "PointCoupling"}, {"S2IKinetics", "S2IKinetics"},
      {"S2IMeshtying", "S2IMeshtying"}, {"ScaTraFluxCalc", "ScaTraFluxCalc"},
      {"ThermoDirichlet", "Dirichlet"}, {"ThermoPointNeumann", "PointNeumann"},
      {"ThermoLineNeumann", "LineNeumann"}, {"ThermoSurfaceNeumann", "SurfaceNeumann"},
      {"ThermoVolumeNeumann", "VolumeNeumann"}, {"ThermoInitfield", "Initfield"},
      {"ThermoRobin", "TransportRobin"}, {"ScatraPartitioning", "ScatraPartitioning"}};
}  // STI::ScatraThermoCloneStrategy::conditions_to_copy()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
bool STI::ScatraThermoCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // set type of cloned element to transport type
  eletype.emplace_back("TRANSP");

  // element should always be cloned
  return true;
}  // STI::ScatraThermoCloneStrategy::determine_ele_type

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::ScatraThermoCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbs)
{
  // cast pointers to current element on source discretization and to current cloned element on
  // target discretization
  auto* oldele_transport = dynamic_cast<Discret::Elements::Transport*>(oldele);
  std::shared_ptr<Discret::Elements::Transport> newele_transport =
      std::dynamic_pointer_cast<Discret::Elements::Transport>(newele);

  // safety check
  if (oldele_transport == nullptr or newele_transport == nullptr)
    FOUR_C_THROW("Expected transport element, but received element of type '%s'!",
        Core::Utils::get_dynamic_type_name(*newele).c_str());

  // provide cloned element with material
  newele_transport->set_material(matid, oldele);

  // provide cloned element with discretization type
  newele_transport->set_dis_type(oldele->shape());

  // provide cloned element with physical implementation type
  switch (oldele_transport->impl_type())
  {
    case Inpar::ScaTra::impltype_elch_diffcond_thermo:
    case Inpar::ScaTra::impltype_elch_diffcond:
    {
      newele_transport->set_impl_type(Inpar::ScaTra::impltype_thermo_elch_diffcond);
      break;
    }
    case Inpar::ScaTra::impltype_elch_electrode_thermo:
    case Inpar::ScaTra::impltype_elch_electrode:
    {
      newele_transport->set_impl_type(Inpar::ScaTra::impltype_thermo_elch_electrode);
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Scatra-thermo interaction not yet implemented for given element implementation type!");
      break;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
