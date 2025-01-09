// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_utils_clonestrategy.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_scatra_ele.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::map<std::string, std::string> ScaTra::ScatraFluidCloneStrategy::conditions_to_copy() const
{
  return {{"TransportDirichlet", "Dirichlet"}, {"TransportPointNeumann", "PointNeumann"},
      {"TransportLineNeumann", "LineNeumann"}, {"TransportSurfaceNeumann", "SurfaceNeumann"},
      {"TransportVolumeNeumann", "VolumeNeumann"},
      {"TransportNeumannInflow", "TransportNeumannInflow"},
      {"TaylorGalerkinOutflow", "TaylorGalerkinOutflow"},
      {"TaylorGalerkinNeumannInflow", "TaylorGalerkinNeumannInflow"},
      {"ReinitializationTaylorGalerkin", "ReinitializationTaylorGalerkin"},
      {"LinePeriodic", "LinePeriodic"}, {"SurfacePeriodic", "SurfacePeriodic"},
      {"TurbulentInflowSection", "TurbulentInflowSection"}, {"LineNeumann", "FluidLineNeumann"},
      {"SurfaceNeumann", "FluidSurfaceNeumann"}, {"VolumeNeumann", "FluidVolumeNeumann"},
      {"KrylovSpaceProjection", "KrylovSpaceProjection"},
      {"ElchBoundaryKinetics", "ElchBoundaryKinetics"}, {"ScaTraFluxCalc", "ScaTraFluxCalc"},
      {"Initfield", "Initfield"}, {"TransportRobin", "TransportRobin"},
      {"FSICoupling", "FSICoupling"}, {"Mortar", "Mortar"}, {"ScaTraCoupling", "ScaTraCoupling"},
      {"LsContact", "LsContact"}, {"SPRboundary", "SPRboundary"},
      {"XFEMLevelsetTwophase", "XFEMLevelsetTwophase"},
      {"XFEMLevelsetCombustion", "XFEMLevelsetCombustion"}};
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScatraFluidCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  Core::Materials::MaterialType mtype =
      Global::Problem::instance()->materials()->parameter_by_id(matid)->type();
  if ((mtype != Core::Materials::m_scatra) && (mtype != Core::Materials::m_sutherland) &&
      (mtype != Core::Materials::m_ion) && (mtype != Core::Materials::m_thermo_fourier) &&
      (mtype != Core::Materials::m_thermostvenant) && (mtype != Core::Materials::m_matlist) &&
      (mtype != Core::Materials::m_matlist_reactions) && (mtype != Core::Materials::m_myocard) &&
      (mtype != Core::Materials::m_scatra_multiporo_fluid) &&
      (mtype != Core::Materials::m_scatra_multiporo_volfrac) &&
      (mtype != Core::Materials::m_scatra_multiporo_solid) &&
      (mtype != Core::Materials::m_scatra_multiporo_temperature))
    FOUR_C_THROW("Material with ID %d is not admissible for scalar transport elements", matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScatraFluidCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // note: set_material() was reimplemented by the transport element!
  auto* trans = dynamic_cast<Discret::Elements::Transport*>(newele.get());
  if (trans != nullptr)
  {
    trans->set_material(matid, oldele);
    trans->set_dis_type(oldele->shape());
  }
  else
  {
    FOUR_C_THROW(
        "unsupported element type '%s'", Core::Utils::get_dynamic_type_name(*newele).c_str());
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool ScaTra::ScatraFluidCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // note: ismyele, actele remain unused here! Used only for ALE creation

  // we only support transport elements here
  eletype.push_back("TRANSP");

  return true;  // yes, we copy EVERY element (no submeshes)
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::map<std::string, std::string> ScaTra::ScatraReactionCloneStrategy::conditions_to_copy() const
{
  return {{"TransportDirichlet", "Dirichlet"}, {"TransportPointNeumann", "PointNeumann"},
      {"TransportLineNeumann", "LineNeumann"}, {"TransportSurfaceNeumann", "SurfaceNeumann"},
      {"TransportVolumeNeumann", "VolumeNeumann"},
      {"KrylovSpaceProjection", "KrylovSpaceProjection"}, {"ScaTraFluxCalc", "ScaTraFluxCalc"},
      {"Initfield", "Initfield"}, {"ScaTraCoupling", "ScaTraCoupling"},
      {"SSICoupling", "SSICoupling"}, {"SSICouplingScatraToSolid", "SSICouplingScatraToSolid"},
      {"SSICouplingSolidToScatra", "SSICouplingSolidToScatra"},
      {"ScatraHeteroReactionMaster", "ScatraHeteroReactionMaster"},
      {"ScatraHeteroReactionSlave", "ScatraHeteroReactionSlave"}};
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScatraReactionCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  Core::Materials::MaterialType mtype =
      Global::Problem::instance()->materials()->parameter_by_id(matid)->type();
  if ((mtype != Core::Materials::m_scatra) && (mtype != Core::Materials::m_matlist) &&
      (mtype != Core::Materials::m_matlist_reactions))
    FOUR_C_THROW("Material with ID %d is not admissible for scalar transport elements", matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScatraReactionCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbsdis)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  // note: set_material() was reimplemented by the transport element!
  Discret::Elements::Transport* trans = dynamic_cast<Discret::Elements::Transport*>(newele.get());
  if (trans != nullptr)
  {
    trans->set_material(matid, oldele);
    trans->set_dis_type(oldele->shape());  // set distype as well!
  }
  else
  {
    FOUR_C_THROW(
        "unsupported element type '%s'", Core::Utils::get_dynamic_type_name(*newele).c_str());
  }
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool ScaTra::ScatraReactionCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // note: ismyele, actele remain unused here! Used only for ALE creation

  // we only support transport elements here
  eletype.push_back("TRANSP");

  return true;  // yes, we copy EVERY element (no submeshes)
}

FOUR_C_NAMESPACE_CLOSE
