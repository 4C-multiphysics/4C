// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_clonestrategy.hpp"

#include "4C_adapter_structure_scatra_ele.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_ssi.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_scatra_ele.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> SSI::ScatraStructureCloneStrategy::conditions_to_copy() const
{
  return {{"TransportDirichlet", "Dirichlet"}, {"TransportPointNeumann", "PointNeumann"},
      {"TransportLineNeumann", "LineNeumann"}, {"TransportSurfaceNeumann", "SurfaceNeumann"},
      {"TransportVolumeNeumann", "VolumeNeumann"},
      {"TransportNeumannInflow", "TransportNeumannInflow"}, {"FSICoupling", "FSICoupling"},
      {"ScaTraCoupling", "ScaTraCoupling"}, {"ScaTraFluxCalc", "ScaTraFluxCalc"},
      {"Initfield", "Initfield"}, {"ssi_interface_meshtying", "S2IMeshtying"},
      {"SSIInterfaceContact", "S2INoEvaluation"}, {"S2IKinetics", "S2IKinetics"},
      {"ScatraPartitioning", "ScatraPartitioning"}, {"ElectrodeSOC", "ElectrodeSOC"},
      {"CellVoltage", "CellVoltage"}, {"CellVoltagePoint", "CellVoltagePoint"},
      {"TotalAndMeanScalar", "TotalAndMeanScalar"}, {"CCCVCycling", "CCCVCycling"},
      {"CCCVHalfCycle", "CCCVHalfCycle"},
      {"TransportThermoConvections", "TransportThermoConvections"},
      {"SSIMeshtying3DomainIntersection", "Meshtying3DomainIntersection"},
      {"SSISurfaceManifold", "SSISurfaceManifold"},
      {"SSISurfaceManifoldKinetics", "SSISurfaceManifoldKinetics"}};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<std::string, std::string> SSI::ScatraStructureCloneStrategyManifold::conditions_to_copy()
    const
{
  return {};
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Inpar::ScaTra::ImplType SSI::ScatraStructureCloneStrategy::get_impl_type(
    Core::Elements::Element* ele)
{
  return Adapter::get_sca_tra_impl_type(ele);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  Core::Materials::MaterialType mtype =
      Global::Problem::instance()->materials()->parameter_by_id(matid)->type();
  if ((mtype != Core::Materials::m_scatra) && (mtype != Core::Materials::m_elchmat) &&
      (mtype != Core::Materials::m_electrode) && (mtype != Core::Materials::m_matlist) &&
      (mtype != Core::Materials::m_matlist_reactions) && (mtype != Core::Materials::m_myocard) &&
      (mtype != Core::Materials::m_thermostvenant))
    FOUR_C_THROW("Material with ID %d is not admissible for scalar transport elements", matid);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategy::set_element_data(
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
    // set distype as well
    trans->set_dis_type(oldele->shape());

    // now check whether ImplType is reasonable and if set the ImplType
    Inpar::ScaTra::ImplType impltype = SSI::ScatraStructureCloneStrategy::get_impl_type(oldele);
    if (impltype == Inpar::ScaTra::impltype_undefined)
    {
      FOUR_C_THROW(
          "ScatraStructureCloneStrategy copies scatra discretization from structure "
          "discretization, but the STRUCTURE elements that are defined in the input file are "
          "either not meant to be copied to scatra elements or the ImplType is set 'Undefined' "
          "which is not meaningful for the created scatra discretization! Use SOLIDSCATRA, "
          "WALLSCATRA, SHELLSCATRA or TRUSS3SCATRA elements with meaningful ImplType instead!");
    }
    else
      trans->set_impl_type(impltype);

    // set material
    trans->set_material(matid, oldele);
  }
  else
  {
    FOUR_C_THROW(
        "unsupported element type '%s'", Core::Utils::get_dynamic_type_name(*newele).c_str());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void SSI::ScatraStructureCloneStrategyManifold::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbsdis)
{
  // determine impl type from manifold condition by identifying the condition for this element
  auto struct_dis = Global::Problem::instance()->get_dis("structure");

  std::vector<Core::Conditions::Condition*> conditions;
  struct_dis->get_condition("SSISurfaceManifold", conditions);

  auto impltype = Inpar::ScaTra::impltype_undefined;
  for (auto* condition : conditions)
  {
    auto cond_eles = condition->geometry();
    if (cond_eles.find(oldele->id()) != cond_eles.end())
    {
      impltype = condition->parameters().get<Inpar::ScaTra::ImplType>("ImplType");
      continue;
    }
  }

  if (impltype != Inpar::ScaTra::impltype_elch_electrode and
      impltype != Inpar::ScaTra::impltype_elch_diffcond and impltype != Inpar::ScaTra::impltype_std)
    FOUR_C_THROW("Scatra Impltype not supported for SSI with transport on manifolds");

  auto* trans = dynamic_cast<Discret::Elements::Transport*>(newele.get());
  if (trans == nullptr or oldele->element_type().name() != "StructuralSurfaceType")
    FOUR_C_THROW("element type not supported");

  // set distype
  trans->set_dis_type(oldele->shape());
  // set impltype
  trans->set_impl_type(impltype);
  // set material
  trans->set_material(matid, oldele);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool SSI::ScatraStructureCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // note: ismyele, actele remain unused here! Used only for ALE creation

  // we only support transport elements here
  eletype.emplace_back("TRANSP");

  return true;  // yes, we copy EVERY element (no submeshes)
}

FOUR_C_NAMESPACE_CLOSE
