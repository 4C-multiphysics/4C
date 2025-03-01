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
#include "4C_so3_hex27.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_nurbs27.hpp"
#include "4C_so3_poro_p1_scatra.hpp"
#include "4C_so3_poro_scatra.hpp"
#include "4C_so3_tet10.hpp"
#include "4C_so3_tet4.hpp"
#include "4C_solid_poro_3D_ele_pressure_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based.hpp"
#include "4C_solid_poro_3D_ele_pressure_velocity_based_p1.hpp"
#include "4C_w1_poro_p1_scatra.hpp"
#include "4C_w1_poro_scatra.hpp"

FOUR_C_NAMESPACE_OPEN


Inpar::ScaTra::ImplType PoroElastScaTra::Utils::PoroScatraCloneStrategy::get_impl_type(
    Core::Elements::Element* ele  //! element whose ScaTra::ImplType shall be determined
)
{
  // the element type name, needed to cast correctly in the following
  const std::string eletypename = ele->element_type().name();

  // TET 4 Elements
  // tet 4 solid poro scatra
  if (eletypename == "So_tet4PoroScatraType")
  {
    return (
        dynamic_cast<
            Discret::Elements::So3PoroScatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>*>(
            ele))
        ->impl_type();
  }
  // tet4 solid porop1 scatra
  else if (eletypename == "So_tet4PoroP1ScatraType")
  {
    return (dynamic_cast<Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoTet4,
                Core::FE::CellType::tet4>*>(ele))
        ->impl_type();
  }
  // tet 10 solid poro scatra
  else if (eletypename == "So_tet10PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::So3PoroScatra<Discret::Elements::SoTet10,
                Core::FE::CellType::tet10>*>(ele))
        ->impl_type();
  }
  // HEX 8 Elements
  // hex8 solid poro scatra
  else if (eletypename == "So_hex8PoroScatraType")
  {
    return (
        dynamic_cast<
            Discret::Elements::So3PoroScatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>*>(
            ele))
        ->impl_type();
  }
  // hex8 solid porop1 scatra
  else if (eletypename == "So_hex8PoroP1ScatraType")
  {
    return (dynamic_cast<Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoHex8,
                Core::FE::CellType::hex8>*>(ele))
        ->impl_type();
  }
  // hex27 solid poro scatra
  else if (eletypename == "So_hex27PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::So3PoroScatra<Discret::Elements::SoHex27,
                Core::FE::CellType::hex27>*>(ele))
        ->impl_type();
  }
  // nurbs 27
  else if (eletypename == "So_nurbs27PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::So3PoroScatra<Discret::Elements::Nurbs::SoNurbs27,
                Core::FE::CellType::nurbs27>*>(ele))
        ->impl_type();
  }
  // Solidporo
  else if (eletypename == "SolidPoroPressureBasedType")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureBased*>(ele))->get_impl_type();
  }
  else if (eletypename == "SolidPoroPressureVelocityBasedType")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBased*>(ele))->get_impl_type();
  }
  else if (eletypename == "SolidPoroPressureVelocityBasedP1Type")
  {
    return (dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBasedP1*>(ele))
        ->get_impl_type();
  }
  // wall poro scatra elements
  // quad 4
  else if (eletypename == "WallQuad4PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroScatra<Core::FE::CellType::quad4>*>(ele))
        ->impl_type();
  }
  // quad 9
  else if (eletypename == "WallQuad9PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroScatra<Core::FE::CellType::quad9>*>(ele))
        ->impl_type();
  }
  // nurbs 4
  else if (eletypename == "WallNurbs4PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroScatra<Core::FE::CellType::nurbs4>*>(ele))
        ->impl_type();
  }
  // nurbs 9
  else if (eletypename == "WallNurbs9PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroScatra<Core::FE::CellType::nurbs9>*>(ele))
        ->impl_type();
  }
  // tri 3
  else if (eletypename == "WallTri3PoroScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroScatra<Core::FE::CellType::tri3>*>(ele))
        ->impl_type();
  }
  // wall poro p1 elements
  // quad 4
  else if (eletypename == "WallQuad4PoroP1ScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad4>*>(ele))
        ->impl_type();
  }
  // quad 9
  else if (eletypename == "WallQuad9PoroP1ScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad9>*>(ele))
        ->impl_type();
  }
  // tri 3
  else if (eletypename == "WallTri3PoroP1ScatraType")
  {
    return (dynamic_cast<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::tri3>*>(ele))
        ->impl_type();
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
    FOUR_C_THROW(
        "unsupported element type '%s'", Core::Utils::get_dynamic_type_name(*newele).c_str());


  // set material
  trans->set_material(matid, oldele);
  // set distype as well!
  trans->set_dis_type(oldele->shape());

  // now check whether ImplType is reasonable and if set the ImplType
  Inpar::ScaTra::ImplType impltype =
      PoroElastScaTra::Utils::PoroScatraCloneStrategy::get_impl_type(oldele);

  if (impltype == Inpar::ScaTra::impltype_undefined)
    FOUR_C_THROW(
        "PoroScatraCloneStrategy copies scatra discretization from structure discretization, but "
        "the STRUCTURE elements that are defined in the input file are either not meant to be "
        "copied "
        "to scatra elements or the ImplType is set 'Undefined' which is not meaningful for the "
        "created scatra discretization! "
        "Use SOLIDSCATRA, WALLSCATRA, SHELLSCATRA, SOLIDPOROSCATRA, SOLIDPOROP1SCATRA, "
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
      (mtype != Core::Materials::m_thermostvenant))
    FOUR_C_THROW("Material with ID %d is not admissible for scalar transport elements", matid);
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
