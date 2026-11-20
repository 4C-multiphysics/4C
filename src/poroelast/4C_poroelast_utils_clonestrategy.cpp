// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_poroelast_utils_clonestrategy.hpp"

#include "4C_fluid_ele_poro.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_solid_poro_ele_pressure_based.hpp"
#include "4C_solid_poro_ele_pressure_velocity_based.hpp"
#include "4C_solid_poro_ele_pressure_velocity_based_p1.hpp"

FOUR_C_NAMESPACE_OPEN

std::map<std::string, std::string> PoroElast::Utils::PoroelastCloneStrategy::conditions_to_copy()
    const
{
  return {{"PoroDirichlet", "Dirichlet"}, {"PoroPointNeumann", "PointNeumann"},
      {"PoroLineNeumann", "LineNeumann"}, {"PoroSurfaceNeumann", "SurfaceNeumann"},
      {"PoroVolumeNeumann", "VolumeNeumann"}, {"no_penetration", "no_penetration"},
      {"PoroPartInt", "PoroPartInt"}, {"PoroCoupling", "PoroCoupling"},
      {"FSICoupling", "FSICoupling"}, {"fpsi_coupling", "fpsi_coupling"},
      {"PoroPresInt", "PoroPresInt"}, {"Mortar", "Mortar"}, {"SurfFlowRate", "SurfFlowRate"},
      {"LineFlowRate", "LineFlowRate"}, {"XFEMSurfFPIMono", "XFEMSurfFPIMono"},
      {"FluidNeumannInflow", "FluidNeumannInflow"}};
}

void PoroElast::Utils::PoroelastCloneStrategy::check_material_type(const int matid)
{
  // We take the material with the ID specified by the user
  // Here we check first, whether this material is of admissible type
  Core::Materials::MaterialType mtype =
      Global::Problem::instance()->materials()->parameter_by_id(matid)->type();
  if ((mtype != Core::Materials::m_fluidporo))
    FOUR_C_THROW("Material with ID {} is not admissible for fluid poroelasticity elements", matid);
}

void PoroElast::Utils::PoroelastCloneStrategy::set_element_data(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele,
    const int matid, const bool isnurbs)
{
  // We need to set material and possibly other things to complete element setup.
  // This is again really ugly as we have to extract the actual
  // element type in order to access the material property

  std::shared_ptr<Discret::Elements::FluidPoro> fluid =
      std::dynamic_pointer_cast<Discret::Elements::FluidPoro>(newele);
  if (fluid != nullptr)
  {
    fluid->set_material(0, Mat::factory(matid));
    // Copy Initial Porosity from StructPoro Material to FluidPoro Material
    int eleGID = oldele->id();
    auto struct_mat = std::static_pointer_cast<Mat::StructPoro>(oldele->material());
    Core::IO::InputField<double> init_porosity_{struct_mat->init_porosity(eleGID)};

    static_cast<Mat::PAR::FluidPoro*>(fluid->material()->parameter())
        ->set_initial_porosity(init_porosity_, eleGID);
    fluid->set_dis_type(oldele->shape());  // set distype as well!
    fluid->set_is_ale(true);

    if (auto* solid_poro_pressure_velocity_based =
            dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBased<2>*>(oldele);
        solid_poro_pressure_velocity_based)
    {
      fluid->set_kinematic_type(solid_poro_pressure_velocity_based->kinematic_type());
    }
    else if (auto* solid_poro_pressure_velocity_based =
                 dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBased<3>*>(oldele);
        solid_poro_pressure_velocity_based)
    {
      fluid->set_kinematic_type(solid_poro_pressure_velocity_based->kinematic_type());
    }
    else if (auto* solid_poro_pressure_velocity_based_p1 =
                 dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBasedP1<2>*>(oldele);
        solid_poro_pressure_velocity_based_p1)
    {
      fluid->set_kinematic_type(solid_poro_pressure_velocity_based_p1->kinematic_type());
    }
    else if (auto* solid_poro_pressure_velocity_based_p1 =
                 dynamic_cast<Discret::Elements::SolidPoroPressureVelocityBasedP1<3>*>(oldele);
        solid_poro_pressure_velocity_based_p1)
    {
      fluid->set_kinematic_type(solid_poro_pressure_velocity_based_p1->kinematic_type());
    }
    else
      FOUR_C_THROW(
          " dynamic cast from Core::Elements::Element* to Discret::Elements::So_base* or "
          "Discret::Elements::SolidPoroPressureVelocityBased failed ");

    set_anisotropic_permeability_directions_onto_fluid(newele, oldele);
    set_anisotropic_permeability_nodal_coeffs_onto_fluid(newele, oldele);
  }
  else
  {
    FOUR_C_THROW("unsupported element type '{}'", Core::Utils::get_dynamic_type_name(*newele));
  }
}

void PoroElast::Utils::PoroelastCloneStrategy::set_anisotropic_permeability_directions_onto_fluid(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele)
{
  std::shared_ptr<Discret::Elements::FluidPoro> fluid =
      std::dynamic_pointer_cast<Discret::Elements::FluidPoro>(newele);

  if (const auto* const solid_poro_ele =
          dynamic_cast<const Discret::Elements::SolidPoroPressureVelocityBased<2>* const>(oldele))
  {
    fluid->set_anisotropic_permeability_directions(
        solid_poro_ele->get_anisotropic_permeability_directions());
  }
  else if (const auto* const solid_poro_ele =
               dynamic_cast<const Discret::Elements::SolidPoroPressureVelocityBased<3>* const>(
                   oldele))
  {
    fluid->set_anisotropic_permeability_directions(
        solid_poro_ele->get_anisotropic_permeability_directions());
  }

  // Anisotropic permeability not yet supported for p1 type elements. Do nothing.
}

void PoroElast::Utils::PoroelastCloneStrategy::set_anisotropic_permeability_nodal_coeffs_onto_fluid(
    std::shared_ptr<Core::Elements::Element> newele, Core::Elements::Element* oldele)
{
  std::shared_ptr<Discret::Elements::FluidPoro> fluid =
      std::dynamic_pointer_cast<Discret::Elements::FluidPoro>(newele);

  if (const auto* const solid_poro_ele =
          dynamic_cast<const Discret::Elements::SolidPoroPressureVelocityBased<2>* const>(oldele))
  {
    fluid->set_anisotropic_permeability_nodal_coeffs(
        solid_poro_ele->get_anisotropic_permeability_nodal_coeffs());
  }
  else if (const auto* const solid_poro_ele =
               dynamic_cast<const Discret::Elements::SolidPoroPressureVelocityBased<3>* const>(
                   oldele))
  {
    fluid->set_anisotropic_permeability_nodal_coeffs(
        solid_poro_ele->get_anisotropic_permeability_nodal_coeffs());
  }

  // Nodal anisotropic permeability not yet supported for higher order or p1 elements.
  // Do nothing.
}

bool PoroElast::Utils::PoroelastCloneStrategy::determine_ele_type(
    Core::Elements::Element* actele, const bool ismyele, std::vector<std::string>& eletype)
{
  // clone the element only if it is a poro element (we support submeshes here)
  if (is_poro_element(actele))
  {
    // we only support fluid elements here
    eletype.emplace_back("FLUIDPORO");
    return true;
  }

  return false;
}

FOUR_C_NAMESPACE_CLOSE
