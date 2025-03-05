// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_problemtype.hpp"

#include "4C_fem_general_shape_function_type.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Inpar::PROBLEMTYPE::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs type{"PROBLEM TYPE"};

  {
    using IntegerType = Core::ProblemType;
    Teuchos::Array<std::string> name;
    Teuchos::Array<IntegerType> label;

    for (const auto& [prb_name, prb_enum] : string_to_problem_type_map())
    {
      name.push_back(prb_name);
      label.push_back(prb_enum);
    }

    Core::Utils::string_to_integral_parameter<IntegerType>(
        "PROBLEMTYPE", "Fluid_Structure_Interaction", "", name, label, type);
  }

  {
    using IntegerType = Core::FE::ShapeFunctionType;
    Teuchos::Array<std::string> name;
    Teuchos::Array<IntegerType> label;

    for (const auto& [prb_name, prb_enum] : Core::FE::string_to_shape_function_type_map())
    {
      name.push_back(prb_name);
      label.push_back(prb_enum);
    }

    Core::Utils::string_to_integral_parameter<IntegerType>("SHAPEFCT", "Polynomial",
        "Defines the function spaces for the spatial approximation", name, label, type);
  }

  Core::Utils::int_parameter("RESTART", 0, "", type);
  Core::Utils::int_parameter("RANDSEED", -1, "Set the random seed. If < 0 use current time.", type);

  type.move_into_collection(list);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::map<std::string, Core::ProblemType> Inpar::PROBLEMTYPE::string_to_problem_type_map()
{
  static std::map<std::string, Core::ProblemType> string2prbtype;

  if (string2prbtype.size() == 0)
  {
    // problem types in alphabetical order
    string2prbtype["Ale"] = Core::ProblemType::ale;
    string2prbtype["ArterialNetwork"] = Core::ProblemType::art_net;
    string2prbtype["Biofilm_Fluid_Structure_Interaction"] = Core::ProblemType::biofilm_fsi;
    string2prbtype["Cardiac_Monodomain"] = Core::ProblemType::cardiac_monodomain;
    string2prbtype["Elastohydrodynamic_Lubrication"] = Core::ProblemType::ehl;
    string2prbtype["Electrochemistry"] = Core::ProblemType::elch;
    string2prbtype["Electromagnetics"] = Core::ProblemType::elemag;
    string2prbtype["Fluid"] = Core::ProblemType::fluid;
    string2prbtype["Fluid_Ale"] = Core::ProblemType::fluid_ale;
    string2prbtype["Fluid_Beam_Interaction"] = Core::ProblemType::fbi;
    string2prbtype["Fluid_Poro_Structure_Interaction_XFEM"] = Core::ProblemType::fpsi_xfem;
    string2prbtype["Fluid_Porous_Structure_Interaction"] = Core::ProblemType::fpsi;
    string2prbtype["Fluid_Porous_Structure_Scalar_Scalar_Interaction"] = Core::ProblemType::fps3i;
    string2prbtype["Fluid_RedModels"] = Core::ProblemType::fluid_redmodels;
    string2prbtype["Fluid_Structure_Interaction"] = Core::ProblemType::fsi;
    string2prbtype["Fluid_Structure_Interaction_RedModels"] = Core::ProblemType::fsi_redmodels;
    string2prbtype["Fluid_Structure_Interaction_XFEM"] = Core::ProblemType::fsi_xfem;
    string2prbtype["Fluid_XFEM"] = Core::ProblemType::fluid_xfem;
    string2prbtype["Fluid_XFEM_LevelSet"] = Core::ProblemType::fluid_xfem_ls;
    string2prbtype["Gas_Fluid_Structure_Interaction"] = Core::ProblemType::gas_fsi;
    string2prbtype["Level_Set"] = Core::ProblemType::level_set;
    string2prbtype["Low_Mach_Number_Flow"] = Core::ProblemType::loma;
    string2prbtype["Lubrication"] = Core::ProblemType::lubrication;
    string2prbtype["NP_Supporting_Procs"] = Core::ProblemType::np_support;
    string2prbtype["Particle"] = Core::ProblemType::particle;
    string2prbtype["Particle_Structure_Interaction"] = Core::ProblemType::pasi;
    string2prbtype["Polymer_Network"] = Core::ProblemType::polymernetwork;
    string2prbtype["Poroelastic_scalar_transport"] = Core::ProblemType::poroscatra;
    string2prbtype["Poroelasticity"] = Core::ProblemType::poroelast;
    string2prbtype["Multiphase_Poroelasticity"] = Core::ProblemType::poromultiphase;
    string2prbtype["Multiphase_Poroelasticity_ScaTra"] = Core::ProblemType::poromultiphasescatra;
    string2prbtype["Multiphase_Porous_Flow"] = Core::ProblemType::porofluidmultiphase;
    string2prbtype["RedAirways_Tissue"] = Core::ProblemType::redairways_tissue;
    string2prbtype["ReducedDimensionalAirWays"] = Core::ProblemType::red_airways;
    string2prbtype["Reduced_Lung"] = Core::ProblemType::reduced_lung;
    string2prbtype["Scalar_Thermo_Interaction"] = Core::ProblemType::sti;
    string2prbtype["Scalar_Transport"] = Core::ProblemType::scatra;
    string2prbtype["Structure"] = Core::ProblemType::structure;
    string2prbtype["Structure_Scalar_Interaction"] = Core::ProblemType::ssi;
    string2prbtype["Structure_Scalar_Thermo_Interaction"] = Core::ProblemType::ssti;
    string2prbtype["Thermo"] = Core::ProblemType::thermo;
    string2prbtype["Thermo_Structure_Interaction"] = Core::ProblemType::tsi;
    string2prbtype["Thermo_Fluid_Structure_Interaction"] = Core::ProblemType::thermo_fsi;
  }

  return string2prbtype;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::ProblemType Inpar::PROBLEMTYPE::string_to_problem_type(std::string name)
{
  std::map<std::string, Core::ProblemType> map = string_to_problem_type_map();
  std::map<std::string, Core::ProblemType>::const_iterator i = map.find(name);
  if (i != map.end()) return i->second;
  FOUR_C_THROW("unsupported problem name '%s'", name.c_str());

  return Core::ProblemType::none;
}

FOUR_C_NAMESPACE_CLOSE
