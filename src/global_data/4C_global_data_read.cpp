// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config_revision.hpp"

#include "4C_global_data_read.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_contact_constitutivelaw_bundle.hpp"
#include "4C_contact_constitutivelaw_valid_laws.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_dofset_independent.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_global_legacy_module_validconditions.hpp"
#include "4C_global_legacy_module_validmaterials.hpp"
#include "4C_global_legacy_module_validparameters.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_exodus.hpp"
#include "4C_io_input_field.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_meshreader.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_mat_elchphase.hpp"
#include "4C_mat_micromaterial.hpp"
#include "4C_mat_newman_multiscale.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_scatra_multiscale.hpp"
#include "4C_particle_engine_particlereader.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_xfem_discretization.hpp"
#include "4C_xfem_discretization_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  /**
   * Gather all known section specs.
   */
  std::vector<Core::IO::InputSpec> gather_all_section_specs()
  {
    using namespace Core::IO::InputSpecBuilders;
    std::vector<Core::IO::InputSpec> section_specs;
    section_specs.push_back(list("CONTACT CONSTITUTIVE LAWS",
        {CONTACT::CONSTITUTIVELAW::valid_contact_constitutive_laws()}, {.required = false}));
    section_specs.push_back(list(
        "CLONING MATERIAL MAP", {Core::FE::valid_cloning_material_map()}, {.required = false}));
    section_specs.push_back(list("RESULT DESCRIPTION",
        {global_legacy_module_callbacks().valid_result_description_lines()}, {.required = false}));

    {
      std::vector<Core::IO::InputSpec> possible_materials;
      std::vector<Core::Materials::MaterialType> material_type;
      {
        auto materials = global_legacy_module_callbacks().materials();
        for (auto&& [type, spec] : materials)
        {
          possible_materials.emplace_back(std::move(spec));
          material_type.push_back(type);
        }
      }

      auto all_materials = all_of({
          parameter<int>("MAT"),
          one_of(possible_materials,
              store_index_as<Core::Materials::MaterialType>("_material_type", material_type)),
      });
      section_specs.push_back(list("MATERIALS", {all_materials}, {.required = false}));
    }

    {
      Core::Utils::FunctionManager functionmanager;
      global_legacy_module_callbacks().AttachFunctionDefinitions(functionmanager);

      auto valid_functions = functionmanager.valid_function_lines();

      // The FUNCT sections are special and do not fit into the usual pattern of sections and the
      // capabilities of InputSpec. The special FUNCT<n> section is not supposed to be entered by
      // users but we use this information inside the input file. Not pretty, but it works.
      // TODO remove this hack by restructuring the input of functions.
      section_specs.push_back(list("FUNCT<n>", valid_functions, {.required = false}));
    }

    {
      auto valid_conditions = global_legacy_module_callbacks().conditions();
      for (const auto& cond : valid_conditions)
      {
        section_specs.emplace_back(cond.spec());
      }
    }

    {
      auto global_specs = Global::valid_parameters();
      section_specs.insert(section_specs.end(), std::make_move_iterator(global_specs.begin()),
          std::make_move_iterator(global_specs.end()));
    }
    return section_specs;
  }
}  // namespace

Core::IO::InputFile Global::set_up_input_file(MPI_Comm comm)
{
  std::vector<Core::IO::InputSpec> valid_sections = gather_all_section_specs();

  std::vector<std::string> legacy_section_names{
      // elements
      "STRUCTURE ELEMENTS",
      "FLUID ELEMENTS",
      "LUBRICATION ELEMENTS",
      "TRANSPORT ELEMENTS",
      "TRANSPORT2 ELEMENTS",
      "ALE ELEMENTS",
      "THERMO ELEMENTS",
      "ARTERY ELEMENTS",
      "REDUCED D AIRWAYS ELEMENTS",
      "PARTICLES",
      "PERIODIC BOUNDINGBOX ELEMENTS",
      // general geometry
      "NODE COORDS",
      "DNODE-NODE TOPOLOGY",
      "DLINE-NODE TOPOLOGY",
      "DSURF-NODE TOPOLOGY",
      "DVOL-NODE TOPOLOGY",
  };

  return Core::IO::InputFile{valid_sections, std::move(legacy_section_names), comm};
}

void Global::emit_general_metadata(Core::IO::YamlNodeRef node)
{
  auto& root = node.node;

  // Basic information.
  {
    auto metadata = root["metadata"];
    metadata |= ryml::MAP;
    metadata["commit_hash"] << VersionControl::git_hash;
    metadata["version"] << FOUR_C_VERSION_FULL;
    metadata["description_section_name"] = Core::IO::InputFile::description_section_name;
  }

  // Element types.
  {
    Core::Elements::ElementDefinition element_definition;

    auto legacy_element_specs = root["legacy_element_specs"];
    legacy_element_specs |= ryml::MAP;

    for (const auto& [element_type, cell_specs] : element_definition.definitions)
    {
      auto element_specs = legacy_element_specs.append_child();
      element_specs << ryml::key(element_type);

      element_specs |= ryml::SEQ;
      for (const auto& [cell_type, spec] : cell_specs)
      {
        auto cell_spec = element_specs.append_child();
        cell_spec |= ryml::MAP;
        cell_spec["cell_type"] << ryml::to_csubstr(Core::FE::cell_type_to_string(cell_type));
        auto spec_node = cell_spec["spec"];
        spec.emit_metadata(Core::IO::YamlNodeRef(spec_node, ""));
      }
    }
  }

  // Particle types.
  {
    auto legacy_particle_spec = root["legacy_particle_specs"];
    legacy_particle_spec |= ryml::MAP;

    Core::IO::YamlNodeRef spec_emitter{legacy_particle_spec, ""};
    Particle::create_particle_spec().emit_metadata(spec_emitter);
  }

  // Cell types.
  {
    auto cell_types = root["cell_types"];
    cell_types |= ryml::MAP;

    for (const auto cell_type : EnumTools::enum_values<Core::FE::CellType>())
    {
      if (cell_type != Core::FE::CellType::dis_none && cell_type != Core::FE::CellType::max_distype)
      {
        auto cell_type_node = cell_types.append_child();
        auto cell_type_str = Core::FE::cell_type_to_string(cell_type);
        cell_type_node << ryml::key(cell_type_str);
        cell_type_node |= ryml::MAP;
        cell_type_node["number_of_nodes"] << Core::FE::num_nodes(cell_type);
      }
    }
  }
}


std::unique_ptr<Core::IO::MeshReader> Global::read_discretization(
    Global::Problem& problem, Core::IO::InputFile& input, const bool read_mesh)
{
  // decide which kind of spatial representation is required
  const Core::FE::ShapeFunctionType distype = problem.spatial_approximation_type();
  auto output_control = problem.output_control_file();

  // the basic mesh reader. now add desired node and element readers to it!
  auto meshreader_out = std::make_unique<Core::IO::MeshReader>(input,
      Core::Rebalance::RebalanceParameters{
          .mesh_partitioning_parameters =
              Problem::instance()->parameters().get<Core::Rebalance::MeshPartitioningParameters>(
                  "MESH PARTITIONING"),
          .geometric_search_parameters = Problem::instance()->geometric_search_params(),
          .io_parameters = Problem::instance()->io_params(),
      });
  auto& meshreader = *meshreader_out;

  MPI_Comm comm = problem.get_communicators().local_comm();

  enum class DiscretizationType
  {
    plain,
    faces,
    nurbs,
    xwall,
    xfem,
    hdg,
  };

  // Store the name of a discretization along with its type and the identifier in the input file.
  // The identifier may be an empty string to indicate that this discretization is not read from
  // input for a specific problem.
  std::map<std::string, std::pair<DiscretizationType, std::string>> discretization_types;

  switch (problem.get_problem_type())
  {
    case Core::ProblemType::fsi:
    case Core::ProblemType::fsi_redmodels:
    {
      if (distype == Core::FE::ShapeFunctionType::nurbs)
      {
        discretization_types["structure"] = {DiscretizationType::nurbs, "STRUCTURE"};
        discretization_types["fluid"] = {DiscretizationType::nurbs, "FLUID"};
        discretization_types["ale"] = {DiscretizationType::nurbs, "ALE"};
      }
      else if (problem.fluid_dynamic_params().sublist("WALL MODEL").get<bool>("X_WALL"))
      {
        discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
        discretization_types["fluid"] = {DiscretizationType::xwall, "FLUID"};
        discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
      }
      else
      {
        discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
        if (problem.x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
        {
          discretization_types["xfluid"] = {DiscretizationType::xfem, "FLUID"};
          // No input for fluid in this case.
          discretization_types["fluid"] = {DiscretizationType::faces, ""};
        }
        else
        {
          discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
        }
        discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
      }

      break;
    }
    case Core::ProblemType::gas_fsi:
    case Core::ProblemType::thermo_fsi:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          FOUR_C_THROW("Nurbs discretization not possible for fs3i!");
          break;
        }
        default:
        {
          discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
          discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
          discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
          discretization_types["scatra1"] = {DiscretizationType::plain, "TRANSPORT"};
          discretization_types["scatra2"] = {DiscretizationType::plain, "TRANSPORT2"};
          break;
        }
      }
      break;
    }
    case Core::ProblemType::biofilm_fsi:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          FOUR_C_THROW("Nurbs discretization not possible for biofilm problems!");
          break;
        }
        default:
        {
          discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
          discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
          discretization_types["ale"] = {DiscretizationType::plain, ""};
          discretization_types["structale"] = {DiscretizationType::plain, ""};
          break;
        }
      }
      // fluid scatra field
      discretization_types["scatra1"] = {DiscretizationType::plain, "TRANSPORT"};

      // structure scatra field
      discretization_types["scatra2"] = {DiscretizationType::plain, "TRANSPORT2"};

      break;
    }
    case Core::ProblemType::fsi_xfem:
    case Core::ProblemType::fluid_xfem:
    {
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};

      if (problem.x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
      {
        discretization_types["fluid"] = {DiscretizationType::faces, ""};

        discretization_types["xfluid"] = {DiscretizationType::xfem, "FLUID"};
      }
      else
      {
        discretization_types["fluid"] = {DiscretizationType::xfem, "FLUID"};
      }

      discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
      break;
    }
    case Core::ProblemType::fpsi_xfem:
    {
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["fluid"] = {DiscretizationType::xfem, "FLUID"};
      discretization_types["porofluid"] = {DiscretizationType::faces, ""};
      discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
      break;
    }
    case Core::ProblemType::ale:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["ale"] = {DiscretizationType::nurbs, "ALE"};
          break;
        }
        default:
        {
          discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
          break;
        }
      }
      break;
    }
    case Core::ProblemType::fluid:
    case Core::ProblemType::fluid_redmodels:
    {
      if (distype == Core::FE::ShapeFunctionType::hdg)
      {
        discretization_types["fluid"] = {DiscretizationType::hdg, "FLUID"};
      }
      else if (distype == Core::FE::ShapeFunctionType::nurbs)
      {
        discretization_types["fluid"] = {DiscretizationType::nurbs, "FLUID"};
      }
      else if (problem.fluid_dynamic_params().sublist("WALL MODEL").get<bool>("X_WALL"))
      {
        discretization_types["fluid"] = {DiscretizationType::xwall, "FLUID"};
      }
      else
      {
        discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
      }

      break;
    }
    case Core::ProblemType::lubrication:
    {
      // create empty discretizations
      discretization_types["lubrication"] = {DiscretizationType::plain, "LUBRICATION"};
      break;
    }
    case Core::ProblemType::cardiac_monodomain:
    case Core::ProblemType::scatra:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["fluid"] = {DiscretizationType::nurbs, "FLUID"};
          discretization_types["scatra"] = {DiscretizationType::nurbs, "TRANSPORT"};
          break;
        }
        case Core::FE::ShapeFunctionType::hdg:
        {
          discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
          discretization_types["scatra"] = {DiscretizationType::hdg, "TRANSPORT"};
          break;
        }
        default:
        {
          discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
          discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};
          break;
        }
      }
      break;
    }
    case Core::ProblemType::sti:
    {
      // safety checks
      if (distype == Core::FE::ShapeFunctionType::nurbs)
        FOUR_C_THROW("Scatra-thermo interaction does not work for nurbs discretizations yet!");

      // create empty discretizations for scalar and thermo fields
      discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};
      discretization_types["thermo"] = {DiscretizationType::plain, "THERMO"};

      break;
    }
    case Core::ProblemType::fluid_ale:
    {
      if (distype == Core::FE::ShapeFunctionType::hdg)
      {
        discretization_types["fluid"] = {DiscretizationType::hdg, "FLUID"};
        discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
      }
      else if (distype == Core::FE::ShapeFunctionType::nurbs)
      {
        discretization_types["fluid"] = {DiscretizationType::nurbs, "FLUID"};
        discretization_types["ale"] = {DiscretizationType::nurbs, "ALE"};
      }
      else if (problem.fluid_dynamic_params().sublist("WALL MODEL").get<bool>("X_WALL"))
      {
        discretization_types["fluid"] = {DiscretizationType::xwall, "FLUID"};
        discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
      }
      else
      {
        if (problem.x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
        {
          discretization_types["xfluid"] = {DiscretizationType::xfem, "FLUID"};
          discretization_types["fluid"] = {DiscretizationType::faces, ""};
        }
        else
        {
          discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
        }
        discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
      }

      break;
    }
    case Core::ProblemType::tsi:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["structure"] = {DiscretizationType::nurbs, "STRUCTURE"};
          discretization_types["thermo"] = {DiscretizationType::nurbs, "THERMO"};
          break;
        }
        default:
        {
          discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
          discretization_types["thermo"] = {DiscretizationType::plain, "THERMO"};
          break;
        }
      }

      break;
    }
    case Core::ProblemType::thermo:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["thermo"] = {DiscretizationType::nurbs, "THERMO"};
          break;
        }
        default:
        {
          discretization_types["thermo"] = {DiscretizationType::plain, "THERMO"};
          break;
        }
      }

      break;
    }

    case Core::ProblemType::structure:
    {
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["structure"] = {DiscretizationType::nurbs, "STRUCTURE"};
          break;
        }
        default:
        {
          discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
          break;
        }
      }

      break;
    }

    case Core::ProblemType::polymernetwork:
    {
      // create empty discretizations
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["boundingbox"] = {DiscretizationType::plain, "PERIODIC BOUNDINGBOX"};
      break;
    }

    case Core::ProblemType::loma:
    {
      // create empty discretizations
      discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
      discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};
      break;
    }

    case Core::ProblemType::elch:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["fluid"] = {DiscretizationType::nurbs, "FLUID"};
          discretization_types["scatra"] = {DiscretizationType::nurbs, "TRANSPORT"};
          discretization_types["ale"] = {DiscretizationType::nurbs, "ALE"};
          discretization_types["scatra_micro"] = {DiscretizationType::nurbs, "TRANSPORT2"};
          break;
        }
        default:
        {
          discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};
          discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};
          discretization_types["ale"] = {DiscretizationType::plain, "ALE"};
          discretization_types["scatra_micro"] = {DiscretizationType::plain, "TRANSPORT2"};
          break;
        }
      }
      break;
    }
    case Core::ProblemType::art_net:  // _1D_ARTERY_
    {
      // create empty discretizations
      discretization_types["artery"] = {DiscretizationType::plain, "ARTERY"};

      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          FOUR_C_THROW("Nurbs discretization not possible for artery");
          break;
        }
        default:
        {
          discretization_types["artery_scatra"] = {DiscretizationType::plain, "TRANSPORT"};
          break;
        }
      }
      break;
    }
    case Core::ProblemType::reduced_lung:
    case Core::ProblemType::red_airways:  // _reduced D airways
    {
      // create empty discretizations
      discretization_types["red_airway"] = {DiscretizationType::plain, "REDUCED D AIRWAYS"};
      break;
    }
    case Core::ProblemType::poroelast:
    case Core::ProblemType::porofluid_pressure_based_elast:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["structure"] = {DiscretizationType::nurbs, "STRUCTURE"};
          discretization_types["porofluid"] = {DiscretizationType::nurbs, "FLUID"};
          break;
        }
        default:
        {
          discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
          discretization_types["porofluid"] = {DiscretizationType::plain, "FLUID"};
          break;
        }
      }
      if (problem.poro_multi_phase_dynamic_params().get<bool>("artery_coupling_active"))
      {
        discretization_types["artery"] = {DiscretizationType::plain, "ARTERY"};
      }

      break;
    }
    case Core::ProblemType::porofluid_pressure_based_elast_scatra:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["structure"] = {DiscretizationType::nurbs, "STRUCTURE"};
          discretization_types["porofluid"] = {DiscretizationType::nurbs, "FLUID"};
          discretization_types["scatra"] = {DiscretizationType::nurbs, "TRANSPORT"};
          break;
        }
        default:
        {
          discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
          discretization_types["porofluid"] = {DiscretizationType::plain, "FLUID"};
          discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};
          break;
        }
      }
      if (problem.poro_multi_phase_scatra_dynamic_params().get<bool>("artery_coupling_active"))
      {
        discretization_types["artery"] = {DiscretizationType::plain, "ARTERY"};

        discretization_types["artery_scatra"] = {DiscretizationType::plain, "TRANSPORT"};
      }

      break;
    }
    case Core::ProblemType::porofluid_pressure_based:
    {
      // create empty discretizations
      switch (distype)
      {
        case Core::FE::ShapeFunctionType::nurbs:
        {
          discretization_types["porofluid"] = {DiscretizationType::nurbs, "FLUID"};
          break;
        }
        default:
        {
          discretization_types["porofluid"] = {DiscretizationType::plain, "FLUID"};
          break;
        }
      }
      if (problem.porofluid_pressure_based_dynamic_params().get<bool>("artery_coupling_active"))
      {
        discretization_types["artery"] = {DiscretizationType::plain, "ARTERY"};
      }
      break;
    }
    case Core::ProblemType::fpsi:
    {
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};

      // No input for these discretizations
      discretization_types["porofluid"] = {DiscretizationType::plain, ""};
      discretization_types["ale"] = {DiscretizationType::plain, ""};
      break;
    }
    case Core::ProblemType::fbi:
    {
      // create empty discretizations
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};

      break;
    }
    case Core::ProblemType::fps3i:
    {
      // create empty discretizations
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["fluid"] = {DiscretizationType::faces, "FLUID"};

      // No input for these discretizations
      discretization_types["porofluid"] = {DiscretizationType::plain, ""};
      discretization_types["ale"] = {DiscretizationType::plain, ""};
      discretization_types["scatra1"] = {DiscretizationType::plain, ""};
      discretization_types["scatra2"] = {DiscretizationType::plain, ""};

      break;
    }
    case Core::ProblemType::poroscatra:
    {
      // create empty discretizations
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["porofluid"] = {DiscretizationType::plain, "FLUID"};
      discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};

      break;
    }
    case Core::ProblemType::ehl:
    {
      // create empty discretizations
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["lubrication"] = {DiscretizationType::plain, "LUBRICATION"};
      break;
    }
    case Core::ProblemType::ssi:
    case Core::ProblemType::ssti:
    {
      // create empty discretizations
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};

      // consider case of additional scatra manifold
      if (problem.ssi_control_params().sublist("MANIFOLD").get<bool>("ADD_MANIFOLD"))
      {
        discretization_types["scatra_manifold"] = {DiscretizationType::plain, ""};
      }

      if (problem.get_problem_type() == Core::ProblemType::ssti)
      {
        discretization_types["thermo"] = {DiscretizationType::plain, "THERMO"};
      }

      break;
    }
    case Core::ProblemType::particle:
    case Core::ProblemType::pasi:
    {
      // create empty discretizations
      discretization_types["structure"] = {DiscretizationType::plain, "STRUCTURE"};
      break;
    }
    case Core::ProblemType::level_set:
    {
      // create empty discretizations
      discretization_types["scatra"] = {DiscretizationType::plain, "TRANSPORT"};

      break;
    }
    case Core::ProblemType::np_support:
    {
      // no discretizations and nodes needed for supporting procs
      break;
    }
    default:
      FOUR_C_THROW("Unknown problem type: {}", problem.get_problem_type());
      break;
  }

  // add artery or airways discretizations only for the following problem types
  switch (problem.get_problem_type())
  {
    case Core::ProblemType::fsi_redmodels:
    case Core::ProblemType::fluid_ale:
    case Core::ProblemType::fluid_redmodels:
    {
      if (distype == Core::FE::ShapeFunctionType::polynomial)
      {
        // create empty discretizations
        discretization_types["artery"] = {DiscretizationType::plain, "ARTERY"};
        discretization_types["red_airway"] = {DiscretizationType::plain, "REDUCED D AIRWAYS"};
      }
    }
    break;
    default:
      break;
  }

  // Result tests in 4C use the internal node numbering. This means that we need to construct
  // the generated "DOMAIN" discretizations in a defined order. Specifically, "structure" needs to
  // come before all other discretizations. Also, "fluid" is expected before other fields. Sort the
  // info from the map in an equivalent vector.
  std::vector<std::pair<std::string, std::pair<DiscretizationType, std::string>>>
      discretization_types_ordered(discretization_types.begin(), discretization_types.end());
  std::vector<std::string> magical_ordering_of_field_input{"structure", "fluid"};
  for (const auto& field : magical_ordering_of_field_input | std::views::reverse)
  {
    std::ranges::stable_partition(
        discretization_types_ordered, [&](const auto& pair) { return pair.first == field; });
  }

  // Instantiate all the enabled discretizations
  for (const auto& [name, dis_info] : discretization_types_ordered)
  {
    const auto& [dis_type, input_file_keyword] = dis_info;
    std::shared_ptr<Core::FE::Discretization> dis;
    switch (dis_type)
    {
      case DiscretizationType::plain:
        dis = std::make_shared<Core::FE::Discretization>(name, comm, problem.n_dim());
        break;
      case DiscretizationType::faces:
        dis = std::make_shared<Core::FE::DiscretizationFaces>(name, comm, problem.n_dim());
        break;
      case DiscretizationType::nurbs:
        dis = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(name, comm, problem.n_dim());
        break;
      case DiscretizationType::xwall:
        dis = std::make_shared<XFEM::DiscretizationXWall>(name, comm, problem.n_dim());
        break;
      case DiscretizationType::xfem:
        dis = std::make_shared<XFEM::DiscretizationXFEM>(name, comm, problem.n_dim());
        break;
      case DiscretizationType::hdg:
        dis = std::make_shared<Core::FE::DiscretizationHDG>(name, comm, problem.n_dim());
        break;
    }

    problem.add_dis(name, dis);

    if (!input_file_keyword.empty()) meshreader.attach_discretization(dis, input_file_keyword);
  }

  // Set the output writer for all discretizations that have been allocated and attached to the
  // global data.
  for (const auto& dis : problem.discretization_range() | std::views::values)
  {
    dis->set_writer(
        std::make_unique<Core::IO::DiscretizationWriter>(*dis, *output_control, distype));
  }

  if (read_mesh)  // now read and allocate!
  {
    // we read nodes and elements for the desired fields as specified above
    meshreader.read_and_partition();

    // care for special applications
    switch (problem.get_problem_type())
    {
      case Core::ProblemType::elch:
      case Core::ProblemType::fsi:
      case Core::ProblemType::fsi_redmodels:
      case Core::ProblemType::scatra:
      case Core::ProblemType::structure:
      {
        // read microscale fields from second, third, ... input file if necessary
        // (in case of multi-scale material models)
        read_micro_fields(problem, input.file_for_section("MATERIALS").parent_path());
        break;
      }
      case Core::ProblemType::np_support:
      {
        // read microscale fields from second, third, ... inputfile for supporting processors
        read_microfields_np_support(problem);
        break;
      }
      default:
        break;
    }
  }
  return meshreader_out;
}

void Global::read_micro_fields(Global::Problem& problem, const std::filesystem::path& input_path)
{
  // check whether micro material is specified
  const int id_struct = Global::Problem::instance()->materials()->first_id_by_type(
      Core::Materials::m_struct_multiscale);
  const int id_scatra = Global::Problem::instance()->materials()->first_id_by_type(
      Core::Materials::m_scatra_multiscale);
  const int id_elch = Global::Problem::instance()->materials()->first_id_by_type(
      Core::Materials::m_newman_multiscale);

  // return if no multiscale material is used
  if (id_struct == -1 and id_scatra == -1 and id_elch == -1) return;

  // safety check
  if ((id_struct != -1 and id_scatra != -1) or (id_struct != -1 and id_elch != -1) or
      (id_scatra != -1 and id_elch != -1))
    FOUR_C_THROW("Cannot have more than one multi-scale material!");

  // store name of macro-scale discretization in string
  std::string macro_dis_name("");
  if (id_struct != -1)
    macro_dis_name = "structure";
  else
    macro_dis_name = "scatra";

  // fetch communicators
  MPI_Comm lcomm = problem.get_communicators().local_comm();
  MPI_Comm gcomm = problem.get_communicators().global_comm();

  Global::Problem* macro_problem = Global::Problem::instance();
  std::shared_ptr<Core::FE::Discretization> macro_dis = macro_problem->get_dis(macro_dis_name);

  // repartition macro problem for a good distribution of elements with micro material
  if (macro_dis_name == "structure")
  {
    // do weighted repartitioning to obtain new row/column maps
    Teuchos::ParameterList rebalanceParams;
    std::shared_ptr<const Core::LinAlg::Graph> nodeGraph = macro_dis->build_node_graph();
    const auto& [nodeWeights, edgeWeights] = Core::Rebalance::build_weights(*macro_dis);
    const auto& [rownodes, colnodes] =
        Core::Rebalance::rebalance_node_maps(*nodeGraph, rebalanceParams, nodeWeights, edgeWeights);

    // rebuild the discretization with new maps
    macro_dis->redistribute({*rownodes, *colnodes});
  }

  // make sure that we read the micro discretizations only on the processors on
  // which elements with the corresponding micro material are evaluated

  std::set<int> my_multimat_IDs;

  // take care also of ghosted elements! -> ElementColMap!
  for (int i = 0; i < macro_dis->element_col_map()->num_my_elements(); ++i)
  {
    Core::Elements::Element* actele = macro_dis->l_col_element(i);
    std::shared_ptr<Core::Mat::Material> actmat = actele->material();

    if (id_elch != -1 and actmat->material_type() == Core::Materials::m_elchmat)
    {
      // extract wrapped material
      auto elchmat = std::dynamic_pointer_cast<const Mat::ElchMat>(actmat);
      auto elchphase = std::dynamic_pointer_cast<const Mat::ElchPhase>(
          elchmat->phase_by_id(elchmat->phase_id(0)));
      actmat = elchphase->mat_by_id(elchphase->mat_id(0));
    }

    if ((actmat->material_type() == Core::Materials::m_struct_multiscale and
            macro_dis_name == "structure") or
        (actmat->material_type() == Core::Materials::m_scatra_multiscale and
            macro_dis_name == "scatra") or
        (actmat->material_type() == Core::Materials::m_newman_multiscale and
            macro_dis_name == "scatra"))
    {
      Core::Mat::PAR::Parameter* actparams = actmat->parameter();
      my_multimat_IDs.insert(actparams->id());
    }
  }

  // check which macro procs have an element with micro material
  int foundmicromat = 0;
  int foundmicromatmyrank = -1;
  if (my_multimat_IDs.size() != 0)
  {
    foundmicromat = 1;
    foundmicromatmyrank = Core::Communication::my_mpi_rank(lcomm);
  }

  // find out how many procs have micro material
  int nummicromat = 0;
  nummicromat = Core::Communication::sum_all(foundmicromat, lcomm);
  // broadcast number of procs that have micro material
  Core::Communication::broadcast(&nummicromat, 1, 0, gcomm);

  // every proc needs to know which procs have micro material in order to distribute colors
  // array is filled with either its local proc id or -1 when no micro mat was found
  std::vector<int> foundmyranks;
  foundmyranks.resize(Core::Communication::num_mpi_ranks(lcomm), -1);
  Core::Communication::gather_all(&foundmicromatmyrank, foundmyranks.data(), 1, lcomm);

  // determine color of macro procs with any contribution to micro material, only important for
  // procs with micro material color starts with 0 and is incremented for each group
  int color = -1;
  if (foundmicromat == 1)
  {
    for (int foundmyrank : foundmyranks)
    {
      if (foundmyrank != -1) ++color;
      if (foundmyrank == foundmicromatmyrank) break;
    }
  }
  else
  {
    color = MPI_UNDEFINED;
  }

  // do the splitting of the communicator (macro proc must always be proc in subcomm with lowest
  // key
  // --> 0 is inserted here)
  MPI_Comm mpi_local_comm;
  MPI_Comm_split(gcomm, color, 0 /*important here*/, &mpi_local_comm);

  // sort out macro procs that do not have micro material
  if (foundmicromat == 1)
  {
    // create the sub communicator that includes one macro proc and some supporting procs
    MPI_Comm subgroupcomm = mpi_local_comm;
    problem.get_communicators().set_sub_comm(subgroupcomm);

    // find out how many micro problems have to be solved on this macro proc
    int microcount = 0;
    for (const auto& material_map : problem.materials()->map())
    {
      int matid = material_map.first;
      if (my_multimat_IDs.find(matid) != my_multimat_IDs.end()) microcount++;
    }
    // and broadcast it to the corresponding group of procs
    Core::Communication::broadcast(&microcount, 1, 0, subgroupcomm);

    for (const auto& material_map : problem.materials()->map())
    {
      int matid = material_map.first;

      if (my_multimat_IDs.find(matid) != my_multimat_IDs.end())
      {
        std::shared_ptr<Core::Mat::Material> mat = Mat::factory(matid);

        // initialize variables storing micro-scale information
        int microdisnum(-1);
        std::string micro_dis_name = "";
        std::string micro_inputfile_name("");
        Global::Problem* micro_problem(nullptr);

        // structure case
        if (macro_dis_name == "structure")
        {
          // access multi-scale structure material
          auto* micromat = static_cast<Mat::MicroMaterial*>(mat.get());

          // extract and broadcast number of micro-scale discretization
          microdisnum = micromat->micro_dis_num();
          Core::Communication::broadcast(&microdisnum, 1, 0, subgroupcomm);

          // set name of micro-scale discretization
          micro_dis_name = "structure";

          // extract name of micro-scale input file
          micro_inputfile_name = micromat->micro_input_file_name();

          // instantiate micro-scale problem
          micro_problem = Global::Problem::instance(microdisnum);
        }

        // scalar transport case
        else
        {
          // access multi-scale scalar transport material
          Mat::ScatraMicroMacroCoupling* micromat = nullptr;
          if (id_scatra != -1)
            micromat = dynamic_cast<Mat::ScatraMultiScale*>(mat.get());
          else if (id_elch != -1)
            micromat = dynamic_cast<Mat::NewmanMultiScale*>(mat.get());
          else
            FOUR_C_THROW("How the heck did you get here?!");

          // extract and broadcast number of micro-scale discretization
          microdisnum = micromat->micro_dis_num();
          Core::Communication::broadcast(&microdisnum, 1, 0, subgroupcomm);

          // set unique name of micro-scale discretization
          std::stringstream name;
          name << "scatra_multiscale_" << microdisnum;
          micro_dis_name = name.str();

          // extract name of micro-scale input file
          micro_inputfile_name = micromat->micro_input_file_name();

          // instantiate micro-scale problem
          micro_problem = Global::Problem::instance(microdisnum);
        }

        if (micro_inputfile_name[0] != '/')
        {
          micro_inputfile_name = input_path / micro_inputfile_name;
        }

        // broadcast micro input file name
        int length = static_cast<int>(micro_inputfile_name.length());
        Core::Communication::broadcast(&length, 1, 0, subgroupcomm);
        Core::Communication::broadcast(
            (const_cast<char*>(micro_inputfile_name.c_str())), length, 0, subgroupcomm);

        // start with actual reading
        Core::IO::InputFile micro_input_file = set_up_input_file(subgroupcomm);
        micro_input_file.read(micro_inputfile_name);

        std::shared_ptr<Core::FE::Discretization> dis_micro =
            std::make_shared<Core::FE::Discretization>(
                micro_dis_name, subgroupcomm, problem.n_dim());

        // replace standard dofset inside micro discretization by independent dofset
        // to avoid inconsistent dof numbering in non-nested parallel settings with more than one
        // micro discretization
        if (problem.get_communicators().np_type() ==
            Core::Communication::NestedParallelismType::no_nested_parallelism)
          dis_micro->replace_dof_set(std::make_shared<Core::DOFSets::IndependentDofSet>());

        // We do not need a writer but the rest of the code wants us to have one.
        // Thus, we create a dummy output control here and a writer which is set to not write ever.
        micro_problem->set_output_control_file(std::make_shared<Core::IO::OutputControl>(
            dis_micro->get_comm(), "dummy", micro_problem->spatial_approximation_type(),
            "micro-input-file-not-known", "", "", dis_micro->n_dim(), false, 0,
            /*write binary output: this flag makes the whole control useless*/ false, false));
        dis_micro->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(*dis_micro,
            *micro_problem->output_control_file(), micro_problem->spatial_approximation_type()));

        micro_problem->add_dis(micro_dis_name, dis_micro);

        read_parameter(*micro_problem, micro_input_file);

        // read materials of microscale
        // CAUTION: materials for microscale cannot be read until
        // micro_reader is activated, since else materials will again be
        // read from macroscale inputfile. Besides, materials MUST be read
        // before elements are read since elements establish a connection
        // to the corresponding material! Thus do not change position of
        // function calls!
        problem.materials()->set_read_from_problem(microdisnum);

        read_materials(*micro_problem, micro_input_file);

        Core::IO::MeshReader micromeshreader(micro_input_file,
            {.mesh_partitioning_parameters = Problem::instance()
                    ->parameters()
                    .get<Core::Rebalance::MeshPartitioningParameters>("MESH PARTITIONING"),
                .geometric_search_parameters = Problem::instance()->geometric_search_params(),
                .io_parameters = Problem::instance()->io_params()});

        if (micro_dis_name == "structure")
        {
          micromeshreader.attach_discretization(dis_micro, "STRUCTURE");
        }
        else
          micromeshreader.attach_discretization(dis_micro, "TRANSPORT");

        micromeshreader.read_and_partition();

        read_conditions(*micro_problem, micro_input_file, micromeshreader);

        {
          Core::Utils::FunctionManager function_manager;
          global_legacy_module_callbacks().AttachFunctionDefinitions(function_manager);
          function_manager.read_input(micro_input_file);
          micro_problem->set_function_manager(std::move(function_manager));
        }

        read_result(*micro_problem, micro_input_file);

        // At this point, everything for the microscale is read,
        // subsequent reading is only for macroscale
        dis_micro->fill_complete();

        // broadcast restart information
        int restart_step = problem.restart();
        Core::Communication::broadcast(&restart_step, 1, 0, subgroupcomm);
        problem.set_restart_step(restart_step);

        // set the problem number from which to call materials again to zero
        // (i.e. macro problem), cf. Mat::factory!
        problem.materials()->reset_read_from_problem();
      }
    }
    problem.materials()->reset_read_from_problem();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_microfields_np_support(Global::Problem& problem)
{
  MPI_Comm lcomm = problem.get_communicators().local_comm();
  MPI_Comm gcomm = problem.get_communicators().global_comm();

  // receive number of procs that have micro material
  int nummicromat = 0;
  Core::Communication::broadcast(&nummicromat, 1, 0, gcomm);

  // prepare the supporting procs for a splitting of gcomm

  // groups should be equally sized
  // in a first step every macro proc that needs support gets procpergroup supporting procs
  int procpergroup = int(floor((Core::Communication::num_mpi_ranks(lcomm)) / nummicromat));
  std::vector<int> supgrouplayout(nummicromat, procpergroup);
  // remaining procs are added to the groups in the beginning
  int remainingProcs = Core::Communication::num_mpi_ranks(lcomm) - procpergroup * nummicromat;
  for (int k = 0; k < remainingProcs; ++k)
  {
    supgrouplayout[k]++;
  }

  // secondly: colors are distributed
  // color starts with 0 and is incremented for each group
  int color = -1;
  int gsum = 0;
  do
  {
    color++;
    gsum += supgrouplayout[color];
  } while (gsum <= Core::Communication::my_mpi_rank(lcomm));

  // do the splitting of the communicator
  MPI_Comm mpi_local_comm;
  MPI_Comm_split(gcomm, color, Core::Communication::my_mpi_rank(gcomm), &mpi_local_comm);

  // create the sub communicator that includes one macro proc and some supporting procs
  MPI_Comm subgroupcomm = mpi_local_comm;
  problem.get_communicators().set_sub_comm(subgroupcomm);

  // number of micro problems for this sub group
  int microcount = 0;
  Core::Communication::broadcast(&microcount, 1, 0, subgroupcomm);

  for (int n = 0; n < microcount; n++)
  {
    // broadcast microdis number
    int microdisnum = -1;
    Core::Communication::broadcast(&microdisnum, 1, 0, subgroupcomm);

    Global::Problem* micro_problem = Global::Problem::instance(microdisnum);

    // broadcast micro input file name
    int length = -1;
    std::string micro_inputfile_name;
    Core::Communication::broadcast(&length, 1, 0, subgroupcomm);
    micro_inputfile_name.resize(length);
    Core::Communication::broadcast(
        (const_cast<char*>(micro_inputfile_name.c_str())), length, 0, subgroupcomm);

    // start with actual reading
    Core::IO::InputFile micro_input_file = set_up_input_file(subgroupcomm);
    micro_input_file.read(micro_inputfile_name);

    std::shared_ptr<Core::FE::Discretization> structdis_micro =
        std::make_shared<Core::FE::Discretization>("structure", subgroupcomm, problem.n_dim());

    // We do not need a writer but the rest of the code wants us to have one.
    // Thus, we create a dummy output control here and a writer which is set to not write ever.
    micro_problem->set_output_control_file(std::make_shared<Core::IO::OutputControl>(
        structdis_micro->get_comm(), "dummy", micro_problem->spatial_approximation_type(),
        "micro-input-file-not-known", "", "", structdis_micro->n_dim(), false, 0,
        /*write binary output: this flag makes the whole control useless*/ false, false));
    structdis_micro->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(*structdis_micro,
        *micro_problem->output_control_file(), micro_problem->spatial_approximation_type()));

    micro_problem->add_dis("structure", structdis_micro);

    read_parameter(*micro_problem, micro_input_file);

    // read materials of microscale
    // CAUTION: materials for microscale cannot be read until
    // micro_reader is activated, since else materials will again be
    // read from macroscale inputfile. Besides, materials MUST be read
    // before elements are read since elements establish a connection
    // to the corresponding material! Thus do not change position of
    // function calls!
    problem.materials()->set_read_from_problem(microdisnum);

    read_materials(*micro_problem, micro_input_file);

    Core::IO::MeshReader micromeshreader(micro_input_file,
        {.mesh_partitioning_parameters =
                Problem::instance()->parameters().get<Core::Rebalance::MeshPartitioningParameters>(
                    "MESH PARTITIONING"),
            .geometric_search_parameters = Problem::instance()->geometric_search_params(),
            .io_parameters = Problem::instance()->io_params()});
    micromeshreader.attach_discretization(structdis_micro, "STRUCTURE");
    micromeshreader.read_and_partition();

    read_conditions(*micro_problem, micro_input_file, micromeshreader);

    {
      Core::Utils::FunctionManager function_manager;
      global_legacy_module_callbacks().AttachFunctionDefinitions(function_manager);
      function_manager.read_input(micro_input_file);
      micro_problem->set_function_manager(std::move(function_manager));
    }

    read_result(*micro_problem, micro_input_file);

    // At this point, everything for the microscale is read,
    // subsequent reading is only for macroscale
    structdis_micro->fill_complete();

    // broadcast restart information
    int restart_step = problem.restart();
    Core::Communication::broadcast(&restart_step, 1, 0, subgroupcomm);
    problem.set_restart_step(restart_step);

    // set the problem number from which to call materials again to zero
    // (i.e. macro problem), cf. Mat::factory!
    problem.materials()->reset_read_from_problem();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_parameter(Global::Problem& problem, Core::IO::InputFile& input)
{
  std::shared_ptr<Teuchos::ParameterList> list = std::make_shared<Teuchos::ParameterList>("ROOT");

  auto parameter_section_specs = global_legacy_module_callbacks().parameters();

  for (const auto& sec : parameter_section_specs)
  {
    Core::IO::read_parameters_in_section(input, sec.name(), *list);
  }

  // check for invalid parameters
  problem.set_parameter_list(list);

  //---------------------------------------------------------------------
  // Now we have successfully read the whole input file. It's time to access some data

  // 1) get the problem type
  const Teuchos::ParameterList& type = problem.problem_type_params();
  problem.set_problem_type(Teuchos::getIntegralValue<Core::ProblemType>(type, "PROBLEMTYPE"));

  // 2) get the spatial approximation type
  problem.set_spatial_approximation_type(
      Teuchos::getIntegralValue<Core::FE::ShapeFunctionType>(type, "SHAPEFCT"));

  int restart_step = problem.restart();
  // 3) do the restart business with the four options we support (partially)
  if (restart_step == 0)
  {
    // no restart flag on the command line, so check the restart flag from the input file
    restart_step = type.get<int>("RESTART");
    problem.set_restart_step(restart_step);
  }
  else  // SetRestartStep() has been called before!
  {
    // There is a non-zero restart flag on the command line, so we ignore the input file.
    // The RESTART flag in the input file should be zero or have the same value!
    const int restartflaginfile = type.get<int>("RESTART");
    if ((restartflaginfile > 0) and (restartflaginfile != restart_step))
      FOUR_C_THROW("Restart flags in input file and command line are non-zero and different!");
  }

  // Set restart time based on walltime
  const double restartinterval = problem.io_params().get<double>("RESTARTWALLTIMEINTERVAL");
  const int restartevry = problem.io_params().get<int>("RESTARTEVERY");
  problem.restart_manager()->setup_restart_manager(restartinterval, restartevry);

  // 4) set random seed
  // time is in seconds, therefore we add the global processor id to obtain a unique seed on each
  // proc
  {
    int rs = type.get<int>("RANDSEED");
    if (rs < 0)
      rs = static_cast<int>(time(nullptr)) +
           42 * Core::Communication::my_mpi_rank(
                    Global::Problem::instance(0)->get_communicators().global_comm());

    srand((unsigned int)rs);  // Set random seed for stdlibrary. This is deprecated, as it does not
    // produce random numbers on some platforms!
    problem.random()->set_rand_seed((unsigned int)rs);  // Use this instead.
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_materials(Global::Problem& problem, Core::IO::InputFile& input)
{
  Core::IO::InputParameterContainer container;
  try
  {
    input.match_section("MATERIALS", container);
  }
  catch (const Core::Exception& e)
  {
    FOUR_C_THROW(
        "Failed to match specification in section 'MATERIALS'. The error was:\n{}.", e.what());
  }

  for (const auto& material_entry :
      container.get_or<std::vector<Core::IO::InputParameterContainer>>("MATERIALS", {}))
  {
    const int mat_id = material_entry.get<int>("MAT");

    FOUR_C_ASSERT_ALWAYS(mat_id >= 0, "Material ID must be non-negative. Found: {}", mat_id);

    if (problem.materials()->id_exists(mat_id))
      FOUR_C_THROW("More than one material with 'MAT {}'", mat_id);

    const auto mat_type = material_entry.get<Core::Materials::MaterialType>("_material_type");

    const auto& group = material_entry.groups();
    FOUR_C_ASSERT_ALWAYS(
        group.size() == 1, "Internal error: material must have exactly one group.");
    const auto& [material_name, material_container] = group.front();

    problem.materials()->insert(
        mat_id, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>(
                    [mat_id, mat_type, container = material_entry.group(material_name)]()
                    { return Mat::make_parameter(mat_id, mat_type, container); }));
  }

  // We have read in all the materials and now we force construction of them all. The LazyPtr
  // ensures that the ordering does not matter. Note that we do not wait any longer for
  // construction, because materials might later be used in code sections that only run on proc 0.
  // Doing anything MPI-parallel inside the material constructors would then fail. Unfortunately,
  // such operations happen in the code base, thus we construct the materials here.
  for (const auto& [id, mat] : problem.materials()->map())
  {
    // This is the point where the material is actually constructed via the side effect that we
    // try to access the material.
    (void)mat.get();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_contact_constitutive_laws(Global::Problem& problem, Core::IO::InputFile& input)
{
  const std::string contact_const_laws = "CONTACT CONSTITUTIVE LAWS";
  Core::IO::InputParameterContainer container;
  input.match_section(contact_const_laws, container);

  const auto* laws = container.get_if<Core::IO::InputParameterContainer::List>(contact_const_laws);
  if (laws)
    for (const auto& law : *laws)
      CONTACT::CONSTITUTIVELAW::create_contact_constitutive_law_from_input(law);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_cloning_material_map(Global::Problem& problem, Core::IO::InputFile& input)
{
  Core::IO::InputParameterContainer container;
  input.match_section("CLONING MATERIAL MAP", container);
  const auto* map_entries =
      container.get_if<Core::IO::InputParameterContainer::List>("CLONING MATERIAL MAP");

  if (!map_entries) return;

  for (const auto& entry : *map_entries)
  {
    std::string src_field = entry.get<std::string>("SRC_FIELD");
    int src_matid = entry.get_or<int>("SRC_MAT", -1);
    std::string tar_field = entry.get<std::string>("TAR_FIELD");
    int tar_matid = entry.get_or<int>("TAR_MAT", -1);

    std::pair<std::string, std::string> fields(src_field, tar_field);
    std::pair<int, int> matmap(src_matid, tar_matid);
    problem.cloning_material_map()[fields].insert(matmap);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_result(Global::Problem& problem, Core::IO::InputFile& input)
{
  // read design nodes <-> nodes, lines <-> nodes, surfaces <-> nodes, volumes <-> nodes
  const auto get_discretization_callback = [](const std::string& name) -> decltype(auto)
  { return *Global::Problem::instance()->get_dis(name); };
  std::vector<std::vector<std::vector<int>>> nodeset(4);
  Core::IO::read_design(input, "DNODE", nodeset[0], get_discretization_callback);
  Core::IO::read_design(input, "DLINE", nodeset[1], get_discretization_callback);
  Core::IO::read_design(input, "DSURF", nodeset[2], get_discretization_callback);
  Core::IO::read_design(input, "DVOL", nodeset[3], get_discretization_callback);
  problem.get_result_test_manager().set_node_set(nodeset);

  Core::IO::InputParameterContainer container;
  input.match_section("RESULT DESCRIPTION", container);

  const auto* result_descriptions =
      container.get_if<Core::IO::InputParameterContainer::List>("RESULT DESCRIPTION");
  if (result_descriptions) problem.get_result_test_manager().set_parsed_lines(*result_descriptions);
}

namespace
{
  void get_node_sets_from_mesh(
      std::map<int, std::vector<int>>& node_sets, const Core::IO::MeshReader& mesh_reader)
  {
    node_sets.clear();
    const int my_rank = Core::Communication::my_mpi_rank(mesh_reader.get_comm());

    // Data is available on rank zero: bring it into the right shape and broadcast it.
    if (my_rank == 0)
    {
      if (const auto* external_mesh = mesh_reader.get_external_mesh_on_rank_zero(); external_mesh)
      {
        const auto& node_sets_from_mesh = external_mesh->point_sets();
        for (const auto& [id, node_set] : node_sets_from_mesh)
        {
          const auto& set = node_set.point_ids;
          node_sets[id] = std::vector<int>(set.begin(), set.end());
        }
      }
      Core::Communication::broadcast(node_sets, 0, mesh_reader.get_comm());
    }
    else
    {
      Core::Communication::broadcast(node_sets, 0, mesh_reader.get_comm());
    }
  }

  void get_element_block_nodes_from_mesh(
      std::map<int, std::vector<int>>& element_block_nodes, const Core::IO::MeshReader& mesh_reader)
  {
    element_block_nodes.clear();
    const int my_rank = Core::Communication::my_mpi_rank(mesh_reader.get_comm());

    // Data is available on rank zero: bring it into the right shape and broadcast it.
    if (my_rank == 0)
    {
      if (const auto* external_mesh = mesh_reader.get_external_mesh_on_rank_zero(); external_mesh)
      {
        for (const auto& [id, eb] : external_mesh->cell_blocks())
        {
          std::set<int> nodes;
          for (const auto& cell : eb.cells())
          {
            nodes.insert(cell.begin(), cell.end());
          }
          element_block_nodes[id] = std::vector<int>(nodes.begin(), nodes.end());
        }
      }
      Core::Communication::broadcast(element_block_nodes, 0, mesh_reader.get_comm());
    }
    else
    {
      Core::Communication::broadcast(element_block_nodes, 0, mesh_reader.get_comm());
    }
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_conditions(
    Global::Problem& problem, Core::IO::InputFile& input, const Core::IO::MeshReader& mesh_reader)
{
  Teuchos::Time time("", true);
  if (Core::Communication::my_mpi_rank(input.get_comm()) == 0)
  {
    Core::IO::cout << "Read/generate conditions                          in....";
    Core::IO::cout.flush();
  }

  //--------------------------------------------- read generic node sets
  const auto get_discretization_callback = [](const std::string& name) -> decltype(auto)
  { return *Global::Problem::instance()->get_dis(name); };

  // read design nodes <-> nodes
  std::vector<std::vector<int>> dnode_fenode;
  Core::IO::read_design(input, "DNODE", dnode_fenode, get_discretization_callback);

  // read design lines <-> nodes
  std::vector<std::vector<int>> dline_fenode;
  Core::IO::read_design(input, "DLINE", dline_fenode, get_discretization_callback);

  // read design surfaces <-> nodes
  std::vector<std::vector<int>> dsurf_fenode;
  Core::IO::read_design(input, "DSURF", dsurf_fenode, get_discretization_callback);

  // read design volumes <-> nodes
  std::vector<std::vector<int>> dvol_fenode;
  Core::IO::read_design(input, "DVOL", dvol_fenode, get_discretization_callback);

  std::map<int, std::vector<int>> node_sets;
  get_node_sets_from_mesh(node_sets, mesh_reader);

  std::map<int, std::vector<int>> element_block_nodes;
  get_element_block_nodes_from_mesh(element_block_nodes, mesh_reader);

  // check for meshfree discretisation to add node set topologies
  std::vector<std::vector<std::vector<int>>*> nodeset(4);
  nodeset[0] = &dnode_fenode;
  nodeset[1] = &dline_fenode;
  nodeset[2] = &dsurf_fenode;
  nodeset[3] = &dvol_fenode;

  // create list of known conditions
  std::vector<Core::Conditions::ConditionDefinition> valid_conditions = Global::valid_conditions();

  // test for each condition definition (input file condition section)
  // - read all conditions that match the definition
  // - add the nodal clouds to the conditions
  // - add the conditions to the appropriate discretizations
  //
  // Note that this will reset (un-fill_complete) the discretizations.
  for (const auto& condition_definition : valid_conditions)
  {
    std::multimap<int, std::shared_ptr<Core::Conditions::Condition>> cond;

    // read conditions from the input file
    condition_definition.read(input, cond);

    // add nodes to conditions
    for (const auto& [entity_id, condition] : cond)
    {
      switch (condition->entity_type())
      {
        case Core::Conditions::EntityType::legacy_id:
        {
          if (dnode_fenode.size() == 0 && dline_fenode.size() == 0 && dsurf_fenode.size() == 0 &&
              dvol_fenode.size() == 0)
          {
            FOUR_C_THROW(
                "{} condition {} uses legacy_id entity type but no legacy entities were defined in "
                "the input file.\n"
                "This is probably because the geometry is handled in an external file.\n"
                "If this is the case, you must specify a specific entity type (node_set_id or "
                "element_block_id).\n",
                condition_definition.name(), entity_id);
          }
          switch (condition->g_type())
          {
            case Core::Conditions::geometry_type_point:
              if (entity_id < 0 or static_cast<unsigned>(entity_id) >= dnode_fenode.size())
              {
                FOUR_C_THROW(
                    "DPoint {} not in range [0:{}[\n"
                    "DPoint condition on non existent DPoint?"
                    "Could not read set from entity type.",
                    entity_id, dnode_fenode.size());
              }
              condition->set_nodes(dnode_fenode[entity_id]);
              break;
            case Core::Conditions::geometry_type_line:
              if (entity_id < 0 or static_cast<unsigned>(entity_id) >= dline_fenode.size())
              {
                FOUR_C_THROW(
                    "DLine {} not in range [0:{}[\n"
                    "DLine condition on non existent DLine?"
                    "Could not read set from entity type.",
                    entity_id, dline_fenode.size());
              }
              condition->set_nodes(dline_fenode[entity_id]);
              break;
            case Core::Conditions::geometry_type_surface:
              if (entity_id < 0 or static_cast<unsigned>(entity_id) >= dsurf_fenode.size())
              {
                FOUR_C_THROW(
                    "DSurface {} not in range [0:{}[\n"
                    "DSurface condition on non existent DSurface?"
                    "Could not read set from entity type.",
                    entity_id, dsurf_fenode.size());
              }
              condition->set_nodes(dsurf_fenode[entity_id]);
              break;
            case Core::Conditions::geometry_type_volume:
              if (entity_id < 0 or static_cast<unsigned>(entity_id) >= dvol_fenode.size())
              {
                FOUR_C_THROW(
                    "DVolume {} not in range [0:{}[\n"
                    "DVolume condition on non existent DVolume?",
                    entity_id, dvol_fenode.size());
              }
              condition->set_nodes(dvol_fenode[entity_id]);
              break;
            default:
              FOUR_C_THROW("geometry type unspecified");
              break;
          }

          break;
        }
        case Core::Conditions::EntityType::node_set_id:
        {
          FOUR_C_ASSERT_ALWAYS(node_sets.contains(entity_id),
              "Cannot apply condition '{}' to node set {} which is not specified in the mesh file.",
              condition_definition.name(), entity_id);
          condition->set_nodes(node_sets[entity_id]);
          break;
        }
        case Core::Conditions::EntityType::element_block_id:
        {
          FOUR_C_ASSERT_ALWAYS(element_block_nodes.contains(entity_id),
              "Cannot apply condition '{}' to element block {} which is not specified in the mesh "
              "file.",
              condition_definition.name(), entity_id);
          condition->set_nodes(element_block_nodes[entity_id]);
          break;
        }
      }

      // Iterate through all discretizations and sort the appropriate condition
      // into the correct discretization it applies to

      for (const auto& [name, dis] : problem.discretization_range())
      {
        const std::vector<int>* nodes = condition->get_nodes();
        if (nodes->size() == 0)
          FOUR_C_THROW(
              "{} condition {} has no nodal cloud", condition_definition.name(), condition->id());

        int foundit = 0;
        for (int node : *nodes)
        {
          foundit = dis->have_global_node(node);
          if (foundit) break;
        }
        int found = 0;
        found = Core::Communication::sum_all(foundit, dis->get_comm());
        if (found)
        {
          // Insert a copy since we might insert the same condition in many discretizations.
          dis->set_condition(condition_definition.name(), condition->copy_without_geometry());
        }
      }
    }
  }

  if (Core::Communication::my_mpi_rank(input.get_comm()) == 0)
  {
    std::cout << time.totalElapsedTime(true) << " secs\n";
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_knots(Global::Problem& problem, Core::IO::InputFile& input)
{
  // get information on the spatial approximation --- we only read knots
  // in the nurbs case
  Core::FE::ShapeFunctionType distype = problem.spatial_approximation_type();

  // Iterate through all discretizations and sort the appropriate condition
  // into the correct discretization it applies to

  for (const auto& [name, dis] : problem.discretization_range())
  {
    if (distype == Core::FE::ShapeFunctionType::nurbs)
    {
      // cast discretisation to nurbs variant to be able
      // to add the knotvector
      auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*dis));

      if (nurbsdis == nullptr)
        FOUR_C_THROW("discretization {} is not a NurbsDiscretization! Panic.", dis->name());

      // read the knotvector data from the input
      auto disknots = Core::IO::read_knots(input, dis->name());

      if (disknots == nullptr)
      {
        FOUR_C_THROW("Knotvector read failed in Nurbs discretisation\n");
      }

      // make sure atdis is fillcompleted, to be able to call
      // ElementRowMap() on it
      // do not initialize elements, since this would require knot
      // vector values
      if (!dis->filled())
      {
        dis->fill_complete(Core::FE::OptionsFillComplete::none());
      }

      // the smallest gid in the discretisation determines the access
      // pattern via the element offset
      int smallest_gid_in_dis = dis->element_row_map()->min_all_gid();

      // consistency checks
      disknots->finish_knots(smallest_gid_in_dis);

      // add knots to discretisation
      nurbsdis->set_knot_vector(std::move(disknots));
    }
  }  // loop fields
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Global::read_particles(Global::Problem& problem, Core::IO::InputFile& input)
{
  // no need to read in particles in case of restart
  if (problem.restart()) return;

  Particle::read_particles(input, "PARTICLES", problem.particles());
}


void Global::read_fields(
    Global::Problem& problem, Core::IO::InputFile& input, const Core::IO::MeshReader& mesh_reader)
{
  Core::IO::InputFieldRegistry& field_registry = Core::IO::global_input_field_registry();
  Core::IO::MeshDataInputFieldRegistry& mesh_data_registry =
      Core::IO::global_mesh_data_input_field_registry();

  Core::IO::InputParameterContainer fields;
  input.match_section("fields", fields);

  const auto& field_entries =
      fields.get_or("fields", std::vector<Core::IO::InputParameterContainer>{});

  // Read the information for the fields, warn if we read a field that is not registered and thus
  // never used.
  std::set<std::string> defined_field_names{};
  for (const auto& field_entry : field_entries)
  {
    const auto& field = field_entry.get<std::string>("name");
    const auto& discretization_name = field_entry.get<std::string>("discretization");
    const auto& source = field_entry.group("source").get<Core::IO::InputFieldSource>("source");

    defined_field_names.emplace(field);

    if (source == Core::IO::InputFieldSource::separate_file)
    {
      const auto& separate_file_input = field_entry.group("source").group("separate_file");
      const auto& file_path = separate_file_input.get<std::filesystem::path>("file");
      auto key = separate_file_input.get<std::optional<std::string>>("key");
      if (!key) key = field;

      auto it = field_registry.fields.find(field);
      if (it == field_registry.fields.end())
      {
        Core::IO::cout << "WARNING: Field '" << field
                       << "' defined but never referenced in input file.\n";
        continue;
      }
      auto& field_data = it->second;

      // Get the discretization
      auto discretization = problem.get_dis(discretization_name);

      // initialize the fields using this reference (only on rank 0)
      if (Core::Communication::my_mpi_rank(discretization->get_comm()) == 0)
      {
        for (auto& function : field_data.init_functions | std::views::values)
        {
          function(file_path, *key);
        }
      }

      // Attach a callback to redistribute the field once the dofs are assigned.
      discretization->callbacks().post_assign_dofs.add(
          [&field_data](const Core::FE::Discretization& dis)
          {
            // Redistribute the field once we assigned the dofs
            const auto& target_map = *dis.element_col_map();
            for (const auto& fn : field_data.redistribute_functions | std::views::values)
              fn(target_map);
          });
    }
    else if (source == Core::IO::InputFieldSource::from_mesh)
    {
      const auto& mesh_input = field_entry.group("source").group("from_mesh");

      auto key = mesh_input.get<std::optional<std::string>>("key");
      auto basis = mesh_input.get<Core::IO::FieldDataBasis>("basis");
      if (!key) key = field;

      auto it = mesh_data_registry.fields.find(field);
      if (it == mesh_data_registry.fields.end())
      {
        Core::IO::cout << "WARNING: Field '" << field
                       << "' defined but never referenced in input file.\n";
        continue;
      }
      auto& field_data = it->second;

      // Get the discretization
      auto discretization = problem.get_dis(discretization_name);
      const Core::IO::MeshInput::Mesh<3>* mesh =
          mesh_reader.get_filtered_external_mesh_on_rank_zero(*discretization);

      // initialize the fields using this reference (only on rank 0)
      if (Core::Communication::my_mpi_rank(discretization->get_comm()) == 0)
      {
        FOUR_C_ASSERT_ALWAYS(mesh,
            "Cannot read field '{}': Field data from mesh requires reading the mesh from an "
            "external mesh. No external mesh found for discretization '{}'",
            field, discretization_name);
        for (auto& function : field_data.init_functions | std::views::values)
        {
          function(*discretization, *mesh, basis, *key);
        }
      }
      else
      {
        const Core::IO::MeshInput::Mesh<3> empty_mesh_on_other_ranks{};
        for (auto& function : field_data.init_functions | std::views::values)
        {
          function(*discretization, empty_mesh_on_other_ranks, basis, *key);
        }
      }

      // Attach a callback to redistribute the field once the dofs are assigned.
      discretization->callbacks().post_assign_dofs.add(
          [&field_data, basis](const Core::FE::Discretization& dis)
          {
            // Redistribute the field once we assigned the dofs
            const auto& target_map = basis == Core::IO::FieldDataBasis::points
                                         ? *dis.node_col_map()
                                         : *dis.element_col_map();
            for (const auto& fn : field_data.redistribute_functions | std::views::values)
              fn(target_map);
          });
    }
  }

  // Now check whether all fields that are referenced in the input file were actually defined in the
  // fields section.
  for (const auto& field_name : field_registry.fields | std::views::keys)
  {
    FOUR_C_ASSERT_ALWAYS(defined_field_names.contains(field_name),
        "You refer to a field '{}' but it was never defined in the top-level 'fields' section. "
        "Add an entry in the top-level 'fields' section.",
        field_name);
  }
}

FOUR_C_NAMESPACE_CLOSE
