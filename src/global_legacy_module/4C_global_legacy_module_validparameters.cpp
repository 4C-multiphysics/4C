// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_global_legacy_module_validparameters.hpp"

#include "4C_ale_input.hpp"
#include "4C_art_net_input.hpp"
#include "4C_beam3_discretization_runtime_output_input.hpp"
#include "4C_beaminteraction_input.hpp"
#include "4C_binstrategy_input.hpp"
#include "4C_browniandyn_input.hpp"
#include "4C_cardiovascular0d_input.hpp"
#include "4C_constraint_framework_input.hpp"
#include "4C_contact_input.hpp"
#include "4C_coupling_volmortar_input.hpp"
#include "4C_cut_input.hpp"
#include "4C_ehl_input.hpp"
#include "4C_elch_input.hpp"
#include "4C_fbi_input.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_geometry_searchtree_input.hpp"
#include "4C_fem_nurbs_discretization_knotvector.hpp"
#include "4C_fpsi_input.hpp"
#include "4C_fs3i_input.hpp"
#include "4C_fsi_input.hpp"
#include "4C_geometric_search_input.hpp"
#include "4C_global_legacy_module_problem_type_string.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_inpar_cardiac_monodomain.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_io.hpp"
#include "4C_inpar_IO_runtime_output.hpp"
#include "4C_inpar_IO_runtime_output_fluid.hpp"
#include "4C_inpar_IO_runtime_vtk_output_structure.hpp"
#include "4C_inpar_IO_runtime_vtp_output_structure.hpp"
#include "4C_inpar_plasticity.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io_gridgenerator.hpp"
#include "4C_io_input_field.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_mesh.hpp"
#include "4C_io_pstream.hpp"
#include "4C_levelset_input.hpp"
#include "4C_linear_solver_method_input.hpp"
#include "4C_lubrication_input.hpp"
#include "4C_mortar_input.hpp"
#include "4C_particle_input.hpp"
#include "4C_pasi_input.hpp"
#include "4C_poroelast_input.hpp"
#include "4C_poroelast_scatra_input.hpp"
#include "4C_porofluid_pressure_based_elast_input.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"
#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_rebalance_input.hpp"
#include "4C_red_airways_input.hpp"
#include "4C_reduced_lung_1d_pipe_flow_input.hpp"
#include "4C_reduced_lung_input.hpp"
#include "4C_solver_nonlin_nox_input.hpp"
#include "4C_ssi_input.hpp"
#include "4C_ssti_input.hpp"
#include "4C_sti_input.hpp"
#include "4C_structure_new_monitor_dbc_input.hpp"
#include "4C_thermo_input.hpp"
#include "4C_tsi_input.hpp"

#include <iostream>
#include <string>


FOUR_C_NAMESPACE_OPEN

// helper overloads: accept either a single InputSpec or a vector of InputSpec
namespace
{
  inline void push_specs(std::vector<Core::IO::InputSpec>& specs, const Core::IO::InputSpec& s)
  {
    specs.push_back(s);
  }
  inline void push_specs(
      std::vector<Core::IO::InputSpec>& specs, const std::vector<Core::IO::InputSpec>& v)
  {
    specs.insert(specs.end(), v.begin(), v.end());
  }
}  // unnamed namespace


namespace CommonParameters
{
  static std::vector<Core::IO::InputSpec> valid_parameters()
  {
    using namespace Core::IO::InputSpecBuilders;
    std::vector<Core::IO::InputSpec> specs;

    specs.push_back(group("DISCRETISATION",
        {

            parameter<int>("NUMFLUIDDIS",
                {.description = "Number of meshes in fluid field", .default_value = 1}),
            parameter<int>("NUMSTRUCDIS",
                {.description = "Number of meshes in structural field", .default_value = 1}),
            parameter<int>(
                "NUMALEDIS", {.description = "Number of meshes in ale field", .default_value = 1}),
            parameter<int>("NUMARTNETDIS",
                {.description = "Number of meshes in arterial network field", .default_value = 1}),
            parameter<int>("NUMTHERMDIS",
                {.description = "Number of meshes in thermal field", .default_value = 1}),
            parameter<int>("NUMAIRWAYSDIS",
                {.description = "Number of meshes in reduced dimensional airways network field",
                    .default_value = 1})},
        {.required = false}));

    specs.push_back(group("PROBLEM SIZE",
        {

            parameter<int>("DIM", {.description = "2d or 3d problem", .default_value = 3}),

            // deactivate all the following (unused) parameters one day
            // they are nice as general info in the input file but should not
            // read into a parameter list. Misuse is possible
            parameter<int>(
                "ELEMENTS", {.description = "Total number of elements", .default_value = 0}),

            parameter<int>("NODES", {.description = "Total number of nodes", .default_value = 0}),

            parameter<int>(
                "NPATCHES", {.description = "number of nurbs patches", .default_value = 0}),

            parameter<int>("MATERIALS", {.description = "number of materials", .default_value = 0}),
            parameter<int>("NUMDF",
                {.description = "maximum number of degrees of freedom", .default_value = 3})},
        {.required = false}));

    specs.push_back(group("PROBLEM TYPE",
        {

            deprecated_selection<Core::ProblemType>("PROBLEMTYPE",
                Core::string_to_problem_type_map(), {.description = "Type of the problem"}),

            deprecated_selection<Core::FE::ShapeFunctionType>("SHAPEFCT",
                Core::FE::string_to_shape_function_type_map(),
                {.description = "Defines the function spaces for the spatial approximation",
                    .default_value = Core::FE::ShapeFunctionType::polynomial}),

            parameter<int>("RESTART", {.description = "", .default_value = 0}),
            parameter<int>(
                "RANDSEED", {.description = "Set the random seed. If < 0 use current time.",
                                .default_value = -1})},
        {.required = true}));

    specs.push_back(group("NURBS",
        {

            parameter<bool>("DO_LS_DBC_PROJECTION",
                {.description = "Determines if a projection is needed for least "
                                "square Dirichlet boundary conditions.",
                    .default_value = false}),

            parameter<int>("SOLVER_LS_DBC_PROJECTION",
                {.description = "Number of linear solver for the projection of least squares "
                                "Dirichlet boundary conditions for NURBS discretizations",
                    .default_value = -1})},
        {.required = false}));

    const Core::Elements::ElementDefinition element_definition;
    auto all_possible_elements_spec = element_definition.element_data_spec();

    auto geometry_specs = all_of({
        parameter<std::filesystem::path>(
            "FILE", {.description = "Path to the external geometry file. Either absolute or "
                                    "relative to the input file."}),
        parameter<Core::IO::MeshInput::VerbosityLevel>("SHOW_INFO",
            {
                .description = "Choose verbosity of reporting element, node and set info for "
                               "the exodus file after reading.",
                .enum_value_description = Core::IO::MeshInput::describe,
                .default_value = Core::IO::MeshInput::VerbosityLevel::none,
            }),
        // Once we support more formats, we should add a "TYPE" parameter for the file format.
        list("ELEMENT_BLOCKS",
            all_of({
                parameter<int>(
                    "ID", {.description = "ID of the element block in the exodus file."}),
                all_possible_elements_spec,
            })),
    });

    const auto add_geometry_section = [&](auto& specs, const std::string& field_identifier)
    {
      specs.push_back(group(field_identifier + " GEOMETRY", {geometry_specs},
          {.description = "Settings related to the geometry of discretization " + field_identifier,
              .required = false}));
    };

    const auto add_domain_section =
        [spec = Core::IO::GridGenerator::RectangularCuboidInputs::spec()](
            auto& specs, const std::string& field_identifier)
    {
      specs.push_back(group(field_identifier + " DOMAIN", {spec},
          {.description = "Generate a mesh for discretization " + field_identifier,
              .required = false}));
    };

    const auto add_knotvector_section = [spec = Core::FE::Nurbs::Knotvector::spec()](
                                            auto& specs, const std::string& field_identifier)
    {
      specs.push_back(group(field_identifier + " KNOTVECTORS", {spec},
          {.description = "Knot vector description for NURBS discretization of " + field_identifier,
              .required = false}));
    };

    const std::vector known_fields = {"STRUCTURE", "FLUID", "LUBRICATION", "TRANSPORT",
        "TRANSPORT2", "ALE", "ARTERY", "REDUCED D AIRWAYS", "THERMO", "PERIODIC BOUNDINGBOX"};
    for (const auto& field : known_fields)
    {
      add_geometry_section(specs, field);
      add_domain_section(specs, field);
      add_knotvector_section(specs, field);
    }

    const auto describe = [](Core::IO::FieldDataBasis basis) -> std::string
    {
      switch (basis)
      {
        case Core::IO::FieldDataBasis::cells:
          return "Field data is defined on the cells";
        case Core::IO::FieldDataBasis::points:
          return "Field data is defined on the points";
        default:
          FOUR_C_THROW("Unknown FieldDataBasis enum value");
      }
    };

    specs.push_back(list("fields",
        all_of({parameter<std::string>(
                    "name", {.description = "Name of the field. This is used to refer to the field "
                                            "in other places. "
                                            "It is recommended to choose a descriptive name. The "
                                            "name must be unique "
                                            "across all fields."}),
            parameter<std::string>("discretization",
                {.description = "Name of the discretization to which this field belongs."}),
            selection<Core::IO::InputFieldSource>("source",
                {
                    group("separate_file",
                        {all_of({
                            parameter<std::filesystem::path>(
                                "file", {.description = "(Relative) path to the file containing "
                                                        "the field data."}),
                            parameter<std::optional<std::string>>("key",
                                {.description =
                                        "The key under which the field data is stored in the "
                                        "file. "
                                        "If not specified, the key is assumed to be equal to the "
                                        "name."}),
                        })},
                        {.description = "Read the field data from a separate json file."}),
                    group("from_mesh",
                        {all_of({
                            parameter<Core::IO::FieldDataBasis>("basis",
                                {.description = "The basis on which the field data is defined.",
                                    .enum_value_description = describe}),
                            parameter<std::optional<std::string>>("key",
                                {.description =
                                        "The key under which the field data is stored in the "
                                        "mesh. "
                                        "If not specified, the key is assumed to be equal to the "
                                        "name."}),
                        })},
                        {.description = "Read the field data from the mesh file."}),
                },
                {.description = "Source of the input field data.",
                    .store_selector = in_container<Core::IO::InputFieldSource>("source")})}),
        {
            .description = "Define a field that can be used in the simulation. "
                           "You can refer to a field by its name in other places.",
            .required = false,
        }));

    return specs;
  }
}  // namespace CommonParameters



std::vector<Core::IO::InputSpec> Global::valid_parameters()
{
  std::vector<Core::IO::InputSpec> specs;

  push_specs(specs, CommonParameters::valid_parameters());

  push_specs(specs, Inpar::Solid::valid_parameters());
  push_specs(specs, Inpar::IO::valid_parameters());
  push_specs(specs, Solid::IOMonitorStructureDBC::valid_parameters());
  push_specs(specs, Inpar::IORuntimeOutput::valid_parameters());
  push_specs(specs, Inpar::IORuntimeVTPStructure::valid_parameters());
  push_specs(specs, Mortar::valid_parameters());
  push_specs(specs, CONTACT::valid_parameters());
  push_specs(specs, Coupling::VolMortar::valid_parameters());
  push_specs(specs, Inpar::Wear::valid_parameters());
  push_specs(specs, Inpar::IORuntimeOutput::FLUID::valid_parameters());
  push_specs(specs, Inpar::IORuntimeOutput::Solid::valid_parameters());
  push_specs(specs, Beam::IORuntimeOutput::valid_parameters());
  push_specs(specs, BeamInteraction::valid_parameters());
  push_specs(specs, BrownianDynamics::valid_parameters());

  push_specs(specs, Inpar::Plasticity::valid_parameters());

  push_specs(specs, Thermo::valid_parameters());
  push_specs(specs, TSI::valid_parameters());

  push_specs(specs, Inpar::FLUID::valid_parameters());
  push_specs(specs, Inpar::LowMach::valid_parameters());
  push_specs(specs, Cut::valid_parameters());
  push_specs(specs, Inpar::XFEM::valid_parameters());
  push_specs(specs, Constraints::valid_parameters());

  push_specs(specs, Lubrication::valid_parameters());
  push_specs(specs, Inpar::ScaTra::valid_parameters());
  push_specs(specs, LevelSet::valid_parameters());
  push_specs(specs, ElCh::valid_parameters());
  push_specs(specs, Inpar::ElectroPhysiology::valid_parameters());
  push_specs(specs, STI::valid_parameters());

  push_specs(specs, Inpar::S2I::valid_parameters());
  push_specs(specs, FS3I::valid_parameters());
  push_specs(specs, FSI::valid_parameters());
  push_specs(specs, PoroElast::valid_parameters());
  push_specs(specs, PoroElastScaTra::valid_parameters());
  push_specs(specs, PoroPressureBased::valid_parameters_porofluid());
  push_specs(specs, PoroPressureBased::valid_parameters_porofluid_elast_scatra());
  push_specs(specs, PoroPressureBased::valid_parameters_porofluid_elast());
  push_specs(specs, EHL::valid_parameters());
  push_specs(specs, SSI::valid_parameters());
  push_specs(specs, SSTI::valid_parameters());
  push_specs(specs, ALE::valid_parameters());

  push_specs(specs, ArtDyn::valid_parameters());
  push_specs(specs, ArteryNetwork::valid_parameters());
  push_specs(specs, Inpar::BioFilm::valid_parameters());
  push_specs(specs, Airway::valid_parameters());
  push_specs(specs, ReducedLung::valid_parameters());
  push_specs(specs, ReducedLung1dPipeFlow::valid_parameters());
  push_specs(specs, Cardiovascular0DInput::valid_parameters());
  push_specs(specs, FPSI::valid_parameters());
  push_specs(specs, FBI::valid_parameters());

  push_specs(specs, Particle::valid_parameters());

  push_specs(specs, Geo::valid_parameters());
  push_specs(specs, Core::Binstrategy::valid_parameters());
  push_specs(specs, Core::GeometricSearch::valid_parameters());
  push_specs(specs, PaSI::valid_parameters());

  push_specs(specs, Core::Rebalance::valid_parameters());
  push_specs(specs, Core::LinearSolver::valid_parameters());
  push_specs(specs, NOX::valid_parameters());

  return specs;
}



FOUR_C_NAMESPACE_CLOSE
