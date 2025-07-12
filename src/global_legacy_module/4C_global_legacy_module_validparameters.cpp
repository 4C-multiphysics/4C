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
#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_binstrategy_input.hpp"
#include "4C_browniandyn_input.hpp"
#include "4C_contact_input.hpp"
#include "4C_cut_input.hpp"
#include "4C_ehl_input.hpp"
#include "4C_elch_input.hpp"
#include "4C_fbi_input.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_geometric_search_input.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_inpar_cardiac_monodomain.hpp"
#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_inpar_constraint_framework.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_inpar_fs3i.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_io.hpp"
#include "4C_inpar_IO_runtime_output.hpp"
#include "4C_inpar_IO_runtime_output_fluid.hpp"
#include "4C_inpar_IO_runtime_vtk_output_structure.hpp"
#include "4C_inpar_IO_runtime_vtp_output_structure.hpp"
#include "4C_inpar_levelset.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_inpar_mpc_rve.hpp"
#include "4C_inpar_particle.hpp"
#include "4C_inpar_pasi.hpp"
#include "4C_inpar_plasticity.hpp"
#include "4C_inpar_poroelast.hpp"
#include "4C_inpar_poroscatra.hpp"
#include "4C_inpar_problemtype.hpp"
#include "4C_inpar_rebalance.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_searchtree.hpp"
#include "4C_inpar_solver.hpp"
#include "4C_inpar_solver_nonlin.hpp"
#include "4C_inpar_ssi.hpp"
#include "4C_inpar_ssti.hpp"
#include "4C_inpar_sti.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_inpar_volmortar.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io_exodus.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_pstream.hpp"
#include "4C_lubrication_input.hpp"
#include "4C_porofluid_pressure_based_elast_input.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_input.hpp"
#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_red_airways_input.hpp"
#include "4C_structure_new_monitor_dbc_input.hpp"
#include "4C_thermo_input.hpp"
#include "4C_tsi_input.hpp"

#include <Teuchos_any.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterEntryValidator.hpp>
#include <Teuchos_StrUtils.hpp>
#include <Teuchos_TypeNameTraits.hpp>

#include <iostream>
#include <string>


FOUR_C_NAMESPACE_OPEN



std::map<std::string, Core::IO::InputSpec> Global::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  std::map<std::string, Core::IO::InputSpec> specs;
  /*----------------------------------------------------------------------*/
  specs["DISCRETISATION"] = group("DISCRETISATION",
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
      {.required = false});

  specs["PROBLEM SIZE"] = group("PROBLEM SIZE",
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
      {.required = false});
  Inpar::PROBLEMTYPE::set_valid_parameters(specs);

  /*----------------------------------------------------------------------*/

  specs["NURBS"] = group("NURBS",
      {

          parameter<bool>("DO_LS_DBC_PROJECTION",
              {.description = "Determines if a projection is needed for least "
                              "square Dirichlet boundary conditions.",
                  .default_value = false}),

          parameter<int>("SOLVER_LS_DBC_PROJECTION",
              {.description = "Number of linear solver for the projection of least squares "
                              "Dirichlet boundary conditions for NURBS discretizations",
                  .default_value = -1})},
      {.required = false});

  const auto add_geometry_section = [](auto& specs, const std::string& field_identifier)
  {
    specs[field_identifier + " GEOMETRY"] = group(field_identifier + " GEOMETRY",
        {
            parameter<std::filesystem::path>(
                "FILE", {.description = "Path to the exodus geometry file. Either absolute or "
                                        "relative to the input file."}),
            parameter<Core::IO::Exodus::VerbosityLevel>("SHOW_INFO",
                {.description =
                        "Print element, node and set info for the exodus file after reading.",
                    .default_value = Core::IO::Exodus::VerbosityLevel::none}),

            // Once we support more formats, we should add a "TYPE" parameter for the file format.
            list("ELEMENT_BLOCKS",
                all_of({
                    parameter<int>(
                        "ID", {.description = "ID of the element block in the exodus file."}),
                    parameter<std::string>("ELEMENT_NAME",
                        {.description =
                                "The name of the element that should be assigned to the block."}),
                    parameter<std::string>("ELEMENT_DATA",
                        {.description = "A dat-style string of parameters for the element."}),
                })),
        },
        {.description = "Settings related to the geometry of discretization " + field_identifier,
            .required = false});
  };

  const std::vector known_fields = {"STRUCTURE", "FLUID", "LUBRICATION", "TRANSPORT", "TRANSPORT2",
      "ALE", "ARTERY", "REDUCED D AIRWAYS", "THERMO", "PERIODIC BOUNDINGBOX"};
  for (const auto& field : known_fields)
  {
    add_geometry_section(specs, field);
  }

  specs["fields"] = list("fields",
      all_of({
          parameter<std::string>("name",
              {.description =
                      "Name of the field. This is used to refer to the field in other places. "
                      "It is recommended to choose a descriptive name. The name must be unique "
                      "across all fields."}),
          parameter<std::string>("discretization",
              {.description = "Name of the discretization to which this field belongs."}),
          parameter<std::filesystem::path>(
              "file", {.description = "(Relative) path to the file containing the field data."}),
          parameter<std::optional<std::string>>("key",
              {.description = "The key under which the field data is stored in the file. "
                              "If not specified, the key is assumed to be equal to the name."}),
      }),
      {
          .description = "Define a field that can be used in the simulation. "
                         "You can refer to a field by its name in other places.",
          .required = false,
      });

  Inpar::Solid::set_valid_parameters(specs);
  Inpar::IO::set_valid_parameters(specs);
  Solid::IOMonitorStructureDBC::set_valid_parameters(specs);
  Inpar::IORuntimeOutput::set_valid_parameters(specs);
  Inpar::IORuntimeVTPStructure::set_valid_parameters(specs);
  Inpar::Mortar::set_valid_parameters(specs);
  CONTACT::set_valid_parameters(specs);
  Inpar::VolMortar::set_valid_parameters(specs);
  Inpar::Wear::set_valid_parameters(specs);
  Inpar::IORuntimeOutput::FLUID::set_valid_parameters(specs);
  Inpar::IORuntimeOutput::Solid::set_valid_parameters(specs);
  Beam::IORuntimeOutput::set_valid_parameters(specs);
  BeamContact::set_valid_parameters(specs);
  BeamPotential::set_valid_parameters(specs);
  Inpar::BeamInteraction::set_valid_parameters(specs);
  Inpar::RveMpc::set_valid_parameters(specs);
  BrownianDynamics::set_valid_parameters(specs);

  Inpar::Plasticity::set_valid_parameters(specs);

  Thermo::set_valid_parameters(specs);
  TSI::set_valid_parameters(specs);

  Inpar::FLUID::set_valid_parameters(specs);
  Inpar::LowMach::set_valid_parameters(specs);
  Cut::set_valid_parameters(specs);
  Inpar::XFEM::set_valid_parameters(specs);
  Inpar::Constraints::set_valid_parameters(specs);

  Lubrication::set_valid_parameters(specs);
  Inpar::ScaTra::set_valid_parameters(specs);
  Inpar::LevelSet::set_valid_parameters(specs);
  ElCh::set_valid_parameters(specs);
  Inpar::ElectroPhysiology::set_valid_parameters(specs);
  Inpar::STI::set_valid_parameters(specs);

  Inpar::S2I::set_valid_parameters(specs);
  Inpar::FS3I::set_valid_parameters(specs);
  Inpar::PoroElast::set_valid_parameters(specs);
  Inpar::PoroScaTra::set_valid_parameters(specs);
  PoroPressureBased::set_valid_parameters_porofluid(specs);
  PoroPressureBased::set_valid_parameters_porofluid_elast_scatra(specs);
  PoroPressureBased::set_valid_parameters_porofluid_elast(specs);
  EHL::set_valid_parameters(specs);
  Inpar::SSI::set_valid_parameters(specs);
  Inpar::SSTI::set_valid_parameters(specs);
  ALE::set_valid_parameters(specs);
  Inpar::FSI::set_valid_parameters(specs);

  ArtDyn::set_valid_parameters(specs);
  ArteryNetwork::set_valid_parameters(specs);
  Inpar::BioFilm::set_valid_parameters(specs);
  Airway::set_valid_parameters(specs);
  Inpar::Cardiovascular0D::set_valid_parameters(specs);
  Inpar::FPSI::set_valid_parameters(specs);
  FBI::set_valid_parameters(specs);

  Inpar::PARTICLE::set_valid_parameters(specs);

  Inpar::Geo::set_valid_parameters(specs);
  Core::Binstrategy::set_valid_parameters(specs);
  Core::GeometricSearch::set_valid_parameters(specs);
  Inpar::PaSI::set_valid_parameters(specs);

  Inpar::Rebalance::set_valid_parameters(specs);
  Inpar::SOLVER::set_valid_parameters(specs);
  Inpar::NlnSol::set_valid_parameters(specs);

  return specs;
}



FOUR_C_NAMESPACE_CLOSE