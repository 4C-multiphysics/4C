// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_validparameters.hpp"

#include "4C_ale_input.hpp"
#include "4C_beamcontact_input.hpp"
#include "4C_browniandyn_input.hpp"
#include "4C_cut_input.hpp"
#include "4C_inpar_beaminteraction.hpp"
#include "4C_inpar_beampotential.hpp"
#include "4C_inpar_binningstrategy.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_inpar_cardiac_monodomain.hpp"
#include "4C_inpar_cardiovascular0d.hpp"
#include "4C_inpar_constraint_framework.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_ehl.hpp"
#include "4C_inpar_elch.hpp"
#include "4C_inpar_elemag.hpp"
#include "4C_inpar_fbi.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_inpar_fs3i.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_geometric_search.hpp"
#include "4C_inpar_io.hpp"
#include "4C_inpar_IO_monitor_structure_dbc.hpp"
#include "4C_inpar_IO_runtime_output.hpp"
#include "4C_inpar_IO_runtime_output_fluid.hpp"
#include "4C_inpar_IO_runtime_output_structure_beams.hpp"
#include "4C_inpar_IO_runtime_vtk_output_structure.hpp"
#include "4C_inpar_IO_runtime_vtp_output_structure.hpp"
#include "4C_inpar_levelset.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_inpar_mpc_rve.hpp"
#include "4C_inpar_particle.hpp"
#include "4C_inpar_pasi.hpp"
#include "4C_inpar_plasticity.hpp"
#include "4C_inpar_poroelast.hpp"
#include "4C_inpar_porofluidmultiphase.hpp"
#include "4C_inpar_poromultiphase.hpp"
#include "4C_inpar_poromultiphase_scatra.hpp"
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
#include "4C_inpar_tsi.hpp"
#include "4C_inpar_volmortar.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_inpar_xfem.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_pstream.hpp"
#include "4C_lubrication_input.hpp"
#include "4C_thermo_input.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_any.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterEntryValidator.hpp>
#include <Teuchos_StrUtils.hpp>
#include <Teuchos_TypeNameTraits.hpp>

#include <iostream>
#include <string>


FOUR_C_NAMESPACE_OPEN



void print_default_dat_header()
{
  auto map = Input::valid_parameters();
  Core::IO::print_dat(std::cout, map);
}

std::map<std::string, Core::IO::InputSpec> Input::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  std::map<std::string, Core::IO::InputSpec> list;
  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs discret{"DISCRETISATION"};

  discret.specs.emplace_back(parameter<int>(
      "NUMFLUIDDIS", {.description = "Number of meshes in fluid field", .default_value = 1}));
  discret.specs.emplace_back(parameter<int>(
      "NUMSTRUCDIS", {.description = "Number of meshes in structural field", .default_value = 1}));
  discret.specs.emplace_back(parameter<int>(
      "NUMALEDIS", {.description = "Number of meshes in ale field", .default_value = 1}));
  discret.specs.emplace_back(parameter<int>("NUMARTNETDIS",
      {.description = "Number of meshes in arterial network field", .default_value = 1}));
  discret.specs.emplace_back(parameter<int>(
      "NUMTHERMDIS", {.description = "Number of meshes in thermal field", .default_value = 1}));
  discret.specs.emplace_back(parameter<int>("NUMAIRWAYSDIS",
      {.description = "Number of meshes in reduced dimensional airways network field",
          .default_value = 1}));

  discret.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs size{"PROBLEM SIZE"};

  size.specs.emplace_back(
      parameter<int>("DIM", {.description = "2d or 3d problem", .default_value = 3}));

  // deactivate all the following (unused) parameters one day
  // they are nice as general info in the input file but should not
  // read into a parameter list. Misuse is possible
  size.specs.emplace_back(
      parameter<int>("ELEMENTS", {.description = "Total number of elements", .default_value = 0}));
  size.specs.emplace_back(
      parameter<int>("NODES", {.description = "Total number of nodes", .default_value = 0}));
  size.specs.emplace_back(
      parameter<int>("NPATCHES", {.description = "number of nurbs patches", .default_value = 0}));
  size.specs.emplace_back(
      parameter<int>("MATERIALS", {.description = "number of materials", .default_value = 0}));
  size.specs.emplace_back(parameter<int>(
      "NUMDF", {.description = "maximum number of degrees of freedom", .default_value = 3}));

  size.move_into_collection(list);

  Inpar::PROBLEMTYPE::set_valid_parameters(list);

  /*----------------------------------------------------------------------*/

  Core::Utils::SectionSpecs nurbs_param{"NURBS"};

  nurbs_param.specs.emplace_back(parameter<bool>(
      "DO_LS_DBC_PROJECTION", {.description = "Determines if a projection is needed for least "
                                              "square Dirichlet boundary conditions.",
                                  .default_value = false}));

  nurbs_param.specs.emplace_back(parameter<int>("SOLVER_LS_DBC_PROJECTION",
      {.description = "Number of linear solver for the projection of least squares "
                      "Dirichlet boundary conditions for NURBS discretizations",
          .default_value = -1}));

  nurbs_param.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* Finally call the problem-specific SetValidParameter functions        */
  /*----------------------------------------------------------------------*/

  Inpar::Solid::set_valid_parameters(list);
  Inpar::IO::set_valid_parameters(list);
  Inpar::IOMonitorStructureDBC::set_valid_parameters(list);
  Inpar::IORuntimeOutput::set_valid_parameters(list);
  Inpar::IORuntimeVTPStructure::set_valid_parameters(list);
  Inpar::Mortar::set_valid_parameters(list);
  Inpar::CONTACT::set_valid_parameters(list);
  Inpar::VolMortar::set_valid_parameters(list);
  Inpar::Wear::set_valid_parameters(list);
  Inpar::IORuntimeOutput::FLUID::set_valid_parameters(list);
  Inpar::IORuntimeOutput::Solid::set_valid_parameters(list);
  Inpar::IORuntimeOutput::Beam::set_valid_parameters(list);
  BeamContact::set_valid_parameters(list);
  Inpar::BeamPotential::set_valid_parameters(list);
  Inpar::BeamInteraction::set_valid_parameters(list);
  Inpar::RveMpc::set_valid_parameters(list);
  BrownianDynamics::set_valid_parameters(list);

  Inpar::Plasticity::set_valid_parameters(list);

  Thermo::set_valid_parameters(list);
  Inpar::TSI::set_valid_parameters(list);

  Inpar::FLUID::set_valid_parameters(list);
  Inpar::LowMach::set_valid_parameters(list);
  Cut::set_valid_parameters(list);
  Inpar::XFEM::set_valid_parameters(list);
  Inpar::CONSTRAINTS::set_valid_parameters(list);

  Lubrication::set_valid_parameters(list);
  Inpar::ScaTra::set_valid_parameters(list);
  Inpar::LevelSet::set_valid_parameters(list);
  Inpar::ElCh::set_valid_parameters(list);
  Inpar::ElectroPhysiology::set_valid_parameters(list);
  Inpar::STI::set_valid_parameters(list);

  Inpar::S2I::set_valid_parameters(list);
  Inpar::FS3I::set_valid_parameters(list);
  Inpar::PoroElast::set_valid_parameters(list);
  Inpar::PoroScaTra::set_valid_parameters(list);
  Inpar::POROMULTIPHASE::set_valid_parameters(list);
  Inpar::PoroMultiPhaseScaTra::set_valid_parameters(list);
  Inpar::POROFLUIDMULTIPHASE::set_valid_parameters(list);
  Inpar::EHL::set_valid_parameters(list);
  Inpar::SSI::set_valid_parameters(list);
  Inpar::SSTI::set_valid_parameters(list);
  ALE::set_valid_parameters(list);
  Inpar::FSI::set_valid_parameters(list);

  Inpar::ArtDyn::set_valid_parameters(list);
  Inpar::ArteryNetwork::set_valid_parameters(list);
  Inpar::BioFilm::set_valid_parameters(list);
  Inpar::ReducedLung::set_valid_parameters(list);
  Inpar::Cardiovascular0D::set_valid_parameters(list);
  Inpar::FPSI::set_valid_parameters(list);
  Inpar::FBI::set_valid_parameters(list);

  Inpar::PARTICLE::set_valid_parameters(list);

  Inpar::EleMag::set_valid_parameters(list);

  Inpar::Geo::set_valid_parameters(list);
  Inpar::BINSTRATEGY::set_valid_parameters(list);
  Inpar::GeometricSearch::set_valid_parameters(list);
  Inpar::PaSI::set_valid_parameters(list);

  Inpar::Rebalance::set_valid_parameters(list);
  Inpar::SOLVER::set_valid_parameters(list);
  Inpar::NlnSol::set_valid_parameters(list);

  return list;
}



FOUR_C_NAMESPACE_CLOSE
