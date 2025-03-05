// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_fs3i.hpp"

#include "4C_inpar_scatra.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::FS3I::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs fs3idyn{"FS3I DYNAMIC"};

  fs3idyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}));
  Core::Utils::int_parameter("NUMSTEP", 20, "Total number of time steps", fs3idyn);
  fs3idyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}));
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", fs3idyn);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", fs3idyn);
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::SolverType>("SCATRA_SOLVERTYPE",
      "nonlinear", "type of scalar transport solver", tuple<std::string>("linear", "nonlinear"),
      tuple<Inpar::ScaTra::SolverType>(
          Inpar::ScaTra::solvertype_linear_incremental, Inpar::ScaTra::solvertype_nonlinear),
      fs3idyn);
  fs3idyn.specs.emplace_back(parameter<bool>(
      "INF_PERM", {.description = "Flag for infinite permeability", .default_value = true}));
  std::vector<std::string> consthermpress_valid_input = {"No_energy", "No_mass", "Yes"};
  fs3idyn.specs.emplace_back(selection<std::string>("CONSTHERMPRESS", consthermpress_valid_input,
      {.description = "treatment of thermodynamic pressure in time", .default_value = "Yes"}));

  // number of linear solver used for fs3i problems
  Core::Utils::int_parameter(
      "COUPLED_LINEAR_SOLVER", -1, "number of linear solver used for fs3i problem", fs3idyn);
  Core::Utils::int_parameter(
      "LINEAR_SOLVER1", -1, "number of linear solver used for fluid problem", fs3idyn);
  Core::Utils::int_parameter(
      "LINEAR_SOLVER2", -1, "number of linear solver used for structural problem", fs3idyn);

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::ConvForm>("STRUCTSCAL_CONVFORM",
      "conservative", "form of convective term of structure scalar",
      tuple<std::string>("convective", "conservative"),
      tuple<Inpar::ScaTra::ConvForm>(
          Inpar::ScaTra::convform_convective, Inpar::ScaTra::convform_conservative),
      fs3idyn);

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::InitialField>("STRUCTSCAL_INITIALFIELD",
      "zero_field", "Initial Field for structure scalar transport problem",
      tuple<std::string>("zero_field", "field_by_function"),
      tuple<Inpar::ScaTra::InitialField>(
          Inpar::ScaTra::initfield_zero_field, Inpar::ScaTra::initfield_field_by_function),
      fs3idyn);

  Core::Utils::int_parameter("STRUCTSCAL_INITFUNCNO", -1,
      "function number for structure scalar transport initial field", fs3idyn);

  // Type of coupling strategy between structure and structure-scalar field
  Core::Utils::string_to_integral_parameter<VolumeCoupling>("STRUCTSCAL_FIELDCOUPLING",
      "volume_matching", "Type of coupling strategy between structure and structure-scalar field",
      tuple<std::string>("volume_matching", "volume_nonmatching"),
      tuple<VolumeCoupling>(coupling_match, coupling_nonmatch), fs3idyn);

  // Type of coupling strategy between fluid and fluid-scalar field
  Core::Utils::string_to_integral_parameter<VolumeCoupling>("FLUIDSCAL_FIELDCOUPLING",
      "volume_matching", "Type of coupling strategy between fluid and fluid-scalar field",
      tuple<std::string>("volume_matching", "volume_nonmatching"),
      tuple<VolumeCoupling>(coupling_match, coupling_nonmatch), fs3idyn);

  // type of scalar transport
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::ImplType>("FLUIDSCAL_SCATRATYPE",
      "ConvectionDiffusion", "Type of scalar transport problem",
      tuple<std::string>("Undefined", "ConvectionDiffusion", "Loma", "Advanced_Reaction",
          "Chemotaxis", "Chemo_Reac"),
      tuple<Inpar::ScaTra::ImplType>(Inpar::ScaTra::impltype_undefined, Inpar::ScaTra::impltype_std,
          Inpar::ScaTra::impltype_loma, Inpar::ScaTra::impltype_advreac,
          Inpar::ScaTra::impltype_chemo, Inpar::ScaTra::impltype_chemoreac),
      fs3idyn);

  // Restart from FSI instead of FS3I
  fs3idyn.specs.emplace_back(parameter<bool>(
      "RESTART_FROM_PART_FSI", {.description = "restart from partitioned fsi problem (e.g. from "
                                               "prestress calculations) instead of fs3i",
                                   .default_value = false}));

  fs3idyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned FS3I */
  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs fs3idynpart{fs3idyn, "PARTITIONED"};

  // Coupling strategy for partitioned FS3I
  Core::Utils::string_to_integral_parameter<SolutionSchemeOverFields>("COUPALGO", "fs3i_IterStagg",
      "Coupling strategies for FS3I solvers",
      tuple<std::string>("fs3i_SequStagg", "fs3i_IterStagg"),
      tuple<SolutionSchemeOverFields>(fs3i_SequStagg, fs3i_IterStagg), fs3idynpart);

  // convergence tolerance of outer iteration loop
  fs3idynpart.specs.emplace_back(parameter<double>("CONVTOL",
      {.description = "tolerance for convergence check of outer iteration within partitioned FS3I",
          .default_value = 1e-6}));

  Core::Utils::int_parameter("ITEMAX", 10, "Maximum number of outer iterations", fs3idynpart);

  fs3idynpart.move_into_collection(list);

  /*----------------------------------------------------------------------  */
  /* parameters for stabilization of the structure-scalar field             */
  /*----------------------------------------------------------------------  */

  /// HACK!
  /// reuse the parameters from scatra
  FOUR_C_ASSERT(list.contains("SCALAR TRANSPORT DYNAMIC/STABILIZATION"),
      "Internal error: Hacky FS3I requires scatra to already have been read.");

  list["FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION"] =
      list["SCALAR TRANSPORT DYNAMIC/STABILIZATION"];
}

FOUR_C_NAMESPACE_CLOSE
