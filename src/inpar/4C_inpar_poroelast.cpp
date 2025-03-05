// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_poroelast.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::PoroElast::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs poroelastdyn{"POROELASTICITY DYNAMIC"};

  // Coupling strategy for (monolithic) porous media solvers
  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::SolutionSchemeOverFields>("COUPALGO",
      "poro_monolithic", "Coupling strategies for poroelasticity solvers",
      tuple<std::string>("poro_partitioned", "poro_monolithic", "poro_monolithicstructuresplit",
          "poro_monolithicfluidsplit", "poro_monolithicnopenetrationsplit",
          "poro_monolithicmeshtying"),
      tuple<Inpar::PoroElast::SolutionSchemeOverFields>(Partitioned, Monolithic,
          Monolithic_structuresplit, Monolithic_fluidsplit, Monolithic_nopenetrationsplit,
          Monolithic_meshtying),
      poroelastdyn);

  // physical type of poro fluid flow (incompressible, varying density, loma, Boussinesq
  // approximation)
  Core::Utils::string_to_integral_parameter<Inpar::FLUID::PhysicalType>("PHYSICAL_TYPE", "Poro",
      "Physical Type of Porofluid", tuple<std::string>("Poro", "Poro_P1"),
      tuple<Inpar::FLUID::PhysicalType>(Inpar::FLUID::poro, Inpar::FLUID::poro_p1), poroelastdyn);

  // physical type of poro fluid flow (incompressible, varying density, loma, Boussinesq
  // approximation)
  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::TransientEquationsOfPoroFluid>(
      "TRANSIENT_TERMS", "all", "which equation includes transient terms",
      tuple<std::string>("none", "momentum", "continuity", "all"),
      tuple<Inpar::PoroElast::TransientEquationsOfPoroFluid>(
          transient_none, transient_momentum_only, transient_continuity_only, transient_all),
      poroelastdyn);

  // Output type
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "write restart possibility every RESTARTEVERY steps", poroelastdyn);

  // Time loop control
  Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of Timesteps", poroelastdyn);
  poroelastdyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}));
  poroelastdyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "time step size dt", .default_value = 0.05}));
  Core::Utils::int_parameter(
      "ITEMAX", 10, "maximum number of iterations over fields", poroelastdyn);
  Core::Utils::int_parameter("ITEMIN", 1, "minimal number of iterations over fields", poroelastdyn);
  Core::Utils::int_parameter("RESULTSEVERY", 1, "increment for writing solution", poroelastdyn);

  // Iterationparameters
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLRES_GLOBAL", {.description = "tolerance in the residual norm for the Newton iteration",
                           .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLINC_GLOBAL", {.description = "tolerance in the increment norm for the Newton iteration",
                           .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLRES_DISP", {.description = "tolerance in the residual norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLINC_DISP", {.description = "tolerance in the increment norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLRES_PORO", {.description = "tolerance in the residual norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLINC_PORO", {.description = "tolerance in the increment norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLRES_VEL", {.description = "tolerance in the residual norm for the Newton iteration",
                        .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLINC_VEL", {.description = "tolerance in the increment norm for the Newton iteration",
                        .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLRES_PRES", {.description = "tolerance in the residual norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLINC_PRES", {.description = "tolerance in the increment norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroelastdyn.specs.emplace_back(parameter<double>(
      "TOLRES_NCOUP", {.description = "tolerance in the residual norm for the Newton iteration",
                          .default_value = 1e-8}));

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::ConvNorm>("NORM_INC",
      "AbsSingleFields", "type of norm for primary variables convergence check",
      tuple<std::string>("AbsGlobal", "AbsSingleFields"),
      tuple<Inpar::PoroElast::ConvNorm>(convnorm_abs_global, convnorm_abs_singlefields),
      poroelastdyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::ConvNorm>("NORM_RESF",
      "AbsSingleFields", "type of norm for residual convergence check",
      tuple<std::string>("AbsGlobal", "AbsSingleFields"),
      tuple<Inpar::PoroElast::ConvNorm>(convnorm_abs_global, convnorm_abs_singlefields),
      poroelastdyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::BinaryOp>("NORMCOMBI_RESFINC", "And",
      "binary operator to combine primary variables and residual force values",
      tuple<std::string>("And", "Or"), tuple<Inpar::PoroElast::BinaryOp>(bop_and, bop_or),
      poroelastdyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::VectorNorm>("VECTORNORM_RESF", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<Inpar::PoroElast::VectorNorm>(norm_l1, norm_l1_scaled, norm_l2, norm_rms, norm_inf),
      poroelastdyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::VectorNorm>("VECTORNORM_INC", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<Inpar::PoroElast::VectorNorm>(norm_l1, norm_l1_scaled, norm_l2, norm_rms, norm_inf),
      poroelastdyn);

  poroelastdyn.specs.emplace_back(parameter<bool>("SECONDORDER",
      {.description = "Second order coupling at the interface.", .default_value = true}));

  poroelastdyn.specs.emplace_back(parameter<bool>("CONTIPARTINT",
      {.description = "Partial integration of porosity gradient in continuity equation",
          .default_value = false}));

  poroelastdyn.specs.emplace_back(parameter<bool>("CONTACT_NO_PENETRATION",
      {.description =
              "No-Penetration Condition on active contact surface in case of poro contact problem!",
          .default_value = false}));

  poroelastdyn.specs.emplace_back(
      parameter<bool>("MATCHINGGRID", {.description = "is matching grid", .default_value = true}));

  poroelastdyn.specs.emplace_back(parameter<bool>(
      "CONVECTIVE_TERM", {.description = "convective term ", .default_value = false}));

  // number of linear solver used for poroelasticity
  Core::Utils::int_parameter("LINEAR_SOLVER", -1,
      "number of linear solver used for poroelasticity problems", poroelastdyn);

  // flag for equilibration of global system of equations
  Core::Utils::string_to_integral_parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
      "none", "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_full,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag),
      poroelastdyn);

  poroelastdyn.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
