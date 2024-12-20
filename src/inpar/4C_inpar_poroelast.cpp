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


void Inpar::PoroElast::set_valid_parameters(Teuchos::ParameterList& list)
{
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& poroelastdyn =
      list.sublist("POROELASTICITY DYNAMIC", false, "Poroelasticity");

  // Coupling strategy for (monolithic) porous media solvers
  setStringToIntegralParameter<Inpar::PoroElast::SolutionSchemeOverFields>("COUPALGO",
      "poro_monolithic", "Coupling strategies for poroelasticity solvers",
      tuple<std::string>("poro_partitioned", "poro_monolithic", "poro_monolithicstructuresplit",
          "poro_monolithicfluidsplit", "poro_monolithicnopenetrationsplit",
          "poro_monolithicmeshtying"),
      tuple<Inpar::PoroElast::SolutionSchemeOverFields>(Partitioned, Monolithic,
          Monolithic_structuresplit, Monolithic_fluidsplit, Monolithic_nopenetrationsplit,
          Monolithic_meshtying),
      &poroelastdyn);

  // physical type of poro fluid flow (incompressible, varying density, loma, Boussinesq
  // approximation)
  setStringToIntegralParameter<Inpar::FLUID::PhysicalType>("PHYSICAL_TYPE", "Poro",
      "Physical Type of Porofluid", tuple<std::string>("Poro", "Poro_P1"),
      tuple<Inpar::FLUID::PhysicalType>(Inpar::FLUID::poro, Inpar::FLUID::poro_p1), &poroelastdyn);

  // physical type of poro fluid flow (incompressible, varying density, loma, Boussinesq
  // approximation)
  setStringToIntegralParameter<Inpar::PoroElast::TransientEquationsOfPoroFluid>("TRANSIENT_TERMS",
      "all", "which equation includes transient terms",
      tuple<std::string>("none", "momentum", "continuity", "all"),
      tuple<Inpar::PoroElast::TransientEquationsOfPoroFluid>(
          transient_none, transient_momentum_only, transient_continuity_only, transient_all),
      &poroelastdyn);

  // Output type
  Core::Utils::int_parameter(
      "RESTARTEVRY", 1, "write restart possibility every RESTARTEVRY steps", &poroelastdyn);

  // Time loop control
  Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of Timesteps", &poroelastdyn);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "total simulation time", &poroelastdyn);
  Core::Utils::double_parameter("TIMESTEP", 0.05, "time step size dt", &poroelastdyn);
  Core::Utils::int_parameter(
      "ITEMAX", 10, "maximum number of iterations over fields", &poroelastdyn);
  Core::Utils::int_parameter(
      "ITEMIN", 1, "minimal number of iterations over fields", &poroelastdyn);
  Core::Utils::int_parameter("RESULTSEVRY", 1, "increment for writing solution", &poroelastdyn);

  // Iterationparameters
  Core::Utils::double_parameter("TOLRES_GLOBAL", 1e-8,
      "tolerance in the residual norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLINC_GLOBAL", 1e-8,
      "tolerance in the increment norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLRES_DISP", 1e-8,
      "tolerance in the residual norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLINC_DISP", 1e-8,
      "tolerance in the increment norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLRES_PORO", 1e-8,
      "tolerance in the residual norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLINC_PORO", 1e-8,
      "tolerance in the increment norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter(
      "TOLRES_VEL", 1e-8, "tolerance in the residual norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLINC_VEL", 1e-8,
      "tolerance in the increment norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLRES_PRES", 1e-8,
      "tolerance in the residual norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLINC_PRES", 1e-8,
      "tolerance in the increment norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter("TOLRES_NCOUP", 1e-8,
      "tolerance in the residual norm for the Newton iteration", &poroelastdyn);
  Core::Utils::double_parameter(
      "POROTIMEFAC", 1.0, "time factor for poroelasticity problem", &poroelastdyn);

  setStringToIntegralParameter<Inpar::PoroElast::ConvNorm>("NORM_INC", "AbsSingleFields",
      "type of norm for primary variables convergence check",
      tuple<std::string>("AbsGlobal", "AbsSingleFields"),
      tuple<Inpar::PoroElast::ConvNorm>(convnorm_abs_global, convnorm_abs_singlefields),
      &poroelastdyn);

  setStringToIntegralParameter<Inpar::PoroElast::ConvNorm>("NORM_RESF", "AbsSingleFields",
      "type of norm for residual convergence check",
      tuple<std::string>("AbsGlobal", "AbsSingleFields"),
      tuple<Inpar::PoroElast::ConvNorm>(convnorm_abs_global, convnorm_abs_singlefields),
      &poroelastdyn);

  setStringToIntegralParameter<Inpar::PoroElast::BinaryOp>("NORMCOMBI_RESFINC", "And",
      "binary operator to combine primary variables and residual force values",
      tuple<std::string>("And", "Or"), tuple<Inpar::PoroElast::BinaryOp>(bop_and, bop_or),
      &poroelastdyn);

  setStringToIntegralParameter<Inpar::PoroElast::VectorNorm>("VECTORNORM_RESF", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<Inpar::PoroElast::VectorNorm>(norm_l1, norm_l1_scaled, norm_l2, norm_rms, norm_inf),
      &poroelastdyn);

  setStringToIntegralParameter<Inpar::PoroElast::VectorNorm>("VECTORNORM_INC", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<Inpar::PoroElast::VectorNorm>(norm_l1, norm_l1_scaled, norm_l2, norm_rms, norm_inf),
      &poroelastdyn);

  Core::Utils::bool_parameter(
      "SECONDORDER", "Yes", "Second order coupling at the interface.", &poroelastdyn);

  Core::Utils::bool_parameter("CONTIPARTINT", "No",
      "Partial integration of porosity gradient in continuity equation", &poroelastdyn);

  Core::Utils::bool_parameter("CONTACTNOPEN", "No",
      "No-Penetration Condition on active contact surface in case of poro contact problem!",
      &poroelastdyn);

  Core::Utils::bool_parameter("MATCHINGGRID", "Yes", "is matching grid", &poroelastdyn);

  Core::Utils::bool_parameter("CONVECTIVE_TERM", "No", "convective term ", &poroelastdyn);

  // number of linear solver used for poroelasticity
  Core::Utils::int_parameter("LINEAR_SOLVER", -1,
      "number of linear solver used for poroelasticity problems", &poroelastdyn);

  // flag for equilibration of global system of equations
  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION", "none",
      "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_full,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag),
      &poroelastdyn);
}

FOUR_C_NAMESPACE_CLOSE
