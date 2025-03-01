// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_poromultiphase.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::POROMULTIPHASE::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  // ----------------------------------------------------------------------
  // (1) general control parameters
  Core::Utils::SectionSpecs poromultiphasedyn{"POROMULTIPHASE DYNAMIC"};

  // Output type
  poromultiphasedyn.specs.emplace_back(parameter<int>("RESTARTEVERY",
      {.description = "write restart possibility every RESTARTEVERY steps", .default_value = 1}));
  // Time loop control
  poromultiphasedyn.specs.emplace_back(parameter<int>(
      "NUMSTEP", {.description = "maximum number of Timesteps", .default_value = 200}));
  poromultiphasedyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}));
  poromultiphasedyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "time step size dt", .default_value = -1.0}));
  poromultiphasedyn.specs.emplace_back(parameter<int>(
      "RESULTSEVERY", {.description = "increment for writing solution", .default_value = 1}));
  poromultiphasedyn.specs.emplace_back(parameter<int>(
      "ITEMAX", {.description = "maximum number of iterations over fields", .default_value = 10}));

  // here the computation of the structure can be skipped, this is helpful if only fluid-scatra
  // coupling should be calculated
  poromultiphasedyn.specs.emplace_back(parameter<bool>("SOLVE_STRUCTURE",
      {.description = "Flag to skip computation of structural field", .default_value = true}));


  // Coupling strategy for solvers
  Core::Utils::string_to_integral_parameter<SolutionSchemeOverFields>("COUPALGO",
      "twoway_partitioned", "Coupling strategies for poro multiphase solvers",
      tuple<std::string>("twoway_partitioned", "twoway_monolithic"),
      tuple<SolutionSchemeOverFields>(solscheme_twoway_partitioned, solscheme_twoway_monolithic),
      poromultiphasedyn);

  // coupling with 1D artery network active
  poromultiphasedyn.specs.emplace_back(parameter<bool>("ARTERY_COUPLING",
      {.description = "Coupling with 1D blood vessels.", .default_value = false}));

  poromultiphasedyn.move_into_collection(list);

  // ----------------------------------------------------------------------
  // (2) monolithic parameters
  Core::Utils::SectionSpecs poromultiphasedynmono{poromultiphasedyn, "MONOLITHIC"};

  // convergence tolerances for monolithic coupling
  poromultiphasedynmono.specs.emplace_back(parameter<double>(
      "TOLRES_GLOBAL", {.description = "tolerance in the residual norm for the Newton iteration",
                           .default_value = 1e-8}));
  poromultiphasedynmono.specs.emplace_back(parameter<double>(
      "TOLINC_GLOBAL", {.description = "tolerance in the increment norm for the Newton iteration",
                           .default_value = 1e-8}));

  // number of linear solver used for poroelasticity
  poromultiphasedynmono.specs.emplace_back(parameter<int>(
      "LINEAR_SOLVER", {.description = "number of linear solver used for poroelasticity problems",
                           .default_value = -1}));

  // parameters for finite difference check
  Core::Utils::string_to_integral_parameter<FdCheck>("FDCHECK", "none",
      "flag for finite difference check: none or global",
      tuple<std::string>("none",
          "global"),  // perform finite difference check on time integrator level
      tuple<FdCheck>(fdcheck_none, fdcheck_global), poromultiphasedynmono);

  Core::Utils::string_to_integral_parameter<VectorNorm>("VECTORNORM_RESF", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(Inpar::POROMULTIPHASE::norm_l1, Inpar::POROMULTIPHASE::norm_l1_scaled,
          Inpar::POROMULTIPHASE::norm_l2, Inpar::POROMULTIPHASE::norm_rms,
          Inpar::POROMULTIPHASE::norm_inf),
      poromultiphasedynmono);

  Core::Utils::string_to_integral_parameter<VectorNorm>("VECTORNORM_INC", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<VectorNorm>(Inpar::POROMULTIPHASE::norm_l1, Inpar::POROMULTIPHASE::norm_l1_scaled,
          Inpar::POROMULTIPHASE::norm_l2, Inpar::POROMULTIPHASE::norm_rms,
          Inpar::POROMULTIPHASE::norm_inf),
      poromultiphasedynmono);

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
      poromultiphasedynmono);

  // convergence criteria adaptivity --> note ADAPTCONV_BETTER set pretty small
  poromultiphasedynmono.specs.emplace_back(parameter<bool>("ADAPTCONV",
      {.description =
              "Switch on adaptive control of linear solver tolerance for nonlinear solution",
          .default_value = false}));
  poromultiphasedynmono.specs.emplace_back(parameter<double>("ADAPTCONV_BETTER",
      {.description = "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
          .default_value = 0.001}));

  poromultiphasedynmono.move_into_collection(list);

  // ----------------------------------------------------------------------
  // (3) partitioned parameters
  Core::Utils::SectionSpecs poromultiphasedynpart{poromultiphasedyn, "PARTITIONED"};

  // convergence tolerance of outer iteration loop
  poromultiphasedynpart.specs.emplace_back(parameter<double>(
      "CONVTOL", {.description = "tolerance for convergence check of outer iteration",
                     .default_value = 1e-6}));

  // flag for relaxation of partitioned scheme
  Core::Utils::string_to_integral_parameter<RelaxationMethods>("RELAXATION", "none",
      "flag for relaxation of partitioned scheme", tuple<std::string>("none", "Constant", "Aitken"),
      tuple<RelaxationMethods>(relaxation_none, relaxation_constant, relaxation_aitken),
      poromultiphasedynpart);

  // parameters for relaxation of partitioned coupling
  poromultiphasedynpart.specs.emplace_back(parameter<double>(
      "STARTOMEGA", {.description = "fixed relaxation parameter", .default_value = 1.0}));
  poromultiphasedynpart.specs.emplace_back(parameter<double>("MINOMEGA",
      {.description = "smallest omega allowed for Aitken relaxation", .default_value = 0.1}));
  poromultiphasedynpart.specs.emplace_back(parameter<double>("MAXOMEGA",
      {.description = "largest omega allowed for Aitken relaxation", .default_value = 10.0}));

  poromultiphasedynpart.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
