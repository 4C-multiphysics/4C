// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "4C_porofluid_pressure_based_elast_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_porofluid_pressure_based_input.hpp"

FOUR_C_NAMESPACE_OPEN


void PoroPressureBased::set_valid_parameters_porofluid_elast(
    std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  // ----------------------------------------------------------------------
  // (1) general control parameters
  list["POROMULTIPHASE DYNAMIC"] = group("POROMULTIPHASE DYNAMIC",
      {

          // Output type
          parameter<int>(
              "RESTARTEVERY", {.description = "write restart possibility every RESTARTEVERY steps",
                                  .default_value = 1}),
          // Time loop control
          parameter<int>(
              "NUMSTEP", {.description = "maximum number of Timesteps", .default_value = 200}),
          parameter<double>(
              "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}),

          parameter<double>(
              "TIMESTEP", {.description = "time step size dt", .default_value = -1.0}),
          parameter<int>("RESULTSEVERY",
              {.description = "increment for writing solution", .default_value = 1}),
          parameter<int>("ITEMAX",
              {.description = "maximum number of iterations over fields", .default_value = 10}),

          // here the computation of the structure can be skipped, this is helpful if only
          // fluid-scatra
          // coupling should be calculated
          parameter<bool>(
              "SOLVE_STRUCTURE", {.description = "Flag to skip computation of structural field",
                                     .default_value = true}),


          // Coupling strategy for solvers
          deprecated_selection<SolutionSchemePorofluidElast>("COUPALGO",
              {
                  {"twoway_partitioned", SolutionSchemePorofluidElast::twoway_partitioned},
                  {"twoway_monolithic", SolutionSchemePorofluidElast::twoway_monolithic},
              },
              {.description = "Coupling strategies for porofluid-elasticity solvers",
                  .default_value = SolutionSchemePorofluidElast::twoway_partitioned}),

          // coupling with 1D artery network active
          parameter<bool>("ARTERY_COUPLING",
              {.description = "Coupling with 1D blood vessels.", .default_value = false})},
      {.defaultable =
              true});  // ----------------------------------------------------------------------
  // (2) monolithic parameters
  list["POROMULTIPHASE DYNAMIC/MONOLITHIC"] = group("POROMULTIPHASE DYNAMIC/MONOLITHIC",
      {

          // convergence tolerances for monolithic coupling
          parameter<double>("TOLRES_GLOBAL",
              {.description = "tolerance in the residual norm for the Newton iteration",
                  .default_value = 1e-8}),
          parameter<double>("TOLINC_GLOBAL",
              {.description = "tolerance in the increment norm for the Newton iteration",
                  .default_value = 1e-8}),

          // number of linear solver used for poroelasticity
          parameter<int>("LINEAR_SOLVER",
              {.description = "number of linear solver used for poroelasticity problems",
                  .default_value = -1}),

          // parameters for finite difference check
          parameter<FdCheck>(
              "FDCHECK", {.description = "flag for finite difference check: none or global",
                             .default_value = FdCheck::none}),

          deprecated_selection<VectorNorm>("VECTORNORM_RESF",
              {
                  {"L1", VectorNorm::l1},
                  {"L1_Scaled", VectorNorm::l1_scaled},
                  {"L2", VectorNorm::l2},
                  {"Rms", VectorNorm::rms},
                  {"Inf", VectorNorm::inf},
              },
              {.description = "type of norm to be applied to residuals",
                  .default_value = VectorNorm::l2}),

          deprecated_selection<VectorNorm>("VECTORNORM_INC",
              {
                  {"L1", VectorNorm::l1},
                  {"L1_Scaled", VectorNorm::l1_scaled},
                  {"L2", VectorNorm::l2},
                  {"Rms", VectorNorm::rms},
                  {"Inf", VectorNorm::inf},
              },
              {.description = "type of norm to be applied to residuals",
                  .default_value = VectorNorm::l2}),

          // flag for equilibration of global system of equations
          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
              {.description = "flag for equilibration of global system of equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),

          // convergence criteria adaptivity --> note ADAPTCONV_BETTER set pretty small
          parameter<bool>("ADAPTCONV", {.description = "Switch on adaptive control of linear "
                                                       "solver tolerance for nonlinear solution",
                                           .default_value = false}),
          parameter<double>("ADAPTCONV_BETTER",
              {.description =
                      "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
                  .default_value = 0.001})},
      {.defaultable = true});

  // ----------------------------------------------------------------------
  // (3) partitioned parameters
  list["POROMULTIPHASE DYNAMIC/PARTITIONED"] = group("POROMULTIPHASE DYNAMIC/PARTITIONED",
      {

          // convergence tolerance of outer iteration loop
          parameter<double>(
              "CONVTOL", {.description = "tolerance for convergence check of outer iteration",
                             .default_value = 1e-6}),

          // flag for relaxation of partitioned scheme
          deprecated_selection<RelaxationMethods>("RELAXATION",
              {
                  {"none", RelaxationMethods::none},
                  {"Constant", RelaxationMethods::constant},
                  {"Aitken", RelaxationMethods::aitken},
              },
              {.description = "flag for relaxation of partitioned scheme",
                  .default_value = RelaxationMethods::none}),

          // parameters for relaxation of partitioned coupling
          parameter<double>(
              "STARTOMEGA", {.description = "fixed relaxation parameter", .default_value = 1.0}),
          parameter<double>(
              "MINOMEGA", {.description = "smallest omega allowed for Aitken relaxation",
                              .default_value = 0.1}),
          parameter<double>(
              "MAXOMEGA", {.description = "largest omega allowed for Aitken relaxation",
                              .default_value = 10.0})},
      {.defaultable = true});
}

FOUR_C_NAMESPACE_CLOSE