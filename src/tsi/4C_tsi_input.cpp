// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_tsi_input.hpp"

#include "4C_contact_input.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_timestepping_time_step_control.hpp"

FOUR_C_NAMESPACE_OPEN

std::vector<Core::IO::InputSpec> TSI::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("TSI DYNAMIC",
      {

          // coupling strategy for (partitioned and monolithic) TSI solvers
          deprecated_selection<SolutionSchemeOverFields>("COUPALGO",
              {
                  {"tsi_oneway", SolutionSchemeOverFields::OneWay},
                  {"tsi_sequstagg", SolutionSchemeOverFields::SequStagg},
                  {"tsi_iterstagg", SolutionSchemeOverFields::IterStagg},
                  {"tsi_iterstagg_aitken", SolutionSchemeOverFields::IterStaggAitken},
                  {"tsi_iterstagg_aitkenirons", SolutionSchemeOverFields::IterStaggAitkenIrons},
                  {"tsi_iterstagg_fixedrelax", SolutionSchemeOverFields::IterStaggFixedRel},
                  {"tsi_monolithic", SolutionSchemeOverFields::Monolithic},
              },
              {.description = "Coupling strategies for TSI solvers",
                  .default_value = SolutionSchemeOverFields::Monolithic}),


          parameter<bool>(
              "MATCHINGGRID", {.description = "is matching grid", .default_value = true}),

          // output type
          parameter<int>(
              "RESTARTEVERY", {.description = "write restart possibility every RESTARTEVERY steps",
                                  .default_value = 1}),

          // time loop control
          parameter<int>(
              "NUMSTEP", {.description = "maximum number of Timesteps", .default_value = 200}),
          parameter<double>(
              "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}),

          parameter<double>(
              "TIMESTEP", {.description = "time step size dt", .default_value = 0.05}),
          parameter<int>("ITEMAX",
              {.description = "maximum number of iterations over fields", .default_value = 10}),
          parameter<int>("ITEMIN",
              {.description = "minimal number of iterations over fields", .default_value = 1}),
          parameter<int>("RESULTSEVERY",
              {.description = "increment for writing solution", .default_value = 1}),

          parameter<ConvNorm>("NORM_INC",
              {.description = "type of norm for convergence check of primary variables in TSI",
                  .default_value = ConvNorm::Abs})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic TSI */
  specs.push_back(group("TSI DYNAMIC/MONOLITHIC",
      {

          // convergence tolerance of tsi residual
          parameter<double>("CONVTOL",
              {.description = "tolerance for convergence check of TSI", .default_value = 1e-6}),
          // Iterationparameters
          parameter<double>("TOLINC",
              {.description = "tolerance for convergence check of TSI-increment in monolithic TSI",
                  .default_value = 1.0e-6}),

          parameter<ConvNorm>(
              "NORM_RESF", {.description = "type of norm for residual convergence check",
                               .default_value = ConvNorm::Abs}),

          deprecated_selection<BinaryOp>("NORMCOMBI_RESFINC",
              {
                  {"And", BinaryOp::bop_and},
                  {"Or", BinaryOp::bop_or},
                  {"Coupl_Or_Single", BinaryOp::bop_coupl_or_single},
                  {"Coupl_And_Single", BinaryOp::bop_coupl_and_single},
                  {"And_Single", BinaryOp::bop_and_single},
                  {"Or_Single", BinaryOp::bop_or_single},
              },
              {.description =
                      "binary operator to combine primary variables and residual force values",
                  .default_value = BinaryOp::bop_coupl_and_single}),

          parameter<VectorNorm>(
              "ITERNORM", {.description = "type of norm to be applied to residuals",
                              .default_value = VectorNorm::Rms}),

          parameter<NlnSolTech>("NLNSOL", {.description = "Nonlinear solution technique",
                                              .default_value = NlnSolTech::fullnewton}),


          parameter<double>(
              "PTCDT", {.description = "pseudo time step for pseudo-transient "
                                       "continuation (PTC) stabilised Newton procedure",
                           .default_value = 0.1}),

          // number of linear solver used for monolithic TSI
          parameter<int>("LINEAR_SOLVER",
              {.description = "number of linear solver used for monolithic TSI problems",
                  .default_value = -1}),

          // convergence criteria adaptivity of monolithic TSI solver
          parameter<bool>("ADAPTCONV", {.description = "Switch on adaptive control of linear "
                                                       "solver tolerance for nonlinear solution",
                                           .default_value = false}),
          parameter<double>("ADAPTCONV_BETTER",
              {.description =
                      "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
                  .default_value = 0.1}),

          parameter<bool>("INFNORMSCALING",
              {.description = "Scale blocks of matrix with row infnorm?", .default_value = true}),

          // merge TSI block matrix to enable use of direct solver in monolithic TSI
          // default: "No", i.e. use block matrix
          parameter<bool>("MERGE_TSI_BLOCK_MATRIX",
              {.description = "Merge TSI block matrix", .default_value = false}),

          deprecated_selection<LineSearch>("TSI_LINE_SEARCH",
              {
                  {"none", LineSearch::LS_none},
                  {"structure", LineSearch::LS_structure},
                  {"thermo", LineSearch::LS_thermo},
                  {"and", LineSearch::LS_and},
                  {"or", LineSearch::LS_or},
              },
              {.description = "line-search strategy", .default_value = LineSearch::LS_none}),

          parameter<DivContAct>("DIVERCONT",
              {.description = "Action to be taken in case the nonlinear solver does not "
                              "converge. If `adapt_step`, the time-step size is adapted "
                              "according to the `time_step_control` settings. `ignore` ignores "
                              "the non-convergence and continues anyway.",
                  .default_value = DivContAct::stop}),

          TimeStepping::TimeStepControlSettings::input_spec()},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned TSI */
  specs.push_back(group("TSI DYNAMIC/PARTITIONED",
      {

          parameter<CouplingVariable>(
              "COUPVARIABLE", {.description = "Coupling variable",
                                  .default_value = CouplingVariable::Displacement}),


          // Solver parameter for relaxation of iterative staggered partitioned TSI
          parameter<double>("MAXOMEGA",
              {.description =
                      "largest omega allowed for Aitken relaxation (0.0 means no constraint)",
                  .default_value = 0.0}),
          parameter<double>(
              "FIXEDOMEGA", {.description = "fixed relaxation parameter", .default_value = 1.0}),

          // convergence tolerance of outer iteration loop
          parameter<double>("CONVTOL",
              {.description =
                      "tolerance for convergence check of outer iteraiton within partitioned TSI",
                  .default_value = 1e-6}),
      },
      {.required = false}));

  /*----------------------------------------------------------------------*/
  /* parameters for tsi contact */
  specs.push_back(group("TSI CONTACT",
      {parameter<double>("HEATTRANSSLAVE",
           {.description = "Heat transfer parameter for slave side in thermal contact",
               .default_value = 0.0}),
          parameter<double>("HEATTRANSMASTER",
              {.description = "Heat transfer parameter for master side in thermal contact",
                  .default_value = 0.0}),
          parameter<double>("TEMP_DAMAGE",
              {.description =
                      "damage temperature at contact interface: friction coefficient zero there",
                  .default_value = 1.0e12}),

          parameter<double>(
              "TEMP_REF", {.description = "reference temperature at contact interface: "
                                          "friction coefficient equals the given value",
                              .default_value = 0.0}),

          parameter<double>("NITSCHE_THETA_TSI",
              {.description = "+1: symmetric, 0: non-symmetric, -1: skew-symmetric",
                  .default_value = 0.0}),

          parameter<CONTACT::NitscheWeighting>("NITSCHE_WEIGHTING_TSI",
              {.description = "how to weight consistency terms in Nitsche contact formulation",
                  .default_value = CONTACT::NitscheWeighting::harmonic}),

          parameter<bool>("NITSCHE_PENALTY_ADAPTIVE_TSI",
              {.description = "adapt penalty parameter after each converged time step",
                  .default_value = true}),

          parameter<double>("PENALTYPARAM_THERMO",
              {.description = "Penalty parameter for Nitsche solution strategy",
                  .default_value = 0.0})},
      {.required = false}));
  return specs;
}

FOUR_C_NAMESPACE_CLOSE