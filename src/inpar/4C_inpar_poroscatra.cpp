// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_poroscatra.hpp"

#include "4C_inpar_poroelast.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::PoroScaTra::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs poroscatradyn{"POROSCATRA CONTROL"};

  // Output type
  poroscatradyn.specs.emplace_back(parameter<int>("RESTARTEVERY",
      {.description = "write restart possibility every RESTARTEVERY steps", .default_value = 1}));
  // Time loop control
  poroscatradyn.specs.emplace_back(parameter<int>(
      "NUMSTEP", {.description = "maximum number of Timesteps", .default_value = 200}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}));
  poroscatradyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "time step size dt", .default_value = 0.05}));
  poroscatradyn.specs.emplace_back(parameter<int>(
      "RESULTSEVERY", {.description = "increment for writing solution", .default_value = 1}));
  poroscatradyn.specs.emplace_back(parameter<int>(
      "ITEMAX", {.description = "maximum number of iterations over fields", .default_value = 10}));
  poroscatradyn.specs.emplace_back(parameter<int>(
      "ITEMIN", {.description = "minimal number of iterations over fields", .default_value = 1}));

  // Iterationparameters
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLRES_GLOBAL", {.description = "tolerance in the residual norm for the Newton iteration",
                           .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLINC_GLOBAL", {.description = "tolerance in the increment norm for the Newton iteration",
                           .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLRES_DISP", {.description = "tolerance in the residual norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLINC_DISP", {.description = "tolerance in the increment norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLRES_VEL", {.description = "tolerance in the residual norm for the Newton iteration",
                        .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLINC_VEL", {.description = "tolerance in the increment norm for the Newton iteration",
                        .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLRES_PRES", {.description = "tolerance in the residual norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLINC_PRES", {.description = "tolerance in the increment norm for the Newton iteration",
                         .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLRES_SCALAR", {.description = "tolerance in the residual norm for the Newton iteration",
                           .default_value = 1e-8}));
  poroscatradyn.specs.emplace_back(parameter<double>(
      "TOLINC_SCALAR", {.description = "tolerance in the increment norm for the Newton iteration",
                           .default_value = 1e-8}));

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::ConvNorm>("NORM_INC",
      "AbsSingleFields", "type of norm for primary variables convergence check",
      tuple<std::string>("AbsGlobal", "AbsSingleFields"),
      tuple<Inpar::PoroElast::ConvNorm>(
          Inpar::PoroElast::convnorm_abs_global, Inpar::PoroElast::convnorm_abs_singlefields),
      poroscatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::ConvNorm>("NORM_RESF",
      "AbsSingleFields", "type of norm for residual convergence check",
      tuple<std::string>("AbsGlobal", "AbsSingleFields"),
      tuple<Inpar::PoroElast::ConvNorm>(
          Inpar::PoroElast::convnorm_abs_global, Inpar::PoroElast::convnorm_abs_singlefields),
      poroscatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::BinaryOp>("NORMCOMBI_RESFINC", "And",
      "binary operator to combine primary variables and residual force values",
      tuple<std::string>("And", "Or"),
      tuple<Inpar::PoroElast::BinaryOp>(Inpar::PoroElast::bop_and, Inpar::PoroElast::bop_or),
      poroscatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::VectorNorm>("VECTORNORM_RESF", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<Inpar::PoroElast::VectorNorm>(Inpar::PoroElast::norm_l1,
          Inpar::PoroElast::norm_l1_scaled, Inpar::PoroElast::norm_l2, Inpar::PoroElast::norm_rms,
          Inpar::PoroElast::norm_inf),
      poroscatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::PoroElast::VectorNorm>("VECTORNORM_INC", "L2",
      "type of norm to be applied to residuals",
      tuple<std::string>("L1", "L1_Scaled", "L2", "Rms", "Inf"),
      tuple<Inpar::PoroElast::VectorNorm>(Inpar::PoroElast::norm_l1,
          Inpar::PoroElast::norm_l1_scaled, Inpar::PoroElast::norm_l2, Inpar::PoroElast::norm_rms,
          Inpar::PoroElast::norm_inf),
      poroscatradyn);

  // number of linear solver used for poroelasticity
  poroscatradyn.specs.emplace_back(parameter<int>("LINEAR_SOLVER",
      {.description = "number of linear solver used for monolithic poroscatra problems",
          .default_value = -1}));

  // Coupling strategy for poroscatra solvers
  Core::Utils::string_to_integral_parameter<SolutionSchemeOverFields>("COUPALGO", "solid_to_scatra",
      "Coupling strategies for poroscatra solvers",
      tuple<std::string>("monolithic", "scatra_to_solid", "solid_to_scatra", "two_way"),
      tuple<SolutionSchemeOverFields>(
          Monolithic, Part_ScatraToPoro, Part_PoroToScatra, Part_TwoWay),
      poroscatradyn);

  poroscatradyn.specs.emplace_back(
      parameter<bool>("MATCHINGGRID", {.description = "is matching grid", .default_value = true}));

  poroscatradyn.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
