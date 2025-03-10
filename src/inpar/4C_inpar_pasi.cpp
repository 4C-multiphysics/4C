// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_pasi.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | set valid parameters for pasi                                             |
 *---------------------------------------------------------------------------*/
void Inpar::PaSI::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs pasidyn{"PASI DYNAMIC"};

  // time loop control
  pasidyn.specs.emplace_back(parameter<int>(
      "RESULTSEVERY", {.description = "Increment for writing solution", .default_value = 1}));
  pasidyn.specs.emplace_back(parameter<int>(
      "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}));
  pasidyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.01}));
  pasidyn.specs.emplace_back(parameter<int>(
      "NUMSTEP", {.description = "Total number of Timesteps", .default_value = 100}));
  pasidyn.specs.emplace_back(
      parameter<double>("MAXTIME", {.description = "Total simulation time", .default_value = 1.0}));

  // type of partitioned coupling
  Core::Utils::string_to_integral_parameter<PartitionedCouplingType>("COUPLING",
      "partitioned_onewaycoup",
      "partitioned coupling strategies for particle structure interaction",
      tuple<std::string>("partitioned_onewaycoup", "partitioned_twowaycoup",
          "partitioned_twowaycoup_disprelax", "partitioned_twowaycoup_disprelaxaitken"),
      tuple<PartitionedCouplingType>(partitioned_onewaycoup, partitioned_twowaycoup,
          partitioned_twowaycoup_disprelax, partitioned_twowaycoup_disprelaxaitken),
      pasidyn);

  // partitioned iteration dependent parameters
  pasidyn.specs.emplace_back(parameter<int>(
      "ITEMAX", {.description = "maximum number of partitioned iterations over fields",
                    .default_value = 10}));

  pasidyn.specs.emplace_back(parameter<double>(
      "CONVTOLSCALEDDISP", {.description = "tolerance of dof and dt scaled interface displacement "
                                           "increments in partitioned iterations",
                               .default_value = -1.0}));

  pasidyn.specs.emplace_back(parameter<double>("CONVTOLRELATIVEDISP",
      {.description =
              "tolerance of relative interface displacement increments in partitioned iterations",
          .default_value = -1.0}));

  pasidyn.specs.emplace_back(parameter<double>("CONVTOLSCALEDFORCE",
      {.description =
              "tolerance of dof and dt scaled interface force increments in partitioned iterations",
          .default_value = -1.0}));

  pasidyn.specs.emplace_back(parameter<double>("CONVTOLRELATIVEFORCE",
      {.description = "tolerance of relative interface force increments in partitioned iterations",
          .default_value = -1.0}));

  pasidyn.specs.emplace_back(parameter<bool>("IGNORE_CONV_CHECK",
      {.description = "ignore convergence check and proceed simulation", .default_value = false}));

  // parameters for relaxation
  pasidyn.specs.emplace_back(parameter<double>(
      "STARTOMEGA", {.description = "fixed relaxation parameter", .default_value = 1.0}));
  pasidyn.specs.emplace_back(parameter<double>("MAXOMEGA",
      {.description = "largest omega allowed for Aitken relaxation", .default_value = 10.0}));
  pasidyn.specs.emplace_back(parameter<double>("MINOMEGA",
      {.description = "smallest omega allowed for Aitken relaxation", .default_value = 0.1}));

  pasidyn.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
