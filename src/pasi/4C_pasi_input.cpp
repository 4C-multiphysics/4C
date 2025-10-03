// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pasi_input.hpp"

#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | valid parameters for pasi                                             |
 *---------------------------------------------------------------------------*/
Core::IO::InputSpec PaSI::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  Core::IO::InputSpec spec = group("PASI DYNAMIC",
      {

          // time loop control
          parameter<int>("RESULTSEVERY",
              {.description = "Increment for writing solution", .default_value = 1}),
          parameter<int>(
              "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}),

          parameter<double>(
              "TIMESTEP", {.description = "Time increment dt", .default_value = 0.01}),
          parameter<int>(
              "NUMSTEP", {.description = "Total number of Timesteps", .default_value = 100}),

          parameter<double>(
              "MAXTIME", {.description = "Total simulation time", .default_value = 1.0}),

          // type of partitioned coupling
          parameter<PartitionedCouplingType>("COUPLING",
              {.description = "partitioned coupling strategies for particle structure interaction",
                  .default_value = partitioned_onewaycoup}),

          // partitioned iteration dependent parameters
          parameter<int>(
              "ITEMAX", {.description = "maximum number of partitioned iterations over fields",
                            .default_value = 10}),

          parameter<double>("CONVTOLSCALEDDISP",
              {.description = "tolerance of dof and dt scaled interface displacement "
                              "increments in partitioned iterations",
                  .default_value = -1.0}),

          parameter<double>(
              "CONVTOLRELATIVEDISP", {.description = "tolerance of relative interface displacement "
                                                     "increments in partitioned iterations",
                                         .default_value = -1.0}),

          parameter<double>("CONVTOLSCALEDFORCE",
              {.description = "tolerance of dof and dt scaled interface force "
                              "increments in partitioned iterations",
                  .default_value = -1.0}),

          parameter<double>("CONVTOLRELATIVEFORCE",
              {.description =
                      "tolerance of relative interface force increments in partitioned iterations",
                  .default_value = -1.0}),

          parameter<bool>("IGNORE_CONV_CHECK",
              {.description = "ignore convergence check and proceed simulation",
                  .default_value = false}),

          // parameters for relaxation
          parameter<double>(
              "STARTOMEGA", {.description = "fixed relaxation parameter", .default_value = 1.0}),
          parameter<double>(
              "MAXOMEGA", {.description = "largest omega allowed for Aitken relaxation",
                              .default_value = 10.0}),
          parameter<double>(
              "MINOMEGA", {.description = "smallest omega allowed for Aitken relaxation",
                              .default_value = 0.1})},
      {.required = false});
  return spec;
}

FOUR_C_NAMESPACE_CLOSE
