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

  Core::Utils::SectionSpecs pasidyn{"PASI DYNAMIC"};

  // time loop control
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", pasidyn);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", pasidyn);
  Core::Utils::double_parameter("TIMESTEP", 0.01, "Time increment dt", pasidyn);
  Core::Utils::int_parameter("NUMSTEP", 100, "Total number of Timesteps", pasidyn);
  Core::Utils::double_parameter("MAXTIME", 1.0, "Total simulation time", pasidyn);

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
  Core::Utils::int_parameter(
      "ITEMAX", 10, "maximum number of partitioned iterations over fields", pasidyn);

  Core::Utils::double_parameter("CONVTOLSCALEDDISP", -1.0,
      "tolerance of dof and dt scaled interface displacement increments in partitioned iterations",
      pasidyn);

  Core::Utils::double_parameter("CONVTOLRELATIVEDISP", -1.0,
      "tolerance of relative interface displacement increments in partitioned iterations", pasidyn);

  Core::Utils::double_parameter("CONVTOLSCALEDFORCE", -1.0,
      "tolerance of dof and dt scaled interface force increments in partitioned iterations",
      pasidyn);

  Core::Utils::double_parameter("CONVTOLRELATIVEFORCE", -1.0,
      "tolerance of relative interface force increments in partitioned iterations", pasidyn);

  Core::Utils::bool_parameter(
      "IGNORE_CONV_CHECK", "no", "ignore convergence check and proceed simulation", pasidyn);

  // parameters for relaxation
  Core::Utils::double_parameter("STARTOMEGA", 1.0, "fixed relaxation parameter", pasidyn);
  Core::Utils::double_parameter(
      "MAXOMEGA", 10.0, "largest omega allowed for Aitken relaxation", pasidyn);
  Core::Utils::double_parameter(
      "MINOMEGA", 0.1, "smallest omega allowed for Aitken relaxation", pasidyn);

  pasidyn.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
