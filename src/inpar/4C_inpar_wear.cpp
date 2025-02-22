// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_wear.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::Wear::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;

  /* parameters for wear */
  Core::Utils::SectionSpecs wear{"WEAR"};

  Core::Utils::string_to_integral_parameter<WearLaw>("WEARLAW", "None", "Type of wear law",
      tuple<std::string>("None", "none", "Archard", "archard"),
      tuple<WearLaw>(wear_none, wear_none, wear_archard, wear_archard), wear);

  Core::Utils::bool_parameter("MATCHINGGRID", true, "is matching grid", wear);

  Core::Utils::string_to_integral_parameter<WearShape>("WEAR_SHAPEFCN", "std",
      "Type of employed set of shape functions for wear",
      tuple<std::string>("Dual", "dual", "Standard", "standard", "std"),
      tuple<WearShape>(wear_shape_dual, wear_shape_dual, wear_shape_standard, wear_shape_standard,
          wear_shape_standard),
      wear);

  Core::Utils::double_parameter("WEARCOEFF", 0.0, "Wear coefficient for slave surface", wear);
  Core::Utils::double_parameter(
      "WEARCOEFF_MASTER", 0.0, "Wear coefficient for master surface", wear);
  Core::Utils::double_parameter(
      "WEAR_TIMERATIO", 1.0, "Time step ratio between wear and spatial time scale", wear);
  Core::Utils::double_parameter("SSSLIP", 1.0, "Fixed slip for steady state wear", wear);

  Core::Utils::bool_parameter("SSWEAR", false, "flag for steady state wear", wear);

  Core::Utils::string_to_integral_parameter<WearSide>("WEAR_SIDE", "slave",
      "Definition of wear side",
      tuple<std::string>("s", "slave", "Slave", "both", "slave_master", "sm"),
      tuple<WearSide>(wear_slave, wear_slave, wear_slave, wear_both, wear_both, wear_both), wear);

  Core::Utils::string_to_integral_parameter<WearType>("WEARTYPE", "internal_state",
      "Definition of wear algorithm",
      tuple<std::string>("intstate", "is", "internal_state", "primvar", "pv", "primary_variable"),
      tuple<WearType>(
          wear_intstate, wear_intstate, wear_intstate, wear_primvar, wear_primvar, wear_primvar),
      wear);

  Core::Utils::string_to_integral_parameter<WearTimInt>("WEARTIMINT", "explicit",
      "Definition of wear time integration",
      tuple<std::string>("explicit", "e", "expl", "implicit", "i", "impl"),
      tuple<WearTimInt>(wear_expl, wear_expl, wear_expl, wear_impl, wear_impl, wear_impl), wear);

  Core::Utils::string_to_integral_parameter<WearTimeScale>("WEAR_TIMESCALE", "equal",
      "Definition wear time scale compares to std. time scale",
      tuple<std::string>("equal", "e", "different", "d"),
      tuple<WearTimeScale>(
          wear_time_equal, wear_time_equal, wear_time_different, wear_time_different),
      wear);

  wear.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
