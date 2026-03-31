// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_wear.hpp"

#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN



Core::IO::InputSpec Inpar::Wear::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  /* parameters for wear */
  Core::IO::InputSpec spec = group("WEAR",
      {

          deprecated_selection<WearLaw>("WEARLAW",
              {
                  {"None", wear_none},
                  {"none", wear_none},
                  {"Archard", wear_archard},
                  {"archard", wear_archard},
              },
              {.description = "Type of wear law", .default_value = wear_none}),


          parameter<bool>(
              "MATCHINGGRID", {.description = "is matching grid", .default_value = true}),

          deprecated_selection<WearShape>("WEAR_SHAPEFCN",
              {
                  {"Dual", wear_shape_dual},
                  {"dual", wear_shape_dual},
                  {"Standard", wear_shape_standard},
                  {"standard", wear_shape_standard},
                  {"std", wear_shape_standard},
              },
              {.description = "Type of employed set of shape functions for wear",
                  .default_value = wear_shape_standard}),

          parameter<double>("WEARCOEFF",
              {.description = "Wear coefficient for slave surface", .default_value = 0.0}),
          parameter<double>("WEARCOEFF_MASTER",
              {.description = "Wear coefficient for master surface", .default_value = 0.0}),
          parameter<double>("WEAR_TIMERATIO",
              {.description = "Time step ratio between wear and spatial time scale",
                  .default_value = 1.0}),
          parameter<double>(
              "SSSLIP", {.description = "Fixed slip for steady state wear", .default_value = 1.0}),

          parameter<bool>(
              "SSWEAR", {.description = "flag for steady state wear", .default_value = false}),

          deprecated_selection<WearSide>("WEAR_SIDE",
              {
                  {"s", wear_slave},
                  {"slave", wear_slave},
                  {"Slave", wear_slave},
                  {"both", wear_both},
                  {"slave_master", wear_both},
                  {"sm", wear_both},
              },
              {.description = "Definition of wear side", .default_value = wear_slave}),

          deprecated_selection<WearType>("WEARTYPE",
              {
                  {"intstate", wear_intstate},
                  {"is", wear_intstate},
                  {"internal_state", wear_intstate},
                  {"primvar", wear_primvar},
                  {"pv", wear_primvar},
                  {"primary_variable", wear_primvar},
              },
              {.description = "Definition of wear algorithm", .default_value = wear_intstate}),

          deprecated_selection<WearTimInt>("WEARTIMINT",
              {
                  {"explicit", wear_expl},
                  {"e", wear_expl},
                  {"expl", wear_expl},
                  {"implicit", wear_impl},
                  {"i", wear_impl},
                  {"impl", wear_impl},
              },
              {.description = "Definition of wear time integration", .default_value = wear_expl}),

          deprecated_selection<WearTimeScale>("WEAR_TIMESCALE",
              {
                  {"equal", wear_time_equal},
                  {"e", wear_time_equal},
                  {"different", wear_time_different},
                  {"d", wear_time_different},
              },
              {.description = "Definition wear time scale compares to std. time scale",
                  .default_value = wear_time_equal})},
      {.required = false});
  return spec;
}

FOUR_C_NAMESPACE_CLOSE