// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_io.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_pstream.hpp"
#include "4C_thermo_input.hpp"
FOUR_C_NAMESPACE_OPEN

std::vector<Core::IO::InputSpec> Inpar::IO::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("IO",
      {

          parameter<bool>("OUTPUT_GMSH", {.description = "", .default_value = false}),
          parameter<bool>("OUTPUT_ROT", {.description = "", .default_value = false}),

          parameter<bool>("OUTPUT_SPRING",
              {.description = "Flag indicating whether quantities related to the spring dashpot "
                              "conditions should be written to output.",
                  .default_value = false}),
          parameter<bool>("OUTPUT_BIN",
              {.description = "Do you want to have binary output?", .default_value = true}),

          // Output every iteration (for debugging purposes)
          parameter<bool>("OUTPUT_EVERY_ITER",
              {.description = "Do you desire structural displ. output every Newton iteration",
                  .default_value = false}),
          parameter<int>("OEI_FILE_COUNTER",
              {.description = "Add an output name affix by introducing a additional number",
                  .default_value = 0}),

          parameter<bool>("ELEMENT_MAT_ID",
              {.description = "Output of the material id of each element", .default_value = false}),

          // Structural output
          parameter<bool>(
              "STRUCT_ELE", {.description = "Output of element properties", .default_value = true}),
          parameter<bool>(
              "STRUCT_DISP", {.description = "Output of displacements", .default_value = true}),
          deprecated_selection<Inpar::Solid::StressType>("STRUCT_STRESS",
              {
                  {"No", Inpar::Solid::stress_none},
                  {"no", Inpar::Solid::stress_none},
                  {"NO", Inpar::Solid::stress_none},
                  {"Yes", Inpar::Solid::stress_2pk},
                  {"yes", Inpar::Solid::stress_2pk},
                  {"YES", Inpar::Solid::stress_2pk},
                  {"Cauchy", Inpar::Solid::stress_cauchy},
                  {"cauchy", Inpar::Solid::stress_cauchy},
                  {"2PK", Inpar::Solid::stress_2pk},
                  {"2pk", Inpar::Solid::stress_2pk},
              },
              {.description = "Output of stress", .default_value = Inpar::Solid::stress_none}),
          deprecated_selection<Inpar::Solid::StrainType>("STRUCT_STRAIN",
              {
                  {"No", Inpar::Solid::strain_none},
                  {"no", Inpar::Solid::strain_none},
                  {"NO", Inpar::Solid::strain_none},
                  {"Yes", Inpar::Solid::strain_gl},
                  {"yes", Inpar::Solid::strain_gl},
                  {"YES", Inpar::Solid::strain_gl},
                  {"EA", Inpar::Solid::strain_ea},
                  {"ea", Inpar::Solid::strain_ea},
                  {"GL", Inpar::Solid::strain_gl},
                  {"gl", Inpar::Solid::strain_gl},
                  {"LOG", Inpar::Solid::strain_log},
                  {"log", Inpar::Solid::strain_log},
              },
              {.description = "Output of strains", .default_value = Inpar::Solid::strain_none}),
          deprecated_selection<Inpar::Solid::StrainType>("STRUCT_PLASTIC_STRAIN",
              {
                  {"No", Inpar::Solid::strain_none},
                  {"no", Inpar::Solid::strain_none},
                  {"NO", Inpar::Solid::strain_none},
                  {"Yes", Inpar::Solid::strain_gl},
                  {"yes", Inpar::Solid::strain_gl},
                  {"YES", Inpar::Solid::strain_gl},
                  {"EA", Inpar::Solid::strain_ea},
                  {"ea", Inpar::Solid::strain_ea},
                  {"GL", Inpar::Solid::strain_gl},
                  {"gl", Inpar::Solid::strain_gl},
              },
              {.description = "", .default_value = Inpar::Solid::strain_none}),

          parameter<bool>("STRUCT_SURFACTANT", {.description = "", .default_value = false}),

          parameter<bool>("STRUCT_JACOBIAN_MATLAB", {.description = "", .default_value = false}),

          deprecated_selection<Inpar::Solid::ConditionNumber>("STRUCT_CONDITION_NUMBER",
              {
                  {"gmres_estimate", Inpar::Solid::ConditionNumber::gmres_estimate},
                  {"max_min_ev_ratio", Inpar::Solid::ConditionNumber::max_min_ev_ratio},
                  {"one-norm", Inpar::Solid::ConditionNumber::one_norm},
                  {"inf-norm", Inpar::Solid::ConditionNumber::inf_norm},
                  {"none", Inpar::Solid::ConditionNumber::none},
              },
              {.description =
                      "Compute the condition number of the structural system matrix and write "
                      "it to a text file.",
                  .default_value = Inpar::Solid::ConditionNumber::none}),

          parameter<bool>("FLUID_STRESS", {.description = "", .default_value = false}),

          parameter<bool>("FLUID_WALL_SHEAR_STRESS", {.description = "", .default_value = false}),

          parameter<bool>("FLUID_ELEDATA_EVERY_STEP", {.description = "", .default_value = false}),

          parameter<bool>("FLUID_NODEDATA_FIRST_STEP", {.description = "", .default_value = false}),

          parameter<bool>("THERM_TEMPERATURE", {.description = "", .default_value = false}),
          deprecated_selection<Thermo::HeatFluxType>("THERM_HEATFLUX",
              {
                  {"None", Thermo::heatflux_none},
                  {"No", Thermo::heatflux_none},
                  {"NO", Thermo::heatflux_none},
                  {"no", Thermo::heatflux_none},
                  {"Current", Thermo::heatflux_current},
                  {"Initial", Thermo::heatflux_initial},
              },
              {.description = "", .default_value = Thermo::heatflux_none}),
          deprecated_selection<Thermo::TempGradType>("THERM_TEMPGRAD",
              {
                  {"None", Thermo::tempgrad_none},
                  {"No", Thermo::tempgrad_none},
                  {"NO", Thermo::tempgrad_none},
                  {"no", Thermo::tempgrad_none},
                  {"Current", Thermo::tempgrad_current},
                  {"Initial", Thermo::tempgrad_initial},
              },
              {.description = "", .default_value = Thermo::tempgrad_none}),

          parameter<int>(
              "FILESTEPS", {.description = "Amount of timesteps written to a single result file",
                               .default_value = 1000}),
          parameter<int>(
              "STDOUTEVERY", {.description = "Print to screen every n step", .default_value = 1}),

          parameter<bool>(
              "WRITE_TO_SCREEN", {.description = "Write screen output", .default_value = true}),
          parameter<bool>("WRITE_TO_FILE",
              {.description = "Write the output into a file", .default_value = false}),

          parameter<bool>("WRITE_INITIAL_STATE",
              {.description = "Do you want to write output for initial state ?",
                  .default_value = true}),
          parameter<bool>("WRITE_FINAL_STATE",
              {.description = "Enforce to write output/restart data at the final "
                              "state regardless of the other output/restart intervals",
                  .default_value = false}),

          parameter<bool>("PREFIX_GROUP_ID",
              {.description = "Put a <GroupID>: in front of every line", .default_value = false}),
          parameter<int>("LIMIT_OUTP_TO_PROC",
              {.description = "Only the specified procs will write output", .default_value = -1}),
          deprecated_selection<Core::IO::Verbositylevel>("VERBOSITY",
              {
                  {"minimal", Core::IO::minimal},
                  {"Minimal", Core::IO::minimal},
                  {"standard", Core::IO::standard},
                  {"Standard", Core::IO::standard},
                  {"verbose", Core::IO::verbose},
                  {"Verbose", Core::IO::verbose},
                  {"debug", Core::IO::debug},
                  {"Debug", Core::IO::debug},
              },
              {.description = "", .default_value = Core::IO::verbose}),

          parameter<double>("RESTARTWALLTIMEINTERVAL",
              {.description = "Enforce restart after this walltime interval "
                              "(in seconds), smaller zero to disable",
                  .default_value = -1.0}),
          parameter<int>("RESTARTEVERY",
              {.description = "write restart every RESTARTEVERY steps", .default_value = -1})},
      {.required = false}));
  specs.push_back(group("IO/EVERY ITERATION",
      {

          // Output every iteration (for debugging purposes)
          parameter<bool>(
              "OUTPUT_EVERY_ITER", {.description = "Do you wish output every Newton iteration?",
                                       .default_value = false}),

          parameter<int>("RUN_NUMBER",
              {.description = "Create a new folder for different runs of the same simulation. "
                              "If equal -1, no folder is created.",
                  .default_value = -1}),

          parameter<int>("STEP_NP_NUMBER",
              {.description =
                      "Give the number of the step (i.e. step_{n+1}) for which you want to write "
                      "the debug output. If a negative step number is provided, all steps will"
                      "be written.",
                  .default_value = -1}),

          parameter<bool>("WRITE_OWNER_EACH_NEWTON_ITER",
              {.description =
                      "If yes, the ownership of elements and nodes are written each Newton step, "
                      "instead of only once per time/load step.",
                  .default_value = false})},
      {.required = false}));
  return specs;
}

FOUR_C_NAMESPACE_CLOSE