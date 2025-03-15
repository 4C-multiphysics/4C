// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_io.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_io_pstream.hpp"
#include "4C_thermo_input.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::IO::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs io{"IO"};

  io.specs.emplace_back(
      parameter<bool>("OUTPUT_GMSH", {.description = "", .default_value = false}));
  io.specs.emplace_back(parameter<bool>("OUTPUT_ROT", {.description = "", .default_value = false}));
  io.specs.emplace_back(
      parameter<bool>("OUTPUT_SPRING", {.description = "", .default_value = false}));
  io.specs.emplace_back(parameter<bool>(
      "OUTPUT_BIN", {.description = "Do you want to have binary output?", .default_value = true}));

  // Output every iteration (for debugging purposes)
  io.specs.emplace_back(parameter<bool>("OUTPUT_EVERY_ITER",
      {.description = "Do you desire structural displ. output every Newton iteration",
          .default_value = false}));
  io.specs.emplace_back(parameter<int>("OEI_FILE_COUNTER",
      {.description = "Add an output name affix by introducing a additional number",
          .default_value = 0}));

  io.specs.emplace_back(parameter<bool>("ELEMENT_MAT_ID",
      {.description = "Output of the material id of each element", .default_value = false}));

  // Structural output
  io.specs.emplace_back(parameter<bool>(
      "STRUCT_ELE", {.description = "Output of element properties", .default_value = true}));
  io.specs.emplace_back(parameter<bool>(
      "STRUCT_DISP", {.description = "Output of displacements", .default_value = true}));
  Core::Utils::string_to_integral_parameter<Inpar::Solid::StressType>("STRUCT_STRESS", "No",
      "Output of stress",
      tuple<std::string>("No", "no", "NO", "Yes", "yes", "YES", "Cauchy", "cauchy", "2PK", "2pk"),
      tuple<Inpar::Solid::StressType>(Inpar::Solid::stress_none, Inpar::Solid::stress_none,
          Inpar::Solid::stress_none, Inpar::Solid::stress_2pk, Inpar::Solid::stress_2pk,
          Inpar::Solid::stress_2pk, Inpar::Solid::stress_cauchy, Inpar::Solid::stress_cauchy,
          Inpar::Solid::stress_2pk, Inpar::Solid::stress_2pk),
      io);
  // in case of a coupled problem (e.g. TSI) the additional stresses are
  // (TSI: thermal stresses) are printed here
  Core::Utils::string_to_integral_parameter<Inpar::Solid::StressType>("STRUCT_COUPLING_STRESS",
      "No", "",
      tuple<std::string>("No", "no", "NO", "Yes", "yes", "YES", "Cauchy", "cauchy", "2PK", "2pk"),
      tuple<Inpar::Solid::StressType>(Inpar::Solid::stress_none, Inpar::Solid::stress_none,
          Inpar::Solid::stress_none, Inpar::Solid::stress_2pk, Inpar::Solid::stress_2pk,
          Inpar::Solid::stress_2pk, Inpar::Solid::stress_cauchy, Inpar::Solid::stress_cauchy,
          Inpar::Solid::stress_2pk, Inpar::Solid::stress_2pk),
      io);
  Core::Utils::string_to_integral_parameter<Inpar::Solid::StrainType>("STRUCT_STRAIN", "No",
      "Output of strains",
      tuple<std::string>(
          "No", "no", "NO", "Yes", "yes", "YES", "EA", "ea", "GL", "gl", "LOG", "log"),
      tuple<Inpar::Solid::StrainType>(Inpar::Solid::strain_none, Inpar::Solid::strain_none,
          Inpar::Solid::strain_none, Inpar::Solid::strain_gl, Inpar::Solid::strain_gl,
          Inpar::Solid::strain_gl, Inpar::Solid::strain_ea, Inpar::Solid::strain_ea,
          Inpar::Solid::strain_gl, Inpar::Solid::strain_gl, Inpar::Solid::strain_log,
          Inpar::Solid::strain_log),
      io);
  Core::Utils::string_to_integral_parameter<Inpar::Solid::StrainType>("STRUCT_PLASTIC_STRAIN", "No",
      "", tuple<std::string>("No", "no", "NO", "Yes", "yes", "YES", "EA", "ea", "GL", "gl"),
      tuple<Inpar::Solid::StrainType>(Inpar::Solid::strain_none, Inpar::Solid::strain_none,
          Inpar::Solid::strain_none, Inpar::Solid::strain_gl, Inpar::Solid::strain_gl,
          Inpar::Solid::strain_gl, Inpar::Solid::strain_ea, Inpar::Solid::strain_ea,
          Inpar::Solid::strain_gl, Inpar::Solid::strain_gl),
      io);
  io.specs.emplace_back(
      parameter<bool>("STRUCT_SURFACTANT", {.description = "", .default_value = false}));
  io.specs.emplace_back(
      parameter<bool>("STRUCT_JACOBIAN_MATLAB", {.description = "", .default_value = false}));
  Core::Utils::string_to_integral_parameter<Inpar::Solid::ConditionNumber>(
      "STRUCT_CONDITION_NUMBER", "none",
      "Compute the condition number of the structural system matrix and write it to a text file.",
      tuple<std::string>("gmres_estimate", "max_min_ev_ratio", "one-norm", "inf-norm", "none"),
      tuple<Inpar::Solid::ConditionNumber>(Inpar::Solid::ConditionNumber::gmres_estimate,
          Inpar::Solid::ConditionNumber::max_min_ev_ratio, Inpar::Solid::ConditionNumber::one_norm,
          Inpar::Solid::ConditionNumber::inf_norm, Inpar::Solid::ConditionNumber::none),
      io);
  io.specs.emplace_back(
      parameter<bool>("FLUID_STRESS", {.description = "", .default_value = false}));
  io.specs.emplace_back(
      parameter<bool>("FLUID_WALL_SHEAR_STRESS", {.description = "", .default_value = false}));
  io.specs.emplace_back(
      parameter<bool>("FLUID_ELEDATA_EVERY_STEP", {.description = "", .default_value = false}));
  io.specs.emplace_back(
      parameter<bool>("FLUID_NODEDATA_FIRST_STEP", {.description = "", .default_value = false}));
  io.specs.emplace_back(
      parameter<bool>("THERM_TEMPERATURE", {.description = "", .default_value = false}));
  Core::Utils::string_to_integral_parameter<Thermo::HeatFluxType>("THERM_HEATFLUX", "None", "",
      tuple<std::string>("None", "No", "NO", "no", "Current", "Initial"),
      tuple<Thermo::HeatFluxType>(Thermo::heatflux_none, Thermo::heatflux_none,
          Thermo::heatflux_none, Thermo::heatflux_none, Thermo::heatflux_current,
          Thermo::heatflux_initial),
      io);
  Core::Utils::string_to_integral_parameter<Thermo::TempGradType>("THERM_TEMPGRAD", "None", "",
      tuple<std::string>("None", "No", "NO", "no", "Current", "Initial"),
      tuple<Thermo::TempGradType>(Thermo::tempgrad_none, Thermo::tempgrad_none,
          Thermo::tempgrad_none, Thermo::tempgrad_none, Thermo::tempgrad_current,
          Thermo::tempgrad_initial),
      io);

  io.specs.emplace_back(parameter<int>(
      "FILESTEPS", {.description = "Amount of timesteps written to a single result file",
                       .default_value = 1000}));
  io.specs.emplace_back(parameter<int>(
      "STDOUTEVERY", {.description = "Print to screen every n step", .default_value = 1}));

  io.specs.emplace_back(parameter<bool>(
      "WRITE_TO_SCREEN", {.description = "Write screen output", .default_value = true}));
  io.specs.emplace_back(parameter<bool>(
      "WRITE_TO_FILE", {.description = "Write the output into a file", .default_value = false}));

  io.specs.emplace_back(parameter<bool>("WRITE_INITIAL_STATE",
      {.description = "Do you want to write output for initial state ?", .default_value = true}));
  io.specs.emplace_back(parameter<bool>(
      "WRITE_FINAL_STATE", {.description = "Enforce to write output/restart data at the final "
                                           "state regardless of the other output/restart intervals",
                               .default_value = false}));

  io.specs.emplace_back(parameter<bool>("PREFIX_GROUP_ID",
      {.description = "Put a <GroupID>: in front of every line", .default_value = false}));
  io.specs.emplace_back(parameter<int>("LIMIT_OUTP_TO_PROC",
      {.description = "Only the specified procs will write output", .default_value = -1}));
  Core::Utils::string_to_integral_parameter<FourC::Core::IO::Verbositylevel>("VERBOSITY", "verbose",
      "",
      tuple<std::string>(
          "minimal", "Minimal", "standard", "Standard", "verbose", "Verbose", "debug", "Debug"),
      tuple<FourC::Core::IO::Verbositylevel>(FourC::Core::IO::minimal, FourC::Core::IO::minimal,
          FourC::Core::IO::standard, FourC::Core::IO::standard, FourC::Core::IO::verbose,
          FourC::Core::IO::verbose, FourC::Core::IO::debug, FourC::Core::IO::debug),
      io);

  io.specs.emplace_back(parameter<double>("RESTARTWALLTIMEINTERVAL",
      {.description =
              "Enforce restart after this walltime interval (in seconds), smaller zero to disable",
          .default_value = -1.0}));
  io.specs.emplace_back(parameter<int>("RESTARTEVERY",
      {.description = "write restart every RESTARTEVERY steps", .default_value = -1}));

  io.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs io_every_iter{io, "EVERY ITERATION"};

  // Output every iteration (for debugging purposes)
  io_every_iter.specs.emplace_back(parameter<bool>("OUTPUT_EVERY_ITER",
      {.description = "Do you wish output every Newton iteration?", .default_value = false}));

  io_every_iter.specs.emplace_back(parameter<int>("RUN_NUMBER",
      {.description = "Create a new folder for different runs of the same simulation. "
                      "If equal -1, no folder is created.",
          .default_value = -1}));

  io_every_iter.specs.emplace_back(parameter<int>("STEP_NP_NUMBER",
      {.description = "Give the number of the step (i.e. step_{n+1}) for which you want to write "
                      "the debug output. If a negative step number is provided, all steps will"
                      "be written.",
          .default_value = -1}));

  io_every_iter.specs.emplace_back(parameter<bool>("WRITE_OWNER_EACH_NEWTON_ITER",
      {.description = "If yes, the ownership of elements and nodes are written each Newton step, "
                      "instead of only once per time/load step.",
          .default_value = false}));

  io_every_iter.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
