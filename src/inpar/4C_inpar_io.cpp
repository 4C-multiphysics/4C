// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_io.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_io_pstream.hpp"
#include "4C_thermo_inpar.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::IO::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;

  Core::Utils::SectionSpecs io{"IO"};

  Core::Utils::bool_parameter("OUTPUT_GMSH", "No", "", io);
  Core::Utils::bool_parameter("OUTPUT_ROT", "No", "", io);
  Core::Utils::bool_parameter("OUTPUT_SPRING", "No", "", io);
  Core::Utils::bool_parameter("OUTPUT_BIN", "yes", "Do you want to have binary output?", io);

  // Output every iteration (for debugging purposes)
  Core::Utils::bool_parameter("OUTPUT_EVERY_ITER", "no",
      "Do you desire structural displ. output every Newton iteration", io);
  Core::Utils::int_parameter(
      "OEI_FILE_COUNTER", 0, "Add an output name affix by introducing a additional number", io);

  Core::Utils::bool_parameter(
      "ELEMENT_MAT_ID", "No", "Output of the material id of each element", io);

  // Structural output
  Core::Utils::bool_parameter("STRUCT_ELE", "Yes", "Output of element properties", io);
  Core::Utils::bool_parameter("STRUCT_DISP", "Yes", "Output of displacements", io);
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
  Core::Utils::string_to_integral_parameter<Inpar::Solid::OptQuantityType>(
      "STRUCT_OPTIONAL_QUANTITY", "No", "Output of an optional quantity",
      tuple<std::string>("No", "no", "NO", "membranethickness"),
      tuple<Inpar::Solid::OptQuantityType>(Inpar::Solid::optquantity_none,
          Inpar::Solid::optquantity_none, Inpar::Solid::optquantity_none,
          Inpar::Solid::optquantity_membranethickness),
      io);
  Core::Utils::bool_parameter("STRUCT_SURFACTANT", "No", "", io);
  Core::Utils::bool_parameter("STRUCT_JACOBIAN_MATLAB", "No", "", io);
  Core::Utils::string_to_integral_parameter<Inpar::Solid::ConditionNumber>(
      "STRUCT_CONDITION_NUMBER", "none",
      "Compute the condition number of the structural system matrix and write it to a text file.",
      tuple<std::string>("gmres_estimate", "max_min_ev_ratio", "one-norm", "inf-norm", "none"),
      tuple<Inpar::Solid::ConditionNumber>(Inpar::Solid::ConditionNumber::gmres_estimate,
          Inpar::Solid::ConditionNumber::max_min_ev_ratio, Inpar::Solid::ConditionNumber::one_norm,
          Inpar::Solid::ConditionNumber::inf_norm, Inpar::Solid::ConditionNumber::none),
      io);
  Core::Utils::bool_parameter("FLUID_STRESS", "No", "", io);
  Core::Utils::bool_parameter("FLUID_WALL_SHEAR_STRESS", "No", "", io);
  Core::Utils::bool_parameter("FLUID_ELEDATA_EVERY_STEP", "No", "", io);
  Core::Utils::bool_parameter("FLUID_NODEDATA_FIRST_STEP", "No", "", io);
  Core::Utils::bool_parameter("THERM_TEMPERATURE", "No", "", io);
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

  Core::Utils::int_parameter(
      "FILESTEPS", 1000, "Amount of timesteps written to a single result file", io);
  Core::Utils::int_parameter("STDOUTEVERY", 1, "Print to screen every n step", io);

  Core::Utils::bool_parameter("WRITE_TO_SCREEN", "Yes", "Write screen output", io);
  Core::Utils::bool_parameter("WRITE_TO_FILE", "No", "Write the output into a file", io);

  Core::Utils::bool_parameter(
      "WRITE_INITIAL_STATE", "yes", "Do you want to write output for initial state ?", io);
  Core::Utils::bool_parameter("WRITE_FINAL_STATE", "no",
      "Enforce to write output/restart data at the final state regardless of the other "
      "output/restart intervals",
      io);

  Core::Utils::bool_parameter(
      "PREFIX_GROUP_ID", "No", "Put a <GroupID>: in front of every line", io);
  Core::Utils::int_parameter(
      "LIMIT_OUTP_TO_PROC", -1, "Only the specified procs will write output", io);
  Core::Utils::string_to_integral_parameter<FourC::Core::IO::Verbositylevel>("VERBOSITY", "verbose",
      "",
      tuple<std::string>(
          "minimal", "Minimal", "standard", "Standard", "verbose", "Verbose", "debug", "Debug"),
      tuple<FourC::Core::IO::Verbositylevel>(FourC::Core::IO::minimal, FourC::Core::IO::minimal,
          FourC::Core::IO::standard, FourC::Core::IO::standard, FourC::Core::IO::verbose,
          FourC::Core::IO::verbose, FourC::Core::IO::debug, FourC::Core::IO::debug),
      io);

  Core::Utils::double_parameter("RESTARTWALLTIMEINTERVAL", -1.0,
      "Enforce restart after this walltime interval (in seconds), smaller zero to disable", io);
  Core::Utils::int_parameter("RESTARTEVERY", -1, "write restart every RESTARTEVERY steps", io);

  io.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs io_every_iter{io, "EVERY ITERATION"};

  // Output every iteration (for debugging purposes)
  Core::Utils::bool_parameter(
      "OUTPUT_EVERY_ITER", "No", "Do you wish output every Newton iteration?", io_every_iter);

  Core::Utils::int_parameter("RUN_NUMBER", -1,
      "Create a new folder for different runs of the same simulation. "
      "If equal -1, no folder is created.",
      io_every_iter);

  Core::Utils::int_parameter("STEP_NP_NUMBER", -1,
      "Give the number of the step (i.e. step_{n+1}) for which you want to write the "
      "debug output. If a negative step number is provided, all steps will"
      "be written.",
      io_every_iter);

  Core::Utils::bool_parameter("WRITE_OWNER_EACH_NEWTON_ITER", "No",
      "If yes, the ownership "
      "of elements and nodes are written each Newton step, instead of only once"
      "per time/load step.",
      io_every_iter);

  io_every_iter.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
