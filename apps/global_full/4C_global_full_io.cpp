// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_global_full_io.hpp"

#include "4C_comm_utils.hpp"
#include "4C_global_data_read.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_io_pstream.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

Core::IO::InputFile setup_input_file(const MPI_Comm comm)
{
  return Global::set_up_input_file(comm);
}


void emit_general_metadata(const Core::IO::YamlNodeRef& root_ref)
{
  Global::emit_general_metadata(root_ref);
}

/**
 * \brief Sets up the parallel output environment.
 */
void setup_parallel_output(
    const CommandlineArguments& arguments, const Core::Communication::Communicators& communicators)
{
  using namespace FourC;

  // configure the parallel output environment
  const Teuchos::ParameterList& io = Global::Problem::instance()->io_params();
  bool screen = io.get<bool>("WRITE_TO_SCREEN");
  bool file = io.get<bool>("WRITE_TO_FILE");
  bool preGrpID = io.get<bool>("PREFIX_GROUP_ID");
  int oproc = io.get<int>("LIMIT_OUTP_TO_PROC");
  auto level = Teuchos::getIntegralValue<Core::IO::Verbositylevel>(io, "VERBOSITY");

  Core::IO::cout.setup(screen, file, preGrpID, level, communicators.local_comm(), oproc,
      communicators.group_id(), arguments.output_file_identifier);
}

void setup_global_problem(Core::IO::InputFile& input_file, const CommandlineArguments& arguments,
    Core::Communication::Communicators& communicators)
{
  Global::Problem* problem = Global::Problem::instance();
  problem->set_restart_step(arguments.restart);
  problem->set_communicators(communicators);
  Global::read_parameter(*problem, input_file);

  setup_parallel_output(arguments, communicators);

  // create control file for output and read restart data if required
  problem->open_control_file(communicators.local_comm(), arguments.input_file_name,
      arguments.output_file_identifier, arguments.restart_file_identifier);

  // input of materials
  Global::read_materials(*problem, input_file);

  // input for multi-scale rough-surface contact
  Global::read_contact_constitutive_laws(*problem, input_file);

  // input of materials of cloned fields (if needed)
  Global::read_cloning_material_map(*problem, input_file);

  {
    Core::Utils::FunctionManager function_manager;
    global_legacy_module_callbacks().AttachFunctionDefinitions(function_manager);
    function_manager.read_input(input_file);
    problem->set_function_manager(std::move(function_manager));
  }

  // input of particles
  Global::read_particles(*problem, input_file);


  // input of fields
  auto mesh_reader = Global::read_discretization(*problem, input_file);
  FOUR_C_ASSERT(mesh_reader, "Internal error: nullptr.");

  // read result tests
  Global::read_result(*problem, input_file);

  // read all types of geometry related conditions (e.g. boundary conditions)
  // Also read time and space functions and local coord systems
  Global::read_conditions(*problem, input_file, *mesh_reader);

  // read all knot information for isogeometric analysis
  // and add it to the (derived) nurbs discretization
  Global::read_knots(*problem, input_file);

  Global::read_fields(*problem, input_file, *mesh_reader);
}

double walltime_in_seconds()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
             .count() *
         1.0e-3;
}

void write_timemonitor(MPI_Comm comm)
{
  std::shared_ptr<const Teuchos::Comm<int>> TeuchosComm =
      Core::Communication::to_teuchos_comm<int>(comm);
  Teuchos::TimeMonitor::summarize(Teuchos::Ptr(TeuchosComm.get()), std::cout, false, true, false);
}

FOUR_C_NAMESPACE_CLOSE
