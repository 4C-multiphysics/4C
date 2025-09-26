// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_global_full_io.hpp"

#include "4C_global_data_read.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_io_pstream.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

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
void setup_parallel_output(const CommandlineArguments& arguments)
{
  using namespace FourC;

  // configure the parallel output environment
  const Teuchos::ParameterList& io = Global::Problem::instance()->io_params();
  bool screen = io.get<bool>("WRITE_TO_SCREEN");
  bool file = io.get<bool>("WRITE_TO_FILE");
  bool preGrpID = io.get<bool>("PREFIX_GROUP_ID");
  int oproc = io.get<int>("LIMIT_OUTP_TO_PROC");
  auto level = Teuchos::getIntegralValue<Core::IO::Verbositylevel>(io, "VERBOSITY");

  Core::IO::cout.setup(screen, file, preGrpID, level, std::move(arguments.comms.local_comm()),
      oproc, arguments.comms.group_id(), arguments.output_file_identifier);
}

void setup_global_problem(Core::IO::InputFile& input_file, const CommandlineArguments& arguments)
{
  Global::Problem* problem = Global::Problem::instance();
  problem->set_restart_step(arguments.restart_step);
  problem->set_communicators(arguments.comms);
  Global::read_parameter(*problem, input_file);

  setup_parallel_output(arguments);

  // create control file for output and read restart data if required
  problem->open_control_file(arguments.comms.local_comm(), arguments.input_file_name,
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

void parse_commandline_arguments(CommandlineArguments& arguments)
{
  int group = arguments.comms.group_id();

  int restart_group = 0;
  int my_rank = Core::Communication::my_mpi_rank(arguments.comms.local_comm());

  std::vector<std::string> inout =
      parse_input_output_files(arguments.argc, arguments.argv, my_rank);

  // number of input/output arguments specified by the user
  auto inout_args = int(inout.size());

  std::string input_filename;
  std::string output_file_identifier;
  std::string restart_file_identifier;
  // set input file name in each group
  switch (arguments.comms.np_type())
  {
    case Core::Communication::NestedParallelismType::no_nested_parallelism:
      input_filename = inout[0];
      output_file_identifier = inout[1];
      restart_group = 0;
      break;
    case Core::Communication::NestedParallelismType::every_group_read_input_file:
    {
      if (inout_args > 4)
        FOUR_C_THROW(
            "You specified too many arguments ({}). A maximum of four args is allowed", inout_args);

      input_filename = inout[0];
      // check whether output_file_identifier includes a dash and in case separate the number at the
      // end
      size_t pos = inout[1].rfind('-');
      if (pos != std::string::npos)
      {
        int number = atoi(inout[1].substr(pos + 1).c_str());
        inout[1] = inout[1].substr(0, pos);
        output_file_identifier = std::format("{}_group_{}_{}", inout[1], group, number);
      }
      else
      {
        output_file_identifier = std::format("{}_group_{}", inout[1], group);
      }
      restart_group = 0;
    }
    break;
    case Core::Communication::NestedParallelismType::separate_input_files:
      if (inout_args % arguments.comms.num_groups() != 0)
        FOUR_C_THROW("Each group needs the same number of arguments for input/output.");
      inout_args /= arguments.comms.num_groups();
      input_filename = inout[group * inout_args];
      output_file_identifier = inout[group * inout_args + 1];
      restart_group = group;
      break;
    default:
      FOUR_C_THROW(
          "-nptype is not correct. Only everyGroupReadInputFile and separateInputFiles "
          "are available");
      break;
  }

  if (my_rank == 0)
  {
    std::cout << "input is read from     " << input_filename << std::endl;
  }
  parse_restart_definition(
      inout, inout_args, restart_file_identifier, output_file_identifier, restart_group, arguments);

  /// set IO file names and identifiers
  arguments.input_file_name = input_filename;
  arguments.output_file_identifier = output_file_identifier;
  arguments.restart_file_identifier = restart_file_identifier;
}


std::vector<std::string> parse_input_output_files(const int argc, char** argv, const int my_rank)
{
  if (argc <= 1)
  {
    if (my_rank == 0)
    {
      printf("You forgot to give the input and output file names!\n");
      printf("Try again!\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  else if (argc <= 2)
  {
    if (my_rank == 0)
    {
      printf("You forgot to give the output file name!\n");
      printf("Try again!\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }


  // parse command line and separate input/output arguments
  std::vector<std::string> inout;
  for (int i = 1; i < argc; i++)
  {
    std::string temp = argv[i];
    if (temp.substr(0, 1) != "-") inout.push_back(temp);
  }
  return inout;
}

void parse_restart_definition(const std::vector<std::string>& inout, const int in_out_args,
    std::string& restart_file_identifier, const std::string& outfile_identifier,
    const int restart_group, CommandlineArguments& arguments)
{
  // Global::Problem* problem = Global::Problem::instance();
  //  bool parameter defining if input argument is given
  bool restartIsGiven = false;
  bool restartfromIsGiven = false;

  // default case is an identical restartfile_identifier and outputfile_identifier
  restart_file_identifier = outfile_identifier;
  for (int i = 2; i < in_out_args; i++)
  {
    std::string restart = inout[restart_group * in_out_args + i];

    if (restart.substr(0, 8) == "restart=")
    {
      const std::string option = restart.substr(8, std::string::npos);
      int r;
      if (option.compare("last_possible") == 0)
      {
        r = -1;  // here we use a negative value to trigger the search in the control file in
                 // the later step. It does not mean a restart from a negative number is allowed
                 // from the user point of view.
      }
      else
      {
        r = atoi(option.c_str());
        if (r < 0) FOUR_C_THROW("Restart number must be a positive value");
      }
      // tell the global problem about the restart step given in the command line
      arguments.restart_step = r;
      restartIsGiven = true;
    }
    else if (restart.substr(0, 12) == "restartfrom=")
    {
      restart_file_identifier = (restart.substr(12, std::string::npos).c_str());

      switch (arguments.comms.np_type())
      {
        case Core::Communication::NestedParallelismType::no_nested_parallelism:
        case Core::Communication::NestedParallelismType::separate_input_files:
          // nothing to add to restartfileidentifier
          break;
        case Core::Communication::NestedParallelismType::every_group_read_input_file:
        {
          // check whether restartfileidentifier includes a dash and in case separate the number
          // at the end
          size_t pos = restart_file_identifier.rfind('-');
          if (pos != std::string::npos)
          {
            int number = atoi(restart_file_identifier.substr(pos + 1).c_str());
            std::string identifier = restart_file_identifier.substr(0, pos);
            restart_file_identifier =
                std::format("{}_group_{}_-{}", identifier, arguments.comms.group_id(), number);
          }
          else
          {
            restart_file_identifier =
                std::format("{}_group_{}", restart_file_identifier, arguments.comms.group_id());
          }
        }
        break;
        default:
          FOUR_C_THROW(
              "-nptype is not correct. Only everyGroupReadInputFile and "
              "separateInputFiles are available");
          break;
      }

      restartfromIsGiven = true;
    }
  }

  // throw error in case restartfrom is given but no restart step is specified
  if (restartfromIsGiven && !restartIsGiven)
  {
    FOUR_C_THROW("You need to specify a restart step when using restartfrom.");
  }
}


FOUR_C_NAMESPACE_CLOSE