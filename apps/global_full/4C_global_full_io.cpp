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

#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <fstream>
#include <memory>


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
  const bool screen = io.get<bool>("WRITE_TO_SCREEN");
  const bool file = io.get<bool>("WRITE_TO_FILE");
  const bool preGrpID = io.get<bool>("PREFIX_GROUP_ID");
  const int oproc = io.get<int>("LIMIT_OUTP_TO_PROC");
  const auto level = Teuchos::getIntegralValue<Core::IO::Verbositylevel>(io, "VERBOSITY");

  Core::IO::cout.setup(screen, file, preGrpID, level, communicators.local_comm(), oproc,
      communicators.group_id(), arguments.output_file_identifier);
}

void setup_global_problem(Core::IO::InputFile& input_file, const CommandlineArguments& arguments,
    const Core::Communication::Communicators& communicators)
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
  const auto mesh_reader = Global::read_discretization(*problem, input_file);
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

void write_timemonitor(const MPI_Comm comm)
{
  std::shared_ptr<const Teuchos::Comm<int>> TeuchosComm =
      Core::Communication::to_teuchos_comm<int>(comm);
  Teuchos::TimeMonitor::summarize(Teuchos::Ptr(TeuchosComm.get()), std::cout, false, true, false);
}

void export_timings(const std::filesystem::path& filename, MPI_Comm comm)
{
  std::shared_ptr<const Teuchos::Comm<int>> TeuchosComm =
      Core::Communication::to_teuchos_comm<int>(comm);

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::parameterList();
  params->set("Report format", "YAML");
  params->set("YAML style", "spacious");

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::ofstream out(filename);
    FOUR_C_ASSERT_ALWAYS(
        out.is_open(), "Could not open file '{}' for writing timings!", filename.string());
    Teuchos::TimeMonitor::report(Teuchos::Ptr(TeuchosComm.get()), out, "", params);
  }
  else
  {
    Teuchos::oblackholestream null_stream;
    Teuchos::TimeMonitor::report(Teuchos::Ptr(TeuchosComm.get()), null_stream, "", params);
  }
}

std::vector<std::pair<std::filesystem::path, std::string>> build_io_pairs(
    const std::vector<std::string>& io_pairs, const std::filesystem::path& primary_input,
    const std::string& primary_output)
{
  std::vector<std::pair<std::filesystem::path, std::string>> io_pairs_new;

  io_pairs_new.emplace_back(primary_input, primary_output);

  if (!io_pairs.empty())
  {
    if (io_pairs.size() % 2 != 0)
    {
      FOUR_C_THROW("Positional arguments must be provided as pairs: <input> <output>.\n");
    }
    for (size_t i = 0; i < io_pairs.size(); i += 2)
      io_pairs_new.emplace_back(std::filesystem::path(io_pairs[i]), io_pairs[i + 1]);
  }
  return io_pairs_new;
}

using NPT = Core::Communication::NestedParallelismType;
void validate_argument_cross_compatibility(const CommandlineArguments& arguments)
{
  if (!arguments.group_layout.empty())
  {
    const int layout_len = static_cast<int>(arguments.group_layout.size());
    if (arguments.n_groups != layout_len)
    {
      FOUR_C_THROW(
          "When --glayout is provided its number of entries must equal --ngroup.\n "
          "Example mpirun -np 4 ./4C --ngroup=2 --glayout=1,3 \n");
    }
  }

  if (arguments.n_groups > 1 && arguments.nptype == NPT::no_nested_parallelism)
  {
    FOUR_C_THROW("when --ngroup > 1, a nested parallelism type must be specified via --nptype.\n");
  }

  if (!arguments.parameters)
  {
    const size_t num_pairs = arguments.io_pairs.size();
    if (arguments.nptype == NPT::no_nested_parallelism ||
        arguments.nptype == NPT::every_group_read_input_file)
    {
      if (num_pairs != 1)
      {
        FOUR_C_THROW(
            "when using 'no_nested_parallelism' or 'everyGroupReadInputFile' the "
            "number of <input> <output> pairs must be exactly 1.\n");
      }
    }
    else if (arguments.nptype == NPT::separate_input_files ||
             arguments.nptype == NPT::nested_multiscale)
    {
      if (static_cast<int>(num_pairs) != arguments.n_groups)
      {
        FOUR_C_THROW(
            "when using 'separateInputFiles' or 'nestedMultiscale' the number of "
            "<input> <output> pairs must equal --ngroup {}.\n",
            arguments.n_groups);
      }
    }
  }

  if (arguments.nptype != NPT::separate_input_files &&
      (arguments.restart_per_group.size() > 1 || arguments.restart_identifier_per_group.size() > 1))
  {
    FOUR_C_THROW(
        "When using --nptype other than 'separateInputFiles', only one restart step and one "
        "restartfrom identifier must be given.");
  }

  for (size_t i = 0; i < arguments.restart_identifier_per_group.size(); ++i)
  {
    if (i >= arguments.restart_per_group.size())
    {
      FOUR_C_THROW("You need to specify a restart step when using restartfrom.");
    }
  }
}

void update_io_identifiers(CommandlineArguments& arguments, int group)
{
  std::filesystem::path input_filename;
  std::string output_file_identifier;

  const int restart_input_index = (arguments.nptype == NPT::separate_input_files) ? group : 0;

  arguments.restart =
      arguments.restart_per_group.empty() ? 0 : arguments.restart_per_group[restart_input_index];
  std::string restart_file_identifier =
      arguments.restart_identifier_per_group.empty()
          ? ""
          : arguments.restart_identifier_per_group[restart_input_index];

  switch (arguments.nptype)
  {
    case NPT::no_nested_parallelism:
      input_filename = arguments.io_pairs[0].first;
      output_file_identifier = arguments.io_pairs[0].second;
      if (restart_file_identifier == "")
      {
        restart_file_identifier = output_file_identifier;
      }
      break;
    case NPT::every_group_read_input_file:
    {
      input_filename = arguments.io_pairs[0].first;
      std::string output_file_identifier_temp = arguments.io_pairs[0].second;
      // check whether output_file_identifier includes a dash and in case separate the number at the
      // end
      size_t pos = output_file_identifier_temp.rfind('-');
      auto extract_number_and_identifier = [](const std::string& str, size_t pos)
      {
        std::string number_str = str.substr(pos + 1);
        std::string identifier = str.substr(0, pos);
        int number = 0;
        try
        {
          size_t idx = 0;
          number = std::stoi(number_str, &idx);
          if (idx != number_str.size())
          {
            FOUR_C_THROW("Invalid numeric value in output identifier: '{}'", number_str);
          }
        }
        catch (const std::exception& e)
        {
          FOUR_C_THROW(
              "Failed to parse number in output identifier '{}': {}", number_str, e.what());
        }
        return std::make_pair(identifier, number);
      };
      if (pos != std::string::npos)
      {
        auto [identifier, number] = extract_number_and_identifier(output_file_identifier_temp, pos);
        output_file_identifier = std::format("{}_group_{}_{}", identifier, group, number);
      }
      else
      {
        output_file_identifier = std::format("{}_group_{}", output_file_identifier_temp, group);
      }
      size_t pos_r = restart_file_identifier.rfind('-');
      if (restart_file_identifier == "")
      {
        restart_file_identifier = output_file_identifier;
      }
      else if (pos_r != std::string::npos)
      {
        auto [identifier, number] = extract_number_and_identifier(restart_file_identifier, pos_r);
        restart_file_identifier = std::format("{}_group_{}-{}", identifier, group, number);
      }
      else
      {
        restart_file_identifier = std::format("{}_group_{}", restart_file_identifier, group);
      }
      break;
    }
    case NPT::separate_input_files:
    case NPT::nested_multiscale:
      input_filename = arguments.io_pairs[group].first;
      output_file_identifier = arguments.io_pairs[group].second;
      if (restart_file_identifier == "")
      {
        restart_file_identifier = output_file_identifier;
      }
      break;
    default:
      FOUR_C_THROW("-nptype value {} is not valid.", static_cast<int>(arguments.nptype));
  }
  arguments.input_file_name = input_filename;
  arguments.output_file_identifier = output_file_identifier;
  arguments.restart_file_identifier = restart_file_identifier;
}
void assign_group_layout(const int& n_groups, std::vector<int>& group_layout)
{
  int myrank = -1;
  int num_procs = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if (n_groups > 1)
  {
    if (group_layout.size() == 0)
    {
      if (myrank == (num_procs - 1))  // myrank == 0 is eventually not within 4C (i.e. coupling to
                                      // external codes)
      {
        printf(
            "\n\n\nINFO: Group layout is not specified. Default is equal size of the "
            "groups.\n");
      }
      if ((num_procs % n_groups) != 0)
      {
        if (myrank == (num_procs - 1))
        {
          printf("\n\nNumber of processors (%d) cannot be divided by the number of groups (%d)!\n",
              num_procs, n_groups);
          printf("Try again!\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
      }

      // equal size of the groups
      for (int k = 0; k < n_groups; k++)
      {
        group_layout.push_back(num_procs / n_groups);
      }
    }
  }
}
FOUR_C_NAMESPACE_CLOSE
