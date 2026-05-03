// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"
#include "4C_config_revision.hpp"

#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_global_full_io.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_io_command_line_helpers.hpp"
#include "4C_io_control.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_pstream.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Kokkos_Core.hpp>
#include <unistd.h>

#include <filesystem>
#include <format>
#include <iostream>

#ifdef FOUR_C_ENABLE_FE_TRAPPING
#include <cfenv>
#endif

using namespace FourC;

// Forward declarations
void entrypoint_switch();
void run(CommandlineArguments& cli_args, Core::Communication::Communicators& communicators);
CommandlineArguments parse_command_line(int argc, char** argv);
void get_memory_high_water_mark(MPI_Comm comm);

/**
 * @brief The main function of the central 4C executable.
 *
 * This function:
 * - sets up and finalizes MPI and Kokkos.
 * - handles certain command line options like `--help` which will only print information before
 *   terminating the program.
 * - delegates the actual reading of the input file and the computation.
 *
 */
int main(int argc, char* argv[])
{
  // Initialize MPI and use RAII to create a guard object that will finalize MPI when it goes out of
  // scope.
  MPI_Init(&argc, &argv);
  struct CleanUpMPI
  {
    ~CleanUpMPI() { MPI_Finalize(); }
  } cleanup_mpi;

  // Kokkos should be initialized right after MPI.
  Kokkos::ScopeGuard kokkos_guard{};

  auto arguments = parse_command_line(argc, argv);
  Core::Communication::CommConfig config{
      .group_layout = arguments.group_layout,
      .np_type = arguments.nptype,
      .diffgroup = arguments.diffgroup,
  };

  // Initialize communicators and use RAII to ensure that they are finalized properly in the end.
  // Note: Communicators must be finalized after singleton cleanup and before MPI finalization
  Core::Communication::Communicators communicators = Core::Communication::create_comm(config);
  struct FinalizeCommunicators
  {
    explicit FinalizeCommunicators(Core::Communication::Communicators& communicators)
        : communicators_(communicators)
    {
    }
    FinalizeCommunicators(const FinalizeCommunicators&) = delete;
    FinalizeCommunicators& operator=(const FinalizeCommunicators&) = delete;
    FinalizeCommunicators(FinalizeCommunicators&&) = delete;
    FinalizeCommunicators& operator=(FinalizeCommunicators&&) = delete;
    ~FinalizeCommunicators() noexcept { communicators_.finalize(); }

   private:
    Core::Communication::Communicators& communicators_;
  } finalize_communicators(communicators);

  // Initialize our own singleton registry to ensure we clean up all singletons properly.
  Core::Utils::SingletonOwnerRegistry::ScopeGuard singleton_owner_guard{};



  if (arguments.interactive)
  {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Global rank %d with PID %d on %s is ready for attach\n",
        Core::Communication::my_mpi_rank(communicators.global_comm()), getpid(), hostname);
    if (Core::Communication::my_mpi_rank(communicators.global_comm()) == 0)
    {
      printf("\n** Enter a character to continue > \n");
      fflush(stdout);
      char go = ' ';
      if (scanf("%c", &go) == EOF)
      {
        FOUR_C_THROW("Error while reading input.\n");
      }
    }
  }

  Core::Communication::barrier(communicators.global_comm());

  if (arguments.parameters)
  {
    if (Core::Communication::my_mpi_rank(communicators.local_comm()) == 0)
    {
      ryml::Tree tree = Core::IO::init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      root |= ryml::MAP;
      Core::IO::YamlNodeRef root_ref(root, "");

      // Write the non-user input metadata that is defined globally for 4C.
      emit_general_metadata(root_ref);

      // Write the user input defined for various physics module.
      Core::IO::InputFile input_file = setup_input_file(communicators.local_comm());
      input_file.emit_metadata(root_ref);

      // Finally, dump everything.
      std::cout << tree;
    }
  }
  else
  {
    if (Core::Communication::my_mpi_rank(communicators.global_comm()) == 0)
    {
      constexpr int box_width = 54;

      const auto print_centered = [&](const std::string& str)
      {
        // Subtract 2 for the asterisks on either side
        constexpr int width = box_width - 2;
        FOUR_C_ASSERT(str.size() < width, "String is too long to be centered.");
        std::cout << '*' << std::format("{:^{}}", str, width) << "*\n";
      };

      std::cout << '\n';
      std::cout << std::string(box_width, '*') << '\n';
      print_centered("");
      print_centered("4C");
      print_centered("");
      print_centered("version " FOUR_C_VERSION_FULL);
      print_centered("");
      print_centered("git SHA1");
      print_centered(VersionControl::git_hash);
      print_centered("");
      std::cout << std::string(box_width, '*') << '\n';
      std::cout << '\n';

      std::cout << "Trilinos Version: " << FOUR_C_TRILINOS_HASH << " (git SHA1)\n";
      std::cout << "Total number of MPI ranks: "
                << Core::Communication::num_mpi_ranks(communicators.global_comm()) << '\n';
    }

#ifdef FOUR_C_ENABLE_FE_TRAPPING
    // This is a GNU extension. Enable floating point exceptions for invalid operands and division
    // by zero. When such an operation occurs, the OS will kill the process with an informative
    // message.
    feenableexcept(FE_INVALID | FE_DIVBYZERO);
#endif

/*----------------------------------------------- everything is in here */
#ifdef FOUR_C_ENABLE_CORE_DUMP
    run(arguments);
#else
    try
    {
      run(arguments, communicators);
    }
    catch (Core::Exception& err)
    {
      char line[] = "=========================================================================\n";
      std::cout << "\n\n" << line << err.what_with_stacktrace() << "\n" << line << "\n" << '\n';

      if (communicators.num_groups() > 1)
      {
        printf("Global processor %d has thrown an error and is waiting for the remaining procs\n\n",
            Core::Communication::my_mpi_rank(communicators.global_comm()));
        Core::Communication::barrier(communicators.global_comm());
      }

      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
#endif
    /*----------------------------------------------------------------------*/

    get_memory_high_water_mark(communicators.global_comm());

    Core::Communication::barrier(communicators.local_comm());
    if (communicators.num_groups() > 1)
    {
      printf("Global processor %d with local rank %d finished normally\n",
          Core::Communication::my_mpi_rank(communicators.global_comm()),
          Core::Communication::my_mpi_rank(communicators.local_comm()));
      Core::Communication::barrier(communicators.global_comm());
    }
    else
    {
      Core::Communication::barrier(communicators.global_comm());
      printf("processor %d finished normally\n",
          Core::Communication::my_mpi_rank(communicators.local_comm()));
    }
  }

  return (0);
}

void run(CommandlineArguments& cli_args, Core::Communication::Communicators& communicators)
{
  update_io_identifiers(cli_args, communicators.group_id());

  /* input phase, input of all information */
  global_legacy_module_callbacks().RegisterParObjectTypes();

  {
    TEUCHOS_FUNC_TIME_MONITOR("Input");

    // and now the actual reading
    Core::IO::InputFile input_file = setup_input_file(communicators.local_comm());
    input_file.read(cli_args.input_file_name);
    setup_global_problem(input_file, cli_args, communicators);
  }

  // we wait till all procs are here. Otherwise a hang up might occur where
  // one proc ended with FOUR_C_THROW but other procs were not finished and waited...
  // we also want to have the printing above being finished.
  Core::Communication::barrier(communicators.local_comm());

  /*--------------------------------------------------calculation phase */

  {
    TEUCHOS_FUNC_TIME_MONITOR("Calculation");

    entrypoint_switch();
  }

  write_timemonitor(communicators.local_comm());

  if (Global::Problem::instance()->io_params().get<bool>("WRITE_TIMINGS"))
  {
    std::filesystem::path filename(
        Global::Problem::instance()->output_control_file()->file_name() + "-timings.yaml");

    export_timings(filename, communicators.global_comm());
  }
}

CommandlineArguments parse_command_line(int argc, char** argv)
{
  CLI::App cli_app{"4C - Multiphysics \nComprehensive Computational Community Code"};
  cli_app.formatter(std::make_shared<SpacedFormatter>());
  CommandlineArguments arguments;
  cli_app.add_flag("-p,--parameters", arguments.parameters,
      "Dumps information about the parameters for consumption by additional tools.");
  cli_app.add_flag("-i,--interactive", arguments.interactive,
      "4C waits at the beginning for keyboard input. "
      "Helpful for parallel debugging when attaching to a single job.");
  cli_app
      .add_option("--ngroup", arguments.n_groups,
          "Specify the number of groups for nested parallelism. (default: 1)")
      ->check(CLI::PositiveNumber);
  cli_app
      .add_option(
          "--glayout",
          [&arguments](const std::vector<std::string>& tokens) -> bool
          {
            if (tokens.empty()) return false;
            arguments.group_layout.clear();

            for (const auto& tok : tokens)
            {
              std::stringstream ss(tok);
              std::string part;
              while (std::getline(ss, part, ','))
              {
                if (part.empty()) continue;
                try
                {
                  size_t pos = 0;
                  int val = std::stoi(part, &pos);
                  if (pos != part.size() || val <= 0)
                  {
                    throw CLI::ValidationError("glayout", "Entries must be positive integers.");
                  }
                  arguments.group_layout.push_back(val);
                }
                catch (const std::invalid_argument&)
                {
                  throw CLI::ValidationError("glayout", "Entries must be positive integers.");
                }
                catch (const std::out_of_range&)
                {
                  throw CLI::ValidationError("glayout", "glayout entry out of range.");
                }
              }
            }
            return true;
          },
          "Specify the number of processors per group. Comma-separated list without spaces, e.g. "
          "--glayout=<a>,<b>.\n"
          "Argument --ngroup is mandatory if a glayout is provided. (default: equal distribution)")
      ->allow_extra_args(false);
  cli_app.add_option(
      "--nptype",
      [&arguments](const std::vector<std::string>& tokens) -> bool
      {
        if (tokens.empty()) return false;
        const std::string& input = tokens.front();
        if (input == "separateInputFiles")
        {
          arguments.nptype = Core::Communication::NestedParallelismType::separate_input_files;
          return true;
        }
        else if (input == "everyGroupReadInputFile")
        {
          arguments.nptype =
              Core::Communication::NestedParallelismType::every_group_read_input_file;
          return true;
        }
        else if (input == "nestedMultiscale")
        {
          arguments.nptype = Core::Communication::NestedParallelismType::nested_multiscale;
          return true;
        }
        else if (input.rfind("diffgroup", 0) == 0)
        {
          // Expect exactly an integer suffix after "diffgroup", and only allow 0 or 1
          const std::string suffix = input.substr(9);
          if (suffix.empty())
          {
            throw CLI::ValidationError(
                "nptype", "Missing suffix for 'diffgroup'; expected 'diffgroup0' or 'diffgroup1'.");
          }
          for (char c : suffix)
          {
            if (!std::isdigit(static_cast<unsigned char>(c)))
            {
              throw CLI::ValidationError(
                  "nptype", "Invalid diffgroup suffix; expected integer after 'diffgroup'.");
            }
          }
          int val = std::stoi(suffix);
          if (val != 0 && val != 1)
          {
            throw CLI::ValidationError("nptype", "Only diffgroup0 and diffgroup1 are allowed.");
          }
          arguments.nptype = Core::Communication::NestedParallelismType::no_nested_parallelism;
          arguments.diffgroup = val;
          return true;
        }
        else
        {
          throw CLI::ValidationError("nptype",
              "Only 'everyGroupReadInputFile', 'separateInputFiles', 'nestedMultiscale', and "
              "'diffgroupx' are available for nptype.");
        }
      },
      "Specify nested parallelism type: "
      "separateInputFiles|everyGroupReadInputFile|\nnestedMultiscale|diffgroup<N> \n"
      "Must be set if --ngroup > 1. \n"
      "'diffgroupx' can be used to compare vectors/matrices/results between two separate "
      "(serial/parallel) 4C runs; x must be 0 and 1 for the respective run");
  cli_app.add_option(
      "--restart",
      [&arguments](const std::vector<std::string>& tokens) -> bool
      {
        if (tokens.empty()) return false;
        for (const auto& tok : tokens)
        {
          // allow comma-separated lists in a single token (e.g. "--restart=3,4")
          std::stringstream ss(tok);
          std::string part;
          while (std::getline(ss, part, ','))
          {
            if (part.empty()) continue;
            if (part == "last_possible")
            {
              arguments.restart_per_group.push_back(-1);  // legacy sentinel for last_possible
              continue;
            }
            try
            {
              size_t pos = 0;
              int val = std::stoi(part, &pos);
              if (pos != part.size() || val < 0)
              {
                throw CLI::ValidationError(
                    "restart", "Restart step must be a non-negative integer or 'last_possible'.");
              }
              arguments.restart_per_group.push_back(val);
            }
            catch (const std::invalid_argument&)
            {
              throw CLI::ValidationError(
                  "restart", "Restart step must be a non-negative integer or 'last_possible'.");
            }
            catch (const std::out_of_range&)
            {
              throw CLI::ValidationError("restart", "Restart step out of range.");
            }
          }
        }
        return true;
      },
      "Restart the simulation from step <y>. Accepts a non-negative integer or 'last_possible'.\n"
      "If nested parallelism with separate input files is used, each group can have a different "
      "restart step defined as a comma-separated list, e.g., --restart=<a>,<b>.");
  cli_app.add_option(
      "--restartfrom",
      [&arguments](const std::vector<std::string>& tokens) -> bool
      {
        if (tokens.empty()) return false;
        for (const auto& tok : tokens)
        {
          std::stringstream ss(tok);
          std::string part;
          while (std::getline(ss, part, ','))
          {
            if (part.empty()) continue;
            arguments.restart_identifier_per_group.push_back(part);
          }
        }
        return true;
      },
      "Restart the simulation from the files prefixed with <restart_file_name>.\n If nested "
      "parallelism with separate input files is used, each group can have a different file prefix "
      "defined as a comma-separated list.");
  std::string primary_input;
  std::string primary_output;
  cli_app.add_option("input", primary_input, "Name of the input file, including the suffix");
  cli_app.add_option("output", primary_output, "Prefix of your output files.");

  std::vector<std::string> io_pairs;
  cli_app
      .add_option("io_pairs", io_pairs,
          "More pairs of simulation <input> and <output> names. Only necessary when using nested "
          "parallelism with multiple groups and separate input files.")
      ->expected(-1);


  std::vector<std::string> raw_args;
  raw_args.reserve(argc);
  for (int i = 1; i < argc; ++i) raw_args.emplace_back(argv[i]);

  LegacyCliOptions legacy_options = {.single_dash_legacy_names = {"ngroup", "glayout", "nptype"},
      .nodash_legacy_names = {"restart", "restartfrom"}};
  std::vector<std::string> sanitized_args = adapt_legacy_cli_arguments(raw_args, legacy_options);

  // Reversed order required when parsing std::vector<string> with CLI11
  std::reverse(sanitized_args.begin(), sanitized_args.end());
  try
  {
    cli_app.parse(sanitized_args);
  }
  catch (const CLI::ParseError& e)
  {
    std::exit(cli_app.exit(e));
  }

  if (!arguments.parameters)
  {
    if (primary_input.empty() || primary_output.empty())
    {
      FOUR_C_THROW("Please provide both <input> and <output> arguments.");
    }
  }

  arguments.io_pairs = build_io_pairs(io_pairs, primary_input, primary_output);
  validate_argument_cross_compatibility(arguments);
  assign_group_layout(arguments.n_groups, arguments.group_layout);
  return arguments;
}

/** Collect and print data on memory high water mark of this run
 *
 * 1. Ask the operating system for memory usage.
 * 2. Compute min/max/average and total memory usage across all MPI ranks.
 * 3. Print a summary to the screen.
 *
 * If status file can't be opened, issue a message to the screen. Do not throw an error, since
 * this is not considered a critical failure during a simulation.
 *
 * \note Currently limited to Linux systems
 *
 * @param[in] comm Global MPI_Comm object
 */
void get_memory_high_water_mark(MPI_Comm comm)
{
#if defined(__linux__)  // This works only on Linux systems
  const std::string status_match = "VmHWM";
  const std::string status_filename = "/proc/self/status";
  std::ifstream status_file(status_filename, std::ios_base::in);

  bool file_is_accessible = false;
  {
    /* Each proc knows about sucess/failure of opening its status file. Communication among all
     * procs will reveal, if _any_ proc has failure status. */
    // Get file status failure indicator on this proc
    auto local_status_failed = static_cast<int>(!status_file.is_open());

    // Check file status among all procs
    int global_status_failed = 0;
    MPI_Reduce(&local_status_failed, &global_status_failed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Mark file as ok if no proc failed to open its file
    if (global_status_failed == 0) file_is_accessible = true;
  }

  // Retrieve local memory use on each process
  if (file_is_accessible)
  {
    double local_mem = std::nan("0");

    std::string line;
    while (std::getline(status_file, line))
    {
      if (line.find(status_match) != std::string::npos)
      {
        size_t start = line.find_first_of("1234567890");
        size_t stop = line.find_last_of("1234567890");

        std::stringstream(line.substr(start, stop + 1)) >> local_mem;
        break;
      }
    }
    status_file.close();

    // Convert memory from KB to GB
    local_mem /= (1 << 20);

    // Gather values
    const int num_procs = Core::Communication::num_mpi_ranks(comm);
    auto recvbuf = std::unique_ptr<double[]>(new double[num_procs]);
    MPI_Gather(&local_mem, 1, MPI_DOUBLE, recvbuf.get(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute and output statistics on proc 0
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      double mem_min = recvbuf[0];
      double mem_max = recvbuf[0];
      double mem_tot = 0.0;
      double mem_avg = 0.0;
      int rank_min = -1;
      int rank_max = -1;

      for (int rank = 0; rank < num_procs; ++rank)
      {
        // Check for rank ID with min/max memory consumption
        if (recvbuf[rank] <= mem_min) rank_min = rank;
        if (recvbuf[rank] >= mem_max) rank_max = rank;

        // Compute memory statistics
        mem_min = std::min(mem_min, recvbuf[rank]);
        mem_max = std::max(mem_max, recvbuf[rank]);
        mem_tot += recvbuf[rank];
      }

      mem_avg = mem_tot / num_procs;

      if (num_procs > 1)
      {
        std::cout << std::scientific << std::setprecision(4) << "\nMemory High Water Mark Summary:"
                  << "\t\tMinOverProcs [PID]\tMeanOverProcs\tMaxOverProcs [PID]\tSumOverProcs\n"
                  << "(in GB)\t\t\t\t\t" << mem_min << "   [p" << rank_min << "]\t" << mem_avg
                  << "\t" << mem_max << "   [p" << rank_max << "]\t" << mem_tot << "\n"
                  << std::endl;
      }
      else
      {
        std::cout << std::scientific << std::setprecision(4)
                  << "\nMemory High Water Mark Summary:\t\tTotal\n"
                  << "(in GB)\t\t\t\t\t" << mem_tot << "\n"
                  << std::endl;
      }
    }
  }
  else  // Failed to open the status file
  {
    std::cout << "Memory High Water Mark summary can not be generated, since\n"
              << "status file '" << status_filename << "' could not be opened on every proc.\n"
              << std::endl;
  }
#else
  if (Core::Communication::my_mpi_rank(comm) == 0)
    std::cout << "Memory High Water Mark summary not available on this operating system.\n"
              << std::endl;
#endif
}
