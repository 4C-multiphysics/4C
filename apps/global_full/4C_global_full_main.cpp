// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"
#include "4C_config_revision.hpp"

#include "4C_comm_utils.hpp"
#include "4C_global_full_io.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
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

namespace
{
  void print_help_message()
  {
    std::cout
        << "NAME\n"
        << "\t"
        << "4C - simulate just about anything\n"
        << "\n"
        << "SYNOPSIS\n"
        << "\t"
        << "4C [-h | --help] [-p | --parameters] [-d | --datfile] [-ngroup=<x>] \\ "
           "\n"
           "\t\t[-glayout=a,b,c,...] [-nptype=<parallelism_type>] \\ \n"
        << "\t\t<input_name> <output_name> [restart=<y>] [restartfrom=restart_file_name] \\ \n"
           "\t\t[ <input_name0> <output_name0> [restart=<y>] [restartfrom=restart_file_name] ... "
           "] \\ \n"
           "\t\t[--interactive]\n"
        << "\n"
        << "DESCRIPTION\n"
        << "\tThe am besten simulation tool in the world.\n"
        << "\n"
        << "OPTIONS\n"
        << "\t--help or -h\n"
        << "\t\tPrint this message.\n"
        << "\n"
        << "\t--parameters or -p\n"
        << "\t\tDumps information about the parameters for consumption by additional tools.\n"
        << "\n"
        << "\t-ngroup=<x>\n"
        << "\t\tSpecify the number of groups for nested parallelism. (default: 1)\n"
        << "\n"
        << "\t-glayout=<a>,<b>,<c>,...\n"
        << "\t\tSpecify the number of processors per group. \n"
           "\t\tArgument \"-ngroup\" is mandatory and must be preceding. \n"
           "\t\t(default: equal distribution)\n"
        << "\n"
        << "\t-nptype=<parallelism_type>\n"
        << "\t\tAvailable options: \"separateInputFiles\" and \"everyGroupReadInputFile\"; \n"
           "\t\tMust be set if \"-ngroup\" > 1.\n"
        << "\t\t\"diffgroupx\" can be used to compare results from separate but parallel 4C "
           "runs; \n"
           "\t\tx must be 0 and 1 for the respective run\n"
        << "\n"
        << "\t<input_name>\n"
        << "\t\tName of the input file, including the suffix\n"
        << "\n"
        << "\t<output_name>\n"
        << "\t\tPrefix of your output files.\n"
        << "\n"
        << "\trestart=<y>\n"
        << "\t\tRestart the simulation from step <y>. \n"
           "\t\tIt always refers to the previously defined <input_name> and <output_name>. \n"
           "\t\t(default: 0 or from <input_name>)\n"
           "\t\tIf y=last_possible, it will restart from the last restart step defined in the "
           "control file.\n"
        << "\n"
        << "\trestartfrom=<restart_file_name>\n"
        << "\t\tRestart the simulation from the files prefixed with <restart_file_name>. \n"
           "\t\t(default: <output_name>)\n"
        << "\n"
        << "\t--interactive\n"
        << "\t\t4C waits at the beginning for keyboard input. \n"
           "\t\tHelpful for parallel debugging when attaching to a single job. \n"
           "\t\tMust be specified at the end in the command line.\n"
        << "\n";
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
      MPI_Reduce(
          &local_status_failed, &global_status_failed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

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
          std::cout << std::scientific << std::setprecision(4)
                    << "\nMemory High Water Mark Summary:"
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
}  // namespace

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void entrypoint_switch();
void run(CommandlineArguments& arguments);

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

  // Initialize our own singleton registry to ensure we clean up all singletons properly.
  Core::Utils::SingletonOwnerRegistry::ScopeGuard singleton_owner_guard{};

  Core::Communication::Communicators communicators =
      Core::Communication::create_comm(std::vector<std::string>(argv, argv + argc));

  CommandlineArguments arguments{
      .argc = argc,
      .argv = argv,
      .input_file_name = "",
      .output_file_identifier = "",
      .restart_file_identifier = "",
      .restart_step = 0,
      .comms = communicators,
  };

  if (strcmp(argv[argc - 1], "--interactive") == 0)
  {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Global rank %d with PID %d on %s is ready for attach\n",
        Core::Communication::my_mpi_rank(arguments.comms.global_comm()), getpid(), hostname);
    if (Core::Communication::my_mpi_rank(arguments.comms.global_comm()) == 0)
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

  Core::Communication::barrier(arguments.comms.global_comm());

  if ((argc == 2) && ((strcmp(argv[1], "-h") == 0) || (strcmp(argv[1], "--help") == 0)))
  {
    if (Core::Communication::my_mpi_rank(arguments.comms.local_comm()) == 0)
    {
      printf("\n\n");
      print_help_message();
      printf("\n\n");
    }
  }
  else if ((argc == 2) && ((strcmp(argv[1], "-p") == 0) || (strcmp(argv[1], "--parameters") == 0)))
  {
    if (Core::Communication::my_mpi_rank(arguments.comms.local_comm()) == 0)
    {
      ryml::Tree tree = Core::IO::init_yaml_tree_with_exceptions();
      ryml::NodeRef root = tree.rootref();
      root |= ryml::MAP;
      Core::IO::YamlNodeRef root_ref(root, "");

      // Write the non-user input metadata that is defined globally for 4C.
      emit_general_metadata(root_ref);

      // Write the user input defined for various physics module.
      Core::IO::InputFile input_file = setup_input_file(arguments.comms.local_comm());
      input_file.emit_metadata(root_ref);

      // Finally, dump everything.
      std::cout << tree;
    }
  }
  else
  {
    if (Core::Communication::my_mpi_rank(arguments.comms.global_comm()) == 0)
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
                << Core::Communication::num_mpi_ranks(arguments.comms.global_comm()) << '\n';
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
      run(arguments);
    }
    catch (Core::Exception& err)
    {
      char line[] = "=========================================================================\n";
      std::cout << "\n\n"
                << line << err.what_with_stacktrace() << "\n"
                << line << "\n"
                << std::endl;

      if (arguments.comms.num_groups() > 1)
      {
        printf("Global processor %d has thrown an error and is waiting for the remaining procs\n\n",
            Core::Communication::my_mpi_rank(arguments.comms.global_comm()));
        Core::Communication::barrier(arguments.comms.global_comm());
      }

      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
#endif
    /*----------------------------------------------------------------------*/

    get_memory_high_water_mark(arguments.comms.global_comm());

    Core::Communication::barrier(arguments.comms.local_comm());
  }

  communicators.finalize();
  return (0);
}

void run(CommandlineArguments& arguments)
{
  parse_commandline_arguments(arguments);

  /* input phase, input of all information */
  global_legacy_module_callbacks().RegisterParObjectTypes();
  double t0 = walltime_in_seconds();

  // and now the actual reading
  Core::IO::InputFile input_file = setup_input_file(arguments.comms.local_comm());
  input_file.read(arguments.input_file_name);
  setup_global_problem(input_file, arguments);

  // we wait till all procs are here. Otherwise a hang up might occur where
  // one proc ended with FOUR_C_THROW but other procs were not finished and waited...
  // we also want to have the printing above being finished.
  Core::Communication::barrier(arguments.comms.local_comm());


  const double ti = walltime_in_seconds() - t0;
  if (Core::Communication::my_mpi_rank(arguments.comms.global_comm()) == 0)
  {
    Core::IO::cout << "\nTotal wall time for INPUT:       " << std::setw(10) << std::setprecision(3)
                   << std::scientific << ti << " sec \n\n";
  }

  /*--------------------------------------------------calculation phase */
  t0 = walltime_in_seconds();

  entrypoint_switch();

  write_timemonitor(arguments.comms.local_comm());

  const double tc = walltime_in_seconds() - t0;
  if (Core::Communication::my_mpi_rank(arguments.comms.global_comm()) == 0)
  {
    Core::IO::cout << "\nTotal wall time for CALCULATION: " << std::setw(10) << std::setprecision(3)
                   << std::scientific << tc << " sec \n\n";
  }
}
