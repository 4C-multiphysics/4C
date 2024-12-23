// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"
#include "4C_config_revision.hpp"
#include "4C_config_trilinos_version.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_inpar_validconditions.hpp"
#include "4C_inpar_validcontactconstitutivelaw.hpp"
#include "4C_inpar_validmaterials.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Epetra_MpiComm.h>
#include <Kokkos_Core.hpp>
#include <unistd.h>

#include <csignal>
#include <iostream>

#ifdef FOUR_C_ENABLE_FE_TRAPPING
#include <cfenv>
#endif

using namespace FourC;

namespace
{

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

#ifdef FOUR_C_ENABLE_FE_TRAPPING
  /*!
   * \brief FPE signal handle
   *
   * A function to handle floating point exceptions by raising a FOUR_C_THROW.
   * So we get a stack-trace also on systems where this is not provided
   * through core-dumps from MPI_Abort() (e.g. OpenMPI does whereas
   * Intel MPI doesn't).
   */
  void sigfpe_handler(int sig)
  {
    std::string exception_string;
    switch (sig)
    {
      case FE_INVALID:
        exception_string = "FE_INVALID";
        break;
      case FE_DIVBYZERO:
        exception_string = "FE_DIVBYZERO";
        break;
      case FE_OVERFLOW:
        exception_string = "FE_OVERFLOW";
        break;
      case FE_UNDERFLOW:
        exception_string = "FE_UNDERFLOW";
        break;
      case FE_INEXACT:
        exception_string = "FE_INEXACT";
        break;
      default:
        FOUR_C_THROW("4C produced an unknown floating point exception.");
        break;
    }
    FOUR_C_THROW("4C produced a %s floating point exception.", exception_string.c_str());
  }
#endif

}  // namespace

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ntam(int argc, char *argv[]);

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
int main(int argc, char *argv[])
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

  std::shared_ptr<Core::Communication::Communicators> communicators =
      Core::Communication::create_comm(std::vector<std::string>(argv, argv + argc));
  Global::Problem::instance()->set_communicators(communicators);
  MPI_Comm lcomm = communicators->local_comm();
  MPI_Comm gcomm = communicators->global_comm();
  int ngroups = communicators->num_groups();

  if (strcmp(argv[argc - 1], "--interactive") == 0)
  {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Global rank %d with PID %d on %s is ready for attach\n",
        Core::Communication::my_mpi_rank(gcomm), getpid(), hostname);
    if (Core::Communication::my_mpi_rank(gcomm) == 0)
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

  Core::Communication::barrier(gcomm);

  global_legacy_module_callbacks().RegisterParObjectTypes();

  if ((argc == 2) && ((strcmp(argv[1], "-h") == 0) || (strcmp(argv[1], "--help") == 0)))
  {
    if (Core::Communication::my_mpi_rank(lcomm) == 0)
    {
      printf("\n\n");
      print_help_message();
      printf("\n\n");
    }
  }
  else if ((argc == 2) && ((strcmp(argv[1], "-p") == 0) || (strcmp(argv[1], "--parameters") == 0)))
  {
    if (Core::Communication::my_mpi_rank(lcomm) == 0)
    {
      auto valid_parameters = Input::valid_parameters();
      Core::IO::InputFileUtils::print_metadata_yaml(std::cout, *valid_parameters);
    }
  }
  else if ((argc == 2) && ((strcmp(argv[1], "-d") == 0) || (strcmp(argv[1], "--datfile") == 0)))
  {
    if (Core::Communication::my_mpi_rank(lcomm) == 0)
    {
      printf("\n\n");
      print_default_dat_header();
      print_condition_dat_header();
      print_material_dat_header();
      print_contact_constitutive_law_dat_header();

      const auto lines = Core::FE::valid_cloning_material_map_lines();
      Core::IO::InputFileUtils::print_section(std::cout, "CLONING MATERIAL MAP", lines);

      print_element_dat_header();

      const std::vector<Input::LineDefinition> result_lines =
          global_legacy_module_callbacks().valid_result_description_lines();
      Core::IO::InputFileUtils::print_section(std::cout, "RESULT DESCRIPTION", result_lines);

      printf("\n\n");
    }
  }
  else
  {
    if (Core::Communication::my_mpi_rank(gcomm) == 0)
    {
      printf(
          "\n"
          "**********************************************\n"
          "*                                            *\n"
          "*                     4C                     *\n"
          "*                                            *\n"
          "*                                            *\n"
          "*             version (git SHA1)             *\n"
          "*  %s  *\n"
          "*                                            *\n"
          "*                                            *\n"
          "**********************************************\n\n",
          VersionControl::git_hash);
      printf(
          "Trilinos Version %s (git SHA1 %s)\n", TrilinosVersion.c_str(), TrilinosGitHash.c_str());
      printf("Total number of processors: %d\n", Core::Communication::num_mpi_ranks(gcomm));
    }

    /* Here we turn the NaN and INF numbers off. No need to calculate
     * those. If those appear, the calculation needs much (!) more
     * time. Better stop immediately if some illegal operation occurs. */
#ifdef FOUR_C_ENABLE_FE_TRAPPING

    /* This is a GNU extension thus it's only available on linux. But
     * it's exactly what we want: SIGFPE just for the given
     * exceptions. We don't care about FE_INEXACT. (It happens all the
     * time.) */
    /* Over- and underflow seem to happen sometimes. Does it worry us?
     * Will that spoil the results? */
    /*feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_UNDERFLOW | FE_OVERFLOW);*/
    feclearexcept(FE_ALL_EXCEPT);
    feenableexcept(FE_INVALID | FE_DIVBYZERO);

    // Initialize a signal handle for SIGFPE
    struct sigaction act;
    act.sa_handler = sigfpe_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(SIGFPE, &act, nullptr);

#endif

/*----------------------------------------------- everything is in here */
#ifdef FOUR_C_ENABLE_CORE_DUMP
    ntam(argc, argv);
#else
    try
    {
      ntam(argc, argv);
    }
    catch (Core::Exception &err)
    {
      char line[] = "=========================================================================\n";
      std::cout << "\n\n"
                << line << err.what_with_stacktrace() << "\n"
                << line << "\n"
                << std::endl;

      if (ngroups > 1)
      {
        printf("Global processor %d has thrown an error and is waiting for the remaining procs\n\n",
            Core::Communication::my_mpi_rank(gcomm));
        Core::Communication::barrier(gcomm);
      }

      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
#endif
    /*----------------------------------------------------------------------*/

    get_memory_high_water_mark(gcomm);

    Core::Communication::barrier(lcomm);
    if (ngroups > 1)
    {
      printf("Global processor %d with local rank %d finished normally\n",
          Core::Communication::my_mpi_rank(gcomm), Core::Communication::my_mpi_rank(lcomm));
      Core::Communication::barrier(gcomm);
    }
    else
    {
      Core::Communication::barrier(gcomm);
      printf("processor %d finished normally\n", Core::Communication::my_mpi_rank(lcomm));
    }
  }

  return (0);
}
