// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pasi_utils.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
void PaSI::Utils::change_time_parameter(const Epetra_Comm& comm,
    const Teuchos::ParameterList& pasi_params, Teuchos::ParameterList& particle_params,
    Teuchos::ParameterList& struct_params)
{
  // the default time step size
  particle_params.set<double>("TIMESTEP", pasi_params.get<double>("TIMESTEP"));
  struct_params.set<double>("TIMESTEP", pasi_params.get<double>("TIMESTEP"));

  // maximum number of timesteps
  particle_params.set<int>("NUMSTEP", pasi_params.get<int>("NUMSTEP"));
  struct_params.set<int>("NUMSTEP", pasi_params.get<int>("NUMSTEP"));

  // maximum simulation time
  particle_params.set<double>("MAXTIME", pasi_params.get<double>("MAXTIME"));
  struct_params.set<double>("MAXTIME", pasi_params.get<double>("MAXTIME"));

  // solution output
  particle_params.set<int>("RESULTSEVRY", pasi_params.get<int>("RESULTSEVRY"));
  struct_params.set<int>("RESULTSEVRY", pasi_params.get<int>("RESULTSEVRY"));

  // restart
  particle_params.set<int>("RESTARTEVRY", pasi_params.get<int>("RESTARTEVRY"));
  struct_params.set<int>("RESTARTEVRY", pasi_params.get<int>("RESTARTEVRY"));

  if (comm.MyPID() == 0)
  {
    std::cout << "================= Overview of chosen time stepping: =================="
              << std::endl;
    std::cout << std::setw(20) << "" << std::setw(15) << "PASI" << std::setw(15) << "Particles"
              << std::setw(15) << "Structure" << std::endl;
    // Timestep
    std::cout << std::setw(20) << "Timestep:" << std::scientific << std::setprecision(4)
              << std::setw(15) << pasi_params.get<double>("TIMESTEP") << std::setw(15)
              << particle_params.get<double>("TIMESTEP") << std::setw(15)
              << struct_params.get<double>("TIMESTEP") << std::endl;
    // Numstep
    std::cout << std::setw(20) << "Numstep:" << std::scientific << std::setprecision(4)
              << std::setw(15) << pasi_params.get<int>("NUMSTEP") << std::setw(15)
              << particle_params.get<int>("NUMSTEP") << std::setw(15)
              << struct_params.get<int>("NUMSTEP") << std::endl;
    // Maxtime
    std::cout << std::setw(20) << "Maxtime:" << std::scientific << std::setprecision(4)
              << std::setw(15) << pasi_params.get<double>("MAXTIME") << std::setw(15)
              << particle_params.get<double>("MAXTIME") << std::setw(15)
              << struct_params.get<double>("MAXTIME") << std::endl;
    // Result every step
    std::cout << std::setw(20) << "Result every step:" << std::setw(15)
              << pasi_params.get<int>("RESULTSEVRY") << std::endl;
    // Restart every step
    std::cout << std::setw(20) << "Restart every step:" << std::setw(15)
              << pasi_params.get<int>("RESTARTEVRY") << std::endl;
    std::cout << "======= currently equal for both structure and particle field ========"
              << std::endl;
  }
}

void PaSI::Utils::logo()
{
  std::cout << "============================ Welcome to =============================="
            << std::endl;
  std::cout << "  ___          _   _    _       ___ _               _                 "
            << std::endl;
  std::cout << " | _ \\__ _ _ _| |_(_)__| |___  / __| |_ _ _ _  _ __| |_ _  _ _ _ ___  "
            << std::endl;
  std::cout << " |  _/ _` | '_|  _| / _| / -_) \\__ \\  _| '_| || / _|  _| || | '_/ -_) "
            << std::endl;
  std::cout << " |_| \\__,_|_|  \\__|_\\__|_\\___| |___/\\__|_|  \\_,_\\__|\\__|\\_,_|_| \\___| "
            << std::endl;
  std::cout << "              ___     _                   _   _                       "
            << std::endl;
  std::cout << "             |_ _|_ _| |_ ___ _ _ __ _ __| |_(_)___ _ _               "
            << std::endl;
  std::cout << "              | || ' \\  _/ -_) '_/ _` / _|  _| / _ \\ ' \\              "
            << std::endl;
  std::cout << "             |___|_||_\\__\\___|_| \\__,_\\__|\\__|_\\___/_||_|             "
            << std::endl;
  std::cout << "" << std::endl;
  std::cout << "======================================================================"
            << std::endl;
}

FOUR_C_NAMESPACE_CLOSE
