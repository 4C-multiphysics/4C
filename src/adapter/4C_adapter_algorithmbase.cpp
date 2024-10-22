// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_algorithmbase.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_io_pstream.hpp"
#include "4C_utils_exceptions.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::AlgorithmBase::AlgorithmBase(
    const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams)
    : comm_(comm), printscreen_(Global::Problem::instance()->io_params().get<int>("STDOUTEVRY"))
{
  step_ = 0;
  time_ = 0.;
  dt_ = timeparams.get<double>("TIMESTEP");
  nstep_ = timeparams.get<int>("NUMSTEP");
  maxtime_ = timeparams.get<double>("MAXTIME");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::AlgorithmBase::set_time_step(const double time, const int step)
{
  step_ = step;
  time_ = time;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::AlgorithmBase::print_header()
{
  if (get_comm().MyPID() == 0 and printscreen_ and (step_ % printscreen_ == 0))
  {
    Core::IO::cout << "\n"
                   << method_ << "\n"
                   << "TIME:  " << std::scientific << time_ << "/" << std::scientific << maxtime_
                   << "     DT = " << std::scientific << dt_ << "     STEP = " << std::setw(4)
                   << step_ << "/" << std::setw(4) << nstep_ << "\n\n";
  }
}

FOUR_C_NAMESPACE_CLOSE
