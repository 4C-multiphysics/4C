// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_adapter.hpp"

#include "4C_global_data.hpp"
#include "4C_io_pstream.hpp"
#include "4C_thermo_input.hpp"
#include "4C_thermo_timint_genalpha.hpp"
#include "4C_thermo_timint_ost.hpp"
#include "4C_thermo_timint_statics.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Thermo::BaseAlgorithm::BaseAlgorithm(
    const Teuchos::ParameterList& prbdyn, std::shared_ptr<Core::FE::Discretization> actdis)
{
  if (not actdis->filled()) actdis->fill_complete();

  // context for output and restart
  std::shared_ptr<Core::IO::DiscretizationWriter> output = actdis->writer();
  output->write_mesh(0, 0.0);

  // get input parameter lists and copy them, because a few parameters are overwritten
  Teuchos::ParameterList ioflags(Global::Problem::instance()->io_params());
  Teuchos::ParameterList parameters = Global::Problem::instance()->thermal_dynamic_params();
  Teuchos::ParameterList xparams;  // add extra parameters (a kind of work-around)

  // overrule certain parameters for coupled problems
  parameters.set<double>("TIMESTEP", prbdyn.get<double>("TIMESTEP"));
  parameters.set<double>("MAXTIME", prbdyn.get<double>("MAXTIME"));
  parameters.set<int>("NUMSTEP", prbdyn.get<int>("NUMSTEP"));
  parameters.set<int>("RESTARTEVERY", prbdyn.get<int>("RESTARTEVERY"));
  parameters.set<int>("RESULTSEVERY", prbdyn.get<int>("RESULTSEVERY"));

  // get the solver number used for thermal solver
  const int linsolvernumber = parameters.get<int>("LINEAR_SOLVER");
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "No linear solver defined for thermal solver. Please set LINEAR_SOLVER in THERMAL DYNAMIC "
        "to a valid number!");

  // create a linear solver
  Teuchos::ParameterList solveparams;
  std::shared_ptr<Core::LinAlg::Solver> solver = std::make_shared<Core::LinAlg::Solver>(
      Global::Problem::instance()->solver_params(linsolvernumber), actdis->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  actdis->compute_null_space_if_necessary(solver->params());

  // create marching time integrator
  auto timinttype = Teuchos::getIntegralValue<Thermo::DynamicType>(parameters, "DYNAMICTYPE");

  switch (timinttype)
  {
    case Thermo::dyna_statics:
    {
      thermo_ = std::make_shared<Thermo::TimIntStatics>(
          ioflags, parameters, xparams, actdis, solver, output);
      break;
    }
    case Thermo::dyna_onesteptheta:
    {
      thermo_ = std::make_shared<Thermo::TimIntOneStepTheta>(
          ioflags, parameters, xparams, actdis, solver, output);
      break;
    }
    case Thermo::dyna_genalpha:
    {
      thermo_ = std::make_shared<Thermo::TimIntGenAlpha>(
          ioflags, parameters, xparams, actdis, solver, output);
      break;
    }
    default:
      FOUR_C_THROW("Unknown time integration scheme '%s'!", timinttype);
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Thermo::Adapter::integrate()
{
  while (not_finished())
  {
    prepare_time_step();

    Thermo::ConvergenceStatus convStatus = solve();

    switch (convStatus)
    {
      case Thermo::conv_success:
        update();
        print_step();
        output();
        break;
      case Thermo::conv_fail_repeat:
        continue;
      default:
        FOUR_C_THROW("Solver failed.");
    }
  }

  Teuchos::TimeMonitor::summarize();
}

FOUR_C_NAMESPACE_CLOSE
