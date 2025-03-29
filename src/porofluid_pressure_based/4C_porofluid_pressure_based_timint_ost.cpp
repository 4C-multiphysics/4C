// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_timint_ost.hpp"

#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_porofluid_pressure_based_ele_action.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                   vuong  08/16 |
 *----------------------------------------------------------------------*/
POROFLUIDMULTIPHASE::TimIntOneStepTheta::TimIntOneStepTheta(
    std::shared_ptr<Core::FE::Discretization> dis,  //!< discretization
    const int linsolvernumber,                      //!< number of linear solver
    const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& poroparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< output writer
    )
    : TimIntImpl(dis, linsolvernumber, probparams, poroparams, output),
      theta_(poroparams.get<double>("THETA"))
{
}



/*----------------------------------------------------------------------*
 |  set parameter for element evaluation                    vuong 06/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::set_element_time_step_parameter() const
{
  Teuchos::ParameterList eleparams;

  // the total time definitely changes
  eleparams.set<double>("total time", time_);
  // we set the time step and related, just in case we want adaptive time stepping
  eleparams.set<double>("time-step length", dt_);
  eleparams.set<double>("time factor", theta_ * dt_);

  Discret::Elements::PoroFluidMultiPhaseEleParameter::instance(discret_->name())
      ->set_time_step_parameters(eleparams);
}


/*-----------------------------------------------------------------------------*
 | set time for evaluation of POINT -Neumann boundary conditions   vuong 08/16 |
 *----------------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::set_time_for_neumann_evaluation(
    Teuchos::ParameterList& params)
{
  params.set("total time", time_);
}


/*----------------------------------------------------------------------*
| Print information about current time step to screen      vuong 08/16  |
*-----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::print_time_step_info()
{
  if (myrank_ == 0)
  {
    std::cout << "| TIME: " << std::setw(11) << std::setprecision(4) << std::scientific << time_
              << "/" << std::setw(11) << std::setprecision(4) << std::scientific << maxtime_
              << "  DT = " << std::setw(11) << std::setprecision(4) << std::scientific << dt_
              << "  "
              << "One-Step-Theta (theta = " << std::setw(3) << std::setprecision(2) << theta_
              << ") STEP = " << std::setw(4) << step_ << "/" << std::setw(4) << stepmax_
              << "            |" << '\n';
  }
}


/*----------------------------------------------------------------------*
 | set part of the residual vector belonging to the old timestep        |
 |                                                          vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::set_old_part_of_righthandside()
{
  // hist_ = phin_ + dt*(1-Theta)*phidtn_
  hist_->update(1.0, *phin_, dt_ * (1.0 - theta_), *phidtn_, 0.0);
}


/*----------------------------------------------------------------------*
 | perform an explicit predictor step                       vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::explicit_predictor()
{
  phinp_->update(dt_, *phidtn_, 1.0);
}


/*----------------------------------------------------------------------*
 | add actual Neumann loads                                             |
 | scaled with a factor resulting from time discretization  vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::add_neumann_to_residual()
{
  residual_->update(theta_ * dt_, *neumann_loads_, 1.0);
}


/*----------------------------------------------------------------------------*
 | add global state vectors specific for time-integration scheme  vuong 08/16 |
 *---------------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::add_time_integration_specific_vectors()
{
  discret_->set_state("hist", *hist_);
  discret_->set_state("phinp_fluid", *phinp_);
  discret_->set_state("phin_fluid", *phin_);
  discret_->set_state("phidtnp", *phidtnp_);
}


/*----------------------------------------------------------------------*
 | compute time derivative                                  vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::compute_time_derivative()
{
  // time derivative of phi:
  // *phidt(n+1) = (phi(n+1)-phi(n)) / (theta*dt) + (1-(1/theta))*phidt(n)
  const double fact1 = 1.0 / (theta_ * dt_);
  const double fact2 = 1.0 - (1.0 / theta_);
  phidtnp_->update(fact2, *phidtn_, 0.0);
  phidtnp_->update(fact1, *phinp_, -fact1, *phin_, 1.0);
}


/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                          vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::update()
{
  // call base class
  POROFLUIDMULTIPHASE::TimIntImpl::update();

  // compute time derivative at time n+1
  compute_time_derivative();

  // solution of this step becomes most recent solution of the last step
  phin_->update(1.0, *phinp_, 0.0);

  // time deriv. of this step becomes most recent time derivative of
  // last step
  phidtn_->update(1.0, *phidtnp_, 0.0);
}


/*----------------------------------------------------------------------*
 | write additional data required for restart               vuong 08/16 |
 *----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::output_restart()
{
  POROFLUIDMULTIPHASE::TimIntImpl::output_restart();

  // additional state vectors that are needed for One-Step-Theta restart
  output_->write_vector("phidtn_fluid", phidtn_);
  output_->write_vector("phin_fluid", phin_);
}


/*----------------------------------------------------------------------*
 |  read restart data                                       vuong 08/16 |
 -----------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::read_restart(const int step)
{
  // call base class
  POROFLUIDMULTIPHASE::TimIntImpl::read_restart(step);

  std::shared_ptr<Core::IO::DiscretizationReader> reader(nullptr);
  reader = std::make_shared<Core::IO::DiscretizationReader>(
      discret_, Global::Problem::instance()->input_control_file(), step);

  time_ = reader->read_double("time");
  step_ = reader->read_int("step");

  if (myrank_ == 0)
    std::cout << "Reading POROFLUIDMULTIPHASE restart data (time=" << time_ << " ; step=" << step_
              << ")" << '\n';

  // read state vectors that are needed for One-Step-Theta restart
  reader->read_vector(phinp_, "phinp_fluid");
  reader->read_vector(phin_, "phin_fluid");
  reader->read_vector(phidtn_, "phidtn_fluid");
}

/*--------------------------------------------------------------------*
 | calculate init time derivatives of state variables kremheller 03/17 |
 *--------------------------------------------------------------------*/
void POROFLUIDMULTIPHASE::TimIntOneStepTheta::calc_initial_time_derivative()
{
  // standard general element parameter without stabilization
  set_element_general_parameters();

  // we also have to modify the time-parameter list (incremental solve)
  // actually we do not need a time integration scheme for calculating the initial time derivatives,
  // but the rhs of the standard element routine is used as starting point for this special system
  // of equations. Therefore, the rhs vector has to be scaled correctly.
  set_element_time_step_parameter();

  // call core algorithm
  TimIntImpl::calc_initial_time_derivative();

  // and finally undo our temporary settings
  set_element_general_parameters();
  set_element_time_step_parameter();
}

FOUR_C_NAMESPACE_CLOSE
