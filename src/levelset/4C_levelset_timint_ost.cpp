// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_levelset_timint_ost.hpp"

#include "4C_inpar_validparameters.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_scatra_ele_action.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                rasthofer 09/13 |
 *----------------------------------------------------------------------*/
ScaTra::LevelSetTimIntOneStepTheta::LevelSetTimIntOneStepTheta(
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, sctratimintparams, extraparams, output),
      LevelSetAlgorithm(actdis, solver, params, sctratimintparams, extraparams, output),
      TimIntOneStepTheta(actdis, solver, sctratimintparams, extraparams, output)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             rauch 09/16 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::init()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntOneStepTheta::init();
  LevelSetAlgorithm::init();

  return;
}

/*----------------------------------------------------------------------*
 |  setup time integration                                  rauch 09/16 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::setup()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntOneStepTheta::setup();
  LevelSetAlgorithm::setup();

  return;
}


/*----------------------------------------------------------------------*
| Print information about current time step to screen   rasthofer 09/13 |
*-----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::print_time_step_info()
{
  if (myrank_ == 0)
  {
    if (not switchreinit_)
      TimIntOneStepTheta::print_time_step_info();
    else
    {
      if (reinitaction_ == Inpar::ScaTra::reinitaction_sussman)
        printf("\nPSEUDOTIMESTEP: %11.4E      %s          THETA = %11.4E   PSEUDOSTEP = %4d/%4d \n",
            dtau_, method_title().c_str(), thetareinit_, pseudostep_, pseudostepmax_);
      else if (reinitaction_ == Inpar::ScaTra::reinitaction_ellipticeq)
        printf("\nREINIT ELLIPTIC:\n");
    }
  }
  return;
}


/*--------------------------------------------------------------------------- *
 | calculate consistent initial scalar time derivatives in compliance with    |
 | initial scalar field                                            fang 09/15 |
 *----------------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::calc_initial_time_derivative()
{
  if (not switchreinit_)
    TimIntOneStepTheta::calc_initial_time_derivative();

  else
  {
    // set element parameters with stabilization and artificial diffusivity deactivated
    set_reinitialization_element_parameters(true);

    // note: time-integration parameter list has not to be overwritten here, since we rely on
    // incremental solve
    //       as already set in prepare_time_loop_reinit()

    // compute time derivative of phi at pseudo-time tau=0
    ScaTraTimIntImpl::calc_initial_time_derivative();

    // eventually, undo changes in general parameter list
    set_reinitialization_element_parameters();
  }

  return;
}


/*----------------------------------------------------------------------*
 | set part of the residual vector belonging to the old timestep        |
 |                                                      rasthofer 12/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::set_old_part_of_righthandside()
{
  if (not switchreinit_)
    TimIntOneStepTheta::set_old_part_of_righthandside();
  else
    // hist_ = phin_ + dt*(1-Theta)*phidtn_
    hist_->Update(1.0, *phin_, dtau_ * (1.0 - thetareinit_), *phidtn_, 0.0);

  return;
}


/*----------------------------------------------------------------------*
 | extended version for coupled level-set problems                      |
 | including reinitialization                           rasthofer 01/14 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::update()
{
  // -----------------------------------------------------------------
  //                     reinitialize level-set
  // -----------------------------------------------------------------
  // will be done only if required
  reinitialization();

  // -------------------------------------------------------------------
  //                         update solution
  //        current solution becomes old solution of next time step
  // -------------------------------------------------------------------
  update_state();

  return;
}


/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next time step      |
 |                                                      rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::update_state()
{
  if (not switchreinit_)
  {
    // compute time derivative at time n+1
    compute_time_derivative();

    // after the next command (time shift of solutions) do NOT call
    // compute_time_derivative() anymore within the current time step!!!

    // solution of this step becomes most recent solution of the last step
    phin_->Update(1.0, *phinp_, 0.0);

    // time deriv. of this step becomes most recent time derivative of
    // last step
    phidtn_->Update(1.0, *phidtnp_, 0.0);
  }
  else
  {
    // solution of this step becomes most recent solution of the last step
    phin_->Update(1.0, *phinp_, 0.0);

    // reinitialization is done, reset flag
    if (switchreinit_ == true) switchreinit_ = false;

    // we also have reset the time-integration parameter list, since incremental solver has to be
    // overwritten if used
    set_element_time_parameter(true);

    // compute time derivative at time n (and n+1)
    ScaTraTimIntImpl::calc_initial_time_derivative();

    // reset element time-integration parameters
    set_element_time_parameter();
  }

  return;
}


/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 | used within reinitialization loop                    rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::update_reinit()
{
  // TODO: Fkt hier raus nehmen
  // compute time derivative at time n+1
  // time derivative of phi:
  // phidt(n+1) = (phi(n+1)-phi(n)) / (theta*dt) + (1-(1/theta))*phidt(n)
  const double fact1 = 1.0 / (thetareinit_ * dtau_);
  const double fact2 = 1.0 - (1.0 / thetareinit_);
  phidtnp_->Update(fact2, *phidtn_, 0.0);
  phidtnp_->Update(fact1, *phinp_, -fact1, *phin_, 1.0);

  // solution of this step becomes most recent solution of the last step
  phin_->Update(1.0, *phinp_, 0.0);

  // time deriv. of this step becomes most recent time derivative of
  // last step
  phidtn_->Update(1.0, *phidtnp_, 0.0);

  return;
}


/*--------------------------------------------------------------------------------------------*
 | Redistribute the scatra discretization and vectors according to nodegraph  rasthofer 07/11 |
 |                                                                            DA wichmann     |
 *--------------------------------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::redistribute(Core::LinAlg::Graph& nodegraph)
{
  // let the base class do the basic redistribution and transfer of the base class members
  LevelSetAlgorithm::redistribute(nodegraph);

  // now do all the ost specific steps
  const Epetra_Map* newdofrowmap = discret_->dof_row_map();
  std::shared_ptr<Core::LinAlg::Vector<double>> old;

  if (fsphinp_ != nullptr)
  {
    old = fsphinp_;
    fsphinp_ = Core::LinAlg::create_vector(*newdofrowmap, true);
    Core::LinAlg::export_to(*old, *fsphinp_);
  }

  return;
}


/*----------------------------------------------------------------------*
 | setup problem after restart                          rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::LevelSetTimIntOneStepTheta::read_restart(
    const int step, std::shared_ptr<Core::IO::InputControl> input)
{
  // do basic restart
  TimIntOneStepTheta::read_restart(step, input);

  return;
}


/*----------------------------------------------------------------------*
 | interpolate phi to intermediate time level n+theta   rasthofer 09/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> ScaTra::LevelSetTimIntOneStepTheta::phinptheta(
    const double theta_inter)
{
  const Epetra_Map* dofrowmap = discret_->dof_row_map();
  std::shared_ptr<Core::LinAlg::Vector<double>> phi_tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  phi_tmp->Update((1.0 - theta_inter), *phin_, theta_inter, *phinp_, 0.0);
  return phi_tmp;
}


/*----------------------------------------------------------------------*
 | interpolate phidt to intermediate time level n+theta rasthofer 09/14 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> ScaTra::LevelSetTimIntOneStepTheta::phidtnptheta(
    const double theta_inter)
{
  const Epetra_Map* dofrowmap = discret_->dof_row_map();
  std::shared_ptr<Core::LinAlg::Vector<double>> phidt_tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  phidt_tmp->Update((1.0 - theta_inter), *phidtn_, theta_inter, *phidtnp_, 0.0);
  return phidt_tmp;
}

FOUR_C_NAMESPACE_CLOSE
