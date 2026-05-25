// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_timeada.hpp"

#include "4C_adapter_str_timeada_joint.hpp"
#include "4C_adapter_str_timeada_zienxie.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::StructureTimeAda::StructureTimeAda(
    Global::Problem& problem, std::shared_ptr<Structure> structure)
    : StructureWrapper(structure), problem_(problem)
{
  stm_ = std::dynamic_pointer_cast<Solid::TimeInt::Base>(structure_);

  if (stm_ == nullptr) FOUR_C_THROW("cast from Adapter::Structure to Solid::TimeInt::Base failed");

  // call the setup once if stm_ has been setup
  if (stm_->is_setup()) setup_time_ada();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Adapter::Structure> Adapter::StructureTimeAda::create(Global::Problem& problem,
    const Teuchos::ParameterList& taflags,  //!< adaptive input flags
    std::shared_ptr<Solid::TimeInt::Base> ti_strategy)
{
  auto kind = Teuchos::getIntegralValue<Solid::TimAdaKind>(taflags, "KIND");
  switch (kind)
  {
    case Solid::timada_kind_zienxie:
      // Adaptive time integration with Zienkiewicz-Xie error indicator
      return std::make_shared<Adapter::StructureTimeAdaZienXie>(problem, ti_strategy);

    case Solid::timada_kind_joint_explicit:
      // Adaptive time integration using auxiliary time integrator
      return std::make_shared<Adapter::StructureTimeAdaJoint>(problem, ti_strategy);

    default:
      // Unknown adaptive time integration
      return nullptr;
  }

  return nullptr;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAda::setup()
{
  // call the wrapper setup
  StructureWrapper::setup();

  // self setup
  setup_time_ada();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAda::setup_time_ada()
{
  const Teuchos::ParameterList& sdynparams = problem_.structural_dynamic_params();

  // initialize the local variables
  timeinitial_ = 0.0;
  timefinal_ = sdynparams.get<double>("MAXTIME");
  if (timefinal_ <= timeinitial_) FOUR_C_THROW("MAXTIME is not positive. It is invalid.");
  timedirect_ = 1.0;
  timestepinitial_ = 0;
  timestepfinal_ = sdynparams.get<int>("NUMSTEP");
  stepsizeinitial_ = sdynparams.get<double>("TIMESTEP");

  const Teuchos::ParameterList& tap = sdynparams.sublist("TIMEADAPTIVITY");

  stepsizemax_ = tap.get<double>("STEPSIZEMAX");
  stepsizemin_ = tap.get<double>("STEPSIZEMIN");
  sizeratiomax_ = tap.get<double>("SIZERATIOMAX");
  sizeratiomin_ = tap.get<double>("SIZERATIOMIN");
  sizeratioscale_ = tap.get<double>("SIZERATIOSCALE");
  errctrl_ = ctrl_dis;  // PROVIDE INPUT PARAMETER
  errnorm_ = Teuchos::getIntegralValue<Solid::VectorNorm>(tap, "LOCERRNORM");
  errtol_ = tap.get<double>("LOCERRTOL");
  errorder_ = 1;  // CHANGE THIS CONSTANT
  adaptstepmax_ = tap.get<int>("ADAPTSTEPMAX");

  time_ = timeinitial_;
  timestep_ = 0;
  stepsizepre_ = stepsizeinitial_;
  stepsize_ = sdynparams.get<double>("TIMESTEP");
  adaptstep_ = 0;

  outsys_ = false;
  outstr_ = false;
  outene_ = false;
  outrest_ = false;
  outsysperiod_ = tap.get<double>("OUTSYSPERIOD");
  outstrperiod_ = tap.get<double>("OUTSTRPERIOD");
  outeneperiod_ = tap.get<double>("OUTENEPERIOD");
  outrestperiod_ = tap.get<double>("OUTRESTPERIOD");
  outsizeevery_ = tap.get<int>("OUTSIZEEVERY");
  outsystime_ = timeinitial_ + outsysperiod_;
  outstrtime_ = timeinitial_ + outstrperiod_;
  outenetime_ = timeinitial_ + outeneperiod_;
  outresttime_ = timeinitial_ + outrestperiod_;
  outsizefile_ = nullptr;

  // allocate displacement local error vector
  locerrdisn_ = std::make_shared<Core::LinAlg::Vector<double>>(*(stm_->dof_row_map()), true);

  // enable restart for adaptive timestepping
  const int restart = problem_.restart();
  if (restart)
  {
    // read restart of marching time-integrator and reset initial time and step for adaptive loop
    stm_->read_restart(restart);
    timeinitial_ = stm_->time_old();
    timestepinitial_ = stm_->step_old();
    Core::IO::DiscretizationReader ioreader(
        *stm_->discretization(), problem_.input_control_file(), timestepinitial_);
    stepsizepre_ = ioreader.read_double("next_delta_time");
    time_ = timeinitial_;

    // update variables which depend on initial time and step
    timedirect_ = timefinal_ > timeinitial_ ? 1.0 : -1.0;
    outsystime_ = timeinitial_ + outsysperiod_;
    outstrtime_ = timeinitial_ + outstrperiod_;
    outenetime_ = timeinitial_ + outeneperiod_;
    outresttime_ = timeinitial_ + outrestperiod_;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeAda::read_restart(int step)
{
  setup_time_ada();
  setup_auxiliary();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Adapter::StructureTimeAda::integrate()
{
  // error checking variables
  Solid::ConvergenceStatus convergencestatus = Solid::conv_success;

  int myrank = Core::Communication::my_mpi_rank(stm_->discretization()->get_comm());

  // finalize initialization
  // (only relevant if an auxiliary time integrator is used)
  // buih STR:TimeInt::Base should be initialized outside
  // stm_->init();

  // Richardson extrapolation to no avail
  if (method_adapt_dis() == ada_ident)
    FOUR_C_THROW(
        "This combination is not implemented ... Richardson's extrapolation ... Yoshida technique "
        "...");

  // initialise time loop
  time_ = timeinitial_;
  timestep_ = timestepinitial_;
  stepsize_ = stepsizepre_;

  // time loop
  while ((time_ < timefinal_) and (timestep_ < timestepfinal_))
  {
    // time step size adapting loop
    adaptstep_ = 0;
    bool accepted = false;
    double stpsiznew;
    while ((not accepted) and (adaptstep_ < adaptstepmax_))
    {
      // modify step-size #stepsize_ according to output period
      // and store output type on #outstep_
      size_for_output();

      // set current step size
      stm_->set_delta_time(stepsize_);
      stm_->set_time_np(time_ + stepsize_);

      // integrate system with auxiliary TIS
      // we hold \f$D_{n+1}^{AUX}\f$ on #locdiserrn_
      // and \f$V_{n+1}^{AUX}\f$ on #locvelerrn_
      integrate_step_auxiliary();

      // call the predictor
      pre_predict();
      prepare_time_step();

      // integrate system with marching TIS and
      // stm_->IntegrateStep();
      pre_solve();
      convergencestatus = solve();

      if (convergencestatus != Solid::conv_success)
      {
        // if not converged, then we have to restart the step over
        accepted = false;

        // get the divergence action
        Solid::DivContAct div_action = stm_->data_sdyn().get_divergence_action();

        convergencestatus = perform_error_action(div_action, stpsiznew);
      }

      if (convergencestatus == Solid::conv_success)
      {
        // get local error vector on locerrdisn_
        evaluate_local_error_dis();

        // check whether step passes
        indicate(accepted, stpsiznew);
      }

      // adjust step-size and prepare repetition of current step
      if (not accepted)
      {
        std::cout << "Repeating step " << timestep_ + 1 << "/" << timestepfinal_ << " at time "
                  << time_ << " with stepsize = " << stpsiznew << std::endl;
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - -"
                  << " - - - - - - - - - - - - - - -" << std::endl;

        stepsize_ = stpsiznew;

        reset_step();
      }

      // increment number of adapted step sizes in a row
      adaptstep_ += 1;
    }

    // update or break
    if (accepted)
    {
      if (myrank == 0) std::cout << "Step size accepted" << std::endl;
    }
    else if (adaptstep_ >= adaptstepmax_)
    {
      if (myrank == 0)
        std::cout << "Could not find acceptable time step size"
                  << " ... continuing" << std::endl;
    }
    else
    {
      FOUR_C_THROW("Do not know what to do");
    }

    // calculate stresses, strains and energies
    // note: this has to be done before the update since otherwise a potential
    // material history is overwritten
    constexpr bool force_prepare = false;
    prepare_output(force_prepare);

    // update displacements, velocities, accelerations
    // after this call we will have disn_==dis_, etc
    // update time and step
    // update everything on the element level
    pre_update();

    update();

    post_update();

    stepsizepre_ = stepsize_;
    stepsize_ = stpsiznew;

    // write output
    output();
    post_output();

    // print info about finished time step
    print_step();

    // update
    timestep_ += 1;
    time_ += stepsizepre_;
    stm_->set_step_n(timestep_);
    stm_->set_time_n(time_);
    stm_->set_delta_time(stepsize_);
    stm_->set_time_np(time_ + stepsize_);

    update_period();
    outrest_ = outsys_ = outstr_ = outene_ = false;

    update_auxiliary();

    // the user reads but rarely listens
    if (myrank == 0)
    {
      std::cout << "Step " << timestep_ + 1 << ", Time " << time_ << ", new StepSize " << stepsize_
                << std::endl;
    }
  }

  // that's it say what went wrong
  return convergencestatus;
}

/*----------------------------------------------------------------------*/
/*  Modify step size to hit precisely output period */
void Adapter::StructureTimeAda::size_for_output()
{
  // check output of restart data first
  if ((fabs(time_ + stepsize_) >= fabs(outresttime_)) and (outrestperiod_ != 0.0))

  {
    stepsize_ = outresttime_ - time_;
    outrest_ = true;
  }

  // check output of system vectors
  if ((fabs(time_ + stepsize_) >= fabs(outsystime_)) and (outsysperiod_ != 0.0))
  {
    stepsize_ = outsystime_ - time_;
    outsys_ = true;
    if (fabs(outsystime_) < fabs(outresttime_)) outrest_ = false;
  }

  // check output of stress/strain
  if ((fabs(time_ + stepsize_) >= fabs(outstrtime_)) and (outstrperiod_ != 0.0))
  {
    stepsize_ = outstrtime_ - time_;
    outstr_ = true;
    if (fabs(outstrtime_) < fabs(outresttime_)) outrest_ = false;
    if (fabs(outstrtime_) < fabs(outsystime_)) outsys_ = false;
  }

  // check output of energy
  if ((fabs(time_ + stepsize_) >= fabs(outenetime_)) and (outeneperiod_ != 0.0))
  {
    stepsize_ = outenetime_ - time_;
    outene_ = true;
    if (fabs(outenetime_) < fabs(outresttime_)) outrest_ = false;
    if (fabs(outenetime_) < fabs(outsystime_)) outsys_ = false;
    if (fabs(outenetime_) < fabs(outstrtime_)) outstr_ = false;
  }

  return;
}

/*----------------------------------------------------------------------*/
/* Output action */
void Adapter::StructureTimeAda::output(bool forced_writerestart)
{
  Solid::TimeInt::BaseDataIO& dataio = stm_->data_io();
  std::shared_ptr<Core::IO::DiscretizationWriter> output_ptr = dataio.get_output_ptr();

  StructureWrapper::output(forced_writerestart);
  output_ptr->write_double("next_delta_time", stepsize_);
}

/*----------------------------------------------------------------------*/
/* Evaluate local error vector */
void Adapter::StructureTimeAda::evaluate_local_error_dis()
{
  const Solid::TimeInt::Base& sti = *stm_;
  const auto& gstate = sti.data_global_state();

  if (method_adapt_dis() == ada_orderequal)
  {
    const double coeffmarch = sti.method_lin_err_coeff_dis();
    const double coeffaux = method_lin_err_coeff_dis();
    locerrdisn_->update(-1.0, *(gstate.get_dis_np()), 1.0);
    locerrdisn_->scale(coeffmarch / (coeffaux - coeffmarch));
  }
  else
  {
    // schemes do not have the same order of accuracy
    locerrdisn_->update(-1.0, *(gstate.get_dis_np()), 1.0);
  }

  // blank Dirichlet DOFs since they always carry the exact solution
  sti.get_dbc().apply_dirichlet_to_vector(*locerrdisn_);
}

/*----------------------------------------------------------------------*/
/* Indicate error and determine new step size */
void Adapter::StructureTimeAda::indicate(bool& accepted, double& stpsiznew)
{
  // norm of local discretisation error vector
  const int numneglect = stm_->get_dbc_map_extractor()->cond_map()->num_global_elements();
  const double norm = calculate_vector_norm(errnorm_, *locerrdisn_, numneglect);

  // check if acceptable
  accepted = (norm < errtol_);

  // debug
  int myrank = Core::Communication::my_mpi_rank(stm_->discretization()->get_comm());
  if (myrank == 0)
  {
    std::cout << "LocErrNorm " << std::scientific << norm << ", LocErrTol " << errtol_
              << ", Accept " << std::boolalpha << accepted << std::endl;
  }

  stpsiznew = calculate_dt(norm);
}

/*----------------------------------------------------------------------*/
/* Prepare repetition of current time step */
void Adapter::StructureTimeAda::reset_step()
{
  outrest_ = outsys_ = outstr_ = outene_ = false;
  // set current step size
  stm_->set_delta_time(stepsize_);
  stm_->set_time_np(time_ + stepsize_);
  // reset the integrator
  stm_->reset_step();
}

/*----------------------------------------------------------------------*/
/* Indicate error and determine new step size */
double Adapter::StructureTimeAda::calculate_dt(const double norm)
{
  // get error order
  if (method_adapt_dis() == ada_upward)
    errorder_ = stm_->method_order_of_accuracy_dis();
  else
    errorder_ = method_order_of_accuracy_dis();

  // optimal size ration with respect to given tolerance
  double sizrat = 1.0;
  if (not(norm == 0.0))  // do not divide by zero
    sizrat = std::pow(errtol_ / norm, 1.0 / (errorder_ + 1.0));
  else  // max increase if error norm == 0
    sizrat = sizeratiomax_ / sizeratioscale_;

  // debug
  int myrank = Core::Communication::my_mpi_rank(stm_->discretization()->get_comm());
  if (myrank == 0)
  {
    printf("sizrat %g, stepsize %g, stepsizepre %g\n", sizrat, stepsize_, stepsizepre_);
  }

  // scaled by safety parameter
  sizrat *= sizeratioscale_;

  // optimal new step size
  double stpsiznew = sizrat * stepsize_;

  // redefine sizrat to be dt*_{n}/dt_{n-1}, ie true optimal ratio
  sizrat = stpsiznew / stepsizepre_;

  // limit #sizrat by maximum and minimum
  if (sizrat > sizeratiomax_)
  {
    stpsiznew = sizeratiomax_ * stepsizepre_;
  }
  else if (sizrat < sizeratiomin_)
  {
    stpsiznew = sizeratiomin_ * stepsizepre_;
  }

  // new step size subject to safety measurements
  if (stpsiznew > stepsizemax_)
  {
    stpsiznew = stepsizemax_;
  }
  else if (stpsiznew < stepsizemin_)
  {
    stpsiznew = stepsizemin_;
  }

  return stpsiznew;
}

/*----------------------------------------------------------------------*/
/* Calculate vector norm */
double Adapter::StructureTimeAda::calculate_vector_norm(
    const Solid::VectorNorm norm, Core::LinAlg::Vector<double>& vect, const int numneglect)
{
  // L1 norm
  if (norm == Solid::norm_l1)
  {
    double vectnorm;
    vect.norm_1(&vectnorm);
    return vectnorm;
  }
  // L2/Euclidian norm
  else if (norm == Solid::norm_l2)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm;
  }
  // RMS norm
  else if (norm == Solid::norm_rms)
  {
    double vectnorm;
    vect.norm_2(&vectnorm);
    return vectnorm / sqrt((double)(vect.global_length() - numneglect));
  }
  // infinity/maximum norm
  else if (norm == Solid::norm_inf)
  {
    double vectnorm;
    vect.norm_inf(&vectnorm);
    return vectnorm;
  }
  else
  {
    FOUR_C_THROW("Cannot handle vector norm");
    return 0;
  }
}

/*----------------------------------------------------------------------*/
/* Update output periods */
void Adapter::StructureTimeAda::update_period()
{
  if (outrest_) outresttime_ += outrestperiod_;
  if (outsys_) outsystime_ += outsysperiod_;
  if (outstr_) outstrtime_ += outstrperiod_;
  if (outene_) outenetime_ += outeneperiod_;

  return;
}

/*----------------------------------------------------------------------*/
Solid::ConvergenceStatus Adapter::StructureTimeAda::perform_error_action(
    const Solid::DivContAct& action, double& stepsizenew)
{
  int myrank = Core::Communication::my_mpi_rank(stm_->discretization()->get_comm());

  // here we handle how we deal with a failed Newton-Raphson, basically:
  // + stop -> error
  // + halve_step -> reduce size
  // + continue -> estimate posteriori error assuming very big error, throw warning for
  // divergence status
  // + adapt_step -> error
  // + rand_adapt_step -> error
  // + rand_adapt_step_ele_err -> error
  // + repeat_simulation -> error
  // + adapt_penaltycontact ??
  // + adapt_3D0Dptc_ele_err ??
  switch (action)
  {
    case Solid::divcont_stop:
      // write output
      output();

      // error and stop the simulation
      FOUR_C_THROW("Nonlinear solver did not converge! ");
      break;

    case Solid::divcont_halve_step:
      if (myrank == 0)
      {
        Core::IO::cout << "Nonlinear solver failed to converge at time t= " << stm_->get_time_np()
                       << ". Divide timestep in half. "
                       << "Old time step: " << stepsize_ << Core::IO::endl
                       << "New time step: " << 0.5 * stepsize_ << Core::IO::endl
                       << Core::IO::endl;
      }

      stepsizenew = 0.5 * stepsize_;
      return Solid::conv_fail_repeat;

    case Solid::divcont_continue:
      if (myrank == 0)
      {
        Core::IO::cout
            << "\n WARNING: We are continuing your simulation although the nonlinear solver\n"
               " did not converge in the current time step. We rely on the error estimator "
               "to \n"
               "give a good step size."
            << Core::IO::endl;
      }

      return Solid::conv_success;  // Do not surprise. We enforce successful
                                   // status flag to force the error estimator
                                   // to compute new step size later on.

    case Solid::divcont_adapt_step:
    case Solid::divcont_rand_adapt_step:
    case Solid::divcont_rand_adapt_step_ele_err:
      FOUR_C_THROW(
          "Adapt the time step is handled by the adaptive time marching integrator. Use\n"
          "DIVERCONT = continue if you want to adapt the step size.");
      break;
    case Solid::divcont_repeat_simulation:
      FOUR_C_THROW("No use to repeat a simulation when it failed. Get a coffee instead.");
      break;
    case Solid::divcont_adapt_penaltycontact:
    default:
      FOUR_C_THROW("I don't know what to do.");
      break;
  }
  return Solid::conv_success;  // make compiler happy
}

FOUR_C_NAMESPACE_CLOSE
