// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timada.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"
#include "4C_structure_timint.hpp"
#include "4C_utils_enum.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN


namespace
{
  std::string map_kind_enum_to_string(Inpar::Solid::TimAdaKind term)
  {
    switch (term)
    {
      case Inpar::Solid::timada_kind_zienxie:
        return "ZienkiewiczXie";
        break;
      case Inpar::Solid::timada_kind_ab2:
        return "AdamsBashforth2";
        break;
      case Inpar::Solid::timada_kind_expleuler:
        return "ExplicitEuler";
        break;
      case Inpar::Solid::timada_kind_centraldiff:
        return "CentralDifference";
        break;
      default:
        FOUR_C_THROW("Cannot cope with name enum {}", term);
        return "";
        break;
    }
  }
}  // namespace

/*----------------------------------------------------------------------*/
/* Constructor */
Solid::TimAda::TimAda(const Teuchos::ParameterList& timeparams,  //!< TIS input parameters
    const Teuchos::ParameterList& tap,                           //!< adaptive input flags
    std::shared_ptr<TimInt> tis                                  //!< marching time integrator
    )
    : sti_(tis),
      discret_(tis->discretization()),
      myrank_(Core::Communication::my_mpi_rank(discret_->get_comm())),
      solver_(tis->solver()),
      output_(tis->disc_writer()),
      //
      timeinitial_(0.0),
      timefinal_(timeparams.get<double>("MAXTIME")),
      timedirect_(sign(timefinal_ - timeinitial_)),
      timestepinitial_(0),
      timestepfinal_(timeparams.get<int>("NUMSTEP")),
      stepsizeinitial_(timeparams.get<double>("TIMESTEP")),
      //
      stepsizemax_(tap.get<double>("STEPSIZEMAX")),
      stepsizemin_(tap.get<double>("STEPSIZEMIN")),
      sizeratiomax_(tap.get<double>("SIZERATIOMAX")),
      sizeratiomin_(tap.get<double>("SIZERATIOMIN")),
      sizeratioscale_(tap.get<double>("SIZERATIOSCALE")),
      errctrl_(ctrl_dis),  // PROVIDE INPUT PARAMETER
      errnorm_(Teuchos::getIntegralValue<Inpar::Solid::VectorNorm>(tap, "LOCERRNORM")),
      errtol_(tap.get<double>("LOCERRTOL")),
      errorder_(1),  // CHANGE THIS CONSTANT
      adaptstepmax_(tap.get<int>("ADAPTSTEPMAX")),
      //
      time_(timeinitial_),
      timestep_(0),
      stepsizepre_(stepsizeinitial_),
      stepsize_(timeparams.get<double>("TIMESTEP")),
      locerrdisn_(nullptr),
      adaptstep_(0),
      //
      outsys_(false),
      outstr_(false),
      outene_(false),
      outrest_(false),
      outsysperiod_(tap.get<double>("OUTSYSPERIOD")),
      outstrperiod_(tap.get<double>("OUTSTRPERIOD")),
      outeneperiod_(tap.get<double>("OUTENEPERIOD")),
      outrestperiod_(tap.get<double>("OUTRESTPERIOD")),
      outsizeevery_(tap.get<int>("OUTSIZEEVERY")),
      outsystime_(timeinitial_ + outsysperiod_),
      outstrtime_(timeinitial_ + outstrperiod_),
      outenetime_(timeinitial_ + outeneperiod_),
      outresttime_(timeinitial_ + outrestperiod_),
      outsizefile_(nullptr)
{
  // allocate displacement local error vector
  locerrdisn_ = Core::LinAlg::create_vector(*(discret_->dof_row_map()), true);

  // check whether energyout_ file handle was attached
  if ((not sti_->attached_energy_file()) and (outeneperiod_ != 0.0) and (myrank_ == 0))
  {
    sti_->attach_energy_file();
  }

  // check if step size file is wanted and attach
  if ((outsizeevery_ != 0) and (myrank_ == 0))
  {
    attach_file_step_size();
  }

  // enable restart for adaptive timestepping - however initial timestep size is still read from
  // datfile! (mhv 01/2015)
  const int restart = Global::Problem::instance()->restart();
  if (restart)
  {
    // read restart of marching time-integrator and reset initial time and step for adaptive loop
    tis->read_restart(restart);
    timeinitial_ = tis->time_old();
    timestepinitial_ = tis->step_old();

    // update variables which depend on initial time and step
    timedirect_ = sign(timefinal_ - timeinitial_);
    outsystime_ = timeinitial_ + outsysperiod_;
    outstrtime_ = timeinitial_ + outstrperiod_;
    outenetime_ = timeinitial_ + outeneperiod_;
    outresttime_ = timeinitial_ + outrestperiod_;
  }

  return;
}

/*----------------------------------------------------------------------*/
/* Integrate adaptively in time */
int Solid::TimAda::integrate()
{
  // finalize initialization
  // (only relevant if an auxiliary time integrator is used)
  init(sti_);

  // Richardson extrapolation to no avail
  if (method_adapt_dis() == ada_ident)
    FOUR_C_THROW(
        "This combination is not implemented ... Richardson's extrapolation ... Yoshida technique "
        "...");

  // initialise time loop
  time_ = timeinitial_;
  timestep_ = timestepinitial_;
  stepsize_ = stepsizeinitial_;
  update_step_size();

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
      sti_->dt_->set_step(0, stepsize_);
      sti_->timen_ = time_ + stepsize_;

      // integrate system with auxiliary TIS
      // we hold \f$D_{n+1}^{AUX}\f$ on #locdiserrn_
      // and \f$V_{n+1}^{AUX}\f$ on #locvelerrn_
      integrate_step_auxiliary();

      // integrate system with marching TIS and
      sti_->integrate_step();

      // get local error vector on #locerrdisn_
      evaluate_local_error_dis();

      // check whether step passes
      indicate(accepted, stpsiznew);

      // adjust step-size and prepare repetition of current step
      if (not accepted)
      {
        Core::IO::cout << "Repeating step with stepsize = " << stpsiznew << Core::IO::endl;
        Core::IO::cout << "- - - - - - - - - - - - - - - - - - - - - - - - -"
                       << " - - - - - - - - - - - - - - -" << Core::IO::endl;

        stepsize_ = stpsiznew;

        reset_step();
      }

      // increment number of adapted step sizes in a row
      adaptstep_ += 1;
    }

    // update or break
    if (accepted)
    {
      if (myrank_ == 0) std::cout << "Step size accepted" << std::endl;
    }
    else if (adaptstep_ >= adaptstepmax_)
    {
      if (myrank_ == 0)
        std::cout << "Could not find acceptable time step size"
                  << " ... continuing" << std::endl;
    }
    else
    {
      FOUR_C_THROW("Do not know what to do");
    }

    // increment time and step in the marching time integrator
    sti_->time_->update_steps(time_ + stepsize_);
    sti_->step_ = timestep_ + 1;
    sti_->dt_->update_steps(stepsize_);

    // printing and output
    constexpr bool force_prepare = false;
    prepare_output_period(force_prepare);
    sti_->pre_update();
    sti_->update_step_state();
    sti_->update_step_element();
    sti_->post_update();
    output_period();
    sti_->post_output();
    output_step_size();
    sti_->print_step();

    // update
    //    update();
    timestep_ += 1;
    time_ += stepsize_;
    update_step_size(stpsiznew);

    update_period();
    outrest_ = outsys_ = outstr_ = outene_ = false;

    // the user reads but rarely listens
    if (myrank_ == 0)
    {
      std::cout << "Step " << timestep_ << ", Time " << time_ << ", StepSize " << stepsize_
                << std::endl;
    }
  }

  return 0;  // ToDo Provide meaningful error code here
}

/*----------------------------------------------------------------------*/
/* Evaluate local error vector */
void Solid::TimAda::evaluate_local_error_dis()
{
  if (method_adapt_dis() == ada_orderequal)
  {
    const double coeffmarch = sti_->method_lin_err_coeff_dis();
    const double coeffaux = method_lin_err_coeff_dis();
    locerrdisn_->update(-1.0, *(sti_->disn_), 1.0);
    locerrdisn_->scale(coeffmarch / (coeffaux - coeffmarch));
  }
  else
  {
    // schemes do not have the same order of accuracy
    locerrdisn_->update(-1.0, *(sti_->disn_), 1.0);
  }

  // blank Dirichlet DOFs since they always carry the exact solution
  std::shared_ptr<Core::LinAlg::Vector<double>> zeros =
      std::make_shared<Core::LinAlg::Vector<double>>(locerrdisn_->get_map(), true);
  Core::LinAlg::apply_dirichlet_to_system(
      *locerrdisn_, *zeros, *(sti_->get_dbc_map_extractor()->cond_map()));
}

/*----------------------------------------------------------------------*/
/* Indicate error and determine new step size */
void Solid::TimAda::indicate(bool& accepted, double& stpsiznew)
{
  // norm of local discretisation error vector
  const int numneglect = sti_->get_dbc_map_extractor()->cond_map()->NumGlobalElements();
  const double norm = Solid::calculate_vector_norm(errnorm_, *locerrdisn_, numneglect);

  // check if acceptable
  accepted = (norm < errtol_);

  // debug
  if (myrank_ == 0)
  {
    std::cout << "LocErrNorm " << std::scientific << norm << ", LocErrTol " << errtol_
              << ", Accept " << std::boolalpha << accepted << std::endl;
  }

  stpsiznew = calculate_dt(norm);

  return;
}

/*----------------------------------------------------------------------*/
/* Indicate error and determine new step size */
double Solid::TimAda::calculate_dt(const double norm)
{
  // get error order
  if (method_adapt_dis() == ada_upward)
    errorder_ = sti_->method_order_of_accuracy_dis();
  else
    errorder_ = method_order_of_accuracy_dis();

  // optimal size ration with respect to given tolerance
  double sizrat = 1.0;
  if (not(norm == 0.0))  // do not divide by zero
    sizrat = std::pow(errtol_ / norm, 1.0 / (errorder_ + 1.0));
  else  // max increase if error norm == 0
    sizrat = sizeratiomax_ / sizeratioscale_;

  // debug
  if (myrank_ == 0)
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
/* Prepare repetition of current time step */
void Solid::TimAda::reset_step()
{
  outrest_ = outsys_ = outstr_ = outene_ = false;
  sti_->reset_step();

  return;
}

/*----------------------------------------------------------------------*/
/*  Modify step size to hit precisely output period */
void Solid::TimAda::size_for_output()
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
/* Prepare output to file(s)                                            */
void Solid::TimAda::prepare_output_period(bool force_prepare)
{
  sti_->prepare_output(force_prepare);
}

/*----------------------------------------------------------------------*/
/* Output to file(s) */
void Solid::TimAda::output_period()
{
  // this flag is passed along subroutines and prevents
  // repeated initialising of output writer, printing of
  // state vectors, or similar
  bool datawritten = false;

  // output restart (try this first)
  // write restart step
  if (outrest_)
  {
    sti_->output_restart(datawritten);
  }

  // output results (not necessary if restart in same step)
  if (outsys_ and (not datawritten))
  {
    sti_->output_state(datawritten);
  }

  // output stress & strain
  if (outstr_)
  {
    sti_->output_stress_strain(datawritten);
  }

  // output energy
  if (outene_)
  {
    sti_->output_energy();
  }

  return;
}

/*----------------------------------------------------------------------*/
/* Update output periods */
void Solid::TimAda::update_period()
{
  if (outrest_) outresttime_ += outrestperiod_;
  if (outsys_) outsystime_ += outsysperiod_;
  if (outstr_) outstrtime_ += outstrperiod_;
  if (outene_) outenetime_ += outeneperiod_;

  return;
}

/*----------------------------------------------------------------------*/
/* Write step size */
void Solid::TimAda::output_step_size()
{
  if ((outsizeevery_ != 0) and (timestep_ % outsizeevery_ == 0) and (myrank_ == 0))
  {
    (*outsizefile_) << " " << std::setw(12) << timestep_ << std::scientific << std::setprecision(8)
                    << " " << time_ << " " << stepsize_ << " " << std::setw(2) << adaptstep_
                    << std::endl;
  }
}

std::string Solid::TimAda::method_title() const { return map_kind_enum_to_string(method_name()); }

/*----------------------------------------------------------------------*/
/* Print constants */
void Solid::TimAda::print_constants(std::ostream& str) const
{
  str << "TimAda:  Constants" << std::endl
      << "   Initial time = " << timeinitial_ << std::endl
      << "   Final time = " << timefinal_ << std::endl
      << "   Initial Step = " << timestepinitial_ << std::endl
      << "   Final Step = " << timestepfinal_ << std::endl
      << "   Initial step size = " << stepsizeinitial_ << std::endl
      << "   Max step size = " << stepsizemax_ << std::endl
      << "   Min step size = " << stepsizemin_ << std::endl
      << "   Max size ratio = " << sizeratiomax_ << std::endl
      << "   Min size ratio = " << sizeratiomin_ << std::endl
      << "   Size ratio scale = " << sizeratioscale_ << std::endl
      << "   Error norm = " << errnorm_ << std::endl
      << "   Error order = " << errorder_ << std::endl
      << "   Error tolerance = " << errtol_ << std::endl
      << "   Max adaptations = " << adaptstepmax_ << std::endl;
  return;
}

/*----------------------------------------------------------------------*/
/* Print variables */
void Solid::TimAda::print_variables(std::ostream& str) const
{
  str << "TimAda:  Variables" << std::endl
      << "   Current time = " << time_ << std::endl
      << "   Previous step size = " << stepsizepre_ << std::endl
      << "   Current step size = " << stepsize_ << std::endl
      << "   Current adaptive step = " << adaptstep_ << std::endl;
  return;
}


/*----------------------------------------------------------------------*/
/* Print */
void Solid::TimAda::print(std::ostream& str) const
{
  str << "TimAda" << std::endl;
  print_constants(str);
  print_variables(str);

  return;
}

/*----------------------------------------------------------------------*/
/* Attach file handle for step size file #outsizefile_                  */
void Solid::TimAda::attach_file_step_size()
{
  if (!outsizefile_)
  {
    std::string filename =
        Global::Problem::instance()->output_control_file()->file_name() + ".stepsize";
    outsizefile_ = std::make_shared<std::ofstream>(filename.c_str());
    (*outsizefile_) << "# timestep time step-size adaptations" << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*/
/* Update step size and set new time step size                          */
void Solid::TimAda::update_step_size(const double dtnew)
{
  update_step_size();
  set_dt(dtnew);
  sti_->update_step_time();

  return;
}

/*----------------------------------------------------------------------*/
/* Update step size                                                     */
void Solid::TimAda::update_step_size()
{
  stepsizepre_ = stepsize_;

  return;
}

/*----------------------------------------------------------------------*/
/* Set new time step size                                               */
void Solid::TimAda::set_dt(const double dtnew)
{
  stepsize_ = dtnew;    // in the adaptive time integrator
  sti_->set_dt(dtnew);  // in the marching time integrator
}

/*======================================================================*/
/* Out stream */
std::ostream& operator<<(std::ostream& str, const Solid::TimAda& ta)
{
  ta.print(str);

  return str;
}

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
