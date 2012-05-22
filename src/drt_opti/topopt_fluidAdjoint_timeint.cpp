/*!------------------------------------------------------------------------------------------------*
\file ad_opt_fluid_adjoint_impl.cpp

\brief adapter for element routines of fluid adjoint equations in topology optimization

<pre>
Maintainer: Martin Winklmaier
            winklmaier@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15241
</pre>
 *------------------------------------------------------------------------------------------------*/


#include "topopt_fluidAdjoint_timeint.H"
#include "../drt_lib/drt_discret.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
TOPOPT::ADJOINT::FluidAdjointTimeInt::FluidAdjointTimeInt(
    Teuchos::RCP<DRT::Discretization> discret,
    Teuchos::RCP<LINALG::Solver> solver,
    Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<IO::DiscretizationWriter> output
)
:
  discret_(discret),
  solver_(solver),
  params_(params),
  output_(output),
  myrank_(discret_->Comm().MyPID()),
  uprestart_(params_->get<int>("write restart every", -1)),
  upres_(params_->get<int>("write solution every", -1)),
  numdim_(params_->get<int>("number of velocity degrees of freedom")),
  dt_(params_->get<double>("time step size")),
  step_(0),
  stepmax_(params_->get<int>("max number timesteps")),
  maxtime_(params_->get<double>("total time"))
{
  timealgo_ = DRT::INPUT::get<INPAR::FLUID::TimeIntegrationScheme>(*params_, "time int algo");

  // set initial time = endtime so that it fits to the fluid parameter setting
  // potentially we do one step less here than in fluid
  // fluid criteria are:
  // o endtime <= numstep * dt
  // o endtime < maxtime + dt
  //
  // additionally evaluate the correct number of time steps since it is
  // required for evaluation of the fluid velocity at the correct step
  if (timealgo_==INPAR::FLUID::timeint_stationary)
  {
    time_ = dt_;
    stepmax_ = 1;
  }
  else
  {
    if (fabs(maxtime_-dt_*stepmax_)>1.0e-14)
    {
      dserror("Fix total simulation time sim_time = %f, time step size dt = %f and number of time steps num_steps = %i\n"
          "so that: sim_time = dt * num_steps",maxtime_,dt_,stepmax_);
    }

    time_ = maxtime_;
  }
}




