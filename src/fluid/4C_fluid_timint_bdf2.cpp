// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_bdf2.hpp"

#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_turbulence_boxfilter.hpp"
#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 11/13 |
 *----------------------------------------------------------------------*/
FLD::TimIntBDF2::TimIntBDF2(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid), theta_(1.0)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntBDF2::init()
{
  // call init()-functions of base classes
  // note: this order is important
  FLD::FluidImplicitTimeInt::init();

  // check, if starting algorithm is desired
  if (numstasteps_ > 0)
    FOUR_C_THROW("no starting algorithm supported for schemes other than af-gen-alpha");

  set_element_time_parameter();

  complete_general_init();

  return;
}



/*----------------------------------------------------------------------*
| Print information about current time step to screen          bk 11/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntBDF2::print_time_step_info()
{
  if (myrank_ == 0)
  {
    printf("TIME: %11.4E/%11.4E  DT = %11.4E       BDF2          STEP = %4d/%4d \n", time_,
        maxtime_, dta_, step_, stepmax_);
  }
  return;
}

/*----------------------------------------------------------------------*
| calculate pseudo-theta for startalgo_                        bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntBDF2::set_theta()
{
  // for BDF2, theta is set by the time-step sizes, 2/3 for const. dt

  if (step_ > 1)
    theta_ = (dta_ + dtp_) / (2.0 * dta_ + dtp_);
  else
  {
    // use backward Euler for the first time step
    velnm_->update(1.0, *veln_, 0.0);  // results in hist_ = veln_
    theta_ = 1.0;
  }

  return;
}

/*----------------------------------------------------------------------*
| set old part of right hand side                              bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntBDF2::set_old_part_of_righthandside()
{
  /*
     BDF2: for constant time step:

                   mom: hist_ = 4/3 veln_  - 1/3 velnm_
                  (con: hist_ = 4/3 densn_ - 1/3 densnm_)

  */

  hist_->update(4. / 3., *veln_, -1. / 3., *velnm_, 0.0);

  return;
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state                        bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntBDF2::set_state_tim_int()
{
  discret_->set_state("velaf", *velnp_);

  return;
}

/*----------------------------------------------------------------------*
| calculate acceleration                                       bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntBDF2::calculate_acceleration(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> velnp,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> veln,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> velnm,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> accn,
    const std::shared_ptr<Core::LinAlg::Vector<double>> accnp)
{
  /*

  BDF2:

                 2*dt(n)+dt(n-1)                  dt(n)+dt(n-1)
   acc(n+1) = --------------------- vel(n+1) - --------------- vel(n)
               dt(n)*[dt(n)+dt(n-1)]              dt(n)*dt(n-1)

                       dt(n)
             + ----------------------- vel(n-1)
               dt(n-1)*[dt(n)+dt(n-1)]

  */

  if (dta_ * dtp_ < 1e-15) FOUR_C_THROW("Zero time step size!!!!!");
  const double sum = dta_ + dtp_;

  accnp->update((2.0 * dta_ + dtp_) / (dta_ * sum), *velnp, -sum / (dta_ * dtp_), *veln, 0.0);
  accnp->update(dta_ / (dtp_ * sum), *velnm, 1.0);

  return;
}

/*----------------------------------------------------------------------*
| set gamma                                                    bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntBDF2::set_gamma(Teuchos::ParameterList& eleparams)
{
  eleparams.set("gamma", 1.0);
  return;
}

/*----------------------------------------------------------------------*
| scale separation                                             bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntBDF2::sep_multiply()
{
  Sep_->multiply(false, *velnp_, *fsvelaf_);
  return;
}

/*----------------------------------------------------------------------*
 | paraview output of filtered velocity                  rasthofer 02/11|
 *----------------------------------------------------------------------*/
void FLD::TimIntBDF2::outputof_filtered_vel(std::shared_ptr<Core::LinAlg::Vector<double>> outvec,
    std::shared_ptr<Core::LinAlg::Vector<double>> fsoutvec)
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
  std::shared_ptr<Core::LinAlg::Vector<double>> row_finescaleveltmp;
  row_finescaleveltmp = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // get fine scale velocity
  if (scale_sep_ == Inpar::FLUID::algebraic_multigrid_operator)
    Sep_->multiply(false, *velnp_, *row_finescaleveltmp);
  else
    FOUR_C_THROW("Unknown separation type!");

  // get filtered or coarse scale velocity
  outvec->update(1.0, *velnp_, -1.0, *row_finescaleveltmp, 0.0);

  fsoutvec->update(1.0, *row_finescaleveltmp, 0.0);

  return;
}

// -------------------------------------------------------------------
// set general time parameter (AE 01/2011)
// -------------------------------------------------------------------
void FLD::TimIntBDF2::set_element_time_parameter()
{
  Teuchos::ParameterList eleparams;

  // set time integration scheme
  eleparams.set<Inpar::FLUID::TimeIntegrationScheme>("TimeIntegrationScheme", timealgo_);

  // set general element parameters
  eleparams.set("dt", dta_);
  eleparams.set("theta", theta_);
  eleparams.set("omtheta", 0.0);

  // set scheme-specific element parameters and vector values
  eleparams.set("total time", time_);

  Discret::Elements::FluidEleParameterTimInt::instance()->set_element_time_parameter(eleparams);
}

/*----------------------------------------------------------------------*
| Return linear error coefficient of velocity             mayr.mt 12/13 |
*-----------------------------------------------------------------------*/
double FLD::TimIntBDF2::method_lin_err_coeff_vel() const
{
  double nominator = (dta_ + dtp_) * (dta_ + dtp_);
  double denominator = 6 * dta_ * (2 * dta_ + dtp_);

  return nominator / denominator;
}

FOUR_C_NAMESPACE_CLOSE
