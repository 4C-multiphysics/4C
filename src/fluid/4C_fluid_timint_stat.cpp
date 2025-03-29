// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_stat.hpp"

#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_turbulence_boxfilter.hpp"
#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_fluid_volumetric_surfaceFlow_condition.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 11/13 |
 *----------------------------------------------------------------------*/
FLD::TimIntStationary::TimIntStationary(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntStationary::init()
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
 | use TimeLoop() to start stationary problem                  bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::TimIntStationary::time_loop()
{
  solve_stationary_problem();
  return;
}


/*----------------------------------------------------------------------*
| set old part of right hand side                              bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationary::set_old_part_of_righthandside()
{
  /*
     Stationary:

                   mom: hist_ = 0.0
                   (con: hist_ = 0.0)
  */

  hist_->put_scalar(0.0);

  return;
}

/*----------------------------------------------------------------------*
 | solve stationary fluid problem                              gjb 10/07|
 *----------------------------------------------------------------------*/
void FLD::TimIntStationary::solve_stationary_problem()
{
  // time measurement: time loop (stationary) --- start TimeMonitor tm2
  TEUCHOS_FUNC_TIME_MONITOR(" + time loop");

  // -------------------------------------------------------------------
  // pseudo time loop (continuation loop)
  // -------------------------------------------------------------------
  // slightly increasing b.c. values by given (pseudo-)timecurves to reach
  // convergence also for higher Reynolds number flows
  // as a side effect, you can do parameter studies for different Reynolds
  // numbers within only ONE simulation when you apply a proper
  // (pseudo-)timecurve

  while (not_finished())
  {
    // -------------------------------------------------------------------
    //              set (pseudo-)time-dependent parameters
    // -------------------------------------------------------------------
    increment_time_and_step();

    // -------------------------------------------------------------------
    //                         out to screen
    // -------------------------------------------------------------------
    if (myrank_ == 0)
    {
      printf("Stationary Fluid Solver - STEP = %4d/%4d \n", step_, stepmax_);
    }

    set_element_time_parameter();

    // -------------------------------------------------------------------
    //         evaluate Dirichlet and Neumann boundary conditions
    // -------------------------------------------------------------------
    set_dirichlet_neumann_bc();

    // -------------------------------------------------------------------
    //           preparation of AVM3-based scale separation
    // -------------------------------------------------------------------
    if (step_ == 1 and fssgv_ != Inpar::FLUID::no_fssgv) avm3_preparation();

    // -------------------------------------------------------------------
    //                     solve equation system
    // -------------------------------------------------------------------
    solve();

    // -------------------------------------------------------------------
    //                        compute flow rates
    // -------------------------------------------------------------------
    compute_flow_rates();

    // -------------------------------------------------------------------
    // evaluate divergence u
    // -------------------------------------------------------------------
    evaluate_div_u();


    // Evaluate integral error
    evaluate_error_compared_to_analytical_sol();



    // -------------------------------------------------------------------
    //                         output of solution
    // -------------------------------------------------------------------
    output();

  }  // end of time loop

}  // TimIntStationary::solve_stationary_problem

/*----------------------------------------------------------------------*
| set integration-scheme-specific state                        bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationary::set_state_tim_int() { discret_->set_state("velaf", *velnp_); }

/*----------------------------------------------------------------------*
| calculate acceleration                                       bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationary::calculate_acceleration(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> velnp,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> veln,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> velnm,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> accn,
    const std::shared_ptr<Core::LinAlg::Vector<double>> accnp)
{
  accnp->put_scalar(0.0);

  return;
}

/*----------------------------------------------------------------------*
| set gamma                                                    bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationary::set_gamma(Teuchos::ParameterList& eleparams)
{
  // do nothing
  return;
}

/*----------------------------------------------------------------------*
| scale separation                                             bk 12/13 |
*-----------------------------------------------------------------------*/
void FLD::TimIntStationary::sep_multiply()
{
  Sep_->multiply(false, *velnp_, *fsvelaf_);
  return;
}

/*----------------------------------------------------------------------*
 | paraview output of filtered velocity                  rasthofer 02/11|
 *----------------------------------------------------------------------*/
void FLD::TimIntStationary::outputof_filtered_vel(
    std::shared_ptr<Core::LinAlg::Vector<double>> outvec,
    std::shared_ptr<Core::LinAlg::Vector<double>> fsoutvec)
{
  // no output since subgrid-scale modeling does not make sense for stationary problems!!!
  return;
}

// -------------------------------------------------------------------
// set general time parameter (AE 01/2011)
// -------------------------------------------------------------------
void FLD::TimIntStationary::set_element_time_parameter()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

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
| return time integration factor                               bk 12/13 |
*-----------------------------------------------------------------------*/
double FLD::TimIntStationary::tim_int_param() const
{
  double retval = 0.0;
  // no FSI with stationary time integrator
  FOUR_C_THROW("FSI does not allow a stationary time integrator.");
  return retval;
}

/*----------------------------------------------------------------------*|
 | Set Eleparams for turbulence models                          bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::TimIntStationary::treat_turbulence_models(Teuchos::ParameterList& eleparams)
{
  FLD::FluidImplicitTimeInt::treat_turbulence_models(eleparams);
  if (reconstructder_)
    FLD::Utils::project_gradient_and_set_param(
        *discret_, eleparams, velnp_, "velafgrad", alefluid_);
  return;
}

FOUR_C_NAMESPACE_CLOSE
