// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_genalpha.hpp"

#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_timint_meshtying_strategy_base.hpp"
#include "4C_scatra_turbulence_hit_scalar_forcing.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                       vg 11/08 |
 *----------------------------------------------------------------------*/
ScaTra::TimIntGenAlpha::TimIntGenAlpha(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, params, extraparams, output),
      phiaf_(nullptr),
      phiam_(nullptr),
      phidtam_(nullptr),
      fsphiaf_(nullptr),
      alphaM_(params_->get<double>("ALPHA_M")),
      alphaF_(params_->get<double>("ALPHA_F")),
      gamma_(params_->get<double>("GAMMA")),
      genalphafac_(0.0)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
}


/*----------------------------------------------------------------------*
 |  initialize time integration                         rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::setup()
{
  // initialize base class
  ScaTraTimIntImpl::setup();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // Vectors passed to the element
  // -----------------------------

  // scalar at times n+alpha_F and n+alpha_M
  phiaf_ = Core::LinAlg::create_vector(*dofrowmap, true);
  phiam_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // temporal derivative of scalar at times n+1, n and n+alpha_M
  phidtam_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // compute specific time factor for generalized-alpha time integration:
  // genalphatimefac = gamma*alpha_F/alpha_M
  if (alphaM_ < 1e-12) FOUR_C_THROW("factor alpha_M lower than or equal zero");
  genalphafac_ = gamma_ / alphaM_;

  // fine-scale vector at time n+alpha_F
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
      turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    fsphiaf_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // -------------------------------------------------------------------
  // set element parameters
  // -------------------------------------------------------------------
  // note: - this has to be done before element routines are called
  //       - order is important here: for safety checks in set_element_general_parameters(),
  //         we have to know the time-integration parameters
  set_element_time_parameter();
  set_element_general_parameters();
  set_element_turbulence_parameters();
  set_element_nodeset_parameters();

  // for initializing phiaf_, phiam_ based on the initial field that was
  // set for phinp_, phin_ in the TimInt base class constructor;
  // otherwise phiaf_ is initialized with zeros instead of the initial field
  compute_intermediate_values();

  // setup krylov
  prepare_krylov_projection();

  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  // note: this constructor has to be called after the forcing_ vector has
  //       been initialized; this is done in ScaTraTimIntImpl::init() called before
  if (special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence")
  {
    if (extraparams_->sublist("TURBULENCE MODEL").get<std::string>("SCALAR_FORCING") == "isotropic")
    {
      homisoturb_forcing_ = std::make_shared<ScaTra::HomoIsoTurbScalarForcing>(this);
      // initialize forcing algorithm
      homisoturb_forcing_->set_initial_spectrum(
          Teuchos::getIntegralValue<Inpar::ScaTra::InitialField>(*params_, "INITIALFIELD"));
    }
  }
}


/*----------------------------------------------------------------------*
 |  set time parameter for element evaluation (usual call)   ehrl 11/13 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::set_element_time_parameter(bool forcedincrementalsolver) const
{
  Teuchos::ParameterList eleparams;

  eleparams.set<bool>("using generalized-alpha time integration", true);
  eleparams.set<bool>("using stationary formulation", false);
  if (!forcedincrementalsolver)
    eleparams.set<bool>("incremental solver", incremental_);
  else
    eleparams.set<bool>("incremental solver", true);

  eleparams.set<double>("time-step length", dta_);
  eleparams.set<double>("total time", time_ - (1 - alphaF_) * dta_);
  eleparams.set<double>("time factor", genalphafac_ * dta_);
  eleparams.set<double>("alpha_F", alphaF_);

  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
      ->set_parameters(eleparams);
}


/*----------------------------------------------------------------------*
 |  set time parameter for element evaluation (usual call)   ehrl 11/13 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::set_element_time_parameter_backward_euler() const
{
  Teuchos::ParameterList eleparams;

  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", false);
  eleparams.set<bool>("incremental solver", true);

  eleparams.set<double>("time-step length", dta_);
  eleparams.set<double>("total time", time_);
  eleparams.set<double>("time factor", 1.0 * dta_);
  eleparams.set<double>("alpha_F", 1.0);

  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
      ->set_parameters(eleparams);
}


/*--------------------------------------------------------------------------*
 | set time for evaluation of POINT -Neumann boundary conditions   vg 12/08 |
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::set_time_for_neumann_evaluation(Teuchos::ParameterList& params)
{
  params.set("total time", time_ - (1 - alphaF_) * dta_);
}


/*----------------------------------------------------------------------*
 | set part of the residual vector belonging to the old timestep        |
 |                                                             vg 11/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::set_old_part_of_righthandside()
{
  // call base class routine
  ScaTraTimIntImpl::set_old_part_of_righthandside();

  // calculation of history vector only for non-incremental formulation:
  // (History vector is used in both cases, but in incremental case, it
  // contains time derivatives of scalar, see below.)
  // hist_ = phin_ + dt*(1-(gamma/alpha_M))*phidtn_
  if (not incremental_) hist_->update(1.0, *phin_, dta_ * (1.0 - genalphafac_), *phidtn_, 0.0);
}


/*----------------------------------------------------------------------*
 | perform an explicit predictor step                          vg 11/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::explicit_predictor() const
{
  // call base class routine
  ScaTraTimIntImpl::explicit_predictor();

  // constant predictor
  phinp_->update(1.0, *phin_, 0.0);
}


/*----------------------------------------------------------------------*
 | compute values at intermediate time steps                   vg 09/09 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::compute_intermediate_values()
{
  // compute phi at n+alpha_F and n+alpha_M
  phiaf_->update(alphaF_, *phinp_, (1.0 - alphaF_), *phin_, 0.0);
  phiam_->update(alphaM_, *phinp_, (1.0 - alphaM_), *phin_, 0.0);

  // accelerations are not independent but rather have to be computed
  // from phinp_, phin_ and phidtn_
  compute_time_derivative();

  // compute time derivative of phi at n+alpha_M
  phidtam_->update(alphaM_, *phidtnp_, (1.0 - alphaM_), *phidtn_, 0.0);
}


/*----------------------------------------------------------------------*
 | add actual Neumann loads                                             |
 | scaled with a factor resulting from time discretization     vg 11/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::add_neumann_to_residual()
{
  residual_->update(genalphafac_ * dta_, *neumann_loads_, 1.0);
}


/*----------------------------------------------------------------------*
 | AVM3-based scale separation                                 vg 03/09 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::avm3_separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // AVM3 separation
  Sep_->multiply(false, *phiaf_, *fsphiaf_);

  // set fine-scale velocity for parallel nightly tests
  // separation matrix depends on the number of proc here
  if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales and
      extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES").get<bool>("SET_FINE_SCALE_VEL"))
    fsphiaf_->put_scalar(0.01);

  // set fine-scale vector
  discret_->set_state("fsphinp", *fsphiaf_);
}


/*----------------------------------------------------------------------*
 | dynamic Smagorinsky model                           rasthofer  08/12 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::dynamic_computation_of_cs()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Prt
    // compute averaged values for LkMk and MkMk
    if (DynSmag_ != nullptr)
    {
      const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet_toggle();
      DynSmag_->apply_filter_for_dynamic_computation_of_prt(
          phiaf_, 0.0, dirichtoggle, *extraparams_, nds_vel());
    }
    else
    {
      FOUR_C_THROW("std::shared_ptr<FLD::DynSmagFilter> DynSmag_ = nullptr");
    }
  }
}

/*----------------------------------------------------------------------*
 | dynamic Vreman model                                krank  09/13     |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::dynamic_computation_of_cv()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet_toggle();
    Vrem_->apply_filter_for_dynamic_computation_of_dt(
        phiaf_, 0.0, dirichtoggle, *extraparams_, nds_vel());
  }
}


/*--------------------------------------------------------------------------*
 | add global state vectors specific for time-integration scheme   vg 11/08 |
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::add_time_integration_specific_vectors(bool forcedincrementalsolver)
{
  // call base class routine
  ScaTraTimIntImpl::add_time_integration_specific_vectors(forcedincrementalsolver);

  discret_->set_state("phinp", *phiaf_);

  if (incremental_ or forcedincrementalsolver)
    discret_->set_state("hist", *phidtam_);
  else
  {
    discret_->set_state("hist", *hist_);
    discret_->set_state("phin", *phin_);
  }
}


/*----------------------------------------------------------------------*
 | compute time derivative                                     vg 09/09 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::compute_time_derivative()
{
  // call base class routine
  ScaTraTimIntImpl::compute_time_derivative();

  // time derivative of phi:
  // phidt(n+1) = (phi(n+1)-phi(n)) / (gamma*dt) + (1-(1/gamma))*phidt(n)
  const double fact1 = 1.0 / (gamma_ * dta_);
  const double fact2 = 1.0 - (1.0 / gamma_);
  phidtnp_->update(fact2, *phidtn_, 0.0);
  phidtnp_->update(fact1, *phinp_, -fact1, *phin_, 1.0);

  // We know the first time derivative on Dirichlet boundaries
  // so we do not need an approximation of these values!
  // However, we do not want to break the linear relationship
  // as stated above. We do not want to set Dirichlet values for
  // dependent values like phidtnp_. This turned out to be inconsistent.
  // Such an inconsistency can cause different results for
  // our different Gen. Alpha formulations (linear_full <-> linear_incremental).
  // We don't want this to happen.
  // apply_dirichlet_bc(time_,nullptr,phidtnp_);
}


/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                             vg 11/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::update()
{
  // set history variable to zero for not spoiling flux calculation
  // if (not incremental_) hist_->PutScalar(0.0);

  // compute flux vector field for later output BEFORE time shift of results
  // is performed below !!
  if (calcflux_domain_ != Inpar::ScaTra::flux_none or
      calcflux_boundary_ != Inpar::ScaTra::flux_none)
  {
    if (is_result_step() or do_boundary_flux_statistics()) calc_flux(true);
  }

  // compute time derivative at time n+1
  compute_time_derivative();

  // call base class routine
  ScaTraTimIntImpl::update();

  // solution of this step becomes most recent solution of last step
  phin_->update(1.0, *phinp_, 0.0);

  // time deriv. of this step becomes most recent time derivative of
  // last step
  phidtn_->update(1.0, *phidtnp_, 0.0);

  // call time update of forcing routine
  if (homisoturb_forcing_ != nullptr) homisoturb_forcing_->time_update_forcing();
}


/*----------------------------------------------------------------------*
 | write additional data required for restart                  vg 11/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::write_restart() const
{
  // call base class routine
  ScaTraTimIntImpl::write_restart();

  // additional state vectors that are needed for generalized-alpha restart
  output_->write_vector("phidtnp", phidtnp_);
  output_->write_vector("phidtn", phidtn_);
  output_->write_vector("phin", phin_);
}


/*----------------------------------------------------------------------*
 |                                                             vg 11/08 |
 -----------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::read_restart(
    const int step, std::shared_ptr<Core::IO::InputControl> input)
{
  // call base class routine
  ScaTraTimIntImpl::read_restart(step, input);

  std::shared_ptr<Core::IO::DiscretizationReader> reader(nullptr);
  if (input == nullptr)
  {
    reader = std::make_shared<Core::IO::DiscretizationReader>(
        discret_, Global::Problem::instance()->input_control_file(), step);
  }
  else
    reader = std::make_shared<Core::IO::DiscretizationReader>(discret_, input, step);

  time_ = reader->read_double("time");
  step_ = reader->read_int("step");

  if (myrank_ == 0)
    std::cout << "Reading ScaTra restart data (time=" << time_ << " ; step=" << step_ << ")"
              << '\n';

  // read state vectors that are needed for generalized-alpha restart
  reader->read_vector(phinp_, "phinp");
  reader->read_vector(phin_, "phin");
  reader->read_vector(phidtnp_, "phidtnp");
  reader->read_vector(phidtn_, "phidtn");

  read_restart_problem_specific(step, *reader);

  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
      turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    avm3_preparation();
}


/*-----------------------------------------------------------------------------------------------------------*
 | calculate consistent initial scalar time derivatives in compliance with initial scalar field fang
 09/15 |
 ------------------------------------------------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::calc_initial_time_derivative()
{
  pre_calc_initial_time_derivative();

  // call core algorithm
  ScaTraTimIntImpl::calc_initial_time_derivative();

  post_calc_initial_time_derivative();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::pre_calc_initial_time_derivative()
{
  // for calculation of initial time derivative, we have to switch off all stabilization and
  // turbulence modeling terms
  // standard general element parameter without stabilization
  set_element_general_parameters(true);

  // we also have to modify the time-parameter list (incremental solve)
  // actually we do not need a time integration scheme for calculating the initial time derivatives,
  // but the rhs of the standard element routine is used as starting point for this special system
  // of equations. Therefore, the rhs vector has to be scaled correctly. Since the genalpha scheme
  // cannot be adapted easily, the backward Euler scheme is used instead.
  set_element_time_parameter_backward_euler();

  // deactivate turbulence settings
  set_element_turbulence_parameters(true);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntGenAlpha::post_calc_initial_time_derivative()
{
  // and finally undo our temporary settings
  set_element_general_parameters();
  set_element_time_parameter();
  set_element_turbulence_parameters();
}

FOUR_C_NAMESPACE_CLOSE
