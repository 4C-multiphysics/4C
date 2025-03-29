// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_ost.hpp"

#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_timint_meshtying_strategy_base.hpp"
#include "4C_scatra_turbulence_hit_scalar_forcing.hpp"
#include "4C_utils_parameter_list.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::TimIntOneStepTheta::TimIntOneStepTheta(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output, const int probnum)
    : ScaTraTimIntImpl(actdis, solver, params, extraparams, output, probnum),
      theta_(params_->get<double>("THETA")),
      fsphinp_(nullptr)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting and
  // other kinds of parallel redistribution.
  // This has to be done before all state vectors are initialized.
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::init()
{
  // initialize base class
  ScaTraTimIntImpl::init();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::setup()
{
  // setup base class
  ScaTraTimIntImpl::setup();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // fine-scale vector at time n+1
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
      turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    fsphinp_ = Core::LinAlg::create_vector(*dofrowmap, true);

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

  // -------------------------------------------------------------------
  // initialize multi-scale material if necessary
  // -------------------------------------------------------------------
  if (macro_scale_)
  {
    // create parameter list for macro-scale elements
    Teuchos::ParameterList eleparams;

    // set action
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::micro_scale_initialize, eleparams);

    // loop over macro-scale elements
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::set_element_time_parameter(bool forcedincrementalsolver) const
{
  Teuchos::ParameterList eleparams;

  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", false);
  if (!forcedincrementalsolver)
    eleparams.set<bool>("incremental solver", incremental_);
  else
    eleparams.set<bool>("incremental solver", true);

  eleparams.set<double>("time-step length", dta_);
  eleparams.set<double>("total time", time_);
  eleparams.set<double>("time factor", theta_ * dta_);
  eleparams.set<double>("time derivative factor", 1.0 / (theta_ * dta_));
  eleparams.set<double>("alpha_F", 1.0);

  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
      ->set_parameters(eleparams);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::set_time_for_neumann_evaluation(Teuchos::ParameterList& params)
{
  params.set("total time", time_);
}

/*----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::print_time_step_info()
{
  if (myrank_ == 0 and not micro_scale_)
  {
    std::cout << '\n'
              << "TIME: " << std::setw(11) << std::setprecision(4) << std::scientific << time_
              << "/" << std::setw(11) << std::setprecision(4) << std::scientific << maxtime_
              << "  DT = " << std::setw(11) << std::setprecision(4) << std::scientific << dta_
              << "  " << method_title() << " (theta = " << std::setw(3) << std::setprecision(2)
              << theta_ << ") STEP = " << std::setw(4) << step_ << "/" << std::setw(4) << stepmax_
              << '\n';
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::set_old_part_of_righthandside()
{
  // call base class routine
  ScaTraTimIntImpl::set_old_part_of_righthandside();

  // hist_ = phin_ + dt*(1-Theta)*phidtn_
  hist_->update(1.0, *phin_, dta_ * (1.0 - theta_), *phidtn_, 0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::explicit_predictor() const
{
  // call base class routine
  ScaTraTimIntImpl::explicit_predictor();

  // predict discrete solution variables
  phinp_->update(dta_, *phidtn_, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::add_neumann_to_residual()
{
  residual_->update(theta_ * dta_, *neumann_loads_, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::avm3_separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // AVM3 separation
  Sep_->multiply(false, *phinp_, *fsphinp_);

  // set fine-scale vector
  discret_->set_state("fsphinp", *fsphinp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::dynamic_computation_of_cs()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Prt
    // compute averaged values for LkMk and MkMk
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet_toggle();
    DynSmag_->apply_filter_for_dynamic_computation_of_prt(
        phinp_, 0.0, dirichtoggle, *extraparams_, nds_vel());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::dynamic_computation_of_cv()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet_toggle();
    Vrem_->apply_filter_for_dynamic_computation_of_dt(
        phinp_, 0.0, dirichtoggle, *extraparams_, nds_vel());
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::add_time_integration_specific_vectors(bool forcedincrementalsolver)
{
  // call base class routine
  ScaTraTimIntImpl::add_time_integration_specific_vectors(forcedincrementalsolver);

  discret_->set_state("hist", *hist_);
  discret_->set_state("phinp", *phinp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::compute_time_derivative()
{
  // call base class routine
  ScaTraTimIntImpl::compute_time_derivative();

  // time derivative of phi:
  // phidt(n+1) = (phi(n+1)-phi(n)) / (theta*dt) + (1-(1/theta))*phidt(n)
  const double fact = 1.0 / (theta_ * dta_);
  phidtnp_->update(fact, *phinp_, -fact, *hist_, 0.0);

  // We know the first time derivative on Dirichlet boundaries
  // so we do not need an approximation of these values!
  // However, we do not want to break the linear relationship
  // as stated above. We do not want to set Dirichlet values for
  // dependent values like phidtnp_. This turned out to be inconsistent.
  // apply_dirichlet_bc(time_,nullptr,phidtnp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::update()
{
  // compute time derivative at time n+1
  compute_time_derivative();

  // call base class routine
  ScaTraTimIntImpl::update();

  // compute flux vector field for later output BEFORE time shift of results
  // is performed below !!
  if (calcflux_domain_ != Inpar::ScaTra::flux_none or
      calcflux_boundary_ != Inpar::ScaTra::flux_none)
  {
    if (is_result_step() or do_boundary_flux_statistics()) calc_flux(true);
  }

  // after the next command (time shift of solutions) do NOT call
  // compute_time_derivative() anymore within the current time step!!!

  // solution of this step becomes most recent solution of the last step
  phin_->update(1.0, *phinp_, 0.0);

  // time deriv. of this step becomes most recent time derivative of
  // last step
  phidtn_->update(1.0, *phidtnp_, 0.0);

  // call time update of forcing routine
  if (homisoturb_forcing_ != nullptr) homisoturb_forcing_->time_update_forcing();

  // update micro scale in multi-scale simulations if necessary
  if (macro_scale_)
  {
    // create parameter list for macro-scale elements
    Teuchos::ParameterList eleparams;

    // set action
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::micro_scale_update, eleparams);

    // loop over macro-scale elements
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::write_restart() const
{
  // call base class routine
  ScaTraTimIntImpl::write_restart();

  // additional state vectors that are needed for One-Step-Theta restart
  output_->write_vector("phidtn", phidtn_);
  output_->write_vector("phin", phin_);

  // write nodal micro concentration
  if (macro_scale_ and nds_micro() != -1) output_->write_vector("phinp_micro", phinp_micro_);
}

/*----------------------------------------------------------------------*
 -----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::read_restart(
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

  // read state vectors that are needed for One-Step-Theta restart
  reader->read_vector(phinp_, "phinp");
  reader->read_vector(phin_, "phin");
  reader->read_vector(phidtn_, "phidtn");

  read_restart_problem_specific(step, *reader);

  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
      turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    avm3_preparation();

  // read restart on micro scale in multi-scale simulations if necessary
  if (macro_scale_)
  {
    if (nds_micro() != -1) reader->read_vector(phinp_micro_, "phinp_micro");

    // create parameter list for macro-scale elements
    Teuchos::ParameterList eleparams;

    // set action
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::micro_scale_read_restart, eleparams);

    // loop over macro-scale elements
    discret_->evaluate(eleparams);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::calc_initial_time_derivative()
{
  pre_calc_initial_time_derivative();

  // call core algorithm
  ScaTraTimIntImpl::calc_initial_time_derivative();

  post_calc_initial_time_derivative();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::pre_calc_initial_time_derivative()
{
  // standard general element parameter without stabilization
  set_element_general_parameters(true);

  // we also have to modify the time-parameter list (incremental solve)
  // actually we do not need a time integration scheme for calculating the initial time derivatives,
  // but the rhs of the standard element routine is used as starting point for this special system
  // of equations. Therefore, the rhs vector has to be scaled correctly.
  set_element_time_parameter(true);

  // deactivate turbulence settings
  set_element_turbulence_parameters(true);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::post_calc_initial_time_derivative()
{  // and finally undo our temporary settings
  set_element_general_parameters(false);
  set_element_time_parameter(false);
  set_element_turbulence_parameters(false);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::set_state(std::shared_ptr<Core::LinAlg::Vector<double>> phin,
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp,
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtn,
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtnp,
    std::shared_ptr<Core::LinAlg::Vector<double>> hist,
    std::shared_ptr<Core::IO::DiscretizationWriter> output,
    std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> visualization_writer,
    const std::vector<double>& phinp_macro, const int step, const double time)
{
  phin_ = phin;
  phinp_ = phinp;
  phidtn_ = phidtn;
  phidtnp_ = phidtnp;
  hist_ = hist;
  output_ = output;
  set_visualization_writer(std::move(visualization_writer));
  phinp_macro_ = phinp_macro;
  dq_dphi_.resize(phinp_macro_.size(), 0.);
  step_ = step;
  time_ = time;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void ScaTra::TimIntOneStepTheta::clear_state()
{
  phin_ = nullptr;
  phinp_ = nullptr;
  phidtn_ = nullptr;
  phidtnp_ = nullptr;
  hist_ = nullptr;
  output_ = nullptr;
  phinp_macro_.clear();
  dq_dphi_.clear();
  step_ = -1;
  time_ = 0.0;
}
FOUR_C_NAMESPACE_CLOSE
