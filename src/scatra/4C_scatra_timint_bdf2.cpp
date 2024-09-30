/*----------------------------------------------------------------------*/
/*! \file
\brief BDF2 time-integration scheme

\level 1


*/
/*----------------------------------------------------------------------*/

#include "4C_scatra_timint_bdf2.hpp"

#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_timint_meshtying_strategy_base.hpp"
#include "4C_scatra_turbulence_hit_scalar_forcing.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::TimIntBDF2::TimIntBDF2(Teuchos::RCP<Core::FE::Discretization> actdis,
    Teuchos::RCP<Core::LinAlg::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<Teuchos::ParameterList> extraparams,
    Teuchos::RCP<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, params, extraparams, output),
      theta_(1.0),
      phinm_(Teuchos::null),
      fsphinp_(Teuchos::null)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::setup()
{
  // initialize base class
  ScaTraTimIntImpl::setup();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Epetra_Map* dofrowmap = discret_->dof_row_map();

  // state vector for solution at time t_{n-1}
  phinm_ = Core::LinAlg::create_vector(*dofrowmap, true);

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
      homisoturb_forcing_ = Teuchos::rcp(new ScaTra::HomIsoTurbScalarForcing(this));
      // initialize forcing algorithm
      homisoturb_forcing_->set_initial_spectrum(
          Teuchos::getIntegralValue<Inpar::ScaTra::InitialField>(*params_, "INITIALFIELD"));
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::set_element_time_parameter(bool forcedincrementalsolver) const
{
  Teuchos::ParameterList eleparams;

  Core::UTILS::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::set_time_parameter, eleparams);
  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", false);
  if (!forcedincrementalsolver)
    eleparams.set<bool>("incremental solver", incremental_);
  else
    eleparams.set<bool>("incremental solver", true);

  eleparams.set<double>("time-step length", dta_);
  eleparams.set<double>("total time", time_);
  eleparams.set<double>("time factor", theta_ * dta_);
  eleparams.set<double>("alpha_F", 1.0);
  if (step() == 1)
    eleparams.set<double>("time derivative factor", 1.0 / dta_);
  else
    eleparams.set<double>("time derivative factor", 3.0 / (2.0 * dta_));

  // call standard loop over elements
  discret_->evaluate(
      eleparams, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::set_time_for_neumann_evaluation(Teuchos::ParameterList& params)
{
  params.set("total time", time_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::set_old_part_of_righthandside()
{
  // call base class routine
  ScaTraTimIntImpl::set_old_part_of_righthandside();

  /*
  BDF2: for variable time step:

                 hist_ = (1+omega)^2/(1+ 2*omega) * phin_
                           - omega^2/(1+ 2*omega) * phinm_

  BDF2: for constant time step:

                 hist_ = 4/3 phin_ - 1/3 phinm_
  */
  if (step_ > 1)
  {
    double fact1 = 4.0 / 3.0;
    double fact2 = -1.0 / 3.0;
    hist_->Update(fact1, *phin_, fact2, *phinm_, 0.0);

    // for BDF2 theta is set to 2/3 for constant time-step length dt
    theta_ = 2.0 / 3.0;
  }
  else
  {
    // for start-up of BDF2 we do one step with backward Euler
    hist_->Update(1.0, *phin_, 0.0);

    // backward Euler => use theta=1.0
    theta_ = 1.0;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::explicit_predictor() const
{
  // call base class routine
  ScaTraTimIntImpl::explicit_predictor();

  if (step_ > 1) phinp_->Update(-1.0, *phinm_, 2.0);
  // for step == 1 phinp_ is already correctly initialized with the
  // initial field phin_
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::add_neumann_to_residual()
{
  residual_->Update(theta_ * dta_, *neumann_loads_, 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::avm3_separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // AVM3 separation
  Sep_->multiply(false, *phinp_, *fsphinp_);

  // set fine-scale vector
  discret_->set_state("fsphinp", fsphinp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::dynamic_computation_of_cs()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Prt
    // compute averaged values for LkMk and MkMk
    const Teuchos::RCP<const Core::LinAlg::Vector> dirichtoggle = dirichlet_toggle();
    DynSmag_->apply_filter_for_dynamic_computation_of_prt(
        phinp_, 0.0, dirichtoggle, *extraparams_, nds_vel());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::dynamic_computation_of_cv()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    const Teuchos::RCP<const Core::LinAlg::Vector> dirichtoggle = dirichlet_toggle();
    Vrem_->apply_filter_for_dynamic_computation_of_dt(
        phinp_, 0.0, dirichtoggle, *extraparams_, nds_vel());
  }
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::add_time_integration_specific_vectors(bool forcedincrementalsolver)
{
  // call base class routine
  ScaTraTimIntImpl::add_time_integration_specific_vectors(forcedincrementalsolver);

  discret_->set_state("hist", hist_);
  discret_->set_state("phinp", phinp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::compute_time_derivative()
{
  // call base class routine
  ScaTraTimIntImpl::compute_time_derivative();

  if (step_ == 1)
  {
    // time derivative of phi for first time step:
    // phidt(n+1) = (phi(n+1)-phi(n))/dt
    const double fact = 1.0 / dta_;
    phidtnp_->Update(fact, *phinp_, -fact, *hist_, 0.0);
  }
  else
  {
    // time derivative of phi:
    // phidt(n+1) = ((3/2)*phi(n+1)-2*phi(n)+(1/2)*phi(n-1))/dt
    const double fact = 3.0 / (2.0 * dta_);
    phidtnp_->Update(fact, *phinp_, -fact, *hist_, 0.0);
  }

  // We know the first time derivative on Dirichlet boundaries
  // so we do not need an approximation of these values!
  // However, we do not want to break the linear relationship
  // as stated above. We do not want to set Dirichlet values for
  // dependent values like phidtnp_. This turned out to be inconsistent.
  // apply_dirichlet_bc(time_,Teuchos::null,phidtnp_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::update()
{
  // call base class routine
  ScaTraTimIntImpl::update();

  // compute flux vector field for later output BEFORE time shift of results
  // is performed below !!
  if (calcflux_domain_ != Inpar::ScaTra::flux_none or
      calcflux_boundary_ != Inpar::ScaTra::flux_none)
  {
    if (is_result_step() or do_boundary_flux_statistics()) calc_flux(true);
  }

  // solution of this step becomes most recent solution of the last step
  phinm_->Update(1.0, *phin_, 0.0);
  phin_->Update(1.0, *phinp_, 0.0);

  // call time update of forcing routine
  if (homisoturb_forcing_ != Teuchos::null) homisoturb_forcing_->time_update_forcing();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::write_restart() const
{
  // call base class routine
  ScaTraTimIntImpl::write_restart();

  // additional state vectors that are needed for BDF2 restart
  output_->write_vector("phin", phin_);
  output_->write_vector("phinm", phinm_);
}

/*----------------------------------------------------------------------*
 -----------------------------------------------------------------------*/
void ScaTra::TimIntBDF2::read_restart(const int step, Teuchos::RCP<Core::IO::InputControl> input)
{
  // call base class routine
  ScaTraTimIntImpl::read_restart(step, input);

  Teuchos::RCP<Core::IO::DiscretizationReader> reader(Teuchos::null);
  if (input == Teuchos::null)
  {
    reader = Teuchos::rcp(new Core::IO::DiscretizationReader(
        discret_, Global::Problem::instance()->input_control_file(), step));
  }
  else
    reader = Teuchos::rcp(new Core::IO::DiscretizationReader(discret_, input, step));
  time_ = reader->read_double("time");
  step_ = reader->read_int("step");

  if (myrank_ == 0)
    std::cout << "Reading ScaTra restart data (time=" << time_ << " ; step=" << step_ << ")"
              << '\n';

  // read state vectors that are needed for BDF2 restart
  reader->read_vector(phinp_, "phinp");
  reader->read_vector(phin_, "phin");
  reader->read_vector(phinm_, "phinm");

  read_restart_problem_specific(step, *reader);

  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
      turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    avm3_preparation();
}

FOUR_C_NAMESPACE_CLOSE
