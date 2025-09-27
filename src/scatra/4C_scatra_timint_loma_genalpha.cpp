// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_loma_genalpha.hpp"

#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_turbulence_hit_scalar_forcing.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       vg 11/08 |
 *----------------------------------------------------------------------*/
ScaTra::TimIntLomaGenAlpha::TimIntLomaGenAlpha(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, sctratimintparams, extraparams, output),
      ScaTraTimIntLoma(actdis, solver, params, sctratimintparams, extraparams, output),
      TimIntGenAlpha(actdis, solver, sctratimintparams, extraparams, output),
      thermpressaf_(0.0),
      thermpressam_(0.0),
      thermpressdtaf_(0.0),
      thermpressdtam_(0.0)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             rauch 09/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::init()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntGenAlpha::init();
  ScaTraTimIntLoma::init();

  return;
}

/*----------------------------------------------------------------------*
 |  setup time integration                                  rauch 09/16 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::setup()
{
  // call init()-functions of base classes
  // note: this order is important
  TimIntGenAlpha::setup();
  ScaTraTimIntLoma::setup();

  return;
}



/*----------------------------------------------------------------------*
 | predict thermodynamic pressure and time derivative          vg 12/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::predict_therm_pressure()
{
  // same-thermodynamic-pressure predictor (not required to be performed,
  // since we just updated the thermodynamic pressure, and thus,
  // thermpressnp_ = thermpressn_)
  // prediction of time derivative:
  double fact = (gamma_ - 1.0) / gamma_;
  thermpressdtnp_ = fact * thermpressdtn_;

  // same-thermodynamic-pressure-derivative predictor (currently not used)
  // thermpressnp_ += dta_*thermpressdtn_;
  // prediction of time derivative not required (would also not be required
  // to be performed, since we just updated the time derivatives of density,
  // and thus, thermpressdtnp_ = thermpressdtn_)

  return;
}


/*----------------------------------------------------------------------*
 | compute values of therm. pressure at intermediate time steps         |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::compute_therm_pressure_intermediate_values()
{
  // thermodynamic pressure at n+alpha_F and n+alpha_M for low-Mach-number case
  // -> required for evaluation of equation of state
  thermpressaf_ = alphaF_ * thermpressnp_ + (1.0 - alphaF_) * thermpressn_;
  thermpressam_ = alphaM_ * thermpressnp_ + (1.0 - alphaM_) * thermpressn_;

  // time derivative of thermodyn. press. at n+alpha_F for low-Mach-number case
  // -> required as right-hand-side contribution to temperature equation,
  // hence, evaluated at n+alpha_F
  thermpressdtaf_ = alphaF_ * thermpressdtnp_ + (1.0 - alphaF_) * thermpressdtn_;

  // time derivative of thermodyn. press. at n+alpha_M for low-Mach-number case
  // -> required for transfer to flow solver and use in continuity equation
  thermpressdtam_ = alphaM_ * thermpressdtnp_ + (1.0 - alphaM_) * thermpressdtn_;

  return;
}


/*----------------------------------------------------------------------*
 | compute thermodynamic pressure for low-Mach-number flow     vg 12/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::compute_therm_pressure()
{
  // compute temperature at n+alpha_F
  phiaf_->update(alphaF_, *phinp_, (1.0 - alphaF_), *phin_, 0.0);

  // define element parameter list
  Teuchos::ParameterList eleparams;

  // DO THIS BEFORE PHINP IS SET (ClearState() is called internally!!!!)
  // compute flux approximation and add it to the parameter list
  add_flux_approx_to_parameter_list(eleparams);

  // set scalar values needed by elements
  discret_->clear_state();
  discret_->set_state("phinp", *phiaf_);

  // set action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_domain_and_bodyforce, eleparams);
  set_element_time_parameter();

  // variables for integrals of domain and bodyforce
  Core::LinAlg::SerialDenseVector scalars(2);

  // evaluate domain and bodyforce integral
  discret_->evaluate_scalars(eleparams, scalars);

  // get global integral values
  double pardomint = (scalars)[0];
  double parbofint = (scalars)[1];

  // set action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_loma_therm_press, eleparams);

  // variables for integrals of normal velocity and diffusive flux
  double normvelint = 0.0;
  double normdifffluxint = 0.0;
  eleparams.set("normal velocity integral", normvelint);
  eleparams.set("normal diffusive flux integral", normdifffluxint);

  // evaluate velocity-divergence and diffusive (minus sign!) flux on boundaries
  // We may use the flux-calculation condition for calculation of fluxes for
  // thermodynamic pressure, since it is usually at the same boundary.
  std::vector<std::string> condnames;
  condnames.push_back("ScaTraFluxCalc");
  for (unsigned int i = 0; i < condnames.size(); i++)
  {
    discret_->evaluate_condition(
        eleparams, nullptr, nullptr, nullptr, nullptr, nullptr, condnames[i]);
  }

  // get integral values on this proc
  normvelint = eleparams.get<double>("normal velocity integral");
  normdifffluxint = eleparams.get<double>("normal diffusive flux integral");

  // get integral values in parallel case
  double parnormvelint = 0.0;
  double parnormdifffluxint = 0.0;
  parnormvelint = Core::Communication::sum_all(normvelint, discret_->get_comm());
  parnormdifffluxint = Core::Communication::sum_all(normdifffluxint, discret_->get_comm());

  // clean up
  discret_->clear_state();

  // compute thermodynamic pressure (with specific heat ratio fixed to be 1.4)
  const double shr = 1.4;
  const double divt = shr * parnormvelint / pardomint;
  const double lhs = alphaF_ * genalphafac_ * dta_ * divt;
  const double rhs =
      genalphafac_ * dta_ * (shr - 1.0) * (-parnormdifffluxint + parbofint) / pardomint;
  const double hist = thermpressn_ - (1.0 - alphaF_) * genalphafac_ * dta_ * divt * thermpressn_ +
                      (1.0 - genalphafac_) * dta_ * thermpressdtn_;
  thermpressnp_ = (rhs + hist) / (1.0 + lhs);

  // print out thermodynamic pressure
  if (myrank_ == 0)
  {
    std::cout << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
    std::cout << "Data output for instationary thermodynamic pressure:" << std::endl;
    std::cout << "Velocity in-/outflow at indicated boundary: " << parnormvelint << std::endl;
    std::cout << "Diffusive flux at indicated boundary: " << parnormdifffluxint << std::endl;
    std::cout << "Thermodynamic pressure: " << thermpressnp_ << std::endl;
    std::cout << "+--------------------------------------------------------------------------------"
                 "------------+"
              << std::endl;
  }

  // compute time derivative of thermodynamic pressure at time step n+1
  compute_therm_pressure_time_derivative();

  // compute values at intermediate time steps
  compute_therm_pressure_intermediate_values();

  return;
}


/*----------------------------------------------------------------------*
 | compute time derivative of thermodynamic pressure           vg 09/09 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::compute_therm_pressure_time_derivative()
{
  // time derivative of thermodynamic pressure:
  // tpdt(n+1) = (tp(n+1)-tp(n)) / (gamma*dt) + (1-(1/gamma))*tpdt(n)
  const double fact1 = 1.0 / (gamma_ * dta_);
  const double fact2 = 1.0 - (1.0 / gamma_);
  thermpressdtnp_ = fact1 * (thermpressnp_ - thermpressn_) + fact2 * thermpressdtn_;

  return;
}


/*----------------------------------------------------------------------*
 | update thermodynamic pressure at n for low-Mach-number flow vg 12/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::update_therm_pressure()
{
  thermpressn_ = thermpressnp_;
  thermpressdtn_ = thermpressdtnp_;

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::write_restart() const
{
  // write standard fields first
  TimIntGenAlpha::write_restart();

  // write additional restart data for loma
  // required for restart of closed systems

  // thermodynamic pressure at time n+1
  output_->write_double("thermpressnp", thermpressnp_);
  // thermodynamic pressure at time n
  output_->write_double("thermpressn", thermpressn_);
  // thermodynamic pressure at time n+alpha_f
  output_->write_double("thermpressaf", thermpressaf_);
  // thermodynamic pressure at time n+alpha_m
  output_->write_double("thermpressam", thermpressam_);
  // time derivative of thermodynamic pressure at time n+1
  output_->write_double("thermpressdtnp", thermpressdtnp_);
  // time derivative of thermodynamic pressure at time n
  output_->write_double("thermpressdtn", thermpressdtn_);
  // time derivative of thermodynamic pressure at time n+alpha_f
  output_->write_double("thermpressdtaf", thermpressdtaf_);
  // time derivative of thermodynamic pressure at time n+alpha_m
  output_->write_double("thermpressdtam", thermpressdtam_);
  // as well as initial mass
  output_->write_double("initialmass", initialmass_);

  return;
}


/*----------------------------------------------------------------------*
 |                                                             vg 11/08 |
 -----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::read_restart(
    const int step, std::shared_ptr<Core::IO::InputControl> input)
{
  // do standard output
  TimIntGenAlpha::read_restart(step, input);

  // restart data of loma problems
  // required for restart of closed systems

  std::shared_ptr<Core::IO::DiscretizationReader> reader(nullptr);
  if (input == nullptr)
    reader = std::make_shared<Core::IO::DiscretizationReader>(
        discret_, Global::Problem::instance()->input_control_file(), step);
  else
    reader = std::make_shared<Core::IO::DiscretizationReader>(discret_, input, step);

  // thermodynamic pressure at time n+1
  thermpressnp_ = reader->read_double("thermpressnp");
  // thermodynamic pressure at time n
  thermpressn_ = reader->read_double("thermpressn");
  // thermodynamic pressure at time n+alpha_f
  thermpressaf_ = reader->read_double("thermpressaf");
  // thermodynamic pressure at time n+alpha_m
  thermpressam_ = reader->read_double("thermpressam");
  // time derivative of thermodynamic pressure at time n+1
  thermpressdtnp_ = reader->read_double("thermpressdtnp");
  // time derivative of thermodynamic pressure at time n
  thermpressdtn_ = reader->read_double("thermpressdtn");
  // time derivative of thermodynamic pressure at time n+alpha_f
  thermpressdtaf_ = reader->read_double("thermpressdtaf");
  // time derivative of thermodynamic pressure at time n+alpha_m
  thermpressdtam_ = reader->read_double("thermpressdtam");
  // as well as initial mass
  initialmass_ = reader->read_double("initialmass");

  return;
}


/*----------------------------------------------------------------------*
 | dynamic Smagorinsky model                           rasthofer  08/12 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::dynamic_computation_of_cs()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Prt
    // compute averaged values for LkMk and MkMk
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet_toggle();
    DynSmag_->apply_filter_for_dynamic_computation_of_prt(
        phiaf_, thermpressaf_, dirichtoggle, *extraparams_, nds_vel());
  }

  return;
}


/*----------------------------------------------------------------------*
 | dynamic Vreman model                                krank  09/13     |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::dynamic_computation_of_cv()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet_toggle();
    Vrem_->apply_filter_for_dynamic_computation_of_dt(
        phiaf_, thermpressaf_, dirichtoggle, *extraparams_, nds_vel());
  }

  return;
}


/*-------------------------------------------------------------------------------------*
 | add thermodynamic pressure to parameter list for element evaluation rasthofer 12/13 |
 *-------------------------------------------------------------------------------------*/
void ScaTra::TimIntLomaGenAlpha::add_therm_press_to_parameter_list(
    Teuchos::ParameterList& params  //!< parameter list
)
{
  params.set("thermodynamic pressure", thermpressaf_);
  params.set("thermodynamic pressure at n+alpha_M", thermpressam_);
  params.set("time derivative of thermodynamic pressure", thermpressdtaf_);
  discret_->set_state("phiam", *phiam_);
  return;
}

FOUR_C_NAMESPACE_CLOSE
