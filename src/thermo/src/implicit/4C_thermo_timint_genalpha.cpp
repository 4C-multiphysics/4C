// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_timint_genalpha.hpp"

#include "4C_thermo_ele_action.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | calc coefficients for given rho_inf                      seitz 03/16 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::calc_coeff()
{
  // rho_inf specified --> calculate optimal parameters
  if (rho_inf_ != -1.)
  {
    if ((rho_inf_ < 0.0) or (rho_inf_ > 1.0)) FOUR_C_THROW("rho_inf out of range [0.0,1.0]");
    if ((gamma_ != 0.5) or (alpham_ != 0.5) or (alphaf_ != 0.5))
      FOUR_C_THROW("you may only specify RHO_INF or the other three parameters");
    alpham_ = 0.5 * (3.0 - rho_inf_) / (rho_inf_ + 1.0);
    alphaf_ = 1.0 / (rho_inf_ + 1.0);
    gamma_ = 0.5 + alpham_ - alphaf_;
  }
}

/*----------------------------------------------------------------------*
 | check if coefficients are in correct regime               dano 10/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::verify_coeff()
{
  // alpha_f
  if ((alphaf_ < 0.0) or (alphaf_ > 1.0)) FOUR_C_THROW("alpha_f out of range [0.0,1.0]");
  // alpha_m
  if ((alpham_ < 0.0) or (alpham_ > 1.5)) FOUR_C_THROW("alpha_m out of range [0.0,1.0]");
  // gamma:
  if ((gamma_ <= 0.0) or (gamma_ > 1.0)) FOUR_C_THROW("gamma out of range (0.0,1.0]");

  // mid-averaging type
  // In principle, there exist two mid-averaging possibilities, namely TR-like and IMR-like,
  // where TR-like means trapezoidal rule and IMR-like means implicit mid-point rule.
  // We used to maintain implementations of both variants, but due to its significantly
  // higher complexity, the IMR-like version has been deleted (popp 02/2013). The nice
  // thing about TR-like mid-averaging is that all element (and thus also material) calls
  // are exclusively(!) carried out at the end-point t_{n+1} of each time interval, but
  // never explicitly at some generalised midpoint, such as t_{n+1-\alpha_f}. Thus, any
  // cumbersome extrapolation of history variables, etc. becomes obsolete.
  if (midavg_ != Thermo::midavg_trlike)
    FOUR_C_THROW("mid-averaging of internal forces only implemented TR-like");

  // done
  return;

}  // VerifyCoeff()


/*----------------------------------------------------------------------*
 | constructor                                               dano 10/09 |
 *----------------------------------------------------------------------*/
Thermo::TimIntGenAlpha::TimIntGenAlpha(const Teuchos::ParameterList& ioparams,
    const Teuchos::ParameterList& tdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : TimIntImpl(ioparams, tdynparams, xparams, actdis, solver, output),
      midavg_(Teuchos::getIntegralValue<Thermo::MidAverageEnum>(
          tdynparams.sublist("GENALPHA"), "GENAVG")),
      /* iterupditer_(false), */
      gamma_(tdynparams.sublist("GENALPHA").get<double>("GAMMA")),
      alphaf_(tdynparams.sublist("GENALPHA").get<double>("ALPHA_F")),
      alpham_(tdynparams.sublist("GENALPHA").get<double>("ALPHA_M")),
      rho_inf_(tdynparams.sublist("GENALPHA").get<double>("RHO_INF")),
      tempm_(nullptr),
      ratem_(nullptr),
      fint_(nullptr),
      fintm_(nullptr),
      fintn_(nullptr),
      fext_(nullptr),
      fextm_(nullptr),
      fextn_(nullptr),
      fcap_(nullptr),
      fcapm_(nullptr),
      fcapn_(nullptr)
{
  // calculate coefficients from given spectral radius
  calc_coeff();

  // info to user
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    // check if coefficients have admissible values
    verify_coeff();

    // print values of time integration parameters to screen
    std::cout << "with generalised-alpha" << std::endl
              << "   alpha_f = " << alphaf_ << std::endl
              << "   alpha_m = " << alpham_ << std::endl
              << "   gamma = " << gamma_ << std::endl
              << "   midavg = " << Thermo::mid_average_string(midavg_) << std::endl;
  }

  // determine capacity and initial temperature rates
  determine_capa_consist_temp_rate();

  // create state vectors

  // mid-temperatures
  tempm_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // mid-temperature rates
  ratem_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

  // create force vectors

  // internal force vector F_{int;n} at last time
  fint_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // internal mid-force vector F_{int;n+alpha_f}
  fintm_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // internal force vector F_{int;n+1} at new time
  fintn_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // stored force vector F_{transient;n} at last time
  fcap_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // stored force vector F_{transient;n+\alpha_m} at new time
  fcapm_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // stored force vector F_{transient;n+1} at new time
  fcapn_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // set initial internal force vector
  apply_force_tang_internal((*time_)[0], (*dt_)[0], (*temp_)(0), zeros_, fcap_, fint_, tang_);

  // external force vector F_ext at last times
  fext_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // external mid-force vector F_{ext;n+alpha_f}
  fextm_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // external force vector F_{n+1} at new time
  fextn_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
  // set initial external force vector
  apply_force_external((*time_)[0], (*temp_)(0), *fext_);
  // set initial external force vector of convective heat transfer boundary
  // conditions
  apply_force_external_conv((*time_)[0], (*temp_)(0), (*temp_)(0), fext_, tang_);

  // have a nice day
  return;

}  // TimIntGenAlpha()


/*----------------------------------------------------------------------*
 | Consistent predictor with constant temperatures           dano 10/09 |
 | and consistent temperature rates and temperatures                    |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::predict_const_temp_consist_rate()
{
  // time step size
  const double dt = (*dt_)[0];

  // constant predictor : temperature in domain
  tempn_->Update(1.0, *(*temp_)(0), 0.0);

  // consistent temperature rates
  // R_{n+1}^{i+1} = (gamma - 1)/gamma . R_n + 1/(gamma . dt) . (T_{n+1}^{i+1} - T_n)
  raten_->Update(1.0, *tempn_, -1.0, *(*temp_)(0), 0.0);
  raten_->Update(-(1 - gamma_) / gamma_, *(*rate_)(0), (1 / (gamma_ * dt)));

  // watch out
  return;

}  // predict_const_temp_consist_rate()


/*----------------------------------------------------------------------*
 | evaluate residual force and its tangent, ie derivative    dano 10/09 |
 | with respect to end-point temperatures \f$T_{n+1}\f$                 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::evaluate_rhs_tang_residual()
{
  // build by last converged state and predicted target state
  // the predicted mid-state
  evaluate_mid_state();

  // build new external forces
  fextn_->PutScalar(0.0);

  // initialise tangent matrix to zero
  tang_->zero();

  // set initial external force vector of convective heat transfer boundary
  // conditions

  // if the boundary condition shall be dependent on the current temperature
  // solution T_n+1 --> linearisation must be uncommented
  // --> use tempn_

  // if the old temperature T_n  is sufficient --> no linearisation needed!
  // --> use (*temp_)(0)
  apply_force_external_conv(timen_, (*temp_)(0), tempn_, fextn_, tang_);

  apply_force_external(timen_, (*temp_)(0), *fextn_);

  // external mid-forces F_{ext;n+1-alpha_f} (fextm)
  //    F_{ext;n+alpha_f} := alpha_f * F_{ext;n+1} + (1. - alpha_f) * F_{ext;n}
  fextm_->Update(alphaf_, *fextn_, (1. - alphaf_), *fext_, 0.0);

  // initialise internal forces
  fintn_->PutScalar(0.0);
  // total capacity mid-forces are calculated in the element
  // F_{cap;n+alpha_m} := M_capa . R_{n+alpha_m}
  fcapm_->PutScalar(0.0);

  // ordinary internal force and tangent
  apply_force_tang_internal(timen_, (*dt_)[0], tempn_, tempi_, fcapm_, fintn_, tang_);

  // total internal mid-forces F_{int;n+alpha_f} ----> TR-like
  // F_{int;n+alpha_f} := alpha_f . F_{int;n+1} + (1. - alpha_f) . F_{int;n}
  fintm_->Update(alphaf_, *fintn_, (1. - alphaf_), *fint_, 0.0);

  // total capacitiy forces F_{cap;n+1}
  // F_{cap;n+1} := 1/alpha_m . F_{cap;n+alpha_m} + (1. - alpha_m)/alpha_m . F_{cap;n}
  // using the interpolation to the midpoint
  // F_{cap;n+alpha_m} := alpha_m . F_{cap;n+1} + (1. - alpha_m) . F_{cap;n}
  fcapn_->Update((1. / alpham_), *fcapm_, (1. - alpham_) / alpham_, *fcap_, 0.0);

  // build residual
  //    Res = F_{cap;n+alpha_m}
  //        + F_{int;n+alpha_f}
  //        - F_{ext;n+alpha_f}
  fres_->Update(1.0, *fcapm_, 0.0);
  fres_->Update(1.0, *fintm_, 1.0);
  fres_->Update(-1.0, *fextm_, 1.0);

  // no further modification on tang_ required
  // tang_ is already effective dynamic tangent matrix
  tang_->complete();  // close tangent matrix

  // hallelujah
  return;

}  // evaluate_rhs_tang_residual()


/*----------------------------------------------------------------------*
 | evaluate mid-state vectors by averaging end-point         dano 05/13 |
 | vectors                                                              |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::evaluate_mid_state()
{
  // be careful: in contrast to temporal discretisation of structural field
  // (1-alpha) is used for OLD solution at t_n
  // mid-temperatures T_{n+1-alpha_f} (tempm)
  // T_{n+alpha_f} := alphaf * T_{n+1} + (1.-alphaf) * T_n
  tempm_->Update(alphaf_, *tempn_, (1. - alphaf_), (*temp_)[0], 0.0);

  // mid-temperature rates R_{n+1-alpha_f} (ratem)
  // R_{n+alpha_m} := alpham * R_{n+1} + (1.-alpham) * R_{n}
  // pass ratem_ to the element to calculate fcapm_
  ratem_->Update(alpham_, *raten_, (1. - alpham_), (*rate_)[0], 0.0);

  // jump
  return;

}  // EvaluateMidState()


/*----------------------------------------------------------------------*
 | calculate characteristic/reference norms for              dano 10/09 |
 | temperatures originally by lw                                        |
 *----------------------------------------------------------------------*/
double Thermo::TimIntGenAlpha::calc_ref_norm_temperature()
{
  // The reference norms are used to scale the calculated iterative
  // temperature norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  double charnormtemp = 0.0;
  charnormtemp = Thermo::Aux::calculate_vector_norm(iternorm_, *(*temp_)(0));

  // rise your hat
  return charnormtemp;

}  // calc_ref_norm_temperature()


/*----------------------------------------------------------------------*
 | calculate characteristic/reference norms for forces       dano 10/09 |
 | originally by lw                                                     |
 *----------------------------------------------------------------------*/
double Thermo::TimIntGenAlpha::calc_ref_norm_force()
{
  // The reference norms are used to scale the calculated iterative
  // temperature norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  // norm of the internal forces
  double fintnorm = 0.0;
  fintnorm = Thermo::Aux::calculate_vector_norm(iternorm_, *fintm_);

  // norm of the external forces
  double fextnorm = 0.0;
  fextnorm = Thermo::Aux::calculate_vector_norm(iternorm_, *fextm_);

  // norm of the capacity forces
  double fcapnorm = 0.0;
  fcapnorm = Thermo::Aux::calculate_vector_norm(iternorm_, *fcapm_);

  // norm of the reaction forces
  double freactnorm = 0.0;
  freactnorm = Thermo::Aux::calculate_vector_norm(iternorm_, *freact_);

  // determine worst value ==> characteristic norm
  return std::max(fcapnorm, std::max(fintnorm, std::max(fextnorm, freactnorm)));

}  // calc_ref_norm_force()


/*----------------------------------------------------------------------*
 | update after time step                                    dano 05/13 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::update_iter_incrementally()
{
  // auxiliary global vector holding new temperature rates
  // by extrapolation/scheme on __all__ DOFs. This includes
  // the Dirichlet DOFs as well. Thus we need to protect those
  // DOFs of overwriting; they already hold the
  // correctly 'predicted', final values.
  std::shared_ptr<Core::LinAlg::Vector<double>> aux =
      Core::LinAlg::create_vector(*discret_->dof_row_map(), true);

  // further auxiliary variables
  // step size \f$\Delta t_{n}\f$
  const double dt = (*dt_)[0];

  // new end-point temperatures
  // T_{n+1}^{i+1} := T_{n+1}^{i} + IncT_{n+1}^{i+1}
  tempn_->Update(1.0, *tempi_, 1.0);

  // new end-point temperature rates
  // R_{n+1}^{i+1} = -(1- gamma)/gamma . R_n + 1/(gamma . dt) . (T_{n+1}^{i+1} - T_n)
  aux->Update(1.0, *tempn_, -1.0, *(*temp_)(0), 0.0);
  aux->Update(-(1.0 - gamma_) / gamma_, *(*rate_)(0), (1 / (gamma_ * dt)));
  // put only to free/non-DBC DOFs
  dbcmaps_->insert_other_vector(*dbcmaps_->extract_other_vector(*aux), *raten_);

  // bye
  return;

}  // update_iter_incrementally()


/*----------------------------------------------------------------------*
 | iterative iteration update of state                       dano 10/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::update_iter_iteratively()
{
  // new end-point temperatures
  // T_{n+1}^{i+1} := T_{n+1}^{i} + IncT_{n+1}^{i}
  tempn_->Update(1.0, *tempi_, 1.0);

  // new end-point temperature rates
  // R_{n+1}^{i+1} := R_{n+1}^{i} + 1/(gamma . dt) IncT_{n+1}^{i+1}
  raten_->Update(1.0 / (gamma_ * (*dt_)[0]), *tempi_, 1.0);

  // bye
  return;

}  // update_iter_iteratively()


/*----------------------------------------------------------------------*
 | update after time step                                    dano 10/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::update_step_state()
{
  // update all old state at t_{n-1} etc
  // important for step size adaptivity
  // new temperatures at t_{n+1} -> t_n
  //    T_{n} := T_{n+1}, etc
  temp_->update_steps(*tempn_);
  // new temperature rates at t_{n+1} -> t_n
  //    R_{n} := R_{n+1}, etc
  rate_->update_steps(*raten_);

  // update new external force
  //    F_{ext;n} := F_{ext;n+1}
  fext_->Update(1.0, *fextn_, 0.0);

  // update new internal force
  //    F_{int;n} := F_{int;n+1}
  fint_->Update(1.0, *fintn_, 0.0);

  // update new stored transient force
  //    F_{cap;n} := F_{cap;n+1}
  fcap_->Update(1.0, *fcapn_, 0.0);

  // look out
  return;

}  // update_step_state()


/*----------------------------------------------------------------------*
 | update after time step after output on element level      dano 05/13 |
 | update anything that needs to be updated at the element level        |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::update_step_element()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // other parameters that might be needed by the elements
  p.set("total time", timen_);
  p.set("delta time", (*dt_)[0]);
  // action for elements
  p.set<Thermo::Action>("action", Thermo::calc_thermo_update_istep);
  // go to elements
  discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);

}  // update_step_element()


/*----------------------------------------------------------------------*
 | read restart forces                                       dano 10/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::read_restart_force()
{
  // read the vectors that were written in WriteRestartForce()
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step_);
  reader.read_vector(fext_, "fexternal");
  reader.read_vector(fint_, "fint");
  reader.read_vector(fcap_, "fcap");

  // bye
  return;

}  // ReadRestartForce()


/*----------------------------------------------------------------------*
 | write internal and external forces for restart            dano 07/13 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::write_restart_force(
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
{
  // in contrast to former implementation we save the current vectors.
  // This is required in case of materials with history.
  // Recalculation of restarted state is not possible.
  output->write_vector("fexternal", fext_);
  output->write_vector("fint", fint_);
  output->write_vector("fcap", fcap_);
  return;

}  // WriteRestartForce()


/*----------------------------------------------------------------------*
 | evaluate the internal force and the tangent               dano 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::apply_force_tang_internal(const double time,  //!< evaluation time
    const double dt,                                                       //!< step size
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,              //!< temperature state
    const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,  //!< residual temperatures
    std::shared_ptr<Core::LinAlg::Vector<double>> fcap,         //!< capacity force
    std::shared_ptr<Core::LinAlg::Vector<double>> fint,         //!< internal force
    std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
)
{
  //! create the parameters for the discretization
  Teuchos::ParameterList p;
  //! set parameters
  p.set<double>("alphaf", alphaf_);
  p.set<double>("alpham", alpham_);
  p.set<double>("gamma", gamma_);
  // set the mid-temperature rate R_{n+alpha_m} required for fcapm_
  p.set<std::shared_ptr<const Core::LinAlg::Vector<double>>>("mid-temprate", ratem_);
  p.set<double>("timefac", alphaf_);

  //! call the base function
  TimInt::apply_force_tang_internal(p, time, dt, temp, tempi, fcap, fint, tang);
  //! finish
  return;

}  // apply_force_tang_internal()


/*----------------------------------------------------------------------*
 | evaluate the internal force                               dano 08/09 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::apply_force_internal(const double time,  //!< evaluation time
    const double dt,                                                  //!< step size
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,         //!< temperature state
    const std::shared_ptr<Core::LinAlg::Vector<double>> tempi,        //!< incremental temperatures
    std::shared_ptr<Core::LinAlg::Vector<double>> fint                //!< internal force
)
{
  //! create the parameters for the discretization
  Teuchos::ParameterList p;
  //! set parameters
  p.set<double>("alphaf", alphaf_);
  p.set<double>("alpham", alpham_);
  p.set<double>("gamma", gamma_);
  //! call the base function
  TimInt::apply_force_internal(p, time, dt, temp, tempi, fint);
  //! finish
  return;

}  // apply_force_internal()


/*----------------------------------------------------------------------*
 | evaluate the convective boundary condition                dano 06/13 |
 *----------------------------------------------------------------------*/
void Thermo::TimIntGenAlpha::apply_force_external_conv(const double time,  //!< evaluation time
    const std::shared_ptr<Core::LinAlg::Vector<double>> tempn,  //!< old temperature state T_n
    const std::shared_ptr<Core::LinAlg::Vector<double>> temp,   //!< temperature state T_n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> fext,         //!< external force
    std::shared_ptr<Core::LinAlg::SparseMatrix> tang            //!< tangent matrix
)
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // set parameters
  p.set<double>("alphaf", alphaf_);

  // call the base function
  TimInt::apply_force_external_conv(p, time, tempn, temp, fext, tang);
  // finish
  return;

}  // apply_force_external_conv()


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
