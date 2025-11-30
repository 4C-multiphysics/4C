// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timint_genalpha.hpp"

#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_aux.hpp"
#include "4C_structure_new_impl_genalpha.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
void Solid::TimIntGenAlpha::calc_coeff()
{
  Solid::IMPLICIT::GenAlpha::Coefficients coeffs;
  // get a copy of the input parameters
  coeffs.beta_ = beta_;
  coeffs.gamma_ = gamma_;
  coeffs.alphaf_ = alphaf_;
  coeffs.alpham_ = alpham_;
  coeffs.rhoinf_ = rho_inf_;

  compute_generalized_alpha_parameters(coeffs);

  beta_ = coeffs.beta_;
  gamma_ = coeffs.gamma_;
  alphaf_ = coeffs.alphaf_;
  alpham_ = coeffs.alpham_;
  rho_inf_ = coeffs.rhoinf_;
}

/*----------------------------------------------------------------------*/
void Solid::TimIntGenAlpha::verify_coeff()
{
  // beta
  if ((beta_ <= 0.0) or (beta_ > 0.5))
    FOUR_C_THROW("beta out of range (0.0,0.5]");
  else
    std::cout << "   beta = " << beta_ << '\n';
  // gamma
  if ((gamma_ <= 0.0) or (gamma_ > 1.0))
    FOUR_C_THROW("gamma out of range (0.0,1.0]");
  else
    std::cout << "   gamma = " << gamma_ << '\n';
  // alpha_f
  if ((alphaf_ < 0.0) or (alphaf_ >= 1.0))
    FOUR_C_THROW("alpha_f out of range [0.0,1.0)");
  else
    std::cout << "   alpha_f = " << alphaf_ << '\n';
  // alpha_m
  if ((alpham_ < -1.0) or (alpham_ >= 1.0))
    FOUR_C_THROW("alpha_m out of range [-1.0,1.0)");
  else
    std::cout << "   alpha_m = " << alpham_ << '\n';

  // mid-averaging type
  // In principle, there exist two mid-averaging possibilities, TR-like and IMR-like,
  // where TR-like means trapezoidal rule and IMR-like means implicit mid-point rule.
  // We used to maintain implementations of both variants, but due to its significantly
  // higher complexity, the IMR-like version has been deleted (popp 02/2013). The nice
  // thing about TR-like mid-averaging is that all element (and thus also material) calls
  // are exclusively(!) carried out at the end-point t_{n+1} of each time interval, but
  // never explicitly at some generalized midpoint, such as t_{n+1-\alpha_f}. Thus, any
  // cumbersome extrapolation of history variables, etc. becomes obsolete.
  if (midavg_ != Inpar::Solid::midavg_trlike)
    FOUR_C_THROW("mid-averaging of internal forces only implemented TR-like");
  else
    std::cout << "   midavg = " << midavg_ << '\n';
}

/*----------------------------------------------------------------------*/
/* constructor */
Solid::TimIntGenAlpha::TimIntGenAlpha(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams, std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::LinAlg::Solver> contactsolver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : TimIntImpl(timeparams, ioparams, sdynparams, xparams, actdis, solver, contactsolver, output),
      midavg_(Teuchos::getIntegralValue<Inpar::Solid::MidAverageEnum>(
          sdynparams.sublist("GENALPHA"), "GENAVG")),
      beta_(sdynparams.sublist("GENALPHA").get<double>("BETA")),
      gamma_(sdynparams.sublist("GENALPHA").get<double>("GAMMA")),
      alphaf_(sdynparams.sublist("GENALPHA").get<double>("ALPHA_F")),
      alpham_(sdynparams.sublist("GENALPHA").get<double>("ALPHA_M")),
      rho_inf_(sdynparams.sublist("GENALPHA").get<double>("RHO_INF")),
      dism_(nullptr),
      velm_(nullptr),
      accm_(nullptr),
      fint_(nullptr),
      fintm_(nullptr),
      fintn_(nullptr),
      fext_(nullptr),
      fextm_(nullptr),
      fextn_(nullptr),
      finert_(nullptr),
      finertm_(nullptr),
      finertn_(nullptr),
      fviscm_(nullptr),
      fint_str_(nullptr)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
}

/*----------------------------------------------------------------------------------------------*
 * Initialize this class                                                            rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntGenAlpha::init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver)
{
  // call init() in base class
  Solid::TimIntImpl::init(timeparams, sdynparams, xparams, actdis, solver);

  // calculate time integration parameters
  calc_coeff();

  // info to user about current time integration scheme and its parametrization
  if (myrank_ == 0)
  {
    Core::IO::cout << "with generalised-alpha" << Core::IO::endl;
    verify_coeff();

    std::cout << "   p_dis = " << method_order_of_accuracy_dis() << '\n'
              << "   p_vel = " << method_order_of_accuracy_vel() << '\n'
              << '\n';
  }
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntGenAlpha::setup()
{
  // call setup() in base class
  Solid::TimIntImpl::setup();

  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // determine mass, damping and initial accelerations
    determine_mass_damp_consist_accel();
  }
  else
  {
    /* the case of nonlinear inertia terms works so far only for examples with
     * vanishing initial accelerations, i.e. the initial external
     * forces and initial velocities have to be chosen consistently!!!
     */
    (*acc_)(0)->put_scalar(0.0);
  }

  // create state vectors

  // mid-displacements
  dism_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // mid-velocities
  velm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // mid-accelerations
  accm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // create force vectors

  // internal force vector F_{int;n} at last time
  fint_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // internal mid-force vector F_{int;n+1-alpha_f}
  fintm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // internal force vector F_{int;n+1} at new time
  fintn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // external force vector F_ext at last times
  fext_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // external mid-force vector F_{ext;n+1-alpha_f}
  fextm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // external force vector F_{n+1} at new time
  fextn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // set initial external force vector
  apply_force_external((*time_)[0], (*dis_)(0), disn_, *(*vel_)(0), *fext_);

  // inertial force vector F_{int;n} at last time
  finert_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // inertial mid-force vector F_{int;n+1-alpha_f}
  finertm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // inertial force vector F_{int;n+1} at new time
  finertn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // viscous mid-point force vector F_visc
  fviscm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // structural rhs for newton line search
  if (fresn_str_ != nullptr)
    fint_str_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // create parameter list
  Teuchos::ParameterList params;

  // for line search
  if (fintn_str_ != nullptr)
  {
    params.set("cond_rhs_norm", 0.);
    params.set("MyPID", myrank_);
  }

  // add initial forces due to 0D cardiovascular coupling conditions - needed in case of initial
  // ventricular pressure!
  Teuchos::ParameterList pwindk;
  pwindk.set("scale_timint", 1.0);
  pwindk.set("time_step_size", (*dt_)[0]);
  apply_force_stiff_cardiovascular0_d((*time_)[0], (*dis_)(0), fint_, stiff_, pwindk);

  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // set initial internal force vector
    apply_force_stiff_internal(
        (*time_)[0], (*dt_)[0], (*dis_)(0), zeros_, *(*vel_)(0), fint_, stiff_, params);
  }
  else
  {
    double timeintfac_dis = beta_ * (*dt_)[0] * (*dt_)[0];
    double timeintfac_vel = gamma_ * (*dt_)[0];

    // Check, if initial residuum really vanishes for acc_ = 0
    apply_force_stiff_internal_and_inertial((*time_)[0], (*dt_)[0], timeintfac_dis, timeintfac_vel,
        (*dis_)(0), zeros_, (*vel_)(0), (*acc_)(0), fint_, finert_, stiff_, mass_, params, beta_,
        gamma_, alphaf_, alpham_);

    nonlinear_mass_sanity_check(fext_, (*dis_)(0), (*vel_)(0), (*acc_)(0), &sdynparams_);

    if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_rotations and
        !solely_beam3_elements(*discret_))
    {
      FOUR_C_THROW(
          "Multiplicative Gen-Alpha time integration scheme only implemented for beam elements so "
          "far!");
    }
  }

  // init old time step value
  if (fintn_str_ != nullptr) fint_str_->update(1., *fintn_str_, 0.);
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant displacements
 * and consistent velocities and displacements */
void Solid::TimIntGenAlpha::predict_const_dis_consist_vel_acc()
{
  // constant predictor : displacement in domain
  disn_->update(1.0, *(*dis_)(0), 0.0);

  // consistent velocities following Newmark formulas
  veln_->update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  veln_->update((beta_ - gamma_) / beta_, *(*vel_)(0),
      (2. * beta_ - gamma_) * (*dt_)[0] / (2. * beta_), *(*acc_)(0), gamma_ / (beta_ * (*dt_)[0]));

  // consistent accelerations following Newmark formulas
  accn_->update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  accn_->update(-1. / (beta_ * (*dt_)[0]), *(*vel_)(0), (2. * beta_ - 1.) / (2. * beta_),
      *(*acc_)(0), 1. / (beta_ * (*dt_)[0] * (*dt_)[0]));

  // reset the residual displacement
  disi_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant velocities,
 * extrapolated displacements and consistent accelerations */
void Solid::TimIntGenAlpha::predict_const_vel_consist_acc()
{
  // extrapolated displacements based upon constant velocities
  // d_{n+1} = d_{n} + dt * v_{n}
  disn_->update(1.0, (*dis_)[0], (*dt_)[0], (*vel_)[0], 0.0);

  // consistent velocities following Newmark formulas
  veln_->update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  veln_->update((beta_ - gamma_) / beta_, *(*vel_)(0),
      (2. * beta_ - gamma_) * (*dt_)[0] / (2. * beta_), *(*acc_)(0), gamma_ / (beta_ * (*dt_)[0]));

  // consistent accelerations following Newmark formulas
  accn_->update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  accn_->update(-1. / (beta_ * (*dt_)[0]), *(*vel_)(0), (2. * beta_ - 1.) / (2. * beta_),
      *(*acc_)(0), 1. / (beta_ * (*dt_)[0] * (*dt_)[0]));

  // reset the residual displacement
  disi_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant accelerations
 * and extrapolated velocities and displacements */
void Solid::TimIntGenAlpha::predict_const_acc()
{
  // extrapolated displacements based upon constant accelerations
  // d_{n+1} = d_{n} + dt * v_{n} + dt^2 / 2 * a_{n}
  disn_->update(1.0, (*dis_)[0], (*dt_)[0], (*vel_)[0], 0.0);
  disn_->update((*dt_)[0] * (*dt_)[0] / 2., (*acc_)[0], 1.0);

  // extrapolated velocities (equal to consistent velocities)
  // v_{n+1} = v_{n} + dt * a_{n}
  veln_->update(1.0, (*vel_)[0], (*dt_)[0], (*acc_)[0], 0.0);

  // constant accelerations (equal to consistent accelerations)
  accn_->update(1.0, (*acc_)[0], 0.0);

  // reset the residual displacement
  disi_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*/
/* evaluate residual force and its stiffness, i.e. derivative
 * with respect to end-point displacements \f$D_{n+1}\f$ */
void Solid::TimIntGenAlpha::evaluate_force_stiff_residual(Teuchos::ParameterList& params)
{
  // get info about prediction step from parameter list
  bool predict = false;
  if (params.isParameter("predict")) predict = params.get<bool>("predict");

  // initialise stiffness matrix to zero
  stiff_->zero();

  // in the case of material damping initialise damping matrix to zero
  if (damping_ == Inpar::Solid::damp_material) damp_->zero();

  // build predicted mid-state by last converged state and predicted target state
  evaluate_mid_state();


  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->put_scalar(0.0);
  apply_force_stiff_external(timen_, (*dis_)(0), *disn_, *(*vel_)(0), *fextn_, stiff_);

  // additional external forces are added (e.g. interface forces)
  fextn_->update(1.0, *fifc_, 1.0);

  // external mid-forces F_{ext;n+1-alpha_f} ----> TR-like
  // F_{ext;n+1-alpha_f} := (1.-alphaf) * F_{ext;n+1} + alpha_f * F_{ext;n}
  fextm_->update(1. - alphaf_, *fextn_, alphaf_, *fext_, 0.0);

  // ************************** (2) INTERNAL FORCES ***************************
  fintn_->put_scalar(0.0);
  // build new internal forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    apply_force_stiff_internal(
        timen_, (*dt_)[0], disn_, disi_, *veln_, fintn_, stiff_, params, damp_);
  }
  else
  {
    if (pred_ != Inpar::Solid::pred_constdis)
    {
      FOUR_C_THROW(
          "Only the predictor predict_const_dis_consist_vel_acc() allowed for dynamic beam3r "
          "simulations!!!");
    }

    // If we have nonlinear inertia forces, the corresponding contributions are computed together
    // with the internal forces
    finertn_->put_scalar(0.0);
    mass_->zero();

    // In general the nonlinear inertia force can depend on displacements, velocities and
    // accelerations, i.e     finertn_=finertn_(disn_, veln_, accn_):
    //
    //    LIN finertn_ = [ d(finertn_)/d(disn_) + gamma_/(beta_*dt_)*d(finertn_)/d(veln_)
    //                 + 1/(beta_*dt_*dt_)*d(finertn_)/d(accn_) ]*disi_
    //
    //    LIN finertm_ = (1-alpha_m)/(beta_*dt_*dt_)[ (beta_*dt_*dt_)*d(finertn_)/d(disn_)
    //                 + (gamma_*dt_)*d(finertn_)/d(veln_) + d(finertn_)/d(accn_)]*disi_
    //
    // While the factor (1-alpha_m/(beta_*dt_*dt_) is applied later on in strtimint_genalpha.cpp the
    // factors timintfac_dis=(beta_*dt_*dt_) and timeintfac_vel=(gamma_*dt_) have directly to be
    // applied on element level before the three contributions of the linearization are summed up in
    // mass_.

    double timintfac_dis = beta_ * (*dt_)[0] * (*dt_)[0];
    double timintfac_vel = gamma_ * (*dt_)[0];
    apply_force_stiff_internal_and_inertial(timen_, (*dt_)[0], timintfac_dis, timintfac_vel, disn_,
        disi_, veln_, accn_, fintn_, finertn_, stiff_, mass_, params, beta_, gamma_, alphaf_,
        alpham_);
  }

  // add forces and stiffness due to constraints
  // (for TR scale constraint matrix with the same value fintn_ is scaled with)
  Teuchos::ParameterList pcon;
  pcon.set("scaleConstrMat", (1.0 - alphaf_));
  apply_force_stiff_constraint(timen_, (*dis_)(0), disn_, fintn_, stiff_, pcon);

  // add forces and stiffness due to 0D cardiovascular coupling conditions
  Teuchos::ParameterList pwindk;
  pwindk.set("scale_timint", (1. - alphaf_));
  pwindk.set("time_step_size", (*dt_)[0]);
  apply_force_stiff_cardiovascular0_d(timen_, disn_, fintn_, stiff_, pwindk);

  // add forces and stiffness due to spring dashpot condition
  Teuchos::ParameterList psprdash;
  psprdash.set("time_fac", gamma_ / (beta_ * (*dt_)[0]));
  psprdash.set("dt", (*dt_)[0]);  // needed only for cursurfnormal option!!
  apply_force_stiff_spring_dashpot(stiff_, fintn_, disn_, *veln_, predict, psprdash);

  // total internal mid-forces F_{int;n+1-alpha_f} ----> TR-like
  // F_{int;n+1-alpha_f} := (1.-alphaf) * F_{int;n+1} + alpha_f * F_{int;n}
  fintm_->update(1. - alphaf_, *fintn_, alphaf_, *fint_, 0.0);

  // ************************** (3) INERTIA FORCES ***************************

  // build new inertia forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // build new inertia forces and stiffness
    finertm_->put_scalar(0.0);
    // inertia forces #finertm_
    mass_->multiply(false, *accm_, *finertm_);
  }
  else
  {
    // total inertia mid-forces F_{inert;n+1-alpha_m} ----> TR-like
    // F_{inert;n+1-alpha_m} := (1.-alpham) * F_{inert;n+1} + alpha_m * F_{inert;n}
    finertm_->update(1. - alpham_, *finertn_, alpham_, *finert_, 0.0);
  }

  // ************************** (4) DAMPING FORCES ****************************

  // viscous forces due to Rayleigh damping
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    damp_->multiply(false, *velm_, *fviscm_);
  }

  // ******************** Finally, put everything together ********************

  // build residual and tangent matrix for standard case
  if (have_nonlinear_mass() != Inpar::Solid::MassLin::ml_rotations)
  {
    // build residual
    //    Res = M . A_{n+1-alpha_m}
    //        + C . V_{n+1-alpha_f}
    //        + F_{int;n+1-alpha_f}
    //        - F_{ext;n+1-alpha_f}
    fres_->update(-1.0, *fextm_, 0.0);
    fres_->update(1.0, *fintm_, 1.0);
    fres_->update(1.0, *finertm_, 1.0);
    if (damping_ == Inpar::Solid::damp_rayleigh)
    {
      fres_->update(1.0, *fviscm_, 1.0);
    }

    // build tangent matrix : effective dynamic stiffness matrix
    //    K_{Teffdyn} = (1 - alpha_m)/(beta*dt^2) M
    //                + (1 - alpha_f)*y/(beta*dt) C
    //                + (1 - alpha_f) K_{T}

    stiff_->add(*mass_, false, (1. - alpham_) / (beta_ * (*dt_)[0] * (*dt_)[0]), 1. - alphaf_);
    if (damping_ != Inpar::Solid::damp_none)
    {
      if (damping_ == Inpar::Solid::damp_material) damp_->complete();
      stiff_->add(*damp_, false, (1. - alphaf_) * gamma_ / (beta_ * (*dt_)[0]), 1.0);
    }
  }
  // build residual vector and tangent matrix if a multiplicative Gen-Alpha scheme for rotations is
  // applied
  else
  {
    build_res_stiff_nl_mass_rot(*fres_, *fextn_, *fintn_, *finertn_, *stiff_, *mass_);
  }

  // apply forces and stiffness due to contact / meshtying ----> TR-like
  // F_{c;n+1-alpha_f} := (1.-alphaf) * F_{c;n+1} + alpha_f * F_{c;n}
  apply_force_stiff_contact_meshtying(stiff_, fres_, disn_, predict);

  // calculate RHS without local condensations (for NewtonLs)
  if (fresn_str_ != nullptr)
  {
    // total internal mid-forces F_{int;n+1-alpha_f} ----> TR-like
    // F_{int;n+1-alpha_f} := (1.-alphaf) * F_{int;n+1} + alpha_f * F_{int;n}
    fresn_str_->update(1., *fintn_str_, 0.);
    fresn_str_->update(alphaf_, *fint_str_, 1. - alphaf_);
    fresn_str_->update(-1.0, *fextm_, 1.0);
    fresn_str_->update(1.0, *finertm_, 1.0);
    if (damping_ == Inpar::Solid::damp_rayleigh) fresn_str_->update(1.0, *fviscm_, 1.0);
    Core::LinAlg::apply_dirichlet_to_system(*fresn_str_, *zeros_, *(dbcmaps_->cond_map()));
  }

  // close stiffness matrix
  stiff_->complete();
}

/*----------------------------------------------------------------------*/
/* Evaluate/define the residual force vector #fres_ for
 * relaxation solution with solve_relaxation_linear */
void Solid::TimIntGenAlpha::evaluate_force_stiff_residual_relax(Teuchos::ParameterList& params)
{
  // compute residual forces #fres_ and stiffness #stiff_
  evaluate_force_stiff_residual(params);

  // overwrite the residual forces #fres_ with interface load
  if (have_nonlinear_mass() != Inpar::Solid::MassLin::ml_rotations)
  {
    // standard case
    fres_->update(-1 + alphaf_, *fifc_, 0.0);
  }
  else
  {
    // Remark: In the case of an multiplicative Gen-Alpha time integration scheme, all forces are
    // evaluated at the end point n+1.
    fres_->update(-1.0, *fifc_, 0.0);
  }
}

/*----------------------------------------------------------------------*/
/* Evaluate residual */
void Solid::TimIntGenAlpha::evaluate_force_residual()
{
  // build predicted mid-state by last converged state and predicted target state
  evaluate_mid_state();

  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->put_scalar(0.0);
  apply_force_external(timen_, (*dis_)(0), disn_, *(*vel_)(0), *fextn_);

  // additional external forces are added (e.g. interface forces)
  fextn_->update(1.0, *fifc_, 1.0);

  // external mid-forces F_{ext;n+1-alpha_f} ----> TR-like
  // F_{ext;n+1-alpha_f} := (1.-alphaf) * F_{ext;n+1} + alpha_f * F_{ext;n}
  fextm_->update(1. - alphaf_, *fextn_, alphaf_, *fext_, 0.0);

  // ************************** (2) INTERNAL FORCES ***************************

  fintn_->put_scalar(0.0);

  // build new internal forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    apply_force_internal(timen_, (*dt_)[0], disn_, disi_, *veln_, fintn_);
  }
  else
  {
    FOUR_C_THROW("Not implemented, yet.");
  }

  // total internal mid-forces F_{int;n+1-alpha_f} ----> TR-like
  // F_{int;n+1-alpha_f} := (1.-alphaf) * F_{int;n+1} + alpha_f * F_{int;n}
  fintm_->update(1. - alphaf_, *fintn_, alphaf_, *fint_, 0.0);

  // ************************** (3) INERTIAL FORCES ***************************

  // build new inertia forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // build new inertia forces and stiffness
    finertm_->put_scalar(0.0);
    // inertia forces #finertm_
    mass_->multiply(false, *accm_, *finertm_);
  }
  else
  {
    FOUR_C_THROW("Not implemented, yet.");
  }

  // ************************** (4) DAMPING FORCES ****************************

  // viscous forces due to Rayleigh damping
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    damp_->multiply(false, *velm_, *fviscm_);
  }

  // ******************** Finally, put everything together ********************

  // build residual and tangent matrix for standard case
  if (have_nonlinear_mass() != Inpar::Solid::MassLin::ml_rotations)
  {
    // build residual
    //    Res = M . A_{n+1-alpha_m}
    //        + C . V_{n+1-alpha_f}
    //        + F_{int;n+1-alpha_f}
    //        - F_{ext;n+1-alpha_f}
    fres_->update(-1.0, *fextm_, 0.0);
    fres_->update(1.0, *fintm_, 1.0);
    fres_->update(1.0, *finertm_, 1.0);
    if (damping_ == Inpar::Solid::damp_rayleigh)
    {
      fres_->update(1.0, *fviscm_, 1.0);
    }
  }
  else /* build residual vector and tangent matrix if a multiplicative Gen-Alpha
          scheme for rotations is applied */
  {
    FOUR_C_THROW("Not implemented, yet.");
  }

  // calculate RHS without local condensations (for NewtonLs)
  if (fresn_str_ != nullptr)
  {
    // total internal mid-forces F_{int;n+1-alpha_f} ----> TR-like
    // F_{int;n+1-alpha_f} := (1.-alphaf) * F_{int;n+1} + alpha_f * F_{int;n}
    fresn_str_->update(1., *fintn_str_, 0.);
    fresn_str_->update(alphaf_, *fint_str_, 1. - alphaf_);
    fresn_str_->update(-1.0, *fextm_, 1.0);
    fresn_str_->update(1.0, *finertm_, 1.0);
    if (damping_ == Inpar::Solid::damp_rayleigh) fresn_str_->update(1.0, *fviscm_, 1.0);

    Core::LinAlg::apply_dirichlet_to_system(*fresn_str_, *zeros_, *(dbcmaps_->cond_map()));
  }
}

/*----------------------------------------------------------------------*/
/* evaluate mid-state vectors by averaging end-point vectors */
void Solid::TimIntGenAlpha::evaluate_mid_state()
{
  // mid-displacements D_{n+1-alpha_f} (dism)
  //    D_{n+1-alpha_f} := (1.-alphaf) * D_{n+1} + alpha_f * D_{n}
  dism_->update(1. - alphaf_, *disn_, alphaf_, (*dis_)[0], 0.0);

  // mid-velocities V_{n+1-alpha_f} (velm)
  //    V_{n+1-alpha_f} := (1.-alphaf) * V_{n+1} + alpha_f * V_{n}
  velm_->update(1. - alphaf_, *veln_, alphaf_, (*vel_)[0], 0.0);

  // mid-accelerations A_{n+1-alpha_m} (accm)
  //    A_{n+1-alpha_m} := (1.-alpha_m) * A_{n+1} + alpha_m * A_{n}
  accm_->update(1. - alpham_, *accn_, alpham_, (*acc_)[0], 0.0);
}

/*----------------------------------------------------------------------*/
/* calculate characteristic/reference norms for forces
 * originally by lw */
double Solid::TimIntGenAlpha::calc_ref_norm_force()
{
  // The reference norms are used to scale the calculated iterative
  // displacement norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  // norm of the internal forces
  double fintnorm = 0.0;
  fintnorm = Solid::calculate_vector_norm(iternorm_, *fintm_);

  // norm of the external forces
  double fextnorm = 0.0;
  fextnorm = Solid::calculate_vector_norm(iternorm_, *fextm_);

  // norm of the inertial forces
  double finertnorm = 0.0;
  finertnorm = Solid::calculate_vector_norm(iternorm_, *finertm_);

  // norm of viscous forces
  double fviscnorm = 0.0;
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    fviscnorm = Solid::calculate_vector_norm(iternorm_, *fviscm_);
  }

  // norm of reaction forces
  double freactnorm = 0.0;
  freactnorm = Solid::calculate_vector_norm(iternorm_, *freact_);

  // determine worst value ==> characteristic norm
  return std::max(
      fviscnorm, std::max(finertnorm, std::max(fintnorm, std::max(fextnorm, freactnorm))));
}

/*----------------------------------------------------------------------*/
void Solid::TimIntGenAlpha::update_iter_incrementally()
{
  // step size \f$\Delta t_{n}\f$
  const double dt = (*dt_)[0];

  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->update(1.0, *disn_, -1.0, (*dis_)[0], 0.0);
  veln_->update((beta_ - gamma_) / beta_, (*vel_)[0], (2.0 * beta_ - gamma_) * dt / (2.0 * beta_),
      (*acc_)[0], gamma_ / (beta_ * dt));

  // new end-point accelerations
  accn_->update(1.0, *disn_, -1.0, (*dis_)[0], 0.0);
  accn_->update(-1.0 / (beta_ * dt), (*vel_)[0], (2.0 * beta_ - 1.0) / (2.0 * beta_), (*acc_)[0],
      1.0 / (beta_ * dt * dt));
}

/*----------------------------------------------------------------------*/
/* iterative iteration update of state */
void Solid::TimIntGenAlpha::update_iter_iteratively()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->update(gamma_ / (beta_ * (*dt_)[0]), *disi_, 1.0);

  // new end-point accelerations
  accn_->update(1.0 / (beta_ * (*dt_)[0] * (*dt_)[0]), *disi_, 1.0);
}

/*----------------------------------------------------------------------*/
/* update after time step */
void Solid::TimIntGenAlpha::update_step_state()
{
  // velocity update for contact
  // (must be called BEFORE the following update steps)
  update_step_contact_vum();

  // update all old state at t_{n-1} etc
  // important for step size adaptivity
  // new displacements at t_{n+1} -> t_n
  //    D_{n} := D_{n+1}, etc
  dis_->update_steps(*disn_);

  // new velocities at t_{n+1} -> t_n
  //    V_{n} := V_{n+1}, etc
  vel_->update_steps(*veln_);
  // new accelerations at t_{n+1} -> t_n
  //    A_{n} := A_{n+1}, etc
  acc_->update_steps(*accn_);

  // update new external force
  //    F_{ext;n} := F_{ext;n+1}
  fext_->update(1.0, *fextn_, 0.0);

  // update new internal force
  //    F_{int;n} := F_{int;n+1}
  fint_->update(1.0, *fintn_, 0.0);

  // update new inertial force
  //    F_{inert;n} := F_{inert;n+1}
  finert_->update(1.0, *finertn_, 0.0);

  // update residual force vector for NewtonLS
  if (fresn_str_ != nullptr) fint_str_->update(1., *fintn_str_, 0.);

  // update constraints
  update_step_constraint();

  // update Cardiovascular0D
  update_step_cardiovascular0_d();

  // update constraints
  update_step_spring_dashpot();

  // update contact / meshtying
  update_step_contact_meshtying();
}

/*----------------------------------------------------------------------*/
/* update after time step after output on element level*/
// update anything that needs to be updated at the element level
void Solid::TimIntGenAlpha::update_step_element()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // other parameters that might be needed by the elements
  p.set("total time", timen_);
  p.set("delta time", (*dt_)[0]);
  // action for elements
  p.set("action", "calc_struct_update_istep");

  // go to elements
  discret_->clear_state();
  discret_->set_state("displacement", *(*dis_)(0));

  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    discret_->evaluate(p, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
  else
  {
    /* In the NonlinearMass-case its possible to make an update of
     * displacements, velocities and accelerations at the end of time step
     * (currently only necessary for Kirchhoff beams). An corresponding update
     * rule has to be implemented in the element, otherwise displacements,
     * velocities and accelerations remain unchanged.
     */
    discret_->set_state("velocity", *(*vel_)(0));
    discret_->set_state("acceleration", *(*acc_)(0));

    std::shared_ptr<Core::LinAlg::Vector<double>> update_disp;
    update_disp = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

    std::shared_ptr<Core::LinAlg::Vector<double>> update_vel;
    update_vel = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

    std::shared_ptr<Core::LinAlg::Vector<double>> update_acc;
    update_acc = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);


    discret_->evaluate(p, nullptr, nullptr, update_disp, update_vel, update_acc);

    disn_->update(1.0, *update_disp, 1.0);
    (*dis_)(0)->update(1.0, *update_disp, 1.0);
    veln_->update(1.0, *update_vel, 1.0);
    (*vel_)(0)->update(1.0, *update_vel, 1.0);
    accn_->update(1.0, *update_acc, 1.0);
    (*acc_)(0)->update(1.0, *update_acc, 1.0);
  }

  discret_->clear_state();
}

/*----------------------------------------------------------------------*/
/* read and/or calculate forces for restart */
void Solid::TimIntGenAlpha::read_restart_force()
{
  Core::IO::DiscretizationReader reader(
      *discret_, Global::Problem::instance()->input_control_file(), step_);
  reader.read_vector(fext_, "fexternal");
  reader.read_vector(fint_, "fint");
  reader.read_vector(finert_, "finert");
}

/*----------------------------------------------------------------------*/
/* write internal and external forces for restart */
void Solid::TimIntGenAlpha::write_restart_force(
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
{
  output->write_vector("fexternal", fext_);
  output->write_vector("fint", fint_);
  output->write_vector("finert", finert_);
}

/*-----------------------------------------------------------------------------*
 * Build total residual vector and effective tangential stiffness    meier 05/14
 * matrix in case of nonlinear, rotational inertia effects
 *----------------------------------------------------------------------------*/
void Solid::TimIntGenAlpha::build_res_stiff_nl_mass_rot(Core::LinAlg::Vector<double>& fres_,
    Core::LinAlg::Vector<double>& fextn_, Core::LinAlg::Vector<double>& fintn_,
    Core::LinAlg::Vector<double>& finertn_, Core::LinAlg::SparseOperator& stiff_,
    Core::LinAlg::SparseOperator& mass_)
{
  /* build residual
   *    Res = F_{inert;n+1}
   *        + F_{int;n+1}
   *        - F_{ext;n+1}
   * Remark: In the case of an multiplicative Gen-Alpha time integration scheme,
   * all forces are evaluated at the end point n+1.
   */
  fres_.update(-1.0, fextn_, 0.0);
  fres_.update(1.0, fintn_, 1.0);
  fres_.update(1.0, finertn_, 1.0);

  /* build tangent matrix : effective dynamic stiffness matrix
   *    K_{Teffdyn} = M
   *                + K_{T}
   * Remark: So far, all time integration pre-factors (only necessary for the
   * mass matrix since internal forces are evaluated at n+1) are already
   * considered at element level (see, e.g., beam3r_evaluate.cpp). Therefore,
   * we don't have to apply them here.
   */
  stiff_.add(mass_, false, 1.0, 1.0);
}

/*-----------------------------------------------------------------------------*
 * Check, if there are solely beam elements in the whole             meier 05/14
 * discretization
 *----------------------------------------------------------------------------*/
bool Solid::TimIntGenAlpha::solely_beam3_elements(Core::FE::Discretization& actdis)
{
  bool solelybeameles = true;

  for (int i = 0; i < actdis.num_my_row_elements(); i++)
  {
    Core::Elements::Element* element = actdis.l_col_element(i);
    Core::Nodes::Node* node = (element->nodes())[0];
    int numdof = actdis.num_dof(node);

    // So far we simply check, if we have at least 6 DoFs per node, which is only true for beam
    // elements
    if (numdof < 6) solelybeameles = false;
  }

  return solelybeameles;
}

FOUR_C_NAMESPACE_CLOSE
