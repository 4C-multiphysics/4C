// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_timint_ost.hpp"

#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
void Solid::TimIntOneStepTheta::verify_coeff()
{
  // check value of theta
  if ((theta_ <= 0.0) or (theta_ > 1.0)) FOUR_C_THROW("theta out of range (0.0,1.0]");

  // done
  return;
}

/*======================================================================*/
/* constructor */
Solid::TimIntOneStepTheta::TimIntOneStepTheta(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams, std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver,
    std::shared_ptr<Core::LinAlg::Solver> contactsolver,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : TimIntImpl(timeparams, ioparams, sdynparams, xparams, actdis, solver, contactsolver, output),
      theta_(sdynparams.sublist("ONESTEPTHETA").get<double>("THETA")),
      dist_(nullptr),
      velt_(nullptr),
      acct_(nullptr),
      fint_(nullptr),
      fintn_(nullptr),
      fext_(nullptr),
      fextn_(nullptr),
      finert_(nullptr),
      finertt_(nullptr),
      finertn_(nullptr),
      fvisct_(nullptr)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during setup() in a base class.
  return;
}

/*----------------------------------------------------------------------------------------------*
 * Initialize this class                                                            rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntOneStepTheta::init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    std::shared_ptr<Core::FE::Discretization> actdis, std::shared_ptr<Core::LinAlg::Solver> solver)
{
  // call init() in base class
  Solid::TimIntImpl::init(timeparams, sdynparams, xparams, actdis, solver);

  // general variable verifications:
  // info to user about current time integration scheme and its parametrization
  if (myrank_ == 0)
  {
    verify_coeff();
    Core::IO::cout << "with one-step-theta" << Core::IO::endl
                   << "   theta = " << theta_ << Core::IO::endl
                   << Core::IO::endl;
  }

  // have a nice day
  return;
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void Solid::TimIntOneStepTheta::setup()
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
    // the case of nonlinear inertia terms works so far only for examples with vanishing initial
    // accelerations, i.e. the initial external forces and initial velocities have to be chosen
    // consistently!!!
    (*acc_)(0)->put_scalar(0.0);
  }

  // create state vectors

  // mid-displacements
  dist_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // mid-velocities
  velt_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // mid-accelerations
  acct_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // create force vectors

  // internal force vector F_{int;n} at last time
  fint_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // internal force vector F_{int;n+1} at new time
  fintn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // external force vector F_ext at last times
  fext_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // external force vector F_{n+1} at new time
  fextn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // set initial external force vector
  apply_force_external((*time_)[0], (*dis_)(0), disn_, *(*vel_)(0), *fext_);

  // inertial force vector F_{int;n} at last time
  finert_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // inertial mid-force vector F_{int;n+1-alpha_f}
  finertt_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // inertial force vector F_{int;n+1} at new time
  finertn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);
  // viscous mid-point force vector F_visc
  fvisct_ = std::make_shared<Core::LinAlg::Vector<double>>(*dof_row_map_view(), true);

  // create parameter list
  Teuchos::ParameterList params;

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
    double timeintfac_dis = theta_ * theta_ * (*dt_)[0] * (*dt_)[0];
    double timeintfac_vel = theta_ * (*dt_)[0];

    // Check, if initial residuum really vanishes for acc_ = 0
    apply_force_stiff_internal_and_inertial((*time_)[0], (*dt_)[0], timeintfac_dis, timeintfac_vel,
        (*dis_)(0), zeros_, (*vel_)(0), (*acc_)(0), fint_, finert_, stiff_, mass_, params);

    nonlinear_mass_sanity_check(fext_, (*dis_)(0), (*vel_)(0), (*acc_)(0), &sdynparams_);
  }

  return;
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant displacements
 * and consistent velocities and displacements */
void Solid::TimIntOneStepTheta::predict_const_dis_consist_vel_acc()
{
  // time step size
  const double dt = (*dt_)[0];

  // constant predictor : displacement in domain
  disn_->update(1.0, *(*dis_)(0), 0.0);

  // new end-point velocities
  veln_->update(1.0 / (theta_ * dt), *disn_, -1.0 / (theta_ * dt), *(*dis_)(0), 0.0);
  veln_->update(-(1.0 - theta_) / theta_, *(*vel_)(0), 1.0);

  // new end-point accelerations
  accn_->update(1.0 / (theta_ * theta_ * dt * dt), *disn_, -1.0 / (theta_ * theta_ * dt * dt),
      *(*dis_)(0), 0.0);
  accn_->update(
      -1.0 / (theta_ * theta_ * dt), *(*vel_)(0), -(1.0 - theta_) / theta_, *(*acc_)(0), 1.0);

  // reset the residual displacement
  disi_->put_scalar(0.0);

  // watch out
  return;
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant velocities,
 * extrapolated displacements and consistent accelerations */
void Solid::TimIntOneStepTheta::predict_const_vel_consist_acc()
{
  // time step size
  const double dt = (*dt_)[0];

  // extrapolated displacements based upon constant velocities
  // d_{n+1} = d_{n} + dt * v_{n}
  disn_->update(1.0, (*dis_)[0], dt, (*vel_)[0], 0.0);

  // new end-point velocities
  veln_->update(1.0 / (theta_ * dt), *disn_, -1.0 / (theta_ * dt), *(*dis_)(0), 0.0);
  veln_->update(-(1.0 - theta_) / theta_, *(*vel_)(0), 1.0);

  // new end-point accelerations
  accn_->update(1.0 / (theta_ * theta_ * dt * dt), *disn_, -1.0 / (theta_ * theta_ * dt * dt),
      *(*dis_)(0), 0.0);
  accn_->update(
      -1.0 / (theta_ * theta_ * dt), *(*vel_)(0), -(1.0 - theta_) / theta_, *(*acc_)(0), 1.0);

  // reset the residual displacement
  disi_->put_scalar(0.0);

  // That's it!
  return;
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant accelerations
 * and extrapolated velocities and displacements */
void Solid::TimIntOneStepTheta::predict_const_acc()
{
  // time step size
  const double dt = (*dt_)[0];

  // extrapolated displacements based upon constant accelerations
  // d_{n+1} = d_{n} + dt * v_{n} + dt^2 / 2 * a_{n}
  disn_->update(1.0, (*dis_)[0], dt, (*vel_)[0], 0.0);
  disn_->update(dt * dt / 2., (*acc_)[0], 1.0);

  // new end-point velocities
  veln_->update(1.0 / (theta_ * dt), *disn_, -1.0 / (theta_ * dt), *(*dis_)(0), 0.0);
  veln_->update(-(1.0 - theta_) / theta_, *(*vel_)(0), 1.0);

  // new end-point accelerations
  accn_->update(1.0 / (theta_ * theta_ * dt * dt), *disn_, -1.0 / (theta_ * theta_ * dt * dt),
      *(*dis_)(0), 0.0);
  accn_->update(
      -1.0 / (theta_ * theta_ * dt), *(*vel_)(0), -(1.0 - theta_) / theta_, *(*acc_)(0), 1.0);

  // reset the residual displacement
  disi_->put_scalar(0.0);

  // That's it!
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate residual force and its stiffness, ie derivative
 * with respect to end-point displacements \f$D_{n+1}\f$ */
void Solid::TimIntOneStepTheta::evaluate_force_stiff_residual(Teuchos::ParameterList& params)
{
  // get info about prediction step from parameter list
  bool predict = false;
  if (params.isParameter("predict")) predict = params.get<bool>("predict");

  // initialise stiffness matrix to zero
  stiff_->zero();
  if (damping_ == Inpar::Solid::damp_material) damp_->zero();

  // theta-interpolate state vectors
  evaluate_mid_state();

  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->put_scalar(0.0);
  apply_force_stiff_external(timen_, (*dis_)(0), *disn_, *(*vel_)(0), *fextn_, stiff_);

  // additional external forces are added (e.g. interface forces)
  fextn_->update(1.0, *fifc_, 1.0);

  // ************************** (2) INTERNAL FORCES ***************************

  // initialise internal forces
  fintn_->put_scalar(0.0);

  // build new internal forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // ordinary internal force and stiffness
    apply_force_stiff_internal(
        timen_, (*dt_)[0], disn_, disi_, *veln_, fintn_, stiff_, params, damp_);
  }
  else
  {
    // Remark: Since our element evaluate routine is only designed for two input matrices
    //(stiffness and damping or stiffness and mass) its not possible, to have nonlinear
    // inertia forces AND material damping. Therefore this case is already captured in
    // strtimint.cpp.

    // If we have nonlinear inertia forces, the corresponding contributions are computed together
    // with the internal forces
    finertn_->put_scalar(0.0);
    mass_->zero();

    // In general the nonlinear inertia force can depend on displacements, velocities and
    // accelerations, i.e     finertn_=finertn_(disn_, veln_, accn_):
    //
    //    LIN finertn_ = [ d(finertn_)/d(disn_) + 1/(theta_*dt_)*d(finertn_)/d(veln_)
    //                 + 1/(theta_*theta_*dt_*dt_)*d(finertn_)/d(accn_) ]*disi_
    //
    //    LIN finertt_ = 1/(theta_*dt_^2)[ (theta_^2*dt_^2)*d(finertn_)/d(disn_)
    //                 + (theta_*dt_)*d(finertn_)/d(veln_) + d(finertn_)/d(accn_)]*disi_
    //
    // While the factor 1/(theta_*dt_^2) is applied later on in strtimint_ost.cpp the
    // factors timintfac_dis=(theta_^2*dt_^2) and timeintfac_vel=(theta_*dt_) have directly to be
    // applied on element level before the three contributions of the linearization are summed up in
    // mass_.

    double timintfac_dis = theta_ * theta_ * (*dt_)[0] * (*dt_)[0];
    double timintfac_vel = theta_ * (*dt_)[0];
    apply_force_stiff_internal_and_inertial(timen_, (*dt_)[0], timintfac_dis, timintfac_vel, disn_,
        disi_, veln_, accn_, fintn_, finertn_, stiff_, mass_, params);
  }

  // add forces and stiffness due to spring dashpot condition
  Teuchos::ParameterList psprdash;
  psprdash.set("time_fac", 1. / (theta_ * (*dt_)[0]));
  psprdash.set("dt", (*dt_)[0]);  // needed only for cursurfnormal option!!
  apply_force_stiff_spring_dashpot(stiff_, fintn_, disn_, *veln_, predict, psprdash);

  // apply forces and stiffness due to constraints
  Teuchos::ParameterList pcon;
  // constraint matrix has to be scaled with the same value fintn_ is scaled with
  pcon.set("scaleConstrMat", theta_);
  apply_force_stiff_constraint(timen_, (*dis_)(0), disn_, fintn_, stiff_, pcon);

  // add forces and stiffness due to 0D cardiovascular coupling conditions
  Teuchos::ParameterList pwindk;
  pwindk.set("scale_timint", theta_);
  pwindk.set("time_step_size", (*dt_)[0]);
  apply_force_stiff_cardiovascular0_d(timen_, disn_, fintn_, stiff_, pwindk);

  // ************************** (3) INERTIAL FORCES ***************************

  // build new internal forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // inertial forces #finertt_
    mass_->multiply(false, *acct_, *finertt_);
  }
  else
  {
    determine_mass();
  }

  // ************************** (4) DAMPING FORCES ****************************

  // viscous forces due Rayleigh damping
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    damp_->multiply(false, *velt_, *fvisct_);
  }

  // ******************** Finally, put everything together ********************

  // build residual  Res = M . A_{n+theta}
  //                     + C . V_{n+theta}
  //                     + F_{int;n+theta}
  //                     - F_{ext;n+theta}
  fres_->update(-theta_, *fextn_, -(1.0 - theta_), *fext_, 0.0);
  fres_->update(theta_, *fintn_, (1.0 - theta_), *fint_, 1.0);
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    fres_->update(1.0, *fvisct_, 1.0);
  }
  fres_->update(1.0, *finertt_, 1.0);

  // std::cout << Solid::calculate_vector_norm(vectornorm_l2, fextn_) << std::endl;

  // build tangent matrix : effective dynamic stiffness matrix
  //    K_{Teffdyn} = 1/(theta*dt^2) M
  //                + 1/dt C
  //                + theta K_{T}
  stiff_->add(*mass_, false, 1.0 / (theta_ * (*dt_)[0] * (*dt_)[0]), theta_);
  if (damping_ != Inpar::Solid::damp_none)
  {
    if (damping_ == Inpar::Solid::damp_material) damp_->complete();
    stiff_->add(*damp_, false, 1.0 / (*dt_)[0], 1.0);
  }

  // apply forces and stiffness due to contact / meshtying
  // Note that we ALWAYS use a TR-like approach to compute the interface
  // forces. This means we never explicitly compute fc at the generalized
  // mid-point n+theta, but use a linear combination of the old end-
  // point n and the new end-point n+1 instead:
  // F_{c;n+theta} := theta * F_{c;n+1} +  (1-theta) * F_{c;n}
  apply_force_stiff_contact_meshtying(stiff_, fres_, disn_, predict);

  // close stiffness matrix
  stiff_->complete();

  return;
}

/*----------------------------------------------------------------------*/
/* Evaluate/define the residual force vector #fres_ for
 * relaxation solution with solve_relaxation_linear */
void Solid::TimIntOneStepTheta::evaluate_force_stiff_residual_relax(Teuchos::ParameterList& params)
{
  // compute residual forces #fres_ and stiffness #stiff_
  evaluate_force_stiff_residual(params);

  // overwrite the residual forces #fres_ with interface load
  fres_->update(-theta_, *fifc_, 0.0);
}

/*----------------------------------------------------------------------*/
/* Evaluate residual */
void Solid::TimIntOneStepTheta::evaluate_force_residual()
{
  // theta-interpolate state vectors
  evaluate_mid_state();

  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->put_scalar(0.0);
  apply_force_external(timen_, (*dis_)(0), disn_, *(*vel_)(0), *fextn_);

  // additional external forces are added (e.g. interface forces)
  fextn_->update(1.0, *fifc_, 1.0);

  // ************************** (2) INTERNAL FORCES ***************************

  // initialise internal forces
  fintn_->put_scalar(0.0);

  // build new internal forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // ordinary internal force and stiffness
    apply_force_internal(timen_, (*dt_)[0], disn_, disi_, *veln_, fintn_);
  }
  else
  {
    FOUR_C_THROW("Not implemented, yet.");
  }

  // ************************** (3) INERTIAL FORCES ***************************

  // build new internal forces and stiffness
  if (have_nonlinear_mass() == Inpar::Solid::MassLin::ml_none)
  {
    // inertial forces #finertt_
    mass_->multiply(false, *acct_, *finertt_);
  }
  else
  {
    FOUR_C_THROW("Not implemented, yet.");
  }

  // ************************** (4) DAMPING FORCES ****************************

  // viscous forces due Rayleigh damping
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    damp_->multiply(false, *velt_, *fvisct_);
  }

  // ******************** Finally, put everything together ********************

  // build residual  Res = M . A_{n+theta}
  //                     + C . V_{n+theta}
  //                     + F_{int;n+theta}
  //                     - F_{ext;n+theta}
  fres_->update(-theta_, *fextn_, -(1.0 - theta_), *fext_, 0.0);
  fres_->update(theta_, *fintn_, (1.0 - theta_), *fint_, 1.0);
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    fres_->update(1.0, *fvisct_, 1.0);
  }
  fres_->update(1.0, *finertt_, 1.0);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Solid::TimIntOneStepTheta::determine_mass()
{
  // F_{inert;1+theta} := theta * F_{inert;n+1} + (1-theta) * F_{inert;n}
  finertt_->update(theta_, *finertn_, (1.0 - theta_), *finert_, 0.0);
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate theta-state vectors by averaging end-point vectors */
void Solid::TimIntOneStepTheta::evaluate_mid_state()
{
  // mid-displacements D_{n+1-alpha_f} (dism)
  //    D_{n+theta} := theta * D_{n+1} + (1-theta) * D_{n}
  dist_->update(theta_, *disn_, 1.0 - theta_, *(*dis_)(0), 0.0);

  // mid-velocities V_{n+1-alpha_f} (velm)
  //    V_{n+theta} := theta * V_{n+1} + (1-theta) * V_{n}
  velt_->update(theta_, *veln_, 1.0 - theta_, *(*vel_)(0), 0.0);

  // mid-accelerations A_{n+1-alpha_m} (accm)
  //    A_{n+theta} := theta * A_{n+1} + (1-theta) * A_{n}
  acct_->update(theta_, *accn_, 1.0 - theta_, *(*acc_)(0), 0.0);

  // jump
  return;
}

/*----------------------------------------------------------------------*/
/* calculate characteristic/reference norms for forces
 * originally by lw */
double Solid::TimIntOneStepTheta::calc_ref_norm_force()
{
  // The reference norms are used to scale the calculated iterative
  // displacement norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  // norm of the internal forces
  double fintnorm = 0.0;
  fintnorm = Solid::calculate_vector_norm(iternorm_, *fintn_);

  // norm of the external forces
  double fextnorm = 0.0;
  fextnorm = Solid::calculate_vector_norm(iternorm_, *fextn_);

  // norm of the inertial forces
  double finertnorm = 0.0;
  finertnorm = Solid::calculate_vector_norm(iternorm_, *finertt_);

  // norm of viscous forces
  double fviscnorm = 0.0;
  if (damping_ == Inpar::Solid::damp_rayleigh)
  {
    fviscnorm = Solid::calculate_vector_norm(iternorm_, *fvisct_);
  }

  // norm of reaction forces
  double freactnorm = 0.0;
  freactnorm = Solid::calculate_vector_norm(iternorm_, *freact_);

  // return char norm
  return std::max(
      fviscnorm, std::max(finertnorm, std::max(fintnorm, std::max(fextnorm, freactnorm))));
}

/*----------------------------------------------------------------------*/
void Solid::TimIntOneStepTheta::update_iter_incrementally()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->update(1.0 / (theta_ * (*dt_)[0]), *disn_, -1.0 / (theta_ * (*dt_)[0]), *(*dis_)(0), 0.0);
  veln_->update(-(1.0 - theta_) / theta_, *(*vel_)(0), 1.0);

  // new end-point accelerations
  accn_->update(1.0 / (theta_ * theta_ * (*dt_)[0] * (*dt_)[0]), *disn_,
      -1.0 / (theta_ * theta_ * (*dt_)[0] * (*dt_)[0]), *(*dis_)(0), 0.0);
  accn_->update(-1.0 / (theta_ * theta_ * (*dt_)[0]), *(*vel_)(0), -(1.0 - theta_) / theta_,
      *(*acc_)(0), 1.0);
}

/*----------------------------------------------------------------------*/
/* iterative iteration update of state */
void Solid::TimIntOneStepTheta::update_iter_iteratively()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->update(1.0 / (theta_ * (*dt_)[0]), *disi_, 1.0);

  // new end-point accelerations
  accn_->update(1.0 / ((*dt_)[0] * (*dt_)[0] * theta_ * theta_), *disi_, 1.0);

  // bye
  return;
}

/*----------------------------------------------------------------------*/
/* update after time step */
void Solid::TimIntOneStepTheta::update_step_state()
{
  // update state
  // new displacements at t_{n+1} -> t_n
  //    D_{n} := D_{n+1}
  dis_->update_steps(*disn_);
  // new velocities at t_{n+1} -> t_n
  //    V_{n} := V_{n+1}
  vel_->update_steps(*veln_);
  // new accelerations at t_{n+1} -> t_n
  //    A_{n} := A_{n+1}
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

  // update constraints
  update_step_constraint();

  // update constraints
  update_step_cardiovascular0_d();

  // update constraints
  update_step_spring_dashpot();

  // update contact / meshtying
  update_step_contact_meshtying();

  // look out
  return;
}

/*----------------------------------------------------------------------*/
/* update after time step after output on element level*/
// update anything that needs to be updated at the element level
void Solid::TimIntOneStepTheta::update_step_element()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // other parameters that might be needed by the elements
  p.set("total time", timen_);
  p.set("delta time", (*dt_)[0]);
  // p.set("alpha f", theta_);
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
    // In the NonlinearMass-case its possible to make an update of displacements, velocities
    // and accelerations at the end of time step (currently only necessary for Kirchhoff beams)
    // An corresponding update rule has to be implemented in the element, otherwise
    // displacements, velocities and accelerations remain unchange.
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
/* read restart forces */
void Solid::TimIntOneStepTheta::read_restart_force()
{
  Core::IO::DiscretizationReader reader(
      *discret_, Global::Problem::instance()->input_control_file(), step_);
  reader.read_vector(fext_, "fexternal");
  reader.read_vector(fint_, "fint");
  reader.read_vector(finert_, "finert");

  return;
}

/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*/
/* write internal and external forces for restart */
void Solid::TimIntOneStepTheta::write_restart_force(
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
{
  output->write_vector("fexternal", fext_);
  output->write_vector("fint", fint_);
  output->write_vector("finert", finert_);
  return;
}

FOUR_C_NAMESPACE_CLOSE
