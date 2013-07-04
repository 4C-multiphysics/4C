 /*----------------------------------------------------------------------*/
/*!
\file strtimint_genalpha.cpp
\brief Structural time integration with generalised-alpha

<pre>
Maintainer: Alexander Popp
            popp@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */

/*----------------------------------------------------------------------*/
/* headers */
#include "strtimint_genalpha.H"
#include "stru_aux.H"
#include "../drt_io/io.H"
#include "../drt_lib/drt_locsys.H"
#include "../linalg/linalg_utils.H"
#include "../drt_io/io_pstream.H"

/*----------------------------------------------------------------------*/
void STR::TimIntGenAlpha::VerifyCoeff()
{
  // beta
  if ( (beta_ <= 0.0) or (beta_ > 0.5) )
    dserror("beta out of range (0.0,0.5]");
  else
    std::cout << "   beta = " << beta_ << std::endl;
  // gamma
  if ( (gamma_ <= 0.0) or (gamma_ > 1.0) )
    dserror("gamma out of range (0.0,1.0]");
  else
    std::cout << "   gamma = " << gamma_ << std::endl;
  // alpha_f
  if ( (alphaf_ < 0.0) or (alphaf_ >= 1.0) )
    dserror("alpha_f out of range [0.0,1.0)");
  else
    std::cout << "   alpha_f = " << alphaf_ << std::endl;
  // alpha_m
  if ( (alpham_ < 0.0) or (alpham_ >= 1.0) )
    dserror("alpha_m out of range [0.0,1.0)");
  else
    std::cout << "   alpha_m = " << alpham_ << std::endl;

  // mid-averaging type
  // In principle, there exist two mid-averaging possibilities, TR-like and IMR-like,
  // where TR-like means trapezoidal rule and IMR-like means implicit mid-point rule.
  // We used to maintain implementations of both variants, but due to its significantly
  // higher complexity, the IMR-like version has been deleted (popp 02/2013). The nice
  // thing about TR-like mid-averaging is that all element (and thus also material) calls
  // are exclusively(!) carried out at the end-point t_{n+1} of each time interval, but
  // never explicitly at some generalized midpoint, such as t_{n+1-\alpha_f}. Thus, any
  // cumbersome extrapolation of history variables, etc. becomes obsolete.
  if (midavg_ != INPAR::STR::midavg_trlike)
    dserror("mid-averaging of internal forces only implemented TR-like");
  else
    std::cout << "   midavg = " << INPAR::STR::MidAverageString(midavg_)<<std::endl;

  // done
  return;
}

/*----------------------------------------------------------------------*/
/* constructor */
STR::TimIntGenAlpha::TimIntGenAlpha
(
  const Teuchos::ParameterList& ioparams,
  const Teuchos::ParameterList& sdynparams,
  const Teuchos::ParameterList& xparams,
  Teuchos::RCP<DRT::Discretization> actdis,
  Teuchos::RCP<LINALG::Solver> solver,
  Teuchos::RCP<LINALG::Solver> contactsolver,
  Teuchos::RCP<IO::DiscretizationWriter> output
)
: TimIntImpl
  (
    ioparams,
    sdynparams,
    xparams,
    actdis,
    solver,
    contactsolver,
    output
  ),
  midavg_(DRT::INPUT::IntegralValue<INPAR::STR::MidAverageEnum>(sdynparams.sublist("GENALPHA"),"GENAVG")),
  /* iterupditer_(false), */
  beta_(sdynparams.sublist("GENALPHA").get<double>("BETA")),
  gamma_(sdynparams.sublist("GENALPHA").get<double>("GAMMA")),
  alphaf_(sdynparams.sublist("GENALPHA").get<double>("ALPHA_F")),
  alpham_(sdynparams.sublist("GENALPHA").get<double>("ALPHA_M")),
  dism_(Teuchos::null),
  velm_(Teuchos::null),
  accm_(Teuchos::null),
  fint_(Teuchos::null),
  fintm_(Teuchos::null),
  fintn_(Teuchos::null),
  fext_(Teuchos::null),
  fextm_(Teuchos::null),
  fextn_(Teuchos::null),
  finert_(Teuchos::null),
  finertm_(Teuchos::null),
  finertn_(Teuchos::null),
  fviscm_(Teuchos::null)
{
  // info to users
  if (myrank_ == 0)
  {
    IO::cout << "with generalised-alpha" << IO::endl;
    VerifyCoeff();

    std::cout << "   p_dis = " << MethodOrderOfAccuracyDis() << std::endl
              << "   p_vel = " << MethodOrderOfAccuracyVel() << std::endl
              << std::endl;
  }

  if (!HaveNonlinearMass())
  {
    // determine mass, damping and initial accelerations
    DetermineMassDampConsistAccel();
  }
  else
  {
    // the case of nonlinear inertia terms works so far only for examples with vanishing initial accelerations, i.e. the initial external
    // forces and initial velocities have to be chosen consistently!!!
    (*acc_)(0)->PutScalar(0.0);
  }

  // create state vectors

  // mid-displacements
  dism_ = LINALG::CreateVector(*dofrowmap_, true);
  // mid-velocities
  velm_ = LINALG::CreateVector(*dofrowmap_, true);
  // mid-accelerations
  accm_ = LINALG::CreateVector(*dofrowmap_, true);

  // create force vectors

  // internal force vector F_{int;n} at last time
  fint_ = LINALG::CreateVector(*dofrowmap_, true);
  // internal mid-force vector F_{int;n+1-alpha_f}
  fintm_ = LINALG::CreateVector(*dofrowmap_, true);
  // internal force vector F_{int;n+1} at new time
  fintn_ = LINALG::CreateVector(*dofrowmap_, true);

  // external force vector F_ext at last times
  fext_ = LINALG::CreateVector(*dofrowmap_, true);
  // external mid-force vector F_{ext;n+1-alpha_f}
  fextm_ = LINALG::CreateVector(*dofrowmap_, true);
  // external force vector F_{n+1} at new time
  fextn_ = LINALG::CreateVector(*dofrowmap_, true);
  // set initial external force vector
  ApplyForceExternal((*time_)[0], (*dis_)(0), disn_, (*vel_)(0), fext_, stiff_);

  // inertial force vector F_{int;n} at last time
  finert_ = LINALG::CreateVector(*dofrowmap_, true);
  // inertial mid-force vector F_{int;n+1-alpha_f}
  finertm_ = LINALG::CreateVector(*dofrowmap_, true);
  // inertial force vector F_{int;n+1} at new time
  finertn_ = LINALG::CreateVector(*dofrowmap_, true);

  // viscous mid-point force vector F_visc
  fviscm_ = LINALG::CreateVector(*dofrowmap_, true);

  if (!HaveNonlinearMass())
  {
    // set initial internal force vector
    ApplyForceStiffInternal((*time_)[0], (*dt_)[0], (*dis_)(0), zeros_, (*vel_)(0), fint_, stiff_);
  }
  else
  {
    double timeintfac_dis=beta_*(*dt_)[0]*(*dt_)[0];
    double timeintfac_vel=gamma_*(*dt_)[0];

    // Check, if initial residuum really vanishes for acc_ = 0
    ApplyForceStiffInternalAndInertial((*time_)[0], (*dt_)[0], timeintfac_dis, timeintfac_vel, (*dis_)(0), zeros_, (*vel_)(0), (*acc_)(0), fint_, finert_, stiff_, mass_);

    NonlinearMassSanityCheck(fext_, (*dis_)(0), (*vel_)(0), (*acc_)(0));
  }

  // have a nice day
  return;
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant displacements
 * and consistent velocities and displacements */
void STR::TimIntGenAlpha::PredictConstDisConsistVelAcc()
{
  // constant predictor : displacement in domain
  disn_->Update(1.0, *(*dis_)(0), 0.0);

  // consistent velocities following Newmark formulas
  veln_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  veln_->Update((beta_-gamma_)/beta_, *(*vel_)(0),
                (2.*beta_-gamma_)*(*dt_)[0]/(2.*beta_), *(*acc_)(0),
                gamma_/(beta_*(*dt_)[0]));

  // consistent accelerations following Newmark formulas
  accn_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  accn_->Update(-1./(beta_*(*dt_)[0]), *(*vel_)(0),
                (2.*beta_-1.)/(2.*beta_), *(*acc_)(0),
                1./(beta_*(*dt_)[0]*(*dt_)[0]));

  // watch out
  return;
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant velocities,
 * extrapolated displacements and consistent accelerations */
void STR::TimIntGenAlpha::PredictConstVelConsistAcc()
{
  // extrapolated displacements based upon constant velocities
  // d_{n+1} = d_{n} + dt * v_{n}
  disn_->Update(1.0, (*dis_)[0], (*dt_)[0], (*vel_)[0], 0.0);

  // consistent velocities following Newmark formulas
  veln_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  veln_->Update((beta_-gamma_)/beta_, *(*vel_)(0),
                (2.*beta_-gamma_)*(*dt_)[0]/(2.*beta_), *(*acc_)(0),
                gamma_/(beta_*(*dt_)[0]));

  // consistent accelerations following Newmark formulas
  accn_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  accn_->Update(-1./(beta_*(*dt_)[0]), *(*vel_)(0),
                (2.*beta_-1.)/(2.*beta_), *(*acc_)(0),
                1./(beta_*(*dt_)[0]*(*dt_)[0]));

  // That's it!
  return;
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant accelerations
 * and extrapolated velocities and displacements */
void STR::TimIntGenAlpha::PredictConstAcc()
{
  // extrapolated displacements based upon constant accelerations
  // d_{n+1} = d_{n} + dt * v_{n} + dt^2 / 2 * a_{n}
  disn_->Update(1.0, (*dis_)[0], (*dt_)[0], (*vel_)[0], 0.0);
  disn_->Update((*dt_)[0] * (*dt_)[0] /2., (*acc_)[0], 1.0);

  // extrapolated velocities (equal to consistent velocities)
  // v_{n+1} = v_{n} + dt * a_{n}
  veln_->Update(1.0, (*vel_)[0], (*dt_)[0], (*acc_)[0],  0.0);

  // constant accelerations (equal to consistent accelerations)
  accn_->Update(1.0, (*acc_)[0], 0.0);

  // That's it!
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate residual force and its stiffness, ie derivative
 * with respect to end-point displacements \f$D_{n+1}\f$ */
void STR::TimIntGenAlpha::EvaluateForceStiffResidual(bool predict)
{
  // initialise stiffness matrix to zero
  stiff_->Zero();

  // build predicted mid-state by last converged state and predicted target state
  EvaluateMidState();

  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->PutScalar(0.0);
  ApplyForceExternal(timen_, (*dis_)(0), disn_, (*vel_)(0), fextn_, stiff_);

  // additional external forces are added (e.g. interface forces)
  fextn_->Update(1.0, *fifc_, 1.0);

  // external mid-forces F_{ext;n+1-alpha_f} ----> TR-like
  // F_{ext;n+1-alpha_f} := (1.-alphaf) * F_{ext;n+1} + alpha_f * F_{ext;n}
  fextm_->Update(1.-alphaf_, *fextn_, alphaf_, *fext_, 0.0);

  // ************************** (2) INTERNAL FORCES ***************************

  fintn_->PutScalar(0.0);
  // build new internal forces and stiffness
  if (!HaveNonlinearMass())
  {
    ApplyForceStiffInternal(timen_, (*dt_)[0], disn_, disi_, veln_, fintn_, stiff_);
  }
  else
  {
    //If we have nonlinear inertia forces, the corresponding contributions are computed together with the internal forces
    finertn_->PutScalar(0.0);
    mass_->Zero();

    // In general the nonlinear inertia force can depend on displacements, velocities and accelerations,
    // i.e     finertn_=finertn_(disn_, veln_, accn_):
    //
    //    LIN finertn_ = [ d(finertn_)/d(disn_) + gamma_/(beta_*dt_)*d(finertn_)/d(veln_)
    //                 + 1/(beta_*dt_*dt_)*d(finertn_)/d(accn_) ]*disi_
    //
    //    LIN finertm_ = (1-alpha_m)/(beta_*dt_*dt_)[ (beta_*dt_*dt_)*d(finertn_)/d(disn_)
    //                 + (gamma_*dt_)*d(finertn_)/d(veln_) + d(finertn_)/d(accn_)]*disi_
    //
    // While the factor (1-alpha_m/(beta_*dt_*dt_) is applied later on in strtimint_genalpha.cpp the
    // factors timintfac_dis=(beta_*dt_*dt_) and timeintfac_vel=(gamma_*dt_) have directly to be applied
    // on element level before the three contributions of the linearization are summed up in mass_.

    double timintfac_dis=beta_*(*dt_)[0]*(*dt_)[0];
    double timintfac_vel=gamma_*(*dt_)[0];
    ApplyForceStiffInternalAndInertial(timen_, (*dt_)[0], timintfac_dis, timintfac_vel, disn_, disi_, veln_, accn_, fintn_, finertn_, stiff_, mass_);
  }

  // add forces and stiffness due to constraints
  // (for TR scale constraint matrix with the same value fintn_ is scaled with)
  ParameterList pcon;
  pcon.set("scaleConstrMat", (1.0-alphaf_));
  ApplyForceStiffConstraint(timen_, (*dis_)(0), disn_, fintn_, stiff_, pcon);

  // add surface stress force
  ApplyForceStiffSurfstress(timen_, (*dt_)[0], disn_, fintn_, stiff_);

  // add potential forces
  ApplyForceStiffPotential(timen_, disn_, fintn_, stiff_);
  TestForceStiffPotential(timen_, disn_, step_);

  // add forces and stiffness due to embedding tissue condition
  ApplyForceStiffEmbedTissue(stiff_,fintn_,disn_,predict);

  // total internal mid-forces F_{int;n+1-alpha_f} ----> TR-like
  // F_{int;n+1-alpha_f} := (1.-alphaf) * F_{int;n+1} + alpha_f * F_{int;n}
  fintm_->Update(1.-alphaf_, *fintn_, alphaf_, *fint_, 0.0);

  // ************************** (3) INERTIAL FORCES ***************************

  // build new internal forces and stiffness
  if (!HaveNonlinearMass())
  {
    // build new internal forces and stiffness
    finertm_->PutScalar(0.0);
    // inertial forces #finertm_
    mass_->Multiply(false, *accm_, *finertm_);
  }
  else
  {
    // total inertial mid-forces F_{inert;n+1-alpha_m} ----> TR-like
    // F_{inert;n+1-alpha_m} := (1.-alpham) * F_{inert;n+1} + alpha_m * F_{inert;n}
    finertm_->Update(1.-alpham_, *finertn_, alpham_, *finert_, 0.0);
  }

  // ************************** (4) DAMPING FORCES ****************************

  // viscous forces due to Rayleigh damping
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    damp_->Multiply(false, *velm_, *fviscm_);
  }

  // build residual
  //    Res = M . A_{n+1-alpha_m}
  //        + C . V_{n+1-alpha_f}
  //        + F_{int;n+1-alpha_f}
  //        - F_{ext;n+1-alpha_f}
  fres_->Update(-1.0, *fextm_, 0.0);
  fres_->Update( 1.0, *fintm_, 1.0);
  fres_->Update( 1.0, *finertm_, 1.0);
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    fres_->Update(1.0, *fviscm_, 1.0);
  }

  // build tangent matrix : effective dynamic stiffness matrix
  //    K_{Teffdyn} = (1 - alpha_m)/(beta*dt^2) M
  //                + (1 - alpha_f)*y/(beta*dt) C
  //                + (1 - alpha_f) K_{T}
  stiff_->Add(*mass_, false, (1.-alpham_)/(beta_*(*dt_)[0]*(*dt_)[0]), 1.-alphaf_);
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    stiff_->Add(*damp_, false, (1.-alphaf_)*gamma_/(beta_*(*dt_)[0]), 1.0);
  }

  // apply forces and stiffness due to contact / meshtying ----> TR-like
  // F_{c;n+1-alpha_f} := (1.-alphaf) * F_{c;n+1} + alpha_f * F_{c;n}
  ApplyForceStiffContactMeshtying(stiff_,fres_,disn_,predict);

  // apply forces and stiffness due to beam contact ----> TR-like
  // F_{c;n+1-alpha_f} := (1.-alphaf) * F_{c;n+1} + alpha_f * F_{c;n}
  ApplyForceStiffBeamContact(stiff_,fres_,disn_,predict);

  // close stiffness matrix
  stiff_->Complete();

  // hallelujah
  return;
}

/*----------------------------------------------------------------------*/
/* Evaluate/define the residual force vector #fres_ for
 * relaxation solution with SolveRelaxationLinear */
void STR::TimIntGenAlpha::EvaluateForceStiffResidualRelax()
{
  // compute residual forces #fres_ and stiffness #stiff_
  EvaluateForceStiffResidual();

  // overwrite the residual forces #fres_ with interface load
  fres_->Update(-1+alphaf_, *fifc_, 0.0);

  // oh gosh
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate mid-state vectors by averaging end-point vectors */
void STR::TimIntGenAlpha::EvaluateMidState()
{
  // mid-displacements D_{n+1-alpha_f} (dism)
  //    D_{n+1-alpha_f} := (1.-alphaf) * D_{n+1} + alpha_f * D_{n}
  dism_->Update(1.-alphaf_, *disn_, alphaf_, (*dis_)[0], 0.0);

  // mid-velocities V_{n+1-alpha_f} (velm)
  //    V_{n+1-alpha_f} := (1.-alphaf) * V_{n+1} + alpha_f * V_{n}
  velm_->Update(1.-alphaf_, *veln_, alphaf_, (*vel_)[0], 0.0);

  // mid-accelerations A_{n+1-alpha_m} (accm)
  //    A_{n+1-alpha_m} := (1.-alpha_m) * A_{n+1} + alpha_m * A_{n}
  accm_->Update(1.-alpham_, *accn_, alpham_, (*acc_)[0], 0.0);

  // jump
  return;
}

/*----------------------------------------------------------------------*/
/* calculate characteristic/reference norms for displacements
 * originally by lw */
double STR::TimIntGenAlpha::CalcRefNormDisplacement()
{
  // The reference norms are used to scale the calculated iterative
  // displacement norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  double charnormdis = 0.0;
  if (pressure_ != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> disp = pressure_->ExtractOtherVector((*dis_)(0));
    charnormdis = STR::AUX::CalculateVectorNorm(iternorm_, disp);
  }
  else
    charnormdis = STR::AUX::CalculateVectorNorm(iternorm_, (*dis_)(0));

  // rise your hat
  return charnormdis;
}

/*----------------------------------------------------------------------*/
/* calculate characteristic/reference norms for forces
 * originally by lw */
double STR::TimIntGenAlpha::CalcRefNormForce()
{
  // The reference norms are used to scale the calculated iterative
  // displacement norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  // norm of the internal forces
  double fintnorm = 0.0;
  fintnorm = STR::AUX::CalculateVectorNorm(iternorm_, fintm_);

  // norm of the external forces
  double fextnorm = 0.0;
  fextnorm = STR::AUX::CalculateVectorNorm(iternorm_, fextm_);

  // norm of the inertial forces
  double finertnorm = 0.0;
  finertnorm = STR::AUX::CalculateVectorNorm(iternorm_, finertm_);

  // norm of viscous forces
  double fviscnorm = 0.0;
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    fviscnorm = STR::AUX::CalculateVectorNorm(iternorm_, fviscm_);
  }

  // norm of reaction forces
  double freactnorm = 0.0;
  freactnorm = STR::AUX::CalculateVectorNorm(iternorm_, freact_);

  // determine worst value ==> charactersitic norm
  return max(fviscnorm, max(finertnorm, max(fintnorm, max(fextnorm, freactnorm))));
}

/*----------------------------------------------------------------------*/
/* incremental iteration update of state */
void STR::TimIntGenAlpha::UpdateIterIncrementally()
{
  // auxiliary global vectors
  Teuchos::RCP<Epetra_Vector> aux
    = LINALG::CreateVector(*dofrowmap_, true);

  // further auxiliary variables
  const double dt = (*dt_)[0];  // step size \f$\Delta t_{n}\f$

  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->Update(1.0, *disi_, 1.0);

  // new end-point velocities
  aux->Update(1.0, *disn_, -1.0, (*dis_)[0], 0.0);
  aux->Update((beta_-gamma_)/beta_, (*vel_)[0],
              (2.0*beta_-gamma_)*dt/(2.0*beta_), (*acc_)[0],
              gamma_/(beta_*dt));
  // put only to free/non-DBC DOFs
  dbcmaps_->InsertOtherVector(dbcmaps_->ExtractOtherVector(aux), veln_);

  // new end-point accelerations
  aux->Update(1.0, *disn_, -1.0, (*dis_)[0], 0.0);
  aux->Update(-1.0/(beta_*dt), (*vel_)[0],
              (2.0*beta_-1.0)/(2.0*beta_), (*acc_)[0],
              1.0/(beta_*dt*dt));
  // put only to free/non-DBC DOFs
  dbcmaps_->InsertOtherVector(dbcmaps_->ExtractOtherVector(aux), accn_);

  // bye
  return;
}

/*----------------------------------------------------------------------*/
/* iterative iteration update of state */
void STR::TimIntGenAlpha::UpdateIterIteratively()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->Update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->Update(gamma_/(beta_*(*dt_)[0]), *disi_, 1.0);

  // new end-point accelerations
  accn_->Update(1.0/(beta_*(*dt_)[0]*(*dt_)[0]), *disi_, 1.0);

  // bye
  return;
}

/*----------------------------------------------------------------------*/
/* update after time step */
void STR::TimIntGenAlpha::UpdateStepState()
{
  // velocity update for contact
  // (must be called BEFORE the following update steps)
  UpdateStepContactVUM();

  // update all old state at t_{n-1} etc
  // important for step size adaptivity
  // new displacements at t_{n+1} -> t_n
  //    D_{n} := D_{n+1}, etc
  dis_->UpdateSteps(*disn_);
  // new velocities at t_{n+1} -> t_n
  //    V_{n} := V_{n+1}, etc
  vel_->UpdateSteps(*veln_);
  // new accelerations at t_{n+1} -> t_n
  //    A_{n} := A_{n+1}, etc
  acc_->UpdateSteps(*accn_);

  // update new external force
  //    F_{ext;n} := F_{ext;n+1}
  fext_->Update(1.0, *fextn_, 0.0);

  // update new internal force
  //    F_{int;n} := F_{int;n+1}
  fint_->Update(1.0, *fintn_, 0.0);

  // update new inertial force
  //    F_{inert;n} := F_{inert;n+1}
  finert_->Update(1.0, *finertn_, 0.0);

  // update surface stress
  UpdateStepSurfstress();

  // update constraints
  UpdateStepConstraint();

  // update contact / meshtying
  UpdateStepContactMeshtying();

  // update beam contact
  UpdateStepBeamContact();

  // look out
  return;
}

/*----------------------------------------------------------------------*/
/* update after time step after output on element level*/
// update anything that needs to be updated at the element level
void STR::TimIntGenAlpha::UpdateStepElement()
{
  // create the parameters for the discretization
  ParameterList p;
  // other parameters that might be needed by the elements
  p.set("total time", timen_);
  p.set("delta time", (*dt_)[0]);
  // action for elements
  p.set("action", "calc_struct_update_istep");

  // go to elements
  discret_->ClearState();
  discret_->SetState("displacement",(*dis_)(0));

  if (!HaveNonlinearMass())
  {
    discret_->Evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
  }
  else
  {
  	// In the NonlinearMass-case its possible to make an update of displacements, velocities
  	// and accelerations at the end of time step (currently only necessary for Kirchhoff beams)
    // An corresponding update rule has to be implemented in the element, otherwise
    // displacements, velocities and accelerations remain unchange.
    discret_->SetState("velocity",(*vel_)(0));
    discret_->SetState("acceleration",(*acc_)(0));

    Teuchos::RCP<Epetra_Vector> update_disp;
    update_disp = LINALG::CreateVector(*dofrowmap_, true);

    Teuchos::RCP<Epetra_Vector> update_vel;
    update_vel = LINALG::CreateVector(*dofrowmap_, true);

    Teuchos::RCP<Epetra_Vector> update_acc;
    update_acc = LINALG::CreateVector(*dofrowmap_, true);


    discret_->Evaluate(p, Teuchos::null, Teuchos::null, update_disp, update_vel, update_acc);

    disn_->Update(1.0,*update_disp,1.0);
    (*dis_)(0)->Update(1.0,*update_disp,1.0);
    veln_->Update(1.0,*update_vel,1.0);
    (*vel_)(0)->Update(1.0,*update_vel,1.0);
    accn_->Update(1.0,*update_acc,1.0);
    (*acc_)(0)->Update(1.0,*update_acc,1.0);

  }

  discret_->ClearState();
}

/*----------------------------------------------------------------------*/
/* read and/or calculate forces for restart */
void STR::TimIntGenAlpha::ReadRestartForce()
{
  IO::DiscretizationReader reader(discret_, step_);
  reader.ReadVector(fext_, "fexternal");
  reader.ReadVector(fint_, "fint");
  reader.ReadVector(finert_, "finert");

  return;
}

/*----------------------------------------------------------------------*/
/* write internal and external forces for restart */
void STR::TimIntGenAlpha::WriteRestartForce(Teuchos::RCP<IO::DiscretizationWriter> output)
{
  output->WriteVector("fexternal", fext_);
  output->WriteVector("fint",fint_);
  output->WriteVector("finert",finert_);
  return;
}
