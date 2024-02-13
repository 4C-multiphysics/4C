/*----------------------------------------------------------------------*/
/*! \file
\brief Structural time integration with generalised energy-momentum method
\level 1
*/

/*----------------------------------------------------------------------*/
/* macros */

/*----------------------------------------------------------------------*/
/* headers */
#include "baci_structure_timint_gemm.hpp"

#include "baci_global_data.hpp"
#include "baci_io.hpp"
#include "baci_lib_locsys.hpp"
#include "baci_linalg_utils_sparse_algebra_create.hpp"
#include "baci_structure_aux.hpp"

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* constructor */
STR::TimIntGEMM::TimIntGEMM(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& ioparams, const Teuchos::ParameterList& sdynparams,
    const Teuchos::ParameterList& xparams, Teuchos::RCP<DRT::Discretization> actdis,
    Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<CORE::LINALG::Solver> contactsolver,
    Teuchos::RCP<IO::DiscretizationWriter> output)
    : TimIntImpl(timeparams, ioparams, sdynparams, xparams, actdis, solver, contactsolver, output),
      beta_(sdynparams.sublist("GEMM").get<double>("BETA")),
      gamma_(sdynparams.sublist("GEMM").get<double>("GAMMA")),
      alphaf_(sdynparams.sublist("GEMM").get<double>("ALPHA_F")),
      alpham_(sdynparams.sublist("GEMM").get<double>("ALPHA_M")),
      xi_(sdynparams.sublist("GEMM").get<double>("XI")),
      dism_(Teuchos::null),
      velm_(Teuchos::null),
      accm_(Teuchos::null),
      fintm_(Teuchos::null),
      fext_(Teuchos::null),
      fextm_(Teuchos::null),
      fextn_(Teuchos::null),
      finertm_(Teuchos::null),
      fviscm_(Teuchos::null)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during Setup() in a base class.
}

/*----------------------------------------------------------------------------------------------*
 * Initialize this class                                                            rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void STR::TimIntGEMM::Init(const Teuchos::ParameterList& timeparams,
    const Teuchos::ParameterList& sdynparams, const Teuchos::ParameterList& xparams,
    Teuchos::RCP<DRT::Discretization> actdis, Teuchos::RCP<CORE::LINALG::Solver> solver)
{
  // call Init() in base class
  STR::TimIntImpl::Init(timeparams, sdynparams, xparams, actdis, solver);

  // info to user about current time integration scheme and its parametrization
  if (myrank_ == 0)
  {
    std::cout << "with generalised energy-momentum method" << '\n'
              << "   alpha_f = " << alphaf_ << '\n'
              << "   alpha_m = " << alpham_ << '\n'
              << "   xi = " << xi_ << '\n'
              << "   p_dis = " << MethodOrderOfAccuracyDis() << '\n'
              << "   p_vel = " << MethodOrderOfAccuracyVel() << '\n'
              << '\n';
  }
}

/*----------------------------------------------------------------------------------------------*
 * Setup this class                                                                 rauch 09/16 |
 *----------------------------------------------------------------------------------------------*/
void STR::TimIntGEMM::Setup()
{
  // call Setup() in base class
  STR::TimIntImpl::Setup();

  // determine mass, damping and initial accelerations
  DetermineMassDampConsistAccel();

  // create state vectors

  // mid-displacements
  dism_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // mid-velocities
  velm_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // mid-accelerations
  accm_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  // create force vectors

  // internal force vector F_{int;m} at mid-time
  fintm_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  // external force vector F_ext at last times
  fext_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // external mid-force vector F_{ext;n+1-alpha_f}
  fextm_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // external force vector F_{n+1} at new time
  fextn_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // set initial external force vector
  ApplyForceExternal((*time_)[0], (*dis_)(0), disn_, (*vel_)(0), fext_);

  // inertia mid-point force vector F_inert
  finertm_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);
  // viscous mid-point force vector F_visc
  fviscm_ = CORE::LINALG::CreateVector(*DofRowMapView(), true);

  // GEMM time integrator cannot handle nonlinear inertia forces
  if (HaveNonlinearMass())
  {
    dserror(
        "Gemm time integrator cannot handle nonlinear inertia forces "
        "(flag: MASSLIN)");
  }
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant displacements
 * and consistent velocities and displacements */
void STR::TimIntGEMM::PredictConstDisConsistVelAcc()
{
  // constant predictor : displacement in domain
  disn_->Update(1.0, *(*dis_)(0), 0.0);

  // consistent velocities following Newmark formulas
  veln_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  veln_->Update((beta_ - gamma_) / beta_, *(*vel_)(0),
      (2. * beta_ - gamma_) * (*dt_)[0] / (2. * beta_), *(*acc_)(0), gamma_ / (beta_ * (*dt_)[0]));

  // consistent accelerations following Newmark formulas
  accn_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  accn_->Update(-1. / (beta_ * (*dt_)[0]), *(*vel_)(0), (2. * beta_ - 1.) / (2. * beta_),
      *(*acc_)(0), 1. / (beta_ * (*dt_)[0] * (*dt_)[0]));

  // reset the residual displacement
  disi_->PutScalar(0.0);
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant velocities,
 * extrapolated displacements and consistent accelerations */
void STR::TimIntGEMM::PredictConstVelConsistAcc()
{
  // extrapolated displacements based upon constant velocities
  // d_{n+1} = d_{n} + dt * v_{n}
  disn_->Update(1.0, (*dis_)[0], (*dt_)[0], (*vel_)[0], 0.0);

  // consistent velocities following Newmark formulas
  veln_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  veln_->Update((beta_ - gamma_) / beta_, *(*vel_)(0),
      (2. * beta_ - gamma_) * (*dt_)[0] / (2. * beta_), *(*acc_)(0), gamma_ / (beta_ * (*dt_)[0]));

  // consistent accelerations following Newmark formulas
  accn_->Update(1.0, *disn_, -1.0, *(*dis_)(0), 0.0);
  accn_->Update(-1. / (beta_ * (*dt_)[0]), *(*vel_)(0), (2. * beta_ - 1.) / (2. * beta_),
      *(*acc_)(0), 1. / (beta_ * (*dt_)[0] * (*dt_)[0]));

  // reset the residual displacement
  disi_->PutScalar(0.0);
}

/*----------------------------------------------------------------------*/
/* Consistent predictor with constant accelerations
 * and extrapolated velocities and displacements */
void STR::TimIntGEMM::PredictConstAcc()
{
  // extrapolated displacements based upon constant accelerations
  // d_{n+1} = d_{n} + dt * v_{n} + dt^2 / 2 * a_{n}
  disn_->Update(1.0, (*dis_)[0], (*dt_)[0], (*vel_)[0], 0.0);
  disn_->Update((*dt_)[0] * (*dt_)[0] / 2., (*acc_)[0], 1.0);

  // extrapolated velocities (equal to consistent velocities)
  // v_{n+1} = v_{n} + dt * a_{n}
  veln_->Update(1.0, (*vel_)[0], (*dt_)[0], (*acc_)[0], 0.0);

  // constant accelerations (equal to consistent accelerations)
  accn_->Update(1.0, (*acc_)[0], 0.0);

  // reset the residual displacement
  disi_->PutScalar(0.0);
}

/*----------------------------------------------------------------------*/
/* evaluate residual force and its stiffness, ie derivative
 * with respect to end-point displacements \f$D_{n+1}\f$ */
void STR::TimIntGEMM::EvaluateForceStiffResidual(Teuchos::ParameterList& params)
{
  // get info about prediction step from parameter list
  bool predict = false;
  if (params.isParameter("predict")) predict = params.get<bool>("predict");

  // build by last converged state and predicted target state
  // the predicted mid-state
  EvaluateMidState();

  // initialise stiffness matrix to zero
  stiff_->Zero();

  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->PutScalar(0.0);
  ApplyForceStiffExternal(timen_, (*dis_)(0), disn_, (*vel_)(0), fextn_, stiff_);

  // additional external forces are added (e.g. interface forces)
  fextn_->Update(1.0, *fifc_, 1.0);

  // external mid-forces F_{ext;n+1-alpha_f} (fextm)
  //    F_{ext;n+1-alpha_f} := (1.-alphaf) * F_{ext;n+1}
  //                         + alpha_f * F_{ext;n}
  fextm_->Update(1. - alphaf_, *fextn_, alphaf_, *fext_, 0.0);

  // ************************** (2) INTERNAL FORCES ***************************

  // initialise internal forces
  fintm_->PutScalar(0.0);

  // ordinary internal force and stiffness
  disi_->Scale(1. - alphaf_);  // CHECK THIS
  ApplyForceStiffInternalMid(timen_, (*dt_)[0], (*dis_)(0), disn_, disi_, veln_, fintm_, stiff_);

  // apply forces and stiffness due to constraints
  Teuchos::ParameterList pcon;  // apply empty parameterlist, no scaling necessary
  ApplyForceStiffConstraint(timen_, (*dis_)(0), disn_, fintm_, stiff_, pcon);

  // add forces and stiffness due to 0D cardiovascular coupling conditions
  Teuchos::ParameterList pwindk;
  pwindk.set("scale_timint", 1.);
  pwindk.set("time_step_size", (*dt_)[0]);
  ApplyForceStiffCardiovascular0D(timen_, disn_, fintm_, stiff_, pwindk);

  // add forces and stiffness due to spring dashpot condition
  Teuchos::ParameterList psprdash;
  psprdash.set("time_fac", gamma_ / (beta_ * (*dt_)[0]));
  psprdash.set("dt", (*dt_)[0]);  // needed only for cursurfnormal option!!
  ApplyForceStiffSpringDashpot(stiff_, fintm_, disn_, veln_, predict, psprdash);

  // ************************** (3) INERTIAL FORCES ***************************

  // inertial forces #finertm_
  mass_->Multiply(false, *accm_, *finertm_);

  // ************************** (4) DAMPING FORCES ****************************

  // viscous forces due Rayleigh damping
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    damp_->Multiply(false, *velm_, *fviscm_);
  }

  // ******************** Finally, put everything together ********************

  // build residual
  //    Res = M . A_{n+1-alpha_m}
  //        + C . V_{n+1-alpha_f}
  //        + F_{int;m}
  //        - F_{ext;n+1-alpha_f}
  fres_->Update(-1.0, *fextm_, 0.0);
  fres_->Update(1.0, *fintm_, 1.0);
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    fres_->Update(1.0, *fviscm_, 1.0);
  }
  fres_->Update(1.0, *finertm_, 1.0);

  // build tangent matrix : effective dynamic stiffness matrix
  //    K_{Teffdyn} = (1 - alpha_m)/(beta*dt^2) M
  //                + (1 - alpha_f)*y/(beta*dt) C
  //                + K_{T;m}
  stiff_->Add(*mass_, false, (1. - alpham_) / (beta_ * (*dt_)[0] * (*dt_)[0]), 1.0);
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    stiff_->Add(*damp_, false, (1. - alphaf_) * gamma_ / (beta_ * (*dt_)[0]), 1.0);
  }

  // apply forces and stiffness due to contact / meshtying
  // Note that we ALWAYS use a TR-like approach to compute the interface
  // forces. This means we never explicitly compute fc at the generalized
  // mid-point n+1-alphaf, but use a linear combination of the old end-
  // point n and the new end-point n+1 instead:
  // F_{c;n+1-alpha_f} := (1-alphaf) * F_{c;n+1} +  alpha_f * F_{c;n}
  ApplyForceStiffContactMeshtying(stiff_, fres_, disn_, predict);

  // close stiffness matrix
  stiff_->Complete();
}

/*----------------------------------------------------------------------*/
/* Evaluate/define the residual force vector #fres_ for
 * relaxation solution with SolveRelaxationLinear */
void STR::TimIntGEMM::EvaluateForceStiffResidualRelax(Teuchos::ParameterList& params)
{
  // compute residual forces #fres_ and stiffness #stiff_
  EvaluateForceStiffResidual(params);

  // overwrite the residual forces #fres_ with interface load
  fres_->Update(-(1.0 - alphaf_), *fifc_, 0.0);
}

/*----------------------------------------------------------------------*/
/* Evaluate residual */
void STR::TimIntGEMM::EvaluateForceResidual()
{
  // build predicted mid-state by last converged state and predicted target state
  EvaluateMidState();

  // ************************** (1) EXTERNAL FORCES ***************************

  // build new external forces
  fextn_->PutScalar(0.0);
  ApplyForceExternal(timen_, (*dis_)(0), disn_, (*vel_)(0), fextn_);

  // additional external forces are added (e.g. interface forces)
  fextn_->Update(1.0, *fifc_, 1.0);

  // external mid-forces F_{ext;n+1-alpha_f} (fextm)
  //    F_{ext;n+1-alpha_f} := (1.-alphaf) * F_{ext;n+1}
  //                         + alpha_f * F_{ext;n}
  fextm_->Update(1. - alphaf_, *fextn_, alphaf_, *fext_, 0.0);

  // ************************** (2) INTERNAL FORCES ***************************

  // initialise internal forces
  fintm_->PutScalar(0.0);

  // ordinary internal force and stiffness
  disi_->Scale(1. - alphaf_);  // CHECK THIS
  ApplyForceInternalMid(timen_, (*dt_)[0], (*dis_)(0), disn_, disi_, veln_, fintm_);

  // ************************** (3) INERTIAL FORCES ***************************

  // inertial forces #finertm_
  mass_->Multiply(false, *accm_, *finertm_);

  // ************************** (4) DAMPING FORCES ****************************

  // viscous forces due Rayleigh damping
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    damp_->Multiply(false, *velm_, *fviscm_);
  }

  // ******************** Finally, put everything together ********************

  // build residual
  //    Res = M . A_{n+1-alpha_m}
  //        + C . V_{n+1-alpha_f}
  //        + F_{int;m}
  //        - F_{ext;n+1-alpha_f}
  fres_->Update(-1.0, *fextm_, 0.0);
  fres_->Update(1.0, *fintm_, 1.0);
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    fres_->Update(1.0, *fviscm_, 1.0);
  }
  fres_->Update(1.0, *finertm_, 1.0);
}

/*----------------------------------------------------------------------*/
/* evaluate mid-state vectors by averaging end-point vectors */
void STR::TimIntGEMM::EvaluateMidState()
{
  // mid-displacements D_{n+1-alpha_f} (dism)
  //    D_{n+1-alpha_f} := (1.-alphaf) * D_{n+1} + alpha_f * D_{n}
  dism_->Update(1. - alphaf_, *disn_, alphaf_, (*dis_)[0], 0.0);

  // mid-velocities V_{n+1-alpha_f} (velm)
  //    V_{n+1-alpha_f} := (1.-alphaf) * V_{n+1} + alpha_f * V_{n}
  velm_->Update(1. - alphaf_, *veln_, alphaf_, (*vel_)[0], 0.0);

  // mid-accelerations A_{n+1-alpha_m} (accm)
  //    A_{n+1-alpha_m} := (1.-alpha_m) * A_{n+1} + alpha_m * A_{n}
  accm_->Update(1. - alpham_, *accn_, alpham_, (*acc_)[0], 0.0);
}

/*----------------------------------------------------------------------*/
/* calculate characteristic/reference norms for forces
 * originally by lw */
double STR::TimIntGEMM::CalcRefNormForce()
{
  // The reference norms are used to scale the calculated iterative
  // displacement norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).

  // norm of the internal forces
  double fintnorm = 0.0;
  fintnorm = STR::CalculateVectorNorm(iternorm_, fintm_);

  // norm of the external forces
  double fextnorm = 0.0;
  fextnorm = STR::CalculateVectorNorm(iternorm_, fextm_);

  // norm of the inertial forces
  double finertnorm = 0.0;
  finertnorm = STR::CalculateVectorNorm(iternorm_, finertm_);

  // norm of viscous forces
  double fviscnorm = 0.0;
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    fviscnorm = STR::CalculateVectorNorm(iternorm_, fviscm_);
  }

  // norm of reaction forces
  double freactnorm = 0.0;
  freactnorm = STR::CalculateVectorNorm(iternorm_, freact_);

  // determine worst value ==> charactersitic norm
  return std::max(
      fviscnorm, std::max(finertnorm, std::max(fintnorm, std::max(fextnorm, freactnorm))));
}

/*----------------------------------------------------------------------*/
void STR::TimIntGEMM::UpdateIterIncrementally()
{
  // step size \f$\Delta t_{n}\f$
  const double dt = (*dt_)[0];

  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->Update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->Update(1.0, *disn_, -1.0, (*dis_)[0], 0.0);
  veln_->Update((beta_ - gamma_) / beta_, (*vel_)[0], (2.0 * beta_ - gamma_) * dt / (2.0 * beta_),
      (*acc_)[0], gamma_ / (beta_ * dt));

  // new end-point accelerations
  accn_->Update(1.0, *disn_, -1.0, (*dis_)[0], 0.0);
  accn_->Update(-1.0 / (beta_ * dt), (*vel_)[0], (2.0 * beta_ - 1.0) / (2.0 * beta_), (*acc_)[0],
      1.0 / (beta_ * dt * dt));
}

/*----------------------------------------------------------------------*/
/* iterative iteration update of state */
void STR::TimIntGEMM::UpdateIterIteratively()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->Update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->Update(gamma_ / (beta_ * (*dt_)[0]), *disi_, 1.0);

  // new end-point accelerations
  accn_->Update(1.0 / (beta_ * (*dt_)[0] * (*dt_)[0]), *disi_, 1.0);
}

/*----------------------------------------------------------------------*/
/* update after time step */
void STR::TimIntGEMM::UpdateStepState()
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
  // nothing to be done

  // update constraints
  UpdateStepConstraint();

  // update Cardiovascular0D
  UpdateStepCardiovascular0D();

  // update constraints
  UpdateStepSpringDashpot();

  // update contact  /meshtying
  UpdateStepContactMeshtying();
}

/*----------------------------------------------------------------------*/
/* update after time step after output on element level*/
// update anything that needs to be updated at the element level
void STR::TimIntGEMM::UpdateStepElement()
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // other parameters that might be needed by the elements
  p.set("total time", timen_);
  // p.set("delta time", (*dt_)[0]);
  // action for elements
  // p.set("alpha f", alphaf_);
  p.set("action", "calc_struct_update_istep");
  // go to elements
  discret_->SetState("displacement", (*dis_)(0));
  discret_->Evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);

  discret_->ClearState();
}

/*----------------------------------------------------------------------*/
/* evaluate ordinary internal force, its stiffness at mid-state */
void STR::TimIntGEMM::ApplyForceStiffInternalMid(const double time, const double dt,
    const Teuchos::RCP<Epetra_Vector> dis,            // displacement state at t_n
    const Teuchos::RCP<Epetra_Vector> disn,           // displacement state at t_{n+1}
    const Teuchos::RCP<Epetra_Vector> disi,           // residual displacements
    const Teuchos::RCP<Epetra_Vector> vel,            // velocity state
    Teuchos::RCP<Epetra_Vector> fint,                 // internal force
    Teuchos::RCP<CORE::LINALG::SparseOperator> stiff  // stiffness matrix
)
{
  // *********** time measurement ***********
  double dtcpu = timer_->wallTime();
  // *********** time measurement ***********

  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  const std::string action = "calc_struct_nlnstiff_gemm";
  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  p.set("alpha f", alphaf_);
  p.set("xi", xi_);
  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("old displacement", dis);
  discret_->SetState("displacement", disn);
  discret_->SetState("residual displacement", disi);
  if (damping_ == INPAR::STR::damp_material) discret_->SetState("velocity", vel);
  // fintn_->PutScalar(0.0);  // initialise internal force vector
  discret_->Evaluate(p, stiff, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // *********** time measurement ***********
  dtele_ = timer_->wallTime() - dtcpu;
  // *********** time measurement ***********
}

/*----------------------------------------------------------------------*/
/* evaluate ordinary internal force at mid-state */
void STR::TimIntGEMM::ApplyForceInternalMid(const double time, const double dt,
    const Teuchos::RCP<Epetra_Vector> dis, const Teuchos::RCP<Epetra_Vector> disn,
    const Teuchos::RCP<Epetra_Vector> disi, const Teuchos::RCP<Epetra_Vector> vel,
    Teuchos::RCP<Epetra_Vector> fint)
{
  // *********** time measurement ***********
  double dtcpu = timer_->wallTime();
  // *********** time measurement ***********

  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  const std::string action = "calc_struct_nlnstiff_gemm";
  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  p.set("alpha f", alphaf_);
  p.set("xi", xi_);
  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("old displacement", dis);
  discret_->SetState("displacement", disn);
  discret_->SetState("residual displacement", disi);
  if (damping_ == INPAR::STR::damp_material) discret_->SetState("velocity", vel);
  // fintn_->PutScalar(0.0);  // initialise internal force vector
  discret_->Evaluate(p, Teuchos::null, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // *********** time measurement ***********
  dtele_ = timer_->wallTime() - dtcpu;
  // *********** time measurement ***********
}

/*----------------------------------------------------------------------*/
/* read restart forces */
void STR::TimIntGEMM::ReadRestartForce()
{
  IO::DiscretizationReader reader(discret_, GLOBAL::Problem::Instance()->InputControlFile(), step_);
  reader.ReadVector(fext_, "fexternal");
}

/*----------------------------------------------------------------------*/
/* write external forces for restart */
void STR::TimIntGEMM::WriteRestartForce(Teuchos::RCP<IO::DiscretizationWriter> output)
{
  output->WriteVector("fexternal", fext_);
}

BACI_NAMESPACE_CLOSE
