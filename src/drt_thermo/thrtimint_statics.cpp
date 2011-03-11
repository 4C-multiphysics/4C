/*!
\file thrtimint_statics.cpp
\brief Statics analysis

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/

/*----------------------------------------------------------------------*
 |  definitions                                              dano 08/09 |
 *----------------------------------------------------------------------*/
#ifdef CCADISCRET

/*----------------------------------------------------------------------*
 |  headers                                                  dano 08/09 |
 *----------------------------------------------------------------------*/
#include "thrtimint_statics.H"

/*----------------------------------------------------------------------*
 |  constructor                                              dano 08/09 |
 *----------------------------------------------------------------------*/
THR::TimIntStatics::TimIntStatics(
  const Teuchos::ParameterList& ioparams,
  const Teuchos::ParameterList& tdynparams,
  const Teuchos::ParameterList& xparams,
  Teuchos::RCP<DRT::Discretization> actdis,
  Teuchos::RCP<LINALG::Solver> solver,
  Teuchos::RCP<IO::DiscretizationWriter> output
  )
: TimIntImpl(
    ioparams,
    tdynparams,
    xparams,
    actdis,
    solver,
    output
    ),
  fint_(Teuchos::null),
  fintn_(Teuchos::null),
  fext_(Teuchos::null),
  fextn_(Teuchos::null)
{
  // info to user
  if (myrank_ == 0)
  {
    std::cout << "with statics" << std::endl;
  }

  //! create force vectors

  //! internal force vector F_{int;n} at last time
  fint_ = LINALG::CreateVector(*dofrowmap_, true);
  //! internal force vector F_{int;n+1} at new time
  fintn_ = LINALG::CreateVector(*dofrowmap_, true);
  //! set initial internal force vector
  ApplyForceTangInternal((*time_)[0], (*dt_)[0], (*temp_)(0), zeros_,
                         fint_, tang_);
  //! external force vector F_ext at last times
  fext_ = LINALG::CreateVector(*dofrowmap_, true);
  //! external force vector F_{n+1} at new time
  fextn_ = LINALG::CreateVector(*dofrowmap_, true);

  // set initial external force vector of convective heat transfer boundary
  // conditions
  ApplyForceExternalConv((*time_)[0], (*temp_)(0), (*temp_)(0), fext_, tang_);

  //! set initial external force vector
  ApplyForceExternal((*time_)[0], (*temp_)(0), fext_);

  //! have a nice day
  return;
}

/*----------------------------------------------------------------------*
 |  Consistent predictor with constant temperatures          dano 08/09 |
 |  and consistent temperature rates and temperatures                   |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::PredictConstTempConsistRate()
{
  //! constant predictor : temperature in domain
  // T_n+1,p = T_n
  tempn_->Update(1.0, *(*temp_)(0), 0.0);

  //! new end-point temperature rates, these stay zero in static calculation
  raten_->PutScalar(0.0);

  //! watch out
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate residual force and its tangent, ie derivative   dano 08/09 |
 |  with respect to end-point temperatures \f$T_{n+1}\f$                |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::EvaluateRhsTangResidual()
{
  //! build new external forces
  fextn_->PutScalar(0.0);

  //! initialize tangent matrix to zero
  tang_->Zero();

  // set initial external force vector of convective heat transfer boundary
  // conditions
  // Warning: do not use convection boundary condition with T_n and statics
  // --> always use T_n+1 for statics!
  ApplyForceExternalConv(timen_, (*temp_)(0), tempn_, fextn_, tang_);

  ApplyForceExternal(timen_, (*temp_)(0), fextn_);

  //! initialize internal forces
  fintn_->PutScalar(0.0);

  //! ordinary internal force and tangent
  ApplyForceTangInternal(timen_, (*dt_)[0], tempn_, tempi_, fintn_, tang_);

  //! build residual  Res = F_{int;n+1} - F_{ext;n+1}
  fres_->Update(-1.0, *fextn_, 0.0);
  fres_->Update(1.0, *fintn_, 1.0);

  //! build tangent matrix : effective dynamic tangent matrix
  //!    K_{Teffdyn} = K_{T}
  //! i.e. do nothing here

  // apply modifications due to thermal contact
  ApplyThermoContact(tang_,fres_,tempn_);

  tang_->Complete();  // close tangent matrix

  //! hallelujah
  return;
}

/*----------------------------------------------------------------------*
 |  calculate characteristic/reference norms for             dano 08/09 |
 |  temperatures originally by lw                                       |
 *----------------------------------------------------------------------*/
double THR::TimIntStatics::CalcRefNormTemperature()
{
  //! The reference norms are used to scale the calculated iterative
  //! temperature norm and/or the residual force norm. For this
  //! purpose we only need the right order of magnitude, so we don't
  //! mind evaluating the corresponding norms at possibly different
  //! points within the timestep (end point, generalized midpoint).

  double charnormtemp = 0.0;
    charnormtemp = THR::AUX::CalculateVectorNorm(iternorm_, (*temp_)(0));

  //! rise your hat
  return charnormtemp;
}

/*----------------------------------------------------------------------*
 |  calculate characteristic/reference norms for forces      dano 08/09 |
 |  originally by lw                                                    |
 *----------------------------------------------------------------------*/
double THR::TimIntStatics::CalcRefNormForce()
{
  //! The reference norms are used to scale the calculated iterative
  //! temperature norm and/or the residual force norm. For this
  //! purpose we only need the right order of magnitude, so we don't
  //! mind evaluating the corresponding norms at possibly different
  //! points within the timestep (end point, generalized midpoint).

  //! norm of the internal forces
  double fintnorm = 0.0;
  fintnorm = THR::AUX::CalculateVectorNorm(iternorm_, fintn_);

  //! norm of the external forces
  double fextnorm = 0.0;
  fextnorm = THR::AUX::CalculateVectorNorm(iternorm_, fextn_);

  //! norm of reaction forces
  double freactnorm = 0.0;
  freactnorm = THR::AUX::CalculateVectorNorm(iternorm_, freact_);

  //! return char norm
  return max(fintnorm, max(fextnorm, freactnorm));
}

/*----------------------------------------------------------------------*
 |  incremental iteration update of state                    dano 08/09 |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::UpdateIterIncrementally()
{
  //! new end-point temperatures
  //! T_{n+1}^{<k+1>} := T_{n+1}^{<k>} + IncT_{n+1}^{<k>}
  tempn_->Update(1.0, *tempi_, 1.0);

  //! bye
  return;
}

/*----------------------------------------------------------------------*
 |  iterative iteration update of state                      dano 08/09 |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::UpdateIterIteratively()
{
  //! new end-point temperatures
  //! T_{n+1}^{<k+1>} := T_{n+1}^{<k>} + IncT_{n+1}^{<k>}
  tempn_->Update(1.0, *tempi_, 1.0);

  //! bye
  return;
}

/*----------------------------------------------------------------------*
 |  update after time step                                   dano 08/09 |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::UpdateStepState()
{
  //! update state
  //! new temperatures at t_{n+1} -> t_n
  //!    T_{n} := T_{n+1}
  temp_->UpdateSteps(*tempn_);
  //! new temperature rates at t_{n+1} -> t_n
  //!    T'_{n} := T'_{n+1}
  rate_->UpdateSteps(*raten_);  // this simply copies zero vectors

  //! update new external force
  //!    F_{ext;n} := F_{ext;n+1}
  fext_->Update(1.0, *fextn_, 0.0);

  //! update new internal force
  //!    F_{int;n} := F_{int;n+1}
  fint_->Update(1.0, *fintn_, 0.0);

  //! update anything that needs to be updated at the element level
  {
    //! create the parameters for the discretization
    Teuchos::ParameterList p;
    //! other parameters that might be needed by the elements
    p.set("total time", timen_);
    p.set("delta time", (*dt_)[0]);
    //! action for elements
    p.set("action", "calc_thermo_update_istep");
    //! go to elements
    discret_->Evaluate(p, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);
  }

  //! look out
  return;
}

/*----------------------------------------------------------------------*
 |  read restart forces                                      dano 08/09 |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::ReadRestartForce()
{
  IO::DiscretizationReader reader(discret_, step_);
  //! set 'initial' external force
  reader.ReadVector(fext_, "fexternal");
  //! set 'initial' internal force vector
  //! Set dt to 0, since we do not propagate in time.
  ApplyForceInternal((*time_)[0], 0.0, (*temp_)(0), zeros_, fint_);

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate the internal force and the tangent              dano 08/09 |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::ApplyForceTangInternal(
  const double time,  //!< evaluation time
  const double dt,  //!< step size
  const Teuchos::RCP<Epetra_Vector> temp,  //!< temperature state
  const Teuchos::RCP<Epetra_Vector> tempi,  //!< residual temperatures
  Teuchos::RCP<Epetra_Vector> fint,  //!< internal force
  Teuchos::RCP<LINALG::SparseMatrix> tang  //!< tangent matrix
  )
{
  //! create the parameters for the discretization
  Teuchos::ParameterList p;
  //! set parameters
  // ...
  //! call the base function
  TimInt::ApplyForceTangInternal(p,time,dt,temp,tempi,fint,tang);
  //! finish
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate the internal force                              dano 08/09 |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::ApplyForceInternal(
  const double time,  //!< evaluation time
  const double dt,  //!< step size
  const Teuchos::RCP<Epetra_Vector> temp,  //!< temperature state
  const Teuchos::RCP<Epetra_Vector> tempi,  //!< incremental temperatures
  Teuchos::RCP<Epetra_Vector> fint  //!< internal force
  )
{
  //! create the parameters for the discretization
  Teuchos::ParameterList p;
  //! set parameters
  // ...
  //! call the base function
  TimInt::ApplyForceInternal(p,time,dt,temp,tempi,fint);
  //! finish
  return;
}


/*----------------------------------------------------------------------*
 |  evaluate the convective boundary condition               dano 01/11 |
 *----------------------------------------------------------------------*/
void THR::TimIntStatics::ApplyForceExternalConv(
  const double time,  //!< evaluation time
  const Teuchos::RCP<Epetra_Vector> tempn,  //!< old temperature state T_n
  const Teuchos::RCP<Epetra_Vector> temp,  //!< temperature state T_n+1
  Teuchos::RCP<Epetra_Vector> fext,  //!< external force
  Teuchos::RCP<LINALG::SparseMatrix> tang  //!< tangent matrix
  )
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // set parameters
  // ...
  // call the base function
  TimInt::ApplyForceExternalConv(p,time,tempn,temp,fext,tang);
  // finish
  return;
}


/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET
