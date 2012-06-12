/*!----------------------------------------------------------------------
\file statmech.cpp
\brief time integration for structural problems with statistical mechanics

<pre>
Maintainer: Kei Müller
            mueller@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15276
</pre>

*----------------------------------------------------------------------*/

#include <Teuchos_Time.hpp>
#include "Teuchos_RCP.hpp"
#include <iostream>
#include <iomanip>

#include "strtimint_statmech.H"

#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_sparseoperator.H"
#include "../linalg/linalg_solver.H"
#include "../drt_statmech/statmech_manager.H"
#include "../drt_inpar/inpar_statmech.H"
#include "../drt_io/io_control.H"
#include "../drt_constraint/constraint_manager.H"
#include "../drt_constraint/constraintsolver.H"
#include "../drt_inpar/inpar_contact.H"
#include "../drt_beamcontact/beam3contact_manager.H"
#include "stru_aux.H"

#include "../drt_beam3/beam3.H"
#include "../drt_beam3ii/beam3ii.H"
#include "../drt_beam2/beam2.H"
#include "../drt_beam2r/beam2r.H"
#include "../drt_truss3/truss3.H"
#include "../drt_truss2/truss2.H"


//#define GMSHPTCSTEPS

/*----------------------------------------------------------------------*
 |  ctor (public)                                             cyron 08/08|
 *----------------------------------------------------------------------*/
STR::TimIntStatMech::TimIntStatMech(const Teuchos::ParameterList& params,
                                    const Teuchos::ParameterList& sdynparams,
                                    const Teuchos::ParameterList& xparams,
                                    Teuchos::RCP<DRT::Discretization> actdis,
                                    Teuchos::RCP<LINALG::Solver> solver,
                                    Teuchos::RCP<LINALG::Solver> contactsolver,
                                    Teuchos::RCP<IO::DiscretizationWriter> output) :
TimIntOneStepTheta(params,sdynparams, xparams, actdis,solver,contactsolver,output),
isconverged_(false)
{
  // create StatMechManager object
  //statmechmanager_ = Teuchos::rcp(new STATMECH::StatMechManager(params_,*actdis));

  //getting number of dimensions for diffusion coefficient calculation
  ndim_= DRT::Problem::Instance()->NDim();

  // print dbc type for this simulation to screen
  StatMechPrintDBCType();

  // retrieve number of random numbers per element and store them in randomnumbersperelement_
  RandomNumbersPerElement();

  //suppress all output printed to screen in case of single filament studies in order not to generate too much output on the cluster
  SuppressOutput();

  // Initial Statistical Mechanics Output
  //getting number of dimensions for diffusion coefficient calculation
  statmechman_->InitOutput(DRT::Problem::Instance()->NDim(),(*dt_)[0]);

  // set up inverted dirichlet toggle vector (old dbc way)
  if(dirichtoggle_!=Teuchos::null)
  {
    invtoggle_ = Teuchos::rcp(new Epetra_Vector(*(discret_->DofRowMap()), true));
    invtoggle_->PutScalar(1.0);
    invtoggle_->Update(-1.0,*dirichtoggle_,1.0);
  }

  //in case that beam contact is activated by respective input parameter, a Beam3cmanager object is created
  InitializeBeamContact();

  return;
} // STR::TimIntStatMech::TimIntStatMech()

/*----------------------------------------------------------------------*
 |  print Dirichlet type to screen (public) mueller 03/12 |
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::StatMechPrintDBCType()
{
  if((statmechman_->GetPeriodLength())->at(0) <= 0.0 && DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"PERIODICDBC"))
    dserror("Set PERIODLENGTH  > 0.0 for all three components if periodic DBCs are to be applied");
  if(!discret_->Comm().MyPID())
  {
    INPAR::STATMECH::DBCType dbctype = DRT::INPUT::IntegralValue<INPAR::STATMECH::DBCType>(statmechman_->GetStatMechParams(),"DBCTYPE");
    switch(dbctype)
    {
      // standard DBC application
      case INPAR::STATMECH::dbctyp_std:
        cout<<"- Conventional Input file based application of DBCs"<<endl;
      break;
      // shear with a fixed Dirichlet node set
      case INPAR::STATMECH::dbctype_shearfixed:
        cout<<"- DBCs for rheological measurements applied: fixed node set"<<endl;
      break;
      // shear with an updated Dirichlet node set (only DOF in direction of oscillation is subject to BC, others free)
      case INPAR::STATMECH::dbctype_sheartrans:
        cout<<"- DBCs for rheological measurements applied: transient node set"<<endl;
      break;
      // pin down and release individual nodes
      case INPAR::STATMECH::dbctype_pinnodes:
        cout<<"- Special DBCs pinning down selected nodes"<<endl;
      break;
      // no DBCs at all
      case INPAR::STATMECH::dbctype_none:
        cout<<"- No application of DBCs (i.e. also no DBCs by Input file)"<<endl;
      break;
      // default: everything involving periodic boundary conditions
      default:
        dserror("Check your DBC type! %i", dbctype);
      break;
    }
    cout<<"================================================================"<<endl;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  random number per element                    (public) mueller 03/12 |
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::RandomNumbersPerElement()
{
  //maximal number of random numbers to be generated per time step for any column map element of this processor
  int randomnumbersperlocalelement = 0;

  /*check maximal number of nodes of an element with stochastic forces on this processor*/
  for (int i=0; i<  discret_->NumMyColElements(); ++i)
  {
    const DRT::ElementType & eot = discret_->lColElement(i)->ElementType();
    /*stochastic forces implemented so far only for the following elements:*/
    if ( eot == DRT::ELEMENTS::Beam3Type::Instance() )
    {
      //see whether current element needs more random numbers per time step than any other before
      randomnumbersperlocalelement = max(randomnumbersperlocalelement,dynamic_cast<DRT::ELEMENTS::Beam3*>(discret_->lColElement(i))->HowManyRandomNumbersINeed());

      //in case of periodic boundary conditions beam3 elements require a special initialization if they are broken by the periodic boundaries in the initial configuration
      if((statmechman_->GetPeriodLength())->at(0) > 0.0)
        statmechman_->PeriodicBoundaryBeam3Init(discret_->lColElement(i));
    }
    else if ( eot == DRT::ELEMENTS::Beam3iiType::Instance() )
    {
      //see whether current element needs more random numbers per time step than any other before
      randomnumbersperlocalelement = max(randomnumbersperlocalelement,dynamic_cast<DRT::ELEMENTS::Beam3ii*>(discret_->lColElement(i))->HowManyRandomNumbersINeed());

      //in case of periodic boundary conditions beam3 elements require a special initialization if they are broken by the periodic boundaries in the initial configuration
      if((statmechman_->GetPeriodLength())->at(0) > 0.0)
        statmechman_->PeriodicBoundaryBeam3iiInit(discret_->lColElement(i));
    }
    else if ( eot == DRT::ELEMENTS::Beam2Type::Instance() )
    {
      //see whether current element needs more random numbers per time step than any other before
      randomnumbersperlocalelement = max(randomnumbersperlocalelement,dynamic_cast<DRT::ELEMENTS::Beam2*>(discret_->lColElement(i))->HowManyRandomNumbersINeed());
    }
    else if ( eot == DRT::ELEMENTS::Beam2rType::Instance() )
    {
      //see whether current element needs more random numbers per time step than any other before
      randomnumbersperlocalelement = max(randomnumbersperlocalelement,dynamic_cast<DRT::ELEMENTS::Beam2r*>(discret_->lColElement(i))->HowManyRandomNumbersINeed());
    }
    else if ( eot == DRT::ELEMENTS::Truss3Type::Instance() )
    {
      //see whether current element needs more random numbers per time step than any other before
      randomnumbersperlocalelement = max(randomnumbersperlocalelement,dynamic_cast<DRT::ELEMENTS::Truss3*>(discret_->lColElement(i))->HowManyRandomNumbersINeed());

      //in case of periodic boundary conditions truss3 elements require a special initialization if they are broken by the periodic boundaries in the initial configuration
      if((statmechman_->GetPeriodLength())->at(0) > 0.0)
        statmechman_->PeriodicBoundaryTruss3Init(discret_->lColElement(i));
    }
    else
      continue;
  } //for (int i=0; i<dis_.NumMyColElements(); ++i)

  /*so far the maximal number of random numbers required per element has been checked only locally on this processor;
   *now we compare the results of each processor and store the maximal one in maxrandomnumbersperglobalelement_*/
  discret_->Comm().MaxAll(&randomnumbersperlocalelement,&maxrandomnumbersperglobalelement_ ,1);

  return;
}

/*----------------------------------------------------------------------*
 |  suppress output in some cases                (public) mueller 03/12 |
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::SuppressOutput()
{
  if( DRT::INPUT::IntegralValue<INPAR::STATMECH::StatOutput>(statmechman_->GetStatMechParams(), "SPECIAL_OUTPUT") == INPAR::STATMECH::statout_endtoendlog ||
      DRT::INPUT::IntegralValue<INPAR::STATMECH::StatOutput>(statmechman_->GetStatMechParams(), "SPECIAL_OUTPUT") == INPAR::STATMECH::statout_endtoendconst ||
      DRT::INPUT::IntegralValue<INPAR::STATMECH::StatOutput>(statmechman_->GetStatMechParams(), "SPECIAL_OUTPUT") == INPAR::STATMECH::statout_orientationcorrelation ||
      DRT::INPUT::IntegralValue<INPAR::STATMECH::StatOutput>(statmechman_->GetStatMechParams(), "SPECIAL_OUTPUT") == INPAR::STATMECH::statout_anisotropic)
  {
    printscreen_ = 0;
    std::cout<<"\n\nPay Attention: from now on regular output to screen suppressed !!!\n\n";
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Initialize beam contact                      (public) mueller 03/12 |
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::InitializeBeamContact()
{
  if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
  {
    //check wheter appropriate parameters are set in the parameter list "CONTACT & MESHTYING"
    const Teuchos::ParameterList& scontact = DRT::Problem::Instance()->MeshtyingAndContactParams();
    if (!DRT::INPUT::IntegralValue<INPAR::CONTACT::ApplicationType>(scontact,"APPLICATION") == INPAR::CONTACT::app_beamcontact)
      dserror("beam contact switched on in parameter list STATISTICAL MECHANICS, but not in in parameter list MESHTYING AND CONTACT!!!");
    else
    {
      // initialize beam contact solution strategy
      buildoctree_ = true;
//        // store integration parameter alphaf into beamcman_ as well
//      double alphaf = 1.0-theta_; // = 0.0 in statmech case
//      beamcman_ = rcp(new CONTACT::Beam3cmanager(*discret_,alphaf));
      // decide wether the tangent field should be smoothed or not
      if (DRT::INPUT::IntegralValue<INPAR::CONTACT::Smoothing>(DRT::Problem::Instance()->MeshtyingAndContactParams(),"BEAMS_SMOOTHING") == INPAR::CONTACT::bsm_none)
      {
        //cout << "Test BEAMS_SMOOTHING" << INPAR::CONTACT::bsm_none << endl;
      }
    }
    // Note: the beam contact manager object (beamcmanager_) is built in Integrate(), not here due to reasons decribed below!
  }
  return;
}

/*----------------------------------------------------------------------*
 |  integrate in time          (static/public)               cyron 08/08|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::Integrate()
{
  double eps = 1.0e-12;
  while( (timen_ <= timemax_+eps) and (stepn_ <= stepmax_) )
  {
    // preparations for contact in this time step
    BeamContactPrepareStep();
    // preparations for statistical mechanics in this time step
    StatMechPrepareStep();

    //redo time step in case of bad random configuration
    do
    {
      // Update of statmech specific quantities as well as new set of random numbers
      StatMechUpdate();

      // Solve system of equations according to parameters and methods chosen by input file
      if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
        BeamContactNonlinearSolve();
      else // standard procedure without beam contact
      {
        //pay attention: for a constant predictor an incremental velocity update is necessary, which has
        //been deleted out of the code in order to simplify it
        Predict();

        if(ndim_ ==3)
          PTC();
        else
          FullNewton();
      }

      /*if iterations have not converged a new trial requires setting all intern element variables, statmechmanager class variables
       *and the state of the discretization to status at the beginning of this time step*/
      StatMechRestoreConvState();
    }
    while(!isconverged_);

    // update all that is relevant
    UpdateAndOutput();

    //special output for statistical mechanics
    StatMechOutput();
  }

  return;
} // void STR::TimIntStatMech::Integrate()


/*----------------------------------------------------------------------*
 |does what it says it does                       (public) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::UpdateAndOutput()
{
  //periodic shift of configuration at the end of the time step in order to avoid improper output
  statmechman_->PeriodicBoundaryShift(*disn_, ndim_, (*dt_)[0]);

  // calculate stresses, strains and energies
  // note: this has to be done before the update since otherwise a potential
  // material history is overwritten
  PrepareOutput();

  // update displacements, velocities, accelerations
  // after this call we will have disn_==dis_, etc
  UpdateStepState();

  // update beam contact
  if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
  {
    UpdateStepBeamContact();
  }

  // update time and step
  UpdateStepTime();

  // update everything on the element level
  UpdateStepElement();

  // write output
  OutputStep();

  // print info about finished time step
  PrintStep();
  return;
}//UpdateAndOutput()

/*----------------------------------------------------------------------*
 |do consistent predictor step for Brownian dynamics (public)cyron 10/09|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::Predict()
{
  // -------------------------------------------------------------------
  // get some parameters from parameter list
  // -------------------------------------------------------------------
  //consistent predictor for backward Euler time integration scheme (theta==1.0)
  if(pred_==INPAR::STR::pred_constdis)
  {
    PredictConstDisConsistVel();
    // note: currently just c&p, have to look into it...
    normdisi_ = 1.0e6;
    normpres_ = 1.0e6;
  }
  else
    dserror("Trouble in determining predictor %i", pred_);

  // Apply Dirichlet Boundary Conditions
  ApplyDirichletBC(timen_, disn_, veln_);

  // calculate internal force and stiffness matrix
  bool predict = true;
  EvaluateForceStiffResidual(predict);

  // reactions are negative to balance residual on DBC
  freact_->Update(-1.0, *fres_, 0.0);
// not necessary, since fres_ DBC DOFs were already blanked
//  dbcmaps_->InsertOtherVector(dbcmaps_->ExtractOtherVector(zeros_), freact_);

  //------------------------------------------------ build residual norm
  CalcRefNorms();

  PrintPredictor();

  return;
} //STR::TimIntStatMech::Predict()

/*----------------------------------------------------------------------*
 |  predictor                                     (public) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::PredictConstDisConsistVel()
{
  // time step size
  const double dt = (*dt_)[0];

  /*special part for STATMECH: initialize disn_ and veln_ with zero; this is necessary only for the following case:
   * assume that an iteration step did not converge and is repeated with new random numbers; if the failure of conver
   * gence lead to disn_ = NaN and veln_ = NaN this would affect also the next trial as e.g. disn_->Update(1.0,*((*dis_)(0)),0.0);
   * would set disn_ to NaN as even 0*NaN = NaN!; this would defeat the purpose of the repeated iterations with new
   * random numbers and has thus to be avoided; therefore we initialized disn_ and veln_ with zero which has no effect
   * in any other case*/
  disn_->PutScalar(0.0);
  veln_->PutScalar(0.0);

  // constant predictor : displacement in domain
  disn_->Update(1.0, *(*dis_)(0), 0.0);

  // new end-point velocities
  veln_->Update(1.0/(theta_*dt), *disn_,
                -1.0/(theta_*dt), *(*dis_)(0),
                0.0);
  veln_->Update(-(1.0-theta_)/theta_, *(*vel_)(0),
                1.0);

  // no accelerations (1st order differential equation)
//  accn_->Update(1.0/(theta_*theta_*dt*dt), *disn_,
//                -1.0/(theta_*theta_*dt*dt), *(*dis_)(0),
//                0.0);
//  accn_->Update(-1.0/(theta_*theta_*dt), *(*vel_)(0),
//                -(1.0-theta_)/theta_, *(*acc_)(0),
//                1.0);
  return;
}//STR::TimIntStatMech::PredictConstDisConsistVel()

/*----------------------------------------------------------------------*
 | apply Dirichlet Boundary Conditions            (public) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::ApplyDirichletBC(const double                time,
                                           Teuchos::RCP<Epetra_Vector> dis,
                                           Teuchos::RCP<Epetra_Vector> vel)
{
  // needed parameters
  ParameterList p;
  p.set("total time", time);  // target time
  p.set("delta time", (*dt_)[0]);

  // set vector values needed by elements
  discret_->ClearState();

  discret_->SetState("displacement",disn_);
  discret_->SetState("velocity",veln_);
  // predicted dirichlet values
  // disn then also holds prescribed new dirichlet displacements

  // determine DBC evaluation mode (new vs. old)
  statmechman_->EvaluateDirichletStatMech(p, disn_, dbcmaps_);

  discret_->ClearState();

  return;
} //STR::TimIntStatMech::ApplyDirichletBC()

/*----------------------------------------------------------------------*
 |  evaluate residual                             (public) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::EvaluateForceStiffResidual(bool predict)
{
  // theta-interpolate state vectors
  EvaluateMidState();

  // build new external forces
  ApplyForceExternal(timen_, (*dis_)(0), disn_, (*vel_)(0), fextn_, stiff_);

  // additional external forces are added (e.g. interface forces)
  fextn_->Update(1.0, *fifc_, 1.0);

  // internal forces and stiffness matrix
  ApplyForceStiffInternal(timen_, (*dt_)[0], disn_, disi_, veln_, fintn_, stiff_);

  // note: neglected in statmech...
  // inertial forces #finertt_
  //mass_->Multiply(false, *acct_, *finertt_);

  // viscous forces due Rayleigh damping
  if (damping_ == INPAR::STR::damp_rayleigh)
    damp_->Multiply(false, *velt_, *fvisct_);

  // Build residual
  BuildResidual();

  // evaluate beam contact
  if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
    ApplyForceStiffBeamContact(stiff_, fres_, disn_, predict);

  // blank residual at DOFs on Dirichlet BC already here (compare with strtimint_impl: there, this is first called on freact_, then again on fres_ which seems unnecessary)
  dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_), fres_);

  // build tangent matrix : effective dynamic stiffness matrix
  //    K_{Teffdyn} = 1/(theta*dt^2) M
  //                + 1/dt C
  //                + theta K_{T}
  // note : no mass terms in statmech case -> zeros in matrix -> Comment it(?)
  //stiff_->Add(*mass_, false, 1.0/(theta_*(*dt_)[0]*(*dt_)[0]), theta_);
  if (damping_ == INPAR::STR::damp_rayleigh)
    stiff_->Add(*damp_, false, 1.0/(*dt_)[0], 1.0);

  // close stiffness matrix
  stiff_->Complete();

//  // blank residual at DOFs on Dirichlet BC
//  {
//    Epetra_Vector frescopy(*fres_);
//    fres_->Multiply(1.0,*invtoggle_,frescopy,0.0);
//  }

  return;
}//STR::TimIntStatMech::EvaluateForceStiffResidual()

/*----------------------------------------------------------------------*
 | evaluate theta-state vectors by averaging end-point vectors          |
 |                                                (public) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::EvaluateMidState()
{
  // mid-displacements D_{n+1-alpha_f} (dism)
  //    D_{n+theta} := theta * D_{n+1} + (1-theta) * D_{n}
  dist_->Update(theta_, *disn_, 1.0-theta_, *(*dis_)(0), 0.0);

  // mid-velocities V_{n+1-alpha_f} (velm)
  //    V_{n+theta} := theta * V_{n+1} + (1-theta) * V_{n}
  velt_->Update(theta_, *veln_, 1.0-theta_, *(*vel_)(0), 0.0);

  // note: no accelerations in statmech...
  // mid-accelerations A_{n+1-alpha_m} (accm)
  //    A_{n+theta} := theta * A_{n+1} + (1-theta) * A_{n}
  // acct_->Update(theta_, *accn_, 1.0-theta_, *(*acc_)(0), 0.0);

  return;
}

/*----------------------------------------------------------------------*
 | internal forces and stiffness (public)                  mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::ApplyForceStiffInternal(const double                         time,
                                                  const double                         dt,
                                                  const Teuchos::RCP<Epetra_Vector>    dis,  // displacement state
                                                  const Teuchos::RCP<Epetra_Vector>    disi,  // residual displacements
                                                  const Teuchos::RCP<Epetra_Vector>    vel,  // velocity state
                                                  Teuchos::RCP<Epetra_Vector>          fint,  // internal force
                                                  Teuchos::RCP<LINALG::SparseOperator> stiff, // stiffness matrix
                                                  double                               t_eval) // time
{
  double t_evaluate = Teuchos::Time::wallTime();

  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  const std::string action = "calc_struct_nlnstiff";
  p.set("action", action);
  p.set("total time", time);
  p.set("delta time", dt);

  // reset displacement increments, internal forces and stiffness matrix in order to be evaluated anew
  //disi->PutScalar(0.0);
  fint->PutScalar(0.0);
  stiff->Zero();

  //passing statistical mechanics parameters to elements
  statmechman_->AddStatMechParamsTo(p, randomnumbers_);

  // set vector values needed by elements
  discret_->ClearState();
  // extended SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0,"residual displacement", disi);
  discret_->SetState(0,"displacement", dis);
  discret_->SetState(0,"velocity", vel);
  discret_->Evaluate(p, stiff, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  t_eval += timer_->WallTime() - t_evaluate;

  // that's it
  return;
}

/*----------------------------------------------------------------------*
 |  calculate reference norms for relative convergence checks           |
 |                                                (public) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::BuildResidual()
{
  // build residual  Res = M . A_{n+theta}
  //                     + C . V_{n+theta}
  //                     + F_{int;n+theta}
  //                     - F_{ext;n+theta}
  fres_->Update(-theta_, *fextn_, -(1.0-theta_), *fext_, 0.0);
  fres_->Update(theta_, *fintn_, (1.0-theta_), *fint_, 1.0);

  if (damping_ == INPAR::STR::damp_rayleigh)
    fres_->Update(1.0, *fvisct_, 1.0);
  // note: finertt_ is zero vector in statmech case
  // fres_->Update(1.0, *finertt_, 1.0);
  return;
}

/*----------------------------------------------------------------------*
 |  calculate reference norms for relative convergence checks           |
 |                                                (public) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::CalcRefNorms()
{
  normfres_ = STR::AUX::CalculateVectorNorm(iternorm_, fres_);
  // determine characteristic norms
  // we set the minumum of CalcRefNormForce() and #tolfres_, because
  // we want to prevent the case of a zero characteristic fnorm
  normcharforce_ = CalcRefNormForce();
  if (normcharforce_ == 0.0) normcharforce_ = tolfres_;
  normchardis_ = CalcRefNormDisplacement();
  if (normchardis_ == 0.0) normchardis_ = toldisi_;

  return;
}

/*----------------------------------------------------------------------*
 |  do Newton iteration (public)                             mwgee 03/07|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::FullNewton()
{
//  // -------------------------------------------------------------------
//  // get some parameters from parameter list
//  // -------------------------------------------------------------------
//  double time      = params_.get<double>("total time"             ,0.0);
//  double dt        = params_.get<double>("delta time"             ,0.01);
//  double timen     = time + dt;
//  int    maxiter   = params_.get<int>   ("max iterations"         ,10);
//  double alphaf    = 1.0-theta;
//  string convcheck = params_.get<string>("convcheck"              ,"AbsRes_Or_AbsDis");
//  double toldisp   = params_.get<double>("tolerance displacements",1.0e-07);
//  double tolres    = params_.get<double>("tolerance residual"     ,1.0e-07);
//  bool printscreen = params_.get<bool>  ("print to screen",true);
//  bool printerr    = params_.get<bool>  ("print to err",false);
//  FILE* errfile    = params_.get<FILE*> ("err file",NULL);
//  if (!errfile) printerr = false;
//
//  //------------------------------ turn adaptive solver tolerance on/off
//  const bool   isadapttol    = params_.get<bool>("ADAPTCONV",true);
//  const double adaptolbetter = params_.get<double>("ADAPTCONV_BETTER",0.01);
//
//
//#ifndef STRUGENALPHA_BE
//  //double delta = beta;
//#endif
//
//  //=================================================== equilibrium loop
//  int numiter=0;
//  double fresmnorm = 1.0e6;
//  double disinorm = 1.0e6;
//  fresm_->Norm2(&fresmnorm);
//  Epetra_Time timer(discret_->Comm());
//  timer.ResetStartTime();
//  bool print_unconv = true;
//
//  // create out-of-balance force for 2nd, 3rd, ... Uzawa iteration
//  if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
//    InitializeNewtonUzawa();
//
//  while (!Converged(convcheck, disinorm, fresmnorm, toldisp, tolres) and numiter<=maxiter)
//  {
//
//
//    //------------------------------------------- effective rhs is fresm
//    //---------------------------------------------- build effective lhs
//    //stiff_->Add(*damp_,false,(1.-alphaf)*gamma/(delta*dt),1.0);
//    //stiff_->Complete();
//
//    //backward Euler
//    stiff_->Complete();
//
//    //----------------------- apply dirichlet BCs to system of equations
//    disi_->PutScalar(0.0);  // Useful? depends on solver and more
//    LINALG::ApplyDirichlettoSystem(stiff_,disi_,fresm_,zeros_,dirichtoggle_);
//
//
//    //--------------------------------------------------- solve for disi
//    // Solve K_Teffdyn . IncD = -R  ===>  IncD_{n+1}
//    if (isadapttol && numiter)
//    {
//      double worst = fresmnorm;
//      double wanted = tolres;
//      solver_.AdaptTolerance(wanted,worst,adaptolbetter);
//    }
//    solver_.Solve(stiff_->EpetraOperator(),disi_,fresm_,true,numiter==0);
//    solver_.ResetTolerance();
//
//
//    //---------------------------------- update mid configuration values
//    // displacements
//    // D_{n+1-alpha_f} := D_{n+1-alpha_f} + (1-alpha_f)*IncD_{n+1}
//
//    dism_->Update(1.-alphaf,*disi_,1.0);
//    disn_->Update(1.0,*disi_,1.0);
//
//    // velocities
//
//    // incremental (required for constant predictor)
//
//    //backward Euler
//    velm_->Update(1.0/dt,*dism_,-1.0/dt,*((*dis_)(0)),0.0);
//
//    //velm_->Update(1.0,*dism_,-1.0,*((*dis_)(0)),0.0);
//    //velm_->Update((delta-(1.0-alphaf)*gamma)/delta,*vel_,gamma/(delta*dt));
//
//
//    //---------------------------- compute internal forces and stiffness
//    {
//      // zero out stiffness
//      stiff_->Zero();
//      // create the parameters for the discretization
//      ParameterList p;
//      // action for elements
//      p.set("action","calc_struct_nlnstiff");
//      // other parameters that might be needed by the elements
//      p.set("total time",timen);
//      p.set("delta time",dt);
//      p.set("alpha f",1-theta_);
//
//      //passing statistical mechanics parameters to elements
////      p.set("ETA",(statmechman_->statmechparams_).get<double>("ETA",0.0));
////      p.set("THERMALBATH",DRT::INPUT::IntegralValue<INPAR::STATMECH::ThermalBathType>(statmechman_->statmechparams_,"THERMALBATH"));
////      p.set<int>("FRICTION_MODEL",DRT::INPUT::IntegralValue<INPAR::STATMECH::FrictionModel>(statmechman_->statmechparams_,"FRICTION_MODEL"));
////      p.set("RandomNumbers",randomnumbers_);
////      p.set("SHEARAMPLITUDE",(statmechman_->statmechparams_).get<double>("SHEARAMPLITUDE",0.0));
////      p.set("CURVENUMBER",(statmechman_->statmechparams_).get<int>("CURVENUMBER",-1));
////      p.set("STARTTIMEACT",(statmechman_->statmechparams_).get<double>("STARTTIMEACT",0.0));
////      p.set("DELTA_T_NEW",(statmechman_->statmechparams_).get<double>("DELTA_T_NEW",0.0));
////      p.set("OSCILLDIR",(statmechman_->statmechparams_).get<int>("OSCILLDIR",-1));
////      p.set("PERIODLENGTH",statmechman_->GetPeriodLength());
//      statmechman_->AddStatMechParamsTo(p, randomnumbers_);
//
//      // set vector values needed by elements
//      discret_->ClearState();
//
//      // scale IncD_{n+1} by (1-alphaf) to obtain mid residual displacements IncD_{n+1-alphaf}
//      disi_->Scale(1.-alphaf);
//
//      discret_->SetState("residual displacement",disi_);
//
//      discret_->SetState("displacement",dism_);
//      discret_->SetState("velocity",velm_);
//
//      //discret_->SetState("velocity",velm_); // not used at the moment
//
//      fint_->PutScalar(0.0);  // initialise internal force vector
//      discret_->Evaluate(p,stiff_,null,fint_,null,null);
//
//      discret_->ClearState();
//
//      // do NOT finalize the stiffness matrix to add masses to it later
//    }
//
//    //------------------------------------------ compute residual forces
//
//    // dynamic residual
//    // Res =  C . V_{n+1-alpha_f}
//    //        + F_int(D_{n+1-alpha_f})
//    //        - F_{ext;n+1-alpha_f}
//    // add mid-inertial force
//
//
//    //RefCountPtr<Epetra_Vector> fviscm = LINALG::CreateVector(*dofrowmap,true);
//    fresm_->Update(-1.0,*fint_,1.0,*fextm_,0.0);
//
//    //**********************************************************************
//    //**********************************************************************
//    // evaluate beam contact
//    if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
//    {
//      beamcman_->Evaluate(*SystemMatrix(),*fresm_,*disn_);
//
//#ifdef GMSHNEWTONSTEPS
//      // Create gmsh-output to visualize every step of newton iteration
//      int step  = params_.get<int>("step",0);
//      int istep = step + 1;
//      beamcman_->GmshOutput(*disn_,istep,numiter+1);
//      beamcman_->ConsoleOutput();
//#endif
//    }
//    //**********************************************************************
//    //**********************************************************************
//
//
//    // blank residual DOFs that are on Dirichlet BC
//    {
//      Epetra_Vector fresmcopy(*fresm_);
//      fresm_->Multiply(1.0,*invtoggle_,fresmcopy,0.0);
//    }
//
//    //---------------------------------------------- build residual norm
//    disi_->Norm2(&disinorm);
//    fresm_->Norm2(&fresmnorm);
//
//
//
//    //if code is compiled with DEBUG flag each iteration is written into file for Gmsh visualization
//#ifdef DEBUG
//    // first index = time step index
//    std::ostringstream filename;
//
//    //creating complete file name dependent on step number with 5 digits and leading zeros
//    if (numiter<100000)
//      filename << "./GmshOutput/konvergenz"<< std::setw(5) << setfill('0') << numiter <<".pos";
//    else
//      dserror("Gmsh output implemented for a maximum of 99999 steps");
//
//    //statmechman_->GmshOutput(*dism_,filename,numiter);
//#endif  // #ifdef DEBUG
//
//
//    // a short message
//    if (!myrank_ and (printscreen or printerr))
//    {
//      PrintNewton(printscreen,printerr,print_unconv,errfile,timer,numiter,maxiter,
//                  fresmnorm,disinorm,convcheck);
//    }
//
//    //--------------------------------- increment equilibrium loop index
//    ++numiter;
//
//  }
//  //=============================================== end equilibrium loop
//  print_unconv = false;
//
//  //-------------------------------- test whether max iterations was hit
//  //if on convergence arises within maxiter iterations the time step is restarted with new random numbers
//  if (numiter>=maxiter)
//  {
//    isconverged_ = false;
//    statmechman_->UpdateNumberOfUnconvergedSteps();
//    if(discret_->Comm().MyPID() == 0)
//      std::cout<<"\n\niteration unconverged - new trial with new random numbers!\n\n";
//     //dserror("PTC unconverged in %d iterations",numiter);
//  }
//  else if(!myrank_ and printscreen)
//  {
//    PrintNewton(printscreen,printerr,print_unconv,errfile,timer,numiter,maxiter,
//                fresmnorm,disinorm,convcheck);
//  }
//
//
//  params_.set<int>("num iterations",numiter);

  return;
} // STR::TimIntStatMech::FullNewton()

/*----------------------------------------------------------------------*
 |  initialize Newton for 2nd, 3rd, ... Uzawa iteration      cyron 12/10|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::InitializeNewtonUzawa()
{
//  bool  loadlin    = params_.get<bool>("LOADLIN",false);

  // create out-of-balance force for 2nd, 3rd, ... Uzawa iteration
  if (beamcman_->GetUzawaIter() > 1)
  {
    //--------------------------- recompute external forces if nonlinear
    // at state n, the external forces and linearization are interpolated at
    // time 1-alphaf in a TR fashion
//    if (loadlin)
//    {
//      ParameterList p;
//      // action for elements
//      p.set("action","calc_struct_eleload");
//      // other parameters needed by the elements
//      p.set("total time",timen_);
//      p.set("delta time",(*dt_)[0]);
//      p.set("alpha f",1-theta_);
//      // set vector values needed by elements
//      discret_->ClearState();
//      discret_->SetState("displacement",disn_);
//      discret_->SetState("velocity",veln_);
////      discret_->SetState("displacement",dism_); // mid point
////      discret_->SetState("velocity",velm_);
//      fextn_->PutScalar(0.0); // TR
////      fextm_->PutScalar(0.0);
//      fextlin_->Zero();
////      discret_->EvaluateNeumann(p,fextm_,fextlin_);
//      discret_->EvaluateNeumann(p,fextn_,fextlin_);
//      fextlin_->Complete();
//      discret_->ClearState();
//      fextm_->Update(1.0,*fextn_,0.0,*fext_,0.0);
//    }

    //---------------------------- compute internal forces and stiffness
    {
      // zero out stiffness
      stiff_->Zero();
      // create the parameters for the discretization
      ParameterList p;
      // action for elements
      p.set("action","calc_struct_nlnstiff");
      // other parameters that might be needed by the elements
      p.set("total time",timen_);
      p.set("delta time",(*dt_)[0]);
      p.set("alpha f",1-theta_);

      //passing statistical mechanics parameters to elements
//      p.set("ETA",(statmechman_->statmechparams_).get<double>("ETA",0.0));
//      p.set("THERMALBATH",DRT::INPUT::IntegralValue<INPAR::STATMECH::ThermalBathType>(statmechman_->statmechparams_,"THERMALBATH"));
//      p.set<int>("FRICTION_MODEL",DRT::INPUT::IntegralValue<INPAR::STATMECH::FrictionModel>(statmechman_->statmechparams_,"FRICTION_MODEL"));
//      p.set("RandomNumbers",randomnumbers_);
//      p.set("SHEARAMPLITUDE",(statmechman_->statmechparams_).get<double>("SHEARAMPLITUDE",0.0));
//      p.set("CURVENUMBER",(statmechman_->statmechparams_).get<int>("CURVENUMBER",-1));
//      p.set("STARTTIMEACT",(statmechman_->statmechparams_).get<double>("STARTTIMEACT",0.0));
//      p.set("DELTA_T_NEW",(statmechman_->statmechparams_).get<double>("DELTA_T_NEW",0.0));
//      p.set("OSCILLDIR",(statmechman_->statmechparams_).get<int>("OSCILLDIR",-1));
//      p.set("PERIODLENGTH",statmechman_->GetPeriodLength());

      statmechman_->AddStatMechParamsTo(p, randomnumbers_);

      // set vector values needed by elements
      discret_->ClearState();

      // scale IncD_{n+1} by (1-alphaf) to obtain mid residual displacements IncD_{n+1-alphaf}
//      disi_->Scale(1.-alphaf);

      discret_->SetState("residual displacement",disi_);
      discret_->SetState("displacement",disn_);
      discret_->SetState("velocity",veln_);

      fint_->PutScalar(0.0);  // initialise internal force vector
      discret_->Evaluate(p,stiff_,null,fint_,null,null);

      discret_->ClearState();
    }

    //------------------------------------------ compute residual forces
    fres_->Update(-1.0,*fint_,1.0,*fextn_,0.0); // fext_ oder fextn_ ???
    //**********************************************************************
    //**********************************************************************
    // evaluate beam contact
    if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
      beamcman_->Evaluate(*SystemMatrix(),*fres_,*disn_);
    //**********************************************************************
    //**********************************************************************

    // blank residual DOFs that are on Dirichlet BC
    Epetra_Vector frescopy(*fres_);
    fres_->Multiply(1.0,*invtoggle_,frescopy,0.0);
  }

  return;
}//STR::TimIntStatMech::InitializeNewtonUzawa()

/*----------------------------------------------------------------------*
 |  read restart (public)                                    cyron 12/08|
 *----------------------------------------------------------------------*/
//void STR::TimIntStatMech::ReadRestart(int step)
//{
//  RCP<DRT::Discretization> rcpdiscret = rcp(&discret_,false);
//  IO::DiscretizationReader reader(rcpdiscret,step);
//  double time  = reader.ReadDouble("time");
//  int    rstep = reader.ReadInt("step");
//  if (rstep != step) dserror("Time step on file not equal to given step");
//
//  reader.ReadVector(dis_, "displacement");
//  reader.ReadVector(vel_, "velocity");
//  reader.ReadVector(acc_, "acceleration");
//  reader.ReadVector(fext_,"fexternal");
//  reader.ReadMesh(step);
//
//  // read restart information for contact
//  statmechmanager_->ReadRestart(reader);
//
//#ifdef INVERSEDESIGNUSE
//  int idrestart = -1;
//  idrestart = reader.ReadInt("InverseDesignRestartFlag");
//  if (idrestart==-1) dserror("expected inverse design restart flag not on file");
//  // if idrestart==0 then the file is from a INVERSEDESIGCREATE phase
//  // and we have to zero out the inverse design displacements.
//  // The stored reference configuration is on record at the element level
//  if (!idrestart)
//  {
//    dis_->PutScalar(0.0);
//    vel_->PutScalar(0.0);
//    acc_->PutScalar(0.0);
//  }
//#endif
//
//  // override current time and step with values from file
//  params_.set<double>("total time",time);
//  params_.set<int>   ("step",rstep);
//
//  if (surf_stress_man_->HaveSurfStress())
//    surf_stress_man_->ReadRestart(rstep, DRT::Problem::Instance()->InputControlFile()->FileName());
//
//  if (constrMan_->HaveConstraint())
//  {
//    double uzawatemp = reader.ReadDouble("uzawaparameter");
//    constrSolv_->SetUzawaParameter(uzawatemp);
//    RCP<Epetra_Map> constrmap=constrMan_->GetConstraintMap();
//    RCP<Epetra_Vector> tempvec = LINALG::CreateVector(*constrmap,true);
//    reader.ReadVector(tempvec, "lagrmultiplier");
//    constrMan_->SetLagrMultVector(tempvec);
//    reader.ReadVector(tempvec, "refconval");
//    constrMan_->SetRefBaseValues(tempvec,time);
//  }
//
//  return;
//}//STR::TimIntStatMech::ReadRestart()

/*----------------------------------------------------------------------*
 |  do output including statistical mechanics data(public)    cyron 12/08|
 *----------------------------------------------------------------------*/
//void STR::TimIntStatMech::Output()
//{
//  // -------------------------------------------------------------------
//  // get some parameters from parameter list
//  // -------------------------------------------------------------------
//  double timen         = params_.get<double>("total time"             ,0.0);
//  double dt            = params_.get<double>("delta time"             ,0.01);
//  double alphaf        = 1.0-theta_;
//  int    istep         = params_.get<int>   ("step"                   ,0);
//  int    nstep         = params_.get<int>   ("nstep"                  ,5);
//  int    numiter       = params_.get<int>   ("num iterations"         ,-1);
//
//  bool   iodisp        = params_.get<bool>  ("io structural disp"     ,true);
//  int    updevrydisp   = params_.get<int>   ("io disp every nstep"    ,10);
//  INPAR::STR::StressType iostress = DRT::INPUT::get<INPAR::STR::StressType>(params_, "io structural stress",INPAR::STR::stress_none);
//  int    updevrystress = params_.get<int>   ("io stress every nstep"  ,10);
//  INPAR::STR::StrainType iostrain      = DRT::INPUT::get<INPAR::STR::StrainType>(params_, "io structural strain",INPAR::STR::strain_none);
//  bool   iosurfactant  = params_.get<bool>  ("io surfactant"          ,false);
//
//  int    writeresevry  = params_.get<int>   ("write restart every"    ,0);
//
//  bool   printscreen   = params_.get<bool>  ("print to screen"        ,true);
//  bool   printerr      = params_.get<bool>  ("print to err"           ,true);
//  FILE*  errfile       = params_.get<FILE*> ("err file"               ,NULL);
//  if (!errfile) printerr = false;
//
//  bool isdatawritten = false;
//
//  //------------------------------------------------- write restart step
//  if ((writeresevry and istep%writeresevry==0) or istep==nstep)
//  {
//    output_.WriteMesh(istep,timen);
//    output_.NewStep(istep, timen);
//    output_.WriteVector("displacement",dis_);
//    output_.WriteVector("velocity",vel_);
//    output_.WriteVector("acceleration",acc_);
//    output_.WriteVector("fexternal",fext_);
//
//#ifdef INVERSEDESIGNCREATE // indicate that this restart is from INVERSEDESIGCREATE phase
//    output_.WriteInt("InverseDesignRestartFlag",0);
//#endif
//#ifdef INVERSEDESIGNUSE // indicate that this restart is from INVERSEDESIGNUSE phase
//    output_.WriteInt("InverseDesignRestartFlag",1);
//#endif
//
//    isdatawritten = true;
//
////____________________________________________________________________________________________________________
////note:the following block is the only difference to Output() in strugenalpha.cpp-----------------------------
///* write restart information for statistical mechanics problems; all the information is saved as class variables
// * of StatMechManager*/
//    statmechman_->WriteRestart(output_);
////------------------------------------------------------------------------------------------------------------
////____________________________________________________________________________________________________________
//
//    if (surf_stress_man_->HaveSurfStress())
//      surf_stress_man_->WriteRestart(istep, timen);
//
//    if (constrMan_->HaveConstraint())
//    {
//      output_.WriteDouble("uzawaparameter",constrSolv_->GetUzawaParameter());
//      output_.WriteVector("lagrmultiplier",constrMan_->GetLagrMultVector());
//      output_.WriteVector("refconval",constrMan_->GetRefBaseValues());
//    }
//
//    if (discret_->Comm().MyPID()==0 and printscreen)
//    {
//      cout << "====== Restart written in step " << istep << endl;
//      fflush(stdout);
//    }
//    if (errfile and printerr)
//    {
//      fprintf(errfile,"====== Restart written in step %d\n",istep);
//      fflush(errfile);
//    }
//  }
//
//  //----------------------------------------------------- output results
//  if (iodisp and updevrydisp and istep%updevrydisp==0 and !isdatawritten)
//  {
//    output_.NewStep(istep, timen);
//    output_.WriteVector("displacement",dis_);
//    output_.WriteVector("velocity",vel_);
//    output_.WriteVector("acceleration",acc_);
//    output_.WriteVector("fexternal",fext_);
//    output_.WriteElementData();
//
//    if (surf_stress_man_->HaveSurfStress() and iosurfactant)
//      surf_stress_man_->WriteResults(istep,timen);
//
//    isdatawritten = true;
//  }
//
//  //------------------------------------- do stress calculation and output
//  if (updevrystress and !(istep%updevrystress) and iostress!=INPAR::STR::stress_none)
//  {
//    // create the parameters for the discretization
//    ParameterList p;
//    // action for elements
//    p.set("action","calc_struct_stress");
//    // other parameters that might be needed by the elements
//    p.set("total time",timen);
//    p.set("delta time",dt);
//    p.set("alpha f",1-theta_);
//    Teuchos::RCP<std::vector<char> > stress = Teuchos::rcp(new std::vector<char>());
//    Teuchos::RCP<std::vector<char> > strain = Teuchos::rcp(new std::vector<char>());
//    p.set("stress", stress);
//    p.set<int>("iostress", iostress);
//    p.set("strain", strain);
//    p.set<int>("iostrain", iostrain);
//    // set vector values needed by elements
//    discret_->ClearState();
//    discret_->SetState("residual displacement",zeros_);
//    discret_->SetState("displacement",dis_);
//    discret_->SetState("velocity",vel_);
//    discret_->Evaluate(p,null,null,null,null,null);
//    discret_->ClearState();
//    if (!isdatawritten) output_.NewStep(istep, timen);
//    isdatawritten = true;
//
//    switch (iostress)
//    {
//    case INPAR::STR::stress_cauchy:
//      output_.WriteVector("gauss_cauchy_stresses_xyz",*stress,*discret_->ElementRowMap());
//      break;
//    case INPAR::STR::stress_2pk:
//      output_.WriteVector("gauss_2PK_stresses_xyz",*stress,*discret_->ElementRowMap());
//      break;
//    case INPAR::STR::stress_none:
//      break;
//    default:
//      dserror ("requested stress type not supported");
//    }
//
//    switch (iostrain)
//    {
//    case INPAR::STR::strain_ea:
//      output_.WriteVector("gauss_EA_strains_xyz",*strain,*discret_->ElementRowMap());
//      break;
//    case INPAR::STR::strain_gl:
//      output_.WriteVector("gauss_GL_strains_xyz",*strain,*discret_->ElementRowMap());
//      break;
//    case INPAR::STR::strain_none:
//      break;
//    default:
//      dserror("requested strain type not supported");
//    }
//  }
//
//  //---------------------------------------------------------- print out
//  if (!myrank_)
//  {
//    if (printscreen)
//    {
//      printf("step %6d | nstep %6d | time %-14.8E | dt %-14.8E | numiter %3d\n",
//             istep,nstep,timen,dt,numiter);
//      printf("----------------------------------------------------------------------------------\n");
//      fflush(stdout);
//    }
//    if (printerr)
//    {
//      fprintf(errfile,"step %6d | nstep %6d | time %-14.8E | dt %-14.8E | numiter %3d\n",
//              istep,nstep,timen,dt,numiter);
//      fprintf(errfile,"----------------------------------------------------------------------------------\n");
//      fflush(errfile);
//    }
//  }
//  return;
//}//STR::TimIntStatMech::Output()

/*----------------------------------------------------------------------*
 |  Pseudo Transient Continuation                 (public)   cyron 12/10|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::PTC()
{
  //---------------------------------------------------------------some sanity checks
  if (not stiff_->Filled())
    dserror("Effective stiffness matrix must be filled here");
  //------------------------------------------------------------ for time measurement
  double sumsolver     = 0;
  double sumevaluation = 0;
  double sumptc = 0;

  const double tbegin = Teuchos::Time::wallTime();

  //--------------------create out-of-balance force for 2nd, 3rd, ... Uzawa iteration
  if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
    InitializeNewtonUzawa();

  //=================================================== equilibrium loop
  // initialise equilibrium loop
  iter_ = 1;
  normfres_ = CalcRefNormForce();
  // normdisi_ was already set in predictor; this is strictly >0
  timer_->ResetStartTime();

  //-----------------------------parameters from statistical mechanics parameter list
  // hard wired ptc parameters
  Teuchos::ParameterList statmechparams = statmechman_->GetStatMechParams();
  double ctransptc = statmechparams.get<double>("CTRANSPTC0",0.0);
  // crotptc is used here as equivalent to dti of the PTC scheme
  double crotptc   = statmechparams.get<double>("CROTPTC0",0.145);
  double alphaptc  = statmechparams.get<double>("ALPHAPTC",6.0);

  // PTC parameters
  double nc;
  fres_->NormInf(&nc);
  double resinit = nc;

  //printf("fresnorm %10.5e disinorm %10.5e nc %10.5e\n",normfres_,normdisi_,nc);

  // flag indicating whether or not the iterative loop was left because the residual norm was going to diverge anyway
  bool fresnormdivergent = false;

  //parameters to make sure that in last iteration step botch PTC parameters have reached zero
  double ctransptcold = ctransptc;
  double crotptcold   = crotptc;

  while (((!Converged() || ctransptcold > 0.0 || crotptcold > 0.0) and iter_<=itermax_) or (iter_ <= itermin_))
  {
    //save PTC parameters of the so far last iteration step
    ctransptcold = ctransptc;
    crotptcold   = crotptc;

    // make negative residual
    fres_->Scale(-1.0);

    //backward Euler
    stiff_->Complete();

    //the following part was especially introduced for Brownian dynamics
    PTCBrownianForcesAndDamping((*dt_)[0],crotptc,ctransptc, sumptc);

    //----------------------- apply dirichlet BCs to system of equations
    disi_->PutScalar(0.0);  // Useful? depends on solver and more

    LINALG::ApplyDirichlettoSystem(stiff_,disi_,fres_,zeros_,*(dbcmaps_->CondMap()));

    //--------------------------------------------------- solve for disi
    const double t_solver = Teuchos::Time::wallTime();
    // Solve K_Teffdyn . IncD = -R  ===>  IncD_{n+1}
    if (solveradapttol_ and (iter_ > 1))
    {
      double worst = normfres_;
      double wanted = tolfres_;
      solver_->AdaptTolerance(wanted, worst, solveradaptolbetter_);
    }
    solver_->Solve(stiff_->EpetraOperator(),disi_,fres_,true,iter_==1);
    solver_->ResetTolerance();

    sumsolver += Teuchos::Time::wallTime() - t_solver;

    // update displacements and velocities for this iteration step
    UpdateIter(iter_);

    //---------------- compute internal forces, stiffness and residual
    EvaluateForceStiffResidual();

    // reactions are negative to balance residual on DBC
    // note: due to the use of the old "dirichtoggle_" vector, fres_ dofs with DBCs have already been blanked
    freact_->Update(-1.0, *fres_, 0.0);

    //---------------------------------------------- build residual norm
    // build residual displacement norm
    normdisi_ = STR::AUX::CalculateVectorNorm(iternorm_, disi_);
    // build residual force norm
    normfres_ = STR::AUX::CalculateVectorNorm(iternorm_, fres_);

    // update dti_ of the PTC scheme
    dti_ = crotptc;
    PrintNewtonIter();

    //------------------------------------ PTC update of artificial time
    // compute inf norm of residual
    PTCStatMechUpdate(ctransptc,crotptc,nc,resinit,alphaptc);

#ifdef GMSHPTCSTEPS
    // GmshOutput
    std::ostringstream filename;
    if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"GMSHOUTPUT") && DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
    {
      filename << "./GmshOutput/network"<< time_ <<"_u"<<std::setw(2) << setfill('0')<<beamcman_->GetUzawaIter()<<"_n"<<std::setw(2) << setfill('0')<<numiter<<".pos";
      statmechman_->GmshOutput(*disn_,filename,istep,beamcman_);
    }
    else
    {
      filename << "./GmshOutput/network"<< time_ <<"_n"<<std::setw(2) << setfill('0')<<numiter<<".pos";
      statmechman_->GmshOutput(*disn_,filename,istep);
    }
#endif
    //--------------------------------- increment equilibrium loop index
    ++iter_;

    // leave the loop without going to maxiter iteration because most probably, the process will not converge anyway from here on
    if(normfres_>1.0e4 && iter_>3)
    {
      fresnormdivergent = true;
      break;
    }
  }
  //============================================= end equilibrium loop
  // iter_ started at 1, so "--"
  iter_--;
//  print_unconv = false;

  //-------------------------------- test whether max iterations was hit
  PTCConvergenceStatus(iter_, itermax_, fresnormdivergent);

  INPAR::CONTACT::SolvingStrategy soltype = INPAR::CONTACT::solution_penalty;
  if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
    soltype = DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(beamcman_->InputParameters(),"STRATEGY");
  if(printscreen_ && !isconverged_ &&  !myrank_ && soltype != INPAR::CONTACT::solution_auglag)
    std::cout<<"\n\niteration unconverged - new trial with new random numbers!\n\n";
  if(isconverged_  and !myrank_ and printscreen_)
    PrintNewtonIter();

  if(!myrank_ and printscreen_)
    std::cout << "\n***\nevaluation time: " << sumevaluation<< " seconds\nptc time: "<< sumptc <<" seconds\nsolver time: "<< sumsolver <<" seconds\ntotal solution time: "<<Teuchos::Time::wallTime() - tbegin<<" seconds\n***\n";
  return;
} // STR::TimIntStatMech::PTC()

/*----------------------------------------------------------------------*
 |  evaluate outcome of PTC and chose action accordingly   mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::PTCBrownianForcesAndDamping(double& dt, double& crotptc, double& ctransptc, double& sumptc)
{
  const double t_ptc = Teuchos::Time::wallTime();
  // create the parameters for the discretization
  ParameterList p;

  p.set("action","calc_struct_ptcstiff");
  p.set("delta time",dt);
  p.set("crotptc",crotptc);
  p.set("ctransptc",ctransptc);

  //add statistical vector to parameter list for statistical forces and damping matrix computation
//  p.set("ETA",(statmechman_->statmechparams_).get<double>("ETA",0.0));
//  p.set("THERMALBATH",DRT::INPUT::IntegralValue<INPAR::STATMECH::ThermalBathType>(statmechman_->statmechparams_,"THERMALBATH"));
//  p.set<int>("FRICTION_MODEL",DRT::INPUT::IntegralValue<INPAR::STATMECH::FrictionModel>(statmechman_->statmechparams_,"FRICTION_MODEL"));
//  p.set("SHEARAMPLITUDE",(statmechman_->statmechparams_).get<double>("SHEARAMPLITUDE",0.0));
//  p.set("CURVENUMBER",(statmechman_->statmechparams_).get<int>("CURVENUMBER",-1));
//  p.set("OSCILLDIR",(statmechman_->statmechparams_).get<int>("OSCILLDIR",-1));
//  p.set("PERIODLENGTH",statmechman_->GetPeriodLength());

  statmechman_->AddStatMechParamsTo(p);

  //evaluate ptc stiffness contribution in all the elements
  discret_->Evaluate(p,stiff_,null,null,null,null);

  sumptc += Teuchos::Time::wallTime() - t_ptc;

  return;
}

/*----------------------------------------------------------------------*
 |  update of rot. and transl. ptc damping for statmech    mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::PTCStatMechUpdate(double& ctransptc, double& crotptc, double& nc, double& resinit, double& alphaptc)
{
  double np;
  fres_->NormInf(&np);

  // SER step size control
  crotptc *= pow((np/nc),alphaptc);
  ctransptc *= pow((np/nc),alphaptc);
  nc = np;

  // modification: turn of ptc once residual is small enough
  if(np < 0.001*resinit || iter_ > 5)
  {
    ctransptc = 0.0;
    crotptc = 0.0;
  }
  return;
}//STR::TimIntStatMech::PTCStatMechUpdate()

/*----------------------------------------------------------------------*
 |  evaluate outcome of PTC and chose action accordingly   mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::PTCConvergenceStatus(int& numiter, int& maxiter, bool fresnormdivergent)
{
  if(numiter>=maxiter || fresnormdivergent)
  {
    ConvergenceStatusUpdate();

    // Only augmented lagrange:
    // We take a look at the change in the contact constraint norm.
    // Reason: when the constraint tolerance is a relative measure (gap compared to the smaller of the two beam radii),
    // configurations arise, where (especially in network simulations) the constraint is fullfilled by almost all of the contact
    // pairs except for a very tiny number of pairs (often only 1 pair), where one radius is significantly smaller than the other
    // (pair linker/filament).
//    INPAR::CONTACT::SolvingStrategy soltype = DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(beamcman_->InputParameters(),"STRATEGY");
//    if(soltype==INPAR::CONTACT::solution_auglag)
//    {
//      double cnorm = 1e6;
//      // get the constraint norm and decrease penalty parameter
//      beamcman_->UpdateConstrNorm(&cnorm);
//
//      if(numiter>=maxiter)
//      {
//        // accept step starting from second uzawa step
//        if(cnorm<0.5 && beamcman_->GetUzawaIter()>=2 && fresnorm_<1e-2)
//          ConvergenceStatusUpdate(true,false);
//        else
//          ConvergenceStatusUpdate(false,false);
//      }
//      else if(fresnormdivergent)
//        ConvergenceStatusUpdate(false,false);
//    }
  }
  else
  {
    if(!myrank_ && printscreen_)
      cout<<"PTC converged with..."<<endl;
  }
  return;
}//STR::TimIntStatMech::PTCConvergenceStatus()

/*----------------------------------------------------------------------*
 |  incremental iteration update of state                  mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::UpdateIter(const int iter)  //!< iteration counter
{
  if (iter <= 1)
  {
    UpdateIterIncrementally();
  }
  else
  {
    UpdateIterIteratively();
  }
  // morning is broken
  return;
}

/*----------------------------------------------------------------------*
 |  incremental iteration update of state                  mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::UpdateIterIncrementally()
{
  // Auxiliar vector holding new velocities and accelerations
  // by extrapolation/scheme on __all__ DOFs. This includes
  // the Dirichlet DOFs as well. Thus we need to protect those
  // DOFs of overwriting; they already hold the
  // correctly 'predicted', final values.

  // this version leads to a segmentation fault if the time step has to be repeated...
  //Teuchos::RCP<Epetra_Vector> aux = LINALG::CreateVector(*dofrowmap_, false);

  Teuchos::RCP<Epetra_Vector> aux = Teuchos::rcp(new Epetra_Vector(*(discret_->DofRowMap()), false));


  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->Update(1.0, *disi_, 1.0);

  // new end-point velocities
  aux->Update(1.0/(theta_*(*dt_)[0]), *disn_,
               -1.0/(theta_*(*dt_)[0]), *(*dis_)(0),
               0.0);
  aux->Update(-(1.0-theta_)/theta_, *(*vel_)(0), 1.0);
  // put only to free/non-DBC DOFs
  // old version
  veln_->Multiply(1.0, *invtoggle_,*aux, 0.0);

  // new version of updating velocity vector
  //dbcmaps_->InsertOtherVector(dbcmaps_->ExtractOtherVector(aux), veln_);

  // note: no accelerations in statmech...
//  // new end-point accelerations
//  aux->Update(1.0/(theta_*theta_*(*dt_)[0]*(*dt_)[0]), *disn_,
//              -1.0/(theta_*theta_*(*dt_)[0]*(*dt_)[0]), *(*dis_)(0),
//              0.0);
//  aux->Update(-1.0/(theta_*theta_*(*dt_)[0]), *(*vel_)(0),
//              -(1.0-theta_)/theta_, *(*acc_)(0),
//              1.0);
//  // put only to free/non-DBC DOFs
//  dbcmaps_->InsertOtherVector(dbcmaps_->ExtractOtherVector(aux), accn_);

  // bye
  return;
}

/*----------------------------------------------------------------------*
 |  iterative iteration update of state                    mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::UpdateIterIteratively()
{
  // new end-point displacements
  // D_{n+1}^{<k+1>} := D_{n+1}^{<k>} + IncD_{n+1}^{<k>}
  disn_->Update(1.0, *disi_, 1.0);

  // new end-point velocities
  veln_->Update(1.0/(theta_*(*dt_)[0]), *disi_, 1.0);

  // note: no accelerations in statmech...
//  // new end-point accelerations
//  accn_->Update(1.0/((*dt_)[0]*(*dt_)[0]*theta_*theta_), *disi_, 1.0);

  return;
}

/*----------------------------------------------------------------------*
 | set relevant variables signaling divergence of PTC      mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::ConvergenceStatusUpdate(bool converged, bool increasestepcount)
{
  if(!converged)
  {
    isconverged_ = false;
    if(increasestepcount)
      statmechman_->UpdateNumberOfUnconvergedSteps();
  }
  else
  {
    isconverged_ = true;
    if(!increasestepcount)
      statmechman_->UpdateNumberOfUnconvergedSteps(false);
  }
  return;
}

/*----------------------------------------------------------------------*
 | Precautions for Contact during one time step (private)  mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::BeamContactPrepareStep()
{
  if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT") && DRT::INPUT::IntegralValue<int>(beamcman_->InputParameters(),"BEAMS_NEWGAP"))
  {
    // set normal vector of last time "normal_" to old normal vector "normal_old_" (maybe this go inside the do loop?)
      beamcman_->ShiftAllNormal();
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate beam contact according to solution strategy                |
 |                                            (private)    mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::BeamContactNonlinearSolve()
{
  INPAR::CONTACT::SolvingStrategy soltype = DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(beamcman_->InputParameters(),"STRATEGY");
  switch (soltype)
  {
    //solving strategy using regularization with penalty method (nonlinear solution approach: ordinary NEWTON (PTC))
    case INPAR::CONTACT::solution_penalty:
      BeamContactPenalty();
    break;
    //solving strategy using regularization with augmented Lagrange method (nonlinear solution approach: nested UZAWA NEWTON (PTC))
    case INPAR::CONTACT::solution_auglag:
    {
      BeamContactAugLag();
    }
    break;
    default:
      dserror("Only penalty and augmented Lagrange implemented in statmech_time.cpp for beam contact");
      break;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate beam contact using Penalty appr. (private)    mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::BeamContactPenalty()
{
  Predict();

  if(ndim_ ==3)
    PTC();
  else
    FullNewton();

  beamcman_->UpdateConstrNorm();
  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate beam contact using Augmented Lagrange                      |
 |                                            (private)    mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::BeamContactAugLag()
{
  // get tolerance and maximum number of Uzawa steps from input file
  double eps = beamcman_->InputParameters().get<double>("UZAWACONSTRTOL");
  int maxuzawaiter = beamcman_->InputParameters().get<int>("UZAWAMAXSTEPS");

  // Initialize Lagrange Multipliers and Uzawa iteration counter for the Augmented Lagrangian loop
  beamcman_->ResetAlllmuzawa();
  beamcman_->ResetUzawaIter();

  if(!discret_->Comm().MyPID())
    cout<<"Predictor:"<<endl;
  Predict();

  // LOOP2: augmented Lagrangian (Uzawa)
  do
  {
    // increase iteration index
    beamcman_->UpdateUzawaIter();

    // if unconverged
    if(BeamContactExitUzawaAt(maxuzawaiter))
      break;

    if (discret_->Comm().MyPID() == 0)
      cout << endl << "Starting Uzawa step No. " << beamcman_->GetUzawaIter() << endl;

    if(ndim_ ==3)
      PTC();
    else
      FullNewton();

    // in case uzawa step did not converge
    if(!isconverged_)
    {
      if(!discret_->Comm().MyPID())
        std::cout<<"\n\nNewton iteration in Uzawa Step "<<beamcman_->GetUzawaIter()<<" unconverged - leaving Uzawa loop and restarting time step...!\n\n";
      // reset pairs to size 0 since the octree is being constructed completely anew
      beamcman_->ResetPairs();
      break;
    }

    // update constraint norm and penalty parameter
    beamcman_->UpdateConstrNorm();
    // update Uzawa Lagrange multipliers
    beamcman_->UpdateAlllmuzawa();
  } while (abs(beamcman_->GetConstrNorm()) >= eps);

  // reset penalty parameter
  beamcman_->ResetCurrentpp();
  return;
}

/*----------------------------------------------------------------------*
 |  Check Uzawa convergence                      (private) mueller 02/12|
 *----------------------------------------------------------------------*/
bool STR::TimIntStatMech::BeamContactExitUzawaAt(int& maxuzawaiter)
{
  bool exituzawa = false;
//
  if (beamcman_->GetUzawaIter() > maxuzawaiter)
  {
    isconverged_ = false;
    // note pair crosslinker/filament is the problem here. Since relative constraint norm, half of the crosslinker radius is ok
    if(beamcman_->GetConstrNorm()<0.5)
      isconverged_ = true;
    else
      cout << "Uzawa unconverged in "<< beamcman_->GetUzawaIter() << " iterations" << endl;
    exituzawa = true;
    //dserror("Uzawa unconverged in %d iterations",maxuzawaiter);
  }
  return exituzawa;
}

/*----------------------------------------------------------------------*
 | Precautions for Contact during one time step (private) mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::StatMechPrepareStep()
{
  if(HaveStatMech())
  {
    Teuchos::ParameterList statmechparams = statmechman_->GetStatMechParams();

    // special preparations for the very first step
    if(step_ == 0)
    {
      /* In case we add an initial amount of already linked crosslinkers, we have to build the octree
       * even before the first statmechman_->Update() call because the octree is needed to decide
       * whether links can be set...*/
      if(statmechparams.get<int>("INITOCCUPIEDBSPOTS",0)>0)
        statmechman_->SetInitialCrosslinkers(beamcman_);

      if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"GMSHOUTPUT"))
      {
        std::ostringstream filename;
          filename << "./GmshOutput/networkInit.pos";
        statmechman_->GmshOutput(*((*dis_)(0)),filename,step_);
      }
    }

    // statmechman_ has its own clock, so we hand over the integrator time in order to keep it up to date.
    // Also, switch time step size at given point in time and update time variable in statmechmanager
    // note: point in time should be the time of the last converged time step (have to check, if maybe timen_)
    statmechman_->UpdateTimeAndStepSize((*dt_)[0],(*time_)[0]);

    //save relevant class variables at the beginning of this time step
    statmechman_->WriteConv();

    //seed random generators of statmechman_ to generate the same random numbers even if the simulation was interrupted by a restart
    statmechman_->SeedRandomGenerators(step_);

    if(!discret_->Comm().MyPID() && printscreen_)
      std::cout<<"\nbegin time step "<<step_+1<<":";
  }

  return;
}//StatMechPrepareStep()

/*----------------------------------------------------------------------*
 |  Call statmechmanager Update according to options chosen             |
 |                                            (private)    mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::StatMechUpdate()
{
  if(HaveStatMech())
  {
    Teuchos::ParameterList statmechparams = statmechman_->GetStatMechParams();
    //assuming that iterations will converge
    isconverged_ = true;

    const double t_admin = Teuchos::Time::wallTime();
    if(DRT::INPUT::IntegralValue<int>(statmechparams,"BEAMCONTACT"))
      statmechman_->Update(step_, (*dt_)[0], *((*dis_)(0)), stiff_,ndim_,beamcman_,buildoctree_, printscreen_);
    else
      statmechman_->Update(step_, (*dt_)[0], *((*dis_)(0)), stiff_,ndim_, Teuchos::null,false,printscreen_);

    // print to screen
    StatMechPrintUpdate(t_admin);

    /*multivector for stochastic forces evaluated by each element; the numbers of vectors in the multivector equals the maximal
     *number of random numbers required by any element in the discretization per time step; therefore this multivector is suitable
     *for synchrinisation of these random numbers in parallel computing*/
    randomnumbers_ = Teuchos::rcp( new Epetra_MultiVector(*(discret_->ElementColMap()),maxrandomnumbersperglobalelement_) );
    /*pay attention: for a constant predictor an incremental velocity update is necessary, which has been deleted out of the code in oder to simplify it*/
    //generate gaussian random numbers for parallel use with mean value 0 and standard deviation (2KT / dt)^0.5
    statmechman_->GenerateGaussianRandomNumbers(randomnumbers_,0,pow(2.0 * statmechparams.get<double>("KT",0.0) / (*dt_)[0],0.5));
  }

  return;
} //StatMechUpdate()

/*----------------------------------------------------------------------*
 | Print Statistical Mechanics Update To Screen (private)  mueller 03/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::StatMechPrintUpdate(const double& t_admin)
{
  if(!myrank_ && printscreen_)
  {
    std::cout<<"\nTime for update of crosslinkers                   : " << Teuchos::Time::wallTime() - t_admin<< " seconds";
    std::cout<<"\nTotal number of elements after crosslinker update : "<<discret_->NumGlobalElements();
    std::cout<<"\nNumber of unconverged steps since simulation start: "<<statmechman_->NumberOfUnconvergedSteps()<<"\n"<<endl;
  }
  return;
}//StatMechPrintUpdate()

/*----------------------------------------------------------------------*
 |  Call statmechmanager Output according to options chosen             |
 |                                            (private)    mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::StatMechOutput()
{
  if(HaveStatMech())
  {
    // note: "step_-1
    if(DRT::INPUT::IntegralValue<int>(statmechman_->GetStatMechParams(),"BEAMCONTACT"))
      statmechman_->Output(ndim_,(*time_)[0],step_-1,(*dt_)[0],*((*dis_)(0)),*fint_,beamcman_, printscreen_);
    else
      statmechman_->Output(ndim_,(*time_)[0],step_-1,(*dt_)[0],*((*dis_)(0)),*fint_, Teuchos::null, printscreen_);
  }
  return;
}// StatMechOutput()

/*----------------------------------------------------------------------*
 |  Reset relevant values, vectors, discretization before repeating     |
 |  the time step                             (private)    mueller 02/12|
 *----------------------------------------------------------------------*/
void STR::TimIntStatMech::StatMechRestoreConvState()
{
  if(HaveStatMech())
  {
    if(!isconverged_)
    {
      ParameterList p;
      p.set("action","calc_struct_reset_istep");
      discret_->Evaluate(p,null,null,null,null,null);
      statmechman_->RestoreConv(stiff_, beamcman_);
      buildoctree_ = true;
    }
  }
  return;
} // StatMechRestoreConvState()

