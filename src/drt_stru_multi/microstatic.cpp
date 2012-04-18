/*!----------------------------------------------------------------------
\file microstatic.cpp
\brief Static control for  microstructural problems in case of multiscale
analyses

<pre>
Maintainer: Lena Yoshihara
            yoshihara@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15303
</pre>

*----------------------------------------------------------------------*/

#ifdef CCADISCRET

#include <Epetra_LinearProblem.h>

#ifndef HAVENOT_UMFPACK
#include <Amesos_Umfpack.h>
#endif

#include "microstatic.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_condition.H"
#include "../drt_surfstress/drt_surfstress_manager.H"
#include "../drt_structure/stru_aux.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_so3/so_hex8.H"
#include "../drt_so3/so_shw6.H"
#include "../drt_lib/drt_elementtype.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_sparsematrix.H"
#include "../drt_io/io_control.H"
#include "../drt_io/io.H"
#include "../drt_structure/strtimint_impl.H"

/*----------------------------------------------------------------------*
 | general problem data                                                 |
 | global variable GENPROB genprob is defined in global_control.c       |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;

/*----------------------------------------------------------------------*
 |  ctor (public)|
 *----------------------------------------------------------------------*/
STRUMULTI::MicroStatic::MicroStatic(const int microdisnum,
                                    const double V0):
microdisnum_(microdisnum),
V0_(V0)
{
  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  discret_ = DRT::Problem::Instance(microdisnum_)->Dis(genprob.numsf,0);

  // set degrees of freedom in the discretization
  if (!discret_->Filled()) discret_->FillComplete();

  // -------------------------------------------------------------------
  // set some pointers and variables
  // -------------------------------------------------------------------
  // time step size etc. need to be consistent in both input files, we
  // choose to use the ones defined in the macroscale input file
  // while other parameters (like output options, convergence checks)
  // can be used individually from the microscale input file
  const Teuchos::ParameterList& sdyn_micro  = DRT::Problem::Instance(microdisnum_)->StructuralDynamicParams();
  const Teuchos::ParameterList& sdyn_macro  = DRT::Problem::Instance()->StructuralDynamicParams();

  // we can use here the parameters of the macroscale input file
  //const Teuchos::ParameterList& probtype = DRT::Problem::Instance()->ProblemTypeParams();

  // i/o options should be read from the corresponding micro-file
  const Teuchos::ParameterList& ioflags  = DRT::Problem::Instance(microdisnum_)->IOParams();

  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  // get the solver number used for structural solver
  const int linsolvernumber = sdyn_micro.get<int>("LINEAR_SOLVER");
  // check if the structural solver has a valid solver number
  if (linsolvernumber == (-1))
    dserror("no linear solver defined for structural field. Please set LINEAR_SOLVER in STRUCTURAL DYNAMIC to a valid number!");

  solver_ = Teuchos::rcp (new LINALG::Solver(DRT::Problem::Instance(microdisnum_)->SolverParams(linsolvernumber),
                                    discret_->Comm(),
                                    DRT::Problem::Instance()->ErrorFile()->Handle()));
  discret_->ComputeNullSpaceIfNecessary(solver_->Params());

  // -------------------------------------------------------------------
  // access dynamic / io / etc. parameters
  // -------------------------------------------------------------------
  // new time intgration implementation -> generalized alpha
  // parameters are located in a sublist
  if (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn_macro,"DYNAMICTYP") == INPAR::STR::dyna_genalpha)
  {
    const Teuchos::ParameterList& genalpha  = DRT::Problem::Instance()->StructuralDynamicParams().sublist("GENALPHA");
    beta_ = genalpha.get<double>("BETA");
    gamma_ = genalpha.get<double>("GAMMA");
    alpham_ = genalpha.get<double>("ALPHA_M");
    alphaf_ = genalpha.get<double>("ALPHA_F");
  }
  else
    dserror("multi-scale problems are only implemented for imr-like generalized alpha time integration schemes");

  INPAR::STR::PredEnum pred = DRT::INPUT::IntegralValue<INPAR::STR::PredEnum>(sdyn_micro, "PREDICT");
  pred_ = pred;
  combdisifres_ = DRT::INPUT::IntegralValue<INPAR::STR::BinaryOp>(sdyn_macro,"NORMCOMBI_RESFDISP");
  normtypedisi_ = DRT::INPUT::IntegralValue<INPAR::STR::ConvNorm>(sdyn_macro,"NORM_DISP");
  normtypefres_ = DRT::INPUT::IntegralValue<INPAR::STR::ConvNorm>(sdyn_macro,"NORM_RESF");
  INPAR::STR::VectorNorm iternorm = DRT::INPUT::IntegralValue<INPAR::STR::VectorNorm>(sdyn_micro,"ITERNORM");
  iternorm_ = iternorm;

  dt_    = sdyn_macro.get<double>("TIMESTEP");
  time_  = 0.0;
  timen_ = time_ + dt_;
  step_  = 0;
  stepn_ = step_ + 1;
  numstep_ = sdyn_macro.get<int>("NUMSTEP");
  maxiter_ = sdyn_micro.get<int>("MAXITER");
  numiter_ = -1;

  tolfres_ = sdyn_micro.get<double>("TOLRES");
  toldisi_ = sdyn_micro.get<double>("TOLDISP");
  printscreen_=(ioflags.get<int>("STDOUTEVRY"));


  restart_ = DRT::Problem::Instance()->Restart();
  restartevry_ = sdyn_macro.get<int>("RESTARTEVRY");
  iodisp_ = DRT::INPUT::IntegralValue<int>(ioflags,"STRUCT_DISP");
  resevrydisp_ = sdyn_micro.get<int>("RESULTSEVRY");
  INPAR::STR::StressType iostress = DRT::INPUT::IntegralValue<INPAR::STR::StressType>(ioflags,"STRUCT_STRESS");
  iostress_ = iostress;
  resevrystrs_ = sdyn_micro.get<int>("RESULTSEVRY");
  INPAR::STR::StrainType iostrain = DRT::INPUT::IntegralValue<INPAR::STR::StrainType>(ioflags,"STRUCT_STRAIN");
  iostrain_ = iostrain;
  INPAR::STR::StrainType ioplstrain = DRT::INPUT::IntegralValue<INPAR::STR::StrainType>(ioflags,"STRUCT_PLASTIC_STRAIN");
  ioplstrain_ = ioplstrain;
  iosurfactant_ = DRT::INPUT::IntegralValue<int>(ioflags,"STRUCT_SURFACTANT");

  isadapttol_ = (DRT::INPUT::IntegralValue<int>(sdyn_micro,"ADAPTCONV")==1);
  adaptolbetter_ = sdyn_micro.get<double>("ADAPTCONV_BETTER");

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  // -------------------------------------------------------------------
  if (!discret_->Filled()) discret_->FillComplete();
  const Epetra_Map* dofrowmap = discret_->DofRowMap();
  myrank_ = discret_->Comm().MyPID();

  // -------------------------------------------------------------------
  // create empty matrices
  // -------------------------------------------------------------------
  stiff_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap,81,true,true));

  // -------------------------------------------------------------------
  // create empty vectors
  // -------------------------------------------------------------------
  // a zero vector of full length
  zeros_ = LINALG::CreateVector(*dofrowmap,true);
  // vector of full length; for each component
  //                /  1   i-th DOF is supported, ie Dirichlet BC
  //    vector_i =  <
  //                \  0   i-th DOF is free
  dirichtoggle_ = LINALG::CreateVector(*dofrowmap,true);
  // opposite of dirichtoggle vector, ie for each component
  //                /  0   i-th DOF is supported, ie Dirichlet BC
  //    vector_i =  <
  //                \  1   i-th DOF is free
  invtoggle_ = LINALG::CreateVector(*dofrowmap,false);

  // displacements D_{n} at last time
  dis_ = LINALG::CreateVector(*dofrowmap,true);

  // displacements D_{n+1} at new time
  disn_ = LINALG::CreateVector(*dofrowmap,true);

  // mid-displacements D_{n+1-alpha_f}
  dism_ = LINALG::CreateVector(*dofrowmap,true);

  // iterative displacement increments IncD_{n+1}
  // also known as residual displacements
  disi_ = LINALG::CreateVector(*dofrowmap,true);

  // internal force vector F_int at different times
  fintm_ = LINALG::CreateVector(*dofrowmap,true);

  // dynamic force residual at mid-time R_{n+1-alpha}
  // also known as out-of-balance-force
  fresm_ = LINALG::CreateVector(*dofrowmap,false);

  // -------------------------------------------------------------------
  // create "empty" EAS history map
  //
  // -------------------------------------------------------------------
  {
    lastalpha_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> >);
    oldalpha_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> >);
    oldfeas_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> >);
    oldKaainv_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> >);
    oldKda_ = Teuchos::rcp(new std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> >);
  }

  // -------------------------------------------------------------------
  // call elements to calculate stiffness and mass
  // -------------------------------------------------------------------
  {
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action","calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time",timen_);
    p.set("delta time",dt_);
    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("residual displacement",zeros_);
    discret_->SetState("displacement",dis_);

    discret_->Evaluate(p,stiff_,null,fintm_,null,null);
    discret_->ClearState();
  }

  // Determine dirichtoggle_ and its inverse since boundary conditions for
  // microscale simulations are due to the MicroBoundary condition
  // (and not Dirichlet BC)

  STRUMULTI::MicroStatic::DetermineToggle();
  STRUMULTI::MicroStatic::SetUpHomogenization();

  // reaction force vector at different times
  freactm_ = LINALG::CreateVector(*pdof_,true);

  //----------------------- compute an inverse of the dirichtoggle vector
  invtoggle_->PutScalar(1.0);
  invtoggle_->Update(-1.0,*dirichtoggle_,1.0);

  if (V0 > 0.0)
    V0_ = V0;
  else
  {
    if (DRT::Problem::Instance()->Dis(0,0)->Comm().MyPID()==0)
      cout << "You have not specified the initial volume of the RVE with number "
           << microdisnum_ << ", therefore it will now be calculated.\n\n"
           << "CAUTION: This calculation works only for RVEs without holes penetrating the surface!\n"
           << endl;

    // -------------------------- Calculate initial volume of microstructure
    ParameterList p;
    // action for elements
    p.set("action","calc_init_vol");
    p.set("V0", 0.0);
    discret_->EvaluateCondition(p, null, null, null, null, null, "MicroBoundary");
    V0_ = p.get<double>("V0", -1.0);
    if (V0_ == -1.0)
      dserror("Calculation of initial volume failed");
  }

  // ------------------------- Calculate initial density of microstructure
  // the macroscopic density has to be averaged over the entire
  // microstructural reference volume

  // create the parameters for the discretization
  ParameterList par;
  // action for elements
  par.set("action","multi_calc_dens");
  // set density to zero
  par.set("homogdens", 0.0);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->Evaluate(par,null,null,null,null,null);
  discret_->ClearState();

  density_ = 1/V0_*par.get<double>("homogdens", 0.0);
  if (density_ == 0.0)
    dserror("Density determined from homogenization procedure equals zero!");

  return;
} // STRUMULTI::MicroStatic::MicroStatic


void STRUMULTI::MicroStatic::Predictor(LINALG::Matrix<3,3>* defgrd)
{
  if (pred_ == INPAR::STR::pred_constdis)
    PredictConstDis(defgrd);
  else if (pred_ == INPAR::STR::pred_tangdis)
    PredictTangDis(defgrd);
  else
    dserror("requested predictor not implemented on the micro-scale");
  return;
}


/*----------------------------------------------------------------------*
 |  do predictor step (public)                               mwgee 03/07|
 *----------------------------------------------------------------------*/
void STRUMULTI::MicroStatic::PredictConstDis(LINALG::Matrix<3,3>* defgrd)
{
  // apply new displacements at DBCs -> this has to be done with the
  // mid-displacements since the given macroscopic deformation
  // gradient is evaluated at the mid-point!
  {
    // dism then also holds prescribed new dirichlet displacements
    EvaluateMicroBC(defgrd, dism_);
    disn_->Update(1.0, *dism_, -alphaf_, *dis_, 0.);
    disn_->Scale(1.0/(1.0-alphaf_));
    discret_->ClearState();
  }

  //--------------------------------- set EAS internal data if necessary

  // this has to be done only once since the elements will remember
  // their EAS data until the end of the microscale simulation
  // (end of macroscopic iteration step)
  SetEASData();

  //------------- eval fint at interpolated state, eval stiffness matrix
  {
    // zero out stiffness
    stiff_->Zero();
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action","calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time",timen_);
    p.set("delta time",dt_);
    p.set("alpha f",alphaf_);
    // set vector values needed by elements
    discret_->ClearState();
    disi_->Scale(0.0);
    discret_->SetState("residual displacement",disi_);
    discret_->SetState("displacement",dism_);
    fintm_->PutScalar(0.0);  // initialise internal force vector

    discret_->Evaluate(p,stiff_,null,fintm_,null,null);
    discret_->ClearState();

    if (surf_stress_man_->HaveSurfStress())
    {
      p.set("surfstr_man", surf_stress_man_);
      surf_stress_man_->EvaluateSurfStress(p,dism_,disn_,fintm_,stiff_);
    }

    // complete stiffness matrix
    stiff_->Complete();

    // set norm of displacement increments
    normdisi_ = 1.0e6;
  }

  //-------------------------------------------- compute residual forces
  // add static mid-balance
  fresm_->Update(-1.0,*fintm_,0.0);

  // extract reaction forces
  int err = freactm_->Import(*fresm_, *importp_, Insert);
  if (err)
    dserror("Importing reaction forces of prescribed dofs using importer returned err=%d",err);

  // blank residual at DOFs on Dirichlet BC
  Epetra_Vector fresmcopy(*fresm_);
  fresm_->Multiply(1.0,*invtoggle_,fresmcopy,0.0);

  // store norm of residual
  normfres_ = STR::AUX::CalculateVectorNorm(iternorm_, fresm_);

  return;
} // STRUMULTI::MicroStatic::Predictor()


/*----------------------------------------------------------------------*
 |  do predictor step (public)                                  lw 01/09|
 *----------------------------------------------------------------------*/
void STRUMULTI::MicroStatic::PredictTangDis(LINALG::Matrix<3,3>* defgrd)
{
  // for displacement increments on Dirichlet boundary
  Teuchos::RCP<Epetra_Vector> dbcinc
    = LINALG::CreateVector(*(discret_->DofRowMap()), true);

  // copy last converged displacements
  dbcinc->Update(1.0, *dism_, 0.0);

  // apply new displacements at DBCs -> this has to be done with the
  // mid-displacements since the given macroscopic deformation
  // gradient is evaluated at the mid-point!
  {
    // dbcinc then also holds prescribed new dirichlet displacements
    EvaluateMicroBC(defgrd, dbcinc);
    discret_->ClearState();
  }

  // subtract the displacements of the last converged step
  // DBC-DOFs hold increments of current step
  // free-DOFs hold zeros
  dbcinc->Update(-1.0, *dism_, 1.0);

  //--------------------------------- set EAS internal data if necessary

  // this has to be done only once since the elements will remember
  // their EAS data until the end of the microscale simulation
  // (end of macroscopic iteration step)
  SetEASData();

  //------------- eval fint at interpolated state, eval stiffness matrix
  {
    // zero out stiffness
    stiff_->Zero();
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action","calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time",timen_);
    p.set("delta time",dt_);
    p.set("alpha f",alphaf_);
    // set vector values needed by elements
    discret_->ClearState();
    disi_->PutScalar(0.0);
    discret_->SetState("residual displacement",disi_);
    discret_->SetState("displacement",dism_);
    fintm_->PutScalar(0.0);  // initialise internal force vector

    discret_->Evaluate(p,stiff_,null,fintm_,null,null);
    discret_->ClearState();

    if (surf_stress_man_->HaveSurfStress())
    {
      p.set("surfstr_man", surf_stress_man_);
      surf_stress_man_->EvaluateSurfStress(p,dism_,disn_,fintm_,stiff_);
    }
  }

  stiff_->Complete();

  //-------------------------------------------- compute residual forces
  // add static mid-balance
  fresm_->Update(-1.0,*fintm_,0.0);

  // add linear reaction forces to residual
  {
    // linear reactions
    Teuchos::RCP<Epetra_Vector> freact
      = LINALG::CreateVector(*(discret_->DofRowMap()), true);
    stiff_->Multiply(false, *dbcinc, *freact);

    // add linear reaction forces due to prescribed Dirichlet BCs
    fresm_->Update(-1.0, *freact, 1.0);
  }

  // blank residual at DOFs on Dirichlet BC
  {
    Epetra_Vector fresmcopy(*fresm_);
    fresm_->Multiply(1.0,*invtoggle_,fresmcopy,0.0);
  }

  // apply Dirichlet BCs to system of equations
  disi_->PutScalar(0.0);
  stiff_->Complete();
  LINALG::ApplyDirichlettoSystem(stiff_,disi_,fresm_,zeros_,dirichtoggle_);

  // solve for disi_
  // Solve K_Teffdyn . IncD = -R  ===>  IncD_{n+1}
  solver_->Reset();
  solver_->Solve(stiff_->EpetraMatrix(), disi_, fresm_, true, true);
  solver_->Reset();

  // store norm of displacement increments
  normdisi_ = STR::AUX::CalculateVectorNorm(iternorm_, disi_);

  //---------------------------------- update mid configuration values
  // set Dirichlet increments in displacement increments
  disi_->Update(1.0, *dbcinc, 1.0);

  // displacements
  // note that disi is not Inc_D{n+1} but Inc_D{n+1-alphaf} since everything
  // on the microscale "lives" exclusively at the pseudo generalized
  // mid point! This is just a quasi-static problem!
  dism_->Update(1.0,*disi_,1.0);
  disn_->Update(1.0/(1.0-alphaf_),*disi_,1.0);

  // reset anything that needs to be reset at the element level

  // strictly speaking, this (as well as the resetting of disi) is not
  // mandatory here, we do it just to be in line with the classical
  // time intgrator sti. there tangdis is assumed to be a predictor only, no
  // update of EAS parameters etc is desired. perhaps this might be
  // changed when speed should be optimized later on.
  {
    // create the parameters for the discretization
    ParameterList p;
    p.set("action", "calc_struct_reset_istep");
    // go to elements
    discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                       Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }

  //------------- eval fint at interpolated state, eval stiffness matrix
  {
    // zero out stiffness
    stiff_->Zero();
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action","calc_struct_nlnstiff");
    // other parameters that might be needed by the elements
    p.set("total time",timen_);
    p.set("delta time",dt_);
    p.set("alpha f",alphaf_);
    // set vector values needed by elements
    discret_->ClearState();
    disi_->PutScalar(0.0);
    discret_->SetState("residual displacement",disi_);
    discret_->SetState("displacement",dism_);
    fintm_->PutScalar(0.0);  // initialise internal force vector

    discret_->Evaluate(p,stiff_,null,fintm_,null,null);
    discret_->ClearState();

    if (surf_stress_man_->HaveSurfStress())
    {
      p.set("surfstr_man", surf_stress_man_);
      surf_stress_man_->EvaluateSurfStress(p,dism_,disn_,fintm_,stiff_);
    }
  }

  //-------------------------------------------- compute residual forces
  // add static mid-balance
  fresm_->Update(-1.0,*fintm_,0.0);

  // extract reaction forces
  int err = freactm_->Import(*fresm_, *importp_, Insert);
  if (err)
    dserror("Importing reaction forces of prescribed dofs using importer returned err=%d",err);

  // blank residual at DOFs on Dirichlet BC
  Epetra_Vector fresmcopy(*fresm_);
  fresm_->Multiply(1.0,*invtoggle_,fresmcopy,0.0);

  // store norm of residual
  normfres_ = STR::AUX::CalculateVectorNorm(iternorm_, fresm_);

  return;
}

/*----------------------------------------------------------------------*
 |  do Newton iteration (public)                             mwgee 03/07|
 *----------------------------------------------------------------------*/
void STRUMULTI::MicroStatic::FullNewton()
{
  //=================================================== equilibrium loop
  numiter_ = 0;

  // if TangDis-Predictor is employed, the number of iterations needs
  // to be increased by one, since it involves already one solution of
  // the non-linear system!
  if (pred_ == INPAR::STR::pred_tangdis)
    numiter_++;

  // store norms of old displacements and maximum of norms of
  // internal, external and inertial forces (needed for relative convergence
  // check)
  CalcRefNorms();

  Epetra_Time timer(discret_->Comm());
  timer.ResetStartTime();
  bool print_unconv = true;

  while (!Converged() && numiter_<=maxiter_)
  {

    //----------------------- apply dirichlet BCs to system of equations
    disi_->PutScalar(0.0);  // Useful? depends on solver and more

    LINALG::ApplyDirichlettoSystem(stiff_,disi_,fresm_,zeros_,dirichtoggle_);

    //--------------------------------------------------- solve for disi
    // Solve K_Teffdyn . IncD = -R  ===>  IncD_{n+1}
    if (isadapttol_ && numiter_)
    {
      double worst = normfres_;
      double wanted = tolfres_;
      solver_->AdaptTolerance(wanted,worst,adaptolbetter_);
    }
    solver_->Solve(stiff_->EpetraMatrix(),disi_,fresm_,true,numiter_==0);
    solver_->ResetTolerance();

    //---------------------------------- update mid configuration values
    // displacements
    // note that disi is not Inc_D{n+1} but Inc_D{n+1-alphaf} since everything
    // on the microscale "lives" exclusively at the pseudo generalized
    // mid point! This is just a quasi-static problem!
    dism_->Update(1.0,*disi_,1.0);
    disn_->Update(1.0/(1.0-alphaf_),*disi_,1.0);

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
      p.set("delta time",dt_);
      p.set("alpha f",alphaf_);
      // set vector values needed by elements
      discret_->ClearState();
      // we do not need to scale disi_ here with 1-alphaf (cf. strugenalpha), since
      // everything on the microscale "lives" at the pseudo generalized midpoint
      // -> we solve our quasi-static problem there and only update data to the "end"
      // of the time step after having finished a macroscopic dt
      discret_->SetState("residual displacement",disi_);
      discret_->SetState("displacement",dism_);
      fintm_->PutScalar(0.0);  // initialise internal force vector

      discret_->Evaluate(p,stiff_,null,fintm_,null,null);
      discret_->ClearState();

      if (surf_stress_man_->HaveSurfStress())
      {
        p.set("surfstr_man", surf_stress_man_);
        surf_stress_man_->EvaluateSurfStress(p,dism_,disn_,fintm_,stiff_);
      }
    }

    // complete stiffness matrix
    stiff_->Complete();

    //------------------------------------------ compute residual forces
    // add static mid-balance
    fresm_->Update(-1.0,*fintm_,0.0);

    // extract reaction forces
    int err = freactm_->Import(*fresm_, *importp_, Insert);
    if (err)
      dserror("Importing reaction forces of prescribed dofs using importer returned err=%d",err);

    // blank residual DOFs which are on Dirichlet BC
    Epetra_Vector fresmcopy(*fresm_);
    fresm_->Multiply(1.0,*invtoggle_,fresmcopy,0.0);

    //---------------------------------------------- build residual norm
    normdisi_ = STR::AUX::CalculateVectorNorm(iternorm_, disi_);

    normfres_ = STR::AUX::CalculateVectorNorm(iternorm_, fresm_);

  //--------------------------------- increment equilibrium loop index
    ++numiter_;
  }
  //============================================= end equilibrium loop
  print_unconv = false;

  //-------------------------------- test whether max iterations was hit
  if (numiter_>=maxiter_)
  {
     dserror("Newton unconverged in %d iterations",numiter_);
  }

  return;
} // STRUMULTI::MicroStatic::FullNewton()


/*----------------------------------------------------------------------*
 |  "prepare" output (public)                                   ly 09/11|
 *----------------------------------------------------------------------*/
void STRUMULTI::MicroStatic::PrepareOutput()
{
  if (resevrystrs_ and !(stepn_%resevrystrs_) and iostress_!=INPAR::STR::stress_none)
  {
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action","calc_struct_stress");
    // other parameters that might be needed by the elements
    p.set("total time",timen_);
    p.set("delta time",dt_);
    p.set("stress", stress_);
    p.set("strain", strain_);
    p.set("plstrain", plstrain_);
    p.set<int>("iostress", iostress_);
    p.set<int>("iostrain", iostrain_);
    p.set<int>("ioplstrain", ioplstrain_);
    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("residual displacement",zeros_);
    discret_->SetState("displacement",disn_);
    discret_->Evaluate(p,null,null,null,null,null);
    discret_->ClearState();
  }
}


/*----------------------------------------------------------------------*
 |  write output (public)                                       lw 02/08|
 *----------------------------------------------------------------------*/
void STRUMULTI::MicroStatic::Output(Teuchos::RCP<IO::DiscretizationWriter> output,
                                    const double time,
                                    const int step,
                                    const double dt)
{
  bool isdatawritten = false;

  //------------------------------------------------- write restart step
  if (restartevry_ and step%restartevry_==0)
  {
    output->WriteMesh(step,time);
    output->NewStep(step, time);
    output->WriteVector("displacement",dis_);
    isdatawritten = true;

    if (surf_stress_man_->HaveSurfStress())
      surf_stress_man_->WriteRestart(step, time);

    //Teuchos::RCP<std::vector<char> > lastalphadata = Teuchos::rcp(new std::vector<char>());

    Teuchos::RCP<Epetra_SerialDenseMatrix> emptyalpha = Teuchos::rcp(new Epetra_SerialDenseMatrix(1, 1));

    DRT::PackBuffer data;

    // note that the microstructure is (currently) serial only i.e. we
    // can use the GLOBAL number of elements!
    for (int i=0;i<discret_->NumGlobalElements();++i)
    {
      if ((*lastalpha_)[i]!=null)
      {
        DRT::ParObject::AddtoPack(data, *(*lastalpha_)[i]);
      }
      else
      {
        DRT::ParObject::AddtoPack(data, *emptyalpha);
      }
    }
    data.StartPacking();
    for (int i=0;i<discret_->NumGlobalElements();++i)
    {
      if ((*lastalpha_)[i]!=null)
      {
        DRT::ParObject::AddtoPack(data, *(*lastalpha_)[i]);
      }
      else
      {
        DRT::ParObject::AddtoPack(data, *emptyalpha);
      }
    }
    output->WriteVector("alpha", data(), *discret_->ElementColMap());
  }

  //----------------------------------------------------- output results
  if (iodisp_ && resevrydisp_ && step%resevrydisp_==0 && !isdatawritten)
  {
    output->NewStep(step, time);
    output->WriteVector("displacement",dis_);
    isdatawritten = true;

    if (surf_stress_man_->HaveSurfStress() && iosurfactant_)
      surf_stress_man_->WriteResults(step, time);
  }

  //------------------------------------- stress/strain output
  if (resevrystrs_ and !(step%resevrystrs_) and iostress_!=INPAR::STR::stress_none)
  {
    if (!isdatawritten) output->NewStep(step, time);
    isdatawritten = true;

    if (stress_ == Teuchos::null or strain_ == Teuchos::null or plstrain_ == Teuchos::null)
      dserror("Missing stresses and strains in micro-structural time integrator");

    switch (iostress_)
    {
    case INPAR::STR::stress_cauchy:
      output->WriteVector("gauss_cauchy_stresses_xyz",*stress_,*discret_->ElementRowMap());
      break;
    case INPAR::STR::stress_2pk:
      output->WriteVector("gauss_2PK_stresses_xyz",*stress_,*discret_->ElementRowMap());
      break;
    case INPAR::STR::stress_none:
      break;
    default:
      dserror ("requested stress type not supported");
    }

    switch (iostrain_)
    {
    case INPAR::STR::strain_ea:
      output->WriteVector("gauss_EA_strains_xyz",*strain_,*discret_->ElementRowMap());
      break;
    case INPAR::STR::strain_gl:
      output->WriteVector("gauss_GL_strains_xyz",*strain_,*discret_->ElementRowMap());
      break;
    case INPAR::STR::strain_none:
      break;
    default:
      dserror("requested strain type not supported");;
    }

    switch (ioplstrain_)
    {
    case INPAR::STR::strain_ea:
      output->WriteVector("gauss_pl_EA_strains_xyz",*plstrain_,*discret_->ElementRowMap());
      break;
    case INPAR::STR::strain_gl:
      output->WriteVector("gauss_pl_GL_strains_xyz",*plstrain_,*discret_->ElementRowMap());
      break;
    case INPAR::STR::strain_none:
      break;
    default:
      dserror("requested plastic strain type not supported");;
    }
  }

  return;
} // STRUMULTI::MicroStatic::Output()


/*----------------------------------------------------------------------*
 |  read restart (public)                                       lw 03/08|
 *----------------------------------------------------------------------*/
void STRUMULTI::MicroStatic::ReadRestart(int step,
                                         Teuchos::RCP<Epetra_Vector> dis,
                                         Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > lastalpha,
                                         Teuchos::RCP<UTILS::SurfStressManager> surf_stress_man,
                                         string name)
{
  Teuchos::RCP<IO::InputControl> inputcontrol = Teuchos::rcp(new IO::InputControl(name, true));
  IO::DiscretizationReader reader(discret_, inputcontrol, step);
  double time  = reader.ReadDouble("time");
  int    rstep = reader.ReadInt("step");
  if (rstep != step) dserror("Time step on file not equal to given step");

  reader.ReadVector(dis, "displacement");
  // It does not make any sense to read the mesh and corresponding
  // element based data because we surely have different element based
  // data at every Gauss point
  // reader.ReadMesh(step);

  // Override current time and step with values from file
  time_  = time;
  timen_ = time_ + dt_;
  step_  = rstep;
  stepn_ = step_ + 1;

  if (surf_stress_man->HaveSurfStress())
  {
    surf_stress_man->ReadRestart(rstep, name, true);
  }

  reader.ReadSerialDenseMatrix(lastalpha, "alpha");

  return;
}


void STRUMULTI::MicroStatic::EvaluateMicroBC(LINALG::Matrix<3,3>* defgrd,
                                             Teuchos::RCP<Epetra_Vector> disp)
{
  std::vector<DRT::Condition*> conds;
  discret_->GetCondition("MicroBoundary", conds);
  for (unsigned i=0; i<conds.size(); ++i)
  {
    const std::vector<int>* nodeids = conds[i]->Get<std::vector<int> >("Node Ids");
    if (!nodeids) dserror("MicroBoundary condition does not have nodal cloud");
    const int nnode = (*nodeids).size();

    for (int j=0; j<nnode; ++j)
    {
      // do only nodes in my row map
      if (!discret_->NodeRowMap()->MyGID((*nodeids)[j])) continue;
      DRT::Node* actnode = discret_->gNode((*nodeids)[j]);
      if (!actnode) dserror("Cannot find global node %d",(*nodeids)[j]);

      // nodal coordinates
      const double* x = actnode->X();

      // boundary displacements are prescribed via the macroscopic
      // deformation gradient
      double disp_prescribed[3];
      LINALG::Matrix<3,3> Du(defgrd->A(),false);
      LINALG::Matrix<3,3> I(true);
      I(0,0)=-1.0;
      I(1,1)=-1.0;
      I(2,2)=-1.0;
      Du+=I;

      for (int k=0; k<3;k++)
      {
        double dis = 0.;

        for (int l=0;l<3;l++)
        {
          dis += Du(k, l) * x[l];
        }

        disp_prescribed[k] = dis;
      }

      std::vector<int> dofs = discret_->Dof(actnode);

      for (int l=0; l<3; ++l)
      {
        const int gid = dofs[l];

        const int lid = disp->Map().LID(gid);
        if (lid<0) dserror("Global id %d not on this proc in system vector",gid);
        (*disp)[lid] = disp_prescribed[l];
      }
    }
  }
}

void STRUMULTI::MicroStatic::SetState(Teuchos::RCP<Epetra_Vector> dis,
                                      Teuchos::RCP<Epetra_Vector> dism,
                                      Teuchos::RCP<Epetra_Vector> disn,
                                      Teuchos::RCP<UTILS::SurfStressManager> surfman,
                                      Teuchos::RCP<std::vector<char> > stress,
                                      Teuchos::RCP<std::vector<char> > strain,
                                      Teuchos::RCP<std::vector<char> > plstrain,
                                      Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > lastalpha,
                                      Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > oldalpha,
                                      Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > oldfeas,
                                      Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > oldKaainv,
                                      Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > oldKda)
{
  dis_ = dis;
  dism_ = dism;
  disn_ = disn;
  surf_stress_man_ = surfman;

  stress_ = stress;
  strain_ = strain;
  plstrain_ = plstrain;

  // using RCP's here means we do not need to return EAS data explicitly
  lastalpha_ = lastalpha;
  oldalpha_  = oldalpha;
  oldfeas_   = oldfeas;
  oldKaainv_ = oldKaainv;
  oldKda_    = oldKda;
}

void STRUMULTI::MicroStatic::UpdateNewTimeStep(Teuchos::RCP<Epetra_Vector> dis,
                                               Teuchos::RCP<Epetra_Vector> dism,
                                               Teuchos::RCP<Epetra_Vector> disn,
                                               Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > alpha,
                                               Teuchos::RCP<std::map<int, Teuchos::RCP<Epetra_SerialDenseMatrix> > > oldalpha,
                                               Teuchos::RCP<UTILS::SurfStressManager> surf_stress_man)
{
  // these updates hold for an imr-like generalized alpha time integration
  // -> if another time integration scheme should be used, this needs
  // to be changed accordingly

  dis->Update(1.0, *disn, 0.0);
  dism->Update(1.0, *dis, 0.0);
  disn->Update(1.0, *dis, 0.0);

  if (surf_stress_man->HaveSurfStress())
  {
    surf_stress_man->Update();
  }

  const Epetra_Map* elemap = discret_->ElementRowMap();
  for (int i=0;i<elemap->NumMyElements();++i)
  {
    RCP<Epetra_SerialDenseMatrix> alphai  = (*alpha)[i];
    RCP<Epetra_SerialDenseMatrix> alphao = (*oldalpha)[i];

    if (alphai!=null && alphao!=null) // update only those elements with EAS
    {
      Epetra_BLAS blas;
      blas.SCAL(alphao->M() * alphao->N(), -alphaf_/(1.0-alphaf_), alphao->A());  // alphao *= -alphaf/(1.0-alphaf)
      blas.AXPY(alphao->M() * alphao->N(), 1.0/(1.0-alphaf_), alphai->A(), alphao->A());  // alphao += 1.0/(1.0-alphaf) * alpha
      blas.COPY(alphai->M() * alphai->N(), alphao->A(), alphai->A());  // alpha := alphao
    }
  }
}

void STRUMULTI::MicroStatic::SetTime(const double time, const double timen, const double dt, const int step, const int stepn)
{
  time_  = time;
  timen_ = timen;
  dt_    = dt;
  step_  = step;
  stepn_ = stepn;
}

//Teuchos::RCP<Epetra_Vector> STRUMULTI::MicroStatic::ReturnNewDism() { return Teuchos::rcp(new Epetra_Vector(*dism_)); }

void STRUMULTI::MicroStatic::ClearState()
{
  dis_ = null;
  dism_ = null;
  disn_ = null;
}

void STRUMULTI::MicroStatic::SetEASData()
{
  for (int lid=0; lid<discret_->ElementRowMap()->NumMyElements(); ++lid)
  {
    DRT::Element* actele = discret_->lRowElement(lid);

    if (actele->ElementType()==DRT::ELEMENTS::So_hex8Type::Instance() or
        actele->ElementType()==DRT::ELEMENTS::So_shw6Type::Instance())
    {
      // create the parameters for the discretization
      ParameterList p;
      // action for elements
      p.set("action","multi_eas_set");

      p.set("oldalpha", oldalpha_);
      p.set("oldfeas", oldfeas_);
      p.set("oldKaainv", oldKaainv_);
      p.set("oldKda", oldKda_);

      Epetra_SerialDenseMatrix elematrix1;
      Epetra_SerialDenseMatrix elematrix2;
      Epetra_SerialDenseVector elevector1;
      Epetra_SerialDenseVector elevector2;
      Epetra_SerialDenseVector elevector3;
      std::vector<int> lm;

      actele->Evaluate(p,*discret_,lm,elematrix1,elematrix2,elevector1,elevector2,elevector3);
    }
  }
}




void STRUMULTI::MicroStatic::StaticHomogenization(LINALG::Matrix<6,1>* stress,
                                                  LINALG::Matrix<6,6>* cmat,
                                                  double *density,
                                                  LINALG::Matrix<3,3>* defgrd,
                                                  const bool mod_newton,
                                                  bool& build_stiff)
{
  // determine macroscopic parameters via averaging (homogenization) of
  // microscopic features accoring to Kouznetsova, Miehe etc.
  // this was implemented against the background of serial usage
  // -> if a parallel version of microscale simulations is EVER wanted,
  // carefully check if/what/where things have to change

  // split microscale stiffness into parts corresponding to prescribed
  // and free dofs -> see thesis of Kouznetsova (Computational
  // homogenization for the multi-scale analysis of multi-phase
  // materials, Eindhoven, 2002)

  // for calculating the stresses, we need to choose the
  // right three components of freactm_ corresponding to a single node and
  // take the inner product with the material coordinates of this
  // node. The sum over all boundary nodes delivers the first
  // Piola-Kirchhoff macroscopic stress which has to be transformed
  // into the second Piola-Kirchhoff counterpart.
  // All these complicated conversions are necessary since only for
  // the energy-conjugated pair of first Piola-Kirchhoff and
  // deformation gradient the averaging integrals can be transformed
  // into integrals over the boundaries only in case of negligible
  // inertial forces (which simplifies matters significantly) whereas
  // the calling macroscopic material routine demands a second
  // Piola-Kirchhoff stress tensor.

  // IMPORTANT: the RVE has to be centered around (0,0,0), otherwise
  // modifications of this approach are necessary.

  freactm_->Scale(-1.0);

  LINALG::Matrix<3,3> P(true);

  for (int i=0; i<3; ++i)
  {
    for (int j=0; j<3; ++j)
    {
      for (int n=0; n<np_/3; ++n)
      {
        P(i,j) += (*freactm_)[n*3+i]*(*Xp_)[n*3+j];
      }
      P(i,j) /= V0_;
    }
  }

  // determine inverse of deformation gradient

  LINALG::Matrix<3,3> F_inv(defgrd->A(),false);
  F_inv.Invert();

  // convert to second Piola-Kirchhoff stresses and store them in
  // vector format
  // assembly of stresses (cf Solid3 Hex8): S11,S22,S33,S12,S23,S13

  stress->Scale(0.);

  for (int i=0; i<3; ++i)
  {
    (*stress)(0) += F_inv(0, i)*P(i,0);                     // S11
    (*stress)(1) += F_inv(1, i)*P(i,1);                     // S22
    (*stress)(2) += F_inv(2, i)*P(i,2);                     // S33
    (*stress)(3) += F_inv(0, i)*P(i,1);                     // S12
    (*stress)(4) += F_inv(1, i)*P(i,2);                     // S23
    (*stress)(5) += F_inv(0, i)*P(i,2);                     // S13
  }

  if (build_stiff)
  {
    // The calculation of the consistent macroscopic constitutive tensor
    // follows
    //
    // C. Miehe, Computational micro-to-macro transitions for
    // discretized micro-structures of heterogeneous materials at finite
    // strains based on a minimization of averaged incremental energy.
    // Computer Methods in Applied Mechanics and Engineering 192: 559-591, 2003.

    const Epetra_Map* dofrowmap = discret_->DofRowMap();
    Epetra_MultiVector cmatpf(D_->Map(), 9);

    Epetra_Vector x(*dofrowmap);
    Epetra_Vector y(*dofrowmap);

    // make a copy
    stiff_dirich_ = Teuchos::rcp(new LINALG::SparseMatrix(*stiff_));

    stiff_->ApplyDirichlet(dirichtoggle_);

    Epetra_LinearProblem linprob(&(*stiff_->EpetraMatrix()), &x, &y);
    int error=linprob.CheckInput();
    if (error)
      dserror("Input for linear problem inconsistent");
#ifndef HAVENOT_UMFPACK
    Amesos_Umfpack solver(linprob);
    int err = solver.NumericFactorization();   // LU decomposition of stiff_ only once
    if (err)
      dserror("Numeric factorization of stiff_ for homogenization failed");

    for (int i=0;i<9;++i)
    {
      x.PutScalar(0.0);
      y.Update(1.0, *((*rhs_)(i)), 0.0);
      solver.Solve();

      Epetra_Vector f(*dofrowmap);
      stiff_dirich_->Multiply(false, x, f);
      Epetra_Vector fexp(*pdof_);
      int err = fexp.Import(f, *importp_, Insert);
      if (err)
        dserror("Export of boundary 'forces' failed with err=%d", err);

      (cmatpf(i))->Multiply('N', 'N', 1.0/V0_, *D_, fexp, 0.0);
    }
#endif

    // We now have to transform the calculated constitutive tensor
    // relating first Piola-Kirchhoff stresses to the deformation
    // gradient into a constitutive tensor relating second
    // Piola-Kirchhoff stresses to Green-Lagrange strains.

    ConvertMat(cmatpf, F_inv, *stress, *cmat);

    // after having constructed the stiffness matrix, this need not be
    // done in case of modified Newton as nonlinear solver of the
    // macroscale until the next update of macroscopic time step, when
    // build_stiff is set to true in the micromaterialgp again!

    if (mod_newton == true)
      build_stiff = false;
  }
  // homogenized density was already determined in the constructor

  *density = density_;

  return;
}

#endif
