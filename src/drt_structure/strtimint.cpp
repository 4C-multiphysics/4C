/*----------------------------------------------------------------------*/
/*!
\file strtimint.cpp
\brief Time integration for spatially discretised structural dynamics

<pre>
Maintainer: Burkhard Bornemann
            bornemann@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15237
</pre>
*/

/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */
#include <iostream>
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Teuchos_TimeMonitor.hpp"

#include "strtimint_mstep.H"
#include "strtimint.H"

#include "../drt_contact/contact_defines.H"
#include "../drt_io/io_control.H"
#include "../drt_fluid/fluid_utils.H"

/*----------------------------------------------------------------------*/
/* print tea time logo */
void STR::TimInt::Logo()
{
  std::cout << "Welcome to Structural Time Integration " << std::endl;
  std::cout << "     __o__                          __o__" << std::endl;
  std::cout << "__  /-----\\__                  __  /-----\\__" << std::endl;
  std::cout << "\\ \\/       \\ \\    |       \\    \\ \\/       \\ \\" << std::endl;
  std::cout << " \\ |  tea  | |    |-------->    \\ |  tea  | |" << std::endl;
  std::cout << "  \\|       |_/    |       /      \\|       |_/" << std::endl;
  std::cout << "    \\_____/   ._                   \\_____/   ._ _|_ /|" << std::endl;
  std::cout << "              | |                            | | |   |" << std::endl;
  std::cout << std::endl;
}

/*----------------------------------------------------------------------*/
/* constructor */
STR::TimInt::TimInt
(
  const Teuchos::ParameterList& ioparams,
  const Teuchos::ParameterList& sdynparams,
  const Teuchos::ParameterList& xparams,
  Teuchos::RCP<DRT::Discretization> actdis,
  Teuchos::RCP<LINALG::Solver> solver,
  Teuchos::RCP<IO::DiscretizationWriter> output
)
: discret_(actdis),
  myrank_(actdis->Comm().MyPID()),
  dofrowmap_(actdis->Filled() ? actdis->DofRowMap() : NULL),
  solver_(solver),
  solveradapttol_(Teuchos::getIntegralValue<int>(sdynparams,"ADAPTCONV")==1),
  solveradaptolbetter_(sdynparams.get<double>("ADAPTCONV_BETTER")),
  dbcmaps_(Teuchos::rcp(new LINALG::MapExtractor())),
  output_(output),
  printlogo_(true),  // DON'T EVEN DARE TO SET THIS TO FALSE
  printscreen_(true),  // ADD INPUT PARAMETER
  errfile_(xparams.get<FILE*>("err file")),
  printerrfile_(true and errfile_),  // ADD INPUT PARAMETER FOR 'true'
  printiter_(true),  // ADD INPUT PARAMETER
  writerestartevery_(sdynparams.get<int>("RESTARTEVRY")),
  writestate_((bool) Teuchos::getIntegralValue<int>(ioparams,"STRUCT_DISP")),
  writestateevery_(sdynparams.get<int>("RESEVRYDISP")),
  writestrevery_(sdynparams.get<int>("RESEVRYSTRS")),
  writestress_(Teuchos::getIntegralValue<INPAR::STR::StressType>(ioparams,"STRUCT_STRESS")),
  writestrain_(Teuchos::getIntegralValue<INPAR::STR::StrainType>(ioparams,"STRUCT_STRAIN")),
  writeenergyevery_(sdynparams.get<int>("RESEVRYERGY")),
  writesurfactant_((bool) Teuchos::getIntegralValue<int>(ioparams,"STRUCT_SURFACTANT")),
  energyfile_(NULL),
  damping_(Teuchos::getIntegralValue<INPAR::STR::DampKind>(sdynparams,"DAMPING")),
  dampk_(sdynparams.get<double>("K_DAMP")),
  dampm_(sdynparams.get<double>("M_DAMP")),
  conman_(Teuchos::null),
  consolv_(Teuchos::null),
  surfstressman_(Teuchos::null),
  potman_(Teuchos::null),
  cmtman_(Teuchos::null),
  locsysman_(Teuchos::null),
  pressure_(Teuchos::null),
  time_(Teuchos::null),
  timen_(0.0),
  dt_(Teuchos::null),
  timemax_(sdynparams.get<double>("MAXTIME")),
  stepmax_(sdynparams.get<int>("NUMSTEP")),
  step_(Teuchos::null),
  stepn_(0),
  zeros_(Teuchos::null),
  dis_(Teuchos::null),
  vel_(Teuchos::null),
  acc_(Teuchos::null),
  disn_(Teuchos::null),
  veln_(Teuchos::null),
  accn_(Teuchos::null),
  stiff_(Teuchos::null),
  mass_(Teuchos::null),
  damp_(Teuchos::null)
{
  // welcome user
  if ( (printlogo_) and (myrank_ == 0) )
  {
    Logo();
  }

  // check wether discretisation has been completed
  if (not discret_->Filled() || not actdis->HaveDofs())
  {
    dserror("Discretisation is not complete or has no dofs!");
  }

  // connect degrees of freedom for periodic boundary conditions
  // (i.e. for multi-patch nurbs computations)
  {
    PeriodicBoundaryConditions pbc(discret_);

    if (pbc.HasPBC())
    {
      pbc.UpdateDofsForPeriodicBoundaryConditions();

      discret_->ComputeNullSpaceIfNecessary(solver->Params(),true);

      dofrowmap_ = discret_->DofRowMap(0);
    }
  }

  // time state
  time_ = Teuchos::rcp(new TimIntMStep<double>(0, 0, 0.0));  // HERE SHOULD BE SOMETHING LIKE (sdynparams.get<double>("TIMEINIT"))
  dt_ = Teuchos::rcp(new TimIntMStep<double>(0, 0, sdynparams.get<double>("TIMESTEP")));
  step_ = 0;
  timen_ = (*time_)[0] + (*dt_)[0];  // set target time to initial time plus step size
  stepn_ = step_ + 1;

  // output file for energy
  if ( (writeenergyevery_ != 0) and (myrank_ == 0) )
    AttachEnergyFile();

  // a zero vector of full length
  zeros_ = LINALG::CreateVector(*dofrowmap_, true);

  // Map containing Dirichlet DOFs
  {
    Teuchos::ParameterList p;
    p.set("total time", timen_);
    discret_->EvaluateDirichlet(p, zeros_, Teuchos::null, Teuchos::null,
                                Teuchos::null, dbcmaps_);
    zeros_->PutScalar(0.0); // just in case of change
  }

  // displacements D_{n}
  dis_ = Teuchos::rcp(new TimIntMStep<Epetra_Vector>(0, 0, dofrowmap_, true));
  // velocities V_{n}
  vel_ = Teuchos::rcp(new TimIntMStep<Epetra_Vector>(0, 0, dofrowmap_, true));
  // accelerations A_{n}
  acc_ = Teuchos::rcp(new TimIntMStep<Epetra_Vector>(0, 0, dofrowmap_, true));

  // displacements D_{n+1} at t_{n+1}
  disn_ = LINALG::CreateVector(*dofrowmap_, true);
  // velocities V_{n+1} at t_{n+1}
  veln_ = LINALG::CreateVector(*dofrowmap_, true);
  // accelerations A_{n+1} at t_{n+1}
  accn_ = LINALG::CreateVector(*dofrowmap_, true);

  // create empty matrices
  stiff_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap_, 81, true, true));
  mass_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap_, 81, true, true));
  if (damping_ == INPAR::STR::damp_rayleigh)
    damp_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap_, 81, true, true));

  // initialize constraint manager
  conman_ = Teuchos::rcp(new UTILS::ConstrManager(discret_,
                                                  (*dis_)(0),
                                                  sdynparams));
  // initialize constraint solver iff constraints are defined
  if (conman_->HaveConstraint())
  {
    consolv_ = Teuchos::rcp(new UTILS::ConstraintSolver(discret_,
                                                        *solver_,
                                                        dbcmaps_,
                                                        sdynparams));
  }

  // check for contact or meshtying conditions
  {
    // If contact or meshtying conditions are detected, then a corresponding
    // manager object stored via #cmtman_ is created and all relevant stuff is
    // initialized. Else, #cmtman_ remains a Teuchos::null pointer.
    PrepareContactMeshtying(sdynparams);
  }

  // fix pointer to #dofrowmap_, which has not really changed, but is
  // located at different place
  // (this is necessary to BOTH constraints and contact / meshtying)
  dofrowmap_ = discret_->DofRowMap();

  // Initialize SurfStressManager for handling surface stress conditions due to interfacial phenomena
  surfstressman_ = Teuchos::rcp(new UTILS::SurfStressManager(discret_,
                                                             sdynparams,
                                                             DRT::Problem::Instance()->OutputControlFile()->FileName()));

  // Check for potential conditions
  {
    std::vector<DRT::Condition*> potentialcond(0);
    discret_->GetCondition("Potential", potentialcond);
    if (potentialcond.size())
    {
      potman_ = Teuchos::rcp(new POTENTIAL::PotentialManager(Discretization(), *discret_));
      stiff_ = Teuchos::rcp(new LINALG::SparseMatrix(*dofrowmap_,81,true,false, LINALG::SparseMatrix::FE_MATRIX));
    }
  }

  // check whether we have locsys BCs and create LocSysManager if so
  // after checking
  {
    std::vector<DRT::Condition*> locsysconditions(0);
    discret_->GetCondition("Locsys", locsysconditions);
    if (locsysconditions.size())
    {
      locsysman_ = Teuchos::rcp(new DRT::UTILS::LocsysManager(*discret_, true));
    }
  }

  // check if we have elements which use a continuous displacement and pressure
  // field
  {
    int locnumsosh8p8 = 0;
    // Loop through all elements on processor
    for (int i=0; i<discret_->NumMyColElements(); ++i)
    {
      // get the actual element
      if (discret_->lColElement(i)->Type() == DRT::Element::element_sosh8p8)
        locnumsosh8p8 += 1;
    }
    // Was at least one SoSh8P8 found on one processor?
    int glonumsosh8p8 = 0;
    discret_->Comm().MaxAll(&locnumsosh8p8, &glonumsosh8p8, 1);
    // Yes, it was. Go ahead for all processors (even if they do not carry any SoSh8P8 elements)
    if (glonumsosh8p8 > 0)
    {
      pressure_ = Teuchos::rcp(new LINALG::MapExtractor());
      const int ndim = 3;
      FLD::UTILS::SetupFluidSplit(*discret_, ndim, *pressure_);
    }
  }

  // stay with us
  return;
}

/*----------------------------------------------------------------------*/
/* Check for contact or meshtying conditions and do preparations */
void STR::TimInt::PrepareContactMeshtying(const Teuchos::ParameterList& sdynparams)
{
  // some parameters
  const Teuchos::ParameterList&   scontact = DRT::Problem::Instance()->MeshtyingAndContactParams();
  INPAR::CONTACT::ApplicationType apptype  = Teuchos::getIntegralValue<INPAR::CONTACT::ApplicationType>(scontact,"APPLICATION");
  INPAR::MORTAR::ShapeFcn         shapefcn = Teuchos::getIntegralValue<INPAR::MORTAR::ShapeFcn>(scontact,"SHAPEFCN");
  INPAR::CONTACT::SolvingStrategy soltype  = Teuchos::getIntegralValue<INPAR::CONTACT::SolvingStrategy>(scontact,"STRATEGY");
  bool semismooth = Teuchos::getIntegralValue<int>(scontact,"SEMI_SMOOTH_NEWTON");

  // check contact conditions
  vector<DRT::Condition*> contactconditions(0);
  discret_->GetCondition("Contact",contactconditions);

  // only continue if contact unmistakably chosen in input file
  if (contactconditions.size() && apptype != INPAR::CONTACT::app_none)
  {
    // store integration parameter alphaf into cmtman_ as well
    // (for all cases except GenAlpha and GEMM this is zero)
    double alphaf = 0.0;
    if (Teuchos::getIntegralValue<INPAR::STR::DynamicType>(sdynparams, "DYNAMICTYP") == INPAR::STR::dyna_genalpha)
      alphaf = sdynparams.sublist("GENALPHA").get<double>("ALPHA_F");
    if (Teuchos::getIntegralValue<INPAR::STR::DynamicType>(sdynparams, "DYNAMICTYP") == INPAR::STR::dyna_gemm)
      alphaf = sdynparams.sublist("GEMM").get<double>("ALPHA_F");

    // decide whether this is meshtying or contact and create manager
    if (apptype == INPAR::CONTACT::app_mortarmeshtying)
      cmtman_ = rcp(new CONTACT::MtManager(*discret_,alphaf));
    else if (apptype == INPAR::CONTACT::app_mortarcontact)
      cmtman_ = rcp(new CONTACT::CoManager(*discret_,alphaf));
    else if (apptype == INPAR::CONTACT::app_beamcontact)
      dserror("ERROR: Beam contact not yet implemented in new STI!");

    // store DBC status in contact nodes
    cmtman_->GetStrategy().StoreDirichletStatus(dbcmaps_);

    // create old style dirichtoggle vector (supposed to go away)
    dirichtoggle_ = rcp(new Epetra_Vector(*(dbcmaps_->FullMap())));
    RCP<Epetra_Vector> temp = rcp(new Epetra_Vector(*(dbcmaps_->CondMap())));
    temp->PutScalar(1.0);
    LINALG::Export(*temp,*dirichtoggle_);

    // contact and constraints together not yet implemented
    if (conman_->HaveConstraint())
      dserror("ERROR: Constraints and contact cannot be treated at the same time yet");

    // print messages for multifield problems (e.g FSI)
    const string probtype = DRT::Problem::Instance()->ProblemType();
    if (probtype != "structure" && !myrank_)
    {
      // warnings
#ifdef CONTACTPSEUDO2D
      cout << RED << "WARNING: The flag CONTACTPSEUDO2D is switched on. If this "
           << "is a real 3D problem, switch it off!" << END_COLOR << endl;
#else
      cout << RED << "WARNING: The flag CONTACTPSEUDO2D is switched off. If this "
           << "is a 2D problem modeled pseudo-3D, switch it on!" << END_COLOR << endl;
#endif // #ifdef CONTACTPSEUDO2D
      cout << RED << "WARNING: Contact and Meshtying are still experimental "
           << "for the chosen problem type \"" << probtype << "\"!\n" << END_COLOR << endl;

      // errors
      if (soltype == INPAR::CONTACT::solution_lagmult && (!semismooth || shapefcn != INPAR::MORTAR::shape_dual))
        dserror("ERROR: Multifield problems with LM strategy for meshtying/contact only for dual+semismooth case!");
      if (soltype == INPAR::CONTACT::solution_auglag)
        dserror("ERROR: Multifield problems with AL strategy for meshtying/contact not yet implemented");
    }

    // visualization of initial configuration
#ifdef CONTACTGMSH3
    cmtman_->GetStrategy().VisualizeGmsh(0,0);
#endif // #ifdef CONTACTGMSH2

    // initialization of contact or meshting
    {
      // FOR MESHTYING (ONLY ONCE), NO FUNCTIONALITY FOR CONTACT CASES
      // (1) Do mortar coupling in reference configuration
      // (2) Perform mesh intialization for rotational invariance
      cmtman_->GetStrategy().MortarCoupling(zeros_);
      cmtman_->GetStrategy().MeshInitialization();

      // FOR FRICTIONAL CONTACT
      // (1) Mortar coupling in reference configuration
      // for frictional contact we need history values (relative velocity) and
      // therefore we store the nodal entries of mortar matrices (reference
      // configuration) before the first time step
      cmtman_->GetStrategy().EvaluateReferenceState(disn_);

      // FOR PENALTY CONTACT (ONLY ONCE), NO FUNCTIONALITY FOR OTHER CASES
      // (1) Explicitly store gap-scaling factor kappa
      cmtman_->GetStrategy().SaveReferenceState(zeros_);
    }

    // output of strategy type to screen
    {
      // output
      if (!myrank_)
      {
        if (soltype == INPAR::CONTACT::solution_lagmult && shapefcn == INPAR::MORTAR::shape_standard)
          cout << "===== Standard Lagrange multiplier strategy ====================\n" << endl;
        else if (soltype == INPAR::CONTACT::solution_lagmult && shapefcn == INPAR::MORTAR::shape_dual)
          cout << "===== Dual Lagrange multiplier strategy ========================\n" << endl;
        else if (soltype == INPAR::CONTACT::solution_penalty && shapefcn == INPAR::MORTAR::shape_standard)
          cout << "===== Standard Penalty strategy ================================\n" << endl;
        else if (soltype == INPAR::CONTACT::solution_penalty && shapefcn == INPAR::MORTAR::shape_dual)
          cout << "===== Dual Penalty strategy ====================================\n" << endl;
        else if (soltype == INPAR::CONTACT::solution_auglag && shapefcn == INPAR::MORTAR::shape_standard)
          cout << "===== Standard Augmented Lagrange strategy =====================\n" << endl;
        else if (soltype == INPAR::CONTACT::solution_auglag && shapefcn == INPAR::MORTAR::shape_dual)
          cout << "===== Dual Augmented Lagrange strategy =========================\n" << endl;
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*/
/* equilibrate system at initial state
 * and identify consistent accelerations */
void STR::TimInt::DetermineMassDampConsistAccel()
{
  // temporary force vectors in this routine
  Teuchos::RCP<Epetra_Vector> fext
    = LINALG::CreateVector(*dofrowmap_, true); // external force
  Teuchos::RCP<Epetra_Vector> fint
    = LINALG::CreateVector(*dofrowmap_, true); // internal force

  // overwrite initial state vectors with DirichletBCs
  ApplyDirichletBC((*time_)[0], (*dis_)(0), (*vel_)(0), (*acc_)(0), false);

  // get external force
  ApplyForceExternal((*time_)[0], (*dis_)(0), (*vel_)(0), fext);

  // initialise matrices
  stiff_->Zero();
  mass_->Zero();

  // get initial internal force and stiffness and mass
  {
    // create the parameters for the discretization
    ParameterList p;
    // action for elements
    p.set("action", "calc_struct_nlnstiffmass");
    // other parameters that might be needed by the elements
    p.set("total time", (*time_)[0]);
    p.set("delta time", (*dt_)[0]);
    if (pressure_ != Teuchos::null) p.set("volume", 0.0);
    // set vector values needed by elements
    discret_->ClearState();
    // extended SetState(0,...) in case of multiple dofsets (e.g. TSI)
    discret_->SetState(0,"residual displacement", zeros_);
    discret_->SetState(0,"displacement", (*dis_)(0));
    if (damping_ == INPAR::STR::damp_material) discret_->SetState(0,"velocity", (*vel_)(0));
    // set the temperature for the coupled problem
    if(tempn_!=Teuchos::null)
    {
      discret_->SetState(1,"temperature",tempn_);
    }
    discret_->Evaluate(p, stiff_, mass_, fint, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }

  // finish mass matrix
  mass_->Complete();

  // close stiffness matrix
  stiff_->Complete();

  // build Rayleigh damping matrix if desired
  if (damping_ == INPAR::STR::damp_rayleigh)
  {
    damp_->Add(*stiff_, false, dampk_, 0.0);
    damp_->Add(*mass_, false, dampm_, 1.0);
    damp_->Complete();
  }

  // in case of C0 pressure field, we need to get rid of
  // pressure equations
  Teuchos::RCP<LINALG::SparseOperator> mass = Teuchos::null;
  if (pressure_ != Teuchos::null)
  {
    mass = Teuchos::rcp(new LINALG::SparseMatrix(*MassMatrix(),Copy));
    mass->ApplyDirichlet(*(pressure_->CondMap()));
  }
  else
  {
    mass = mass_;
  }

  // calculate consistent initial accelerations
  // WE MISS:
  //   - surface stress forces
  //   - potential forces
  {
    Teuchos::RCP<Epetra_Vector> rhs = LINALG::CreateVector(*dofrowmap_, true);
    if (damping_ == INPAR::STR::damp_rayleigh)
    {
      damp_->Multiply(false, (*vel_)[0], *rhs);
    }
    rhs->Update(-1.0, *fint, 1.0, *fext, -1.0);
    // blank RHS on DBC DOFs
    dbcmaps_->InsertCondVector(dbcmaps_->ExtractCondVector(zeros_), rhs);
    if (pressure_ != Teuchos::null)
      pressure_->InsertCondVector(pressure_->ExtractCondVector(zeros_), rhs);
    solver_->Solve(mass->EpetraOperator(), (*acc_)(0), rhs, true, true);
  }

  // We need to reset the stiffness matrix because its graph (topology)
  // is not finished yet in case of constraints and posssibly other side
  // effects (basically managers).
  stiff_->Reset();

  // leave this hell
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate Dirichlet BC at t_{n+1} */
void STR::TimInt::ApplyDirichletBC
(
  const double time,
  Teuchos::RCP<Epetra_Vector> dis,
  Teuchos::RCP<Epetra_Vector> vel,
  Teuchos::RCP<Epetra_Vector> acc,
  bool recreatemap
)
{
  // in the case of local systems we have to rotate forward ...
  if (locsysman_ != Teuchos::null)
  {
    locsysman_->RotateGlobalToLocal(dis);
    locsysman_->RotateGlobalToLocal(vel);
    locsysman_->RotateGlobalToLocal(acc);
  }


  // apply DBCs
  // needed parameters
  ParameterList p;
  p.set("total time", time);  // target time

  // predicted Dirichlet values
  // \c dis then also holds prescribed new Dirichlet displacements
  discret_->ClearState();
  if (recreatemap)
  {
    discret_->EvaluateDirichlet(p, dis, vel, acc,
                                Teuchos::null, dbcmaps_);
  }
  else
  {
    discret_->EvaluateDirichlet(p, dis, vel, acc,
                                Teuchos::null, Teuchos::null);
  }
  discret_->ClearState();

  // in the case of local systems we have to rotate back into global Cartesian frame
  if (locsysman_ != Teuchos::null)
  {
    locsysman_->RotateLocalToGlobal(dis);
    locsysman_->RotateLocalToGlobal(vel);
    locsysman_->RotateLocalToGlobal(acc);
  }

  // ciao
  return;
}

/*----------------------------------------------------------------------*/
/* Update time and step counter */
void STR::TimInt::UpdateStepTime()
{
  // update time and step
  time_->UpdateSteps(timen_);  // t_{n} := t_{n+1}, etc
  step_ = stepn_;  // n := n+1
  //
  timen_ += (*dt_)[0];
  stepn_ += 1;

  // new deal
  return;
}

/*----------------------------------------------------------------------*/
/* Reset configuration after time step */
void STR::TimInt::ResetStep()
{
  // reset state vectors
  disn_->Update(1.0, (*dis_)[0], 0.0);
  veln_->Update(1.0, (*vel_)[0], 0.0);
  accn_->Update(1.0, (*acc_)[0], 0.0);

  // reset anything that needs to be reset at the element level
  {
    // create the parameters for the discretization
    ParameterList p;
    p.set("action", "calc_struct_reset_istep");
    // go to elements
    discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                       Teuchos::null, Teuchos::null, Teuchos::null);
    discret_->ClearState();
  }

  // I am gone
  return;
}

/*----------------------------------------------------------------------*/
/* Read and set restart values */
void STR::TimInt::ReadRestart
(
  const int step
)
{
  IO::DiscretizationReader reader(discret_, step);
  if (step != reader.ReadInt("step"))
    dserror("Time step on file not equal to given step");

  step_ = step;
  stepn_ = step_ + 1;
  time_ = Teuchos::rcp(new TimIntMStep<double>(0, 0, reader.ReadDouble("time")));
  timen_ = (*time_)[0] + (*dt_)[0];

  ReadRestartState();
  ReadRestartConstraint();
  ReadRestartContactMeshtying();
  ReadRestartForce();
  ReadRestartSurfstress();
  ReadRestartMultiScale();
  // fix pointer to #dofrowmap_, which has not really changed, but is
  // located at different place
  dofrowmap_ = discret_->DofRowMap();
}

/*----------------------------------------------------------------------*/
/* Read and set restart state */
void STR::TimInt::ReadRestartState()
{
  IO::DiscretizationReader reader(discret_, step_);
  reader.ReadVector(disn_, "displacement");
  dis_->UpdateSteps(*disn_);
  reader.ReadVector(veln_, "velocity");
  vel_->UpdateSteps(*veln_);
  reader.ReadVector(accn_, "acceleration");
  acc_->UpdateSteps(*accn_);
  reader.ReadMesh(step_);
  return;
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for constraints */
void STR::TimInt::ReadRestartConstraint()
{
  if (conman_->HaveConstraint())
  {
    IO::DiscretizationReader reader(discret_, step_);
    double uzawatemp = reader.ReadDouble("uzawaparameter");
    consolv_->SetUzawaParameter(uzawatemp);
    Teuchos::RCP<Epetra_Map> constrmap=conman_->GetConstraintMap();
    Teuchos::RCP<Epetra_Vector> tempvec = LINALG::CreateVector(*constrmap, true);
    reader.ReadVector(tempvec, "lagrmultiplier");
    conman_->SetLagrMultVector(tempvec);
    reader.ReadVector(tempvec, "refconval");
    conman_->SetRefBaseValues(tempvec, (*time_)[0]);

  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for contact / meshtying */
void STR::TimInt::ReadRestartContactMeshtying()
{
  if (cmtman_ != Teuchos::null)
  {
    //**********************************************************************
    // NOTE: There is an important difference here between contact and
    // meshtying simulations. In both cases, the current coupling operators
    // have to be re-computed for restart, but in the meshtying case this
    // means evaluating DM in the reference configuration!
    // Thus, both dis_ (current displacement state) and zero_ are handed
    // in and contact / meshtying managers choose the correct state.
    //**********************************************************************
    IO::DiscretizationReader reader(discret_,step_);
    cmtman_->ReadRestart(reader,(*dis_)(0),zeros_);
  }
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for constraints */
void STR::TimInt::ReadRestartSurfstress()
{
  if (surfstressman_ -> HaveSurfStress())
    surfstressman_->ReadRestart(step_, DRT::Problem::Instance()->InputControlFile()->FileName());
}

/*----------------------------------------------------------------------*/
/* Read and set restart values for multi-scale */
void STR::TimInt::ReadRestartMultiScale()
{
  Teuchos::RCP<MAT::PAR::Bundle> materials = DRT::Problem::Instance()->Materials();
  for (std::map<int,Teuchos::RCP<MAT::PAR::Material> >::const_iterator i=materials->Map()->begin();
       i!=materials->Map()->end();
       ++i)
  {
    Teuchos::RCP<MAT::PAR::Material> mat = i->second;
    if (mat->Type() == INPAR::MAT::m_struct_multiscale)
    {
      // create the parameters for the discretization
      ParameterList p;
      // action for elements
      p.set("action", "multi_readrestart");
      discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                         Teuchos::null, Teuchos::null, Teuchos::null);
      discret_->ClearState();
      break;
    }
  }
}

/*----------------------------------------------------------------------*/
/* output to file
 * originally by mwgee 03/07 */
void STR::TimInt::OutputStep()
{
  // this flag is passed along subroutines and prevents
  // repeated initialising of output writer, printing of
  // state vectors, or similar
  bool datawritten = false;

  // output restart (try this first)
  // write restart step
  if (writerestartevery_ and (step_%writerestartevery_ == 0) )
  {
    OutputRestart(datawritten);
  }

  // output results (not necessary if restart in same step)
  if ( writestate_
       and writestateevery_ and (step_%writestateevery_ == 0)
       and (not datawritten) )
  {
    OutputState(datawritten);
  }

  // output stress & strain
  if ( writestrevery_
       and ( (writestress_ != INPAR::STR::stress_none)
             or (writestrain_ != INPAR::STR::strain_none) )
       and (step_%writestrevery_ == 0) )
  {
    OutputStressStrain(datawritten);
  }

  // output energy
  if ( writeenergyevery_ and (step_%writeenergyevery_ == 0) )
  {
    OutputEnergy();
  }

  // print active contact / meshtying set
  if (cmtman_ != Teuchos::null)
    cmtman_->GetStrategy().PrintActiveSet();

  // what's next?
  return;
}

/*----------------------------------------------------------------------*/
/* write restart
 * originally by mwgee 03/07 */
void STR::TimInt::OutputRestart
(
  bool& datawritten
)
{
  // Yes, we are going to write...
  datawritten = true;

  // write restart output, please
  output_->WriteMesh(step_, (*time_)[0]);
  output_->NewStep(step_, (*time_)[0]);
  output_->WriteVector("displacement", (*dis_)(0));
  output_->WriteVector("velocity", (*vel_)(0));
  output_->WriteVector("acceleration", (*acc_)(0));
  output_->WriteVector("fexternal", Fext());
  output_->WriteElementData();

  // surface stress
  if (surfstressman_->HaveSurfStress())
  {
    surfstressman_->WriteRestart(step_, (*time_)[0]);
  }

  // constraints
  if (conman_->HaveConstraint())
  {
    output_->WriteDouble("uzawaparameter",
                          consolv_->GetUzawaParameter());
    output_->WriteVector("lagrmultiplier",
                          conman_->GetLagrMultVector());
    output_->WriteVector("refconval",
                          conman_->GetRefBaseValues());
  }

  // contact / meshtying
  if (cmtman_ != Teuchos::null)
  {
      cmtman_->WriteRestart(*output_);
      cmtman_->PostprocessTractions(*output_);
  }

  // info dedicated to user's eyes staring at standard out
  if ( (myrank_ == 0) and printscreen_)
  {
    printf("====== Restart written in step %d\n", step_);
    fflush(stdout);
  }

  // info dedicated to processor error file
  if (printerrfile_)
  {
    fprintf(errfile_, "====== Restart written in step %d\n", step_);
    fflush(errfile_);
  }

  // we will say what we did
  return;
}

/*----------------------------------------------------------------------*/
/* output displacements, velocities and accelerations
 * originally by mwgee 03/07 */
void STR::TimInt::OutputState
(
  bool& datawritten
)
{
  // Yes, we are going to write...
  datawritten = true;

  // write now
  output_->NewStep(step_, (*time_)[0]);
  output_->WriteVector("displacement", (*dis_)(0));
  output_->WriteVector("velocity", (*vel_)(0));
  output_->WriteVector("acceleration", (*acc_)(0));
  output_->WriteVector("fexternal", Fext());
  output_->WriteElementData();

  if (surfstressman_->HaveSurfStress() && writesurfactant_)
    surfstressman_->WriteResults(step_, (*time_)[0]);

  // contact / meshtying
  if (cmtman_ != Teuchos::null)
    cmtman_->PostprocessTractions(*output_);

  // leave for good
  return;
}

/*----------------------------------------------------------------------*/
/* stress calculation and output
 * originally by lw */
void STR::TimInt::OutputStressStrain
(
  bool& datawritten
)
{
  // create the parameters for the discretization
  ParameterList p;
  // action for elements
  p.set("action", "calc_struct_stress");
  // other parameters that might be needed by the elements
  p.set("total time", (*time_)[0]);
  p.set("delta time", (*dt_)[0]);

  Teuchos::RCP<std::vector<char> > stressdata
    = Teuchos::rcp(new std::vector<char>());
  p.set("stress", stressdata);
  p.set("iostress", writestress_);

  Teuchos::RCP<std::vector<char> > straindata
    = Teuchos::rcp(new std::vector<char>());
  p.set("strain", straindata);
  p.set("iostrain", writestrain_);

  // set vector values needed by elements
  discret_->ClearState();
  // extended SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0,"residual displacement", zeros_);
  discret_->SetState(0,"displacement", (*dis_)(0));
  // 29.04.10
  // set the temperature for the coupled problem
  if(tempn_!=Teuchos::null)
  {
    discret_->SetState(1,"temperature",tempn_);
  }
  discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                     Teuchos::null, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // Make new step
  if (not datawritten)
  {
    output_->NewStep(step_, (*time_)[0]);
  }
  datawritten = true;

  // write stress
  if (writestress_ != INPAR::STR::stress_none)
  {
    std::string stresstext = "";
    if (writestress_ == INPAR::STR::stress_cauchy)
    {
      stresstext = "gauss_cauchy_stresses_xyz";
    }
    else if (writestress_ == INPAR::STR::stress_2pk)
    {
      stresstext = "gauss_2PK_stresses_xyz";
    }
    else
    {
      dserror("requested stress type not supported");
    }
    output_->WriteVector(stresstext, *stressdata,
                         *(discret_->ElementColMap()));
  }

  // write strain
  if (writestrain_ != INPAR::STR::strain_none)
  {
    std::string straintext = "";
    if (writestrain_ == INPAR::STR::strain_ea)
    {
      straintext = "gauss_EA_strains_xyz";
    }
    else if (writestrain_ == INPAR::STR::strain_gl)
    {
      straintext = "gauss_GL_strains_xyz";
    }
    else
    {
      dserror("requested strain type not supported");
    }
    output_->WriteVector(straintext, *straindata,
                         *(discret_->ElementColMap()));
  }

  // leave me alone
  return;
}

/*----------------------------------------------------------------------*/
/* output system energies */
void STR::TimInt::OutputEnergy()
{
  // internal/strain energy
  double intergy = 0.0;  // total internal energy
  {
    ParameterList p;
    // other parameters needed by the elements
    p.set("action", "calc_struct_energy");

    // set vector values needed by elements
    discret_->ClearState();
    discret_->SetState("displacement", (*dis_)(0));
    // get energies
    Teuchos::RCP<Epetra_SerialDenseVector> energies
      = Teuchos::rcp(new Epetra_SerialDenseVector(1));
    discret_->EvaluateScalars(p, energies);
    discret_->ClearState();
    intergy = (*energies)(0);
  }

  // global calculation of kinetic energy
  double kinergy = 0.0;  // total kinetic energy
  {
    Teuchos::RCP<Epetra_Vector> linmom
      = LINALG::CreateVector(*dofrowmap_, true);
    mass_->Multiply(false, (*vel_)[0], *linmom);
    linmom->Dot((*vel_)[0], &kinergy);
    kinergy *= 0.5;
  }

  // external energy
  double extergy = 0.0;  // total external energy
  {
    // WARNING: This will only work with dead loads!!!
    Teuchos::RCP<Epetra_Vector> fext = Fext();
    fext->Dot((*dis_)[0], &extergy);
  }

  // total energy
  double totergy = kinergy + intergy - extergy;

  // the output
  if (myrank_ == 0)
  {
    *energyfile_ << " " << std::setw(9) << step_
                 << std::scientific  << std::setprecision(16)
                 << " " << (*time_)[0]
                 << " " << totergy
                 << " " << kinergy
                 << " " << intergy
                 << " " << extergy
                 << std::endl;
  }

  // in God we trust
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate external forces at t_{n+1} */
void STR::TimInt::ApplyForceExternal
(
  const double time,  //!< evaluation time
  const Teuchos::RCP<Epetra_Vector> dis,  //!< displacement state
  const Teuchos::RCP<Epetra_Vector> vel,  //!< velocity state
  Teuchos::RCP<Epetra_Vector>& fext  //!< external force
)
{
  ParameterList p;
  // other parameters needed by the elements
  p.set("total time", time);

  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState(0,"displacement", dis);
  if (damping_ == INPAR::STR::damp_material) discret_->SetState(0,"velocity", vel);
  // get load vector
  discret_->EvaluateNeumann(p, *fext);

  // go away
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate ordinary internal force, its stiffness at state */
void STR::TimInt::ApplyForceStiffInternal
(
  const double time,
  const double dt,
  const Teuchos::RCP<Epetra_Vector> dis,  // displacement state
  const Teuchos::RCP<Epetra_Vector> disi,  // residual displacements
  const Teuchos::RCP<Epetra_Vector> vel,  // velocity state
  Teuchos::RCP<Epetra_Vector> fint,  // internal force
  Teuchos::RCP<LINALG::SparseOperator> stiff  // stiffness matrix
)
{
  // create the parameters for the discretization
  Teuchos::ParameterList p;
  // action for elements
  const std::string action = "calc_struct_nlnstiff";
  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  if (pressure_ != Teuchos::null) p.set("volume", 0.0);
  // set vector values needed by elements
  discret_->ClearState();
  // extended SetState(0,...) in case of multiple dofsets (e.g. TSI)
  discret_->SetState(0,"residual displacement", disi);
  discret_->SetState(0,"displacement", dis);
  if (damping_ == INPAR::STR::damp_material) discret_->SetState(0,"velocity", vel);
  //fintn_->PutScalar(0.0);  // initialise internal force vector
  // set the temperature for the coupled problem
  if(tempn_!=Teuchos::null)
  {
    discret_->SetState(1,"temperature",tempn_);
  }
  discret_->Evaluate(p, stiff, Teuchos::null, fint, Teuchos::null, Teuchos::null);
  discret_->ClearState();

#if 0
  if (pressure_ != Teuchos::null)
    cout << "Total volume=" << std::scientific << p.get<double>("volume") << endl;
#endif

  // that's it
  return;
}

/*----------------------------------------------------------------------*/
/* evaluate ordinary internal force */
void STR::TimInt::ApplyForceInternal
(
  const double time,
  const double dt,
  const Teuchos::RCP<Epetra_Vector> dis,  // displacement state
  const Teuchos::RCP<Epetra_Vector> disi,  // incremental displacements
  const Teuchos::RCP<Epetra_Vector> vel,  // velocity state
  Teuchos::RCP<Epetra_Vector> fint  // internal force
)
{
  // create the parameters for the discretization
  ParameterList p;
  // action for elements
  const std::string action = "calc_struct_internalforce";
  p.set("action", action);
  // other parameters that might be needed by the elements
  p.set("total time", time);
  p.set("delta time", dt);
  if (pressure_ != Teuchos::null) p.set("volume", 0.0);
  // set vector values needed by elements
  discret_->ClearState();
  discret_->SetState("residual displacement", disi);  // these are incremental
  discret_->SetState("displacement", dis);
  if (damping_ == INPAR::STR::damp_material) discret_->SetState("velocity", vel);
  //fintn_->PutScalar(0.0);  // initialise internal force vector
  discret_->Evaluate(p, Teuchos::null, Teuchos::null,
                     fint, Teuchos::null, Teuchos::null);
  discret_->ClearState();

  // where the fun starts
  return;
}

/*----------------------------------------------------------------------*/
/* integrate */
void STR::TimInt::Integrate()
{
  // target time #timen_ and step #stepn_ already set

  // time loop
  // (NOTE: popp 03/2010: we have to add eps to avoid the very
  // awkward effect that the time loop stops one step too early)
  double eps = 1.0e-12;
  while ( (timen_ <= timemax_+eps) and (stepn_ <= stepmax_) )
  {
    // integrate time step
    // after this step we hold disn_, etc
    IntegrateStep();

    // update displacements, velocities, accelerations
    // after this call we will have disn_==dis_, etc
    UpdateStepState();

    // update time and step
    UpdateStepTime();

    // update everything on the element level
    UpdateStepElement();

    // write output
    OutputStep();

    // print info about finished time step
    PrintStep();

  }

  // print monitoring of time consumption
  TimeMonitor::summarize();

  // that's it
  return;
}

/*----------------------------------------------------------------------*/
/* get the temperature from the temperature discretization   dano 03/10 */
void STR::TimInt::ApplyTemperatures(
  Teuchos::RCP<Epetra_Vector> itemp  ///< the current temperature
  )
{
  tempn_ = itemp;
  // where the fun starts
  return;
}


/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET
