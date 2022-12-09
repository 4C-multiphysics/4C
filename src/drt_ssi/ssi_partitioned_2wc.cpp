/*----------------------------------------------------------------------*/
/*! \file
 \brief two way coupled partitioned scalar structure interaction

 \level 2


 *------------------------------------------------------------------------------------------------*/

#include "ssi_partitioned_2wc.H"
#include "drt_globalproblem.H"
#include "linalg_utils_sparse_algebra_create.H"

#include "ad_str_ssiwrapper.H"
#include "adapter_scatra_base_algorithm.H"

#include "scatra_ele.H"
#include "scatra_timint_implicit.H"
#include <Teuchos_Time.hpp>

/*----------------------------------------------------------------------*
 | constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WC::SSIPart2WC(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart(comm, globaltimeparams),
      scaincnp_(Teuchos::null),
      dispincnp_(Teuchos::null),
      ittol_(-1.0),
      itmax_(-1)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during Setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Init this class                                          rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::Init(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIPart::Init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  // call the SSI parameter lists
  const Teuchos::ParameterList& ssicontrol = DRT::Problem::Instance()->SSIControlParams();
  const Teuchos::ParameterList& ssicontrolpart =
      DRT::Problem::Instance()->SSIControlParams().sublist("PARTITIONED");

  // do some checks
  {
    auto structtimealgo =
        DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(structparams, "DYNAMICTYP");
    if (structtimealgo == INPAR::STR::dyna_statics)
    {
      dserror(
          "If you use statics as the structural time integrator no velocities will be calculated "
          "and hence"
          "the deformations will not be applied to the scalar transport problem!");
    }

    auto convform = DRT::INPUT::IntegralValue<INPAR::SCATRA::ConvForm>(scatraparams, "CONVFORM");
    if (convform == INPAR::SCATRA::convform_convective)
    {
      // get scatra discretization
      Teuchos::RCP<DRT::Discretization> scatradis =
          DRT::Problem::Instance()->GetDis(scatra_disname);

      // loop over all elements of scatra discretization to check if impltype is correct or not
      for (int i = 0; i < scatradis->NumMyColElements(); ++i)
      {
        if ((dynamic_cast<DRT::ELEMENTS::Transport*>(scatradis->lColElement(i)))->ImplType() !=
            INPAR::SCATRA::impltype_refconcreac)
        {
          dserror(
              "If the scalar transport problem is solved on the deforming domain, the conservative "
              "form must be "
              "used to include volume changes! Set 'CONVFORM' to 'conservative' in the SCALAR "
              "TRANSPORT DYNAMIC section or use RefConcReac as ScatraType!");
        }
      }
    }
  }

  if (DiffTimeStepSize())
  {
    dserror("Different time stepping for two way coupling not implemented yet.");
  }

  // Get the parameters for the ConvergenceCheck
  itmax_ = ssicontrol.get<int>("ITEMAX");          // default: =10
  ittol_ = ssicontrolpart.get<double>("CONVTOL");  // default: =1e-6
}

/*----------------------------------------------------------------------*
 | Setup this class                                          rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::Setup()
{
  // call setup in base class
  SSI::SSIPart::Setup();

  // construct increment vectors
  scaincnp_ = LINALG::CreateVector(*ScaTraField()->Discretization()->DofRowMap(0), true);
  dispincnp_ = LINALG::CreateVector(*StructureField()->DofRowMap(0), true);
}

/*----------------------------------------------------------------------*
 | Timeloop for 2WC SSI problems                             Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::Timeloop()
{
  // safety checks
  CheckIsInit();
  CheckIsSetup();

  PrepareTimeLoop();

  // time loop
  while (NotFinished() and ScaTraField()->NotFinished())
  {
    PrepareTimeStep();

    // store time before calling nonlinear solver
    double time = Teuchos::Time::wallTime();

    OuterLoop();

    // determine time spent by nonlinear solver and take maximum over all processors via
    // communication
    double mydtnonlinsolve(Teuchos::Time::wallTime() - time), dtnonlinsolve(0.);
    Comm().MaxAll(&mydtnonlinsolve, &dtnonlinsolve, 1);

    // output performance statistics associated with nonlinear solver into *.csv file if applicable
    if (DRT::INPUT::IntegralValue<int>(
            *ScaTraField()->ScatraParameterList(), "OUTPUTNONLINSOLVERSTATS"))
      ScaTraField()->OutputNonlinSolverStats(IterationCount(), dtnonlinsolve, Step(), Comm());

    UpdateAndOutput();
  }
}

/*----------------------------------------------------------------------*
 | Solve structure filed                                     Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::DoStructStep()
{
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n***********************\n STRUCTURE SOLVER \n***********************\n";
  }

  // Newton-Raphson iteration
  StructureField()->Solve();

  if (IsS2IKineticsWithPseudoContact()) StructureField()->DetermineStressStrain();

  //  set mesh displacement and velocity fields
  return SetStructSolution(
      StructureField()->Dispnp(), StructureField()->Velnp(), IsS2IKineticsWithPseudoContact());
}

/*----------------------------------------------------------------------*
 | Solve Scatra field                                        Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::DoScatraStep()
{
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n***********************\n  TRANSPORT SOLVER \n***********************\n";
  }

  // -------------------------------------------------------------------
  //                  solve nonlinear / linear equation
  // -------------------------------------------------------------------
  ScaTraField()->Solve();

  // set structure-based scalar transport values
  SetScatraSolution(ScaTraField()->Phinp());

  // set micro scale value (projected to macro scale) to structure field
  if (MacroScale()) SetMicroScatraSolution(ScaTraField()->PhinpMicro());

  // evaluate temperature from function and set to structural discretization
  EvaluateAndSetTemperatureField();
}

/*----------------------------------------------------------------------*
 | Solve Scatra field                                        Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::PreOperator1()
{
  if (IterationCount() != 1 and UseOldStructureTimeInt())
  {
    // NOTE: the predictor is NOT called in here. Just the screen output is not correct.
    // we only get norm of the evaluation of the structure problem
    StructureField()->PreparePartitionStep();
  }
}

/*----------------------------------------------------------------------*/
// prepare time step
/*----------------------------------------------------------------------*/
void SSI::SSIPart2WC::PrepareTimeLoop()
{
  // initial output
  constexpr bool force_prepare = true;
  StructureField()->PrepareOutput(force_prepare);
  StructureField()->Output();
  SetStructSolution(StructureField()->Dispnp(), StructureField()->Velnp(), false);
  ScaTraField()->PrepareTimeLoop();
}

/*----------------------------------------------------------------------*/
// prepare time step
/*----------------------------------------------------------------------*/
void SSI::SSIPart2WC::PrepareTimeStep(bool printheader)
{
  IncrementTimeAndStep();

  SetStructSolution(StructureField()->Dispnp(), StructureField()->Velnp(), false);
  ScaTraField()->PrepareTimeStep();

  // if adaptive time stepping and different time step size: calculate time step in scatra
  // (PrepareTimeStep() of Scatra) and pass to other fields
  if (ScaTraField()->TimeStepAdapted()) SetDtFromScaTraToSSI();

  SetScatraSolution(ScaTraField()->Phinp());
  if (MacroScale()) SetMicroScatraSolution(ScaTraField()->PhinpMicro());

  // NOTE: the predictor of the structure is called in here
  StructureField()->PrepareTimeStep();

  if (printheader) PrintHeader();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIPart2WC::UpdateAndOutput()
{
  constexpr bool force_prepare = false;
  StructureField()->PrepareOutput(force_prepare);

  StructureField()->Update();
  ScaTraField()->Update();

  ScaTraField()->EvaluateErrorComparedToAnalyticalSol();

  StructureField()->Output();
  ScaTraField()->Output();
}


/*----------------------------------------------------------------------*
 | update the current states in every iteration             rauch 05/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::IterUpdateStates()
{
  // store last solutions (current states).
  // will be compared in ConvergenceCheck to the solutions,
  // obtained from the next Struct and Scatra steps.
  scaincnp_->Update(1.0, *ScaTraField()->Phinp(), 0.0);
  dispincnp_->Update(1.0, *StructureField()->Dispnp(), 0.0);
}  // IterUpdateStates()


/*----------------------------------------------------------------------*
 | Outer Timeloop for 2WC SSi without relaxation            rauch 06/17 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WC::OuterLoop()
{
  // reset iteration number
  ResetIterationCount();
  bool stopnonliniter = false;

  if (Comm().MyPID() == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";
  }

  while (!stopnonliniter)
  {
    // increment iteration number
    IncrementIterationCount();

    // update the states to the last solutions obtained
    IterUpdateStates();

    // standard ssi_2wc :
    // 1.) solve structural system
    // 2.) set disp and vel states in scatra field
    PreOperator1();
    Operator1();
    PostOperator1();

    // standard ssi_2wc :
    // 1.) solve scalar transport equation
    // 2.) set phi state in structure field
    PreOperator2();
    Operator2();
    PostOperator2();

    // check convergence for all fields
    // stop iteration loop if converged
    stopnonliniter = ConvergenceCheck(IterationCount());
  }
}

/*----------------------------------------------------------------------*
 | convergence check for both fields (scatra & structure) (copied form tsi)
 *----------------------------------------------------------------------*/
bool SSI::SSIPart2WC::ConvergenceCheck(int itnum)
{
  // convergence check based on the scalar increment
  bool stopnonliniter = false;

  //    | scalar increment |_2
  //  -------------------------------- < Tolerance
  //    | scalar state n+1 |_2
  //
  // AND
  //
  //    | scalar increment |_2
  //  -------------------------------- < Tolerance , with n := global length of vector
  //        dt * sqrt(n)
  //
  // The same is checked for the structural displacements.
  //

  // variables to save different L2 - Norms
  // define L2-norm of incremental scalar and scalar
  double scaincnorm_L2(0.0);
  double scanorm_L2(0.0);
  double dispincnorm_L2(0.0);
  double dispnorm_L2(0.0);

  // build the current scalar increment Inc T^{i+1}
  // \f Delta T^{k+1} = Inc T^{k+1} = T^{k+1} - T^{k}  \f
  scaincnp_->Update(1.0, *(ScaTraField()->Phinp()), -1.0);
  dispincnp_->Update(1.0, *(StructureField()->Dispnp()), -1.0);

  // build the L2-norm of the scalar increment and the scalar
  scaincnp_->Norm2(&scaincnorm_L2);
  ScaTraField()->Phinp()->Norm2(&scanorm_L2);
  dispincnp_->Norm2(&dispincnorm_L2);
  StructureField()->Dispnp()->Norm2(&dispnorm_L2);

  // care for the case that there is (almost) zero scalar
  if (scanorm_L2 < 1e-6) scanorm_L2 = 1.0;
  if (dispnorm_L2 < 1e-6) dispnorm_L2 = 1.0;

  // print the incremental based convergence check to the screen
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n";
    std::cout
        << "***********************************************************************************\n";
    std::cout << "    OUTER ITERATION STEP    \n";
    std::cout
        << "***********************************************************************************\n";
    printf(
        "+--------------+---------------------+----------------+------------------+----------------"
        "----+------------------+\n");
    printf(
        "|-  step/max  -|-  tol      [norm]  -|-  scalar-inc  -|-  disp-inc      -|-  "
        "scalar-rel-inc  -|-  disp-rel-inc  -|\n");
    printf(
        "|   %3d/%3d    |  %10.3E[L_2 ]   |  %10.3E    |  %10.3E      |  %10.3E        |  %10.3E   "
        "   |",
        itnum, itmax_, ittol_, scaincnorm_L2 / Dt() / sqrt(scaincnp_->GlobalLength()),
        dispincnorm_L2 / Dt() / sqrt(dispincnp_->GlobalLength()), scaincnorm_L2 / scanorm_L2,
        dispincnorm_L2 / dispnorm_L2);
    printf("\n");
    printf(
        "+--------------+---------------------+----------------+------------------+----------------"
        "----+------------------+\n");
  }

  // converged
  if (((scaincnorm_L2 / scanorm_L2) <= ittol_) and ((dispincnorm_L2 / dispnorm_L2) <= ittol_) and
      ((dispincnorm_L2 / Dt() / sqrt(dispincnp_->GlobalLength())) <= ittol_) and
      ((scaincnorm_L2 / Dt() / sqrt(scaincnp_->GlobalLength())) <= ittol_))
  {
    stopnonliniter = true;
    if (Comm().MyPID() == 0)
    {
      printf(
          "|  Outer Iteration loop converged after iteration %3d/%3d !                             "
          "                         |\n",
          itnum, itmax_);
      printf(
          "+--------------+---------------------+----------------+------------------+--------------"
          "------+------------------+\n");
    }
  }

  // stop if itemax is reached without convergence
  // timestep
  if ((itnum == itmax_) and
      (((scaincnorm_L2 / scanorm_L2) > ittol_) or ((dispincnorm_L2 / dispnorm_L2) > ittol_) or
          ((dispincnorm_L2 / Dt() / sqrt(dispincnp_->GlobalLength())) > ittol_) or
          (scaincnorm_L2 / Dt() / sqrt(scaincnp_->GlobalLength())) > ittol_))
  {
    stopnonliniter = true;
    if ((Comm().MyPID() == 0))
    {
      printf(
          "|     >>>>>> not converged in itemax steps!                                             "
          "                         |\n");
      printf(
          "+--------------+---------------------+----------------+------------------+--------------"
          "------+------------------+\n");
      printf("\n");
      printf("\n");
    }
    dserror("The partitioned SSI solver did not converge in ITEMAX steps!");
  }

  return stopnonliniter;
}

/*----------------------------------------------------------------------*
 | calculate velocities by a FD approximation                Thon 14/11 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> SSI::SSIPart2WC::CalcVelocity(Teuchos::RCP<const Epetra_Vector> dispnp)
{
  Teuchos::RCP<Epetra_Vector> vel = Teuchos::null;
  // copy D_n onto V_n+1
  vel = Teuchos::rcp(new Epetra_Vector(*(StructureField()->Dispn())));
  // calculate velocity with timestep Dt()
  //  V_n+1^k = (D_n+1^k - D_n) / Dt
  vel->Update(1. / Dt(), *dispnp, -1. / Dt());

  return vel;
}  // CalcVelocity()

/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCSolidToScatraRelax::SSIPart2WCSolidToScatraRelax(
    const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WC(comm, globaltimeparams), omega_(-1.0)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during Setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Init this class                                          rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelax::Init(const Epetra_Comm& comm,
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& scatraparams,
    const Teuchos::ParameterList& structparams, const std::string& struct_disname,
    const std::string& scatra_disname, bool isAle)
{
  // call init of base class
  SSI::SSIPart2WC::Init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  const Teuchos::ParameterList& ssicontrolpart =
      DRT::Problem::Instance()->SSIControlParams().sublist("PARTITIONED");

  // Get minimal relaxation parameter from input file
  omega_ = ssicontrolpart.get<double>("STARTOMEGA");
}

/*----------------------------------------------------------------------*
 | Outer Timeloop for 2WC SSi with relaxed displacements     Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelax::OuterLoop()
{
  ResetIterationCount();
  bool stopnonliniter = false;

  if (Comm().MyPID() == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";
  }

  // these are the relaxed inputs
  Teuchos::RCP<Epetra_Vector> dispnp =
      LINALG::CreateVector(*(StructureField()->DofRowMap(0)), true);
  Teuchos::RCP<Epetra_Vector> velnp = LINALG::CreateVector(*(StructureField()->DofRowMap(0)), true);

  while (!stopnonliniter)
  {
    IncrementIterationCount();

    if (IterationCount() == 1)
    {
      dispnp->Update(1.0, *(StructureField()->Dispnp()), 0.0);  // TSI does Dispn()
      velnp->Update(1.0, *(StructureField()->Velnp()), 0.0);
    }

    // store scalars and displacements for the convergence check later
    scaincnp_->Update(1.0, *ScaTraField()->Phinp(), 0.0);
    dispincnp_->Update(1.0, *dispnp, 0.0);

    // begin nonlinear solver / outer iteration ***************************

    // set relaxed mesh displacements and velocity field
    SetStructSolution(dispnp, velnp, IsS2IKineticsWithPseudoContact());

    // solve scalar transport equation
    DoScatraStep();

    // prepare a partitioned structure step
    if (IterationCount() != 1 and UseOldStructureTimeInt())
      StructureField()->PreparePartitionStep();

    // solve structural system
    DoStructStep();

    // end nonlinear solver / outer iteration *****************************

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = ConvergenceCheck(IterationCount());

    // get relaxation parameter
    CalcOmega(omega_, IterationCount());

    // do the relaxation
    // d^{i+1} = omega^{i+1} . d^{i+1} + (1- omega^{i+1}) d^i
    //         = d^i + omega^{i+1} * ( d^{i+1} - d^i )
    dispnp->Update(omega_, *dispincnp_, 1.0);

    // since the velocity field has to fit to the relaxated displacements we also have to relaxate
    // them.
    if (UseOldStructureTimeInt())
    {
      // since the velocity depends nonlinear on the displacements we can just approximate them via
      // finite differences here.
      velnp = CalcVelocity(dispnp);
    }
    else
    {
      // consistent derivation of velocity field from displacement field
      // SetState(dispnp) will automatically be undone during the next evaluation of the structural
      // field
      StructureField()->SetState(dispnp);
      velnp->Update(1., *StructureField()->Velnp(), 0.);
    }
  }
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter                            Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelax::CalcOmega(double& omega, const int itnum)
{
  // nothing to do in here since we have a constant relaxation parameter: omega != startomega_;
  if (Comm().MyPID() == 0)
    std::cout << "Fixed relaxation parameter omega is: " << omega << std::endl;
}

/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCSolidToScatraRelaxAitken::SSIPart2WCSolidToScatraRelaxAitken(
    const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WCSolidToScatraRelax(comm, globaltimeparams), dispincnpold_(Teuchos::null)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call he setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during Setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                          fang 01/18 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelaxAitken::Setup()
{
  // call setup of base class
  SSI::SSIPart2WC::Setup();

  // setup old scatra increment vector
  dispincnpold_ = LINALG::CreateVector(*StructureField()->DofRowMap(0), true);
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter via Aitken                 Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCSolidToScatraRelaxAitken::CalcOmega(double& omega, const int itnum)
{
  const Teuchos::ParameterList& ssicontrolpart =
      DRT::Problem::Instance()->SSIControlParams().sublist("PARTITIONED");

  // Get maximal relaxation parameter from input file
  const double maxomega = ssicontrolpart.get<double>("MAXOMEGA");
  // Get minimal relaxation parameter from input file
  const double minomega = ssicontrolpart.get<double>("MINOMEGA");

  // calculate difference of current (i+1) and old (i) residual vector
  // dispincnpdiff = ( r^{i+1}_{n+1} - r^i_{n+1} )
  Teuchos::RCP<Epetra_Vector> dispincnpdiff =
      LINALG::CreateVector(*(StructureField()->DofRowMap(0)), true);
  dispincnpdiff->Update(
      1.0, *dispincnp_, (-1.0), *dispincnpold_, 0.0);  // update r^{i+1}_{n+1} - r^i_{n+1}

  double dispincnpdiffnorm = 0.0;
  dispincnpdiff->Norm2(&dispincnpdiffnorm);
  if (dispincnpdiffnorm <= 1e-06 and Comm().MyPID() == 0)
  {
    std::cout << "Warning: The structure increment is to small in order to use it for Aitken "
                 "relaxation. Using the previous Omega instead!"
              << std::endl;
  }

  // calculate dot product
  double dispincsdot = 0.0;  // delsdot = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
  dispincnpdiff->Dot(*dispincnp_, &dispincsdot);

  if (itnum != 1 and dispincnpdiffnorm > 1e-06)
  {  // relaxation parameter
    // omega^{i+1} = 1- mu^{i+1} and nu^{i+1} = nu^i + (nu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) /
    // |r^{i+1} - r^{i}|^2 results in
    omega = omega *
            (1 - (dispincsdot) / (dispincnpdiffnorm *
                                     dispincnpdiffnorm));  // compare e.g. PhD thesis U. Kuettler

    // we force omega to be in the range defined in the input file
    if (omega < minomega)
    {
      if (Comm().MyPID() == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value smaller than MINOMEGA!"
                  << std::endl;
      }
      omega = minomega;
    }
    if (omega > maxomega)
    {
      if (Comm().MyPID() == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value bigger than MAXOMEGA!"
                  << std::endl;
      }
      omega = maxomega;
    }
  }

  // else //if itnum==1 nothing is to do here since we want to take the last omega from the previous
  // step
  if (Comm().MyPID() == 0)
    std::cout << "Using Aitken the relaxation parameter omega was estimated to: " << omega
              << std::endl;

  // update history vector old increment r^i_{n+1}
  dispincnpold_->Update(1.0, *dispincnp_, 0.0);
}


/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCScatraToSolidRelax::SSIPart2WCScatraToSolidRelax(
    const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WC(comm, globaltimeparams), omega_(-1.0)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during Setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelax::Init(const Epetra_Comm& comm,
    const Teuchos::ParameterList& globaltimeparams, const Teuchos::ParameterList& scatraparams,
    const Teuchos::ParameterList& structparams, const std::string& struct_disname,
    const std::string& scatra_disname, bool isAle)
{
  // call setup of base class
  SSI::SSIPart2WC::Init(
      comm, globaltimeparams, scatraparams, structparams, struct_disname, scatra_disname, isAle);

  const Teuchos::ParameterList& ssicontrolpart =
      DRT::Problem::Instance()->SSIControlParams().sublist("PARTITIONED");

  // Get start relaxation parameter from input file
  omega_ = ssicontrolpart.get<double>("STARTOMEGA");

  if (Comm().MyPID() == 0)
  {
    std::cout << "\n#########################################################################\n  "
              << std::endl;
    std::cout << "The ScatraToSolid relaxations are not well tested . Keep your eyes open!  "
              << std::endl;
    std::cout << "\n#########################################################################\n  "
              << std::endl;
  }
}

/*----------------------------------------------------------------------*
 | Outer Timeloop for 2WC SSi with relaxed scalar             Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelax::OuterLoop()
{
  ResetIterationCount();
  bool stopnonliniter = false;

  if (Comm().MyPID() == 0)
  {
    std::cout << "\n****************************************\n          OUTER ITERATION "
                 "LOOP\n****************************************\n";
  }

  // this is the relaxed input
  Teuchos::RCP<Epetra_Vector> phinp =
      LINALG::CreateVector(*ScaTraField()->Discretization()->DofRowMap(0), true);

  while (!stopnonliniter)
  {
    IncrementIterationCount();

    if (IterationCount() == 1)
    {
      phinp->Update(1.0, *(ScaTraField()->Phinp()), 0.0);  // TSI does Dispn()
    }

    // store scalars and displacements for the convergence check later
    scaincnp_->Update(1.0, *phinp, 0.0);
    dispincnp_->Update(1.0, *StructureField()->Dispnp(), 0.0);


    // begin nonlinear solver / outer iteration ***************************

    // set relaxed scalars
    SetScatraSolution(phinp);

    // evaluate temperature from function and set to structural discretization
    EvaluateAndSetTemperatureField();

    // prepare partitioned structure step
    if (IterationCount() != 1 and UseOldStructureTimeInt())
      StructureField()->PreparePartitionStep();

    // solve structural system
    DoStructStep();

    // solve scalar transport equation
    DoScatraStep();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = ConvergenceCheck(IterationCount());

    // get relaxation parameter
    CalcOmega(omega_, IterationCount());

    // do the relaxation
    // d^{i+1} = omega^{i+1} . d^{i+1} + (1- omega^{i+1}) d^i
    //         = d^i + omega^{i+1} * ( d^{i+1} - d^i )
    phinp->Update(omega_, *scaincnp_, 1.0);
  }
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter                            Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelax::CalcOmega(double& omega, const int itnum)
{
  // nothing to do in here since we have a constant relaxation parameter: omega != startomega_;
  if (Comm().MyPID() == 0)
    std::cout << "Fixed relaxation parameter omega is: " << omega << std::endl;
}

/*----------------------------------------------------------------------*
 | Constructor                                               Thon 12/14 |
 *----------------------------------------------------------------------*/
SSI::SSIPart2WCScatraToSolidRelaxAitken::SSIPart2WCScatraToSolidRelaxAitken(
    const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams)
    : SSIPart2WCScatraToSolidRelax(comm, globaltimeparams), scaincnpold_(Teuchos::null)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during Setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Setup this class                                          fang 01/18 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelaxAitken::Setup()
{
  // call setup of base class
  SSI::SSIPart2WC::Setup();

  // setup old scatra increment vector
  scaincnpold_ = LINALG::CreateVector(*ScaTraField()->Discretization()->DofRowMap(), true);
}

/*----------------------------------------------------------------------*
 | Calculate relaxation parameter via Aitken                 Thon 12/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIPart2WCScatraToSolidRelaxAitken::CalcOmega(double& omega, const int itnum)
{
  const Teuchos::ParameterList& ssicontrolpart =
      DRT::Problem::Instance()->SSIControlParams().sublist("PARTITIONED");

  // Get maximal relaxation parameter from input file
  const double maxomega = ssicontrolpart.get<double>("MAXOMEGA");
  // Get minimal relaxation parameter from input file
  const double minomega = ssicontrolpart.get<double>("MINOMEGA");

  // scaincnpdiff =  r^{i+1}_{n+1} - r^i_{n+1}
  Teuchos::RCP<Epetra_Vector> scaincnpdiff =
      LINALG::CreateVector(*ScaTraField()->Discretization()->DofRowMap(0), true);
  scaincnpdiff->Update(1.0, *scaincnp_, (-1.0), *scaincnpold_, 0.0);

  double scaincnpdiffnorm = 0.0;
  scaincnpdiff->Norm2(&scaincnpdiffnorm);

  if (scaincnpdiffnorm <= 1e-06 and Comm().MyPID() == 0)
  {
    std::cout << "Warning: The scalar increment is to small in order to use it for Aitken "
                 "relaxation. Using the previous omega instead!"
              << std::endl;
  }

  // calculate dot product
  double scaincsdot = 0.0;  // delsdot = ( r^{i+1}_{n+1} - r^i_{n+1} )^T . r^{i+1}_{n+1}
  scaincnpdiff->Dot(*scaincnp_, &scaincsdot);

  if (itnum != 1 and scaincnpdiffnorm > 1e-06)
  {  // relaxation parameter
    // omega^{i+1} = 1- mu^{i+1} and nu^{i+1} = nu^i + (nu^i -1) . (r^{i+1} - r^i)^T . (-r^{i+1}) /
    // |r^{i+1} - r^{i}|^2 results in
    omega = omega *
            (1 - (scaincsdot) /
                     (scaincnpdiffnorm * scaincnpdiffnorm));  // compare e.g. PhD thesis U. Kuettler

    // we force omega to be in the range defined in the input file
    if (omega < minomega)
    {
      if (Comm().MyPID() == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value smaller than MINOMEGA!"
                  << std::endl;
      }
      omega = minomega;
    }
    if (omega > maxomega)
    {
      if (Comm().MyPID() == 0)
      {
        std::cout << "Warning: The calculation of the relaxation parameter omega via Aitken did "
                     "lead to a value bigger than MAXOMEGA!"
                  << std::endl;
      }
      omega = maxomega;
    }
  }

  // else //if itnum==1 nothing is to do here since we want to take the last omega from the previous
  // step
  if (Comm().MyPID() == 0)
    std::cout << "Using Aitken the relaxation parameter omega was estimated to: " << omega
              << std::endl;

  // update history vector old increment r^i_{n+1}
  scaincnpold_->Update(1.0, *scaincnp_, 0.0);
}
