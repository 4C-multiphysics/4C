/*
 * ssi_partitioned_1wc.cpp
 *
 *  Created on: Jun 29, 2012
 *      Author: vuong
 */

#include "ssi_partitioned_2wc.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_utils.H"

SSI::SSI_Part2WC::SSI_Part2WC(const Epetra_Comm& comm,
    const Teuchos::ParameterList& timeparams)
  : SSI_Part(comm, timeparams),
    scaincnp_(rcp(new Epetra_Vector(*(scatra_->ScaTraField().Phinp())))),
    dispincnp_(rcp(new Epetra_Vector(*(structure_()->Dispnp()))))
{
    // build a proxy of the structure discretization for the temperature field
    Teuchos::RCP<DRT::DofSet> structdofset
      = structure_->Discretization()->GetDofSetProxy();
    // build a proxy of the temperature discretization for the structure field
    Teuchos::RCP<DRT::DofSet> scatradofset
      = scatra_->ScaTraField().Discretization()->GetDofSetProxy();

    // check if scatra field has 2 discretizations, so that coupling is possible
    if (scatra_->ScaTraField().Discretization()->AddDofSet(structdofset)!=1)
      dserror("unexpected dof sets in scatra field");
    if (structure_->Discretization()->AddDofSet(scatradofset)!=1)
      dserror("unexpected dof sets in structure field");

    const Teuchos::ParameterList& ssicontrol = DRT::Problem::Instance()->SSIControlParams();
    // Get the parameters for the ConvergenceCheck
    itmax_ = ssicontrol.get<int>("ITEMAX"); // default: =10
    ittol_ = ssicontrol.get<double>("CONVTOL"); // default: =1e-6
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSI_Part2WC::Timeloop()
{
  //InitialCalculations();

  while (NotFinished())
  {
    PrepareTimeStep();

    OuterLoop();

    UpdateAndOutput();

  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSI_Part2WC::DoStructStep()
{
  if (Comm().MyPID() == 0)
  {
    cout
        << "\n***********************\n STRUCTURE SOLVER \n***********************\n";
  }

  // Newton-Raphson iteration
  structure_-> Solve();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSI_Part2WC::DoScatraStep()
{
  if (Comm().MyPID() == 0)
  {
    cout
        << "\n***********************\n  TRANSPORT SOLVER \n***********************\n";
  }

  // -------------------------------------------------------------------
  //                  solve nonlinear / linear equation
  // -------------------------------------------------------------------
  scatra_->ScaTraField().Solve();

}

/*----------------------------------------------------------------------*/
//prepare time step
/*----------------------------------------------------------------------*/
void SSI::SSI_Part2WC::PrepareTimeStep()
{
  IncrementTimeAndStep();
  PrintHeader();

  structure_-> PrepareTimeStep();
  SetStructSolution();
  scatra_->ScaTraField().PrepareTimeStep();
  SetScatraSolution();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSI_Part2WC::UpdateAndOutput()
{
  structure_-> PrepareOutput();

  structure_-> Update();
  scatra_->ScaTraField().Update();

  scatra_->ScaTraField().EvaluateErrorComparedToAnalyticalSol();

  structure_-> Output();
  scatra_->ScaTraField().Output();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSI_Part2WC::SetScatraSolution()
{
  structure_->ApplyTemperatures(scatra_->ScaTraField().Phinp());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSI_Part2WC::OuterLoop()
{
  int  itnum = 0;
  bool stopnonliniter = false;

  if (Comm().MyPID()==0)
  {
    cout<<"\n****************************************\n          OUTER ITERATION LOOP\n****************************************\n";
  }

  // initially solve coupled scalar transport equation system
  //DoScatraStep();

  while (stopnonliniter==false)
  {
    itnum++;

    // store temperature from first solution for convergence check (like in
    // elch_algorithm: use current values)
    scaincnp_->Update(1.0,*scatra_->ScaTraField().Phinp(),0.0);
    dispincnp_->Update(1.0,*structure_->Dispnp(),0.0);

    // set fluid- and structure-based scalar transport values required in FSI
    SetScatraSolution();

    if(itnum!=1)
      structure_->PreparePartitionStep();
    // solve structural system
    DoStructStep();

    // set mesh displacement and velocity fields
    SetStructSolution();

    // solve scalar transport equation
    DoScatraStep();
    //ScatraEvaluateSolveIterUpdate();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = ConvergenceCheck(itnum);
  }

  return;
}

/*----------------------------------------------------------------------*
 | convergence check for both fields (scatra & structure)
 *----------------------------------------------------------------------*/
bool SSI::SSI_Part2WC::ConvergenceCheck(
  int itnum
  )
{

  // convergence check based on the temperature increment
  bool stopnonliniter = false;

  //    | temperature increment |_2
  //  -------------------------------- < Tolerance
  //     | temperature_n+1 |_2

  // variables to save different L2 - Norms
  // define L2-norm of incremental temperature and temperature
  // here: only the temperature field is checked for convergence!!!
  double scaincnorm_L2(0.0);
  double scanorm_L2(0.0);
  double dispincnorm_L2(0.0);
  double dispnorm_L2(0.0);

  // build the current temperature increment Inc T^{i+1}
  // \f Delta T^{k+1} = Inc T^{k+1} = T^{k+1} - T^{k}  \f
  scaincnp_->Update(1.0,*(scatra_->ScaTraField().Phinp()),-1.0);
  dispincnp_->Update(1.0,*(structure_->Dispnp()),-1.0);

  // build the L2-norm of the temperature increment and the temperature
  scaincnp_->Norm2(&scaincnorm_L2);
  scatra_->ScaTraField().Phinp()->Norm2(&scanorm_L2);
  dispincnp_->Norm2(&dispincnorm_L2);
  structure_->Dispnp()->Norm2(&dispnorm_L2);

  // care for the case that there is (almost) zero temperature
  // (usually not required for temperature)
  if (scanorm_L2 < 1e-6) scanorm_L2 = 1.0;
  if (dispnorm_L2 < 1e-6) dispnorm_L2 = 1.0;

  // print the incremental based convergence check to the screen
  if (Comm().MyPID()==0 )//and PrintScreenEvry() and (Step()%PrintScreenEvry()==0))
  {
    cout<<"\n";
    cout<<"***********************************************************************************\n";
    cout<<"    OUTER ITERATION STEP    \n";
    cout<<"***********************************************************************************\n";
    printf("+--------------+------------------------+--------------------+--------------------+\n");
    printf("|-  step/max  -|-  tol      [norm]     -|--  temp-inc      --|--  disp-inc      --|\n");
    printf("|   %3d/%3d    |  %10.3E[L_2 ]      | %10.3E         | %10.3E         |",
         itnum,itmax_,ittol_,scaincnorm_L2/scanorm_L2,dispincnorm_L2/dispnorm_L2);
    printf("\n");
    printf("+--------------+------------------------+--------------------+--------------------+\n");
  }

  // converged
  if ((scaincnorm_L2/scanorm_L2 <= ittol_) &&
      (dispincnorm_L2/dispnorm_L2 <= ittol_))
  {
    stopnonliniter = true;
    if (Comm().MyPID()==0 ) //and PrintScreenEvry() and (Step()%PrintScreenEvry()==0))
    {
      printf("\n");
      printf("|  Outer Iteration loop converged after iteration %3d/%3d !                       |\n", itnum,itmax_);
      printf("+--------------+------------------------+--------------------+--------------------+\n");
    }
  }

  // warn if itemax is reached without convergence, but proceed to next
  // timestep
  if ((itnum==itmax_) and
       ((scaincnorm_L2/scanorm_L2 > ittol_) || (dispincnorm_L2/dispnorm_L2 > ittol_))
     )
  {
    stopnonliniter = true;
    if ((Comm().MyPID()==0) )//and PrintScreenEvry() and (Step()%PrintScreenEvry()==0))
    {
      printf("|     >>>>>> not converged in itemax steps!                                       |\n");
      printf("+--------------+------------------------+--------------------+--------------------+\n");
      printf("\n");
      printf("\n");
    }
  }

  return stopnonliniter;
}
