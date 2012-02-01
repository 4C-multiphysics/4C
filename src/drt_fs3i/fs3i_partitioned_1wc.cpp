/*!----------------------------------------------------------------------
\file fs3i_partitioned_1wc.cpp
\brief Algorithmic routines for partitioned solution approaches to
       fluid-structure-scalar-scalar interaction (FS3I) specifically
       related to one-way-coupled problem configurations

<pre>
Maintainers: Lena Yoshihara & Volker Gravemeier
             {yoshihara,vgravem}@lnm.mw.tum.de
             089/289-15303,-15245
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "fs3i_partitioned_1wc.H"

#include "../drt_fsi/fsi_monolithic_nox.H"
#include "../drt_scatra/passive_scatra_algorithm.H"
#include "../drt_inpar/inpar_scatra.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FS3I::PartFS3I_1WC::PartFS3I_1WC(const Epetra_Comm& comm)
  :PartFS3I(comm)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_1WC::Timeloop()
{
  // output of initial state
  ScatraOutput();

  fsi_->PrepareTimeloop();

  while (NotFinished())
  {
    IncrementTimeAndStep();
    DoFSIStep();
    SetFSISolution();
    DoScatraStep();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_1WC::DoFSIStep()
{
  fsi_->PrepareTimeStep();
  fsi_->TimeStep(fsi_);
  fsi_->PrepareOutput();
  fsi_->Update();
  fsi_->Output();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_1WC::DoScatraStep()
{
  if (Comm().MyPID()==0)
  {
    cout<<"\n***********************\n GAS TRANSPORT SOLVER \n***********************\n";
  }

  // first scatra field is associated with fluid, second scatra field is
  // associated with structure

  bool stopnonliniter=false;
  int itnum = 0;

  PrepareTimeStep();

  while (stopnonliniter==false)
  {
    ScatraEvaluateSolveIterUpdate();
    itnum++;
    if (ScatraConvergenceCheck(itnum))
      break;
  }

  UpdateScatraFields();
  ScatraOutput();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_1WC::PrepareTimeStep()
{
  // set mesh displacement field for present time step
  SetMeshDisp();

  // set velocity fields from fluid and structure solution
  // for present time step
  SetVelocityFields();

  // prepare time step for both fluid- and structure-based scatra field
  for (unsigned i=0; i<scatravec_.size(); ++i)
  {
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->ScaTraField().PrepareTimeStep();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FS3I::PartFS3I_1WC::ScatraConvergenceCheck(const int itnum)
{
  const Teuchos::ParameterList& fs3icontrol = DRT::Problem::Instance()->FS3IControlParams();
  INPAR::SCATRA::SolverType scatra_solvtype = DRT::INPUT::IntegralValue<INPAR::SCATRA::SolverType>(fs3icontrol,"SCATRA_SOLVERTYPE");

  double conresnorm(0.0);
  scatrarhs_->Norm2(&conresnorm);
  double incconnorm(0.0);
  scatraincrement_->Norm2(&incconnorm);

  switch(scatra_solvtype)
  {
  case INPAR::SCATRA::solvertype_linear_incremental:
  {
    // print the screen info
    if (Comm().MyPID()==0)
    {
      printf("\n+-------------------+-------------------+\n");
      printf("| norm of residual  | norm of increment |\n");
      printf("+-------------------+-------------------+\n");
      printf("|    %10.3E     |    %10.3E     |\n",
             conresnorm,incconnorm);
      printf("+-------------------+-------------------+\n\n");
    }
    return true;
  }
  break;
  case INPAR::SCATRA::solvertype_nonlinear:
  {
    // some input parameters for the scatra fields
    const Teuchos::ParameterList& scatradyn = DRT::Problem::Instance()->ScalarTransportDynamicParams();
    const int itemax = scatradyn.sublist("NONLINEAR").get<int>("ITEMAX");
    const double ittol = scatradyn.sublist("NONLINEAR").get<double>("CONVTOL");
    const double abstolres = scatradyn.sublist("NONLINEAR").get<double>("ABSTOLRES");

    double connorm(0.0);
    // set up vector of absolute concentrations
    Teuchos::RCP<Epetra_Vector> con = rcp(new Epetra_Vector(scatraincrement_->Map()));
    Teuchos::RCP<const Epetra_Vector> scatra1 = scatravec_[0]->ScaTraField().Phinp();
    Teuchos::RCP<const Epetra_Vector> scatra2 = scatravec_[1]->ScaTraField().Phinp();
    SetupCoupledScatraVector(con,scatra1,scatra2);
    con->Norm2(&connorm);

    // care for the case that nothing really happens in the concentration field
    if (connorm < 1e-5)
      connorm = 1.0;

    // print the screen info
    if (Comm().MyPID()==0)
    {
      printf("|  %3d/%3d   | %10.3E[L_2 ]  | %10.3E   | %10.3E   |",
             itnum,itemax,ittol,conresnorm,incconnorm/connorm);
    }

    // this is the convergence check
    // We always require at least one solve. We test the L_2-norm of the
    // current residual. Norm of residual is just printed for information
    if (conresnorm <= ittol and incconnorm/connorm <= ittol)
    {
      if (Comm().MyPID()==0)
      {
        // print 'finish line'
        printf("+------------+-------------------+--------------+--------------+\n");
      }
      return true;
    }

    // abort iteration, when there's nothing more to do! -> more robustness
    else if (conresnorm < abstolres)
    {
      // print 'finish line'
      if (Comm().MyPID()==0)
      {
        printf("+------------+-------------------+--------------+--------------+\n");
      }
      return true;
    }

    // warn if itemax is reached without convergence, but proceed to
    // next timestep...
    else if (itnum == itemax)
    {
      if (Comm().MyPID()==0)
      {
        printf("+---------------------------------------------------------------+\n");
        printf("|            >>>>>> not converged in itemax steps!              |\n");
        printf("+---------------------------------------------------------------+\n");
      }
      // yes, we stop the iteration
      return true;
    }
    else
      return false;
  }
  break;
  default:
    dserror("Illegal ScaTra solvertype in FS3I");
  }
  return false;
}

#endif
