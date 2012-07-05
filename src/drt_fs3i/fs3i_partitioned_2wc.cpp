/*!----------------------------------------------------------------------
\file fs3i_partitioned_2wc.cpp
\brief Algorithmic routines for partitioned solution approaches to
       fluid-structure-scalar-scalar interaction (FS3I) specifically
       related to two-way-coupled problem configurations

<pre>
Maintainers: Volker Gravemeier & Lena Yoshihara
             {vgravem,yoshihara}@lnm.mw.tum.de
             089/289-15245,-15303
</pre>

*----------------------------------------------------------------------*/


#include "fs3i_partitioned_2wc.H"

#include "../drt_fsi/fsi_monolithic.H"
#include "../drt_adapter/ad_str_fsiwrapper.H"
#include "../drt_scatra/passive_scatra_algorithm.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FS3I::PartFS3I_2WC::PartFS3I_2WC(const Epetra_Comm& comm)
  :PartFS3I(comm)
{
  //---------------------------------------------------------------------
  // get input parameters for two-way-coupled problems, which is
  // thermo-fluid-structure interaction, for the time being
  //---------------------------------------------------------------------
  const Teuchos::ParameterList& fs3icontrol = DRT::Problem::Instance()->FS3IControlParams();
  ittol_ = fs3icontrol.get<double>("CONVTOL");
  itmax_ = fs3icontrol.get<int>("ITEMAX");

  // flag for constant thermodynamic pressure
  consthermpress_ = fs3icontrol.get<string>("CONSTHERMPRESS");

  // define fluid- and structure-based scalar transport problem
  fluidscatra_     = scatravec_[0];
  structurescatra_ = scatravec_[1];

  // generate proxy of dof set for structure-based scalar transport
  // problem to be used by structure field
  Teuchos::RCP<DRT::DofSet> structurescatradofset = structurescatra_->ScaTraField().Discretization()->GetDofSetProxy();

  // check number of dof sets in structure field
  if (fsi_->StructureField()->Discretization()->AddDofSet(structurescatradofset)!=1)
    dserror("Incorrect number of dof sets in structure field!");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_2WC::Timeloop()
{
  InitialCalculations();

  while (NotFinished())
  {
    IncrementTimeAndStep();

    PrepareTimeStep();

    OuterLoop();

    TimeUpdateAndOutput();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_2WC::InitialCalculations()
{
  // set initial values for mesh displacement field
  SetMeshDisp();

  // set initial fluid velocity field for evaluation of initial scalar
  // time derivative in fluid-based scalar transport
  fluidscatra_->ScaTraField().SetVelocityField(fsi_->FluidField().Velnp(),
                                               Teuchos::null,
                                               Teuchos::null,
                                               Teuchos::null,
                                               Teuchos::null,
                                               fsi_->FluidField().Discretization());

  // set initial value of thermodynamic pressure in fluid-based scalar
  // transport
  fluidscatra_->ScaTraField().SetInitialThermPressure();

  // energy conservation: compute initial time derivative of therm. pressure
  // mass conservation: compute initial mass (initial time deriv. assumed zero)
  if (consthermpress_=="No_energy")
    fluidscatra_->ScaTraField().ComputeInitialThermPressureDeriv();
  else if (consthermpress_=="No_mass")
    fluidscatra_->ScaTraField().ComputeInitialMass();

  // set initial scalar field and thermodynamic pressure for evaluation of
  // Neumann boundary conditions in fluid at beginning of first time step
  fsi_->FluidField().SetTimeLomaFields(fluidscatra_->ScaTraField().Phinp(),
                                       fluidscatra_->ScaTraField().ThermPressNp(),
                                       null,
                                       fluidscatra_->ScaTraField().Discretization());
  // prepare time loop for FSI
  fsi_->PrepareTimeloop();

  // output of initial state for scalar values
  //ScatraOutput();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_2WC::PrepareTimeStep()
{
  // set mesh displacement and velocity fields for present time step
  SetFSISolution();

  // prepare time step for both fluid- and structure-based scatra field
  // (+ computation of initial scalar time derivative in first time step)
  fluidscatra_->ScaTraField().PrepareTimeStep();
  structurescatra_->ScaTraField().PrepareTimeStep();

  // predict thermodynamic pressure and time derivative
  // (if not constant or based on mass conservation)
  if (consthermpress_=="No_energy")
    fluidscatra_->ScaTraField().PredictThermPressure();

  // prepare time step for fluid, structure and ALE fields
  fsi_->PrepareTimeStep();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_2WC::OuterLoop()
{
#ifdef PARALLEL
  const Epetra_Comm& comm = scatravec_[0]->ScaTraField().Discretization()->Comm();
#else
  Epetra_SerialComm comm;
#endif

  int  itnum = 0;
  bool stopnonliniter = false;

  if (comm.MyPID()==0)
  {
    cout<<"\n****************************************\n          OUTER ITERATION LOOP\n****************************************\n";

    printf("TIME: %11.4E/%11.4E  DT = %11.4E  %s  STEP = %4d/%4d\n",
           fluidscatra_->ScaTraField().Time(),timemax_,dt_,fluidscatra_->ScaTraField().MethodTitle().c_str(),fluidscatra_->ScaTraField().Step(),numstep_);
  }

  // the following already done in PrepareTimeStep:
  // set FSI values required in scatra
  //SetFSIValuesInScaTra();

  // initially solve coupled scalar transport equation system
  // (values for intermediate time steps were calculated at the end of PrepareTimeStep)
  if (comm.MyPID()==0) cout<<"\n****************************************\n        SCALAR TRANSPORT SOLVER\n****************************************\n";
  ScatraEvaluateSolveIterUpdate();

  while (stopnonliniter==false)
  {
    itnum++;

    // in case of non-constant thermodynamic pressure: compute
    // (either based on energy conservation or based on mass conservation)
    if (consthermpress_=="No_energy")
      fluidscatra_->ScaTraField().ComputeThermPressure();
    else if (consthermpress_=="No_mass")
      fluidscatra_->ScaTraField().ComputeThermPressureFromMassCons();

    // set fluid- and structure-based scalar transport values required in FSI
    SetScaTraValuesInFSI();

    // solve FSI system
    if (comm.MyPID()==0) cout<<"\n****************************************\n               FSI SOLVER\n****************************************\n";
    fsi_->TimeStep(fsi_);

    // set FSI values required in scatra (will be done in the following
    // routine, for the time being)
    //SetFSIValuesInScaTra();
    // set mesh displacement and velocity fields
    SetFSISolution();

    // solve scalar transport equation
    if (comm.MyPID()==0) cout<<"\n****************************************\n        SCALAR TRANSPORT SOLVER\n****************************************\n";
    ScatraEvaluateSolveIterUpdate();

    // check convergence for all fields and stop iteration loop if
    // convergence is achieved overall
    stopnonliniter = ConvergenceCheck(itnum);
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*void FS3I::PartFS3I_2WC::SetFSIValuesInScaTra()
{
  // set respective field vectors for velocity/pressure, acceleration
  // and discretization based on time-integration scheme
  if (fsi_->FluidField().TimIntScheme() == INPAR::FLUID::timeint_afgenalpha)
    fluidscatra_->ScaTraField().SetVelocityField(fsi_->FluidField().Velaf(),
                                                fsi_->FluidField().Accam(),
                                                Teuchos::null,
                                                fsi_->FluidField().FsVel(),
                                                Teuchos::null,
                                                fsi_->FluidField().Discretization());
  else
    fluidscatra_->ScaTraField().SetVelocityField(fsi_->FluidField().Velnp(),
                                                fsi_->FluidField().Hist(),
                                                Teuchos::null,
                                                fsi_->FluidField().FsVel(),
                                                Teuchos::null,
                                                fsi_->FluidField().Discretization());
}*/


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_2WC::SetScaTraValuesInFSI()
{
    // set scalar and thermodynamic pressure values as well as time
    // derivatives and discretization based on time-integration scheme
    /*if (fsi_->FluidField().TimIntScheme() == INPAR::FLUID::timeint_afgenalpha)
    {
      fsi_->FluidField().SetIterLomaFields(fluidscatra_->ScaTraField().Phiaf(),
                                           fluidscatra_->ScaTraField().Phiam(),
                                           fluidscatra_->ScaTraField().Phidtam(),
                                           Teuchos::null,
                                           fluidscatra_->ScaTraField().ThermPressAf(),
                                           fluidscatra_->ScaTraField().ThermPressAm(),
                                           fluidscatra_->ScaTraField().ThermPressDtAf(),
                                           fluidscatra_->ScaTraField().ThermPressDtAm(),
                                           fluidscatra_->ScaTraField().Discretization());

      fsi_->StructureField()->ApplyTemperatures(structurescatra_->ScaTraField().Phiaf());
    }
    else
    {*/
      fsi_->FluidField().SetIterLomaFields(fluidscatra_->ScaTraField().Phinp(),
                                           fluidscatra_->ScaTraField().Phin(),
                                           fluidscatra_->ScaTraField().Phidtnp(),
                                           Teuchos::null,
                                           fluidscatra_->ScaTraField().ThermPressNp(),
                                           fluidscatra_->ScaTraField().ThermPressN(),
                                           fluidscatra_->ScaTraField().ThermPressDtNp(),
                                           fluidscatra_->ScaTraField().ThermPressDtNp(),
                                           fluidscatra_->ScaTraField().Discretization());
      fsi_->StructureField()->ApplyTemperatures(structurescatra_->ScaTraField().Phinp());
    //}
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FS3I::PartFS3I_2WC::ConvergenceCheck(int itnum)
{
#ifdef PARALLEL
  const Epetra_Comm& comm = scatravec_[0]->ScaTraField().Discretization()->Comm();
#else
  Epetra_SerialComm comm;
#endif

  // define flags for fluid and scatra convergence check
  bool fluidstopnonliniter  = false;
  bool scatrastopnonliniter = false;

  // dump on screen
  if (comm.MyPID()==0) cout<<"\n****************************************\n  CONVERGENCE CHECK FOR ITERATION STEP\n****************************************\n";

  // fsi convergence check
  if (fsi_->NoxStatus() == NOX::StatusTest::Converged) fluidstopnonliniter = true;

  // scatra convergence check
  scatrastopnonliniter = ScatraConvergenceCheck(itnum);

  if (fluidstopnonliniter == true and scatrastopnonliniter == true) return true;
  else                                                              return false;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FS3I::PartFS3I_2WC::ScatraConvergenceCheck(int itnum)
{
#ifdef PARALLEL
  const Epetra_Comm& comm = scatravec_[0]->ScaTraField().Discretization()->Comm();
#else
  Epetra_SerialComm comm;
#endif

  // define flags for convergence check for scatra fields
  bool scatra1stopnonliniter = false;
  bool scatra2stopnonliniter = false;

  // convergence check of scatra fields
  if (comm.MyPID() == 0)
  {
    cout<<"\n****************************************\n         SCALAR TRANSPORT CHECK\n****************************************\n";
    cout<<"\n****************************************\n   FLUID-BASED SCALAR TRANSPORT CHECK\n****************************************\n";
  }
  scatra1stopnonliniter = scatravec_[0]->ScaTraField().ConvergenceCheck(itnum,itmax_,ittol_);

  if (comm.MyPID() == 0) cout<<"\n****************************************\n STRUCTURE-BASED SCALAR TRANSPORT CHECK\n****************************************\n";
  scatra2stopnonliniter = scatravec_[1]->ScaTraField().ConvergenceCheck(itnum,itmax_,ittol_);

  if (scatra1stopnonliniter == true and scatra2stopnonliniter == true) return true;
  else                                                                 return false;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::PartFS3I_2WC::TimeUpdateAndOutput()
{
  // prepare output for FSI
  fsi_->PrepareOutput();

  // update fluid- and structure-based scalar transport
  UpdateScatraFields();

  // in case of non-constant thermodynamic pressure: update
  if (consthermpress_=="No_energy" or consthermpress_=="No_mass")
    fluidscatra_->ScaTraField().UpdateThermPressure();

  // update structure, fluid and ALE
  fsi_->Update();

  // set scalar and thermodynamic pressure at n+1 and SCATRA trueresidual
  // for statistical evaluation and evaluation of Neumann boundary
  // conditions at the beginning of the subsequent time step
  fsi_->FluidField().SetTimeLomaFields(fluidscatra_->ScaTraField().Phinp(),
                                       fluidscatra_->ScaTraField().ThermPressNp(),
                                       fluidscatra_->ScaTraField().TrueResidual(),
                                       fluidscatra_->ScaTraField().Discretization());

  // Note: The order is important here! Herein, control file entries are
  // written, defining the order in which the filters handle the
  // discretizations, which in turn defines the dof number ordering of the
  // discretizations.
  //fsi_->FluidField().StatisticsAndOutput();
  fsi_->Output();

  // output of fluid- and structure-based scalar transport
  ScatraOutput();

  return;
}


