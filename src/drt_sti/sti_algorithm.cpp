/*----------------------------------------------------------------------*/
/*!
\file sti_algorithm.cpp

\brief monolithic algorithm for scatra-thermo interaction

<pre>
Maintainer: Rui Fang
            fang@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/
            089-289-15251
</pre>
*/
/*----------------------------------------------------------------------*/
#include "sti_algorithm.H"

#include <Epetra_Time.h>

#include "../drt_adapter/adapter_coupling.H"
#include "../drt_adapter/adapter_scatra_base_algorithm.H"

#include "../drt_fsi/fsi_matrixtransform.H"

#include "../drt_io/io_control.H"

#include "../drt_lib/drt_assemblestrategy.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../drt_scatra/scatra_timint_implicit.H"
#include "../drt_scatra/scatra_timint_meshtying_strategy_s2i.H"

#include "../drt_scatra_ele/scatra_ele_action.H"

#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_utils.H"

/*--------------------------------------------------------------------------------*
 | constructor                                                         fang 04/15 |
 *--------------------------------------------------------------------------------*/
STI::Algorithm::Algorithm(
    const Epetra_Comm&              comm,          //! communicator
    const Teuchos::ParameterList&   stidyn,        //! parameter list for scatra-thermo interaction
    const Teuchos::ParameterList&   scatradyn,     //! scalar transport parameter list for scatra and thermo fields
    const Teuchos::ParameterList&   solverparams   //! solver parameter list
    ) :
    // instantiate base class
    AlgorithmBase(comm,scatradyn),

    scatra_(Teuchos::null),
    thermo_(Teuchos::null),
    strategyscatra_(Teuchos::null),
    strategythermo_(Teuchos::null),
    stiparameters_(Teuchos::rcp(new Teuchos::ParameterList(stidyn))),
    fieldparameters_(Teuchos::rcp(new Teuchos::ParameterList(scatradyn))),
    iter_(0),
    itermax_(fieldparameters_->sublist("NONLINEAR").get<int>("ITEMAX")),
    itertol_(fieldparameters_->sublist("NONLINEAR").get<double>("CONVTOL")),
    restol_(fieldparameters_->sublist("NONLINEAR").get<double>("ABSTOLRES")),
    maps_(Teuchos::null),
    systemmatrix_(Teuchos::null),
    scatrathermoblock_(Teuchos::null),
    thermoscatrablock_(Teuchos::null),
    increment_(Teuchos::null),

    // initialize timer for Newton-Raphson iteration
    timer_(Teuchos::rcp(new Epetra_Time(comm))),

    residual_(Teuchos::null),
    dtsolve_(0.),

    // initialize algebraic solver for global system of equations
    solver_(Teuchos::rcp(new LINALG::Solver(
        solverparams,
        comm,
        DRT::Problem::Instance()->ErrorFile()->Handle()
        ))),

    // initialize L2 norms for Newton-Raphson convergence check
    scatradofnorm_(0.),
    scatraresnorm_(0.),
    scatraincnorm_(0.),
    thermodofnorm_(0.),
    thermoresnorm_(0.),
    thermoincnorm_(0.),

    icoupscatra_(Teuchos::null),
    icoupthermo_(Teuchos::null),
    islavetomasterrowtransformscatraod_(Teuchos::null),
    islavetomastercoltransformthermood_(Teuchos::null),
    imastertoslaverowtransformthermood_(Teuchos::null)
{
  // check input parameters for scatra and thermo fields
  if(DRT::INPUT::IntegralValue<INPAR::SCATRA::VelocityField>(*fieldparameters_,"VELOCITYFIELD") != INPAR::SCATRA::velocity_zero)
    dserror("Scatra-thermo interaction with convection not yet implemented!");

  // initialize scatra time integrator
  scatra_ = Teuchos::rcp(new ADAPTER::ScaTraBaseAlgorithm(*fieldparameters_,*fieldparameters_,solverparams))->ScaTraField();

  // modify field parameters for thermo field
  ModifyFieldParametersForThermoField();

  // initialize thermo time integrator
  thermo_ = Teuchos::rcp(new ADAPTER::ScaTraBaseAlgorithm(*fieldparameters_,*fieldparameters_,solverparams,"thermo"))->ScaTraField();

  // check maps from scatra and thermo discretizations
  if(scatra_->Discretization()->DofRowMap()->NumGlobalElements() == 0)
    dserror("Scatra discretization does not have any degrees of freedom!");
  if(thermo_->Discretization()->DofRowMap()->NumGlobalElements() == 0)
    dserror("Thermo discretization does not have any degrees of freedom!");

  // additional safety checks
  if(!scatra_->IsIncremental())
    dserror("Must have incremental solution approach for scatra-thermo interaction!");
  if(thermo_->NumScal() != 1)
    dserror("Thermo field must involve exactly one transported scalar!");

  // extract meshtying strategies for scatra-scatra interface coupling from scatra and thermo time integrators
  strategyscatra_ = Teuchos::rcp_dynamic_cast<SCATRA::MeshtyingStrategyS2I>(scatra_->Strategy());
  strategythermo_ = Teuchos::rcp_dynamic_cast<SCATRA::MeshtyingStrategyS2I>(thermo_->Strategy());

  // initialize global map extractor
  maps_ = Teuchos::rcp(new LINALG::MapExtractor(
      *LINALG::MergeMap(
          *scatra_->Discretization()->DofRowMap(),
          *thermo_->Discretization()->DofRowMap(),
          false
          ),
      thermo_->DofRowMap(),
      scatra_->DofRowMap()
      ));

  // check global map extractor
  maps_->CheckForValidMapExtractor();

  // initialize global increment vector for Newton-Raphson iteration
  increment_ = LINALG::CreateVector(
      *DofRowMap(),
      true
      );

  // initialize global residual vector
  residual_ = LINALG::CreateVector(
      *DofRowMap(),
      true
      );

  // initialize global system matrix
  systemmatrix_ = Teuchos::rcp(new LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>(
          *maps_,
          *maps_,
          81,
          false,
          true
          ));

  // initialize scatra-thermo block of global system matrix
  scatrathermoblock_ = Teuchos::rcp(new LINALG::SparseMatrix(
         *scatra_->Discretization()->DofRowMap(),
         81,
         true,
         true
         ));

  // initialize thermo-scatra block of global system matrix
  thermoscatrablock_ = Teuchos::rcp(new LINALG::SparseMatrix(
         *thermo_->Discretization()->DofRowMap(),
         81,
         true,
         true
         ));

  // initialize coupling adapter
  icoupscatra_ = strategyscatra_->CouplingAdapter();
  icoupthermo_ = strategythermo_->CouplingAdapter();

  // initialize transformation operators
  islavetomasterrowtransformscatraod_ = Teuchos::rcp(new FSI::UTILS::MatrixRowTransform);
  islavetomastercoltransformthermood_ = Teuchos::rcp(new FSI::UTILS::MatrixColTransform);
  imastertoslaverowtransformthermood_ = Teuchos::rcp(new FSI::UTILS::MatrixRowTransform);

  return;
} // STI::Algorithm::Algorithm


/*----------------------------------------------------------------------*
 | assemble global system of equations                       fang 07/15 |
 *----------------------------------------------------------------------*/
void STI::Algorithm::AssembleMatAndRHS()
{
  // pass scatra degrees of freedom to thermo discretization and vice versa
  ExchangeStateVectors();

  // build system matrix and residual for scatra field
  scatra_->PrepareLinearSolve();

  // pass master-side scatra degrees of freedom to thermo discretization for evaluation of scatra-scatra interface coupling
  if(thermo_->S2ICoupling())
    thermo_->Discretization()->SetState(2,"imasterscatra",strategyscatra_->MasterPhinp());

  // build system matrix and residual for thermo field
  thermo_->PrepareLinearSolve();

  // assemble off-diagonal scatra-thermo block of global system matrix (derivatives of scatra residuals w.r.t. thermo degrees of freedom)
  AssembleODBlockScatraThermo();

  // assemble off-diagonal thermo-scatra block of global system matrix (derivatives of thermo residuals w.r.t. scatra degrees of freedom)
  AssembleODBlockThermoScatra();

  // build global system matrix
  systemmatrix_->Assign(0,0,LINALG::View,*scatra_->SystemMatrix());
  systemmatrix_->Assign(0,1,LINALG::View,*scatrathermoblock_);
  systemmatrix_->Assign(1,0,LINALG::View,*thermoscatrablock_);
  systemmatrix_->Assign(1,1,LINALG::View,*thermo_->SystemMatrix());
  systemmatrix_->Complete();

  // create full monolithic right-hand side vector
  maps_->InsertVector(scatra_->Residual(),0,residual_);
  maps_->InsertVector(thermo_->Residual(),1,residual_);

  return;
} // STI::Algorithm::AssembleMatAndRHS()


/*----------------------------------------------------------------------*
 | global map of degrees of freedom                          fang 04/15 |
 *----------------------------------------------------------------------*/
const Teuchos::RCP<const Epetra_Map>& STI::Algorithm::DofRowMap() const
{
  return maps_->FullMap();
} // STI::Algorithm::DofRowMap()


/*-------------------------------------------------------------------------------------*
 | pass scatra degrees of freedom to thermo discretization and vice versa   fang 04/15 |
 *-------------------------------------------------------------------------------------*/
void STI::Algorithm::ExchangeStateVectors()
{
  // pass scatra degrees of freedom to thermo discretization and vice versa
  scatra_->Discretization()->SetState(2,"thermo",thermo_->Phiafnp());
  thermo_->Discretization()->SetState(2,"scatra",scatra_->Phiafnp());

  return;
} // STI::Algorithm::ExchangeStateVectors()


/*-----------------------------------------------------------------------*
 | check termination criterion for Newton-Raphson iteration   fang 04/15 |
 *-----------------------------------------------------------------------*/
bool STI::Algorithm::ExitNewtonRaphson()
{
  // initialize exit flag
  bool exit(false);

  // compute vector norms for convergence check
  scatra_->Phinp()->Norm2(&scatradofnorm_);
  maps_->ExtractVector(residual_,0)->Norm2(&scatraresnorm_);
  maps_->ExtractVector(increment_,0)->Norm2(&scatraincnorm_);
  thermo_->Phinp()->Norm2(&thermodofnorm_);
  maps_->ExtractVector(residual_,1)->Norm2(&thermoresnorm_);
  maps_->ExtractVector(increment_,1)->Norm2(&thermoincnorm_);

  // safety checks
  if(std::isnan(scatradofnorm_) or
     std::isnan(scatraresnorm_) or
     std::isnan(scatraincnorm_) or
     std::isnan(thermodofnorm_) or
     std::isnan(thermoresnorm_) or
     std::isnan(thermoincnorm_))
    dserror("Vector norm is not a number!");
  if(std::isinf(scatradofnorm_) or
     std::isinf(scatraresnorm_) or
     std::isinf(scatraincnorm_) or
     std::isinf(thermodofnorm_) or
     std::isinf(thermoresnorm_) or
     std::isinf(thermoincnorm_))
    dserror("Vector norm is infinity!");

  // prevent division by zero
  if(scatradofnorm_ < 1.e-5)
    scatradofnorm_ = 1.e-5;
  if(thermodofnorm_ < 1.e-5)
    scatradofnorm_ = 1.e-5;

  // first Newton-Raphson iteration
  if(iter_ == 1)
  {
    // print first line of convergence table to screen
    // solution increment not yet available during first Newton-Raphson iteration
    if(Comm().MyPID() == 0)
      std::cout << "|  " << std::setw(3) << iter_ << "/" << std::setw(3) << itermax_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << itertol_ << "[L_2 ]  | "
                << std::setw(10) << std::setprecision(3) << std::scientific << scatraresnorm_
                << "   |      --      | "
                << std::setw(10) << std::setprecision(3) << std::scientific << thermoresnorm_
                << "   |      --      | "
                << "(       --      , te = "
                << std::setw(10) << std::setprecision(3) << scatra_->DtEle()+thermo_->DtEle() << ")" << std::endl;
  }

  // subsequent Newton-Raphson iterations
  else
  {
    // print current line of convergence table to screen
    if(Comm().MyPID() == 0)
      std::cout << "|  " << std::setw(3) << iter_ << "/" << std::setw(3) << itermax_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << itertol_ << "[L_2 ]  | "
                << std::setw(10) << std::setprecision(3) << std::scientific << scatraresnorm_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << scatraincnorm_/scatradofnorm_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << thermoresnorm_ << "   | "
                << std::setw(10) << std::setprecision(3) << std::scientific << thermoincnorm_/thermodofnorm_ << "   | (ts = "
                << std::setw(10) << std::setprecision(3) << dtsolve_ << ", te = "
                << std::setw(10) << std::setprecision(3) << scatra_->DtEle()+thermo_->DtEle() << ")" << std::endl;

    // convergence check
    if(scatraresnorm_ <= itertol_ and
       thermoresnorm_ <= itertol_ and
       scatraincnorm_/scatradofnorm_ <= itertol_ and
       thermoincnorm_/thermodofnorm_ <= itertol_)
      // exit Newton-Raphson iteration upon convergence
      exit = true;
  }

  // exit Newton-Raphson iteration when residuals are small enough to prevent unnecessary additional solver calls
  if(scatraresnorm_ < restol_ and thermoresnorm_ < restol_)
    exit = true;

  // print warning to screen if maximum number of Newton-Raphson iterations is reached without convergence
  if(iter_ == itermax_)
  {
    if(Comm().MyPID() == 0)
    {
      std::cout << "+------------+-------------------+--------------+--------------+--------------+--------------+" << std::endl;
      std::cout << "|      Newton-Raphson method has not converged after a maximum number of " << std::setw(2) << itermax_ << " iterations!      |" << std::endl;
    }

    // proceed to next time step
    exit = true;
  }

  // print finish line of convergence table to screen
  if(exit and Comm().MyPID() == 0)
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--------------+" << std::endl << std::endl;

  return exit;
} // STI::Algorithm::ExitNewtonRaphson()


/*----------------------------------------------------------------------*
 | modify field parameters for thermo field                  fang 06/15 |
 *----------------------------------------------------------------------*/
void STI::Algorithm::ModifyFieldParametersForThermoField()
{
  // extract parameters for initial temperature field from parameter list for scatra-thermo interaction
  // and overwrite corresponding parameters in parameter list for thermo field
  if(!fieldparameters_->isParameter("INITIALFIELD") or !fieldparameters_->isParameter("INITFUNCNO"))
    dserror("Initial field parameters not properly set in input file section SCALAR TRANSPORT DYNAMIC!");
  if(!stiparameters_->isParameter("THERMO_INITIALFIELD") or !stiparameters_->isParameter("THERMO_INITFUNCNO"))
    dserror("Initial field parameters not properly set in input file section SCALAR TRANSPORT DYNAMIC!");
  fieldparameters_->set<std::string>("INITIALFIELD",stiparameters_->get<std::string>("THERMO_INITIALFIELD"));
  fieldparameters_->set<int>("INITFUNCNO",stiparameters_->get<int>("THERMO_INITFUNCNO"));

  // set flag in thermo meshtying strategy for evaluation of interface linearizations and residuals on slave side only
  if(scatra_->S2ICoupling())
    fieldparameters_->sublist("S2I COUPLING").set<std::string>("SLAVEONLY","Yes");

  return;
} // STI::Algorithm::ModifyFieldParametersForThermoField()


/*----------------------------------------------------------------------*
 | output solution to screen and files                       fang 04/15 |
 *----------------------------------------------------------------------*/
void STI::Algorithm::Output()
{
  // output scatra field
  scatra_->Output();

  // output thermo field
  thermo_->Output();

  return;
}


/*----------------------------------------------------------------------*
 | prepare time step                                         fang 04/15 |
 *----------------------------------------------------------------------*/
void STI::Algorithm::PrepareTimeStep()
{
  // update time and time step
  IncrementTimeAndStep();

  // provide scatra and thermo fields with velocities
  scatra_->SetVelocityField(1);
  thermo_->SetVelocityField(1);

  // pass scatra degrees of freedom to thermo discretization and vice versa
  ExchangeStateVectors();

  // prepare time step for scatra field
  scatra_->PrepareTimeStep();

  // pass scatra degrees of freedom to thermo discretization and vice versa
  // this only needs to be done for the first time step, when the initial values of the electric potential state variables are computed
  if(Step() == 1)
  {
    ExchangeStateVectors();

    // pass master-side scatra degrees of freedom to thermo discretization for evaluation of scatra-scatra interface coupling
    if(thermo_->S2ICoupling())
      thermo_->Discretization()->SetState(2,"imasterscatra",strategyscatra_->MasterPhinp());
  }

  // prepare time step for thermo field
  thermo_->PrepareTimeStep();

  // print time step information to screen
  scatra_->PrintTimeStepInfo();

  return;
} // STI::Algorithm::PrepareTimeStep()


/*----------------------------------------------------------------------*
 | read restart data                                         fang 04/15 |
 *----------------------------------------------------------------------*/
void STI::Algorithm::ReadRestart(
    int step   //! time step for restart
    )
{
  // read scatra and thermo restart variables
  scatra_->ReadRestart(step);
  thermo_->ReadRestart(step);

  // pass scatra degrees of freedom to thermo discretization and vice versa
  ExchangeStateVectors();

  // set time and time step
  SetTimeStep(scatra_->Time(),step);

  // ToDo: check and remove
  dserror("Restart functionality for scatra-thermo interaction has not been tested yet. Feel free to do it and remove this error.");

  return;
} // STI::Algorithm::ReadRestart


/*----------------------------------------------------------------------*
 | evaluate time step using Newton-Raphson iteration         fang 04/15 |
 *----------------------------------------------------------------------*/
void STI::Algorithm::Solve()
{
  // initialize counter for Newton-Raphson iterations
  iter_ = 0;

  // print header of convergence table to screen
  if(Comm().MyPID() == 0)
  {
    std::cout << "+------------+-------------------+--------------+--------------+--------------+--------------+" << std::endl;
    std::cout << "|- step/max -|- tolerance[norm] -|- scatra-res -|- scatra-inc -|- thermo-res -|- thermo-inc -|" << std::endl;
  }

  // start Newton-Raphson iteration
  while(true)
  {
    // update iteration counter
    iter_ += 1;

    // reset timer
    timer_->ResetStartTime();

    // assemble global system of equations
    AssembleMatAndRHS();

    // safety check
    if(!systemmatrix_->Filled())
      dserror("Complete() has not been called on global system matrix yet!");

    // perform finite difference check on time integrator level
    if(scatra_->FDCheckType() == INPAR::SCATRA::fdcheck_global)
      FDCheck();

    // check termination criterion for Newton-Raphson iteration
    if(ExitNewtonRaphson())
      break;

    // initialize global increment vector
    increment_->PutScalar(0.);

    // store time before solving global system of equations
    const double time = timer_->WallTime();

    // solve global system of equations
    // Dirichlet boundary conditions have already been applied to global system of equations
    solver_->Solve(
        systemmatrix_->EpetraOperator(),
        increment_,
        residual_,
        true,
        iter_==1
        );

    // determine time needed for solving global system of equations
    dtsolve_ = timer_->WallTime()-time;

    // update scatra field
    scatra_->UpdateIter(maps_->ExtractVector(increment_,0));
    scatra_->ComputeIntermediateValues();

    // update thermo field
    thermo_->UpdateIter(maps_->ExtractVector(increment_,1));
    thermo_->ComputeIntermediateValues();
  } // Newton-Raphson iteration

  return;
} // STI::Algorithm::Solve


/*--------------------------------------------------------------------------------*
 | assemble off-diagonal scatra-thermo block of global system matrix   fang 12/15 |
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::AssembleODBlockScatraThermo()
{
  // initialize scatra-thermo matrix block
  scatrathermoblock_->Zero();

  // create parameter list for element evaluation
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<int>("action",SCATRA::calc_scatra_mono_odblock_scatrathermo);

  // number of dofset associated with velocity-related dofs on scatra discretization
  eleparams.set<int>("ndsvel",1);

  // remove state vectors from scatra discretization
  scatra_->Discretization()->ClearState();

  // add state vectors to scatra discretization
  scatra_->AddTimeIntegrationSpecificVectors();

  // create strategy for assembly of scatra-thermo matrix block
  DRT::AssembleStrategy strategyscatrathermo(
      0,                    // row assembly based on number of dofset associated with scatra dofs on scatra discretization
      2,                    // column assembly based on number of dofset associated with thermo dofs on scatra discretization
      scatrathermoblock_,   // scatra-thermo matrix block
      Teuchos::null,        // no additional matrices or vectors
      Teuchos::null,
      Teuchos::null,
      Teuchos::null
      );

  // assemble scatra-thermo matrix block
  scatra_->Discretization()->Evaluate(eleparams,strategyscatrathermo);

  // provide scatra-thermo matrix block with contributions from scatra-scatra interface coupling if applicable
  if(scatra_->S2ICoupling())
  {
    // initialize auxiliary system matrix for linearizations of slave-side scatra fluxes w.r.t. slave-side thermo dofs
    strategyscatra_->SlaveMatrix()->Zero();

    // create parameter list for element evaluation
    Teuchos::ParameterList condparams;

    // action for elements
    condparams.set<int>("action",SCATRA::bd_calc_s2icoupling_od);

    // add state vector containing master-side scatra degrees of freedom to scatra discretization
    scatra_->Discretization()->SetState("imasterphinp",strategyscatra_->MasterPhinp());

    // create strategy for assembly of auxiliary system matrix
    DRT::AssembleStrategy strategyscatrathermos2i(
        0,                                // row assembly based on number of dofset associated with scatra dofs on scatra discretization
        2,                                // column assembly based on number of dofset associated with thermo dofs on scatra discretization
        strategyscatra_->SlaveMatrix(),   // auxiliary system matrix
        Teuchos::null,                    // no additional matrices of vectors
        Teuchos::null,
        Teuchos::null,
        Teuchos::null
        );

    // evaluate scatra-scatra interface coupling
    std::vector<DRT::Condition*> conditions;
    scatra_->Discretization()->GetCondition("S2ICoupling",conditions);
    for(unsigned icondition=0; icondition<conditions.size(); ++icondition)
      if(conditions[icondition]->GetInt("interface side") == INPAR::S2I::side_slave)
        scatra_->Discretization()->EvaluateCondition(condparams,strategyscatrathermos2i,"S2ICoupling",conditions[icondition]->GetInt("ConditionID"));

    // finalize auxiliary system matrix
    strategyscatra_->SlaveMatrix()->Complete(*maps_->Map(1),*maps_->Map(0));

    // assemble linearizations of slave-side scatra fluxes w.r.t. slave-side thermo dofs into scatra-thermo matrix block
    scatrathermoblock_->Add(*strategyscatra_->SlaveMatrix(),false,1.,1.);

    // derive linearizations of master-side scatra fluxes w.r.t. slave-side thermo dofs and assemble into scatra-thermo matrix block
    (*islavetomasterrowtransformscatraod_)(
        *strategyscatra_->SlaveMatrix(),
        -1.,
        ADAPTER::CouplingSlaveConverter(*icoupscatra_),
        *scatrathermoblock_,
        true
        );

    // linearizations of scatra fluxes w.r.t. master-side thermo dofs are not needed, since these dofs will be condensed out later
  }

  // finalize scatra-thermo matrix block
  scatrathermoblock_->Complete(*maps_->Map(1),*maps_->Map(0));

  // apply Dirichlet boundary conditions to scatra-thermo matrix block
  scatrathermoblock_->ApplyDirichlet(*scatra_->DirichMaps()->CondMap(),false);

  // remove state vectors from scatra discretization
  scatra_->Discretization()->ClearState();

  return;
} // STI::Algorithm::AssembleODBlockScatraThermo()


/*--------------------------------------------------------------------------------*
 | assemble off-diagonal thermo-scatra block of global system matrix   fang 12/15 |
 *--------------------------------------------------------------------------------*/
void STI::Algorithm::AssembleODBlockThermoScatra()
{
  // initialize thermo-scatra matrix block
  thermoscatrablock_->Zero();

  // create parameter list for element evaluation
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<int>("action",SCATRA::calc_scatra_mono_odblock_thermoscatra);

  // number of dofset associated with velocity-related dofs on thermo discretization
  eleparams.set<int>("ndsvel",1);

  // remove state vectors from thermo discretization
  thermo_->Discretization()->ClearState();

  // add state vectors to thermo discretization
  thermo_->AddTimeIntegrationSpecificVectors();

  // create strategy for assembly of thermo-scatra matrix block
  DRT::AssembleStrategy strategythermoscatra(
      0,                    // row assembly based on number of dofset associated with thermo dofs on thermo discretization
      2,                    // column assembly based on number of dofset associated with scatra dofs on thermo discretization
      thermoscatrablock_,   // thermo-scatra matrix block
      Teuchos::null,        // no additional matrices or vectors
      Teuchos::null,
      Teuchos::null,
      Teuchos::null
      );

  // assemble thermo-scatra matrix block
  thermo_->Discretization()->Evaluate(eleparams,strategythermoscatra);

  // provide thermo-scatra matrix block with contributions from scatra-scatra interface coupling if applicable
  if(thermo_->S2ICoupling())
  {
    // initialize auxiliary system matrix for linearizations of slave-side thermo fluxes w.r.t. slave-side scatra dofs
    strategythermo_->SlaveMatrix()->Zero();

    // initialize auxiliary system matrix for linearizations of slave-side thermo fluxes w.r.t. master-side scatra dofs
    strategythermo_->MasterMatrix()->Zero();

    // create parameter list for element evaluation
    Teuchos::ParameterList condparams;

    // action for elements
    condparams.set<int>("action",SCATRA::bd_calc_s2icoupling_od);

    // create strategy for assembly of auxiliary system matrices
    DRT::AssembleStrategy strategythermoscatras2i(
        0,                                // row assembly based on number of dofset associated with thermo dofs on thermo discretization
        2,                                // column assembly based on number of dofset associated with scatra dofs on thermo discretization
        strategythermo_->SlaveMatrix(),   // auxiliary system matrices
        strategythermo_->MasterMatrix(),
        Teuchos::null,                    // no additional matrices of vectors
        Teuchos::null,
        Teuchos::null
        );

    // evaluate scatra-scatra interface coupling
    std::vector<DRT::Condition*> conditions;
    thermo_->Discretization()->GetCondition("S2ICoupling",conditions);
    for(unsigned icondition=0; icondition<conditions.size(); ++icondition)
      if(conditions[icondition]->GetInt("interface side") == INPAR::S2I::side_slave)
        thermo_->Discretization()->EvaluateCondition(condparams,strategythermoscatras2i,"S2ICoupling",conditions[icondition]->GetInt("ConditionID"));

    // finalize auxiliary system matrices
    strategythermo_->SlaveMatrix()->Complete(*icoupscatra_->SlaveDofMap(),*icoupthermo_->SlaveDofMap());
    strategythermo_->MasterMatrix()->Complete(*icoupscatra_->SlaveDofMap(),*icoupthermo_->SlaveDofMap());

    // assemble linearizations of slave-side thermo fluxes w.r.t. slave-side scatra dofs into thermo-scatra matrix block
    thermoscatrablock_->Add(*strategythermo_->SlaveMatrix(),false,1.,1.);

    // derive linearizations of slave-side thermo fluxes w.r.t. master-side scatra dofs and assemble into thermo-scatra matrix block
    (*islavetomastercoltransformthermood_)(
        strategythermo_->MasterMatrix()->RowMap(),
        strategythermo_->MasterMatrix()->ColMap(),
        *strategythermo_->MasterMatrix(),
        1.,
        ADAPTER::CouplingSlaveConverter(*icoupscatra_),
        *thermoscatrablock_,
        true,
        true
        );

    // linearizations of master-side thermo fluxes w.r.t. scatra dofs are not needed, since thermo fluxes are source terms and thus only evaluated once on slave side

    // initialize temporary matrix for master-side rows of thermo-scatra matrix block
    LINALG::SparseMatrix thermoscatrarowsmaster(*icoupthermo_->MasterDofMap(),81);

    // loop over all master-side rows of thermo-scatra matrix block
    for(int masterdoflid=0; masterdoflid<icoupthermo_->MasterDofMap()->NumMyElements(); ++masterdoflid)
    {
      // determine global ID of current matrix row
      const int masterdofgid = icoupthermo_->MasterDofMap()->GID(masterdoflid);
      if(masterdofgid < 0)
        dserror("Couldn't find local ID %d in map!",masterdoflid);

      // extract current matrix row from thermo-scatra matrix block
      const int length = thermoscatrablock_->EpetraMatrix()->NumGlobalEntries(masterdofgid);
      int numentries(0);
      std::vector<double> values(length,0.);
      std::vector<int> indices(length,0);
      if(thermoscatrablock_->EpetraMatrix()->ExtractGlobalRowCopy(masterdofgid,length,numentries,&values[0],&indices[0]) != 0)
        dserror("Cannot extract matrix row with global ID %d from thermo-scatra matrix block!",masterdofgid);

      // copy current matrix row of thermo-scatra matrix block into temporary matrix
      if(thermoscatrarowsmaster.EpetraMatrix()->InsertGlobalValues(masterdofgid,numentries,&values[0],&indices[0]) < 0)
        dserror("Cannot insert matrix row with global ID %d into temporary matrix!",masterdofgid);
    }

    // finalize temporary matrix with master-side rows of thermo-scatra matrix block
    thermoscatrarowsmaster.Complete(*maps_->Map(0),*icoupthermo_->MasterDofMap());

    // add master-side rows of thermo-scatra matrix block to corresponding slave-side rows
    (*imastertoslaverowtransformthermood_)(
        thermoscatrarowsmaster,
        1.,
        ADAPTER::CouplingMasterConverter(*icoupthermo_),
        *thermoscatrablock_,
        true
        );
  }

  // finalize thermo-scatra matrix block
  thermoscatrablock_->Complete(*maps_->Map(0),*maps_->Map(1));

  // apply Dirichlet boundary conditions to scatra-thermo matrix block
  thermoscatrablock_->ApplyDirichlet(*thermo_->DirichMaps()->CondMap(),false);

  if(thermo_->S2ICoupling())
    // zero out master-side rows of thermo-scatra matrix block after having added them to the
    // corresponding slave-side rows to finalize condensation of master-side thermo dofs
    thermoscatrablock_->ApplyDirichlet(*icoupthermo_->MasterDofMap(),false);

  // remove state vectors from thermo discretization
  thermo_->Discretization()->ClearState();

  return;
} // STI::Algorithm::AssembleODBlockThermoScatra()


/*----------------------------------------------------------------------*
 | time loop                                                 fang 04/15 |
 *----------------------------------------------------------------------*/
void STI::Algorithm::TimeLoop()
{
  // output initial solution to screen and files
  if(Step() == 0)
    Output();

  // time loop
  while(NotFinished())
  {
    // prepare time step
    PrepareTimeStep();

    // evaluate time step
    Solve();

    // update scatra and thermo fields
    Update();

    // output solution to screen and files
    Output();
  } // while(NotFinished())

  return;
} // STI::Algorithm::TimeLoop()


/*-------------------------------------------------------------------------*
 | update scatra and thermo fields after time step evaluation   fang 04/15 |
 *-------------------------------------------------------------------------*/
void STI::Algorithm::Update()
{
  // update scatra field
  scatra_->Update();

  // compare scatra field to analytical solution if applicable
  scatra_->EvaluateErrorComparedToAnalyticalSol();

  // update thermo field
  thermo_->Update();

  // compare thermo field to analytical solution if applicable
  thermo_->EvaluateErrorComparedToAnalyticalSol();

  return;
} // STI::Algorithm::Update()


/*---------------------------------------------------------------------------------------------*
 | finite difference check for global system matrix (for debugging only)            fang 07/15 |
 *---------------------------------------------------------------------------------------------*/
void STI::Algorithm::FDCheck()
{
  // initial screen output
  if(Comm().MyPID() == 0)
    std::cout << std::endl << "FINITE DIFFERENCE CHECK FOR STI SYSTEM MATRIX" << std::endl;

  // create global state vector
  Teuchos::RCP<Epetra_Vector> statenp(LINALG::CreateVector(*DofRowMap(),true));
  maps_->InsertVector(scatra_->Phinp(),0,statenp);
  maps_->InsertVector(thermo_->Phinp(),1,statenp);

  // make a copy of global state vector to undo perturbations later
  Teuchos::RCP<Epetra_Vector> statenp_original = Teuchos::rcp(new Epetra_Vector(*statenp));

  // make a copy of system matrix as Epetra_CrsMatrix
  Teuchos::RCP<Epetra_CrsMatrix> sysmat_original = Teuchos::null;
  if(Teuchos::rcp_dynamic_cast<LINALG::BlockSparseMatrixBase>(systemmatrix_) != Teuchos::null)
    sysmat_original = (new LINALG::SparseMatrix(*(Teuchos::rcp_static_cast<LINALG::BlockSparseMatrixBase>(systemmatrix_)->Merge())))->EpetraMatrix();
  else
    dserror("Global system matrix must be a block sparse matrix!");
  sysmat_original->FillComplete();

  // make a copy of system right-hand side vector
  Teuchos::RCP<Epetra_Vector> rhs_original = Teuchos::rcp(new Epetra_Vector(*residual_));

  // initialize counter for system matrix entries with failing finite difference check
  int counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  for (int colgid=0; colgid<=sysmat_original->ColMap().MaxAllGID(); ++colgid)
  {
    // check whether current column index is a valid global column index and continue loop if not
    int collid(sysmat_original->ColMap().LID(colgid));
    int maxcollid(-1);
    Comm().MaxAll(&collid,&maxcollid,1);
    if(maxcollid < 0)
      continue;

    // fill global state vector with original state variables
    statenp->Update(1.,*statenp_original,0.);

    // impose perturbation
    if(statenp->Map().MyGID(colgid))
      if(statenp->SumIntoGlobalValue(colgid,0,scatra_->FDCheckEps()))
        dserror("Perturbation could not be imposed on state vector for finite difference check!");
    scatra_->Phinp()->Update(1.,*maps_->ExtractVector(statenp,0),0.);
    thermo_->Phinp()->Update(1.,*maps_->ExtractVector(statenp,1),0.);

    // carry perturbation over to state vectors at intermediate time stages if necessary
    scatra_->ComputeIntermediateValues();
    thermo_->ComputeIntermediateValues();

    // calculate element right-hand side vector for perturbed state
    AssembleMatAndRHS();

    // Now we compare the difference between the current entries in the system matrix
    // and their finite difference approximations according to
    // entries ?= (residual_perturbed - residual_original) / epsilon

    // Note that the residual_ vector actually denotes the right-hand side of the linear
    // system of equations, i.e., the negative system residual.
    // To account for errors due to numerical cancellation, we additionally consider
    // entries + residual_original / epsilon ?= residual_perturbed / epsilon

    // Note that we still need to evaluate the first comparison as well. For small entries in the system
    // matrix, the second comparison might yield good agreement in spite of the entries being wrong!
    for(int rowlid=0; rowlid<DofRowMap()->NumMyElements(); ++rowlid)
    {
      // get global index of current matrix row
      const int rowgid = sysmat_original->RowMap().GID(rowlid);
      if(rowgid < 0)
        dserror("Invalid global ID of matrix row!");

      // get relevant entry in current row of original system matrix
      double entry(0.);
      int length = sysmat_original->NumMyEntries(rowlid);
      int numentries;
      std::vector<double> values(length);
      std::vector<int> indices(length);
      sysmat_original->ExtractMyRowCopy(rowlid,length,numentries,&values[0],&indices[0]);
      for(int ientry=0; ientry<length; ++ientry)
      {
        if(sysmat_original->ColMap().GID(indices[ientry]) == colgid)
        {
          entry = values[ientry];
          break;
        }
      }

      // finite difference suggestion (first divide by epsilon and then add for better conditioning)
      const double fdval = -(*residual_)[rowlid] / scatra_->FDCheckEps() + (*rhs_original)[rowlid] / scatra_->FDCheckEps();

      // confirm accuracy of first comparison
      if(abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
        dserror("Finite difference check involves values too close to numerical zero!");

      // absolute and relative errors in first comparison
      const double abserr1 = entry - fdval;
      if(abs(abserr1) > maxabserr)
        maxabserr = abs(abserr1);
      double relerr1(0.);
      if(abs(entry) > 1.e-17)
        relerr1 = abserr1 / abs(entry);
      else if(abs(fdval) > 1.e-17)
        relerr1 = abserr1 / abs(fdval);
      if(abs(relerr1) > maxrelerr)
        maxrelerr = abs(relerr1);

      // evaluate first comparison
      if(abs(relerr1) > scatra_->FDCheckTol())
      {
        std::cout << "sysmat[" << rowgid << "," << colgid << "]:  " << entry << "   ";
        std::cout << "finite difference suggestion:  " << fdval << "   ";
        std::cout << "absolute error:  " << abserr1 << "   ";
        std::cout << "relative error:  " << relerr1 << std::endl;

        counter++;
      }

      // first comparison OK
      else
      {
        // left-hand side in second comparison
        const double left  = entry - (*rhs_original)[rowlid] / scatra_->FDCheckEps();

        // right-hand side in second comparison
        const double right = -(*residual_)[rowlid] / scatra_->FDCheckEps();

        // confirm accuracy of second comparison
        if(abs(right) > 1.e-17 and abs(right) < 1.e-15)
          dserror("Finite difference check involves values too close to numerical zero!");

        // absolute and relative errors in second comparison
        const double abserr2 = left - right;
        if(abs(abserr2) > maxabserr)
          maxabserr = abs(abserr2);
        double relerr2(0.);
        if(abs(left) > 1.e-17)
          relerr2 = abserr2 / abs(left);
        else if(abs(right) > 1.e-17)
          relerr2 = abserr2 / abs(right);
        if(abs(relerr2) > maxrelerr)
          maxrelerr = abs(relerr2);

        // evaluate second comparison
        if(abs(relerr2) > scatra_->FDCheckTol())
        {
          std::cout << "sysmat[" << rowgid << "," << colgid << "]-rhs[" << rowgid << "]/eps:  " << left << "   ";
          std::cout << "-rhs_perturbed[" << rowgid << "]/eps:  " << right << "   ";
          std::cout << "absolute error:  " << abserr2 << "   ";
          std::cout << "relative error:  " << relerr2 << std::endl;

          counter++;
        }
      }
    }
  }

  // communicate tracking variables
  int counterglobal(0);
  Comm().SumAll(&counter,&counterglobal,1);
  double maxabserrglobal(0.);
  Comm().MaxAll(&maxabserr,&maxabserrglobal,1);
  double maxrelerrglobal(0.);
  Comm().MaxAll(&maxrelerr,&maxrelerrglobal,1);

  // final screen output
  if(Comm().MyPID() == 0)
  {
    if(counterglobal)
    {
      printf("--> FAILED AS LISTED ABOVE WITH %d CRITICAL MATRIX ENTRIES IN TOTAL\n\n",counterglobal);
      dserror("Finite difference check failed for STI system matrix!");
    }
    else
      printf("--> PASSED WITH MAXIMUM ABSOLUTE ERROR %+12.5e AND MAXIMUM RELATIVE ERROR %+12.5e\n\n",maxabserrglobal,maxrelerrglobal);
  }

  // undo perturbations of state variables
  scatra_->Phinp()->Update(1.,*maps_->ExtractVector(statenp_original,0),0.);
  scatra_->ComputeIntermediateValues();
  thermo_->Phinp()->Update(1.,*maps_->ExtractVector(statenp_original,1),0.);
  thermo_->ComputeIntermediateValues();

  // recompute system matrix and right-hand side vector based on original state variables
  AssembleMatAndRHS();

  return;
}
