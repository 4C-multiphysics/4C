/*----------------------------------------------------------------------*/
/*! \file
 \brief routines for coupling between 1D arterial network and 2D/3D
        scatra-algorithm

   \level 3

 *----------------------------------------------------------------------*/

#include "4C_scatra_timint_meshtying_strategy_artery.hpp"

#include "4C_adapter_art_net.hpp"
#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_poromultiphase_scatra_artery_coupling_nodebased.hpp"
#include "4C_poromultiphase_scatra_utils.hpp"
#include "4C_scatra_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                         kremheller 04/18 |
 *----------------------------------------------------------------------*/
SCATRA::MeshtyingStrategyArtery::MeshtyingStrategyArtery(
    SCATRA::ScaTraTimIntImpl* scatratimint  //!< scalar transport time integrator
    )
    : MeshtyingStrategyBase(scatratimint)
{
}

/*----------------------------------------------------------------------*
 | init                                                kremheller 04/18 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::InitMeshtying()
{
  // instantiate strategy for Newton-Raphson convergence check
  init_conv_check_strategy();

  const Teuchos::ParameterList& globaltimeparams =
      GLOBAL::Problem::Instance()->poro_multi_phase_scatra_dynamic_params();
  const Teuchos::ParameterList& myscatraparams =
      GLOBAL::Problem::Instance()->scalar_transport_dynamic_params();
  if (CORE::UTILS::IntegralValue<INPAR::SCATRA::VelocityField>(myscatraparams, "VELOCITYFIELD") !=
      INPAR::SCATRA::velocity_zero)
    FOUR_C_THROW("set your velocity field to zero!");

  // construct artery scatra problem
  Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> art_scatra =
      Teuchos::rcp(new ADAPTER::ScaTraBaseAlgorithm(globaltimeparams, myscatraparams,
          GLOBAL::Problem::Instance()->SolverParams(myscatraparams.get<int>("LINEAR_SOLVER")),
          "artery_scatra", false));

  // initialize the base algo.
  // scatra time integrator is initialized inside.
  art_scatra->Init();

  // only now we must call Setup() on the scatra time integrator.
  // all objects relying on the parallel distribution are
  // created and pointers are set.
  // calls Setup() on the scatra time integrator inside.
  art_scatra->ScaTraField()->Setup();
  GLOBAL::Problem::Instance()->AddFieldTest(art_scatra->create_sca_tra_field_test());

  // set the time integrator
  set_artery_scatra_time_integrator(art_scatra->ScaTraField());

  // get the two discretizations
  artscatradis_ = artscatratimint_->discretization();
  scatradis_ = scatratimint_->discretization();

  if (scatratimint_->discretization()->Comm().MyPID() == 0)
  {
    std::cout << "\n";
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "<                                                  >" << std::endl;
    std::cout << "< ScaTra-Coupling with 1D Artery Network activated >" << std::endl;
  }

  const bool evaluate_on_lateral_surface = CORE::UTILS::IntegralValue<int>(
      GLOBAL::Problem::Instance()->poro_fluid_multi_phase_dynamic_params().sublist(
          "ARTERY COUPLING"),
      "LATERAL_SURFACE_COUPLING");

  // set coupling condition name
  const std::string couplingcondname = std::invoke(
      [&]()
      {
        if (CORE::UTILS::IntegralValue<INPAR::ARTNET::ArteryPoroMultiphaseScatraCouplingMethod>(
                GLOBAL::Problem::Instance()->poro_fluid_multi_phase_dynamic_params().sublist(
                    "ARTERY COUPLING"),
                "ARTERY_COUPLING_METHOD") ==
            INPAR::ARTNET::ArteryPoroMultiphaseScatraCouplingMethod::ntp)
        {
          return "ArtScatraCouplConNodeToPoint";
        }
        else
        {
          return "ArtScatraCouplConNodebased";
        }
      });

  // init the mesh tying object, which does all the work
  arttoscatracoupling_ = POROMULTIPHASESCATRA::UTILS::CreateAndInitArteryCouplingStrategy(
      artscatradis_, scatradis_, myscatraparams.sublist("ARTERY COUPLING"), couplingcondname,
      "COUPLEDDOFS_ARTSCATRA", "COUPLEDDOFS_SCATRA", evaluate_on_lateral_surface);

  initialize_linear_solver(myscatraparams);
}

/*----------------------------------------------------------------------*
 | setup                                               kremheller 04/18 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::setup_meshtying()
{
  // Initialize rhs vector
  rhs_ = Teuchos::rcp(new Epetra_Vector(*arttoscatracoupling_->FullMap(), true));

  // Initialize increment vector
  comb_increment_ = Teuchos::rcp(new Epetra_Vector(*arttoscatracoupling_->FullMap(), true));

  // initialize scatra-artery_scatra-systemmatrix_
  comb_systemmatrix_ =
      Teuchos::rcp(new CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>(
          *arttoscatracoupling_->GlobalExtractor(), *arttoscatracoupling_->GlobalExtractor(), 81,
          false, true));

  arttoscatracoupling_->Setup();

  return;
}

/*----------------------------------------------------------------------*
 | initialize the linear solver                        kremheller 07/20 |
 *----------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::initialize_linear_solver(
    const Teuchos::ParameterList& scatraparams)
{
  const int linsolvernumber = scatraparams.get<int>("LINEAR_SOLVER");
  const Teuchos::ParameterList& solverparams =
      GLOBAL::Problem::Instance()->SolverParams(linsolvernumber);
  const auto solvertype =
      Teuchos::getIntegralValue<CORE::LINEAR_SOLVER::SolverType>(solverparams, "SOLVER");
  // no need to do the rest for direct solvers
  if (solvertype == CORE::LINEAR_SOLVER::SolverType::umfpack or
      solvertype == CORE::LINEAR_SOLVER::SolverType::superlu)
    return;

  if (solvertype != CORE::LINEAR_SOLVER::SolverType::belos)
    FOUR_C_THROW("Iterative solver expected");

  const auto azprectype =
      Teuchos::getIntegralValue<CORE::LINEAR_SOLVER::PreconditionerType>(solverparams, "AZPREC");

  // plausibility check
  switch (azprectype)
  {
    case CORE::LINEAR_SOLVER::PreconditionerType::multigrid_nxn:
    {
      // no plausibility checks here
      // if you forget to declare an xml file you will get an error message anyway
    }
    break;
    default:
      FOUR_C_THROW("AMGnxn preconditioner expected");
      break;
  }

  // equip smoother for fluid matrix block with empty parameter sublists to trigger null space
  // computation
  Teuchos::ParameterList& blocksmootherparams1 = Solver().Params().sublist("Inverse1");
  blocksmootherparams1.sublist("Belos Parameters");
  blocksmootherparams1.sublist("MueLu Parameters");

  scatradis_->compute_null_space_if_necessary(blocksmootherparams1);

  Teuchos::ParameterList& blocksmootherparams2 = Solver().Params().sublist("Inverse2");
  blocksmootherparams2.sublist("Belos Parameters");
  blocksmootherparams2.sublist("MueLu Parameters");

  artscatradis_->compute_null_space_if_necessary(blocksmootherparams2);
}

/*-----------------------------------------------------------------------*
 | return global map of degrees of freedom              kremheller 04/18 |
 *-----------------------------------------------------------------------*/
const Epetra_Map& SCATRA::MeshtyingStrategyArtery::dof_row_map() const
{
  return *arttoscatracoupling_->FullMap();
}

/*-----------------------------------------------------------------------*
 | return global map of degrees of freedom              kremheller 04/18 |
 *-----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Map> SCATRA::MeshtyingStrategyArtery::ArtScatraDofRowMap() const
{
  return arttoscatracoupling_->ArteryDofRowMap();
}

/*-------------------------------------------------------------------------------*
 | return linear solver for global system of linear equations   kremheller 04/18 |
 *-------------------------------------------------------------------------------*/
const CORE::LINALG::Solver& SCATRA::MeshtyingStrategyArtery::Solver() const
{
  if (scatratimint_->Solver() == Teuchos::null) FOUR_C_THROW("Invalid linear solver!");

  return *scatratimint_->Solver();
}

/*------------------------------------------------------------------------------*
 | instantiate strategy for Newton-Raphson convergence check   kremheller 04/18 |
 *------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::init_conv_check_strategy()
{
  convcheckstrategy_ = Teuchos::rcp(new SCATRA::ConvCheckStrategyPoroMultiphaseScatraArtMeshTying(
      scatratimint_->ScatraParameterList()->sublist("NONLINEAR")));

  return;
}

/*------------------------------------------------------------------------------------------*
 | solve linear system of equations for scatra-scatra interface coupling   kremheller 04/18 |
 *------------------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::Solve(
    const Teuchos::RCP<CORE::LINALG::Solver>& solver,                //!< solver
    const Teuchos::RCP<CORE::LINALG::SparseOperator>& systemmatrix,  //!< system matrix
    const Teuchos::RCP<Epetra_Vector>& increment,                    //!< increment vector
    const Teuchos::RCP<Epetra_Vector>& residual,                     //!< residual vector
    const Teuchos::RCP<Epetra_Vector>& phinp,                        //!< state vector at time n+1
    const int iteration,  //!< number of current Newton-Raphson iteration
    CORE::LINALG::SolverParams& solver_params) const
{
  // setup the system (evaluate mesh tying)
  // reason for this being done here is that we need the system matrix of the continuous scatra
  // problem with DBCs applied which is performed directly before calling solve

  SetupSystem(systemmatrix, residual);

  comb_systemmatrix_->Complete();

  // solve
  comb_increment_->PutScalar(0.0);
  solver_params.refactor = true;
  solver_params.reset = iteration == 1;
  solver->Solve(comb_systemmatrix_->EpetraOperator(), comb_increment_, rhs_, solver_params);

  // extract increments of scatra and artery-scatra field
  Teuchos::RCP<const Epetra_Vector> artscatrainc;
  Teuchos::RCP<const Epetra_Vector> myinc;
  extract_single_field_vectors(comb_increment_, myinc, artscatrainc);

  // update the scatra increment, update iter is performed outside
  increment->Update(1.0, *(myinc), 1.0);
  // update the artery-scatra field
  artscatratimint_->UpdateIter(artscatrainc);

  return;
}

/*------------------------------------------------------------------------------------------*
 | solve linear system of equations for scatra-scatra interface coupling   kremheller 04/18 |
 *------------------------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::SetupSystem(
    const Teuchos::RCP<CORE::LINALG::SparseOperator>& systemmatrix,  //!< system matrix
    const Teuchos::RCP<Epetra_Vector>& residual                      //!< residual vector
) const
{
  arttoscatracoupling_->SetSolutionVectors(
      scatratimint_->Phinp(), Teuchos::null, artscatratimint_->Phinp());

  // evaluate the 1D-3D coupling
  arttoscatracoupling_->Evaluate(comb_systemmatrix_, rhs_);

  // evaluate 1D sub-problem
  artscatratimint_->PrepareLinearSolve();

  // setup the entire system
  arttoscatracoupling_->SetupSystem(comb_systemmatrix_, rhs_,
      Teuchos::rcp_dynamic_cast<CORE::LINALG::SparseMatrix>(systemmatrix),
      artscatratimint_->SystemMatrix(), residual, artscatratimint_->Residual(),
      scatratimint_->DirichMaps(), artscatratimint_->DirichMaps());
}

/*-------------------------------------------------------------------------*
 | set time integrator for scalar transport in arteries   kremheller 04/18 |
 *------------------------------------------------------------------------ */
void SCATRA::MeshtyingStrategyArtery::UpdateArtScatraIter(
    Teuchos::RCP<const Epetra_Vector> combined_inc)
{
  Teuchos::RCP<const Epetra_Vector> artscatrainc;
  Teuchos::RCP<const Epetra_Vector> myinc;
  extract_single_field_vectors(combined_inc, myinc, artscatrainc);

  artscatratimint_->UpdateIter(artscatrainc);

  return;
}

/*-------------------------------------------------------------------------*
 | extract single field vectors                           kremheller 10/20 |
 *------------------------------------------------------------------------ */
void SCATRA::MeshtyingStrategyArtery::extract_single_field_vectors(
    Teuchos::RCP<const Epetra_Vector> globalvec, Teuchos::RCP<const Epetra_Vector>& vec_cont,
    Teuchos::RCP<const Epetra_Vector>& vec_art) const
{
  arttoscatracoupling_->extract_single_field_vectors(globalvec, vec_cont, vec_art);

  return;
}

/*-------------------------------------------------------------------------*
 | set time integrator for scalar transport in arteries   kremheller 04/18 |
 *------------------------------------------------------------------------ */
void SCATRA::MeshtyingStrategyArtery::set_artery_scatra_time_integrator(
    Teuchos::RCP<SCATRA::ScaTraTimIntImpl> artscatratimint)
{
  artscatratimint_ = artscatratimint;
  if (artscatratimint_ == Teuchos::null)
    FOUR_C_THROW("could not set artery scatra time integrator");

  return;
}

/*-------------------------------------------------------------------------*
 | set time integrator for artery problems                kremheller 04/18 |
 *------------------------------------------------------------------------ */
void SCATRA::MeshtyingStrategyArtery::set_artery_time_integrator(
    Teuchos::RCP<ADAPTER::ArtNet> arttimint)
{
  arttimint_ = arttimint;
  if (arttimint_ == Teuchos::null) FOUR_C_THROW("could not set artery time integrator");

  return;
}

/*-------------------------------------------------------------------------*
 | set element pairs that are close                       kremheller 03/19 |
 *------------------------------------------------------------------------ */
void SCATRA::MeshtyingStrategyArtery::SetNearbyElePairs(
    const std::map<int, std::set<int>>* nearbyelepairs)
{
  arttoscatracoupling_->SetNearbyElePairs(nearbyelepairs);

  return;
}

/*--------------------------------------------------------------------------*
 | setup the coupled matrix                                kremheller 04/18 |
 *--------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::prepare_time_step() const
{
  artscatratimint_->prepare_time_step();
  return;
}

/*--------------------------------------------------------------------------*
 | setup the coupled matrix                                kremheller 04/18 |
 *--------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::SetArteryPressure() const
{
  artscatradis_->set_state(2, "one_d_artery_pressure", arttimint_->Pressurenp());
  return;
}

/*--------------------------------------------------------------------------*
 | apply mesh movement on artery coupling                  kremheller 07/18 |
 *--------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::ApplyMeshMovement()
{
  arttoscatracoupling_->ApplyMeshMovement();
  return;
}

/*--------------------------------------------------------------------------*
 | check if initial fields match                           kremheller 04/18 |
 *--------------------------------------------------------------------------*/
void SCATRA::MeshtyingStrategyArtery::CheckInitialFields() const
{
  arttoscatracoupling_->CheckInitialFields(scatratimint_->Phinp(), artscatratimint_->Phinp());

  return;
}

FOUR_C_NAMESPACE_CLOSE