/*----------------------------------------------------------------------*/
/*! \file
\brief H-file associated with algorithmic routines for partitioned
       solution approaches to fluid-porous-structure-scalar-scalar interaction
       (FPS3I) specifically related to one-way-coupled problem
       configurations

\level 3


*----------------------------------------------------------------------*/
#include "baci_fs3i_fps3i_partitioned_1wc.hpp"

#include "baci_adapter_fld_poro.hpp"
#include "baci_adapter_str_fpsiwrapper.hpp"
#include "baci_coupling_adapter.hpp"
#include "baci_fluid_implicit_integration.hpp"
#include "baci_fluid_result_test.hpp"
#include "baci_fluid_utils.hpp"
#include "baci_fpsi_monolithic.hpp"
#include "baci_fpsi_monolithic_plain.hpp"
#include "baci_fpsi_utils.hpp"
#include "baci_fsi_monolithic.hpp"
#include "baci_global_data.hpp"
#include "baci_inpar_scatra.hpp"
#include "baci_inpar_validparameters.hpp"
#include "baci_io_control.hpp"
#include "baci_lib_condition_selector.hpp"
#include "baci_lib_condition_utils.hpp"
#include "baci_lib_discret.hpp"
#include "baci_lib_utils_createdis.hpp"
#include "baci_linalg_utils_sparse_algebra_math.hpp"
#include "baci_linear_solver_method_linalg.hpp"
#include "baci_poroelast_monolithic.hpp"
#include "baci_scatra_algorithm.hpp"
#include "baci_scatra_timint_implicit.hpp"
#include "baci_scatra_utils_clonestrategy.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor                                           hemmler 07/14 |
 *----------------------------------------------------------------------*/
FS3I::PartFpS3I1Wc::PartFpS3I1Wc(const Epetra_Comm& comm) : PartFPS3I(comm)
{
  // keep constructor empty
  return;
}


/*----------------------------------------------------------------------*
 |  Init                                                    rauch 09/16 |
 *----------------------------------------------------------------------*/
void FS3I::PartFpS3I1Wc::Init()
{
  FS3I::PartFPS3I::Init();



  return;
}


/*----------------------------------------------------------------------*
 |  Setup                                                   rauch 09/16 |
 *----------------------------------------------------------------------*/
void FS3I::PartFpS3I1Wc::Setup()
{
  FS3I::PartFPS3I::Setup();

  // add proxy of fluid degrees of freedom to scatra discretization
  if (scatravec_[0]->ScaTraField()->Discretization()->AddDofSet(
          fpsi_->FluidField()->Discretization()->GetDofSetProxy()) != 1)
    dserror("Scatra discretization has illegal number of dofsets!");

  // check if scatra field has 2 discretizations, so that coupling is possible
  if (scatravec_[1]->ScaTraField()->Discretization()->AddDofSet(
          fpsi_->PoroField()->StructureField()->Discretization()->GetDofSetProxy()) != 1)
    dserror("unexpected dof sets in structure field");
  // check if scatra field has 3 discretizations, so that coupling is possible
  if (scatravec_[1]->ScaTraField()->Discretization()->AddDofSet(
          fpsi_->PoroField()->FluidField()->Discretization()->GetDofSetProxy()) != 2)
    dserror("unexpected dof sets in structure field");

  // Note: in the scatra fields we have now the following dof-sets:
  // fluidscatra dofset 0: fluidscatra dofset
  // fluidscatra dofset 1: fluid dofset
  // structscatra dofset 0: structscatra dofset
  // structscatra dofset 1: structure dofset
  // structscatra dofset 2: porofluid dofset

  return;
}


/*----------------------------------------------------------------------*
 |  Timeloop                                              hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFpS3I1Wc::Timeloop()
{
  CheckIsInit();
  CheckIsSetup();

  // write FPSI solution into scatra field
  SetFPSISolution();

  // output of initial state
  ScatraOutput();

  fpsi_->PrepareTimeloop();

  while (NotFinished())
  {
    IncrementTimeAndStep();

    DoFPSIStep();  // TODO: One could think about skipping the very costly FSI/FPSI calculation for
                   // the case that it is stationary at some point (Thon)
    SetFPSISolution();  // write FPSI solution into scatra field
    DoScatraStep();
  }
}


/*----------------------------------------------------------------------*
 |  FPSI step                                             hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFpS3I1Wc::DoFPSIStep()
{
  fpsi_->PrepareTimeStep();
  fpsi_->SetupNewton();
  fpsi_->TimeStep();

  constexpr bool force_prepare = false;
  fpsi_->PrepareOutput(force_prepare);
  fpsi_->Update();
  fpsi_->Output();
}


/*----------------------------------------------------------------------*
 |  Scatra step                                           hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFpS3I1Wc::DoScatraStep()
{
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n***********************\n SCALAR TRANSPORT SOLVER \n***********************\n";
  }

  // first scatra field is associated with fluid, second scatra field is
  // associated with structure

  bool stopnonliniter = false;
  int itnum = 0;

  PrepareTimeStep();

  while (stopnonliniter == false)
  {
    ScatraEvaluateSolveIterUpdate();
    itnum++;
    if (ScatraConvergenceCheck(itnum)) break;
  }

  UpdateScatraFields();
  ScatraOutput();
}


/*----------------------------------------------------------------------*
 |  Prepare time step                                     hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFpS3I1Wc::PrepareTimeStep()
{
  CheckIsInit();
  CheckIsSetup();

  // prepare time step for both fluid- and poro-based scatra field
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->ScaTraField()->PrepareTimeStep();
  }
}


/*----------------------------------------------------------------------*
 |  Check of convergence of scatra solver                 hemmler 07/14 |
 *----------------------------------------------------------------------*/
bool FS3I::PartFpS3I1Wc::ScatraConvergenceCheck(const int itnum)
{
  const Teuchos::ParameterList& fs3idyn = GLOBAL::Problem::Instance()->FS3IDynamicParams();
  INPAR::SCATRA::SolverType scatra_solvtype =
      CORE::UTILS::IntegralValue<INPAR::SCATRA::SolverType>(fs3idyn, "SCATRA_SOLVERTYPE");

  double conresnorm(0.0);
  scatrarhs_->Norm2(&conresnorm);
  double incconnorm(0.0);
  scatraincrement_->Norm2(&incconnorm);

  switch (scatra_solvtype)
  {
    case INPAR::SCATRA::solvertype_linear_incremental:
    {
      // print the screen info
      if (Comm().MyPID() == 0)
      {
        printf("\n+-------------------+-------------------+\n");
        printf("| norm of residual  | norm of increment |\n");
        printf("+-------------------+-------------------+\n");
        printf("|    %10.3E     |    %10.3E     |\n", conresnorm, incconnorm);
        printf("+-------------------+-------------------+\n\n");
      }
      return true;
    }
    break;
    case INPAR::SCATRA::solvertype_nonlinear:
    {
      // some input parameters for the scatra fields
      const Teuchos::ParameterList& scatradyn =
          GLOBAL::Problem::Instance()->ScalarTransportDynamicParams();
      const int itemax = scatradyn.sublist("NONLINEAR").get<int>("ITEMAX");
      const double ittol = scatradyn.sublist("NONLINEAR").get<double>("CONVTOL");
      const double abstolres = scatradyn.sublist("NONLINEAR").get<double>("ABSTOLRES");

      double connorm(0.0);
      // set up vector of absolute concentrations
      Teuchos::RCP<Epetra_Vector> con = Teuchos::rcp(new Epetra_Vector(scatraincrement_->Map()));
      Teuchos::RCP<const Epetra_Vector> scatra1 = scatravec_[0]->ScaTraField()->Phinp();
      Teuchos::RCP<const Epetra_Vector> scatra2 = scatravec_[1]->ScaTraField()->Phinp();
      SetupCoupledScatraVector(con, scatra1, scatra2);
      con->Norm2(&connorm);

      // care for the case that nothing really happens in the concentration field
      if (connorm < 1e-5) connorm = 1.0;

      // print the screen info
      if (Comm().MyPID() == 0)
      {
        printf("|  %3d/%3d   |   %10.3E [L_2 ]  | %10.3E   |   %10.3E [L_2 ]  | %10.3E   |\n",
            itnum, itemax, abstolres, conresnorm, ittol, incconnorm / connorm);
      }

      // this is the convergence check
      // We always require at least one solve. We test the L_2-norm of the
      // current residual. Norm of residual is just printed for information
      if (conresnorm <= abstolres and incconnorm / connorm <= ittol)
      {
        if (Comm().MyPID() == 0)
        {
          // print 'finish line'
          printf(
              "+------------+----------------------+--------------+----------------------+---------"
              "-----+\n\n");
        }
        return true;
      }
      // warn if itemax is reached without convergence, but proceed to
      // next timestep...
      else if (itnum == itemax)
      {
        if (Comm().MyPID() == 0)
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
      dserror("Illegal ScaTra solvertype in FPS3I");
      break;
  }
  return false;
}

FOUR_C_NAMESPACE_CLOSE
