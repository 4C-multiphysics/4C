/*----------------------------------------------------------------------*/
/*! \file

\brief main control routine for monolithic scalar-thermo interaction

\level 2

*/
/*----------------------------------------------------------------------*/
#include "sti_dyn.H"

#include "sti_clonestrategy.H"
#include "sti_monolithic.H"
#include "sti_partitioned.H"
#include "sti_resulttest.H"

#include <Teuchos_TimeMonitor.hpp>

#include "drt_dofset_predefineddofnumber.H"
#include "drt_utils_createdis.H"

#include "scatra_resulttest_elch.H"
#include "scatra_timint_elch.H"

#include "adapter_scatra_base_algorithm.H"

/*--------------------------------------------------------------------------------*
 | entry point for simulations of scalar-thermo interaction problems   fang 04/15 |
 *--------------------------------------------------------------------------------*/
void sti_dyn(const int& restartstep  //! time step for restart
)
{
  // access global problem
  DRT::Problem* problem = DRT::Problem::Instance();

  // access communicator
  const Epetra_Comm& comm = problem->GetDis("scatra")->Comm();

  // access scatra discretization
  Teuchos::RCP<DRT::Discretization> scatradis = problem->GetDis("scatra");

  // add dofset for velocity-related quantities to scatra discretization
  Teuchos::RCP<DRT::DofSetInterface> dofsetaux =
      Teuchos::rcp(new DRT::DofSetPredefinedDoFNumber(problem->NDim() + 1, 0, 0, true));
  if (scatradis->AddDofSet(dofsetaux) != 1)
    dserror("Scatra discretization has illegal number of dofsets!");

  // finalize scatra discretization
  scatradis->FillComplete();

  // safety check
  if (scatradis->NumGlobalNodes() == 0)
    dserror(
        "The scatra discretization must not be empty, since the thermo discretization needs to be "
        "cloned from it!");

  // access thermo discretization
  Teuchos::RCP<DRT::Discretization> thermodis = problem->GetDis("thermo");

  // add dofset for velocity-related quantities to thermo discretization
  dofsetaux = Teuchos::rcp(new DRT::DofSetPredefinedDoFNumber(problem->NDim() + 1, 0, 0, true));
  if (thermodis->AddDofSet(dofsetaux) != 1)
    dserror("Thermo discretization has illegal number of dofsets!");

  // equip thermo discretization with noderowmap for subsequent safety check
  // final FillComplete() is called at the end of discretization cloning
  thermodis->FillComplete(false, false, false);

  // safety check
  if (thermodis->NumGlobalNodes() != 0)
    dserror(
        "The thermo discretization must be empty, since it is cloned from the scatra "
        "discretization!");

  // clone thermo discretization from scatra discretization, using clone strategy for scatra-thermo
  // interaction
  DRT::UTILS::CloneDiscretization<STI::ScatraThermoCloneStrategy>(scatradis, thermodis);
  thermodis->FillComplete(false, true, true);

  // add proxy of scalar transport degrees of freedom to thermo discretization and vice versa
  if (thermodis->AddDofSet(scatradis->GetDofSetProxy()) != 2)
    dserror("Thermo discretization has illegal number of dofsets!");
  if (scatradis->AddDofSet(thermodis->GetDofSetProxy()) != 2)
    dserror("Scatra discretization has illegal number of dofsets!");
  thermodis->FillComplete(true, false, false);
  scatradis->FillComplete(true, false, false);

  // add material of scatra elements to thermo elements and vice versa
  for (int i = 0; i < scatradis->NumMyColElements(); ++i)
  {
    DRT::Element* scatraele = scatradis->lColElement(i);
    DRT::Element* thermoele = thermodis->gElement(scatraele->Id());

    thermoele->AddMaterial(scatraele->Material());
    scatraele->AddMaterial(thermoele->Material());
  }

  // access parameter lists for scatra-thermo interaction and scalar transport field
  const Teuchos::ParameterList& stidyn = problem->STIDynamicParams();
  const Teuchos::ParameterList& scatradyn = problem->ScalarTransportDynamicParams();

  // extract and check ID of linear solver for scatra field
  const int solver_id_scatra = scatradyn.get<int>("LINEAR_SOLVER");
  if (solver_id_scatra == -1)
    dserror(
        "No linear solver for scalar transport field was specified in input file section 'SCALAR "
        "TRANSPORT DYNAMIC'!");

  // extract and check ID of linear solver for thermo field
  const int solver_id_thermo = stidyn.get<int>("THERMO_LINEAR_SOLVER");
  if (solver_id_thermo == -1)
    dserror(
        "No linear solver for temperature field was specified in input file section 'STI "
        "DYNAMIC'!");

  // instantiate coupling algorithm for scatra-thermo interaction
  Teuchos::RCP<STI::Algorithm> sti_algorithm(Teuchos::null);

  switch (Teuchos::getIntegralValue<INPAR::STI::CouplingType>(stidyn, "COUPLINGTYPE"))
  {
    // monolithic algorithm
    case INPAR::STI::CouplingType::monolithic:
    {
      // extract and check ID of monolithic linear solver
      const int solver_id = stidyn.sublist("MONOLITHIC").get<int>("LINEAR_SOLVER");
      if (solver_id == -1)
        dserror(
            "No global linear solver was specified in input file section 'STI "
            "DYNAMIC/MONOLITHIC'!");

      sti_algorithm = Teuchos::rcp(new STI::Monolithic(comm, stidyn, scatradyn,
          DRT::Problem::Instance()->SolverParams(solver_id),
          DRT::Problem::Instance()->SolverParams(solver_id_scatra),
          DRT::Problem::Instance()->SolverParams(solver_id_thermo)));

      break;
    }

    // partitioned algorithm
    case INPAR::STI::CouplingType::oneway_scatratothermo:
    case INPAR::STI::CouplingType::oneway_thermotoscatra:
    case INPAR::STI::CouplingType::twoway_scatratothermo:
    case INPAR::STI::CouplingType::twoway_scatratothermo_aitken:
    case INPAR::STI::CouplingType::twoway_scatratothermo_aitken_dofsplit:
    case INPAR::STI::CouplingType::twoway_thermotoscatra:
    case INPAR::STI::CouplingType::twoway_thermotoscatra_aitken:
    {
      sti_algorithm = Teuchos::rcp(new STI::Partitioned(comm, stidyn, scatradyn,
          DRT::Problem::Instance()->SolverParams(solver_id_scatra),
          DRT::Problem::Instance()->SolverParams(solver_id_thermo)));

      break;
    }

    // unknown algorithm
    default:
    {
      dserror("Unknown coupling algorithm for scatra-thermo interaction!");
      break;
    }
  }

  // read restart data if necessary
  if (restartstep) sti_algorithm->ReadRestart(restartstep);

  // provide scatra and thermo fields with velocities
  sti_algorithm->ScaTraField()->SetVelocityField();
  sti_algorithm->ThermoField()->SetVelocityField();

  // enter time loop and solve scatra-thermo interaction problem
  sti_algorithm->TimeLoop();

  // summarize performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform result tests
  problem->AddFieldTest(Teuchos::rcp<DRT::ResultTest>(new STI::STIResultTest(sti_algorithm)));
  if (Teuchos::getIntegralValue<INPAR::STI::ScaTraTimIntType>(
          problem->STIDynamicParams(), "SCATRATIMINTTYPE") == INPAR::STI::ScaTraTimIntType::elch)
    problem->AddFieldTest(Teuchos::rcp<DRT::ResultTest>(new SCATRA::ElchResultTest(
        Teuchos::rcp_dynamic_cast<SCATRA::ScaTraTimIntElch>(sti_algorithm->ScaTraField()))));
  else
    dserror(
        "Scatra-thermo interaction is currently only available for thermodynamic electrochemistry, "
        "but not for other kinds of thermodynamic scalar transport!");
  problem->AddFieldTest(
      Teuchos::rcp<DRT::ResultTest>(new SCATRA::ScaTraResultTest(sti_algorithm->ThermoField())));
  problem->TestAll(comm);

  return;
}  // sti_dyn()
