/*---------------------------------------------------------------------*/
/*! \file
\brief Control routine for structural dynamics (outsourced to adapter layer)

\level 1

*/
/*---------------------------------------------------------------------*/

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include "baci_adapter_str_factory.H"
#include "baci_adapter_str_structure_new.H"
#include "baci_adapter_str_structure.H"
#include "baci_structure_dyn_nln_drt.H"
#include "baci_io.H"
#include "baci_io_control.H"
#include "baci_lib_globalproblem.H"
#include "baci_comm_utils.H"
#include "baci_inpar_structure.H"
#include "baci_inpar_invanalysis.H"
#include "baci_inpar_statinvanalysis.H"
#include "baci_structure_resulttest.H"

#include "baci_lib_discret.H"
#include "baci_linalg_utils_sparse_algebra_math.H"
#include "baci_linear_solver_method_linalg.H"

// periodic boundary conditions
#include "baci_lib_periodicbc.H"

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void caldyn_drt()
{
  // get input lists
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  // major switch to different time integrators
  switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP"))
  {
    case INPAR::STR::dyna_statics:
    case INPAR::STR::dyna_genalpha:
    case INPAR::STR::dyna_genalpha_liegroup:
    case INPAR::STR::dyna_onesteptheta:
    case INPAR::STR::dyna_gemm:
    case INPAR::STR::dyna_expleuler:
    case INPAR::STR::dyna_centrdiff:
    case INPAR::STR::dyna_ab2:
    case INPAR::STR::dyna_euma:
    case INPAR::STR::dyna_euimsto:
      dyn_nlnstructural_drt();
      break;
    default:
      dserror("unknown time integration scheme '%s'", sdyn.get<std::string>("DYNAMICTYP").c_str());
      break;
  }

  return;
}


/*----------------------------------------------------------------------*
 | structural nonlinear dynamics                                        |
 *----------------------------------------------------------------------*/
void dyn_nlnstructural_drt()
{
  // get input lists
  const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
  // access the structural discretization
  Teuchos::RCP<DRT::Discretization> structdis = DRT::Problem::Instance()->GetDis("structure");

  // connect degrees of freedom for periodic boundary conditions
  {
    PeriodicBoundaryConditions pbc_struct(structdis);

    if (pbc_struct.HasPBC())
    {
      pbc_struct.UpdateDofsForPeriodicBoundaryConditions();
    }
  }

  // create an adapterbase and adapter
  Teuchos::RCP<ADAPTER::Structure> structadapter = Teuchos::null;
  // FixMe The following switch is just a temporal hack, such we can jump between the new and the
  // old structure implementation. Has to be deleted after the clean-up has been finished!
  const enum INPAR::STR::IntegrationStrategy intstrat =
      DRT::INPUT::IntegralValue<INPAR::STR::IntegrationStrategy>(sdyn, "INT_STRATEGY");
  switch (intstrat)
  {
    // -------------------------------------------------------------------
    // old implementation
    // -------------------------------------------------------------------
    case INPAR::STR::int_old:
    {
      Teuchos::RCP<ADAPTER::StructureBaseAlgorithm> adapterbase_old_ptr =
          Teuchos::rcp(new ADAPTER::StructureBaseAlgorithm(
              sdyn, const_cast<Teuchos::ParameterList&>(sdyn), structdis));
      structadapter = adapterbase_old_ptr->StructureField();
      structadapter->Setup();
      break;
    }
    // -------------------------------------------------------------------
    // new implementation
    // -------------------------------------------------------------------
    default:
    {
      Teuchos::RCP<ADAPTER::StructureBaseAlgorithmNew> adapterbase_ptr =
          ADAPTER::STR::BuildStructureAlgorithm(sdyn);
      adapterbase_ptr->Init(sdyn, const_cast<Teuchos::ParameterList&>(sdyn), structdis);
      adapterbase_ptr->Setup();
      structadapter = adapterbase_ptr->StructureField();
      break;
    }
  }

  const bool write_initial_state =
      DRT::INPUT::IntegralValue<int>(DRT::Problem::Instance()->IOParams(), "WRITE_INITIAL_STATE");
  const bool write_final_state =
      DRT::INPUT::IntegralValue<int>(DRT::Problem::Instance()->IOParams(), "WRITE_FINAL_STATE");

  // do restart
  const int restart = DRT::Problem::Instance()->Restart();
  if (restart)
  {
    structadapter->ReadRestart(restart);
  }
  // write output at beginnning of calc
  else
  {
    if (write_initial_state)
    {
      constexpr bool force_prepare = true;
      structadapter->PrepareOutput(force_prepare);
      structadapter->Output();
      structadapter->PostOutput();
    }
  }

  // run time integration
  structadapter->Integrate();

  if (write_final_state && !structadapter->HasFinalStateBeenWritten())
  {
    constexpr bool forceWriteRestart = true;
    constexpr bool force_prepare = true;
    structadapter->PrepareOutput(force_prepare);
    structadapter->Output(forceWriteRestart);
    structadapter->PostOutput();
  }

  // test results
  DRT::Problem::Instance()->AddFieldTest(structadapter->CreateFieldTest());
  DRT::Problem::Instance()->TestAll(structadapter->DofRowMap()->Comm());

  // print monitoring of time consumption
  Teuchos::RCP<const Teuchos::Comm<int>> TeuchosComm =
      COMM_UTILS::toTeuchosComm<int>(structdis->Comm());
  Teuchos::TimeMonitor::summarize(TeuchosComm.ptr(), std::cout, false, true, true);

  // time to go home...
  return;

}  // end of dyn_nlnstructural_drt()
