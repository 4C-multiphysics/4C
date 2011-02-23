/*----------------------------------------------------------------------*/
/*!
\file scatra_dyn.cpp
\brief entry point for (passive) scalar transport problems

<pre>
Maintainer: Volker Gravemeier
            vgravem@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15245
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <Teuchos_TimeMonitor.hpp>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include "scatra_dyn.H"
#include "passive_scatra_algorithm.H"
#include "scatra_utils.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_utils_createdis.H"
#include "../drt_adapter/adapter_scatra_base_algorithm.H"
#include "scatra_resulttest.H"
#include <Teuchos_StandardParameterEntryValidators.hpp>


/*----------------------------------------------------------------------*
 * Main control routine for scalar transport problems, icl. various solvers
 *
 *        o Laplace-/ Poisson equation (zero velocity field)
 *          (with linear and nonlinear boundary conditons)
 *        o transport of passive scalar in velocity field given by spatial function
 *        o transport of passive scalar in velocity field given by Navier-Stokes
 *          (one-way coupling)
 *
 *----------------------------------------------------------------------*/
void scatra_dyn(int disnumff, int disnumscatra, int restart)
{
  // create a communicator
#ifdef PARALLEL
  Epetra_MpiComm comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm comm;
#endif

  // access the problem-specific parameter list
  const Teuchos::ParameterList& scatradyn = DRT::Problem::Instance()->ScalarTransportDynamicParams();

  // access the fluid discretization
  RefCountPtr<DRT::Discretization> fluiddis = DRT::Problem::Instance()->Dis(disnumff,0);
  // access the scatra discretization
  RefCountPtr<DRT::Discretization> scatradis = DRT::Problem::Instance()->Dis(disnumscatra,0);

  // ensure that all dofs are assigned in the right order; this creates dof numbers with
  //       fluid dof < scatra dof
  fluiddis->FillComplete();
  scatradis->FillComplete();

  // set velocity field
  const INPAR::SCATRA::VelocityField veltype
    = DRT::INPUT::IntegralValue<INPAR::SCATRA::VelocityField>(scatradyn,"VELOCITYFIELD");
  switch (veltype)
  {
    case INPAR::SCATRA::velocity_zero:  // zero  (see case 1)
    case INPAR::SCATRA::velocity_function:  // function
    {
      // we directly use the elements from the scalar transport elements section
      if (scatradis->NumGlobalNodes()==0)
        dserror("No elements in the ---TRANSPORT ELEMENTS section");

      // create instance of scalar transport basis algorithm (empty fluid discretization)
      Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatraonly = rcp(new ADAPTER::ScaTraBaseAlgorithm(scatradyn,false));

      // read the restart information, set vectors and variables
      if (restart) scatraonly->ScaTraField().ReadRestart(restart);

      // set velocity field
      //(this is done only once. Time-dependent velocity fields are not supported)
      (scatraonly->ScaTraField()).SetVelocityField();

      // enter time loop to solve problem with given convective velocity
      (scatraonly->ScaTraField()).TimeLoop();

      // perform the result test if required
      DRT::Problem::Instance()->AddFieldTest(scatraonly->CreateScaTraFieldTest());
      DRT::Problem::Instance()->TestAll(comm);

      break;
    }
    case INPAR::SCATRA::velocity_Navier_Stokes:  // Navier_Stokes
    {
      // we use the fluid discretization as layout for the scalar transport discretization
      if (fluiddis->NumGlobalNodes()==0) dserror("Fluid discretization is empty!");

      // create scatra elements if the scatra discretization is empty
      if (scatradis->NumGlobalNodes()==0)
      {
        Epetra_Time time(comm);

        // fetch the desired material id for the transport elements
        const int matid = scatradyn.get<int>("MATID");
        // create the scatra discretization
        {
        Teuchos::RCP<DRT::UTILS::DiscretizationCreator<SCATRA::ScatraFluidCloneStrategy> > clonewizard =
              Teuchos::rcp(new DRT::UTILS::DiscretizationCreator<SCATRA::ScatraFluidCloneStrategy>() );

        clonewizard->CreateMatchingDiscretization(fluiddis,scatradis,matid);
        }
        if (comm.MyPID()==0)
        cout<<"Created scalar transport discretization from fluid field in...."
        <<time.ElapsedTime() << " secs\n\n";
      }
      else
        dserror("Fluid AND ScaTra discretization present. This is not supported.");

      // we need a non-const list in order to be able to add sublists below!
      Teuchos::ParameterList prbdyn(scatradyn);
      // support for turbulent flow statistics
      const Teuchos::ParameterList& fdyn = (DRT::Problem::Instance()->FluidDynamicParams());
      prbdyn.sublist("TURBULENCE MODEL")=fdyn.sublist("TURBULENCE MODEL");

      // create an one-way coupling algorithm instance
      Teuchos::RCP<SCATRA::PassiveScaTraAlgorithm> algo = Teuchos::rcp(new SCATRA::PassiveScaTraAlgorithm(comm,prbdyn));

      if (restart)
      {
        // read the restart information, set vectors and variables
        algo->ReadRestart(restart);
      }

      // solve the whole (one-way-coupled) problem
      algo->TimeLoop();

      // summarize the performance measurements
      Teuchos::TimeMonitor::summarize();

      // perform the result test
      DRT::Problem::Instance()->AddFieldTest(algo->FluidField().CreateFieldTest());
      DRT::Problem::Instance()->AddFieldTest(algo->CreateScaTraFieldTest());
      DRT::Problem::Instance()->TestAll(comm);

      break;
    } // case 2
    default:
      dserror("unknown velocity field type for transport of passive scalar");
  }

  return;

} // end of scatra_dyn()

#endif  // #ifdef CCADISCRET
