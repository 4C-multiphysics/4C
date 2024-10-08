/*-----------------------------------------------------------*/
/*! \file

\brief Main control routine for all fluid (in)stationary solvers,

     including instationary solvers based on

     o one-step-theta time-integration scheme

     o two-step BDF2 time-integration scheme
       (with potential one-step-theta start algorithm)

     o generalized-alpha time-integration scheme

     and stationary solver.


\level 1

*/
/*-----------------------------------------------------------*/

#include "4C_fluid_dyn_nln_drt.hpp"

#include "4C_adapter_fld_base_algorithm.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_turbulence_turbulent_flow_algorithm.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 * Main control routine for fluid including various solvers:
 *
 *        o instationary one-step-theta
 *        o instationary BDF2
 *        o instationary generalized-alpha (two versions)
 *        o stationary
 *
 *----------------------------------------------------------------------*/
void dyn_fluid_drt(const int restart)
{
  // create a communicator
  const Epetra_Comm& comm = Global::Problem::instance()->get_dis("fluid")->get_comm();

  // access to some parameter lists
  // const Teuchos::ParameterList& probtype = Global::Problem::instance()->ProblemTypeParams();
  const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();

  // prepares a turbulent flow simulation with generation of turbulent inflow during the
  // actual simulation
  // this is done in two steps
  // 1. computation of inflow until it reaches a fully turbulent state
  // 2. computation of the main problem after restart
  // Remark: we restart the simulation to save procs!
  if ((fdyn.sublist("TURBULENT INFLOW").get<bool>("TURBULENTINFLOW")) and
      (restart < fdyn.sublist("TURBULENT INFLOW").get<int>("NUMINFLOWSTEP")))
  {
    if (comm.MyPID() == 0)
    {
      std::cout << "#-----------------------------------------------#" << std::endl;
      std::cout << "#      ENTER TURBULENT INFLOW COMPUTATION       #" << std::endl;
      std::cout << "#-----------------------------------------------#" << std::endl;
    }

    // create instance of fluid turbulent flow algorithm
    Teuchos::RCP<FLD::TurbulentFlowAlgorithm> turbfluidalgo =
        Teuchos::make_rcp<FLD::TurbulentFlowAlgorithm>(comm, fdyn);

    // read the restart information, set vectors and variables
    if (restart) turbfluidalgo->read_restart(restart);

    // run simulation for a separate part of the complete domain to get turbulent flow in it
    // after restart a turbulent inflow profile is computed in the separate inflow section and
    // transferred as dirichlet boundary condition to the problem domain of interest
    // this finally allows to get high quality turbulent inflow conditions during simulation of the
    // actual flow
    turbfluidalgo->time_loop();

    // perform result tests if required
    Global::Problem::instance()->add_field_test(turbfluidalgo->do_result_check());
    Global::Problem::instance()->test_all(comm);
  }
  // solve a simple fluid problem
  else
  {
    // create instance of fluid basis algorithm
    Teuchos::RCP<Adapter::FluidBaseAlgorithm> fluidalgo =
        Teuchos::make_rcp<Adapter::FluidBaseAlgorithm>(fdyn, fdyn, "fluid", false);

    // read the restart information, set vectors and variables
    if (restart) fluidalgo->fluid_field()->read_restart(restart);

    // run the simulation
    //    fluidalgo->fluid_field()->TimeLoop();
    fluidalgo->fluid_field()->integrate();

    // perform result tests if required
    Global::Problem::instance()->add_field_test(fluidalgo->fluid_field()->create_field_test());
    Global::Problem::instance()->test_all(comm);
  }

  // have fun with your results!
  return;

}  // end of dyn_fluid_drt()

FOUR_C_NAMESPACE_CLOSE
