// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_turbulence_turbulent_flow_algorithm.hpp"

#include "4C_fluid_discret_extractor.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 | Constructor (public)                                  rasthofer 06/11|
 *----------------------------------------------------------------------*/
FLD::TurbulentFlowAlgorithm::TurbulentFlowAlgorithm(
    MPI_Comm comm, const Teuchos::ParameterList& fdyn)
    : step_(0)
{
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "#-----------------------------------------------#" << std::endl;
    std::cout << "#       INITIALIZE BASIC FLUID ALGORITHM        #" << std::endl;
    std::cout << "#-----------------------------------------------#" << std::endl;
  }
  // initialize fluid algorithm
  // this is the first and main fluid algorithm
  fluidalgo_ = std::make_shared<Adapter::FluidBaseAlgorithm>(fdyn, fdyn, "fluid", false);

  // get the compete fluid discretization
  fluiddis_ = fluidalgo_->fluid_field()->discretization();
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "#-----------------------------------------------#" << std::endl;
    std::cout << "#         EXTRACT INFLOW DISCRETIZATION         #" << std::endl;
    std::cout << "#-----------------------------------------------#" << std::endl;
  }
  // build extra discretization for turbulent inflow generation
  inflowgenerator_ =
      std::make_shared<FluidDiscretExtractor>(fluiddis_, "TurbulentInflowSection", true);
  // and get this discretization
  inflowdis_ = inflowgenerator_->get_child_discretization();

  // set number of time steps
  numtimesteps_ = fdyn.sublist("TURBULENT INFLOW").get<int>("NUMINFLOWSTEP");

  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "#-----------------------------------------------#" << std::endl;
    std::cout << "#       INITIALIZE INFLOW FLUID ALGORITHM       #" << std::endl;
    std::cout << "#-----------------------------------------------#" << std::endl;
  }

  // initialize fluid inflow algorithm
  // this is a second fluid algorithm
  inflowfluidalgo_ = std::make_shared<Adapter::FluidBaseAlgorithm>(fdyn, inflowdis_);

  return;
}


/*--------------------------------------------------------------------------------*
 | Algorithm for development of turbulent flow in inflow section   rasthofer 06/11|
 *--------------------------------------------------------------------------------*/
void FLD::TurbulentFlowAlgorithm::time_loop()
{
  if (Core::Communication::my_mpi_rank(fluiddis_->get_comm()) == 0)
  {
    std::cout << "#-----------------------------------------------#" << std::endl;
    std::cout << "#       START TURBULENT INFLOW COMPUTATION      #" << std::endl;
    std::cout << "#-----------------------------------------------#\n" << std::endl;
  }

  while (step_ < numtimesteps_)
  {
    step_++;

    // prepare time integration
    inflowfluidalgo_->fluid_field()->prepare_time_step();
    if (Core::Communication::my_mpi_rank(fluiddis_->get_comm()) == 0)
      printf("#   STEP = %4d/%4d     TIME: %11.4E  DT = %11.4E \n", step_, numtimesteps_,
          inflowfluidalgo_->fluid_field()->time(), inflowfluidalgo_->fluid_field()->dt());
    // slove nonlinear problem
    inflowfluidalgo_->fluid_field()->solve();
    // update time integration
    inflowfluidalgo_->fluid_field()->update();
    // write output of statistics only
    // remark: does also gmsh-output if required
    inflowfluidalgo_->fluid_field()->statistics_output();

    // transfer solution of inflow section to fluid discretization
    transfer_inflow_velocity();

    // increase time and step only
    fluidalgo_->fluid_field()->increment_time_and_step();
    // velnp is set manually instead of being computed in Solve()
    // replaces Solve
    fluidalgo_->fluid_field()->set_velocity_field(velnp_);
    // update time integration with given velocity field
    fluidalgo_->fluid_field()->update();
    // write output
    fluidalgo_->fluid_field()->output();
  }

  if (Core::Communication::my_mpi_rank(fluiddis_->get_comm()) == 0)
  {
    std::cout << "#-----------------------------------------------#" << std::endl;
    std::cout << "#     FINISHED TURBULENT INFLOW COMPUTATION     #" << std::endl;
    std::cout << "#     -> problem ready for restart              #" << std::endl;
    std::cout << "#-----------------------------------------------#\n" << std::endl;
  }
}


/*-------------------------------------------------------------------------------------------*
 | transfer solution of inflow section to the complete fluid discretization   rasthofer 06/11|
 *-------------------------------------------------------------------------------------------*/
void FLD::TurbulentFlowAlgorithm::transfer_inflow_velocity()
{
  if (Core::Communication::my_mpi_rank(fluiddis_->get_comm()) == 0)
    std::cout << "#   transfer solution of inflow section ..." << std::flush;

  // velocity/pressure at time n+1 of inflow section
  std::shared_ptr<const Core::LinAlg::Vector<double>> inflowvelnp =
      inflowfluidalgo_->fluid_field()->velnp();

  // velocity/pressure at time n+1 to be transferred to the complete fluid field
  // get a vector layout from the complete discretization
  velnp_ = Core::LinAlg::create_vector(*fluiddis_->dof_row_map(), true);

  // get exporter for transfer of dofs from inflow discretization to complete fluid discretization
  Core::LinAlg::Export exporter(inflowvelnp->get_map(), velnp_->get_map());
  // export inflow velocity
  velnp_->export_to(*inflowvelnp, exporter, Insert);

  if (Core::Communication::my_mpi_rank(fluiddis_->get_comm()) == 0)
    std::cout << "done\n" << std::endl;

  return;
}


/*---------------------------------------------------------------------------*
 | read restart                                               rasthofer 06/11|
 *---------------------------------------------------------------------------*/
void FLD::TurbulentFlowAlgorithm::read_restart(const int restart)
{
  if (Core::Communication::my_mpi_rank(fluiddis_->get_comm()) == 0)
  {
    std::cout << "#-----------------------------------------------#" << std::endl;
    std::cout << "#                 READ RESTART                  #" << std::endl;
    std::cout << "#-----------------------------------------------#\n" << std::endl;
  }
  // As we don't write a separate output for the inflow section, we first read
  // the values of the complete discretization, then extract the values belonging
  // to the inflow section and, finally, set them manually as restart values
  // in the fluid time integration.

  // set step
  step_ = restart;

  // read restart for complete discretization
  fluidalgo_->fluid_field()->read_restart(restart);

  // vectors to be transferred to the inflow field
  // get a vector layout from the inflow discretization
  std::shared_ptr<Core::LinAlg::Vector<double>> velnp;
  velnp = Core::LinAlg::create_vector(*inflowdis_->dof_row_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> veln;
  veln = Core::LinAlg::create_vector(*inflowdis_->dof_row_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> velnm;
  velnm = Core::LinAlg::create_vector(*inflowdis_->dof_row_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> accnp;
  accnp = Core::LinAlg::create_vector(*inflowdis_->dof_row_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> accn;
  accn = Core::LinAlg::create_vector(*inflowdis_->dof_row_map(), true);

  // get all vectors of restart
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidvelnp =
      fluidalgo_->fluid_field()->velnp();
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidveln = fluidalgo_->fluid_field()->veln();
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidvelnm =
      fluidalgo_->fluid_field()->velnm();
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidaccnp =
      fluidalgo_->fluid_field()->accnp();
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidaccn = fluidalgo_->fluid_field()->accn();

  // export vectors to inflow discretization
  Core::LinAlg::Export exportvelnp(fluidvelnp->get_map(), velnp->get_map());
  velnp->export_to(*fluidvelnp, exportvelnp, Insert);
  Core::LinAlg::Export exportveln(fluidveln->get_map(), veln->get_map());
  veln->export_to(*fluidveln, exportveln, Insert);
  Core::LinAlg::Export exportvelnm(fluidvelnm->get_map(), velnm->get_map());
  velnm->export_to(*fluidvelnm, exportvelnm, Insert);
  Core::LinAlg::Export exportaccnp(fluidaccnp->get_map(), accnp->get_map());
  accnp->export_to(*fluidaccnp, exportaccnp, Insert);
  Core::LinAlg::Export exportaccn(fluidaccn->get_map(), accn->get_map());
  accn->export_to(*fluidaccn, exportaccn, Insert);


  // set values in the inflow field
  inflowfluidalgo_->fluid_field()->set_restart(
      restart, fluidalgo_->fluid_field()->time(), velnp, veln, velnm, accnp, accn);

  if (Core::Communication::my_mpi_rank(fluiddis_->get_comm()) == 0)
    std::cout << "#   ... done \n" << std::endl;

  return;
}

FOUR_C_NAMESPACE_CLOSE
