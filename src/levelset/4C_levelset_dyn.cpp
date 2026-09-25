// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_levelset_dyn.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_global_data.hpp"
#include "4C_levelset_algorithm.hpp"
#include "4C_scatra_input.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>
#include <memory>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 * Main control routine for level set problems
 *----------------------------------------------------------------------*/
void levelset_dyn(Global::Problem& problem, int restart)
{
  // access the scatra discretization
  std::shared_ptr<Core::FE::Discretization> scatradis = problem.get_dis("scatra");

  // access the communicator
  MPI_Comm comm = scatradis->get_comm();

  // print warning to screen
  if (Core::Communication::my_mpi_rank(comm) == 0)
    std::cout << "You are now about to enter the module for level-set problems!" << std::endl;

  // access the level-set-specific parameter list
  const Teuchos::ParameterList& levelsetcontrol = problem.level_set_control();

  // access the scatra-specific parameter list
  const Teuchos::ParameterList& scatradyn = problem.scalar_transport_dynamic_params();

  // check velocity field
  const auto veltype = Teuchos::getIntegralValue<ScaTra::VelocityField>(scatradyn, "VELOCITYFIELD");
  if (veltype != ScaTra::VelocityField::function)
    FOUR_C_THROW(
        "Other velocity fields than a field given by a function not yet supported for level-set "
        "problems");

  // get linear solver id from SCALAR TRANSPORT DYNAMIC
  const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for SCALAR_TRANSPORT problem. "
        "Please set LINEAR_SOLVER in SCALAR TRANSPORT DYNAMIC to a valid number!");

  // create instance of scalar transport basis algorithm (empty fluid discretization)
  std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatrabase =
      std::make_shared<Adapter::ScaTraBaseAlgorithm>(
          problem, levelsetcontrol, scatradyn, problem.solver_params(linsolvernumber));

  // add proxy of velocity related degrees of freedom to scatra discretization
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(problem.n_dim() + 1, 0, 0, true);
  if (scatradis->add_dof_set(dofsetaux) != 1)
    FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
  scatrabase->scatra_field()->set_number_of_dof_set_velocity(1);

  // finalize discretization
  scatradis->fill_complete();

  // we directly use the elements from the scalar transport elements section
  if (scatradis->num_global_nodes() == 0)
    FOUR_C_THROW("No elements in the ---TRANSPORT ELEMENTS section");

  // first we initialize the base algorithm
  // time integrator is initialized inside.
  scatrabase->init();

  // only now we must call setup() on the base algo.
  // all objects relying on the parallel distribution are
  // created and pointers are set.
  // calls setup() in time integrator inside.
  scatrabase->setup();

  // get pointer to time integrator
  std::shared_ptr<ScaTra::ScaTraTimIntImpl> levelsetalgo = scatrabase->scatra_field();

  // read the restart information, set vectors and variables
  if (restart) levelsetalgo->read_restart(restart);

  // set initial velocity field
  // note: The order read_restart() before set_velocity_field() is important here!!
  //       The velocity field is not initialized in the constructor of the basic scalar field.
  //       Moreover, it is not read from restart data. Therefore, we first have to set the restart
  //       time in the function read_restart() and then in case of time-dependent velocity fields to
  //       evaluate the velocity function and curve.
  // bool true allows for setting old convective velocity required for particle coupling
  // old particle framework removed -> todo: requires clean up
  std::dynamic_pointer_cast<LevelSet::LevelSetAlgorithm>(levelsetalgo)->set_velocity_field(true);

  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR("LEVEL SET:  + time loop");

  // enter time loop
  levelsetalgo->time_loop();

  // perform result test if required
  std::dynamic_pointer_cast<LevelSet::LevelSetAlgorithm>(levelsetalgo)->test_results();
}  // end of levelset_dyn()

FOUR_C_NAMESPACE_CLOSE
