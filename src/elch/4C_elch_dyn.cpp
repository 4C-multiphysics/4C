// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_elch_dyn.hpp"

#include "4C_ale_utils_clonestrategy.hpp"
#include "4C_elch_algorithm.hpp"
#include "4C_elch_input.hpp"
#include "4C_elch_moving_boundary_algorithm.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_timint_elch.hpp"
#include "4C_scatra_utils_clonestrategy.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void elch_dyn(Global::Problem& problem, int restart)
{
  // access the communicator
  const auto& comm = problem.get_dis("fluid")->get_comm();

  // print ELCH-Logo to screen
  if (Core::Communication::my_mpi_rank(comm) == 0) printlogo();

  // access the fluid discretization
  auto fluiddis = problem.get_dis("fluid");
  // access the scatra discretization
  auto scatradis = problem.get_dis("scatra");

  // ensure that all dofs are assigned in the right order; this creates dof numbers with
  //       fluid dof < scatra/elch dof
  fluiddis->fill_complete();
  scatradis->fill_complete();

  // access the problem-specific parameter list
  const auto& elchcontrol = problem.elch_control_params();

  // access the scalar transport parameter list
  const auto& scatradyn = problem.scalar_transport_dynamic_params();
  const auto veltype = Teuchos::getIntegralValue<ScaTra::VelocityField>(scatradyn, "VELOCITYFIELD");

  // choose algorithm depending on velocity field type
  switch (veltype)
  {
    case ScaTra::VelocityField::zero:
    case ScaTra::VelocityField::function:
    {
      // we directly use the elements from the scalar transport elements section
      if (scatradis->num_global_nodes() == 0)
        FOUR_C_THROW("No elements in the ---TRANSPORT ELEMENTS section");

      // get linear solver id from SCALAR TRANSPORT DYNAMIC
      const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
      if (linsolvernumber == -1)
      {
        FOUR_C_THROW(
            "no linear solver defined for ELCH problem. Please set LINEAR_SOLVER in SCALAR "
            "TRANSPORT DYNAMIC to a valid number!");
      }

      // create instance of scalar transport basis algorithm (empty fluid discretization)
      Adapter::ScaTraBaseAlgorithm scatraonly(
          problem, scatradyn, scatradyn, problem.solver_params(linsolvernumber));

      // add proxy of velocity related degrees of freedom to scatra discretization
      auto dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
          problem.n_dim() + 1, 0, 0, true);
      if (scatradis->add_dof_set(dofsetaux) != 1)
        FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
      scatraonly.scatra_field()->set_number_of_dof_set_velocity(1);

      // now me may redistribute or ghost the scatra discretization
      // finalize discretization
      scatradis->fill_complete();

      // now we can call init() on the base algorithm
      // scatra time integrator is constructed and initialized inside
      scatraonly.init();

      // now me may redistribute or ghost the scatra discretization
      // finalize discretization
      scatradis->fill_complete();

      // only now we must call setup() on the base algorithm.
      // all objects relying on the parallel distribution are
      // created and pointers are set.
      // calls setup() on time integrator inside.
      scatraonly.setup();

      // read the restart information, set vectors and variables
      if (restart) scatraonly.scatra_field()->read_restart(restart);

      // set velocity field
      // note: The order read_restart() before set_velocity_field_from_function() is important
      // here!! for time-dependent velocity fields, set_velocity_field_from_function() is
      // additionally called in each prepare_time_step()-call
      scatraonly.scatra_field()->set_velocity_field_from_function();

      // enter time loop to solve problem with given convective velocity
      scatraonly.scatra_field()->time_loop();

      // perform the result test if required
      scatraonly.scatra_field()->test_results();

      break;
    }
    case ScaTra::VelocityField::from_other_field:
    {
      // we use the fluid discretization as layout for the scalar transport discretization
      if (fluiddis->num_global_nodes() == 0) FOUR_C_THROW("Fluid discretization is empty!");

      // create scatra elements if the scatra discretization is empty
      if (scatradis->num_global_nodes() == 0)
      {
        // fill scatra discretization by cloning fluid discretization
        Core::FE::clone_discretization<ScaTra::ScatraFluidCloneStrategy>(
            *fluiddis, *scatradis, problem.cloning_material_map());
        scatradis->fill_complete();
        // determine implementation type of cloned scatra elements
        ScaTra::ImplType impltype = ScaTra::impltype_undefined;
        if (elchcontrol.get<bool>("DIFFCOND_FORMULATION"))
          impltype = ScaTra::impltype_elch_diffcond;
        else
          impltype = ScaTra::impltype_elch_NP;

        // set implementation type
        for (int i = 0; i < scatradis->num_my_col_elements(); ++i)
        {
          auto* element = dynamic_cast<Discret::Elements::Transport*>(scatradis->l_col_element(i));
          FOUR_C_ASSERT_ALWAYS(element != nullptr, "Invalid element type!");
          element->set_impl_type(impltype);
        }
      }
      else
      {
        FOUR_C_THROW("Fluid AND ScaTra discretization present. This is not supported.");
      }

      // support for turbulent flow statistics
      const auto& fdyn = (problem.fluid_dynamic_params());

      std::shared_ptr<Core::FE::Discretization> aledis = problem.get_dis("ale");
      if (!aledis->filled()) aledis->fill_complete(Core::FE::OptionsFillComplete::none());
      // is ALE needed or not?
      const auto withale =
          Teuchos::getIntegralValue<ElCh::ElchMovingBoundary>(elchcontrol, "MOVINGBOUNDARY");

      if (withale != ElCh::elch_mov_bndry_no)
      {
        // create ale elements only if the ale discretization is empty
        if (aledis->num_global_nodes() == 0)
        {
          // clone ALE discretization from fluid discretization
          Core::FE::clone_discretization<ALE::Utils::AleCloneStrategy>(
              *fluiddis, *aledis, problem.cloning_material_map());

          aledis->fill_complete({
              .assign_degrees_of_freedom = true,
              .init_elements = true,
              .do_boundary_conditions = false,
          });
          // setup material in every ALE element
          Teuchos::ParameterList params;
          params.set<std::string>("action", "setup_material");
          aledis->evaluate(params);
        }
        else
          FOUR_C_THROW("Providing an ALE mesh is not supported for problemtype Electrochemistry.");

        // get linear solver id from SCALAR TRANSPORT DYNAMIC
        const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
        if (linsolvernumber == -1)
        {
          FOUR_C_THROW(
              "no linear solver defined for ELCH problem. Please set LINEAR_SOLVER in SCALAR "
              "TRANSPORT DYNAMIC to a valid number!");
        }

        // create an ElCh::MovingBoundaryAlgorithm instance
        // NOTE: elch reads time parameters from scatra dynamic section!
        ElCh::MovingBoundaryAlgorithm elch(
            problem, comm, elchcontrol, scatradyn, problem.solver_params(linsolvernumber));

        // add proxy of fluid degrees of freedom to scatra discretization
        if (scatradis->add_dof_set(fluiddis->get_dof_set_proxy()) != 1)
          FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
        elch.scatra_field()->set_number_of_dof_set_velocity(1);

        // add proxy of ALE degrees of freedom to scatra discretization
        if (scatradis->add_dof_set(aledis->get_dof_set_proxy()) != 2)
          FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
        elch.scatra_field()->set_number_of_dof_set_displacement(2);

        // now we must call init()
        elch.init();

        // NOTE : At this point we may redistribute and/or
        //        ghost our discretizations at will.
        scatradis->fill_complete();
        fluiddis->fill_complete();
        aledis->fill_complete();

        // now we can call setup() on the scatra time integrator
        elch.setup();

        // read the restart information, set vectors and variables
        if (restart) elch.read_restart(restart);

        // solve the whole electrochemistry problem
        elch.time_loop();

        // perform the result test
        elch.test_results();
      }
      else
      {
        // get linear solver id from SCALAR TRANSPORT DYNAMIC
        const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
        if (linsolvernumber == -1)
        {
          FOUR_C_THROW(
              "no linear solver defined for ELCH problem. Please set LINEAR_SOLVER in SCALAR "
              "TRANSPORT DYNAMIC to a valid number!");
        }

        // create an ElCh::Algorithm instance
        // NOTE: elch reads time parameters from scatra dynamic section!
        ElCh::Algorithm elch(
            problem, comm, elchcontrol, scatradyn, fdyn, problem.solver_params(linsolvernumber));

        // add proxy of fluid degrees of freedom to scatra discretization
        if (scatradis->add_dof_set(fluiddis->get_dof_set_proxy()) != 1)
          FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
        elch.scatra_field()->set_number_of_dof_set_velocity(1);

        // now we must call init()
        elch.init();

        // NOTE : At this point we may redistribute and/or
        //        ghost our discretizations at will.
        scatradis->fill_complete();
        fluiddis->fill_complete();
        aledis->fill_complete();

        // discretizations are done, now we can call setup() on the algorithm
        elch.setup();

        // read the restart information, set vectors and variables
        if (restart) elch.read_restart(restart);

        // solve the whole electrochemistry problem
        elch.time_loop();

        // perform the result test
        elch.test_results();
      }

      break;
    }
    default:
      FOUR_C_THROW("Unknown velocity field type for transport of passive scalar: {}", veltype);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void printlogo()
{
  // more at http://www.ascii-art.de under entry "moose" (or "elk")
  std::cout << "     ___            ___    " << '\n';
  std::cout << "    /   \\          /   \\ " << '\n';
  std::cout << "    \\_   \\        /  __/ " << '\n';
  std::cout << "     _\\   \\      /  /__  "
            << "     _____ _     _____  _   _   " << '\n';
  std::cout << "     \\___  \\____/   __/  "
            << "    |  ___| |   /  __ \\| | | |  " << '\n';
  std::cout << "         \\_       _/      "
            << "   | |__ | |   | /  \\/| |_| |  " << '\n';
  std::cout << "           | @ @  \\_      "
            << "   |  __|| |   | |    |  _  |   " << '\n';
  std::cout << "           |               "
            << "  | |___| |___| \\__/\\| | | | " << '\n';
  std::cout << "         _/     /\\        "
            << "   \\____/\\_____/\\____/\\_| |_/ " << '\n';
  std::cout << "        /o)  (o/\\ \\_     " << '\n';
  std::cout << "        \\_____/ /         " << '\n';
  std::cout << "          \\____/          " << '\n';
  std::cout << "                           " << '\n';
}

FOUR_C_NAMESPACE_CLOSE
