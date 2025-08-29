// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_dyn.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_scatra_algorithm.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_resulttest.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_utils_clonestrategy.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 * Main control routine for scalar transport problems, incl. various solvers
 *
 *        o Laplace-/ Poisson equation (zero velocity field)
 *          (with linear and nonlinear boundary conditions)
 *        o transport of passive scalar in velocity field given by spatial function
 *        o transport of passive scalar in velocity field given by Navier-Stokes
 *          (one-way coupling)
 *        o scalar transport in velocity field given by Navier-Stokes with natural convection
 *          (two-way coupling)
 *
 *----------------------------------------------------------------------*/
void scatra_dyn(int restart)
{
  // access the communicator
  MPI_Comm comm = Global::Problem::instance()->get_dis("fluid")->get_comm();

  // print problem type
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "###################################################" << '\n';
    std::cout << "# YOUR PROBLEM TYPE: " << Global::Problem::instance()->problem_name() << '\n';
    std::cout << "###################################################" << '\n';
  }

  // access the problem-specific parameter list
  const auto& scatradyn = Global::Problem::instance()->scalar_transport_dynamic_params();

  // access the fluid discretization
  auto fluiddis = Global::Problem::instance()->get_dis("fluid");
  // access the scatra discretization
  auto scatradis = Global::Problem::instance()->get_dis("scatra");

  // ensure that all dofs are assigned in the right order;
  // this creates dof numbers with fluid dof < scatra dof
  fluiddis->fill_complete(true, true, true);
  scatradis->fill_complete(true, true, true);

  // determine coupling type
  const auto fieldcoupling = Teuchos::getIntegralValue<Inpar::ScaTra::FieldCoupling>(
      Global::Problem::instance()->scalar_transport_dynamic_params(), "FIELDCOUPLING");

  // determine velocity type
  const auto veltype =
      Teuchos::getIntegralValue<Inpar::ScaTra::VelocityField>(scatradyn, "VELOCITYFIELD");

  if (scatradis->num_global_nodes() == 0)
  {
    if (fieldcoupling != Inpar::ScaTra::coupling_match and
        veltype != Inpar::ScaTra::velocity_Navier_Stokes)
    {
      FOUR_C_THROW(
          "If you want matching fluid and scatra meshes, do clone you fluid mesh and use "
          "FIELDCOUPLING match!");
    }
  }
  else
  {
    if (fieldcoupling != Inpar::ScaTra::coupling_volmortar and
        veltype == Inpar::ScaTra::velocity_Navier_Stokes)
    {
      FOUR_C_THROW(
          "If you want non-matching fluid and scatra meshes, "
          "you need to use FIELDCOUPLING volmortar!");
    }
  }

  switch (veltype)
  {
    case Inpar::ScaTra::velocity_zero:      // zero  (see case 1)
    case Inpar::ScaTra::velocity_function:  // function
    {
      // we directly use the elements from the scalar transport elements section
      if (scatradis->num_global_nodes() == 0)
        FOUR_C_THROW("No elements in the ---TRANSPORT ELEMENTS section");

      // get linear solver id from SCALAR TRANSPORT DYNAMIC
      const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
      if (linsolvernumber == -1)
      {
        FOUR_C_THROW(
            "no linear solver defined for SCALAR_TRANSPORT problem. Please set LINEAR_SOLVER in "
            "SCALAR TRANSPORT DYNAMIC to a valid number!");
      }

      // create instance of scalar transport basis algorithm (empty fluid discretization)
      Adapter::ScaTraBaseAlgorithm scatraonly(
          scatradyn, scatradyn, Global::Problem::instance()->solver_params(linsolvernumber));

      // add proxy of velocity related degrees of freedom to scatra discretization
      auto dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
          Global::Problem::instance()->n_dim() + 1, 0, 0, true);
      if (scatradis->add_dof_set(dofsetaux) != 1)
        FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
      scatraonly.scatra_field()->set_number_of_dof_set_velocity(1);

      // allow TRANSPORT conditions, too
      // NOTE: we can not use the conditions given by 'conditions_to_copy =
      // clonestrategy.conditions_to_copy()' since we than may have some scatra condition twice. So
      // we only copy the Dirichlet and Neumann conditions:
      const std::map<std::string, std::string> conditions_to_copy = {
          {"TransportDirichlet", "Dirichlet"}, {"TransportPointNeumann", "PointNeumann"},
          {"TransportLineNeumann", "LineNeumann"}, {"TransportSurfaceNeumann", "SurfaceNeumann"},
          {"TransportVolumeNeumann", "VolumeNeumann"}};

      Core::FE::DiscretizationCreatorBase creator;
      creator.copy_conditions(*scatradis, *scatradis, conditions_to_copy);

      // finalize discretization
      scatradis->fill_complete(true, false, true);

      // now we can call init() on the base algo.
      // time integrator is initialized inside
      scatraonly.init();

      // redistribution between init(...) and setup()
      // redistribute scatra elements in case of heterogeneous reactions
      if (scatradis->has_condition("ScatraHeteroReactionSlave"))
      {
        // create vector of discr.
        std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
        dis.push_back(scatradis);

        Teuchos::ParameterList binning_params =
            Global::Problem::instance()->binning_strategy_params();
        Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
            "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
            binning_params);
        Core::Rebalance::rebalance_discretizations_by_binning(binning_params,
            Global::Problem::instance()->output_control_file(), dis, nullptr, nullptr, false);
      }

      // assign degrees of freedom and rebuild geometries
      scatradis->fill_complete(true, false, true);

      // now we must call setup()
      scatraonly.setup();

      // read the restart information, set vectors and variables
      if (restart) scatraonly.scatra_field()->read_restart(restart);

      // set initial velocity field
      // note: The order read_restart() before set_velocity_field_from_function() is important
      // here!! for time-dependent velocity fields, set_velocity_field_from_function() is
      // additionally called in each prepare_time_step()-call
      scatraonly.scatra_field()->set_velocity_field_from_function();

      // set external force
      if (scatraonly.scatra_field()->has_external_force())
        scatraonly.scatra_field()->set_external_force();

      // enter time loop to solve problem with given convective velocity
      scatraonly.scatra_field()->time_loop();

      // perform the result test if required
      scatraonly.scatra_field()->test_results();
      break;
    }
    case Inpar::ScaTra::velocity_Navier_Stokes:  // Navier_Stokes
    {
      // we use the fluid discretization as layout for the scalar transport discretization
      if (fluiddis->num_global_nodes() == 0) FOUR_C_THROW("Fluid discretization is empty!");

      // create scatra elements by cloning from fluid dis in matching case
      if (fieldcoupling == Inpar::ScaTra::coupling_match)
      {
        // fill scatra discretization by cloning fluid discretization
        Core::FE::clone_discretization<ScaTra::ScatraFluidCloneStrategy>(
            *fluiddis, *scatradis, Global::Problem::instance()->cloning_material_map());

        // set implementation type of cloned scatra elements
        for (int i = 0; i < scatradis->num_my_col_elements(); ++i)
        {
          auto* element = dynamic_cast<Discret::Elements::Transport*>(scatradis->l_col_element(i));
          if (element == nullptr)
            FOUR_C_THROW("Invalid element type!");
          else
            element->set_impl_type(Inpar::ScaTra::impltype_std);
        }
      }

      // support for turbulent flow statistics
      const Teuchos::ParameterList& fdyn = (Global::Problem::instance()->fluid_dynamic_params());

      // get linear solver id from SCALAR TRANSPORT DYNAMIC
      const int linsolvernumber = scatradyn.get<int>("LINEAR_SOLVER");
      if (linsolvernumber == -1)
      {
        FOUR_C_THROW(
            "no linear solver defined for SCALAR_TRANSPORT problem. Please set LINEAR_SOLVER in "
            "SCALAR TRANSPORT DYNAMIC to a valid number!");
      }

      // create a scalar transport algorithm instance
      ScaTra::ScaTraAlgorithm algo(comm, scatradyn, fdyn, "scatra",
          Global::Problem::instance()->solver_params(linsolvernumber));

      // create scatra elements by cloning from fluid dis in matching case
      if (fieldcoupling == Inpar::ScaTra::coupling_match)
      {
        // add proxy of fluid transport degrees of freedom to scatra discretization
        if (scatradis->add_dof_set(fluiddis->get_dof_set_proxy()) != 1)
          FOUR_C_THROW("Scatra discretization has illegal number of dofsets!");
        algo.scatra_field()->set_number_of_dof_set_velocity(1);
      }

      // we create  the aux dofsets before init(...)
      // volmortar adapter init(...) relies on this
      if (fieldcoupling == Inpar::ScaTra::coupling_volmortar)
      {
        // allow TRANSPORT conditions, too
        ScaTra::ScatraFluidCloneStrategy clonestrategy;
        const auto conditions_to_copy = clonestrategy.conditions_to_copy();
        Core::FE::DiscretizationCreatorBase creator;
        creator.copy_conditions(*scatradis, *scatradis, conditions_to_copy);

        // build the element and node maps
        scatradis->fill_complete(false, false, false);
        fluiddis->fill_complete(false, false, false);

        // build auxiliary dofsets, i.e. pseudo dofs on each discretization
        const int ndofpernode_scatra = scatradis->num_dof(0, scatradis->l_row_node(0));
        const int ndofperelement_scatra = 0;
        const int ndofpernode_fluid = fluiddis->num_dof(0, fluiddis->l_row_node(0));
        const int ndofperelement_fluid = 0;

        // add proxy of velocity related degrees of freedom to scatra discretization
        std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux;
        dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
            ndofpernode_scatra, ndofperelement_scatra, 0, true);
        if (fluiddis->add_dof_set(dofsetaux) != 1)
          FOUR_C_THROW("unexpected dof sets in fluid field");
        dofsetaux = std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
            ndofpernode_fluid, ndofperelement_fluid, 0, true);
        if (scatradis->add_dof_set(dofsetaux) != 1)
          FOUR_C_THROW("unexpected dof sets in scatra field");
        algo.scatra_field()->set_number_of_dof_set_velocity(1);

        // call assign_degrees_of_freedom also for auxiliary dofsets
        // note: the order of fill_complete() calls determines the gid numbering!
        // 1. fluid dofs
        // 2. scatra dofs
        // 3. fluid auxiliary dofs
        // 4. scatra auxiliary dofs
        fluiddis->fill_complete(true, false, false);
        scatradis->fill_complete(true, false, false);
      }

      // init algo (init fluid time integrator and scatra time integrator inside)
      algo.init();

      // redistribution between init(...) and setup()
      // redistribute scatra elements if the scatra discretization is not empty
      if (fieldcoupling == Inpar::ScaTra::coupling_volmortar)
      {
        // create vector of discr.
        std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
        dis.push_back(fluiddis);
        dis.push_back(scatradis);

        Teuchos::ParameterList binning_params =
            Global::Problem::instance()->binning_strategy_params();
        Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
            "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
            binning_params);
        Core::Rebalance::rebalance_discretizations_by_binning(binning_params,
            Global::Problem::instance()->output_control_file(), dis, nullptr, nullptr, false);
      }

      // ensure that all dofs are assigned in the right order;
      // this creates dof numbers with fluid dof < scatra dof
      fluiddis->fill_complete(true, false, true);
      scatradis->fill_complete(true, false, true);

      // setup algo
      //(setup fluid time integrator and scatra time integrator inside)
      algo.setup();

      // read restart information
      // in case a inflow generation in the inflow section has been performed, there are not any
      // scatra results available and the initial field is used
      if (restart)
      {
        if (fdyn.sublist("TURBULENT INFLOW").get<bool>("TURBULENTINFLOW") and
            restart == fdyn.sublist("TURBULENT INFLOW").get<int>("NUMINFLOWSTEP"))
          algo.read_inflow_restart(restart);
        else
          algo.read_restart(restart);
      }
      else if (fdyn.sublist("TURBULENT INFLOW").get<bool>("TURBULENTINFLOW"))
      {
        FOUR_C_THROW(
            "Turbulent inflow generation for passive scalar transport should be performed as fluid "
            "problem!");
      }

      // solve the whole scalar transport problem
      algo.time_loop();

      // summarize the performance measurements
      Teuchos::TimeMonitor::summarize();

      // perform the result test
      algo.test_results();

      break;
    }
    default:
    {
      FOUR_C_THROW("unknown velocity field type for transport of passive scalar");
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
