// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_cardiac_monodomain_dyn.hpp"

#include "4C_adapter_scatra_base_algorithm.hpp"
#include "4C_binstrategy.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
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
void scatra_cardiac_monodomain_dyn(int restart)
{
  // pointer to problem
  Global::Problem* problem = Global::Problem::instance();

  // access the communicator
  MPI_Comm comm = problem->get_dis("fluid")->get_comm();

  //  // print problem type
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "###################################################" << std::endl;
    std::cout << "# YOUR PROBLEM TYPE: " << Global::Problem::instance()->problem_name()
              << std::endl;
    std::cout << "###################################################" << std::endl;
  }


  // print CardiacMonodomain-Logo to screen
  if (Core::Communication::my_mpi_rank(comm) == 0) printheartlogo();

  // access the problem-specific parameter list
  const Teuchos::ParameterList& scatradyn = problem->scalar_transport_dynamic_params();

  // access the fluid discretization
  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis("fluid");
  // access the scatra discretization
  std::shared_ptr<Core::FE::Discretization> scatradis = problem->get_dis("scatra");

  // ensure that all dofs are assigned in the right order; this creates dof numbers with
  //       fluid dof < scatra dof
  fluiddis->fill_complete();
  scatradis->fill_complete();

  // set velocity field
  const auto veltype = Teuchos::getIntegralValue<ScaTra::VelocityField>(scatradyn, "VELOCITYFIELD");
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
            "no linear solver defined for SCALAR_TRANSPORT problem. Please set LINEAR_SOLVER in "
            "SCALAR TRANSPORT DYNAMIC to a valid number!");
      }

      // create instance of scalar transport basis algorithm (empty fluid discretization)
      Adapter::ScaTraBaseAlgorithm scatraonly(*Global::Problem::instance(), scatradyn, scatradyn,
          Global::Problem::instance()->solver_params(linsolvernumber));

      // add proxy of velocity related degrees of freedom to scatra discretization
      std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux =
          std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
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
      scatradis->fill_complete();



      // We have do use the binningstrategy and extended ghosting when we use p-adaptivity. This
      // guarantees that the elements at the boarder between the processor calculate correctly since
      // one face is shared with the neighboring element (which is owned by an other processor =
      // ghosted element) which again is sharing other faces with elements on other processors
      // (extended ghosted element)
      if (scatradyn.get<bool>("PADAPTIVITY"))
      {
        // redistribute discr. with help of binning strategy
        if (Core::Communication::num_mpi_ranks(scatradis->get_comm()) > 1)
        {
          // create vector of discr.
          std::vector<std::shared_ptr<Core::FE::Discretization>> dis;
          dis.push_back(scatradis);

          // binning strategy for parallel redistribution
          std::shared_ptr<Core::Binstrategy::BinningStrategy> binningstrategy;

          std::vector<std::shared_ptr<Core::LinAlg::Map>> stdelecolmap;
          std::vector<std::shared_ptr<Core::LinAlg::Map>> stdnodecolmap;

          // binning strategy is created and parallel redistribution is performed
          Teuchos::ParameterList binning_params =
              Global::Problem::instance()->binning_strategy_params();
          Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
              "spatial_approximation_type",
              Global::Problem::instance()->spatial_approximation_type(), binning_params);
          binningstrategy = std::make_shared<Core::Binstrategy::BinningStrategy>(binning_params,
              Global::Problem::instance()->output_control_file(), scatradis->get_comm(),
              Core::Communication::my_mpi_rank(scatradis->get_comm()), std::nullopt, nullptr,
              nullptr, dis);
          binningstrategy
              ->do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
                  dis, stdelecolmap, stdnodecolmap);
        }
      }

      // now we can call init() on the base algo.
      // time integrator is constructed and initialized inside
      scatraonly.init();

      // NOTE : At this point we may redistribute and/or
      //        ghost our discretizations at will.

      // now we must call setup()
      scatraonly.setup();

      // read the restart information, set vectors and variables
      if (restart) scatraonly.scatra_field()->read_restart(restart);

      // set initial velocity field
      // note: The order read_restart() before set_velocity_field_from_function() is important
      // here!! for time-dependent velocity fields, set_velocity_field_from_function() is
      // additionally called in each prepare_time_step()-call
      (scatraonly.scatra_field())->set_velocity_field_from_function();

      // enter time loop to solve problem with given convective velocity
      (scatraonly.scatra_field())->time_loop();

      // perform the result test if required
      Global::Problem::instance()->add_field_test(scatraonly.create_scatra_field_test());
      Global::Problem::instance()->test_all(comm);

      break;
    }
    case ScaTra::VelocityField::from_other_field:
    {
      FOUR_C_THROW(
          "Applying the velocity field from another field is not implemented for cardiac "
          "monodomain scalar transport problem");
    }
    default:
    {
      FOUR_C_THROW("unknown velocity field type for transport of passive scalar");
      break;
    }
  }

  return;

}  // end of scatra_cardiac_monodomain_dyn()


/*----------------------------------------------------------------------*/
// print ELCH-Module logo
/*----------------------------------------------------------------------*/
void printheartlogo()
{
  // more at http://www.ascii-art.de
  std::cout << "                                                         " << std::endl;
  std::cout << "               |  \\ \\ | |/ /                           " << std::endl;
  std::cout << "               |  |\\ `' ' /                             " << std::endl;
  std::cout << "               |  ;'aorta \\      / , pulmonary          " << std::endl;
  std::cout << "               | ;    _,   |    / / ,  arteries          " << std::endl;
  std::cout << "      superior | |   (  `-.;_,-' '-' ,                   " << std::endl;
  std::cout << "     vena cava | `,   `-._       _,-'_                   " << std::endl;
  std::cout
      << "               |,-`.    `.)    ,<_,-'_, pulmonary                     ______ _____    "
      << std::endl;
  std::cout
      << "              ,'    `.   /   ,'  `;-' _,  veins                      |  ____|  __ \\   "
      << std::endl;
  std::cout
      << "             ;        `./   /`,    \\-'                               | |__  | |__) |  "
      << std::endl;
  std::cout
      << "             | right   /   |  ;\\   |\\                                |  __| |  ___/   "
      << std::endl;
  std::cout
      << "             | atrium ;_,._|_,  `, ' \\                               | |____| |       "
      << std::endl;
  std::cout
      << "             |        \\    \\ `       `,                              |______|_|       "
      << std::endl;
  std::cout
      << "             `      __ `    \\   left  ;,                                              "
      << std::endl;
  std::cout
      << "              \\   ,'  `      \\,  ventricle                                            "
      << std::endl;
  std::cout << "               \\_(            ;,      ;;                " << std::endl;
  std::cout << "               |  \\           `;,     ;;                " << std::endl;
  std::cout << "      inferior |  |`.          `;;,   ;'                 " << std::endl;
  std::cout << "     vena cava |  |  `-.        ;;;;,;'                  " << std::endl;
  std::cout << "               |  |    |`-.._  ,;;;;;'                   " << std::endl;
  std::cout << "               |  |    |   | ``';;;'                     " << std::endl;
  std::cout << "                       aorta                             " << std::endl;
  std::cout << "                                                         " << std::endl;
}

FOUR_C_NAMESPACE_CLOSE
