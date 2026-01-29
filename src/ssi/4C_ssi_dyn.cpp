// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_dyn.hpp"

#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_ssi_monolithic.hpp"
#include "4C_ssi_monolithic_meshtying_strategy.hpp"
#include "4C_ssi_partitioned_1wc.hpp"
#include "4C_ssi_partitioned_2wc.hpp"
#include "4C_ssi_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ssi_drt()
{
  // 1.- Initialization
  std::shared_ptr<SSI::SSIBase> ssi = nullptr;
  Global::Problem* problem = Global::Problem::instance();
  MPI_Comm comm = problem->get_dis("structure")->get_comm();

  {
    TEUCHOS_FUNC_TIME_MONITOR("SSI: setup");

    // 2.- Parameter reading
    auto& ssiparams = const_cast<Teuchos::ParameterList&>(problem->ssi_control_params());
    // access scatra params list
    auto& scatradyn =
        const_cast<Teuchos::ParameterList&>(problem->scalar_transport_dynamic_params());
    // access the structural dynamic params list which will be possibly modified while creating the
    // time integrator
    auto& sdyn = const_cast<Teuchos::ParameterList&>(
        Global::Problem::instance()->structural_dynamic_params());

    FOUR_C_ASSERT_ALWAYS(sdyn.get<Inpar::Solid::IntegrationStrategy>("INT_STRATEGY") ==
                             Inpar::Solid::IntegrationStrategy::int_standard,
        "Only the new solid time integration is supported for SSI problems. Set `INT_STRATEGY` to "
        "`Standard`!");

    // introduce additional scatra field on manifold?
    const bool is_scatra_manifold = ssiparams.sublist("MANIFOLD").get<bool>("ADD_MANIFOLD");

    // Modification of time parameter list
    SSI::Utils::change_time_parameter(comm, ssiparams, scatradyn, sdyn);

    const auto coupling =
        Teuchos::getIntegralValue<SSI::SolutionSchemeOverFields>(ssiparams, "COUPALGO");


    // 3.- Creation of Structure + Scalar_Transport problem.
    // by default we employ an ale formulation for our scatra
    bool isale = true;

    // 3.1 choose algorithm depending on solution type
    switch (coupling)
    {
      case SSI::SolutionSchemeOverFields::ssi_OneWay_ScatraToSolid:
      {
        ssi = std::make_shared<SSI::SSIPart1WCScatraToSolid>(comm, ssiparams);
        isale = false;
      }
      break;
      case SSI::SolutionSchemeOverFields::ssi_OneWay_SolidToScatra:
        ssi = std::make_shared<SSI::SSIPart1WCSolidToScatra>(comm, ssiparams);
        break;
      case SSI::SolutionSchemeOverFields::ssi_IterStagg:
        ssi = std::make_shared<SSI::SSIPart2WC>(comm, ssiparams);
        break;
      case SSI::SolutionSchemeOverFields::ssi_IterStaggFixedRel_ScatraToSolid:
        ssi = std::make_shared<SSI::SSIPart2WCScatraToSolidRelax>(comm, ssiparams);
        break;
      case SSI::SolutionSchemeOverFields::ssi_IterStaggFixedRel_SolidToScatra:
        ssi = std::make_shared<SSI::SSIPart2WCSolidToScatraRelax>(comm, ssiparams);
        break;
      case SSI::SolutionSchemeOverFields::ssi_IterStaggAitken_ScatraToSolid:
        ssi = std::make_shared<SSI::SSIPart2WCScatraToSolidRelaxAitken>(comm, ssiparams);
        break;
      case SSI::SolutionSchemeOverFields::ssi_IterStaggAitken_SolidToScatra:
        ssi = std::make_shared<SSI::SSIPart2WCSolidToScatraRelaxAitken>(comm, ssiparams);
        break;
      case SSI::SolutionSchemeOverFields::ssi_Monolithic:
        ssi = std::make_shared<SSI::SsiMono>(comm, ssiparams);
        break;
      default:
        FOUR_C_THROW("unknown coupling algorithm for SSI!");
    }

    // 3.1.1 initial fill_complete
    problem->get_dis("structure")->fill_complete();
    problem->get_dis("scatra")->fill_complete();
    if (is_scatra_manifold) problem->get_dis("scatra_manifold")->fill_complete();

    // 3.1.2 init the chosen ssi algorithm
    // Construct time integrators of subproblems inside.
    ssi->init(comm, ssiparams, scatradyn, sdyn, "structure", "scatra", isale);

    // now we can finally fill our discretizations
    // reinitialization of the structural elements is
    // vital for parallelization here!
    problem->get_dis("structure")->fill_complete();
    problem->get_dis("scatra")->fill_complete({
        .assign_degrees_of_freedom = true,
        .init_elements = false,
        .do_boundary_conditions = true,
    });
    if (is_scatra_manifold)
      problem->get_dis("scatra_manifold")
          ->fill_complete({
              .assign_degrees_of_freedom = true,
              .init_elements = false,
              .do_boundary_conditions = true,
          });

    Core::Rebalance::print_parallel_distribution(*problem->get_dis("structure"));
    Core::Rebalance::print_parallel_distribution(*problem->get_dis("scatra"));
    if (is_scatra_manifold)
      Core::Rebalance::print_parallel_distribution(*problem->get_dis("scatra_manifold"));

    // 3.1.4 Setup the coupled problem
    // now as we redistributed our discretizations we can construct all
    // objects relying on the parallel distribution
    ssi->setup();

    // 3.2 - Read restart if needed / Call post_setup of structure.(discretization called inside)
    if (ssi->is_restart())
    {
      ssi->read_restart(problem->restart());
    }
    else
    {
      // call post_setup for the structure field
      ssi->post_setup();
    }

    // 3.3 AFTER restart: reset input filename of the problem so that results from other runs can be
    // read
    bool flag_readscatra = ssiparams.get<bool>("SCATRA_FROM_RESTART_FILE");
    if (coupling == SSI::SolutionSchemeOverFields::ssi_OneWay_ScatraToSolid and flag_readscatra)
    {
      std::string filename = Teuchos::getNumericStringParameter(ssiparams, "SCATRA_FILENAME");
      auto inputscatra = std::make_shared<Core::IO::InputControl>(filename, comm);
      problem->set_input_control_file(std::move(inputscatra));
    }

    // 4.- Run of the actual problem.
    // 4.1.- Some setup needed for the subproblems.
    ssi->setup_system();
  }

  // 4.2.- Solve the whole problem
  ssi->timeloop();

  // 5. - perform the result test
  ssi->test_results(comm);
}

FOUR_C_NAMESPACE_CLOSE
