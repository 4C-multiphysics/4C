// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_dyn.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_porofluid_pressure_based_algorithm.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*-------------------------------------------------------------------------------*
 | Main control routine for poro fluid multiphase problems           vuong 08/16 |
 *-------------------------------------------------------------------------------*/
void porofluid_pressure_based_dyn(int restart)
{
  // define the discretization names
  const std::string fluid_disname = "porofluid";
  const std::string struct_disname = "structure";
  const std::string artery_disname = "artery";

  // access the communicator
  MPI_Comm comm = Global::Problem::instance()->get_dis(fluid_disname)->get_comm();

  // access the problem
  Global::Problem* problem = Global::Problem::instance();

  // print problem type
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    std::cout << "###################################################" << std::endl;
    std::cout << "# YOUR PROBLEM TYPE: " << Global::Problem::instance()->problem_name()
              << std::endl;
    std::cout << "###################################################" << std::endl;
  }

  // Parameter reading
  // access structural dynamic params list which will be possibly modified while creating the time
  // integrator
  const Teuchos::ParameterList& porodyn = problem->poro_fluid_multi_phase_dynamic_params();

  // get the solver number used for poro fluid solver
  const int linsolvernumber = porodyn.get<int>("LINEAR_SOLVER");

  // -------------------------------------------------------------------
  // access the discretization(s)
  // -------------------------------------------------------------------
  std::shared_ptr<Core::FE::Discretization> actdis = nullptr;
  actdis = Global::Problem::instance()->get_dis(fluid_disname);

  // possible interaction partners as seen from the artery elements
  // [artelegid; contelegid_1, ...contelegid_n]
  std::map<int, std::set<int>> nearby_ele_pairs;

  if (Global::Problem::instance()->does_exist_dis(artery_disname))
  {
    std::shared_ptr<Core::FE::Discretization> arterydis = nullptr;
    arterydis = Global::Problem::instance()->get_dis(artery_disname);
    // get the coupling method
    auto arterycoupl =
        Teuchos::getIntegralValue<Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
            porodyn.sublist("ARTERY COUPLING"), "ARTERY_COUPLING_METHOD");

    // lateral surface coupling active?
    const bool evaluate_on_lateral_surface =
        porodyn.sublist("ARTERY COUPLING").get<bool>("LATERAL_SURFACE_COUPLING");

    switch (arterycoupl)
    {
      case Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gpts:
      case Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mp:
      case Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::ntp:
      {
        actdis->fill_complete();
        nearby_ele_pairs = PoroPressureBased::extended_ghosting_artery_discretization(
            *actdis, arterydis, evaluate_on_lateral_surface, arterycoupl);
        break;
      }
      default:
      {
        break;
      }
    }
  }

  // -------------------------------------------------------------------
  // assign dof set for solid pressures
  // -------------------------------------------------------------------
  std::shared_ptr<Core::DOFSets::DofSetInterface> dofsetaux =
      std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(1, 0, 0, false);
  const int nds_solidpressure = actdis->add_dof_set(dofsetaux);

  // -------------------------------------------------------------------
  // set degrees of freedom in the discretization
  // -------------------------------------------------------------------
  actdis->fill_complete();

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  std::shared_ptr<Core::IO::DiscretizationWriter> output = actdis->writer();
  output->write_mesh(0, 0.0);

  // -------------------------------------------------------------------
  // algorithm construction depending on
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------
  auto timintscheme =
      Teuchos::getIntegralValue<PoroPressureBased::TimeIntegrationScheme>(porodyn, "TIMEINTEGR");

  // build poro fluid time integrator
  std::shared_ptr<Adapter::PoroFluidMultiphase> algo = PoroPressureBased::create_algorithm(
      timintscheme, actdis, linsolvernumber, porodyn, porodyn, output);

  // initialize
  algo->init(false,        // eulerian formulation
      -1,                  //  no displacements
      -1,                  // no velocities
      nds_solidpressure,   // dof set for post processing solid pressure
      -1,                  // no scalar field
      &nearby_ele_pairs);  // possible interaction pairs

  // read the restart information, set vectors and variables
  if (restart) algo->read_restart(restart);

  // assign poro material for evaluation of porosity
  // note: to be done after potential restart, as in read_restart()
  //       the secondary material is destroyed
  PoroPressureBased::setup_material(comm, struct_disname, fluid_disname);

  // 4.- Run of the actual problem.
  algo->time_loop();

  // 4.3.- Summarize the performance measurements
  Teuchos::TimeMonitor::summarize();

  // perform the result test if required
  problem->add_field_test(algo->create_field_test());
  problem->test_all(comm);

  return;

}  // poromultiphase_dyn

FOUR_C_NAMESPACE_CLOSE
