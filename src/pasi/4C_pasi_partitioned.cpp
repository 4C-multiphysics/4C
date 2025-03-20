// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pasi_partitioned.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_pasiwrapper.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_particle_algorithm.hpp"
#include "4C_particle_wall_datastate.hpp"
#include "4C_particle_wall_interface.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PaSI::PartitionedAlgo::PartitionedAlgo(MPI_Comm comm, const Teuchos::ParameterList& params)
    : AlgorithmBase(comm, params), isinit_(false), issetup_(false)
{
  // empty constructor
}

void PaSI::PartitionedAlgo::init()
{
  // reset setup flag
  set_is_setup(false);

  // init structure field
  init_structure_field();

  // init particle algorithm
  init_particle_algorithm();

  // set communication object at the interface
  interface_ = structurefield_->interface();

  // construct interface states
  intfdispnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
  intfvelnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);
  intfaccnp_ = Core::LinAlg::create_vector(*interface_->pasi_cond_map(), true);

  // set init flag
  set_is_init(true);
}

void PaSI::PartitionedAlgo::setup()
{
  // check correct initialization
  check_is_init();

  // setup particle algorithm
  particlealgorithm_->setup();

  // write initial output
  structurefield_->output();

  // set setup flag
  set_is_setup(true);
}

void PaSI::PartitionedAlgo::post_init()
{
  // call post setup routine of the structure field
  structurefield_->post_setup();
}


void PaSI::PartitionedAlgo::read_restart(int restartstep)
{
  // read restart information for structure field
  structurefield_->read_restart(restartstep);

  // read restart information for particle algorithm
  particlealgorithm_->read_restart(restartstep);

  // set time and step after restart
  set_time_step(structurefield_->time_old(), restartstep);

  // extract interface states
  extract_interface_states();

  // set interface states
  set_interface_states(intfdispnp_, intfvelnp_, intfaccnp_);
}

void PaSI::PartitionedAlgo::test_results(MPI_Comm comm)
{
  // get instance of global problem
  Global::Problem* problem = Global::Problem::instance();

  // add structure field specific result test object
  problem->add_field_test(structurefield_->create_field_test());

  // create particle field specific result test objects
  std::vector<std::shared_ptr<Core::Utils::ResultTest>> allresulttests =
      particlealgorithm_->create_result_tests();

  // add particle field specific result test objects
  for (auto& resulttest : allresulttests)
    if (resulttest) problem->add_field_test(resulttest);

  // perform all tests
  problem->test_all(comm);
}

void PaSI::PartitionedAlgo::prepare_time_step(bool printheader)
{
  // increment time and step
  increment_time_and_step();

  // print header
  if (printheader) print_header();

  // prepare time step of structure field and particle algorithm
  structurefield_->prepare_time_step();
  particlealgorithm_->prepare_time_step(false);
}

void PaSI::PartitionedAlgo::pre_evaluate_time_step()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PartitionedAlgo::pre_evaluate_time_step");

  // pre evaluate time step
  particlealgorithm_->pre_evaluate_time_step();
}

void PaSI::PartitionedAlgo::struct_step()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PartitionedAlgo::StructStep");

  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
    printf("-------------------- STRUCTURE SOLVER --------------------\n");

  // integrate structural time step
  structurefield_->solve();
}

void PaSI::PartitionedAlgo::particle_step()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PartitionedAlgo::particle_step");

  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and print_screen_every() and
      (step() % print_screen_every() == 0))
    printf("-------------------- PARTICLE SOLVER ---------------------\n");

  // integrate time step
  particlealgorithm_->integrate_time_step();
}

void PaSI::PartitionedAlgo::post_evaluate_time_step()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PartitionedAlgo::post_evaluate_time_step");

  // post evaluate time step
  particlealgorithm_->post_evaluate_time_step();
}

void PaSI::PartitionedAlgo::extract_interface_states()
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PartitionedAlgo::extract_interface_states");

  intfdispnp_ = interface_->extract_pasi_cond_vector(*structurefield_->dispnp());
  intfvelnp_ = interface_->extract_pasi_cond_vector(*structurefield_->velnp());
  intfaccnp_ = interface_->extract_pasi_cond_vector(*structurefield_->accnp());
}

void PaSI::PartitionedAlgo::set_interface_states(
    std::shared_ptr<const Core::LinAlg::Vector<double>> intfdispnp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> intfvelnp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> intfaccnp)
{
  TEUCHOS_FUNC_TIME_MONITOR("PaSI::PartitionedAlgo::set_interface_states");

  // get interface to particle wall handler
  std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface =
      particlealgorithm_->get_particle_wall_handler_interface();

  // get wall data state container
  std::shared_ptr<PARTICLEWALL::WallDataState> walldatastate =
      particlewallinterface->get_wall_data_state();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (walldatastate->get_disp_row() == nullptr or walldatastate->get_disp_col() == nullptr)
    FOUR_C_THROW("wall displacements not initialized!");
  if (walldatastate->get_vel_col() == nullptr) FOUR_C_THROW("wall velocities not initialized!");
  if (walldatastate->get_acc_col() == nullptr) FOUR_C_THROW("wall accelerations not initialized!");
#endif

  // export displacement, velocity and acceleration states
  Core::LinAlg::export_to(*intfdispnp, *walldatastate->get_disp_col());
  Core::LinAlg::export_to(*intfvelnp, *walldatastate->get_vel_col());
  Core::LinAlg::export_to(*intfaccnp, *walldatastate->get_acc_col());

  // export column to row displacements (no communication)
  Core::LinAlg::export_to(*walldatastate->get_disp_col(), *walldatastate->get_disp_row());

  // print norm of interface displacement to the screen
  if (print_screen_every() and (step() % print_screen_every() == 0))
  {
    double normintfdisp(0.0);
    intfdispnp->norm_2(&normintfdisp);

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
      printf("--> Norm of interface displacement: %10.5E\n", normintfdisp);
  }
}

void PaSI::PartitionedAlgo::struct_output()
{
  // calculate stresses, strains, energies
  constexpr bool force_prepare = false;
  structurefield_->prepare_output(force_prepare);

  // update all single field solvers
  structurefield_->update();

  // write output
  structurefield_->output();
}

void PaSI::PartitionedAlgo::particle_output()
{
  // write output
  particlealgorithm_->write_output();

  // write restart information
  particlealgorithm_->write_restart();
}

void PaSI::PartitionedAlgo::init_structure_field()
{
  // get instance of global problem
  Global::Problem* problem = Global::Problem::instance();

  // get parameter list
  const Teuchos::ParameterList& params = problem->structural_dynamic_params();

  // access the structural discretization
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");

  // build structure
  if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(params, "INT_STRATEGY") ==
      Inpar::Solid::IntegrationStrategy::int_standard)
  {
    // create and init structure base algorithm
    struct_adapterbase_ptr_ = Adapter::build_structure_algorithm(params);
    struct_adapterbase_ptr_->init(params, const_cast<Teuchos::ParameterList&>(params), structdis);
  }
  else if (Teuchos::getIntegralValue<Inpar::Solid::IntegrationStrategy>(params, "INT_STRATEGY") ==
           Inpar::Solid::IntegrationStrategy::int_old)
    FOUR_C_THROW(
        "Old time integration not supported in particle structure interaction!\n"
        "Set parameter INT_STRATEGY to Standard in ---STRUCTURAL DYNAMIC section!");
  else
    FOUR_C_THROW(
        "Unknown time integration requested!\n"
        "Set parameter INT_STRATEGY to Standard in ---STRUCTURAL DYNAMIC section!");

  // build and register structure model evaluator
  build_structure_model_evaluator();
}

void PaSI::PartitionedAlgo::init_particle_algorithm()
{
  // get instance of global problem
  Global::Problem* problem = Global::Problem::instance();

  // get parameter list
  const Teuchos::ParameterList& params = problem->particle_params();

  // reference to vector of initial particles
  std::vector<PARTICLEENGINE::ParticleObjShrdPtr>& initialparticles = problem->particles();

  // create and init particle algorithm
  particlealgorithm_ = std::make_shared<PARTICLEALGORITHM::ParticleAlgorithm>(get_comm(), params);
  particlealgorithm_->init(initialparticles);
}

void PaSI::PartitionedAlgo::build_structure_model_evaluator()
{
  // if adapter base has not already been set up outside.
  if (not struct_adapterbase_ptr_->is_setup())
  {
    // build and register pasi model evaluator
    std::shared_ptr<Solid::ModelEvaluator::Generic> pasi_model_ptr =
        std::make_shared<Solid::ModelEvaluator::PartitionedPASI>();

    struct_adapterbase_ptr_->register_model_evaluator("Partitioned Coupling Model", pasi_model_ptr);

    // call setup() on structure base algorithm (wrapper is created inside)
    struct_adapterbase_ptr_->setup();

    // get wrapper and cast it to specific type
    structurefield_ = std::dynamic_pointer_cast<Adapter::PASIStructureWrapper>(
        struct_adapterbase_ptr_->structure_field());

    if (structurefield_ == nullptr)
      FOUR_C_THROW("No valid pointer to Adapter::PASIStructureWrapper set!");

    // set pointer to model evaluator in PASIStructureWrapper
    structurefield_->set_model_evaluator_ptr(
        std::dynamic_pointer_cast<Solid::ModelEvaluator::PartitionedPASI>(pasi_model_ptr));
  }
}

FOUR_C_NAMESPACE_CLOSE
