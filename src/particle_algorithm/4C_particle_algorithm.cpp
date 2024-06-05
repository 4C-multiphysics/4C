/*---------------------------------------------------------------------------*/
/*! \file
\brief algorithm to control particle simulations
\level 1
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_particle_algorithm.hpp"

#include "4C_inpar_particle.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_particle_algorithm_gravity.hpp"
#include "4C_particle_algorithm_initial_field.hpp"
#include "4C_particle_algorithm_input_generator.hpp"
#include "4C_particle_algorithm_result_test.hpp"
#include "4C_particle_algorithm_timint.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_algorithm_viscous_damping.hpp"
#include "4C_particle_engine.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_interaction_base.hpp"
#include "4C_particle_interaction_dem.hpp"
#include "4C_particle_interaction_sph.hpp"
#include "4C_particle_rigidbody.hpp"
#include "4C_particle_rigidbody_result_test.hpp"
#include "4C_particle_wall.hpp"
#include "4C_particle_wall_result_test.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_result_test.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEALGORITHM::ParticleAlgorithm::ParticleAlgorithm(
    const Epetra_Comm& comm, const Teuchos::ParameterList& params)
    : AlgorithmBase(comm, params),
      myrank_(comm.MyPID()),
      params_(params),
      numparticlesafterlastloadbalance_(0),
      transferevery_(CORE::UTILS::IntegralValue<int>(params_, "TRANSFER_EVERY")),
      writeresultsevery_(params.get<int>("RESULTSEVRY")),
      writerestartevery_(params.get<int>("RESTARTEVRY")),
      writeresultsthisstep_(true),
      writerestartthisstep_(false),
      isrestarted_(false)
{
  // empty constructor
}

PARTICLEALGORITHM::ParticleAlgorithm::~ParticleAlgorithm() = default;

void PARTICLEALGORITHM::ParticleAlgorithm::Init(
    std::vector<PARTICLEENGINE::ParticleObjShrdPtr>& initialparticles)
{
  // init particle engine
  init_particle_engine();

  // init particle wall handler
  init_particle_wall();

  // init rigid body handler
  init_particle_rigid_body();

  // init particle time integration
  init_particle_time_integration();

  // init particle interaction handler
  init_particle_interaction();

  // init particle gravity handler
  init_particle_gravity();

  // init viscous damping handler
  init_viscous_damping();

  // set initial particles to vector of particles to be distributed
  particlestodistribute_ = initialparticles;

  // clear vector of initial particles in global problem
  initialparticles.clear();
}

void PARTICLEALGORITHM::ParticleAlgorithm::Setup()
{
  // generate initial particles
  if (not isrestarted_) generate_initial_particles();

  // determine all particle types
  determine_particle_types();

  // determine particle states of all particle types
  determine_particle_states_of_particle_types();

  // setup particle engine
  particleengine_->Setup(particlestatestotypes_);

  // setup wall handler
  if (particlewall_) particlewall_->Setup(particleengine_, Time());

  // setup rigid body handler
  if (particlerigidbody_) particlerigidbody_->Setup(particleengine_);

  // setup particle time integration
  particletimint_->Setup(particleengine_, particlerigidbody_);

  // setup particle interaction handler
  if (particleinteraction_) particleinteraction_->Setup(particleengine_, particlewall_);

  // setup gravity handler
  if (particlegravity_) particlegravity_->Setup();

  // setup viscous damping handler
  if (viscousdamping_) viscousdamping_->Setup(particleengine_);

  // setup initial particles
  setup_initial_particles();

  // setup initial rigid bodies
  if (particlerigidbody_) setup_initial_rigid_bodies();

  // distribute load among processors
  distribute_load_among_procs();

  // ghost particles on other processors
  particleengine_->GhostParticles();

  // build global id to local index map
  particleengine_->build_global_id_to_local_index_map();

  // build potential neighbor relation
  if (particleinteraction_) build_potential_neighbor_relation();

  // setup initial states
  if (not isrestarted_) setup_initial_states();

  // write initial output
  if (not isrestarted_) WriteOutput();
}

void PARTICLEALGORITHM::ParticleAlgorithm::read_restart(const int restartstep)
{
  // clear vector of particles to be distributed
  particlestodistribute_.clear();

  // create discretization reader
  const std::shared_ptr<CORE::IO::DiscretizationReader> reader =
      particleengine_->BinDisReader(restartstep);

  // safety check
  if (restartstep != reader->ReadInt("step"))
    FOUR_C_THROW("time step on file not equal to given step!");

  // get restart time
  double restarttime = reader->ReadDouble("time");

  // read restart of particle engine
  particleengine_->read_restart(reader, particlestodistribute_);

  // read restart of rigid body handler
  if (particlerigidbody_) particlerigidbody_->read_restart(reader);

  // read restart of particle interaction handler
  if (particleinteraction_) particleinteraction_->read_restart(reader);

  // read restart of wall handler
  if (particlewall_) particlewall_->read_restart(restartstep);

  // set time and step after restart
  SetTimeStep(restarttime, restartstep);

  // set flag indicating restart to true
  isrestarted_ = true;

  // short screen output
  if (myrank_ == 0)
    CORE::IO::cout << "====== restart of the particle simulation from step " << restartstep
                   << CORE::IO::endl;
}

void PARTICLEALGORITHM::ParticleAlgorithm::Timeloop()
{
  // time loop
  while (NotFinished())
  {
    // counter and print header
    prepare_time_step();

    // pre evaluate time step
    pre_evaluate_time_step();

    // integrate time step
    IntegrateTimeStep();

    // post evaluate time step
    post_evaluate_time_step();

    // write output
    WriteOutput();

    // write restart information
    write_restart();
  }
}

void PARTICLEALGORITHM::ParticleAlgorithm::prepare_time_step(bool do_print_header)
{
  // increment time and step
  increment_time_and_step();

  // set current time
  set_current_time();

  // set current step size
  set_current_step_size();

  // print header
  if (do_print_header) print_header();

  // update result and restart control flags
  writeresultsthisstep_ = (writeresultsevery_ and (Step() % writeresultsevery_ == 0));
  writerestartthisstep_ = (writerestartevery_ and (Step() % writerestartevery_ == 0));

  // set current write result flag
  set_current_write_result_flag();
}

void PARTICLEALGORITHM::ParticleAlgorithm::pre_evaluate_time_step()
{
  // pre evaluate time step
  if (particleinteraction_) particleinteraction_->pre_evaluate_time_step();
}

void PARTICLEALGORITHM::ParticleAlgorithm::IntegrateTimeStep()
{
  // time integration scheme specific pre-interaction routine
  particletimint_->pre_interaction_routine();

  // update connectivity
  update_connectivity();

  // evaluate time step
  evaluate_time_step();

  // time integration scheme specific post-interaction routine
  particletimint_->post_interaction_routine();
}

void PARTICLEALGORITHM::ParticleAlgorithm::post_evaluate_time_step()
{
  // post evaluate time step
  std::vector<PARTICLEENGINE::ParticleTypeToType> particlesfromphasetophase;
  if (particleinteraction_)
    particleinteraction_->post_evaluate_time_step(particlesfromphasetophase);

  if (particlerigidbody_)
  {
    // have rigid body phase change
    if (particlerigidbody_->have_rigid_body_phase_change(particlesfromphasetophase))
    {
      // update connectivity
      update_connectivity();

      // evaluate rigid body phase change
      particlerigidbody_->evaluate_rigid_body_phase_change(particlesfromphasetophase);
    }
  }
}

void PARTICLEALGORITHM::ParticleAlgorithm::WriteOutput() const
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEALGORITHM::ParticleAlgorithm::WriteOutput");

  // write result step
  if (writeresultsthisstep_)
  {
    // write particle runtime output
    particleengine_->write_particle_runtime_output(Step(), Time());

    // write binning discretization output (debug feature)
    particleengine_->WriteBinDisOutput(Step(), Time());

    // write rigid body runtime output
    if (particlerigidbody_) particlerigidbody_->write_rigid_body_runtime_output(Step(), Time());

    // write interaction runtime output
    if (particleinteraction_)
      particleinteraction_->write_interaction_runtime_output(Step(), Time());

    // write wall runtime output
    if (particlewall_) particlewall_->write_wall_runtime_output(Step(), Time());
  }
}

void PARTICLEALGORITHM::ParticleAlgorithm::write_restart() const
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEALGORITHM::ParticleAlgorithm::write_restart");

  // write restart step
  if (writerestartthisstep_)
  {
    // write restart of particle engine
    particleengine_->write_restart(Step(), Time());

    // write restart of rigid body handler
    if (particlerigidbody_) particlerigidbody_->write_restart();

    // write restart of particle interaction handler
    if (particleinteraction_) particleinteraction_->write_restart();

    // write restart of wall handler
    if (particlewall_) particlewall_->write_restart(Step(), Time());

    // short screen output
    if (myrank_ == 0)
      CORE::IO::cout(CORE::IO::verbose)
          << "====== restart of the particle simulation written in step " << Step()
          << CORE::IO::endl;
  }
}

std::vector<std::shared_ptr<CORE::UTILS::ResultTest>>
PARTICLEALGORITHM::ParticleAlgorithm::CreateResultTests()
{
  // build global id to local index map
  particleengine_->build_global_id_to_local_index_map();

  // particle field specific result test objects
  std::vector<std::shared_ptr<CORE::UTILS::ResultTest>> allresulttests(0);

  // particle result test
  {
    // create and init particle result test
    std::shared_ptr<PARTICLEALGORITHM::ParticleResultTest> particleresulttest =
        std::make_shared<PARTICLEALGORITHM::ParticleResultTest>();
    particleresulttest->Init();

    // setup particle result test
    particleresulttest->Setup(particleengine_);

    allresulttests.push_back(particleresulttest);
  }

  // wall result test
  if (particlewall_)
  {
    // create and init wall result test
    std::shared_ptr<PARTICLEWALL::WallResultTest> wallresulttest =
        std::make_shared<PARTICLEWALL::WallResultTest>();
    wallresulttest->Init();

    // setup wall result test
    wallresulttest->Setup(particlewall_);

    allresulttests.push_back(wallresulttest);
  }

  if (particlerigidbody_)
  {
    // create and init rigid body result test
    std::shared_ptr<PARTICLERIGIDBODY::RigidBodyResultTest> rigidbodyresulttest =
        std::make_shared<PARTICLERIGIDBODY::RigidBodyResultTest>();
    rigidbodyresulttest->Init();

    // setup rigid body result test
    rigidbodyresulttest->Setup(particlerigidbody_);

    allresulttests.push_back(rigidbodyresulttest);
  }

  return allresulttests;
}

void PARTICLEALGORITHM::ParticleAlgorithm::init_particle_engine()
{
  // create and init particle engine
  particleengine_ = std::make_shared<PARTICLEENGINE::ParticleEngine>(Comm(), params_);
  particleengine_->Init();
}

void PARTICLEALGORITHM::ParticleAlgorithm::init_particle_wall()
{
  // get type of particle wall source
  INPAR::PARTICLE::ParticleWallSource particlewallsource =
      CORE::UTILS::IntegralValue<INPAR::PARTICLE::ParticleWallSource>(
          params_, "PARTICLE_WALL_SOURCE");

  // create particle wall handler
  switch (particlewallsource)
  {
    case INPAR::PARTICLE::NoParticleWall:
    {
      particlewall_ = std::shared_ptr<PARTICLEWALL::WallHandlerBase>(nullptr);
      break;
    }
    case INPAR::PARTICLE::DiscretCondition:
    {
      particlewall_ = std::make_shared<PARTICLEWALL::WallHandlerDiscretCondition>(Comm(), params_);
      break;
    }
    case INPAR::PARTICLE::BoundingBox:
    {
      particlewall_ = std::make_shared<PARTICLEWALL::WallHandlerBoundingBox>(Comm(), params_);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown type of particle wall source!");
      break;
    }
  }

  // init particle wall handler
  if (particlewall_) particlewall_->Init(particleengine_->GetBinningStrategy());
}

void PARTICLEALGORITHM::ParticleAlgorithm::init_particle_rigid_body()
{
  // create rigid body handler
  if (CORE::UTILS::IntegralValue<int>(params_, "RIGID_BODY_MOTION"))
    particlerigidbody_ = std::make_shared<PARTICLERIGIDBODY::RigidBodyHandler>(Comm(), params_);

  // init rigid body handler
  if (particlerigidbody_) particlerigidbody_->Init();
}

void PARTICLEALGORITHM::ParticleAlgorithm::init_particle_time_integration()
{
  // get particle time integration scheme
  INPAR::PARTICLE::DynamicType timinttype =
      CORE::UTILS::IntegralValue<INPAR::PARTICLE::DynamicType>(params_, "DYNAMICTYP");

  // create particle time integration
  switch (timinttype)
  {
    case INPAR::PARTICLE::dyna_semiimpliciteuler:
    {
      particletimint_ = std::unique_ptr<PARTICLEALGORITHM::TimIntSemiImplicitEuler>(
          new PARTICLEALGORITHM::TimIntSemiImplicitEuler(params_));
      break;
    }
    case INPAR::PARTICLE::dyna_velocityverlet:
    {
      particletimint_ = std::unique_ptr<PARTICLEALGORITHM::TimIntVelocityVerlet>(
          new PARTICLEALGORITHM::TimIntVelocityVerlet(params_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown particle time integration scheme!");
      break;
    }
  }

  // init particle time integration
  particletimint_->Init();
}

void PARTICLEALGORITHM::ParticleAlgorithm::init_particle_interaction()
{
  // get particle interaction type
  INPAR::PARTICLE::InteractionType interactiontype =
      CORE::UTILS::IntegralValue<INPAR::PARTICLE::InteractionType>(params_, "INTERACTION");

  // create particle interaction handler
  switch (interactiontype)
  {
    case INPAR::PARTICLE::interaction_none:
    {
      particleinteraction_ = std::unique_ptr<PARTICLEINTERACTION::ParticleInteractionBase>(nullptr);
      break;
    }
    case INPAR::PARTICLE::interaction_sph:
    {
      particleinteraction_ = std::unique_ptr<PARTICLEINTERACTION::ParticleInteractionSPH>(
          new PARTICLEINTERACTION::ParticleInteractionSPH(Comm(), params_));
      break;
    }
    case INPAR::PARTICLE::interaction_dem:
    {
      particleinteraction_ = std::unique_ptr<PARTICLEINTERACTION::ParticleInteractionDEM>(
          new PARTICLEINTERACTION::ParticleInteractionDEM(Comm(), params_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown particle interaction type!");
      break;
    }
  }

  // init particle interaction handler
  if (particleinteraction_) particleinteraction_->Init();
}

void PARTICLEALGORITHM::ParticleAlgorithm::init_particle_gravity()
{
  // init gravity acceleration vector
  std::vector<double> gravity;
  std::string value;
  std::istringstream gravitystream(
      Teuchos::getNumericStringParameter(params_, "GRAVITY_ACCELERATION"));

  while (gravitystream >> value) gravity.push_back(std::atof(value.c_str()));

  // safety check
  if (static_cast<int>(gravity.size()) != 3)
    FOUR_C_THROW("dimension (dim = %d) of gravity acceleration vector is wrong!",
        static_cast<int>(gravity.size()));

  // get magnitude of gravity
  double temp = 0.0;
  for (double g : gravity) temp += g * g;
  const double gravity_norm = std::sqrt(temp);

  // create particle gravity handler
  if (gravity_norm > 0.0)
    particlegravity_ = std::unique_ptr<PARTICLEALGORITHM::GravityHandler>(
        new PARTICLEALGORITHM::GravityHandler(params_));

  // init particle gravity handler
  if (particlegravity_) particlegravity_->Init(gravity);
}

void PARTICLEALGORITHM::ParticleAlgorithm::init_viscous_damping()
{
  // get viscous damping factor
  const double viscdampfac = params_.get<double>("VISCOUS_DAMPING");

  // create viscous damping handler
  if (viscdampfac > 0.0)
    viscousdamping_ = std::unique_ptr<PARTICLEALGORITHM::ViscousDampingHandler>(
        new PARTICLEALGORITHM::ViscousDampingHandler(viscdampfac));

  // init viscous damping handler
  if (viscousdamping_) viscousdamping_->Init();
}

void PARTICLEALGORITHM::ParticleAlgorithm::generate_initial_particles()
{
  // create and init particle input generator
  std::unique_ptr<PARTICLEALGORITHM::InputGenerator> particleinputgenerator =
      std::unique_ptr<PARTICLEALGORITHM::InputGenerator>(
          new PARTICLEALGORITHM::InputGenerator(Comm(), params_));
  particleinputgenerator->Init();

  // generate particles
  particleinputgenerator->GenerateParticles(particlestodistribute_);
}

void PARTICLEALGORITHM::ParticleAlgorithm::determine_particle_types()
{
  // init map relating particle types to dynamic load balance factor
  std::map<PARTICLEENGINE::TypeEnum, double> typetodynloadbal;

  // read parameters relating particle types to values
  PARTICLEALGORITHM::UTILS::ReadParamsTypesRelatedToValues(
      params_, "PHASE_TO_DYNLOADBALFAC", typetodynloadbal);

  // insert into map of particle types and corresponding states with empty set
  for (auto& typeIt : typetodynloadbal)
    particlestatestotypes_.insert(
        std::make_pair(typeIt.first, std::set<PARTICLEENGINE::StateEnum>()));

  // safety check
  for (auto& particle : particlestodistribute_)
    if (not particlestatestotypes_.count(particle->ReturnParticleType()))
      FOUR_C_THROW("particle type '%s' of initial particle not defined!",
          PARTICLEENGINE::EnumToTypeName(particle->ReturnParticleType()).c_str());
}

void PARTICLEALGORITHM::ParticleAlgorithm::determine_particle_states_of_particle_types()
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes_)
  {
    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    // insert default particle states
    particlestates.insert({PARTICLEENGINE::Position, PARTICLEENGINE::Velocity,
        PARTICLEENGINE::Acceleration, PARTICLEENGINE::LastTransferPosition});
  }

  // insert integration dependent states of all particle types
  particletimint_->insert_particle_states_of_particle_types(particlestatestotypes_);

  // insert interaction dependent states of all particle types
  if (particleinteraction_)
    particleinteraction_->insert_particle_states_of_particle_types(particlestatestotypes_);

  // insert wall handler dependent states of all particle types
  if (particlewall_)
    particlewall_->insert_particle_states_of_particle_types(particlestatestotypes_);

  // insert rigid body handler dependent states of all particle types
  if (particlerigidbody_)
    particlerigidbody_->insert_particle_states_of_particle_types(particlestatestotypes_);
}

void PARTICLEALGORITHM::ParticleAlgorithm::setup_initial_particles()
{
  // get unique global ids for all particles
  if (not isrestarted_)
    particleengine_->get_unique_global_ids_for_all_particles(particlestodistribute_);

  // erase particles outside bounding box
  particleengine_->erase_particles_outside_bounding_box(particlestodistribute_);

  // distribute particles to owning processor
  particleengine_->DistributeParticles(particlestodistribute_);

  // distribute interaction history
  if (particleinteraction_) particleinteraction_->distribute_interaction_history();
}

void PARTICLEALGORITHM::ParticleAlgorithm::setup_initial_rigid_bodies()
{
  // set initial affiliation pair data
  if (not isrestarted_) particlerigidbody_->set_initial_affiliation_pair_data();

  // set unique global ids for all rigid bodies
  if (not isrestarted_) particlerigidbody_->set_unique_global_ids_for_all_rigid_bodies();

  // allocate rigid body states
  if (not isrestarted_) particlerigidbody_->allocate_rigid_body_states();

  // distribute rigid body
  particlerigidbody_->DistributeRigidBody();
}

void PARTICLEALGORITHM::ParticleAlgorithm::setup_initial_states()
{
  // set initial states
  if (particleinteraction_) particleinteraction_->SetInitialStates();

  // initialize rigid body mass quantities and orientation
  if (particlerigidbody_)
    particlerigidbody_->initialize_rigid_body_mass_quantities_and_orientation();

  // set initial conditions
  set_initial_conditions();

  // time integration scheme specific initialization routine
  particletimint_->SetInitialStates();

  // evaluate consistent initial states
  {
    // pre evaluate time step
    pre_evaluate_time_step();

    // update connectivity
    update_connectivity();

    // evaluate time step
    evaluate_time_step();

    // post evaluate time step
    post_evaluate_time_step();
  }
}

void PARTICLEALGORITHM::ParticleAlgorithm::update_connectivity()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEALGORITHM::ParticleAlgorithm::update_connectivity");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check number of unique global ids
  particleengine_->check_number_of_unique_global_ids();
#endif

  // check particle interaction distance concerning bin size
  if (particleinteraction_)
    particleinteraction_->check_particle_interaction_distance_concerning_bin_size();

  // check that wall nodes are located in bounding box
  if (particlewall_) particlewall_->check_wall_nodes_located_in_bounding_box();

  if (check_load_transfer_needed())
  {
    // transfer load between processors
    transfer_load_between_procs();

    // distribute load among processors
    if (check_load_redistribution_needed()) distribute_load_among_procs();

    // ghost particles on other processors
    particleengine_->GhostParticles();

    // build global id to local index map
    particleengine_->build_global_id_to_local_index_map();

    // build potential neighbor relation
    if (particleinteraction_) build_potential_neighbor_relation();
  }
  else
  {
    // refresh particles being ghosted on other processors
    particleengine_->RefreshParticles();
  }
}

bool PARTICLEALGORITHM::ParticleAlgorithm::check_load_transfer_needed()
{
  bool transferload = transferevery_ or writeresultsthisstep_ or writerestartthisstep_;

  // check max position increment
  transferload |= check_max_position_increment();

  // check for valid particle connectivity
  transferload |= (not particleengine_->have_valid_particle_connectivity());

  // check for valid particle neighbors
  if (particleinteraction_) transferload |= (not particleengine_->have_valid_particle_neighbors());

  // check for valid wall neighbors
  if (particleinteraction_ and particlewall_)
    transferload |= (not particlewall_->have_valid_wall_neighbors());

  return transferload;
}

bool PARTICLEALGORITHM::ParticleAlgorithm::check_max_position_increment()
{
  // get maximum particle interaction distance
  double allprocmaxinteractiondistance = 0.0;
  if (particleinteraction_)
  {
    double maxinteractiondistance = particleinteraction_->max_interaction_distance();
    Comm().MaxAll(&maxinteractiondistance, &allprocmaxinteractiondistance, 1);
  }

  // get max particle position increment since last transfer
  double maxparticlepositionincrement = 0.0;
  get_max_particle_position_increment(maxparticlepositionincrement);

  // get max wall position increment since last transfer
  double maxwallpositionincrement = 0.0;
  if (particlewall_) particlewall_->get_max_wall_position_increment(maxwallpositionincrement);

  // get max overall position increment since last transfer
  const double maxpositionincrement =
      std::max(maxparticlepositionincrement, maxwallpositionincrement);

  // get allowed position increment
  const double allowedpositionincrement =
      0.5 * (particleengine_->MinBinSize() - allprocmaxinteractiondistance);

  // check if a particle transfer is needed based on a worst case scenario:
  // two particles approach each other with maximum position increment in one spatial dimension
  return (maxpositionincrement > allowedpositionincrement);
}

void PARTICLEALGORITHM::ParticleAlgorithm::get_max_particle_position_increment(
    double& allprocmaxpositionincrement)
{
  // maximum position increment since last particle transfer
  double maxpositionincrement = 0.0;

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengine_->get_particle_container_bundle();

  // iterate over particle types
  for (auto& typeEnum : particlecontainerbundle->GetParticleTypes())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(typeEnum, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->ParticlesStored();

    // no owned particles of current particle type
    if (particlestored == 0) continue;

    // get particle state dimension
    int statedim = container->GetStateDim(PARTICLEENGINE::Position);

    // position increment of particle
    double positionincrement[3];

    // iterate over owned particles of current type
    for (int i = 0; i < particlestored; ++i)
    {
      // get pointer to particle states
      const double* pos = container->GetPtrToState(PARTICLEENGINE::Position, i);
      const double* lasttransferpos =
          container->GetPtrToState(PARTICLEENGINE::LastTransferPosition, i);

      // position increment of particle considering periodic boundaries
      particleengine_->distance_between_particles(pos, lasttransferpos, positionincrement);

      // iterate over spatial dimension
      for (int dim = 0; dim < statedim; ++dim)
        maxpositionincrement = std::max(maxpositionincrement, std::abs(positionincrement[dim]));
    }
  }

  // bin size safety check
  if (maxpositionincrement > particleengine_->MinBinSize())
    FOUR_C_THROW("a particle traveled more than one bin on this processor!");

  // get maximum particle position increment on all processors
  Comm().MaxAll(&maxpositionincrement, &allprocmaxpositionincrement, 1);
}

void PARTICLEALGORITHM::ParticleAlgorithm::transfer_load_between_procs()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEALGORITHM::ParticleAlgorithm::transfer_load_between_procs");

  // transfer particles to new bins and processors
  particleengine_->TransferParticles();

  // transfer wall elements and nodes
  if (particlewall_) particlewall_->transfer_wall_elements_and_nodes();

  // communicate rigid body
  if (particlerigidbody_) particlerigidbody_->communicate_rigid_body();

  // communicate interaction history
  if (particleinteraction_) particleinteraction_->communicate_interaction_history();

  // short screen output
  if (myrank_ == 0)
    CORE::IO::cout(CORE::IO::verbose) << "transfer load in step " << Step() << CORE::IO::endl;
}

bool PARTICLEALGORITHM::ParticleAlgorithm::check_load_redistribution_needed()
{
  bool redistributeload = writerestartthisstep_;

  // percentage limit
  const double percentagelimit = 0.1;

  // get number of particles on this processor
  int numberofparticles = particleengine_->get_number_of_particles();

  // percentage change of particles on this processor
  double percentagechange = 0.0;
  if (numparticlesafterlastloadbalance_ > 0)
    percentagechange =
        std::abs(static_cast<double>(numberofparticles - numparticlesafterlastloadbalance_) /
                 numparticlesafterlastloadbalance_);

  // get maximum percentage change of particles
  double maxpercentagechange = 0.0;
  Comm().MaxAll(&percentagechange, &maxpercentagechange, 1);

  // criterion for load redistribution based on maximum percentage change of the number of particles
  redistributeload |= (maxpercentagechange > percentagelimit);

  return redistributeload;
}

void PARTICLEALGORITHM::ParticleAlgorithm::distribute_load_among_procs()
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEALGORITHM::ParticleAlgorithm::distribute_load_among_procs");

  // dynamic load balancing
  particleengine_->dynamic_load_balancing();

  // get number of particles on this processor
  numparticlesafterlastloadbalance_ = particleengine_->get_number_of_particles();

  if (particlewall_)
  {
    // update bin row and column map
    particlewall_->update_bin_row_and_col_map(
        particleengine_->GetBinRowMap(), particleengine_->GetBinColMap());

    // distribute wall elements and nodes
    particlewall_->distribute_wall_elements_and_nodes();
  }

  // communicate rigid body
  if (particlerigidbody_) particlerigidbody_->communicate_rigid_body();

  // communicate interaction history
  if (particleinteraction_) particleinteraction_->communicate_interaction_history();

  // short screen output
  if (myrank_ == 0)
    CORE::IO::cout(CORE::IO::verbose) << "distribute load in step " << Step() << CORE::IO::endl;
}

void PARTICLEALGORITHM::ParticleAlgorithm::build_potential_neighbor_relation()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PARTICLEALGORITHM::ParticleAlgorithm::build_potential_neighbor_relation");

  // build particle to particle neighbors
  particleengine_->build_particle_to_particle_neighbors();

  if (particlewall_)
  {
    // relate bins to column wall elements
    particlewall_->relate_bins_to_col_wall_eles();

    // build particle to wall neighbors
    particlewall_->build_particle_to_wall_neighbors(particleengine_->GetParticlesToBins());
  }
}

void PARTICLEALGORITHM::ParticleAlgorithm::set_initial_conditions()
{
  // create and init particle initial field handler
  std::unique_ptr<PARTICLEALGORITHM::InitialFieldHandler> initialfield =
      std::unique_ptr<PARTICLEALGORITHM::InitialFieldHandler>(
          new PARTICLEALGORITHM::InitialFieldHandler(params_));
  initialfield->Init();

  // setup particle initial field handler
  initialfield->Setup(particleengine_);

  // set initial fields
  initialfield->SetInitialFields();

  // set rigid body initial conditions
  if (particlerigidbody_) particlerigidbody_->set_initial_conditions();
}

void PARTICLEALGORITHM::ParticleAlgorithm::set_current_time()
{
  // set current time in particle time integration
  particletimint_->set_current_time(Time());

  // set current time in particle interaction
  if (particleinteraction_) particleinteraction_->set_current_time(Time());
}

void PARTICLEALGORITHM::ParticleAlgorithm::set_current_step_size()
{
  // set current step size in particle interaction
  if (particleinteraction_) particleinteraction_->set_current_step_size(Dt());
}

void PARTICLEALGORITHM::ParticleAlgorithm::set_current_write_result_flag()
{
  // set current write result flag in particle interaction
  if (particleinteraction_)
    particleinteraction_->set_current_write_result_flag(writeresultsthisstep_);
}

void PARTICLEALGORITHM::ParticleAlgorithm::evaluate_time_step()
{
  // clear forces and torques
  if (particlerigidbody_) particlerigidbody_->clear_forces_and_torques();

  // set gravity acceleration
  if (particlegravity_) set_gravity_acceleration();

  // evaluate particle interactions
  if (particleinteraction_) particleinteraction_->evaluate_interactions();

  // apply viscous damping contribution
  if (viscousdamping_) viscousdamping_->ApplyViscousDamping();

  // compute accelerations of rigid bodies
  if (particlerigidbody_ and particleinteraction_) particlerigidbody_->compute_accelerations();
}

void PARTICLEALGORITHM::ParticleAlgorithm::set_gravity_acceleration()
{
  std::vector<double> scaled_gravity(3);

  // get gravity acceleration
  particlegravity_->get_gravity_acceleration(Time(), scaled_gravity);

  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengine_->get_particle_container_bundle();

  // iterate over particle types
  for (auto& typeEnum : particlecontainerbundle->GetParticleTypes())
  {
    // gravity is not set for boundary or rigid particles
    if (typeEnum == PARTICLEENGINE::BoundaryPhase or typeEnum == PARTICLEENGINE::RigidPhase)
      continue;

    // gravity is not set for open boundary particles
    if (typeEnum == PARTICLEENGINE::DirichletPhase or typeEnum == PARTICLEENGINE::NeumannPhase)
      continue;

    // set gravity acceleration for all particles of current type
    particlecontainerbundle->set_state_specific_container(
        scaled_gravity, PARTICLEENGINE::Acceleration, typeEnum);
  }

  // add gravity acceleration
  if (particlerigidbody_) particlerigidbody_->add_gravity_acceleration(scaled_gravity);

  // set scaled gravity in particle interaction handler
  if (particleinteraction_) particleinteraction_->SetGravity(scaled_gravity);
}

FOUR_C_NAMESPACE_CLOSE
