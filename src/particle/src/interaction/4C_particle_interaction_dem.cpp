// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_dem_adhesion.hpp"
#include "4C_particle_interaction_dem_contact.hpp"
#include "4C_particle_interaction_dem_history_pairs.hpp"
#include "4C_particle_interaction_dem_neighbor_pairs.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_runtime_writer.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_interface.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::ParticleInteractionDEM::ParticleInteractionDEM(
    MPI_Comm comm, const Teuchos::ParameterList& params)
    : ParticleInteraction::ParticleInteractionBase(comm, params),
      params_dem_(params.sublist("DEM")),
      writeparticleenergy_(params_dem_.get<bool>("WRITE_PARTICLE_ENERGY"))
{
  // empty constructor
}

ParticleInteraction::ParticleInteractionDEM::~ParticleInteractionDEM() = default;

void ParticleInteraction::ParticleInteractionDEM::init()
{
  // call base class init
  ParticleInteractionBase::init();

  // init neighbor pair handler
  init_neighbor_pair_handler();

  // init history pair handler
  init_history_pair_handler();

  // init contact handler
  init_contact_handler();

  // init adhesion handler
  init_adhesion_handler();
}

void ParticleInteraction::ParticleInteractionDEM::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface)
{
  // call base class setup
  ParticleInteractionBase::setup(particleengineinterface, particlewallinterface);

  // setup neighbor pair handler
  neighborpairs_->setup(particleengineinterface, particlewallinterface);

  // setup history pair handler
  historypairs_->setup(particleengineinterface);

  // setup contact handler
  contact_->setup(particleengineinterface, particlewallinterface, particlematerial_,
      particleinteractionwriter_, neighborpairs_, historypairs_);

  // setup adhesion handler
  if (adhesion_)
    adhesion_->setup(particleengineinterface, particlewallinterface, particleinteractionwriter_,
        neighborpairs_, historypairs_, contact_->get_normal_contact_stiffness());

  // setup particle interaction writer
  setup_particle_interaction_writer();
}

void ParticleInteraction::ParticleInteractionDEM::write_restart() const
{
  // call base class function
  ParticleInteractionBase::write_restart();

  // write restart of history pair handler
  historypairs_->write_restart();
}

void ParticleInteraction::ParticleInteractionDEM::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader)
{
  // call base class function
  ParticleInteractionBase::read_restart(reader);

  // read restart of history pair handler
  historypairs_->read_restart(*reader);
}

void ParticleInteraction::ParticleInteractionDEM::insert_particle_states_of_particle_types(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    // insert states of regular phase particles
    particlestates.insert({PARTICLEENGINE::Force, PARTICLEENGINE::Mass, PARTICLEENGINE::Radius});
  }

  // states for contact evaluation scheme
  contact_->insert_particle_states_of_particle_types(particlestatestotypes);
}

void ParticleInteraction::ParticleInteractionDEM::set_initial_states()
{
  // set initial radius
  set_initial_radius();

  // set initial mass
  set_initial_mass();

  // set initial inertia
  set_initial_inertia();
}

void ParticleInteraction::ParticleInteractionDEM::pre_evaluate_time_step()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::ParticleInteractionDEM::pre_evaluate_time_step");
}

void ParticleInteraction::ParticleInteractionDEM::evaluate_interactions()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::ParticleInteractionDEM::evaluate_interactions");

  // clear force and moment states of particles
  clear_force_and_moment_states();

  // evaluate neighbor pairs
  neighborpairs_->evaluate_neighbor_pairs();

  // evaluate adhesion neighbor pairs
  if (adhesion_)
    neighborpairs_->evaluate_neighbor_pairs_adhesion(adhesion_->get_adhesion_distance());

  // check critical time step
  contact_->check_critical_time_step();

  // add contact contribution to force and moment field
  contact_->add_force_and_moment_contribution();

  // add adhesion contribution to force field
  if (adhesion_) adhesion_->add_force_contribution();

  // compute acceleration from force and moment
  compute_acceleration();

  // update history pairs
  historypairs_->update_history_pairs();
}

void ParticleInteraction::ParticleInteractionDEM::post_evaluate_time_step(
    std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase)
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::ParticleInteractionDEM::post_evaluate_time_step");

  // evaluate particle energy
  if (particleinteractionwriter_->get_current_write_result_flag() and writeparticleenergy_)
    evaluate_particle_energy();
}

double ParticleInteraction::ParticleInteractionDEM::max_interaction_distance() const
{
  // particle contact interaction distance
  double interactiondistance = 2.0 * max_particle_radius();

  // add adhesion distance
  if (adhesion_) interactiondistance += adhesion_->get_adhesion_distance();

  return interactiondistance;
}

void ParticleInteraction::ParticleInteractionDEM::distribute_interaction_history() const
{
  // distribute history pairs
  historypairs_->distribute_history_pairs();
}

void ParticleInteraction::ParticleInteractionDEM::communicate_interaction_history() const
{
  // communicate history pairs
  historypairs_->communicate_history_pairs();
}

void ParticleInteraction::ParticleInteractionDEM::set_current_step_size(
    const double currentstepsize)
{
  // call base class method
  ParticleInteractionBase::set_current_step_size(currentstepsize);

  // set current step size
  contact_->set_current_step_size(currentstepsize);
}

void ParticleInteraction::ParticleInteractionDEM::init_neighbor_pair_handler()
{
  // create neighbor pair handler
  neighborpairs_ = std::make_shared<ParticleInteraction::DEMNeighborPairs>();

  // init neighbor pair handler
  neighborpairs_->init();
}

void ParticleInteraction::ParticleInteractionDEM::init_history_pair_handler()
{
  // create history pair handler
  historypairs_ = std::make_shared<ParticleInteraction::DEMHistoryPairs>(comm_);

  // init history pair handler
  historypairs_->init();
}

void ParticleInteraction::ParticleInteractionDEM::init_contact_handler()
{
  // create contact handler
  contact_ = std::unique_ptr<ParticleInteraction::DEMContact>(
      new ParticleInteraction::DEMContact(params_dem_));

  // init contact handler
  contact_->init();
}

void ParticleInteraction::ParticleInteractionDEM::init_adhesion_handler()
{
  // get type of adhesion law
  auto adhesionlaw = Teuchos::getIntegralValue<PARTICLE::AdhesionLaw>(params_dem_, "ADHESIONLAW");

  // create adhesion handler
  if (adhesionlaw != PARTICLE::NoAdhesion)
    adhesion_ = std::unique_ptr<ParticleInteraction::DEMAdhesion>(
        new ParticleInteraction::DEMAdhesion(params_dem_));

  // init adhesion handler
  if (adhesion_) adhesion_->init();
}

void ParticleInteraction::ParticleInteractionDEM::setup_particle_interaction_writer()
{
  if (writeparticleenergy_)
  {
    // register specific runtime csv writer
    particleinteractionwriter_->register_specific_runtime_csv_writer("particle-energy");

    // get specific runtime csv writer
    Core::IO::RuntimeCsvWriter* runtime_csv_writer =
        particleinteractionwriter_->get_specific_runtime_csv_writer("particle-energy");

    // register all data vectors
    runtime_csv_writer->register_data_vector("kin_energy", 1, 10);
    runtime_csv_writer->register_data_vector("grav_pot_energy", 1, 10);
    runtime_csv_writer->register_data_vector("elast_pot_energy", 1, 10);
  }
}

void ParticleInteraction::ParticleInteractionDEM::set_initial_radius()
{
  // get allowed bounds for particle radius
  double r_min = params_dem_.get<double>("MIN_RADIUS");
  double r_max = params_dem_.get<double>("MAX_RADIUS");

  // safety checks
  if (r_min < 0.0) FOUR_C_THROW("negative minimum allowed particle radius!");
  if (not(r_max > 0.0)) FOUR_C_THROW("non-positive maximum allowed particle radius!");
  if (r_min > r_max)
    FOUR_C_THROW("minimum allowed particle radius larger than maximum allowed particle radius!");

  // get type of initial particle radius assignment
  auto radiusdistributiontype =
      Teuchos::getIntegralValue<PARTICLE::InitialRadiusAssignment>(params_dem_, "INITIAL_RADIUS");

  switch (radiusdistributiontype)
  {
    // particle radius from particle material
    case PARTICLE::RadiusFromParticleMaterial:
    {
      // iterate over particle types
      for (const auto& type_i : particlecontainerbundle_->get_particle_types())
      {
        // get container of owned particles of current particle type
        PARTICLEENGINE::ParticleContainer* container =
            particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

        // get number of particles stored in container
        const int particlestored = container->particles_stored();

        // no owned particles of current particle type
        if (particlestored <= 0) continue;

        // get material for current particle type
        const Mat::PAR::ParticleMaterialBase* material =
            particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

        // safety checks
        if (material->initRadius_ < r_min)
          FOUR_C_THROW("material particle radius smaller than minimum allowed particle radius!");

        if (material->initRadius_ > r_max)
          FOUR_C_THROW("material particle radius larger than maximum allowed particle radius!");

        // (initial) radius of current phase
        std::vector<double> initradius(1);
        initradius[0] = material->initRadius_;

        // set initial radius for all particles of current type
        container->set_state(initradius, PARTICLEENGINE::Radius);
      }

      break;
    }
    // particle radius from particle input
    case PARTICLE::RadiusFromParticleInput:
    {
      // note: particle radius set as read in from input file, only safety checks here

      // iterate over particle types
      for (const auto& type_i : particlecontainerbundle_->get_particle_types())
      {
        // get container of owned particles of current particle type
        PARTICLEENGINE::ParticleContainer* container =
            particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

        // get number of particles stored in container
        const int particlestored = container->particles_stored();

        // no owned particles of current particle type
        if (particlestored <= 0) continue;

        // safety checks
        if (container->get_min_value_of_state(PARTICLEENGINE::Radius) < r_min)
          FOUR_C_THROW("minimum particle radius smaller than minimum allowed particle radius!");

        if (container->get_max_value_of_state(PARTICLEENGINE::Radius) > r_max)
          FOUR_C_THROW("maximum particle radius larger than maximum allowed particle radius!");
      }

      break;
    }
    // normal or log-normal random particle radius distribution
    case PARTICLE::NormalRadiusDistribution:
    case PARTICLE::LogNormalRadiusDistribution:
    {
      // get sigma of random particle radius distribution
      auto sigma = params_dem_.get<std::optional<double>>("RADIUSDISTRIBUTION_SIGMA");

      // safety check
      if (!sigma.has_value())
        FOUR_C_THROW("RADIUSDISTRIBUTION_SIGMA is not set but required for a radius distribution.");

      // iterate over particle types
      for (const auto& type_i : particlecontainerbundle_->get_particle_types())
      {
        // get container of owned particles of current particle type
        PARTICLEENGINE::ParticleContainer* container =
            particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

        // get number of particles stored in container
        const int particlestored = container->particles_stored();

        // no owned particles of current particle type
        if (particlestored <= 0) continue;

        // get material for current particle type
        const Mat::PAR::ParticleMaterialBase* material =
            particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

        // get pointer to particle state
        double* radius = container->get_ptr_to_state(PARTICLEENGINE::Radius, 0);

        // determine mu of random particle radius distribution
        const double mu = (radiusdistributiontype == PARTICLE::NormalRadiusDistribution)
                              ? material->initRadius_
                              : std::log(material->initRadius_);

        // initialize random number generator
        Global::Problem::instance()->random()->set_mean_stddev(mu, *sigma);

        // iterate over particles stored in container
        for (int i = 0; i < particlestored; ++i)
        {
          // generate random value
          const double randomvalue = Global::Problem::instance()->random()->normal();

          // set normal or log-normal distributed random value for particle radius
          radius[i] = (radiusdistributiontype == PARTICLE::NormalRadiusDistribution)
                          ? randomvalue
                          : std::exp(randomvalue);

          // adjust radius to allowed bounds
          if (radius[i] > r_max)
            radius[i] = r_max;
          else if (radius[i] < r_min)
            radius[i] = r_min;
        }
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("invalid type of (random) particle radius distribution!");
      break;
    }
  }
}

void ParticleInteraction::ParticleInteractionDEM::set_initial_mass()
{
  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* material =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // get pointer to particle states
    const double* radius = container->get_ptr_to_state(PARTICLEENGINE::Radius, 0);
    double* mass = container->get_ptr_to_state(PARTICLEENGINE::Mass, 0);

    // compute mass via particle volume and initial density
    const double fac = material->initDensity_ * 4.0 / 3.0 * std::numbers::pi;
    for (int i = 0; i < particlestored; ++i) mass[i] = fac * Utils::pow<3>(radius[i]);
  }
}

void ParticleInteraction::ParticleInteractionDEM::set_initial_inertia()
{
  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // no inertia state for current particle type
    if (not container->have_stored_state(PARTICLEENGINE::Inertia)) continue;

    // get pointer to particle states
    const double* radius = container->get_ptr_to_state(PARTICLEENGINE::Radius, 0);
    const double* mass = container->get_ptr_to_state(PARTICLEENGINE::Mass, 0);
    double* inertia = container->get_ptr_to_state(PARTICLEENGINE::Inertia, 0);

    // compute mass via particle volume and initial density
    for (int i = 0; i < particlestored; ++i) inertia[i] = 0.4 * mass[i] * Utils::pow<2>(radius[i]);
  }
}

void ParticleInteraction::ParticleInteractionDEM::clear_force_and_moment_states() const
{
  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // clear force of all particles
    container->clear_state(PARTICLEENGINE::Force);

    // clear moment of all particles
    if (container->have_stored_state(PARTICLEENGINE::Moment))
      container->clear_state(PARTICLEENGINE::Moment);
  }
}

void ParticleInteraction::ParticleInteractionDEM::compute_acceleration() const
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::ParticleInteractionDEM::compute_acceleration");

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get particle state dimension
    const int statedim = container->get_state_dim(PARTICLEENGINE::Acceleration);

    // get pointer to particle states
    const double* radius = container->get_ptr_to_state(PARTICLEENGINE::Radius, 0);
    const double* mass = container->get_ptr_to_state(PARTICLEENGINE::Mass, 0);
    const double* force = container->get_ptr_to_state(PARTICLEENGINE::Force, 0);
    const double* moment = container->cond_get_ptr_to_state(PARTICLEENGINE::Moment, 0);
    double* acc = container->get_ptr_to_state(PARTICLEENGINE::Acceleration, 0);
    double* angacc = container->cond_get_ptr_to_state(PARTICLEENGINE::AngularAcceleration, 0);

    // compute acceleration
    for (int i = 0; i < particlestored; ++i)
      Utils::vec_add_scale(&acc[statedim * i], (1.0 / mass[i]), &force[statedim * i]);

    // compute angular acceleration
    if (angacc and moment)
    {
      for (int i = 0; i < particlestored; ++i)
        Utils::vec_add_scale(&angacc[statedim * i],
            (5.0 / (2.0 * mass[i] * Utils::pow<2>(radius[i]))), &moment[statedim * i]);
    }
  }
}

void ParticleInteraction::ParticleInteractionDEM::evaluate_particle_energy() const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::ParticleInteractionDEM::evaluate_particle_energy");

  // evaluate particle kinetic energy contribution
  std::vector<double> kinenergy(1, 0.0);
  {
    std::vector<double> localkinenergy(1, 0.0);
    evaluate_particle_kinetic_energy(localkinenergy[0]);
    kinenergy = Core::Communication::sum_all(localkinenergy, comm_);
  }

  // evaluate particle gravitational potential energy contribution
  std::vector<double> gravpotenergy(1, 0.0);
  {
    std::vector<double> localgravpotenergy(1, 0.0);
    evaluate_particle_gravitational_potential_energy(localgravpotenergy[0]);
    gravpotenergy = Core::Communication::sum_all(localgravpotenergy, comm_);
  }

  // evaluate elastic potential energy contribution
  std::vector<double> elastpotenergy(1, 0.0);
  {
    std::vector<double> localelastpotenergy(1, 0.0);
    contact_->evaluate_elastic_potential_energy(localelastpotenergy[0]);
    elastpotenergy = Core::Communication::sum_all(localelastpotenergy, comm_);
  }

  // get specific runtime csv writer
  Core::IO::RuntimeCsvWriter* runtime_csv_writer =
      particleinteractionwriter_->get_specific_runtime_csv_writer("particle-energy");

  // append data vector
  runtime_csv_writer->append_data_vector("kin_energy", kinenergy);
  runtime_csv_writer->append_data_vector("grav_pot_energy", gravpotenergy);
  runtime_csv_writer->append_data_vector("elast_pot_energy", elastpotenergy);
}

void ParticleInteraction::ParticleInteractionDEM::evaluate_particle_kinetic_energy(
    double& kineticenergy) const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::ParticleInteractionDEM::evaluate_particle_kinetic_energy");

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get particle state dimension
    const int statedim = container->get_state_dim(PARTICLEENGINE::Position);

    // get pointer to particle states
    const double* radius = container->get_ptr_to_state(PARTICLEENGINE::Radius, 0);
    const double* mass = container->get_ptr_to_state(PARTICLEENGINE::Mass, 0);
    const double* vel = container->get_ptr_to_state(PARTICLEENGINE::Velocity, 0);
    double* angvel = container->cond_get_ptr_to_state(PARTICLEENGINE::AngularVelocity, 0);

    // add translational kinetic energy contribution
    for (int i = 0; i < particlestored; ++i)
      kineticenergy += 0.5 * mass[i] * Utils::vec_dot(&vel[statedim * i], &vel[statedim * i]);

    // add rotational kinetic energy contribution
    if (angvel)
    {
      for (int i = 0; i < particlestored; ++i)
        kineticenergy += 0.5 * (0.4 * mass[i] * Utils::pow<2>(radius[i])) *
                         Utils::vec_dot(&angvel[statedim * i], &angvel[statedim * i]);
    }
  }
}

void ParticleInteraction::ParticleInteractionDEM::evaluate_particle_gravitational_potential_energy(
    double& gravitationalpotentialenergy) const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::ParticleInteractionDEM::evaluate_particle_gravitational_potential_"
      "energy");

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get particle state dimension
    const int statedim = container->get_state_dim(PARTICLEENGINE::Position);

    // get pointer to particle states
    const double* pos = container->get_ptr_to_state(PARTICLEENGINE::Position, 0);
    const double* mass = container->get_ptr_to_state(PARTICLEENGINE::Mass, 0);

    // add gravitational potential energy contribution
    for (int i = 0; i < particlestored; ++i)
      gravitationalpotentialenergy -= mass[i] * Utils::vec_dot(gravity_.data(), &pos[statedim * i]);
  }
}

FOUR_C_NAMESPACE_CLOSE
