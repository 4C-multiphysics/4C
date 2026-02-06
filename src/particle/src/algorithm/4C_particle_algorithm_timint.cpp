// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_timint.hpp"

#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_particle_algorithm_constraints.hpp"
#include "4C_particle_algorithm_dirichlet_bc.hpp"
#include "4C_particle_algorithm_temperature_bc.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_rigidbody_interface.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::TimInt::TimInt(const Teuchos::ParameterList& params)
    : params_(params), time_(0.0), dt_(params.get<double>("TIMESTEP"))
{
  init_dirichlet_boundary_condition();

  init_temperature_boundary_condition();
}

Particle::TimInt::~TimInt() = default;

void Particle::TimInt::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::RigidBodyHandlerInterface> particlerigidbodyinterface,
    const std::shared_ptr<Particle::ConstraintsHandler> constraints)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set interface to rigid body handler
  particlerigidbodyinterface_ = particlerigidbodyinterface;

  // setup dirichlet boundary condition handler
  if (dirichletboundarycondition_) dirichletboundarycondition_->setup(particleengineinterface_);

  // setup temperature boundary condition handler
  if (temperatureboundarycondition_) temperatureboundarycondition_->setup(particleengineinterface_);

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // set of particle types being excluded from time integration
  std::set<Particle::TypeEnum> typesexludedfromtimeintegration;

  // boundary and rigid particles are not integrated in time
  typesexludedfromtimeintegration.insert({Particle::BoundaryPhase, Particle::RigidPhase});

  if (dirichletboundarycondition_)
  {
    // get reference to set of particle types subjected to dirichlet boundary conditions
    const std::set<Particle::TypeEnum>& typessubjectedtodirichletbc =
        dirichletboundarycondition_->get_particle_types_subjected_to_dirichlet_bc_set();

    // particles subjected to dirichlet boundary conditions are not integrated in time
    for (Particle::TypeEnum currtype : typessubjectedtodirichletbc)
      typesexludedfromtimeintegration.insert(currtype);
  }

  // determine set of particle types to be integrated in time
  for (auto& typeEnum : particlecontainerbundle->get_particle_types())
    if (not typesexludedfromtimeintegration.contains(typeEnum)) typestointegrate_.insert(typeEnum);

  // set constraints handler
  constraints_ = constraints;
}

void Particle::TimInt::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // insert dbc dependent states of all particle types
  if (dirichletboundarycondition_)
    dirichletboundarycondition_->insert_particle_states_of_particle_types(particlestatestotypes);

  // insert tempbc dependent states of all particle types
  if (temperatureboundarycondition_)
    temperatureboundarycondition_->insert_particle_states_of_particle_types(particlestatestotypes);
}

void Particle::TimInt::set_initial_states()
{
  // add initial random noise to particle position
  add_initial_random_noise_to_position();

  // set particle reference position
  if (dirichletboundarycondition_) dirichletboundarycondition_->set_particle_reference_position();

  // set particle reference position
  if (temperatureboundarycondition_)
    temperatureboundarycondition_->set_particle_reference_position();

  // evaluate dirichlet boundary condition
  if (dirichletboundarycondition_)
    dirichletboundarycondition_->evaluate_dirichlet_boundary_condition(0.0, true, true, true);

  // evaluate temperature boundary condition
  if (temperatureboundarycondition_)
    temperatureboundarycondition_->evaluate_temperature_boundary_condition(0.0);
}

void Particle::TimInt::set_current_time(const double currenttime) { time_ = currenttime; }

void Particle::TimInt::init_dirichlet_boundary_condition()
{
  // create dirichlet boundary condition handler
  dirichletboundarycondition_ =
      std::make_unique<Particle::DirichletBoundaryConditionHandler>(params_);

  // get reference to set of particle types subjected to dirichlet boundary conditions
  const std::set<Particle::TypeEnum>& typessubjectedtodirichletbc =
      dirichletboundarycondition_->get_particle_types_subjected_to_dirichlet_bc_set();

  // no particle types are subjected to dirichlet boundary conditions
  if (typessubjectedtodirichletbc.empty()) dirichletboundarycondition_.reset();
}

void Particle::TimInt::init_temperature_boundary_condition()
{
  // create temperature boundary condition handler
  temperatureboundarycondition_ =
      std::make_unique<Particle::TemperatureBoundaryConditionHandler>(params_);

  // get reference to set of particle types subjected to temperature boundary conditions
  const std::set<Particle::TypeEnum>& typessubjectedtotempbc =
      temperatureboundarycondition_->get_particle_types_subjected_to_temperature_bc_set();

  // no particle types are subjected to temperature boundary conditions
  if (typessubjectedtotempbc.empty()) temperatureboundarycondition_.reset();
}

void Particle::TimInt::add_initial_random_noise_to_position()
{
  // init vector of initial position amplitude for each spatial direction
  std::vector<double> amplitude;
  double value;
  std::istringstream amplitudestream(
      Teuchos::getNumericStringParameter(params_, "INITIAL_POSITION_AMPLITUDE"));

  while (amplitudestream >> value) amplitude.push_back(value);

  // safety check
  if (static_cast<int>(amplitude.size()) != 3)
    FOUR_C_THROW("dimension (dim = {}) of initial position amplitude vector is wrong!",
        static_cast<int>(amplitude.size()));

  // safety check
  for (double a : amplitude)
    if (a < 0.0)
      FOUR_C_THROW(
          "no negative initial position amplitude allowed (set a positive or zero value)!");

  // get magnitude of initial position amplitude
  double temp = 0.0;
  for (double a : amplitude) temp += a * a;
  const double amplitude_norm = std::sqrt(temp);

  // no initial position amplitude defined
  if (not(amplitude_norm > 0.0)) return;

  // safety check
  const double max_amplitude = *std::max_element(amplitude.begin(), amplitude.end());
  if (max_amplitude > particleengineinterface_->min_bin_size())
    FOUR_C_THROW(
        "amplitude of noise added to initial position larger than minimum relevant bin size!");

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over particle types
  for (auto& particleType : typestointegrate_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get pointer to particle state
    double* pos = container->get_ptr_to_state(Particle::Position, 0);

    // get particle state dimension
    int statedim = container->get_state_dim(Particle::Position);

    // iterate over owned particles of current type
    for (int i = 0; i < particlestored; ++i)
    {
      // iterate over spatial dimension
      for (int dim = 0; dim < statedim; ++dim)
      {
        // generate random value
        const double randomvalue = Global::Problem::instance()->random()->uni();

        // update position of particle
        pos[statedim * i + dim] += randomvalue * amplitude[dim];
      }
    }
  }
}

Particle::TimIntSemiImplicitEuler::TimIntSemiImplicitEuler(const Teuchos::ParameterList& params)
    : Particle::TimInt(params)
{
  // empty constructor
}

void Particle::TimIntSemiImplicitEuler::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::RigidBodyHandlerInterface> particlerigidbodyinterface,
    const std::shared_ptr<Particle::ConstraintsHandler> constraints)
{
  // call base class setup
  Particle::TimInt::setup(particleengineinterface, particlerigidbodyinterface);

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over particle types
  for (auto& particleType : typestointegrate_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // safety check
    if (container->have_stored_state(Particle::ModifiedVelocity) or
        container->have_stored_state(Particle::ModifiedAcceleration))
      FOUR_C_THROW(
          "modified velocity and acceleration states not implemented yet for semi-implicit Euler "
          "time integration scheme!");
  }

  // set constraints handler
  constraints_ = constraints;
}

void Particle::TimIntSemiImplicitEuler::pre_interaction_routine()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::TimIntSemiImplicitEuler::pre_interaction_routine");

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // apply kinematic constraints
  if (constraints_) constraints_->apply(particlecontainerbundle, typestointegrate_, time_);

  // iterate over particle types
  for (auto& particleType : typestointegrate_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // update velocity of all particles
    container->update_state(1.0, Particle::Velocity, dt_, Particle::Acceleration);

    // clear acceleration of all particles
    container->clear_state(Particle::Acceleration);

    // angular velocity and acceleration states
    if (container->have_stored_state(Particle::AngularVelocity) and
        container->have_stored_state(Particle::AngularAcceleration))
    {
      // update angular velocity of all particles
      container->update_state(1.0, Particle::AngularVelocity, dt_, Particle::AngularAcceleration);

      // clear angular acceleration of all particles
      container->clear_state(Particle::AngularAcceleration);
    }

    // update position of all particles
    container->update_state(1.0, Particle::Position, dt_, Particle::Velocity);
  }

  if (particlerigidbodyinterface_)
  {
    // update velocities of all rigid bodies and corresponding rigid particles
    particlerigidbodyinterface_->update_velocities(dt_);

    // clear accelerations of all rigid bodies
    particlerigidbodyinterface_->clear_accelerations();

    // update positions of all rigid bodies and corresponding rigid particles
    particlerigidbodyinterface_->update_positions(dt_);
  }

  // evaluate dirichlet boundary condition
  if (dirichletboundarycondition_)
    dirichletboundarycondition_->evaluate_dirichlet_boundary_condition(time_, true, true, true);

  // evaluate temperature boundary condition
  if (temperatureboundarycondition_)
    temperatureboundarycondition_->evaluate_temperature_boundary_condition(time_);
}

void Particle::TimIntSemiImplicitEuler::post_interaction_routine()
{
  // nothing to do
}

Particle::TimIntVelocityVerlet::TimIntVelocityVerlet(const Teuchos::ParameterList& params)
    : Particle::TimInt(params), dthalf_(0.5 * dt_)
{
  // empty constructor
}

void Particle::TimIntVelocityVerlet::set_initial_states()
{
  // call base class method
  TimInt::set_initial_states();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over particle types
  for (auto& particleType : typestointegrate_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // modified velocity states
    if (container->have_stored_state(Particle::ModifiedVelocity))
    {
      // update modified velocity of all particles
      container->update_state(0.0, Particle::ModifiedVelocity, 1.0, Particle::Velocity);
    }
  }
}

void Particle::TimIntVelocityVerlet::pre_interaction_routine()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::TimIntVelocityVerlet::pre_interaction_routine");

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // apply kinematic constraints
  if (constraints_) constraints_->apply(particlecontainerbundle, typestointegrate_, time_);

  // iterate over particle types
  for (auto& particleType : typestointegrate_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // update velocity of all particles
    container->update_state(1.0, Particle::Velocity, dthalf_, Particle::Acceleration);

    // clear acceleration of all particles
    container->clear_state(Particle::Acceleration);

    // angular velocity and acceleration states
    if (container->have_stored_state(Particle::AngularVelocity) and
        container->have_stored_state(Particle::AngularAcceleration))
    {
      // update angular velocity of all particles
      container->update_state(
          1.0, Particle::AngularVelocity, dthalf_, Particle::AngularAcceleration);

      // clear angular acceleration of all particles
      container->clear_state(Particle::AngularAcceleration);
    }

    // modified velocity and acceleration states
    if (container->have_stored_state(Particle::ModifiedVelocity) and
        container->have_stored_state(Particle::ModifiedAcceleration))
    {
      // update modified velocity of all particles
      container->update_state(0.0, Particle::ModifiedVelocity, 1.0, Particle::Velocity);
      container->update_state(
          1.0, Particle::ModifiedVelocity, dthalf_, Particle::ModifiedAcceleration);

      // clear modified acceleration of all particles
      container->clear_state(Particle::ModifiedAcceleration);

      // update position of all particles
      container->update_state(1.0, Particle::Position, dt_, Particle::ModifiedVelocity);
    }
    else
    {
      // update position of all particles
      container->update_state(1.0, Particle::Position, dt_, Particle::Velocity);
    }
  }

  if (particlerigidbodyinterface_)
  {
    // update velocities of all rigid bodies and corresponding rigid particles
    particlerigidbodyinterface_->update_velocities(dthalf_);

    // clear accelerations of all rigid bodies
    particlerigidbodyinterface_->clear_accelerations();

    // update positions of all rigid bodies and corresponding rigid particles
    particlerigidbodyinterface_->update_positions(dt_);
  }

  // evaluate dirichlet boundary condition
  if (dirichletboundarycondition_)
  {
    dirichletboundarycondition_->evaluate_dirichlet_boundary_condition(time_, true, false, true);
    dirichletboundarycondition_->evaluate_dirichlet_boundary_condition(
        time_ - dthalf_, false, true, false);
  }

  // evaluate temperature boundary condition
  if (temperatureboundarycondition_)
    temperatureboundarycondition_->evaluate_temperature_boundary_condition(time_);
}

void Particle::TimIntVelocityVerlet::post_interaction_routine()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::TimIntVelocityVerlet::post_interaction_routine");

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over particle types
  for (auto& particleType : typestointegrate_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // update velocity of all particles
    container->update_state(1.0, Particle::Velocity, dthalf_, Particle::Acceleration);

    // angular velocity and acceleration states
    if (container->have_stored_state(Particle::AngularVelocity) and
        container->have_stored_state(Particle::AngularAcceleration))
    {
      // update angular velocity of all particles
      container->update_state(
          1.0, Particle::AngularVelocity, dthalf_, Particle::AngularAcceleration);
    }
  }

  // update velocities of all rigid bodies and corresponding rigid particles
  if (particlerigidbodyinterface_) particlerigidbodyinterface_->update_velocities(dthalf_);

  // evaluate dirichlet boundary condition
  if (dirichletboundarycondition_)
    dirichletboundarycondition_->evaluate_dirichlet_boundary_condition(time_, false, true, false);
}

FOUR_C_NAMESPACE_CLOSE
