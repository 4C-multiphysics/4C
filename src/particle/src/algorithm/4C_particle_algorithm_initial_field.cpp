// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_initial_field.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::InitialFieldHandler::InitialFieldHandler(const Teuchos::ParameterList& params)
    : params_(params)
{
  // get control parameters for initial/boundary conditions
  const Teuchos::ParameterList& params_conditions =
      params_.sublist("INITIAL AND BOUNDARY CONDITIONS");

  // relate particle state to input name
  std::map<std::string, Particle::StateEnum> initialfieldtostateenum = {
      std::make_pair("INITIAL_TEMP_FIELD", Particle::Temperature),
      std::make_pair("INITIAL_VELOCITY_FIELD", Particle::Velocity),
      std::make_pair("INITIAL_ANGULAR_VELOCITY_FIELD", Particle::AngularVelocity),
      std::make_pair("INITIAL_ACCELERATION_FIELD", Particle::Acceleration),
      std::make_pair("INITIAL_ANGULAR_ACCELERATION_FIELD", Particle::AngularAcceleration)};

  // iterate over particle states
  for (auto& stateIt : initialfieldtostateenum)
  {
    // get reference to sub-map
    std::map<Particle::Type, int>& currentstatetypetofunctidmap =
        statetotypetofunctidmap_[stateIt.second];

    // read parameters relating particle types to values
    ParticleUtils::read_params_types_related_to_values(
        params_conditions, stateIt.first, currentstatetypetofunctidmap);
  }
}

void Particle::InitialFieldHandler::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;
}

void Particle::InitialFieldHandler::set_initial_fields()
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  for (auto& stateIt : statetotypetofunctidmap_)
  {
    // get state of particles
    Particle::StateEnum particleState = stateIt.first;

    // iterate over particle types
    for (auto& initialFieldIt : stateIt.second)
    {
      // get type of particles
      Particle::Type particleType = initialFieldIt.first;

      // get container of owned particles of current particle type
      Particle::ParticleContainer* container =
          particlecontainerbundle->get_specific_container(particleType, Particle::Status::Owned);

      // get number of particles stored in container
      const int particlestored = container->particles_stored();

      // no owned particles of current particle type
      if (particlestored <= 0) continue;

      if (not container->have_stored_state(particleState)) continue;

      // get id of function
      const int functid = initialFieldIt.second;

      // get reference to function
      const auto& function =
          Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(functid);

      // get pointer to particle states
      const double* pos = container->get_ptr_to_state(Particle::Position, 0);
      double* state = container->get_ptr_to_state_writable(particleState, 0);

      // get particle state dimensions
      int posstatedim = container->get_state_dim(Particle::Position);
      int statedim = container->get_state_dim(particleState);

      // safety check
      if (static_cast<std::size_t>(statedim) != function.number_components())
        FOUR_C_THROW(
            "dimensions of function defining initial field and of state '{}' not matching!",
            Particle::enum_to_state_name(particleState).c_str());

      // iterate over owned particles of current type
      for (int i = 0; i < particlestored; ++i)
      {
        // evaluate function to set initial field
        for (int dim = 0; dim < statedim; ++dim)
          state[statedim * i + dim] =
              function.evaluate(std::span(&(pos[posstatedim * i]), 3), 0.0, dim);
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
