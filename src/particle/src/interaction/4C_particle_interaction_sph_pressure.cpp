// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_pressure.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

Particle::SPHPressure::SPHPressure() : fluidtypes_({ParticleType::Phase1, ParticleType::Phase2})
{
  // empty constructor
}

void Particle::SPHPressure::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial,
    const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // set equation of state handler
  equationofstatebundle_ = equationofstatebundle;

  // update with actual fluid particle types
  const auto fluidtypes = fluidtypes_;
  for (const auto& type_i : fluidtypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      fluidtypes_.erase(type_i);

  // setup pressure of ghosted particles to refresh
  {
    std::vector<ParticleState> states{ParticleState::Pressure};

    for (const auto& type_i : fluidtypes_)
      pressuretorefresh_.push_back(std::make_pair(type_i, states));
  }
}

void Particle::SPHPressure::compute_pressure() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPressure::ComputePressure");

  // get pointers to particle states
  ConstParticleContainerBundleStatePtrs& dens = particlecontainerbundle_->try_get_ptrs_to_state(
      ParticleState::Density, fluidtypes_, ParticleStatus::Owned);
  ParticleContainerBundleStatePtrs& press =
      particlecontainerbundle_->try_get_ptrs_to_state_writable(
          ParticleState::Pressure, fluidtypes_, ParticleStatus::Owned);

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    const ParticleStatus status_i = ParticleStatus::Owned;
    Particle::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* material =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);
    const double initDensity = material->initDensity_;

    // get equation of state for current particle type
    const Particle::SPHEquationOfStateBase* equationofstate =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // iterate over owned particles of current type
    for (int particle_i = 0; particle_i < particlestored; ++particle_i)
    {
      const double* dens_i = Particle::bundle_state_ptrs_index(dens, type_i, status_i, particle_i);
      double* press_i = Particle::bundle_state_ptrs_index(press, type_i, status_i, particle_i);

      press_i[0] = equationofstate->density_to_pressure(dens_i[0], initDensity);
    }
  }

  // refresh pressure of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(pressuretorefresh_);
}

FOUR_C_NAMESPACE_CLOSE
