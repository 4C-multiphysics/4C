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

Particle::SPHPressure::SPHPressure() : fluidtypes_({Particle::Phase1, Particle::Phase2})
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
    std::vector<Particle::StateEnum> states{Particle::Pressure};

    for (const auto& type_i : fluidtypes_)
      pressuretorefresh_.push_back(std::make_pair(type_i, states));
  }
}

void Particle::SPHPressure::compute_pressure() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPressure::ComputePressure");

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get pointer to particle state
    const double* dens = container->get_ptr_to_state(Particle::Density, 0);
    double* press = container->get_ptr_to_state(Particle::Pressure, 0);

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* material =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // get equation of state for current particle type
    const Particle::SPHEquationOfStateBase* equationofstate =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // iterate over owned particles of current type
    for (int i = 0; i < particlestored; ++i)
      press[i] = equationofstate->density_to_pressure(dens[i], material->initDensity_);
  }

  // refresh pressure of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(pressuretorefresh_);
}

FOUR_C_NAMESPACE_CLOSE
