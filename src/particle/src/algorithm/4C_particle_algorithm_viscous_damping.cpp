// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_viscous_damping.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ViscousDampingHandler::ViscousDampingHandler(const double viscdampfac)
    : viscdampfac_(viscdampfac)
{
  // empty constructor
}

void Particle::ViscousDampingHandler::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;
}

void Particle::ViscousDampingHandler::apply_viscous_damping()
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over particle types
  for (const auto& type : particlecontainerbundle->get_particle_types())
  {
    // no viscous damping contribution for boundary or rigid particles
    if (type == Particle::BoundaryPhase or type == Particle::RigidPhase) continue;

    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(type, Particle::Status::Owned);

    // apply viscous damping contribution
    container->update_state(1.0, Particle::Acceleration, -viscdampfac_, Particle::Velocity);
  }
}

FOUR_C_NAMESPACE_CLOSE
