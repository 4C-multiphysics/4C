// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_constraints.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

void Particle::ConstraintsProjection2D::apply(
    Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
    const std::set<Particle::TypeEnum>& types_to_integrate, const double time) const
{
  // iterate over particle types
  for (auto& particleType : types_to_integrate)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particle_container_bundle->get_specific_container(particleType, Particle::Owned);

    // get number of particles stored in container
    const int n_particle_stored = container->particles_stored();

    // no owned particles of current particle type
    if (n_particle_stored <= 0) continue;

    // get pointer to particle velocity and acceleration
    double* vel = container->get_ptr_to_state(Particle::Velocity, 0);
    double* acc = container->get_ptr_to_state(Particle::Acceleration, 0);

    double* modvel = nullptr;
    if (container->have_stored_state(Particle::ModifiedVelocity))
      modvel = container->get_ptr_to_state(Particle::ModifiedVelocity, 0);

    double* modacc = nullptr;
    if (container->have_stored_state(Particle::ModifiedAcceleration))
      modacc = container->get_ptr_to_state(Particle::ModifiedAcceleration, 0);

    // get particle state dimension
    int pos_state_dim = container->get_state_dim(Particle::Position);

    // iterate over owned particles of current type
    for (int i = 0; i < n_particle_stored; ++i)
    {
      vel[pos_state_dim * i + 2] = 0.0;
      acc[pos_state_dim * i + 2] = 0.0;
      if (modvel) modvel[pos_state_dim * i + 2] = 0.0;
      if (modacc) modacc[pos_state_dim * i + 2] = 0.0;
    }
  }
}

std::unique_ptr<Particle::ConstraintsHandler> Particle::create_constraints(
    const Teuchos::ParameterList& params)
{
  std::unique_ptr<Particle::ConstraintsHandler> constraints = nullptr;

  const auto restrain_to_2d =
      params.sublist("INITIAL AND BOUNDARY CONDITIONS").get<bool>("RESTRAIN_TO_2D");

  if (restrain_to_2d) constraints = std::make_unique<Particle::ConstraintsProjection2D>();

  return constraints;
}

FOUR_C_NAMESPACE_CLOSE
