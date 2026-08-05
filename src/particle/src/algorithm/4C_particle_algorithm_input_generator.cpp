// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_input_generator.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_particle_engine_object.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::InputGenerator::InputGenerator(MPI_Comm comm, const Teuchos::ParameterList& params)
    : myrank_(Core::Communication::my_mpi_rank(comm)), params_(params)
{
  // empty constructor
}

void Particle::InputGenerator::generate_particles(
    std::vector<Particle::ParticleObjShrdPtr>& particlesgenerated) const
{
  // generate initial particles
}

void Particle::InputGenerator::add_generated_particle(const std::vector<double>& position,
    const Particle::Type particletype,
    std::vector<Particle::ParticleObjShrdPtr>& particlesgenerated) const
{
  // safety check
  if (position.size() != 3)
    FOUR_C_THROW("particle can not be generated since position vector needs three entries!");

  // allocate memory to hold particle states
  Particle::ParticleStates particlestates;
  particlestates.assign((Particle::Position + 1), std::vector<double>(0));

  // set position state
  particlestates[Particle::Position] = position;

  // construct and store generated particle object
  particlesgenerated.emplace_back(
      std::make_shared<Particle::ParticleObject>(particletype, -1, particlestates));
}

FOUR_C_NAMESPACE_CLOSE
