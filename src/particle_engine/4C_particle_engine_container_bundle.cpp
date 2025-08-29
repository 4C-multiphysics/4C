// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_container_bundle.hpp"

#include "4C_particle_engine_object.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEENGINE::ParticleContainerBundle::ParticleContainerBundle()
{
  // empty constructor
}

void PARTICLEENGINE::ParticleContainerBundle::init()
{
  // nothing to do
}

void PARTICLEENGINE::ParticleContainerBundle::setup(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes)
{
  std::shared_ptr<ParticleContainer> container;

  // determine necessary size of vector for particle types
  const int typevectorsize = ((--particlestatestotypes.end())->first) + 1;

  // allocate memory to hold particle types
  containers_.resize(typevectorsize);

  // iterate over particle types
  for (const auto& typeIt : particlestatestotypes)
  {
    // get particle type
    ParticleType type = typeIt.first;

    // insert particle type into set of stored containers
    storedtypes_.insert(type);

    // allocate memory for container of owned and ghosted particles
    (containers_[type]).resize(2);

    // set of particle state enums of current particle type (equal for owned and ghosted particles)
    const std::set<ParticleState>& stateset = typeIt.second;

    // initial size of particle container
    int initialsize = 1;

    // create and init container of owned particles
    container = std::make_shared<ParticleContainer>();
    container->init();
    // setup container of owned particles
    container->setup(initialsize, stateset);
    // set container of owned particles
    (containers_[type])[Owned] = container;

    // create and init container of ghosted particles
    container = std::make_shared<ParticleContainer>();
    container->init();
    // setup container of ghosted particles
    container->setup(initialsize, stateset);
    // set container of ghosted particles
    (containers_[type])[Ghosted] = container;
  }
}

void PARTICLEENGINE::ParticleContainerBundle::get_packed_particle_objects_of_all_containers(
    std::vector<char>& particlebuffer) const
{
  // iterate over particle types
  for (const auto& type : storedtypes_)
  {
    // get container of owned particles
    ParticleContainer* container = (containers_[type])[Owned].get();

    // loop over particles in container
    for (int index = 0; index < container->particles_stored(); ++index)
    {
      int globalid(0);
      ParticleStates states;
      container->get_particle(index, globalid, states);

      ParticleObjShrdPtr particleobject = std::make_shared<ParticleObject>(type, globalid, states);

      // pack data for writing
      Core::Communication::PackBuffer data;
      particleobject->pack(data);
      particlebuffer.insert(particlebuffer.end(), data().begin(), data().end());
    }
  }
}

void PARTICLEENGINE::ParticleContainerBundle::get_vector_of_particle_objects_of_all_containers(
    std::vector<ParticleObjShrdPtr>& particlesstored) const
{
  // iterate over particle types
  for (const auto& type : storedtypes_)
  {
    // get container of owned particles
    ParticleContainer* container = (containers_[type])[Owned].get();

    // loop over particles in container
    for (int index = 0; index < container->particles_stored(); ++index)
    {
      int globalid(0);
      ParticleStates states;
      container->get_particle(index, globalid, states);

      particlesstored.emplace_back(std::make_shared<ParticleObject>(type, globalid, states));
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
