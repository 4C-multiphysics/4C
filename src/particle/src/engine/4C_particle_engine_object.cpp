// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_object.hpp"

#include "4C_comm_pack_helpers.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ParticleObjectType Particle::ParticleObjectType::instance_;

Core::Communication::ParObject* Particle::ParticleObjectType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  ParticleObject* my_particleobject = new ParticleObject();
  my_particleobject->unpack(buffer);
  return my_particleobject;
}

Particle::ParticleObject::ParticleObject()
    : type_(ParticleType::Phase1), globalid_(0), bingid_(-1), index_(-1)
{
  // empty constructor
}

Particle::ParticleObject::ParticleObject(
    ParticleType type, int globalid, const ParticleStates& states, int bingid, int index)
    : type_(type), globalid_(globalid), states_(states), bingid_(bingid), index_(index)
{
  // empty constructor
}

void Particle::ParticleObject::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // particletype_
  add_to_pack(data, type_);

  // particleglobalid_
  add_to_pack(data, globalid_);

  // particle states
  int numstates = states_.size();
  add_to_pack(data, numstates);
  for (int i = 0; i < numstates; ++i) add_to_pack(data, states_[i]);

  // bingid_
  add_to_pack(data, bingid_);

  // containerindex_
  add_to_pack(data, index_);
}

void Particle::ParticleObject::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // particletype_
  extract_from_pack(buffer, type_);

  // particleglobalid_
  extract_from_pack(buffer, globalid_);

  // particle states
  int numstates = 0;
  extract_from_pack(buffer, numstates);
  states_.resize(numstates);
  for (int i = 0; i < numstates; ++i) extract_from_pack(buffer, states_[i]);

  // bingid_
  extract_from_pack(buffer, bingid_);

  // containerindex_
  extract_from_pack(buffer, index_);
}

FOUR_C_NAMESPACE_CLOSE
