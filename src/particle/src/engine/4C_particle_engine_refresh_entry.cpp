// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_refresh_entry.hpp"

#include "4C_comm_pack_helpers.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
void Particle::ParticleRefreshEntry::pack(ParticleType type, int ghostedindex,
    const Core::Communication::PackBuffer& prepacked_states, std::vector<char>& sendbuffer)
{
  // pack type and ghosted index header
  Core::Communication::PackBuffer header;
  add_to_pack(header, static_cast<int>(type));
  add_to_pack(header, ghostedindex);

  // append header + pre-packed states to send buffer
  sendbuffer.insert(sendbuffer.end(), header().begin(), header().end());
  sendbuffer.insert(sendbuffer.end(), prepacked_states().begin(), prepacked_states().end());
}

Particle::ParticleRefreshEntry Particle::ParticleRefreshEntry::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  ParticleRefreshEntry entry;

  // unpack particle type
  int type_idx;

  extract_from_pack(buffer, type_idx);
  entry.type = static_cast<ParticleType>(type_idx);

  // unpack ghosted index
  extract_from_pack(buffer, entry.ghostedindex);

  // unpack states
  extract_from_pack(buffer, entry.states);

  return entry;
}

FOUR_C_NAMESPACE_CLOSE
