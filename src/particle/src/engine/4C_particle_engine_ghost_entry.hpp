// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_GHOST_ENTRY_HPP
#define FOUR_C_PARTICLE_ENGINE_GHOST_ENTRY_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief entry for ghosting an owned particle on another processor
   *
   * pack() and unpack() together define the wire format used by
   * ParticleEngine::pack_particles_to_be_ghosted() and
   * ParticleEngine::communicate_and_insert_ghost_particles() to create ghosted particles directly
   * from send buffers, bypassing ParticleObject creation. Both must be kept in sync: unpack()
   * extracts fields in exactly the order pack() writes them.
   */
  struct ParticleGhostEntry
  {
    //! particle type
    Particle::Type type;

    //! global id of the particle
    int globalid;

    //! global id of the bin the particle is ghosted in on the target processor
    int bingid;

    //! local index of the particle in the container of owned particles on the sending processor
    int ownedindex;

    //! states of the particle
    ParticleStates states;

    /*!
     * \brief pack header (type, global id, bin id, owned index) and append pre-packed states to a
     *        send buffer
     *
     * The states are passed in already packed since, for a single owned particle that is ghosted
     * on multiple processors, the same packed bytes are appended to every target processor's send
     * buffer without re-serializing the states for each target.
     *
     * \param[in]  type             particle type
     * \param[in]  globalid         global id of the particle
     * \param[in]  bingid           global id of the bin the particle is ghosted in
     * \param[in]  ownedindex       local index of the particle in the container of owned particles
     * \param[in]  prepacked_states pre-packed particle states
     * \param[out] sendbuffer       send buffer of the target processor to append to
     */
    static void pack(Particle::Type type, int globalid, int bingid, int ownedindex,
        const Core::Communication::PackBuffer& prepacked_states, std::vector<char>& sendbuffer);

    /*!
     * \brief unpack one ghost entry previously packed by pack()
     *
     *
     * \param[in,out] buffer buffer to unpack from
     * \return unpacked ghost entry
     */
    static ParticleGhostEntry unpack(Core::Communication::UnpackBuffer& buffer);
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
