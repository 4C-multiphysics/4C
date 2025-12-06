// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_TYPEDEFS_HPP
#define FOUR_C_PARTICLE_ENGINE_TYPEDEFS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"

#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleContainer;
  class ParticleContainerBundle;
  class ParticleObject;
}  // namespace Particle

namespace Core::Elements
{
  class Element;
}

/*---------------------------------------------------------------------------*
 | type definitions                                                          |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  //! particle type enum
  using TypeEnum = ParticleType;

  //! particle status enum
  using StatusEnum = ParticleStatus;

  //! particle state enum
  using StateEnum = ParticleState;

  //! states of particle indexed by particle state enum
  using ParticleStates = std::vector<std::vector<double>>;

  //! shared pointer to particle object
  using ParticleObjShrdPtr = std::shared_ptr<ParticleObject>;

  //! collection of particle containers indexed by particle type enum and particle status enum
  using TypeStatusContainers = std::vector<std::vector<std::shared_ptr<ParticleContainer>>>;

  //! shared pointer to particle container bundle
  using ParticleContainerBundleShrdPtr = std::shared_ptr<ParticleContainerBundle>;

  /*!
   * \brief local index tuple of a particle
   *
   * Local index tuple of a particle consisting of particle type enum, particle status enum, and
   * index of particle in container.
   */
  using LocalIndexTuple = std::tuple<ParticleType, ParticleStatus, int>;

  //! shared pointer to local index tuple of a particle
  using LocalIndexTupleShrdPtr = std::shared_ptr<LocalIndexTuple>;

  /*!
   * \brief local and global index tuple of a particle
   *
   * Local and global index tuple of a particle consisting of particle type enum, particle status
   * enum, and local index of particle in container and global index.
   */
  using LocalGlobalIndexTuple = std::tuple<ParticleType, ParticleStatus, int, int>;

  /*!
   * \brief relate particles to corresponding bins being located in
   *
   * Relate all particles (consisting of particle type enum and index of particle in container) to
   * their respective corresponding bin the particles are located in. The bins are indexed by the
   * local id of the bin column map.
   *
   * \note The information of the particle status (owned or ghosted) can be deduced directly from
   *       the status of the bin.
   */
  using ParticlesToBins = std::vector<std::vector<std::pair<ParticleType, int>>>;

  //! potential particle neighbor pairs
  using PotentialParticleNeighbors = std::vector<std::pair<LocalIndexTuple, LocalIndexTuple>>;

  //! relate particle type enums and particle state enums to be refreshed
  using StatesOfTypesToRefresh = std::vector<std::pair<ParticleType, std::vector<ParticleState>>>;

  /*!
   * \brief relate range of owned bins to corresponding column wall element
   *
   * Relate the range of owned bins (global id of bin) to the column wall element occupying that
   * bins. The column wall elements are indexed by the local id of their column map.
   */
  using BinsToColWallEles = std::vector<std::vector<int>>;

  //! potential particle wall neighbor pairs
  using PotentialWallNeighbors = std::vector<std::pair<LocalIndexTuple, Core::Elements::Element*>>;

  //! relate particle source to target type after phase change
  using ParticleTypeToType = std::tuple<ParticleType, ParticleType, int>;

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
