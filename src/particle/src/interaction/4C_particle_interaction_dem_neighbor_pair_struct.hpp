// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_NEIGHBOR_PAIR_STRUCT_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_NEIGHBOR_PAIR_STRUCT_HPP

#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  class Element;
}

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  //! struct to store quantities of interacting particles
  struct DEMParticlePair final
  {
    //! local index tuple of particles i and j
    Particle::LocalIndexTuple tuple_i_;
    Particle::LocalIndexTuple tuple_j_;

    //! gap between particles
    double gap_;

    //! absolute distance between particles
    double absdist_;

    //! versor from particle i to j
    double e_ji_[3];

    //! effective mass of particles i and j
    double m_eff_;
  };

  //! struct to store quantities of interacting particles and wall elements
  struct DEMParticleWallPair final
  {
    //! local index tuple of particle i
    Particle::LocalIndexTuple tuple_i_;

    //! pointer to column wall element
    Core::Elements::Element* ele_;

    //! gap between particle and wall contact point
    double gap_;

    //! versor from particle i to wall contact point j
    double e_ji_[3];

    //! parameter space coordinates of wall contact point
    double elecoords_[2];

    //! global ids of relevant wall elements in penetration volume for interaction history
    std::set<int> histeles_;
  };
}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
