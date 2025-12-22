// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_PD_NEIGHBOR_PAIR_STRUCT_HPP
#define FOUR_C_PARTICLE_INTERACTION_PD_NEIGHBOR_PAIR_STRUCT_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  //! struct to store quantities of interacting particles
  struct PDParticlePair final
  {
    //! local index tuple of particles i and j
    Particle::LocalIndexTuple tuple_i_;
    Particle::LocalIndexTuple tuple_j_;

    //! absolute distance between particles
    double absdist_;

    //! versor from particle j to i
    double e_ij_[3];
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
