// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_NEIGHBOR_PAIRS_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_NEIGHBOR_PAIRS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_dem_neighbor_pair_struct.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleContainerBundle;
  class ParticleEngineInterface;
  class WallHandlerInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | type definitions                                                          |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  using DEMParticlePairData = std::vector<Particle::DEMParticlePair>;
  using DEMParticleWallPairData = std::vector<Particle::DEMParticleWallPair>;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class DEMNeighborPairs final
  {
   public:
    //! constructor
    explicit DEMNeighborPairs();

    //! setup neighbor pair handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface);

    //! get reference to particle pair data
    inline const DEMParticlePairData& get_ref_to_particle_pair_data() const
    {
      return particlepairdata_;
    };

    //! get reference to particle-wall pair data
    inline const DEMParticleWallPairData& get_ref_to_particle_wall_pair_data() const
    {
      return particlewallpairdata_;
    };

    //! get reference to adhesion particle pair data
    inline const DEMParticlePairData& get_ref_to_particle_pair_adhesion_data() const
    {
      return particlepairadhesiondata_;
    };

    //! get reference to adhesion particle-wall pair data
    inline const DEMParticleWallPairData& get_ref_to_particle_wall_pair_adhesion_data() const
    {
      return particlewallpairadhesiondata_;
    };

    //! evaluate neighbor pairs
    void evaluate_neighbor_pairs();

    //! evaluate adhesion neighbor pairs
    void evaluate_neighbor_pairs_adhesion(const double& adhesion_distance);

   private:
    //! evaluate particle pairs
    void evaluate_particle_pairs();

    //! evaluate particle-wall pairs
    void evaluate_particle_wall_pairs();

    //! evaluate adhesion particle pairs
    void evaluate_particle_pairs_adhesion(const double& adhesion_distance);

    //! evaluate adhesion particle-wall pairs
    void evaluate_particle_wall_pairs_adhesion(const double& adhesion_distance);

    //! particle pair data with evaluated quantities
    DEMParticlePairData particlepairdata_;

    //! particle-wall pair data with evaluated quantities
    DEMParticleWallPairData particlewallpairdata_;

    //! adhesion particle pair data with evaluated quantities
    DEMParticlePairData particlepairadhesiondata_;

    //! adhesion particle-wall pair data with evaluated quantities
    DEMParticleWallPairData particlewallpairadhesiondata_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
