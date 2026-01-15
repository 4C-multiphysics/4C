// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_VIRTUAL_WALL_PARTICLE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_VIRTUAL_WALL_PARTICLE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleContainerBundle;
  class ParticleEngineInterface;
  class SPHKernelBase;
  class SPHNeighborPairs;
  class WallHandlerInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHVirtualWallParticle
  {
   public:
    //! constructor
    explicit SPHVirtualWallParticle(const Teuchos::ParameterList& params);

    //! setup virtual wall particle handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs);

    //! get reference to relative positions of virtual particles
    inline const std::vector<std::vector<double>>& get_relative_positions_of_virtual_particles()
        const
    {
      return virtualparticles_;
    };

    //! get reference to weighted fluid particle pressure
    inline const std::vector<double>& get_weighted_pressure() const { return weightedpressure_; };

    //! get reference to weighted fluid particle pressure gradient
    inline const std::vector<std::vector<double>>& get_weighted_pressure_gradient() const
    {
      return weightedpressuregradient_;
    };

    //! get reference to weighted fluid particle distance vector
    inline const std::vector<std::vector<double>>& get_weighted_distance_vector() const
    {
      return weighteddistancevector_;
    };

    //! get reference to weighted fluid particle velocity
    inline const std::vector<std::vector<double>>& get_weighted_velocity() const
    {
      return weightedvelocity_;
    };

    //! init relative positions of virtual particles
    void init_relative_positions_of_virtual_particles(const double maxinteractiondistance);

    //! init states at wall contact points
    void init_states_at_wall_contact_points(std::vector<double>& gravity);

   private:
    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! kernel handler
    std::shared_ptr<Particle::SPHKernelBase> kernel_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! relative positions of virtual particles
    std::vector<std::vector<double>> virtualparticles_;

    //! weighted fluid particle pressure
    std::vector<double> weightedpressure_;

    //! weighted fluid particle pressure gradient
    std::vector<std::vector<double>> weightedpressuregradient_;

    //! weighted fluid particle distance vector
    std::vector<std::vector<double>> weighteddistancevector_;

    //! weighted fluid particle velocity
    std::vector<std::vector<double>> weightedvelocity_;

    //! set of all fluid particle types
    std::set<Particle::TypeEnum> allfluidtypes_;

    //! set of integrated fluid particle types
    std::set<Particle::TypeEnum> intfluidtypes_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
