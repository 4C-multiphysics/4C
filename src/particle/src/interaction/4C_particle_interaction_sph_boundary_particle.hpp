// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_BOUNDARY_PARTICLE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_BOUNDARY_PARTICLE_HPP

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
  class SPHNeighborPairs;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHBoundaryParticleBase
  {
   public:
    //! constructor
    explicit SPHBoundaryParticleBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHBoundaryParticleBase() = default;

    //! setup boundary particle handler
    virtual void setup(
        const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs);

    //! init boundary particle states
    virtual void init_boundary_particle_states(std::vector<double>& gravity) = 0;

   protected:
    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! set of fluid particle types
    std::set<Particle::TypeEnum> fluidtypes_;

    //! set of boundary particle types
    std::set<Particle::TypeEnum> boundarytypes_;
  };

  class SPHBoundaryParticleAdami : public SPHBoundaryParticleBase
  {
   public:
    //! constructor
    explicit SPHBoundaryParticleAdami(const Teuchos::ParameterList& params);

    //! setup boundary particle handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs) override;

    //! init boundary particle states
    void init_boundary_particle_states(std::vector<double>& gravity) override;

   private:
    //! modified states of ghosted boundary particles to refresh
    Particle::StatesOfTypesToRefresh boundarystatestorefresh_;

    //! contributions of neighboring particles
    std::vector<std::vector<double>> sumj_wij_;
    std::vector<std::vector<double>> sumj_press_j_wij_;
    std::vector<std::vector<std::vector<double>>> sumj_dens_j_r_ij_wij_;
    std::vector<std::vector<std::vector<double>>> sumj_vel_j_wij_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
