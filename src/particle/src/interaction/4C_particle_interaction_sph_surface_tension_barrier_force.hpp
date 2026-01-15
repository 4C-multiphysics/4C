// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_SURFACE_TENSION_BARRIER_FORCE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_SURFACE_TENSION_BARRIER_FORCE_HPP

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
  class SPHBarrierForce
  {
   public:
    //! constructor
    explicit SPHBarrierForce(const Teuchos::ParameterList& params);

    //! setup barrier force handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs);

    //! compute barrier force contribution
    void compute_barrier_force_contribution() const;

   protected:
    //! compute barrier force contribution (particle contribution)
    void compute_barrier_force_particle_contribution() const;

    //! compute barrier force contribution (particle-boundary contribution)
    void compute_barrier_force_particle_boundary_contribution() const;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! liquid particle type
    Particle::TypeEnum liquidtype_;

    //! gas particle type
    Particle::TypeEnum gastype_;

    //! set of fluid particle types
    std::set<Particle::TypeEnum> fluidtypes_;

    //! set of boundary particle types
    std::set<Particle::TypeEnum> boundarytypes_;

    //! barrier force distance
    const double dist_;

    //! barrier force temperature scaling
    const double cr_;

    //! transition reference temperature
    const double trans_ref_temp_;

    //! transition temperature difference for barrier force evaluation
    const double trans_dT_barrier_;

    //! barrier force stiffness of heavy phase
    const double stiff_h_;

    //! barrier force damping parameter of heavy phase
    const double damp_h_;

    //! barrier force stiffness of gas phase
    const double stiff_g_;

    //! barrier force damping parameter of gas phase
    const double damp_g_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
