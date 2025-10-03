// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_MOMENTUM_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_MOMENTUM_HPP

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
namespace PARTICLEENGINE
{
  class ParticleEngineInterface;
  class ParticleContainerBundle;
}  // namespace PARTICLEENGINE

namespace PARTICLEWALL
{
  class WallHandlerInterface;
}

namespace ParticleInteraction
{
  class SPHKernelBase;
  class MaterialHandler;
  class InteractionWriter;
  class SPHEquationOfStateBundle;
  class SPHNeighborPairs;
  class SPHVirtualWallParticle;
  class SPHMomentumFormulationBase;
  class SPHArtificialViscosity;
}  // namespace ParticleInteraction

namespace Mat
{
  namespace PAR
  {
    class ParticleMaterialSPHFluid;
  }
}  // namespace Mat

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHMomentum final
  {
   public:
    //! constructor
    explicit SPHMomentum(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~SPHMomentum();

    //! init momentum handler
    void init();

    //! setup momentum handler
    void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel,
        const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
        const std::shared_ptr<ParticleInteraction::InteractionWriter> particleinteractionwriter,
        const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs,
        const std::shared_ptr<ParticleInteraction::SPHVirtualWallParticle> virtualwallparticle);

    //! insert momentum evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>&
            particlestatestotypes) const;

    //! add momentum contribution to acceleration field
    void add_acceleration_contribution() const;

   private:
    //! init momentum formulation handler
    void init_momentum_formulation_handler();

    //! init artificial viscosity handler
    void init_artificial_viscosity_handler();

    //! setup particle interaction writer
    void setup_particle_interaction_writer();

    //! momentum equation (particle contribution)
    void momentum_equation_particle_contribution() const;

    //! momentum equation (particle-boundary contribution)
    void momentum_equation_particle_boundary_contribution() const;

    //! momentum equation (particle-wall contribution)
    void momentum_equation_particle_wall_contribution() const;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface_;

    //! kernel handler
    std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel_;

    //! particle material handler
    std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial_;

    //! particle interaction writer
    std::shared_ptr<ParticleInteraction::InteractionWriter> particleinteractionwriter_;

    //! equation of state bundle
    std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle_;

    //! neighbor pair handler
    std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs_;

    //! virtual wall particle handler
    std::shared_ptr<ParticleInteraction::SPHVirtualWallParticle> virtualwallparticle_;

    //! momentum formulation handler
    std::unique_ptr<ParticleInteraction::SPHMomentumFormulationBase> momentumformulation_;

    //! artificial viscosity handler
    std::unique_ptr<ParticleInteraction::SPHArtificialViscosity> artificialviscosity_;

    //! type of boundary particle interaction
    PARTICLE::BoundaryParticleInteraction boundaryparticleinteraction_;

    //! type of transport velocity formulation
    PARTICLE::TransportVelocityFormulation transportvelocityformulation_;

    //! pointer to fluid material of particle types
    std::vector<const Mat::PAR::ParticleMaterialSPHFluid*> fluidmaterial_;

    //! write particle-wall interaction output
    const bool writeparticlewallinteraction_;

    //! set of all fluid particle types
    std::set<PARTICLEENGINE::TypeEnum> allfluidtypes_;

    //! set of integrated fluid particle types
    std::set<PARTICLEENGINE::TypeEnum> intfluidtypes_;

    //! set of pure fluid particle types
    std::set<PARTICLEENGINE::TypeEnum> purefluidtypes_;

    //! set of boundary particle types
    std::set<PARTICLEENGINE::TypeEnum> boundarytypes_;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
