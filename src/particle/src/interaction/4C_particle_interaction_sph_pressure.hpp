// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_PRESSURE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_PRESSURE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class MaterialHandler;
  class ParticleContainerBundle;
  class ParticleEngineInterface;
  class SPHEquationOfStateBundle;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHPressure final
  {
   public:
    //! constructor
    explicit SPHPressure();

    //! setup pressure handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle);

    //! compute pressure using equation of state and density
    void compute_pressure() const;

   protected:
    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<Particle::MaterialHandler> particlematerial_;

    //! equation of state bundle
    std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle_;

    //! pressure of ghosted particles to refresh
    Particle::StatesOfTypesToRefresh pressuretorefresh_;

    //! set of fluid particle types
    std::set<Particle::TypeEnum> fluidtypes_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
