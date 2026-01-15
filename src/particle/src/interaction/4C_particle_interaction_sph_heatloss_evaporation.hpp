// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_HEATLOSS_EVAPORATION_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_HEATLOSS_EVAPORATION_HPP

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
  class MaterialHandler;
  class ParticleContainerBundle;
  class ParticleEngineInterface;
}  // namespace Particle

namespace Mat
{
  namespace PAR
  {
    class ParticleMaterialThermo;
  }
}  // namespace Mat

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHHeatLossEvaporation
  {
   public:
    //! constructor
    explicit SPHHeatLossEvaporation(const Teuchos::ParameterList& params);

    //! setup evaporation induced heat loss handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial);

    //! evaluate evaporation induced heat loss
    void evaluate_evaporation_induced_heat_loss() const;

   protected:
    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<Particle::MaterialHandler> particlematerial_;

    //! pointer to thermo material of particle types
    std::vector<const Mat::PAR::ParticleMaterialThermo*> thermomaterial_;

    //! evaporating phase
    Particle::TypeEnum evaporatingphase_;

    //! boiling temperature in recoil pressure formula
    double recoilboilingtemp_;

    //! pressure factor in recoil pressure formula
    double recoil_pfac_;

    //! temperature factor in recoil pressure formula
    double recoil_tfac_;

    //! latent heat in heat loss formula
    double latentheat_;

    //! enthalpy reference temperature in heat loss formula
    double enthalpyreftemp_;

    //! pressure factor in heat loss formula
    double heatloss_pfac_;

    //! temperature factor in heat loss formula
    double heatloss_tfac_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
