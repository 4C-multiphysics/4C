// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_HEATSOURCE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_HEATSOURCE_HPP

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

namespace ParticleInteraction
{
  class MaterialHandler;
  class SPHNeighborPairs;
}  // namespace ParticleInteraction

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
namespace ParticleInteraction
{
  class SPHHeatSourceBase
  {
   public:
    //! constructor
    explicit SPHHeatSourceBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHHeatSourceBase() = default;

    //! init heat source handler
    virtual void init();

    //! setup heat source handler
    virtual void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
        const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs);

    //! evaluate heat source
    virtual void evaluate_heat_source(const double& evaltime) const = 0;

   protected:
    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial_;

    //! neighbor pair handler
    std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs_;

    //! pointer to thermo material of particle types
    std::vector<const Mat::PAR::ParticleMaterialThermo*> thermomaterial_;

    //! heat source function number
    const int heatsourcefctnumber_;

    //! set of absorbing particle types
    std::set<PARTICLEENGINE::TypeEnum> absorbingtypes_;

    //! set of non-absorbing particle types
    std::set<PARTICLEENGINE::TypeEnum> nonabsorbingtypes_;
  };

  class SPHHeatSourceVolume : public SPHHeatSourceBase
  {
   public:
    //! constructor
    explicit SPHHeatSourceVolume(const Teuchos::ParameterList& params);

    //! evaluate heat source
    void evaluate_heat_source(const double& evaltime) const override;
  };

  class SPHHeatSourceSurface : public SPHHeatSourceBase
  {
   public:
    //! constructor
    explicit SPHHeatSourceSurface(const Teuchos::ParameterList& params);

    //! init heat source handler
    void init() override;

    //! evaluate heat source
    void evaluate_heat_source(const double& evaltime) const override;

   private:
    //! heat source direction vector
    std::vector<double> direction_;

    //! evaluate heat source direction
    bool eval_direction_;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
