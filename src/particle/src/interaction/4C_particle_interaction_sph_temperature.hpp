// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_TEMPERATURE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_TEMPERATURE_HPP

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
  class SPHHeatSourceBase;
  class SPHHeatLossEvaporation;
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
  class SPHTemperature final
  {
   public:
    //! constructor
    explicit SPHTemperature(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~SPHTemperature();

    //! init temperature handler
    void init();

    //! setup temperature handler
    void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
        const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs);

    //! set current time
    void set_current_time(const double currenttime);

    //! set current step size
    void set_current_step_size(const double currentstepsize);

    //! insert temperature evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>&
            particlestatestotypes) const;

    //! compute temperature field using energy equation
    void compute_temperature() const;

   private:
    //! init heat source handler
    void init_heat_source_handler();

    //! init evaporation induced heat loss handler
    void init_heat_loss_evaporation_handler();

    //! evaluate energy equation
    void energy_equation() const;

    //! evaluate temperature gradient
    void temperature_gradient() const;

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

    //! heat source handler
    std::unique_ptr<ParticleInteraction::SPHHeatSourceBase> heatsource_;

    //! evaporation induced heat loss handler
    std::unique_ptr<ParticleInteraction::SPHHeatLossEvaporation> heatlossevaporation_;

    //! temperature of ghosted particles to refresh
    PARTICLEENGINE::StatesOfTypesToRefresh temptorefresh_;

    //! current time
    double time_;

    //! time step size
    double dt_;

    //! evaluate temperature gradient
    bool temperaturegradient_;

    //! pointer to thermo material of particle types
    std::vector<const Mat::PAR::ParticleMaterialThermo*> thermomaterial_;

    //! set of integrated thermo particle types
    std::set<PARTICLEENGINE::TypeEnum> intthermotypes_;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
