/*---------------------------------------------------------------------------*/
/*! \file
\brief temperature handler for smoothed particle hydrodynamics (SPH) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef BACI_PARTICLE_INTERACTION_SPH_TEMPERATURE_HPP
#define BACI_PARTICLE_INTERACTION_SPH_TEMPERATURE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_config.hpp"

#include "baci_inpar_particle.hpp"
#include "baci_particle_engine_enums.hpp"
#include "baci_particle_engine_typedefs.hpp"

BACI_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace PARTICLEENGINE
{
  class ParticleEngineInterface;
  class ParticleContainerBundle;
}  // namespace PARTICLEENGINE

namespace PARTICLEINTERACTION
{
  class MaterialHandler;
  class SPHNeighborPairs;
  class SPHHeatSourceBase;
  class SPHHeatLossEvaporation;
}  // namespace PARTICLEINTERACTION

namespace MAT
{
  namespace PAR
  {
    class ParticleMaterialThermo;
  }
}  // namespace MAT

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace PARTICLEINTERACTION
{
  class SPHTemperature final
  {
   public:
    //! constructor
    explicit SPHTemperature(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     * \author Sebastian Fuchs \date 10/2018
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~SPHTemperature();

    //! init temperature handler
    void Init();

    //! setup temperature handler
    void Setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial,
        const std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs);

    //! set current time
    void SetCurrentTime(const double currenttime);

    //! set current step size
    void SetCurrentStepSize(const double currentstepsize);

    //! insert temperature evaluation dependent states
    void InsertParticleStatesOfParticleTypes(
        std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>&
            particlestatestotypes) const;

    //! compute temperature field using energy equation
    void ComputeTemperature() const;

   private:
    //! init heat source handler
    void InitHeatSourceHandler();

    //! init evaporation induced heat loss handler
    void InitHeatLossEvaporationHandler();

    //! evaluate energy equation
    void EnergyEquation() const;

    //! evaluate temperature gradient
    void TemperatureGradient() const;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial_;

    //! neighbor pair handler
    std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs_;

    //! heat source handler
    std::unique_ptr<PARTICLEINTERACTION::SPHHeatSourceBase> heatsource_;

    //! evaporation induced heat loss handler
    std::unique_ptr<PARTICLEINTERACTION::SPHHeatLossEvaporation> heatlossevaporation_;

    //! temperature of ghosted particles to refresh
    PARTICLEENGINE::StatesOfTypesToRefresh temptorefresh_;

    //! current time
    double time_;

    //! time step size
    double dt_;

    //! evaluate temperature gradient
    bool temperaturegradient_;

    //! pointer to thermo material of particle types
    std::vector<const MAT::PAR::ParticleMaterialThermo*> thermomaterial_;

    //! set of integrated thermo particle types
    std::set<PARTICLEENGINE::TypeEnum> intthermotypes_;
  };

}  // namespace PARTICLEINTERACTION

/*---------------------------------------------------------------------------*/
BACI_NAMESPACE_CLOSE

#endif
