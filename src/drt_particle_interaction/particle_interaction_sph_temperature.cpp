/*---------------------------------------------------------------------------*/
/*!
\file particle_interaction_sph_temperature.cpp

\brief temperature handler for smoothed particle hydrodynamics (SPH) interactions

\level 3

\maintainer  Sebastian Fuchs
             fuchs@lnm.mw.tum.de
             http://www.lnm.mw.tum.de
             089 - 289 -15262

*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                     meier 09/2018 |
 *---------------------------------------------------------------------------*/
#include "particle_interaction_sph_temperature.H"

#include "particle_interaction_sph_kernel.H"
#include "particle_interaction_material_handler.H"
#include "particle_interaction_sph_neighbor_pairs.H"

#include "../drt_particle_engine/particle_engine_interface.H"
#include "../drt_particle_engine/particle_container.H"

#include "../drt_lib/drt_dserror.H"

/*---------------------------------------------------------------------------*
 | constructor                                                 meier 09/2018 |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHTemperatureBase::SPHTemperatureBase(const Teuchos::ParameterList& params)
    : params_sph_(params), dt_(0.0)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | init temperature handler                                    meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureBase::Init()
{
  // nothing to do
}

/*---------------------------------------------------------------------------*
 | setup temperature handler                                   meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureBase::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial,
    const std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->GetParticleContainerBundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;
}

/*---------------------------------------------------------------------------*
 | write restart of temperature handler                        meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureBase::WriteRestart(const int step, const double time) const
{
  // nothing to do
}

/*---------------------------------------------------------------------------*
 | read restart of temperature handler                         meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureBase::ReadRestart(
    const std::shared_ptr<IO::DiscretizationReader> reader)
{
  // nothing to do
}

/*---------------------------------------------------------------------------*
 | set current step size                                       meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureBase::SetCurrentStepSize(const double currentstepsize)
{
  dt_ = currentstepsize;
}

/*---------------------------------------------------------------------------*
 | refresh temperature of ghosted particles                    meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureBase::RefreshTemperature() const
{
  // init map
  std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>> particlestatestotypes;

  // iterate over particle types
  for (auto& typeIt : particlecontainerbundle_->GetRefToAllContainersMap())
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum type = typeIt.first;

    // no refreshing of temperature states for boundary particles
    if (type == PARTICLEENGINE::BoundaryPhase) continue;

    // set state enums to map
    particlestatestotypes[type].insert(PARTICLEENGINE::Temperature);
  }

  // refresh specific states of particles of specific types
  particleengineinterface_->RefreshSpecificStatesOfParticlesOfSpecificTypes(particlestatestotypes);
}

/*---------------------------------------------------------------------------*
 | constructor                                                 meier 09/2018 |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHTemperatureIntegration::SPHTemperatureIntegration(
    const Teuchos::ParameterList& params)
    : SPHTemperatureBase::SPHTemperatureBase(params)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | insert temperature evaluation dependent states              meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureIntegration::InsertParticleStatesOfParticleTypes(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
    const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    // states for temperature evaluation scheme
    particlestates.insert(PARTICLEENGINE::Temperature);
    particlestates.insert(PARTICLEENGINE::TemperatureDot);
  }
}

/*---------------------------------------------------------------------------*
 | compute temperature field using energy equation             meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureIntegration::ComputeTemperature() const
{
  // evaluate energy equation
  EnergyEquation();

  // iterate over particle types
  for (auto& typeIt : particlecontainerbundle_->GetRefToAllContainersMap())
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum particleType = typeIt.first;

    // no temperature calculation for boundary particles
    if (particleType == PARTICLEENGINE::BoundaryPhase) continue;

    // update temperature of all particles
    particlecontainerbundle_->UpdateStateSpecificContainer(
        1.0, PARTICLEENGINE::Temperature, dt_, PARTICLEENGINE::TemperatureDot, particleType);
  }

  // refresh temperature of ghosted particles
  RefreshTemperature();
}

/*---------------------------------------------------------------------------*
 | evaluate energy equation                                    meier 09/2018 |
 *---------------------------------------------------------------------------*/
void PARTICLEINTERACTION::SPHTemperatureIntegration::EnergyEquation() const
{
  // iterate over particle types
  for (auto& typeIt : neighborpairs_->GetRefToNeighborPairsMap())
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum type_i = typeIt.first;

    // no temperature integration for boundary particles
    if (type_i == PARTICLEENGINE::BoundaryPhase) continue;

    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainerShrdPtr container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, PARTICLEENGINE::Owned);

    // get material for current particle type
    const MAT::PAR::ParticleMaterialThermo* thermomaterial_i =
        dynamic_cast<const MAT::PAR::ParticleMaterialThermo*>(
            particlematerial_->GetPtrToParticleMatParameter(type_i));

    // particles of current type with neighbors
    const auto& currparticles = typeIt.second;

    // iterate over particles of current type
    for (auto& particleIt : currparticles)
    {
      // get local index of particle i
      const int particle_i = particleIt.first;

      // declare pointer variables for particle i
      const double *dens_i, *temp_i;
      double* tempdot_i;

      // get pointer to particle states
      dens_i = container_i->GetPtrToParticleState(PARTICLEENGINE::Density, particle_i);
      temp_i = container_i->GetPtrToParticleState(PARTICLEENGINE::Temperature, particle_i);
      tempdot_i = container_i->GetPtrToParticleState(PARTICLEENGINE::TemperatureDot, particle_i);

      // initialize sum of evaluated kernel values for particle i due to neighbor particles j
      double sumj_tempdot_ij(0.0);

      // iterate over particle types of neighboring particles
      for (auto& neighborTypeIt : particleIt.second)
      {
        // get type of neighboring particles
        PARTICLEENGINE::TypeEnum type_j = neighborTypeIt.first;

        // get material for current particle type
        const MAT::PAR::ParticleMaterialThermo* thermomaterial_j = NULL;
        if (type_i == type_j)
          thermomaterial_j = thermomaterial_i;
        else
          thermomaterial_j = dynamic_cast<const MAT::PAR::ParticleMaterialThermo*>(
              particlematerial_->GetPtrToParticleMatParameter(type_j));

        const MAT::PAR::ParticleMaterialBase* basematerial_j = NULL;
        if (type_j == PARTICLEENGINE::BoundaryPhase)
          basematerial_j = particlematerial_->GetPtrToParticleMatParameter(type_j);

        // iterate over particle status of neighboring particles
        for (auto& neighborStatusIt : neighborTypeIt.second)
        {
          // get status of neighboring particles of current type
          PARTICLEENGINE::StatusEnum status_j = neighborStatusIt.first;

          // get container of neighboring particles of current particle type and state
          PARTICLEENGINE::ParticleContainerShrdPtr container_j =
              particlecontainerbundle_->GetSpecificContainer(type_j, status_j);

          // iterate over neighboring particles of current type and status
          for (auto& neighborParticleIt : neighborStatusIt.second)
          {
            // get local index of neighbor particle j
            const int particle_j = neighborParticleIt.first;

            // get reference to particle pair
            const ParticlePairSPH& particlepair = neighborParticleIt.second;

            // declare pointer variables for neighbor particle j
            const double *mass_j, *dens_j, *temp_j;

            // get pointer to particle states
            mass_j = container_j->GetPtrToParticleState(PARTICLEENGINE::Mass, particle_j);
            temp_j = container_j->GetPtrToParticleState(PARTICLEENGINE::Temperature, particle_j);

            if (type_j == PARTICLEENGINE::BoundaryPhase)
              dens_j = &(basematerial_j->initDensity_);
            else
              dens_j = container_j->GetPtrToParticleState(PARTICLEENGINE::Density, particle_j);

            // sum contribution of neighbor particle j
            sumj_tempdot_ij +=
                mass_j[0] / (dens_i[0] * dens_j[0]) *
                (thermomaterial_i->thermalConductivity_ + thermomaterial_j->thermalConductivity_) *
                (temp_i[0] - temp_j[0]) * (particlepair.dWdrij_ / particlepair.absdist_);
          }
        }
      }

      // add contributions of neighbor particles
      tempdot_i[0] = sumj_tempdot_ij;
    }
  }
}