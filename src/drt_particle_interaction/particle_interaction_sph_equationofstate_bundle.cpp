/*---------------------------------------------------------------------------*/
/*! \file
\brief class holding all equation of state handlers
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "particle_interaction_sph_equationofstate_bundle.H"

#include "particle_interaction_sph_equationofstate.H"
#include "particle_interaction_material_handler.H"

#include "inpar_particle.H"

#include "drt_dserror.H"

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHEquationOfStateBundle::SPHEquationOfStateBundle(
    const Teuchos::ParameterList& params)
    : params_sph_(params)
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHEquationOfStateBundle::Init(
    const std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial)
{
  // get type of smoothed particle hydrodynamics equation of state
  INPAR::PARTICLE::EquationOfStateType equationofstatetype =
      DRT::INPUT::IntegralValue<INPAR::PARTICLE::EquationOfStateType>(
          params_sph_, "EQUATIONOFSTATE");

  // determine size of vector indexed by particle types
  const int typevectorsize = *(--particlematerial->GetParticleTypes().end()) + 1;

  // allocate memory to hold particle types
  phasetypetoequationofstate_.resize(typevectorsize);

  // iterate over particle types
  for (const auto& type_i : particlematerial->GetParticleTypes())
  {
    // no equation of state for boundary or rigid particles
    if (type_i == PARTICLEENGINE::BoundaryPhase or type_i == PARTICLEENGINE::RigidPhase) continue;

    // add to set of particle types of stored equation of state handlers
    storedtypes_.insert(type_i);

    // get material for current particle type
    const MAT::PAR::ParticleMaterialSPHFluid* material =
        dynamic_cast<const MAT::PAR::ParticleMaterialSPHFluid*>(
            particlematerial->GetPtrToParticleMatParameter(type_i));

    // create equation of state handler
    switch (equationofstatetype)
    {
      case INPAR::PARTICLE::GenTait:
      {
        const double speedofsound = material->SpeedOfSound();
        const double refdensfac = material->refDensFac_;
        const double exponent = material->exponent_;

        phasetypetoequationofstate_[type_i] =
            std::unique_ptr<PARTICLEINTERACTION::SPHEquationOfStateGenTait>(
                new PARTICLEINTERACTION::SPHEquationOfStateGenTait(
                    speedofsound, refdensfac, exponent));
        break;
      }
      case INPAR::PARTICLE::IdealGas:
      {
        const double speedofsound = material->SpeedOfSound();

        phasetypetoequationofstate_[type_i] =
            std::unique_ptr<PARTICLEINTERACTION::SPHEquationOfStateIdealGas>(
                new PARTICLEINTERACTION::SPHEquationOfStateIdealGas(speedofsound));
        break;
      }
      default:
      {
        dserror("unknown equation of state type!");
        break;
      }
    }

    // init equation of state handler
    phasetypetoequationofstate_[type_i]->Init();
  }
}

void PARTICLEINTERACTION::SPHEquationOfStateBundle::Setup()
{
  for (PARTICLEENGINE::TypeEnum type_i : storedtypes_) phasetypetoequationofstate_[type_i]->Setup();
}
