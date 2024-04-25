/*---------------------------------------------------------------------------*/
/*! \file
\brief particle input generator for particle simulations
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_particle_algorithm_input_generator.hpp"

#include "4C_particle_engine_object.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEALGORITHM::InputGenerator::InputGenerator(
    const Epetra_Comm& comm, const Teuchos::ParameterList& params)
    : myrank_(comm.MyPID()), params_(params)
{
  // empty constructor
}

void PARTICLEALGORITHM::InputGenerator::Init()
{
  // nothing to do
}

void PARTICLEALGORITHM::InputGenerator::GenerateParticles(
    std::vector<PARTICLEENGINE::ParticleObjShrdPtr>& particlesgenerated) const
{
  // generate initial particles
}

void PARTICLEALGORITHM::InputGenerator::AddGeneratedParticle(const std::vector<double>& position,
    const PARTICLEENGINE::TypeEnum particletype,
    std::vector<PARTICLEENGINE::ParticleObjShrdPtr>& particlesgenerated) const
{
  // safety check
  if (position.size() != 3)
    FOUR_C_THROW("particle can not be generated since position vector needs three entries!");

  // allocate memory to hold particle states
  PARTICLEENGINE::ParticleStates particlestates;
  particlestates.assign((PARTICLEENGINE::Position + 1), std::vector<double>(0));

  // set position state
  particlestates[PARTICLEENGINE::Position] = position;

  // construct and store generated particle object
  particlesgenerated.emplace_back(
      std::make_shared<PARTICLEENGINE::ParticleObject>(particletype, -1, particlestates));
}

FOUR_C_NAMESPACE_CLOSE
