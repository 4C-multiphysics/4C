#include "4C_particle_interaction_sph_pressure.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::SPHPressure::SPHPressure()
{
  // empty constructor
}

void ParticleInteraction::SPHPressure::init()
{
  // init with potential fluid particle types
  fluidtypes_ = {PARTICLEENGINE::Phase1, PARTICLEENGINE::Phase2};
}

void ParticleInteraction::SPHPressure::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
    const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // set equation of state handler
  equationofstatebundle_ = equationofstatebundle;

  // update with actual fluid particle types
  const auto fluidtypes = fluidtypes_;
  for (const auto& type_i : fluidtypes)
    if (not particlecontainerbundle_->get_particle_types().count(type_i)) fluidtypes_.erase(type_i);

  // setup pressure of ghosted particles to refresh
  {
    std::vector<PARTICLEENGINE::StateEnum> states{PARTICLEENGINE::Pressure};

    for (const auto& type_i : fluidtypes_)
      pressuretorefresh_.push_back(std::make_pair(type_i, states));
  }
}

void ParticleInteraction::SPHPressure::compute_pressure() const
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::SPHPressure::ComputePressure");

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get pointer to particle state
    const double* dens = container->get_ptr_to_state(PARTICLEENGINE::Density, 0);
    double* press = container->get_ptr_to_state(PARTICLEENGINE::Pressure, 0);

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* material =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // get equation of state for current particle type
    const ParticleInteraction::SPHEquationOfStateBase* equationofstate =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // iterate over owned particles of current type
    for (int i = 0; i < particlestored; ++i)
      press[i] = equationofstate->density_to_pressure(dens[i], material->initDensity_);
  }

  // refresh pressure of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(pressuretorefresh_);
}

FOUR_C_NAMESPACE_CLOSE
