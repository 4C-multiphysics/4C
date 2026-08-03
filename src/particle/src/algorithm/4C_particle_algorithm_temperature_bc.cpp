// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_temperature_bc.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::TemperatureBoundaryConditionHandler::TemperatureBoundaryConditionHandler(
    const Teuchos::ParameterList& params)
    : params_(params)
{
  // get control parameters for conditions
  const Teuchos::ParameterList& params_conditions =
      params_.sublist("INITIAL AND BOUNDARY CONDITIONS");

  // read parameters relating particle types to values
  ParticleUtils::read_params_types_related_to_values(
      params_conditions, "TEMPERATURE_BOUNDARY_CONDITION", temperaturebctypetofunctid_);

  // iterate over particle types and insert into set
  for (auto& typeIt : temperaturebctypetofunctid_)
    typessubjectedtotemperaturebc_.insert(typeIt.first);
}

void Particle::TemperatureBoundaryConditionHandler::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;
}

void Particle::TemperatureBoundaryConditionHandler::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types subjected to temperature boundary conditions
  for (auto& particleType : typessubjectedtotemperaturebc_)
  {
    // insert states for types subjected to temperature boundary conditions
    particlestatestotypes[particleType].insert(Particle::ReferencePosition);
  }
}

void Particle::TemperatureBoundaryConditionHandler::set_particle_reference_position() const
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over particle types subjected to temperature boundary conditions
  for (auto& particleType : typessubjectedtotemperaturebc_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // set particle reference position
    container->update_state(0.0, Particle::ReferencePosition, 1.0, Particle::Position);
  }
}

void Particle::TemperatureBoundaryConditionHandler::evaluate_temperature_boundary_condition(
    const double& evaltime) const
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // iterate over particle types subjected to temperature boundary conditions
  for (auto& typeIt : temperaturebctypetofunctid_)
  {
    // get type of particles
    Particle::TypeEnum particleType = typeIt.first;

    // get container of owned particles of current particle type
    Particle::ParticleContainer* container =
        particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get id of function
    const int functid = typeIt.second;

    // get reference to function
    const auto& function =
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(functid);

    // get pointer to particle states
    const double* refpos = container->get_ptr_to_state(Particle::ReferencePosition, 0);
    double* temp = container->get_ptr_to_state_writable(Particle::Temperature, 0);

    // get particle state dimension
    int statedim = container->get_state_dim(Particle::Position);

    // safety check
    if (function.number_components() != 1)
      FOUR_C_THROW("dimension of function defining temperature boundary condition is not one!");

    // iterate over owned particles of current type
    for (int i = 0; i < particlestored; ++i)
    {
      // evaluate function
      temp[i] = function.evaluate(std::span(&(refpos[statedim * i]), 3), evaltime, 0);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
