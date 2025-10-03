// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_temperature.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_heatloss_evaporation.hpp"
#include "4C_particle_interaction_sph_heatsource.hpp"
#include "4C_particle_interaction_sph_kernel.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::SPHTemperature::SPHTemperature(const Teuchos::ParameterList& params)
    : params_sph_(params),
      time_(0.0),
      dt_(0.0),
      temperaturegradient_(params_sph_.get<bool>("TEMPERATUREGRADIENT"))
{
  // empty constructor
}

ParticleInteraction::SPHTemperature::~SPHTemperature() = default;

void ParticleInteraction::SPHTemperature::init()
{
  // init heat source handler
  init_heat_source_handler();

  // init evaporation induced heat loss handler
  init_heat_loss_evaporation_handler();

  // init with potential integrated thermo particle types
  intthermotypes_ = {PARTICLEENGINE::Phase1, PARTICLEENGINE::Phase2, PARTICLEENGINE::RigidPhase};
}

void ParticleInteraction::SPHTemperature::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
    const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // update with actual integrated thermo particle types
  const auto intthermotypes = intthermotypes_;
  for (const auto& type_i : intthermotypes)
    if (not particlecontainerbundle_->get_particle_types().count(type_i))
      intthermotypes_.erase(type_i);

  // setup temperature of ghosted particles to refresh
  {
    std::vector<PARTICLEENGINE::StateEnum> states{PARTICLEENGINE::Temperature};

    for (const auto& type_i : intthermotypes_)
      temptorefresh_.push_back(std::make_pair(type_i, states));
  }

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  // allocate memory to hold particle types
  thermomaterial_.resize(typevectorsize);

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    thermomaterial_[type_i] = dynamic_cast<const Mat::PAR::ParticleMaterialThermo*>(
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i));

    // safety check
    if (not(thermomaterial_[type_i]->thermalCapacity_ > 0.0))
      FOUR_C_THROW("thermal capacity for particles of type '{}' not positive!",
          PARTICLEENGINE::enum_to_type_name(type_i));
  }

  // setup heat source handler
  if (heatsource_) heatsource_->setup(particleengineinterface, particlematerial, neighborpairs);

  // setup evaporation induced heat loss handler
  if (heatlossevaporation_) heatlossevaporation_->setup(particleengineinterface, particlematerial);
}

void ParticleInteraction::SPHTemperature::set_current_time(const double currenttime)
{
  time_ = currenttime;
}

void ParticleInteraction::SPHTemperature::set_current_step_size(const double currentstepsize)
{
  dt_ = currentstepsize;
}

void ParticleInteraction::SPHTemperature::insert_particle_states_of_particle_types(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
    const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum type = typeIt.first;

    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    // set temperature state
    particlestates.insert(PARTICLEENGINE::Temperature);

    // current particle type is not an integrated thermo particle type
    if (not intthermotypes_.count(type)) continue;

    // state for temperature evaluation scheme
    particlestates.insert(PARTICLEENGINE::TemperatureDot);

    // state for temperature gradient evaluation
    if (temperaturegradient_) particlestates.insert(PARTICLEENGINE::temperature_gradient);
  }
}

void ParticleInteraction::SPHTemperature::compute_temperature() const
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::SPHTemperature::ComputeTemperature");

  // evaluate energy equation
  energy_equation();

  // evaluate heat source
  if (heatsource_) heatsource_->evaluate_heat_source(time_);

  // evaluate evaporation induced heat loss
  if (heatlossevaporation_) heatlossevaporation_->evaluate_evaporation_induced_heat_loss();

  // iterate over integrated thermo particle types
  for (const auto& type_i : intthermotypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // update temperature of all particles
    container_i->update_state(
        1.0, PARTICLEENGINE::Temperature, dt_, PARTICLEENGINE::TemperatureDot);
  }

  // refresh temperature of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(temptorefresh_);

  // evaluate temperature gradient
  if (temperaturegradient_) temperature_gradient();
}

void ParticleInteraction::SPHTemperature::init_heat_source_handler()
{
  // get type of heat source
  auto heatsourcetype =
      Teuchos::getIntegralValue<PARTICLE::HeatSourceType>(params_sph_, "HEATSOURCETYPE");

  // create heat source handler
  switch (heatsourcetype)
  {
    case PARTICLE::NoHeatSource:
    {
      heatsource_ = std::unique_ptr<ParticleInteraction::SPHHeatSourceBase>(nullptr);
      break;
    }
    case PARTICLE::VolumeHeatSource:
    {
      heatsource_ = std::unique_ptr<ParticleInteraction::SPHHeatSourceVolume>(
          new ParticleInteraction::SPHHeatSourceVolume(params_sph_));
      break;
    }
    case PARTICLE::SurfaceHeatSource:
    {
      heatsource_ = std::unique_ptr<ParticleInteraction::SPHHeatSourceSurface>(
          new ParticleInteraction::SPHHeatSourceSurface(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown type of heat source!");
      break;
    }
  }

  // init heat source handler
  if (heatsource_) heatsource_->init();
}

void ParticleInteraction::SPHTemperature::init_heat_loss_evaporation_handler()
{
  if (params_sph_.get<bool>("VAPOR_HEATLOSS"))
    heatlossevaporation_ = std::unique_ptr<ParticleInteraction::SPHHeatLossEvaporation>(
        new ParticleInteraction::SPHHeatLossEvaporation(params_sph_));

  // init evaporation induced heat loss handler
  if (heatlossevaporation_) heatlossevaporation_->init();
}

void ParticleInteraction::SPHTemperature::energy_equation() const
{
  // iterate over integrated thermo particle types
  for (const auto& type_i : intthermotypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // clear temperature dot state
    container_i->clear_state(PARTICLEENGINE::TemperatureDot);
  }

  // iterate over particle pairs
  for (auto& particlepair : neighborpairs_->get_ref_to_particle_pair_data())
  {
    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get material for particle types
    const Mat::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);
    const Mat::PAR::ParticleMaterialBase* basematerial_j =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

    const Mat::PAR::ParticleMaterialThermo* thermomaterial_i = thermomaterial_[type_i];
    const Mat::PAR::ParticleMaterialThermo* thermomaterial_j = thermomaterial_[type_j];

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);

    const double* dens_i = (container_i->have_stored_state(PARTICLEENGINE::Density))
                               ? container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i)
                               : &(basematerial_i->initDensity_);

    const double* temp_i = container_i->get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);
    double* tempdot_i =
        container_i->cond_get_ptr_to_state(PARTICLEENGINE::TemperatureDot, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);

    const double* dens_j = (container_j->have_stored_state(PARTICLEENGINE::Density))
                               ? container_j->get_ptr_to_state(PARTICLEENGINE::Density, particle_j)
                               : &(basematerial_j->initDensity_);

    const double* temp_j = container_j->get_ptr_to_state(PARTICLEENGINE::Temperature, particle_j);
    double* tempdot_j =
        container_j->cond_get_ptr_to_state(PARTICLEENGINE::TemperatureDot, particle_j);

    // thermal conductivities
    const double& k_i = thermomaterial_i->thermalConductivity_;
    const double& k_j = thermomaterial_j->thermalConductivity_;

    // factor containing effective conductivity and temperature difference
    const double fac = (4.0 * k_i * k_j) * (temp_i[0] - temp_j[0]) /
                       (dens_i[0] * dens_j[0] * (k_i + k_j) * particlepair.absdist_);

    // sum contribution of neighboring particle j
    if (tempdot_i)
      tempdot_i[0] +=
          mass_j[0] * fac * particlepair.dWdrij_ * thermomaterial_i->invThermalCapacity_;

    // sum contribution of neighboring particle i
    if (tempdot_j and status_j == PARTICLEENGINE::Owned)
      tempdot_j[0] -=
          mass_i[0] * fac * particlepair.dWdrji_ * thermomaterial_j->invThermalCapacity_;
  }
}

void ParticleInteraction::SPHTemperature::temperature_gradient() const
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::SPHTemperature::temperature_gradient");

  // iterate over integrated thermo particle types
  for (const auto& type_i : intthermotypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // clear temperature gradient state
    container_i->clear_state(PARTICLEENGINE::temperature_gradient);
  }

  // iterate over particle pairs
  for (auto& particlepair : neighborpairs_->get_ref_to_particle_pair_data())
  {
    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get material for particle types
    const Mat::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);
    const Mat::PAR::ParticleMaterialBase* basematerial_j =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);

    const double* dens_i = (container_i->have_stored_state(PARTICLEENGINE::Density))
                               ? container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i)
                               : &(basematerial_i->initDensity_);

    const double* temp_i = container_i->get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);
    double* tempgrad_i =
        container_i->cond_get_ptr_to_state(PARTICLEENGINE::temperature_gradient, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);

    const double* dens_j = (container_j->have_stored_state(PARTICLEENGINE::Density))
                               ? container_j->get_ptr_to_state(PARTICLEENGINE::Density, particle_j)
                               : &(basematerial_j->initDensity_);

    const double* temp_j = container_j->get_ptr_to_state(PARTICLEENGINE::Temperature, particle_j);
    double* tempgrad_j =
        container_j->cond_get_ptr_to_state(PARTICLEENGINE::temperature_gradient, particle_j);

    const double temp_ji = temp_j[0] - temp_i[0];

    // sum contribution of neighboring particle j
    if (tempgrad_i)
      Utils::vec_add_scale(
          tempgrad_i, (mass_j[0] / dens_j[0]) * temp_ji * particlepair.dWdrij_, particlepair.e_ij_);

    // sum contribution of neighboring particle i
    if (tempgrad_j and status_j == PARTICLEENGINE::Owned)
      Utils::vec_add_scale(
          tempgrad_j, (mass_i[0] / dens_i[0]) * temp_ji * particlepair.dWdrji_, particlepair.e_ij_);
  }
}

FOUR_C_NAMESPACE_CLOSE
