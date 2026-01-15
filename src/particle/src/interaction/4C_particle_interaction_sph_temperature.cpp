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

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHTemperature::SPHTemperature(const Teuchos::ParameterList& params)
    : params_sph_(params),
      time_(0.0),
      dt_(0.0),
      temperaturegradient_(params_sph_.get<bool>("TEMPERATUREGRADIENT")),
      intthermotypes_({Particle::Phase1, Particle::Phase2, Particle::RigidPhase})
{
  init_heat_source_handler();

  if (params_sph_.get<bool>("VAPOR_HEATLOSS"))
    heatlossevaporation_ = std::make_unique<Particle::SPHHeatLossEvaporation>(params_sph_);
}

Particle::SPHTemperature::~SPHTemperature() = default;

void Particle::SPHTemperature::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs)
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
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      intthermotypes_.erase(type_i);

  // setup temperature of ghosted particles to refresh
  {
    std::vector<Particle::StateEnum> states{Particle::Temperature};

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
          Particle::enum_to_type_name(type_i));
  }

  // setup heat source handler
  if (heatsource_) heatsource_->setup(particleengineinterface, particlematerial, neighborpairs);

  // setup evaporation induced heat loss handler
  if (heatlossevaporation_) heatlossevaporation_->setup(particleengineinterface, particlematerial);
}

void Particle::SPHTemperature::set_current_time(const double currenttime) { time_ = currenttime; }

void Particle::SPHTemperature::set_current_step_size(const double currentstepsize)
{
  dt_ = currentstepsize;
}

void Particle::SPHTemperature::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    Particle::TypeEnum type = typeIt.first;

    // set of particle states for current particle type
    std::set<Particle::StateEnum>& particlestates = typeIt.second;

    // set temperature state
    particlestates.insert(Particle::Temperature);

    // current particle type is not an integrated thermo particle type
    if (not intthermotypes_.contains(type)) continue;

    // state for temperature evaluation scheme
    particlestates.insert(Particle::TemperatureDot);

    // state for temperature gradient evaluation
    if (temperaturegradient_) particlestates.insert(Particle::temperature_gradient);
  }
}

void Particle::SPHTemperature::compute_temperature() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHTemperature::ComputeTemperature");

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
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // update temperature of all particles
    container_i->update_state(1.0, Particle::Temperature, dt_, Particle::TemperatureDot);
  }

  // refresh temperature of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(temptorefresh_);

  // evaluate temperature gradient
  if (temperaturegradient_) temperature_gradient();
}

void Particle::SPHTemperature::init_heat_source_handler()
{
  // get type of heat source
  auto heatsourcetype =
      Teuchos::getIntegralValue<Particle::HeatSourceType>(params_sph_, "HEATSOURCETYPE");

  // create heat source handler
  switch (heatsourcetype)
  {
    case Particle::NoHeatSource:
    {
      heatsource_ = std::unique_ptr<Particle::SPHHeatSourceBase>(nullptr);
      break;
    }
    case Particle::VolumeHeatSource:
    {
      heatsource_ = std::unique_ptr<Particle::SPHHeatSourceVolume>(
          new Particle::SPHHeatSourceVolume(params_sph_));
      break;
    }
    case Particle::SurfaceHeatSource:
    {
      heatsource_ = std::unique_ptr<Particle::SPHHeatSourceSurface>(
          new Particle::SPHHeatSourceSurface(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown type of heat source!");
      break;
    }
  }
}

void Particle::SPHTemperature::energy_equation() const
{
  // iterate over integrated thermo particle types
  for (const auto& type_i : intthermotypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // clear temperature dot state
    container_i->clear_state(Particle::TemperatureDot);
  }

  // iterate over particle pairs
  for (auto& particlepair : neighborpairs_->get_ref_to_particle_pair_data())
  {
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get material for particle types
    const Mat::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);
    const Mat::PAR::ParticleMaterialBase* basematerial_j =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

    const Mat::PAR::ParticleMaterialThermo* thermomaterial_i = thermomaterial_[type_i];
    const Mat::PAR::ParticleMaterialThermo* thermomaterial_j = thermomaterial_[type_j];

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);

    const double* dens_i = (container_i->have_stored_state(Particle::Density))
                               ? container_i->get_ptr_to_state(Particle::Density, particle_i)
                               : &(basematerial_i->initDensity_);

    const double* temp_i = container_i->get_ptr_to_state(Particle::Temperature, particle_i);
    double* tempdot_i = container_i->cond_get_ptr_to_state(Particle::TemperatureDot, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(Particle::Mass, particle_j);

    const double* dens_j = (container_j->have_stored_state(Particle::Density))
                               ? container_j->get_ptr_to_state(Particle::Density, particle_j)
                               : &(basematerial_j->initDensity_);

    const double* temp_j = container_j->get_ptr_to_state(Particle::Temperature, particle_j);
    double* tempdot_j = container_j->cond_get_ptr_to_state(Particle::TemperatureDot, particle_j);

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
    if (tempdot_j and status_j == Particle::Owned)
      tempdot_j[0] -=
          mass_i[0] * fac * particlepair.dWdrji_ * thermomaterial_j->invThermalCapacity_;
  }
}

void Particle::SPHTemperature::temperature_gradient() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHTemperature::temperature_gradient");

  // iterate over integrated thermo particle types
  for (const auto& type_i : intthermotypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // clear temperature gradient state
    container_i->clear_state(Particle::temperature_gradient);
  }

  // iterate over particle pairs
  for (auto& particlepair : neighborpairs_->get_ref_to_particle_pair_data())
  {
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get material for particle types
    const Mat::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);
    const Mat::PAR::ParticleMaterialBase* basematerial_j =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);

    const double* dens_i = (container_i->have_stored_state(Particle::Density))
                               ? container_i->get_ptr_to_state(Particle::Density, particle_i)
                               : &(basematerial_i->initDensity_);

    const double* temp_i = container_i->get_ptr_to_state(Particle::Temperature, particle_i);
    double* tempgrad_i =
        container_i->cond_get_ptr_to_state(Particle::temperature_gradient, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(Particle::Mass, particle_j);

    const double* dens_j = (container_j->have_stored_state(Particle::Density))
                               ? container_j->get_ptr_to_state(Particle::Density, particle_j)
                               : &(basematerial_j->initDensity_);

    const double* temp_j = container_j->get_ptr_to_state(Particle::Temperature, particle_j);
    double* tempgrad_j =
        container_j->cond_get_ptr_to_state(Particle::temperature_gradient, particle_j);

    const double temp_ji = temp_j[0] - temp_i[0];

    // sum contribution of neighboring particle j
    if (tempgrad_i)
      ParticleUtils::vec_add_scale(
          tempgrad_i, (mass_j[0] / dens_j[0]) * temp_ji * particlepair.dWdrij_, particlepair.e_ij_);

    // sum contribution of neighboring particle i
    if (tempgrad_j and status_j == Particle::Owned)
      ParticleUtils::vec_add_scale(
          tempgrad_j, (mass_i[0] / dens_i[0]) * temp_ji * particlepair.dWdrji_, particlepair.e_ij_);
  }
}

FOUR_C_NAMESPACE_CLOSE
