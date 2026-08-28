// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_heatsource.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHHeatSourceBase::SPHHeatSourceBase(const Teuchos::ParameterList& params)
    : params_sph_(params), heatsourcefctnumber_(params.get<int>("HEATSOURCE_FUNCT"))
{
  // empty constructor
}

void Particle::SPHHeatSourceBase::setup(
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

  // determine size of vectors indexed by particle types
  const int typevectorsize =
      static_cast<int>(*(--particlecontainerbundle_->get_particle_types().end())) + 1;

  // allocate memory to hold particle types
  thermomaterial_.resize(typevectorsize);

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
    thermomaterial_[static_cast<int>(type_i)] =
        dynamic_cast<const Mat::PAR::ParticleMaterialThermo*>(
            particlematerial_->get_ptr_to_particle_mat_parameter(type_i));

  // set of potential absorbing particle types
  std::set<ParticleType> potentialabsorbingtypes = {
      ParticleType::Phase1, ParticleType::Phase2, ParticleType::RigidPhase, ParticleType::PDPhase};

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // determine absorbing particle types
    if (thermomaterial_[static_cast<int>(type_i)]->thermalAbsorptivity_ > 0.0)
    {
      // safety check
      if (not potentialabsorbingtypes.contains(type_i))
        FOUR_C_THROW("thermal absorptivity for particles of type '{}' not possible!",
            Particle::enum_to_type_name(type_i));

      absorbingtypes_.insert(type_i);
    }
    // determine non-absorbing particle types
    else if (potentialabsorbingtypes.contains(type_i))
    {
      nonabsorbingtypes_.insert(type_i);
    }
  }
}

Particle::SPHHeatSourceVolume::SPHHeatSourceVolume(const Teuchos::ParameterList& params)
    : Particle::SPHHeatSourceBase(params)
{
  // empty constructor
}

void Particle::SPHHeatSourceVolume::evaluate_heat_source(const double& evaltime) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHHeatSourceVolume::EvaluateHeatSource");

  // init vector containing evaluated function
  std::vector<double> funct(1);

  // get reference to function
  const auto& function =
      Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
          heatsourcefctnumber_);

  // safety check
  if (function.number_components() != 1)
    FOUR_C_THROW("dimension of function defining heat source is not one!");

  // iterate over absorbing particle types
  for (const auto& type_i : absorbingtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, ParticleStatus::Owned);

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    const Mat::PAR::ParticleMaterialThermo* thermomaterial_i =
        thermomaterial_[static_cast<int>(type_i)];

    // get pointers to particle states
    const int statedim = Particle::enum_to_state_dim(ParticleState::Position);
    const double* dens = container_i->try_get_ptr_to_state(ParticleState::Density);
    const double* pos = container_i->get_ptr_to_state(ParticleState::Position);
    double* tempdot = container_i->get_ptr_to_state_writable(ParticleState::TemperatureDot);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointers to particle states
      const double* dens_i = dens ? &dens[particle_i] : &(basematerial_i->initDensity_);
      const double* pos_i = &pos[particle_i * statedim];
      double* tempdot_i = &tempdot[particle_i];

      // evaluate function defining heat source
      funct = function.evaluate_time_derivative(std::span(pos_i, 3), evaltime, 0, 0);

      // add contribution of heat source
      tempdot_i[0] += thermomaterial_i->thermalAbsorptivity_ * funct[0] *
                      thermomaterial_i->invThermalCapacity_ / dens_i[0];
    }
  }
}

Particle::SPHHeatSourceSurface::SPHHeatSourceSurface(const Teuchos::ParameterList& params)
    : Particle::SPHHeatSourceBase(params), eval_direction_(false)
{
  init_heat_source_direction();
}

void Particle::SPHHeatSourceSurface::init_heat_source_direction()
{
  // init heat source direction vector
  double value;
  std::istringstream directionstream(
      Teuchos::getNumericStringParameter(params_sph_, "HEATSOURCE_DIRECTION"));

  while (directionstream >> value) direction_.push_back(value);

  // safety check
  if (static_cast<int>(direction_.size()) != 3)
    FOUR_C_THROW("dimension (dim = {}) of heat source direction vector is wrong!",
        static_cast<int>(direction_.size()));

  // normalize heat source direction vector
  const double direction_norm = ParticleUtils::vec_norm_two(direction_.data());
  if (direction_norm > 0.0)
  {
    eval_direction_ = true;
    ParticleUtils::vec_set_scale(direction_.data(), 1.0 / direction_norm, direction_.data());
  }
}

void Particle::SPHHeatSourceSurface::evaluate_heat_source(const double& evaltime) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHHeatSourceSurface::EvaluateHeatSource");

  // determine size of vectors indexed by particle types
  const int typevectorsize = static_cast<int>(*(--absorbingtypes_.end())) + 1;

  // colorfield gradient of absorbing interface particles
  std::vector<std::vector<std::vector<double>>> cfg_i(typevectorsize);

  // iterate over absorbing particle types
  for (const auto& type_i : absorbingtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, ParticleStatus::Owned);

    // get number of particles stored in container
    const int particlestored = container_i->particles_stored();

    // allocate memory
    cfg_i[static_cast<int>(type_i)].assign(particlestored, std::vector<double>(3, 0.0));
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      absorbingtypes_, nonabsorbingtypes_, relindices);

  // get pointers to particle states
  ConstParticleContainerBundleStatePtrs& mass =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Mass);
  ConstParticleContainerBundleStatePtrs& dens =
      particlecontainerbundle_->try_get_ptrs_to_state(ParticleState::Density);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    ParticleType type_i;
    ParticleStatus status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    ParticleType type_j;
    ParticleStatus status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get material for particle types
    const Mat::PAR::ParticleMaterialBase* material_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    const Mat::PAR::ParticleMaterialBase* material_j =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

    // get pointer to particle states
    const double* mass_i = Particle::bundle_state_ptrs_index(mass, type_i, status_i, particle_i);
    const double* dens_i = Particle::bundle_state_ptrs_index(
        dens, &(material_i->initDensity_), type_i, status_i, particle_i);

    const double* mass_j = Particle::bundle_state_ptrs_index(mass, type_j, status_j, particle_j);
    const double* dens_j = Particle::bundle_state_ptrs_index(
        dens, &(material_j->initDensity_), type_j, status_j, particle_j);

    // (current) volume of particle i and j
    const double V_i = mass_i[0] / dens_i[0];
    const double V_j = mass_j[0] / dens_j[0];

    const double fac =
        (ParticleUtils::pow<2>(V_i) + ParticleUtils::pow<2>(V_j)) / (dens_i[0] + dens_j[0]);

    // evaluate contribution of neighboring particle j
    if (absorbingtypes_.contains(type_i))
    {
      // sum contribution of neighboring particle j
      ParticleUtils::vec_add_scale(cfg_i[static_cast<int>(type_i)][particle_i].data(),
          dens_i[0] / V_i * fac * particlepair.dWdrij_, particlepair.e_ij_);
    }

    // evaluate contribution of neighboring particle i
    if (absorbingtypes_.contains(type_j) and status_j == ParticleStatus::Owned)
    {
      // sum contribution of neighboring particle i
      ParticleUtils::vec_add_scale(cfg_i[static_cast<int>(type_j)][particle_j].data(),
          -dens_j[0] / V_j * fac * particlepair.dWdrji_, particlepair.e_ij_);
    }
  }

  // init vector containing evaluated function
  std::vector<double> funct(1);

  // get reference to function
  const auto& function =
      Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
          heatsourcefctnumber_);

  // safety check
  if (function.number_components() != 1)
    FOUR_C_THROW("dimension of function defining heat source is not one!");

  // get pointer to particle states
  const int statedim = Particle::enum_to_state_dim(ParticleState::Position);
  ConstParticleContainerBundleStatePtrs& pos = particlecontainerbundle_->try_get_ptrs_to_state(
      ParticleState::Position, absorbingtypes_, ParticleStatus::Owned);
  ParticleContainerBundleStatePtrs& tempdot =
      particlecontainerbundle_->try_get_ptrs_to_state_writable(
          ParticleState::TemperatureDot, absorbingtypes_, ParticleStatus::Owned);

  // iterate over absorbing particle types
  for (const auto& type_i : absorbingtypes_)
  {
    // get container of owned particles of current particle type
    ParticleStatus status_i = ParticleStatus::Owned;
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, ParticleStatus::Owned);

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    const Mat::PAR::ParticleMaterialThermo* thermomaterial_i =
        thermomaterial_[static_cast<int>(type_i)];

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // norm of colorfield gradient of absorbing interface particles
      const double f_i =
          ParticleUtils::vec_norm_two(cfg_i[static_cast<int>(type_i)][particle_i].data());

      // no heat source contribution to current particle
      if (not(f_i > 0.0)) continue;

      // projection of colorfield gradient with heat source direction
      const double f_i_proj = eval_direction_
                                  ? -ParticleUtils::vec_dot(direction_.data(),
                                        cfg_i[static_cast<int>(type_i)][particle_i].data())
                                  : f_i;

      // heat source contribution only for surface opposing heat source
      if (f_i_proj < 0.0) continue;

      // get pointer to particle states
      const double* dens_i = Particle::bundle_state_ptrs_index(
          dens, &(basematerial_i->initDensity_), type_i, status_i, particle_i);
      const double* pos_i =
          Particle::bundle_state_ptrs_index(pos, type_i, status_i, particle_i, statedim);
      double* tempdot_i = Particle::bundle_state_ptrs_index(tempdot, type_i, status_i, particle_i);

      // evaluate function defining heat source
      funct = function.evaluate_time_derivative(std::span(pos_i, 3), evaltime, 0, 0);

      // add contribution of heat source
      tempdot_i[0] += f_i_proj * thermomaterial_i->thermalAbsorptivity_ * funct[0] *
                      thermomaterial_i->invThermalCapacity_ / dens_i[0];
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
