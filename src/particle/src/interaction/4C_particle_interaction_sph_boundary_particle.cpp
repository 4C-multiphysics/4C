// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_boundary_particle.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

Particle::SPHBoundaryParticleBase::SPHBoundaryParticleBase(const Teuchos::ParameterList& params)
    : params_sph_(params),
      fluidtypes_({ParticleType::Phase1, ParticleType::Phase2, ParticleType::DirichletPhase,
          ParticleType::NeumannPhase}),
      boundarytypes_({ParticleType::BoundaryPhase, ParticleType::RigidPhase, ParticleType::PDPhase})
{
  // empty constructor
}

void Particle::SPHBoundaryParticleBase::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // update with actual fluid particle types
  const auto fluidtypes = fluidtypes_;
  for (const auto& type_i : fluidtypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      fluidtypes_.erase(type_i);

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      boundarytypes_.erase(type_i);

  // safety check
  if (boundarytypes_.empty())
    FOUR_C_THROW(
        "no boundary or rigid particles defined but a boundary particle formulation is set!");
}

Particle::SPHBoundaryParticleAdami::SPHBoundaryParticleAdami(const Teuchos::ParameterList& params)
    : Particle::SPHBoundaryParticleBase(params)
{
  // empty constructor
}

void Particle::SPHBoundaryParticleAdami::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs)
{
  // call base class setup
  SPHBoundaryParticleBase::setup(particleengineinterface, neighborpairs);

  // setup modified states of ghosted boundary particles to refresh
  {
    std::vector<ParticleState> states{
        ParticleState::BoundaryPressure, ParticleState::BoundaryVelocity};

    for (const auto& type_i : boundarytypes_)
      boundarystatestorefresh_.push_back(std::make_pair(type_i, states));
  }

  // determine size of vectors indexed by particle types
  const int typevectorsize = static_cast<int>(*(--boundarytypes_.end())) + 1;

  // allocate memory to hold contributions of neighboring particles
  sumj_wij_.resize(typevectorsize);
  sumj_press_j_wij_.resize(typevectorsize);
  sumj_dens_j_r_ij_wij_.resize(typevectorsize);
  sumj_vel_j_wij_.resize(typevectorsize);
}

void Particle::SPHBoundaryParticleAdami::init_boundary_particle_states(std::vector<double>& gravity)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHBoundaryParticleAdami::init_boundary_particle_states");

  // iterate over boundary particle types
  for (const auto& type_i : boundarytypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, ParticleStatus::Owned);

    // get number of particles stored in container
    const int particlestored = container_i->particles_stored();

    // allocate memory
    sumj_wij_[static_cast<int>(type_i)].assign(particlestored, 0.0);
    sumj_press_j_wij_[static_cast<int>(type_i)].assign(particlestored, 0.0);
    sumj_dens_j_r_ij_wij_[static_cast<int>(type_i)].assign(
        particlestored, std::vector<double>(3, 0.0));
    sumj_vel_j_wij_[static_cast<int>(type_i)].assign(particlestored, std::vector<double>(3, 0.0));
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      boundarytypes_, fluidtypes_, relindices);

  // get pointers to particle states
  const int statedim = Particle::enum_to_state_dim(ParticleState::Position);
  ConstParticleContainerBundleStatePtrs& vel =
      particlecontainerbundle_->get_ptrs_to_state(ParticleState::Velocity);
  ConstParticleContainerBundleStatePtrs& acc =
      particlecontainerbundle_->get_ptrs_to_state(ParticleState::Acceleration);
  ConstParticleContainerBundleStatePtrs& dens =
      particlecontainerbundle_->get_ptrs_to_state(ParticleState::Density);
  ConstParticleContainerBundleStatePtrs& press =
      particlecontainerbundle_->get_ptrs_to_state(ParticleState::Pressure);

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

    // evaluate contribution of neighboring fluid particle j
    if (boundarytypes_.contains(type_i))
    {
      // get pointer to particle states
      const double* vel_j =
          Particle::bundle_state_ptrs_index(vel, type_j, status_j, particle_j, statedim);
      const double* dens_j = Particle::bundle_state_ptrs_index(dens, type_j, status_j, particle_j);
      const double* press_j =
          Particle::bundle_state_ptrs_index(press, type_j, status_j, particle_j);

      // sum contribution of neighboring particle j
      sumj_wij_[static_cast<int>(type_i)][particle_i] += particlepair.Wij_;
      sumj_press_j_wij_[static_cast<int>(type_i)][particle_i] += press_j[0] * particlepair.Wij_;

      const double fac = dens_j[0] * particlepair.absdist_ * particlepair.Wij_;
      ParticleUtils::vec_add_scale(
          sumj_dens_j_r_ij_wij_[static_cast<int>(type_i)][particle_i].data(), fac,
          particlepair.e_ij_);

      ParticleUtils::vec_add_scale(
          sumj_vel_j_wij_[static_cast<int>(type_i)][particle_i].data(), particlepair.Wij_, vel_j);
    }

    // evaluate contribution of neighboring fluid particle i
    if (boundarytypes_.contains(type_j) and status_j == ParticleStatus::Owned)
    {
      // get pointer to particle states
      const double* vel_i =
          Particle::bundle_state_ptrs_index(vel, type_i, status_i, particle_i, statedim);
      const double* dens_i = Particle::bundle_state_ptrs_index(dens, type_i, status_i, particle_i);
      const double* press_i =
          Particle::bundle_state_ptrs_index(press, type_i, status_i, particle_i);

      // sum contribution of neighboring particle i
      sumj_wij_[static_cast<int>(type_j)][particle_j] += particlepair.Wji_;
      sumj_press_j_wij_[static_cast<int>(type_j)][particle_j] += press_i[0] * particlepair.Wji_;

      const double fac = -dens_i[0] * particlepair.absdist_ * particlepair.Wji_;
      ParticleUtils::vec_add_scale(
          sumj_dens_j_r_ij_wij_[static_cast<int>(type_j)][particle_j].data(), fac,
          particlepair.e_ij_);

      ParticleUtils::vec_add_scale(
          sumj_vel_j_wij_[static_cast<int>(type_j)][particle_j].data(), particlepair.Wji_, vel_i);
    }
  }

  // iterate over boundary particle types
  for (const auto& type_i : boundarytypes_)
  {
    // get container of owned particles
    ParticleStatus status_i = ParticleStatus::Owned;
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // clear modified boundary particle states
    container_i->clear_state(ParticleState::BoundaryPressure);
    container_i->clear_state(ParticleState::BoundaryVelocity);

    // get pointers to particle states
    double* boundarypress =
        container_i->get_ptr_to_state_writable(Particle::State::BoundaryPressure);
    double* boundaryvel = container_i->get_ptr_to_state_writable(Particle::State::BoundaryVelocity);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // set modified boundary particle states
      if (sumj_wij_[static_cast<int>(type_i)][particle_i] > 0.0)
      {
        // get pointers to particle states
        const double* vel_i =
            Particle::bundle_state_ptrs_index(vel, type_i, status_i, particle_i, statedim);
        const double* acc_i =
            Particle::bundle_state_ptrs_index(acc, type_i, status_i, particle_i, statedim);
        double* boundarypress_i = &boundarypress[particle_i];
        double* boundaryvel_i = &boundaryvel[particle_i * statedim];

        // get relative acceleration of boundary particle
        double relacc[3];
        ParticleUtils::vec_set(relacc, gravity.data());
        ParticleUtils::vec_sub(relacc, acc_i);

        const double inv_sumj_Wij = 1.0 / sumj_wij_[static_cast<int>(type_i)][particle_i];

        // set modified boundary pressure
        boundarypress_i[0] =
            (sumj_press_j_wij_[static_cast<int>(type_i)][particle_i] +
                ParticleUtils::vec_dot(
                    relacc, sumj_dens_j_r_ij_wij_[static_cast<int>(type_i)][particle_i].data())) *
            inv_sumj_Wij;

        // set modified boundary velocity
        ParticleUtils::vec_set_scale(boundaryvel_i, 2.0, vel_i);
        ParticleUtils::vec_add_scale(boundaryvel_i, -inv_sumj_Wij,
            sumj_vel_j_wij_[static_cast<int>(type_i)][particle_i].data());
      }
    }
  }

  // refresh modified states of ghosted boundary particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(
      boundarystatestorefresh_);
}

FOUR_C_NAMESPACE_CLOSE
