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
      fluidtypes_(
          {Particle::Phase1, Particle::Phase2, Particle::DirichletPhase, Particle::NeumannPhase}),
      boundarytypes_({Particle::BoundaryPhase, Particle::RigidPhase})
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
    std::vector<Particle::StateEnum> states{Particle::BoundaryPressure, Particle::BoundaryVelocity};

    for (const auto& type_i : boundarytypes_)
      boundarystatestorefresh_.push_back(std::make_pair(type_i, states));
  }

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--boundarytypes_.end()) + 1;

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
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // get number of particles stored in container
    const int particlestored = container_i->particles_stored();

    // allocate memory
    sumj_wij_[type_i].assign(particlestored, 0.0);
    sumj_press_j_wij_[type_i].assign(particlestored, 0.0);
    sumj_dens_j_r_ij_wij_[type_i].assign(particlestored, std::vector<double>(3, 0.0));
    sumj_vel_j_wij_[type_i].assign(particlestored, std::vector<double>(3, 0.0));
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      boundarytypes_, fluidtypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // evaluate contribution of neighboring fluid particle j
    if (boundarytypes_.contains(type_i))
    {
      // get container of owned particles
      Particle::ParticleContainer* container_j =
          particlecontainerbundle_->get_specific_container(type_j, status_j);

      // get pointer to particle states
      const double* vel_j = container_j->get_ptr_to_state(Particle::Velocity, particle_j);
      const double* dens_j = container_j->get_ptr_to_state(Particle::Density, particle_j);
      const double* press_j = container_j->get_ptr_to_state(Particle::Pressure, particle_j);

      // sum contribution of neighboring particle j
      sumj_wij_[type_i][particle_i] += particlepair.Wij_;
      sumj_press_j_wij_[type_i][particle_i] += press_j[0] * particlepair.Wij_;

      const double fac = dens_j[0] * particlepair.absdist_ * particlepair.Wij_;
      ParticleUtils::vec_add_scale(
          sumj_dens_j_r_ij_wij_[type_i][particle_i].data(), fac, particlepair.e_ij_);

      ParticleUtils::vec_add_scale(
          sumj_vel_j_wij_[type_i][particle_i].data(), particlepair.Wij_, vel_j);
    }

    // evaluate contribution of neighboring fluid particle i
    if (boundarytypes_.contains(type_j) and status_j == Particle::Owned)
    {
      // get container of owned particles
      Particle::ParticleContainer* container_i =
          particlecontainerbundle_->get_specific_container(type_i, status_i);

      // get pointer to particle states
      const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
      const double* dens_i = container_i->get_ptr_to_state(Particle::Density, particle_i);
      const double* press_i = container_i->get_ptr_to_state(Particle::Pressure, particle_i);

      // sum contribution of neighboring particle i
      sumj_wij_[type_j][particle_j] += particlepair.Wji_;
      sumj_press_j_wij_[type_j][particle_j] += press_i[0] * particlepair.Wji_;

      const double fac = -dens_i[0] * particlepair.absdist_ * particlepair.Wji_;
      ParticleUtils::vec_add_scale(
          sumj_dens_j_r_ij_wij_[type_j][particle_j].data(), fac, particlepair.e_ij_);

      ParticleUtils::vec_add_scale(
          sumj_vel_j_wij_[type_j][particle_j].data(), particlepair.Wji_, vel_i);
    }
  }

  // iterate over boundary particle types
  for (const auto& type_i : boundarytypes_)
  {
    // get container of owned particles
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // clear modified boundary particle states
    container_i->clear_state(Particle::BoundaryPressure);
    container_i->clear_state(Particle::BoundaryVelocity);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // set modified boundary particle states
      if (sumj_wij_[type_i][particle_i] > 0.0)
      {
        // get pointer to particle states
        const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
        const double* acc_i = container_i->get_ptr_to_state(Particle::Acceleration, particle_i);
        double* boundarypress_i =
            container_i->get_ptr_to_state(Particle::BoundaryPressure, particle_i);
        double* boundaryvel_i =
            container_i->get_ptr_to_state(Particle::BoundaryVelocity, particle_i);

        // get relative acceleration of boundary particle
        double relacc[3];
        ParticleUtils::vec_set(relacc, gravity.data());
        ParticleUtils::vec_sub(relacc, acc_i);

        const double inv_sumj_Wij = 1.0 / sumj_wij_[type_i][particle_i];

        // set modified boundary pressure
        boundarypress_i[0] =
            (sumj_press_j_wij_[type_i][particle_i] +
                ParticleUtils::vec_dot(relacc, sumj_dens_j_r_ij_wij_[type_i][particle_i].data())) *
            inv_sumj_Wij;

        // set modified boundary velocity
        ParticleUtils::vec_set_scale(boundaryvel_i, 2.0, vel_i);
        ParticleUtils::vec_add_scale(
            boundaryvel_i, -inv_sumj_Wij, sumj_vel_j_wij_[type_i][particle_i].data());
      }
    }
  }

  // refresh modified states of ghosted boundary particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(
      boundarystatestorefresh_);
}

FOUR_C_NAMESPACE_CLOSE
