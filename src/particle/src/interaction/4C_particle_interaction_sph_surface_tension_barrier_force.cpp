// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_surface_tension_barrier_force.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHBarrierForce::SPHBarrierForce(const Teuchos::ParameterList& params)
    : params_sph_(params),
      liquidtype_(Particle::Phase1),
      gastype_(Particle::Phase2),
      fluidtypes_({liquidtype_, gastype_}),
      boundarytypes_({Particle::BoundaryPhase, Particle::RigidPhase}),
      dist_(params_sph_.get<double>("BARRIER_FORCE_DISTANCE")),
      cr_(params_sph_.get<double>("BARRIER_FORCE_TEMPSCALE")),
      trans_ref_temp_(params_sph_.get<double>("TRANS_REF_TEMPERATURE")),
      trans_dT_barrier_(params_sph_.get<double>("TRANS_DT_BARRIER")),
      stiff_h_(params_sph_.get<double>("BARRIER_FORCE_STIFF_HEAVY")),
      damp_h_(params_sph_.get<double>("BARRIER_FORCE_DAMP_HEAVY")),
      stiff_g_(params_sph_.get<double>("BARRIER_FORCE_STIFF_GAS")),
      damp_g_(params_sph_.get<double>("BARRIER_FORCE_DAMP_GAS"))
{
  if (not(dist_ > 0.0)) FOUR_C_THROW("barrier force distance not positive!");

  if (not(stiff_h_ > 0.0)) FOUR_C_THROW("stiffness of heavy phase not positive!");
  if (not(stiff_g_ > 0.0)) FOUR_C_THROW("stiffness of gas phase not positive!");

  if (damp_h_ < 0.0) FOUR_C_THROW("damping parameter of heavy phase not positive or zero!");
  if (damp_g_ < 0.0) FOUR_C_THROW("damping parameter of gas phase not positive or zero!");

  if (trans_dT_barrier_ > 0.0)
  {
    if (Teuchos::getIntegralValue<Particle::TemperatureEvaluationScheme>(
            params_sph_, "TEMPERATUREEVALUATION") == Particle::NoTemperatureEvaluation)
      FOUR_C_THROW("temperature evaluation needed for linear transition of surface tension!");
  }
}

void Particle::SPHBarrierForce::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // safety check
  for (const auto& type_i : fluidtypes_)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      FOUR_C_THROW("no particle container for particle type '{}' found!",
          Particle::enum_to_type_name(type_i));

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      boundarytypes_.erase(type_i);
}

void Particle::SPHBarrierForce::compute_barrier_force_contribution() const
{
  // compute barrier force contribution (particle contribution)
  compute_barrier_force_particle_contribution();

  // compute barrier force contribution (particle-boundary contribution)
  compute_barrier_force_particle_boundary_contribution();
}

void Particle::SPHBarrierForce::compute_barrier_force_particle_contribution() const
{
  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_equal_combination(fluidtypes_, relindices);

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

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
    const double* temp_i = container_i->cond_get_ptr_to_state(Particle::Temperature, particle_i);
    double* acc_i = container_i->get_ptr_to_state(Particle::Acceleration, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(Particle::Mass, particle_j);
    const double* vel_j = container_j->get_ptr_to_state(Particle::Velocity, particle_j);
    const double* temp_j = container_j->cond_get_ptr_to_state(Particle::Temperature, particle_j);
    double* acc_j = container_j->get_ptr_to_state(Particle::Acceleration, particle_j);

    // evaluate transition factor above reference temperature
    double tempfac_i = 0.0;
    double tempfac_j = 0.0;

    if (type_i != gastype_ and trans_dT_barrier_ > 0.0)
      tempfac_i = ParticleUtils::comp_lin_trans(
          temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_dT_barrier_);

    if (type_j != gastype_ and trans_dT_barrier_ > 0.0)
      tempfac_j = ParticleUtils::comp_lin_trans(
          temp_j[0], trans_ref_temp_, trans_ref_temp_ + trans_dT_barrier_);

    // evaluate active barrier force distance
    const double activedist = std::max(1.0 + cr_ * tempfac_i, 1.0 + cr_ * tempfac_j) * dist_;

    if (particlepair.absdist_ < activedist)
    {
      const double gap = particlepair.absdist_ - activedist;
      const double gapdot = ParticleUtils::vec_dot(vel_i, particlepair.e_ij_) -
                            ParticleUtils::vec_dot(vel_j, particlepair.e_ij_);

      const double stiff = (type_i == gastype_ or type_j == gastype_) ? stiff_g_ : stiff_h_;
      const double damp = (type_i == gastype_ or type_j == gastype_) ? damp_g_ : damp_h_;

      // magnitude of barrier force
      const double fac = (stiff * gap + damp * std::abs(gap) * gapdot);

      // sum contribution of neighboring particle j
      ParticleUtils::vec_add_scale(acc_i, -fac / mass_i[0], particlepair.e_ij_);

      // sum contribution of neighboring particle i
      if (status_j == Particle::Owned)
        ParticleUtils::vec_add_scale(acc_j, fac / mass_j[0], particlepair.e_ij_);
    }
  }
}

void Particle::SPHBarrierForce::compute_barrier_force_particle_boundary_contribution() const
{
  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      fluidtypes_, boundarytypes_, relindices);

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

    // swap fluid particle and boundary particle
    const bool swapparticles = boundarytypes_.contains(type_i);
    if (swapparticles)
    {
      std::tie(type_i, status_i, particle_i) = particlepair.tuple_j_;
      std::tie(type_j, status_j, particle_j) = particlepair.tuple_i_;
    }

    // absolute distance between particles
    const double absdist = particlepair.absdist_;

    // versor from particle j to i
    double e_ij[3];
    ParticleUtils::vec_set(e_ij, particlepair.e_ij_);
    if (swapparticles) ParticleUtils::vec_scale(e_ij, -1.0);

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
    const double* temp_i = container_i->cond_get_ptr_to_state(Particle::Temperature, particle_i);

    double* acc_i = nullptr;
    if (status_i == Particle::Owned)
      acc_i = container_i->get_ptr_to_state(Particle::Acceleration, particle_i);

    // get pointer to boundary particle states
    const double* vel_j = container_j->get_ptr_to_state(Particle::Velocity, particle_j);
    const double* temp_j = container_j->cond_get_ptr_to_state(Particle::Temperature, particle_j);

    // evaluate transition factor above reference temperature
    double tempfac_i = 0.0;
    double tempfac_j = 0.0;

    if (type_i != gastype_ and trans_dT_barrier_ > 0.0)
      tempfac_i = ParticleUtils::comp_lin_trans(
          temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_dT_barrier_);

    if (trans_dT_barrier_ > 0.0)
      tempfac_j = ParticleUtils::comp_lin_trans(
          temp_j[0], trans_ref_temp_, trans_ref_temp_ + trans_dT_barrier_);

    // evaluate active barrier force distance
    const double activedist = std::max(1.0 + cr_ * tempfac_i, 1.0 + cr_ * tempfac_j) * dist_;

    if (absdist < activedist)
    {
      const double gap = absdist - activedist;
      const double gapdot =
          ParticleUtils::vec_dot(vel_i, e_ij) - ParticleUtils::vec_dot(vel_j, e_ij);

      const double stiff = (type_i == gastype_) ? stiff_g_ : stiff_h_;
      const double damp = (type_i == gastype_) ? damp_g_ : damp_h_;

      // magnitude of barrier force
      const double fac = (stiff * gap + damp * std::abs(gap) * gapdot);

      // sum contribution of neighboring particle j
      if (acc_i) ParticleUtils::vec_add_scale(acc_i, -fac / mass_i[0], e_ij);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
