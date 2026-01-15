// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_density.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_density_correction.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_particle_interaction_sph_kernel.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_sph_virtual_wall_particle.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_datastate.hpp"
#include "4C_particle_wall_interface.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHDensityBase::SPHDensityBase(const Teuchos::ParameterList& params)
    : params_sph_(params), fluidtypes_({Particle::Phase1, Particle::Phase2}), dt_(0.0)
{
  // empty constructor
}

void Particle::SPHDensityBase::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
    const std::shared_ptr<Particle::SPHKernelBase> kernel,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial,
    const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs,
    const std::shared_ptr<Particle::SPHVirtualWallParticle> virtualwallparticle)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set interface to particle wall handler
  particlewallinterface_ = particlewallinterface;

  // set kernel handler
  kernel_ = kernel;

  // set particle material handler
  particlematerial_ = particlematerial;

  // set equation of state handler
  equationofstatebundle_ = equationofstatebundle;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // set virtual wall particle handler
  virtualwallparticle_ = virtualwallparticle;

  // update with actual fluid particle types
  const auto fluidtypes = fluidtypes_;
  for (const auto& type_i : fluidtypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      fluidtypes_.erase(type_i);

  // setup density of ghosted particles to refresh
  {
    std::vector<Particle::StateEnum> states{Particle::Density};

    for (const auto& type_i : fluidtypes_)
      densitytorefresh_.push_back(std::make_pair(type_i, states));
  }
}

void Particle::SPHDensityBase::set_current_step_size(const double currentstepsize)
{
  dt_ = currentstepsize;
}

void Particle::SPHDensityBase::sum_weighted_mass() const
{
  // clear density sum state
  clear_density_sum_state();

  // sum weighted mass (self contribution)
  sum_weighted_mass_self_contribution();

  // sum weighted mass (particle contribution)
  sum_weighted_mass_particle_contribution();

  // sum weighted mass (particle-wall contribution)
  if (virtualwallparticle_) sum_weighted_mass_particle_wall_contribution();
}

void Particle::SPHDensityBase::clear_density_sum_state() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // clear density sum state
    container_i->clear_state(Particle::DensitySum);
  }
}

void Particle::SPHDensityBase::sum_weighted_mass_self_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityBase::sum_weighted_mass_self_contribution");

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
      const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
      double* denssum_i = container_i->get_ptr_to_state(Particle::DensitySum, particle_i);

      // evaluate kernel
      const double Wii = kernel_->w0(rad_i[0]);

      // add self contribution
      denssum_i[0] += Wii * mass_i[0];
    }
  }
}

void Particle::SPHDensityBase::sum_weighted_mass_particle_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityBase::sum_weighted_mass_particle_contribution");

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

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    double* denssum_i = container_i->cond_get_ptr_to_state(Particle::DensitySum, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(Particle::Mass, particle_j);
    double* denssum_j = container_j->cond_get_ptr_to_state(Particle::DensitySum, particle_j);

    // sum contribution of neighboring particle j
    if (denssum_i) denssum_i[0] += particlepair.Wij_ * mass_i[0];

    // sum contribution of neighboring particle i
    if (denssum_j and status_j == Particle::Owned) denssum_j[0] += particlepair.Wji_ * mass_j[0];
  }
}

void Particle::SPHDensityBase::sum_weighted_mass_particle_wall_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::SPHDensityBase::sum_weighted_mass_particle_wall_contribution");

  // get relevant particle wall pair indices for specific particle types
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_wall_pair_indices(fluidtypes_, relindices);

  // iterate over relevant particle-wall pairs
  for (const int particlewallpairindex : relindices)
  {
    const SPHParticleWallPair& particlewallpair =
        neighborpairs_->get_ref_to_particle_wall_pair_data()[particlewallpairindex];

    // access values of local index tuple of particle i
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlewallpair.tuple_i_;

    // get corresponding particle container
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    double* denssum_i = container_i->get_ptr_to_state(Particle::DensitySum, particle_i);

    // compute vector from wall contact point j to particle i
    double r_ij[3];
    ParticleUtils::vec_set_scale(r_ij, particlewallpair.absdist_, particlewallpair.e_ij_);

    // unit surface tangent vectors in wall contact point j
    double t_j_1[3];
    double t_j_2[3];
    ParticleUtils::unit_surface_tangents(particlewallpair.e_ij_, t_j_1, t_j_2);

    // iterate over virtual particles
    for (const std::vector<double>& virtualparticle :
        virtualwallparticle_->get_relative_positions_of_virtual_particles())
    {
      // vector from virtual particle k to particle i
      double r_ik[3];
      ParticleUtils::vec_set(r_ik, r_ij);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[0], particlewallpair.e_ij_);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[1], t_j_1);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[2], t_j_2);

      // absolute distance between virtual particle k and particle i
      const double absdist = ParticleUtils::vec_norm_two(r_ik);

      // virtual particle within interaction distance
      if (absdist < rad_i[0])
      {
        // evaluate kernel
        const double Wik = kernel_->w(absdist, rad_i[0]);

        // sum contribution of virtual particle k
        denssum_i[0] += Wik * mass_i[0];
      }
    }
  }
}

void Particle::SPHDensityBase::sum_colorfield() const
{
  // clear colorfield state
  clear_colorfield_state();

  // sum colorfield (self contribution)
  sum_colorfield_self_contribution();

  // sum colorfield (particle contribution)
  sum_colorfield_particle_contribution();

  // sum colorfield (particle-wall contribution)
  if (virtualwallparticle_) sum_colorfield_particle_wall_contribution();
}

void Particle::SPHDensityBase::clear_colorfield_state() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // clear colorfield state
    container_i->clear_state(Particle::Colorfield);
  }
}

void Particle::SPHDensityBase::sum_colorfield_self_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityBase::sum_colorfield_self_contribution");

  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
    {
      // get pointer to particle states
      const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
      const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
      const double* dens_i = container_i->get_ptr_to_state(Particle::Density, particle_i);
      double* colorfield_i = container_i->get_ptr_to_state(Particle::Colorfield, particle_i);

      // evaluate kernel
      const double Wii = kernel_->w0(rad_i[0]);

      // add self contribution
      colorfield_i[0] += (Wii / dens_i[0]) * mass_i[0];
    }
  }
}

void Particle::SPHDensityBase::sum_colorfield_particle_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityBase::sum_colorfield_particle_contribution");

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
    const Mat::PAR::ParticleMaterialBase* material_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    const Mat::PAR::ParticleMaterialBase* material_j =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);

    const double* dens_i = container_i->have_stored_state(Particle::Density)
                               ? container_i->get_ptr_to_state(Particle::Density, particle_i)
                               : &(material_j->initDensity_);

    double* colorfield_i = container_i->cond_get_ptr_to_state(Particle::Colorfield, particle_i);

    const double* mass_j = container_j->get_ptr_to_state(Particle::Mass, particle_j);

    const double* dens_j = container_j->have_stored_state(Particle::Density)
                               ? container_j->get_ptr_to_state(Particle::Density, particle_j)
                               : &(material_i->initDensity_);

    double* colorfield_j = container_j->cond_get_ptr_to_state(Particle::Colorfield, particle_j);

    // sum contribution of neighboring particle j
    if (colorfield_i) colorfield_i[0] += (particlepair.Wij_ / dens_j[0]) * mass_j[0];

    // sum contribution of neighboring particle i
    if (colorfield_j and status_j == Particle::Owned)
      colorfield_j[0] += (particlepair.Wji_ / dens_i[0]) * mass_i[0];
  }
}

void Particle::SPHDensityBase::sum_colorfield_particle_wall_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityBase::sum_colorfield_particle_wall_contribution");

  // get relevant particle wall pair indices for specific particle types
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_wall_pair_indices(fluidtypes_, relindices);

  // iterate over relevant particle-wall pairs
  for (const int particlewallpairindex : relindices)
  {
    const SPHParticleWallPair& particlewallpair =
        neighborpairs_->get_ref_to_particle_wall_pair_data()[particlewallpairindex];

    // access values of local index tuple of particle i
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlewallpair.tuple_i_;

    // get corresponding particle container
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get material for particle types
    const Mat::PAR::ParticleMaterialBase* material_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
    double* colorfield_i = container_i->get_ptr_to_state(Particle::Colorfield, particle_i);

    // get pointer to virtual particle states
    const double* mass_k = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* dens_k = &(material_i->initDensity_);

    // (current) volume of virtual particle k
    const double V_k = mass_k[0] / dens_k[0];

    // compute vector from wall contact point j to particle i
    double r_ij[3];
    ParticleUtils::vec_set_scale(r_ij, particlewallpair.absdist_, particlewallpair.e_ij_);

    // unit surface tangent vectors in wall contact point j
    double t_j_1[3];
    double t_j_2[3];
    ParticleUtils::unit_surface_tangents(particlewallpair.e_ij_, t_j_1, t_j_2);

    // iterate over virtual particles
    for (const std::vector<double>& virtualparticle :
        virtualwallparticle_->get_relative_positions_of_virtual_particles())
    {
      // vector from virtual particle k to particle i
      double r_ik[3];
      ParticleUtils::vec_set(r_ik, r_ij);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[0], particlewallpair.e_ij_);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[1], t_j_1);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[2], t_j_2);

      // absolute distance between virtual particle k and particle i
      const double absdist = ParticleUtils::vec_norm_two(r_ik);

      // virtual particle within interaction distance
      if (absdist < rad_i[0])
      {
        // evaluate kernel
        const double Wik = kernel_->w(absdist, rad_i[0]);

        // sum contribution of virtual particle k
        colorfield_i[0] += V_k * Wik;
      }
    }
  }
}

void Particle::SPHDensityBase::continuity_equation() const
{
  // clear density dot state
  clear_density_dot_state();

  // continuity equation (particle contribution)
  continuity_equation_particle_contribution();

  // continuity equation (particle-wall contribution)
  if (virtualwallparticle_) continuity_equation_particle_wall_contribution();
}

void Particle::SPHDensityBase::clear_density_dot_state() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // clear density dot state
    container_i->clear_state(Particle::DensityDot);
  }
}

void Particle::SPHDensityBase::continuity_equation_particle_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityBase::continuity_equation_particle_contribution");

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
    const Mat::PAR::ParticleMaterialBase* material_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    const Mat::PAR::ParticleMaterialBase* material_j =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_j);

    // get pointer to particle states
    const double* vel_i =
        container_i->have_stored_state(Particle::ModifiedVelocity)
            ? container_i->get_ptr_to_state(Particle::ModifiedVelocity, particle_i)
            : container_i->get_ptr_to_state(Particle::Velocity, particle_i);

    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);

    const double* dens_i = container_i->have_stored_state(Particle::Density)
                               ? container_i->get_ptr_to_state(Particle::Density, particle_i)
                               : &(material_j->initDensity_);

    double* densdot_i = container_i->cond_get_ptr_to_state(Particle::DensityDot, particle_i);

    const double* vel_j =
        container_j->have_stored_state(Particle::ModifiedVelocity)
            ? container_j->get_ptr_to_state(Particle::ModifiedVelocity, particle_j)
            : container_j->get_ptr_to_state(Particle::Velocity, particle_j);

    const double* mass_j = container_j->get_ptr_to_state(Particle::Mass, particle_j);

    const double* dens_j = container_j->have_stored_state(Particle::Density)
                               ? container_j->get_ptr_to_state(Particle::Density, particle_j)
                               : &(material_i->initDensity_);

    double* densdot_j = container_j->cond_get_ptr_to_state(Particle::DensityDot, particle_j);

    // relative velocity (use modified velocities in case of transport velocity formulation)
    double vel_ij[3];
    ParticleUtils::vec_set(vel_ij, vel_i);
    ParticleUtils::vec_sub(vel_ij, vel_j);

    const double e_ij_vel_ij = ParticleUtils::vec_dot(particlepair.e_ij_, vel_ij);

    // sum contribution of neighboring particle j
    if (densdot_i)
      densdot_i[0] += dens_i[0] * (mass_j[0] / dens_j[0]) * particlepair.dWdrij_ * e_ij_vel_ij;

    // sum contribution of neighboring particle i
    if (densdot_j and status_j == Particle::Owned)
      densdot_j[0] += dens_j[0] * (mass_i[0] / dens_i[0]) * particlepair.dWdrji_ * e_ij_vel_ij;
  }
}

void Particle::SPHDensityBase::continuity_equation_particle_wall_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::SPHDensityBase::continuity_equation_particle_wall_contribution");

  // get wall data state container
  std::shared_ptr<Particle::WallDataState> walldatastate =
      particlewallinterface_->get_wall_data_state();

  // get relevant particle wall pair indices for specific particle types
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_wall_pair_indices(fluidtypes_, relindices);

  // iterate over relevant particle-wall pairs
  for (const int particlewallpairindex : relindices)
  {
    const SPHParticleWallPair& particlewallpair =
        neighborpairs_->get_ref_to_particle_wall_pair_data()[particlewallpairindex];

    // access values of local index tuple of particle i
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlewallpair.tuple_i_;

    // get corresponding particle container
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get material for particle types
    const Mat::PAR::ParticleMaterialBase* material_i =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // get pointer to particle states
    const double* vel_i =
        container_i->have_stored_state(Particle::ModifiedVelocity)
            ? container_i->get_ptr_to_state(Particle::ModifiedVelocity, particle_i)
            : container_i->get_ptr_to_state(Particle::Velocity, particle_i);

    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(Particle::Density, particle_i);
    double* densdot_i = container_i->get_ptr_to_state(Particle::DensityDot, particle_i);

    // get pointer to column wall element
    Core::Elements::Element* ele = particlewallpair.ele_;

    // number of nodes of wall element
    const int numnodes = ele->num_node();

    // shape functions and location vector of wall element
    Core::LinAlg::SerialDenseVector funct(numnodes);
    std::vector<int> lmele;

    if (walldatastate->get_vel_col() != nullptr)
    {
      // evaluate shape functions of element at wall contact point
      Core::FE::shape_function_2d(
          funct, particlewallpair.elecoords_[0], particlewallpair.elecoords_[1], ele->shape());

      // get location vector of wall element
      lmele.reserve(numnodes * 3);
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(
          *particlewallinterface_->get_wall_discretization(), lmele, lmowner, lmstride);
    }

    // velocity of wall contact point j
    std::array<double, 3> vel_j = {0.0, 0.0, 0.0};

    if (walldatastate->get_vel_col() != nullptr)
    {
      // get nodal velocities
      std::vector<double> nodal_vel =
          Core::FE::extract_values(*walldatastate->get_vel_col(), lmele);

      // determine velocity of wall contact point j
      for (int node = 0; node < numnodes; ++node)
        for (int dim = 0; dim < 3; ++dim) vel_j[dim] += funct[node] * nodal_vel[node * 3 + dim];
    }

    // get pointer to virtual particle states
    const double* mass_k = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* dens_k = &(material_i->initDensity_);
    const double* vel_k = vel_j.data();

    // (current) volume of virtual particle k
    const double V_k = mass_k[0] / dens_k[0];

    // compute vector from wall contact point j to particle i
    double r_ij[3];
    ParticleUtils::vec_set_scale(r_ij, particlewallpair.absdist_, particlewallpair.e_ij_);

    // relative velocity (use modified velocities in case of transport velocity formulation)
    double vel_ik[3];
    ParticleUtils::vec_set(vel_ik, vel_i);
    ParticleUtils::vec_sub(vel_ik, vel_k);

    // unit surface tangent vectors in wall contact point j
    double t_j_1[3];
    double t_j_2[3];
    ParticleUtils::unit_surface_tangents(particlewallpair.e_ij_, t_j_1, t_j_2);

    // iterate over virtual particles
    for (const std::vector<double>& virtualparticle :
        virtualwallparticle_->get_relative_positions_of_virtual_particles())
    {
      // vector from virtual particle k to particle i
      double r_ik[3];
      ParticleUtils::vec_set(r_ik, r_ij);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[0], particlewallpair.e_ij_);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[1], t_j_1);
      ParticleUtils::vec_add_scale(r_ik, virtualparticle[2], t_j_2);

      // absolute distance between virtual particle k and particle i
      const double absdist = ParticleUtils::vec_norm_two(r_ik);

      // virtual particle within interaction distance
      if (absdist < rad_i[0])
      {
        const double e_ik_vel_ik = ParticleUtils::vec_dot(r_ik, vel_ik) / absdist;

        // evaluate first derivative of kernel
        const double dWdrik = kernel_->d_wdrij(absdist, rad_i[0]);

        // sum contribution of virtual particle k
        densdot_i[0] += dens_i[0] * V_k * dWdrik * e_ik_vel_ik;
      }
    }
  }
}

void Particle::SPHDensityBase::set_density_sum() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // update density of all particles
    container_i->update_state(0.0, Particle::Density, 1.0, Particle::DensitySum);
  }
}

void Particle::SPHDensityBase::add_time_step_scaled_density_dot() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // update density of all particles
    container_i->update_state(1.0, Particle::Density, dt_, Particle::DensityDot);
  }
}

Particle::SPHDensitySummation::SPHDensitySummation(const Teuchos::ParameterList& params)
    : Particle::SPHDensityBase(params)
{
  // empty constructor
}

void Particle::SPHDensitySummation::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    Particle::TypeEnum type_i = typeIt.first;

    // set of particle states for current particle type
    std::set<Particle::StateEnum>& particlestates = typeIt.second;

    // current particle type is not a fluid particle type
    if (not fluidtypes_.contains(type_i)) continue;

    // states for density evaluation scheme
    particlestates.insert(Particle::DensitySum);
  }
}

void Particle::SPHDensitySummation::compute_density() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensitySummation::ComputeDensity");

  // evaluate sum of weighted mass
  sum_weighted_mass();

  // set density sum to density field
  set_density_sum();

  // refresh density of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(densitytorefresh_);
}

Particle::SPHDensityIntegration::SPHDensityIntegration(const Teuchos::ParameterList& params)
    : Particle::SPHDensityBase(params)
{
  // empty constructor
}

void Particle::SPHDensityIntegration::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    Particle::TypeEnum type_i = typeIt.first;

    // set of particle states for current particle type
    std::set<Particle::StateEnum>& particlestates = typeIt.second;

    // current particle type is not a fluid particle type
    if (not fluidtypes_.contains(type_i)) continue;

    // states for density evaluation scheme
    particlestates.insert(Particle::DensityDot);
  }
}

void Particle::SPHDensityIntegration::compute_density() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityIntegration::ComputeDensity");

  // evaluate continuity equation
  continuity_equation();

  // add time step scaled density dot to density field
  add_time_step_scaled_density_dot();

  // refresh density of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(densitytorefresh_);
}

Particle::SPHDensityPredictCorrect::SPHDensityPredictCorrect(const Teuchos::ParameterList& params)
    : Particle::SPHDensityBase(params)
{
  init_density_correction_handler();
}

Particle::SPHDensityPredictCorrect::~SPHDensityPredictCorrect() = default;

void Particle::SPHDensityPredictCorrect::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
    const std::shared_ptr<Particle::SPHKernelBase> kernel,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial,
    const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs,
    const std::shared_ptr<Particle::SPHVirtualWallParticle> virtualwallparticle)
{
  // call base class setup
  SPHDensityBase::setup(particleengineinterface, particlewallinterface, kernel, particlematerial,
      equationofstatebundle, neighborpairs, virtualwallparticle);
}

void Particle::SPHDensityPredictCorrect::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    Particle::TypeEnum type_i = typeIt.first;

    // set of particle states for current particle type
    std::set<Particle::StateEnum>& particlestates = typeIt.second;

    // current particle type is not a fluid particle type
    if (not fluidtypes_.contains(type_i)) continue;

    // states for density evaluation scheme
    particlestates.insert({Particle::DensityDot, Particle::DensitySum, Particle::Colorfield});
  }
}

void Particle::SPHDensityPredictCorrect::compute_density() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHDensityPredictCorrect::ComputeDensity");

  // evaluate continuity equation
  continuity_equation();

  // add time step scaled density dot to density field
  add_time_step_scaled_density_dot();

  // refresh density of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(densitytorefresh_);

  // evaluate sum of weighted mass
  sum_weighted_mass();

  // evaluate sum of colorfield
  sum_colorfield();

  // correct density of interior/surface particles
  correct_density();

  // refresh density of ghosted particles
  particleengineinterface_->refresh_particles_of_specific_states_and_types(densitytorefresh_);
}

void Particle::SPHDensityPredictCorrect::init_density_correction_handler()
{
  // get type of density correction scheme
  auto densitycorrectionscheme = Teuchos::getIntegralValue<Particle::DensityCorrectionScheme>(
      params_sph_, "DENSITYCORRECTION");

  // create density correction handler
  switch (densitycorrectionscheme)
  {
    case Particle::InteriorCorrection:
    {
      densitycorrection_ = std::unique_ptr<Particle::SPHDensityCorrectionInterior>(
          new Particle::SPHDensityCorrectionInterior());
      break;
    }
    case Particle::NormalizedCorrection:
    {
      densitycorrection_ = std::unique_ptr<Particle::SPHDensityCorrectionNormalized>(
          new Particle::SPHDensityCorrectionNormalized());
      break;
    }
    case Particle::RandlesCorrection:
    {
      densitycorrection_ = std::unique_ptr<Particle::SPHDensityCorrectionRandles>(
          new Particle::SPHDensityCorrectionRandles());
      break;
    }
    default:
    {
      FOUR_C_THROW("no density correction scheme set via parameter 'DENSITYCORRECTION'!");
      break;
    }
  }
}

void Particle::SPHDensityPredictCorrect::correct_density() const
{
  // iterate over fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    // get container of owned particles of current particle type
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, Particle::Owned);

    // get number of particles stored in container
    const int particlestored = container_i->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get pointer to particle state
    double* dens = container_i->get_ptr_to_state(Particle::Density, 0);
    const double* denssum = container_i->get_ptr_to_state(Particle::DensitySum, 0);
    const double* colorfield = container_i->get_ptr_to_state(Particle::Colorfield, 0);

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* material =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // get equation of state for current particle type
    const Particle::SPHEquationOfStateBase* equationofstate =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // iterate over owned particles of current type
    for (int i = 0; i < particlestored; ++i)
    {
      if (colorfield[i] >= 1.0)
      {
        // set corrected density of interior particles
        densitycorrection_->corrected_density_interior(&denssum[i], &dens[i]);
      }
      else
      {
        double dens_bc = 0.0;
        if (densitycorrection_->compute_density_bc())
        {
          double press_bc = 0.0;
          dens_bc = equationofstate->pressure_to_density(press_bc, material->initDensity_);
        }

        // set corrected density of free surface particles
        densitycorrection_->corrected_density_free_surface(
            &denssum[i], &colorfield[i], &dens_bc, &dens[i]);
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
