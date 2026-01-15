// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_rigid_particle_contact.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_runtime_writer.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_datastate.hpp"
#include "4C_particle_wall_interface.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHRigidParticleContactBase::SPHRigidParticleContactBase(
    const Teuchos::ParameterList& params)
    : params_sph_(params),
      writeparticlewallinteraction_(params_sph_.get<bool>("WRITE_PARTICLE_WALL_INTERACTION")),
      boundarytypes_({Particle::BoundaryPhase, Particle::RigidPhase})
{
  // empty constructor
}

void Particle::SPHRigidParticleContactBase::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
    const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set interface to particle wall handler
  particlewallinterface_ = particlewallinterface;

  // set particle interaction writer
  particleinteractionwriter_ = particleinteractionwriter;

  // setup particle interaction writer
  setup_particle_interaction_writer();

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      boundarytypes_.erase(type_i);

  // safety check
  if (not boundarytypes_.contains(Particle::RigidPhase))
    FOUR_C_THROW("no rigid particles defined but a rigid particle contact formulation is set!");
}

void Particle::SPHRigidParticleContactBase::setup_particle_interaction_writer()
{
  // register specific runtime output writer
  if (writeparticlewallinteraction_)
    particleinteractionwriter_->register_specific_runtime_output_writer(
        "rigidparticle-wall-contact");
}

Particle::SPHRigidParticleContactElastic::SPHRigidParticleContactElastic(
    const Teuchos::ParameterList& params)
    : Particle::SPHRigidParticleContactBase(params),
      stiff_(params_sph_.get<double>("RIGIDPARTICLECONTACTSTIFF")),
      damp_(params_sph_.get<double>("RIGIDPARTICLECONTACTDAMP"))
{
  // empty constructor
}

void Particle::SPHRigidParticleContactElastic::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
    const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs)
{
  // call base class setup
  SPHRigidParticleContactBase::setup(
      particleengineinterface, particlewallinterface, particleinteractionwriter, neighborpairs);

  // safety check
  if (not(stiff_ > 0.0)) FOUR_C_THROW("rigid particle contact stiffness not positive!");
  if (damp_ < 0.0) FOUR_C_THROW("rigid particle contact damping parameter not positive or zero!");
}

void Particle::SPHRigidParticleContactElastic::add_force_contribution()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHRigidParticleContactElastic::add_force_contribution");

  // elastic contact (particle contribution)
  elastic_contact_particle_contribution();

  // elastic contact (particle-wall contribution)
  if (particlewallinterface_) elastic_contact_particle_wall_contribution();
}

void Particle::SPHRigidParticleContactElastic::elastic_contact_particle_contribution()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::SPHRigidParticleContactElastic::elastic_contact_particle_contribution");

  // get initial particle spacing
  const double initialparticlespacing = params_sph_.get<double>("INITIALPARTICLESPACING");

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_equal_combination(
      boundarytypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // evaluate contact condition
    if (not(particlepair.absdist_ < initialparticlespacing)) continue;

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
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
    double* force_i = container_i->cond_get_ptr_to_state(Particle::Force, particle_i);

    // get pointer to particle states
    const double* vel_j = container_j->get_ptr_to_state(Particle::Velocity, particle_j);

    double* force_j = nullptr;
    if (status_j == Particle::Owned)
      force_j = container_j->cond_get_ptr_to_state(Particle::Force, particle_j);

    // compute normal gap and rate of normal gap
    const double gap = particlepair.absdist_ - initialparticlespacing;
    const double gapdot = ParticleUtils::vec_dot(vel_i, particlepair.e_ij_) -
                          ParticleUtils::vec_dot(vel_j, particlepair.e_ij_);

    // magnitude of rigid particle contact force
    const double fac = std::min(0.0, (stiff_ * gap + damp_ * gapdot));

    // add contributions
    if (force_i) ParticleUtils::vec_add_scale(force_i, -fac, particlepair.e_ij_);
    if (force_j) ParticleUtils::vec_add_scale(force_j, fac, particlepair.e_ij_);
  }
}

void Particle::SPHRigidParticleContactElastic::elastic_contact_particle_wall_contribution()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::SPHRigidParticleContactElastic::"
      "elastic_contact_particle_wall_contribution");

  // get initial particle spacing
  const double initialparticlespacing = params_sph_.get<double>("INITIALPARTICLESPACING");

  // get wall data state container
  std::shared_ptr<Particle::WallDataState> walldatastate =
      particlewallinterface_->get_wall_data_state();

  // get reference to particle-wall pair data
  const SPHParticleWallPairData& particlewallpairdata =
      neighborpairs_->get_ref_to_particle_wall_pair_data();

  // get number of particle-wall pairs
  const int numparticlewallpairs = particlewallpairdata.size();

  // write interaction output
  const bool writeinteractionoutput =
      particleinteractionwriter_->get_current_write_result_flag() and writeparticlewallinteraction_;

  // init storage for interaction output
  std::vector<double> attackpoints;
  std::vector<double> contactforces;
  std::vector<double> normaldirection;

  // prepare storage for interaction output
  if (writeinteractionoutput)
  {
    attackpoints.reserve(3 * numparticlewallpairs);
    contactforces.reserve(3 * numparticlewallpairs);
    normaldirection.reserve(3 * numparticlewallpairs);
  }

  // get relevant particle wall pair indices for specific particle types
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_wall_pair_indices({Particle::RigidPhase}, relindices);

  // iterate over relevant particle-wall pairs
  for (const int particlewallpairindex : relindices)
  {
    const SPHParticleWallPair& particlewallpair = particlewallpairdata[particlewallpairindex];

    // evaluate contact condition
    if (not(particlewallpair.absdist_ < 0.5 * initialparticlespacing)) continue;

    // access values of local index tuple of particle i
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlewallpair.tuple_i_;

    // get corresponding particle container
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
    double* force_i = container_i->cond_get_ptr_to_state(Particle::Force, particle_i);

    // get pointer to column wall element
    Core::Elements::Element* ele = particlewallpair.ele_;

    // number of nodes of wall element
    const int numnodes = ele->num_node();

    // shape functions and location vector of wall element
    Core::LinAlg::SerialDenseVector funct(numnodes);
    std::vector<int> lmele;

    if (walldatastate->get_vel_col() != nullptr or walldatastate->get_force_col() != nullptr)
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
    double vel_j[3] = {0.0, 0.0, 0.0};

    if (walldatastate->get_vel_col() != nullptr)
    {
      // get nodal velocities
      std::vector<double> nodal_vel =
          Core::FE::extract_values(*walldatastate->get_vel_col(), lmele);

      // determine velocity of wall contact point j
      for (int node = 0; node < numnodes; ++node)
        for (int dim = 0; dim < 3; ++dim) vel_j[dim] += funct[node] * nodal_vel[node * 3 + dim];
    }

    // compute normal gap and rate of normal gap
    const double gap = particlewallpair.absdist_ - 0.5 * initialparticlespacing;
    const double gapdot = ParticleUtils::vec_dot(vel_i, particlewallpair.e_ij_) -
                          ParticleUtils::vec_dot(vel_j, particlewallpair.e_ij_);

    // magnitude of rigid particle contact force
    const double fac = std::min(0.0, (stiff_ * gap + damp_ * gapdot));

    // add contributions
    if (force_i) ParticleUtils::vec_add_scale(force_i, -fac, particlewallpair.e_ij_);

    // calculation of wall contact force
    double wallcontactforce[3] = {0.0, 0.0, 0.0};
    if (writeinteractionoutput or walldatastate->get_force_col() != nullptr)
      ParticleUtils::vec_set_scale(wallcontactforce, fac, particlewallpair.e_ij_);

    // write interaction output
    if (writeinteractionoutput)
    {
      // compute vector from wall contact point j to particle i
      double r_ij[3];
      ParticleUtils::vec_set_scale(r_ij, particlewallpair.absdist_, particlewallpair.e_ij_);

      // calculate wall contact point
      double wallcontactpoint[3];
      ParticleUtils::vec_set(wallcontactpoint, pos_i);
      ParticleUtils::vec_sub(wallcontactpoint, r_ij);

      // set wall attack point and states
      for (int dim = 0; dim < 3; ++dim) attackpoints.push_back(wallcontactpoint[dim]);
      for (int dim = 0; dim < 3; ++dim) contactforces.push_back(wallcontactforce[dim]);
      for (int dim = 0; dim < 3; ++dim) normaldirection.push_back(particlewallpair.e_ij_[dim]);
    }

    // assemble contact force acting on wall element
    if (walldatastate->get_force_col() != nullptr)
    {
      // determine nodal forces
      std::vector<double> nodal_force(numnodes * 3);
      for (int node = 0; node < numnodes; ++node)
        for (int dim = 0; dim < 3; ++dim)
          nodal_force[node * 3 + dim] = funct[node] * wallcontactforce[dim];

      // assemble nodal forces
      walldatastate->get_force_col()->sum_into_global_values(
          numnodes * 3, nodal_force.data(), lmele.data());
    }
  }

  if (writeinteractionoutput)
  {
    // get specific runtime output writer
    Core::IO::VisualizationManager* visualization_manager =
        particleinteractionwriter_->get_specific_runtime_output_writer(
            "rigidparticle-wall-contact");
    auto& visualization_data = visualization_manager->get_visualization_data();

    // set wall attack points
    visualization_data.get_point_coordinates() = attackpoints;

    // append states
    visualization_data.set_point_data_vector<double>("contact force", contactforces, 3);
    visualization_data.set_point_data_vector<double>("normal direction", normaldirection, 3);
  }
}

FOUR_C_NAMESPACE_CLOSE
