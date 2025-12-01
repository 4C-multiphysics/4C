// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_peridynamic.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_dem_neighbor_pairs.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_pd_neighbor_pairs.hpp"
#include "4C_particle_interaction_sph.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <iterator>

FOUR_C_NAMESPACE_OPEN

double calculate_volume_correction_factor(double xi, double dx, double horizon);

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHPeridynamic::SPHPeridynamic(const Teuchos::ParameterList& params)
    : params_pd_(params),
      bondlist_(std::make_shared<
          std::list<std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>>())
{
  // empty constructor
}

void Particle::SPHPeridynamic::init(
    const std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs_solid_type)
{
  // init with potential peridynamic phases
  pdtypes_ = {Particle::RigidPhase};

  // set bondlist to pd neighbor pairs, we do it here because readrestart relies on the bondlist
  // initialization
  std::static_pointer_cast<Particle::PDNeighborPairs>(neighborpairs_solid_type)
      ->set_bond_list(bondlist_);
}

void Particle::SPHPeridynamic::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // update with actual peridynamic types
  const auto pdtypes = pdtypes_;
  for (const auto& type_i : pdtypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i)) pdtypes_.erase(type_i);
}

void Particle::SPHPeridynamic::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    if (typeIt.first == Particle::RigidPhase)
    {
      // set of particle states for current particle type
      std::set<Particle::StateEnum>& particlestates = typeIt.second;

      // set temperature state
      particlestates.insert({Particle::PDBodyId, Particle::ReferencePosition, Particle::Young,
          Particle::CriticalStretch, Particle::InitialConnectedBonds,
          Particle::CurrentConnectedBonds, Particle::PDDamageVariable});
    }
  }
}

void Particle::SPHPeridynamic::setup_peridynamic_bondlist()
{
  // internal variable in future PDInteractionHandler
  const double horizon_pd = params_pd_.get<double>("INTERACTION_HORIZON");
#ifdef FOUR_C_ENABLE_ASSERTIONS
  // get material for rigid phase
  const Mat::PAR::ParticleMaterialBase* material =
      particlematerial_->get_ptr_to_particle_mat_parameter(Particle::RigidPhase);

  // (initial) radius of current phase
  const double initradius = material->initRadius_;
#endif

  // // careful: inside minbinsize_ is used -> maybe use pd_neighborhood instead

  const Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();
  // iterate over potential particle neighbors
  for (const auto& potentialneighbors :
      particleengineinterface_->get_potential_particle_neighbors())
  {
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

    // TODO: maybe remove if optimized version of neighborhood search is included
    if (type_i != Particle::RigidPhase || type_j != Particle::RigidPhase) continue;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);
    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* pdbodyid_i = container_i->get_ptr_to_state(Particle::PDBodyId, particle_i);
    double* initialconnectedbonds_i =
        container_i->get_ptr_to_state(Particle::InitialConnectedBonds, particle_i);

    const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);
    const double* pdbodyid_j = container_j->get_ptr_to_state(Particle::PDBodyId, particle_j);
    double* initialconnectedbonds_j =
        container_j->get_ptr_to_state(Particle::InitialConnectedBonds, particle_j);

    // vector from particle i to j
    double r_ji[3];
    // distance between particles considering periodic boundaries
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);
    // absolute distance between particles
    const double absdist = ParticleUtils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * initradius) or absdist < (1.0e-10 * initradius))
      FOUR_C_THROW("absolute distance %f between particles close to zero!", absdist);
#endif

    // neighboring particles within interaction distance
    if (absdist <= horizon_pd)
    {
      // initialize particle pair
      const int id_i = std::round(pdbodyid_i[0]);
      const int id_j = std::round(pdbodyid_j[0]);
      if (id_j == id_i)
      {
        // get global id of particle i
        const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);
        const int* globalid_j = container_j->get_ptr_to_global_id(particle_j);

        // creat the content of the bond between particles i & j
        Particle::LocalGlobalIndexTuple tuple_i =
            std::make_tuple(type_i, status_i, particle_i, globalid_i[0]);
        Particle::LocalGlobalIndexTuple tuple_j =
            std::make_tuple(type_j, status_j, particle_j, globalid_j[0]);

        bondlist_->push_back(std::make_pair(tuple_i, tuple_j));

        //  increase bond counter for particles i and j
        initialconnectedbonds_i[0] += 1.0;
        initialconnectedbonds_j[0] += 1.0;
      }
    }
  }

  // initialize the current number of connected bonds having the initial connected bonds
  Particle::ParticleContainer* container =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);
  container->update_state(
      0.0, Particle::CurrentConnectedBonds, 1.0, Particle::InitialConnectedBonds);
}

bool Particle::SPHPeridynamic::check_valid_peridynamic_bond_entry(
    const int localid, const int globalid, Particle::ParticleContainer* container) const
{
  // get number of particles stored in container
  const int particlestored = container->particles_stored();

  // check if the localid is smaller than the number of stored particles in the container
  if (localid >= particlestored) return false;
  // get global id of particle i
  const int* globalid_provided = container->get_ptr_to_global_id(localid);

  // check if the provided globalid by the container matches that of the bondlist
  if (globalid == globalid_provided[0])
    return true;
  else
    return false;
}


void Particle::SPHPeridynamic::compute_peridynamic_forces() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::compute_peridynamic_forces");

  // internal variable in future PDInteractionHandler
  const double horizon_pd = params_pd_.get<double>("INTERACTION_HORIZON");
  const double dx_pd = params_pd_.get<double>("PERIDYNAMIC_GRID_SPACING");
  const Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();
  //  iterate over particle pairs
  std::list<std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>::iterator
      it = bondlist_->begin();

  while (it != bondlist_->end())
  {
    // const auto particlepair = *it;
    std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>& particlepair = *it;
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i, globalid_i;
    std::tie(type_i, status_i, particle_i, globalid_i) = particlepair.first;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j, globalid_j;
    std::tie(type_j, status_j, particle_j, globalid_j) = particlepair.second;

    // this occurs if bond pair is communicated to this processor; however, particles
    if (particle_i == -1 or particle_j == -1)
    {
      it = bondlist_->erase(it);
      continue;
    }

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    // check for valid bond if one of its particles has left the processor
    if (check_valid_peridynamic_bond_entry(particle_i, globalid_i, container_i) == false or
        check_valid_peridynamic_bond_entry(particle_j, globalid_j, container_j) == false)
    {
      it = bondlist_->erase(it);
      continue;
    }

    // get pointer to particle states
    const double* ref_pos_i =
        container_i->get_ptr_to_state(Particle::ReferencePosition, particle_i);

    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* young_i = container_i->get_ptr_to_state(Particle::Young, particle_i);
    double* force_i = container_i->cond_get_ptr_to_state(Particle::Force, particle_i);
    const double* critical_stretch_i =
        container_i->get_ptr_to_state(Particle::CriticalStretch, particle_i);

    const double* ref_pos_j =
        container_j->get_ptr_to_state(Particle::ReferencePosition, particle_j);
    const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);
    const double* young_j = container_j->get_ptr_to_state(Particle::Young, particle_j);
    double* force_j = container_j->get_ptr_to_state(Particle::Force, particle_j);

    const double* critical_stretch_j =
        container_j->get_ptr_to_state(Particle::CriticalStretch, particle_j);
    // calculate the bond between two particles
    double xi[3];
    ParticleUtils::vec_set(xi, ref_pos_j);
    ParticleUtils::vec_sub(xi, ref_pos_i);

    // calculate the relative position of the pair
    double xi_eta[3];
    ParticleUtils::vec_set(xi_eta, pos_j);
    ParticleUtils::vec_sub(xi_eta, pos_i);

    // calculate the required norms of the pair
    const double xi_norm = ParticleUtils::vec_norm_two(xi);
    const double xi_eta_norm = ParticleUtils::vec_norm_two(xi_eta);

    // calculate the bond stretch
    double s = (xi_eta_norm - xi_norm) / xi_norm;

    // check the stretch
    const double s_critical = 0.5 * (critical_stretch_i[0] + critical_stretch_j[0]);

    // if critical stretch is not reached
    if (s < s_critical)
    {
      double m[3];
      ParticleUtils::vec_set_scale(m, 1.0 / xi_eta_norm, xi_eta);

      // calculate the bond force of the pair
      double const fac = (12.00 * (young_i[0] + young_j[0]) * 0.5) /
                         (M_PI * horizon_pd * horizon_pd * horizon_pd * horizon_pd) * s *
                         calculate_volume_correction_factor(xi_norm, dx_pd, horizon_pd) * dx_pd *
                         dx_pd * dx_pd * dx_pd * dx_pd * dx_pd;

      // add bond force contribution
      ParticleUtils::vec_add_scale(force_i, fac, m);
      if (status_j == Particle::Owned) ParticleUtils::vec_add_scale(force_j, -fac, m);

      // only increment if you do not delete
      ++it;
    }
    else
    {
      double* currentconnectedbonds_i =
          container_i->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_i);
      double* currentconnectedbonds_j =
          container_j->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_j);

      currentconnectedbonds_i[0] -= 1.0;
      currentconnectedbonds_j[0] -= 1.0;

      // remove the broken PD bond from the bond list
      it = bondlist_->erase(it);
    }
  }
}

void Particle::SPHPeridynamic::damage_evaluation()
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();
  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

  // loop over particles in container
  for (int particle_i = 0; particle_i < container->particles_stored(); ++particle_i)
  {
    const double* initialconnectedbonds_i =
        container->get_ptr_to_state(Particle::InitialConnectedBonds, particle_i);

    const double* currentconnectedbonds_i =
        container->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_i);

    double* pddamagevariable_i =
        container->get_ptr_to_state(Particle::PDDamageVariable, particle_i);

    pddamagevariable_i[0] = 1.0 - currentconnectedbonds_i[0] / initialconnectedbonds_i[0];
  }
}

// the beta correction volume function in PD
double calculate_volume_correction_factor(const double xi, const double dx, const double horizon)
{
  if (xi <= horizon - dx * 0.5)
  {
    return 1.0;
  }
  else if (xi <= horizon + dx * 0.5)
  {
    return (horizon + dx * 0.5 - xi) / (dx);
  }
  else
  {
    return 0.0;
  }
}

FOUR_C_NAMESPACE_CLOSE
