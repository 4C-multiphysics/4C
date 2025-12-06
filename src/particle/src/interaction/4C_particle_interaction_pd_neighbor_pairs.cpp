// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_pd_neighbor_pairs.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_io.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <tuple>

FOUR_C_NAMESPACE_OPEN

long int compute_hash_key(const long int i, const long int j)
{
  // create same hashes independent of order
  return static_cast<long int>(0.5 * (i + j) * (i + j + 1)) + std::min(i, j);
}

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/

Particle::PDNeighborPairs::PDNeighborPairs(const MPI_Comm& comm)
    : Particle::DEMNeighborPairs(),
      blacklistmaphashes_(std::make_unique<std::map<long int,
              std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>*>>()),
      comm_(comm)
{
  // empty constructor
}

void Particle::PDNeighborPairs::evaluate_neighbor_pairs()
{
  // evaluate particle pairs
  evaluate_filtered_particle_pairs();

  // evaluate particle-wall pairs
  if (particlewallinterface_) evaluate_particle_wall_pairs();
}

void Particle::PDNeighborPairs::evaluate_filtered_particle_pairs()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::PDNeighborPairs::EvaluateParticlePairs");

  // clear particle pair data
  particlepairdata_.clear();

  // recreate list of hash keys for pd bond pairs
  setup_peridynamic_pair_hashes();

  // iterate over potential particle neighbors
  for (auto& potentialneighbors : particleengineinterface_->get_potential_particle_neighbors())
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

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);
    const int* globalid_j = container_j->get_ptr_to_global_id(particle_j);

    // compute hash key for the pair
    const long int currenthashkey = compute_hash_key(globalid_i[0], globalid_j[0]);

    if (blacklistmaphashes_->contains(currenthashkey))
    {
      // access peridnamic bond list and update values
      std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>* currentpdbond =
          (*blacklistmaphashes_)[currenthashkey];

      std::get<0>(currentpdbond->first) = type_i;
      std::get<1>(currentpdbond->first) = status_i;
      std::get<2>(currentpdbond->first) = particle_i;
      std::get<3>(currentpdbond->first) = globalid_i[0];
      std::get<0>(currentpdbond->second) = type_j;
      std::get<1>(currentpdbond->second) = status_j;
      std::get<2>(currentpdbond->second) = particle_j;
      std::get<3>(currentpdbond->second) = globalid_j[0];

      continue;
    }

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);

    const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);
    const double* rad_j = container_j->get_ptr_to_state(Particle::Radius, particle_j);

    // vector from particle i to j
    double r_ji[3];

    // distance between particles considering periodic boundaries
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);

    // absolute distance between particles
    const double absdist = ParticleUtils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * rad_i[0]) or absdist < (1.0e-10 * rad_j[0]))
      FOUR_C_THROW("absolute distance %f between particles close to zero!", absdist);
#endif

    // gap between particles
    const double gap = absdist - rad_i[0] - rad_j[0];

    // neighboring particles within interaction distance
    if (gap < 0.0)
    {
      // initialize particle pair
      particlepairdata_.push_back(DEMParticlePair());
      // get reference to current particle pair
      DEMParticlePair& particlepair = particlepairdata_.back();

      // set local index tuple of particles i and j
      particlepair.tuple_i_ = potentialneighbors.first;
      particlepair.tuple_j_ = potentialneighbors.second;
      // set gap between particles
      particlepair.gap_ = gap;

      // versor from particle i to j
      ParticleUtils::vec_set_scale(particlepair.e_ji_, (1.0 / absdist), r_ji);
    }
  }
}

void Particle::PDNeighborPairs::setup_peridynamic_pair_hashes()
{
  blacklistmaphashes_->clear();
  // iterate over the peridynamic bond list
  for (auto& pair : *bondlist_)
  {
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i, globalid_i;
    std::tie(type_i, status_i, particle_i, globalid_i) = pair.first;
    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j, globalid_j;
    std::tie(type_j, status_j, particle_j, globalid_j) = pair.second;
    long int hash = compute_hash_key(globalid_i, globalid_j);

    blacklistmaphashes_->insert(std::make_pair(hash, &pair));
  }
};

void Particle::PDNeighborPairs::communicate_bond_list(
    const std::vector<std::vector<int>>& particletargets)
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  Particle::TypeEnum type_i;
  Particle::StatusEnum status_i;
  int particle_i, globalid_i;

  Particle::TypeEnum type_j;
  Particle::StatusEnum status_j;
  int particle_j, globalid_j;

  // pack bond pair information
  // do not delete information on proc just add bond information while receiving
  // during later evaluation if any particle of that bond is not available remove the bond
  const int num_procs = Core::Communication::num_mpi_ranks(comm_);
  for (int torank = 0; torank < num_procs; ++torank)
  {
    if (particletargets[torank].empty()) continue;

    for (int globalid : particletargets[torank])
    {
      for (const auto& pair : *bondlist_)
      {
        std::tie(type_i, status_i, particle_i, globalid_i) = pair.first;
        std::tie(type_j, status_j, particle_j, globalid_j) = pair.second;

        // if particle to be sent is in the bondlist also send the bondlist information
        if (globalid_i == globalid or globalid_j == globalid)
        {
          Core::Communication::PackBuffer data;
          data.add_to_pack(globalid_i);
          data.add_to_pack(globalid_j);
          sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
        }
      }
    }
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack global ids and initialize remaining bond data
  for (auto& p : rdata) unpack_peridynamic_bond_list_data(p.second);
}

void Particle::PDNeighborPairs::unpack_peridynamic_bond_list_data(const std::vector<char>& buffer)
{
  Core::Communication::UnpackBuffer data(buffer);
  while (!data.at_end())
  {
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;

    int globalid_i;
    extract_from_pack(data, globalid_i);
    int globalid_j;
    extract_from_pack(data, globalid_j);

    // setup default tuples with proper global ids
    Particle::LocalGlobalIndexTuple tuple_i = std::make_tuple(type_i, status_i, -1, globalid_i);
    Particle::LocalGlobalIndexTuple tuple_j = std::make_tuple(type_j, status_j, -1, globalid_j);

    // add bond pair
    bondlist_->push_back(std::make_pair(tuple_i, tuple_j));
  }
}

void Particle::PDNeighborPairs::write_restart() const
{
  // get bin discretization writer
  std::shared_ptr<Core::IO::DiscretizationWriter> binwriter =
      particleengineinterface_->get_bin_discretization_writer();

  // prepare buffer
  Core::Communication::PackBuffer buffer;

  // peridynamic bond list
  if (not bondlist_->empty()) pack_bond_list_pairs(buffer);

  binwriter->write_char_data("PeridynamicBondList", buffer());
}

void Particle::PDNeighborPairs::pack_bond_list_pairs(Core::Communication::PackBuffer& buffer) const
{
  Particle::TypeEnum type_i;
  Particle::StatusEnum status_i;
  int particle_i, globalid_i;

  Particle::TypeEnum type_j;
  Particle::StatusEnum status_j;
  int particle_j, globalid_j;

  // iterate over bond list
  for (const auto& pair : *bondlist_)
  {
    std::tie(type_i, status_i, particle_i, globalid_i) = pair.first;
    std::tie(type_j, status_j, particle_j, globalid_j) = pair.second;

    // add bond pair to buffer
    buffer.add_to_pack(globalid_i);
    buffer.add_to_pack(globalid_j);
  }
}

void Particle::PDNeighborPairs::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader)
{
  // peridynamic bond list
  std::shared_ptr<std::vector<char>> buffer = std::make_shared<std::vector<char>>();
  reader->read_char_vector(buffer, "PeridynamicBondList");
  if (buffer->size() > 0) unpack_peridynamic_bond_list_data(*buffer);
}

FOUR_C_NAMESPACE_CLOSE
