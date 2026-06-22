// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_pd_neighbor_pairs.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_geometry_element_coordtrafo.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_geometry_searchtree_nearestobject.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_io.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_interface.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <algorithm>
#include <tuple>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace
{
  // Peridynamic bonds are identified by the unordered pair of particle global ids.
  // Canonicalize to (min, max) whenever a bond is serialized or deduplicated.
  std::pair<int, int> canonical_pd_bond_ids(const int i, const int j)
  {
    return {std::min(i, j), std::max(i, j)};
  }

  long compute_pd_pair_hash_key(const long i, const long j)
  {
    // create same hashes independent of order of i and j
    const long sum = i + j;
    return (sum * (sum + 1)) / 2 + std::min(i, j);
  }

  Particle::LocalGlobalIndexTuple make_default_peridynamic_bond_tuple(const int globalid)
  {
    // Initialize a bond tuple where only the global id is known so far. Type and status remain
    // unresolved until the particle is found in a local container, and local id = -1 marks that
    // "not yet resolved on this rank" state
    return std::make_tuple(
        Particle::UninitializedType, Particle::UninitializedStatus, -1, globalid);
  }

  std::set<long> build_known_peridynamic_bond_hashes(const std::shared_ptr<std::vector<
          std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>>& bondlist)
  {
    std::set<long> known_bond_hashes;

    for (const auto& [first, second] : *bondlist)
    {
      const int globalid_i = std::get<3>(first);
      const int globalid_j = std::get<3>(second);

      known_bond_hashes.insert(compute_pd_pair_hash_key(globalid_i, globalid_j));
    }

    return known_bond_hashes;
  }

  bool is_valid_peridynamic_bond_entry(
      const int localid, const int globalid, Particle::ParticleContainer* container);
}  // namespace

Particle::PDNeighborPairs::PDNeighborPairs(
    const MPI_Comm& comm, const Teuchos::ParameterList& params_pd)
    : comm_(comm), peridynamic_grid_spacing_(params_pd.get<double>("PERIDYNAMIC_GRID_SPACING"))
{
  // empty constructor
}

void Particle::PDNeighborPairs::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set interface to particle wall handler
  particlewallinterface_ = particlewallinterface;
}

void Particle::PDNeighborPairs::evaluate_neighbor_pairs()
{
  // evaluate particle pairs
  evaluate_particle_pairs();

  // evaluate particle-wall pairs
  if (particlewallinterface_) evaluate_particle_wall_pairs();
}

void Particle::PDNeighborPairs::evaluate_particle_pairs()
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

    // filter rigid phase and boundary phase to find close pairs which interact in peridynamic or
    // DEM way
    if ((type_i != PDPhase and type_i != BoundaryPhase) or
        (type_j != PDPhase and type_j != BoundaryPhase))
      continue;

    if (type_i == BoundaryPhase and type_j == BoundaryPhase) continue;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);
    const int* globalid_j = container_j->get_ptr_to_global_id(particle_j);

    // compute hash key for the pair
    const long current_hashkey = compute_pd_pair_hash_key(globalid_i[0], globalid_j[0]);

    if (map_hashkey_to_bondlist_entry_.contains(current_hashkey))
    {
      // access peridynamic bond list and update values
      std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>& current_pdbond =
          (*bondlist_)[map_hashkey_to_bondlist_entry_[current_hashkey]];

      std::get<0>(current_pdbond.first) = type_i;
      std::get<1>(current_pdbond.first) = status_i;
      std::get<2>(current_pdbond.first) = particle_i;
      std::get<3>(current_pdbond.first) = globalid_i[0];
      std::get<0>(current_pdbond.second) = type_j;
      std::get<1>(current_pdbond.second) = status_j;
      std::get<2>(current_pdbond.second) = particle_j;
      std::get<3>(current_pdbond.second) = globalid_j[0];

      // particles with peridynamic interaction can leave here
      continue;
    }

    // all close and non-bonded particle pairs are considered as potential colliding partners
    // undergoing short range force interaction
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);

    // vector from particle i to j
    double r_ji[3];

    // distance between particles considering periodic boundaries
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);

    // absolute distance between particles
    const double absdist = ParticleUtils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
    const double* rad_j = container_j->get_ptr_to_state(Particle::Radius, particle_j);

    if (absdist < (1.0e-10 * rad_i[0]) or absdist < (1.0e-10 * rad_j[0]))
      FOUR_C_THROW("absolute distance %f between particles close to zero!", absdist);
#endif

    // gap calculation based on initial spacing
    const double gap = absdist - peridynamic_grid_spacing_;

    // neighboring particles within interaction distance
    if (gap < 0.0)
    {
      // initialize particle pair
      particlepairdata_.push_back(PDParticlePair());
      // get reference to current particle pair
      PDParticlePair& particlepair = particlepairdata_.back();

      // set local index tuple of particles i and j
      particlepair.tuple_i_ = potentialneighbors.first;
      particlepair.tuple_j_ = potentialneighbors.second;
      // set gap between particles
      particlepair.gap_ = gap;

      // versor from particle i to j
      ParticleUtils::vec_set_scale(particlepair.e_ji_, (1.0 / absdist), r_ji);
    }
  }

  // clear bond list from invalid entries (introduced due to changes in parallel distribution)
  std::size_t iter = 0;
  while (iter < bondlist_->size())
  {
    auto& particlepair = (*bondlist_)[iter];
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i, globalid_i;
    std::tie(type_i, status_i, particle_i, globalid_i) = particlepair.first;
    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j, globalid_j;
    std::tie(type_j, status_j, particle_j, globalid_j) = particlepair.second;

    // one of the bond list partners may no longer reside on this proc
    if (particle_i == -1 or particle_j == -1)
    {
      (*bondlist_)[iter] = std::move(bondlist_->back());
      bondlist_->pop_back();
      continue;
    }

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // check for valid bond if one of its particles has left the processor
    if (not is_valid_peridynamic_bond_entry(particle_i, globalid_i, container_i) or
        not is_valid_peridynamic_bond_entry(particle_j, globalid_j, container_j))
    {
      (*bondlist_)[iter] = std::move(bondlist_->back());
      bondlist_->pop_back();
      continue;
    }

    ++iter;
  }
}

void Particle::PDNeighborPairs::setup_peridynamic_pair_hashes()
{
  map_hashkey_to_bondlist_entry_.clear();
  // iterate over the peridynamic bond list
  for (size_t iter = 0; iter < bondlist_->size(); ++iter)
  {
    const auto& pair = (*bondlist_)[iter];
    const int globalid_i = std::get<3>(pair.first);
    const int globalid_j = std::get<3>(pair.second);

    const long hash = compute_pd_pair_hash_key(globalid_i, globalid_j);

    const auto insert_result = map_hashkey_to_bondlist_entry_.insert(std::make_pair(hash, iter));
    const bool inserted = insert_result.second;
    if (!inserted)
      FOUR_C_THROW(
          "duplicate peridynamic bond entry detected for particle global ids ({}, {}) at bond "
          "list entry {}!",
          globalid_i, globalid_j, iter);
  }
}

void Particle::PDNeighborPairs::evaluate_particle_wall_pairs()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::PDNeighborPairs::evaluate_particle_wall_pairs");

  // clear particle-wall pair data
  particlewallpairdata_.clear();

  // relate particles to index of particle-wall pairs (considering object type of contact point)
  std::unordered_map<int, std::vector<std::pair<Core::Geo::ObjectType, int>>>
      particletoindexofparticlewallpairs;

  // index of particle-wall pairs
  int particlewallpairindex = 0;

  // iterate over potential wall neighbors
  for (const auto& potentialneighbors : particlewallinterface_->get_potential_wall_neighbors())
  {
    // access values of local index tuple of particle i
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    // get corresponding particle container
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);

    // get position of particle i
    const Core::LinAlg::Matrix<3, 1> pos_i(
        container_i->get_ptr_to_state(Particle::Position, particle_i));

    // get pointer to column wall element
    Core::Elements::Element* ele = potentialneighbors.second;

    // determine nodal positions of column wall element
    std::map<int, Core::LinAlg::Matrix<3, 1>> colelenodalpos;
    particlewallinterface_->determine_col_wall_ele_nodal_pos(ele, colelenodalpos);

    // get coordinates of closest point on current column wall element to particle
    Core::LinAlg::Matrix<3, 1> closestpos;
    Core::Geo::ObjectType objecttype =
        Core::Geo::nearest_3d_object_on_element(ele, colelenodalpos, pos_i, closestpos);

    // vector from particle i to wall contact point j
    double r_ji[3];
    for (int i = 0; i < 3; i++) r_ji[i] = closestpos(i) - pos_i(i);

    // absolute distance between particle and wall contact point
    const double absdist = ParticleUtils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * rad_i[0]))
      FOUR_C_THROW("absolute distance {} between particle and wall close to zero!", absdist);
#endif

    // gap between particle and wall contact point
    const double gap = absdist - rad_i[0];

    // neighboring particle and wall element within interaction distance
    if (gap < 0.0)
    {
      // initialize particle-wall pair
      particlewallpairdata_.push_back(PDParticleWallPair());

      // get reference to current particle-wall pair
      PDParticleWallPair& particlewallpair = particlewallpairdata_[particlewallpairindex];

      // store index of particle-wall pair
      particletoindexofparticlewallpairs[globalid_i[0]].push_back(
          std::make_pair(objecttype, particlewallpairindex));

      // increase index
      ++particlewallpairindex;

      // set local index tuple of particle i
      particlewallpair.tuple_i_ = potentialneighbors.first;

      // set pointer to column wall element
      particlewallpair.ele_ = potentialneighbors.second;

      // set gap between particle and wall contact point
      particlewallpair.gap_ = gap;

      // versor from particle i to wall contact point j
      ParticleUtils::vec_set_scale(particlewallpair.e_ji_, (1.0 / absdist), r_ji);

      // get coordinates of wall contact point in element parameter space
      Core::LinAlg::Matrix<2, 1> elecoords(Core::LinAlg::Initialization::zero);
      const Core::LinAlg::SerialDenseMatrix xyze(
          Core::Geo::get_current_nodal_positions(ele, colelenodalpos));
      Core::Geo::current_to_surface_element_coordinates(ele->shape(), xyze, closestpos, elecoords);

      // set parameter space coordinates of wall contact point
      particlewallpair.elecoords_[0] = elecoords(0, 0);
      particlewallpair.elecoords_[1] = elecoords(1, 0);
    }
  }

  // set of particle-wall pairs to remove
  std::set<int> particlewallpairstoremove;

  // iterate over particles with neighboring wall contact points
  for (auto& particleIt : particletoindexofparticlewallpairs)
  {
    // get reference to index of particle-wall pairs for current particle
    std::vector<std::pair<Core::Geo::ObjectType, int>>& indexofparticlewallpairs =
        particleIt.second;

    // only one particle-wall pair for current particle
    if (indexofparticlewallpairs.size() == 1) continue;

    // get local index tuple of current particle
    Particle::LocalIndexTuple tuple_i =
        particlewallpairdata_[indexofparticlewallpairs[0].second].tuple_i_;

    // access values of local index tuple of particle i
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = tuple_i;

    // get corresponding particle container
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);

    // define tolerance dependent on the particle radius
    const double adaptedtol = 1.0e-7 * rad_i[0];

    // iterate over particle-wall pairs (master)
    for (std::pair<Core::Geo::ObjectType, int>& master : indexofparticlewallpairs)
    {
      // get reference to particle-wall pair (master)
      PDParticleWallPair& masterpair = particlewallpairdata_[master.second];

      // intersection radius of particle with column wall element in wall contact point
      const double intersectionradius = std::sqrt(
          ParticleUtils::pow<2>(rad_i[0]) - ParticleUtils::pow<2>(rad_i[0] + masterpair.gap_));

      // check with other particle-wall pairs (slave)
      for (std::pair<Core::Geo::ObjectType, int>& slave : indexofparticlewallpairs)
      {
        // no-self checking
        if (master.second == slave.second) continue;

        // get reference to particle-wall pair (slave)
        PDParticleWallPair& slavepair = particlewallpairdata_[slave.second];

        // vector between detected wall contact points
        double dist[3];
        ParticleUtils::vec_set_scale(dist, (rad_i[0] + masterpair.gap_), masterpair.e_ji_);
        ParticleUtils::vec_add_scale(dist, -(rad_i[0] + slavepair.gap_), slavepair.e_ji_);

        // absolute distance between wall contact points
        const double absdist = ParticleUtils::vec_norm_two(dist);

        bool removeslavepair = false;

        // check for coincident contact points of same type (e.g. on line between two surfaces)
        if (master.first == slave.first)
        {
          // contact point already detected (e.g. on line between two surfaces)
          if (absdist <= adaptedtol)
          {
            if (master.second < slave.second) removeslavepair = true;
          }
        }
        // check for line/node contact points within penetration volume of a surface contact point
        else if (master.first == Core::Geo::SURFACE_OBJECT)
        {
          if (absdist <= intersectionradius) removeslavepair = true;
        }
        // check for node contact points within penetration volume of a line contact point
        else if (master.first == Core::Geo::LINE_OBJECT and slave.first == Core::Geo::NODE_OBJECT)
        {
          if (absdist <= intersectionradius) removeslavepair = true;
        }

        if (removeslavepair)
        {
          // mark particle-wall pair (slave) to be removed
          particlewallpairstoremove.insert(slave.second);
          // add global id of slave wall element for interaction history
          masterpair.histeles_.insert(slavepair.ele_->id());
        }
      }
    }
  }

  // erase particle-wall pairs to be removed
  {
    int numparticlewallpairs = particlewallpairdata_.size();

    std::set<int>::reverse_iterator rit;
    for (rit = particlewallpairstoremove.rbegin(); rit != particlewallpairstoremove.rend(); ++rit)
      particlewallpairdata_[*rit] = particlewallpairdata_[--numparticlewallpairs];

    particlewallpairdata_.resize(numparticlewallpairs);
  }
}

void Particle::PDNeighborPairs::communicate_bond_list(
    const std::vector<std::vector<int>>& particletargets)
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;
  std::vector<std::set<std::pair<int, int>>> communicated_bonds_per_target(
      Core::Communication::num_mpi_ranks(comm_));

  // build an index mapping particle global id -> bond indices for O(1) lookup
  std::unordered_map<int, std::vector<std::size_t>> particle_to_bond_indices;
  for (std::size_t bond_idx = 0; bond_idx < bondlist_->size(); ++bond_idx)
  {
    const auto& bond = (*bondlist_)[bond_idx];
    particle_to_bond_indices[std::get<3>(bond.first)].push_back(bond_idx);
    particle_to_bond_indices[std::get<3>(bond.second)].push_back(bond_idx);
  }

  // pack bond pair information
  // do not delete information on proc just add bond information while receiving
  // during later evaluation if any particle of that bond is not available remove the bond
  const int num_procs = Core::Communication::num_mpi_ranks(comm_);
  for (int torank = 0; torank < num_procs; ++torank)
  {
    if (particletargets[torank].empty()) continue;

    for (int globalid : particletargets[torank])
    {
      auto it = particle_to_bond_indices.find(globalid);
      if (it == particle_to_bond_indices.end()) continue;

      for (std::size_t bond_idx : it->second)
      {
        const auto& bond = (*bondlist_)[bond_idx];
        const int globalid_i = std::get<3>(bond.first);
        const int globalid_j = std::get<3>(bond.second);

        const auto canonical_bond_ids = canonical_pd_bond_ids(globalid_i, globalid_j);
        // A bond can be matched twice when both particles move to the same rank.
        // Send each canonical bond at most once per target rank.
        const bool inserted =
            communicated_bonds_per_target[torank].insert(canonical_bond_ids).second;
        if (!inserted) continue;

        Core::Communication::PackBuffer data;
        data.add_to_pack(canonical_bond_ids.first);
        data.add_to_pack(canonical_bond_ids.second);
        sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
      }
    }
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  std::set<long> known_bond_hashes = build_known_peridynamic_bond_hashes(bondlist_);

  // unpack global ids and initialize remaining bond data
  for (auto& p : rdata) unpack_peridynamic_bond_list_data(p.second, known_bond_hashes);
}

void Particle::PDNeighborPairs::unpack_peridynamic_bond_list_data(
    const std::vector<char>& buffer, std::set<long>& known_bond_hashes)
{
  Core::Communication::UnpackBuffer data(buffer);
  while (!data.at_end())
  {
    int globalid_i;
    extract_from_pack(data, globalid_i);
    int globalid_j;
    extract_from_pack(data, globalid_j);

    const auto canonical_bond_ids = canonical_pd_bond_ids(globalid_i, globalid_j);
    const long bond_hash = compute_pd_pair_hash_key(globalid_i, globalid_j);
    // Guard against duplicate communication and duplicate restart payload entries.
    const bool inserted = known_bond_hashes.insert(bond_hash).second;
    if (!inserted) continue;

    // setup default tuples with proper global ids
    Particle::LocalGlobalIndexTuple tuple_i =
        make_default_peridynamic_bond_tuple(canonical_bond_ids.first);
    Particle::LocalGlobalIndexTuple tuple_j =
        make_default_peridynamic_bond_tuple(canonical_bond_ids.second);

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
  std::set<std::pair<int, int>> packed_canonical_bonds;

  // iterate over bond list
  for (const auto& [first, second] : *bondlist_)
  {
    auto [type_i, status_i, particle_i, globalid_i] = first;
    auto [type_j, status_j, particle_j, globalid_j] = second;
    const auto canonical_bond_ids = canonical_pd_bond_ids(globalid_i, globalid_j);

    // Keep restart payload deterministic and free of duplicate unordered bonds.
    const bool inserted = packed_canonical_bonds.insert(canonical_bond_ids).second;
    if (!inserted) continue;

    // add bond pair to buffer
    buffer.add_to_pack(canonical_bond_ids.first);
    buffer.add_to_pack(canonical_bond_ids.second);
  }
}

void Particle::PDNeighborPairs::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader)
{
  // peridynamic bond list
  std::shared_ptr<std::vector<char>> buffer = std::make_shared<std::vector<char>>();
  reader->read_char_vector(buffer, "PeridynamicBondList");
  if (!buffer->empty())
  {
    std::set<long> known_bond_hashes = build_known_peridynamic_bond_hashes(bondlist_);
    unpack_peridynamic_bond_list_data(*buffer, known_bond_hashes);
  }
}

namespace
{
  bool is_valid_peridynamic_bond_entry(
      const int localid, const int globalid, Particle::ParticleContainer* container)
  {
    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // check if the localid is smaller than the number of stored particles in the container
    if (localid >= particlestored) return false;
    // get global id of particle i
    const int* globalid_provided = container->get_ptr_to_global_id(localid);

    // check if the provided global id by the container matches that of the bond list
    if (globalid == globalid_provided[0]) return true;

    return false;
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE
