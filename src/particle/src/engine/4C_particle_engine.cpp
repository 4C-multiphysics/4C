// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine.hpp"

#include "4C_binstrategy.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_engine_runtime_vtp_writer.hpp"
#include "4C_particle_engine_unique_global_id.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ParticleEngine::ParticleEngine(MPI_Comm comm, const Teuchos::ParameterList& params)
    : comm_(comm),
      myrank_(Core::Communication::my_mpi_rank(comm)),
      params_(params),
      minbinsize_(0.0),
      typevectorsize_(0),
      particlecontainerbundle_(std::make_shared<ParticleContainerBundle>()),
      particleuniqueglobalidhandler_(std::make_unique<UniqueGlobalIdHandler>(comm_, "particle")),
      validownedparticles_(false),
      validghostedparticles_(false),
      validparticleneighbors_(false),
      validglobalidtolocalindex_(false),
      validdirectghosting_(false),
      validhalfneighboringbins_(false)
{
  // init binning strategy
  init_binning_strategy();

  // init particle runtime vtp writer
  init_particle_vtp_writer();
}

Particle::ParticleEngine::~ParticleEngine() = default;

void Particle::ParticleEngine::setup(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes)
{
  // setup binning strategy
  setup_binning_strategy();

  // setup particle container bundle
  setup_particle_container_bundle(particlestatestotypes);

  // setup data storage
  setup_data_storage(particlestatestotypes);

  // setup particle runtime vtp writer
  setup_particle_vtp_writer();

  // setup particle type weights for dynamic load balancing
  setup_type_weights();
}

void Particle::ParticleEngine::write_restart(const int step, const double time) const
{
  // get bin discretization writer
  std::shared_ptr<Core::IO::DiscretizationWriter> binwriter = binstrategy_->bin_discret()->writer();

  binwriter->new_step(step, time);

  // pack particles of all containers
  std::shared_ptr<std::vector<char>> particlebuffer = std::make_shared<std::vector<char>>();
  particlecontainerbundle_->get_packed_particle_objects_of_all_containers(*particlebuffer);

  // write particle data
  binwriter->write_char_data("ParticleData", *Teuchos::rcp(particlebuffer));

  // write restart of unique global identifier handler
  particleuniqueglobalidhandler_->write_restart(binwriter);
}

void Particle::ParticleEngine::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader,
    std::vector<ParticleObjShrdPtr>& particlestoread) const
{
  // read particle data
  std::shared_ptr<std::vector<char>> particledata = std::make_shared<std::vector<char>>();
  reader->read_char_vector(particledata, "ParticleData");

  Core::Communication::UnpackBuffer buffer(*particledata);
  while (!buffer.at_end())
  {
    std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
    ParticleObjShrdPtr particleobject = std::dynamic_pointer_cast<ParticleObject>(object);
    if (particleobject == nullptr) FOUR_C_THROW("received object is not a particle object!");

    // store read particle
    particlestoread.push_back(particleobject);
  }

  // read restart of unique global identifier handler
  particleuniqueglobalidhandler_->read_restart(reader);

  // read restart of runtime vtp writer
  particlevtpwriter_->read_restart(reader);
}

void Particle::ParticleEngine::write_particle_runtime_output(
    const int step, const double time) const
{
  particlevtpwriter_->set_particle_positions_and_states();
  particlevtpwriter_->write_to_disk(time, step);
}

void Particle::ParticleEngine::free_unique_global_ids(std::vector<int>& freeuniquegids)
{
  // insert freed global id
  for (const int currfreegid : freeuniquegids)
    particleuniqueglobalidhandler_->insert_freed_global_id(currfreegid);

  // clear after all unique global ids are freed
  freeuniquegids.clear();
}

void Particle::ParticleEngine::get_unique_global_ids_for_all_particles(
    std::vector<ParticleObjShrdPtr>& particlestogetuniquegids)
{
  std::vector<int> requesteduniqueglobalids;
  requesteduniqueglobalids.reserve(particlestogetuniquegids.size());

  // draw requested number of global ids
  particleuniqueglobalidhandler_->draw_requested_number_of_global_ids(requesteduniqueglobalids);

  // number of particles to get unique ids
  int numparticles = particlestogetuniquegids.size();

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
    particlestogetuniquegids[i]->set_particle_global_id(requesteduniqueglobalids[i]);
}

void Particle::ParticleEngine::check_number_of_unique_global_ids()
{
  // get number of particles on all processors
  int numberofparticles = get_number_of_particles();
  MPI_Allreduce(MPI_IN_PLACE, &numberofparticles, 1, MPI_INT, MPI_SUM, comm_);

  // get number of reusable global ids on all processors
  int numberofreusableglobalids =
      particleuniqueglobalidhandler_->get_number_of_reusable_global_ids();
  MPI_Allreduce(MPI_IN_PLACE, &numberofreusableglobalids, 1, MPI_INT, MPI_SUM, comm_);

  // maximum global id
  const int maxglobalid = particleuniqueglobalidhandler_->get_max_global_id();

  // safety check
  if (numberofparticles + numberofreusableglobalids != (maxglobalid + 1))
    FOUR_C_THROW("sum of particles and reusable global ids unequal total global ids: {} + {} != {}",
        numberofparticles, numberofreusableglobalids, (maxglobalid + 1));
}

void Particle::ParticleEngine::erase_particles_outside_bounding_box(
    std::vector<ParticleObjShrdPtr>& particlestocheck)
{
  // get bounding box dimensions
  Core::LinAlg::Matrix<3, 2> boundingbox = binstrategy_->domain_bounding_box_corner_positions();

  // set of particles located outside bounding box
  std::set<int> particlesoutsideboundingbox;

  // number of particles to check
  int numparticles = particlestocheck.size();

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
  {
    // get states of particle
    const ParticleStates& states = particlestocheck[i]->return_particle_states();

    // get position of particle
    const std::vector<double>& pos = states[Position];

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // get type of particles
    ParticleType type = particlestocheck[i]->return_particle_type();

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    if (static_cast<int>(pos.size()) != container->get_state_dim(Position))
      FOUR_C_THROW("dimension of particle state '{}' not valid!", enum_to_state_name(Position));
#endif

    // check particle location with respect to bounding box in each spatial directions
    for (int dim = 0; dim < 3; ++dim)
    {
      // particle located outside bounding box
      if ((pos[dim] < boundingbox(dim, 0)) or (pos[dim] > boundingbox(dim, 1)))
      {
        // insert particle into set
        particlesoutsideboundingbox.insert(i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (particlestocheck[i]->return_particle_global_id() < 0)
          FOUR_C_THROW("no global id assigned to particle!");
#endif

        // insert freed global id
        particleuniqueglobalidhandler_->insert_freed_global_id(
            particlestocheck[i]->return_particle_global_id());

        break;
      }
    }
  }

  // number of particles located outside bounding box
  const int numparticlesoutside = particlesoutsideboundingbox.size();

  // no particles located outside of bounding box
  if (numparticlesoutside == 0) return;

  // put particles to be erased at the end of the vector
  int swapposition = numparticles - 1;

  // iterate in reversed order over particles to be erased
  std::set<int>::reverse_iterator rit;
  for (rit = particlesoutsideboundingbox.rbegin(); rit != particlesoutsideboundingbox.rend(); ++rit)
  {
    // swap position of particle to be erase
    std::swap(particlestocheck[*rit], particlestocheck[swapposition]);

    // set new swap position
    --swapposition;
  }

  // erase particles located outside bounding box
  particlestocheck.resize(numparticles - numparticlesoutside);

  // short screen output
  if (numparticlesoutside)
    Core::IO::cout << "on processor " << myrank_ << " removed " << numparticlesoutside
                   << " particle(s) being outside the computational domain!" << Core::IO::endl;
}

void Particle::ParticleEngine::distribute_particles(
    std::vector<ParticleObjShrdPtr>& particlestodistribute)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::DistributeParticles");

  std::vector<std::vector<ParticleObjShrdPtr>> particlestosend(
      Core::Communication::num_mpi_ranks(comm_));
  std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>> particlestoinsert(typevectorsize_);

  // determine particles that need to be distributed
  determine_particles_to_be_distributed(particlestodistribute, particlestosend, particlestoinsert);

  // communicate particles
  communicate_particles(particlestosend, particlestoinsert);

  // insert owned particles received from other processors
  insert_owned_particles(particlestoinsert);

  // store particle positions after transfer of particles
  store_positions_after_particle_transfer();

  // relate owned particles to bins
  relate_owned_particles_to_bins();
}

void Particle::ParticleEngine::transfer_particles()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::TransferParticles");

  std::vector<std::set<int>> particlestoremove(typevectorsize_);
  std::vector<std::vector<ParticleObjShrdPtr>> particlestosend(
      Core::Communication::num_mpi_ranks(comm_));
  std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>> particlestoinsert(typevectorsize_);

  // relate owned particles to bins
  if (not validownedparticles_) relate_owned_particles_to_bins();

  // check particles for periodic boundaries/leaving domain
  check_particles_at_boundaries(particlestoremove);

  // determine particles that need to be transferred
  determine_particles_to_be_transferred(particlestoremove, particlestosend);

  // remove particles from containers
  remove_particles_from_containers(particlestoremove);

  // communicate particles
  communicate_particles(particlestosend, particlestoinsert);

  // insert owned particles received from other processors
  insert_owned_particles(particlestoinsert);

  // store particle positions after transfer of particles
  store_positions_after_particle_transfer();

  // relate owned particles to bins
  relate_owned_particles_to_bins();

  // check and decrease the size of all containers of owned particles
  particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Owned);
}

void Particle::ParticleEngine::ghost_particles()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::GhostParticles");

  std::vector<std::vector<ParticleObjShrdPtr>> particlestosend(
      Core::Communication::num_mpi_ranks(comm_));
  std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>> particlestoinsert(typevectorsize_);
  std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>> directghosting;

  // clear all containers of ghosted particles
  particlecontainerbundle_->clear_all_containers_of_specific_status(Ghosted);

  // determine particles that need to be ghosted
  determine_particles_to_be_ghosted(particlestosend);

  // communicate particles
  communicate_particles(particlestosend, particlestoinsert);

  // insert ghosted particles received from other processors
  insert_ghosted_particles(particlestoinsert, directghosting);

  // communicate and build map for direct ghosting
  communicate_direct_ghosting_map(directghosting);

  // check and decrease the size of all containers of ghosted particles
  particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Ghosted);
}

void Particle::ParticleEngine::refresh_particles() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::RefreshParticles");

  std::vector<std::vector<ParticleObjShrdPtr>> particlestosend(
      Core::Communication::num_mpi_ranks(comm_));
  std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>> particlestoinsert(typevectorsize_);

  // determine particles that need to be refreshed
  determine_particles_to_be_refreshed(particlestosend);

  // communicate particles
  communicate_particles(particlestosend, particlestoinsert);

  // insert refreshed particles received from other processors
  insert_refreshed_particles(particlestoinsert);
}

void Particle::ParticleEngine::refresh_particles_of_specific_states_and_types(
    const StatesOfTypesToRefresh& particlestatestotypes) const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::ParticleEngine::refresh_particles_of_specific_states_and_types");

  std::vector<std::vector<ParticleObjShrdPtr>> particlestosend(
      Core::Communication::num_mpi_ranks(comm_));
  std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>> particlestoinsert(typevectorsize_);

  // determine particles that need to be refreshed
  determine_specific_states_of_particles_of_specific_types_to_be_refreshed(
      particlestatestotypes, particlestosend);

  // communicate particles
  communicate_particles(particlestosend, particlestoinsert);

  // insert refreshed particles received from other processors
  insert_refreshed_particles(particlestoinsert);
}

void Particle::ParticleEngine::dynamic_load_balancing()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::dynamic_load_balancing");

  // determine bin weights needed for repartitioning
  determine_bin_weights();

  // distribute bins via recursive coordinate bisection
  binstrategy_->distribute_bins_recurs_coord_bisection(binrowmap_, bincenters_, binweights_);

  // export elements to new layout
  binstrategy_->bin_discret()->export_row_elements(*binrowmap_);

  // setup ghosting of bins
  setup_bin_ghosting();

  // determine bin distribution dependent maps/sets
  determine_bin_dis_dependent_maps_and_sets();

  // determine ghosting dependent maps/sets for communication
  determine_ghosting_dependent_maps_and_sets();

  // prepare storage for particle objects
  std::vector<ParticleObjShrdPtr> particlestodistribute;
  particlestodistribute.reserve(get_number_of_particles());

  // get vector of particle objects of all containers
  particlecontainerbundle_->get_vector_of_particle_objects_of_all_containers(particlestodistribute);

  // clear all containers of owned particles
  particlecontainerbundle_->clear_all_containers_of_specific_status(Owned);

  // invalidate particle safety flags
  invalidate_particle_safety_flags();

  // invalidate flag denoting valid relation of half surrounding neighboring bins to owned bins
  validhalfneighboringbins_ = false;

  // distribute particles to owning processor
  distribute_particles(particlestodistribute);

  // check and decrease the size of all containers of owned particles
  particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Owned);
}

void Particle::ParticleEngine::hand_over_particles_to_be_removed(
    std::vector<std::set<int>>& particlestoremove)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::hand_over_particles_to_be_removed");

  // number of particles to remove
  int numparticlestoremove = 0;
  for (auto typeIt : particlestoremove) numparticlestoremove += typeIt.size();

  if (numparticlestoremove)
  {
    // remove particles from containers
    remove_particles_from_containers(particlestoremove);

    // check and decrease the size of all containers of owned particles
    particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(Owned);
  }
}

void Particle::ParticleEngine::hand_over_particles_to_be_inserted(
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::hand_over_particles_to_be_inserted");

  // number of particles to insert
  int numparticlestoinsert = 0;
  for (auto typeIt : particlestoinsert) numparticlestoinsert += typeIt.size();

  if (numparticlestoinsert)
  {
    // insert owned particles
    insert_owned_particles(particlestoinsert);
  }
}

void Particle::ParticleEngine::build_particle_to_particle_neighbors()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::build_particle_to_particle_neighbors");

  // safety check
  if ((not validownedparticles_) or (not validghostedparticles_))
    FOUR_C_THROW("invalid relation of particles to bins!");

  // relate half neighboring bins to owned bins
  if (not validhalfneighboringbins_) relate_half_neighboring_bins_to_owned_bins();

  // clear potential particle neighbors
  potentialparticleneighbors_.clear();

  // invalidate flag denoting validity of particle neighbors map
  validparticleneighbors_ = false;

  // loop over row bins
  for (int rowlidofbin = 0; rowlidofbin < binrowmap_->num_my_elements(); ++rowlidofbin)
  {
    // get global id of bin
    const int gidofbin = binrowmap_->gid(rowlidofbin);

    // get local id of bin
    const int collidofbin = bincolmap_->lid(gidofbin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particles of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      // get global id of particle
      const int* currglobalid = container->get_ptr_to_global_id(ownedindex);

      // get position of particle
      const double* currpos = container->get_ptr_to_state(Position, ownedindex);

      // iterate over neighboring bins (including current bin)
      for (int gidofneighborbin : halfneighboringbinstobins_[rowlidofbin])
      {
        // get local id of neighboring bin
        const int collidofneighboringbin = bincolmap_->lid(gidofneighborbin);

        // check if current neighboring bin contains particles
        if (particlestobins_[collidofneighboringbin].empty()) continue;

        // get status of neighboring particles
        ParticleStatus neighborstatus = (binrowmap_->lid(gidofneighborbin) < 0) ? Ghosted : Owned;

        // iterate over particles in current neighboring bin
        for (const auto& neighborParticleIt : particlestobins_[collidofneighboringbin])
        {
          // get type of neighboring particle
          ParticleType neighbortype = neighborParticleIt.first;

          // get local index of neighboring particle
          const int neighborindex = neighborParticleIt.second;

          // get container of neighboring particle of current particle type
          ParticleContainer* neighborcontainer =
              particlecontainerbundle_->get_specific_container(neighbortype, neighborstatus);

          // get global id of neighboring particle
          const int* neighborglobalid = neighborcontainer->get_ptr_to_global_id(neighborindex);

          // avoid duplicate neighbor pairs and self-neighboring
          if (gidofbin == gidofneighborbin and neighborglobalid[0] <= currglobalid[0]) continue;

          // get position of neighboring particle
          const double* neighborpos = neighborcontainer->get_ptr_to_state(Position, neighborindex);

          // distance vector from owned particle to neighboring particle
          double dist[3];

          // distance between particles considering periodic boundaries
          distance_between_particles(currpos, neighborpos, dist);

          // distance between particles larger than minimum bin size
          if (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] >
              (minbinsize_ * minbinsize_))
            continue;

          // append potential particle neighbor pair
          potentialparticleneighbors_.push_back(
              std::make_pair(std::make_tuple(type, Owned, ownedindex),
                  std::make_tuple(neighbortype, neighborstatus, neighborindex)));
        }
      }
    }
  }

  // validate flag denoting validity of particle neighbors map
  validparticleneighbors_ = true;
}

void Particle::ParticleEngine::build_global_id_to_local_index_map()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::ParticleEngine::build_global_id_to_local_index_map");

  // clear map
  globalidtolocalindex_.clear();

  // invalidate flag denoting validity of map relating particle global ids to local index
  validglobalidtolocalindex_ = false;

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // iterate over particle statuses
    for (const auto& status : {Owned, Ghosted})
    {
      // get container of current particle type and current status
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, status);

      // get number of particles stored in container
      const int particlestored = container->particles_stored();

      // no particles of current type and current status
      if (particlestored <= 0) continue;

      // get pointer to global id of particles
      int* globalids = container->get_ptr_to_global_id(0);

      // loop over particles in container
      for (int index = 0; index < particlestored; ++index)
      {
        // get global id of particle
        int globalid = globalids[index];

        // add entry to map
        globalidtolocalindex_[globalid] = std::make_shared<LocalIndexTuple>(type, status, index);
      }
    }
  }

  // validate flag denoting validity of map relating particle global ids to local index
  validglobalidtolocalindex_ = true;
}

bool Particle::ParticleEngine::have_valid_particle_connectivity() const
{
  int localcheck = ((validownedparticles_ and validghostedparticles_ and
                     validglobalidtolocalindex_ and validdirectghosting_));

  // check among all processors
  int globalcheck = 0;
  globalcheck = Core::Communication::min_all(localcheck, comm_);

  return globalcheck;
}

bool Particle::ParticleEngine::have_valid_particle_neighbors() const
{
  int localcheck = validparticleneighbors_;

  // check among all processors
  int globalcheck = 0;
  globalcheck = Core::Communication::min_all(localcheck, comm_);

  return globalcheck;
}

const Particle::ParticlesToBins& Particle::ParticleEngine::get_particles_to_bins() const
{
  // safety check
  if ((not validownedparticles_) or (not validghostedparticles_))
    FOUR_C_THROW("invalid map relating particles to bins!");

  return particlestobins_;
}

const Particle::PotentialParticleNeighbors&
Particle::ParticleEngine::get_potential_particle_neighbors() const
{
  // safety check
  if (not validparticleneighbors_) FOUR_C_THROW("invalid particle neighbors!");

  return potentialparticleneighbors_;
}

Particle::LocalIndexTupleShrdPtr Particle::ParticleEngine::get_local_index_in_specific_container(
    int globalid) const
{
  // safety check
  if (not validglobalidtolocalindex_) FOUR_C_THROW("invalid global id to local index map!");

  // get local index of particle in specific container
  auto globalidIt = globalidtolocalindex_.find(globalid);
  if (globalidIt == globalidtolocalindex_.end()) return nullptr;

  return globalidIt->second;
}

std::shared_ptr<Core::IO::DiscretizationWriter>
Particle::ParticleEngine::get_bin_discretization_writer() const
{
  return binstrategy_->bin_discret()->writer();
}

void Particle::ParticleEngine::relate_all_particles_to_all_procs(
    std::vector<int>& particlestoproc) const
{
  // global ids on this processor
  std::vector<int> thisprocglobalids;

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no particles of current type and current status
    if (particlestored <= 0) continue;

    // get pointer to global id of particles
    int* globalids = container->get_ptr_to_global_id(0);

    // insert global id of particles
    thisprocglobalids.insert(thisprocglobalids.end(), globalids, globalids + particlestored);
  }

  // get maximum global id on this processor
  int thisprocmaxglobalid = 0;
  if (not thisprocglobalids.empty())
    thisprocmaxglobalid = *std::max_element(thisprocglobalids.begin(), thisprocglobalids.end());

  // get maximum global id on all processors
  int allprocmaxglobalid(0);
  allprocmaxglobalid = Core::Communication::max_all(thisprocmaxglobalid, comm_);

  // resize to hold all particles
  const int vecsize = allprocmaxglobalid + 1;
  particlestoproc.resize(vecsize, -1);

  // relate this processor id to its global ids
  for (int globalid : thisprocglobalids) particlestoproc[globalid] = myrank_;

  // communicate global ids between all processors
  MPI_Allreduce(MPI_IN_PLACE, particlestoproc.data(), vecsize, MPI_INT, MPI_MAX, comm_);
}

void Particle::ParticleEngine::get_particles_within_radius(const double* position,
    const double radius, std::vector<LocalIndexTuple>& neighboringparticles) const
{
  // safety check
  if ((not validownedparticles_) or (not validghostedparticles_))
    FOUR_C_THROW("invalid relation of particles to bins!");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // bin size safety check
  if (radius > minbinsize_)
    FOUR_C_THROW(
        "the given radius is larger than the minimal bin size ({} > {})!", radius, minbinsize_);
#endif

  // get global id of bin
  const int gidofbin = binstrategy_->convert_pos_to_gid(position);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // position outside computational domain
  if (gidofbin == -1) FOUR_C_THROW("position outside of computational domain!");

  // bin not owned or ghosted by this processor
  if (bincolmap_->lid(gidofbin) < 0)
    FOUR_C_THROW("position not in owned or ghosted bin on this processor!");
#endif

  // get neighboring bins to current bin
  std::vector<int> binvec;
  binstrategy_->get_neighbor_and_own_bin_ids(gidofbin, binvec);

  // iterate over neighboring bins
  for (int gidofneighborbin : binvec)
  {
    // get local id of neighboring bin
    const int collidofneighboringbin = bincolmap_->lid(gidofneighborbin);

    // neighboring bin not ghosted by this processor
    if (collidofneighboringbin < 0) continue;

    // check if current neighboring bin contains particles
    if (particlestobins_[collidofneighboringbin].empty()) continue;

    // get status of neighboring particles
    ParticleStatus neighborstatus = (binrowmap_->lid(gidofneighborbin) < 0) ? Ghosted : Owned;

    // iterate over particles in current neighboring bin
    for (const auto& neighborParticleIt : particlestobins_[collidofneighboringbin])
    {
      // get type of neighboring particle
      ParticleType neighbortype = neighborParticleIt.first;

      // get local index of neighboring particle
      const int neighborindex = neighborParticleIt.second;

      // get container of neighboring particle of current particle type
      ParticleContainer* neighborcontainer =
          particlecontainerbundle_->get_specific_container(neighbortype, neighborstatus);

      // get position of neighboring particle
      const double* neighborpos = neighborcontainer->get_ptr_to_state(Position, neighborindex);

      // distance vector from position to neighboring particle
      double dist[3];

      // distance between position and neighboring particle considering periodic boundaries
      distance_between_particles(position, neighborpos, dist);

      // distance between position and neighboring particle larger than radius
      if (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2] > (radius * radius)) continue;

      // append neighboring particle
      neighboringparticles.push_back(std::make_tuple(neighbortype, neighborstatus, neighborindex));
    }
  }
}

std::array<double, 3> Particle::ParticleEngine::bin_size() const
{
  return binstrategy_->bin_size();
}

bool Particle::ParticleEngine::have_periodic_boundary_conditions() const
{
  return binstrategy_->have_periodic_boundary_conditions_applied();
}

bool Particle::ParticleEngine::have_periodic_boundary_conditions_in_spatial_direction(
    const int dim) const
{
  return binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(dim);
}

double Particle::ParticleEngine::length_of_binning_domain_in_a_spatial_direction(
    const int dim) const
{
  return binstrategy_->length_of_binning_domain_in_a_spatial_direction(dim);
}

Core::LinAlg::Matrix<3, 2> const& Particle::ParticleEngine::domain_bounding_box_corner_positions()
    const
{
  return binstrategy_->domain_bounding_box_corner_positions();
}

void Particle::ParticleEngine::distance_between_particles(
    const double* pos_i, const double* pos_j, double* r_ji) const
{
  // loop over all spatial directions
  for (int dim = 0; dim < 3; ++dim)
  {
    // vector from particle i to j
    r_ji[dim] = pos_j[dim] - pos_i[dim];

    // check for periodic boundary condition in current spatial direction
    if (binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(dim))
    {
      // binning domain length in current spatial direction
      double binningdomainlength =
          binstrategy_->length_of_binning_domain_in_a_spatial_direction(dim);

      // shift by periodic length if particles are closer over periodic boundary
      if (std::abs(r_ji[dim]) > (0.5 * binningdomainlength))
      {
        if (pos_i[dim] < pos_j[dim])
          r_ji[dim] -= binningdomainlength;
        else
          r_ji[dim] += binningdomainlength;
      }
    }
  }
}

std::shared_ptr<Core::IO::DiscretizationReader> Particle::ParticleEngine::bin_dis_reader(
    int restartstep) const
{
  return std::make_shared<Core::IO::DiscretizationReader>(
      *binstrategy_->bin_discret(), Global::Problem::instance()->input_control_file(), restartstep);
}

int Particle::ParticleEngine::get_number_of_particles() const
{
  int numberofparticles = 0;

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // add number of particles stored in container
    numberofparticles += container->particles_stored();
  }

  return numberofparticles;
}

int Particle::ParticleEngine::get_number_of_particles_of_specific_type(
    const ParticleType type) const
{
  if (not particlecontainerbundle_->get_particle_types().contains(type)) return 0;

  // get container of owned particles of specific particle type
  ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

  return container->particles_stored();
}

void Particle::ParticleEngine::write_bin_dis_output(const int step, const double time) const
{
  // write bins to output file
  binstrategy_->write_bin_output(step, time);
}

void Particle::ParticleEngine::init_binning_strategy()
{
  // create and init binning strategy and create bins
  Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
  Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
      "spatial_approximation_type", Global::Problem::instance()->spatial_approximation_type(),
      binning_params);
  binstrategy_ = std::make_shared<Core::Binstrategy::BinningStrategy>(binning_params,
      Global::Problem::instance()->output_control_file(), comm_,
      Core::Communication::my_mpi_rank(comm_));
}

void Particle::ParticleEngine::setup_binning_strategy()
{
  // determine minimum relevant bin size
  determine_min_relevant_bin_size();

  // create an initial linear distribution of row bins
  binrowmap_ = binstrategy_->create_linear_map_for_numbin(comm_);

  // initialize vector for storage of bin center coordinates and bin weights
  bincenters_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*binrowmap_, 3);
  binweights_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*binrowmap_, 1);

  // get all bin centers needed for repartitioning
  binstrategy_->get_all_bin_centers(*binrowmap_, *bincenters_);

  // initialize weights of all bins
  binweights_->put_scalar(1.0e-05);

  // distribute bins via recursive coordinate bisection
  binstrategy_->distribute_bins_recurs_coord_bisection(binrowmap_, bincenters_, binweights_);

  // create bins and fill bins into binning discretization
  binstrategy_->fill_bins_into_bin_discretization(*binrowmap_);

  // setup ghosting of bins
  setup_bin_ghosting();

  // determine bin distribution dependent maps/sets
  determine_bin_dis_dependent_maps_and_sets();

  // determine ghosting dependent maps/sets for communication
  determine_ghosting_dependent_maps_and_sets();
}

void Particle::ParticleEngine::setup_bin_ghosting()
{
  // gather bins of row map and all its neighbors (row + ghost)
  std::set<int> bins;
  for (int lid = 0; lid < binrowmap_->num_my_elements(); ++lid)
  {
    int gidofbin = binrowmap_->gid(lid);
    std::vector<int> binvec;
    // get neighboring bins
    binstrategy_->get_neighbor_and_own_bin_ids(gidofbin, binvec);
    bins.insert(binvec.begin(), binvec.end());
  }

  // remove non-existing ghost bins from original bin set
  {
    // create copy of column bins
    std::set<int> ghostbins(bins);
    // find ghost bins and check for existence
    for (int lid = 0; lid < binrowmap_->num_my_elements(); ++lid)
    {
      const int gid = binrowmap_->gid(lid);
      std::set<int>::iterator iter = ghostbins.find(gid);
      if (iter != ghostbins.end()) ghostbins.erase(iter);
    }
    // only ghost bins remain
    std::vector<int> ghostbins_vec(ghostbins.begin(), ghostbins.end());
    const int size = static_cast<int>(ghostbins.size());
    std::vector<int> pidlist(size);
    const int err = binrowmap_->remote_id_list(size, ghostbins_vec.data(), pidlist.data(), nullptr);
    if (err < 0) FOUR_C_THROW("Core::LinAlg::Map::RemoteIDList returned err={}", err);

    for (int i = 0; i < size; ++i)
    {
      if (pidlist[i] == -1)
      {
        std::set<int>::iterator iter = bins.find(ghostbins_vec[i]);
        if (iter == bins.end()) FOUR_C_THROW("bin id is missing in bin set");
        // erase non-existing id
        bins.erase(iter);
      }
    }
  }

  // copy bin gids to a vector and create bincolmap
  std::vector<int> bincolmapvec(bins.begin(), bins.end());
  bincolmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, static_cast<int>(bincolmapvec.size()), bincolmapvec.data(), 0, comm_);

  if (bincolmap_->num_global_elements() == 1 && Core::Communication::num_mpi_ranks(comm_) > 1)
    FOUR_C_THROW("one bin cannot be run in parallel -> reduce BIN_SIZE_LOWER_BOUND");

  // make sure that all processors are either filled or unfilled
  binstrategy_->bin_discret()->check_filled_globally();

  // create ghosting for bins
  binstrategy_->bin_discret()->extended_ghosting(*bincolmap_, true, false, true, false);
}

void Particle::ParticleEngine::setup_particle_container_bundle(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes) const
{
  // setup particle container bundle
  particlecontainerbundle_->setup(particlestatestotypes);
}

void Particle::ParticleEngine::setup_data_storage(
    const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes)
{
  // determine size of vectors indexed by particle types
  typevectorsize_ = ((--particlestatestotypes.end())->first) + 1;

  // allocate memory to hold particle types
  directghostingtargets_.resize(typevectorsize_);

  // allocate memory for particles being communicated to target processors
  communicatedparticletargets_.assign(
      Core::Communication::num_mpi_ranks(comm_), std::vector<int>(0));
}

void Particle::ParticleEngine::init_particle_vtp_writer()
{
  // construct and init particle runtime vtp writer
  particlevtpwriter_ = std::make_unique<ParticleRuntimeVtpWriter>(comm_);
  particlevtpwriter_->init(particlecontainerbundle_);
}

void Particle::ParticleEngine::setup_particle_vtp_writer() const
{
  // get flag to determine output of ghosted particles (debug feature)
  bool write_ghosted_particles = params_.get<bool>("WRITE_GHOSTED_PARTICLES");

  // setup particle runtime vtp writer
  particlevtpwriter_->setup(write_ghosted_particles);
}

void Particle::ParticleEngine::setup_type_weights()
{
  // allocate memory to hold particle types
  typeweights_.resize(typevectorsize_);

  // init map relating particle types to dynamic load balance factor
  std::map<ParticleType, double> typetodynloadbal;

  // read parameters relating particle types to values
  ParticleUtils::read_params_types_related_to_values(
      params_, "PHASE_TO_DYNLOADBALFAC", typetodynloadbal);

  // insert weight of particle type
  for (const auto& typeIt : typetodynloadbal) typeweights_[typeIt.first] = typeIt.second;
}

void Particle::ParticleEngine::determine_bin_dis_dependent_maps_and_sets()
{
  // clear sets and maps
  boundarybins_.clear();
  touchedbins_.clear();
  firstlayerbinsownedby_.clear();

  // check for finalized construction of binning discretization
  if (binstrategy_->bin_discret()->filled() == false)
    FOUR_C_THROW("construction of binning discretization not finalized!");

  // loop over row bins
  for (int rowlidofbin = 0; rowlidofbin < binrowmap_->num_my_elements(); ++rowlidofbin)
  {
    int currbin = binrowmap_->gid(rowlidofbin);

    // first insert all owned bins
    boundarybins_.insert(currbin);

    // get neighboring bins
    std::vector<int> binvec;
    binstrategy_->get_neighbor_bin_ids(currbin, binvec);

    // iterate over neighboring bins
    for (int neighbin : binvec)
    {
      // neighboring bin not owned by this processor
      if (binrowmap_->lid(neighbin) < 0)
      {
        // insert owned bin
        touchedbins_.insert(currbin);

        // insert owner of neighbouring bin
        int neighbinowner = binstrategy_->bin_discret()->g_element(neighbin)->owner();
        firstlayerbinsownedby_.insert(std::make_pair(neighbin, neighbinowner));
      }
    }
  }

  // determine all non-boundary bins
  std::set<int> innerbinids;

  // get number of bins in all spatial directions
  const auto binperdir = binstrategy_->bin_per_dir();

  // safety check
  for (int dim = 0; dim < 3; ++dim)
    if (binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(dim) and
        binperdir[dim] < 3)
      FOUR_C_THROW("at least 3 bins in direction with periodic boundary conditions necessary!");

  // determine range of all inner bins
  std::vector<int> ijk_min(3);
  std::vector<int> ijk_max(3);
  for (int dim = 0; dim < 3; ++dim)
  {
    ijk_min[dim] = (binperdir[dim] > 2) ? 1 : 0;
    ijk_max[dim] = (binperdir[dim] > 2) ? (binperdir[dim] - 2) : (binperdir[dim] - 1);
  }

  // ijk_range of inner bins (contains: i_min i_max j_min j_max k_min k_max)
  int ijk_range[] = {ijk_min[0], ijk_max[0], ijk_min[1], ijk_max[1], ijk_min[2], ijk_max[2]};

  // get corresponding owned bin ids in ijk range
  binstrategy_->gids_in_ijk_range(ijk_range, innerbinids, true);

  // subtract non-boundary bins from all owned bins to obtain boundary bins
  for (int currbin : innerbinids) boundarybins_.erase(currbin);
}

void Particle::ParticleEngine::determine_ghosting_dependent_maps_and_sets()
{
  // clear sets and maps
  ghostedbins_.clear();
  thisbinsghostedby_.clear();

  // check for finalized construction of binning discretization
  if (binstrategy_->bin_discret()->filled() == false)
    FOUR_C_THROW("construction of binning discretization not finalized!");

  // loop over col bins
  for (int collidofbin = 0; collidofbin < bincolmap_->num_my_elements(); ++collidofbin)
  {
    int currbin = bincolmap_->gid(collidofbin);

    // current bin not owned by this processor
    if (binrowmap_->lid(currbin) < 0) ghostedbins_.insert(currbin);
  }

  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // pack data for sending
  Core::Communication::PackBuffer data;
  add_to_pack(data, ghostedbins_);

  // communicate ghosted bins between all processors
  for (int torank = 0; torank < Core::Communication::num_mpi_ranks(comm_); ++torank)
  {
    if (torank == myrank_) continue;

    sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // init receiving vector
  std::set<int> receivedbins;

  // unpack and store received data
  for (const auto& p : rdata)
  {
    const int msgsource = p.first;
    const std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      extract_from_pack(buffer, receivedbins);

      // iterate over received bins
      for (int receivedbin : receivedbins)
      {
        // received bin is owned by this processor
        if (binrowmap_->lid(receivedbin) >= 0) (thisbinsghostedby_[receivedbin]).insert(msgsource);
      }
    }
  }
}

void Particle::ParticleEngine::relate_half_neighboring_bins_to_owned_bins()
{
  // allocate memory for neighbors of owned bins
  halfneighboringbinstobins_.assign(binrowmap_->num_my_elements(), std::set<int>());

  // loop over row bins
  for (int rowlidofbin = 0; rowlidofbin < binrowmap_->num_my_elements(); ++rowlidofbin)
  {
    // get global id of bin
    const int gidofbin = binrowmap_->gid(rowlidofbin);

    // get ijk of current bin
    int ijk[3];
    binstrategy_->convert_gid_to_ijk(gidofbin, ijk);

    // get reference to neighboring bins (including current bin) of current bin
    std::set<int>& neighboringbins = halfneighboringbinstobins_[rowlidofbin];

    // insert current bin id
    neighboringbins.insert(gidofbin);

    // insert half of the surrounding bins following a specific stencil
    int ijk_range_9bin[] = {ijk[0] - 1, ijk[0] + 1, ijk[1] - 1, ijk[1] + 1, ijk[2] + 1, ijk[2] + 1};
    binstrategy_->gids_in_ijk_range(ijk_range_9bin, neighboringbins, false);

    int ijk_range_3bin[] = {ijk[0] + 1, ijk[0] + 1, ijk[1] - 1, ijk[1] + 1, ijk[2], ijk[2]};
    binstrategy_->gids_in_ijk_range(ijk_range_3bin, neighboringbins, false);

    int ijk_range_1bin[] = {ijk[0], ijk[0], ijk[1] + 1, ijk[1] + 1, ijk[2], ijk[2]};
    binstrategy_->gids_in_ijk_range(ijk_range_1bin, neighboringbins, false);
  }

  // iterate over bins being ghosted on this processor
  for (int gidofbin : ghostedbins_)
  {
    // get neighboring bins
    std::vector<int> binvec;
    binstrategy_->get_neighbor_bin_ids(gidofbin, binvec);

    // iterate over neighboring bins
    for (int neighbin : binvec)
    {
      // get local id of bin
      const int rowlidofbin = binrowmap_->lid(neighbin);

      // neighboring bin not owned by this processor
      if (rowlidofbin < 0) continue;

      // insert neighboring bins being ghosted on this processor
      halfneighboringbinstobins_[rowlidofbin].insert(gidofbin);
    }
  }

  // validate flag denoting valid relation of half surrounding neighboring bins to owned bins
  validhalfneighboringbins_ = true;
}

void Particle::ParticleEngine::check_particles_at_boundaries(
    std::vector<std::set<int>>& particlestoremove) const
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // get bounding box dimensions
  Core::LinAlg::Matrix<3, 2> boundingbox = binstrategy_->domain_bounding_box_corner_positions();

  // count particles that left the computational domain
  int numparticlesoutside = 0;

  // iterate over owned bins at the boundary
  for (int bdrybin : boundarybins_)
  {
    // get local id of bin
    const int collidofbin = bincolmap_->lid(bdrybin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particle of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      // get position of particle
      double* currpos = container->get_ptr_to_state(Position, ownedindex);

      // get global id of bin
      const int gidofbin = binstrategy_->convert_pos_to_gid(currpos);

      // particle left computational domain
      if (gidofbin == -1)
      {
        (particlestoremove[type]).insert(ownedindex);

        // get global id of particle
        const int* currglobalid = container->get_ptr_to_global_id(ownedindex);

#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (currglobalid[0] < 0) FOUR_C_THROW("no global id assigned to particle!");
#endif

        // insert freed global id
        particleuniqueglobalidhandler_->insert_freed_global_id(currglobalid[0]);

        ++numparticlesoutside;

        continue;
      }

      // no periodic boundary conditions
      if (not binstrategy_->have_periodic_boundary_conditions_applied()) continue;

      // check for periodic boundary in each spatial directions
      for (int dim = 0; dim < 3; ++dim)
      {
        if (binstrategy_->have_periodic_boundary_conditions_applied_in_spatial_direction(dim))
        {
          // binning domain length in current spatial direction
          double binningdomainlength =
              binstrategy_->length_of_binning_domain_in_a_spatial_direction(dim);

          // shift position by periodic length
          if (currpos[dim] < boundingbox(dim, 0))
            currpos[dim] += binningdomainlength;
          else if (currpos[dim] > boundingbox(dim, 1))
            currpos[dim] -= binningdomainlength;
        }
      }
    }
  }

  // short screen output
  if (numparticlesoutside)
    Core::IO::cout << "on processor " << myrank_ << " removed " << numparticlesoutside
                   << " particle(s) being outside the computational domain!" << Core::IO::endl;
}

void Particle::ParticleEngine::determine_particles_to_be_distributed(
    std::vector<ParticleObjShrdPtr>& particlestodistribute,
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend,
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestokeep)
{
  // clear particles being communicated to target processors
  communicatedparticletargets_.assign(
      Core::Communication::num_mpi_ranks(comm_), std::vector<int>(0));

  // number of particles to distribute
  int numparticles = particlestodistribute.size();

  // global ids of bins containing particles
  std::vector<int> bingidlist(numparticles);

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
  {
    // get states of particle
    const ParticleStates& states = particlestodistribute[i]->return_particle_states();

    // get position of particle
    const std::vector<double>& pos = states[Position];

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // get type of particles
    ParticleType type = particlestodistribute[i]->return_particle_type();

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    if (static_cast<int>(pos.size()) != container->get_state_dim(Position))
      FOUR_C_THROW("dimension of particle state '{}' not valid!", enum_to_state_name(Position));
#endif

    // get global id of bin
    bingidlist[i] = binstrategy_->convert_pos_to_gid(pos.data());
  }

  // get corresponding processor id
  std::vector<int> pidlist(numparticles);
  {
    // only unique id lists are accepted in RemoteIDList
    // 1) make gid list unique
    std::set<int> unique_bingidlist(bingidlist.begin(), bingidlist.end());
    std::vector<int> uniquevec_bingidlist(unique_bingidlist.begin(), unique_bingidlist.end());
    const int uniquesize = static_cast<int>(uniquevec_bingidlist.size());

    // 2) communication
    std::vector<int> unique_pidlist(uniquesize);
    int err = binrowmap_->remote_id_list(
        uniquesize, uniquevec_bingidlist.data(), unique_pidlist.data(), nullptr);
    if (err < 0) FOUR_C_THROW("RemoteIDList returned err={}", err);

    // 3) build full pid list via lookup table
    std::map<int, int> lookuptable;
    for (int s = 0; s < uniquesize; ++s)
      lookuptable.insert(
          lookuptable.end(), std::pair<int, int>(uniquevec_bingidlist[s], unique_pidlist[s]));
    for (int s = 0; s < numparticles; ++s) pidlist[s] = lookuptable[bingidlist[s]];
  }

  // count particles that are outside of the computational domain
  int numparticlesoutside = 0;

  // iterate over particles objects
  for (int i = 0; i < numparticles; ++i)
  {
    // get type of particle
    ParticleType type = particlestodistribute[i]->return_particle_type();

    // get owner of particle
    int ownerofparticle = pidlist[i];

    // particle outside of computational domain
    if (ownerofparticle == -1)
    {
      ++numparticlesoutside;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (particlestodistribute[i]->return_particle_global_id() < 0)
        FOUR_C_THROW("no global id assigned to particle!");
#endif

      // insert freed global id
      particleuniqueglobalidhandler_->insert_freed_global_id(
          particlestodistribute[i]->return_particle_global_id());
    }
    // particle is owned by this processor
    else if (myrank_ == ownerofparticle)
      particlestokeep[type].push_back(std::make_pair(ownerofparticle, particlestodistribute[i]));
    // particle is owned by another processor
    else
    {
      // append particle to be send
      particlestosend[ownerofparticle].push_back(particlestodistribute[i]);

      // append global id of particle to be send
      communicatedparticletargets_[ownerofparticle].emplace_back(
          particlestodistribute[i]->return_particle_global_id());
    }
  }

  // short screen output
  if (numparticlesoutside)
    Core::IO::cout << "on processor " << myrank_ << " removed " << numparticlesoutside
                   << " particle(s) being outside the computational domain!" << Core::IO::endl;

  // clear after all particles are prepared for distribution
  particlestodistribute.clear();
}

void Particle::ParticleEngine::determine_particles_to_be_transferred(
    std::vector<std::set<int>>& particlestoremove,
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend)
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // clear particles being communicated to target processors
  communicatedparticletargets_.assign(
      Core::Communication::num_mpi_ranks(comm_), std::vector<int>(0));

  // iterate over this processors bins being touched by other processors
  for (int touchedbin : touchedbins_)
  {
    // get local id of bin
    const int collidofbin = bincolmap_->lid(touchedbin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particle of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      // get position of particle
      const double* currpos = container->get_ptr_to_state(Position, ownedindex);

      // get global id of bin
      const int gidofbin = binstrategy_->convert_pos_to_gid(currpos);

      // particle left computational domain
      if (gidofbin == -1)
      {
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (not particlestoremove[type].contains(ownedindex))
          FOUR_C_THROW(
              "on processor {} a particle left the computational domain without being detected!",
              myrank_);
#endif
        continue;
      }

      // particle remains owned on this processor
      if (binrowmap_->lid(gidofbin) >= 0) continue;

      // get owning processor
      auto targetIt = firstlayerbinsownedby_.find(gidofbin);
      if (targetIt == firstlayerbinsownedby_.end())
        FOUR_C_THROW("particle not owned on this proc but target processor is unknown!");
      int sendtoproc = targetIt->second;

      int globalid(0);
      ParticleStates states;
      container->get_particle(ownedindex, globalid, states);

      // append particle to be send
      particlestosend[sendtoproc].emplace_back(
          std::make_shared<ParticleObject>(type, globalid, states, gidofbin));

      // store index of particle to be removed from containers after particle transfer
      (particlestoremove[type]).insert(ownedindex);

      // append global id of particle to be send
      communicatedparticletargets_[sendtoproc].emplace_back(globalid);
    }
  }
}

void Particle::ParticleEngine::determine_particles_to_be_ghosted(
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend) const
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // iterate over this processors bins being ghosted by other processors
  for (const auto& targetIt : thisbinsghostedby_)
  {
    // bin being ghosted on other processors
    const int ghostedbin = targetIt.first;

    // get local id of bin
    const int collidofbin = bincolmap_->lid(ghostedbin);

    // check if current bin contains owned particles
    if (particlestobins_[collidofbin].empty()) continue;

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[collidofbin])
    {
      // get type of particle
      ParticleType type = particleIt.first;

      // get local index of owned particle
      const int ownedindex = particleIt.second;

      // get container of owned particle of current particle type
      ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

      int globalid(0);
      ParticleStates states;
      container->get_particle(ownedindex, globalid, states);

      // iterate over target processors
      for (int sendtoproc : targetIt.second)
      {
        // append particle to be send
        particlestosend[sendtoproc].emplace_back(
            std::make_shared<ParticleObject>(type, globalid, states, ghostedbin, ownedindex));
      }
    }
  }
}

void Particle::ParticleEngine::determine_particles_to_be_refreshed(
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend) const
{
  // safety check
  if (not validdirectghosting_) FOUR_C_THROW("invalid direct ghosting!");

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type to be sent
    if (directghostingtargets_[type].empty()) continue;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate over owned particles of current type
    for (const auto& indexIt : directghostingtargets_[type])
    {
      int ownedindex = indexIt.first;

      int globalid(0);
      ParticleStates states;
      container->get_particle(ownedindex, globalid, states);

      // iterate over target processors
      for (const auto& targetIt : indexIt.second)
      {
        int sendtoproc = targetIt.first;
        int ghostedindex = targetIt.second;

        // append particle to be send
        particlestosend[sendtoproc].emplace_back(
            std::make_shared<ParticleObject>(type, -1, states, -1, ghostedindex));
      }
    }
  }
}

void Particle::ParticleEngine::
    determine_specific_states_of_particles_of_specific_types_to_be_refreshed(
        const StatesOfTypesToRefresh& particlestatestotypes,
        std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend) const
{
  // safety check
  if (not validdirectghosting_) FOUR_C_THROW("invalid direct ghosting!");

  // iterate over particle types
  for (const auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    ParticleType type = typeIt.first;

    // check for particles of current type to be sent
    if (directghostingtargets_[type].empty()) continue;

    // determine necessary size of vector for states
    int statesvectorsize = *std::max_element(typeIt.second.begin(), typeIt.second.end()) + 1;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate over owned particles of current type
    for (const auto& indexIt : directghostingtargets_[type])
    {
      int ownedindex = indexIt.first;

      // allocate memory to hold particle states
      ParticleStates states;
      states.assign(statesvectorsize, std::vector<double>(0));

      // iterate over states to be sent
      for (const auto& state : typeIt.second)
      {
        // get particle state dimension
        int statedim = container->get_state_dim(state);

        // get pointer to particle state
        const double* state_ptr = container->get_ptr_to_state(state, ownedindex);

        // fill particle state
        states[state].assign(state_ptr, state_ptr + statedim);
      }

      // iterate over target processors
      for (const auto& targetIt : indexIt.second)
      {
        int sendtoproc = targetIt.first;
        int ghostedindex = targetIt.second;

        // append particle to be send
        particlestosend[sendtoproc].emplace_back(
            std::make_shared<ParticleObject>(type, -1, states, -1, ghostedindex));
      }
    }
  }
}

void Particle::ParticleEngine::communicate_particles(
    std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend,
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoreceive) const
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // pack data for sending
  for (int torank = 0; torank < Core::Communication::num_mpi_ranks(comm_); ++torank)
  {
    if (particlestosend[torank].empty()) continue;

    for (const auto& iter : particlestosend[torank])
    {
      Core::Communication::PackBuffer data;
      iter->pack(data);
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
    }
  }

  // clear after all particles are packed
  particlestosend.clear();

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (const auto& p : rdata)
  {
    const int msgsource = p.first;
    const std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
      ParticleObjShrdPtr particleobject = std::dynamic_pointer_cast<ParticleObject>(object);
      if (particleobject == nullptr) FOUR_C_THROW("received object is not a particle object!");

      // store received particle
      particlestoreceive[particleobject->return_particle_type()].push_back(
          std::make_pair(msgsource, particleobject));
    }
  }
}

void Particle::ParticleEngine::communicate_direct_ghosting_map(
    std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>>& directghosting)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
    directghostingtargets_[type].clear();

  // invalidate flags denoting validity of direct ghosting
  validdirectghosting_ = false;

  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // pack data for sending
  for (const auto& p : directghosting)
  {
    Core::Communication::PackBuffer data;
    add_to_pack(data, p.second);
    std::swap(sdata[p.first], data());
  }

  // clear after all ghosting information is packed
  directghosting.clear();

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // init receiving map
  std::map<ParticleType, std::map<int, std::pair<int, int>>> receiveddirectghosting;

  // unpack and store received data
  for (const auto& p : rdata)
  {
    const std::vector<char>& rmsg = p.second;

    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      extract_from_pack(buffer, receiveddirectghosting);

      // iterate over particle types
      for (const auto& typeIt : receiveddirectghosting)
      {
        // get type of particles
        ParticleType type = typeIt.first;

        // iterate over this processors local indices of owned particles
        for (const auto& indexIt : typeIt.second)
        {
          // get index of owned particle
          int ownedindex = indexIt.first;

          (directghostingtargets_[type])[ownedindex].push_back(indexIt.second);
        }
      }
    }
  }

  // validate flags denoting validity of direct ghosting
  validdirectghosting_ = true;
}

void Particle::ParticleEngine::insert_owned_particles(
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type
    if (particlestoinsert[type].empty()) continue;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate over particle objects pairs
    for (const auto& objectpair : particlestoinsert[type])
    {
      // get particle object
      ParticleObjShrdPtr particleobject = objectpair.second;

      // get global id of particle
      int globalid = particleobject->return_particle_global_id();

      // get states of particle
      const ParticleStates& states = particleobject->return_particle_states();

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (globalid < 0) FOUR_C_THROW("no global id assigned to particle!");

      // get bin of particle
      int gidofbin = particleobject->return_bin_gid();

      // bin particle
      if (gidofbin < 0)
      {
        // get position of particle
        const std::vector<double>& pos = states[Position];

        // get type of particles
        ParticleType type = particleobject->return_particle_type();

        // get container of owned particles of current particle type
        ParticleContainer* container =
            particlecontainerbundle_->get_specific_container(type, Owned);

        if (static_cast<int>(pos.size()) != container->get_state_dim(Position))
          FOUR_C_THROW("dimension of particle state '{}' not valid!", enum_to_state_name(Position));

        // get global id of bin
        gidofbin = binstrategy_->convert_pos_to_gid(pos.data());
      }

      // particle not owned by this processor
      if (binrowmap_->lid(gidofbin) < 0) FOUR_C_THROW("particle received not owned on this proc!");
#endif

      // add particle to container of owned particles
      int index(0);
      container->add_particle(index, globalid, states);
    }
  }

  // clear after all particles are inserted
  particlestoinsert.clear();

  // invalidate particle safety flags
  invalidate_particle_safety_flags();
}

void Particle::ParticleEngine::insert_ghosted_particles(
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert,
    std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>>& directghosting)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type
    if (particlestoinsert[type].empty()) continue;

    // get container of ghosted particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Ghosted);

    // iterate over particle objects pairs
    for (const auto& objectpair : particlestoinsert[type])
    {
      // get owner of sending processor
      int sendingproc = objectpair.first;

      // get particle object
      ParticleObjShrdPtr particleobject = objectpair.second;

      // get global id of particle
      int globalid = particleobject->return_particle_global_id();

      // get states of particle
      const ParticleStates& states = particleobject->return_particle_states();

      // get bin of particle
      const int gidofbin = particleobject->return_bin_gid();
      if (gidofbin < 0)
        FOUR_C_THROW("received ghosted particle contains no information about its bin gid!");

      // add particle to container of ghosted particles
      int ghostedindex(0);
      container->add_particle(ghostedindex, globalid, states);

      // add index relating (owned and ghosted) particles to col bins
      particlestobins_[bincolmap_->lid(gidofbin)].push_back(std::make_pair(type, ghostedindex));

      // get local index of particle in container of owned particles of sending processor
      int ownedindex = particleobject->return_container_index();

      // insert necessary information being communicated to other processors for direct ghosting
      (((directghosting[sendingproc])[type])[ownedindex]) = std::make_pair(myrank_, ghostedindex);
    }
  }

  // clear after all particles are inserted
  particlestoinsert.clear();

  // validate flag denoting valid relation of ghosted particles to bins
  validghostedparticles_ = true;

  // invalidate safety flags dependent on ghosting
  validparticleneighbors_ = false;
  validglobalidtolocalindex_ = false;
  validdirectghosting_ = false;
}

void Particle::ParticleEngine::insert_refreshed_particles(
    std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert) const
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type
    if (particlestoinsert[type].empty()) continue;

    // get container of ghosted particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Ghosted);

    // iterate over particle objects pairs
    for (const auto& objectpair : particlestoinsert[type])
    {
      // get particle object
      ParticleObjShrdPtr particleobject = objectpair.second;

      // get states of particle
      const ParticleStates& states = particleobject->return_particle_states();

      // get local index of particle in container of ghosted particles on this processor
      int ghostedindex = particleobject->return_container_index();

      // replace particle in container of ghosted particles
      container->replace_particle(ghostedindex, -1, states);
    }
  }

  // clear after all particles are inserted
  particlestoinsert.clear();
}

void Particle::ParticleEngine::remove_particles_from_containers(
    std::vector<std::set<int>>& particlestoremove)
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // check for particles of current type
    if (particlestoremove[type].empty()) continue;

    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // iterate in reversed order over particles to be removed
    std::set<int>::reverse_iterator rit;
    for (rit = particlestoremove[type].rbegin(); rit != particlestoremove[type].rend(); ++rit)
      container->remove_particle(*rit);
  }

  // clear after all particles are removed
  particlestoremove.clear();

  // invalidate particle safety flags
  invalidate_particle_safety_flags();
}

void Particle::ParticleEngine::store_positions_after_particle_transfer()
{
  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored == 0) continue;

    // get pointer to particle states
    const double* pos = container->get_ptr_to_state(Position, 0);
    double* lasttransferpos = container->get_ptr_to_state(LastTransferPosition, 0);

    // get particle state dimension
    int statedim = container->get_state_dim(Position);

    // copy particle position data
    for (int i = 0; i < (statedim * particlestored); ++i) lasttransferpos[i] = pos[i];
  }
}

void Particle::ParticleEngine::relate_owned_particles_to_bins()
{
  // clear vector relating (owned and ghosted) particles to col bins
  particlestobins_.resize(bincolmap_->num_my_elements());
  for (auto& binIt : particlestobins_) binIt.clear();

  // invalidate particle safety flags
  invalidate_particle_safety_flags();

  // iterate over particle types
  for (const auto& type : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    ParticleContainer* container = particlecontainerbundle_->get_specific_container(type, Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get pointer to position of particle after last transfer
    const double* lasttransferpos = container->get_ptr_to_state(LastTransferPosition, 0);

    // get particle state dimension
    int statedim = container->get_state_dim(Position);

    // loop over particles in container
    for (int index = 0; index < particlestored; ++index)
    {
      // get global id of bin
      const int gidofbin = binstrategy_->convert_pos_to_gid(&(lasttransferpos[statedim * index]));

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (gidofbin < 0)
        FOUR_C_THROW("particle out of bounding box but not removed from container!");

      if (binrowmap_->lid(gidofbin) < 0)
        FOUR_C_THROW("particle not owned by this proc but not removed from container!");
#endif

      // add index relating (owned and ghosted) particles to col bins
      particlestobins_[bincolmap_->lid(gidofbin)].push_back(std::make_pair(type, index));
    }
  }

  // validate flag denoting valid relation of owned particles to bins
  validownedparticles_ = true;
}

void Particle::ParticleEngine::determine_min_relevant_bin_size()
{
  // get number of bins in all spatial directions
  const auto binperdir = binstrategy_->bin_per_dir();

  // get bin size
  const std::array<double, 3> binsize = binstrategy_->bin_size();

  // initialize minimum bin size to maximum bin size
  minbinsize_ = binstrategy_->get_max_bin_size();

  // check for minimum bin size in spatial directions with more than one bin layer
  for (int i = 0; i < 3; ++i)
    if (binperdir[i] > 1) minbinsize_ = std::min(minbinsize_, binsize[i]);
}

void Particle::ParticleEngine::determine_bin_weights()
{
  // safety check
  if (not validownedparticles_) FOUR_C_THROW("invalid relation of owned particles to bins!");

  // initialize weights of all bins
  binweights_->put_scalar(1.0e-05);

  // loop over row bins
  for (int rowlidofbin = 0; rowlidofbin < binrowmap_->num_my_elements(); ++rowlidofbin)
  {
    // get global id of bin
    const int gidofbin = binrowmap_->gid(rowlidofbin);

    // iterate over owned particles in current bin
    for (const auto& particleIt : particlestobins_[bincolmap_->lid(gidofbin)])
    {
      // add weight of particle of specific type
      binweights_->get_vector(0).get_values()[rowlidofbin] += typeweights_[particleIt.first];
    }
  }
}

void Particle::ParticleEngine::invalidate_particle_safety_flags()
{
  validownedparticles_ = false;
  validghostedparticles_ = false;
  validparticleneighbors_ = false;
  validglobalidtolocalindex_ = false;
  validdirectghosting_ = false;
}

FOUR_C_NAMESPACE_CLOSE
