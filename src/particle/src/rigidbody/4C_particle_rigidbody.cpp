// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_rigidbody.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_engine_unique_global_id.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_rigidbody_affiliation_pairs.hpp"
#include "4C_particle_rigidbody_datastate.hpp"
#include "4C_particle_rigidbody_initial_field.hpp"
#include "4C_particle_rigidbody_runtime_vtp_writer.hpp"
#include "4C_particle_rigidbody_utils.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::RigidBodyHandler::RigidBodyHandler(MPI_Comm comm, const Teuchos::ParameterList& params)
    : comm_(comm), myrank_(Core::Communication::my_mpi_rank(comm)), params_(params)
{
  init_rigid_body_unique_global_id_handler();

  init_rigid_body_data_state();

  init_rigid_body_vtp_writer();

  init_affiliation_pair_handler();
}

Particle::RigidBodyHandler::~RigidBodyHandler() = default;

void Particle::RigidBodyHandler::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // setup affiliation pair handler
  affiliationpairs_->setup(particleengineinterface);

  // safety check
  {
    // get particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
        particleengineinterface_->get_particle_container_bundle();

    if (not particlecontainerbundle->get_particle_types().contains(Particle::RigidPhase))
      FOUR_C_THROW("no particle container for particle type '{}' found!",
          Particle::enum_to_type_name(Particle::RigidPhase));
  }

  // short screen output
  if (particleengineinterface_->have_periodic_boundary_conditions() and myrank_ == 0)
    Core::IO::cout << "Warning: rigid bodies not transferred over periodic boundary!"
                   << Core::IO::endl;
}

void Particle::RigidBodyHandler::write_restart() const
{
  // get bin discretization writer
  std::shared_ptr<Core::IO::DiscretizationWriter> binwriter =
      particleengineinterface_->get_bin_discretization_writer();

  // write restart of unique global identifier handler
  rigidbodyuniqueglobalidhandler_->write_restart(binwriter);

  // write restart of affiliation pair handler
  affiliationpairs_->write_restart();

  // get packed rigid body state data
  std::shared_ptr<std::vector<char>> buffer = std::make_shared<std::vector<char>>();
  get_packed_rigid_body_states(*buffer);

  // write rigid body state data
  binwriter->write_char_data("RigidBodyStateData", *buffer);
}

void Particle::RigidBodyHandler::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader)
{
  // read restart of unique global identifier handler
  rigidbodyuniqueglobalidhandler_->read_restart(reader);

  // read restart of runtime vtp writer
  rigidbodyvtpwriter_->read_restart(reader);

  // read restart of affiliation pair handler
  affiliationpairs_->read_restart(reader);

  // allocate rigid body states
  allocate_rigid_body_states();

  // read rigid body state data
  std::shared_ptr<std::vector<char>> buffer = std::make_shared<std::vector<char>>();
  reader->read_char_vector(buffer, "RigidBodyStateData");

  // extract packed rigid body state data
  extract_packed_rigid_body_states(*buffer);
}

void Particle::RigidBodyHandler::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    Particle::TypeEnum type = typeIt.first;

    // set of particle states for current particle type
    std::set<Particle::StateEnum>& particlestates = typeIt.second;

    if (type == Particle::RigidPhase)
    {
      // insert states of rigid particles
      particlestates.insert({Particle::RigidBodyColor, Particle::RelativePositionBodyFrame,
          Particle::RelativePosition, Particle::Inertia, Particle::Force});
    }
  }
}

void Particle::RigidBodyHandler::write_rigid_body_runtime_output(
    const int step, const double time) const
{
  rigidbodyvtpwriter_->set_rigid_body_positions_and_states(ownedrigidbodies_);
  rigidbodyvtpwriter_->write_to_disk(time, step);
}

void Particle::RigidBodyHandler::set_initial_affiliation_pair_data()
{
  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    // get pointer to particle states
    const double* rigidbodycolor_i =
        container_i->get_ptr_to_state(Particle::RigidBodyColor, particle_i);

    // get global id of affiliated rigid body k
    const int rigidbody_k = std::round(rigidbodycolor_i[0]);

    // insert affiliation pair
    affiliationpairdata.insert(std::make_pair(globalid_i[0], rigidbody_k));
  }

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif
}

void Particle::RigidBodyHandler::set_unique_global_ids_for_all_rigid_bodies()
{
  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // maximum global id of rigid bodies on this processor
  int maxglobalid = -1;

  // get maximum global id of rigid bodies on this processor
  for (const auto& it : affiliationpairdata) maxglobalid = std::max(maxglobalid, it.second);

  // get maximum global id of rigid bodies on all processors
  int allprocmaxglobalid = -1;
  allprocmaxglobalid = Core::Communication::max_all(maxglobalid, comm_);

  // number of global ids on all processors
  const int numglobalids = allprocmaxglobalid + 1;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not(rigidbodyuniqueglobalidhandler_->get_max_global_id() < 0))
    FOUR_C_THROW(
        "maximum global id of rigid body unique global identifier handler already touched!");
#endif

  // request number of global ids of all rigid bodies on processor 0
  std::vector<int> requesteduniqueglobalids;
  if (myrank_ == 0) requesteduniqueglobalids.reserve(numglobalids);

  // draw requested number of global ids
  rigidbodyuniqueglobalidhandler_->draw_requested_number_of_global_ids(requesteduniqueglobalids);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (myrank_ == 0)
    for (int i = 0; i < numglobalids; ++i)
      if (requesteduniqueglobalids[i] != i)
        FOUR_C_THROW("drawn requested global ids not consecutive!");
#endif

  // used global ids on all processors
  std::vector<int> usedglobalids(numglobalids, 0);

  // get used global ids on this processor
  for (const auto& it : affiliationpairdata) usedglobalids[it.second] = 1;


  // get used global ids on all processors
  MPI_Allreduce(MPI_IN_PLACE, usedglobalids.data(), numglobalids, MPI_INT, MPI_MAX, comm_);

  // free unused global ids on processor 0
  if (myrank_ == 0)
    for (int i = 0; i < numglobalids; ++i)
      if (usedglobalids[i] == 0)
        rigidbodyuniqueglobalidhandler_->insert_freed_global_id(requesteduniqueglobalids[i]);
}

void Particle::RigidBodyHandler::allocate_rigid_body_states()
{
  // number of global ids
  const int numglobalids = rigidbodyuniqueglobalidhandler_->get_max_global_id() + 1;

  // allocate stored states
  rigidbodydatastate_->allocate_stored_states(numglobalids);
}

void Particle::RigidBodyHandler::initialize_rigid_body_mass_quantities_and_orientation()
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::RigidBodyHandler::initialize_rigid_body_mass_quantities_and_orientation");

  // compute mass quantities of rigid bodies
  compute_rigid_body_mass_quantities();

  // clear orientation of rigid bodies
  clear_rigid_body_orientation();

  // broadcast positions of rigid bodies
  broadcast_rigid_body_positions();

  // set relative position of rigid particles in body frame
  set_rigid_particle_relative_position_in_body_frame();

  // update relative position of rigid particles
  update_rigid_particle_relative_position();
}

void Particle::RigidBodyHandler::distribute_rigid_body()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::DistributeRigidBody");

  // distribute affiliation pairs
  affiliationpairs_->distribute_affiliation_pairs();

  // update rigid body ownership
  update_rigid_body_ownership();
}

void Particle::RigidBodyHandler::communicate_rigid_body()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::communicate_rigid_body");

  // communicate affiliation pairs
  affiliationpairs_->communicate_affiliation_pairs();

  // update rigid body ownership
  update_rigid_body_ownership();
}

void Particle::RigidBodyHandler::clear_forces_and_torques()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::clear_forces_and_torques");

  // clear force and torque acting on rigid bodies
  clear_rigid_body_force_and_torque();

  // clear force acting on rigid particles
  clear_rigid_particle_force();
}

void Particle::RigidBodyHandler::add_gravity_acceleration(std::vector<double>& gravity)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k].data();

    // set gravity acceleration
    ParticleUtils::vec_add(acc_k, gravity.data());
  }
}

void Particle::RigidBodyHandler::compute_accelerations()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::compute_accelerations");

  // compute partial force and torque acting on rigid bodies
  compute_partial_force_and_torque();

  // gather partial and compute full force and torque acting on rigid bodies
  gather_partial_and_compute_full_force_and_torque();

  // compute accelerations of rigid bodies from force and torque
  compute_accelerations_from_force_and_torque();

  // broadcast accelerations of rigid bodies
  broadcast_rigid_body_accelerations();

  // set accelerations of rigid particles
  set_rigid_particle_accelerations();
}

void Particle::RigidBodyHandler::update_positions(const double timeincrement)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::UpdatePositions");

  // update positions of rigid bodies with given time increment
  update_rigid_body_positions(timeincrement);

  // broadcast positions of rigid bodies
  broadcast_rigid_body_positions();

  // update relative position of rigid particles
  update_rigid_particle_relative_position();

  // set position of rigid particles
  set_rigid_particle_position();
}

void Particle::RigidBodyHandler::update_velocities(const double timeincrement)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::UpdateVelocities");

  // update velocities of rigid bodies with given time increment
  update_rigid_body_velocities(timeincrement);

  // broadcast velocities of rigid bodies
  broadcast_rigid_body_velocities();

  // set velocities of rigid particles
  set_rigid_particle_velocities();
}

void Particle::RigidBodyHandler::clear_accelerations()
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::ClearAccelerations");

  // clear accelerations of rigid bodies
  clear_rigid_body_accelerations();
}

bool Particle::RigidBodyHandler::have_rigid_body_phase_change(
    const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::have_rigid_body_phase_change");

  int localhavephasechange = 0;

  // iterate over particle phase change tuples
  for (const auto& particletypetotype : particlesfromphasetophase)
  {
    Particle::TypeEnum type_source;
    Particle::TypeEnum type_target;
    std::tie(type_source, type_target, std::ignore) = particletypetotype;

    if (type_source == Particle::RigidPhase or type_target == Particle::RigidPhase)
    {
      localhavephasechange = 1;
      break;
    }
  }

  // check among all processors
  int globalhavephasechange = 0;
  globalhavephasechange = Core::Communication::max_all(localhavephasechange, comm_);

  return globalhavephasechange;
}

void Particle::RigidBodyHandler::evaluate_rigid_body_phase_change(
    const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase)
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::RigidBodyHandler::evaluate_rigid_body_phase_change");

  // evaluate melting of rigid bodies
  evaluate_rigid_body_melting(particlesfromphasetophase);

  // evaluate solidification of rigid bodies
  evaluate_rigid_body_solidification(particlesfromphasetophase);

  // update rigid body ownership
  update_rigid_body_ownership();

  // store previous position of rigid bodies
  const std::vector<std::vector<double>> previousposition = rigidbodydatastate_->get_ref_position();

  // initialize rigid body mass quantities and orientation
  initialize_rigid_body_mass_quantities_and_orientation();

  // set velocities of rigid bodies after phase change
  set_rigid_body_velocities_after_phase_change(previousposition);

  // broadcast velocities of rigid bodies
  broadcast_rigid_body_velocities();

  // set velocities of rigid particles
  set_rigid_particle_velocities();
}

void Particle::RigidBodyHandler::init_rigid_body_unique_global_id_handler()
{
  // create and init unique global identifier handler
  rigidbodyuniqueglobalidhandler_ = std::unique_ptr<Particle::UniqueGlobalIdHandler>(
      new Particle::UniqueGlobalIdHandler(comm_, "rigidbody"));
}

void Particle::RigidBodyHandler::init_rigid_body_data_state()
{
  // create rigid body data state container
  rigidbodydatastate_ = std::make_shared<Particle::RigidBodyDataState>();
}

void Particle::RigidBodyHandler::init_rigid_body_vtp_writer()
{
  // construct and init rigid body runtime vtp writer
  rigidbodyvtpwriter_ = std::unique_ptr<Particle::RigidBodyRuntimeVtpWriter>(
      new Particle::RigidBodyRuntimeVtpWriter(comm_));
  rigidbodyvtpwriter_->init(rigidbodydatastate_);
}

void Particle::RigidBodyHandler::init_affiliation_pair_handler()
{
  // create affiliation pair handler
  affiliationpairs_ = std::unique_ptr<Particle::RigidBodyAffiliationPairs>(
      new Particle::RigidBodyAffiliationPairs(comm_));
}

void Particle::RigidBodyHandler::get_packed_rigid_body_states(std::vector<char>& buffer) const
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to rigid body states
    const double& mass_k = rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    const std::vector<double>& inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k];
    const std::vector<double>& pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k];
    const std::vector<double>& rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k];
    const std::vector<double>& vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k];
    const std::vector<double>& angvel_k =
        rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k];
    const std::vector<double>& acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k];
    const std::vector<double>& angacc_k =
        rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k];

    // pack data for sending
    Core::Communication::PackBuffer data;

    data.add_to_pack(rigidbody_k);
    data.add_to_pack(mass_k);
    for (int i = 0; i < 6; ++i) data.add_to_pack(inertia_k[i]);
    for (int i = 0; i < 3; ++i) data.add_to_pack(pos_k[i]);
    for (int i = 0; i < 4; ++i) data.add_to_pack(rot_k[i]);
    for (int i = 0; i < 3; ++i) data.add_to_pack(vel_k[i]);
    for (int i = 0; i < 3; ++i) data.add_to_pack(angvel_k[i]);
    for (int i = 0; i < 3; ++i) data.add_to_pack(acc_k[i]);
    for (int i = 0; i < 3; ++i) data.add_to_pack(angacc_k[i]);

    buffer.insert(buffer.end(), data().begin(), data().end());
  }
}

void Particle::RigidBodyHandler::extract_packed_rigid_body_states(std::vector<char>& buffer)
{
  Core::Communication::UnpackBuffer data(buffer);
  while (!data.at_end())
  {
    int rigidbody_k;
    extract_from_pack(data, rigidbody_k);

    // get global ids of rigid bodies owned by this processor
    ownedrigidbodies_.push_back(rigidbody_k);

    // get reference to rigid body states
    double& mass_k = rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    std::vector<double>& inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k];
    std::vector<double>& pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k];
    std::vector<double>& rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k];
    std::vector<double>& vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k];
    std::vector<double>& angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k];
    std::vector<double>& acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k];
    std::vector<double>& angacc_k =
        rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k];

    extract_from_pack(data, mass_k);
    for (int i = 0; i < 6; ++i) extract_from_pack(data, inertia_k[i]);
    for (int i = 0; i < 3; ++i) extract_from_pack(data, pos_k[i]);
    for (int i = 0; i < 4; ++i) extract_from_pack(data, rot_k[i]);
    for (int i = 0; i < 3; ++i) extract_from_pack(data, vel_k[i]);
    for (int i = 0; i < 3; ++i) extract_from_pack(data, angvel_k[i]);
    for (int i = 0; i < 3; ++i) extract_from_pack(data, acc_k[i]);
    for (int i = 0; i < 3; ++i) extract_from_pack(data, angacc_k[i]);
  }
}

void Particle::RigidBodyHandler::update_rigid_body_ownership()
{
  // store rigid bodies previously owned by this processor
  std::vector<int> previouslyownedrigidbodies = ownedrigidbodies_;

  // determine owned and hosted rigid bodies
  determine_owned_and_hosted_rigid_bodies();

  // relate owned rigid bodies to all hosting processors
  relate_owned_rigid_bodies_to_hosting_procs();

  // communicate rigid body states
  communicate_rigid_body_states(previouslyownedrigidbodies);
}

void Particle::RigidBodyHandler::determine_owned_and_hosted_rigid_bodies()
{
  ownedrigidbodies_.clear();
  hostedrigidbodies_.clear();
  ownerofrigidbodies_.clear();

  // number of global ids
  const int numglobalids = rigidbodyuniqueglobalidhandler_->get_max_global_id() + 1;

  // maximum number of particles per rigid body over all processors
  std::vector<std::pair<int, int>> maxnumberofparticlesperrigidbodyonproc(
      numglobalids, std::make_pair(0, myrank_));

  // get number of particle per rigid body on this processor
  for (const auto& it : affiliationpairs_->get_ref_to_affiliation_pair_data())
    maxnumberofparticlesperrigidbodyonproc[it.second].first++;

  // get global ids of rigid bodies hosted (owned and non-owned) by this processor
  for (int rigidbody_k = 0; rigidbody_k < numglobalids; ++rigidbody_k)
    if (maxnumberofparticlesperrigidbodyonproc[rigidbody_k].first > 0)
      hostedrigidbodies_.push_back(rigidbody_k);


  // get maximum number of particles per rigid body over all processors
  MPI_Allreduce(MPI_IN_PLACE, maxnumberofparticlesperrigidbodyonproc.data(), numglobalids, MPI_2INT,
      MPI_MAXLOC, comm_);

  // get owner of all rigid bodies
  ownerofrigidbodies_.reserve(numglobalids);
  for (const auto& it : maxnumberofparticlesperrigidbodyonproc)
    ownerofrigidbodies_.push_back(it.second);

  // get global ids of rigid bodies owned by this processor
  for (const int rigidbody_k : hostedrigidbodies_)
    if (ownerofrigidbodies_[rigidbody_k] == myrank_) ownedrigidbodies_.push_back(rigidbody_k);
}

void Particle::RigidBodyHandler::relate_owned_rigid_bodies_to_hosting_procs()
{
  // number of global ids
  const int numglobalids = rigidbodyuniqueglobalidhandler_->get_max_global_id() + 1;

  // allocate memory
  ownedrigidbodiestohostingprocs_.assign(numglobalids, std::vector<int>(0));

  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // communicate global id of rigid body to owning processor
    if (owner_k != myrank_)
    {
      // pack data for sending
      Core::Communication::PackBuffer data;

      data.add_to_pack(rigidbody_k);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    int msgsource = p.first;
    std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      int rigidbody_k;
      extract_from_pack(buffer, rigidbody_k);

      // insert processor id the gathered global id of rigid body is received from
      ownedrigidbodiestohostingprocs_[rigidbody_k].push_back(msgsource);
    }
  }
}

void Particle::RigidBodyHandler::communicate_rigid_body_states(
    std::vector<int>& previouslyownedrigidbodies)
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over previously owned rigid bodies
  for (const int rigidbody_k : previouslyownedrigidbodies)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // get reference to rigid body states
    const double& mass_k = rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    const std::vector<double>& inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k];
    const std::vector<double>& pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k];
    const std::vector<double>& rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k];
    const std::vector<double>& vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k];
    const std::vector<double>& angvel_k =
        rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k];
    const std::vector<double>& acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k];
    const std::vector<double>& angacc_k =
        rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k];

    // communicate states to owning processor
    if (owner_k != myrank_)
    {
      // pack data for sending
      Core::Communication::PackBuffer data;

      data.add_to_pack(rigidbody_k);
      data.add_to_pack(mass_k);
      for (int i = 0; i < 6; ++i) data.add_to_pack(inertia_k[i]);
      for (int i = 0; i < 3; ++i) data.add_to_pack(pos_k[i]);
      for (int i = 0; i < 4; ++i) data.add_to_pack(rot_k[i]);
      for (int i = 0; i < 3; ++i) data.add_to_pack(vel_k[i]);
      for (int i = 0; i < 3; ++i) data.add_to_pack(angvel_k[i]);
      for (int i = 0; i < 3; ++i) data.add_to_pack(acc_k[i]);
      for (int i = 0; i < 3; ++i) data.add_to_pack(angacc_k[i]);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      int rigidbody_k;
      extract_from_pack(buffer, rigidbody_k);

      // get reference to rigid body states
      double& mass_k = rigidbodydatastate_->get_ref_mass()[rigidbody_k];
      std::vector<double>& inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k];
      std::vector<double>& pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k];
      std::vector<double>& rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k];
      std::vector<double>& vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k];
      std::vector<double>& angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k];
      std::vector<double>& acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k];
      std::vector<double>& angacc_k =
          rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k];

      extract_from_pack(buffer, mass_k);
      for (int i = 0; i < 6; ++i) extract_from_pack(buffer, inertia_k[i]);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, pos_k[i]);
      for (int i = 0; i < 4; ++i) extract_from_pack(buffer, rot_k[i]);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, vel_k[i]);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, angvel_k[i]);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, acc_k[i]);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, angacc_k[i]);
    }
  }
}

void Particle::RigidBodyHandler::compute_rigid_body_mass_quantities()
{
  // clear partial mass quantities of rigid bodies
  clear_partial_mass_quantities();

  // compute partial mass quantities of rigid bodies
  compute_partial_mass_quantities();

  // gathered partial mass quantities of rigid bodies from all corresponding processors
  std::unordered_map<int, std::vector<double>> gatheredpartialmass;
  std::unordered_map<int, std::vector<std::vector<double>>> gatheredpartialinertia;
  std::unordered_map<int, std::vector<std::vector<double>>> gatheredpartialposition;

  // gather partial mass quantities of rigid bodies
  gather_partial_mass_quantities(
      gatheredpartialmass, gatheredpartialinertia, gatheredpartialposition);

  // compute full mass quantities of rigid bodies
  compute_full_mass_quantities(
      gatheredpartialmass, gatheredpartialinertia, gatheredpartialposition);
}

void Particle::RigidBodyHandler::clear_partial_mass_quantities()
{
  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // get pointer to rigid body states
    double* mass_k = &rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    double* inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k].data();
    double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();

    // clear mass quantities
    mass_k[0] = 0.0;
    for (int i = 0; i < 6; ++i) inertia_k[i] = 0.0;
    ParticleUtils::vec_clear(pos_k);
  }
}

void Particle::RigidBodyHandler::compute_partial_mass_quantities()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    double* mass_k = &rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);

    // sum contribution of particle i
    mass_k[0] += mass_i[0];
    ParticleUtils::vec_add_scale(pos_k, mass_i[0], pos_i);
  }

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // get pointer to rigid body states
    const double* mass_k = &rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (not(mass_k[0] > 0.0)) FOUR_C_THROW("partial mass of rigid body {} is zero!", rigidbody_k);
#endif

    // determine center of gravity of (partial) rigid body k
    ParticleUtils::vec_scale(pos_k, 1.0 / mass_k[0]);
  }

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();
    double* inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k].data();

    // get pointer to particle states
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* inertia_i = container_i->get_ptr_to_state(Particle::Inertia, particle_i);

    double r_ki[3];
    ParticleUtils::vec_set(r_ki, pos_k);
    ParticleUtils::vec_sub(r_ki, pos_i);

    // sum contribution of particle i
    inertia_k[0] += inertia_i[0] + (r_ki[1] * r_ki[1] + r_ki[2] * r_ki[2]) * mass_i[0];
    inertia_k[1] += inertia_i[0] + (r_ki[0] * r_ki[0] + r_ki[2] * r_ki[2]) * mass_i[0];
    inertia_k[2] += inertia_i[0] + (r_ki[0] * r_ki[0] + r_ki[1] * r_ki[1]) * mass_i[0];
    inertia_k[3] -= r_ki[0] * r_ki[1] * mass_i[0];
    inertia_k[4] -= r_ki[0] * r_ki[2] * mass_i[0];
    inertia_k[5] -= r_ki[1] * r_ki[2] * mass_i[0];
  }
}

void Particle::RigidBodyHandler::gather_partial_mass_quantities(
    std::unordered_map<int, std::vector<double>>& gatheredpartialmass,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialinertia,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialposition)
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // get reference to rigid body states
    const double& mass_k = rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    const std::vector<double>& inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k];
    const std::vector<double>& pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k];

    // rigid body k owned by this processor
    if (owner_k == myrank_)
    {
      // append to gathered partial mass quantities
      gatheredpartialmass[rigidbody_k].push_back(mass_k);
      gatheredpartialinertia[rigidbody_k].push_back(inertia_k);
      gatheredpartialposition[rigidbody_k].push_back(pos_k);
    }
    // communicate partial mass quantities to owning processor
    else
    {
      // pack data for sending
      Core::Communication::PackBuffer data;

      data.add_to_pack(rigidbody_k);
      data.add_to_pack(mass_k);
      for (int i = 0; i < 6; ++i) data.add_to_pack(inertia_k[i]);
      for (int i = 0; i < 3; ++i) data.add_to_pack(pos_k[i]);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      int rigidbody_k;
      extract_from_pack(buffer, rigidbody_k);
      double mass_k;
      extract_from_pack(buffer, mass_k);

      std::vector<double> inertia_k(6);
      for (int i = 0; i < 6; ++i) extract_from_pack(buffer, inertia_k[i]);

      std::vector<double> pos_k(3);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, pos_k[i]);

      // append to gathered partial mass quantities
      gatheredpartialmass[rigidbody_k].push_back(mass_k);
      gatheredpartialinertia[rigidbody_k].push_back(inertia_k);
      gatheredpartialposition[rigidbody_k].push_back(pos_k);
    }
  }
}

void Particle::RigidBodyHandler::compute_full_mass_quantities(
    std::unordered_map<int, std::vector<double>>& gatheredpartialmass,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialinertia,
    std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialposition)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    std::vector<double>& partialmass_k = gatheredpartialmass[rigidbody_k];
    std::vector<std::vector<double>>& partialpos_k = gatheredpartialposition[rigidbody_k];

    // number of partial mass quantities of rigid body k including this processor
    const int numpartial_k = ownedrigidbodiestohostingprocs_[rigidbody_k].size() + 1;

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (static_cast<int>(partialmass_k.size()) != numpartial_k or
        static_cast<int>(partialpos_k.size()) != numpartial_k)
      FOUR_C_THROW(
          "the number of partial mass quantities of rigid body {} do not match!", rigidbody_k);
#endif

    // get pointer to rigid body states
    double* mass_k = &rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();

    // clear mass and position
    mass_k[0] = 0.0;
    ParticleUtils::vec_clear(pos_k);

    // iterate over partial quantities
    for (int p = 0; p < numpartial_k; ++p)
    {
      // sum contribution of partial quantity
      mass_k[0] += partialmass_k[p];
      ParticleUtils::vec_add_scale(pos_k, partialmass_k[p], partialpos_k[p].data());
    }

    // determine center of gravity of rigid body k
    ParticleUtils::vec_scale(pos_k, 1.0 / mass_k[0]);
  }

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    std::vector<double>& partialmass_k = gatheredpartialmass[rigidbody_k];
    std::vector<std::vector<double>>& partialpos_k = gatheredpartialposition[rigidbody_k];
    std::vector<std::vector<double>>& partialinertia_k = gatheredpartialinertia[rigidbody_k];

    // number of partial mass quantities of rigid body k including this processor
    const int numpartial_k = ownedrigidbodiestohostingprocs_[rigidbody_k].size() + 1;

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (static_cast<int>(partialmass_k.size()) != numpartial_k or
        static_cast<int>(partialinertia_k.size()) != numpartial_k)
      FOUR_C_THROW(
          "the number of partial mass quantities of rigid body {} do not match!", rigidbody_k);
#endif

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();
    double* inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k].data();

    // clear inertia
    for (int i = 0; i < 6; ++i) inertia_k[i] = 0.0;

    // iterate over partial quantities
    for (int p = 0; p < numpartial_k; ++p)
    {
      double r_kp[3];
      ParticleUtils::vec_set(r_kp, pos_k);
      ParticleUtils::vec_sub(r_kp, partialpos_k[p].data());

      // sum contribution of partial quantity
      for (int i = 0; i < 6; ++i) inertia_k[i] += partialinertia_k[p][i];

      inertia_k[0] += (r_kp[1] * r_kp[1] + r_kp[2] * r_kp[2]) * partialmass_k[p];
      inertia_k[1] += (r_kp[0] * r_kp[0] + r_kp[2] * r_kp[2]) * partialmass_k[p];
      inertia_k[2] += (r_kp[0] * r_kp[0] + r_kp[1] * r_kp[1]) * partialmass_k[p];
      inertia_k[3] -= r_kp[0] * r_kp[1] * partialmass_k[p];
      inertia_k[4] -= r_kp[0] * r_kp[2] * partialmass_k[p];
      inertia_k[5] -= r_kp[1] * r_kp[2] * partialmass_k[p];
    }
  }
}

void Particle::RigidBodyHandler::clear_rigid_body_force_and_torque()
{
  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // get pointer to rigid body states
    double* force_k = rigidbodydatastate_->get_ref_force()[rigidbody_k].data();
    double* torque_k = rigidbodydatastate_->get_ref_torque()[rigidbody_k].data();

    // clear force and torque of rigid body k
    ParticleUtils::vec_clear(force_k);
    ParticleUtils::vec_clear(torque_k);
  }
}

void Particle::RigidBodyHandler::clear_rigid_particle_force()
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

  // clear force of all particles
  container_i->clear_state(Particle::Force);
}

void Particle::RigidBodyHandler::compute_partial_force_and_torque()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    double* force_k = rigidbodydatastate_->get_ref_force()[rigidbody_k].data();
    double* torque_k = rigidbodydatastate_->get_ref_torque()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i = container_i->get_ptr_to_state(Particle::RelativePosition, particle_i);
    const double* force_i = container_i->get_ptr_to_state(Particle::Force, particle_i);

    // sum contribution of particle i
    ParticleUtils::vec_add(force_k, force_i);
    ParticleUtils::vec_add_cross(torque_k, relpos_i, force_i);
  }
}

void Particle::RigidBodyHandler::gather_partial_and_compute_full_force_and_torque()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over hosted rigid bodies
  for (const int rigidbody_k : hostedrigidbodies_)
  {
    // owner of rigid body k
    const int owner_k = ownerofrigidbodies_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& force_k = rigidbodydatastate_->get_ref_force()[rigidbody_k];
    const std::vector<double>& torque_k = rigidbodydatastate_->get_ref_torque()[rigidbody_k];

    // communicate partial force and torque to owning processor
    if (owner_k != myrank_)
    {
      // pack data for sending
      Core::Communication::PackBuffer data;

      data.add_to_pack(rigidbody_k);
      for (int i = 0; i < 3; ++i) data.add_to_pack(force_k[i]);
      for (int i = 0; i < 3; ++i) data.add_to_pack(torque_k[i]);

      sdata[owner_k].insert(sdata[owner_k].end(), data().begin(), data().end());
    }
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      int rigidbody_k;
      extract_from_pack(buffer, rigidbody_k);

      std::vector<double> tmp_force_k(3);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, tmp_force_k[i]);

      std::vector<double> tmp_torque_k(3);
      for (int i = 0; i < 3; ++i) extract_from_pack(buffer, tmp_torque_k[i]);

      // get pointer to rigid body states
      double* force_k = rigidbodydatastate_->get_ref_force()[rigidbody_k].data();
      double* torque_k = rigidbodydatastate_->get_ref_torque()[rigidbody_k].data();

      // sum gathered contribution to full force and torque
      ParticleUtils::vec_add(force_k, tmp_force_k.data());
      ParticleUtils::vec_add(torque_k, tmp_torque_k.data());
    }
  }
}

void Particle::RigidBodyHandler::compute_accelerations_from_force_and_torque()
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    const double* mass_k = &rigidbodydatastate_->get_ref_mass()[rigidbody_k];
    const double* inertia_k = rigidbodydatastate_->get_ref_inertia()[rigidbody_k].data();
    const double* rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k].data();
    const double* force_k = rigidbodydatastate_->get_ref_force()[rigidbody_k].data();
    const double* torque_k = rigidbodydatastate_->get_ref_torque()[rigidbody_k].data();
    double* acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k].data();
    double* angacc_k = rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k].data();

    // compute acceleration of rigid body k
    ParticleUtils::vec_add_scale(acc_k, 1 / mass_k[0], force_k);

    // compute inverse of rotation
    double invrot_k[4];
    ParticleUtils::quaternion_invert(invrot_k, rot_k);

    // get torque in the reference frame
    double reftorque_k[3];
    ParticleUtils::quaternion_rotate_vector(reftorque_k, invrot_k, torque_k);

    // determinant of mass moment of inertia
    const double det_inertia_k =
        inertia_k[0] * inertia_k[1] * inertia_k[2] + inertia_k[3] * inertia_k[4] * inertia_k[5] +
        inertia_k[3] * inertia_k[4] * inertia_k[5] - inertia_k[1] * inertia_k[4] * inertia_k[4] -
        inertia_k[2] * inertia_k[3] * inertia_k[3] - inertia_k[0] * inertia_k[5] * inertia_k[5];

    // no mass moment of inertia
    if (std::abs(det_inertia_k) < 1E-14) continue;

    // evaluate angular acceleration of rigid body k in the reference frame
    double refangacc_k[3];
    refangacc_k[0] = reftorque_k[0] * (inertia_k[1] * inertia_k[2] - inertia_k[5] * inertia_k[5]) +
                     reftorque_k[1] * (inertia_k[4] * inertia_k[5] - inertia_k[2] * inertia_k[3]) +
                     reftorque_k[2] * (inertia_k[3] * inertia_k[5] - inertia_k[1] * inertia_k[4]);

    refangacc_k[1] = reftorque_k[0] * (inertia_k[4] * inertia_k[5] - inertia_k[2] * inertia_k[3]) +
                     reftorque_k[1] * (inertia_k[0] * inertia_k[2] - inertia_k[4] * inertia_k[4]) +
                     reftorque_k[2] * (inertia_k[3] * inertia_k[4] - inertia_k[0] * inertia_k[5]);

    refangacc_k[2] = reftorque_k[0] * (inertia_k[3] * inertia_k[5] - inertia_k[1] * inertia_k[4]) +
                     reftorque_k[1] * (inertia_k[3] * inertia_k[4] - inertia_k[0] * inertia_k[5]) +
                     reftorque_k[2] * (inertia_k[0] * inertia_k[1] - inertia_k[3] * inertia_k[3]);

    ParticleUtils::vec_scale(refangacc_k, 1.0 / det_inertia_k);

    // compute angular acceleration of rigid body k in the rotating frame
    double temp[3];
    ParticleUtils::quaternion_rotate_vector(temp, rot_k, refangacc_k);
    ParticleUtils::vec_add(angacc_k, temp);
  }
}

void Particle::RigidBodyHandler::clear_rigid_body_orientation()
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k].data();

    // initialize rotation of rigid body k
    ParticleUtils::quaternion_clear(rot_k);
  }
}

void Particle::RigidBodyHandler::update_rigid_body_positions(const double timeincrement)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();
    double* rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k].data();
    const double* vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k].data();
    const double* angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k].data();

    // update position
    ParticleUtils::vec_add_scale(pos_k, timeincrement, vel_k);

    // save current rotation
    double curr_rot_k[4];
    ParticleUtils::quaternion_set(curr_rot_k, rot_k);

    // get rotation increment
    double phi_k[3];
    ParticleUtils::vec_set_scale(phi_k, timeincrement, angvel_k);

    double incr_rot_k[4];
    ParticleUtils::quaternion_from_angle(incr_rot_k, phi_k);

    // update rotation
    ParticleUtils::quaternion_product(rot_k, incr_rot_k, curr_rot_k);
  }
}

void Particle::RigidBodyHandler::update_rigid_body_velocities(const double timeincrement)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k].data();
    double* angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k].data();
    const double* acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k].data();
    const double* angacc_k =
        rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k].data();

    // update velocities
    ParticleUtils::vec_add_scale(vel_k, timeincrement, acc_k);
    ParticleUtils::vec_add_scale(angvel_k, timeincrement, angacc_k);
  }
}

void Particle::RigidBodyHandler::clear_rigid_body_accelerations()
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    double* acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k].data();
    double* angacc_k = rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k].data();

    // clear accelerations of rigid body k
    ParticleUtils::vec_clear(acc_k);
    ParticleUtils::vec_clear(angacc_k);
  }
}

void Particle::RigidBodyHandler::broadcast_rigid_body_positions()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to hosting processors of rigid body k
    std::vector<int>& hostingprocs_k = ownedrigidbodiestohostingprocs_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k];
    const std::vector<double>& rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k];

    // pack data for sending
    Core::Communication::PackBuffer data;

    data.add_to_pack(rigidbody_k);

    add_to_pack(data, pos_k);
    add_to_pack(data, rot_k);

    for (int torank : hostingprocs_k)
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      int rigidbody_k;
      extract_from_pack(buffer, rigidbody_k);

      // get reference to rigid body states
      std::vector<double>& pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k];
      std::vector<double>& rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k];

      extract_from_pack(buffer, pos_k);
      extract_from_pack(buffer, rot_k);
    }
  }
}

void Particle::RigidBodyHandler::broadcast_rigid_body_velocities()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to hosting processors of rigid body k
    std::vector<int>& hostingprocs_k = ownedrigidbodiestohostingprocs_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k];
    const std::vector<double>& angvel_k =
        rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k];

    // pack data for sending
    Core::Communication::PackBuffer data;

    data.add_to_pack(rigidbody_k);

    add_to_pack(data, vel_k);
    add_to_pack(data, angvel_k);

    for (int torank : hostingprocs_k)
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      int rigidbody_k;
      extract_from_pack(buffer, rigidbody_k);

      // get reference to rigid body states
      std::vector<double>& vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k];
      std::vector<double>& angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k];

      extract_from_pack(buffer, vel_k);
      extract_from_pack(buffer, angvel_k);
    }
  }
}

void Particle::RigidBodyHandler::broadcast_rigid_body_accelerations()
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get reference to hosting processors of rigid body k
    std::vector<int>& hostingprocs_k = ownedrigidbodiestohostingprocs_[rigidbody_k];

    // get reference to rigid body states
    const std::vector<double>& acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k];
    const std::vector<double>& angacc_k =
        rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k];

    // pack data for sending
    Core::Communication::PackBuffer data;

    data.add_to_pack(rigidbody_k);

    add_to_pack(data, acc_k);
    add_to_pack(data, angacc_k);

    for (int torank : hostingprocs_k)
      sdata[torank].insert(sdata[torank].end(), data().begin(), data().end());
  }

  // communicate data via non-buffered send from proc to proc
  ParticleUtils::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack and store received data
  for (auto& p : rdata)
  {
    std::vector<char>& rmsg = p.second;



    Core::Communication::UnpackBuffer buffer(rmsg);
    while (!buffer.at_end())
    {
      int rigidbody_k;
      extract_from_pack(buffer, rigidbody_k);

      // get reference to rigid body states
      std::vector<double>& acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k];
      std::vector<double>& angacc_k =
          rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k];

      extract_from_pack(buffer, acc_k);
      extract_from_pack(buffer, angacc_k);
    }
  }
}

void Particle::RigidBodyHandler::set_rigid_particle_relative_position_in_body_frame()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    double* relposbody_i =
        container_i->get_ptr_to_state(Particle::RelativePositionBodyFrame, particle_i);

    ParticleUtils::vec_set(relposbody_i, pos_i);
    ParticleUtils::vec_sub(relposbody_i, pos_k);
  }
}

void Particle::RigidBodyHandler::update_rigid_particle_relative_position()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* rot_k = rigidbodydatastate_->get_ref_rotation()[rigidbody_k].data();

    // get pointer to particle states
    const double* relposbody_i =
        container_i->get_ptr_to_state(Particle::RelativePositionBodyFrame, particle_i);
    double* relpos_i = container_i->get_ptr_to_state(Particle::RelativePosition, particle_i);

    // update relative position of particle i
    ParticleUtils::quaternion_rotate_vector(relpos_i, rot_k, relposbody_i);
  }
}

void Particle::RigidBodyHandler::set_rigid_particle_position()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i = container_i->get_ptr_to_state(Particle::RelativePosition, particle_i);
    double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);

    // set position of particle i
    ParticleUtils::vec_set(pos_i, pos_k);
    ParticleUtils::vec_add(pos_i, relpos_i);
  }
}

void Particle::RigidBodyHandler::set_rigid_particle_velocities()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k].data();
    const double* angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i = container_i->get_ptr_to_state(Particle::RelativePosition, particle_i);
    double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
    double* angvel_i = container_i->cond_get_ptr_to_state(Particle::AngularVelocity, particle_i);

    // set velocities of particle i
    ParticleUtils::vec_set(vel_i, vel_k);
    ParticleUtils::vec_add_cross(vel_i, angvel_k, relpos_i);
    if (angvel_i) ParticleUtils::vec_set(angvel_i, angvel_k);
  }
}

void Particle::RigidBodyHandler::set_rigid_particle_accelerations()
{
  // get reference to affiliation pair data
  const std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (static_cast<int>(affiliationpairdata.size()) != container_i->particles_stored())
    FOUR_C_THROW("number of affiliation pairs and rigid particles not equal!");
#endif

  // loop over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    auto it = affiliationpairdata.find(globalid_i[0]);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // no affiliation pair for current global id
    if (it == affiliationpairdata.end())
      FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i[0]);
#endif

    // get global id of affiliated rigid body k
    const int rigidbody_k = it->second;

    // get pointer to rigid body states
    const double* angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k].data();
    const double* acc_k = rigidbodydatastate_->get_ref_acceleration()[rigidbody_k].data();
    const double* angacc_k =
        rigidbodydatastate_->get_ref_angular_acceleration()[rigidbody_k].data();

    // get pointer to particle states
    const double* relpos_i = container_i->get_ptr_to_state(Particle::RelativePosition, particle_i);
    double* acc_i = container_i->get_ptr_to_state(Particle::Acceleration, particle_i);
    double* angacc_i =
        container_i->cond_get_ptr_to_state(Particle::AngularAcceleration, particle_i);

    // evaluate relative velocity of particle i
    double relvel_i[3];
    ParticleUtils::vec_set_cross(relvel_i, angvel_k, relpos_i);

    // set accelerations of particle i
    ParticleUtils::vec_set(acc_i, acc_k);
    ParticleUtils::vec_add_cross(acc_i, angacc_k, relpos_i);
    ParticleUtils::vec_add_cross(acc_i, angvel_k, relvel_i);
    if (angacc_i) ParticleUtils::vec_set(angacc_i, angacc_k);
  }
}

void Particle::RigidBodyHandler::evaluate_rigid_body_melting(
    const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase)
{
  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // iterate over particle phase change tuples
  for (const auto& particletypetotype : particlesfromphasetophase)
  {
    Particle::TypeEnum type_source;
    int globalid_i;
    std::tie(type_source, std::ignore, globalid_i) = particletypetotype;

    if (type_source == Particle::RigidPhase)
    {
      auto it = affiliationpairdata.find(globalid_i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      // no affiliation pair for current global id
      if (it == affiliationpairdata.end())
        FOUR_C_THROW("no affiliated rigid body found for particle with global id {}", globalid_i);
#endif

      // erase affiliation pair
      affiliationpairdata.erase(it);
    }
  }
}

void Particle::RigidBodyHandler::evaluate_rigid_body_solidification(
    const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase)
{
  // get search radius
  const double searchradius = params_.get<double>("RIGID_BODY_PHASECHANGE_RADIUS");
  if (not(searchradius > 0.0)) FOUR_C_THROW("search radius not positive!");

  // get reference to affiliation pair data
  std::unordered_map<int, int>& affiliationpairdata =
      affiliationpairs_->get_ref_to_affiliation_pair_data();

  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  // get container of owned particles of rigid phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle->get_specific_container(Particle::RigidPhase, Particle::Owned);

  // iterate over particle phase change tuples
  for (const auto& particletypetotype : particlesfromphasetophase)
  {
    Particle::TypeEnum type_target;
    int globalid_i;
    std::tie(std::ignore, type_target, globalid_i) = particletypetotype;

    if (type_target == Particle::RigidPhase)
    {
      // get local index in specific particle container
      Particle::LocalIndexTupleShrdPtr localindextuple =
          particleengineinterface_->get_local_index_in_specific_container(globalid_i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not localindextuple)
        FOUR_C_THROW("particle with global id {} not found on this processor!", globalid_i);

      // access values of local index tuples of particle i
      Particle::TypeEnum type_i;
      Particle::StatusEnum status_i;
      std::tie(type_i, status_i, std::ignore) = *localindextuple;

      if (type_i != Particle::RigidPhase)
        FOUR_C_THROW("particle with global id {} not of particle type '{}'!", globalid_i,
            Particle::enum_to_type_name(Particle::RigidPhase));

      if (status_i == Particle::Ghosted)
        FOUR_C_THROW("particle with global id {} not owned on this processor!", globalid_i);
#endif

      // access values of local index tuples of particle i
      int particle_i;
      std::tie(std::ignore, std::ignore, particle_i) = *localindextuple;

      // get pointer to particle states
      const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
      double* rigidbodycolor_i =
          container_i->get_ptr_to_state(Particle::RigidBodyColor, particle_i);

      // get particles within radius
      std::vector<Particle::LocalIndexTuple> neighboringparticles;
      particleengineinterface_->get_particles_within_radius(
          pos_i, searchradius, neighboringparticles);

      // minimum distance between particles
      double mindist = searchradius;

      // iterate over neighboring particles
      for (const auto& neighboringparticle : neighboringparticles)
      {
        // access values of local index tuple of particle j
        Particle::TypeEnum type_j;
        Particle::StatusEnum status_j;
        int particle_j;
        std::tie(type_j, status_j, particle_j) = neighboringparticle;

        // evaluation only for rigid particles
        if (type_j != Particle::RigidPhase) continue;

        // get container of particles of current particle type
        Particle::ParticleContainer* container_j =
            particlecontainerbundle->get_specific_container(type_j, status_j);

        // get pointer to particle states
        const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);
        const double* rigidbodycolor_j =
            container_j->get_ptr_to_state(Particle::RigidBodyColor, particle_j);

        // vector from particle i to j
        double r_ji[3];
        ParticleUtils::vec_set(r_ji, pos_j);
        ParticleUtils::vec_sub(r_ji, pos_i);

        // absolute distance between particles
        const double absdist = ParticleUtils::vec_norm_two(r_ji);

        // set rigid body color to the one of the closest rigid particle j
        if (absdist < mindist)
        {
          mindist = absdist;
          rigidbodycolor_i[0] = rigidbodycolor_j[0];
        }
      }

      // get global id of affiliated rigid body k
      const int rigidbody_k = std::round(rigidbodycolor_i[0]);

      // insert affiliation pair
      affiliationpairdata.insert(std::make_pair(globalid_i, rigidbody_k));
    }
  }
}

void Particle::RigidBodyHandler::set_rigid_body_velocities_after_phase_change(
    const std::vector<std::vector<double>>& previousposition)
{
  // iterate over owned rigid bodies
  for (const int rigidbody_k : ownedrigidbodies_)
  {
    // get pointer to rigid body states
    const double* pos_k = rigidbodydatastate_->get_ref_position()[rigidbody_k].data();
    const double* angvel_k = rigidbodydatastate_->get_ref_angular_velocity()[rigidbody_k].data();
    double* vel_k = rigidbodydatastate_->get_ref_velocity()[rigidbody_k].data();

    const double* prevpos_k = previousposition[rigidbody_k].data();

    // vector from previous to current position of rigid body k
    double prev_r_kk[3];
    ParticleUtils::vec_set(prev_r_kk, pos_k);
    ParticleUtils::vec_sub(prev_r_kk, prevpos_k);

    // update velocity
    ParticleUtils::vec_add_cross(vel_k, angvel_k, prev_r_kk);
  }
}

void Particle::RigidBodyHandler::set_initial_conditions()
{
  Particle::set_initial_fields(params_, ownedrigidbodies_, *rigidbodydatastate_);
}

FOUR_C_NAMESPACE_CLOSE
