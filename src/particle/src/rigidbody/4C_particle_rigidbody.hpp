// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_RIGIDBODY_HPP
#define FOUR_C_PARTICLE_RIGIDBODY_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_rigidbody_interface.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <mpi.h>

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::IO
{
  class DiscretizationReader;
}

namespace Particle
{
  class ParticleEngineInterface;
  class RigidBodyAffiliationPairs;
  class RigidBodyDataState;
  class RigidBodyRuntimeVtpWriter;
  class UniqueGlobalIdHandler;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief rigid body handler for particle problem
   *
   */
  class RigidBodyHandler final : public RigidBodyHandlerInterface
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle simulation parameter list
     */
    explicit RigidBodyHandler(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~RigidBodyHandler() override;

    /*!
     * \brief setup rigid body handler
     *
     */
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface);

    /*!
     * \brief write restart of rigid body handler
     *
     */
    void write_restart() const;

    /*!
     * \brief read restart of rigid body handler
     *
     *
     * \param[in] reader discretization reader
     */
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader);

    /*!
     * \brief insert rigid body handler dependent states of all particle types
     *
     *
     * \param[out] particlestatestotypes map of particle types and corresponding states
     */
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const;

    /*!
     * \brief write rigid body runtime output
     *
     *
     * \param[in] step output step
     * \param[in] time output time
     */
    void write_rigid_body_runtime_output(const int step, const double time) const;

    /*!
     * \brief set initial affiliation pair data
     *
     * The rigid body color of each rigid particle given via the input file in the particle section
     * defines the affiliation of a rigid particle to a rigid body. Upon this information the
     * affiliation pair data is initialized.
     *
     */
    void set_initial_affiliation_pair_data();

    /*!
     * \brief set unique global ids for all rigid bodies
     *
     * The rigid body unique global identifier handler is initialized. This requires knowledge of
     * all global ids currently assigned to rigid bodies over all processors.
     *
     */
    void set_unique_global_ids_for_all_rigid_bodies();

    /*!
     * \brief allocate rigid body states
     *
     */
    void allocate_rigid_body_states();

    /*!
     * \brief initialize rigid body mass quantities and orientation
     *
     * Compute the mass quantities and clear the orientation (describing the body frame) of rigid
     * bodies. This includes setting the relative position of rigid particles in the body frame and
     * in the reference frame.
     *
     */
    void initialize_rigid_body_mass_quantities_and_orientation();

    /*!
     * \brief distribute rigid body
     *
     */
    void distribute_rigid_body();

    /*!
     * \brief communicate rigid body
     *
     */
    void communicate_rigid_body();

    /*!
     * \brief clear forces and torques
     *
     * Clear the forces and torques acting on rigid bodies and on the corresponding rigid particles
     * affiliated with the rigid bodies.
     *
     */
    void clear_forces_and_torques();

    /*!
     * \brief add gravity acceleration
     *
     * Add the gravity acceleration acting on rigid bodies.
     *
     *
     * \param[in] gravity gravity acceleration
     */
    void add_gravity_acceleration(std::vector<double>& gravity);

    /*!
     * \brief compute accelerations of rigid bodies
     *
     * First compute the force and the torque acting on rigid bodies over all processors. Then, the
     * accelerations are computed from force and torque using the mass quantities on the owning
     * processor. Finally, the corresponding accelerations of rigid particles affiliated with the
     * rigid bodies are set.
     *
     */
    void compute_accelerations();

    /*!
     * \brief update positions with given time increment
     *
     * Update the positions of rigid bodies with a given time increment and set the corresponding
     * positions of rigid particles affiliated with the rigid bodies.
     *
     *
     * \param[in] timeincrement time increment
     */
    void update_positions(const double timeincrement) override;

    /*!
     * \brief update velocities with given time increment
     *
     * Update the velocities of rigid bodies with a given time increment and set the corresponding
     * velocities of rigid particles affiliated with the rigid bodies.
     *
     *
     * \param[in] timeincrement time increment
     */
    void update_velocities(const double timeincrement) override;

    /*!
     * \brief clear accelerations
     *
     * Clear the accelerations of rigid bodies. However, note that the accelerations of rigid
     * particles remain untouched!
     *
     */
    void clear_accelerations() override;

    /*!
     * \brief have rigid body phase change
     *
     * Check for phase change, i.e., melting or solidification, of rigid bodies over all processors.
     *
     *
     * \param[in] particlesfromphasetophase particle phase change tuples
     *
     * \return have phase change
     */
    bool have_rigid_body_phase_change(
        const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase);

    /*!
     * \brief evaluate rigid body phase change
     *
     * Evaluate phase change, i.e., melting or solidification, of rigid bodies by tracking the phase
     * change of rigid particles owned on this processor from or to the rigid phase.
     *
     *
     * \param[in] particlesfromphasetophase particle phase change tuples
     */
    void evaluate_rigid_body_phase_change(
        const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase);

    std::shared_ptr<Particle::RigidBodyDataState> get_rigid_body_data_state() const override
    {
      return rigidbodydatastate_;
    }

    const std::vector<int>& get_owned_rigid_bodies() const override { return ownedrigidbodies_; }

    /*!
     * \brief set initial states of rigid particles
     *
     * Evaluate initial conditions of rigid particles.
     */
    void set_initial_conditions();

   private:
    /*!
     * \brief init rigid body unique global identifier handler
     *
     */
    void init_rigid_body_unique_global_id_handler();

    /*!
     * \brief init rigid body data state container
     *
     */
    void init_rigid_body_data_state();

    /*!
     * \brief init rigid body runtime vtp writer
     *
     */
    void init_rigid_body_vtp_writer();

    /*!
     * \brief init affiliation pair handler
     *
     */
    void init_affiliation_pair_handler();

    /*!
     * \brief get packed rigid body state data
     *
     * Get the packed required states of all rigid bodies owned by this processor.
     *
     *
     * \param[out] buffer buffer of packed rigid body state data
     */
    void get_packed_rigid_body_states(std::vector<char>& buffer) const;

    /*!
     * \brief extract packed rigid body state data
     *
     *
     * \param[in] buffer buffer of packed rigid body state data
     */
    void extract_packed_rigid_body_states(std::vector<char>& buffer);

    /*!
     * \brief update rigid body ownership
     *
     * Update the ownership of rigid bodies on all processors.
     *
     */
    void update_rigid_body_ownership();

    /*!
     * \brief determine owned and hosted rigid bodies
     *
     * First, the global ids of rigid bodies hosted, i.e., owned and non-owned, by this processor
     * are determined using the affiliation of rigid particles to rigid bodies. Then, based on the
     * maximum number of rigid particles per processor the owning processor of all rigid bodies is
     * determined. Finally, each processor stores the global ids of rigid bodies owned.
     *
     */
    void determine_owned_and_hosted_rigid_bodies();

    /*!
     * \brief relate owned rigid bodies to all hosting processors
     *
     * Relate the global ids of rigid bodies owned by this processor to the corresponding hosting
     * processors.
     *
     */
    void relate_owned_rigid_bodies_to_hosting_procs();

    /*!
     * \brief communicate rigid body states
     *
     * Communicate required states of all rigid bodies previously owned by this processor to the new
     * owning processor.
     *
     *
     * \param[in] previouslyownedrigidbodies rigid bodies previously owned by this processor
     */
    void communicate_rigid_body_states(std::vector<int>& previouslyownedrigidbodies);

    /*!
     * \brief compute mass quantities of rigid bodies
     *
     * Compute the mass quantities, i.e., mass, center of gravity, and mass moment of inertia, of
     * rigid bodies over all processors.
     *
     */
    void compute_rigid_body_mass_quantities();

    /*!
     * \brief clear partial mass quantities of rigid bodies
     *
     * Clear partial mass quantities of hosted rigid bodies on this processor.
     *
     */
    void clear_partial_mass_quantities();

    /*!
     * \brief compute partial mass quantities of rigid bodies
     *
     * Compute the partial mass quantities of hosted rigid bodies on respective hosting processors.
     *
     */
    void compute_partial_mass_quantities();

    /*!
     * \brief gather partial mass quantities of rigid bodies
     *
     * Gather the partial mass quantities for owned rigid bodies from respective hosting processors
     * on owning processor.
     *
     *
     * \param[out] gatheredpartialmass     gathered partial mass
     * \param[out] gatheredpartialinertia  gathered partial inertia
     * \param[out] gatheredpartialposition gathered partial position
     */
    void gather_partial_mass_quantities(
        std::unordered_map<int, std::vector<double>>& gatheredpartialmass,
        std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialinertia,
        std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialposition);

    /*!
     * \brief compute full mass quantities of rigid bodies
     *
     * Compute the full mass quantities of owned rigid bodies on owning processor using all gathered
     * partial mass quantities.
     *
     *
     * \param[in] gatheredpartialmass     gathered partial mass
     * \param[in] gatheredpartialinertia  gathered partial inertia
     * \param[in] gatheredpartialposition gathered partial position
     */
    void compute_full_mass_quantities(
        std::unordered_map<int, std::vector<double>>& gatheredpartialmass,
        std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialinertia,
        std::unordered_map<int, std::vector<std::vector<double>>>& gatheredpartialposition);

    /*!
     * \brief clear force and torque acting on rigid bodies
     *
     * Clear the force and torque of hosted rigid bodies on this processor.
     *
     */
    void clear_rigid_body_force_and_torque();

    /*!
     * \brief clear force acting on rigid particles
     *
     * Clear the force of rigid particles owned on this processor.
     *
     */
    void clear_rigid_particle_force();

    /*!
     * \brief compute partial force and torque acting on rigid bodies
     *
     * Compute the partial force and torque acting on hosted rigid bodies on respective hosting
     * processors.
     *
     */
    void compute_partial_force_and_torque();

    /*!
     * \brief gather partial and compute full force and torque acting on rigid bodies
     *
     * Gather the partial force and torque acting on owned rigid bodies from respective hosting
     * processors on owning processor and compute the full force and torque.
     *
     */
    void gather_partial_and_compute_full_force_and_torque();

    /*!
     * \brief compute accelerations of rigid bodies from force and torque
     *
     * Compute the accelerations of owned rigid bodies from force and torque using mass quantities
     * on the owning processor.
     *
     */
    void compute_accelerations_from_force_and_torque();

    /*!
     * \brief clear orientation of rigid bodies
     *
     * Clear the orientation of rigid bodies owned on this processor.
     *
     */
    void clear_rigid_body_orientation();

    /*!
     * \brief update positions of rigid bodies with given time increment
     *
     * Update the positions of rigid bodies with a given time increment on the owning processor.
     *
     *
     * \param[in] timeincrement time increment
     */
    void update_rigid_body_positions(const double timeincrement);

    /*!
     * \brief update velocities of rigid bodies with given time increment
     *
     * Update the velocities of rigid bodies with a given time increment on the owning processor.
     *
     *
     * \param[in] timeincrement time increment
     */
    void update_rigid_body_velocities(const double timeincrement);

    /*!
     * \brief clear accelerations of rigid bodies
     *
     * Clear the accelerations of rigid bodies on the owning processor.
     *
     */
    void clear_rigid_body_accelerations();

    /*!
     * \brief broadcast positions of rigid bodies
     *
     * Broadcast positions of rigid bodies from the owning processor to respective local processors.
     *
     */
    void broadcast_rigid_body_positions();

    /*!
     * \brief broadcast velocities of rigid bodies
     *
     * Broadcast velocities of rigid bodies from the owning processor to respective local
     * processors.
     *
     */
    void broadcast_rigid_body_velocities();

    /*!
     * \brief broadcast accelerations of rigid bodies
     *
     * Broadcast accelerations of rigid bodies from the owning processor to respective local
     * processors.
     *
     */
    void broadcast_rigid_body_accelerations();

    /*!
     * \brief set relative position of rigid particles in body frame
     *
     * Set the relative position of rigid particles in the body frame owned on this processor.
     *
     */
    void set_rigid_particle_relative_position_in_body_frame();

    /*!
     * \brief update relative position of rigid particles
     *
     * Update the relative position of rigid particles in the rotating frame owned on this
     * processor.
     *
     */
    void update_rigid_particle_relative_position();

    /*!
     * \brief set position of rigid particles
     *
     * Set the position of rigid particles owned on this processor consistent to the affiliated
     * rigid body.
     *
     */
    void set_rigid_particle_position();

    /*!
     * \brief set velocities of rigid particles
     *
     * Set the velocities of rigid particles owned on this processor consistent to the affiliated
     * rigid body.
     *
     */
    void set_rigid_particle_velocities();

    /*!
     * \brief set accelerations of rigid particles
     *
     * Set the accelerations of rigid particles owned on this processor consistent to the affiliated
     * rigid body.
     *
     */
    void set_rigid_particle_accelerations();

    /*!
     * \brief evaluate melting of rigid bodies
     *
     *
     * \param[in] particlesfromphasetophase particle phase change tuples
     */
    void evaluate_rigid_body_melting(
        const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase);

    /*!
     * \brief evaluate solidification of rigid bodies
     *
     *
     * \param[in] particlesfromphasetophase particle phase change tuples
     */
    void evaluate_rigid_body_solidification(
        const std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase);

    /*!
     * \brief set velocities of rigid bodies after phase change
     *
     * Set the velocities of rigid bodies after a phase change on the owning processor.
     *
     *
     * \param[in] previousposition previous position of rigid bodies
     */
    void set_rigid_body_velocities_after_phase_change(
        const std::vector<std::vector<double>>& previousposition);

    //! communicator
    MPI_Comm comm_;

    //! processor id
    const int myrank_;

    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! rigid body unique global identifier handler
    std::unique_ptr<Particle::UniqueGlobalIdHandler> rigidbodyuniqueglobalidhandler_;

    //! rigid body data state container
    std::shared_ptr<Particle::RigidBodyDataState> rigidbodydatastate_;

    //! rigid body runtime vtp writer
    std::unique_ptr<Particle::RigidBodyRuntimeVtpWriter> rigidbodyvtpwriter_;

    //! affiliation pair handler
    std::unique_ptr<Particle::RigidBodyAffiliationPairs> affiliationpairs_;

    //! rigid bodies owned by this processor
    std::vector<int> ownedrigidbodies_;

    //! rigid bodies hosted (owned and non-owned) by this processor
    std::vector<int> hostedrigidbodies_;

    //! owner of all rigid bodies
    std::vector<int> ownerofrigidbodies_;

    //! owned rigid bodies related to hosting processors (without this processor)
    std::vector<std::vector<int>> ownedrigidbodiestohostingprocs_;
  };
}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
