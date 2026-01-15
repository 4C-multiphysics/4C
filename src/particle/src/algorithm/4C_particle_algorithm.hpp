// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_HPP
#define FOUR_C_PARTICLE_ALGORITHM_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_particle_engine.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_wall.hpp"
#include "4C_utils_result_test.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class GravityHandler;
  class ParticleInteractionBase;
  class ParticleObject;
  class RigidBodyHandler;
  class TimInt;
  class ViscousDampingHandler;
}  // namespace Particle

namespace Discret
{
  class ResultTest;
}

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief algorithm to control particle simulations
   *
   * The particle algorithm controls all major steps of particle simulations. Besides initialization
   * and setup of the problem this mainly concerns the control of the time loop (with the
   * integration of the particle step) consisting of the following steps:
   * - update and maintain valid connectivity (parallel distribution, load transfer, ghosting,
   *   neighbor pair relations)
   * - evaluation of particle accelerations from gravity acceleration, interactions, and viscous
   *   damping
   * - (runtime) output of fields
   * - write restart information
   *
   */

  class ParticleAlgorithm final : public Adapter::AlgorithmBase
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle simulation parameter list
     */
    explicit ParticleAlgorithm(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~ParticleAlgorithm() override;

    /*!
     * \brief init particle algorithm
     *
     *
     * \param[in] initialparticles particle objects read in from restart
     */
    void init(std::vector<Particle::ParticleObjShrdPtr>& initialparticles);

    /*!
     * \brief setup particle algorithm
     *
     */
    void setup();

    /*!
     * \brief read restart information for given time step
     *
     * Read the restart information in the same order as it is written.
     *
     *
     * \param[in] restartstep restart step
     */
    void read_restart(const int restartstep) override;

    /*!
     * \brief time loop for particle problem
     *
     */
    void timeloop();

    /*!
     * \brief prepare time step
     *
     *
     * \param[in] do_print_header flag to control output of time step header
     */
    void prepare_time_step(bool do_print_header = true);

    /*!
     * \brief pre evaluate time step
     *
     */
    void pre_evaluate_time_step();

    /*!
     * \brief integrate time step
     *
     */
    void integrate_time_step();

    /*!
     * \brief post evaluate time step
     *
     */
    void post_evaluate_time_step();

    /*!
     * \brief write output
     *
     */
    void write_output() const;

    /*!
     * \brief write restart information
     *
     * The order of writing the restart information in the particle framework is important: First
     * restart information is written by the bin discretization writer, afterwards restart
     * information is written by the wall discretization writer. This order is important, otherwise
     * the restart fails.
     *
     */
    void write_restart() const;

    /*!
     * \brief create particle field specific result test objects
     *
     *
     * \return particle field specific result test objects
     */
    std::vector<std::shared_ptr<Core::Utils::ResultTest>> create_result_tests();

    /*!
     * \brief get interface to particle engine
     *
     *
     * \return interface to particle engine
     */
    std::shared_ptr<Particle::ParticleEngineInterface> get_particle_engine_interface() const
    {
      return particleengine_;
    }

    /*!
     * \brief get interface to particle wall handler
     *
     *
     * \return interface to particle wall handler
     */
    std::shared_ptr<Particle::WallHandlerInterface> get_particle_wall_handler_interface() const
    {
      return particlewall_;
    }

   private:
    //! \name init and setup methods
    //! @{

    /*!
     * \brief init particle engine
     *
     */
    void init_particle_engine();

    /*!
     * \brief init particle wall handler
     *
     */
    void init_particle_wall();

    /*!
     * \brief init rigid body handler
     *
     */
    void init_particle_rigid_body();

    /*!
     * \brief init particle time integration
     *
     */
    void init_particle_time_integration();

    /*!
     * \brief init particle interaction handler
     *
     */
    void init_particle_interaction();

    /*!
     * \brief init particle gravity handler
     *
     */
    void init_particle_gravity();

    /*!
     * \brief init viscous damping handler
     *
     */
    void init_viscous_damping();

    /*!
     * \brief generate initial particles
     *
     */
    void generate_initial_particles();

    /*!
     * \brief determine all particle types
     *
     */
    void determine_particle_types();

    /*!
     * \brief determine particle states of all particle types
     *
     */
    void determine_particle_states_of_particle_types();

    /*!
     * \brief setup initial particles
     *
     */
    void setup_initial_particles();

    /*!
     * \brief setup initial rigid bodies
     *
     */
    void setup_initial_rigid_bodies();

    /*!
     * \brief setup initial states
     *
     */
    void setup_initial_states();

    //! @}

    /*!
     * \brief update and maintain valid connectivity in particle problems
     *
     * The main control routine of particle problems to maintain valid particle connectivity:
     * - parallel distribution of particles
     * - transfer of particles to new bins
     * - ghosting of particles
     * - neighbor pair relations
     *
     * This method includes several safety checks that are part of the called sub-methods.
     *
     */
    void update_connectivity();

    /*!
     * \brief check load transfer
     *
     * Check if a load transfer is needed based on the following criteria:
     * - maximum position increment of particles or walls
     * - invalid particle connectivity
     * - invalid particle neighbor pair relation
     * - invalid particle wall neighbor pair relation
     *
     *
     * \return flag indicating load transfer is needed
     */
    bool check_load_transfer_needed();

    /*!
     * \brief check maximum position increment
     *
     *
     * \return flag indicating load transfer is needed due to maximum position increment
     */
    bool check_max_position_increment();

    /*!
     * \brief get maximum particle position increment since last transfer
     *
     *
     * \param allprocmaxpositionincrement maximum particle position increment
     */
    void get_max_particle_position_increment(double& allprocmaxpositionincrement);

    /*!
     * \brief transfer load between processors
     *
     */
    void transfer_load_between_procs();

    /*!
     * \brief check load redistribution
     *
     * Check if a load redistribution is needed based on the difference in the absolute number of
     * particles since the last redistribution.
     *
     *
     * \return flag indicating load redistribution is needed
     */
    bool check_load_redistribution_needed();

    /*!
     * \brief distribute load among processors
     *
     */
    void distribute_load_among_procs();

    /*!
     * \brief build potential neighbor relation
     *
     */
    void build_potential_neighbor_relation();

    /*!
     * \brief set initial conditions
     *
     */
    void set_initial_conditions();

    /*!
     * \brief set current time
     *
     */
    void set_current_time();

    /*!
     * \brief set current step size
     *
     */
    void set_current_step_size();

    /*!
     * \brief set current write result flag
     *
     */
    void set_current_write_result_flag();

    /*!
     * \brief evaluate time step
     *
     */
    void evaluate_time_step();

    /*!
     * \brief set gravity acceleration
     *
     */
    void set_gravity_acceleration();

    //! processor id
    const int myrank_;

    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! particle engine
    std::shared_ptr<Particle::ParticleEngine> particleengine_;

    //! particle wall handler
    std::shared_ptr<Particle::WallHandlerBase> particlewall_;

    //! rigid body handler
    std::shared_ptr<Particle::RigidBodyHandler> particlerigidbody_;

    //! particle time integration
    std::unique_ptr<Particle::TimInt> particletimint_;

    //! particle interaction
    std::unique_ptr<Particle::ParticleInteractionBase> particleinteraction_;

    //! particle gravity handler
    std::unique_ptr<Particle::GravityHandler> particlegravity_;

    //! particle viscous damping handler
    std::unique_ptr<Particle::ViscousDampingHandler> viscousdamping_;

    //! map of particle types and corresponding states
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>> particlestatestotypes_;

    //! vector of initial or generated particles to distribute
    std::vector<Particle::ParticleObjShrdPtr> particlestodistribute_;

    //! number of particles on this processor after last load balance
    int numparticlesafterlastloadbalance_;

    //! transfer particles to new bins every time step
    bool transferevery_;

    //! write results interval
    const int writeresultsevery_;

    //! write restart interval
    const int writerestartevery_;

    //! result control flag
    bool writeresultsthisstep_;

    //! restart control flag
    bool writerestartthisstep_;

    //! simulation is restarted
    bool isrestarted_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
