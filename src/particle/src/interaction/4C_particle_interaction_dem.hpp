// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_input.hpp"
#include "4C_particle_interaction_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class DEMAdhesion;
  class DEMContact;
  class DEMHistoryPairs;
  class DEMNeighborPairs;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief discrete element method (DEM) interaction
   *
   */

  class ParticleInteractionDEM final : public ParticleInteractionBase
  {
   public:
    //! constructor
    explicit ParticleInteractionDEM(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~ParticleInteractionDEM() override;

    //! setup particle interaction handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface) override;

    //! write restart of particle interaction handler
    void write_restart() const override;

    //! read restart of particle interaction handler
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader) override;

    //! insert interaction dependent states of all particle types
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes)
        override;

    //! set initial states
    void set_initial_states() override;

    //! pre evaluate time step
    void pre_evaluate_time_step() override;

    //! evaluate particle interactions
    void evaluate_interactions() override;

    //! post evaluate time step
    void post_evaluate_time_step(
        std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase) override;

    //! maximum interaction distance (on this processor)
    double max_interaction_distance() const override;

    //! distribute interaction history
    void distribute_interaction_history() const override;

    //! communicate interaction history
    void communicate_interaction_history() const override;

    //! set current step size
    void set_current_step_size(const double currentstepsize) override;

   private:
    //! setup particle interaction writer
    void setup_particle_interaction_writer();

    //! set initial radius
    void set_initial_radius();

    //! set initial mass
    void set_initial_mass();

    //! set initial inertia
    void set_initial_inertia();

    //! clear force and moment states of particles
    void clear_force_and_moment_states() const;

    //! compute acceleration from force and moment
    void compute_acceleration() const;

    //! evaluate particle energy
    void evaluate_particle_energy() const;

    //! evaluate particle kinetic energy contribution
    void evaluate_particle_kinetic_energy(double& kineticenergy) const;

    //! evaluate particle gravitational potential energy contribution
    void evaluate_particle_gravitational_potential_energy(
        double& gravitationalpotentialenergy) const;

    //! discrete element method specific parameter list
    const Teuchos::ParameterList& params_dem_;

    //! neighbor pair handler
    std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs_;

    //! history pair handler
    std::shared_ptr<Particle::DEMHistoryPairs> historypairs_;

    //! contact handler
    std::unique_ptr<Particle::DEMContact> contact_;

    //! adhesion handler
    std::unique_ptr<Particle::DEMAdhesion> adhesion_;

    //! write particle energy output
    const bool writeparticleenergy_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
