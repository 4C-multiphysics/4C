#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_inpar_particle.hpp"
#include "4C_particle_interaction_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class DEMNeighborPairs;
  class DEMHistoryPairs;
  class DEMContact;
  class DEMAdhesion;
}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  /*!
   * \brief discrete element method (DEM) interaction
   *
   * \author Sebastian Fuchs \date 05/2018
   */

  class ParticleInteractionDEM final : public ParticleInteractionBase
  {
   public:
    //! constructor
    explicit ParticleInteractionDEM(const Epetra_Comm& comm, const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     * \author Sebastian Fuchs \date 10/2018
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~ParticleInteractionDEM() override;

    //! init particle interaction handler
    void init() override;

    //! setup particle interaction handler
    void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface) override;

    //! write restart of particle interaction handler
    void write_restart() const override;

    //! read restart of particle interaction handler
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader) override;

    //! insert interaction dependent states of all particle types
    void insert_particle_states_of_particle_types(
        std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>&
            particlestatestotypes) override;

    //! set initial states
    void set_initial_states() override;

    //! pre evaluate time step
    void pre_evaluate_time_step() override;

    //! evaluate particle interactions
    void evaluate_interactions() override;

    //! post evaluate time step
    void post_evaluate_time_step(
        std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase) override;

    //! maximum interaction distance (on this processor)
    double max_interaction_distance() const override;

    //! distribute interaction history
    void distribute_interaction_history() const override;

    //! communicate interaction history
    void communicate_interaction_history() const override;

    //! set current step size
    void set_current_step_size(const double currentstepsize) override;

   private:
    //! init neighbor pair handler
    void init_neighbor_pair_handler();

    //! init history pair handler
    void init_history_pair_handler();

    //! init contact handler
    void init_contact_handler();

    //! init adhesion handler
    void init_adhesion_handler();

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
    std::shared_ptr<ParticleInteraction::DEMNeighborPairs> neighborpairs_;

    //! history pair handler
    std::shared_ptr<ParticleInteraction::DEMHistoryPairs> historypairs_;

    //! contact handler
    std::unique_ptr<ParticleInteraction::DEMContact> contact_;

    //! adhesion handler
    std::unique_ptr<ParticleInteraction::DEMAdhesion> adhesion_;

    //! write particle energy output
    const bool writeparticleenergy_;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
