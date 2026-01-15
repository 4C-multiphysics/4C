// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_HPP

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
  class SPHBoundaryParticleBase;
  class SPHDensityBase;
  class SPHEquationOfStateBundle;
  class SPHKernelBase;
  class SPHMomentum;
  class SPHNeighborPairs;
  class SPHOpenBoundaryBase;
  class SPHPhaseChangeBase;
  class SPHPressure;
  class SPHRigidParticleContactBase;
  class SPHSurfaceTension;
  class SPHTemperature;
  class SPHVirtualWallParticle;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief smoothed particle hydrodynamics (SPH) interaction
   *
   */
  class ParticleInteractionSPH final : public ParticleInteractionBase
  {
   public:
    //! constructor
    explicit ParticleInteractionSPH(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~ParticleInteractionSPH() override;

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

    //! set current time
    void set_current_time(const double currenttime) override;

    //! set current step size
    void set_current_step_size(const double currentstepsize) override;

   private:
    /*!
     * Initialize the members by assigning the respective derived classes based on the input
     * parameters
     */
    void initialize_members();

    //! init kernel handler
    void init_kernel_handler();

    //! init equation of state bundle
    void init_equation_of_state_bundle();

    //! init neighbor pair handler
    void init_neighbor_pair_handler();

    //! init density handler
    void init_density_handler();

    //! init pressure handler
    void init_pressure_handler();

    //! init temperature handler
    void init_temperature_handler();

    //! init momentum handler
    void init_momentum_handler();

    //! init surface tension handler
    void init_surface_tension_handler();

    //! init boundary particle handler
    void init_boundary_particle_handler();

    //! init dirichlet open boundary handler
    void init_dirichlet_open_boundary_handler();

    //! init neumann open boundary handler
    void init_neumann_open_boundary_handler();

    //! init virtual wall particle handler
    void init_virtual_wall_particle_handler();

    //! init phase change handler
    void init_phase_change_handler();

    //! init rigid particle contact handler
    void init_rigid_particle_contact_handler();

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! kernel handler
    std::shared_ptr<Particle::SPHKernelBase> kernel_;

    //! equation of state bundle
    std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! density handler
    std::unique_ptr<Particle::SPHDensityBase> density_;

    //! pressure handler
    std::unique_ptr<Particle::SPHPressure> pressure_;

    //! temperature handler
    std::unique_ptr<Particle::SPHTemperature> temperature_;

    //! momentum handler
    std::unique_ptr<Particle::SPHMomentum> momentum_;

    //! surface tension handler
    std::unique_ptr<Particle::SPHSurfaceTension> surfacetension_;

    //! boundary particle handler
    std::unique_ptr<Particle::SPHBoundaryParticleBase> boundaryparticle_;

    //! dirichlet open boundary handler
    std::unique_ptr<Particle::SPHOpenBoundaryBase> dirichletopenboundary_;

    //! neumann open boundary handler
    std::unique_ptr<Particle::SPHOpenBoundaryBase> neumannopenboundary_;

    //! virtual wall particle handler
    std::shared_ptr<Particle::SPHVirtualWallParticle> virtualwallparticle_;

    //! phase change handler
    std::unique_ptr<Particle::SPHPhaseChangeBase> phasechange_;

    //! rigid particle contact handler
    std::unique_ptr<Particle::SPHRigidParticleContactBase> rigidparticlecontact_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
