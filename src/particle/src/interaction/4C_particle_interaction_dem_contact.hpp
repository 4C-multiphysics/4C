// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_CONTACT_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_CONTACT_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class DEMContactNormalBase;
  class DEMContactRollingBase;
  class DEMContactTangentialBase;
  class DEMHistoryPairs;
  class DEMNeighborPairs;
  class InteractionWriter;
  class MaterialHandler;
  class ParticleContainerBundle;
  class ParticleEngineInterface;
  class WallHandlerInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class DEMContact final
  {
   public:
    //! constructor
    explicit DEMContact(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~DEMContact();

    //! setup contact handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
        const std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs,
        const std::shared_ptr<Particle::DEMHistoryPairs> historypairs);

    //! set current step size
    void set_current_step_size(const double currentstepsize);

    //! insert contact evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const;

    //! get normal contact stiffness
    double get_normal_contact_stiffness() const;

    //! check critical time step (on this processor)
    void check_critical_time_step() const;

    //! add contact contribution to force and moment field
    void add_force_and_moment_contribution();

    //! evaluate elastic potential energy contribution
    void evaluate_elastic_potential_energy(double& elasticpotentialenergy) const;

   private:
    //! init normal contact handler
    void init_normal_contact_handler();

    //! init tangential contact handler
    void init_tangential_contact_handler();

    //! init rolling contact handler
    void init_rolling_contact_handler();

    //! setup particle interaction writer
    void setup_particle_interaction_writer();

    //! get maximum density of all materials
    double get_max_density_of_all_materials() const;

    //! evaluate particle contact contribution
    void evaluate_particle_contact();

    //! evaluate particle-wall contact contribution
    void evaluate_particle_wall_contact();

    //! evaluate particle elastic potential energy contribution
    void evaluate_particle_elastic_potential_energy(double& elasticpotentialenergy) const;

    //! evaluate particle-wall elastic potential energy contribution
    void evaluate_particle_wall_elastic_potential_energy(double& elasticpotentialenergy) const;

    //! discrete element method specific parameter list
    const Teuchos::ParameterList& params_dem_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! particle material handler
    std::shared_ptr<Particle::MaterialHandler> particlematerial_;

    //! particle interaction writer
    std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter_;

    //! neighbor pair handler
    std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs_;

    //! history pair handler
    std::shared_ptr<Particle::DEMHistoryPairs> historypairs_;

    //! normal contact handler
    std::unique_ptr<Particle::DEMContactNormalBase> contactnormal_;

    //! tangential contact handler
    std::unique_ptr<Particle::DEMContactTangentialBase> contacttangential_;

    //! rolling contact handler
    std::unique_ptr<Particle::DEMContactRollingBase> contactrolling_;

    //! time step size
    double dt_;

    //! tension cutoff of normal contact force
    const bool tension_cutoff_;

    //! write particle-wall interaction output
    const bool writeparticlewallinteraction_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
