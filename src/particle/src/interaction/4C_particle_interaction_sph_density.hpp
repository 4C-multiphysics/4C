// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_DENSITY_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_DENSITY_HPP

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
  class MaterialHandler;
  class ParticleContainerBundle;
  class ParticleEngineInterface;
  class SPHDensityCorrectionBase;
  class SPHEquationOfStateBundle;
  class SPHKernelBase;
  class SPHNeighborPairs;
  class SPHVirtualWallParticle;
  class WallHandlerInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHDensityBase
  {
   public:
    //! constructor
    explicit SPHDensityBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHDensityBase() = default;

    //! setup density handler
    virtual void setup(
        const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs,
        const std::shared_ptr<Particle::SPHVirtualWallParticle> virtualwallparticle);

    //! set current step size
    virtual void set_current_step_size(const double currentstepsize) final;

    //! insert density evaluation dependent states
    virtual void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes)
        const = 0;

    //! compute density field
    virtual void compute_density() const = 0;

   protected:
    //! evaluate sum of weighted mass
    virtual void sum_weighted_mass() const final;

    //! clear density sum state
    virtual void clear_density_sum_state() const final;

    //! sum weighted mass (self contribution)
    virtual void sum_weighted_mass_self_contribution() const final;

    //! sum weighted mass (particle contribution)
    virtual void sum_weighted_mass_particle_contribution() const final;

    //! sum weighted mass (particle-wall contribution)
    virtual void sum_weighted_mass_particle_wall_contribution() const final;

    //! evaluate sum of colorfield
    virtual void sum_colorfield() const final;

    //! clear colorfield state
    virtual void clear_colorfield_state() const final;

    //! sum colorfield (self contribution)
    virtual void sum_colorfield_self_contribution() const final;

    //! sum colorfield (particle contribution)
    virtual void sum_colorfield_particle_contribution() const final;

    //! sum colorfield (particle-wall contribution)
    virtual void sum_colorfield_particle_wall_contribution() const final;

    //! evaluate continuity equation
    virtual void continuity_equation() const final;

    //! clear density dot state
    virtual void clear_density_dot_state() const final;

    //! continuity equation (particle contribution)
    virtual void continuity_equation_particle_contribution() const final;

    //! continuity equation (particle-wall contribution)
    virtual void continuity_equation_particle_wall_contribution() const final;

    //! set density sum to density field
    virtual void set_density_sum() const final;

    //! add time step scaled density dot to density field
    virtual void add_time_step_scaled_density_dot() const final;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! kernel handler
    std::shared_ptr<Particle::SPHKernelBase> kernel_;

    //! particle material handler
    std::shared_ptr<Particle::MaterialHandler> particlematerial_;

    //! equation of state bundle
    std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! virtual wall particle handler
    std::shared_ptr<Particle::SPHVirtualWallParticle> virtualwallparticle_;

    //! density of ghosted particles to refresh
    Particle::StatesOfTypesToRefresh densitytorefresh_;

    //! set of fluid particle types
    std::set<Particle::TypeEnum> fluidtypes_;

    //! time step size
    double dt_;
  };

  class SPHDensitySummation final : public SPHDensityBase
  {
   public:
    //! constructor
    explicit SPHDensitySummation(const Teuchos::ParameterList& params);

    //! insert density evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes)
        const override;

    //! compute density field
    void compute_density() const override;
  };

  class SPHDensityIntegration final : public SPHDensityBase
  {
   public:
    //! constructor
    explicit SPHDensityIntegration(const Teuchos::ParameterList& params);

    //! insert density evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes)
        const override;

    //! compute density field
    void compute_density() const override;
  };

  class SPHDensityPredictCorrect final : public SPHDensityBase
  {
   public:
    //! constructor
    explicit SPHDensityPredictCorrect(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~SPHDensityPredictCorrect() override;

    //! setup density handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs,
        const std::shared_ptr<Particle::SPHVirtualWallParticle> virtualwallparticle) override;

    //! insert density evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes)
        const override;

    //! compute density field
    void compute_density() const override;

   private:
    //! init density correction handler
    void init_density_correction_handler();

    //! correct density of interior/surface particles
    void correct_density() const;

    //! density correction handler
    std::unique_ptr<Particle::SPHDensityCorrectionBase> densitycorrection_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
