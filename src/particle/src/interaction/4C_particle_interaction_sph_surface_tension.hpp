// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_SURFACE_TENSION_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_SURFACE_TENSION_HPP

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
  class SPHBarrierForce;
  class SPHEquationOfStateBundle;
  class SPHInterfaceViscosity;
  class SPHKernelBase;
  class SPHNeighborPairs;
  class SPHRecoilPressureEvaporation;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHSurfaceTension
  {
   public:
    //! constructor
    explicit SPHSurfaceTension(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~SPHSurfaceTension();

    //! setup surface tension handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs);

    //! set current time
    void set_current_time(const double currenttime);

    //! insert surface tension evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const;

    //! compute interface quantities
    void compute_interface_quantities();

    //! add surface tension contribution to acceleration field
    void add_acceleration_contribution();

   private:
    //! compute colorfield gradient
    void compute_colorfield_gradient() const;

    //! compute interface normal
    void compute_interface_normal() const;

    //! compute wall colorfield and wall interface normal
    void compute_wall_colorfield_and_wall_interface_normal() const;

    //! correct normal vector of particles close to triple point
    void correct_triple_point_normal() const;

    //! compute curvature
    void compute_curvature() const;

    //! compute surface tension contribution
    void compute_surface_tension_contribution() const;

    //! compute temperature gradient driven contribution
    void compute_temp_grad_driven_contribution() const;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! kernel handler
    std::shared_ptr<Particle::SPHKernelBase> kernel_;

    //! particle material handler
    std::shared_ptr<Particle::MaterialHandler> particlematerial_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! interface viscosity handler
    std::unique_ptr<Particle::SPHInterfaceViscosity> interfaceviscosity_;

    //! evaporation induced recoil pressure handler
    std::unique_ptr<Particle::SPHRecoilPressureEvaporation> recoilpressureevaporation_;

    //! barrier force handler
    std::unique_ptr<Particle::SPHBarrierForce> barrierforce_;

    //! liquid particle type
    Particle::TypeEnum liquidtype_;

    //! gas particle type
    Particle::TypeEnum gastype_;

    //! set of fluid particle types
    std::set<Particle::TypeEnum> fluidtypes_;

    //! set of boundary particle types
    std::set<Particle::TypeEnum> boundarytypes_;

    //! interface normal of ghosted particles to refresh
    Particle::StatesOfTypesToRefresh intnormtorefresh_;

    //! current time
    double time_;

    //! surface tension time ramp function
    const int timerampfct_;

    //! constant part of surface tension coefficient
    const double alpha0_;

    //! minimum surface tension coefficient in case of temperature dependence
    const double alphamin_;

    //! factor of dependence of surface tension coefficient on temperature
    const double alpha_t_;

    //! surface tension coefficient reference temperature
    const double surf_ref_temp_;

    //! static contact angle
    const double staticcontactangle_;

    //! triple point normal correction wall color field low
    const double tpn_corr_cf_low_;

    //! triple point normal correction wall color field up
    const double tpn_corr_cf_up_;

    //! transition reference temperature
    const double trans_ref_temp_;

    //! transition temperature difference for surface tension evaluation
    const double trans_d_t_surf_;

    //! transition temperature difference for marangoni evaluation
    const double trans_d_t_mara_;

    //! transition temperature difference for curvature evaluation
    const double trans_d_t_curv_;

    //! transition temperature difference for wetting evaluation
    const double trans_d_t_wet_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
