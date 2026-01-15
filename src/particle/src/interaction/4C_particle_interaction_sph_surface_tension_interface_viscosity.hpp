// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_SURFACE_TENSION_INTERFACE_VISCOSITY_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_SURFACE_TENSION_INTERFACE_VISCOSITY_HPP

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
  class SPHArtificialViscosity;
  class SPHEquationOfStateBundle;
  class SPHKernelBase;
  class SPHNeighborPairs;
}  // namespace Particle

namespace Mat
{
  namespace PAR
  {
    class ParticleMaterialSPHFluid;
  }
}  // namespace Mat

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHInterfaceViscosity
  {
   public:
    //! constructor
    explicit SPHInterfaceViscosity(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~SPHInterfaceViscosity();

    //! setup interface viscosity handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        Particle::MaterialHandler& particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs);

    //! compute interface viscosity contribution
    void compute_interface_viscosity_contribution() const;

   private:
    //! init artificial viscosity handler
    void init_artificial_viscosity_handler();

    //! compute interface viscosity contribution (particle contribution)
    void compute_interface_viscosity_particle_contribution() const;

    //! compute interface viscosity contribution (particle-boundary contribution)
    void compute_interface_viscosity_particle_boundary_contribution() const;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! kernel handler
    std::shared_ptr<Particle::SPHKernelBase> kernel_;

    //! equation of state bundle
    std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! artificial viscosity handler
    std::unique_ptr<Particle::SPHArtificialViscosity> artificialviscosity_;

    //! pointer to fluid material of particle types
    std::vector<const Mat::PAR::ParticleMaterialSPHFluid*> fluidmaterial_;

    //! liquid particle type
    Particle::TypeEnum liquidtype_;

    //! gas particle type
    Particle::TypeEnum gastype_;

    //! set of fluid particle types
    std::set<Particle::TypeEnum> fluidtypes_;

    //! set of boundary particle types
    std::set<Particle::TypeEnum> boundarytypes_;

    //! artificial viscosity on liquid-gas interface
    const double artvisc_lg_int_;

    //! artificial viscosity on solid-liquid interface
    const double artvisc_sl_int_;

    //! transition reference temperature
    const double trans_ref_temp_;

    //! transition temperature difference for interface viscosity evaluation
    const double trans_d_t_intvisc_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
