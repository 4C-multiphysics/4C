// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_OPEN_BOUNDARY_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_OPEN_BOUNDARY_HPP

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
  class SPHEquationOfStateBundle;
  class SPHKernelBase;
  class SPHNeighborPairs;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHOpenBoundaryBase
  {
   public:
    //! constructor
    explicit SPHOpenBoundaryBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHOpenBoundaryBase() = default;

    //! setup open boundary handler
    virtual void setup(
        const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs);

    //! prescribe open boundary states
    virtual void prescribe_open_boundary_states(const double& evaltime) = 0;

    //! interpolate open boundary states
    virtual void interpolate_open_boundary_states() = 0;

    //! check open boundary phase change
    virtual void check_open_boundary_phase_change(const double maxinteractiondistance) final;

   protected:
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

    //! equation of state bundle
    std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! states of ghosted particles to refresh
    Particle::StatesOfTypesToRefresh statestorefresh_;

    //! function id of prescribed state
    int prescribedstatefunctid_;

    //! outward normal
    std::vector<double> outwardnormal_;

    //! plane point
    std::vector<double> planepoint_;

    //! fluid phase
    Particle::TypeEnum fluidphase_;

    //! open boundary phase
    Particle::TypeEnum openboundaryphase_;
  };

  class SPHOpenBoundaryDirichlet : public SPHOpenBoundaryBase
  {
   public:
    //! constructor
    explicit SPHOpenBoundaryDirichlet(const Teuchos::ParameterList& params);

    //! setup open boundary handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs) override;

    //! prescribe open boundary states
    void prescribe_open_boundary_states(const double& evaltime) override;

    //! interpolate open boundary states
    void interpolate_open_boundary_states() override;
  };

  class SPHOpenBoundaryNeumann : public SPHOpenBoundaryBase
  {
   public:
    //! constructor
    explicit SPHOpenBoundaryNeumann(const Teuchos::ParameterList& params);

    //! setup open boundary handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::SPHKernelBase> kernel,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial,
        const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs) override;

    //! prescribe open boundary states
    void prescribe_open_boundary_states(const double& evaltime) override;

    //! interpolate open boundary states
    void interpolate_open_boundary_states() override;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
