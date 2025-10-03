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
namespace PARTICLEENGINE
{
  class ParticleEngineInterface;
  class ParticleContainerBundle;
}  // namespace PARTICLEENGINE

namespace ParticleInteraction
{
  class SPHKernelBase;
  class MaterialHandler;
  class SPHEquationOfStateBundle;
  class SPHNeighborPairs;
}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHOpenBoundaryBase
  {
   public:
    //! constructor
    explicit SPHOpenBoundaryBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHOpenBoundaryBase() = default;

    //! init open boundary handler
    virtual void init();

    //! setup open boundary handler
    virtual void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel,
        const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
        const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs);

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
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! kernel handler
    std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel_;

    //! particle material handler
    std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial_;

    //! equation of state bundle
    std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle_;

    //! neighbor pair handler
    std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs_;

    //! states of ghosted particles to refresh
    PARTICLEENGINE::StatesOfTypesToRefresh statestorefresh_;

    //! function id of prescribed state
    int prescribedstatefunctid_;

    //! outward normal
    std::vector<double> outwardnormal_;

    //! plane point
    std::vector<double> planepoint_;

    //! fluid phase
    PARTICLEENGINE::TypeEnum fluidphase_;

    //! open boundary phase
    PARTICLEENGINE::TypeEnum openboundaryphase_;
  };

  class SPHOpenBoundaryDirichlet : public SPHOpenBoundaryBase
  {
   public:
    //! constructor
    explicit SPHOpenBoundaryDirichlet(const Teuchos::ParameterList& params);

    //! init open boundary handler
    void init() override;

    //! setup open boundary handler
    void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel,
        const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
        const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs) override;

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

    //! init open boundary handler
    void init() override;

    //! setup open boundary handler
    void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<ParticleInteraction::SPHKernelBase> kernel,
        const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
        const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<ParticleInteraction::SPHNeighborPairs> neighborpairs) override;

    //! prescribe open boundary states
    void prescribe_open_boundary_states(const double& evaltime) override;

    //! interpolate open boundary states
    void interpolate_open_boundary_states() override;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
