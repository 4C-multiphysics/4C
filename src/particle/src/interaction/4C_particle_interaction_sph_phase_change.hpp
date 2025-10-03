// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_PHASE_CHANGE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_PHASE_CHANGE_HPP

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
  class MaterialHandler;
  class SPHEquationOfStateBundle;
}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHPhaseChangeBase
  {
   public:
    //! constructor
    explicit SPHPhaseChangeBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHPhaseChangeBase() = default;

    //! init phase change handler
    virtual void init();

    //! setup phase change handler
    virtual void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
        const std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle);

    //! evaluate phase change
    virtual void evaluate_phase_change(
        std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase) const = 0;

   protected:
    //! evaluate phase change from below to above phase
    virtual void evaluate_phase_change_from_below_to_above_phase(
        std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase,
        std::vector<std::set<int>>& particlestoremove,
        std::vector<std::vector<std::pair<int, PARTICLEENGINE::ParticleObjShrdPtr>>>&
            particlestoinsert) const final;

    //! evaluate phase change from above to below phase
    virtual void evaluate_phase_change_from_above_to_below_phase(
        std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase,
        std::vector<std::set<int>>& particlestoremove,
        std::vector<std::vector<std::pair<int, PARTICLEENGINE::ParticleObjShrdPtr>>>&
            particlestoinsert) const final;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial_;

    //! equation of state bundle
    std::shared_ptr<ParticleInteraction::SPHEquationOfStateBundle> equationofstatebundle_;

    //! phase below transition value
    PARTICLEENGINE::TypeEnum belowphase_;

    //! phase above transition value
    PARTICLEENGINE::TypeEnum abovephase_;

    //! transition state of phase change
    PARTICLEENGINE::StateEnum transitionstate_;

    //! transition value of phase change
    double transitionvalue_;

    //! hysteresis gap at transition value
    double hysteresisgap_;
  };

  class SPHPhaseChangeOneWayScalarBelowToAbove : public SPHPhaseChangeBase
  {
   public:
    //! constructor
    explicit SPHPhaseChangeOneWayScalarBelowToAbove(const Teuchos::ParameterList& params);

    //! evaluate phase change
    void evaluate_phase_change(
        std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase) const override;
  };

  class SPHPhaseChangeOneWayScalarAboveToBelow : public SPHPhaseChangeBase
  {
   public:
    //! constructor
    explicit SPHPhaseChangeOneWayScalarAboveToBelow(const Teuchos::ParameterList& params);

    //! evaluate phase change
    void evaluate_phase_change(
        std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase) const override;
  };

  class SPHPhaseChangeTwoWayScalar : public SPHPhaseChangeBase
  {
   public:
    //! constructor
    explicit SPHPhaseChangeTwoWayScalar(const Teuchos::ParameterList& params);

    //! evaluate phase change
    void evaluate_phase_change(
        std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase) const override;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
