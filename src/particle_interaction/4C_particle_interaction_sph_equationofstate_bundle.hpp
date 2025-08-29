// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_EQUATIONOFSTATE_BUNDLE_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_EQUATIONOFSTATE_BUNDLE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHEquationOfStateBase;
  class MaterialHandler;
}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHEquationOfStateBundle final
  {
   public:
    //! constructor
    explicit SPHEquationOfStateBundle(const Teuchos::ParameterList& params);

    //! init equation of state bundle
    void init(ParticleInteraction::MaterialHandler& particlematerial);

    //! setup equation of state bundle
    void setup();

    //! return pointer to specific equation of state
    inline const ParticleInteraction::SPHEquationOfStateBase* get_ptr_to_specific_equation_of_state(
        PARTICLEENGINE::TypeEnum type_i) const
    {
      return phasetypetoequationofstate_[type_i].get();
    };

   private:
    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! equation of state handler for all particle types
    std::vector<std::unique_ptr<ParticleInteraction::SPHEquationOfStateBase>>
        phasetypetoequationofstate_;

    //! set of particle types of stored equation of state handlers
    std::set<PARTICLEENGINE::TypeEnum> storedtypes_;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
