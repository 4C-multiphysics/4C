// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_INITIAL_FIELD_HPP
#define FOUR_C_PARTICLE_ALGORITHM_INITIAL_FIELD_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleEngineInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief initial field handler for particle simulations
   *
   */
  class InitialFieldHandler
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] params particle simulation parameter list
     */
    explicit InitialFieldHandler(const Teuchos::ParameterList& params);

    /*!
     * \brief setup initial field handler
     *
     *
     * \param[in] particleengineinterface interface to particle engine
     */
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface);

    /*!
     * \brief set initial fields
     *
     */
    void set_initial_fields();

   protected:
    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! relating particle types to function ids
    std::map<Particle::StateEnum, std::map<Particle::TypeEnum, int>> statetotypetofunctidmap_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
