// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_VISCOUS_DAMPING_HPP
#define FOUR_C_PARTICLE_ALGORITHM_VISCOUS_DAMPING_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"

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
   * \brief viscous damping handler for particle simulations
   *
   */
  class ViscousDampingHandler
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] viscdampfac viscous damping factor
     */
    explicit ViscousDampingHandler(const double viscdampfac);

    /*!
     * \brief setup viscous damping handler
     *
     *
     * \param[in] particleengineinterface interface to particle engine
     */
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface);

    /*!
     * \brief apply viscous damping contribution
     *
     */
    void apply_viscous_damping();

   private:
    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! viscous damping factor
    const double viscdampfac_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
