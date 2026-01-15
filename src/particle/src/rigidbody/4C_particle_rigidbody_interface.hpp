// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_RIGIDBODY_INTERFACE_HPP
#define FOUR_C_PARTICLE_RIGIDBODY_INTERFACE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class RigidBodyDataState;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief interface to provide restricted access to rigid body handler
   *
   */
  class RigidBodyHandlerInterface
  {
   public:
    //! virtual destructor
    virtual ~RigidBodyHandlerInterface() = default;

    /*!
     * \brief update positions with given time increment
     *
     *
     * \param[in] timeincrement time increment
     */
    virtual void update_positions(const double timeincrement) = 0;

    /*!
     * \brief update velocities with given time increment
     *
     *
     * \param[in] timeincrement time increment
     */
    virtual void update_velocities(const double timeincrement) = 0;

    /*!
     * \brief clear accelerations
     *
     */
    virtual void clear_accelerations() = 0;

    /*!
     * \brief get rigid body data state container
     *
     *
     * \return rigid body data state container
     */
    virtual std::shared_ptr<Particle::RigidBodyDataState> get_rigid_body_data_state() const = 0;

    /*!
     * \brief get owned rigid bodies by this processor
     *
     *
     * \return owned rigid bodies by this processor
     */
    virtual const std::vector<int>& get_owned_rigid_bodies() const = 0;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
