// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_RIGIDBODY_DATASTATE_HPP
#define FOUR_C_PARTICLE_RIGIDBODY_DATASTATE_HPP

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
  class ParticleEngineInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief rigid body data state container
   *
   */
  class RigidBodyDataState final
  {
   public:
    /*!
     * \brief allocate stored states
     *
     *
     * \param[in] numrigidbodies number of rigid bodies
     */
    void allocate_stored_states(const int numrigidbodies);

    //! @name get states (read only access)
    //! @{

    //! get mass of rigid bodies
    inline const std::vector<double>& get_ref_mass() const { return mass_; };

    //! get mass moment of inertia of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_inertia() const { return inertia_; };

    //! get position of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_position() const { return position_; };

    //! get velocity of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_velocity() const { return velocity_; };

    //! get acceleration of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_acceleration() const
    {
      return acceleration_;
    };

    //! get rotation of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_rotation() const { return rotation_; };

    //! get angular velocity of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_angular_velocity() const
    {
      return angularvelocity_;
    };

    //! get angular acceleration of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_angular_acceleration() const
    {
      return angularacceleration_;
    };

    //! get force of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_force() const { return force_; };

    //! get torque of rigid bodies
    inline const std::vector<std::vector<double>>& get_ref_torque() const { return torque_; };

    //! @}

    //! @name get states (read and write access)
    //! @{

    //! get mass of rigid bodies
    inline std::vector<double>& get_ref_mass() { return mass_; };

    //! get mass moment of inertia of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_inertia() { return inertia_; };

    //! get position of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_position() { return position_; };

    //! get velocity of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_velocity() { return velocity_; };

    //! get acceleration of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_acceleration() { return acceleration_; };

    //! get rotation of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_rotation() { return rotation_; };

    //! get angular velocity of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_angular_velocity()
    {
      return angularvelocity_;
    };

    //! get angular acceleration of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_angular_acceleration()
    {
      return angularacceleration_;
    };

    //! get force of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_force() { return force_; };

    //! get torque of rigid bodies
    inline std::vector<std::vector<double>>& get_ref_torque() { return torque_; };

    //! @}

   private:
    //! @name stored states
    //! @{

    //! mass of rigid bodies
    std::vector<double> mass_;

    //! mass moment of inertia of rigid bodies
    std::vector<std::vector<double>> inertia_;

    //! position of rigid bodies
    std::vector<std::vector<double>> position_;

    //! velocity of rigid bodies
    std::vector<std::vector<double>> velocity_;

    //! acceleration of rigid bodies
    std::vector<std::vector<double>> acceleration_;

    //! rotation of rigid bodies
    std::vector<std::vector<double>> rotation_;

    //! angular velocity of rigid bodies
    std::vector<std::vector<double>> angularvelocity_;

    //! angular acceleration of rigid bodies
    std::vector<std::vector<double>> angularacceleration_;

    //! force of rigid bodies
    std::vector<std::vector<double>> force_;

    //! torque of rigid bodies
    std::vector<std::vector<double>> torque_;

    //! @}
  };
}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
