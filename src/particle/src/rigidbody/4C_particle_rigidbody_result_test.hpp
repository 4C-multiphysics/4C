// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_RIGIDBODY_RESULT_TEST_HPP
#define FOUR_C_PARTICLE_RIGIDBODY_RESULT_TEST_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_utils_result_test.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class RigidBodyHandlerInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief rigid body result test handler
   *
   */
  class RigidBodyResultTest final : public Core::Utils::ResultTest
  {
   public:
    //! constructor
    explicit RigidBodyResultTest();

    /*!
     * \brief setup rigid body result test
     *
     *
     * \param[in] particlerigidbodyinterface interface to rigid body handler
     */
    void setup(
        const std::shared_ptr<Particle::RigidBodyHandlerInterface> particlerigidbodyinterface);

    /*!
     * \brief test special quantity
     *
     * \param[in]  res        result parameter container
     * \param[out] nerr       number of tests with errors
     * \param[out] test_count number of tests performed
     */
    void test_special(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

   private:
    //! interface to rigid body handler
    std::shared_ptr<Particle::RigidBodyHandlerInterface> particlerigidbodyinterface_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
