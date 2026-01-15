// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_WALL_RESULT_TEST_HPP
#define FOUR_C_PARTICLE_WALL_RESULT_TEST_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_result_test.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class WallHandlerInterface;
}  // namespace Particle

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief particle wall result test handler
   *
   */
  class WallResultTest final : public Core::Utils::ResultTest
  {
   public:
    //! constructor
    explicit WallResultTest();

    /*!
     * \brief setup wall result test
     *
     *
     * \param[in] particleengineinterface interface to particle engine
     */
    void setup(const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface);

    /*!
     * \brief test node value
     *
     * \param[in]  res        result parameter container
     * \param[out] nerr       number of tests with errors
     * \param[out] test_count number of tests performed
     */
    void test_node(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

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
    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! wall discretization
    std::shared_ptr<const Core::FE::Discretization> walldiscretization_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
