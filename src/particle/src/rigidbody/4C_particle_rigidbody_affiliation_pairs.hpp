// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_RIGIDBODY_AFFILIATION_PAIRS_HPP
#define FOUR_C_PARTICLE_RIGIDBODY_AFFILIATION_PAIRS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include <mpi.h>

#include <memory>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::IO
{
  class DiscretizationReader;
}

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
   * \brief affiliation pair handler for rigid bodies
   *
   * The affiliation pair handler relates the global ids of rigid particles to the corresponding
   * global ids of rigid bodies.
   *
   */
  class RigidBodyAffiliationPairs final
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     */
    explicit RigidBodyAffiliationPairs(MPI_Comm comm);

    /*!
     * \brief setup affiliation pair handler
     *
     */
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface);

    /*!
     * \brief write restart of affiliation pair handler
     *
     */
    void write_restart() const;

    /*!
     * \brief read restart of affiliation pair handler
     *
     *
     * \param[in] reader discretization reader
     */
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader);

    /*!
     * \brief get reference to affiliation pair data
     *
     *
     * \return reference to affiliation pair data
     */
    inline std::unordered_map<int, int>& get_ref_to_affiliation_pair_data()
    {
      return affiliationdata_;
    };

    /*!
     * \brief distribute affiliation pairs
     *
     */
    void distribute_affiliation_pairs();

    /*!
     * \brief communicate affiliation pairs
     *
     */
    void communicate_affiliation_pairs();

   private:
    /*!
     * \brief communicate specific affiliation pairs
     *
     */
    void communicate_specific_affiliation_pairs(
        const std::vector<std::vector<int>>& particletargets);

    /*!
     * \brief pack all affiliation pairs
     *
     *
     * \param[in] buffer buffer containing affiliation data
     */
    void pack_all_affiliation_pairs(std::vector<char>& buffer) const;

    /*!
     * \brief unpack affiliation pairs
     *
     * Unpack affiliation pairs relating rigid particles to rigid bodies.
     *
     *
     * \param[in] buffer buffer containing affiliation data
     */
    void unpack_affiliation_pairs(const std::vector<char>& buffer);

    /*!
     * \brief add affiliation pair to buffer
     *
     *
     * \param[in,out] buffer    buffer containing affiliation data
     * \param[in]     globalid  global id of rigid particle
     * \param[in]     rigidbody rigid body
     */
    void add_affiliation_pair_to_buffer(
        std::vector<char>& buffer, int globalid, int rigidbody) const;

    //! communicator
    MPI_Comm comm_;

    //! processor id
    const int myrank_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! affiliation pair data relating rigid particles to rigid bodies
    std::unordered_map<int, int> affiliationdata_;
  };
}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
