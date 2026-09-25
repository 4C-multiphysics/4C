// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PASI_PARTITIONED_ONEWAYCOUP_HPP
#define FOUR_C_PASI_PARTITIONED_ONEWAYCOUP_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_pasi_partitioned.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace PaSI
{
  /*!
   * \brief one way coupled partitioned algorithm
   *
   * One way coupled partitioned particle structure interaction algorithm with structure to particle
   * coupling of the interface states.
   *
   */
  class PasiPartOneWayCoup : public PartitionedAlgo
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle structure interaction parameter list
     */
    explicit PasiPartOneWayCoup(
        Global::Problem& problem, MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief setup pasi algorithm
     *
     */
    void setup() override;

    /*!
     * \brief partitioned one way coupled timeloop
     *
     */
    void timeloop() override;

   private:
    /*!
     * \brief output of fields
     *
     */
    void output() override;
  };

}  // namespace PaSI

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
