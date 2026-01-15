// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_COMMUNICATION_UTILS_HPP
#define FOUR_C_PARTICLE_ENGINE_COMMUNICATION_UTILS_HPP

#include "4C_config.hpp"

#include <mpi.h>

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Particle::ParticleUtils
{
  //! @name methods for the communication of particles
  //@{

  /*!
   * \brief communicate data via non-buffered send from processor to processor
   *
   * Communicate data via non-buffered send from processor to processor via point-to-point
   * communication. Collective communication is required to inform the target processors about the
   * size of data to be received. The send buffer only relates data to be send to the target
   * processor.
   *
   * \note This method has to be called by all processors of the communicator as it contains
   * collective communication.
   *
   *
   * \param[in]  comm  communicator
   * \param[in]  sdata send buffers related to corresponding target processors
   * \param[out] rdata receive buffers related to corresponding source processors
   */
  void immediate_recv_blocking_send(MPI_Comm comm, std::map<int, std::vector<char>>& sdata,
      std::map<int, std::vector<char>>& rdata);

  //@}

}  // namespace Particle::ParticleUtils

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
