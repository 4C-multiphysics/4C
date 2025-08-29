// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_history_pairs.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_pack_buffer.hpp"
#include "4C_io.hpp"
#include "4C_particle_engine_communication_utils.hpp"
#include "4C_particle_engine_interface.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMHistoryPairs::DEMHistoryPairs(MPI_Comm comm) : comm_(comm)
{
  // empty constructor
}

void ParticleInteraction::DEMHistoryPairs::init()
{
  // nothing to do
}

void ParticleInteraction::DEMHistoryPairs::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;
}

void ParticleInteraction::DEMHistoryPairs::write_restart() const
{
  // get bin discretization writer
  std::shared_ptr<Core::IO::DiscretizationWriter> binwriter =
      particleengineinterface_->get_bin_discretization_writer();

  // prepare buffer
  std::shared_ptr<std::vector<char>> buffer;

  // particle tangential history data
  {
    buffer = std::make_shared<std::vector<char>>();

    if (not particletangentialhistorydata_.empty())
      pack_all_history_pairs(*buffer, particletangentialhistorydata_);

    binwriter->write_char_data("ParticleTangentialHistoryData", *buffer);
  }

  // particle-wall tangential history pair data
  {
    buffer = std::make_shared<std::vector<char>>();

    if (not particlewalltangentialhistorydata_.empty())
      pack_all_history_pairs(*buffer, particlewalltangentialhistorydata_);

    binwriter->write_char_data("ParticleWallTangentialHistoryData", *buffer);
  }

  // particle rolling history data
  {
    buffer = std::make_shared<std::vector<char>>();

    if (not particlerollinghistorydata_.empty())
      pack_all_history_pairs(*buffer, particlerollinghistorydata_);

    binwriter->write_char_data("ParticleRollingHistoryData", *buffer);
  }

  // particle-wall rolling history pair data
  {
    buffer = std::make_shared<std::vector<char>>();

    if (not particlewallrollinghistorydata_.empty())
      pack_all_history_pairs(*buffer, particlewallrollinghistorydata_);

    binwriter->write_char_data("ParticleWallRollingHistoryData", *buffer);
  }

  // particle adhesion history data
  {
    buffer = std::make_shared<std::vector<char>>();

    if (not particleadhesionhistorydata_.empty())
      pack_all_history_pairs(*buffer, particleadhesionhistorydata_);

    binwriter->write_char_data("ParticleAdhesionHistoryData", *buffer);
  }

  // particle-wall adhesion history pair data
  {
    buffer = std::make_shared<std::vector<char>>();

    if (not particlewalladhesionhistorydata_.empty())
      pack_all_history_pairs(*buffer, particlewalladhesionhistorydata_);

    binwriter->write_char_data("ParticleWallAdhesionHistoryData", *buffer);
  }
}

void ParticleInteraction::DEMHistoryPairs::read_restart(Core::IO::DiscretizationReader& reader)
{
  // prepare buffer
  std::shared_ptr<std::vector<char>> buffer;

  // particle tangential history data
  {
    buffer = std::make_shared<std::vector<char>>();

    reader.read_char_vector(buffer, "ParticleTangentialHistoryData");

    if (buffer->size() > 0) unpack_history_pairs(*buffer, particletangentialhistorydata_);
  }

  // particle-wall tangential history pair data
  {
    buffer = std::make_shared<std::vector<char>>();

    reader.read_char_vector(buffer, "ParticleWallTangentialHistoryData");

    if (buffer->size() > 0) unpack_history_pairs(*buffer, particlewalltangentialhistorydata_);
  }

  // particle rolling history data
  {
    buffer = std::make_shared<std::vector<char>>();

    reader.read_char_vector(buffer, "ParticleRollingHistoryData");

    if (buffer->size() > 0) unpack_history_pairs(*buffer, particlerollinghistorydata_);
  }

  // particle-wall rolling history pair data
  {
    buffer = std::make_shared<std::vector<char>>();

    reader.read_char_vector(buffer, "ParticleWallRollingHistoryData");

    if (buffer->size() > 0) unpack_history_pairs(*buffer, particlewallrollinghistorydata_);
  }

  // particle adhesion history data
  {
    buffer = std::make_shared<std::vector<char>>();

    reader.read_char_vector(buffer, "ParticleAdhesionHistoryData");

    if (buffer->size() > 0) unpack_history_pairs(*buffer, particleadhesionhistorydata_);
  }

  // particle-wall adhesion history pair data
  {
    buffer = std::make_shared<std::vector<char>>();

    reader.read_char_vector(buffer, "ParticleWallAdhesionHistoryData");

    if (buffer->size() > 0) unpack_history_pairs(*buffer, particlewalladhesionhistorydata_);
  }
}

void ParticleInteraction::DEMHistoryPairs::distribute_history_pairs()
{
  // relate all particles to all processors
  std::vector<int> particlestoproc;
  particleengineinterface_->relate_all_particles_to_all_procs(particlestoproc);

  // allocate memory
  std::vector<std::vector<int>> particletargets(Core::Communication::num_mpi_ranks(comm_));

  // iterate over all particle global ids
  for (int gid = 0; gid < static_cast<int>(particlestoproc.size()); ++gid)
  {
    // processor id of current particle
    const int currproc = particlestoproc[gid];

    // no need to send history pairs
    if (currproc == Core::Communication::my_mpi_rank(comm_)) continue;

    // no particle with current global id in simulation
    if (currproc < 0) continue;

    // append current particle global id
    particletargets[currproc].push_back(gid);
  }

  // communicate specific history pairs
  communicate_specific_history_pairs(particletargets, particletangentialhistorydata_);
  communicate_specific_history_pairs(particletargets, particlewalltangentialhistorydata_);

  communicate_specific_history_pairs(particletargets, particlerollinghistorydata_);
  communicate_specific_history_pairs(particletargets, particlewallrollinghistorydata_);

  communicate_specific_history_pairs(particletargets, particleadhesionhistorydata_);
  communicate_specific_history_pairs(particletargets, particlewalladhesionhistorydata_);
}

void ParticleInteraction::DEMHistoryPairs::communicate_history_pairs()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMHistoryPairs::communicate_history_pairs");

  // get reference to particles being communicated to target processors
  const std::vector<std::vector<int>>& particletargets =
      particleengineinterface_->get_communicated_particle_targets();

  // communicate specific history pairs
  communicate_specific_history_pairs(particletargets, particletangentialhistorydata_);
  communicate_specific_history_pairs(particletargets, particlewalltangentialhistorydata_);

  communicate_specific_history_pairs(particletargets, particlerollinghistorydata_);
  communicate_specific_history_pairs(particletargets, particlewallrollinghistorydata_);

  communicate_specific_history_pairs(particletargets, particleadhesionhistorydata_);
  communicate_specific_history_pairs(particletargets, particlewalladhesionhistorydata_);
}

void ParticleInteraction::DEMHistoryPairs::update_history_pairs()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMHistoryPairs::UpdateHistoryPairs");

  // erase untouched history pairs
  if (not particletangentialhistorydata_.empty())
    erase_untouched_history_pairs(particletangentialhistorydata_);

  if (not particlewalltangentialhistorydata_.empty())
    erase_untouched_history_pairs(particlewalltangentialhistorydata_);

  if (not particlerollinghistorydata_.empty())
    erase_untouched_history_pairs(particlerollinghistorydata_);

  if (not particlewallrollinghistorydata_.empty())
    erase_untouched_history_pairs(particlewallrollinghistorydata_);

  if (not particleadhesionhistorydata_.empty())
    erase_untouched_history_pairs(particleadhesionhistorydata_);

  if (not particlewalladhesionhistorydata_.empty())
    erase_untouched_history_pairs(particlewalladhesionhistorydata_);
}

template <typename Historypairtype>
void ParticleInteraction::DEMHistoryPairs::communicate_specific_history_pairs(
    const std::vector<std::vector<int>>& particletargets,
    std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>& historydata)
{
  // prepare buffer for sending and receiving
  std::map<int, std::vector<char>> sdata;
  std::map<int, std::vector<char>> rdata;

  // pack history pairs
  for (int torank = 0; torank < Core::Communication::num_mpi_ranks(comm_); ++torank)
  {
    if (particletargets[torank].empty()) continue;

    for (int globalid : particletargets[torank])
    {
      // no history pairs for current global id
      if (not historydata.count(globalid)) continue;

      for (auto& it_j : historydata[globalid])
      {
        // get reference to history pair
        const Historypairtype& historypair = (it_j.second).second;

        // add history pair to buffer
        add_history_pair_to_buffer(sdata[torank], globalid, it_j.first, historypair);
      }
    }
  }

  // communicate data via non-buffered send from proc to proc
  PARTICLEENGINE::COMMUNICATION::immediate_recv_blocking_send(comm_, sdata, rdata);

  // unpack history pairs
  for (auto& p : rdata) unpack_history_pairs(p.second, historydata);
}

template <typename Historypairtype>
void ParticleInteraction::DEMHistoryPairs::erase_untouched_history_pairs(
    std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>& historydata)
{
  // iterate over nested unordered maps of stored history pairs
  for (auto it_i = historydata.begin(); it_i != historydata.end();)
  {
    for (auto it_j = (it_i->second).begin(); it_j != (it_i->second).end();)
    {
      // remove untouched history pair
      if ((it_j->second).first == false) (it_i->second).erase(it_j++);
      // invalidate touched history pair
      else
        ((it_j++)->second).first = false;
    }

    // no history pairs left: erase entry
    if ((it_i->second).empty()) historydata.erase(it_i++);
    // increment iterator
    else
      ++it_i;
  }
}

template <typename Historypairtype>
void ParticleInteraction::DEMHistoryPairs::pack_all_history_pairs(std::vector<char>& buffer,
    const std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>&
        historydata) const
{
  // iterate over nested unordered maps of stored history pairs
  for (auto& it_i : historydata)
  {
    for (auto& it_j : it_i.second)
    {
      // get reference to history pair
      const Historypairtype& historypair = (it_j.second).second;

      // add history pair to buffer
      add_history_pair_to_buffer(buffer, it_i.first, it_j.first, historypair);
    }
  }
}

template <typename Historypairtype>
void ParticleInteraction::DEMHistoryPairs::unpack_history_pairs(const std::vector<char>& buffer,
    std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>& historydata)
{
  Core::Communication::UnpackBuffer data(buffer);
  while (!data.at_end())
  {
    // get global ids
    int globalid_i;
    extract_from_pack(data, globalid_i);
    int globalid_j;
    extract_from_pack(data, globalid_j);

    // unpack history pair data
    Historypairtype historypair = Historypairtype();
    historypair.unpack(data);

    // add history pair data
    historydata[globalid_i][globalid_j] = std::make_pair(true, historypair);
  }
}

template <typename Historypairtype>
void ParticleInteraction::DEMHistoryPairs::add_history_pair_to_buffer(std::vector<char>& buffer,
    int globalid_i, int globalid_j, const Historypairtype& historypair) const
{
  Core::Communication::PackBuffer data;
  // add global ids
  data.add_to_pack(globalid_i);
  data.add_to_pack(globalid_j);

  // pack history pair data
  historypair.pack(data);

  // append packed history pair to buffer
  buffer.insert(buffer.end(), data().begin(), data().end());
}

/*---------------------------------------------------------------------------*
 | template instantiations                                                   |
 *---------------------------------------------------------------------------*/
template void ParticleInteraction::DEMHistoryPairs::communicate_specific_history_pairs<
    ParticleInteraction::DEMHistoryPairTangential>(
    const std::vector<std::vector<int>>&, DEMHistoryPairTangentialData&);

template void ParticleInteraction::DEMHistoryPairs::communicate_specific_history_pairs<
    ParticleInteraction::DEMHistoryPairRolling>(
    const std::vector<std::vector<int>>&, DEMHistoryPairRollingData&);

template void ParticleInteraction::DEMHistoryPairs::communicate_specific_history_pairs<
    ParticleInteraction::DEMHistoryPairAdhesion>(
    const std::vector<std::vector<int>>&, DEMHistoryPairAdhesionData&);

template void ParticleInteraction::DEMHistoryPairs::erase_untouched_history_pairs<
    ParticleInteraction::DEMHistoryPairTangential>(DEMHistoryPairTangentialData&);

template void ParticleInteraction::DEMHistoryPairs::erase_untouched_history_pairs<
    ParticleInteraction::DEMHistoryPairRolling>(DEMHistoryPairRollingData&);

template void ParticleInteraction::DEMHistoryPairs::erase_untouched_history_pairs<
    ParticleInteraction::DEMHistoryPairAdhesion>(DEMHistoryPairAdhesionData&);

template void ParticleInteraction::DEMHistoryPairs::pack_all_history_pairs<
    ParticleInteraction::DEMHistoryPairTangential>(
    std::vector<char>&, const DEMHistoryPairTangentialData&) const;

template void ParticleInteraction::DEMHistoryPairs::pack_all_history_pairs<
    ParticleInteraction::DEMHistoryPairRolling>(
    std::vector<char>&, const DEMHistoryPairRollingData&) const;

template void ParticleInteraction::DEMHistoryPairs::pack_all_history_pairs<
    ParticleInteraction::DEMHistoryPairAdhesion>(
    std::vector<char>&, const DEMHistoryPairAdhesionData&) const;

template void ParticleInteraction::DEMHistoryPairs::unpack_history_pairs<
    ParticleInteraction::DEMHistoryPairTangential>(
    const std::vector<char>&, DEMHistoryPairTangentialData&);

template void ParticleInteraction::DEMHistoryPairs::unpack_history_pairs<
    ParticleInteraction::DEMHistoryPairRolling>(
    const std::vector<char>&, DEMHistoryPairRollingData&);

template void ParticleInteraction::DEMHistoryPairs::unpack_history_pairs<
    ParticleInteraction::DEMHistoryPairAdhesion>(
    const std::vector<char>&, DEMHistoryPairAdhesionData&);

template void ParticleInteraction::DEMHistoryPairs::add_history_pair_to_buffer<
    ParticleInteraction::DEMHistoryPairTangential>(
    std::vector<char>&, int, int, const ParticleInteraction::DEMHistoryPairTangential&) const;

template void ParticleInteraction::DEMHistoryPairs::add_history_pair_to_buffer<
    ParticleInteraction::DEMHistoryPairRolling>(
    std::vector<char>&, int, int, const ParticleInteraction::DEMHistoryPairRolling&) const;

template void ParticleInteraction::DEMHistoryPairs::add_history_pair_to_buffer<
    ParticleInteraction::DEMHistoryPairAdhesion>(
    std::vector<char>&, int, int, const ParticleInteraction::DEMHistoryPairAdhesion&) const;

FOUR_C_NAMESPACE_CLOSE
