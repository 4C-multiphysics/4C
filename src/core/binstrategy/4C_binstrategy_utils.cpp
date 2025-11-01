// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_binstrategy_utils.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_pstream.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_rebalance_print.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Binstrategy::Utils
{
  /*-----------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------*/
  void extend_discretization_ghosting(Core::FE::Discretization& discret,
      Core::LinAlg::Map& extendedelecolmap, bool assigndegreesoffreedom, bool initelements,
      bool doboundaryconditions)
  {
    // make sure that all procs are either filled or unfilled
    // oldmap in export_column_elements must be reset() on every proc or nowhere
    discret.check_filled_globally();

    // adapt layout to extended ghosting in discret
    // first export the elements according to the processor local element column maps
    discret.export_column_elements(extendedelecolmap);

    // get the node ids of the elements that are to be ghosted
    // and create a proper node column map for their export
    std::set<int> nodes;
    for (int lid = 0; lid < extendedelecolmap.num_my_elements(); ++lid)
    {
      Core::Elements::Element* ele = discret.g_element(extendedelecolmap.gid(lid));
      const int* nodeids = ele->node_ids();
      for (int inode = 0; inode < ele->num_node(); ++inode) nodes.insert(nodeids[inode]);
    }

    std::vector<int> colnodes(nodes.begin(), nodes.end());
    Core::LinAlg::Map nodecolmap(-1, (int)colnodes.size(), colnodes.data(), 0, discret.get_comm());

    // now ghost the nodes
    discret.export_column_nodes(nodecolmap);

    // fillcomplete discret with extended ghosting
    discret.fill_complete({
        .assign_degrees_of_freedom = assigndegreesoffreedom,
        .init_elements = initelements,
        .do_boundary_conditions = doboundaryconditions,
    });

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // print distribution after standard ghosting
    if (Core::Communication::my_mpi_rank(discret.get_comm()) == 0)
      std::cout << "parallel distribution with extended ghosting" << std::endl;
    Core::Rebalance::print_parallel_distribution(discret);
#endif

    return;
  }

  /*-----------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------*/
  void communicate_elements(Core::FE::Discretization& discret,
      std::map<int, std::vector<Core::Elements::Element*>> const& toranktosendeles)
  {
    // build exporter
    Core::Communication::Exporter exporter(discret.get_comm());
    int const numproc = Core::Communication::num_mpi_ranks(discret.get_comm());

    // -----------------------------------------------------------------------
    // send
    // -----------------------------------------------------------------------
    // ---- pack data for sending -----
    std::map<int, std::vector<char>> sdata;
    std::vector<int> targetprocs(numproc, 0);
    std::map<int, std::vector<Core::Elements::Element*>>::const_iterator p;
    for (p = toranktosendeles.begin(); p != toranktosendeles.end(); ++p)
    {
      std::vector<Core::Elements::Element*>::const_iterator iter;
      for (iter = p->second.begin(); iter != p->second.end(); ++iter)
      {
        Core::Communication::PackBuffer data;
        (*iter)->pack(data);
        sdata[p->first].insert(sdata[p->first].end(), data().begin(), data().end());
      }
      targetprocs[p->first] = 1;
    }

    // ---- send ----
    const int length = sdata.size();
    std::vector<MPI_Request> request(length);
    int tag = 0;
    for (std::map<int, std::vector<char>>::const_iterator p = sdata.begin(); p != sdata.end(); ++p)
    {
      exporter.i_send(Core::Communication::my_mpi_rank(discret.get_comm()), p->first,
          (p->second).data(), (int)(p->second).size(), 1234, request[tag]);
      ++tag;
    }
    if (tag != length) FOUR_C_THROW("Number of messages is mixed up");

    // -----------------------------------------------------------------------
    // receive
    // -----------------------------------------------------------------------
    // ---- prepare receiving procs -----
    std::vector<int> summedtargets(numproc, 0);
    summedtargets = Core::Communication::sum_all(targetprocs, discret.get_comm());

    // ---- receive ----
    for (int rec = 0; rec < summedtargets[Core::Communication::my_mpi_rank(discret.get_comm())];
        ++rec)
    {
      std::vector<char> rdata;
      int length = 0;
      int tag = -1;
      int from = -1;
      exporter.receive_any(from, tag, rdata, length);
      if (tag != 1234)
        FOUR_C_THROW("Received on proc {} data with wrong tag from proc {}",
            Core::Communication::my_mpi_rank(discret.get_comm()), from);

      // ---- unpack ----
      {
        // Put received nodes into discretization
        Communication::UnpackBuffer buffer(rdata);
        while (!buffer.at_end())
        {
          auto object =
              std::shared_ptr<Core::Communication::ParObject>(Core::Communication::factory(buffer));
          std::shared_ptr<Core::Elements::Element> element =
              std::dynamic_pointer_cast<Core::Elements::Element>(object);
          if (element == nullptr) FOUR_C_THROW("Received object is not a element");

          // safety check
          if (discret.have_global_element(element->id()) != true)
            FOUR_C_THROW(
                "{} is getting owner of element {} without having it ghosted before, "
                "this is not intended.",
                Core::Communication::my_mpi_rank(discret.get_comm()), element->id());

          // delete already existing element (as it has wrong internal variables)
          discret.delete_element(element->id());
          // add node (ownership already adapted on sending proc)
          discret.add_element(element);
        }
      }
    }

    // wait for all communications to finish
    for (int i = 0; i < length; ++i) exporter.wait(request[i]);
    // safety, should be a no time operation if everything works fine before
    Core::Communication::barrier(discret.get_comm());
  }

  /*-----------------------------------------------------------------------------*
   *-----------------------------------------------------------------------------*/
  void communicate_distribution_of_transferred_elements_to_bins(Core::FE::Discretization& discret,
      std::map<int, std::vector<std::pair<int, std::vector<int>>>> const& toranktosendbinids,
      std::map<int, std::set<int>>& bintorowelemap)
  {
    // build exporter
    Core::Communication::Exporter exporter(discret.get_comm());
    int const numproc = Core::Communication::num_mpi_ranks(discret.get_comm());

    // -----------------------------------------------------------------------
    // send
    // -----------------------------------------------------------------------
    // ---- pack data for sending -----
    std::map<int, std::vector<char>> sdata;
    std::vector<int> targetprocs(numproc, 0);

    for (const auto& [rank, bin_list] : toranktosendbinids)
    {
      Core::Communication::PackBuffer data;
      for (const auto& bin : bin_list)
      {
        add_to_pack(data, bin);
      }
      auto& buffer = sdata[rank];
      buffer.insert(buffer.end(), data().begin(), data().end());
      targetprocs[rank] = 1;
    }

    // ---- send ----
    const int length = sdata.size();
    std::vector<MPI_Request> request(length);
    int tag = 0;
    for (std::map<int, std::vector<char>>::const_iterator p = sdata.begin(); p != sdata.end(); ++p)
    {
      exporter.i_send(Core::Communication::my_mpi_rank(discret.get_comm()), p->first,
          (p->second).data(), (int)(p->second).size(), 1234, request[tag]);
      ++tag;
    }
    if (tag != length) FOUR_C_THROW("Number of messages is mixed up");

    // -----------------------------------------------------------------------
    // receive
    // -----------------------------------------------------------------------
    // ---- prepare receiving procs -----
    std::vector<int> summedtargets(numproc, 0);
    summedtargets = Core::Communication::sum_all(targetprocs, discret.get_comm());

    // ---- receive ----
    for (int rec = 0; rec < summedtargets[Core::Communication::my_mpi_rank(discret.get_comm())];
        ++rec)
    {
      std::vector<char> rdata;
      int length = 0;
      int tag = -1;
      int from = -1;
      exporter.receive_any(from, tag, rdata, length);
      if (tag != 1234)
        FOUR_C_THROW("Received on proc {} data with wrong tag from proc {}",
            Core::Communication::my_mpi_rank(discret.get_comm()), from);

      // ---- unpack ----
      {
        // Put received nodes into discretization
        Communication::UnpackBuffer buffer(rdata);
        while (!buffer.at_end())
        {
          std::pair<int, std::vector<int>> pair;
          extract_from_pack(buffer, pair);
          std::vector<int>::const_iterator j;
          for (j = pair.second.begin(); j != pair.second.end(); ++j)
            bintorowelemap[*j].insert(pair.first);
        }
      }
    }

    // wait for all communications to finish
    for (int i = 0; i < length; ++i) exporter.wait(request[i]);
    // safety, should be a no time operation if everything works fine before
    Core::Communication::barrier(discret.get_comm());
  }

  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void get_current_node_pos(const Core::FE::Discretization& discret, Core::Nodes::Node const* node,
      std::shared_ptr<const Core::LinAlg::Vector<double>> const disnp, double* currpos)
  {
    const auto x = node->x();
    if (disnp != nullptr)
    {
      const int gid = discret.dof(node, 0);
      const int lid = disnp->get_map().lid(gid);
      if (lid < 0)
        FOUR_C_THROW(
            "Your displacement is incomplete (need to be based on a column map"
            " as this function is also called from a loop over elements and "
            "each proc does (usually) not own all nodes of its row elements ");
      for (size_t dim = 0; dim < x.size(); ++dim)
      {
        currpos[dim] = x[dim] + disnp->local_values_as_span()[lid + dim];
      }
      for (size_t dim = x.size(); dim < 3; ++dim) currpos[dim] = 0.0;
    }
    else
    {
      for (size_t dim = 0; dim < x.size(); ++dim) currpos[dim] = x[dim];
      for (size_t dim = x.size(); dim < 3; ++dim) currpos[dim] = 0.0;
    }
  }
}  // namespace Core::Binstrategy::Utils

FOUR_C_NAMESPACE_CLOSE
