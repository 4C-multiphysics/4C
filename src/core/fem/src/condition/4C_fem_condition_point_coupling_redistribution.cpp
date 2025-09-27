// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_point_coupling_redistribution.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_rebalance_print.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
void put_all_slaves_to_masters_proc(
    const std::vector<const Core::Conditions::Condition*>& all_point_coupling_conditions,
    Core::FE::Discretization& discret);
void redistribute(const std::vector<int>& rank_to_hold_condition,
    const std::vector<const Core::Conditions::Condition*>& all_point_coupling_conditions,
    Core::FE::Discretization& discret);

void Core::Conditions::redistribute_for_point_coupling_conditions(Core::FE::Discretization& discret)
{
  // MPI rank of this processor
  const int myrank = Communication::my_mpi_rank(discret.get_comm());

  // vector of point coupling conditions
  std::vector<const Core::Conditions::Condition*> all_point_coupling_conditions;
  discret.get_condition("PointCoupling", all_point_coupling_conditions);

  if (Core::Communication::num_mpi_ranks(discret.get_comm()) == 1 ||
      all_point_coupling_conditions.empty())
  {
    // no redistribution necessary if only one processor or no point coupling condition
    // hence, fill complete discretization including assign degrees of freedom and leave
    discret.fill_complete();
    return;
  }

  if (myrank == 0)
  {
    std::cout << "\nRepair node distribution for point coupling condition(s) on discretization "
              << discret.name() << std::endl;
  }

  // fetch all slaves to the proc of the master
  put_all_slaves_to_masters_proc(all_point_coupling_conditions, discret);

  if (myrank == 0)
  {
    std::cout
        << "Finished repair of node distribution for point coupling conditions on discretization "
        << discret.name() << std::endl;
  }
}

void put_all_slaves_to_masters_proc(
    const std::vector<const Core::Conditions::Condition*>& all_point_coupling_conditions,
    Core::FE::Discretization& discret)
{
  const int myrank = Core::Communication::my_mpi_rank(discret.get_comm());

  std::vector<std::set<int>> conditioned_node_sets;
  for (const auto& current_condition : all_point_coupling_conditions)
  {
    const std::vector<int>* conditioned_nodes = current_condition->get_nodes();
    conditioned_node_sets.emplace_back(conditioned_nodes->begin(), conditioned_nodes->end());
  }

  // identify conditions with overlapping conditioned nodes
  // these conditions need to be owned by the same processor
  std::vector<std::set<int>> overlapping_conditions;
  for (size_t outer = 0; outer < all_point_coupling_conditions.size(); ++outer)
  {
    // create new entry for current condition with its own id
    overlapping_conditions.push_back(std::set<int>{static_cast<int>(outer)});
    for (size_t inner = 0; inner < outer; ++inner)
    {
      std::set<int> combined_set = conditioned_node_sets[inner];
      combined_set.insert(conditioned_node_sets[outer].begin(), conditioned_node_sets[outer].end());

      // overlapping sets "loose" entries, so size check is appropriate to identify overlap
      if (combined_set.size() !=
          conditioned_node_sets[inner].size() + conditioned_node_sets[outer].size())
      {
        // overlapping case
        overlapping_conditions[outer].insert(static_cast<int>(inner));
      }
    }
  }

  // it is important to know which condition overlaps with which others;
  // this double loop ensures that each overlapping condition knows all the other overlapping ones
  for (size_t outer = 0; outer < all_point_coupling_conditions.size(); ++outer)
  {
    for (size_t inner = 0; inner < all_point_coupling_conditions.size(); ++inner)
    {
      if (overlapping_conditions[inner].contains(static_cast<int>(outer)))
      {
        overlapping_conditions[outer].insert(static_cast<int>(inner));
      }
    }
  }

  // find out which processor holds master node of overlapping conditions;
  // arbitrarily, the smallest MPI rank is chosen to hold finally all row nodes of the (overlapping)
  // conditions
  std::vector<int> rank_to_hold_condition;
  for (size_t inner = 0; inner < all_point_coupling_conditions.size(); ++inner)
  {
    // first node in condition is selected to be master node (needs to be consistent with assumption
    // in dof set assignment procedure)
    int master_id = *(conditioned_node_sets[inner].begin());
    int rank_with_master_id = -1;
    if (discret.have_global_node(master_id))
    {
      // find rank which owns the master node
      Core::Nodes::Node* actnode = discret.g_node(master_id);
      if (actnode->owner() == myrank)
      {
        rank_with_master_id = myrank;
      }
    }
    // communicate such that all processor know which one owns the master node
    int rank_with_master_id_global;
    rank_with_master_id_global =
        Core::Communication::max_all(rank_with_master_id, discret.get_comm());
    rank_to_hold_condition.push_back(rank_with_master_id_global);
  }

  // redistribute such that all master and corresponding slave nodes are owned by the same processor
  redistribute(rank_to_hold_condition, all_point_coupling_conditions, discret);
}

void redistribute(const std::vector<int>& rank_to_hold_condition,
    const std::vector<const Core::Conditions::Condition*>& all_point_coupling_conditions,
    Core::FE::Discretization& discret)
{
  const int myrank = Core::Communication::my_mpi_rank(discret.get_comm());

  // make sure we have a filled discretization at this place
  // dofs are not required yet, they are assigned after redistribution
  // accessing the noderowmap requires a 'completed' discretization
  if (!discret.filled())
  {
    discret.fill_complete(Core::FE::OptionsFillComplete::none());
  }

  // get all currently owned node gids of this proc
  const std::vector<int> new_row_nodes = std::invoke(
      [&]()
      {
        std::vector<int> row_node_ids_on_this_proc(discret.node_row_map()->num_my_elements());
        discret.node_row_map()->my_global_elements(row_node_ids_on_this_proc.data());
        std::set<int> row_node_set(
            row_node_ids_on_this_proc.begin(), row_node_ids_on_this_proc.end());
        row_node_ids_on_this_proc.clear();

        // insert/remove gids such that all conditioned nodes are on the target processor
        int myerase = 0;
        int numerase = 0;
        int myadd = 0;
        int numadd = 0;

        for (size_t i = 0; i < all_point_coupling_conditions.size(); ++i)
        {
          const auto& current_condition = all_point_coupling_conditions[i];
          const std::vector<int>* conditioned_nodes = current_condition->get_nodes();

          // add conditioned nodes to target processor and remove from others
          if (rank_to_hold_condition[i] == myrank)
          {
            const int size_before = static_cast<int>(row_node_set.size());
            row_node_set.insert(conditioned_nodes->begin(), conditioned_nodes->end());
            myadd += static_cast<int>(row_node_set.size()) - size_before;
          }
          else
          {
            const int size_before = static_cast<int>(row_node_set.size());
            for (int idtodel : (*conditioned_nodes))
            {
              row_node_set.erase(idtodel);
            }
            myerase -= static_cast<int>(row_node_set.size()) - size_before;
          }
        }

        // print information
        numerase = Core::Communication::sum_all(myerase, discret.get_comm());
        numadd = Core::Communication::sum_all(myadd, discret.get_comm());
        if (myrank == 0)
        {
          std::cout << "Erased " << numerase << " nodes in total from row node list.\n";
          std::cout << "Added " << numadd << " nodes in total from row node list.\n";
        }

        {
          // safety check
          int myn = static_cast<int>(row_node_set.size());
          int gn = 0;

          gn = Core::Communication::sum_all(myn, discret.get_comm());
          FOUR_C_ASSERT_ALWAYS(gn == discret.num_global_nodes(),
              "Unmatching numbers of nodes before and after call Redistribution. Nodemap "
              "constructor will crash.\n");
        }

        return std::vector<int>(row_node_set.begin(), row_node_set.end());
      });

  //--------------------------------------------------
  // build new row node map
  Core::LinAlg::Map new_row_node_map(discret.num_global_nodes(), new_row_nodes.size(),
      new_row_nodes.data(), 0, discret.get_comm());

  // create nodal graph of problem, according to old row node map
  std::shared_ptr<Core::LinAlg::Graph> old_node_graph = discret.build_node_graph();

  // build graph based on the new row node map
  Core::LinAlg::Graph node_graph(Copy, new_row_node_map, 108, false);

  // export nodal graph to new row node layout
  {
    const Core::LinAlg::Export exporter(*discret.node_row_map(), new_row_node_map);
    node_graph.export_to(old_node_graph->get_epetra_crs_graph(), exporter, Add);
  }
  node_graph.fill_complete();
  node_graph.optimize_storage();

  // build node col map for new distribution of nodes
  const Core::LinAlg::Map new_col_node_block_map = node_graph.col_map();
  const Core::LinAlg::Map new_col_node_map(-1, new_col_node_block_map.num_my_elements(),
      new_col_node_block_map.my_global_elements(), 0, discret.get_comm());

  // rearrange node/element distribution according to target layout (accept extended ghosting)
  discret.redistribute(
      {
          .row_map = new_row_node_map,
          .col_map = new_col_node_map,
      },
      {
          .do_extended_ghosting = true,
      });

  if (myrank == 0)
    std::cout << "\nparallel redistributed discretization due to point coupling condition(s)"
              << std::endl;

  Core::Rebalance::print_parallel_distribution(discret);
}

FOUR_C_NAMESPACE_CLOSE
