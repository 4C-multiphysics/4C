// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_rebalance_print.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_pstream.hpp"

FOUR_C_NAMESPACE_OPEN

void Core::Rebalance::print_parallel_distribution(const Core::FE::Discretization& dis)
{
  const int numproc = Core::Communication::num_mpi_ranks(dis.get_comm());
  const int myrank = Core::Communication::my_mpi_rank(dis.get_comm());

  if (numproc > 1)
  {
    std::vector<int> my_n_nodes(numproc, 0);
    std::vector<int> n_nodes(numproc, 0);
    std::vector<int> my_n_ghostnodes(numproc, 0);
    std::vector<int> n_ghostnodes(numproc, 0);
    std::vector<int> my_n_elements(numproc, 0);
    std::vector<int> n_elements(numproc, 0);
    std::vector<int> my_n_ghostele(numproc, 0);
    std::vector<int> n_ghostele(numproc, 0);

    my_n_nodes[myrank] = dis.num_my_row_nodes();
    my_n_ghostnodes[myrank] = dis.num_my_col_nodes() - my_n_nodes[myrank];
    my_n_elements[myrank] = dis.num_my_row_elements();
    my_n_ghostele[myrank] = dis.num_my_col_elements() - my_n_elements[myrank];

    n_nodes = Core::Communication::sum_all(my_n_nodes, dis.get_comm());
    n_ghostnodes = Core::Communication::sum_all(my_n_ghostnodes, dis.get_comm());
    n_elements = Core::Communication::sum_all(my_n_elements, dis.get_comm());
    n_ghostele = Core::Communication::sum_all(my_n_ghostele, dis.get_comm());

    if (myrank == 0)
    {
      Core::IO::cout(Core::IO::verbose) << "\n   discretization: " << dis.name() << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "   +-----+---------------+--------------+-----------------+----------------+"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "   | PID |  n_rownodes   | n_ghostnodes |  n_rowelements  |   n_ghostele   |"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "   +-----+---------------+--------------+-----------------+----------------+"
          << Core::IO::endl;

      for (int npid = 0; npid < numproc; ++npid)
      {
        Core::IO::cout(Core::IO::verbose)
            << "   | " << std::setw(3) << npid << " | " << std::setw(13) << n_nodes[npid] << " | "
            << std::setw(12) << n_ghostnodes[npid] << " | " << std::setw(15) << n_elements[npid]
            << " | " << std::setw(14) << n_ghostele[npid] << " | " << Core::IO::endl;
        Core::IO::cout(Core::IO::verbose)
            << "   +-----+---------------+--------------+-----------------+----------------+"
            << Core::IO::endl;
      }
      Core::IO::cout(Core::IO::verbose) << Core::IO::endl;
    }
  }
}
FOUR_C_NAMESPACE_CLOSE
