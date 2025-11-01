// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fs3i_biofilm_fsi_utils.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FS3I::BioFilm::Utils::scatra_change_config(Core::FE::Discretization& scatradis,
    Core::FE::Discretization& dis, Core::LinAlg::Vector<double>& disp)
{
  const int numnode = (scatradis.node_col_map())->num_my_elements();

  // Create Vector which holds all col-displacements of processor
  Core::LinAlg::Vector<double> coldisp(*(dis.dof_col_map()));

  // Export row-displacements to col-displacements
  Core::LinAlg::export_to(disp, coldisp);


  const Core::LinAlg::Vector<double>& gvector = coldisp;

  // loop over all nodes
  for (int index = 0; index < numnode; ++index)
  {
    // get current node
    int gid = (scatradis.node_col_map())->gid(index);
    Core::Nodes::Node* mynode = scatradis.g_node(gid);

    // get local fluid/structure node with the same local node id
    Core::Nodes::Node* lnode = dis.l_col_node(index);

    // get degrees of freedom associated with this fluid/structure node
    std::vector<int> nodedofs = dis.dof(0, lnode);

    // Since ChangePos requires vector of length 3, we use hardcoded length here
    // init with zero just to be sure
    std::vector<double> nvector(3, 0.0);

    // determine number of space dimensions
    const int numdim = Global::Problem::instance()->n_dim();

    for (int i = 0; i < numdim; ++i)
    {
      const int lid = gvector.get_map().lid(nodedofs[i]);

      if (lid < 0)
        FOUR_C_THROW("Proc {}: Cannot find gid={} in Core::LinAlg::Vector<double>",
            Core::Communication::my_mpi_rank(gvector.get_comm()), nodedofs[i]);
      nvector[i] += gvector.local_values_as_span()[lid];
    }

    mynode->change_pos(nvector);
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
