// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_extract_values.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_utils_exceptions.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::extract_my_values(const Core::LinAlg::Vector<double>& global,
    Core::LinAlg::SerialDenseVector& local, const std::vector<int>& lm)
{
  const size_t ldim = lm.size();
  local.size(ldim);
  for (size_t i = 0; i < ldim; ++i)
  {
    const int lid = global.get_map().lid(lm[i]);
    if (lid < 0)
      FOUR_C_THROW("Proc {}: Cannot find gid={} in Core::LinAlg::Vector<double>",
          Core::Communication::my_mpi_rank(global.get_comm()), lm[i]);
    local[i] = global.local_values_as_span()[lid];
  }
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::extract_my_node_based_values(const Core::Elements::Element* ele,
    std::vector<double>& local, const Core::LinAlg::MultiVector<double>& global)
{
  const int numnode = ele->num_node();
  const int numcol = global.num_vectors();
  local.resize(numnode * numcol);

  // loop over element nodes
  for (int i = 0; i < numnode; ++i)
  {
    const int nodegid = (ele->nodes()[i])->id();
    const int lid = global.get_map().lid(nodegid);
    if (lid < 0)
      FOUR_C_THROW("Proc {}: Cannot find gid={} in Core::LinAlg::Vector<double>",
          Core::Communication::my_mpi_rank(global.get_comm()), nodegid);

    // loop over multi vector columns (numcol=1 for Core::LinAlg::Vector<double>)
    for (int col = 0; col < numcol; col++)
    {
      local[col + (numcol * i)] = global(col).local_values_as_span()[lid];
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::extract_my_node_based_values(const Core::Elements::Element* ele,
    Core::LinAlg::SerialDenseVector& local, Core::LinAlg::MultiVector<double>& global,
    const int nsd)
{
  if (nsd > global.num_vectors())
    FOUR_C_THROW("Requested {} of {} available columns", nsd, global.num_vectors());
  const int iel = ele->num_node();  // number of nodes
  if (local.length() != (iel * nsd)) FOUR_C_THROW("vector size mismatch.");

  // TODO: might we do change the loops?
  for (int i = 0; i < nsd; i++)
  {
    // loop over the element nodes
    for (int j = 0; j < iel; j++)
    {
      const int nodegid = (ele->nodes()[j])->id();
      const int lid = global.get_map().lid(nodegid);
      if (lid < 0)
        FOUR_C_THROW("Proc {}: Cannot find gid={} in Core::LinAlg::MultiVector<double>",
            Core::Communication::my_mpi_rank(global.get_comm()), nodegid);
      local(i + (nsd * j)) = global(i).local_values_as_span()[lid];
    }
  }
  return;
}

FOUR_C_NAMESPACE_CLOSE
