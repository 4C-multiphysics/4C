// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cardiovascular0d_dofset.hpp"

#include "4C_linalg_sparse_pod.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor (public)                                             ukue 04/07|
 *----------------------------------------------------------------------*/
Utils::Cardiovascular0DDofSet::Cardiovascular0DDofSet() : Core::DOFSets::DofSet() { return; }



/*----------------------------------------------------------------------*
 |  reset everything  (public)                                ukue 04/07|
 *----------------------------------------------------------------------*/
void Utils::Cardiovascular0DDofSet::reset()
{
  dofrowmap_ = nullptr;
  dofcolmap_ = nullptr;
}


/*----------------------------------------------------------------------*
 |  setup everything  (public)                                ukue 04/07|
 *----------------------------------------------------------------------*/
int Utils::Cardiovascular0DDofSet::assign_degrees_of_freedom(
    const std::shared_ptr<Core::FE::Discretization> dis, const int ndofs, const int start,
    const std::shared_ptr<Core::LinAlg::ProperOrthogonalDecomposition> mor)
{
  // A definite offset is currently not supported.
  if (start != 0) FOUR_C_THROW("right now user specified dof offsets are not supported");

  // Add DofSets in order of assignment to list. Once it is there it has its
  // place and will get its starting id from the previous DofSet.
  add_dof_set_to_list();

  // We assume that all dof sets before this one have been set up. Otherwise
  // we'd have to reorder the list.
  //
  // There is no test anymore to make sure that all prior dof sets have been
  // assigned. It seems people like to manipulate dof sets. People do create
  // dof sets that do not contain any dofs (on its first assignment), people
  // even shift dof set numbers to create overlapping dof sets. This is
  // perfectly fine.
  //
  // However if you rely on non-overlapping dof sets, you have to
  // fill_complete() your discretizations in the order of their creation. This
  // is guaranteed for all discretizations read from the input file since the
  // input reader calls fill_complete(). If you create your own discretizations
  // try to understand what you do.

  // Get highest GID used so far and add one
  // (In case of POD-MOR, the highest GID will be projmatrix->NumVectors()-1 because indexing starts
  // from 0. Therefore, there is no need to add anything.)
  int count;
  if (mor == nullptr)
    count = max_gid_in_list(dis->get_comm()) + 1;
  else
    count = mor->get_red_dim();

  // dofrowmap with index base = count, which is undesired
  Core::LinAlg::Map dofrowmap(ndofs, count, dis->get_comm());

  std::vector<int> gids;
  for (int i = 0; i < dofrowmap.num_my_elements(); i++) gids.push_back(dofrowmap.gid(i));

  // dofrowmap with index base = 0
  dofrowmap_ =
      std::make_shared<Core::LinAlg::Map>(-1, gids.size(), gids.data(), 0, dis->get_comm());

  return count;
}

FOUR_C_NAMESPACE_CLOSE
