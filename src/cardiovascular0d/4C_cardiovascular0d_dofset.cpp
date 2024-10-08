/*----------------------------------------------------------------------*/
/*! \file

\brief A set of degrees of freedom

\level 2


*----------------------------------------------------------------------*/

#include "4C_cardiovascular0d_dofset.hpp"

#include "4C_cardiovascular0d_mor_pod.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor (public)                                             ukue 04/07|
 *----------------------------------------------------------------------*/
UTILS::Cardiovascular0DDofSet::Cardiovascular0DDofSet() : Core::DOFSets::DofSet() { return; }



/*----------------------------------------------------------------------*
 |  reset everything  (public)                                ukue 04/07|
 *----------------------------------------------------------------------*/
void UTILS::Cardiovascular0DDofSet::reset()
{
  dofrowmap_ = Teuchos::null;
  dofcolmap_ = Teuchos::null;
}


/*----------------------------------------------------------------------*
 |  setup everything  (public)                                ukue 04/07|
 *----------------------------------------------------------------------*/
int UTILS::Cardiovascular0DDofSet::assign_degrees_of_freedom(
    const Teuchos::RCP<Core::FE::Discretization> dis, const int ndofs, const int start,
    const Teuchos::RCP<FourC::Cardiovascular0D::ProperOrthogonalDecomposition> mor)
{
  // A definite offset is currently not supported.
  if (start != 0) FOUR_C_THROW("right now user specified dof offsets are not supported");

  // Add DofSets in order of assignment to list. Once it is there it has its
  // place and will get its starting id from the previous DofSet.
  add_dof_setto_list();

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
  if (mor == Teuchos::null or not mor->have_mor())
    count = max_gi_din_list(dis->get_comm()) + 1;
  else
    count = mor->get_red_dim();

  // dofrowmap with index base = count, which is undesired
  Teuchos::RCP<Epetra_Map> dofrowmap = Teuchos::RCP(new Epetra_Map(ndofs, count, dis->get_comm()));

  std::vector<int> gids;
  for (int i = 0; i < dofrowmap->NumMyElements(); i++) gids.push_back(dofrowmap->GID(i));

  // dofrowmap with index base = 0
  dofrowmap_ = Teuchos::RCP(new Epetra_Map(-1, gids.size(), gids.data(), 0, dis->get_comm()));

  return count;
}

FOUR_C_NAMESPACE_CLOSE
