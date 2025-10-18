// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_discretization.hpp"

#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_dofset.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             ager 11/14|
 |  comm             (in)  a communicator object                        |
 *----------------------------------------------------------------------*/
XFEM::DiscretizationXFEM::DiscretizationXFEM(
    const std::string name, MPI_Comm comm, const unsigned int n_dim)
    : Core::FE::DiscretizationFaces(name, comm, n_dim),
      initialized_(false),
      initialfulldofrowmap_(nullptr),
      initialpermdofrowmap_(nullptr)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Finalize construction (public)                             ager 11/14|
 *----------------------------------------------------------------------*/
int XFEM::DiscretizationXFEM::initial_fill_complete(const std::vector<int>& nds,
    bool assigndegreesoffreedom, bool initelements, bool doboundaryconditions)
{
  // Call from BaseClass
  int val = Core::FE::Discretization::fill_complete(
      {assigndegreesoffreedom, initelements, doboundaryconditions});

  if (!assigndegreesoffreedom)
    FOUR_C_THROW(
        "DiscretizationXFEM: Call InitialFillComplete() with assigndegreesoffreedom = true!");

  // Store initial dofs of the discretisation
  store_initial_dofs(nds);
  return val;
}

/*----------------------------------------------------------------------*
 |  checks if discretization is initialized (protected)  ager 11/14|
 *----------------------------------------------------------------------*/
bool XFEM::DiscretizationXFEM::initialized() const
{
  if (!initialized_)
    FOUR_C_THROW("DiscretizationXFEM is not initialized! - Call InitialFillComplete() once!");
  return initialized_;
}

/*----------------------------------------------------------------------*
 |  Store Initial Dofs (private)                               ager 11/14|
 *----------------------------------------------------------------------*/
void XFEM::DiscretizationXFEM::store_initial_dofs(const std::vector<int>& nds)
{
  if (nds.size() != 1)
    FOUR_C_THROW(
        "DiscretizationXFEM: At the moment just one initial dofset to be initialized is supported "
        "by DiscretisationXFEM!");

  // store copy of initial dofset
  initialdofsets_.clear();

  initialdofsets_.push_back(
      std::dynamic_pointer_cast<Core::DOFSets::DofSet>(dofsets_[nds[0]])->clone());

  // store map required for export to active dofs
  if (initialdofsets_.size() > 1)
    FOUR_C_THROW(
        "DiscretizationXFEM: At the moment just one initial dofset is supported by "
        "DiscretisationXFEM!");

  std::shared_ptr<Core::DOFSets::FixedSizeDofSet> fsds =
      std::dynamic_pointer_cast<Core::DOFSets::FixedSizeDofSet>(initialdofsets_[0]);
  if (fsds == nullptr)
    FOUR_C_THROW("DiscretizationXFEM: Cast to Core::DOFSets::FixedSizeDofSet failed!");

  std::shared_ptr<XFEM::XFEMDofSet> xfds =
      std::dynamic_pointer_cast<XFEM::XFEMDofSet>(initialdofsets_[0]);
  if (xfds != nullptr)
    FOUR_C_THROW("DiscretizationXFEM: Initial Dofset shouldn't be a XFEM::XFEMDofSet!");

  int numdofspernode = 0;
  fsds->get_reserved_max_num_dofper_node(numdofspernode);

  if (num_my_col_nodes() == 0) FOUR_C_THROW("no column node on this proc available!");
  int numdofspernodedofset = fsds->num_dof(l_col_node(0));
  int numdofsetspernode = 0;

  if (numdofspernode % numdofspernodedofset)
    FOUR_C_THROW("DiscretizationXFEM: Dividing numdofspernode / numdofspernodedofset failed!");
  else
    numdofsetspernode = numdofspernode / numdofspernodedofset;

  initialfulldofrowmap_ =
      extend_map(fsds->dof_row_map(), numdofspernodedofset, numdofsetspernode, true);
  initialpermdofrowmap_ =
      extend_map(fsds->dof_row_map(), numdofspernodedofset, numdofsetspernode, false);

  initialized_ = true;

  return;
}

/*------------------------------------------------------------------------------*
 * Export Vector with initialdofrowmap (all nodes have one dofset) - to Vector  |
 * with all active dofs (public)                                       ager 11/14|
 *  *---------------------------------------------------------------------------*/
void XFEM::DiscretizationXFEM::export_initialto_active_vector(
    const Core::LinAlg::Vector<double>& initialvec, Core::LinAlg::Vector<double>& activevec)
{
  // Is the discretization initialized?
  initialized();

  Core::LinAlg::Vector<double> fullvec(*initialpermdofrowmap_, true);

  {  // Export manually as target.Map().UniqueGIDs() gives = true, although this shouldn't be the
     // case
    //(UniqueGIDs() just checks if gid occurs on more procs!)
    if (Core::Communication::num_mpi_ranks(initialvec.get_comm()) == 1 &&
        Core::Communication::num_mpi_ranks(activevec.get_comm()) ==
            1)  // for one proc , Export works fine!
    {
      Core::LinAlg::export_to(initialvec, fullvec);
    }
    else
    {
      Core::LinAlg::Import importer(fullvec.get_map(), initialvec.get_map());
      fullvec.import(initialvec, importer, Insert);
    }
  }
  fullvec.replace_map(*initialfulldofrowmap_);  /// replace |1 2 3 4|1 2 3 4| -> |1 2 3 4|5 6 7 8|
  Core::LinAlg::export_to(fullvec, activevec);
}

/*------------------------------------------------------------------------------*
 * Export Vector with initialdofrowmap (all nodes have one dofset) - to Vector  |
 * with all active dofs (public)                                       ager 11/14|
 *  *---------------------------------------------------------------------------*/
void XFEM::DiscretizationXFEM::export_activeto_initial_vector(
    const Core::LinAlg::Vector<double>& activevec, Core::LinAlg::Vector<double>& initialvec)
{
  // Is the discretization initialized?
  initialized();

  Core::LinAlg::export_to(activevec, initialvec);
}

/*----------------------------------------------------------------------*
 |  get dof row map (public)                                 ager 11/14 |
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* XFEM::DiscretizationXFEM::initial_dof_row_map(unsigned nds) const
{
  initialized();
  FOUR_C_ASSERT(nds < initialdofsets_.size(), "undefined initial dof set");

  return initialdofsets_[nds]->dof_row_map();
}


/*----------------------------------------------------------------------*
 |  get dof column map (public)                              ager 11/14 |
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* XFEM::DiscretizationXFEM::initial_dof_col_map(unsigned nds) const
{
  initialized();
  FOUR_C_ASSERT(nds < initialdofsets_.size(), "undefined initial dof set");

  return initialdofsets_[nds]->dof_col_map();
}

/*---------------------------------------------------------------------------*
 * Takes dof_row_map with just one xfem-Dofset and duplicates                  |
 * the dof gids for export to active dofs                          ager 11/14|
 *---------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> XFEM::DiscretizationXFEM::extend_map(
    const Core::LinAlg::Map* srcmap, int numdofspernodedofset, int numdofsets, bool uniquenumbering)
{
  int numsrcelements = srcmap->num_my_elements();
  const int* srcgids = srcmap->my_global_elements();
  std::vector<int> dstgids;
  for (int i = 0; i < numsrcelements; i += numdofspernodedofset)
  {
    if (numsrcelements < i + numdofspernodedofset) FOUR_C_THROW("extend_map(): Check your srcmap!");
    for (int dofset = 0; dofset < numdofsets; ++dofset)
    {
      for (int dof = 0; dof < numdofspernodedofset; ++dof)
      {
        dstgids.push_back(srcgids[i + dof] + uniquenumbering * dofset * numdofspernodedofset);
      }
    }
  }

  return std::make_shared<Core::LinAlg::Map>(
      -1, dstgids.size(), dstgids.data(), 0, srcmap->get_comm());
}

/*----------------------------------------------------------------------*
 |  set a reference to a data vector (public)                mwgee 12/06|
 *----------------------------------------------------------------------*/
void XFEM::DiscretizationXFEM::set_initial_state(unsigned nds, const std::string& name,
    std::shared_ptr<const Core::LinAlg::Vector<double>> state)
{
  TEUCHOS_FUNC_TIME_MONITOR("XFEM::DiscretizationXFEM::SetInitialState");

  if (!have_dofs()) FOUR_C_THROW("fill_complete() was not called");
  const Core::LinAlg::Map* colmap = initial_dof_col_map(nds);
  const Core::LinAlg::Map& vecmap = state->get_map();

  if (state_.size() <= nds) state_.resize(nds + 1);

  // if it's already in column map just set a reference
  // This is a rough test, but it might be ok at this place. It is an
  // error anyway to hand in a vector that is not related to our dof
  // maps.
  if (vecmap.point_same_as(*colmap))
  {
    state_[nds][name] = state;
  }
  else  // if it's not in column map export and allocate
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (not initial_dof_row_map(nds)->same_as(state->get_map()))
    {
      FOUR_C_THROW(
          "row map of discretization and state vector {} are different. This is a fatal bug!",
          name.c_str());
    }
#endif
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp = Core::LinAlg::create_vector(*colmap, false);
    Core::LinAlg::export_to(*state, *tmp);
    state_[nds][name] = tmp;
  }
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool XFEM::DiscretizationXFEM::is_equal_x_dof_set(
    int nds, const XFEM::XFEMDofSet& xdofset_new) const
{
  const XFEM::XFEMDofSet* xdofset_old = dynamic_cast<XFEM::XFEMDofSet*>(dofsets_[nds].get());
  if (not xdofset_old) return false;

  return ((*xdofset_old) == xdofset_new);
}

FOUR_C_NAMESPACE_CLOSE
