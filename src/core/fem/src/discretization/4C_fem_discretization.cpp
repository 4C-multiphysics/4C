// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_dofset_pbc.hpp"
#include "4C_fem_dofset_proxy.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <algorithm>
#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::FE::Discretization::Discretization(const std::string& name, MPI_Comm comm, unsigned int n_dim)
    : name_(name), comm_(comm), writer_(nullptr), filled_(false), havedof_(false), n_dim_(n_dim)
{
  dofsets_.emplace_back(std::make_shared<Core::DOFSets::DofSet>());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::add_element(std::shared_ptr<Core::Elements::Element> ele)
{
  element_[ele->id()] = ele;
  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::check_filled_globally()
{
  // global filled flag (is true / one if and only if filled_ == true on each processor
  int globalfilled = 0;

  // convert filled_ flag on this processor  into integer (no Epetra communicator for type bool)
  int localfilled = (int)filled_;

  /*the global filled flag is set to the minimal value of any local filled flag
   * i.e. if on any processor filled_ == false, the flag globalfilled is set to
   * zero*/
  Core::Communication::min_all(&localfilled, &globalfilled, 1, get_comm());

  // if not Filled() == true on all the processors call reset()
  if (!globalfilled) reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::add_node(std::shared_ptr<Core::Nodes::Node> node)
{
  node_[node->id()] = node;
  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::delete_node(Core::Nodes::Node& node)
{
  auto it_node = node_.find(node.id());
  if (it_node == node_.end()) return false;
  node_.erase(it_node);
  reset();
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::delete_node(const int gid)
{
  auto it_node = node_.find(gid);
  if (it_node == node_.end()) return false;
  node_.erase(it_node);
  reset();
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::delete_nodes()
{
  node_.clear();
  reset();
  check_filled_globally();
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::delete_elements()
{
  element_.clear();
  reset();
  check_filled_globally();
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::delete_element(Core::Elements::Element& ele)
{
  auto it_ele = element_.find(ele.id());
  if (it_ele == element_.end()) return false;
  element_.erase(it_ele);
  reset();
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::delete_element(const int gid)
{
  auto it_ele = element_.find(gid);
  if (it_ele == element_.end()) return false;
  element_.erase(it_ele);
  reset();
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::clear_discret()
{
  element_.clear();
  node_.clear();
  condition_.clear();
  reset();
  check_filled_globally();
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::Discretization::node_row_map() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return noderowmap_.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::Discretization::node_col_map() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return nodecolmap_.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::Discretization::element_row_map() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return elerowmap_.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::Discretization::element_col_map() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return elecolmap_.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::FE::Discretization::num_global_elements() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return element_row_map()->num_global_elements();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::FE::Discretization::num_my_row_elements() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return element_row_map()->num_my_elements();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::FE::Discretization::num_my_col_elements() const
{
  if (filled())
    return element_col_map()->num_my_elements();
  else
    return (int)element_.size();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::FE::Discretization::num_global_nodes() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return node_row_map()->num_global_elements();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::FE::Discretization::num_my_row_nodes() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before for discretization {}!", name_);
  return node_row_map()->num_my_elements();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::FE::Discretization::num_my_col_nodes() const
{
  if (filled())
    return node_col_map()->num_my_elements();
  else
    return (int)node_.size();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::have_global_element(const int gid) const
{
  return element_.find(gid) != element_.end();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Core::FE::Discretization::g_element(const int gid) const
{
  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator curr = element_.find(gid);
  FOUR_C_ASSERT(
      curr != element_.end(), "Element with global id gid={} not stored on this proc!", gid);
  return curr->second.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::have_global_node(const int gid) const
{
  return node_.find(gid) != node_.end();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Nodes::Node* Core::FE::Discretization::g_node(int gid) const
{
  std::map<int, std::shared_ptr<Core::Nodes::Node>>::const_iterator curr = node_.find(gid);
  FOUR_C_ASSERT(curr != node_.end(), "Node with global id gid={} not stored on this proc!", gid);
  return curr->second.get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const Core::FE::Discretization& dis)
{
  dis.print(os);
  return os;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::print(std::ostream& os) const
{
  int numglobalelements = 0;
  int numglobalnodes = 0;
  if (filled())
  {
    numglobalelements = num_global_elements();
    numglobalnodes = num_global_nodes();
  }
  else
  {
    int nummynodes = 0;
    std::map<int, std::shared_ptr<Core::Nodes::Node>>::const_iterator ncurr;
    for (ncurr = node_.begin(); ncurr != node_.end(); ++ncurr)
      if (ncurr->second->owner() == Core::Communication::my_mpi_rank(get_comm())) nummynodes++;

    int nummyele = 0;
    std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator ecurr;
    for (ecurr = element_.begin(); ecurr != element_.end(); ++ecurr)
      if (ecurr->second->owner() == Core::Communication::my_mpi_rank(get_comm())) nummyele++;

    Core::Communication::sum_all(&nummynodes, &numglobalnodes, 1, get_comm());
    Core::Communication::sum_all(&nummyele, &numglobalelements, 1, get_comm());
  }

  // print head
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    os << "--------------------------------------------------\n";
    os << "discretization: " << name() << std::endl;
    os << "--------------------------------------------------\n";
    os << numglobalelements << " Elements " << numglobalnodes << " Nodes (global)\n";
    os << "--------------------------------------------------\n";
    if (filled())
      os << "Filled() = true\n";
    else
      os << "Filled() = false\n";
    os << "--------------------------------------------------\n";
  }
  Core::Communication::barrier(get_comm());
  for (int proc = 0; proc < Core::Communication::num_mpi_ranks(get_comm()); ++proc)
  {
    if (proc == Core::Communication::my_mpi_rank(get_comm()))
    {
      // loop over dofsets
      for (int nds = 0; nds < num_dof_sets(); ++nds)
      {
        os << "\n------------------------ Dofset " << nds << " out of " << num_dof_sets()
           << " :\n\n";
        // print elements
        {
          os << "-------------------------- Proc " << proc << " :\n";
          std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator curr;
          for (curr = element_.begin(); curr != element_.end(); ++curr)
          {
            os << *(curr->second);
            if (filled() && have_dofs())
            {
              std::vector<int> dofs = dof(nds, &*(curr->second));
              if (dofs.size())
              {
                os << " Dofs ";
                for (int i : dofs) os << std::setw(6) << i << " ";
              }
            }
            os << std::endl;
          }
          os << std::endl;
        }
        // print nodes
        {
          os << "-------------------------- Proc " << proc << " :\n";
          std::map<int, std::shared_ptr<Core::Nodes::Node>>::const_iterator curr;
          for (curr = node_.begin(); curr != node_.end(); ++curr)
          {
            os << *(curr->second);
            if (filled() && have_dofs())
            {
              std::vector<int> dofs = dof(nds, &*(curr->second));
              if (dofs.size())
              {
                os << " Dofs ";
                for (int i : dofs) os << std::setw(6) << i << " ";
              }
            }
            os << std::endl;
          }
          os << std::endl;
        }
      }
      // print conditions
      {
        const unsigned numcond = condition_.size();
        if (numcond) os << "-------------------------- Proc " << proc << " :\n";
        if (numcond)
        {
          os << numcond << " Conditions:\n";
          std::map<std::string, std::shared_ptr<Core::Conditions::Condition>>::const_iterator curr;
          for (curr = condition_.begin(); curr != condition_.end(); ++curr)
          {
            os << curr->first << " ";
            os << *(curr->second) << std::endl;
          }
        }
        os << std::endl;
      }
    }
    Core::Communication::barrier(get_comm());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::Discretization::dof_row_map(const unsigned nds) const
{
  FOUR_C_ASSERT(nds < dofsets_.size(), "undefined dof set found in discretization {}!", name_);
  FOUR_C_ASSERT_ALWAYS(filled(), "fill_complete was not called on discretization {}!", name_);
  FOUR_C_ASSERT_ALWAYS(
      have_dofs(), "assign_degrees_of_freedom() not called on discretization {}!", name_);

  return dofsets_[nds]->dof_row_map();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::Discretization::dof_col_map(const unsigned nds) const
{
  FOUR_C_ASSERT(nds < dofsets_.size(), "undefined dof set found in discretization {}!", name_);
  FOUR_C_ASSERT_ALWAYS(filled(), "fill_complete was not called on discretization {}!", name_);
  FOUR_C_ASSERT_ALWAYS(
      have_dofs(), "assign_degrees_of_freedom() not called on discretization {}!", name_);

  return dofsets_[nds]->dof_col_map();
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::replace_dof_set(const unsigned nds,
    std::shared_ptr<Core::DOFSets::DofSetInterface> newdofset, const bool replaceinstatdofsets)
{
  FOUR_C_ASSERT(nds < dofsets_.size(), "undefined dof set found in discretization {}!", name_);
  // if we already have our dofs here and we add a properly filled (proxy)
  // DofSet, we do not need (and do not want) to refill.
  havedof_ = havedof_ and newdofset->filled() and nds != 0;
  if (replaceinstatdofsets) newdofset->replace_in_static_dofsets(dofsets_[nds]);
  dofsets_[nds] = newdofset;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Core::FE::Discretization::add_dof_set(std::shared_ptr<Core::DOFSets::DofSetInterface> newdofset)
{
  // if we already have our dofs here and we add a properly filled (proxy)
  // DofSet, we do not need (and do not want) to refill.
  havedof_ = havedof_ and newdofset->filled();
  dofsets_.push_back(newdofset);
  return static_cast<int>(dofsets_.size() - 1);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::DOFSets::DofSetInterface> Core::FE::Discretization::get_dof_set_proxy(
    const int nds)
{
  FOUR_C_ASSERT(nds < (int)dofsets_.size(), "undefined dof set found in discretization {}!", name_);
  return std::make_shared<Core::DOFSets::DofSetProxy>(&*dofsets_[nds]);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::replace_dof_set(
    std::shared_ptr<Core::DOFSets::DofSetInterface> newdofset, const bool replaceinstatdofsets)
{
  FOUR_C_ASSERT(dofsets_.size() == 1, "Discretization {} expects just one dof set!", name_);
  havedof_ = false;
  if (replaceinstatdofsets) newdofset->replace_in_static_dofsets(dofsets_[0]);
  dofsets_[0] = newdofset;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<int, std::vector<int>>* Core::FE::Discretization::get_all_pbc_coupled_col_nodes()
{
  // check for pbcs
  for (int nds = 0; nds < num_dof_sets(); nds++)
  {
    std::shared_ptr<Core::DOFSets::PBCDofSet> pbcdofset =
        std::dynamic_pointer_cast<Core::DOFSets::PBCDofSet>(dofsets_[nds]);

    if (pbcdofset != nullptr)
    {
      // it is assumed that, if one pbc set is available, all other potential dofsets hold the same
      // layout
      return pbcdofset->get_coupled_nodes();
    }
  }

  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<std::map<int, int>>
Core::FE::Discretization::get_pbc_slave_to_master_node_connectivity()
{
  // check for pbcs
  for (int nds = 0; nds < num_dof_sets(); nds++)
  {
    std::shared_ptr<Core::DOFSets::PBCDofSet> pbcdofset =
        std::dynamic_pointer_cast<Core::DOFSets::PBCDofSet>(dofsets_[nds]);

    if (pbcdofset != nullptr)
    {
      // it is assumed that, if one pbc set is available, all other potential dofsets hold the same
      // layout
      return pbcdofset->get_slave_to_master_node_connectivity();
    }
  }

  return nullptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::set_state(
    const unsigned nds, const std::string& name, const LinAlg::Vector<double>& state)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::FE::Discretization::set_state");

  FOUR_C_ASSERT_ALWAYS(have_dofs(), "fill_complete() was not called for discretization {}!", name_);
  const Core::LinAlg::Map* colmap = dof_col_map(nds);
  const Core::LinAlg::Map& vecmap = state.get_map();

  if (state_.size() <= nds) state_.resize(nds + 1);

  // if it's already in column map just set a reference
  // This is a rough test, but it might be ok at this place. It is an
  // error anyway to hand in a vector that is not related to our dof
  // maps.
  if (vecmap.point_same_as(*colmap))
  {
    FOUR_C_ASSERT(colmap->same_as(vecmap),
        "col map of discretization {} and state vector {} are different. This is a fatal bug!",
        name_.c_str(), name.c_str());
    // make a copy as in parallel such that no additional RCP points to the state vector
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp = Core::LinAlg::create_vector(*colmap, false);
    tmp->update(1.0, state, 0.0);
    state_[nds][name] = tmp;
  }
  else  // if it's not in column map export and allocate
  {
    FOUR_C_ASSERT(dof_row_map(nds)->same_as(state.get_map()),
        "row map of discretization {} and state vector {} are different. This is a fatal bug!",
        name_.c_str(), name.c_str());
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp = Core::LinAlg::create_vector(*colmap, false);

    // this is necessary to find out the number of nodesets in the beginning
    if (stateimporter_.size() <= nds)
    {
      stateimporter_.resize(nds + 1);
      for (unsigned i = 0; i <= nds; ++i) stateimporter_[i] = nullptr;
    }
    // (re)build importer if necessary
    if (stateimporter_[nds] == nullptr or
        not stateimporter_[nds]->source_map().same_as(state.get_map()) or
        not stateimporter_[nds]->target_map().same_as(*colmap))
    {
      stateimporter_[nds] = std::make_shared<Core::LinAlg::Import>(*colmap, state.get_map());
    }

    // transfer data
    int err = tmp->import(state, (*stateimporter_[nds]), Insert);
    FOUR_C_ASSERT_ALWAYS(!err,
        "Export using importer failed for Core::LinAlg::Vector<double>: return value = {}", err);

    // save state
    state_[nds][name] = tmp;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::set_condition(
    const std::string& name, std::shared_ptr<Core::Conditions::Condition> cond)
{
  condition_.insert(
      std::pair<std::string, std::shared_ptr<Core::Conditions::Condition>>(name, cond));
  filled_ = false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::replace_conditions(
    const std::string& name, const std::vector<std::shared_ptr<Core::Conditions::Condition>>& conds)
{
  if (condition_.contains(name)) condition_.erase(name);

  std::vector<std::shared_ptr<Core::Conditions::Condition>>::const_iterator cit;
  for (cit = conds.begin(); cit != conds.end(); ++cit)
  {
    // skip null pointers (these conditions will be deleted only and
    // therefore may disappear completely from this discretization)
    if (*cit != nullptr)
      condition_.insert(
          std::pair<std::string, std::shared_ptr<Core::Conditions::Condition>>(name, *cit));
  }
  filled_ = false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::get_condition(
    const std::string& name, std::vector<const Core::Conditions::Condition*>& out) const
{
  const unsigned num = condition_.count(name);
  out.resize(num);
  unsigned count = 0;

  auto range = condition_.equal_range(name);
  for (auto cond = range.first; cond != range.second; ++cond)
  {
    out[count++] = cond->second.get();
  }
  FOUR_C_ASSERT_ALWAYS(
      count == num, "Mismatch in number of conditions found in discretization {}!", name_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Core::FE::Discretization::has_condition(const std::string& name) const
{
  return condition_.contains(name);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::get_condition_names(std::vector<std::string>& names) const
{
  std::set<std::string> n;
  for (const auto& [name, cond] : condition_) n.insert(name);

  names.reserve(n.size());
  names.assign(n.begin(), n.end());
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::FE::Discretization::pack_my_elements() const
{
  FOUR_C_ASSERT_ALWAYS(filled(), "fill_complete was not called on discretization {}!", name_);

  Core::Communication::PackBuffer buffer;

  for (auto* ele : elerowptr_) ele->pack(buffer);

  auto block = std::make_shared<std::vector<char>>();
  std::swap(*block, buffer());
  return block;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<char>> Core::FE::Discretization::pack_my_nodes() const
{
  FOUR_C_ASSERT_ALWAYS(filled(), "fill_complete was not called on discretization {}!", name_);

  Core::Communication::PackBuffer buffer;

  for (auto* node : noderowptr_) node->pack(buffer);

  auto block = std::make_shared<std::vector<char>>();
  std::swap(*block, buffer());
  return block;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::unpack_my_elements(std::vector<char>& e)
{
  Communication::UnpackBuffer buffer(e);
  while (!buffer.at_end())
  {
    Core::Communication::ParObject* o = Core::Communication::factory(buffer);
    auto* ele = dynamic_cast<Core::Elements::Element*>(o);
    FOUR_C_ASSERT_ALWAYS(ele != nullptr,
        "Failed to build an element from the element data for discretization {}", name_);
    ele->set_owner(Core::Communication::my_mpi_rank(comm_));
    add_element(std::shared_ptr<Core::Elements::Element>(ele));
  }
  // in case add_element forgets...
  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::unpack_my_nodes(std::vector<char>& e)
{
  Communication::UnpackBuffer buffer(e);
  while (!buffer.at_end())
  {
    Core::Communication::ParObject* o = Core::Communication::factory(buffer);
    auto* node = dynamic_cast<Core::Nodes::Node*>(o);
    FOUR_C_ASSERT_ALWAYS(
        node != nullptr, "Failed to build a node from the node data for discretization {}", name_);
    node->set_owner(Core::Communication::my_mpi_rank(comm_));
    add_node(std::shared_ptr<Core::Nodes::Node>(node));
  }
  // in case add_node forgets...
  reset();
}


void Core::FE::Discretization::compute_null_space_if_necessary(
    Teuchos::ParameterList& solveparams, bool recompute)
{
  // see whether we have a list for an iterative solver
  if (!solveparams.isSublist("Belos Parameters") || solveparams.isSublist("IFPACK Parameters"))
  {
    return;
  }

  // adapt multigrid settings (if a multigrid preconditioner is used)
  if (!solveparams.isSublist("MueLu Parameters") && !solveparams.isSublist("Teko Parameters"))
    return;
  Teuchos::ParameterList* mllist_ptr = nullptr;
  if (solveparams.isSublist("MueLu Parameters"))
    mllist_ptr = &(solveparams.sublist("MueLu Parameters"));
  else if (solveparams.isSublist("Teko Parameters"))
    mllist_ptr = &(solveparams);
  else
    return;

  // see whether we have previously computed the nullspace
  // and recomputation is enforced
  Teuchos::ParameterList& mllist = *mllist_ptr;
  std::shared_ptr<Core::LinAlg::MultiVector<double>> ns =
      mllist.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullptr);
  if (ns != nullptr && !recompute) return;

  // compute solver parameters and set them into list
  Core::LinearSolver::Parameters::compute_solver_parameters(*this, mllist);
}

/*----------------------------------------------------------------------*
 |  set_state surrogate for node based vectors                  (public) |
 |                                                            gjb 06/09 |
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::add_multi_vector_to_parameter_list(Teuchos::ParameterList& p,
    const std::string name, std::shared_ptr<const Core::LinAlg::MultiVector<double>> vec)
{
  // provide data in node-based multi-vector for usage on element level
  // -> export to column map is necessary for parallel evaluation
  // set_state cannot be used since this multi-vector is nodebased and not dofbased!
  if (vec != nullptr)
  {
    const Core::LinAlg::Map* nodecolmap = node_col_map();
    const int numcol = vec->NumVectors();

    // if it's already in column map just copy it
    // This is a rough test, but it might be ok at this place.
    if (vec->get_map().point_same_as(*nodecolmap))
    {
      // make a copy as in parallel such that no additional RCP points to the state vector
      std::shared_ptr<Core::LinAlg::MultiVector<double>> tmp =
          std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, numcol);
      tmp->Update(1.0, *vec, 0.0);
      p.set(name, tmp);
    }
    else  // if it's not in column map export and allocate
    {
      std::shared_ptr<Core::LinAlg::MultiVector<double>> tmp =
          std::make_shared<Core::LinAlg::MultiVector<double>>(*nodecolmap, numcol);
      Core::LinAlg::export_to(*vec, *tmp);
      p.set(name, tmp);
    }
  }
  else
    p.set(name, nullptr);

  return;
}

FOUR_C_NAMESPACE_CLOSE
