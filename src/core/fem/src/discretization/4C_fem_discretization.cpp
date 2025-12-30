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
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_io_mesh.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_rebalance.hpp"
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
  ele->discretization_ = this;
  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::check_filled_globally()
{
  // global filled flag (is true / one if and only if filled_ == true on each processor
  int globalfilled = 0;

  // convert filled_ flag on this processor into integer (no communicator for type bool)
  int localfilled = (int)filled_;

  /*the global filled flag is set to the minimal value of any local filled flag
   * i.e. if on any processor filled_ == false, the flag globalfilled is set to
   * zero*/
  globalfilled = Core::Communication::min_all(localfilled, get_comm());

  // if not Filled() == true on all the processors call reset()
  if (!globalfilled) reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::FE::Discretization::add_node(
    std::span<const double> x, GlobalIndexType gid, std::shared_ptr<Core::Nodes::Node> user_node)
{
  FOUR_C_ASSERT(x.size() == n_dim_, "Node coordinate dimension mismatch in discretization.");

  // As long as user nodes are required, we need to ensure that a node exists, so we create one.
  // Once we can live without user nodes, this fixup may be removed.
  if (user_node == nullptr)
    user_node = std::make_shared<Core::Nodes::Node>(gid, x, Communication::my_mpi_rank(comm_));

  user_node->discretization_ = this;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  FOUR_C_ASSERT(user_node->id() == gid,
      "Node GID mismatch in discretization (user node). Expected {}, got {}.", gid,
      user_node->id());

  const auto user_x = user_node->x();
  FOUR_C_ASSERT(
      user_x.size() == n_dim_, "Node coordinate dimension mismatch in discretization (user node).");
  for (unsigned int i = 0; i < n_dim_; i++)
    FOUR_C_ASSERT(user_x[i] == x[i],
        "Node coordinate mismatch in discretization (user node). Expected {}, got {}.", x[i],
        user_x[i]);

#endif

  node_[gid] = user_node;
  FOUR_C_ASSERT(node_gid_.size() * n_dim_ == node_coordinates_.size(),
      "Internal error: mismatch in size of node data.");

  node_gid_.emplace_back() = gid;
  node_coordinates_.insert(node_coordinates_.end(), x.begin(), x.end());
  reset();
}


void Core::FE::Discretization::fill_from_mesh(const IO::MeshInput::Mesh<3>& mesh,
    const std::map<int, std::shared_ptr<Elements::Element>>& user_elements,
    const std::map<int, std::shared_ptr<Nodes::Node>>& user_nodes,
    const Rebalance::RebalanceParameters& params)
{
  FOUR_C_ASSERT_ALWAYS(!filled() && element_.empty() && node_.empty(),
      "Discretization must be empty before calling fill_from_mesh().");

  const int my_rank = Core::Communication::my_mpi_rank(comm_);
  if (my_rank == 0)
  {
    Core::Elements::ElementDefinition element_definition;

    unsigned ele_count = 0;
    for (const auto& [eb_id, eb] : mesh.cell_blocks())
    {
      for (const auto& connectivity : eb.cells())
      {
        (void)connectivity;
        std::shared_ptr<Core::Elements::Element> user_element = nullptr;

        if (auto it = user_elements.find(ele_count); it != user_elements.end())
          user_element = it->second;

        // Currently, we always require a user element
        FOUR_C_ASSERT_ALWAYS(user_element,
            "Need a user element for element ID {} in block with ID {}.", ele_count, eb_id);

        add_element(user_element);
        ele_count++;
      }
    }

    // Now add all the nodes to the discretization on rank 0. They are distributed later during
    // the rebalancing process.
    for (const auto& point : mesh.points_with_data())
    {
      const auto coordinate = point.coordinate();
      // Discard additional coordinates but only if they are zero.
      for (auto i = coordinate.size() - 1; i >= n_dim_; --i)
      {
        if (std::abs(coordinate[i]) != 0)
        {
          FOUR_C_THROW(
              "Node {} has a non-zero coordinate {} in direction {} but discretization "
              "is {}D!",
              point.id(), coordinate[i], i, n_dim_);
        }
      }

      std::shared_ptr<Core::Nodes::Node> user_node = nullptr;
      if (auto it = user_nodes.find(point.id()); it != user_nodes.end()) user_node = it->second;

      add_node(std::span(coordinate.data(), n_dim_), point.id(), user_node);
    }
  }
  else
  {
    FOUR_C_ASSERT_ALWAYS(mesh.points().size() == 0 && mesh.cell_blocks().size() == 0,
        "Non-zero number of nodes or elements on rank {} in discretization fill_from_mesh(). "
        "Currently we only expect elements and nodes on rank 0.",
        my_rank);
  }

  // Currently, all elements live on rank 0. Create a map for this situation.
  const int num_my_elements = (my_rank == 0) ? element_.size() : 0;
  Core::LinAlg::Map element_map(-1, num_my_elements, 0, comm_);

  // Now rebalance the discretization
  Core::Rebalance::rebalance_discretization(*this, element_map, params, comm_);
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
  auto curr = element_.find(gid);
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

    numglobalnodes = Core::Communication::sum_all(nummynodes, get_comm());
    numglobalelements = Core::Communication::sum_all(nummyele, get_comm());
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
const std::map<int, std::vector<int>>* Core::FE::Discretization::get_all_pbc_coupled_col_nodes()
    const
{
  // check for pbcs
  for (int nds = 0; nds < num_dof_sets(); nds++)
  {
    std::shared_ptr<const Core::DOFSets::PBCDofSet> pbcdofset =
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
std::shared_ptr<const std::map<int, int>>
Core::FE::Discretization::get_pbc_slave_to_master_node_connectivity() const
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
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
        std::make_shared<Core::LinAlg::Vector<double>>(*colmap, false);
    tmp->update(1.0, state, 0.0);
    state_[nds][name] = tmp;
  }
  else  // if it's not in column map export and allocate
  {
    FOUR_C_ASSERT(dof_row_map(nds)->same_as(state.get_map()),
        "row map of discretization {} and state vector {} are different. This is a fatal bug!",
        name_.c_str(), name.c_str());
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
        std::make_shared<Core::LinAlg::Vector<double>>(*colmap, false);

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
    tmp->import(state, (*stateimporter_[nds]), Core::LinAlg::CombineMode::insert);

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

std::vector<const Core::Conditions::Condition*> Core::FE::Discretization::get_conditions_on_node(
    const std::string& name, const Core::Nodes::Node* node) const
{
  std::vector<const Core::Conditions::Condition*> out;
  for (auto [begin, end] = condition_.equal_range(name);
      const auto& cond : std::ranges::subrange(begin, end) | std::views::values)
  {
    if (cond->contains_node(node->id())) out.emplace_back(cond.get());
  }
  return out;
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

  for (auto node : my_row_node_range())
  {
    node.user_node()->pack(buffer);
  }

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
    add_node(node->x(), node->id(), std::shared_ptr<Core::Nodes::Node>(node));
  }
  // in case add_node forgets...
  reset();
}


FOUR_C_NAMESPACE_CLOSE
