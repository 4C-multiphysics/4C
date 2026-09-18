// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_adapter.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_geometric_search_matchingoctree.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"

#include <algorithm>
#include <numeric>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Coupling::Adapter::Coupling::Coupling()
    : target_dof_map_(nullptr),
      permuted_target_dof_map_(nullptr),
      source_dof_map_(nullptr),
      permuted_source_dof_map_(nullptr),
      target_export_(nullptr),
      source_export_(nullptr),
      matmm_(nullptr),
      matsm_(nullptr),
      matmm_trans_(nullptr),
      matsm_trans_(nullptr)
{
  // empty
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_condition_coupling(
    const Core::FE::Discretization& target_dis,
    std::shared_ptr<const Core::LinAlg::Map> target_cond_map,
    const Core::FE::Discretization& source_dis,
    std::shared_ptr<const Core::LinAlg::Map> source_cond_map, const std::string& condname,
    const std::vector<int>& target_dofs, const std::vector<int>& source_dofs, bool matchall,
    const int target_dofset_number, const int source_dofset_number)
{
  const int numdof = target_dofs.size();
  const int num_dof_source = source_dofs.size();
  if (numdof != num_dof_source)
    FOUR_C_THROW("Received {} target DOFs, but {} source DOFs", numdof, num_dof_source);

  auto target_nodes_set = Core::Conditions::find_conditioned_node_ids(
      target_dis, condname, Core::Conditions::LookFor::locally_owned);
  std::vector<int> target_nodes(target_nodes_set.begin(), target_nodes_set.end());
  auto source_nodes_set = Core::Conditions::find_conditioned_node_ids(
      source_dis, condname, Core::Conditions::LookFor::locally_owned);
  std::vector<int> source_nodes(source_nodes_set.begin(), source_nodes_set.end());

  int local_target_count = static_cast<int>(target_nodes.size());
  int target_count;
  int local_source_count = static_cast<int>(source_nodes.size());
  int source_count;

  target_count = Core::Communication::sum_all(local_target_count, target_dis.get_comm());
  source_count = Core::Communication::sum_all(local_source_count, source_dis.get_comm());

  if (target_count != source_count)
    FOUR_C_THROW(
        "got {} target nodes but {} source nodes for coupling", target_count, source_count);

  setup_coupling(target_dis, source_dis, target_nodes, source_nodes, target_dofs, source_dofs,
      matchall, 1.0e-3, target_dofset_number, source_dofset_number);

  // test for completeness
  if (static_cast<int>(target_nodes.size()) * numdof != target_dof_map_->num_my_elements())
    FOUR_C_THROW("failed to setup target nodes properly");
  if (static_cast<int>(source_nodes.size()) * numdof != source_dof_map_->num_my_elements())
    FOUR_C_THROW("failed to setup source nodes properly");

  // Now swap in the maps we already had.
  // So we did a little more work than required. But there are cases
  // where we have to do that work (fluid-ale coupling) and we want to
  // use just one setup implementation.
  //
  // The point is to make sure there is only one map for each
  // interface.

  if (not target_dof_map_->point_same_as(*target_cond_map)) FOUR_C_THROW("target dof map mismatch");

  if (not source_dof_map_->point_same_as(*source_cond_map))
  {
    FOUR_C_THROW("source dof map mismatch");
  }

  target_dof_map_ = target_cond_map;
  target_export_ =
      std::make_shared<Core::LinAlg::Export>(*permuted_target_dof_map_, *target_dof_map_);

  source_dof_map_ = source_cond_map;
  source_export_ =
      std::make_shared<Core::LinAlg::Export>(*permuted_source_dof_map_, *source_dof_map_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_condition_coupling(
    const Core::FE::Discretization& target_dis,
    std::shared_ptr<const Core::LinAlg::Map> target_cond_map,
    const Core::FE::Discretization& source_dis,
    std::shared_ptr<const Core::LinAlg::Map> source_cond_map, const std::string& condname,
    const int numdof, bool matchall, const int target_dofset_number, const int source_dofset_number)
{
  setup_condition_coupling(target_dis, target_cond_map, source_dis, source_cond_map, condname,
      build_dof_vector_from_num_dof(numdof), build_dof_vector_from_num_dof(numdof), matchall,
      target_dofset_number, source_dofset_number);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis, const std::vector<int>& target_nodes,
    const std::vector<int>& source_nodes, const std::vector<int>& target_dofs,
    const std::vector<int>& source_dofs, const bool matchall, const double tolerance,
    const int target_dofset_number, const int source_dofset_number)
{
  std::vector<int> patched_target_nodes(target_nodes);
  std::vector<int> permuted_source_nodes;
  match_nodes(target_dis, source_dis, patched_target_nodes, permuted_source_nodes, source_nodes,
      matchall, tolerance);

  // maps in original distribution

  std::shared_ptr<Core::LinAlg::Map> target_node_map = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(patched_target_nodes), 0, target_dis.get_comm());

  std::shared_ptr<Core::LinAlg::Map> source_node_map = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(source_nodes), 0, source_dis.get_comm());

  std::shared_ptr<Core::LinAlg::Map> permuted_source_node_map = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(permuted_source_nodes), 0, source_dis.get_comm());

  finish_coupling(target_dis, source_dis, target_node_map, source_node_map,
      permuted_source_node_map, target_dofs, source_dofs, target_dofset_number,
      source_dofset_number);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis, const std::vector<int>& target_nodes,
    const std::vector<int>& source_nodes, const int numdof, const bool matchall,
    const double tolerance, const int target_dofset_number, const int source_dofset_number)
{
  setup_coupling(target_dis, source_dis, target_nodes, source_nodes,
      build_dof_vector_from_num_dof(numdof), build_dof_vector_from_num_dof(numdof), matchall,
      tolerance, target_dofset_number, source_dofset_number);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(
    std::shared_ptr<const Core::LinAlg::Map> source_dof_map,
    std::shared_ptr<const Core::LinAlg::Map> permuted_source_dof_map,
    std::shared_ptr<const Core::LinAlg::Map> target_dof_map,
    std::shared_ptr<const Core::LinAlg::Map> permuted_target_dof_map)
{
  target_dof_map_ = target_dof_map;
  source_dof_map_ = source_dof_map;
  permuted_target_dof_map_ = permuted_target_dof_map;
  permuted_source_dof_map_ = permuted_source_dof_map;

  target_export_ =
      std::make_shared<Core::LinAlg::Export>(*permuted_target_dof_map_, *target_dof_map_);
  source_export_ =
      std::make_shared<Core::LinAlg::Export>(*permuted_source_dof_map_, *source_dof_map_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis, const Core::LinAlg::Map& target_nodes,
    const Core::LinAlg::Map& source_nodes, const int numdof, const bool matchall,
    const double tolerance, const int target_dofset_number, const int source_dofset_number)
{
  if (target_nodes.num_global_elements() != source_nodes.num_global_elements() and matchall)
    FOUR_C_THROW("got {} target nodes but {} source nodes for coupling",
        target_nodes.num_global_elements(), source_nodes.num_global_elements());

  std::vector<int> target_vect(target_nodes.my_global_elements(),
      target_nodes.my_global_elements() + target_nodes.num_my_elements());
  std::vector<int> source_vect(source_nodes.my_global_elements(),
      source_nodes.my_global_elements() + source_nodes.num_my_elements());
  std::vector<int> permuted_source_nodes;

  match_nodes(
      target_dis, source_dis, target_vect, permuted_source_nodes, source_vect, matchall, tolerance);

  // maps in original distribution

  std::shared_ptr<Core::LinAlg::Map> target_node_map = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(target_vect), 0, target_dis.get_comm());

  std::shared_ptr<Core::LinAlg::Map> source_node_map =
      std::make_shared<Core::LinAlg::Map>(source_nodes);

  std::shared_ptr<Core::LinAlg::Map> permuted_source_node_map = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(permuted_source_nodes), 0, source_dis.get_comm());

  finish_coupling(target_dis, source_dis, target_node_map, source_node_map,
      permuted_source_node_map, build_dof_vector_from_num_dof(numdof),
      build_dof_vector_from_num_dof(numdof), target_dofset_number, source_dofset_number);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis, const Core::LinAlg::Map& target_node_map,
    const Core::LinAlg::Map& source_node_map, const Core::LinAlg::Map& permuted_source_node_map,
    const int numdof)
{
  if (target_node_map.num_global_elements() != source_node_map.num_global_elements())
    FOUR_C_THROW("got {} target nodes but {} source nodes for coupling",
        target_node_map.num_global_elements(), source_node_map.num_global_elements());

  // just copy maps

  std::shared_ptr<Core::LinAlg::Map> my_target_node_map =
      std::make_shared<Core::LinAlg::Map>(target_node_map);

  std::shared_ptr<Core::LinAlg::Map> my_source_node_map =
      std::make_shared<Core::LinAlg::Map>(source_node_map);

  std::shared_ptr<Core::LinAlg::Map> my_permuted_source_node_map =
      std::make_shared<Core::LinAlg::Map>(permuted_source_node_map);

  // build source to target permutation and dof all maps
  finish_coupling(target_dis, source_dis, my_target_node_map, my_source_node_map,
      my_permuted_source_node_map, build_dof_vector_from_num_dof(numdof),
      build_dof_vector_from_num_dof(numdof));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(
    const Core::FE::Discretization& target_dis, const Core::FE::Discretization& source_dis)
{
  // safety check
  if (target_dis.dof_row_map()->num_global_elements() !=
      source_dis.dof_row_map()->num_global_elements())
    FOUR_C_THROW("got {} target nodes but {} source nodes for coupling",
        target_dis.dof_row_map()->num_global_elements(),
        source_dis.dof_row_map()->num_global_elements());

  // get target dof maps and build exporter
  permuted_target_dof_map_ = std::make_shared<Core::LinAlg::Map>(*source_dis.dof_row_map());
  target_dof_map_ = std::make_shared<Core::LinAlg::Map>(*target_dis.dof_row_map());
  target_export_ =
      std::make_shared<Core::LinAlg::Export>(*permuted_target_dof_map_, *target_dof_map_);

  // get source dof maps and build exporter
  permuted_source_dof_map_ = std::make_shared<Core::LinAlg::Map>(*target_dis.dof_row_map());
  source_dof_map_ = std::make_shared<Core::LinAlg::Map>(*source_dis.dof_row_map());
  source_export_ =
      std::make_shared<Core::LinAlg::Export>(*permuted_source_dof_map_, *source_dof_map_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis,
    const std::vector<std::vector<int>>& target_nodes_vec,
    const std::vector<std::vector<int>>& source_nodes_vec, const int numdof, const bool matchall,
    const double tolerance, const int target_dofset_number, const int source_dofset_number)
{
  // vectors with target and source node maps (from input) for every coupling condition
  // Permuted source node map for each coupling conditions from match_nodes()
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> target_node_map_cond;
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> source_node_map_cond;
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> permuted_source_node_map_cond;

  for (unsigned i = 0; i < target_nodes_vec.size(); ++i)
  {
    std::vector<int> target_nodes = target_nodes_vec.at(i);
    std::vector<int> source_nodes = source_nodes_vec.at(i);

    std::vector<int> permuted_source_nodes;

    match_nodes(target_dis, source_dis, target_nodes, permuted_source_nodes, source_nodes, matchall,
        tolerance);

    target_node_map_cond.push_back(std::make_shared<Core::LinAlg::Map>(
        -1, std::span<const int>(target_nodes), 0, target_dis.get_comm()));
    source_node_map_cond.push_back(std::make_shared<Core::LinAlg::Map>(
        -1, std::span<const int>(source_nodes), 0, source_dis.get_comm()));
    permuted_source_node_map_cond.push_back(std::make_shared<Core::LinAlg::Map>(
        -1, std::span<const int>(permuted_source_nodes), 0, source_dis.get_comm()));
  }

  // merge maps for all conditions, but keep order (= keep assignment of permuted source node map
  // and target map)
  auto target_node_map =
      Core::LinAlg::MultiMapExtractor::merge_maps_keep_order(target_node_map_cond);
  auto source_node_map =
      Core::LinAlg::MultiMapExtractor::merge_maps_keep_order(source_node_map_cond);
  auto permuted_source_node_map =
      Core::LinAlg::MultiMapExtractor::merge_maps_keep_order(permuted_source_node_map_cond);

  finish_coupling(target_dis, source_dis, target_node_map, source_node_map,
      permuted_source_node_map, build_dof_vector_from_num_dof(numdof),
      build_dof_vector_from_num_dof(numdof), target_dofset_number, source_dofset_number);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::match_nodes(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis, std::vector<int>& target_nodes,
    std::vector<int>& permuted_source_nodes, const std::vector<int>& source_nodes,
    const bool matchall, const double tolerance)
{
  // match target and source nodes using octree
  auto tree = Core::GeometricSearch::NodeMatchingOctree();
  tree.init(target_dis, target_nodes, 150, tolerance);
  tree.setup();

  std::map<int, std::pair<int, double>> coupling;
  tree.find_match(source_dis, source_nodes, coupling);

  if (target_nodes.size() != coupling.size() and matchall)
    FOUR_C_THROW(
        "Did not get 1:1 correspondence. \ntargetnodes.size()={} ({}), coupling.size()={} ({})",
        target_nodes.size(), target_dis.name().c_str(), coupling.size(), source_dis.name().c_str());

  // extract permutation

  std::vector<int> patched_target_nodes;
  patched_target_nodes.reserve(coupling.size());
  permuted_source_nodes.reserve(source_nodes.size());

  for (int gid : target_nodes)
  {
    // We allow to hand in target nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int, double>& coupled = coupling[gid];
      patched_target_nodes.push_back(gid);
      permuted_source_nodes.push_back(coupled.first);
    }
  }

  // return new list of target nodes via reference
  swap(target_nodes, patched_target_nodes);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::finish_coupling(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis, std::shared_ptr<Core::LinAlg::Map> target_node_map,
    std::shared_ptr<Core::LinAlg::Map> source_node_map,
    std::shared_ptr<Core::LinAlg::Map> permuted_source_node_map,
    const std::vector<int>& target_dofs, const std::vector<int>& source_dofs,
    const int target_dofset_number, const int source_dofset_number)
{
  // we expect to get maps of exactly the same shape
  if (not target_node_map->point_same_as(*permuted_source_node_map))
    FOUR_C_THROW("target and permuted source node maps do not match");

  // export target nodes to source node distribution

  // To do so we create vectors that contain the values of the target
  // maps, assigned to the source maps. On the target side we actually
  // create just a view on the map! This vector must not be changed!
  std::shared_ptr<Core::LinAlg::Vector<int>> target_node_vec =
      std::make_shared<Core::LinAlg::Vector<int>>(
          *permuted_source_node_map, target_node_map->my_global_elements());

  std::shared_ptr<Core::LinAlg::Vector<int>> permuted_target_node_vec =
      std::make_shared<Core::LinAlg::Vector<int>>(*source_node_map);

  Core::LinAlg::Export target_node_export(*permuted_source_node_map, *source_node_map);
  permuted_target_node_vec->export_to(
      *target_node_vec, target_node_export, Core::LinAlg::CombineMode::insert);

  std::shared_ptr<const Core::LinAlg::Map> permuted_target_node_map =
      std::make_shared<Core::LinAlg::Map>(-1,
          std::span<const int>(permuted_target_node_vec->get_local_values().data(),
              permuted_target_node_vec->local_length()),
          0, target_dis.get_comm());

  if (not source_node_map->point_same_as(*permuted_target_node_map))
    FOUR_C_THROW("source and permuted target node maps do not match");

  target_node_vec = nullptr;
  permuted_target_node_vec = nullptr;

  build_dof_maps(target_dis, source_dis, target_node_map, source_node_map, permuted_target_node_map,
      permuted_source_node_map, target_dofs, source_dofs, target_dofset_number,
      source_dofset_number);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::build_dof_maps(const Core::FE::Discretization& target_dis,
    const Core::FE::Discretization& source_dis,
    const std::shared_ptr<const Core::LinAlg::Map>& target_node_map,
    const std::shared_ptr<const Core::LinAlg::Map>& source_node_map,
    const std::shared_ptr<const Core::LinAlg::Map>& permuted_target_node_map,
    const std::shared_ptr<const Core::LinAlg::Map>& permuted_source_node_map,
    const std::vector<int>& target_dofs, const std::vector<int>& source_dofs,
    const int target_dofset_number, const int source_dofset_number)
{
  build_dof_maps(target_dis, *target_node_map, *permuted_target_node_map, target_dof_map_,
      permuted_target_dof_map_, target_export_, target_dofs, target_dofset_number);
  build_dof_maps(source_dis, *source_node_map, *permuted_source_node_map, source_dof_map_,
      permuted_source_dof_map_, source_export_, source_dofs, source_dofset_number);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<int> Coupling::Adapter::Coupling::build_dof_vector_from_num_dof(const int numdof)
{
  std::vector<int> dofvec;
  if (numdof > 0)
  {
    dofvec.resize(numdof);
    std::iota(dofvec.begin(), dofvec.end(), 0);
  }
  else
    dofvec = {-1};

  return dofvec;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::build_dof_maps(const Core::FE::Discretization& dis,
    const Core::LinAlg::Map& nodemap, const Core::LinAlg::Map& permnodemap,
    std::shared_ptr<const Core::LinAlg::Map>& dofmap,
    std::shared_ptr<const Core::LinAlg::Map>& permdofmap,
    std::shared_ptr<Core::LinAlg::Export>& exporter, const std::vector<int>& coupled_dofs,
    const int nds) const
{
  // communicate dofs

  std::vector<int> dofmapvec;
  std::map<int, std::vector<int>> dofs;


  auto surface_periodic_conditions = Core::Conditions::find_conditioned_node_ids_and_conditions(
      dis, "SurfacePeriodic", Core::Conditions::LookFor::locally_owned);
  auto line_periodic_conditions = Core::Conditions::find_conditioned_node_ids_and_conditions(
      dis, "LinePeriodic", Core::Conditions::LookFor::locally_owned);

  const int* nodes = nodemap.my_global_elements();
  const int numnode = nodemap.num_my_elements();

  for (int i = 0; i < numnode; ++i)
  {
    const Core::Nodes::Node* actnode = dis.g_node(nodes[i]);

    std::vector<const Core::Conditions::Condition*> thiscond;

    const auto copy_relevant_conditions = [&](const auto& conditions_map)
    {
      auto range = conditions_map.equal_range(actnode->id());
      std::ranges::copy(std::ranges::subrange(range.first, range.second) | std::views::values,
          std::back_inserter(thiscond));
    };
    copy_relevant_conditions(surface_periodic_conditions);
    copy_relevant_conditions(line_periodic_conditions);

    if (!thiscond.empty())
    {
      // loop them and check, whether this is a pbc pure target node
      // for all previous conditions
      unsigned n_times_target = 0;
      for (auto& cond : thiscond)
      {
        const auto& target_source_toggle = cond->parameters().get<std::string>("MASTER_OR_SLAVE");

        if (target_source_toggle == "Master")
        {
          ++n_times_target;
        }
      }

      if (n_times_target < thiscond.size())
      {
        // this node is not a target and does not own its own dofs
        continue;
      }
    }

    const std::vector<int> dof = dis.dof(nds, actnode);
    const int numdof = coupled_dofs.size();
    if (numdof > static_cast<int>(dof.size()))
      FOUR_C_THROW(
          "got just {} dofs at node {} (lid={}) but expected {}", dof.size(), nodes[i], i, numdof);
    for (int idof = 0; idof < numdof; idof++)
    {
      copy(dof.data() + coupled_dofs[idof], dof.data() + coupled_dofs[idof] + 1,
          back_inserter(dofs[nodes[i]]));
      copy(dof.data() + coupled_dofs[idof], dof.data() + coupled_dofs[idof] + 1,
          back_inserter(dofmapvec));
    }
  }

  std::vector<int>::const_iterator pos = std::min_element(dofmapvec.begin(), dofmapvec.end());
  if (pos != dofmapvec.end() and *pos < 0) FOUR_C_THROW("illegal dof number {}", *pos);

  // dof map is the original, unpermuted distribution of dofs
  dofmap =
      std::make_shared<Core::LinAlg::Map>(-1, std::span<const int>(dofmapvec), 0, dis.get_comm());

  dofmapvec.clear();

  Core::Communication::Exporter exportdofs(nodemap, permnodemap, dis.get_comm());
  exportdofs.do_export(dofs);

  const int* permnodes = permnodemap.my_global_elements();
  const int permnumnode = permnodemap.num_my_elements();

  for (int i = 0; i < permnumnode; ++i)
  {
    const std::vector<int>& dof = dofs[permnodes[i]];
    copy(dof.begin(), dof.end(), back_inserter(dofmapvec));
  }

  dofs.clear();

  // permuted dof map according to a given permuted node map
  permdofmap =
      std::make_shared<Core::LinAlg::Map>(-1, std::span<const int>(dofmapvec), 0, dis.get_comm());

  // prepare communication plan to create a dofmap out of a permuted
  // dof map
  exporter = std::make_shared<Core::LinAlg::Export>(*permdofmap, *dofmap);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::Coupling::target_to_source(
    const Core::LinAlg::Vector<double>& tv) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> sv =
      std::make_shared<Core::LinAlg::Vector<double>>(*source_dof_map_);

  target_to_source(tv, *sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::Coupling::source_to_target(
    const Core::LinAlg::Vector<double>& sv) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> tv =
      std::make_shared<Core::LinAlg::Vector<double>>(*target_dof_map_);

  source_to_target(sv, *tv);

  return tv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::FEVector<double>> Coupling::Adapter::Coupling::target_to_source(
    const Core::LinAlg::FEVector<double>& tv) const
{
  std::shared_ptr<Core::LinAlg::FEVector<double>> sv =
      std::make_shared<Core::LinAlg::FEVector<double>>(*source_dof_map_, tv.num_vectors());

  target_to_source(tv.as_multi_vector(), sv->as_multi_vector());

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::FEVector<double>> Coupling::Adapter::Coupling::source_to_target(
    const Core::LinAlg::FEVector<double>& sv) const
{
  std::shared_ptr<Core::LinAlg::FEVector<double>> tv =
      std::make_shared<Core::LinAlg::FEVector<double>>(*target_dof_map_, sv.num_vectors());

  source_to_target(sv.as_multi_vector(), tv->as_multi_vector());

  return tv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Coupling::Adapter::Coupling::target_to_source(
    const Core::LinAlg::MultiVector<double>& tv) const
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> sv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*source_dof_map_, tv.num_vectors());

  target_to_source(tv, *sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Coupling::Adapter::Coupling::source_to_target(
    const Core::LinAlg::MultiVector<double>& sv) const
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> tv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*target_dof_map_, sv.num_vectors());

  source_to_target(sv, *tv);

  return tv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::target_to_source(
    const Core::LinAlg::MultiVector<double>& tv, Core::LinAlg::MultiVector<double>& sv) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not tv.get_map().point_same_as(*target_dof_map_))
    FOUR_C_THROW("target dof map vector expected");
  if (not sv.get_map().point_same_as(*source_dof_map_))
    FOUR_C_THROW("source dof map vector expected");
  if (sv.num_vectors() != tv.num_vectors())
    FOUR_C_THROW("column number mismatch {}!={}", sv.num_vectors(), tv.num_vectors());
#endif

  Core::LinAlg::MultiVector<double> perm(*permuted_source_dof_map_, tv.num_vectors());
  std::copy(
      tv.get_values(), tv.get_values() + (tv.local_length() * tv.num_vectors()), perm.get_values());

  sv.export_to(perm, *source_export_, Core::LinAlg::CombineMode::insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::target_to_source(
    const Core::LinAlg::Vector<int>& tv, Core::LinAlg::Vector<int>& sv) const
{
  Core::LinAlg::Vector<int> perm(*permuted_source_dof_map_);
  std::ranges::copy(tv.get_local_values(), perm.get_local_values().begin());

  sv.export_to(perm, *source_export_, Core::LinAlg::CombineMode::insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::source_to_target(
    const Core::LinAlg::MultiVector<double>& sv, Core::LinAlg::MultiVector<double>& tv) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not tv.get_map().point_same_as(*target_dof_map_))
    FOUR_C_THROW("target dof map vector expected");
  if (not sv.get_map().point_same_as(*source_dof_map_))
  {
    std::cout << "source_dof_map_" << std::endl;
    std::cout << *source_dof_map_ << std::endl;
    std::cout << "sv" << std::endl;
    std::cout << sv.get_map() << std::endl;
    FOUR_C_THROW("source dof map vector expected");
  }
  if (sv.num_vectors() != tv.num_vectors())
    FOUR_C_THROW("column number mismatch {}!={}", sv.num_vectors(), tv.num_vectors());
#endif

  Core::LinAlg::MultiVector<double> perm(*permuted_target_dof_map_, sv.num_vectors());
  std::copy(
      sv.get_values(), sv.get_values() + (sv.local_length() * sv.num_vectors()), perm.get_values());

  tv.export_to(perm, *target_export_, Core::LinAlg::CombineMode::insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::source_to_target(
    const Core::LinAlg::Vector<int>& sv, Core::LinAlg::Vector<int>& tv) const
{
  Core::LinAlg::Vector<int> perm(*permuted_target_dof_map_);
  std::ranges::copy(sv.get_local_values(), perm.get_local_values().begin());

  tv.export_to(perm, *target_export_, Core::LinAlg::CombineMode::insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::fill_target_to_source_map(std::map<int, int>& rowmap) const
{
  for (int i = 0; i < target_dof_map_->num_my_elements(); ++i)
  {
    rowmap[target_dof_map_->gid(i)] = permuted_source_dof_map_->gid(i);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::fill_source_to_target_map(std::map<int, int>& rowmap) const
{
  for (int i = 0; i < source_dof_map_->num_my_elements(); ++i)
  {
    rowmap[source_dof_map_->gid(i)] = permuted_target_dof_map_->gid(i);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Coupling::Adapter::Coupling::source_to_target_map(
    Core::LinAlg::Map& source)
{
  int nummyele = 0;
  std::vector<int> globalelements;
  const std::shared_ptr<Core::LinAlg::Map> source_map = Core::LinAlg::allreduce_e_map(source);
  for (int i = 0; i < source_map->num_my_elements(); ++i)
  {
    int lid = permuted_source_dof_map_->lid(source_map->gid(i));
    if (lid != -1)
    {
      globalelements.push_back(target_dof_map_->gid(lid));
      nummyele++;
    }
  }

  return std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(globalelements.data(), nummyele), 0, source.get_comm());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Coupling::Adapter::Coupling::target_to_source_map(
    Core::LinAlg::Map& target)
{
  int nummyele = 0;
  std::vector<int> globalelements;
  const std::shared_ptr<Core::LinAlg::Map> target_map = Core::LinAlg::allreduce_e_map(target);
  for (int i = 0; i < target_map->num_my_elements(); ++i)
  {
    int lid = permuted_target_dof_map_->lid(target_map->gid(i));
    if (lid != -1)
    {
      globalelements.push_back(source_dof_map_->gid(lid));
      nummyele++;
    }
  }

  return std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(globalelements.data(), nummyele), 0, target.get_comm());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling_matrices(
    const Core::LinAlg::Map& shifted_target_map, const Core::LinAlg::Map& target_domain_map,
    const Core::LinAlg::Map& source_domain_map)
{
  // we always use the target_dof_map for the domain
  matmm_ = std::make_shared<Core::LinAlg::SparseMatrix>(shifted_target_map, 1);
  matsm_ = std::make_shared<Core::LinAlg::SparseMatrix>(shifted_target_map, 1);
  matmm_trans_ = std::make_shared<Core::LinAlg::SparseMatrix>(target_domain_map, 1);
  matsm_trans_ = std::make_shared<Core::LinAlg::SparseMatrix>(*permuted_source_dof_map(), 1);

  int length = shifted_target_map.num_my_elements();
  double one = 1.;
  for (int i = 0; i < length; ++i)
  {
    int sgid = permuted_source_dof_map()->gid(i);
    int mgid = target_dof_map()->gid(i);
    int shiftedmgid = shifted_target_map.gid(i);

    matmm_->insert_global_values(shiftedmgid, 1, &one, &mgid);
    matsm_->insert_global_values(shiftedmgid, 1, &one, &sgid);
    matmm_trans_->insert_global_values(mgid, 1, &one, &shiftedmgid);
    matsm_trans_->insert_global_values(sgid, 1, &one, &shiftedmgid);
  }

  matmm_->complete(target_domain_map, shifted_target_map);
  matsm_->complete(source_domain_map, shifted_target_map);
  matmm_trans_->complete(shifted_target_map, target_domain_map);
  matsm_trans_->complete(shifted_target_map, *permuted_source_dof_map());

  // communicate source to target matrix
  auto tmp = std::make_shared<Core::LinAlg::SparseMatrix>(source_domain_map, 1);

  Core::LinAlg::Import exporter(source_domain_map, *permuted_source_dof_map());
  tmp->import(*matsm_trans_, exporter, Core::LinAlg::CombineMode::insert);
  tmp->complete(shifted_target_map, source_domain_map);
  matsm_trans_ = tmp;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::target_dof_map_ptr()
{
  return target_dof_map_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::permuted_target_dof_map_ptr()
{
  return permuted_target_dof_map_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::source_dof_map_ptr()
{
  return source_dof_map_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::permuted_source_dof_map_ptr()
{
  return permuted_source_dof_map_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Export>& Coupling::Adapter::Coupling::target_exporter_ptr()
{
  return target_export_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Export>& Coupling::Adapter::Coupling::source_exporter_ptr()
{
  return source_export_;
}

FOUR_C_NAMESPACE_CLOSE
