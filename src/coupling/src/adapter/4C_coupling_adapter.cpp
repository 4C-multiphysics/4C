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
    : masterdofmap_(nullptr),
      permmasterdofmap_(nullptr),
      slavedofmap_(nullptr),
      permslavedofmap_(nullptr),
      masterexport_(nullptr),
      slaveexport_(nullptr),
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
    const Core::FE::Discretization& masterdis,
    std::shared_ptr<const Core::LinAlg::Map> mastercondmap,
    const Core::FE::Discretization& slavedis, std::shared_ptr<const Core::LinAlg::Map> slavecondmap,
    const std::string& condname, const std::vector<int>& masterdofs,
    const std::vector<int>& slavedofs, bool matchall, const int nds_master, const int nds_slave)
{
  const int numdof = masterdofs.size();
  const int numdof_slave = slavedofs.size();
  if (numdof != numdof_slave)
    FOUR_C_THROW("Received {} master DOFs, but {} slave DOFs", numdof, numdof_slave);

  auto masternodes_set = Core::Conditions::find_conditioned_node_ids(
      masterdis, condname, Core::Conditions::LookFor::locally_owned);
  std::vector<int> masternodes(masternodes_set.begin(), masternodes_set.end());
  auto slavenodes_set = Core::Conditions::find_conditioned_node_ids(
      slavedis, condname, Core::Conditions::LookFor::locally_owned);
  std::vector<int> slavenodes(slavenodes_set.begin(), slavenodes_set.end());

  int localmastercount = static_cast<int>(masternodes.size());
  int mastercount;
  int localslavecount = static_cast<int>(slavenodes.size());
  int slavecount;

  mastercount = Core::Communication::sum_all(localmastercount, masterdis.get_comm());
  slavecount = Core::Communication::sum_all(localslavecount, slavedis.get_comm());

  if (mastercount != slavecount)
    FOUR_C_THROW("got {} master nodes but {} slave nodes for coupling", mastercount, slavecount);

  setup_coupling(masterdis, slavedis, masternodes, slavenodes, masterdofs, slavedofs, matchall,
      1.0e-3, nds_master, nds_slave);

  // test for completeness
  if (static_cast<int>(masternodes.size()) * numdof != masterdofmap_->num_my_elements())
    FOUR_C_THROW("failed to setup master nodes properly");
  if (static_cast<int>(slavenodes.size()) * numdof != slavedofmap_->num_my_elements())
    FOUR_C_THROW("failed to setup slave nodes properly");

  // Now swap in the maps we already had.
  // So we did a little more work than required. But there are cases
  // where we have to do that work (fluid-ale coupling) and we want to
  // use just one setup implementation.
  //
  // The point is to make sure there is only one map for each
  // interface.

  if (not masterdofmap_->point_same_as(*mastercondmap)) FOUR_C_THROW("master dof map mismatch");

  if (not slavedofmap_->point_same_as(*slavecondmap))
  {
    FOUR_C_THROW("slave dof map mismatch");
  }

  masterdofmap_ = mastercondmap;
  masterexport_ = std::make_shared<Core::LinAlg::Export>(*permmasterdofmap_, *masterdofmap_);

  slavedofmap_ = slavecondmap;
  slaveexport_ = std::make_shared<Core::LinAlg::Export>(*permslavedofmap_, *slavedofmap_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_condition_coupling(
    const Core::FE::Discretization& masterdis,
    std::shared_ptr<const Core::LinAlg::Map> mastercondmap,
    const Core::FE::Discretization& slavedis, std::shared_ptr<const Core::LinAlg::Map> slavecondmap,
    const std::string& condname, const int numdof, bool matchall, const int nds_master,
    const int nds_slave)
{
  setup_condition_coupling(masterdis, mastercondmap, slavedis, slavecondmap, condname,
      build_dof_vector_from_num_dof(numdof), build_dof_vector_from_num_dof(numdof), matchall,
      nds_master, nds_slave);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis, const std::vector<int>& masternodes,
    const std::vector<int>& slavenodes, const std::vector<int>& masterdofs,
    const std::vector<int>& slavedofs, const bool matchall, const double tolerance,
    const int nds_master, const int nds_slave)
{
  std::vector<int> patchedmasternodes(masternodes);
  std::vector<int> permslavenodes;
  match_nodes(
      masterdis, slavedis, patchedmasternodes, permslavenodes, slavenodes, matchall, tolerance);

  // maps in original distribution

  std::shared_ptr<Core::LinAlg::Map> masternodemap = std::make_shared<Core::LinAlg::Map>(
      -1, patchedmasternodes.size(), patchedmasternodes.data(), 0, masterdis.get_comm());

  std::shared_ptr<Core::LinAlg::Map> slavenodemap = std::make_shared<Core::LinAlg::Map>(
      -1, slavenodes.size(), slavenodes.data(), 0, slavedis.get_comm());

  std::shared_ptr<Core::LinAlg::Map> permslavenodemap = std::make_shared<Core::LinAlg::Map>(
      -1, permslavenodes.size(), permslavenodes.data(), 0, slavedis.get_comm());

  finish_coupling(masterdis, slavedis, masternodemap, slavenodemap, permslavenodemap, masterdofs,
      slavedofs, nds_master, nds_slave);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis, const std::vector<int>& masternodes,
    const std::vector<int>& slavenodes, const int numdof, const bool matchall,
    const double tolerance, const int nds_master, const int nds_slave)
{
  setup_coupling(masterdis, slavedis, masternodes, slavenodes,
      build_dof_vector_from_num_dof(numdof), build_dof_vector_from_num_dof(numdof), matchall,
      tolerance, nds_master, nds_slave);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(
    std::shared_ptr<const Core::LinAlg::Map> slavedofmap,
    std::shared_ptr<const Core::LinAlg::Map> permslavedofmap,
    std::shared_ptr<const Core::LinAlg::Map> masterdofmap,
    std::shared_ptr<const Core::LinAlg::Map> permmasterdofmap)
{
  masterdofmap_ = masterdofmap;
  slavedofmap_ = slavedofmap;
  permmasterdofmap_ = permmasterdofmap;
  permslavedofmap_ = permslavedofmap;

  masterexport_ = std::make_shared<Core::LinAlg::Export>(*permmasterdofmap_, *masterdofmap_);
  slaveexport_ = std::make_shared<Core::LinAlg::Export>(*permslavedofmap_, *slavedofmap_);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis, const Core::LinAlg::Map& masternodes,
    const Core::LinAlg::Map& slavenodes, const int numdof, const bool matchall,
    const double tolerance, const int nds_master, const int nds_slave)
{
  if (masternodes.num_global_elements() != slavenodes.num_global_elements() and matchall)
    FOUR_C_THROW("got {} master nodes but {} slave nodes for coupling",
        masternodes.num_global_elements(), slavenodes.num_global_elements());

  std::vector<int> mastervect(masternodes.my_global_elements(),
      masternodes.my_global_elements() + masternodes.num_my_elements());
  std::vector<int> slavevect(slavenodes.my_global_elements(),
      slavenodes.my_global_elements() + slavenodes.num_my_elements());
  std::vector<int> permslavenodes;

  match_nodes(masterdis, slavedis, mastervect, permslavenodes, slavevect, matchall, tolerance);

  // maps in original distribution

  std::shared_ptr<Core::LinAlg::Map> masternodemap = std::make_shared<Core::LinAlg::Map>(
      -1, mastervect.size(), mastervect.data(), 0, masterdis.get_comm());

  std::shared_ptr<Core::LinAlg::Map> slavenodemap = std::make_shared<Core::LinAlg::Map>(slavenodes);

  std::shared_ptr<Core::LinAlg::Map> permslavenodemap = std::make_shared<Core::LinAlg::Map>(
      -1, permslavenodes.size(), permslavenodes.data(), 0, slavedis.get_comm());

  finish_coupling(masterdis, slavedis, masternodemap, slavenodemap, permslavenodemap,
      build_dof_vector_from_num_dof(numdof), build_dof_vector_from_num_dof(numdof), nds_master,
      nds_slave);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis, const Core::LinAlg::Map& masternodemap,
    const Core::LinAlg::Map& slavenodemap, const Core::LinAlg::Map& permslavenodemap,
    const int numdof)
{
  if (masternodemap.num_global_elements() != slavenodemap.num_global_elements())
    FOUR_C_THROW("got {} master nodes but {} slave nodes for coupling",
        masternodemap.num_global_elements(), slavenodemap.num_global_elements());

  // just copy maps

  std::shared_ptr<Core::LinAlg::Map> mymasternodemap =
      std::make_shared<Core::LinAlg::Map>(masternodemap);

  std::shared_ptr<Core::LinAlg::Map> myslavenodemap =
      std::make_shared<Core::LinAlg::Map>(slavenodemap);

  std::shared_ptr<Core::LinAlg::Map> mypermslavenodemap =
      std::make_shared<Core::LinAlg::Map>(permslavenodemap);

  // build slave to master permutation and dof all maps
  finish_coupling(masterdis, slavedis, mymasternodemap, myslavenodemap, mypermslavenodemap,
      build_dof_vector_from_num_dof(numdof), build_dof_vector_from_num_dof(numdof));
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(
    const Core::FE::Discretization& masterdis, const Core::FE::Discretization& slavedis)
{
  // safety check
  if (masterdis.dof_row_map()->num_global_elements() !=
      slavedis.dof_row_map()->num_global_elements())
    FOUR_C_THROW("got {} master nodes but {} slave nodes for coupling",
        masterdis.dof_row_map()->num_global_elements(),
        slavedis.dof_row_map()->num_global_elements());

  // get master dof maps and build exporter
  permmasterdofmap_ = std::make_shared<Core::LinAlg::Map>(*slavedis.dof_row_map());
  masterdofmap_ = std::make_shared<Core::LinAlg::Map>(*masterdis.dof_row_map());
  masterexport_ = std::make_shared<Core::LinAlg::Export>(*permmasterdofmap_, *masterdofmap_);

  // get slave dof maps and build exporter
  permslavedofmap_ = std::make_shared<Core::LinAlg::Map>(*masterdis.dof_row_map());
  slavedofmap_ = std::make_shared<Core::LinAlg::Map>(*slavedis.dof_row_map());
  slaveexport_ = std::make_shared<Core::LinAlg::Export>(*permslavedofmap_, *slavedofmap_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis, const std::vector<std::vector<int>>& masternodes_vec,
    const std::vector<std::vector<int>>& slavenodes_vec, const int numdof, const bool matchall,
    const double tolerance, const int nds_master, const int nds_slave)
{
  // vectors with master and slave node maps (from input) for every coupling condition
  // Permuted slave node map for each coupling conditions from match_nodes()
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> masternodemap_cond;
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> slavenodemap_cond;
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> permslavenodemap_cond;

  for (unsigned i = 0; i < masternodes_vec.size(); ++i)
  {
    std::vector<int> masternodes = masternodes_vec.at(i);
    std::vector<int> slavenodes = slavenodes_vec.at(i);

    std::vector<int> permslavenodes;

    match_nodes(masterdis, slavedis, masternodes, permslavenodes, slavenodes, matchall, tolerance);

    masternodemap_cond.push_back(std::make_shared<Core::LinAlg::Map>(
        -1, masternodes.size(), masternodes.data(), 0, masterdis.get_comm()));
    slavenodemap_cond.push_back(std::make_shared<Core::LinAlg::Map>(
        -1, slavenodes.size(), slavenodes.data(), 0, slavedis.get_comm()));
    permslavenodemap_cond.push_back(std::make_shared<Core::LinAlg::Map>(
        -1, permslavenodes.size(), permslavenodes.data(), 0, slavedis.get_comm()));
  }

  // merge maps for all conditions, but keep order (= keep assignment of permuted slave node map and
  // master map)
  auto masternodemap = Core::LinAlg::MultiMapExtractor::merge_maps_keep_order(masternodemap_cond);
  auto slavenodemap = Core::LinAlg::MultiMapExtractor::merge_maps_keep_order(slavenodemap_cond);
  auto permslavenodemap =
      Core::LinAlg::MultiMapExtractor::merge_maps_keep_order(permslavenodemap_cond);

  finish_coupling(masterdis, slavedis, masternodemap, slavenodemap, permslavenodemap,
      build_dof_vector_from_num_dof(numdof), build_dof_vector_from_num_dof(numdof), nds_master,
      nds_slave);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::match_nodes(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis, std::vector<int>& masternodes,
    std::vector<int>& permslavenodes, const std::vector<int>& slavenodes, const bool matchall,
    const double tolerance)
{
  // match master and slave nodes using octree
  auto tree = Core::GeometricSearch::NodeMatchingOctree();
  tree.init(masterdis, masternodes, 150, tolerance);
  tree.setup();

  std::map<int, std::pair<int, double>> coupling;
  tree.find_match(slavedis, slavenodes, coupling);

  if (masternodes.size() != coupling.size() and matchall)
    FOUR_C_THROW(
        "Did not get 1:1 correspondence. \nmasternodes.size()={} ({}), coupling.size()={} ({})",
        masternodes.size(), masterdis.name().c_str(), coupling.size(), slavedis.name().c_str());

  // extract permutation

  std::vector<int> patchedmasternodes;
  patchedmasternodes.reserve(coupling.size());
  permslavenodes.reserve(slavenodes.size());

  for (int gid : masternodes)
  {
    // We allow to hand in master nodes that do not take part in the
    // coupling. If this is undesired behaviour the user has to make
    // sure all nodes were used.
    if (coupling.find(gid) != coupling.end())
    {
      std::pair<int, double>& coupled = coupling[gid];
      patchedmasternodes.push_back(gid);
      permslavenodes.push_back(coupled.first);
    }
  }

  // return new list of master nodes via reference
  swap(masternodes, patchedmasternodes);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::finish_coupling(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis, std::shared_ptr<Core::LinAlg::Map> masternodemap,
    std::shared_ptr<Core::LinAlg::Map> slavenodemap,
    std::shared_ptr<Core::LinAlg::Map> permslavenodemap, const std::vector<int>& masterdofs,
    const std::vector<int>& slavedofs, const int nds_master, const int nds_slave)
{
  // we expect to get maps of exactly the same shape
  if (not masternodemap->point_same_as(*permslavenodemap))
    FOUR_C_THROW("master and permuted slave node maps do not match");

  // export master nodes to slave node distribution

  // To do so we create vectors that contain the values of the master
  // maps, assigned to the slave maps. On the master side we actually
  // create just a view on the map! This vector must not be changed!
  std::shared_ptr<Core::LinAlg::Vector<int>> masternodevec =
      std::make_shared<Core::LinAlg::Vector<int>>(
          *permslavenodemap, masternodemap->my_global_elements());

  std::shared_ptr<Core::LinAlg::Vector<int>> permmasternodevec =
      std::make_shared<Core::LinAlg::Vector<int>>(*slavenodemap);

  Core::LinAlg::Export masternodeexport(*permslavenodemap, *slavenodemap);
  const int err = permmasternodevec->export_to(*masternodevec, masternodeexport, Insert);
  if (err) FOUR_C_THROW("failed to export master nodes");

  std::shared_ptr<const Core::LinAlg::Map> permmasternodemap =
      std::make_shared<Core::LinAlg::Map>(-1, permmasternodevec->local_length(),
          permmasternodevec->get_local_values().data(), 0, masterdis.get_comm());

  if (not slavenodemap->point_same_as(*permmasternodemap))
    FOUR_C_THROW("slave and permuted master node maps do not match");

  masternodevec = nullptr;
  permmasternodevec = nullptr;

  build_dof_maps(masterdis, slavedis, masternodemap, slavenodemap, permmasternodemap,
      permslavenodemap, masterdofs, slavedofs, nds_master, nds_slave);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::build_dof_maps(const Core::FE::Discretization& masterdis,
    const Core::FE::Discretization& slavedis,
    const std::shared_ptr<const Core::LinAlg::Map>& masternodemap,
    const std::shared_ptr<const Core::LinAlg::Map>& slavenodemap,
    const std::shared_ptr<const Core::LinAlg::Map>& permmasternodemap,
    const std::shared_ptr<const Core::LinAlg::Map>& permslavenodemap,
    const std::vector<int>& masterdofs, const std::vector<int>& slavedofs, const int nds_master,
    const int nds_slave)
{
  build_dof_maps(masterdis, *masternodemap, *permmasternodemap, masterdofmap_, permmasterdofmap_,
      masterexport_, masterdofs, nds_master);
  build_dof_maps(slavedis, *slavenodemap, *permslavenodemap, slavedofmap_, permslavedofmap_,
      slaveexport_, slavedofs, nds_slave);
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
      // loop them and check, whether this is a pbc pure master node
      // for all previous conditions
      unsigned ntimesmaster = 0;
      for (auto& cond : thiscond)
      {
        const auto& mymasterslavetoggle = cond->parameters().get<std::string>("MASTER_OR_SLAVE");

        if (mymasterslavetoggle == "Master")
        {
          ++ntimesmaster;
        }
      }

      if (ntimesmaster < thiscond.size())
      {
        // this node is not a master and does not own its own dofs
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
  dofmap = std::make_shared<Core::LinAlg::Map>(
      -1, dofmapvec.size(), dofmapvec.data(), 0, dis.get_comm());

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
  permdofmap = std::make_shared<Core::LinAlg::Map>(
      -1, dofmapvec.size(), dofmapvec.data(), 0, dis.get_comm());

  // prepare communication plan to create a dofmap out of a permuted
  // dof map
  exporter = std::make_shared<Core::LinAlg::Export>(*permdofmap, *dofmap);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::Coupling::master_to_slave(
    const Core::LinAlg::Vector<double>& mv) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> sv =
      std::make_shared<Core::LinAlg::Vector<double>>(*slavedofmap_);

  master_to_slave(mv, *sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::Coupling::slave_to_master(
    const Core::LinAlg::Vector<double>& sv) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> mv =
      std::make_shared<Core::LinAlg::Vector<double>>(*masterdofmap_);

  slave_to_master(sv, *mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::FEVector<double>> Coupling::Adapter::Coupling::master_to_slave(
    const Core::LinAlg::FEVector<double>& mv) const
{
  std::shared_ptr<Core::LinAlg::FEVector<double>> sv =
      std::make_shared<Core::LinAlg::FEVector<double>>(*slavedofmap_, mv.num_vectors());

  Core::LinAlg::View sv_view(sv->get_ref_of_epetra_fevector());
  Core::LinAlg::View mv_view(mv.get_ref_of_epetra_fevector());
  master_to_slave(mv_view.underlying().as_multi_vector(), sv_view.underlying().as_multi_vector());

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::FEVector<double>> Coupling::Adapter::Coupling::slave_to_master(
    const Core::LinAlg::FEVector<double>& sv) const
{
  std::shared_ptr<Core::LinAlg::FEVector<double>> mv =
      std::make_shared<Core::LinAlg::FEVector<double>>(*masterdofmap_, sv.num_vectors());

  Core::LinAlg::View sv_view(sv.get_ref_of_epetra_fevector());
  Core::LinAlg::View mv_view(mv->get_ref_of_epetra_fevector());
  slave_to_master(sv_view.underlying().as_multi_vector(), mv_view.underlying().as_multi_vector());

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Coupling::Adapter::Coupling::master_to_slave(
    const Core::LinAlg::MultiVector<double>& mv) const
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> sv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*slavedofmap_, mv.num_vectors());

  master_to_slave(mv, *sv);

  return sv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::MultiVector<double>> Coupling::Adapter::Coupling::slave_to_master(
    const Core::LinAlg::MultiVector<double>& sv) const
{
  std::shared_ptr<Core::LinAlg::MultiVector<double>> mv =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*masterdofmap_, sv.num_vectors());

  slave_to_master(sv, *mv);

  return mv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::master_to_slave(
    const Core::LinAlg::MultiVector<double>& mv, Core::LinAlg::MultiVector<double>& sv) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not mv.get_map().point_same_as(*masterdofmap_))
    FOUR_C_THROW("master dof map vector expected");
  if (not sv.get_map().point_same_as(*slavedofmap_)) FOUR_C_THROW("slave dof map vector expected");
  if (sv.num_vectors() != mv.num_vectors())
    FOUR_C_THROW("column number mismatch {}!={}", sv.num_vectors(), mv.num_vectors());
#endif

  Core::LinAlg::MultiVector<double> perm(*permslavedofmap_, mv.num_vectors());
  std::copy(
      mv.get_values(), mv.get_values() + (mv.local_length() * mv.num_vectors()), perm.get_values());

  sv.export_to(perm, *slaveexport_, Insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::master_to_slave(
    const Core::LinAlg::Vector<int>& mv, Core::LinAlg::Vector<int>& sv) const
{
  Core::LinAlg::Vector<int> perm(*permslavedofmap_);
  std::ranges::copy(mv.get_local_values(), perm.get_local_values().begin());

  const int err = sv.export_to(perm, *slaveexport_, Insert);
  if (err) FOUR_C_THROW("Export to slave distribution returned err={}", err);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::slave_to_master(
    const Core::LinAlg::MultiVector<double>& sv, Core::LinAlg::MultiVector<double>& mv) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not mv.get_map().point_same_as(*masterdofmap_))
    FOUR_C_THROW("master dof map vector expected");
  if (not sv.get_map().point_same_as(*slavedofmap_))
  {
    std::cout << "slavedofmap_" << std::endl;
    std::cout << *slavedofmap_ << std::endl;
    std::cout << "sv" << std::endl;
    std::cout << sv.get_map() << std::endl;
    FOUR_C_THROW("slave dof map vector expected");
  }
  if (sv.num_vectors() != mv.num_vectors())
    FOUR_C_THROW("column number mismatch {}!={}", sv.num_vectors(), mv.num_vectors());
#endif

  Core::LinAlg::MultiVector<double> perm(*permmasterdofmap_, sv.num_vectors());
  std::copy(
      sv.get_values(), sv.get_values() + (sv.local_length() * sv.num_vectors()), perm.get_values());

  mv.export_to(perm, *masterexport_, Insert);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::slave_to_master(
    const Core::LinAlg::Vector<int>& sv, Core::LinAlg::Vector<int>& mv) const
{
  Core::LinAlg::Vector<int> perm(*permmasterdofmap_);
  std::ranges::copy(sv.get_local_values(), perm.get_local_values().begin());

  const int err = mv.export_to(perm, *masterexport_, Insert);
  if (err) FOUR_C_THROW("Export to master distribution returned err={}", err);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::fill_master_to_slave_map(std::map<int, int>& rowmap) const
{
  for (int i = 0; i < masterdofmap_->num_my_elements(); ++i)
  {
    rowmap[masterdofmap_->gid(i)] = permslavedofmap_->gid(i);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::fill_slave_to_master_map(std::map<int, int>& rowmap) const
{
  for (int i = 0; i < slavedofmap_->num_my_elements(); ++i)
  {
    rowmap[slavedofmap_->gid(i)] = permmasterdofmap_->gid(i);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Coupling::Adapter::Coupling::slave_to_master_map(
    Core::LinAlg::Map& slave)
{
  int nummyele = 0;
  std::vector<int> globalelements;
  const std::shared_ptr<Core::LinAlg::Map> slavemap = Core::LinAlg::allreduce_e_map(slave);
  for (int i = 0; i < slavemap->num_my_elements(); ++i)
  {
    int lid = permslavedofmap_->lid(slavemap->gid(i));
    if (lid != -1)
    {
      globalelements.push_back(masterdofmap_->gid(lid));
      nummyele++;
    }
  }

  return std::make_shared<Core::LinAlg::Map>(
      -1, nummyele, globalelements.data(), 0, slave.get_comm());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Coupling::Adapter::Coupling::master_to_slave_map(
    Core::LinAlg::Map& master)
{
  int nummyele = 0;
  std::vector<int> globalelements;
  const std::shared_ptr<Core::LinAlg::Map> mastermap = Core::LinAlg::allreduce_e_map(master);
  for (int i = 0; i < mastermap->num_my_elements(); ++i)
  {
    int lid = permmasterdofmap_->lid(mastermap->gid(i));
    if (lid != -1)
    {
      globalelements.push_back(slavedofmap_->gid(lid));
      nummyele++;
    }
  }

  return std::make_shared<Core::LinAlg::Map>(
      -1, nummyele, globalelements.data(), 0, master.get_comm());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Coupling::Adapter::Coupling::master_to_perm_master(
    const Core::LinAlg::SparseMatrix& sm) const
{
  auto permsm =
      std::make_shared<Core::LinAlg::SparseMatrix>(*permmasterdofmap_, sm.max_num_entries());

  // OK. You cannot use the same exporter for different matrices. So we
  // recreate one all the time... This has to be optimized later on.
  Core::LinAlg::Export exporter(*permmasterdofmap_, *masterdofmap_);
  int err = permsm->import(sm, exporter, Insert);

  if (err) FOUR_C_THROW("Import failed with err={}", err);

  permsm->complete(sm.domain_map(), *permmasterdofmap_);

  return permsm;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Coupling::Adapter::Coupling::slave_to_perm_slave(
    const Core::LinAlg::SparseMatrix& sm) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not sm.row_map().point_same_as(*slavedofmap_)) FOUR_C_THROW("slave dof map vector expected");
  if (not sm.filled()) FOUR_C_THROW("matrix must be filled");
#endif

  auto permsm =
      std::make_shared<Core::LinAlg::SparseMatrix>(*permslavedofmap_, sm.max_num_entries());

  // OK. You cannot use the same exporter for different matrices. So we
  // recreate one all the time... This has to be optimized later on.
  Core::LinAlg::Export exporter(*permslavedofmap_, *slavedofmap_);
  int err = permsm->import(sm, exporter, Insert);

  if (err) FOUR_C_THROW("Import failed with err={}", err);

  permsm->complete(sm.domain_map(), *permslavedofmap_);

  return permsm;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::Coupling::setup_coupling_matrices(const Core::LinAlg::Map& shiftedmastermap,
    const Core::LinAlg::Map& masterdomainmap, const Core::LinAlg::Map& slavedomainmap)
{
  // we always use the masterdofmap for the domain
  matmm_ = std::make_shared<Core::LinAlg::SparseMatrix>(shiftedmastermap, 1);
  matsm_ = std::make_shared<Core::LinAlg::SparseMatrix>(shiftedmastermap, 1);
  matmm_trans_ = std::make_shared<Core::LinAlg::SparseMatrix>(masterdomainmap, 1);
  matsm_trans_ = std::make_shared<Core::LinAlg::SparseMatrix>(*perm_slave_dof_map(), 1);

  int length = shiftedmastermap.num_my_elements();
  double one = 1.;
  for (int i = 0; i < length; ++i)
  {
    int sgid = perm_slave_dof_map()->gid(i);
    int mgid = master_dof_map()->gid(i);
    int shiftedmgid = shiftedmastermap.gid(i);

    int err = matmm_->insert_global_values(shiftedmgid, 1, &one, &mgid);
    if (err != 0)
      FOUR_C_THROW(
          "insert_global_values() for entry ({},{}) failed with err={}", shiftedmgid, mgid, err);

    err = matsm_->insert_global_values(shiftedmgid, 1, &one, &sgid);
    if (err != 0)
      FOUR_C_THROW(
          "insert_global_values() for entry ({},{}) failed with err={}", shiftedmgid, sgid, err);

    err = matmm_trans_->insert_global_values(mgid, 1, &one, &shiftedmgid);
    if (err != 0)
      FOUR_C_THROW(
          "insert_global_values() for entry ({},{}) failed with err={}", mgid, shiftedmgid, err);

    err = matsm_trans_->insert_global_values(sgid, 1, &one, &shiftedmgid);
    if (err != 0)
      FOUR_C_THROW(
          "insert_global_values() for entry ({},{}) failed with err={}", sgid, shiftedmgid, err);
  }

  matmm_->complete(masterdomainmap, shiftedmastermap);
  matsm_->complete(slavedomainmap, shiftedmastermap);
  matmm_trans_->complete(shiftedmastermap, masterdomainmap);
  matsm_trans_->complete(shiftedmastermap, *perm_slave_dof_map());

  // communicate slave to master matrix
  auto tmp = std::make_shared<Core::LinAlg::SparseMatrix>(slavedomainmap, 1);

  Core::LinAlg::Import exporter(slavedomainmap, *perm_slave_dof_map());
  int err = tmp->import(*matsm_trans_, exporter, Insert);
  if (err) FOUR_C_THROW("Import failed with err={}", err);

  tmp->complete(shiftedmastermap, slavedomainmap);
  matsm_trans_ = tmp;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::ma_dof_map_ptr()
{
  return masterdofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::LinAlg::Map& Coupling::Adapter::Coupling::ma_dof_map() const
{
  if (!masterdofmap_) FOUR_C_THROW("The masterdofmap_ has not been initialized correctly!");
  return *masterdofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::permuted_ma_dof_map_ptr()
{
  return permmasterdofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::LinAlg::Map& Coupling::Adapter::Coupling::permuted_ma_dof_map() const
{
  if (!permmasterdofmap_) FOUR_C_THROW("The permmasterdofmap_ has not been initialized correctly!");
  return *permmasterdofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::sl_dof_map_ptr()
{
  return slavedofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::LinAlg::Map& Coupling::Adapter::Coupling::sl_dof_map() const
{
  if (!slavedofmap_) FOUR_C_THROW("The slavedofmap_ has not been initialized correctly!");
  return *slavedofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>& Coupling::Adapter::Coupling::permuted_sl_dof_map_ptr()
{
  return permslavedofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::LinAlg::Map& Coupling::Adapter::Coupling::permuted_sl_dof_map() const
{
  if (!permslavedofmap_) FOUR_C_THROW("The permslavedofmap_ has not been initialized correctly!");
  return *permslavedofmap_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Export>& Coupling::Adapter::Coupling::ma_exporter_ptr()
{
  return masterexport_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::LinAlg::Export& Coupling::Adapter::Coupling::ma_exporter() const
{
  if (!masterexport_) FOUR_C_THROW("The masterexport_ has not been initialized correctly!");
  return *masterexport_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Export>& Coupling::Adapter::Coupling::sl_exporter_ptr()
{
  return slaveexport_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::LinAlg::Export& Coupling::Adapter::Coupling::sl_exporter() const
{
  if (!slaveexport_) FOUR_C_THROW("The slaveexport_ has not been initialized correctly!");
  return *slaveexport_;
}

FOUR_C_NAMESPACE_CLOSE
