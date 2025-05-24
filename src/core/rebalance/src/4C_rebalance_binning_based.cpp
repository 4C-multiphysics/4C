// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_rebalance_binning_based.hpp"

#include "4C_binstrategy.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_geometric_search_matchingoctree.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Rebalance using BinningStrategy                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void Core::Rebalance::rebalance_discretizations_by_binning(
    const Teuchos::ParameterList& binning_params,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const std::vector<std::shared_ptr<Core::FE::Discretization>>& vector_of_discretizations,
    std::function<const Core::Nodes::Node&(const Core::Nodes::Node& node)> correct_node,
    std::function<std::vector<std::array<double, 3>>(const Core::FE::Discretization&,
        const Core::Elements::Element&, std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)>
        determine_relevant_points,
    bool revertextendedghosting)
{
  // safety check
  if (vector_of_discretizations.size() == 0)
    FOUR_C_THROW("No discretizations provided for binning !");

  // get communicator
  MPI_Comm comm = vector_of_discretizations[0]->get_comm();

  // rebalance discr. with help of binning strategy
  if (Core::Communication::num_mpi_ranks(comm) > 1)
  {
    if (Core::Communication::my_mpi_rank(comm) == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "+---------------------------------------------------------------" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "| Rebalance discretizations using Binning Strategy ...          " << Core::IO::endl;
      for (const auto& curr_dis : vector_of_discretizations)
      {
        if (!curr_dis->filled()) FOUR_C_THROW("fill_complete(false,false,false) was not called");
        Core::IO::cout(Core::IO::verbose)
            << "| Rebalance discretization " << std::setw(11) << curr_dis->name() << Core::IO::endl;
      }
      Core::IO::cout(Core::IO::verbose)
          << "+---------------------------------------------------------------" << Core::IO::endl;
    }

    std::vector<std::shared_ptr<Core::LinAlg::Map>> stdelecolmap;
    std::vector<std::shared_ptr<Core::LinAlg::Map>> stdnodecolmap;

    // binning strategy is created and parallel redistribution is performed
    Binstrategy::BinningStrategy binningstrategy(binning_params, output_control,
        vector_of_discretizations[0]->get_comm(),
        Core::Communication::my_mpi_rank(vector_of_discretizations[0]->get_comm()),
        std::move(correct_node), std::move(determine_relevant_points), vector_of_discretizations);

    binningstrategy
        .do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
            vector_of_discretizations, stdelecolmap, stdnodecolmap);

    // revert extended ghosting if requested
    if (revertextendedghosting)
      binningstrategy.revert_extended_ghosting(
          vector_of_discretizations, stdelecolmap, stdnodecolmap);
  }  // if more than 1 proc
  else
    for (const auto& curr_dis : vector_of_discretizations) curr_dis->fill_complete();

}  // Core::Rebalance::rebalance_discretizations_by_binning

/*----------------------------------------------------------------------*
 |  Ghost input discr. redundantly on all procs             rauch 09/16 |
 *----------------------------------------------------------------------*/
void Core::Rebalance::ghost_discretization_on_all_procs(Core::FE::Discretization& distobeghosted)
{
  MPI_Comm com = distobeghosted.get_comm();
  if (Core::Communication::my_mpi_rank(com) == 0)
  {
    Core::IO::cout(Core::IO::verbose)
        << "+-----------------------------------------------------------------------+"
        << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << "|   Ghost discretization " << std::setw(11) << distobeghosted.name()
        << " redundantly on all procs ...       |" << Core::IO::endl;
    Core::IO::cout(Core::IO::verbose)
        << "+-----------------------------------------------------------------------+"
        << Core::IO::endl;
  }

  std::vector<int> allproc(Core::Communication::num_mpi_ranks(com));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(com); ++i) allproc[i] = i;

  // fill my own row node ids
  const Core::LinAlg::Map* noderowmap = distobeghosted.node_row_map();
  std::vector<int> sdata;
  for (int lid = 0; lid < noderowmap->num_my_elements(); ++lid)
  {
    int gid = noderowmap->gid(lid);
    sdata.push_back(gid);
  }

  // gather all master row node gids redundantly in rdata
  std::vector<int> rdata;
  Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), com);

  // build new node column map (on ALL processors)
  Core::LinAlg::Map newnodecolmap(-1, (int)rdata.size(), rdata.data(), 0, com);
  sdata.clear();
  rdata.clear();

  // fill my own row element ids
  const Core::LinAlg::Map* elerowmap = distobeghosted.element_row_map();
  sdata.resize(0);
  for (int i = 0; i < elerowmap->num_my_elements(); ++i)
  {
    int gid = elerowmap->gid(i);
    sdata.push_back(gid);
  }

  // gather all gids of elements redundantly
  rdata.resize(0);
  Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), com);

  // build new element column map (on ALL processors)
  Core::LinAlg::Map newelecolmap(-1, (int)rdata.size(), rdata.data(), 0, com);
  sdata.clear();
  rdata.clear();
  allproc.clear();

  // rebalance the nodes and elements of the discr. according to the
  // new node / element column layout (i.e. master = full overlap)
  distobeghosted.export_column_nodes(newnodecolmap);
  distobeghosted.export_column_elements(newelecolmap);

  // Safety checks in DEBUG
#ifdef FOUR_C_ENABLE_ASSERTIONS
  int nummycolnodes = newnodecolmap.num_my_elements();
  std::vector<int> sizelist(Core::Communication::num_mpi_ranks(com));
  Core::Communication::gather_all(&nummycolnodes, sizelist.data(), 1, com);
  Core::Communication::barrier(com);
  for (int k = 1; k < Core::Communication::num_mpi_ranks(com); ++k)
  {
    if (sizelist[k - 1] != nummycolnodes)
      FOUR_C_THROW(
          "Since whole dis. is ghosted every processor should have the same number of colnodes.\n"
          "This is not the case."
          "Fix this!");
  }
#endif
}  // Core::Rebalance::ghost_discretization_on_all_procs

/*---------------------------------------------------------------------*
|  Rebalance Elements Matching Template discretization     rauch 09/16 |
*----------------------------------------------------------------------*/
void Core::Rebalance::match_element_distribution_of_matching_discretizations(
    Core::FE::Discretization& dis_template, Core::FE::Discretization& dis_to_rebalance)
{
  // clone communicator of target discretization
  MPI_Comm com(dis_template.get_comm());
  if (Core::Communication::num_mpi_ranks(com) > 1)
  {
    // print to screen
    if (Core::Communication::my_mpi_rank(com) == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "|   Match element distribution of discr. " << std::setw(11) << dis_to_rebalance.name()
          << "to discr. " << std::setw(11) << dis_template.name() << " ... |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
    }

    ////////////////////////////////////////
    // MATCH ELEMENTS
    ////////////////////////////////////////
    std::vector<int> rebalance_rowelegid_vec;
    std::vector<int> rebalance_colelegid_vec;

    // match elements to be rebalanced to template elements and fill vectors
    // with desired row and col gids for redistribution.
    match_element_row_col_distribution(
        dis_template, dis_to_rebalance, rebalance_rowelegid_vec, rebalance_colelegid_vec);

    // construct rebalanced element row map
    Core::LinAlg::Map rebalanced_elerowmap(
        -1, rebalance_rowelegid_vec.size(), rebalance_rowelegid_vec.data(), 0, com);

    // construct rebalanced element col map
    Core::LinAlg::Map rebalanced_elecolmap(
        -1, rebalance_colelegid_vec.size(), rebalance_colelegid_vec.data(), 0, com);

    ////////////////////////////////////////
    // MATCH NODES
    ////////////////////////////////////////
    std::vector<int> rebalance_nodegid_vec;
    std::vector<int> rebalance_colnodegid_vec;

    // match nodes to be rebalanced to template nodes and fill vectors
    // with desired row and col gids for redistribution.
    match_nodal_row_col_distribution(
        dis_template, dis_to_rebalance, rebalance_nodegid_vec, rebalance_colnodegid_vec);

    // construct rebalanced node row map
    Core::LinAlg::Map rebalanced_noderowmap(
        -1, rebalance_nodegid_vec.size(), rebalance_nodegid_vec.data(), 0, com);

    // construct rebalanced node col map
    Core::LinAlg::Map rebalanced_nodecolmap(
        -1, rebalance_colnodegid_vec.size(), rebalance_colnodegid_vec.data(), 0, com);

    ////////////////////////////////////////
    // REBALANCE
    ////////////////////////////////////////
    // export the nodes
    dis_to_rebalance.export_row_nodes(rebalanced_noderowmap, false, false);
    dis_to_rebalance.export_column_nodes(rebalanced_nodecolmap, false, false);
    // export the elements
    dis_to_rebalance.export_row_elements(rebalanced_elerowmap, false, false);
    dis_to_rebalance.export_column_elements(rebalanced_elecolmap, false, false);

    ////////////////////////////////////////
    // FINISH
    ////////////////////////////////////////
    int err = dis_to_rebalance.fill_complete(false, false, false);

    if (err) FOUR_C_THROW("fill_complete() returned err={}", err);

    // print to screen
    Core::Rebalance::Utils::print_parallel_distribution(dis_to_rebalance);
  }  // if more than one proc
}  // Core::Rebalance::match_element_distribution_of_matching_discretizations


/*---------------------------------------------------------------------*
|  Rebalance Conditioned Elements Matching Template        rauch 10/16 |
*----------------------------------------------------------------------*/
void Core::Rebalance::match_element_distribution_of_matching_conditioned_elements(
    Core::FE::Discretization& dis_template, Core::FE::Discretization& dis_to_rebalance,
    const std::string& condname_template, const std::string& condname_rebalance, const bool print)
{
  // clone communicator of target discretization
  MPI_Comm com(dis_template.get_comm());
  if (Core::Communication::num_mpi_ranks(com) > 1)
  {
    // print to screen
    if (Core::Communication::my_mpi_rank(com) == 0)
    {
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "|   Match element distribution of discr. " << std::setw(11) << dis_to_rebalance.name()
          << "                          |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "|   Condition : " << std::setw(35) << condname_rebalance
                                        << "                           |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "|   to template discr. " << std::setw(11) << dis_template.name()
          << "                                            |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose) << "|   Condition : " << std::setw(35) << condname_template
                                        << "                           |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "+-----------------------------------------------------------------------------+"
          << Core::IO::endl;
    }

    // create vectors for element matching
    std::vector<int> my_template_colelegid_vec;
    std::vector<int> rebalance_rowelegid_vec;

    // create vectors for node matching
    std::vector<int> my_template_nodegid_vec;
    std::vector<int> rebalance_rownodegid_vec;
    std::vector<int> rebalance_colnodegid_vec;

    // geometry iterator
    std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator geom_it;

    // fill condition discretization by cloning scatra discretization
    std::shared_ptr<Core::FE::Discretization> dis_from_template_condition;

    Core::FE::DiscretizationCreatorBase discreator;
    std::vector<std::string> conditions_to_copy;
    dis_from_template_condition = discreator.create_matching_discretization_from_condition(
        dis_template,        ///< discretization with condition
        condname_template,   ///< name of the condition, by which the derived discretization is
                             ///< identified
        "aux_dis",           ///< name of the new discretization
        "TRANSP",            ///< name/type of the elements to be created
        conditions_to_copy,  ///< list of conditions that will be copied to the new discretization
        -1  ///< coupling id, only elements conditioned with this coupling id are considered
    );

    // get element col map of conditioned template dis
    const Core::LinAlg::Map* template_cond_dis_elecolmap =
        dis_from_template_condition->element_col_map();
    // get element row map of dis to be rebalanced
    const Core::LinAlg::Map* rebalance_elerowmap = dis_to_rebalance.element_row_map();


    ////////////////////////////////////////
    // MATCH CONDITIONED ELEMENTS
    ////////////////////////////////////////
    // fill element gid vectors
    for (int lid = 0; lid < template_cond_dis_elecolmap->num_my_elements(); lid++)
      my_template_colelegid_vec.push_back(template_cond_dis_elecolmap->gid(lid));

    for (int lid = 0; lid < rebalance_elerowmap->num_my_elements(); lid++)
      rebalance_rowelegid_vec.push_back(rebalance_elerowmap->gid(lid));


    // initialize search tree for matching with template (source,master) elements
    auto elementmatchingtree = Core::GeometricSearch::ElementMatchingOctree();
    elementmatchingtree.init(*dis_from_template_condition, my_template_colelegid_vec, 150, 1e-06);
    elementmatchingtree.setup();

    // map that will be filled with matched elements.
    // mapping: redistr. ele gid to (template ele gid, dist.).
    // note: 'fill_slave_to_master_gid_mapping' loops over all
    //        template eles and finds corresponding redistr. eles.
    std::map<int, std::vector<double>> matched_ele_map;
    // match target (slave) elements to source (master) elements using octtree
    elementmatchingtree.fill_slave_to_master_gid_mapping(
        dis_to_rebalance, rebalance_rowelegid_vec, matched_ele_map);

    // now we have a map matching the geometry ids of slave elements
    // to the geometry id of master elements (always starting from 0).
    // for redistribution we need to translate the geometry ids to the
    // actual element gids.
    // fill vectors with row and col gids for new distribution
    std::vector<int> rebalance_colelegid_vec;
    rebalance_rowelegid_vec.clear();
    for (const auto& it : matched_ele_map)
    {
      // if this proc owns the template element we also want to own
      // the element of the rebalanced discretization.
      // we also want to own all nodes of this element.
      if (static_cast<int>((it.second)[2]) == 1) rebalance_rowelegid_vec.push_back(it.first);

      rebalance_colelegid_vec.push_back(it.first);
    }

    if (print)
    {
      Core::Communication::barrier(dis_to_rebalance.get_comm());
      for (const auto& it : matched_ele_map)
      {
        std::cout << "ELEMENT : " << it.first << " ->  ( " << it.second[0] << ", " << it.second[1]
                  << ", " << it.second[2] << " )"
                  << " on PROC " << Core::Communication::my_mpi_rank(dis_to_rebalance.get_comm())
                  << " map size = " << matched_ele_map.size() << std::endl;
      }
    }


    ////////////////////////////////////////
    // ALSO APPEND UNCONDITIONED ELEMENTS
    ////////////////////////////////////////

    auto conditioned_node_ids = Conditions::find_conditioned_node_ids(
        dis_to_rebalance, condname_rebalance, Conditions::LookFor::locally_owned_and_ghosted);

    for (int lid = 0; lid < dis_to_rebalance.element_col_map()->num_my_elements(); lid++)
    {
      bool conditionedele = false;
      Core::Elements::Element* ele =
          dis_to_rebalance.g_element(dis_to_rebalance.element_col_map()->gid(lid));
      Core::Nodes::Node** nodes = ele->nodes();
      for (int node = 0; node < ele->num_node(); node++)
      {
        if (conditioned_node_ids.contains(nodes[node]->id()))
        {
          conditionedele = true;
          break;
        }
      }  // loop over nodes

      if (not conditionedele)
      {
        // append unconditioned ele id to col gid vec
        rebalance_colelegid_vec.push_back(ele->id());

        // append unconditioned ele id to row gid vec
        if (ele->owner() == Core::Communication::my_mpi_rank(com))
          rebalance_rowelegid_vec.push_back(ele->id());
      }

    }  // loop over col elements


    // construct rebalanced element row map
    Core::LinAlg::Map rebalanced_elerowmap(
        -1, rebalance_rowelegid_vec.size(), rebalance_rowelegid_vec.data(), 0, com);

    // construct rebalanced element col map
    Core::LinAlg::Map rebalanced_elecolmap(
        -1, rebalance_colelegid_vec.size(), rebalance_colelegid_vec.data(), 0, com);


    ////////////////////////////////////////
    // MATCH CONDITIONED NODES
    ////////////////////////////////////////
    // fill vector with processor local conditioned node gids for template dis
    for (int lid = 0; lid < dis_template.node_col_map()->num_my_elements(); ++lid)
    {
      if (dis_template.g_node(dis_template.node_col_map()->gid(lid))
              ->get_condition(condname_template) != nullptr)
        my_template_nodegid_vec.push_back(dis_template.node_col_map()->gid(lid));
    }

    // fill vec with processor local node gids of dis to be rebalanced
    std::vector<const Core::Conditions::Condition*> rebalance_conds;
    dis_to_rebalance.get_condition(condname_rebalance, rebalance_conds);

    for (auto* const rebalance_cond : rebalance_conds)
    {
      const std::vector<int>* rebalance_cond_nodes = rebalance_cond->get_nodes();
      for (int rebalance_cond_node : *rebalance_cond_nodes)
      {
        if (dis_to_rebalance.have_global_node(rebalance_cond_node))
          if (dis_to_rebalance.g_node(rebalance_cond_node)->owner() ==
              Core::Communication::my_mpi_rank(com))
            rebalance_rownodegid_vec.push_back(rebalance_cond_node);
      }
    }

    // initialize search tree for matching with template (source) nodes
    auto nodematchingtree = Core::GeometricSearch::NodeMatchingOctree();
    nodematchingtree.init(dis_template, my_template_nodegid_vec, 150, 1e-06);
    nodematchingtree.setup();

    // map that will be filled with matched nodes.
    // mapping: redistr. node gid to (template node gid, dist.).
    // note: FindMatch loops over all template nodes
    //       and finds corresponding redistr. nodes.
    std::map<int, std::vector<double>> matched_node_map;
    // match target nodes to source nodes using octtree
    nodematchingtree.fill_slave_to_master_gid_mapping(
        dis_to_rebalance, rebalance_rownodegid_vec, matched_node_map);

    // fill vectors with row gids for new distribution
    rebalance_rownodegid_vec.clear();
    // std::vector<int> rebalance_colnodegid_vec;
    for (const auto& it : matched_node_map)
    {
      // if this proc owns the template node we also want to own
      // the node of the rebalanced discretization
      if (static_cast<int>((it.second)[2]) == 1) rebalance_rownodegid_vec.push_back(it.first);

      rebalance_colnodegid_vec.push_back(it.first);
    }
    if (print)
    {
      Core::Communication::barrier(dis_to_rebalance.get_comm());
      for (const auto& it : matched_node_map)
      {
        std::cout << "NODE : " << it.first << " ->  ( " << it.second[0] << ", " << it.second[1]
                  << ", " << it.second[2] << " )"
                  << " on PROC " << Core::Communication::my_mpi_rank(dis_to_rebalance.get_comm())
                  << " map size = " << matched_node_map.size() << std::endl;
      }
    }


    ////////////////////////////////////////
    // ALSO APPEND UNCONDITIONED  NODES
    ////////////////////////////////////////
    // add row nodes
    for (int lid = 0; lid < dis_to_rebalance.node_row_map()->num_my_elements(); lid++)
    {
      Core::Conditions::Condition* testcond =
          dis_to_rebalance.g_node(dis_to_rebalance.node_row_map()->gid(lid))
              ->get_condition(condname_rebalance);
      if (testcond == nullptr)
        rebalance_rownodegid_vec.push_back(
            dis_to_rebalance.g_node(dis_to_rebalance.node_row_map()->gid(lid))->id());
    }
    // add col nodes
    for (int lid = 0; lid < dis_to_rebalance.node_col_map()->num_my_elements(); lid++)
    {
      Core::Conditions::Condition* testcond =
          dis_to_rebalance.g_node(dis_to_rebalance.node_col_map()->gid(lid))
              ->get_condition(condname_rebalance);
      if (testcond == nullptr)
        rebalance_colnodegid_vec.push_back(
            dis_to_rebalance.g_node(dis_to_rebalance.node_col_map()->gid(lid))->id());
    }

    // construct rebalanced node row map
    Core::LinAlg::Map rebalanced_noderowmap(
        -1, rebalance_rownodegid_vec.size(), rebalance_rownodegid_vec.data(), 0, com);

    // construct rebalanced node col map
    Core::LinAlg::Map rebalanced_nodecolmap(
        -1, rebalance_colnodegid_vec.size(), rebalance_colnodegid_vec.data(), 0, com);


    ////////////////////////////////////////
    // REBALANCE
    ////////////////////////////////////////
    // export the nodes
    dis_to_rebalance.export_row_nodes(rebalanced_noderowmap, false, false);
    dis_to_rebalance.export_column_nodes(rebalanced_nodecolmap, false, false);
    // export the elements
    dis_to_rebalance.export_row_elements(rebalanced_elerowmap, false, false);
    dis_to_rebalance.export_column_elements(rebalanced_elecolmap, false, false);


    ////////////////////////////////////////
    // FINISH
    ////////////////////////////////////////
    int err = dis_to_rebalance.fill_complete(false, false, false);

    if (err) FOUR_C_THROW("fill_complete() returned err={}", err);

    // print to screen
    Core::Rebalance::Utils::print_parallel_distribution(dis_to_rebalance);

  }  // if more than one proc
}  // match_element_distribution_of_matching_conditioned_elements

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> Core::Rebalance::get_col_version_of_row_vector(
    const Core::FE::Discretization& dis,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> state, const int nds)
{
  // note that this routine has the same functionality as set_state,
  // although here we do not store the new vector anywhere
  // maybe this routine can be used in set_state or become a member function of the discretization
  // class

  if (!dis.have_dofs()) FOUR_C_THROW("fill_complete() was not called");
  const Core::LinAlg::Map* colmap = dis.dof_col_map(nds);
  const Core::LinAlg::Map& vecmap = state->get_map();

  // if it's already in column map just set a reference
  // This is a rought test, but it might be ok at this place. It is an
  // error anyway to hand in a vector that is not related to our dof
  // maps.
  if (vecmap.point_same_as(colmap->get_epetra_block_map())) return state;
  // if it's not in column map export and allocate
  else
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp = Core::LinAlg::create_vector(*colmap, false);
    Core::LinAlg::export_to(*state, *tmp);
    return tmp;
  }
}  // get_col_version_of_row_vector

/*----------------------------------------------------------------------*
 |(private)                                                   tk 06/10  |
 |recompute nodecolmap of standard discretization to include all        |
 |nodes as of subdicretization                                          |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Core::Rebalance::compute_node_col_map(
    const Core::FE::Discretization& sourcedis,  ///< standard discretization we want to rebalance
    const Core::FE::Discretization& subdis      ///< subdiscretization prescribing ghosting
)
{
  const Core::LinAlg::Map* oldcolnodemap = sourcedis.node_col_map();

  std::vector<int> mycolnodes(oldcolnodemap->num_my_elements());
  oldcolnodemap->my_global_elements(mycolnodes.data());
  for (int inode = 0; inode != subdis.num_my_col_nodes(); ++inode)
  {
    const Core::Nodes::Node* newnode = subdis.l_col_node(inode);
    const int gid = newnode->id();
    if (!(sourcedis.have_global_node(gid)))
    {
      mycolnodes.push_back(gid);
    }
  }

  // now reconstruct the extended colmap
  std::shared_ptr<Core::LinAlg::Map> newcolnodemap = std::make_shared<Core::LinAlg::Map>(
      -1, mycolnodes.size(), mycolnodes.data(), 0, sourcedis.get_comm());
  return newcolnodemap;
}  // Core::Rebalance::ComputeNodeColMap

/*----------------------------------------------------------------------*
 *                                                          rauch 10/16 |
 *----------------------------------------------------------------------*/
void Core::Rebalance::match_element_row_col_distribution(
    const Core::FE::Discretization& dis_template, const Core::FE::Discretization& dis_to_rebalance,
    std::vector<int>& row_id_vec_to_fill, std::vector<int>& col_id_vec_to_fill)
{
  // preliminary work
  const Core::LinAlg::Map* rebalance_elerowmap = dis_to_rebalance.element_row_map();
  const Core::LinAlg::Map* template_elecolmap = dis_template.element_col_map();
  std::vector<int> my_template_elegid_vec(template_elecolmap->num_my_elements());
  std::vector<int> my_rebalance_elegid_vec;

  // fill vector with processor local ele gids for template dis
  for (int lid = 0; lid < template_elecolmap->num_my_elements(); ++lid)
    my_template_elegid_vec[lid] = template_elecolmap->gid(lid);

  // fill vec with processor local ele gids of dis to be rebalanced
  for (int lid = 0; lid < rebalance_elerowmap->num_my_elements(); ++lid)
    my_rebalance_elegid_vec.push_back(rebalance_elerowmap->gid(lid));

  // initialize search tree for matching with template (source,master) elements
  auto elementmatchingtree = Core::GeometricSearch::ElementMatchingOctree();
  elementmatchingtree.init(dis_template, my_template_elegid_vec, 150, 1e-07);
  elementmatchingtree.setup();

  // map that will be filled with matched elements.
  // mapping: redistr. ele gid to (template ele gid, dist.).
  // note: 'fill_slave_to_master_gid_mapping' loops over all
  //        template eles and finds corresponding redistr. eles.
  std::map<int, std::vector<double>> matched_ele_map;
  // match target (slave) nodes to source (master) nodes using octtree
  elementmatchingtree.fill_slave_to_master_gid_mapping(
      dis_to_rebalance, my_rebalance_elegid_vec, matched_ele_map);

  // declare iterator
  std::map<int, std::vector<double>>::iterator it;

  // fill vectors with row and col gids for new distribution
  for (it = matched_ele_map.begin(); it != matched_ele_map.end(); ++it)
  {
    // if this proc owns the template element we also want to own
    // the element of the rebalanced discretization.
    // we also want to own all nodes of this element.
    if (static_cast<int>((it->second)[2]) == 1) row_id_vec_to_fill.push_back(it->first);

    col_id_vec_to_fill.push_back(it->first);
  }
}  // Core::Rebalance::MatchElementRowColDistribution

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Core::Rebalance::match_nodal_row_col_distribution(const Core::FE::Discretization& dis_template,
    const Core::FE::Discretization& dis_to_rebalance, std::vector<int>& row_id_vec_to_fill,
    std::vector<int>& col_id_vec_to_fill)
{
  // temp sets
  std::set<int> temprowset;
  std::set<int> tempcolset;

  for (int& row_id_to_fill : row_id_vec_to_fill) temprowset.insert(row_id_to_fill);

  for (int& col_id_to_fill : col_id_vec_to_fill) tempcolset.insert(col_id_to_fill);

  // preliminary work
  const Core::LinAlg::Map* rebalance_noderowmap = dis_to_rebalance.node_row_map();
  const Core::LinAlg::Map* template_nodecolmap = dis_template.node_col_map();
  std::vector<int> my_template_nodegid_vec(template_nodecolmap->num_my_elements());
  std::vector<int> my_rebalance_nodegid_vec;

  // fill vector with processor local node gids for template dis
  for (int lid = 0; lid < template_nodecolmap->num_my_elements(); ++lid)
    my_template_nodegid_vec[lid] = template_nodecolmap->gid(lid);

  // fill vec with processor local node gids of dis to be rebalanced
  for (int lid = 0; lid < rebalance_noderowmap->num_my_elements(); ++lid)
    my_rebalance_nodegid_vec.push_back(rebalance_noderowmap->gid(lid));

  // initialize search tree for matching with template (source) nodes
  auto nodematchingtree = Core::GeometricSearch::NodeMatchingOctree();
  nodematchingtree.init(dis_template, my_template_nodegid_vec, 150, 1e-07);
  nodematchingtree.setup();

  // map that will be filled with matched nodes.
  // mapping: redistr. node gid to (template node gid, dist.).
  // note: FindMatch loops over all template nodes
  //       and finds corresponding redistr. nodes.
  std::map<int, std::vector<double>> matched_node_map;
  // match target nodes to source nodes using octtree
  nodematchingtree.fill_slave_to_master_gid_mapping(
      dis_to_rebalance, my_rebalance_nodegid_vec, matched_node_map);

  // declare iterator
  std::map<int, std::vector<double>>::iterator it;

  // fill vectors with row gids for new distribution
  // std::vector<int> rebalance_colnodegid_vec;
  for (it = matched_node_map.begin(); it != matched_node_map.end(); ++it)
  {
    // if this proc owns the template node we also want to own
    // the node of the rebalanced discretization
    if (static_cast<int>((it->second)[2]) == 1) temprowset.insert(it->first);

    tempcolset.insert(it->first);
  }

  // assign temporary sets to vectors
  row_id_vec_to_fill.clear();
  row_id_vec_to_fill.reserve(temprowset.size());
  row_id_vec_to_fill.assign(temprowset.begin(), temprowset.end());

  col_id_vec_to_fill.clear();
  col_id_vec_to_fill.reserve(tempcolset.size());
  col_id_vec_to_fill.assign(tempcolset.begin(), tempcolset.end());
}  // Core::Rebalance::MatchNodalRowColDistribution

FOUR_C_NAMESPACE_CLOSE
