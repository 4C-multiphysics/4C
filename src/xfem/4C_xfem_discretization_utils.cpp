// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_discretization_utils.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_dofset_fixed_size.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_rebalance_print.hpp"
#include "4C_xfem_discretization.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::Utils::print_discretization_to_stream(std::shared_ptr<Core::FE::Discretization> dis,
    const std::string& disname, bool elements, bool elecol, bool nodes, bool nodecol, bool faces,
    bool facecol, std::ostream& s, std::map<int, Core::LinAlg::Matrix<3, 1>>* curr_pos)
{
  if (elements)
  {
    // draw bg elements with associated gid
    s << "View \" " << disname;
    if (elecol)
    {
      s << " col e->Id() \" {\n";
      for (int i = 0; i < dis->num_my_col_elements(); ++i)
      {
        const Core::Elements::Element* actele = dis->l_col_element(i);
        if (curr_pos == nullptr)
          Core::IO::Gmsh::element_at_initial_position_to_stream(double(actele->id()), actele, s);
        else
          Core::IO::Gmsh::element_at_current_position_to_stream(
              double(actele->id()), actele, *curr_pos, s);
      };
    }
    else
    {
      s << " row e->Id() \" {\n";
      for (int i = 0; i < dis->num_my_row_elements(); ++i)
      {
        const Core::Elements::Element* actele = dis->l_row_element(i);
        if (curr_pos == nullptr)
          Core::IO::Gmsh::element_at_initial_position_to_stream(double(actele->id()), actele, s);
        else
          Core::IO::Gmsh::element_at_current_position_to_stream(
              double(actele->id()), actele, *curr_pos, s);
      };
    }
    s << "};\n";
  }

  if (nodes)
  {
    s << "View \" " << disname;
    if (nodecol)
    {
      s << " col n->Id() \" {\n";
      for (int i = 0; i < dis->num_my_col_nodes(); ++i)
      {
        const Core::Nodes::Node* actnode = dis->l_col_node(i);
        Core::LinAlg::Matrix<3, 1> pos(true);

        if (curr_pos != nullptr)
        {
          const Core::LinAlg::Matrix<3, 1>& curr_x = curr_pos->find(actnode->id())->second;
          pos(0) = curr_x(0);
          pos(1) = curr_x(1);
          pos(2) = curr_x(2);
        }
        else
        {
          const Core::LinAlg::Matrix<3, 1> x(actnode->x().data());
          pos(0) = x(0);
          pos(1) = x(1);
          pos(2) = x(2);
        }
        Core::IO::Gmsh::cell_with_scalar_to_stream(
            Core::FE::CellType::point1, actnode->id(), pos, s);
      }
    }
    else
    {
      s << " row n->Id() \" {\n";
      for (int i = 0; i < dis->num_my_row_nodes(); ++i)
      {
        const Core::Nodes::Node* actnode = dis->l_row_node(i);
        Core::LinAlg::Matrix<3, 1> pos(true);

        if (curr_pos != nullptr)
        {
          const Core::LinAlg::Matrix<3, 1>& curr_x = curr_pos->find(actnode->id())->second;
          pos(0) = curr_x(0);
          pos(1) = curr_x(1);
          pos(2) = curr_x(2);
        }
        else
        {
          const Core::LinAlg::Matrix<3, 1> x(actnode->x().data());
          pos(0) = x(0);
          pos(1) = x(1);
          pos(2) = x(2);
        }
        Core::IO::Gmsh::cell_with_scalar_to_stream(
            Core::FE::CellType::point1, actnode->id(), pos, s);
      }
    }
    s << "};\n";
  }

  if (faces)
  {
    // cast to DiscretizationXFEM
    std::shared_ptr<Core::FE::DiscretizationFaces> xdis =
        std::dynamic_pointer_cast<Core::FE::DiscretizationFaces>(dis);
    if (xdis == nullptr)
      FOUR_C_THROW(
          "Failed to cast Core::FE::Discretization to "
          "Core::FE::DiscretizationFaces.");

    s << "View \" " << disname;

    if (xdis->filled_extension() == true)  // faces output
    {
      if (facecol)
      {
        s << " col f->Id() \" {\n";
        for (int i = 0; i < xdis->num_my_col_faces(); ++i)
        {
          const Core::Elements::Element* actele = xdis->l_col_face(i);
          if (curr_pos == nullptr)
            Core::IO::Gmsh::element_at_initial_position_to_stream(double(actele->id()), actele, s);
          else
            Core::IO::Gmsh::element_at_current_position_to_stream(
                double(actele->id()), actele, *curr_pos, s);
        };
      }
      else
      {
        s << " row f->Id() \" {\n";
        for (int i = 0; i < xdis->num_my_row_faces(); ++i)
        {
          const Core::Elements::Element* actele = xdis->l_row_face(i);
          if (curr_pos == nullptr)
            Core::IO::Gmsh::element_at_initial_position_to_stream(double(actele->id()), actele, s);
          else
            Core::IO::Gmsh::element_at_current_position_to_stream(
                double(actele->id()), actele, *curr_pos, s);
        };
      }
      s << "};\n";
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::Utils::XFEMDiscretizationBuilder::setup_xfem_discretization(
    const Teuchos::ParameterList& xgen_params, std::shared_ptr<Core::FE::Discretization> dis,
    int numdof) const
{
  std::shared_ptr<XFEM::DiscretizationXFEM> xdis =
      std::dynamic_pointer_cast<XFEM::DiscretizationXFEM>(dis);
  //
  if (xdis == nullptr)
  {
    FOUR_C_THROW("No XFEM discretization for XFEM problem available!");

    // REMARK: standard fluid could also step into this routine, as a special case! (remove
    // FOUR_C_THROW)
    if (!dis->filled()) dis->fill_complete();

    return;
  }

  if (!xdis->filled()) xdis->fill_complete();

  const Epetra_Map* noderowmap = xdis->node_row_map();
  if (noderowmap == nullptr) FOUR_C_THROW("we expect a fill-complete call before!");

  // now we can reserve dofs for xfem discretization
  int nodeindexrange =
      noderowmap->MaxAllGID() - noderowmap->MinAllGID() + 1;  // if id's are not continuous numbered
  int maxNumMyReservedDofsperNode = (xgen_params.get<int>("MAX_NUM_DOFSETS")) * numdof;
  std::shared_ptr<Core::DOFSets::FixedSizeDofSet> maxdofset =
      std::make_shared<Core::DOFSets::FixedSizeDofSet>(maxNumMyReservedDofsperNode, nodeindexrange);

  const int fluid_nds = 0;
  xdis->replace_dof_set(fluid_nds, maxdofset, true);  // fluid dofset has nds = 0
  std::vector<int> nds;
  nds.push_back(fluid_nds);
  xdis->initial_fill_complete(nds);

  // print all dofsets
  xdis->get_dof_set_proxy()->print_all_dofsets(xdis->get_comm());

  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::Utils::XFEMDiscretizationBuilder::setup_xfem_discretization(
    const Teuchos::ParameterList& xgen_params, std::shared_ptr<Core::FE::Discretization> dis,
    Core::FE::Discretization& embedded_dis, const std::string& embedded_cond_name, int numdof) const
{
  if (!embedded_dis.filled()) embedded_dis.fill_complete();

  std::shared_ptr<XFEM::DiscretizationXFEM> xdis =
      std::dynamic_pointer_cast<XFEM::DiscretizationXFEM>(dis);
  if (!xdis->filled()) xdis->fill_complete();

  // get fluid mesh conditions: hereby we specify standalone embedded discretizations
  std::vector<Core::Conditions::Condition*> conditions;
  xdis->get_condition(embedded_cond_name, conditions);

  std::vector<std::string> conditions_to_copy;
  xdis->get_condition_names(conditions_to_copy);

  split_discretization_by_condition(*xdis, embedded_dis, conditions, conditions_to_copy);

  setup_xfem_discretization(xgen_params, xdis, numdof);

  Core::Rebalance::Utils::print_parallel_distribution(*dis);
  Core::Rebalance::Utils::print_parallel_distribution(embedded_dis);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int XFEM::Utils::XFEMDiscretizationBuilder::setup_xfem_discretization(
    const Teuchos::ParameterList& xgen_params, std::shared_ptr<Core::FE::Discretization> src_dis,
    std::shared_ptr<Core::FE::Discretization> target_dis,
    const std::vector<Core::Conditions::Condition*>& boundary_conds) const
{
  if (!target_dis->filled()) target_dis->fill_complete();

  if (!src_dis->filled()) src_dis->fill_complete();

  // get the number of DoF's per node
  int gid_node = src_dis->node_row_map()->MinMyGID();
  Core::Nodes::Node* node_ptr = src_dis->g_node(gid_node);
  int num_dof_per_node = src_dis->num_dof(node_ptr);

  std::vector<std::string> conditions_to_copy;
  src_dis->get_condition_names(conditions_to_copy);

  split_discretization_by_boundary_condition(
      *src_dis, *target_dis, boundary_conds, conditions_to_copy);

  if (std::dynamic_pointer_cast<XFEM::DiscretizationXFEM>(src_dis))
    setup_xfem_discretization(xgen_params, src_dis, num_dof_per_node);
  if (std::dynamic_pointer_cast<XFEM::DiscretizationXFEM>(target_dis))
    setup_xfem_discretization(xgen_params, target_dis, num_dof_per_node);

  Core::Rebalance::Utils::print_parallel_distribution(*src_dis);
  Core::Rebalance::Utils::print_parallel_distribution(*target_dis);

  return num_dof_per_node;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::Utils::XFEMDiscretizationBuilder::split_discretization_by_condition(
    Core::FE::Discretization& sourcedis, Core::FE::Discretization& targetdis,
    std::vector<Core::Conditions::Condition*>& conditions,
    const std::vector<std::string>& conditions_to_copy) const
{
  // row node map (id -> pointer)
  std::map<int, Core::Nodes::Node*> sourcenodes;

  // column node map
  std::map<int, Core::Nodes::Node*> sourcegnodes;

  // element map
  std::map<int, std::shared_ptr<Core::Elements::Element>> sourceelements;

  // find conditioned nodes (owned and ghosted) and elements
  Core::Conditions::find_condition_objects(
      sourcedis, sourcenodes, sourcegnodes, sourceelements, conditions);

  split_discretization(
      sourcedis, targetdis, sourcenodes, sourcegnodes, sourceelements, conditions_to_copy);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::Utils::XFEMDiscretizationBuilder::split_discretization(
    Core::FE::Discretization& sourcedis, Core::FE::Discretization& targetdis,
    const std::map<int, Core::Nodes::Node*>& sourcenodes,
    const std::map<int, Core::Nodes::Node*>& sourcegnodes,
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& sourceelements,
    const std::vector<std::string>& conditions_to_copy) const
{
  if (!sourcedis.filled()) FOUR_C_THROW("sourcedis is not filled");
  const int myrank = Core::Communication::my_mpi_rank(targetdis.get_comm());

  const int numothernoderow = sourcedis.num_my_row_nodes();
  const int numothernodecol = sourcedis.num_my_col_nodes();

  // add the conditioned elements
  for (std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator sourceele_iter =
           sourceelements.begin();
      sourceele_iter != sourceelements.end(); ++sourceele_iter)
  {
    if (sourceele_iter->second->owner() == myrank)
    {
      targetdis.add_element(Core::Utils::shared_ptr_from_ref(*sourceele_iter->second->clone()));
    }
  }

  // row/col sets of conditioned node ids
  std::set<int> condnoderowset;
  std::set<int> condnodecolset;
  // row/col vectors of target node ids
  std::vector<int> targetnoderowvec;
  targetnoderowvec.reserve(sourcenodes.size());
  std::vector<int> targetnodecolvec;
  targetnodecolvec.reserve(sourcegnodes.size());

  // ------------------------------------------------------------------------
  // add conditioned nodes and fill the id vectors
  // ------------------------------------------------------------------------
  for (std::map<int, Core::Nodes::Node*>::const_iterator sourcegnode_iter = sourcegnodes.begin();
      sourcegnode_iter != sourcegnodes.end(); ++sourcegnode_iter)
  {
    const int nid = sourcegnode_iter->first;
    if (sourcegnode_iter->second->owner() == myrank)
    {
      std::shared_ptr<Core::Nodes::Node> sourcegnode =
          std::make_shared<Core::Nodes::Node>(nid, sourcegnode_iter->second->x(), myrank);
      targetdis.add_node(sourcegnode);
      condnoderowset.insert(nid);
      targetnoderowvec.push_back(nid);
    }
    condnodecolset.insert(nid);
    targetnodecolvec.push_back(nid);
  }

  // ------------------------------------------------------------------------
  // copy selected conditions to the new discretization
  // ------------------------------------------------------------------------
  for (std::vector<std::string>::const_iterator conditername = conditions_to_copy.begin();
      conditername != conditions_to_copy.end(); ++conditername)
  {
    std::vector<Core::Conditions::Condition*> conds;
    sourcedis.get_condition(*conditername, conds);
    for (unsigned i = 0; i < conds.size(); ++i)
    {
      std::shared_ptr<Core::Conditions::Condition> cond_to_copy =
          split_condition(conds[i], targetnodecolvec, targetdis.get_comm());
      if (cond_to_copy) targetdis.set_condition(*conditername, cond_to_copy);
    }
  }

  redistribute(targetdis, targetnoderowvec, targetnodecolvec);

  // ------------------------------------------------------------------------
  // remove all nodes from the condnodecol and condnoderow sets, which also
  // belong to a not deleted source element
  // ------------------------------------------------------------------------
  for (unsigned j = 0; j < static_cast<unsigned>(sourcedis.num_my_col_elements()); ++j)
  {
    int source_ele_gid = sourcedis.element_col_map()->GID(j);
    // continue, if we are going to delete this element
    if (sourceelements.find(source_ele_gid) != sourceelements.end()) continue;
    Core::Elements::Element* source_ele = sourcedis.g_element(source_ele_gid);
    const int* nid = source_ele->node_ids();
    for (unsigned i = 0; i < static_cast<unsigned>(source_ele->num_node()); ++i)
    {
      // Remove all nodes from the condition sets, which should stay in
      // the source discretization, since they belong to elements
      // which are not going to be deleted!
      std::set<int>::iterator pos = condnodecolset.find(nid[i]);
      if (pos != condnodecolset.end()) condnodecolset.erase(pos);
      pos = condnoderowset.find(nid[i]);
      if (pos != condnoderowset.end()) condnoderowset.erase(pos);
    }
  }

  // row/col vectors of non-conditioned node ids
  std::vector<int> othernoderowvec;
  othernoderowvec.reserve(numothernoderow - condnoderowset.size());
  std::vector<int> othernodecolvec;
  othernodecolvec.reserve(numothernodecol - condnodecolset.size());

  // determine non-conditioned nodes
  for (int lid = 0; lid < sourcedis.node_col_map()->NumMyElements(); ++lid)
  {
    const int nid = sourcedis.node_col_map()->GID(lid);

    // if we erase this node, we do not add it and just go on
    if (condnodecolset.find(nid) != condnodecolset.end()) continue;

    othernodecolvec.push_back(nid);

    if (sourcedis.node_row_map()->LID(nid) > -1) othernoderowvec.push_back(nid);
  }
  // delete conditioned nodes, which are not connected to any unconditioned elements
  for (std::set<int>::iterator it = condnodecolset.begin(); it != condnodecolset.end(); ++it)
    if (not sourcedis.delete_node(*it)) FOUR_C_THROW("Node %d could not be deleted!", *it);

  // delete conditioned elements from source discretization
  for (std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator sourceele_iter =
           sourceelements.begin();
      sourceele_iter != sourceelements.end(); ++sourceele_iter)
  {
    sourcedis.delete_element(sourceele_iter->first);
  }

  // ------------------------------------------------------------------------
  // validate the source conditions
  // ------------------------------------------------------------------------
  std::vector<std::string> src_conditions;
  sourcedis.get_condition_names(src_conditions);
  for (std::vector<std::string>::const_iterator conditername = src_conditions.begin();
      conditername != src_conditions.end(); ++conditername)
  {
    std::vector<Core::Conditions::Condition*> conds;
    sourcedis.get_condition(*conditername, conds);
    std::vector<std::shared_ptr<Core::Conditions::Condition>> src_conds(conds.size(), nullptr);
    for (unsigned i = 0; i < conds.size(); ++i)
      src_conds[i] = split_condition(conds[i], othernodecolvec, sourcedis.get_comm());
    sourcedis.replace_conditions(*conditername, src_conds);
  }
  // re-partitioning
  redistribute(sourcedis, othernoderowvec, othernodecolvec);


  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::Utils::XFEMDiscretizationBuilder::redistribute(
    Core::FE::Discretization& dis, std::vector<int>& noderowvec, std::vector<int>& nodecolvec) const
{
  dis.check_filled_globally();

  MPI_Comm comm(dis.get_comm());

  std::shared_ptr<Epetra_Map> noderowmap = std::make_shared<Epetra_Map>(
      -1, noderowvec.size(), noderowvec.data(), 0, Core::Communication::as_epetra_comm(comm));

  std::shared_ptr<Epetra_Map> nodecolmap = std::make_shared<Epetra_Map>(
      -1, nodecolvec.size(), nodecolvec.data(), 0, Core::Communication::as_epetra_comm(comm));
  if (!dis.filled()) dis.redistribute(*noderowmap, *nodecolmap);

  Epetra_Map elerowmap(*dis.element_row_map());
  std::shared_ptr<const Core::LinAlg::Graph> nodegraph =
      Core::Rebalance::build_graph(dis, elerowmap);

  Teuchos::ParameterList rebalanceParams;
  rebalanceParams.set("num parts", std::to_string(Core::Communication::num_mpi_ranks(comm)));
  std::tie(noderowmap, nodecolmap) =
      Core::Rebalance::rebalance_node_maps(*nodegraph, rebalanceParams);

  auto const& [roweles, coleles] = dis.build_element_row_column(*noderowmap, *nodecolmap);

  dis.export_row_nodes(*noderowmap);
  dis.export_row_elements(*roweles);

  dis.export_column_nodes(*nodecolmap);
  dis.export_column_elements(*coleles);

  dis.fill_complete();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XFEM::Utils::XFEMDiscretizationBuilder::split_discretization_by_boundary_condition(
    Core::FE::Discretization& sourcedis, Core::FE::Discretization& targetdis,
    const std::vector<Core::Conditions::Condition*>& boundary_conds,
    const std::vector<std::string>& conditions_to_copy) const
{
  if (not sourcedis.filled()) FOUR_C_THROW("sourcedis is not filled");
  const int myrank = Core::Communication::my_mpi_rank(targetdis.get_comm());

  // element map
  std::map<int, std::shared_ptr<Core::Elements::Element>> src_cond_elements;

  // find conditioned nodes (owned and ghosted) and elements
  Core::Conditions::find_condition_objects(src_cond_elements, boundary_conds);

  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator cit;
  std::map<int, std::shared_ptr<Core::Elements::Element>> src_elements;
  // row node map (id -> pointer)
  std::map<int, Core::Nodes::Node*> src_my_gnodes;
  std::vector<int> condnoderowvec;
  // column node map
  std::map<int, Core::Nodes::Node*> src_gnodes;
  std::vector<int> condnodecolvec;
  // find all parent elements
  for (cit = src_cond_elements.begin(); cit != src_cond_elements.end(); ++cit)
  {
    Core::Elements::FaceElement* src_face_element =
        dynamic_cast<Core::Elements::FaceElement*>(cit->second.get());
    if (src_face_element == nullptr)
      FOUR_C_THROW("Dynamic cast failed! The src element %d is no Core::Elements::FaceElement!",
          cit->second->id());
    // get the parent element
    Core::Elements::Element* src_ele = src_face_element->parent_element();
    int src_ele_gid = src_face_element->parent_element_id();
    src_elements[src_ele_gid] = Core::Utils::shared_ptr_from_ref(*src_ele);
    const int* n = src_ele->node_ids();
    for (unsigned i = 0; i < static_cast<unsigned>(src_ele->num_node()); ++i)
    {
      const int gid = n[i];
      if (sourcedis.have_global_node(gid))
      {
        Core::Nodes::Node* node = sourcedis.g_node(gid);
        src_gnodes[gid] = node;

        if (node->owner() == myrank) src_my_gnodes[gid] = node;
      }
      else
        FOUR_C_THROW("All nodes of known elements must be known!");
    }
  }

  split_discretization(
      sourcedis, targetdis, src_my_gnodes, src_gnodes, src_elements, conditions_to_copy);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::Conditions::Condition>
XFEM::Utils::XFEMDiscretizationBuilder::split_condition(const Core::Conditions::Condition* src_cond,
    const std::vector<int>& nodecolvec, MPI_Comm comm) const
{
  const std::vector<int>* cond_node_gids = src_cond->get_nodes();
  std::set<int> nodecolset;
  nodecolset.insert(nodecolvec.begin(), nodecolvec.end());

  int lcount = 0;
  int gcount = 0;
  for (unsigned i = 0; i < cond_node_gids->size(); ++i)
  {
    int ngid = cond_node_gids->at(i);
    // add the node GID, if it is also a part of the new discretization
    if (nodecolset.find(ngid) != nodecolset.end()) lcount++;
  }

  Core::Communication::sum_all(&lcount, &gcount, 1, comm);
  // return a nullptr pointer, if there is nothing to copy
  if (gcount == 0) return nullptr;

  // copy and keep this src condition
  return src_cond->copy_without_geometry();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
XFEM::DiscretizationXWall::DiscretizationXWall(
    const std::string name, MPI_Comm comm, const unsigned int n_dim)
    : DiscretizationFaces(name, comm, n_dim)  // use base class constructor
{};

FOUR_C_NAMESPACE_CLOSE
