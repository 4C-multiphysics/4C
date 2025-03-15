// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_interface.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  store the required ghosting within a round              farah 10/13 |
 |  robin iteration for current interface (public)                      |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::round_robin_extend_ghosting(bool firstevaluation)
{
  std::vector<int> element_GIDs_to_be_ghosted;
  std::vector<int> node_GIDs_to_be_ghosted;
  for (int k = 0; k < slave_col_elements()->NumMyElements(); ++k)
  {
    int gid = slave_col_elements()->GID(k);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find ele with gid %i", gid);
    Element* slave_ele = dynamic_cast<Element*>(ele);

    for (int j = 0; j < slave_ele->mo_data().num_search_elements(); ++j)
    {
      int gid2 = slave_ele->mo_data().search_elements()[j];
      Core::Elements::Element* ele2 = idiscret_->g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find master element with gid %", gid2);
      Element* melement = dynamic_cast<Element*>(ele2);

      element_GIDs_to_be_ghosted.push_back(melement->id());

      for (int z = 0; z < melement->num_node(); ++z)
      {
        int gidn = melement->node_ids()[z];
        node_GIDs_to_be_ghosted.push_back(gidn);
      }
    }
    // reset found elements
    slave_ele->delete_search_elements();
  }

  std::shared_ptr<Epetra_Map> currently_ghosted_elements =
      std::make_shared<Epetra_Map>(-1, (int)element_GIDs_to_be_ghosted.size(),
          element_GIDs_to_be_ghosted.data(), 0, Core::Communication::as_epetra_comm(get_comm()));
  std::shared_ptr<Epetra_Map> currently_ghosted_nodes =
      std::make_shared<Epetra_Map>(-1, (int)node_GIDs_to_be_ghosted.size(),
          node_GIDs_to_be_ghosted.data(), 0, Core::Communication::as_epetra_comm(get_comm()));

  if (firstevaluation)
  {
    eextendedghosting_ = std::make_shared<Epetra_Map>(*currently_ghosted_elements);
    nextendedghosting_ = std::make_shared<Epetra_Map>(*currently_ghosted_nodes);
  }
  else
  {
    eextendedghosting_ =
        Core::LinAlg::merge_map(eextendedghosting_, currently_ghosted_elements, true);
    nextendedghosting_ = Core::LinAlg::merge_map(nextendedghosting_, currently_ghosted_nodes, true);
  }

  return;
}

/*----------------------------------------------------------------------*
 | perform the ownership change within a round robin        farah 10/13 |
 | iteration                                                            |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::round_robin_change_ownership()
{
  // pack/unpack friction nodes is only required for wear problems
  // so we should create a slightly redundant function for the wear-
  // interface and exclude the friction node packing from here

  // get friction type
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");

  // change master-side proc ownership
  // some local variables
  MPI_Comm comm_v(get_comm());
  const int myrank = Core::Communication::my_mpi_rank(comm_v);
  const int numproc = Core::Communication::num_mpi_ranks(comm_v);
  const int torank = (myrank + 1) % numproc;              // to
  const int fromrank = (myrank + numproc - 1) % numproc;  // from

  // new elements, new nodes
  std::vector<int> ncol, nrow;
  std::vector<int> ecol, erow;

  // create dummy
  Epetra_Map MasterColNodesdummy(*master_col_nodes());
  Epetra_Map MasterColelesdummy(*master_col_elements());

  // create origin maps
  std::shared_ptr<Epetra_Map> SCN = std::make_shared<Epetra_Map>(*slave_col_nodes());
  std::shared_ptr<Epetra_Map> SCE = std::make_shared<Epetra_Map>(*slave_col_elements());
  std::shared_ptr<Epetra_Map> SRN = std::make_shared<Epetra_Map>(*slave_row_nodes());
  std::shared_ptr<Epetra_Map> SRE = std::make_shared<Epetra_Map>(*slave_row_elements());

  // *****************************************
  // Elements
  // *****************************************
  std::vector<char> sdataeles;
  std::vector<char> rdataeles;

  // vector containing all proc ids
  std::vector<int> allproc(numproc);
  for (int i = 0; i < numproc; ++i) allproc[i] = i;

  // get exporter
  Core::Communication::Exporter exporter(idiscret_->get_comm());

  // create data buffer
  Core::Communication::PackBuffer dataeles;

  // now pack/store
  for (int i = 0; i < (int)MasterColelesdummy.NumMyElements(); ++i)
  {
    int gid = MasterColelesdummy.GID(i);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find ele with gid %i", gid);
    Mortar::Element* mele = dynamic_cast<Mortar::Element*>(ele);

    mele->pack(dataeles);

    const bool locally_owned = mele->owner() == myrank;
    add_to_pack(dataeles, locally_owned);
  }
  std::swap(sdataeles, dataeles());

  // delete the elements from discretization
  for (int i = 0; i < (int)MasterColelesdummy.NumMyElements(); ++i)
  {
    int gid = MasterColelesdummy.GID(i);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find ele with gid %i", gid);
    Mortar::Element* mele = dynamic_cast<Mortar::Element*>(ele);

    // check for ghosting
    if (mele->owner() == myrank)
    {
      idiscret_->delete_element(mele->id());
    }
  }

  // send the information
  MPI_Request request;
  exporter.i_send(myrank, torank, sdataeles.data(), (int)sdataeles.size(), 1234, request);

  // receive the information
  int length = rdataeles.size();
  int tag = -1;
  int from = -1;
  exporter.receive_any(from, tag, rdataeles, length);
  if (tag != 1234 or from != fromrank)
    FOUR_C_THROW("Received data from the wrong proc soll(%i -> %i) is(%i -> %i)", fromrank, myrank,
        from, myrank);

  // ---- unpack ----
  {
    // Put received nodes into discretization
    Core::Communication::UnpackBuffer buffer(rdataeles);
    while (!buffer.at_end())
    {
      // this Teuchos::rcp holds the memory of the ele
      std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));
      std::shared_ptr<Mortar::Element> ele = std::dynamic_pointer_cast<Mortar::Element>(object);
      if (ele == nullptr) FOUR_C_THROW("Received object is not an ele");

      bool locally_owned;
      extract_from_pack(buffer, locally_owned);

      // add whether its a row ele
      if (locally_owned)
      {
        ele->set_owner(myrank);
        idiscret_->add_element(ele);

        // to new ele
        erow.push_back(ele->id());
        ecol.push_back(ele->id());
      }
      else
      {
        ecol.push_back(ele->id());
      }
    }
  }  // end unpack

  // wait for all communication to finish
  exporter.wait(request);
  Core::Communication::barrier(comm_v);

  // *****************************************
  // NODES
  // *****************************************
  std::vector<char> sdatanodes;
  std::vector<char> rdatanodes;

  // get exporter
  Core::Communication::Exporter exportern(idiscret_->get_comm());

  Core::Communication::PackBuffer datanodes;

  for (int i = 0; i < (int)MasterColNodesdummy.NumMyElements(); ++i)
  {
    int gid = MasterColNodesdummy.GID(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find ele with gid %i", gid);

    // check for ghosting
    int ghost;

    if (ftype == CONTACT::FrictionType::None)
    {
      Mortar::Node* cnode = dynamic_cast<Mortar::Node*>(node);
      cnode->pack(datanodes);

      if (cnode->owner() == myrank)
        ghost = 1;
      else
        ghost = 0;
    }
    else
    {
      FriNode* cnode = dynamic_cast<FriNode*>(node);
      cnode->pack(datanodes);

      if (cnode->owner() == myrank)
        ghost = 1;
      else
        ghost = 0;
    }

    add_to_pack(datanodes, ghost);
  }
  std::swap(sdatanodes, datanodes());

  // DELETING
  for (int i = 0; i < (int)MasterColNodesdummy.NumMyElements(); ++i)
  {
    int gid = MasterColNodesdummy.GID(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find ele with gid %i", gid);

    if (ftype == CONTACT::FrictionType::None)
    {
      Mortar::Node* cnode = dynamic_cast<Mortar::Node*>(node);
      if (cnode->owner() == myrank) idiscret_->delete_node(cnode->id());
    }
    else
    {
      FriNode* cnode = dynamic_cast<FriNode*>(node);
      if (cnode->owner() == myrank) idiscret_->delete_node(cnode->id());
    }
  }

  // ---- send ----
  MPI_Request requestn;
  exportern.i_send(myrank, torank, sdatanodes.data(), (int)sdatanodes.size(), 1234, requestn);

  // ---- receive ----
  int lengthn = rdatanodes.size();
  int tagn = -1;
  int fromn = -1;
  exportern.receive_any(fromn, tagn, rdatanodes, lengthn);
  if (tagn != 1234 or fromn != fromrank)
    FOUR_C_THROW("Received data from the wrong proc soll(%i -> %i) is(%i -> %i)", fromrank, myrank,
        fromn, myrank);

  // ---- unpack ----
  {
    // Put received nodes into discretization
    Core::Communication::UnpackBuffer buffer(rdatanodes);
    while (!buffer.at_end())
    {
      std::shared_ptr<Core::Communication::ParObject> object(Core::Communication::factory(buffer));

      int ghost;
      extract_from_pack(buffer, ghost);

      if (ftype == CONTACT::FrictionType::None)
      {
        std::shared_ptr<Mortar::Node> node = std::dynamic_pointer_cast<Mortar::Node>(object);
        if (node == nullptr) FOUR_C_THROW("Received object is not a node");

        if (ghost == 1)
        {
          node->set_owner(myrank);
          idiscret_->add_node(node);

          nrow.push_back(node->id());
          ncol.push_back(node->id());
        }
        else
        {
          // all others to col
          ncol.push_back(node->id());
        }
      }
      else  // if friction...
      {
        std::shared_ptr<FriNode> node = std::dynamic_pointer_cast<FriNode>(object);
        if (node == nullptr) FOUR_C_THROW("Received object is not a node");

        if (ghost == 1)
        {
          node->set_owner(myrank);
          idiscret_->add_node(node);

          nrow.push_back(node->id());
          ncol.push_back(node->id());
        }
        else
        {
          // all others to col
          ncol.push_back(node->id());
        }
      }
    }
  }  // end unpack

  // wait for all communication to finish
  exportern.wait(requestn);
  Core::Communication::barrier(comm_v);

  // create maps from sending
  std::shared_ptr<Epetra_Map> noderowmap = std::make_shared<Epetra_Map>(
      -1, (int)nrow.size(), nrow.data(), 0, Core::Communication::as_epetra_comm(get_comm()));
  std::shared_ptr<Epetra_Map> nodecolmap = std::make_shared<Epetra_Map>(
      -1, (int)ncol.size(), ncol.data(), 0, Core::Communication::as_epetra_comm(get_comm()));

  std::shared_ptr<Epetra_Map> elerowmap = std::make_shared<Epetra_Map>(
      -1, (int)erow.size(), erow.data(), 0, Core::Communication::as_epetra_comm(get_comm()));
  std::shared_ptr<Epetra_Map> elecolmap = std::make_shared<Epetra_Map>(
      -1, (int)ecol.size(), ecol.data(), 0, Core::Communication::as_epetra_comm(get_comm()));

  // Merge s/m column maps for eles and nodes
  std::shared_ptr<Epetra_Map> colnodesfull = Core::LinAlg::merge_map(nodecolmap, SCN, true);
  std::shared_ptr<Epetra_Map> colelesfull = Core::LinAlg::merge_map(elecolmap, SCE, true);

  // Merge s/m row maps for eles and nodes
  std::shared_ptr<Epetra_Map> rownodesfull = Core::LinAlg::merge_map(noderowmap, SRN, false);
  std::shared_ptr<Epetra_Map> rowelesfull = Core::LinAlg::merge_map(elerowmap, SRE, false);

  // to discretization
  // export nodes and elements to the row map
  discret().export_row_nodes(*rownodesfull);
  discret().export_row_elements(*rowelesfull);

  // export nodes and elements to the col map
  discret().export_column_nodes(*colnodesfull);
  discret().export_column_elements(*colelesfull);

  // ********************************************
  // call the (very) expensive FILLCOMPLETE()!
  // ********************************************
  // make sure discretization is complete
  fill_complete(Global::Problem::instance()->discretization_map(),
      Global::Problem::instance()->binning_strategy_params(),
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type(), true);

  return;
}


/*----------------------------------------------------------------------*
 |  change master ownership clockwise for contact            farah 10/13|
 |  interface without evaluation of the interface                       |
 *----------------------------------------------------------------------*/
void CONTACT::Interface::round_robin_detect_ghosting()
{
  if (search_alg() == Inpar::Mortar::search_bfele)
    evaluate_search_brute_force(search_param());
  else if (search_alg() == Inpar::Mortar::search_binarytree)
    evaluate_search_binarytree();
  else
    FOUR_C_THROW("Invalid search algorithm");

  // first ghosting for std. distribution
  round_robin_extend_ghosting(true);

  // Init Maps
  std::shared_ptr<Epetra_Map> initial_slave_node_column_map =
      std::make_shared<Epetra_Map>(*slave_col_nodes());
  std::shared_ptr<Epetra_Map> initial_slave_element_column_map =
      std::make_shared<Epetra_Map>(*slave_col_elements());
  std::shared_ptr<Epetra_Map> initial_master_node_column_map =
      std::make_shared<Epetra_Map>(*master_col_nodes());
  std::shared_ptr<Epetra_Map> initial_master_element_column_map =
      std::make_shared<Epetra_Map>(*master_col_elements());

  // *************************************
  // start RR loop for current interface
  // *************************************
  // loop over all procs
  if (Core::Communication::num_mpi_ranks(get_comm()) > 1)
    for (int proc = 0; proc < (int)(Core::Communication::num_mpi_ranks(get_comm())); ++proc)
    {
      // status output
      if (Core::Communication::my_mpi_rank(get_comm()) == 0 && proc == 0)
        std::cout << "Round-Robin-Iteration #" << proc;
      if (Core::Communication::my_mpi_rank(get_comm()) == 0 && proc > 0) std::cout << " #" << proc;

      // perform the ownership change
      round_robin_change_ownership();

      // build new search tree or do nothing for bruteforce
      if (search_alg() == Inpar::Mortar::search_binarytree)
        create_search_tree();
      else if (search_alg() != Inpar::Mortar::search_bfele)
        FOUR_C_THROW("Invalid search algorithm");

      // evaluate interfaces
      if (proc < (int)(Core::Communication::num_mpi_ranks(get_comm()) - 1))
      {
        if (search_alg() == Inpar::Mortar::search_bfele)
          evaluate_search_brute_force(search_param());
        else if (search_alg() == Inpar::Mortar::search_binarytree)
          evaluate_search_binarytree();
        else
          FOUR_C_THROW("Invalid search algorithm");

        // other ghostings per iteration
        round_robin_extend_ghosting(false);
      }
      else
      {
        // do nothing -- just switch
      }
    }  // end RR

  // Append ghost nodes/elements to be ghosted to existing column maps
  eextendedghosting_ =
      Core::LinAlg::merge_map(eextendedghosting_, initial_slave_element_column_map, true);
  nextendedghosting_ =
      Core::LinAlg::merge_map(nextendedghosting_, initial_slave_node_column_map, true);
  eextendedghosting_ =
      Core::LinAlg::merge_map(eextendedghosting_, initial_master_element_column_map, true);
  nextendedghosting_ =
      Core::LinAlg::merge_map(nextendedghosting_, initial_master_node_column_map, true);

  // finally extend ghosting
  discret().export_column_elements(*eextendedghosting_);
  discret().export_column_nodes(*nextendedghosting_);
  fill_complete(Global::Problem::instance()->discretization_map(),
      Global::Problem::instance()->binning_strategy_params(),
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type(), true);

  // reset extended ghosting maps
  eextendedghosting_ = nullptr;
  nextendedghosting_ = nullptr;

  // build new search tree or do nothing for bruteforce
  if (search_alg() == Inpar::Mortar::search_binarytree)
    create_search_tree();
  else if (search_alg() != Inpar::Mortar::search_bfele)
    FOUR_C_THROW("Invalid search algorithm");

  // final output for loop
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << " Round-Robin loop done!" << std::endl;

  return;
}

FOUR_C_NAMESPACE_CLOSE
