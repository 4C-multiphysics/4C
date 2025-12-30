// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization_faces.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor (public)                                    schott 03/12|
 *----------------------------------------------------------------------*/
Core::FE::DiscretizationFaces::DiscretizationFaces(
    const std::string name, MPI_Comm comm, const unsigned int n_dim)
    : Discretization(name, comm, n_dim),  // use base class constructor
      extension_filled_(false),
      doboundaryfaces_(false) {};

/*----------------------------------------------------------------------*
 |  Finalize construction (public)                          schott 03/12|
 *----------------------------------------------------------------------*/
int Core::FE::DiscretizationFaces::fill_complete_faces(
    OptionsFillComplete options, bool createinternalfaces)

{
  // call standard FillComplete of base class
  Core::FE::Discretization::fill_complete(options);

  if (createinternalfaces)
  {
    create_internal_faces_extension();
  }

  return 0;
}



/*----------------------------------------------------------------------*
 |  Build internal faces extension (public)                 schott 03/12|
 *----------------------------------------------------------------------*/
void Core::FE::DiscretizationFaces::create_internal_faces_extension(const bool verbose)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::FE::DiscretizationFaces::CreateInternalFaces");

  // create internal faces for stabilization along edges
  build_faces(verbose);

  // (re)build map of internal faces
  build_face_row_map();
  build_face_col_map();

  extension_filled_ = true;

  if (verbose)
  {
    int summyfaces = facerowptr_.size();
    int summall = 0;
    summall = Core::Communication::sum_all(summyfaces, comm_);

    if (Core::Communication::my_mpi_rank(comm_) == 0)
      std::cout << "number of created faces:   " << summall << "\n" << std::endl;
  }

  return;
}



/*----------------------------------------------------------------------*
 |  Build internal faces geometry (public)                  schott 03/12|
 *----------------------------------------------------------------------*/
void Core::FE::DiscretizationFaces::build_faces(const bool verbose)
{
  faces_.clear();

  if (verbose and Core::Communication::my_mpi_rank(comm_) == 0)
  {
    std::cout << "Create internal faces ..." << std::endl;
  }

  //----------------------------------------------------------------------
  /* First: Create the surface objects between to elements . */

  // map of surfaces in this cloud: (sorted node_ids) -> (surface)
  std::map<std::vector<int>, InternalFacesData> surfmapdata;

  // loop col elements and find all surfaces attached to them
  //
  // REMARK: in a first step: find all surfaces and adjacent elements and fill InternalFacesData
  //         without creating the internal faces elements

  std::vector<Core::Elements::Element*>::iterator fool;

  for (fool = elecolptr_.begin(); fool != elecolptr_.end(); ++fool)
  {
    Core::Elements::Element* ele = *fool;

    //-------------------------------------------
    // create

    Core::Communication::BoundaryBuildType buildtype = Core::Communication::buildNothing;

    // 3D elements
    if (ele->num_surface() > 1)  // 2D boundary element and 3D parent element
    {
      buildtype = Core::Communication::buildSurfaces;
    }
    else if (ele->num_surface() == 1)  // 1D boundary element and 2D parent element
    {
      buildtype = Core::Communication::buildLines;
    }
    else
      FOUR_C_THROW("creating internal faces for 1D elements (would be points) not implemented yet");


    // get node connectivity for specific distype of parent element
    unsigned int nele = 0;
    const Core::FE::CellType distype = ele->shape();
    std::vector<std::vector<int>> connectivity;
    switch (buildtype)
    {
      case Core::Communication::buildSurfaces:
      {
        nele = ele->num_surface();
        connectivity = Core::FE::get_ele_node_numbering_surfaces(distype);
        break;
      }
      case Core::Communication::buildLines:
      {
        nele = ele->num_line();
        connectivity = Core::FE::get_ele_node_numbering_lines(distype);
        break;
      }
      default:
        FOUR_C_THROW("Core::FE::build... not supported");
        break;
    }


    // does Core::FE::UTILS convention match your implementation of NumSurface() or
    // NumLine()?
    if (nele != connectivity.size()) FOUR_C_THROW("number of surfaces or lines does not match!");

    // now, get the nodal information for the new surface/line faces
    for (unsigned int iele = 0; iele < nele; iele++)
    {
      // allocate node vectors
      unsigned int nnode = connectivity[iele].size();  // this number changes for pyramids or wedges
      std::vector<int> nodeids(nnode);
      std::vector<Core::Nodes::Node*> nodes(nnode);

      // get connectivity info
      for (unsigned int inode = 0; inode < nnode; inode++)
      {
        nodeids[inode] = ele->node_ids()[connectivity[iele][inode]];
        nodes[inode] = ele->nodes()[connectivity[iele][inode]];
      }

      // sort the nodes. Used to identify surfaces that are created multiple
      std::sort(nodeids.begin(), nodeids.end());

      // find existing InternalFacesData
      std::map<std::vector<int>, InternalFacesData>::iterator surf_it = surfmapdata.find(nodeids);
      if (surf_it == surfmapdata.end())
      {
        // not found -> generate new Data
        // add the faces information to the map (key is the sorted vector of nodeids)
        surfmapdata.insert(std::pair<std::vector<int>, InternalFacesData>(
            nodeids, InternalFacesData(ele->id(), nodes, iele)));
      }
      else
      {
        if (surf_it->second.get_slave_peid() != -1) FOUR_C_THROW("slave peid should not be set!!!");
        // if found -> add second neighbor data to existing data
        surf_it->second.set_slave_peid(ele->id());
        surf_it->second.set_l_surface_slave(iele);

        std::vector<int> localtrafomap;

        // get the face's nodes sorted w.r.t local coordinate system of the parent's face element
        const std::vector<Core::Nodes::Node*> nodes_face_master = surf_it->second.get_nodes();
        if (nodes_face_master.size() != nnode)
          FOUR_C_THROW(
              "the number of the face w.r.t parent element and slave element are not the same. "
              "That is wrong!");

        // find the nodes given with the master element node numbering also for the slave element
        // to define a connectivity map between the local face's coordinate systems
        for (unsigned int inode = 0; inode < nnode; inode++)  // master face nodes
        {
          int position = -1;
          for (std::size_t knode = 0; knode < nodes.size(); knode++)
          {
            if (nodes[knode] == nodes_face_master[inode]) position = knode;
          }

          if (position >= 0)
            localtrafomap.push_back(position);
          else
            FOUR_C_THROW(
                "face's node from master's face element not found in slave's face element!");
        }

        surf_it->second.set_local_numbering_map(localtrafomap);
      }
    }  // loop iele

  }  // loop elecolptr_

  //----------------------------------------------------------------------
  // in a second step: create the internal faces elements ( sorted nids -> surface element)
  // REMARK: internal faces are created and distributed on procs as following:
  // * faces are created whenever two adjacent elements are available on this proc (sometimes faces
  // are created multiply on more procs)
  // * each face is created at least once (at least one node of the surface is on a proc a row node
  // and a 1-ring of elements around
  //   this node is available as col elements)
  // * how to set the owner for this face on all procs equally?
  //    -> if one set has been created on a proc, there are both parent elements available as row or
  //    col elements
  //    -> therefore for each node of this surface both parent elements are available
  //    -> choose the node with smallest global id
  //    -> the owner of this node will be the owner for the face
  //       (this criterion is working in the same way on all procs holding this face)

  std::map<std::vector<int>, std::shared_ptr<Core::Elements::Element>> faces;

  // get pbcs
  const std::map<int, std::vector<int>>* col_pbcmapmastertoslave = get_all_pbc_coupled_col_nodes();

  std::map<std::vector<int>, InternalFacesData>::iterator face_it;
  for (face_it = surfmapdata.begin(); face_it != surfmapdata.end(); ++face_it)
  {
    int master_peid = face_it->second.get_master_peid();
    int slave_peid = face_it->second.get_slave_peid();
    if (master_peid == -1) FOUR_C_THROW("Face master expected!");

    FOUR_C_ASSERT(master_peid == g_element(master_peid)->id(), "Internal error");
    FOUR_C_ASSERT(slave_peid == -1 || slave_peid == g_element(slave_peid)->id(), "Internal error");

    // check for potential periodic boundary conditions and connect respective faces/elements
    if (col_pbcmapmastertoslave)
    {
      // unconnected face is potential pbc face
      if (slave_peid == -1)
      {
        // get node ids of current face
        std::vector<int> mynodeids = face_it->first;

        // get periodic surface boundary conditions
        // number of pairs of periodic boundary conditions
        int numpbcpairs;
        // vector of periodic surface boundary conditions
        std::vector<const Core::Conditions::Condition*> mypbcs;
        get_condition("SurfacePeriodic", mypbcs);
        if (mypbcs.empty())
        {
          get_condition("LinePeriodic", mypbcs);
        }
        // set number of pairs of periodic boundary conditions
        numpbcpairs = mypbcs.size() / 2;

        // sets of pbc id and related node ids
        // for master and slave
        std::map<int, std::set<int>> mastertopbcset;
        std::map<int, std::set<int>> slavetopbcset;
        for (auto& mypbc : mypbcs)
        {
          const int zero_based_id = mypbc->parameters().get<int>("ID") - 1;

          const auto mymasterslavetoggle = mypbc->parameters().get<std::string>("MASTER_OR_SLAVE");

          if (mymasterslavetoggle == "Master")
          {
            // get global master node ids
            const std::vector<int>* masteridstoadd = mypbc->get_nodes();

            // store them in list depending on the pbc id
            for (int idtoadd : *masteridstoadd)
            {
              (mastertopbcset[zero_based_id]).insert(idtoadd);
            }
          }
          else if (mymasterslavetoggle == "Slave")
          {
            // get global slave node ids
            const std::vector<int>* slaveidstoadd = mypbc->get_nodes();

            // store them in list depending on the pbc id
            for (int idtoadd : *slaveidstoadd)
            {
              (slavetopbcset[zero_based_id]).insert(idtoadd);
            }
          }
          else
            FOUR_C_THROW("Unknown type for pbc!");
        }

        // provide vectors for master and slave node ids
        std::vector<int> mymasternodeids;
        std::vector<int> myslavenodeids;
        // provide vector for undefined nodes
        // i.e., special nodes on edges or in corners
        // for multiple pbc sets master nodes of boundary condition
        // become slave nodes
        // e.g. for two sets two master nodes at the corners become slave nodes
        //
        //                PBC M surface
        //           M------------------------S
        //           |                        |
        //  PBC M    |                        | PBC S
        //  surface  |                        | surface
        //           |                        |
        //           S------------------------S
        //                PBC S surface
        // these nodes are not contained in the list col_pbcmapmastertoslave as master nodes
        // but result in more than one slave node for the corner or edge master
        std::vector<int> myfurthermasternodeids;

        // local (or face) master to slave coupling
        std::map<int, int> local_pbcmapmastertoslave;

        // bool to indicate if slave element has been found and should be added to the patch
        bool add_salve_ele_to_face = true;

        // loop node ids of current face and check if they are contained in the list
        // of all master node ids
        for (std::size_t inode = 0; inode < mynodeids.size(); inode++)
        {
          if (col_pbcmapmastertoslave->find(mynodeids[inode]) != col_pbcmapmastertoslave->end())
          {
            // add node id to list of current master nodes
            mymasternodeids.push_back(mynodeids[inode]);
          }
          else
          {
            // if node is not in (master) list col_pbcmapmastertoslave, it may be special
            // node as explained above
            // check whether node is master and slave due to several pbcs
            bool found = false;
            // loop all master sets
            for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
            {
              if ((mastertopbcset[ipbc]).find(mynodeids[inode]) != (mastertopbcset[ipbc]).end())
                found = true;
            }

            // yes, we have a master node here
            // add to list of further master nodes which require special care
            if (found) myfurthermasternodeids.push_back(mynodeids[inode]);
          }
        }

        if (mymasternodeids.size() > 0)
        {
          //          std::cout << "current sets" << std::endl;
          //          std::cout << "master nodes" << std::endl;
          //          for (std::size_t rr=0; rr < mymasternodeids.size(); rr++)
          //            std::cout << mymasternodeids[rr] << std::endl;
          //          std::cout << "further master nodes" << std::endl;
          //          for (std::size_t rr=0; rr < myfurthermasternodeids.size(); rr++)
          //            std::cout << myfurthermasternodeids[rr] << std::endl;

          // check if all nodes of the face are masters of pbcs
          // -> this is a master face
          if ((mymasternodeids.size() + myfurthermasternodeids.size()) == mynodeids.size())
          {
            // get corresponding slave ids
            // do the standard master nodes of col_pbcmapmastertoslave first
            for (std::size_t rr = 0; rr < mymasternodeids.size(); rr++)
            {
              // this master node has one slave node
              if (((*col_pbcmapmastertoslave).at(mymasternodeids[rr])).size() == 1)
              {
                myslavenodeids.push_back(((*col_pbcmapmastertoslave).at(mymasternodeids[rr]))[0]);
                local_pbcmapmastertoslave[mymasternodeids[rr]] =
                    ((*col_pbcmapmastertoslave).at(mymasternodeids[rr]))[0];
              }
              // this master node has several slave nodes
              // it is a corner or edge node of two or three pbc sets
              else
              {
                // this is only possible for multiple pbcs
                if (numpbcpairs == 1) FOUR_C_THROW("Two or three pbs sets expected");

                // identify the pbc condition (i.e., pbc id) to which the current face belongs

                std::map<int, int> pbcspermaster;
                // initialize with zeros
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++) pbcspermaster[ipbc] = 0;

                // identify pbc set to which master nodes belong
                for (std::size_t imnode = 0; imnode < mymasternodeids.size(); imnode++)
                {
                  for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                  {
                    std::set<int>::iterator iter =
                        (mastertopbcset[ipbc]).find(mymasternodeids[imnode]);
                    if (iter != (mastertopbcset[ipbc]).end()) pbcspermaster[ipbc] += 1;
                  }
                }

                // all master nodes of current surface share the same pbc id
                int masterpbcid = -1;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (pbcspermaster[ipbc] == (int)mymasternodeids.size())
                  {
                    masterpbcid = ipbc;
                    break;
                  }
                }

                // find the corresponding slave of the current master node

                // the corresponding slave node is
                // for 2 pbc sets
                // (i) slave node with respect to the pbc id of the master face
                // (ii) master with respect to the remaining pbc sets
                // for 3 pbc sets
                // here to cases may occur
                // master has 7 slaves -> corner node
                // this results as for 2 sets in
                // (i) slave node with respect to the pbc id of the master face
                // (ii) master with respect to the remaining pbc sets
                // master has 3 slaves -> edge node
                // this results in
                // (i) slave node with respect to the pbc id of the master face
                // (ii) master with respect to one of the two remaining pbc sets
                //  this special case is marked by flag
                bool three_sets_edge_node = false;
                if (numpbcpairs == 3 and
                    ((*col_pbcmapmastertoslave).at(mymasternodeids[rr])).size() == 3)
                  three_sets_edge_node = true;

                // pbc id of master face also for the slave
                int slavepbcid = masterpbcid;
                // identify the remaining pbc sets via their id
                std::vector<int> remainingmasterpbcids;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (ipbc != slavepbcid) remainingmasterpbcids.push_back(ipbc);
                }

                // loop all slave nodes of the current master and
                // check which node fulfills above conditions
                int actslaveid = -999;
                for (std::size_t islave = 0;
                    islave < ((*col_pbcmapmastertoslave).at(mymasternodeids[rr])).size(); islave++)
                {
                  // get id
                  actslaveid = ((*col_pbcmapmastertoslave).at(mymasternodeids[rr]))[islave];

                  // check first criterion -> (i)
                  if ((slavetopbcset[slavepbcid]).find(actslaveid) !=
                      (slavetopbcset[slavepbcid]).end())
                  {
                    std::size_t found = 0;
                    // if satisfied
                    // check second criterion -> (ii)
                    for (std::size_t k = 0; k < remainingmasterpbcids.size(); k++)
                    {
                      if ((mastertopbcset[remainingmasterpbcids[k]]).find(actslaveid) !=
                          (mastertopbcset[remainingmasterpbcids[k]]).end())
                        found++;
                    }

                    if ((not three_sets_edge_node) and found == remainingmasterpbcids.size())
                      break;
                    else if (three_sets_edge_node and found == 1)
                      break;
                  }
                }

                // store in list
                myslavenodeids.push_back(actslaveid);
                local_pbcmapmastertoslave[mymasternodeids[rr]] = actslaveid;
              }
            }

            // next go to the special masters which occur as slaves in the list
            // col_pbcmapmastertoslave and are indeed edge or corner nodes of master surfaces
            if (myfurthermasternodeids.size() > 0)
            {
              // identify the pbc condition (i.e., id) to which the current face belongs
              // perform as explained above of the special master nodes with several slaves

              std::map<int, int> pbcspermaster;
              for (int ipbc = 0; ipbc < numpbcpairs; ipbc++) pbcspermaster[ipbc] = 0;

              // identify pbc set to which master nodes belong
              for (std::size_t imnode = 0; imnode < mymasternodeids.size(); imnode++)
              {
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  std::set<int>::iterator iter =
                      (mastertopbcset[ipbc]).find(mymasternodeids[imnode]);
                  if (iter != (mastertopbcset[ipbc]).end()) pbcspermaster[ipbc] += 1;
                }
              }
              // identify pbc set to which additional master nodes belong
              for (std::size_t imnode = 0; imnode < myfurthermasternodeids.size(); imnode++)
              {
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  std::set<int>::iterator iter =
                      (mastertopbcset[ipbc]).find(myfurthermasternodeids[imnode]);
                  if (iter != (mastertopbcset[ipbc]).end()) pbcspermaster[ipbc] += 1;
                }
              }

              // all master nodes of current surface share the same pbc id
              int masterpbcid = -1;
              for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
              {
                if (pbcspermaster[ipbc] ==
                    (int)(mymasternodeids.size() + myfurthermasternodeids.size()))
                {
                  masterpbcid = ipbc;
                  break;
                }
              }

              // find the corresponding slaves of the additional master nodes

              for (std::size_t ifnode = 0; ifnode < myfurthermasternodeids.size(); ifnode++)
              {
                // get node id of master
                int actnodeid = myfurthermasternodeids[ifnode];

                // get list of all potential slave nodes
                // i.e., further edge or corner nodes
                std::vector<int> mypotslaveids;

                // first, look for masters with more than one slave
                // then, check whether node id (i.e. actnodeid) is contained in slave list
                std::map<int, std::vector<int>>::const_iterator master_it;
                for (master_it = col_pbcmapmastertoslave->begin();
                    master_it != col_pbcmapmastertoslave->end(); master_it++)
                {
                  if ((master_it->second).size() > 1)
                  {
                    bool found = false;

                    for (std::size_t k = 0; k < (master_it->second).size(); k++)
                    {
                      if ((master_it->second)[k] == actnodeid)
                      {
                        found = true;
                      }
                    }

                    if (found)
                    {
                      for (std::size_t k = 0; k < (master_it->second).size(); k++)
                        mypotslaveids.push_back((master_it->second)[k]);
                    }

                    if (found) break;
                  }
                }

                if (mypotslaveids.size() == 0) FOUR_C_THROW("Expected to find node!");

                // find the corresponding slave of the current master node

                // the corresponding slave node is
                // for 2 pbc sets
                // (i) slave node with respect to the pbc id of the master face
                // (ii) slave with respect to the remaining pbc sets
                // for 3 pbc sets
                // master has 3 slaves -> edge node (special case 1)
                // this results in
                // (i) slave node with respect to the pbc id of the master face
                // (ii) slave with respect to one of the two remaining pbc sets
                // master has 7 slaves -> corner node (special case 2)
                // (i) slave node with respect to the pbc id of the master face
                // (ii) slave or master with respect to the remaining pbc sets
                //      depending on the status of the current node with respect to those pbcs
                // special case 1 marked by flag
                bool three_sets_edge_node = false;
                if (numpbcpairs == 3 and mypotslaveids.size() == 3) three_sets_edge_node = true;

                int slavepbcid = masterpbcid;

                std::vector<int> remainingslavepbcids;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (ipbc != slavepbcid) remainingslavepbcids.push_back(ipbc);
                }

                std::vector<int> remainingmasterpbcids;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (ipbc != slavepbcid) remainingmasterpbcids.push_back(ipbc);
                }

                // special case 2 marked by flag
                bool corner_node = false;
                int furthermastercond = -1;
                int slavecond_1 = -1;

                // get status of corresponding slave with respect to the
                // remaining to pbc sets
                // may be master and slave or slave and slave
                if (numpbcpairs == 3 and mypotslaveids.size() == 7)
                {
                  corner_node = true;

                  // set the sets to check master or slave
                  for (std::size_t k = 0; k < remainingmasterpbcids.size(); k++)
                  {
                    if ((mastertopbcset[remainingmasterpbcids[k]]).find(actnodeid) !=
                        (mastertopbcset[remainingmasterpbcids[k]]).end())
                      furthermastercond = remainingmasterpbcids[k];
                  }

                  // corresponding slave is pure slave
                  // -> we just have to check the slave lists
                  // -> similar to the 2 pbc-sets case
                  if (furthermastercond == -1)
                  {
                    corner_node = false;
                  }
                  // corresponding slave is master and slave
                  // -> we set the slave list to check
                  else
                  {
                    if (furthermastercond == 0)
                    {
                      if (slavepbcid == 1)
                        slavecond_1 = 2;
                      else if (slavepbcid == 2)
                        slavecond_1 = 1;
                      else
                        FOUR_C_THROW("Same pbc ids?");
                    }
                    else if (furthermastercond == 1)
                    {
                      if (slavepbcid == 0)
                        slavecond_1 = 2;
                      else if (slavepbcid == 2)
                        slavecond_1 = 0;
                      else
                        FOUR_C_THROW("Same pbc ids?");
                    }
                    else if (furthermastercond == 2)
                    {
                      if (slavepbcid == 0)
                        slavecond_1 = 1;
                      else if (slavepbcid == 1)
                        slavecond_1 = 0;
                      else
                        FOUR_C_THROW("Same pbc ids?");
                    }
                    else
                      FOUR_C_THROW("Unknown pbc id!");
                  }
                }

                //                std::cout << "furthermastercond " <<furthermastercond<< std::endl;
                //                std::cout << "slavecond_1 " <<slavecond_1<< std::endl;

                //                std::cout<< "master 1 \n" << masterpbcid << std::endl;
                //                std::set<int>::iterator myiter;
                //                for (myiter=mastertopbcset[remainingslavepbcids[0]].begin();
                //                myiter!=mastertopbcset[remainingslavepbcids[0]].end(); myiter++)
                //                {
                //                  std::cout<< *myiter << std::endl;
                //                }
                //                std::cout<< "master 2 \n" << masterpbcid << std::endl;
                //                for (myiter=mastertopbcset[remainingslavepbcids[1]].begin();
                //                myiter!=mastertopbcset[remainingslavepbcids[1]].end(); myiter++)
                //                  std::cout<< *myiter << std::endl;

                // loop all slave nodes of the current master and
                // check which node fulfills above conditions
                int actslaveid = -999;
                for (std::size_t islave = 0; islave < mypotslaveids.size(); islave++)
                {
                  // get slave node id
                  actslaveid = mypotslaveids[islave];

                  // check first criterion
                  if ((slavetopbcset[slavepbcid]).find(actslaveid) !=
                      (slavetopbcset[slavepbcid]).end())
                  {
                    std::size_t found = 0;
                    // if satisfied
                    // check second criterion
                    if (not corner_node)
                    {
                      // check to slave conditions
                      for (std::size_t k = 0; k < remainingslavepbcids.size(); k++)
                      {
                        if ((slavetopbcset[remainingslavepbcids[k]]).find(actslaveid) !=
                            (slavetopbcset[remainingslavepbcids[k]]).end())
                          found++;
                      }
                    }
                    else
                    {
                      // check a master and a slave condition
                      if ((mastertopbcset[furthermastercond]).find(actslaveid) !=
                          (mastertopbcset[furthermastercond]).end())
                        found++;
                      if ((slavetopbcset[slavecond_1]).find(actslaveid) !=
                          (slavetopbcset[slavecond_1]).end())
                        found++;
                    }

                    if ((not three_sets_edge_node) and found == remainingslavepbcids.size())
                      break;
                    else if (three_sets_edge_node and found == 1)
                      break;
                  }
                }

                // store in list
                myslavenodeids.push_back(actslaveid);
                local_pbcmapmastertoslave[myfurthermasternodeids[ifnode]] = actslaveid;
              }
            }

            // sort the slave node ids
            std::sort(myslavenodeids.begin(), myslavenodeids.end());
          }
          else
          {
            add_salve_ele_to_face = false;
          }

          // this criterion ensures that slave set is available on this proc
          int counter = 0;
          for (std::size_t kk = 0; kk < mymasternodeids.size(); kk++)
          {
            for (auto node : my_row_node_range())
            {
              if (node.global_id() == mymasternodeids[kk]) counter++;
            }
          }
          for (std::size_t kk = 0; kk < myfurthermasternodeids.size(); kk++)
          {
            for (auto node : my_row_node_range())
            {
              if (node.global_id() == myfurthermasternodeids[kk]) counter++;
            }
          }
          if (counter == 0)
          {
            add_salve_ele_to_face = false;
          }

          // add slave element to the patch
          if (add_salve_ele_to_face)
          {
            // get master element
            Core::Elements::Element* master_ele = elecolptr_[0];
            for (fool = elecolptr_.begin(); fool != elecolptr_.end(); ++fool)
            {
              if ((*fool)->id() == master_peid) master_ele = *fool;
            }

            // look for the corresponding slave face in the list of all faces
            std::map<std::vector<int>, InternalFacesData>::iterator pbc_surf_it =
                surfmapdata.find(myslavenodeids);
            if (pbc_surf_it == surfmapdata.end())
            {
              // print some helpful information first
              master_ele->print(std::cout);

              std::cout << "\n slave " << std::endl;
              for (std::size_t kk = 0; kk < myslavenodeids.size(); kk++)
                std::cout << myslavenodeids[kk] << std::endl;

              FOUR_C_THROW("Expected to find slave face!");
            }

            // add slave data to master data
            face_it->second.set_slave_peid(pbc_surf_it->second.get_master_peid());
            slave_peid = face_it->second.get_slave_peid();
            face_it->second.set_l_surface_slave(pbc_surf_it->second.get_l_surface_master());

            // add connection of coordinate systems for master and slave
            std::vector<int> localtrafomap;

            // get the face's nodes sorted w.r.t local coordinate system of the parent's face
            // element
            const std::vector<Core::Nodes::Node*> nodes_face_master = face_it->second.get_nodes();
            // get number of nodes
            unsigned int nnode = nodes_face_master.size();

            // get slave nodes
            std::vector<Core::Nodes::Node*> slave_nodes = pbc_surf_it->second.get_nodes();

            // find the nodes given with the master element node numbering also for the slave
            // element to define a connectivity map between the local face's coordinate systems
            for (unsigned int inode = 0; inode < nnode; inode++)  // master face nodes
            {
              int position = -1;

              for (std::size_t knode = 0; knode < slave_nodes.size(); knode++)
              {
                if (slave_nodes[knode]->id() ==
                    local_pbcmapmastertoslave[nodes_face_master[inode]->id()])
                  position = knode;
              }

              if (position >= 0)
                localtrafomap.push_back(position);
              else
                FOUR_C_THROW(
                    "face's node from master's face element not found in slave's face element!");
            }
            // set in face
            face_it->second.set_local_numbering_map(localtrafomap);

            //            if (Core::Communication::my_mpi_rank(comm_)==1)
            //            {
            //            std::cout << "\n added pbc face "  << std::endl;
            //
            //            std::cout << "master nodes" << std::endl;
            //            for (std::size_t rr=0; rr < mymasternodeids.size(); rr++)
            //              std::cout << mymasternodeids[rr] << std::endl;
            //
            //            std::cout << "further master nodes" << std::endl;
            //            for (std::size_t rr=0; rr < myfurthermasternodeids.size(); rr++)
            //              std::cout << myfurthermasternodeids[rr] << std::endl;
            //
            //              std::cout << "slave node ids "  << std::endl;
            //            for (std::size_t kk=0; kk<myslavenodeids.size(); kk++)
            //               std::cout << myslavenodeids[kk] << std::endl;
            //
            ////            std::cout << "local trafo map  " << std::endl;
            ////            for (std::size_t kk=0; kk<localtrafomap.size(); kk++)
            ////            {
            ////              std::cout << "master node id " << nodes_face_master[kk]->Id() <<
            /// std::endl; /              std::cout << "slave node id " << slave_nodes[kk]->Id() <<
            /// std::endl; /              std::cout << "slave node position " << localtrafomap[kk]
            /// << std::endl; /            }
            //            }
          }
        }
      }
    }

    // create faces
    if (doboundaryfaces_ || (master_peid != -1 && slave_peid != -1))
    {
      FOUR_C_ASSERT(master_peid != -1, "At least the master element should be present");
      Core::Elements::Element* parent_master = g_element(master_peid);
      Core::Elements::Element* parent_slave = slave_peid != -1 ? g_element(slave_peid) : nullptr;

      FOUR_C_ASSERT(master_peid == parent_master->id(), "Internal error");
      FOUR_C_ASSERT(slave_peid == -1 || slave_peid == parent_slave->id(), "Internal error");

      // get the unsorted nodes
      std::vector<Core::Nodes::Node*> nodes = face_it->second.get_nodes();

      // get corresponding nodeids
      std::vector<int> nodeids(nodes.size());
      std::transform(
          nodes.begin(), nodes.end(), nodeids.begin(), std::mem_fn(&Core::Nodes::Node::id));

      // create the internal face element
      std::shared_ptr<Core::Elements::FaceElement> surf =
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(parent_master->create_face_element(
              parent_slave, nodeids.size(), nodeids.data(), nodes.data(),
              face_it->second.get_l_surface_master(), face_it->second.get_l_surface_slave(),
              face_it->second.get_local_numbering_map()));
      FOUR_C_ASSERT(surf != nullptr,
          "Creating a face element failed. Check overloading of CreateFaceElement");

      // create a clone (the internally created element does not exist anymore when all
      // std::shared_ptr's finished)
      std::shared_ptr<Core::Elements::FaceElement> surf_clone(
          dynamic_cast<Core::Elements::FaceElement*>(surf->clone()));
      if (surf_clone.get() == nullptr)
        FOUR_C_THROW("Invalid element detected. Expected face element");

      // Set owning process of surface to node with smallest gid
      // REMARK: see below
      sort(nodeids.begin(), nodeids.end());
      int owner = g_node(nodeids[0])->owner();

      // set the owner
      surf_clone->set_owner(owner);

      // insert the newly created element
      faces.insert(std::pair<std::vector<int>, std::shared_ptr<Core::Elements::Element>>(
          face_it->first, surf_clone));

      // set face to elements
      parent_master->set_face(face_it->second.get_l_surface_master(), surf_clone.get());
      if (slave_peid != -1)
        parent_slave->set_face(face_it->second.get_l_surface_slave(), surf_clone.get());
    }
  }

  // Surfaces be added to the faces_-map: (line_id) -> (surface).
  // this clear is important to have here
  // if the discretization has been redistributed (combustion module), we have to
  // rebuild the faces and therefore we have to be sure that the map faces_ is clear
  // therefore, the old faces are deleted and replaced by new ones
  std::map<int, std::shared_ptr<Core::Elements::Element>> finalFaces;
  assign_global_ids(get_comm(), faces, finalFaces);
  for (std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator faceit =
           finalFaces.begin();
      faceit != finalFaces.end(); ++faceit)
    faces_[faceit->first] = std::dynamic_pointer_cast<Core::Elements::FaceElement>(faceit->second);

  if (verbose and Core::Communication::my_mpi_rank(comm_) == 0)
  {
    std::cout << "... done!" << std::endl;
  }

  return;
}  // Core::FE::DiscretizationFaces::BuildInternalFaces



/*----------------------------------------------------------------------*
 |  Build intfacerowmap_ (private)                          schott 03/12|
 *----------------------------------------------------------------------*/
void Core::FE::DiscretizationFaces::build_face_row_map()
{
  const int myrank = Core::Communication::my_mpi_rank(get_comm());
  int nummyeles = 0;
  std::map<int, std::shared_ptr<Core::Elements::FaceElement>>::iterator curr;
  for (curr = faces_.begin(); curr != faces_.end(); ++curr)
    if (curr->second->owner() == myrank) nummyeles++;
  std::vector<int> eleids(nummyeles);
  facerowptr_.resize(nummyeles);
  int count = 0;
  for (curr = faces_.begin(); curr != faces_.end(); ++curr)
    if (curr->second->owner() == myrank)
    {
      eleids[count] = curr->second->id();
      facerowptr_[count] = curr->second.get();
      ++count;
    }
  if (count != nummyeles) FOUR_C_THROW("Mismatch in no. of internal faces");
  facerowmap_ = std::make_shared<Core::LinAlg::Map>(-1, nummyeles, eleids.data(), 0, get_comm());
  return;
}


/*----------------------------------------------------------------------*
 |  Build intfacecolmap_ (private)                          schott 03/12|
 *----------------------------------------------------------------------*/
void Core::FE::DiscretizationFaces::build_face_col_map()
{
  int nummyeles = (int)faces_.size();
  std::vector<int> eleids(nummyeles);
  facecolptr_.resize(nummyeles);
  std::map<int, std::shared_ptr<Core::Elements::FaceElement>>::iterator curr;
  int count = 0;
  for (curr = faces_.begin(); curr != faces_.end(); ++curr)
  {
    eleids[count] = curr->second->id();
    facecolptr_[count] = curr->second.get();
    curr->second->set_lid(count);
    ++count;
  }
  if (count != nummyeles) FOUR_C_THROW("Mismatch in no. of elements");
  facecolmap_ = std::make_shared<Core::LinAlg::Map>(-1, nummyeles, eleids.data(), 0, get_comm());
  return;
}


/*----------------------------------------------------------------------*
 |  get internal faces row map (public)                     schott 03/12|
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::DiscretizationFaces::face_row_map() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before call to FaceRowMap()");
  return facerowmap_.get();
}


/*----------------------------------------------------------------------*
 |  get internal faces col map (public)                     schott 03/12|
 *----------------------------------------------------------------------*/
const Core::LinAlg::Map* Core::FE::DiscretizationFaces::face_col_map() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before call to FaceColMap()");
  return facecolmap_.get();
}


/*----------------------------------------------------------------------*
 |  get global no of internal faces (public)                schott 03/12|
 *----------------------------------------------------------------------*/
int Core::FE::DiscretizationFaces::num_global_faces() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before call to NumGlobalFaces()");
  return face_row_map()->num_global_elements();
}


/*----------------------------------------------------------------------*
 |  get no of my row internal faces (public)                schott 03/12|
 *----------------------------------------------------------------------*/
int Core::FE::DiscretizationFaces::num_my_row_faces() const
{
  FOUR_C_ASSERT(filled(), "fill_complete() must be called before call to NumMyRowFaces()");
  return face_row_map()->num_my_elements();
}


/*----------------------------------------------------------------------*
 |  get no of my column internal faces (public)             schott 03/12|
 *----------------------------------------------------------------------*/
int Core::FE::DiscretizationFaces::num_my_col_faces() const
{
  if (filled())
    return face_col_map()->num_my_elements();
  else
    return (int)faces_.size();
}



/*----------------------------------------------------------------------*
 |  << operator                                             schott 03/12|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const Core::FE::DiscretizationFaces& dis)
{
  // print standard discretization info
  dis.print(os);
  // print additional info about internal faces
  dis.print_faces(os);

  return os;
}


/*----------------------------------------------------------------------*
 |  Print internal faces discretization (public)            schott 03/12|
 *----------------------------------------------------------------------*/
void Core::FE::DiscretizationFaces::print_faces(std::ostream& os) const
{
  int numglobalfaces = 0;
  if (filled())
  {
    numglobalfaces = num_global_faces();
  }
  else
  {
    int nummyfaces = 0;
    std::map<int, std::shared_ptr<Core::Elements::FaceElement>>::const_iterator ecurr;
    for (ecurr = faces_.begin(); ecurr != faces_.end(); ++ecurr)
      if (ecurr->second->owner() == Core::Communication::my_mpi_rank(get_comm())) nummyfaces++;

    numglobalfaces = Core::Communication::sum_all(nummyfaces, get_comm());
  }

  // print head
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    os << "--------------------------------------------------\n";
    os << "discretization: " << name() << std::endl;
    os << "--------------------------------------------------\n";
    os << numglobalfaces << " Faces (global)\n";
    os << "--------------------------------------------------\n";
    if (filled())
      os << "Filled() = true\n";
    else
      os << "Filled() = false\n";
    os << "--------------------------------------------------\n";
  }
  // print elements
  for (int proc = 0; proc < Core::Communication::num_mpi_ranks(get_comm()); ++proc)
  {
    if (proc == Core::Communication::my_mpi_rank(get_comm()))
    {
      if ((int)faces_.size()) os << "-------------------------- Proc " << proc << " :\n";
      std::map<int, std::shared_ptr<Core::Elements::FaceElement>>::const_iterator curr;
      for (curr = faces_.begin(); curr != faces_.end(); ++curr)
      {
        os << *(curr->second);
        os << std::endl;
      }
      os << std::endl;
    }
    Core::Communication::barrier(get_comm());
  }

  return;
}

FOUR_C_NAMESPACE_CLOSE
