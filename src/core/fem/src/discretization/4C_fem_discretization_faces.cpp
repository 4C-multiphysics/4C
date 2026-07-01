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
        if (surf_it->second.get_source_peid() != -1)
          FOUR_C_THROW("source peid should not be set!!!");
        // if found -> add second neighbor data to existing data
        surf_it->second.set_source_peid(ele->id());
        surf_it->second.set_l_surface_source(iele);

        std::vector<int> localtrafomap;

        // get the face's nodes sorted w.r.t local coordinate system of the parent's face element
        const std::vector<Core::Nodes::Node*> nodes_face_target = surf_it->second.get_nodes();
        if (nodes_face_target.size() != nnode)
          FOUR_C_THROW(
              "the number of the face w.r.t parent element and source element are not the same. "
              "That is wrong!");

        // find the nodes given with the target element node numbering also for the source element
        // to define a connectivity map between the local face's coordinate systems
        for (unsigned int inode = 0; inode < nnode; inode++)  // target face nodes
        {
          int position = -1;
          for (std::size_t knode = 0; knode < nodes.size(); knode++)
          {
            if (nodes[knode] == nodes_face_target[inode]) position = knode;
          }

          if (position >= 0)
            localtrafomap.push_back(position);
          else
            FOUR_C_THROW(
                "face's node from target's face element not found in source's face element!");
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
  const std::map<int, std::vector<int>>* col_pbc_map_target_to_source =
      get_all_pbc_coupled_col_nodes();

  std::map<std::vector<int>, InternalFacesData>::iterator face_it;
  for (face_it = surfmapdata.begin(); face_it != surfmapdata.end(); ++face_it)
  {
    int target_peid = face_it->second.get_target_peid();
    int source_peid = face_it->second.get_source_peid();
    if (target_peid == -1) FOUR_C_THROW("Face target expected!");

    FOUR_C_ASSERT(target_peid == g_element(target_peid)->id(), "Internal error");
    FOUR_C_ASSERT(
        source_peid == -1 || source_peid == g_element(source_peid)->id(), "Internal error");

    // check for potential periodic boundary conditions and connect respective faces/elements
    if (col_pbc_map_target_to_source)
    {
      // unconnected face is potential pbc face
      if (source_peid == -1)
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
        // for target and source
        std::map<int, std::set<int>> target_to_pbc_set;
        std::map<int, std::set<int>> source_to_pbc_set;
        for (auto& mypbc : mypbcs)
        {
          const int zero_based_id = mypbc->parameters().get<int>("ID") - 1;

          const auto my_target_source_toggle =
              mypbc->parameters().get<std::string>("MASTER_OR_SLAVE");

          if (my_target_source_toggle == "Master")
          {
            // get global target node ids
            const std::vector<int>* target_ids_to_add = mypbc->get_nodes();

            // store them in list depending on the pbc id
            for (int idtoadd : *target_ids_to_add)
            {
              (target_to_pbc_set[zero_based_id]).insert(idtoadd);
            }
          }
          else if (my_target_source_toggle == "Slave")
          {
            // get global source node ids
            const std::vector<int>* source_ids_to_add = mypbc->get_nodes();

            // store them in list depending on the pbc id
            for (int idtoadd : *source_ids_to_add)
            {
              (source_to_pbc_set[zero_based_id]).insert(idtoadd);
            }
          }
          else
            FOUR_C_THROW("Unknown type for pbc!");
        }

        // provide vectors for target and source node ids
        std::vector<int> my_target_node_ids;
        std::vector<int> my_source_node_ids;
        // provide vector for undefined nodes
        // i.e., special nodes on edges or in corners
        // for multiple pbc sets target nodes of boundary condition
        // become source nodes
        // e.g. for two sets two target nodes at the corners become source nodes
        //
        //                PBC T surface
        //           M------------------------S
        //           |                        |
        //  PBC T    |                        | PBC S
        //  surface  |                        | surface
        //           |                        |
        //           S------------------------S
        //                PBC S surface
        // these nodes are not contained in the list col_pbc_map_target_to_source as target nodes
        // but result in more than one source node for the corner or edge target
        std::vector<int> further_target_node_ids;

        // local (or face) target to source coupling
        std::map<int, int> local_pbc_map_target_to_source;

        // bool to indicate if source element has been found and should be added to the patch
        bool add_source_ele_to_face = true;

        // loop node ids of current face and check if they are contained in the list
        // of all target node ids
        for (std::size_t inode = 0; inode < mynodeids.size(); inode++)
        {
          if (col_pbc_map_target_to_source->find(mynodeids[inode]) !=
              col_pbc_map_target_to_source->end())
          {
            // add node id to list of current target nodes
            my_target_node_ids.push_back(mynodeids[inode]);
          }
          else
          {
            // if node is not in (target) list col_pbc_map_target_to_source, it may be special
            // node as explained above
            // check whether node is target and source due to several pbcs
            bool found = false;
            // loop all target sets
            for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
            {
              if ((target_to_pbc_set[ipbc]).find(mynodeids[inode]) !=
                  (target_to_pbc_set[ipbc]).end())
                found = true;
            }

            // yes, we have a target node here
            // add to list of further target nodes which require special care
            if (found) further_target_node_ids.push_back(mynodeids[inode]);
          }
        }

        if (my_target_node_ids.size() > 0)
        {
          // check if all nodes of the face are targets of pbcs
          // -> this is a target face
          if ((my_target_node_ids.size() + further_target_node_ids.size()) == mynodeids.size())
          {
            // get corresponding source ids
            // do the standard target nodes of col_pbc_map_target_to_source first
            for (std::size_t rr = 0; rr < my_target_node_ids.size(); rr++)
            {
              // this target node has one source node
              if (((*col_pbc_map_target_to_source).at(my_target_node_ids[rr])).size() == 1)
              {
                my_source_node_ids.push_back(
                    ((*col_pbc_map_target_to_source).at(my_target_node_ids[rr]))[0]);
                local_pbc_map_target_to_source[my_target_node_ids[rr]] =
                    ((*col_pbc_map_target_to_source).at(my_target_node_ids[rr]))[0];
              }
              // this target node has several source nodes
              // it is a corner or edge node of two or three pbc sets
              else
              {
                // this is only possible for multiple pbcs
                if (numpbcpairs == 1) FOUR_C_THROW("Two or three pbs sets expected");

                // identify the pbc condition (i.e., pbc id) to which the current face belongs

                std::map<int, int> pbcs_per_target;
                // initialize with zeros
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++) pbcs_per_target[ipbc] = 0;

                // identify pbc set to which target nodes belong
                for (std::size_t imnode = 0; imnode < my_target_node_ids.size(); imnode++)
                {
                  for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                  {
                    std::set<int>::iterator iter =
                        (target_to_pbc_set[ipbc]).find(my_target_node_ids[imnode]);
                    if (iter != (target_to_pbc_set[ipbc]).end()) pbcs_per_target[ipbc] += 1;
                  }
                }

                // all target nodes of current surface share the same pbc id
                int target_pbc_id = -1;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (pbcs_per_target[ipbc] == (int)my_target_node_ids.size())
                  {
                    target_pbc_id = ipbc;
                    break;
                  }
                }

                // find the corresponding source of the current target node

                // the corresponding source node is
                // for 2 pbc sets
                // (i) source node with respect to the pbc id of the target face
                // (ii) target with respect to the remaining pbc sets
                // for 3 pbc sets
                // here to cases may occur
                // target has 7 sources -> corner node
                // this results as for 2 sets in
                // (i) source node with respect to the pbc id of the target face
                // (ii) target with respect to the remaining pbc sets
                // target has 3 sources -> edge node
                // this results in
                // (i) source node with respect to the pbc id of the target face
                // (ii) target with respect to one of the two remaining pbc sets
                //  this special case is marked by flag
                bool three_sets_edge_node = false;
                if (numpbcpairs == 3 and
                    ((*col_pbc_map_target_to_source).at(my_target_node_ids[rr])).size() == 3)
                  three_sets_edge_node = true;

                // pbc id of target face also for the source
                int source_pbc_id = target_pbc_id;
                // identify the remaining pbc sets via their id
                std::vector<int> remaining_target_pbc_ids;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (ipbc != source_pbc_id) remaining_target_pbc_ids.push_back(ipbc);
                }

                // loop all source nodes of the current target and
                // check which node fulfills above conditions
                int act_source_id = -999;
                for (std::size_t isource = 0;
                    isource < ((*col_pbc_map_target_to_source).at(my_target_node_ids[rr])).size();
                    isource++)
                {
                  // get id
                  act_source_id =
                      ((*col_pbc_map_target_to_source).at(my_target_node_ids[rr]))[isource];

                  // check first criterion -> (i)
                  if ((source_to_pbc_set[source_pbc_id]).find(act_source_id) !=
                      (source_to_pbc_set[source_pbc_id]).end())
                  {
                    std::size_t found = 0;
                    // if satisfied
                    // check second criterion -> (ii)
                    for (std::size_t k = 0; k < remaining_target_pbc_ids.size(); k++)
                    {
                      if ((target_to_pbc_set[remaining_target_pbc_ids[k]]).find(act_source_id) !=
                          (target_to_pbc_set[remaining_target_pbc_ids[k]]).end())
                        found++;
                    }

                    if ((not three_sets_edge_node) and found == remaining_target_pbc_ids.size())
                      break;
                    else if (three_sets_edge_node and found == 1)
                      break;
                  }
                }

                // store in list
                my_source_node_ids.push_back(act_source_id);
                local_pbc_map_target_to_source[my_target_node_ids[rr]] = act_source_id;
              }
            }

            // next go to the special masters which occur as slaves in the list
            // col_pbc_map_target_to_source and are indeed edge or corner nodes of target surfaces
            if (further_target_node_ids.size() > 0)
            {
              // identify the pbc condition (i.e., id) to which the current face belongs
              // perform as explained above of the special target nodes with several sources

              std::map<int, int> pbcs_per_target;
              for (int ipbc = 0; ipbc < numpbcpairs; ipbc++) pbcs_per_target[ipbc] = 0;

              // identify pbc set to which target nodes belong
              for (std::size_t imnode = 0; imnode < my_target_node_ids.size(); imnode++)
              {
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  std::set<int>::iterator iter =
                      (target_to_pbc_set[ipbc]).find(my_target_node_ids[imnode]);
                  if (iter != (target_to_pbc_set[ipbc]).end()) pbcs_per_target[ipbc] += 1;
                }
              }
              // identify pbc set to which additional target nodes belong
              for (std::size_t imnode = 0; imnode < further_target_node_ids.size(); imnode++)
              {
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  std::set<int>::iterator iter =
                      (target_to_pbc_set[ipbc]).find(further_target_node_ids[imnode]);
                  if (iter != (target_to_pbc_set[ipbc]).end()) pbcs_per_target[ipbc] += 1;
                }
              }

              // all target nodes of current surface share the same pbc id
              int target_pbc_id = -1;
              for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
              {
                if (pbcs_per_target[ipbc] ==
                    (int)(my_target_node_ids.size() + further_target_node_ids.size()))
                {
                  target_pbc_id = ipbc;
                  break;
                }
              }

              // find the corresponding sources of the additional target nodes

              for (std::size_t ifnode = 0; ifnode < further_target_node_ids.size(); ifnode++)
              {
                // get node id of target
                int act_node_id = further_target_node_ids[ifnode];

                // get list of all potential source nodes
                // i.e., further edge or corner nodes
                std::vector<int> potential_source_ids;

                // first, look for targets with more than one source
                // then, check whether node id (i.e. act_node_id) is contained in source list
                std::map<int, std::vector<int>>::const_iterator target_it;
                for (target_it = col_pbc_map_target_to_source->begin();
                    target_it != col_pbc_map_target_to_source->end(); target_it++)
                {
                  if ((target_it->second).size() > 1)
                  {
                    bool found = false;

                    for (std::size_t k = 0; k < (target_it->second).size(); k++)
                    {
                      if ((target_it->second)[k] == act_node_id)
                      {
                        found = true;
                      }
                    }

                    if (found)
                    {
                      for (std::size_t k = 0; k < (target_it->second).size(); k++)
                        potential_source_ids.push_back((target_it->second)[k]);
                    }

                    if (found) break;
                  }
                }

                if (potential_source_ids.size() == 0) FOUR_C_THROW("Expected to find node!");

                // find the corresponding source of the current target node

                // the corresponding source node is
                // for 2 pbc sets
                // (i) source node with respect to the pbc id of the target face
                // (ii) source with respect to the remaining pbc sets
                // for 3 pbc sets
                // target has 3 sources -> edge node (special case 1)
                // this results in
                // (i) source node with respect to the pbc id of the target face
                // (ii) source with respect to one of the two remaining pbc sets
                // target has 7 sources -> corner node (special case 2)
                // (i) source node with respect to the pbc id of the target face
                // (ii) source or target with respect to the remaining pbc sets
                //      depending on the status of the current node with respect to those pbcs
                // special case 1 marked by flag
                bool three_sets_edge_node = false;
                if (numpbcpairs == 3 and potential_source_ids.size() == 3)
                  three_sets_edge_node = true;

                int source_pbc_id = target_pbc_id;

                std::vector<int> remaining_source_pbc_ids;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (ipbc != source_pbc_id) remaining_source_pbc_ids.push_back(ipbc);
                }

                std::vector<int> remaining_target_pbc_ids;
                for (int ipbc = 0; ipbc < numpbcpairs; ipbc++)
                {
                  if (ipbc != source_pbc_id) remaining_target_pbc_ids.push_back(ipbc);
                }

                // special case 2 marked by flag
                bool corner_node = false;
                int further_target_cond = -1;
                int source_cond_1 = -1;

                // get status of corresponding source with respect to the
                // remaining to pbc sets
                // may be target and source or source and source
                if (numpbcpairs == 3 and potential_source_ids.size() == 7)
                {
                  corner_node = true;

                  // set the sets to check target or source
                  for (std::size_t k = 0; k < remaining_target_pbc_ids.size(); k++)
                  {
                    if ((target_to_pbc_set[remaining_target_pbc_ids[k]]).find(act_node_id) !=
                        (target_to_pbc_set[remaining_target_pbc_ids[k]]).end())
                      further_target_cond = remaining_target_pbc_ids[k];
                  }

                  // corresponding source is pure source
                  // -> we just have to check the source lists
                  // -> similar to the 2 pbc-sets case
                  if (further_target_cond == -1)
                  {
                    corner_node = false;
                  }
                  // corresponding source is target and source
                  // -> we set the source list to check
                  else
                  {
                    if (further_target_cond == 0)
                    {
                      if (source_pbc_id == 1)
                        source_cond_1 = 2;
                      else if (source_pbc_id == 2)
                        source_cond_1 = 1;
                      else
                        FOUR_C_THROW("Same pbc ids?");
                    }
                    else if (further_target_cond == 1)
                    {
                      if (source_pbc_id == 0)
                        source_cond_1 = 2;
                      else if (source_pbc_id == 2)
                        source_cond_1 = 0;
                      else
                        FOUR_C_THROW("Same pbc ids?");
                    }
                    else if (further_target_cond == 2)
                    {
                      if (source_pbc_id == 0)
                        source_cond_1 = 1;
                      else if (source_pbc_id == 1)
                        source_cond_1 = 0;
                      else
                        FOUR_C_THROW("Same pbc ids?");
                    }
                    else
                      FOUR_C_THROW("Unknown pbc id!");
                  }
                }

                // loop all source nodes of the current target and
                // check which node fulfills above conditions
                int act_source_id = -999;
                for (std::size_t isource = 0; isource < potential_source_ids.size(); isource++)
                {
                  // get source node id
                  act_source_id = potential_source_ids[isource];

                  // check first criterion
                  if ((source_to_pbc_set[source_pbc_id]).find(act_source_id) !=
                      (source_to_pbc_set[source_pbc_id]).end())
                  {
                    std::size_t found = 0;
                    // if satisfied
                    // check second criterion
                    if (not corner_node)
                    {
                      // check to source conditions
                      for (std::size_t k = 0; k < remaining_source_pbc_ids.size(); k++)
                      {
                        if ((source_to_pbc_set[remaining_source_pbc_ids[k]]).find(act_source_id) !=
                            (source_to_pbc_set[remaining_source_pbc_ids[k]]).end())
                          found++;
                      }
                    }
                    else
                    {
                      // check a target and a source condition
                      if ((target_to_pbc_set[further_target_cond]).find(act_source_id) !=
                          (target_to_pbc_set[further_target_cond]).end())
                        found++;
                      if ((source_to_pbc_set[source_cond_1]).find(act_source_id) !=
                          (source_to_pbc_set[source_cond_1]).end())
                        found++;
                    }

                    if ((not three_sets_edge_node) and found == remaining_source_pbc_ids.size())
                      break;
                    else if (three_sets_edge_node and found == 1)
                      break;
                  }
                }

                // store in list
                my_source_node_ids.push_back(act_source_id);
                local_pbc_map_target_to_source[further_target_node_ids[ifnode]] = act_source_id;
              }
            }

            // sort the source node ids
            std::sort(my_source_node_ids.begin(), my_source_node_ids.end());
          }
          else
          {
            add_source_ele_to_face = false;
          }

          // this criterion ensures that source set is available on this proc
          int counter = 0;
          for (std::size_t kk = 0; kk < my_target_node_ids.size(); kk++)
          {
            for (auto node : my_row_node_range())
            {
              if (node.global_id() == my_target_node_ids[kk]) counter++;
            }
          }
          for (std::size_t kk = 0; kk < further_target_node_ids.size(); kk++)
          {
            for (auto node : my_row_node_range())
            {
              if (node.global_id() == further_target_node_ids[kk]) counter++;
            }
          }
          if (counter == 0)
          {
            add_source_ele_to_face = false;
          }

          // add source element to the patch
          if (add_source_ele_to_face)
          {
            // get target element
            Core::Elements::Element* target_ele = elecolptr_[0];
            for (fool = elecolptr_.begin(); fool != elecolptr_.end(); ++fool)
            {
              if ((*fool)->id() == target_peid) target_ele = *fool;
            }

            // look for the corresponding source face in the list of all faces
            std::map<std::vector<int>, InternalFacesData>::iterator pbc_surf_it =
                surfmapdata.find(my_source_node_ids);
            if (pbc_surf_it == surfmapdata.end())
            {
              // print some helpful information first
              target_ele->print(std::cout);

              std::cout << "\n source " << std::endl;
              for (std::size_t kk = 0; kk < my_source_node_ids.size(); kk++)
                std::cout << my_source_node_ids[kk] << std::endl;

              FOUR_C_THROW("Expected to find source face!");
            }

            // add source data to target data
            face_it->second.set_source_peid(pbc_surf_it->second.get_target_peid());
            source_peid = face_it->second.get_source_peid();
            face_it->second.set_l_surface_source(pbc_surf_it->second.get_l_surface_target());

            // add connection of coordinate systems for target and source
            std::vector<int> localtrafomap;

            // get the face's nodes sorted w.r.t local coordinate system of the parent's face
            // element
            const std::vector<Core::Nodes::Node*> nodes_face_target = face_it->second.get_nodes();
            // get number of nodes
            unsigned int nnode = nodes_face_target.size();

            // get source nodes
            std::vector<Core::Nodes::Node*> source_nodes = pbc_surf_it->second.get_nodes();

            // find the nodes given with the target element node numbering also for the source
            // element to define a connectivity map between the local face's coordinate systems
            for (unsigned int inode = 0; inode < nnode; inode++)  // target face nodes
            {
              int position = -1;

              for (std::size_t knode = 0; knode < source_nodes.size(); knode++)
              {
                if (source_nodes[knode]->id() ==
                    local_pbc_map_target_to_source[nodes_face_target[inode]->id()])
                  position = knode;
              }

              if (position >= 0)
                localtrafomap.push_back(position);
              else
                FOUR_C_THROW(
                    "face's node from target's face element not found in source's face element!");
            }
            // set in face
            face_it->second.set_local_numbering_map(localtrafomap);
          }
        }
      }
    }

    // create faces
    if (doboundaryfaces_ || (target_peid != -1 && source_peid != -1))
    {
      FOUR_C_ASSERT(target_peid != -1, "At least the target element should be present");
      Core::Elements::Element* parent_target = g_element(target_peid);
      Core::Elements::Element* parent_source = source_peid != -1 ? g_element(source_peid) : nullptr;

      FOUR_C_ASSERT(target_peid == parent_target->id(), "Internal error");
      FOUR_C_ASSERT(source_peid == -1 || source_peid == parent_source->id(), "Internal error");

      // get the unsorted nodes
      std::vector<Core::Nodes::Node*> nodes = face_it->second.get_nodes();

      // get corresponding nodeids
      std::vector<int> nodeids(nodes.size());
      std::transform(
          nodes.begin(), nodes.end(), nodeids.begin(), std::mem_fn(&Core::Nodes::Node::id));

      // create the internal face element
      std::shared_ptr<Core::Elements::FaceElement> surf =
          std::dynamic_pointer_cast<Core::Elements::FaceElement>(parent_target->create_face_element(
              parent_source, nodeids.size(), nodeids.data(), nodes.data(),
              face_it->second.get_l_surface_target(), face_it->second.get_l_surface_source(),
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
      parent_target->set_face(face_it->second.get_l_surface_target(), surf_clone.get());
      if (source_peid != -1)
        parent_source->set_face(face_it->second.get_l_surface_source(), surf_clone.get());
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
  facerowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(eleids.data(), nummyeles), 0, get_comm());
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
  facecolmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, std::span<const int>(eleids.data(), nummyeles), 0, get_comm());
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
}

FOUR_C_NAMESPACE_CLOSE
