// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometric_search_matchingoctree.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::GeometricSearch::MatchingOctree::MatchingOctree()
    : discret_(nullptr),
      tol_(-1.0),
      masterentityids_(nullptr),
      maxtreenodesperleaf_(-1),
      issetup_(false),
      isinit_(false)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::GeometricSearch::MatchingOctree::init(const Core::FE::Discretization& actdis,
    const std::vector<int>& masternodeids, const int maxnodeperleaf, const double tol)
{
  set_is_setup(false);

  discret_ = &actdis;
  masterentityids_ = &masternodeids;
  maxtreenodesperleaf_ = maxnodeperleaf;
  tol_ = tol;

  set_is_init(true);
  return 0;
}  // MatchingOctree::Init

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::GeometricSearch::MatchingOctree::setup()
{
  check_is_init();

  const unsigned int nummygids = masterentityids_->size();

  // extract all masternodes on this proc from the list masternodeids
  std::vector<int> masternodesonthisproc;

  // construct octree if proc has masternodes
  if (nummygids > 0)
  {
    // create initial bounding box for all nodes
    //
    //                 +-            -+
    //                 |  xmin  xmax  |
    //                 |  ymin  ymax  |
    //                 |  zmin  zmax  |
    //                 +-            -+
    //
    Core::LinAlg::SerialDenseMatrix initialboundingbox(3, 2);
    std::array<double, 3> pointcoord;
    calc_point_coordinate(discret_, masterentityids_->at(0), pointcoord.data());
    for (int dim = 0; dim < 3; dim++)
    {
      initialboundingbox(dim, 0) = pointcoord[dim] - tol_;
      initialboundingbox(dim, 1) = pointcoord[dim] + tol_;

      // store coordinates of one point in master plane (later on, one
      // coordinate of the masternode will be substituted by the coordinate
      // of the master plane)
      masterplanecoords_.push_back(pointcoord[dim]);
    }

    for (unsigned locn = 0; locn < nummygids; locn++)
    {
      // check if entity is on this proc
      if (not check_have_entity(discret_, masterentityids_->at(locn)))
      {
        FOUR_C_THROW(
            "MatchingOctree can only be constructed with entities,\n"
            "which are either owned, or ghosted by calling proc.");
      }

      masternodesonthisproc.push_back(masterentityids_->at(locn));

      calc_point_coordinate(discret_, masternodesonthisproc[locn], pointcoord.data());

      for (int dim = 0; dim < 3; dim++)
      {
        initialboundingbox(dim, 0) = std::min(initialboundingbox(dim, 0), pointcoord[dim] - tol_);
        initialboundingbox(dim, 1) = std::max(initialboundingbox(dim, 1), pointcoord[dim] + tol_);
      }
    }

    // create octree root --- initial layer is 0
    // all other layers are generated down here by recursive calls
    int initlayer = 0;

    octreeroot_ = create_octree_element(masternodesonthisproc, initialboundingbox, initlayer);
  }

  set_is_setup(true);
  return 0;
}  // MatchingOctree::Setup

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::GeometricSearch::MatchingOctree::search_closest_entity_on_this_proc(
    const std::vector<double>& x, int& idofclosestpoint, double& distofclosestpoint,
    bool searchsecond)
{
  // flag
  bool nodeisinbox = false;

  nodeisinbox = octreeroot_->is_point_in_bounding_box(x);

  if (nodeisinbox)
  {
    // the node is inside the bounding box. So maybe the closest one is
    // here on this proc -> search for it

    if (octreeroot_ == nullptr)
    {
      FOUR_C_THROW("No root for octree on proc");
    }

    std::shared_ptr<OctreeElement> octreeele = octreeroot_;

    while (!octreeele->is_leaf())
    {
      octreeele = octreeele->return_child_containing_point(x);

      if (octreeele == nullptr)
      {
        FOUR_C_THROW("Child is nullpointer");
      }
    }

    // now get closest point in leaf
    octreeele->search_closest_node_in_leaf(
        x, idofclosestpoint, distofclosestpoint, tol_, searchsecond);
  }

  return nodeisinbox;
}  // MatchingOctree::SearchClosestNodeOnThisProc

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::MatchingOctree::create_global_entity_matching(
    const std::vector<int>& slavenodeids, const std::vector<int>& dofsforpbcplane,
    const double rotangle, std::map<int, std::vector<int>>& midtosid)
{
  check_is_init();
  check_is_setup();

  int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());
  int numprocs = Core::Communication::num_mpi_ranks(discret_->get_comm());

  // map from global masternodeids to distances to their global slave
  // counterpart
  std::map<int, double> diststom;

  // 1) each proc generates a list of its slavenodes
  // 2) the list is communicated in a round robin pattern to all the
  //    other procs.
  //
  // 3) the proc checks the package from each proc and calcs the min
  //    distance on each --- the result is kept if distance is smaller
  //    than on the preceding processors

  //--------------------------------------------------------------------
  // -> 1) create a list of slave nodes on this proc. Pack it.
  std::vector<char> sblockofnodes;
  std::vector<char> rblockofnodes;

  sblockofnodes.clear();
  rblockofnodes.clear();

  Core::Communication::PackBuffer pack_data;

  for (int slavenodeid : slavenodeids)
  {
    if (check_have_entity(discret_, slavenodeid))
    {
      pack_entity(pack_data, discret_, slavenodeid);
    }
  }

  swap(sblockofnodes, pack_data());

  //--------------------------------------------------------------------
  // -> 2) round robin loop
  // create an exporter for point to point communication
  Core::Communication::Exporter exporter(discret_->get_comm());

  for (int np = 0; np < numprocs; np++)
  {
    //--------------------------------------------------
    // Send block to next proc. Receive a block from the last proc
    if (np > 0)  // in the first step, we keep all nodes on this proc
    {
      MPI_Request request;
      int tag = myrank;

      int frompid = myrank;
      int topid = (myrank + 1) % numprocs;

      int length = static_cast<int>(sblockofnodes.size());

      exporter.i_send(frompid, topid, sblockofnodes.data(), length, tag, request);

      // make sure that you do not think you received something if
      // you didn't
      if (!rblockofnodes.empty())
      {
        FOUR_C_THROW("rblockofnodes not empty");
      }

      rblockofnodes.clear();

      // receive from predecessor
      frompid = (myrank + numprocs - 1) % numprocs;
      exporter.receive_any(frompid, tag, rblockofnodes, length);

      if (tag != (myrank + numprocs - 1) % numprocs)
      {
        FOUR_C_THROW("received wrong message (ReceiveAny)");
      }

      exporter.wait(request);

      {
        // for safety
        Core::Communication::barrier(exporter.get_comm());
      }
    }
    else
    {
      // dummy communication
      rblockofnodes = sblockofnodes;
    }

    //--------------------------------------------------
    // Unpack block.
    Communication::UnpackBuffer buffer(rblockofnodes);
    while (!buffer.at_end())
    {
      // extract node data from blockofnodes
      std::vector<char> data;
      extract_from_pack(buffer, data);

      // allocate an "empty node". Fill it with info from
      // extracted node data
      Communication::UnpackBuffer data_buffer(data);
      auto o = std::shared_ptr<Core::Communication::ParObject>(
          Core::Communication::factory(data_buffer));

      // check type of ParObject, and return gid
      const int id = check_valid_entity_type(o);

      //----------------------------------------------------------------
      // there is nothing to do if there are no master nodes on this
      // proc
      if (!masterplanecoords_.empty())
      {
        std::array<double, 3> pointcoord;
        calc_point_coordinate(o.get(), pointcoord.data());

        // get its coordinates
        std::vector<double> x(3);

        if (abs(rotangle) < 1e-13)
        {
          for (int dim = 0; dim < 3; dim++)
          {
            x[dim] = pointcoord[dim];
          }
        }
        else
        {
          // if there is a rotationally symmetric periodic boundary condition:
          // rotate slave plane for making it parallel to the master plane
          x[0] = pointcoord[0] * cos(rotangle) + pointcoord[1] * sin(rotangle);
          x[1] = pointcoord[0] * (-sin(rotangle)) + pointcoord[1] * cos(rotangle);
          x[2] = pointcoord[2];
        }

        // Substitute the coordinate normal to the master plane by the
        // coordinate of the masterplane
        //
        //     |                           |
        //     |                           |
        //     |      parallel planes      |
        //     |-------------------------->|
        //     |                           |
        //     |                           |
        //     |                           |
        //   slave                      master
        //
        //

        // get direction for parallel translation
        if (!dofsforpbcplane.empty())
        {
          int dir = -1;

          for (int dim = 0; dim < 3; dim++)
          {
            if (dofsforpbcplane[0] == dim || dofsforpbcplane[1] == dim)
            {
              // direction dim is in plane
              continue;
            }
            else
            {
              dir = dim;
            }
          }

          if (dir < 0)
          {
            FOUR_C_THROW("Unable to get direction orthogonal to plane");
          }

          // substitute x value
          x[dir] = masterplanecoords_[dir];
        }
        //--------------------------------------------------------
        // 3) now search for closest master point on this proc
        int idofclosestpoint;
        double distofclosestpoint;

        bool nodeisinbox;

        nodeisinbox =
            this->search_closest_entity_on_this_proc(x, idofclosestpoint, distofclosestpoint);

        // If x is not in the bounding box on this proc, its probably not
        // matching a point in the box. We do nothing.
        if (nodeisinbox)
        {
          auto found = midtosid.find(idofclosestpoint);

          if (found != midtosid.end())
          {
            // if this is true, we already have an entry in the list and
            // have to check whether this is a better value
            if (diststom[idofclosestpoint] > distofclosestpoint)
            {
              (midtosid[idofclosestpoint]).clear();
              (midtosid[idofclosestpoint]).push_back(id);
              diststom[idofclosestpoint] = distofclosestpoint;
            }
            else if (diststom[idofclosestpoint] < distofclosestpoint + 1e-9 &&
                     diststom[idofclosestpoint] > distofclosestpoint - 1e-9)
            {
              (midtosid[idofclosestpoint]).push_back(id);
            }
          }
          else
          {
            // this is the first estimate for a closest point
            (midtosid[idofclosestpoint]).clear();
            (midtosid[idofclosestpoint]).push_back(id);
            diststom[idofclosestpoint] = distofclosestpoint;
          }
        }  // end nodeisinbox==true

      }  // end if (masterplanecoords_.empty()!=true)
    }



    //----------------------------------------------------------------
    // prepare to send nodes to next proc (keep list).

    // the received nodes will be sent to the next proc
    sblockofnodes = rblockofnodes;

    // we need a new receive buffer
    rblockofnodes.clear();

    {
      // for safety
      Core::Communication::barrier(exporter.get_comm());
    }
  }  // end loop np
}  // MatchingOctree::CreateGlobalNodeMatching

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::MatchingOctree::find_match(const Core::FE::Discretization& slavedis,
    const std::vector<int>& slavenodeids, std::map<int, std::pair<int, double>>& coupling)
{
  check_is_init();
  check_is_setup();

  int numprocs = Core::Communication::num_mpi_ranks(discret_->get_comm());

  if (Core::Communication::num_mpi_ranks(slavedis.get_comm()) != numprocs)
    FOUR_C_THROW("compared discretizations must live on same procs");

  // 1) each proc generates a list of its slavenodes
  //
  // 2) the list is communicated in a round robin pattern to all the
  //    other procs.
  //
  // 3) the proc checks the package from each proc and calcs the min
  //    distance on each --- the result is kept if distance is smaller
  //    than on the preceding processors

  //--------------------------------------------------------------------
  // -> 1) create a list of slave nodes on this proc. Pack it.
  std::vector<char> sblockofnodes;
  std::vector<char> rblockofnodes;

  Core::Communication::PackBuffer pack_data;

  for (int slavenodeid : slavenodeids)
  {
    if (check_have_entity(&slavedis, slavenodeid))
    {
      pack_entity(pack_data, &slavedis, slavenodeid);
    }
  }

  swap(sblockofnodes, pack_data());

  //--------------------------------------------------------------------
  // -> 2) round robin loop

  // create an exporter for point to point communication
  // We do all communication with the communicator of the original
  // discretization.
  Core::Communication::Exporter exporter(discret_->get_comm());

  for (int np = 0; np < numprocs; np++)
  {
    //--------------------------------------------------
    // Send block to next proc. Receive a block from the last proc
    if (np > 0)  // in the first step, we keep all nodes on this proc
    {
      int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());
      MPI_Request request;
      int tag = myrank;

      int frompid = myrank;
      int topid = (myrank + 1) % numprocs;

      int length = static_cast<int>(sblockofnodes.size());

      exporter.i_send(frompid, topid, sblockofnodes.data(), length, tag, request);

      // make sure that you do not think you received something if
      // you didn't
      if (not rblockofnodes.empty())
      {
        FOUR_C_THROW("rblockofnodes not empty");
      }

      rblockofnodes.clear();

      // receive from predecessor
      frompid = (myrank + numprocs - 1) % numprocs;
      exporter.receive_any(frompid, tag, rblockofnodes, length);

      if (tag != (myrank + numprocs - 1) % numprocs)
      {
        FOUR_C_THROW("received wrong message (ReceiveAny)");
      }

      exporter.wait(request);
    }
    else
    {
      // no need to communicate
      swap(rblockofnodes, sblockofnodes);
    }

    //--------------------------------------------------
    // Unpack block.
    Communication::UnpackBuffer buffer(rblockofnodes);
    while (!buffer.at_end())
    {
      // extract node data from blockofnodes
      std::vector<char> data;
      extract_from_pack(buffer, data);

      // allocate an "empty node". Fill it with info from
      // extracted node data
      Communication::UnpackBuffer data_buffer(data);
      std::shared_ptr<Core::Communication::ParObject> o(Core::Communication::factory(data_buffer));

      // cast ParObject to specific type and return id
      const int id = check_valid_entity_type(o);

      //----------------------------------------------------------------
      // there is nothing to do if there are no master nodes on this
      // proc
      if (not masterplanecoords_.empty())
      {
        std::array<double, 3> pointcoord;
        calc_point_coordinate(o.get(), pointcoord.data());

        // get its coordinates
        std::vector<double> x(pointcoord.begin(), pointcoord.end());

        //--------------------------------------------------------
        // 3) now search for closest master point on this proc
        int gid;
        double dist;

        // If x is not in the bounding box on this proc, its probably not
        // matching a point in the box. We do nothing.
        if (search_closest_entity_on_this_proc(x, gid, dist))
        {
          auto found = coupling.find(gid);

          // search for second point with same distance, if found gid is already in coupling
          if (found != coupling.end())
          {
            if (search_closest_entity_on_this_proc(x, gid, dist, true))
            {
              found = coupling.find(gid);
            }
          }

          // we are interested in the closest match
          if (found == coupling.end() or coupling[gid].second > dist)
          {
            coupling[gid] = std::make_pair(id, dist);
          }
        }
      }
    }

    //----------------------------------------------------------------
    // prepare to send nodes to next proc (keep list).

    // the received nodes will be sent to the next proc
    swap(sblockofnodes, rblockofnodes);

    // we need a new receive buffer
    rblockofnodes.clear();
  }
}  // MatchingOctree::FindMatch

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::MatchingOctree::fill_slave_to_master_gid_mapping(
    const Core::FE::Discretization& slavedis, const std::vector<int>& slavenodeids,
    std::map<int, std::vector<double>>& coupling)
{
  check_is_init();
  check_is_setup();

  int numprocs = Core::Communication::num_mpi_ranks(discret_->get_comm());

  if (Core::Communication::num_mpi_ranks(slavedis.get_comm()) != numprocs)
    FOUR_C_THROW("compared discretizations must live on same procs");

  // 1) each proc generates a list of its slavenodes
  //
  // 2) the list is communicated in a round robin pattern to all the
  //    other procs.
  //
  // 3) the proc checks the package from each proc and calcs the min
  //    distance on each --- the result is kept if distance is smaller
  //    than on the preceding processors

  //--------------------------------------------------------------------
  // -> 1) create a list of slave nodes on this proc. Pack it.
  std::vector<char> sblockofnodes;
  std::vector<char> rblockofnodes;

  Core::Communication::PackBuffer pack_data;

  for (int slavenodeid : slavenodeids) pack_entity(pack_data, &slavedis, slavenodeid);

  swap(sblockofnodes, pack_data());

  //--------------------------------------------------------------------
  // -> 2) round robin loop

  // create an exporter for point to point communication
  // We do all communication with the communicator of the original
  // discretization.
  Core::Communication::Exporter exporter(discret_->get_comm());

  for (int np = 0; np < numprocs; np++)
  {
    //--------------------------------------------------
    // Send block to next proc. Receive a block from the last proc
    if (np > 0)  // in the first step, we keep all nodes on this proc
    {
      int myrank = Core::Communication::my_mpi_rank(discret_->get_comm());
      MPI_Request request;
      int tag = myrank;

      int frompid = myrank;
      int topid = (myrank + 1) % numprocs;

      int length = static_cast<int>(sblockofnodes.size());

      exporter.i_send(frompid, topid, sblockofnodes.data(), length, tag, request);

      // make sure that you do not think you received something if
      // you didn't
      if (not rblockofnodes.empty())
      {
        FOUR_C_THROW("rblockofnodes not empty");
      }

      rblockofnodes.clear();

      // receive from predecessor
      frompid = (myrank + numprocs - 1) % numprocs;
      exporter.receive_any(frompid, tag, rblockofnodes, length);

      if (tag != (myrank + numprocs - 1) % numprocs)
      {
        FOUR_C_THROW("received wrong message (ReceiveAny)");
      }

      exporter.wait(request);
    }
    else
    {
      // no need to communicate
      swap(rblockofnodes, sblockofnodes);
    }

    //--------------------------------------------------
    // Unpack block.
    Communication::UnpackBuffer buffer(rblockofnodes);
    while (!buffer.at_end())
    {
      // extract node data from blockofnodes
      std::vector<char> data;
      unpack_entity(buffer, data);

      // allocate an "empty node". Fill it with info from
      // extracted node data
      Communication::UnpackBuffer data_buffer(data);
      std::shared_ptr<Core::Communication::ParObject> o(Core::Communication::factory(data_buffer));

      const int id = check_valid_entity_type(o);

      //----------------------------------------------------------------
      // there is nothing to do if there are no master nodes on this
      // proc
      if (not masterplanecoords_.empty())
      {
        std::array<double, 3> pointcoord;
        calc_point_coordinate(o.get(), pointcoord.data());

        // get its coordinates
        std::vector<double> x(pointcoord.begin(), pointcoord.end());

        //--------------------------------------------------------
        // 3) now search for closest master point on this proc
        int gid;
        double dist;

        // If x is not in the bounding box on this proc, its probably not
        // matching a point in the box. We do nothing.
        if (search_closest_entity_on_this_proc(x, gid, dist))
        {
          auto found = coupling.find(id);

          // search for second point with same distance,
          // if found gid is already in coupling
          if (found != coupling.end())
          {
            if (search_closest_entity_on_this_proc(x, gid, dist, true))
            {
              found = coupling.find(id);
            }
          }

          // we are interested in the closest match
          if (found == coupling.end() or (coupling[id])[1] > dist)
          {
            if (dist <= tol_)
            {
              bool isrownode = check_entity_owner(discret_, gid);
              std::vector<double> myvec(3);  // initialize vector
              myvec[0] = (double)gid;        // save master gid in vector
              myvec[1] = dist;               // save distance in vector
              myvec[2] = (double)isrownode;  // save master row col info in vector
              coupling[id] = myvec;          // copy vector to map
            }
          }
        }
      }
    }

    //----------------------------------------------------------------
    // prepare to send nodes to next proc (keep list).

    // the received nodes will be sent to the next proc
    swap(sblockofnodes, rblockofnodes);

    // we need a new receive buffer
    rblockofnodes.clear();
  }
}  // MatchingOctree::fill_slave_to_master_gid_mapping


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::GeometricSearch::NodeMatchingOctree::NodeMatchingOctree()
    : MatchingOctree() {}  // NodeMatchingOctree::NodeMatchingOctree

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::NodeMatchingOctree::calc_point_coordinate(
    const Core::FE::Discretization* dis, const int id, double* coord)
{
  Core::Nodes::Node* actnode = dis->g_node(id);

  const int dim = 3;

  const auto x = actnode->x();
  for (size_t idim = 0; idim < x.size(); idim++) coord[idim] = x[idim];
  for (size_t idim = x.size(); idim < dim; idim++) coord[idim] = 0.0;
}  // NodeMatchingOctree::calc_point_coordinate

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
//! calc unique coordinate of entity
void Core::GeometricSearch::NodeMatchingOctree::calc_point_coordinate(
    Core::Communication::ParObject* entity, double* coord)
{
  auto* actnode = dynamic_cast<Core::Nodes::Node*>(entity);
  if (actnode == nullptr) FOUR_C_THROW("dynamic_cast failed");

  const int dim = 3;

  const auto x = actnode->x();
  for (size_t idim = 0; idim < x.size(); idim++) coord[idim] = x[idim];
  for (size_t idim = x.size(); idim < dim; idim++) coord[idim] = 0.0;

}  // NodeMatchingOctree::calc_point_coordinate

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::GeometricSearch::NodeMatchingOctree::check_have_entity(
    const Core::FE::Discretization* dis, const int id)
{
  return dis->have_global_node(id);
}  // NodeMatchingOctree::check_have_entity

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::GeometricSearch::NodeMatchingOctree::check_entity_owner(
    const Core::FE::Discretization* dis, const int id)
{
  return (dis->g_node(id)->owner() == Core::Communication::my_mpi_rank(dis->get_comm()));
}  // NodeMatchingOctree::check_entity_owner

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::NodeMatchingOctree::pack_entity(
    Core::Communication::PackBuffer& data, const Core::FE::Discretization* dis, const int id)
{
  // get the slavenode
  Core::Nodes::Node* actnode = dis->g_node(id);
  // Add node to list of nodes which will be sent to the next proc
  Core::Communication::PackBuffer dummy;
  add_to_pack(dummy, *actnode);
  add_to_pack(data, dummy());

}  // NodeMatchingOctree::PackEntity

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::NodeMatchingOctree::unpack_entity(
    Communication::UnpackBuffer& buffer, std::vector<char>& data)
{
  extract_from_pack(buffer, data);
}  // NodeMatchingOctree::unpack_entity

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::GeometricSearch::NodeMatchingOctree::check_valid_entity_type(
    std::shared_ptr<Core::Communication::ParObject> o)
{
  // cast ParObject to Node
  auto* actnode = dynamic_cast<Core::Nodes::Node*>(o.get());
  if (actnode == nullptr) FOUR_C_THROW("unpack of invalid data");

  return actnode->id();
}  // NodeMatchingOctree::check_valid_entity_type

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::GeometricSearch::OctreeElement>
Core::GeometricSearch::NodeMatchingOctree::create_octree_element(
    std::vector<int>& nodeidstoadd, Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer)
{
  std::shared_ptr<Core::GeometricSearch::OctreeElement> newtreeelement =
      std::make_shared<OctreeNodalElement>();

  newtreeelement->init(
      *discret_, nodeidstoadd, boundingboxtoadd, layer, maxtreenodesperleaf_, tol_);

  newtreeelement->setup();

  return newtreeelement;
}  // NodeMatchingOctree::create_octree_element


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::GeometricSearch::ElementMatchingOctree::ElementMatchingOctree()
    : MatchingOctree() {}  // ElementMatchingOctree::ElementMatchingOctree

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::ElementMatchingOctree::calc_point_coordinate(
    const Core::FE::Discretization* dis, const int id, double* coord)
{
  Core::Elements::Element* actele = dis->g_element(id);

  const int dim = 3;

  for (int idim = 0; idim < dim; idim++) coord[idim] = 0.0;

  for (auto node : actele->node_range())
  {
    const auto x = node.x();
    for (size_t idim = 0; idim < x.size(); idim++) coord[idim] += x[idim];
  }
}  // ElementMatchingOctree::calc_point_coordinate

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::ElementMatchingOctree::calc_point_coordinate(
    Core::Communication::ParObject* entity, double* coord)
{
  auto* actele = dynamic_cast<Core::Elements::Element*>(entity);
  if (actele == nullptr) FOUR_C_THROW("dynamic_cast failed");

  Core::Nodes::Node** nodes = actele->nodes();
  if (nodes == nullptr) FOUR_C_THROW("could not get pointer to nodes");

  const int dim = 3;

  for (int idim = 0; idim < dim; idim++) coord[idim] = 0.0;

  for (auto node : actele->node_range())
  {
    const auto x = node.x();
    for (size_t idim = 0; idim < x.size(); idim++) coord[idim] += x[idim];
  }
}  // ElementMatchingOctree::calc_point_coordinate

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::GeometricSearch::ElementMatchingOctree::check_have_entity(
    const Core::FE::Discretization* dis, const int id)
{
  return dis->have_global_element(id);
}  // ElementMatchingOctree::check_have_entity

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::GeometricSearch::ElementMatchingOctree::check_entity_owner(
    const Core::FE::Discretization* dis, const int id)
{
  return (dis->g_element(id)->owner() == Core::Communication::my_mpi_rank(dis->get_comm()));
}  // ElementMatchingOctree::check_have_entity

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::ElementMatchingOctree::pack_entity(
    Core::Communication::PackBuffer& data, const Core::FE::Discretization* dis, const int id)
{
  // get the slavenode
  Core::Elements::Element* actele = dis->g_element(id);
  Core::Nodes::Node** nodes = actele->nodes();
  // Add node to list of nodes which will be sent to the next proc
  add_to_pack(data, actele->num_node());

  {
    Core::Communication::PackBuffer dummy_buffer;
    add_to_pack(dummy_buffer, *actele);

    add_to_pack(data, dummy_buffer());
  }

  for (int node = 0; node < actele->num_node(); node++) add_to_pack(data, *nodes[node]);
}  // ElementMatchingOctree::PackEntity

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::ElementMatchingOctree::unpack_entity(
    Communication::UnpackBuffer& buffer, std::vector<char>& data)
{
  nodes_.clear();
  int numnode;
  extract_from_pack(buffer, numnode);
  extract_from_pack(buffer, data);

  for (int node = 0; node < numnode; node++)
  {
    std::shared_ptr<Core::Communication::ParObject> o(Core::Communication::factory(buffer));
    std::shared_ptr<Core::Nodes::Node> actnode = std::dynamic_pointer_cast<Core::Nodes::Node>(o);
    if (actnode == nullptr) FOUR_C_THROW("cast from ParObject to Node failed");
    nodes_.insert(std::pair<int, std::shared_ptr<Core::Nodes::Node>>(actnode->id(), actnode));
  }

}  // ElementMatchingOctree::unpack_entity

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::GeometricSearch::ElementMatchingOctree::check_valid_entity_type(
    std::shared_ptr<Core::Communication::ParObject> o)
{
  // cast ParObject to element
  auto* actele = dynamic_cast<Core::Elements::Element*>(o.get());
  if (actele == nullptr) FOUR_C_THROW("unpack of invalid data");

  // set nodal pointers for this element
  actele->build_nodal_pointers(nodes_);

  return actele->id();
}  // ElementMatchingOctree::check_valid_entity_type

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::GeometricSearch::OctreeElement>
Core::GeometricSearch::ElementMatchingOctree::create_octree_element(
    std::vector<int>& nodeidstoadd, Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer)
{
  std::shared_ptr<Core::GeometricSearch::OctreeElement> newtreeelement =
      std::make_shared<OctreeElementElement>();

  newtreeelement->init(
      *discret_, nodeidstoadd, boundingboxtoadd, layer, maxtreenodesperleaf_, tol_);

  newtreeelement->setup();

  return newtreeelement;
}  // ElementMatchingOctree::create_octree_element


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::GeometricSearch::OctreeNodalElement::OctreeNodalElement()
    : OctreeElement() {}  // OctreeElement()

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::OctreeNodalElement::calc_point_coordinate(
    const Core::FE::Discretization* dis, const int id, double* coord)
{
  Core::Nodes::Node* actnode = dis->g_node(id);

  const int dim = 3;

  const auto x = actnode->x();
  for (size_t idim = 0; idim < x.size(); idim++) coord[idim] = x[idim];
  for (size_t idim = x.size(); idim < dim; idim++) coord[idim] = 0.0;
}  // OctreeNodalElement::calc_point_coordinate

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::GeometricSearch::OctreeElement>
Core::GeometricSearch::OctreeNodalElement::create_octree_element(
    std::vector<int>& nodeidstoadd, Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer)
{
  std::shared_ptr<Core::GeometricSearch::OctreeElement> newtreeelement =
      std::make_shared<OctreeNodalElement>();

  newtreeelement->init(
      *discret_, nodeidstoadd, boundingboxtoadd, layer, maxtreenodesperleaf_, tol_);

  newtreeelement->setup();

  return newtreeelement;
}  // OctreeNodalElement::create_octree_element


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::GeometricSearch::OctreeElementElement::OctreeElementElement()
    : OctreeElement() {}  // OctreeElementElement::OctreeElementElement

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::OctreeElementElement::calc_point_coordinate(
    const Core::FE::Discretization* dis, const int id, double* coord)
{
  Core::Elements::Element* actele = dis->g_element(id);

  const int dim = 3;

  for (int idim = 0; idim < dim; idim++) coord[idim] = 0.0;

  for (auto node : actele->node_range())
  {
    const auto x = node.x();
    for (size_t idim = 0; idim < x.size(); idim++) coord[idim] += x[idim];
  }
}  // OctreeElementElement::calc_point_coordinate

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::GeometricSearch::OctreeElement>
Core::GeometricSearch::OctreeElementElement::create_octree_element(
    std::vector<int>& nodeidstoadd, Core::LinAlg::SerialDenseMatrix& boundingboxtoadd, int layer)
{
  std::shared_ptr<Core::GeometricSearch::OctreeElement> newtreeelement =
      std::make_shared<OctreeElementElement>();

  newtreeelement->init(
      *discret_, nodeidstoadd, boundingboxtoadd, layer, maxtreenodesperleaf_, tol_);

  newtreeelement->setup();

  return newtreeelement;
}  // OctreeElementElement::create_octree_element


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::GeometricSearch::OctreeElement::OctreeElement()
    : discret_(nullptr),
      layer_(-1),
      maxtreenodesperleaf_(-1),
      tol_(-1.0),
      issetup_(false),
      isinit_(false)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::GeometricSearch::OctreeElement::init(const Core::FE::Discretization& actdis,
    std::vector<int>& nodeidstoadd, const Core::LinAlg::SerialDenseMatrix& boundingboxtoadd,
    const int layer, const int maxnodeperleaf, const double tol)
{
  set_is_setup(false);

  discret_ = &actdis;
  boundingbox_ = boundingboxtoadd;
  nodeids_ = nodeidstoadd;
  layer_ = layer;
  maxtreenodesperleaf_ = maxnodeperleaf;
  tol_ = tol;

  set_is_init(true);
  return 0;
}  // OctreeElement::Init

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Core::GeometricSearch::OctreeElement::setup()
{
  check_is_init();

  if (layer_ > 200)
  {
    FOUR_C_THROW("max.depth of octree: 200. Can't append further children\n");
  }

  const int numnodestoadd = static_cast<int>(nodeids_.size());
  // if number of slavenodes on this proc is too large split the element
  if (numnodestoadd > maxtreenodesperleaf_)
  {
    // mean coordinate value in direction of the longest edge
    std::array<double, 3> mean;
    std::array<double, 3> pointcoord;

    for (int dim = 0; dim < 3; dim++)
    {
      mean[dim] = 0.0;
      pointcoord[dim] = 0.0;
    }

    // calculate mean coordinate for all directions
    for (int locn = 0; locn < numnodestoadd; locn++)
    {
      calc_point_coordinate(discret_, nodeids_.at(locn), pointcoord.data());

      for (int dim = 0; dim < 3; dim++) mean[dim] += pointcoord[dim];
    }

    for (int dim = 0; dim < 3; dim++) mean[dim] = mean[dim] / numnodestoadd;

    // direction specifies which side will be cut (the one with the largest
    // value of the mean distance to the "lower" boundary)
    int direction = 0;


    // the maximum distance of the mean value from the boundary of the box
    double maxdist = 0;
    double wheretocut = 0;

    // get the coordinate with the maximum distance
    for (int dim = 0; dim < 3; dim++)
    {
      double thisdist =
          std::min(mean[dim] - boundingbox_(dim, 0), boundingbox_(dim, 1) - mean[dim]);

      if (maxdist < thisdist)
      {
        maxdist = thisdist;
        wheretocut = mean[dim];
        direction = dim;
      }
    }

    // Why choose the coordinate with the maximum distance from both edges
    // and not simply the longest edge?
    //
    // Here is what happens if you do so:
    //
    //
    //  +-------------+
    //  |             |
    //  X             |
    //  |             |     (*)
    //  X             |
    //  |             |
    //  |             |
    //  +-------------+
    //   longest edge
    //
    // This leads to two children:
    //
    //
    // +--------------+
    // |              |
    // |X             |
    // |              |
    // |X             |
    // |              |
    // |              |
    // +--------------+
    //
    // and
    //
    // +-+
    // | |
    // |X|
    // | |
    // |X|
    // | |
    // | |
    // +-+
    //
    // This means an endless loop from the problem (*) ...

    // create bounding boxes for the children
    Core::LinAlg::SerialDenseMatrix childboundingbox1(3, 2);
    Core::LinAlg::SerialDenseMatrix childboundingbox2(3, 2);

    // initialise the new bounding boxes with the old values
    for (int dim = 0; dim < 3; dim++)
    {
      childboundingbox1(dim, 0) = boundingbox_(dim, 0);
      childboundingbox1(dim, 1) = boundingbox_(dim, 1);

      childboundingbox2(dim, 0) = boundingbox_(dim, 0);
      childboundingbox2(dim, 1) = boundingbox_(dim, 1);
    }

    // replace the boundaries of component direction to divide cells
    // create initial bounding box for all nodes
    //
    // example direction = 1 , e. g. y-direction:
    //
    // mean = ymin + maxdist
    //
    //   +-            -+      +-               -+   +-               -+
    //   |  xmin  xmax  |      |  xmin  xmax     |   |  xmin     xmax  |
    //   |  ymin  ymax  | ---> |  ymin  mean+eps | + |  mean-eps ymax  |
    //   |  zmin  zmax  |      |  zmin  zmax     |   |  zmin     zmax  |
    //   +-            -+      +-               -+   +-               -+
    //
    //                         lower bounding box    upper bounding box
    //


    childboundingbox1(direction, 1) = wheretocut + tol_;
    childboundingbox2(direction, 0) = wheretocut - tol_;

    // distribute nodes to children
    std::vector<int> childnodeids1;
    std::vector<int> childnodeids2;
    for (int& nodeid : nodeids_)
    {
      calc_point_coordinate(discret_, nodeid, pointcoord.data());

      // node is in "lower" bounding box
      if (pointcoord[direction] < childboundingbox1(direction, 1))
      {
        childnodeids1.push_back(nodeid);
      }
      // node is in "upper" bounding box
      if (pointcoord[direction] > childboundingbox2(direction, 0))
      {
        childnodeids2.push_back(nodeid);
      }
    }

    // we do not need the full node id vector anymore --- it was distributed
    // to the children --> throw it away
    nodeids_.clear();

    // append children to parent
    octreechild1_ = create_octree_element(childnodeids1, childboundingbox1, layer_ + 1);

    octreechild2_ = create_octree_element(childnodeids2, childboundingbox2, layer_ + 1);

  }  // end number of slavenodes on this proc is too large split the element
  else
  {
    if ((int)nodeids_.size() == 0)
    {
      FOUR_C_THROW("Trying to create leaf with no nodes. Stop.");
    }
  }

  set_is_setup(true);
  return 0;
}  // OctreeElement::Setup

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::OctreeElement::search_closest_node_in_leaf(const std::vector<double>& x,
    int& idofclosestpoint, double& distofclosestpoint, const double& elesize, bool searchsecond)
{
  check_is_init();
  check_is_setup();

  double thisdist;
  std::vector<double> dx(3);
  std::array<double, 3> pointcoord;

  // the first node is the guess for the closest node
  calc_point_coordinate(discret_, nodeids_.at(0), pointcoord.data());
  for (int dim = 0; dim < 3; dim++)
  {
    dx[dim] = pointcoord[dim] - x[dim];
  }

  distofclosestpoint = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
  idofclosestpoint = nodeids_.at(0);

  // now loop the others and check whether they are better
  for (int nn = 1; nn < (int)nodeids_.size(); nn++)
  {
    calc_point_coordinate(discret_, nodeids_.at(nn), pointcoord.data());

    for (int dim = 0; dim < 3; dim++)
    {
      dx[dim] = pointcoord[dim] - x[dim];
    }
    thisdist = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);

    if (thisdist < (distofclosestpoint - 1e-02 * elesize))
    {
      distofclosestpoint = thisdist;
      idofclosestpoint = nodeids_.at(nn);
    }
    else
    {
      if ((abs(thisdist - distofclosestpoint) < 1e-02 * elesize) && searchsecond)
      {
        distofclosestpoint = thisdist;
        idofclosestpoint = nodeids_.at(nn);
      }
    }
  }

}  // OctreeElement::search_closest_node_in_leaf

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::GeometricSearch::OctreeElement::is_point_in_bounding_box(const std::vector<double>& x)
{
  check_is_init();
  check_is_setup();

  bool nodeinboundingbox = true;

  for (int dim = 0; dim < 3; dim++)
  {
    // check whether node is outside of bounding box
    if ((x[dim] < this->boundingbox_(dim, 0)) || (x[dim] > this->boundingbox_(dim, 1)))
    {
      nodeinboundingbox = false;
    }
  }

  return nodeinboundingbox;
}  // OctreeElement::is_point_in_bounding_box

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::GeometricSearch::OctreeElement>
Core::GeometricSearch::OctreeElement::return_child_containing_point(const std::vector<double>& x)
{
  check_is_init();
  check_is_setup();

  std::shared_ptr<OctreeElement> nextelement;

  if (this->octreechild1_ == nullptr || this->octreechild2_ == nullptr)
  {
    FOUR_C_THROW("Asked leaf element for further children.");
  }

  if (this->octreechild1_->is_point_in_bounding_box(x))
  {
    nextelement = this->octreechild1_;
  }
  else if (this->octreechild2_->is_point_in_bounding_box(x))
  {
    nextelement = this->octreechild2_;
  }
  else
  {
    FOUR_C_THROW("point in no bounding box of children, but in parent bounding box!");
  }

  return nextelement;
}  // OctreeElement::return_child_containing_point

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Core::GeometricSearch::OctreeElement::is_leaf()
{
  bool isleaf = true;

  if (this->nodeids_.size() == 0)
  {
    isleaf = false;
  }

  return isleaf;
}  // OctreeElement::IsLeaf

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::GeometricSearch::OctreeElement::print(std::ostream& os) const
{
  // Print id and coordinates
  os << "Leaf in Layer " << layer_ << " Nodes ";

  for (int nodeid : nodeids_)
  {
    os << nodeid << " ";
  }
  os << '\n';
}  // OctreeElement::print(ostream& os)

FOUR_C_NAMESPACE_CLOSE
