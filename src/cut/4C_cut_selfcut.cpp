// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_selfcut.hpp"

#include "4C_cut_kernel.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_meshhandle.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_output.hpp"
#include "4C_cut_point_impl.hpp"
#include "4C_cut_pointgraph.hpp"
#include "4C_cut_pointpool.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_triangulateFacet.hpp"
#include "4C_fem_geometry_searchtree.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

// #define DEBUG_SELFCUT

/*-------------------------------------------------------------------------------------*
 * constructor                                                              wirtz 05/13
 *-------------------------------------------------------------------------------------*/
Cut::SelfCut::SelfCut(MeshHandle& cut_mesh_handle, int myrank)
    : myrank_(myrank), mesh_(cut_mesh_handle.linear_mesh()), meshhandle_(cut_mesh_handle)
{
  meshsizeparam_ = 1e200;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator sit = mesh_.sides().begin();
      sit != mesh_.sides().end(); ++sit)
  {
    std::shared_ptr<BoundingBox> sbb(BoundingBox::create(*(sit->second)));
    meshsizeparam_ = std::min(meshsizeparam_, sbb->diagonal());
  }
}

void Cut::SelfCut::perform_self_cut()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 2/6 --- Cut_Mesh --- SELFCUT");
  if (collision_detection())
  {
    mesh_intersection();
    element_selection();
    if (mesh_.get_options().self_cut_do_mesh_correction()) mesh_correction();
  }
}

/*-------------------------------------------------------------------------------------*
 * detects cutsides which cut each other by finding the respective cutpoints
 *                                                                          wirtz 05/13
 *-------------------------------------------------------------------------------------*/
bool Cut::SelfCut::collision_detection()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 2/6 --- Cut_Mesh --- SELFCUT --- CollisionDetection");

  find_cutting_sides();
  find_self_cut_points();
  get_self_cut_objects();

  return (selfcut_sides_.size() > 0);
}

/*-------------------------------------------------------------------------------------*
 * replaces cut sides by creating new nodes, edges and sides             wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::mesh_intersection()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 2/6 --- Cut_Mesh --- SELFCUT --- MeshIntersection");

  create_self_cut_nodes();
  create_self_cut_edges();
  find_self_cut_triangulation();
  create_self_cut_sides();
  erase_cut_sides();
  erase_cut_edges();
}

/*-------------------------------------------------------------------------------------*
 * erases nodes, edges and sides which lies inside a structure body by
 * locating there position                                                  wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::element_selection()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 2/6 --- Cut_Mesh --- SELFCUT --- ElementSelection");

  determine_self_cut_position();
  propagate_self_cut_position();
  erase_inside_sides();
  erase_inside_edges();
  erase_inside_nodes();
}

/*-------------------------------------------------------------------------------------*
 * repair the mesh from potential islands                                   wirtz 07/16
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::mesh_correction()
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 2/6 --- Cut_Mesh --- SELFCUT --- MeshCorrection");

  construct_connectivity();
  find_islands();
  erase_inside_sides();
  erase_inside_edges();
  erase_inside_nodes();
}

/*-------------------------------------------------------------------------------------*
 * represents the result by using console or gmsh                           wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::status_inspection()
{
  cut_side_status_text();
  cut_mesh_status_text();
  cut_side_status_gmsh("selfcut.pos");
  s_cmgm_gmsh("SCmgmGmsh.pos");
}

/*-------------------------------------------------------------------------------------*
 * finds cutsides which possibly cuts the considered side                   wirtz 05/13
 * we construct bounding box over each side and collect all cut sides that intersect
 * this bounding box. Hence possible self cut sides that we collect here may not
 * actually produce self-cut --> we check this in subsequent procedure
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::find_cutting_sides()
{
  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = mesh_.sides();
  const std::map<int, Core::LinAlg::Matrix<3, 2>>& selfcutbvs = mesh_.self_cut_bvs();
  const std::shared_ptr<Core::Geo::SearchTree>& selfcuttree = mesh_.self_cut_tree();
  const std::map<int, Side*>& shadowsides = mesh_.shadow_sides();

  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    Core::LinAlg::Matrix<3, 2> cutsideBV = cutside->get_bounding_volume().get_bounding_volume();
    if (selfcutbvs.size() != 0)  // ******************************************************* possible
                                 // in case of parallel computing and using only relevant elements
    {
      std::set<int> collisions;
      // search collision between cutside and elements in the static search tree
      selfcuttree->search_collisions(selfcutbvs, cutsideBV, 0, collisions);

      // Quick check if another side consists of the same points
      bool skip_cutside = false;
      for (std::set<int>::iterator ic = collisions.begin(); ic != collisions.end(); ++ic)
      {
        int collisionid = *ic;
        std::map<int, Side*>::const_iterator is = shadowsides.find(collisionid);
        if (is != shadowsides.end())
        {
          Side* s = is->second;
          if (s == cutside) continue;

          if (cutside->all_points_common(*s)) skip_cutside = true;
        }
        if (skip_cutside) break;
      }
      if (skip_cutside) continue;
      // add the cutside to all the elements which have been found
      for (std::set<int>::iterator ic = collisions.begin(); ic != collisions.end(); ++ic)
      {
        int collisionid = *ic;
        std::map<int, Side*>::const_iterator is = shadowsides.find(collisionid);
        if (is != shadowsides.end())
        {
          Side* s = is->second;
          if (s == cutside) continue;
          if (not cutside->have_common_node(*s))
          {
            // if "side" and "cutside" has two nodes at the same position, we merge them
            // if these two sides overlap within BB only because of coinciding nodes,
            // we do not take this as self-cut
            if (merge_coinciding_nodes(cutside, s))
            {
              // even after merging nodes, if two sides not have a common node --> self-cut possible
              if (not cutside->have_common_node(*s))
              {
                cutside->get_cutting_side(s);
              }
            }
            else
            {
              cutside->get_cutting_side(s);
            }
          }
        }
      }
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * If two nodes are at the same position, then this function will carry out
 * equivalent operations such that only one node is available at this place
 * Connectivity information are incorporated accordingly                       sudhakar 10/13
 *
 * In the following, nodes 4-5 and 3-6 are at the same position, and the two
 * Quads are not connected. In order to avoid many complicated routines of handling such
 * coinciding nodes in self-cut libraries, we delete one corresponding node and ensure
 * connectivity between the Quads. The cut library can easily handle this modified confi.
 *
 *   1     4,5     8               1      4      8
 *    +-----+-----+                 +-----+-----+
 *    |     |     |                 |     |     |
 *    |     |     |     ===>        |     |     |
 *    |     |     |                 |     |     |
 *    +-----+-----+                 +-----+-----+
 *   2     3,6     7               2      6     7
 *
 *   Remember : Which one of the nodes is considered among the coinciding nodes is
 *   not reproducible (just like many operations in cut libraries)
 *-------------------------------------------------------------------------------------*/
bool Cut::SelfCut::merge_coinciding_nodes(Side* keep, Side* replace)
{
  bool merge = false;

  const std::vector<Node*>& cutnodes = keep->nodes();

  const std::vector<Node*>& scnodes = replace->nodes();

  // store the nodes that should be replaced with other ids
  std::vector<std::pair<const Node*, const Node*>> ids_replace;

  for (std::vector<Node*>::const_iterator no_it = scnodes.begin(); no_it != scnodes.end(); no_it++)
  {
    const Node* scnod = *no_it;

    for (std::vector<Node*>::const_iterator cutit = cutnodes.begin(); cutit != cutnodes.end();
        cutit++)
    {
      const Node* cutnod = *cutit;

      // if both nodes are the same - no need to do anything
      if (scnod->id() == cutnod->id()) continue;

      // Two nodes are at the same location in the initial mesh
      if (scnod->point()->id() == cutnod->point()->id())
      {
        std::pair<const Node*, const Node*> nodpair;
        nodpair.first = scnod;
        nodpair.second = cutnod;

        ids_replace.push_back(nodpair);
      }

      // In moving boundary simulation, two nodes from different discretizations
      // move such that two nodes reached to the same position
      // if we implement this, the previous loop (two nodes at the same position in initial mesh)
      // can be erased
      // I am not even sure whether this special check is necessary (Sudhakar)
      // because we always works with a copy of cut mesh, and only one point is
      // created for two coinciding nodes ---> need to be checked
      else if (cutnod->is_at_same_location(scnod))
      {
        FOUR_C_THROW("not implemented yet");
      }

      // now that we have all the information about coinciding nodes
      // Do the necessary operations in the mesh to consider only one node
      if (ids_replace.size() > 0)
      {
        operations_for_node_merging(ids_replace, true);
        merge = true;
      }
    }
  }

  return merge;
}

/*-----------------------------------------------------------------------------------------*
 * After all information about coinciding nodes are available, perform the       sudhakar 10/13
 * operations in our data structure such that only one node is available and
 * ensure connectivity between the corresponding elements
 *-----------------------------------------------------------------------------------------*/
void Cut::SelfCut::operations_for_node_merging(
    std::vector<std::pair<const Node*, const Node*>> repl, bool initial)
{
  // The following "edge" and "side" data structure has to be modified accordingly
  // the node numbers has to be set accordingly
  std::map<plain_int_set, std::shared_ptr<Edge>>& edges =
      const_cast<std::map<plain_int_set, std::shared_ptr<Edge>>&>(mesh_.edges());
  std::map<plain_int_set, std::shared_ptr<Side>>& sides =
      const_cast<std::map<plain_int_set, std::shared_ptr<Side>>&>(mesh_.sides());

  /*for (std::map<plain_int_set, std::shared_ptr<Edge> >::iterator it =
  edges.begin();it!=edges.end();it++)
  {
    plain_int_set pis = it->first;
    std::cout<<"edge ";
    for( unsigned i=0;i<pis.size();i++ )
      std::cout<<pis[i]<<" ";
    std::cout<<"\n";
  }*/

  // consider each node that needs to be replaced
  for (std::vector<std::pair<const Node*, const Node*>>::iterator it = repl.begin();
      it != repl.end(); it++)
  {
    std::pair<const Node*, const Node*>& pai = *it;
    Node* nod = const_cast<Node*>(pai.first);
    Node* replwith = const_cast<Node*>(pai.second);

    const plain_side_set& sideset = nod->sides();

    // consider each side that are associated with this node
    // and replace it with correct node
    // Within this, internally edge information gets modified correspondingly
    for (plain_side_set::const_iterator i = sideset.begin(); i != sideset.end(); ++i)
    {
      Side* s = *i;
      s->replace_nodes(nod, replwith);
    }

    // modify the "edge" and "side data structure in the mesh
    // delete node "nod" by "replwith" in both edges and sides data-structure
    modify_edge_or_side_map<plain_int_set, std::shared_ptr<Edge>>(edges, nod->id(), replwith->id());
    modify_edge_or_side_map<plain_int_set, std::shared_ptr<Side>>(sides, nod->id(), replwith->id());
  }
}
/*----------------------------------------------------------------------------------------------------*
 * Templated function to delete "nod" by "replwith" in both edge and side data-structures sudhakar
 *10/13 This is a bit tricky because we need to modify the "key" of this std::map "Key" of a
 *std::map cannot be modified directly. So, A new element with correct information is added, and old
 *element is deleted
 *----------------------------------------------------------------------------------------------------*/
template <typename A, typename B>
void Cut::SelfCut::modify_edge_or_side_map(std::map<A, B>& data, int nod, int replwith)
{
  typedef typename std::map<A, B>::iterator ittype;
  typedef typename A::iterator A_ittype;

  std::vector<ittype> eraseThese;

  // -----
  // loop over each element in data to check whether "nod" is available
  // if found, then we create a new element for data with correct "key" in the map
  // we add this and delete the element with wrong key
  // -----
  for (ittype itmap = data.begin(); itmap != data.end(); itmap++)
  {
    A idset = itmap->first;
    A newids;
    bool modi = false;

    for (A_ittype ii = idset.begin(); ii != idset.end(); ii++)
    {
      if (*ii == nod)
      {
        newids.insert(replwith);
        modi = true;
      }
      else
        newids.insert(*ii);
    }

    // insert correct elements
    if (modi)
    {
      data[newids] = data[idset];
      eraseThese.push_back(itmap);
    }
  }

  typedef typename std::vector<ittype>::iterator itc;

  // delete the elements with "wrong" key
  for (itc ite = eraseThese.begin(); ite != eraseThese.end(); ite++)
  {
    ittype era = *ite;
    data.erase(era);
  }
}

/*-------------------------------------------------------------------------------------*
 * finds cutpoints by searching the intersection between the edges
 * of one side with the other side and vice versa                           wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::find_self_cut_points()
{
  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = mesh_.sides();
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const plain_side_set& possiblecuttingsides = cutside->cutting_sides();
    plain_side_set nocuttingsides;
    for (plain_side_set::const_iterator i = possiblecuttingsides.begin();
        i != possiblecuttingsides.end(); ++i)
    {
      Side* possiblecuttingside = *i;

      PointSet selfcutpoints;
      perform_self_cut(*cutside, *possiblecuttingside, selfcutpoints);
      perform_self_cut(*possiblecuttingside, *cutside, selfcutpoints);
      cutside->get_self_cut_points(selfcutpoints);

      // possible cut sides that are intersecting the bounding box of "this" side
      // but not actually cutting, are stored for deletion
      if (selfcutpoints.size() == 0)
      {
        nocuttingsides.insert(possiblecuttingside);
        possiblecuttingside->erase_cutting_side(cutside);
      }
    }
    for (plain_side_set::iterator i = nocuttingsides.begin(); i != nocuttingsides.end(); ++i)
    {
      Side* nocuttingside = *i;
      cutside->erase_cutting_side(nocuttingside);
    }
  }
#ifdef DEBUG_SELFCUT
  Debug("FindSelfCutPoints");
#endif
}

/*-------------------------------------------------------------------------------------*
 * gets all cut sides and their nodes and edges to store them as
 * private variables                                                         wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::get_self_cut_objects()
{
  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = mesh_.sides();
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    std::shared_ptr<Side> cutside = i->second;
    if (cutside->cutting_sides().size() != 0)
    {
      plain_int_set cutsidenodeids = i->first;
      selfcut_sides_[cutsidenodeids] = cutside;
    }
  }

  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    get_self_cut_edges(*cutside);
  }
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    const std::vector<Node*>& cutsidenodes = cutside->nodes();
    for (std::vector<Node*>::const_iterator i = cutsidenodes.begin(); i != cutsidenodes.end(); ++i)
    {
      int cutsidenodeid = (*i)->id();
      const std::map<int, std::shared_ptr<Node>>& cutsidenodercp = mesh_.nodes();
      std::map<int, std::shared_ptr<Node>>::const_iterator nodeiterator =
          cutsidenodercp.find(cutsidenodeid);
      selfcut_nodes_[cutsidenodeid] = nodeiterator->second;
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * creates a node at every selfcutpoint with respect to the
 * corresponding cutsides                                                   wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::create_self_cut_nodes()
{
  std::vector<Node*> cutsidesnodes;
  for (std::map<int, std::shared_ptr<Node>>::iterator i = selfcut_nodes_.begin();
      i != selfcut_nodes_.end(); ++i)
  {
    Node* cutsidesnode = &*i->second;
    cutsidesnodes.push_back(cutsidesnode);
  }

  // we take all selfcut points of each side
  // 1.if these points are at any of the nodes of cut side, then no need to create a new node
  //   just set their position as "oncutsurface"
  // 2.If not we create a new node at this location and add it to the mesh, and corresponding cut
  // side
  //   set their position as "oncutsurface"
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    const PointSet& cutsideselfcutpoints = cutside->self_cut_points();
    for (PointSet::const_iterator i = cutsideselfcutpoints.begin(); i != cutsideselfcutpoints.end();
        ++i)
    {
      Point* cutsideselfcutpoint = *i;
      if (cutsideselfcutpoint->nodal_point(cutsidesnodes))
      {
        for (std::vector<Node*>::iterator i = cutsidesnodes.begin(); i != cutsidesnodes.end(); ++i)
        {
          Node* cutsidesnode = *i;
          if (cutsidesnode->point() == cutsideselfcutpoint)
          {
            cutsidesnode->self_cut_position(Point::oncutsurface);
            cutside->get_self_cut_node(cutsidesnode);
            break;
          }
        }
      }
      else
      {
        const std::map<int, std::shared_ptr<Node>>& cutsidenodes = mesh_.nodes();
        int selfcutnodeid = cutsidenodes.begin()->first - 1;
        Node* selfcutnode = new Node(selfcutnodeid, cutsideselfcutpoint, 0.0);
        selfcutnode->self_cut_position(Point::oncutsurface);
        mesh_.get_node(selfcutnodeid, selfcutnode);
        const std::map<int, std::shared_ptr<Node>>& selfcutnodercp = mesh_.nodes();
        std::map<int, std::shared_ptr<Node>>::const_iterator nodeiterator =
            selfcutnodercp.find(selfcutnodeid);
        selfcut_nodes_[selfcutnodeid] = nodeiterator->second;
        cutside->get_self_cut_node(selfcutnode);
        cutsidesnodes.push_back(selfcutnode);
      }
    }
  }
#ifdef DEBUG_SELFCUT
  Debug("CreateSelfCutNodes");
#endif
}

/*-------------------------------------------------------------------------------------*
 * creates a edge between two selfcutnodes with respect to the
 * corresponding cutsides                                                   wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::create_self_cut_edges()
{
  // Find common nodes between a cut side and its self-cutting sides
  // create edges between these common nodes
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    const plain_side_set& cuttingsides = cutside->cutting_sides();
    const plain_node_set& cutside_nodes = cutside->self_cut_nodes();
    for (plain_side_set::const_iterator i = cuttingsides.begin(); i != cuttingsides.end(); ++i)
    {
      Side* cuttingside = *i;
      const plain_node_set& cuttingside_nodes = cuttingside->self_cut_nodes();
      std::vector<Node*> commonselfcutnodes;
      plain_int_set commonselfcutnodeids;
      for (plain_node_set::const_iterator i = cutside_nodes.begin(); i != cutside_nodes.end(); ++i)
      {
        Node* cutsideselfcutnode = *i;
        plain_node_set::const_iterator nodeiterator = cuttingside_nodes.find(cutsideselfcutnode);
        if (nodeiterator != cuttingside_nodes.end())
        {
          Node* commonselfcutnode = *nodeiterator;
          commonselfcutnodes.push_back(commonselfcutnode);
          commonselfcutnodeids.insert(commonselfcutnode->id());
        }
      }
      if (commonselfcutnodeids.size() == 2)
      {
        std::map<plain_int_set, std::shared_ptr<Edge>>::iterator edgeiterator =
            selfcut_edges_.find(commonselfcutnodeids);
        if (edgeiterator !=
            selfcut_edges_.end())  // this edge is already formed when processing other side
        {
          Edge* commonedge = &*edgeiterator->second;
          commonedge->self_cut_position(Point::oncutsurface);
          cutside->get_self_cut_edge(commonedge);
        }
        else
        {
          std::shared_ptr<Edge> selfcutedge =
              Cut::Edge::create(Core::FE::CellType::line2, commonselfcutnodes);
          selfcutedge->self_cut_position(Point::oncutsurface);
          mesh_.get_edge(commonselfcutnodeids, selfcutedge);
          const std::map<plain_int_set, std::shared_ptr<Edge>>& cutsideedgercp = mesh_.edges();
          std::map<plain_int_set, std::shared_ptr<Edge>>::const_iterator edgeiterator =
              cutsideedgercp.find(commonselfcutnodeids);
          selfcut_edges_[commonselfcutnodeids] = edgeiterator->second;
          cutside->get_self_cut_edge(selfcutedge.get());
        }
      }
    }
  }
#ifdef DEBUG_SELFCUT
  Debug("CreateSelfCutEdges");
#endif
}

/*-------------------------------------------------------------------------------------*
 * finds triangles with respect to the self cut by using
 * a pointgraph and a triangulation method                                  wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::find_self_cut_triangulation()
{
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = i->second.get();
    const std::vector<Node*>& cutsidenodes = cutside->nodes();

    // find the equation of plane of this SIDE
    // this eqn is used to make sure that the axillary sides that resulting from
    // the triangulation preserve the normal direction of SIDE
    std::vector<Point*> cutsidepoints(cutsidenodes.size());
    for (unsigned ic = 0; ic < cutsidenodes.size(); ic++)
      cutsidepoints[ic] = cutsidenodes[ic]->point();

    cutsidepoints[0] = cutsidenodes[0]->point();
    cutsidepoints[1] = cutsidenodes[1]->point();
    cutsidepoints[2] = cutsidenodes[2]->point();
    std::vector<double> cutsideplane = Kernel::eqn_plane_of_polygon(cutsidepoints);

    // -----
    // STEP 1 : Create facets on cut side by taking into account the self-cut points
    // -----
    Impl::PointGraph pointgraph(cutside);
    Cycle* maincycle = &*(pointgraph.fbegin());
    bool IsCorrectNormal = check_normal(cutsideplane, *maincycle);
    std::vector<Cycle*> maincycles;
    std::vector<std::shared_ptr<Cycle>> newmaincyclercp;
    for (Impl::PointGraph::facet_iterator i = pointgraph.fbegin(); i != pointgraph.fend(); ++i)
    {
      Cycle* maincycle = &*i;
      if (not IsCorrectNormal)
      {
        // normal is not in correct direction
        // we reverse the ordering of points to correct this
        maincycle->reverse();
      }
      maincycles.push_back(maincycle);
    }

    std::vector<Cycle*> holecycles;
    bool holenormalconservation = true;
    for (Impl::PointGraph::hole_iterator i = pointgraph.hbegin(); i != pointgraph.hend(); ++i)
    {
      std::vector<Cycle> oldholecycles = *i;
      Cycle* holecycle = oldholecycles.data();
      holenormalconservation = check_normal(cutsideplane, *holecycle);
      break;
    }
    bool firstinnerloop = true;
    for (Impl::PointGraph::hole_iterator i = pointgraph.hbegin(); i != pointgraph.hend(); ++i)
    {
      std::vector<Cycle>* holecyclesold = &*i;
      firstinnerloop = true;
      for (std::vector<Cycle>::iterator i = holecyclesold->begin(); i != holecyclesold->end(); ++i)
      {
        Cycle* holecycle = &*i;
        if ((holenormalconservation and firstinnerloop) or
            (not holenormalconservation and not firstinnerloop))
        {
          maincycles.push_back(holecycle);
        }
        if ((holenormalconservation and not firstinnerloop) or
            (not holenormalconservation and firstinnerloop))
        {
          holecycles.push_back(holecycle);
        }
        firstinnerloop = false;
      }
    }
    std::vector<std::vector<Point*>> mainholecyclepoints;
    for (std::vector<Cycle*>::iterator i = holecycles.begin(); i != holecycles.end(); ++i)
    {
      Cycle* holecycle = *i;
      std::vector<Point*> holecyclepoints = (*holecycle)();
      mainholecyclepoints.push_back(holecyclepoints);
    }

    // -----
    // STEP 2 : Triangulate each facet using Earclipping algorithm
    // -----
    std::vector<std::vector<Point*>> allmaincycletriangles;
    for (std::vector<Cycle*>::iterator i = maincycles.begin(); i != maincycles.end(); ++i)
    {
      Cycle* maincycle = *i;
      std::vector<Point*> maincyclepoints = (*maincycle)();
      TriangulateFacet triangulatefacet(maincyclepoints, mainholecyclepoints);
      triangulatefacet.ear_clipping_with_holes(cutside);
      std::vector<std::vector<Point*>> maincycletriangles = triangulatefacet.get_split_cells();
      for (std::vector<std::vector<Point*>>::iterator i = maincycletriangles.begin();
          i != maincycletriangles.end(); ++i)
      {
        std::vector<Point*> maincycletriangle = *i;
        if (Kernel::is_on_line(maincycletriangle[0], maincycletriangle[1], maincycletriangle[2]))
        {
          error_status_text(*cutside);
          error_gmsh("triangle_with_collinear_points.pos", *cutside);
          //          output(*cutside, "find_self_cut_triangulation");
          std::cout
              << "WARNING:::selfcut algorithm produced a triangle with all points on a line\n";
        }
        if (maincycletriangle.size() != 3)
        {
          //          output(*cutside, "find_self_cut_triangulation");
          FOUR_C_THROW("SelfCut: triangulation unsuccessful; triangle without 3 points");
        }
        if (maincycletriangle[0]->id() == maincycletriangle[1]->id() or
            maincycletriangle[0]->id() == maincycletriangle[2]->id() or
            maincycletriangle[1]->id() == maincycletriangle[2]->id())
        {
          FOUR_C_THROW("SelfCut: triangulation unsuccessful; triangle with two identical points");
        }
        cutside->get_self_cut_triangle(maincycletriangle);
      }
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * creates triangular sides out of the self cut triangles                   wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::create_self_cut_sides()
{
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    plain_node_set allcutsidenodes = cutside->self_cut_nodes();
    const std::vector<Node*>& cutsidenodes = cutside->nodes();

    // store all the nodes (self-cut nodes + side nodes)
    // this is used to check we really have a node at each point of triangle
    for (std::vector<Cut::Node*>::const_iterator i = cutsidenodes.begin(); i != cutsidenodes.end();
        ++i)
    {
      Node* cutsidenode = *i;
      allcutsidenodes.insert(cutsidenode);
    }

    const std::vector<std::vector<Point*>>& selfcuttriangles = cutside->self_cut_triangles();

    for (std::vector<std::vector<Point*>>::const_iterator i = selfcuttriangles.begin();
        i != selfcuttriangles.end(); ++i)
    {
      std::vector<Point*> selfcuttriangle = *i;
      std::vector<int> selfcutsidenodeids;
      plain_int_set selfcutsidenodeidsset;
      for (std::vector<Cut::Point*>::iterator i = selfcuttriangle.begin();
          i != selfcuttriangle.end(); ++i)
      {
        Point* selfcuttrianglepoint = *i;
        for (plain_node_set::iterator i = allcutsidenodes.begin(); i != allcutsidenodes.end(); ++i)
        {
          Node* allcutsidenode = *i;
          if (allcutsidenode->point() == selfcuttrianglepoint)
          {
            int allcutsidenodeid = allcutsidenode->id();
            selfcutsidenodeids.push_back(allcutsidenodeid);
            selfcutsidenodeidsset.insert(allcutsidenodeid);
          }
        }
      }
      if (selfcutsidenodeidsset.size() != 3)
      {
        std::cout << "selfcutsidenodeidsset.size(): " << selfcutsidenodeidsset.size() << "\n";
        const std::vector<Node*>& cutsidenodes = cutside->nodes();
        for (std::vector<Node*>::const_iterator i = cutsidenodes.begin(); i != cutsidenodes.end();
            ++i)
        {
          Node* cutsidenode = *i;
          std::cout << "Ids: " << cutsidenode->id() << "\n";
        }
        //        output(*cutside, "CreateSelfCutSides");
        FOUR_C_THROW("SelfCut: creating sides unsuccessful; didn't find 3-id-set");
      }
      if (selfcutsidenodeids.size() != 3)
      {
        std::cout << "selfcutsidenodeids.size(): " << selfcutsidenodeids.size() << "\n";
        //        output(*cutside, "CreateSelfCutSides");
        FOUR_C_THROW("SelfCut: creating sides unsuccessful; didn't find 3 ids");
      }
      if (selfcutsidenodeids[0] == selfcutsidenodeids[1] or
          selfcutsidenodeids[0] == selfcutsidenodeids[2] or
          selfcutsidenodeids[1] == selfcutsidenodeids[2])
      {
        //        output(*cutside, "CreateSelfCutSides");
        FOUR_C_THROW("SelfCut: triangulation unsuccessful; triangle with two identical points");
      }
      Side* selfcutside =
          mesh_.create_side(cutside->id(), selfcutsidenodeids, Core::FE::CellType::tri3);
      meshhandle_.add_sub_side(selfcutside);
      const std::map<plain_int_set, std::shared_ptr<Side>>& cutsideedgercp = mesh_.sides();
      std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator sideiterator =
          cutsideedgercp.find(selfcutsidenodeidsset);
      selfcut_sides_[selfcutsidenodeidsset] = sideiterator->second;
      get_self_cut_edges(*selfcutside);
    }
  }
#ifdef DEBUG_SELFCUT
  Debug("CreateSelfCutSides");
#endif
}

/*-------------------------------------------------------------------------------------*
 * erases all cutsides which are cut by another side                        wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_cut_sides()
{
  std::vector<plain_int_set> cutsideids;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    if (cutside->cutting_sides().size() != 0 and cutside->self_cut_triangles().size() != 1)
    {
      cutsideids.push_back(i->first);
      erase_side_pointer(*cutside, true);
    }
  }
  erase_side(cutsideids);
}

/*-------------------------------------------------------------------------------------*
 * erases all edges which are cut by a cutside                              wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_cut_edges()
{
  std::vector<plain_int_set> cutsideedgeids;
  for (std::map<plain_int_set, std::shared_ptr<Edge>>::iterator i = selfcut_edges_.begin();
      i != selfcut_edges_.end(); ++i)
  {
    Edge* cutsideedge = &*i->second;
    if (cutsideedge->cut_points().size() != 2 or cutsideedge->sides().size() == 0)
    {
      if (!connectedto_background(cutsideedge))
      {
        cutsideedgeids.push_back(i->first);
        erase_edge_pointer(*cutsideedge);
      }
    }
  }
  erase_edge(cutsideedgeids);
}

bool Cut::SelfCut::connectedto_background(Edge* edge)
{
  for (plain_side_set::const_iterator sit = edge->sides().begin(); sit != edge->sides().end();
      ++sit)
  {
    if ((*sit)->id() < 0) return true;
  }
  return false;
}

bool Cut::SelfCut::connectedto_background(Node* node)
{
  for (plain_side_set::const_iterator sit = node->sides().begin(); sit != node->sides().end();
      ++sit)
  {
    if ((*sit)->id() < 0) return true;
  }
  return false;
}

/*-------------------------------------------------------------------------------------*
 * locates the position of nodes, edges and sides of
 * a structure body with respect to the other bodys                         wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::determine_self_cut_position()
{
  for (std::map<plain_int_set, std::shared_ptr<Edge>>::iterator i = selfcut_edges_.begin();
      i != selfcut_edges_.end(); ++i)
  {
    Edge* cutsideedge = &*i->second;
    std::vector<Side*> selfcutsides;
    plain_side_set cutsides;
    if (cutsideedge->self_cut_position() == Point::oncutsurface)
    {
      cutsides = cutsideedge->sides();  // always 4 sides
      for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
      {
        Side* cutside = *i;
        const std::vector<Node*>& cutsidenodes = cutside->nodes();
        for (std::vector<Node*>::const_iterator i = cutsidenodes.begin(); i != cutsidenodes.end();
            ++i)
        {
          Node* cutsidenode = *i;
          if (cutsidenode->self_cut_position() == Point::undecided)
          {
            selfcutsides.push_back(cutside);
            continue;
          }
        }
      }
    }

    for (std::vector<Side*>::iterator i = selfcutsides.begin(); i != selfcutsides.end(); ++i)
    {
      Side* selfcutside = *i;
      const std::vector<Node*>& selfcutsidenodes = selfcutside->nodes();
      Node* undecidednode = nullptr;
      Node* onselfcutedgenode = nullptr;
      for (std::vector<Node*>::const_iterator i = selfcutsidenodes.begin();
          i != selfcutsidenodes.end(); ++i)
      {
        Node* selfcutsidenode = *i;
        if ((selfcutsidenode->self_cut_position() == Point::oncutsurface) &&
            (onselfcutedgenode == nullptr))
        {
          onselfcutedgenode = selfcutsidenode;
        }
        else if (selfcutsidenode->self_cut_position() == Point::undecided)
        {
          undecidednode = selfcutsidenode;
        }
      }
      Side* otherselfcutside = nullptr;
      for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
      {
        Side* anotherselfcutside = *i;
        if (anotherselfcutside->id() != selfcutside->id())
        {
          Core::LinAlg::Matrix<3, 1> normal;
          Core::LinAlg::Matrix<2, 1> center(Core::LinAlg::Initialization::zero);
          anotherselfcutside->normal(center, normal, false);
          double norm = normal.norm2();
          if (norm > 1e-8)
          {
            otherselfcutside = anotherselfcutside;
            break;
          }
          else
          {
            std::cout << "==| WARNING: Avoided anotherside with normal norm of " << norm << " ( "
                      << myrank_ << ")! |==" << std::endl;
          }
        }
      }
      if (otherselfcutside == nullptr or undecidednode == nullptr)
      {
        continue;
      }

      // this means that we are deling with self-cut with coinciding nodes
      // we already merged the nodes appropriately
      // hence in this case, all nodes associated are always "outside" --> should be included
      /*if( selfcutside->HaveCommonNode(*otherselfcutside) )
      {
        undecidednode->SelfCutPosition(Point::oncutsurface);
        continue;
      }*/

      Core::LinAlg::Matrix<3, 1> oncut_cord;
      Core::LinAlg::Matrix<3, 1> oncut_cord_loc;
      Core::LinAlg::Matrix<2, 1> oncut_cord_loc2;
      Core::LinAlg::Matrix<3, 1> otherSideNormal;
      onselfcutedgenode->point()->coordinates(oncut_cord.data());
      otherselfcutside->local_coordinates(oncut_cord, oncut_cord_loc, false);
      oncut_cord_loc2(0) = oncut_cord_loc(0);
      oncut_cord_loc2(1) = oncut_cord_loc(1);
      otherselfcutside->normal(oncut_cord_loc2, otherSideNormal);
      Core::LinAlg::Matrix<3, 1> undecidedpointcoordinates;
      Core::LinAlg::Matrix<3, 1> differencebetweenpoints;
      undecidednode->point()->coordinates(undecidedpointcoordinates.data());
      differencebetweenpoints.update(1.0, oncut_cord, -1.0, undecidedpointcoordinates);
      double norm = differencebetweenpoints.norm2();
      if (norm < SELF_CUT_POS_TOL)
      {
        std::cout << "==| WARNING: Avoided this point with distance norm of " << norm << " ( "
                  << myrank_ << ")! |==" << std::endl;
        continue;
      }
      else
        differencebetweenpoints.scale(1. / norm);
      Core::LinAlg::Matrix<1, 1> innerproduct;
      innerproduct.multiply_tn(differencebetweenpoints, otherSideNormal);
      if (innerproduct(0) > 0 and fabs(innerproduct(0)) > SELF_CUT_POS_TOL)
      {
        undecidednode->self_cut_position(Point::inside);
      }
      else if (innerproduct(0) < 0 and fabs(innerproduct(0)) > SELF_CUT_POS_TOL)
      {
        undecidednode->self_cut_position(Point::outside);
      }
      else if (innerproduct(0) < 0)
      {
        std::cout << "==| WARNING: I assign outside for an innerproduct of " << innerproduct(0)
                  << " ( " << myrank_ << ")! |==" << std::endl;
        undecidednode->self_cut_position(Point::outside);
      }
      else
      {
        std::cout << "==| WARNING: I assign inside for an innerproduct of " << innerproduct(0)
                  << " ( " << myrank_ << ")! |==" << std::endl;
        undecidednode->self_cut_position(Point::inside);
      }
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * creates triangular sides with respect to the corresponding
 * cutsides by using a pointgraph and a triangulation method                wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::propagate_self_cut_position()
{
  plain_side_set undecidedsides;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    if (cutside->self_cut_position() == Point::undecided)
    {
      const std::vector<Edge*>& cutsideedges = cutside->edges();
      Point::PointPosition cutsideedgeposition = Point::undecided;
      for (std::vector<Edge*>::const_iterator i = cutsideedges.begin(); i != cutsideedges.end();
          ++i)
      {
        Edge* cutsideedge = *i;
        Point::PointPosition cutsideedgepos = cutsideedge->self_cut_position();
        switch (cutsideedgepos)
        {
          case Point::undecided:
          case Point::oncutsurface:
            break;
          case Point::inside:
          case Point::outside:
            if ((cutsideedgeposition != Point::undecided) and
                (cutsideedgepos != cutsideedgeposition))
            {
              FOUR_C_THROW("SelfCut: mixed PointPosition of cutsideedges");
            }
            cutsideedgeposition = cutsideedgepos;
            break;
        }
      }
      if (cutsideedgeposition != Point::undecided)
      {
        cutside->get_self_cut_position(cutsideedgeposition);
      }
      else
      {
        undecidedsides.insert(cutside);
      }
    }
  }
  if (selfcut_sides_.size() == undecidedsides.size() and selfcut_sides_.size() > 0)
  {
    FOUR_C_THROW("SelfCut: all cutsides undecided");
  }
  while (undecidedsides.size() > 0)
  {
    unsigned undecidedsidesize = undecidedsides.size();
    for (plain_side_set::iterator ui = undecidedsides.begin(); ui != undecidedsides.end();)
    {
      Side* undecidedside = *ui;
      if (undecidedside->self_cut_position() == Point::undecided)
      {
        bool done = false;
        const std::vector<Edge*>& undecidedcutsideedges = undecidedside->edges();
        for (std::vector<Edge*>::const_iterator i = undecidedcutsideedges.begin();
            i != undecidedcutsideedges.end(); ++i)
        {
          Edge* undecidedcutsideedge = *i;
          const plain_side_set& undecidedcutsideedgesides = undecidedcutsideedge->sides();
          Side* siblingside = nullptr;
          for (plain_side_set::const_iterator i = undecidedcutsideedgesides.begin();
              i != undecidedcutsideedgesides.end(); ++i)
          {
            Side* undecidedcutsideedgeside = *i;
            if (undecidedcutsideedgeside->id() == undecidedside->id() and
                undecidedcutsideedgeside->self_cut_position() != Point::undecided and
                undecidedcutsideedgeside != undecidedside)
            {
              siblingside = undecidedcutsideedgeside;
              break;
            }
          }
          if (siblingside != nullptr)
          {
            Point::PointPosition siblingsideposition = siblingside->self_cut_position();
            switch (siblingsideposition)
            {
              case Point::undecided:
                if (undecidedsides.count(siblingside))
                {
                  FOUR_C_THROW("SelfCut: incomplete set of undecided cutsides");
                }
                break;
              case Point::oncutsurface:
                FOUR_C_THROW("SelfCut: illegal side position");
                break;
              case Point::inside:
              case Point::outside:
                if (undecidedcutsideedge->self_cut_position() == Point::oncutsurface)
                {
                  undecidedside->get_self_cut_position(
                      siblingsideposition == Point::inside ? Point::outside : Point::inside);
                }
                else
                {
                  undecidedside->get_self_cut_position(siblingsideposition);
                }
                done = true;
                break;
            }
            if (done)
            {
              break;
            }
          }
        }
        if (done)
        {
          set_erase(undecidedsides, ui);
        }
        else
        {
          ui++;
        }
      }
      else
      {
        set_erase(undecidedsides, ui);
      }
    }
    if (undecidedsidesize == undecidedsides.size())
      FOUR_C_THROW("SelfCut: no progress in cutside position");
  }
}

/*-------------------------------------------------------------------------------------*
 * erases sides which lies inside a structure body by locating
 * their position                                                           wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_inside_sides()
{
  // Get some information of the sides stored in the mesh_ in case that a cuttest fails
  int initial_cutsides = mesh_.sides().size();
  int final_cutsides = 0;

  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = mesh_.sides();
  std::vector<plain_int_set> cutsideids;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    if (cutside->self_cut_position() == Point::inside)
    {
      cutsideids.push_back(i->first);
      erase_side_pointer(*cutside, false);
      final_cutsides++;
    }
  }
  erase_inside_side(cutsideids);
  if (mesh_.sides().size() == 0)
    FOUR_C_THROW(
        "All self-cut positions are undecided\n. The initial number of cutsides is {}, the number "
        "of erased cutsides is {}",
        initial_cutsides, final_cutsides);
}

/*-------------------------------------------------------------------------------------*
 * erases edges which lies inside a structure body by locating
 * their position                                                           wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_inside_edges()
{
  const std::map<plain_int_set, std::shared_ptr<Edge>>& cutsideedges = mesh_.edges();
  std::vector<plain_int_set> cutsideedgeids;
  for (std::map<plain_int_set, std::shared_ptr<Edge>>::const_iterator i = cutsideedges.begin();
      i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = &*i->second;
    if (cutsideedge->self_cut_position() == Point::inside)
    {
      if (!connectedto_background(cutsideedge))
      {
        cutsideedgeids.push_back(i->first);
        erase_edge_pointer(*cutsideedge);
      }
    }
  }
  erase_edge(cutsideedgeids);
}

/*-------------------------------------------------------------------------------------*
 * erases nodes which lies inside a structure body by locating
 * there position                                                           wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_inside_nodes()
{
  const std::map<int, std::shared_ptr<Node>>& cutsidenodes = mesh_.nodes();
  std::vector<int> cutsidenodeids;
  for (std::map<int, std::shared_ptr<Node>>::const_iterator i = cutsidenodes.begin();
      i != cutsidenodes.end(); ++i)
  {
    Node* cutsidenode = &*i->second;
    if (cutsidenode->self_cut_position() == Point::inside)
    {
      if (!connectedto_background(cutsidenode))
      {
        cutsidenodeids.push_back(i->first);
      }
    }
  }
  for (std::vector<int>::iterator i = cutsidenodeids.begin(); i != cutsidenodeids.end(); ++i)
  {
    int cutsidenodeid = *i;
    selfcut_nodes_.erase(cutsidenodeid);
    mesh_.move_nodeto_storage(cutsidenodeid);
  }
}

/*-------------------------------------------------------------------------------------*
 * construct the connectivity of the nodes to find potential islands in the cut mesh
 *                                                                          wirtz 07/16
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::construct_connectivity()
{
  const std::map<int, std::shared_ptr<Node>>& cutsidenodes = mesh_.nodes();
  plain_node_set remainingnodes;
  int count = 0;

  for (std::map<int, std::shared_ptr<Node>>::const_iterator i = cutsidenodes.begin();
      i != cutsidenodes.end(); ++i)
  {
    Node* cutsidenode = &*i->second;
    remainingnodes.insert(cutsidenode);
  }

  while (remainingnodes.size() > 0)
  {
    Node* node = *remainingnodes.begin();
    selfcut_connectivity_[count].insert(node);
    remainingnodes.erase(node);
    next_node(node, remainingnodes, count);
    count++;
  }
}

/*-------------------------------------------------------------------------------------*
 * find the next node for the construction of the connectivity              wirtz 07/16
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::next_node(Node* node, plain_node_set& remainingnodes, int count)
{
  const plain_edge_set& nodeedges = node->edges();
  for (plain_edge_set::const_iterator i = nodeedges.begin(); i != nodeedges.end(); ++i)
  {
    Edge* nodeedge = *i;
    const std::vector<Node*>& nodeedgenodes = nodeedge->nodes();
    for (std::vector<Node*>::const_iterator i = nodeedgenodes.begin(); i != nodeedgenodes.end();
        ++i)
    {
      Node* nodeedgenode = *i;
      plain_node_set::iterator remainingnodesiter = remainingnodes.find(nodeedgenode);
      if (remainingnodesiter != remainingnodes.end())
      {
        selfcut_connectivity_[count].insert(nodeedgenode);
        remainingnodes.erase(nodeedgenode);
        next_node(nodeedgenode, remainingnodes, count);
      }
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * identify the islands                                                     wirtz 07/16
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::find_islands()
{
  plain_side_set selfcutsides;

  const std::map<plain_int_set, std::shared_ptr<Side>>& fullcut_sides = mesh_.sides();

  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = fullcut_sides.begin();
      i != fullcut_sides.end(); ++i)
  {
    Side* cutside = &*i->second;
    if (cutside->self_cut_position() == Point::outside) selfcutsides.insert(cutside);
  }

  std::vector<plain_edge_set> nonconnected_selfcutedges;
  while (selfcutsides.size() > 0)
  {
    plain_side_set islandsides;
    bool IsIsland = false;
    Side* cutside = *selfcutsides.begin();
    islandsides.insert(cutside);
    selfcutsides.erase(cutside);

    std::shared_ptr<BoundingBox> tmp_bb(BoundingBox::create(*cutside));
    next_sides(cutside, tmp_bb, selfcutsides, islandsides, IsIsland);

    if (tmp_bb->diagonal() <=
            mesh_.get_options().self_cut_island_geom_multiplicator() * meshsizeparam_ &&
        IsIsland)
    {
      for (plain_side_set::iterator i = islandsides.begin(); i != islandsides.end(); ++i)
      {
        Side* cutside = *i;
        cutside->get_self_cut_position(Point::inside);  // overwrite?
      }
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * finds the next sides in a recursive way
 *                                                                          wirtz 02/15
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::next_sides(Side* cutside, std::shared_ptr<Cut::BoundingBox>& tmpbb,
    plain_side_set& selfcutsides, plain_side_set& islandsides, bool& IsIsland)
{
  const std::vector<Edge*>& cutsideedges = cutside->edges();
  for (std::vector<Edge*>::const_iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    {
      const plain_side_set& cutsidenodesides = cutsideedge->sides();
      for (plain_side_set::const_iterator i = cutsidenodesides.begin(); i != cutsidenodesides.end();
          ++i)
      {
        Side* cutsidenodeside = *i;
        if (cutsidenodeside->self_cut_position() == Point::outside)
        {
          plain_side_set::iterator islandsideiter = islandsides.find(cutsidenodeside);
          if (islandsideiter == islandsides.end())  // side has not been treated jet
          {
            if (tmpbb->diagonal() <=
                mesh_.get_options().self_cut_island_geom_multiplicator() * meshsizeparam_)
            {
              IsIsland = true;
              tmpbb->add_points(cutsidenodeside->nodes());
            }
            else
              IsIsland = false;
            islandsides.insert(cutsidenodeside);
            selfcutsides.erase(cutsidenodeside);
            next_sides(cutsidenodeside, tmpbb, selfcutsides, islandsides, IsIsland);
          }
        }
      }
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * Status of the cut sides for text viewer                               wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::cut_side_status_text()
{
  int selfcutsidesize = selfcut_sides_.size();
  int j = 1;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    plot_across();
    std::cout << "\nCutside " << j << "/" << selfcutsidesize << ":\n";
    side_plot_head();
    side_plot(*cutside);

    std::cout << "\nCutting Sides:\n";
    side_plot_head();
    const plain_side_set& cuttingsides = cutside->cutting_sides();
    for (plain_side_set::const_iterator i = cuttingsides.begin(); i != cuttingsides.end(); ++i)
    {
      Side* cuttingside = *i;
      side_plot(*cuttingside);
    }

    const PointSet& selfcutpoints = cutside->self_cut_points();
    std::cout << "\nSelfcutpoints:\n";
    point_plot_head();
    for (PointSet::const_iterator i = selfcutpoints.begin(); i != selfcutpoints.end(); ++i)
    {
      Point* selfcutpoint = *i;
      point_plot(*selfcutpoint);
      std::cout << "\n";
    }

    const plain_node_set& selfcutnodes = cutside->self_cut_nodes();
    std::cout << "\nSelfcutnodes:\n";
    node_plot_head();
    for (plain_node_set::const_iterator i = selfcutnodes.begin(); i != selfcutnodes.end(); ++i)
    {
      Node* selfcutnode = *i;
      node_plot(*selfcutnode);
    }

    const plain_edge_set& selfcutedges = cutside->self_cut_edges();
    std::cout << "\nSelfcutedges:\n";
    edge_plot_head();
    for (plain_edge_set::const_iterator i = selfcutedges.begin(); i != selfcutedges.end(); ++i)
    {
      Edge* selfcutedge = *i;
      edge_plot(*selfcutedge);
    }

    const std::vector<std::vector<Cut::Point*>>& selfcuttriangles = cutside->self_cut_triangles();
    std::cout << "\nSelfcuttriangles:\n";
    for (std::vector<std::vector<Cut::Point*>>::const_iterator i = selfcuttriangles.begin();
        i != selfcuttriangles.end(); ++i)
    {
      std::vector<Cut::Point*> selfcuttriangle = *i;
      point_plot_head();
      for (std::vector<Cut::Point*>::iterator i = selfcuttriangle.begin();
          i != selfcuttriangle.end(); ++i)
      {
        Point* selfcuttrianglepoint = *i;
        point_plot(*selfcuttrianglepoint);
        std::cout << "\n";
      }
    }
    plot_across();
    j++;
  }
}

/*-------------------------------------------------------------------------------------*
 * Status of the cutmesh for text viewer                                    wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::cut_mesh_status_text()
{
  plot_across();
  std::cout << "\n" << selfcut_nodes_.size() << " Nodes: \n";
  node_plot_head();
  for (std::map<int, std::shared_ptr<Node>>::iterator i = selfcut_nodes_.begin();
      i != selfcut_nodes_.end(); ++i)
  {
    Node* node = &*i->second;
    node_plot(*node);
  }
  plot_across();
  std::cout << "\n" << selfcut_edges_.size() << " Edges: \n";
  edge_plot_head();
  for (std::map<plain_int_set, std::shared_ptr<Edge>>::iterator i = selfcut_edges_.begin();
      i != selfcut_edges_.end(); ++i)
  {
    Edge* edge = &*i->second;
    edge_plot(*edge);
  }
  plot_across();
  std::cout << "\n" << selfcut_sides_.size() << " Sides: \n";
  side_plot_head();
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* side = &*i->second;
    side_plot(*side);
  }
  plot_across();
}

/*-------------------------------------------------------------------------------------*
 * Status of one problematic side for text viewer                           wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::error_status_text(Side& cutside)
{
  int selfcutsidesize = 1;
  int j = 1;
  plot_across();
  std::cout << "\nCutside " << j << "/" << selfcutsidesize << ":\n";
  side_plot_head();
  side_plot(cutside);
  side_node_plot(cutside);

  std::cout << "\nCutting Sides:\n";
  side_plot_head();
  const plain_side_set& cuttingsides = cutside.cutting_sides();
  for (plain_side_set::const_iterator i = cuttingsides.begin(); i != cuttingsides.end(); ++i)
  {
    Side* cuttingside = *i;
    side_plot(*cuttingside);
    side_node_plot(*cuttingside);
  }

  const PointSet& selfcutpoints = cutside.self_cut_points();
  std::cout << "\nSelfcutpoints:\n";
  point_plot_head();
  for (PointSet::const_iterator i = selfcutpoints.begin(); i != selfcutpoints.end(); ++i)
  {
    Point* selfcutpoint = *i;
    point_plot(*selfcutpoint);
    std::cout << "\n";
  }

  const plain_node_set& selfcutnodes = cutside.self_cut_nodes();
  std::cout << "\nSelfcutnodes:\n";
  node_plot_head();
  for (plain_node_set::const_iterator i = selfcutnodes.begin(); i != selfcutnodes.end(); ++i)
  {
    Node* selfcutnode = *i;
    node_plot(*selfcutnode);
  }

  const plain_edge_set& selfcutedges = cutside.self_cut_edges();
  std::cout << "\nSelfcutedges:\n";
  edge_plot_head();
  for (plain_edge_set::const_iterator i = selfcutedges.begin(); i != selfcutedges.end(); ++i)
  {
    Edge* selfcutedge = *i;
    edge_plot(*selfcutedge);
  }

  const std::vector<std::vector<Cut::Point*>>& selfcuttriangles = cutside.self_cut_triangles();
  std::cout << "\nSelfcuttriangles:\n";
  for (std::vector<std::vector<Cut::Point*>>::const_iterator i = selfcuttriangles.begin();
      i != selfcuttriangles.end(); ++i)
  {
    std::vector<Cut::Point*> selfcuttriangle = *i;
    point_plot_head();
    for (std::vector<Cut::Point*>::iterator i = selfcuttriangle.begin(); i != selfcuttriangle.end();
        ++i)
    {
      Point* selfcuttrianglepoint = *i;
      point_plot(*selfcuttrianglepoint);
      std::cout << "\n";
    }
  }
  plot_across();
  j++;
}

/*-------------------------------------------------------------------------------------*
 * Status of the cut sides for gmsh                                      wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::cut_side_status_gmsh(const std::string& name)
{
  std::ofstream file(name.c_str());
  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = mesh_.sides();
  int cutsidessize = cutsides.size();
  file << "View \"" << cutsidessize << " Sides\" {\n";
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    file.precision(16);
    int cutsidetype = i->first.size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      FOUR_C_THROW("SelfCut: irregular side");
    }

    file << "){";
    Point::PointPosition cutsideselfcutposition = cutside->self_cut_position();
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  const std::map<plain_int_set, std::shared_ptr<Edge>>& cutsideedges = mesh_.edges();
  int cutedgessize = cutsideedges.size();
  file << "View \"" << cutedgessize << " Edges\" {\n";
  for (std::map<plain_int_set, std::shared_ptr<Edge>>::const_iterator i = cutsideedges.begin();
      i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = &*i->second;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    cutsideedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsideedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  const std::map<int, std::shared_ptr<Node>>& cutsidenodes = mesh_.nodes();
  int cutnodessize = cutsidenodes.size();
  file << "View \"" << cutnodessize << " Nodes\" {\n";
  for (std::map<int, std::shared_ptr<Node>>::const_iterator i = cutsidenodes.begin();
      i != cutsidenodes.end(); ++i)
  {
    Node* cutsidenode = &*i->second;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    cutsidenode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsidenode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";
}

/*-------------------------------------------------------------------------------------*
 * Status of the cutmesh in gmsh                                            wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::wall_gmsh(const std::string& name)
{
  std::stringstream str;
  str << name << "_" << myrank_ << ".pos";
  std::ofstream file(str.str().c_str());

  /*  BoundingBox elementbox;
   double x[3];
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   plain_side_set cutsides;
   for ( std::map<plain_int_set, std::shared_ptr<Side> >::iterator i=selfcut_sides_.begin();
   i!=selfcut_sides_.end();
   ++i )
   {
   Side * cutside = &*i->second;
   BoundingBox sidebox;
   sidebox.Assign( *cutside );
   if( sidebox.Within(1.0, elementbox) )
   {
   cutsides.insert( cutside );
   }
   }*/

  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = mesh_.sides();

  int cutsidessize = cutsides.size();
  file << "View \"" << cutsidessize << " Sides\" {\n";
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    file.precision(16);
    int cutsidetype = cutside->nodes().size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      FOUR_C_THROW("SelfCut: irregular side");
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutside->self_cut_position();
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_edge_set cutsideedges;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const std::vector<Edge*>& cutedges = cutside->edges();
    for (std::vector<Edge*>::const_iterator i = cutedges.begin(); i != cutedges.end(); ++i)
    {
      Edge* cutsideedge = *i;
      cutsideedges.insert(cutsideedge);
    }
  }
  int cutsideedgessize = cutsideedges.size();
  file << "View \"" << cutsideedgessize << " Edges\" {\n";
  for (plain_edge_set::iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    cutsideedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsideedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_node_set cutsidenodes;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const std::vector<Node*>& cutnodes = cutside->nodes();
    for (std::vector<Node*>::const_iterator i = cutnodes.begin(); i != cutnodes.end(); ++i)
    {
      Node* cutsidenode = *i;
      cutsidenodes.insert(cutsidenode);
    }
  }
  int cutnodessize = cutsidenodes.size();
  file << "View \"" << cutnodessize << " Nodes\" {\n";
  for (plain_node_set::iterator i = cutsidenodes.begin(); i != cutsidenodes.end(); ++i)
  {
    Node* cutsidenode = *i;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    cutsidenode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsidenode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";

  plain_side_set cuttingsides;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const plain_side_set& selfcutsides = cutside->cutting_sides();
    for (plain_side_set::const_iterator i = selfcutsides.begin(); i != selfcutsides.end(); ++i)
    {
      Side* cuttingside = *i;
      cuttingsides.insert(cuttingside);
    }
  }
  int cuttingsidessize = cuttingsides.size();
  file << "View \"" << cuttingsidessize << " Cuttingsides\" {\n";
  for (plain_side_set::iterator i = cuttingsides.begin(); i != cuttingsides.end(); ++i)
  {
    Side* cuttingside = *i;
    file.precision(16);
    int cutsidetype = cuttingside->nodes().size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cuttingside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cuttingside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      FOUR_C_THROW("SelfCut: irregular side");
    }
    file << "){";
    double cutsideselfcutposition = -0.5;
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_node_set selfcutnodes;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const plain_node_set& selfnodes = cutside->self_cut_nodes();
    for (plain_node_set::const_iterator i = selfnodes.begin(); i != selfnodes.end(); ++i)
    {
      Node* selfcutnode = *i;
      selfcutnodes.insert(selfcutnode);
    }
  }
  int selfcutnodessize = selfcutnodes.size();
  file << "View \"" << selfcutnodessize << " Selfcutnodes\" {\n";
  for (plain_node_set::iterator i = selfcutnodes.begin(); i != selfcutnodes.end(); ++i)
  {
    Node* selfcutnode = *i;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    selfcutnode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = selfcutnode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";

  plain_edge_set selfcutedges;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const plain_edge_set& selfedges = cutside->self_cut_edges();
    for (plain_edge_set::const_iterator i = selfedges.begin(); i != selfedges.end(); ++i)
    {
      Edge* selfcutedge = *i;
      selfcutedges.insert(selfcutedge);
    }
  }
  int selfcutedgessize = selfcutedges.size();
  file << "View \"" << selfcutedgessize << " Selfcutedges\" {\n";
  for (plain_edge_set::iterator i = selfcutedges.begin(); i != selfcutedges.end(); ++i)
  {
    Edge* selfcutedge = *i;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    selfcutedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = selfcutedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  std::vector<std::vector<Point*>> selfcuttriangles;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const std::vector<std::vector<Point*>>& selftriangles = cutside->self_cut_triangles();
    for (std::vector<std::vector<Point*>>::const_iterator i = selftriangles.begin();
        i != selftriangles.end(); ++i)
    {
      std::vector<Point*> selfcuttriangle = *i;
      selfcuttriangles.push_back(selfcuttriangle);
    }
  }
  int selfcuttrianglessize = selfcuttriangles.size();
  file << "View \"" << selfcuttrianglessize << " Selfcuttriangles\" {\n";
  for (std::vector<std::vector<Point*>>::iterator i = selfcuttriangles.begin();
      i != selfcuttriangles.end(); ++i)
  {
    std::vector<Point*> selfcuttriangle = *i;
    int selfcuttriangleposition = 0;
    int elementfacetpointssize = selfcuttriangle.size();
    for (std::vector<Point*>::iterator i = selfcuttriangle.begin(); i != selfcuttriangle.end(); ++i)
    {
      Point* elementfacetpoint1 = *i;
      Point* elementfacetpoint2 = nullptr;
      if (i + 1 != selfcuttriangle.end())
      {
        elementfacetpoint2 = *(i + 1);
      }
      else
      {
        elementfacetpoint2 = *(i + 1 - elementfacetpointssize);
      }
      file << "SL (";
      Core::LinAlg::Matrix<3, 1> elementfacetpoint1coordinates;
      elementfacetpoint1->coordinates(elementfacetpoint1coordinates.data());
      Core::LinAlg::Matrix<3, 1> elementfacetpoint2coordinates;
      elementfacetpoint2->coordinates(elementfacetpoint2coordinates.data());
      file << elementfacetpoint1coordinates(0, 0) << "," << elementfacetpoint1coordinates(1, 0)
           << "," << elementfacetpoint1coordinates(2, 0) << ","
           << elementfacetpoint2coordinates(0, 0) << "," << elementfacetpoint2coordinates(1, 0)
           << "," << elementfacetpoint2coordinates(2, 0);
      file << "){";
      for (int i = 0; i < 2; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << selfcuttriangleposition;
      }
      file << "};\n";
    }
  }
  file << "};\n";
}


/*-------------------------------------------------------------------------------------*
 * Status of the selfcut objects in gmsh                                    wirtz 07/16
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::sc_objects_gmsh(const std::string& name)
{
  std::ofstream file(name.c_str());

  /*  BoundingBox elementbox;
   double x[3];
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   plain_side_set cutsides;
   for ( std::map<plain_int_set, std::shared_ptr<Side> >::iterator i=selfcut_sides_.begin();
   i!=selfcut_sides_.end();
   ++i )
   {
   Side * cutside = &*i->second;
   BoundingBox sidebox;
   sidebox.Assign( *cutside );
   if( sidebox.Within(1.0, elementbox) )
   {
   cutsides.insert( cutside );
   }
   }*/

  const std::map<plain_int_set, std::shared_ptr<Side>>& cutsides = mesh_.sides();

  int cutsidessize = cutsides.size();
  file << "View \"" << cutsidessize << " Sides\" {\n";
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    file.precision(16);
    int cutsidetype = cutside->nodes().size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      // FOUR_C_THROW("SelfCut: irregular side");
      std::cout << "==| WARNING: SelfCut: irregular side 4 |==" << std::endl;
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutside->self_cut_position();
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_edge_set cutsideedges;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const std::vector<Edge*>& cutedges = cutside->edges();
    for (std::vector<Edge*>::const_iterator i = cutedges.begin(); i != cutedges.end(); ++i)
    {
      Edge* cutsideedge = *i;
      cutsideedges.insert(cutsideedge);
    }
  }
  int cutsideedgessize = cutsideedges.size();
  file << "View \"" << cutsideedgessize << " Edges\" {\n";
  for (plain_edge_set::iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    cutsideedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsideedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_node_set cutsidenodes;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = cutsides.begin();
      i != cutsides.end(); ++i)
  {
    Side* cutside = &*i->second;
    const std::vector<Node*>& cutnodes = cutside->nodes();
    for (std::vector<Node*>::const_iterator i = cutnodes.begin(); i != cutnodes.end(); ++i)
    {
      Node* cutsidenode = *i;
      cutsidenodes.insert(cutsidenode);
    }
  }
  int cutnodessize = cutsidenodes.size();
  file << "View \"" << cutnodessize << " Nodes\" {\n";
  for (plain_node_set::iterator i = cutsidenodes.begin(); i != cutsidenodes.end(); ++i)
  {
    Node* cutsidenode = *i;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    cutsidenode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsidenode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";



  int selfcutsidessize = selfcut_sides_.size();
  file << "View \"" << selfcutsidessize << " Sides\" {\n";
  for (std::map<plain_int_set, std::shared_ptr<Side>>::const_iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* selfcutside = &*i->second;
    file.precision(16);
    int selfcutsidetype = selfcutside->nodes().size();
    if (selfcutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> selfcutsidecoordinates;
      selfcutside->coordinates(selfcutsidecoordinates.data());
      for (int i = 0; i < selfcutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << selfcutsidecoordinates(0, i) << "," << selfcutsidecoordinates(1, i) << ","
             << selfcutsidecoordinates(2, i);
      }
    }
    else if (selfcutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> selfcutsidecoordinates;
      selfcutside->coordinates(selfcutsidecoordinates.data());
      for (int i = 0; i < selfcutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << selfcutsidecoordinates(0, i) << "," << selfcutsidecoordinates(1, i) << ","
             << selfcutsidecoordinates(2, i);
      }
    }
    else
    {
      // FOUR_C_THROW("SelfCut: irregular side");
      std::cout << "==| WARNING: SelfCut: irregular side 5 |==" << std::endl;
    }
    file << "){";
    Point::PointPosition selfcutsideselfcutposition = selfcutside->self_cut_position();
    for (int i = 0; i < selfcutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << selfcutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  int selfcutedgessize = selfcut_edges_.size();
  file << "View \"" << selfcutedgessize << " Edges\" {\n";
  for (std::map<plain_int_set, std::shared_ptr<Edge>>::iterator i = selfcut_edges_.begin();
      i != selfcut_edges_.end(); ++i)
  {
    Edge* selfcutedge = &*i->second;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> selfcutedgecoordinates;
    selfcutedge->coordinates(selfcutedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << selfcutedgecoordinates(0, i) << "," << selfcutedgecoordinates(1, i) << ","
           << selfcutedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition selfcutedgeposition = selfcutedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << selfcutedgeposition;
    }
    file << "};\n";
  }
  file << "};\n";

  int selfcutnodessize = selfcut_nodes_.size();
  file << "View \"" << selfcutnodessize << " Nodes\" {\n";
  for (std::map<int, std::shared_ptr<Node>>::iterator i = selfcut_nodes_.begin();
      i != selfcut_nodes_.end(); ++i)
  {
    Node* selfcutnode = &*i->second;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> selfcutnodecoordinates;
    selfcutnode->coordinates(selfcutnodecoordinates.data());
    file << selfcutnodecoordinates(0, 0) << "," << selfcutnodecoordinates(1, 0) << ","
         << selfcutnodecoordinates(2, 0);
    file << "){";
    Point::PointPosition selfcutnodeposition = selfcutnode->self_cut_position();
    file << selfcutnodeposition;
    file << "};\n";
  }
  file << "};\n";
}


/*-------------------------------------------------------------------------------------*
 * Status of the cutmesh in gmsh for my CMGM (old)                          wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::s_cmgm_gmsh(const std::string& name)
{
  std::ofstream file(name.c_str());

  /*  BoundingBox elementbox;
   double x[3];
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   plain_side_set cutsides;
   for ( std::map<plain_int_set, std::shared_ptr<Side> >::iterator i=selfcut_sides_.begin();
   i!=selfcut_sides_.end();
   ++i )
   {
   Side * cutside = &*i->second;
   BoundingBox sidebox;
   sidebox.Assign( *cutside );
   if( sidebox.Within(1.0, elementbox) )
   {
   cutsides.insert( cutside );
   }
   }*/
  plain_side_set cutsides;
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    cutsides.insert(cutside);
  }
  int cutsidessize = cutsides.size();
  file << "View \"" << cutsidessize << " Sides\" {\n";
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    file.precision(16);
    int cutsidetype = cutside->nodes().size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      // FOUR_C_THROW("SelfCut: irregular side");
      std::cout << "==| WARNING: SelfCut: irregular side 6 |==" << std::endl;
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutside->self_cut_position();
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_edge_set cutsideedges;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const std::vector<Edge*>& cutedges = cutside->edges();
    for (std::vector<Edge*>::const_iterator i = cutedges.begin(); i != cutedges.end(); ++i)
    {
      Edge* cutsideedge = *i;
      cutsideedges.insert(cutsideedge);
    }
  }
  int cutsideedgessize = cutsideedges.size();
  file << "View \"" << cutsideedgessize << " Edges\" {\n";
  for (plain_edge_set::iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    cutsideedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsideedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_node_set cutsidenodes;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const std::vector<Node*>& cutnodes = cutside->nodes();
    for (std::vector<Node*>::const_iterator i = cutnodes.begin(); i != cutnodes.end(); ++i)
    {
      Node* cutsidenode = *i;
      cutsidenodes.insert(cutsidenode);
    }
  }
  int cutnodessize = cutsidenodes.size();
  file << "View \"" << cutnodessize << " Nodes\" {\n";
  for (plain_node_set::iterator i = cutsidenodes.begin(); i != cutsidenodes.end(); ++i)
  {
    Node* cutsidenode = *i;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    cutsidenode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsidenode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";

  plain_side_set cuttingsides;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const plain_side_set& selfcutsides = cutside->cutting_sides();
    for (plain_side_set::const_iterator i = selfcutsides.begin(); i != selfcutsides.end(); ++i)
    {
      Side* cuttingside = *i;
      cuttingsides.insert(cuttingside);
    }
  }
  int cuttingsidessize = cuttingsides.size();
  file << "View \"" << cuttingsidessize << " Cuttingsides\" {\n";
  for (plain_side_set::iterator i = cuttingsides.begin(); i != cuttingsides.end(); ++i)
  {
    Side* cuttingside = *i;
    file.precision(16);
    int cutsidetype = cuttingside->nodes().size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cuttingside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cuttingside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      // FOUR_C_THROW("SelfCut: irregular side");
      std::cout << "==| WARNING: SelfCut: irregular side 7 |==" << std::endl;
    }
    file << "){";
    double cutsideselfcutposition = -0.5;
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_node_set selfcutnodes;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const plain_node_set& selfnodes = cutside->self_cut_nodes();
    for (plain_node_set::const_iterator i = selfnodes.begin(); i != selfnodes.end(); ++i)
    {
      Node* selfcutnode = *i;
      selfcutnodes.insert(selfcutnode);
    }
  }
  int selfcutnodessize = selfcutnodes.size();
  file << "View \"" << selfcutnodessize << " Selfcutnodes\" {\n";
  for (plain_node_set::iterator i = selfcutnodes.begin(); i != selfcutnodes.end(); ++i)
  {
    Node* selfcutnode = *i;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    selfcutnode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = selfcutnode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";

  plain_edge_set selfcutedges;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const plain_edge_set& selfedges = cutside->self_cut_edges();
    for (plain_edge_set::const_iterator i = selfedges.begin(); i != selfedges.end(); ++i)
    {
      Edge* selfcutedge = *i;
      selfcutedges.insert(selfcutedge);
    }
  }
  int selfcutedgessize = selfcutedges.size();
  file << "View \"" << selfcutedgessize << " Selfcutedges\" {\n";
  for (plain_edge_set::iterator i = selfcutedges.begin(); i != selfcutedges.end(); ++i)
  {
    Edge* selfcutedge = *i;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    selfcutedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = selfcutedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  std::vector<std::vector<Point*>> selfcuttriangles;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const std::vector<std::vector<Point*>>& selftriangles = cutside->self_cut_triangles();
    for (std::vector<std::vector<Point*>>::const_iterator i = selftriangles.begin();
        i != selftriangles.end(); ++i)
    {
      std::vector<Point*> selfcuttriangle = *i;
      selfcuttriangles.push_back(selfcuttriangle);
    }
  }
  int selfcuttrianglessize = selfcuttriangles.size();
  file << "View \"" << selfcuttrianglessize << " Selfcuttriangles\" {\n";
  for (std::vector<std::vector<Point*>>::iterator i = selfcuttriangles.begin();
      i != selfcuttriangles.end(); ++i)
  {
    std::vector<Point*> selfcuttriangle = *i;
    int selfcuttriangleposition = 0;
    int elementfacetpointssize = selfcuttriangle.size();
    for (std::vector<Point*>::iterator i = selfcuttriangle.begin(); i != selfcuttriangle.end(); ++i)
    {
      Point* elementfacetpoint1 = *i;
      Point* elementfacetpoint2 = nullptr;
      if (i + 1 != selfcuttriangle.end())
      {
        elementfacetpoint2 = *(i + 1);
      }
      else
      {
        elementfacetpoint2 = *(i + 1 - elementfacetpointssize);
      }
      file << "SL (";
      Core::LinAlg::Matrix<3, 1> elementfacetpoint1coordinates;
      elementfacetpoint1->coordinates(elementfacetpoint1coordinates.data());
      Core::LinAlg::Matrix<3, 1> elementfacetpoint2coordinates;
      elementfacetpoint2->coordinates(elementfacetpoint2coordinates.data());
      file << elementfacetpoint1coordinates(0, 0) << "," << elementfacetpoint1coordinates(1, 0)
           << "," << elementfacetpoint1coordinates(2, 0) << ","
           << elementfacetpoint2coordinates(0, 0) << "," << elementfacetpoint2coordinates(1, 0)
           << "," << elementfacetpoint2coordinates(2, 0);
      file << "){";
      for (int i = 0; i < 2; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << selfcuttriangleposition;
      }
      file << "};\n";
    }
  }
  file << "};\n";
}

/*-------------------------------------------------------------------------------------*
 * Status of all sides in separates files for gmsh                          wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::all_single_gmsh(const std::string& location)
{
  for (std::map<plain_int_set, std::shared_ptr<Side>>::iterator i = selfcut_sides_.begin();
      i != selfcut_sides_.end(); ++i)
  {
    Side* cutside = &*i->second;
    std::stringstream filenamegmsh;

    plain_int_set::const_iterator it = (i->first).begin();
    const int first = *it;
    it++;
    const int second = *it;
    it++;
    const int third = *it;

    filenamegmsh << location << "_SelfCutSide_" << first << "_" << second << "_" << third << ".pos";
    error_gmsh(filenamegmsh.str(), *cutside);
  }
}

/*-------------------------------------------------------------------------------------*
 * Status of one problematic side for gmsh                                  wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::error_gmsh(const std::string& name, Side& cutside)
{
  std::ofstream file(name.c_str());

  /*  BoundingBox elementbox;
   double x[3];
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.52;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = 0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.035;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = -0.005;
   elementbox.add_point( x );
   x[0] = 0.53;
   x[1] = 0.045;
   x[2] = 0.005;
   elementbox.add_point( x );
   plain_side_set cutsides;
   for ( std::map<plain_int_set, std::shared_ptr<Side> >::iterator i=selfcut_sides_.begin();
   i!=selfcut_sides_.end();
   ++i )
   {
   Side * cutside = &*i->second;
   BoundingBox sidebox;
   sidebox.Assign( *cutside );
   if( sidebox.Within(1.0, elementbox) )
   {
   cutsides.insert( cutside );
   }
   }*/
  plain_side_set cutsides;
  /*  for ( std::map<plain_int_set, std::shared_ptr<Side> >::iterator i=selfcut_sides_.begin();
   i!=selfcut_sides_.end();
   ++i )
   {
   Side * cutside = &*i->second;
   cutsides.insert( cutside );
   }*/
  cutsides.insert(&cutside);

  int cutsidessize = cutsides.size();
  file << "View \"" << cutsidessize << " Sides\" {\n";
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    file.precision(16);
    int cutsidetype = cutside->nodes().size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cutside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      // FOUR_C_THROW("SelfCut: irregular side");
      std::cout << "==| WARNING: SelfCut: irregular side 8 |==" << std::endl;
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutside->self_cut_position();
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_edge_set cutsideedges;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const std::vector<Edge*>& cutedges = cutside->edges();
    for (std::vector<Edge*>::const_iterator i = cutedges.begin(); i != cutedges.end(); ++i)
    {
      Edge* cutsideedge = *i;
      cutsideedges.insert(cutsideedge);
    }
  }
  int cutsideedgessize = cutsideedges.size();
  file << "View \"" << cutsideedgessize << " Edges\" {\n";
  for (plain_edge_set::iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    cutsideedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsideedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_node_set cutsidenodes;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const std::vector<Node*>& cutnodes = cutside->nodes();
    for (std::vector<Node*>::const_iterator i = cutnodes.begin(); i != cutnodes.end(); ++i)
    {
      Node* cutsidenode = *i;
      cutsidenodes.insert(cutsidenode);
    }
  }
  int cutnodessize = cutsidenodes.size();
  file << "View \"" << cutnodessize << " Nodes\" {\n";
  for (plain_node_set::iterator i = cutsidenodes.begin(); i != cutsidenodes.end(); ++i)
  {
    Node* cutsidenode = *i;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    cutsidenode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = cutsidenode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";

  plain_side_set cuttingsides;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const plain_side_set& selfcutsides = cutside->cutting_sides();
    for (plain_side_set::const_iterator i = selfcutsides.begin(); i != selfcutsides.end(); ++i)
    {
      Side* cuttingside = *i;
      cuttingsides.insert(cuttingside);
    }
  }
  int cuttingsidessize = cuttingsides.size();
  file << "View \"" << cuttingsidessize << " Cuttingsides\" {\n";
  for (plain_side_set::iterator i = cuttingsides.begin(); i != cuttingsides.end(); ++i)
  {
    Side* cuttingside = *i;
    file.precision(16);
    int cutsidetype = cuttingside->nodes().size();
    if (cutsidetype == 3)
    {
      file << "ST (";
      Core::LinAlg::Matrix<3, 3> cutsidecoordinates;
      cuttingside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else if (cutsidetype == 4)
    {
      file << "SQ (";
      Core::LinAlg::Matrix<3, 4> cutsidecoordinates;
      cuttingside->coordinates(cutsidecoordinates.data());
      for (int i = 0; i < cutsidetype; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << cutsidecoordinates(0, i) << "," << cutsidecoordinates(1, i) << ","
             << cutsidecoordinates(2, i);
      }
    }
    else
    {
      // FOUR_C_THROW("SelfCut: irregular side");
      std::cout << "==| WARNING: SelfCut: irregular side 9 |==" << std::endl;
    }
    file << "){";
    double cutsideselfcutposition = -0.5;
    for (int i = 0; i < cutsidetype; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  plain_node_set selfcutnodes;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const plain_node_set& selfnodes = cutside->self_cut_nodes();
    for (plain_node_set::const_iterator i = selfnodes.begin(); i != selfnodes.end(); ++i)
    {
      Node* selfcutnode = *i;
      selfcutnodes.insert(selfcutnode);
    }
  }
  int selfcutnodessize = selfcutnodes.size();
  file << "View \"" << selfcutnodessize << " Selfcutnodes\" {\n";
  for (plain_node_set::iterator i = selfcutnodes.begin(); i != selfcutnodes.end(); ++i)
  {
    Node* selfcutnode = *i;
    file.precision(16);
    file << "SP (";
    Core::LinAlg::Matrix<3, 1> cutsidenodecoordinates;
    selfcutnode->coordinates(cutsidenodecoordinates.data());
    file << cutsidenodecoordinates(0, 0) << "," << cutsidenodecoordinates(1, 0) << ","
         << cutsidenodecoordinates(2, 0);
    file << "){";
    Point::PointPosition cutsideselfcutposition = selfcutnode->self_cut_position();
    file << cutsideselfcutposition;
    file << "};\n";
  }
  file << "};\n";

  plain_edge_set selfcutedges;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const plain_edge_set& selfedges = cutside->self_cut_edges();
    for (plain_edge_set::const_iterator i = selfedges.begin(); i != selfedges.end(); ++i)
    {
      Edge* selfcutedge = *i;
      selfcutedges.insert(selfcutedge);
    }
  }
  int selfcutedgessize = selfcutedges.size();
  file << "View \"" << selfcutedgessize << " Selfcutedges\" {\n";
  for (plain_edge_set::iterator i = selfcutedges.begin(); i != selfcutedges.end(); ++i)
  {
    Edge* selfcutedge = *i;
    file.precision(16);
    file << "SL (";
    Core::LinAlg::Matrix<3, 2> cutsideedgecoordinates;
    selfcutedge->coordinates(cutsideedgecoordinates.data());
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideedgecoordinates(0, i) << "," << cutsideedgecoordinates(1, i) << ","
           << cutsideedgecoordinates(2, i);
    }
    file << "){";
    Point::PointPosition cutsideselfcutposition = selfcutedge->self_cut_position();
    for (int i = 0; i < 2; ++i)
    {
      if (i > 0)
      {
        file << ", ";
      }
      file << cutsideselfcutposition;
    }
    file << "};\n";
  }
  file << "};\n";

  std::vector<std::vector<Point*>> selfcuttriangles;
  for (plain_side_set::iterator i = cutsides.begin(); i != cutsides.end(); ++i)
  {
    Side* cutside = *i;
    const std::vector<std::vector<Point*>>& selftriangles = cutside->self_cut_triangles();
    for (std::vector<std::vector<Point*>>::const_iterator i = selftriangles.begin();
        i != selftriangles.end(); ++i)
    {
      std::vector<Point*> selfcuttriangle = *i;
      selfcuttriangles.push_back(selfcuttriangle);
    }
  }
  int selfcuttrianglessize = selfcuttriangles.size();
  file << "View \"" << selfcuttrianglessize << " Selfcuttriangles\" {\n";
  for (std::vector<std::vector<Point*>>::iterator i = selfcuttriangles.begin();
      i != selfcuttriangles.end(); ++i)
  {
    std::vector<Point*> selfcuttriangle = *i;
    int selfcuttriangleposition = 0;
    int elementfacetpointssize = selfcuttriangle.size();
    for (std::vector<Point*>::iterator i = selfcuttriangle.begin(); i != selfcuttriangle.end(); ++i)
    {
      Point* elementfacetpoint1 = *i;
      Point* elementfacetpoint2 = nullptr;
      if (i + 1 != selfcuttriangle.end())
      {
        elementfacetpoint2 = *(i + 1);
      }
      else
      {
        elementfacetpoint2 = *(i + 1 - elementfacetpointssize);
      }
      file << "SL (";
      Core::LinAlg::Matrix<3, 1> elementfacetpoint1coordinates;
      elementfacetpoint1->coordinates(elementfacetpoint1coordinates.data());
      Core::LinAlg::Matrix<3, 1> elementfacetpoint2coordinates;
      elementfacetpoint2->coordinates(elementfacetpoint2coordinates.data());
      file << elementfacetpoint1coordinates(0, 0) << "," << elementfacetpoint1coordinates(1, 0)
           << "," << elementfacetpoint1coordinates(2, 0) << ","
           << elementfacetpoint2coordinates(0, 0) << "," << elementfacetpoint2coordinates(1, 0)
           << "," << elementfacetpoint2coordinates(2, 0);
      file << "){";
      for (int i = 0; i < 2; ++i)
      {
        if (i > 0)
        {
          file << ", ";
        }
        file << selfcuttriangleposition;
      }
      file << "};\n";
    }
  }
  file << "};\n";
}

/*-------------------------------------------------------------------------------------*
 * cuts the edges of the first cutside with the other cutside               wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::perform_self_cut(Side& cutside, Side& otherside, PointSet& selfcutpoints)
{
  const std::vector<Edge*>& cutsideedges = cutside.edges();
  for (std::vector<Edge*>::const_iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    PointSet sidepairselfcutpoints;
    cutsideedge->find_cut_points_mesh_cut(
        mesh_, nullptr, cutside, otherside, &sidepairselfcutpoints);
    for (PointSet::iterator i = sidepairselfcutpoints.begin(); i != sidepairselfcutpoints.end();
        ++i)
    {
      Point* edgeselfcutpoint = *i;
      cutsideedge->add_point(edgeselfcutpoint);
      selfcutpoints.insert(edgeselfcutpoint);
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * gets the edges of the new created sides                                  wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::get_self_cut_edges(Side& cutside)
{
  const std::vector<Edge*>& cutsideedges = cutside.edges();
  for (std::vector<Edge*>::const_iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    plain_int_set cutsideedgenodeids;
    const std::vector<Node*>& cutsideedgenodes = cutsideedge->nodes();
    for (std::vector<Node*>::const_iterator i = cutsideedgenodes.begin();
        i != cutsideedgenodes.end(); ++i)
    {
      Node* cutsideedgenode = *i;
      int cutsideedgenodeid = cutsideedgenode->id();
      cutsideedgenodeids.insert(cutsideedgenodeid);
    }

    std::map<plain_int_set, std::shared_ptr<Edge>>::iterator edgeiterator =
        selfcut_edges_.find(cutsideedgenodeids);
    if (edgeiterator == selfcut_edges_.end())
    {
      const std::map<plain_int_set, std::shared_ptr<Edge>>& cutsideedgercp = mesh_.edges();
      std::map<plain_int_set, std::shared_ptr<Edge>>::const_iterator edgeiterator =
          cutsideedgercp.find(cutsideedgenodeids);
      selfcut_edges_[cutsideedgenodeids] = edgeiterator->second;
    }
  }
}

/*-------------------------------------------------------------------------------------*
 * checks if the direction of rotation of the cycle is correct              wirtz 05/13
 *-------------------------------------------------------------------------------------*/
bool Cut::SelfCut::check_normal(std::vector<double> cutsideplane, Cycle& maincycle)
{
  std::vector<Point*> maincyclepoints = maincycle();
  std::vector<double> maincycleplane = Kernel::eqn_plane_of_polygon(maincyclepoints);
  if (fabs(cutsideplane[0]) > TOL_EQN_PLANE and cutsideplane[0] * maincycleplane[0] < 0.0)
  {
    return false;
  }
  else if (fabs(cutsideplane[1]) > TOL_EQN_PLANE and cutsideplane[1] * maincycleplane[1] < 0.0)
  {
    return false;
  }
  else if (fabs(cutsideplane[2]) > TOL_EQN_PLANE and cutsideplane[2] * maincycleplane[2] < 0.0)
  {
    return false;
  }
  else
  {
    return true;
  }
}

/*-------------------------------------------------------------------------------------*
 * deletes all pointers which are pointing to a soon to be erased side      wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_side_pointer(Side& cutside, bool erase_nodepointers)
{
  const std::vector<Edge*>& cutsideedges = cutside.edges();
  for (std::vector<Edge*>::const_iterator i = cutsideedges.begin(); i != cutsideedges.end(); ++i)
  {
    Edge* cutsideedge = *i;
    cutsideedge->erase_cut_side(&cutside);
  }
  const PointSet& cutsidepoints = cutside.cut_points();
  for (PointSet::const_iterator i = cutsidepoints.begin(); i != cutsidepoints.end(); ++i)
  {
    Point* cutsidepoint = *i;
    if (erase_nodepointers) cutsidepoint->erase_cut_side(&cutside);
    cutsidepoint->erased_containing_cut_pairs(&cutside);
  }
}

/*-------------------------------------------------------------------------------------*
 * erases a side                                                            wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_side(std::vector<plain_int_set>& cutsideids)
{
  for (std::vector<plain_int_set>::iterator i = cutsideids.begin(); i != cutsideids.end(); ++i)
  {
    const plain_int_set& cutsideid = *i;
    selfcut_sides_.erase(cutsideid);
    meshhandle_.remove_sub_side(&(*mesh_.sides().at(cutsideid)));
    mesh_.erase_side(cutsideid);
  }
}

void Cut::SelfCut::erase_inside_side(std::vector<plain_int_set>& cutsideids)
{
  for (std::vector<plain_int_set>::iterator i = cutsideids.begin(); i != cutsideids.end(); ++i)
  {
    const plain_int_set& cutsideid = *i;
    selfcut_sides_.erase(cutsideid);
    meshhandle_.mark_sub_sideas_unphysical(&(*mesh_.sides().at(cutsideid)));
    mesh_.move_sideto_storage(cutsideid);
  }
}

/*-------------------------------------------------------------------------------------*
 * deletes all pointers which are pointing to a soon to be erased edge      wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_edge_pointer(Edge& cutsideedge)
{
  const std::vector<Node*>& cutsideedgenodes = cutsideedge.nodes();
  for (std::vector<Node*>::const_iterator i = cutsideedgenodes.begin(); i != cutsideedgenodes.end();
      ++i)
  {
    Node* cutsideedgenode = *i;
    cutsideedgenode->erase_cut_side_edge(&cutsideedge);
  }
  const PointPositionSet& cutsideedgepoints = cutsideedge.cut_points();
  for (PointPositionSet::const_iterator i = cutsideedgepoints.begin(); i != cutsideedgepoints.end();
      ++i)
  {
    Point* cutsideedgepoint = *i;
    cutsideedgepoint->erase_cut_side_edge(&cutsideedge);
    cutsideedgepoint->erased_containing_cut_pairs(&cutsideedge);
  }
}

/*-------------------------------------------------------------------------------------*
 * erases an edge                                                           wirtz 05/13
 *-------------------------------------------------------------------------------------*/
void Cut::SelfCut::erase_edge(std::vector<plain_int_set>& cutsideedgeids)
{
  for (std::vector<plain_int_set>::iterator i = cutsideedgeids.begin(); i != cutsideedgeids.end();
      ++i)
  {
    const plain_int_set& cutsideedgeid = *i;
    selfcut_edges_.erase(cutsideedgeid);
    mesh_.erase_edge(cutsideedgeid);
  }
}

void Cut::SelfCut::plot_across()
{
  std::cout << "\n================================================================================="
               "===================================\n";
}

void Cut::SelfCut::point_plot_head()
{
  std::cout << "x\ty\tz\t# pid\n"
            << "-----------------------------\n";
}

void Cut::SelfCut::node_plot_head()
{
  std::cout << "x\ty\tz\t# pid\t# nid\tSelfCutPosition\n"
            << "-------------------------------------------------------\n";
}

void Cut::SelfCut::edge_plot_head()
{
  for (int i = 1; i < 3; ++i)
  {
    std::cout << "# nid_" << i << "\t";
  }
  std::cout << "SelfCutPosition\n"
            << "-------------------------------\n";
}

void Cut::SelfCut::side_plot_head()
{
  for (int i = 1; i < 4; ++i)
  {
    std::cout << "# nid_" << i << "\t";
  }
  std::cout << "# sid\tSelfCutPosition\n"
            << "-----------------------------------------------\n";
}

void Cut::SelfCut::point_plot(Point& point)
{
  Core::LinAlg::Matrix<3, 1> pointcoordinates;
  point.coordinates(pointcoordinates.data());
  std::cout << std::setprecision(16) << pointcoordinates(0) << "\t" << std::setprecision(16)
            << pointcoordinates(1) << "\t" << std::setprecision(16) << pointcoordinates(2) << "\t# "
            << point.id();
}

void Cut::SelfCut::node_plot(Node& node)
{
  point_plot(*node.point());
  std::cout << "\t# " << node.id() << "\t" << node.self_cut_position() << "\n";
}

void Cut::SelfCut::edge_plot(Edge& edge)
{
  const std::vector<Node*>& edgenodes = edge.nodes();
  for (unsigned int i = 0; i < edgenodes.size(); ++i)
  {
    std::cout << "# " << edgenodes[i]->id() << "\t";
  }
  std::cout << edge.self_cut_position() << "\n";
}

void Cut::SelfCut::side_plot(Side& side)
{
  const std::vector<Node*>& sidenodes = side.nodes();
  for (unsigned int i = 0; i < sidenodes.size(); ++i)
  {
    std::cout << "# " << sidenodes[i]->id() << "\t";
  }
  std::cout << "# " << side.id() << "\t" << side.self_cut_position() << "\n";
}

void Cut::SelfCut::side_node_plot(Side& side)
{
  const std::vector<Node*>& sidenodes = side.nodes();
  node_plot_head();
  for (std::vector<Node*>::const_iterator i = sidenodes.begin(); i != sidenodes.end(); ++i)
  {
    Node* sidenode = *i;
    node_plot(*sidenode);
  }
}

FOUR_C_NAMESPACE_CLOSE
