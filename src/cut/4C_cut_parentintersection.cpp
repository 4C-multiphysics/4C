// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_parentintersection.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_discretization.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------------------------------*
 * Create nodal dofset sets within the parallel cut framework
 *------------------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::create_nodal_dof_set(
    bool include_inner, const Core::FE::Discretization& dis)
{
  Core::Communication::barrier(dis.get_comm());

  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets --- CreateNodalDofSet");


  std::set<int> eids;  // eids of elements that are involved in CUT and include
                       // ele_vc_set_inside/outside (no duplicates!)

  Mesh& m = normal_mesh();

  const Cut::NodalDofSetStrategy strategy = options_.get_nodal_dof_set_strategy();

  // nodes used for CUT std::map<node->ID, Node>, shadow nodes have ID<0
  std::map<int, Node*> nodes;
  m.get_node_map(nodes);

  //===============
  // STEP 1: create for each node involved in CUT nodal cell sets for all nodes of adjacent elements
  // of this element
  //         nodal cell sets are sets of volumecells, that are connected via subelements in a real
  //         element degrees for linear elements a set contains just one single volumecell in its
  //         plain_volumecell_set(with usually more than one integrationcells) degrees for quadratic
  //         elements a set contains connected volumecell sets (sorted for inside and outside
  //         connections)
  //
  //         FindDOFSets() finds also the connections of volumecell sets between adjacent elements
  //         finally, each found DOFSet around a 1-ring of this node maintains its own set of DOFs
  //===============
  // for each node (which is not a shadow node) we have to built nodalDofSets
  // each connected set of volumecells (connected via adjacent elements - not only via subelements
  // -) gets an own dofset
  for (std::map<int, Node*>::iterator i = nodes.begin(); i != nodes.end(); ++i)
  {
    Node* n = i->second;
    int n_gid = n->id();

    std::vector<int> surrounding_elements;

    // get all adjacent elements to this node if this is a real (- not a shadow -) node
    if (n_gid >= 0)
    {
      Core::Nodes::Node* node = dis.g_node(n_gid);

      // get adjacent elements for this node
      const Core::Elements::Element* const* adjelements = node->elements();

      for (int iele = 0; iele < node->num_element(); iele++)
      {
        int adj_eid = adjelements[iele]->id();

        // get its elementhandle
        Cut::ElementHandle* e = get_element(adj_eid);

        if (e != nullptr)
        {
          surrounding_elements.push_back(adj_eid);
        }
      }  // end loop over adjacent elements



      // each node stores all its sets of volumecells,
      // includes all volumecell_sets that are connected (within a whole adjacent element) via
      // subelements, inside and outside sets (appended for all adjacent elements of a node)
      std::map<Node*, std::vector<plain_volumecell_set>>
          nodal_cell_sets_inside;  // for each node a vector of volumecell_sets
      std::map<Node*, std::vector<plain_volumecell_set>>
          nodal_cell_sets_outside;  // for each node a vector of volumecell_sets


      // includes all volumecell_sets that are connected (within a whole adjacent element) via
      // subelements, inside and outside sets
      std::vector<plain_volumecell_set> cell_sets;  // vector of all volumecell sets connected
                                                    // within elements adjacent to this node

      // split for inside and outside
      std::vector<plain_volumecell_set>
          cell_sets_inside;  // sets of volumecells connected between subelements
      std::vector<plain_volumecell_set> cell_sets_outside;

      find_nodal_cell_sets(include_inner, eids, surrounding_elements, nodal_cell_sets_inside,
          nodal_cell_sets_outside, cell_sets_inside, cell_sets_outside, cell_sets);


      // finds also the connections of volumecell sets between adjacent elements
      // finally, each found DOFSet around a 1-ring of the node maintains its own set of DOFs
      if (include_inner)
      {
        // WARNING:
        // This is necessary to have to set the "standard values" at the first DOF-set, and the
        // ghost values on the following DOFS.
        // It is important for result check AND later for time-integration!
        if (n->position() == Point::outside)
        {
          n->find_dof_sets_new(nodal_cell_sets_outside, cell_sets_outside);
          n->find_dof_sets_new(nodal_cell_sets_inside, cell_sets_inside);
        }
        else
        {
          n->find_dof_sets_new(nodal_cell_sets_inside, cell_sets_inside);
          n->find_dof_sets_new(nodal_cell_sets_outside, cell_sets_outside);
        }
      }
      else
      {
        n->find_dof_sets_new(nodal_cell_sets_outside, cell_sets_outside);
      }

      // sort the dofsets for this node after FindDOFSetsNEW
      n->sort_nodal_dof_sets();


      if (strategy == NDS_Strategy_OneDofset_PerNodeAndPosition)
      {
        /* combine the (ghost and standard) dofsets for this node w.r.t each phase to avoid
         * multiple ghost nodal dofsets for a certain phase */
        n->collect_nodal_dof_sets(true);
      }  // otherwise do nothing
      else if (strategy == NDS_Strategy_ConnectGhostDofsets_PerNodeAndPosition)
      {
        /* combine the only the ghost dofsets for this node w.r.t each phase to avoid
         * multiple ghost nodal dofsets for a certain phase */
        n->collect_nodal_dof_sets(false);
      }

    }  // end if n_gid >= 0
  }  // end loop over nodes


  //===============
  // STEP 2: for each element that contains volumecell_sets (connections via subelements...),
  // all nodes of this element have to know the dofset_number for each set of volumecells
  //===============
  for (std::set<int>::iterator i = eids.begin(); i != eids.end(); i++)
  {
    TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets --- STEP 2");


    int eid = *i;

    // get the nodes of this element
    // get the element via discret
    Core::Elements::Element* e = dis.g_element(eid);

    if (e == nullptr) FOUR_C_THROW(" element not found, this should not be! ");

    // get the nodes of this element
    int numnode = e->num_node();
    const int* nids = e->node_ids();
    std::vector<Node*> nodes(numnode);

    for (int i = 0; i < numnode; i++)
    {
      Node* node = get_node(nids[i]);

      if (node == nullptr) FOUR_C_THROW("node not found!");

      nodes[i] = node;
    }

    ElementHandle* eh = get_element(eid);

    // get inside and outside cell_sets connected within current element

    const std::vector<plain_volumecell_set>& ele_vc_sets_inside = eh->get_vc_sets_inside();
    const std::vector<plain_volumecell_set>& ele_vc_sets_outside = eh->get_vc_sets_outside();

    std::vector<std::vector<int>>& nodaldofset_vc_sets_inside =
        eh->get_nodal_dof_set_vc_sets_inside();
    std::vector<std::vector<int>>& nodaldofset_vc_sets_outside =
        eh->get_nodal_dof_set_vc_sets_outside();

    std::vector<std::map<int, int>>& vcsets_nid_dofsetnumber_map_toComm_inside =
        eh->get_node_dofset_map_vc_sets_inside_for_communication();
    std::vector<std::map<int, int>>& vcsets_nid_dofsetnumber_map_toComm_outside =
        eh->get_node_dofset_map_vc_sets_outside_for_communication();

    if (include_inner)
    {
      connect_nodal_dof_sets(nodes, include_inner, dis, ele_vc_sets_inside,
          nodaldofset_vc_sets_inside, vcsets_nid_dofsetnumber_map_toComm_inside);
    }

    connect_nodal_dof_sets(nodes, include_inner, dis, ele_vc_sets_outside,
        nodaldofset_vc_sets_outside, vcsets_nid_dofsetnumber_map_toComm_outside);
  }
}


/*--------------------------------------------------------------------------------------*
 | fill parallel DofSetData with information that has to be communicated   schott 03/12 |
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::fill_parallel_dof_set_data(
    std::vector<std::shared_ptr<DofSetData>>& parallel_dofSetData,
    const Core::FE::Discretization& dis, bool include_inner)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets --- fill_parallel_dof_set_data");

  // find volumecell sets and non-row nodes for that dofset numbers has to be communicated parallel
  // the communication is done element wise for all its sets of volumecells when there is a non-row
  // node in this element
  for (int k = 0; k < dis.num_my_col_elements(); ++k)
  {
    Core::Elements::Element* ele = dis.l_col_element(k);
    int eid = ele->id();
    Cut::ElementHandle* e = get_element(eid);

    if (e != nullptr)
    {
      if (include_inner)
      {
        // get inside cell_sets connected within current element
        const std::vector<plain_volumecell_set>& ele_vc_sets_inside = e->get_vc_sets_inside();
        std::vector<std::map<int, int>>& vcsets_nid_dofsetnumber_map_toComm_inside =
            e->get_node_dofset_map_vc_sets_inside_for_communication();

        int set_index = 0;
        // decide for each set of connected volumecells, if communication is necessary
        for (std::vector<std::map<int, int>>::iterator set_it =
                 vcsets_nid_dofsetnumber_map_toComm_inside.begin();
            set_it != vcsets_nid_dofsetnumber_map_toComm_inside.end(); set_it++)
        {
          // does the current set contain dofset data to communicate
          if (set_it->size() > 0)
          {
            // communicate data for the first Volumecell in this set
            // REMARK: all cells contained in a set carry the same dofset information

            // first vc in set
            VolumeCell* cell = *(ele_vc_sets_inside[set_index].begin());

            if (cell == nullptr) FOUR_C_THROW("pointer to first Volumecell of set is nullptr!");

            create_parallel_dof_set_data_vc(
                parallel_dofSetData, eid, set_index, true, cell, *set_it);
          }

          set_index++;
        }
      }

      // standard case for outside elements
      {
        // get outside cell_sets connected within current element
        const std::vector<plain_volumecell_set>& ele_vc_sets_outside = e->get_vc_sets_outside();
        std::vector<std::map<int, int>>& vcsets_nid_dofsetnumber_map_toComm_outside =
            e->get_node_dofset_map_vc_sets_outside_for_communication();

        int set_index = 0;
        // decide for each set of connected volumecells, if communication is necessary
        for (std::vector<std::map<int, int>>::iterator set_it =
                 vcsets_nid_dofsetnumber_map_toComm_outside.begin();
            set_it != vcsets_nid_dofsetnumber_map_toComm_outside.end(); set_it++)
        {
          // does the current set contain dofset data to communicate
          if (set_it->size() > 0)
          {
            // communicate data for the first Volumecell in this set
            // REMARK: all cells contained in a set carry the same dofset information

            // first vc in set
            VolumeCell* cell = *(ele_vc_sets_outside[set_index].begin());

            if (cell == nullptr) FOUR_C_THROW("pointer to first Volumecell of set is nullptr!");

            create_parallel_dof_set_data_vc(
                parallel_dofSetData, eid, set_index, false, cell, *set_it);
          }



          set_index++;
        }
      }
    }

  }  // end col elements
}


/*--------------------------------------------------------------------------------------*
 | create parallel DofSetData for a volumecell that has to be communicated schott 03/12 |
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::create_parallel_dof_set_data_vc(
    std::vector<std::shared_ptr<DofSetData>>& parallel_dofSetData, int eid, int set_index,
    bool inside, VolumeCell* cell, std::map<int, int>& node_dofset_map)
{
  if (node_dofset_map.size() > 0)
  {
    // get volumecell information
    // REMARK: identify volumecells using the volumecells (its facets) points
    plain_point_set
        cut_points;  // use sets here such that common points of facets are not stored twice

    {
      // get all the facets points
      const plain_facet_set& facets = cell->facets();

      for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
      {
        Facet* f = *i;

        // decide which points has to be send!!
        // Points, CornerPoints, AllPoints
        const std::vector<Point*>& facetpoints = f->points();

        std::copy(
            facetpoints.begin(), facetpoints.end(), std::inserter(cut_points, cut_points.begin()));
      }
    }

    Core::LinAlg::Matrix<3, 1> coords(Core::LinAlg::Initialization::zero);

    std::vector<Core::LinAlg::Matrix<3, 1>> cut_points_coords(cut_points.size(), coords);

    for (plain_point_set::iterator p = cut_points.begin(); p != cut_points.end(); ++p)
    {
      const int idx = std::distance(cut_points.begin(), p);
      Core::LinAlg::Matrix<3, 1>& xyz = cut_points_coords[idx];

      std::copy((*p)->x(), (*p)->x() + 3, &xyz(0, 0));
    }


    // get the parent element Id
    // REMARK: for quadratic elements use the eid for the base element, not -1 for subelements

    // create dofset data for this volumecell for Communication
    parallel_dofSetData.push_back(
        std::make_shared<DofSetData>(set_index, inside, cut_points_coords, eid, node_dofset_map));
  }
  else
    FOUR_C_THROW("communication for empty node-dofset map not necessary!");
}


/*--------------------------------------------------------------------------------------*
 | find cell sets around each node (especially for quadratic elements)     schott 03/12 |
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::find_nodal_cell_sets(bool include_inner, std::set<int>& eids,
    std::vector<int>& surrounding_elements,
    std::map<Node*, std::vector<plain_volumecell_set>>& nodal_cell_sets_inside,
    std::map<Node*, std::vector<plain_volumecell_set>>& nodal_cell_sets_outside,
    std::vector<plain_volumecell_set>& cell_sets_inside,
    std::vector<plain_volumecell_set>& cell_sets_outside,
    std::vector<plain_volumecell_set>& cell_sets)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets --- FindNodalCellSets");

  for (std::vector<int>::iterator i = surrounding_elements.begin(); i != surrounding_elements.end();
      ++i)
  {
    int eid = *i;

    ElementHandle* e = get_element(eid);

    const std::vector<plain_volumecell_set>& ele_vc_sets_inside = e->get_vc_sets_inside();
    const std::vector<plain_volumecell_set>& ele_vc_sets_outside = e->get_vc_sets_outside();
    //    e->VolumeCellSets( include_inner, ele_vc_sets_inside, ele_vc_sets_outside);

    // copy into cell_sets that collects all sets of adjacent elements
    if (include_inner)
    {
      std::copy(ele_vc_sets_inside.begin(), ele_vc_sets_inside.end(),
          std::inserter(cell_sets, cell_sets.end()));
      std::copy(ele_vc_sets_inside.begin(), ele_vc_sets_inside.end(),
          std::inserter(cell_sets_inside, cell_sets_inside.end()));
    }

    std::copy(ele_vc_sets_outside.begin(), ele_vc_sets_outside.end(),
        std::inserter(cell_sets, cell_sets.end()));
    std::copy(ele_vc_sets_outside.begin(), ele_vc_sets_outside.end(),
        std::inserter(cell_sets_outside, cell_sets_outside.end()));


    if ((ele_vc_sets_inside.size() > 0 and include_inner) or ele_vc_sets_outside.size() > 0)
    {
      eids.insert(eid);  // no duplicates in std::set
    }

    const std::vector<Node*>& nodes = e->nodes();


    for (std::vector<Node*>::const_iterator n = nodes.begin(); n != nodes.end(); ++n)
    {
      Node* node = *n;

      // call once for inside and once for outside
      {
        if (include_inner)
        {
          node->assign_nodal_cell_set(ele_vc_sets_inside, nodal_cell_sets_inside);
        }

        node->assign_nodal_cell_set(ele_vc_sets_outside, nodal_cell_sets_outside);
      }
    }  // end loop over nodes of current surrounding element
  }
}

/*--------------------------------------------------------------------------------------*
 | connect sets of volumecells for neighboring elements around a node      schott 03/12 |
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::connect_nodal_dof_sets(std::vector<Node*>& nodes, bool include_inner,
    const Core::FE::Discretization& dis,
    const std::vector<plain_volumecell_set>&
        connected_vc_sets,  // connections of volumecells within one element connected via
                            // subelements
    std::vector<std::vector<int>>& nodaldofset_vc_sets,
    std::vector<std::map<int, int>>& vcsets_nid_dofsetnumber_map_toComm)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 5/6 --- cut_positions_dofsets --- ConnectNodalDOFSets");


  for (std::vector<plain_volumecell_set>::const_iterator s =
           connected_vc_sets.begin();  // connections within this element
      s != connected_vc_sets.end(); s++)
  {
    const plain_volumecell_set& cells =
        *s;  // this is one connection of volumecells, connected via subelements, within one element

    std::vector<int> nds;

    // fill the map with nids, whose dofsets for the current set of volumecells has to filled by the
    // nodes row proc initialize the value (dofset_number with -1)
    std::map<int, int> nids_dofsetnumber_map_toComm;

    // find this plain_volumecell_set in dof_cellsets_ vector of each node
    {
      for (std::vector<Node*>::iterator i = nodes.begin(); i != nodes.end(); ++i)
      {
        //                Node * n = *i;
        //
        //                if( n->Id() >= 0) nds.push_back( n->DofSetNumberNEW( cells ) );
        //                else FOUR_C_THROW("node with negative Id gets no dofnumber!");

        Node* n = *i;

        int nid = n->id();

        if (nid >= 0)
        {
          Core::Nodes::Node* drt_node = dis.g_node(nid);

          // decide if the information for this cell has to be ordered from row-node or not
          // REMARK:
          if (drt_node->owner() == Core::Communication::my_mpi_rank(dis.get_comm()))
          {
            nds.push_back(n->dof_set_number_new(cells));
          }
          else
          {
            // insert the required pair of nid and unset dofsetnumber value (-1)
            nids_dofsetnumber_map_toComm[nid] = -1;

            // set dofset number to minus one, not a valid dofset number
            nds.push_back(-1);
          }
        }
        else
          FOUR_C_THROW("node with negative Id gets no dofnumber!");
      }
    }

    vcsets_nid_dofsetnumber_map_toComm.push_back(nids_dofsetnumber_map_toComm);

    // set the nds vector for each volumecell of the current set
    for (plain_volumecell_set::const_iterator c = cells.begin(); c != cells.end(); c++)
    {
      VolumeCell* cell = *c;
      cell->set_nodal_dof_set(nds);
    }

    nodaldofset_vc_sets.push_back(nds);
  }
}


/*------------------------------------------------------------------------------------------------*
 * standard Cut routine for parallel XFSI, XFLUIDFLUID and Level set cut where dofsets and        *
 * node positions have to be parallelized                                            schott 03/12 *
 *------------------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::cut_finalize(bool include_inner, VCellGaussPts VCellgausstype,
    Cut::BCellGaussPts BCellgausstype, bool tetcellsonly, bool screenoutput)
{
  TEUCHOS_FUNC_TIME_MONITOR("Cut --- 6/6 --- Cut_Finalize");

  if (myrank_ == 0 and screenoutput) Core::IO::cout << "\t * 6/6 Cut_Finalize ...\t";

  //  const double t_start = Teuchos::Time::wallTime();

  //----------------------------------------------------------

  Mesh& m = normal_mesh();

  if (VCellgausstype == VCellGaussPts_Tessellation)
  {
    TEUCHOS_FUNC_TIME_MONITOR("XFEM::FluidWizard::Cut::Tessellation");
    m.create_integration_cells(
        0, tetcellsonly);  // boundary cells will be created within TetMesh.CreateElementTets
    // m.remove_empty_volume_cells();

    // Test:
    m.test_element_volume(true, VCellgausstype);
    if (myrank_ == 0 and screenoutput) Core::IO::cout << "\n\t *     TestElementVolume ...";
    m.test_facet_area();
    if (myrank_ == 0 and screenoutput) Core::IO::cout << "\n\t *     TestFacetArea ...";
  }
  else if (VCellgausstype == VCellGaussPts_MomentFitting)
  {
    TEUCHOS_FUNC_TIME_MONITOR("XFEM::FluidWizard::Cut::MomentFitting");
    m.moment_fit_gauss_weights(include_inner, BCellgausstype);
    m.test_facet_area();
  }
  else if (VCellgausstype == VCellGaussPts_DirectDivergence)
  {
    TEUCHOS_FUNC_TIME_MONITOR("XFEM::FluidWizard::Cut::DirectDivergence");

    m.direct_divergence_gauss_rule(include_inner, BCellgausstype);
  }
  else
    FOUR_C_THROW("Undefined option of volumecell gauss points generation");
}

/*--------------------------------------------------------------------------------------*
 * get the node based on node id
 *-------------------------------------------------------------------------------------*/
Cut::Node* Cut::ParentIntersection::get_node(int nid) const { return mesh_.get_node(nid); }

/*--------------------------------------------------------------------------------------*
 * get the mesh's side based on node ids and return the side
 *-------------------------------------------------------------------------------------*/
Cut::SideHandle* Cut::ParentIntersection::get_side(std::vector<int>& nodeids) const
{
  return mesh_.get_side(nodeids);
}

/*--------------------------------------------------------------------------------------*
 * get the mesh's side based on side id and return the sidehandle
 *-------------------------------------------------------------------------------------*/
Cut::SideHandle* Cut::ParentIntersection::get_side(int sid) const { return mesh_.get_side(sid); }

/*--------------------------------------------------------------------------------------*
 * get the mesh's element based on element id
 *-------------------------------------------------------------------------------------*/
Cut::ElementHandle* Cut::ParentIntersection::get_element(int eid) const
{
  return mesh_.get_element(eid);
}

/*--------------------------------------------------------------------------------------*
 * print cell statistics
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::print_cell_stats() { normal_mesh().print_cell_stats(); }

/*--------------------------------------------------------------------------------------*
 * write gmsh debug output for nodal cell sets
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::dump_gmsh_nodal_cell_set(
    std::map<Node*, std::vector<plain_volumecell_set>>& nodal_cell_sets,
    const Core::FE::Discretization& dis)
{
  std::string filename =
      "cut_test";  // ::Global::Problem::instance()->output_control_file()->file_name();
  std::stringstream str;
  str << filename << "CUT_NodalCellSet." << Core::Communication::my_mpi_rank(dis.get_comm())
      << ".pos";


  std::string name = str.str();

  std::ofstream file(name.c_str());



  // Gmsh output for the sets of volumecells (connected within a global element) that are assigned
  // to a node all the cells of a set get the node id of the node they are assigned to

  file << "View \"NodalCellSet\" {\n";

  for (std::map<Node*, std::vector<plain_volumecell_set>>::iterator i = nodal_cell_sets.begin();
      i != nodal_cell_sets.end(); i++)
  {
    Node* n = i->first;

    int nid = n->id();

    std::vector<plain_volumecell_set>& sets = i->second;

    for (std::vector<plain_volumecell_set>::iterator s = sets.begin(); s != sets.end(); s++)
    {
      const plain_volumecell_set& volumes = *s;

      for (plain_volumecell_set::const_iterator i = volumes.begin(); i != volumes.end(); ++i)
      {
        VolumeCell* vc = *i;

        const plain_integrationcell_set& integrationcells = vc->integration_cells();
        for (plain_integrationcell_set::const_iterator i = integrationcells.begin();
            i != integrationcells.end(); ++i)
        {
          IntegrationCell* ic = *i;
          ic->dump_gmsh(file, &nid);
        }
      }
    }
  }

  file << "};\n";



  // Gmsh output, additional information (node Ids)

  file << "View \"NodeID\" {\n";

  for (std::map<Node*, std::vector<plain_volumecell_set>>::iterator i = nodal_cell_sets.begin();
      i != nodal_cell_sets.end(); i++)
  {
    Node* n = i->first;

    int nid = n->id();


    Point* p = n->point();
    const double* x = p->x();

    // output just for real nodes of elements, not for shadow nodes
    file << "SP(" << x[0] << "," << x[1] << "," << x[2] << "){" << nid << "};\n";
  }

  file << "};\n";
}

/*--------------------------------------------------------------------------------------*
 * write gmsh debug output for CellSets
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::dump_gmsh_cell_sets(
    std::vector<plain_volumecell_set>& cell_sets, const Core::FE::Discretization& dis)
{
  std::string filename =
      "cut_test";  // ::Global::Problem::instance()->output_control_file()->file_name();
  std::stringstream str;
  str << filename << "CUT_CellSets." << Core::Communication::my_mpi_rank(dis.get_comm()) << ".pos";


  std::string name = str.str();

  std::ofstream file(name.c_str());


  // Gmsh output for all sets of connected volumecells (connected within a global element) that are
  // assigned to a node

  plain_volumecell_set cells;

  // get cell_sets as a plain_volume_set
  for (std::vector<plain_volumecell_set>::iterator i = cell_sets.begin(); i != cell_sets.end(); i++)
  {
    std::copy((*i).begin(), (*i).end(), std::inserter(cells, cells.begin()));
  }

  file << "View \"CellSet\" {\n";
  int count = 0;

  for (plain_volumecell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
  {
    count++;
    VolumeCell* vc = *i;

    const plain_integrationcell_set& integrationcells = vc->integration_cells();
    for (plain_integrationcell_set::const_iterator i = integrationcells.begin();
        i != integrationcells.end(); ++i)
    {
      IntegrationCell* ic = *i;
      ic->dump_gmsh(file, &count);
    }
  }


  file << "};\n";
}


/*--------------------------------------------------------------------------------------*
 * write gmsh cut output for number of dofsets and the connected vc sets
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::dump_gmsh_num_dof_sets(
    std::string filename, bool include_inner, const Core::FE::Discretization& dis)
{
  std::stringstream str;
  str << filename << ".CUT_NumDOFSets." << Core::Communication::my_mpi_rank(dis.get_comm())
      << ".pos";


  Mesh& m = normal_mesh();



  std::string name = str.str();

  std::ofstream file(name.c_str());


  // Gmsh output for all sets of connected volumecells (connected within a global element) separated
  // for inside and outside each set gets its own number ( inside (negative) , outside(positive) )

  file << "View \"ConnectedVcSets\" {\n";
  int count_inside = -1;
  int count_outside = 0;

  int num_row_ele = dis.num_my_row_elements();

  for (int lid = 0; lid < num_row_ele;
      lid++)  // std::set<int>::iterator i= eids.begin(); i!= eids.end(); i++)
  {
    Core::Elements::Element* e = dis.l_row_element(lid);
    int eid = e->id();

    ElementHandle* eh = get_element(eid);

    if (eh != nullptr)
    {
      // get inside and outside cell_sets connected within current element
      const std::vector<plain_volumecell_set>& ele_vc_sets_inside = eh->get_vc_sets_inside();
      const std::vector<plain_volumecell_set>& ele_vc_sets_outside = eh->get_vc_sets_outside();


      for (std::vector<plain_volumecell_set>::const_iterator i = ele_vc_sets_outside.begin();
          i != ele_vc_sets_outside.end(); ++i)
      {
        plain_volumecell_set volumes = *i;

        for (plain_volumecell_set::const_iterator i = volumes.begin(); i != volumes.end(); ++i)
        {
          VolumeCell* vc = *i;

          const plain_integrationcell_set& integrationcells = vc->integration_cells();
          for (plain_integrationcell_set::const_iterator i = integrationcells.begin();
              i != integrationcells.end(); ++i)
          {
            IntegrationCell* ic = *i;
            ic->dump_gmsh(file, &count_outside);
          }
        }
        count_outside += 1;
      }

      // for inside cells
      if (include_inner)
      {
        for (std::vector<plain_volumecell_set>::const_iterator i = ele_vc_sets_inside.begin();
            i != ele_vc_sets_inside.end(); ++i)
        {
          const plain_volumecell_set& volumes = *i;

          for (plain_volumecell_set::const_iterator i = volumes.begin(); i != volumes.end(); ++i)
          {
            VolumeCell* vc = *i;

            const plain_integrationcell_set& integrationcells = vc->integration_cells();
            for (plain_integrationcell_set::const_iterator i = integrationcells.begin();
                i != integrationcells.end(); ++i)
            {
              IntegrationCell* ic = *i;
              ic->dump_gmsh(file, &count_inside);
            }
          }
          count_inside -= 1;
        }
      }
    }
  }

  file << "};\n";



  // Gmsh output for all dof sets of one node (connected via adjacent elements of this node)
  // each set gets its own number ( inside (negative) , outside(positive) )


  //    file << "View \"DofSets for special node\" {\n";
  //
  //
  //    Node* n = m.GetNode( nid );
  //    Node* n = m.GetNode( 70 );
  //    std::vector<std::set<plain_volumecell_set> > dof_cellsets = n->DofCellSets();
  //
  //    int count=0;
  ////    int count_vc = 0;
  //    for(std::vector<std::set<plain_volumecell_set> >::iterator i=dof_cellsets.begin();
  //    i!=dof_cellsets.end(); i++)
  //    {
  //        std::set<plain_volumecell_set> & cellset = *i;
  //
  //        for(std::set<plain_volumecell_set>::iterator vc_set=cellset.begin();
  //        vc_set!=cellset.end(); vc_set++)
  //        {
  //            const plain_volumecell_set & volumes = *vc_set;
  //
  //            for ( plain_volumecell_set::const_iterator i=volumes.begin(); i!=volumes.end(); ++i
  //            ) { //count_vc++;
  //                VolumeCell * vc = *i;
  //
  //                const plain_integrationcell_set & integrationcells = vc->IntegrationCells();
  //                for ( plain_integrationcell_set::const_iterator i=integrationcells.begin();
  //                      i!=integrationcells.end();
  //                      ++i )
  //                {
  //                    IntegrationCell * ic = *i;
  //                    ic->DumpGmsh( file, &count );
  //
  //                }
  //
  //            }
  //        }
  //        count++;
  //    }
  //    file << "};\n";

  // nodes used for CUT std::map<node->ID, Node>, shadow nodes have ID<0
  // print the dofsets just for the row nodes
  std::map<int, Node*> nodes;
  m.get_node_map(nodes);

  file << "View \"NumDofSets\" {\n";
  for (std::map<int, Node*>::iterator i = nodes.begin(); i != nodes.end(); ++i)
  {
    int nid = i->first;

    if (nid >= 0)
    {
      if (dis.node_row_map()->LID(nid) == -1) continue;  // non-local row node

      Node* n = i->second;
      Point* p = n->point();
      const double* x = p->x();

      // output just for real nodes of elements, not for shadow nodes
      if (n->id() >= 0)
        file << "SP(" << x[0] << "," << x[1] << "," << x[2] << "){" << n->num_dof_sets() << "};\n";
    }
  }
  file << "};\n";
}

/*--------------------------------------------------------------------------------------*
 * write gmsh output for volumecells
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::dump_gmsh_volume_cells(std::string name, bool include_inner)
{
  normal_mesh().dump_gmsh_volume_cells(name, include_inner);
}

/*--------------------------------------------------------------------------------------*
 * write gmsh output for volumecells
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::dump_gmsh_integration_cells(std::string name)
{
  normal_mesh().dump_gmsh_integration_cells(name);
}

/*--------------------------------------------------------------------------------------*
 * write gmsh output for volumecells
 *-------------------------------------------------------------------------------------*/
void Cut::ParentIntersection::dump_gmsh_volume_cells(std::string name)
{
  normal_mesh().dump_gmsh_volume_cells(name);
}

FOUR_C_NAMESPACE_CLOSE
