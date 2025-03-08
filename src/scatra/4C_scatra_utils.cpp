// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_utils.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraUtils::check_consistency_of_s2_i_conditions(
    std::shared_ptr<Core::FE::Discretization> discretization)
{
  // check if the number of s2i condition definition is correct
  std::vector<Core::Conditions::Condition*> s2ikinetics_conditions, s2isclcoupling_condition,
      s2imeshtying_conditions, s2inoevaluation_conditions;
  discretization->get_condition("S2IKinetics", s2ikinetics_conditions);
  discretization->get_condition("S2ISCLCoupling", s2isclcoupling_condition);
  discretization->get_condition("S2IMeshtying", s2imeshtying_conditions);
  discretization->get_condition("S2INoEvaluation", s2inoevaluation_conditions);

  if ((s2ikinetics_conditions.size() + s2isclcoupling_condition.size()) !=
      (s2imeshtying_conditions.size() + s2inoevaluation_conditions.size()))
  {
    FOUR_C_THROW(
        "For each 'S2IKinetics' or 'S2ISCLCoupling' condition a corresponding 'S2IMeshtying' or "
        "'S2INoEvaluation' condition has to be defined!");
  }

  // combine conditions that define the physics and the evaluation type
  std::vector<Core::Conditions::Condition*> s2ievaluation_conditions(s2imeshtying_conditions);
  s2ievaluation_conditions.insert(s2ievaluation_conditions.end(),
      s2inoevaluation_conditions.begin(), s2inoevaluation_conditions.end());
  std::vector<Core::Conditions::Condition*> s2iphysics_conditions(s2ikinetics_conditions);
  s2iphysics_conditions.insert(s2iphysics_conditions.end(), s2isclcoupling_condition.begin(),
      s2isclcoupling_condition.end());

  std::vector<int> s2ievaluation_nodes, s2iphysics_nodes;
  Core::Conditions::find_conditioned_nodes(
      *discretization, s2ievaluation_conditions, s2ievaluation_nodes);
  Core::Conditions::find_conditioned_nodes(
      *discretization, s2iphysics_conditions, s2iphysics_nodes);

  if (s2iphysics_nodes != s2ievaluation_nodes)
  {
    FOUR_C_THROW(
        "Definition of 'S2IKinetics' or 'S2ISCLCoupling' conditions and corresponding "
        "'S2IMeshtying' or 'S2INoEvaluation' conditions is inconsistent! The nodes the conditions "
        "are defined on do not match!");
  }

  ScaTraUtils::check_consistency_with_s2_i_kinetics_condition("S2IMeshtying", discretization);
  ScaTraUtils::check_consistency_with_s2_i_kinetics_condition("S2INoEvaluation", discretization);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraUtils::check_consistency_with_s2_i_kinetics_condition(
    const std::string& condition_to_be_tested,
    std::shared_ptr<Core::FE::Discretization> discretization)
{
  std::vector<Core::Conditions::Condition*> allConditionsToBeTested;
  discretization->get_condition(condition_to_be_tested, allConditionsToBeTested);
  std::vector<Core::Conditions::Condition*> s2ikinetics_conditions;
  discretization->get_condition("S2IKinetics", s2ikinetics_conditions);

  // loop over all conditions to be tested and check for a consistent initialization of the s2i
  // conditions
  for (const auto& conditionToBeTested : allConditionsToBeTested)
  {
    if (conditionToBeTested->g_type() != Core::Conditions::geometry_type_surface) continue;
    bool isslave(true);
    const int s2ikinetics_id = conditionToBeTested->parameters().get<int>("S2I_KINETICS_ID");

    // check the interface side
    switch (conditionToBeTested->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE"))
    {
      case Inpar::S2I::side_slave:
      {
        isslave = true;
        break;
      }
      case Inpar::S2I::side_master:
      {
        isslave = false;
        break;
      }
      default:
      {
        FOUR_C_THROW("interface side of %s has to be either 'Slave' or 'Master'",
            condition_to_be_tested.c_str());
        break;
      }
    }

    // loop over all s2i conditions to find the one that is matching the current ssi condition
    for (const auto& s2ikinetics_cond : s2ikinetics_conditions)
    {
      const int s2ikinetics_cond_id = s2ikinetics_cond->parameters().get<int>("ConditionID");
      // only do further checks if Ids match
      if (s2ikinetics_id != s2ikinetics_cond_id) continue;

      // check the interface side
      switch (s2ikinetics_cond->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE"))
      {
        case Inpar::S2I::side_slave:
        {
          if (isslave)
            Core::Conditions::have_same_nodes(conditionToBeTested, s2ikinetics_cond, true);

          break;
        }
        case Inpar::S2I::side_master:
        {
          if (!isslave)
            Core::Conditions::have_same_nodes(conditionToBeTested, s2ikinetics_cond, true);

          break;
        }
        default:
        {
          FOUR_C_THROW(
              "interface side of 'S2IKinetics' condition has to be either 'Slave' or 'Master'");
          break;
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <const int dim>
std::shared_ptr<Core::LinAlg::MultiVector<double>>
ScaTra::ScaTraUtils::compute_gradient_at_nodes_mean_average(Core::FE::Discretization& discret,
    const Core::LinAlg::Vector<double>& state, const int scatra_dofid)
{
  // number space dimensions
  const size_t nsd = dim;
  // const int scatra_dofid = 0; //<  this is the first DoFSet (i.e. the scalar one!!)

  if (nsd != 3)
    FOUR_C_THROW("Only implemented for 3D elements. Should be simple enough to extend...");

  // DOF-COL-MAP
  Core::LinAlg::Vector<double> phinp_col(*discret.dof_col_map());
  // export dof_row_map to DofColMap phinp
  Core::LinAlg::export_to(state, phinp_col);

  // ---------------------------------------------------------------

  // before we can export to NodeColMap we need reconstruction with a NodeRowMap
  const std::shared_ptr<Core::LinAlg::MultiVector<double>> gradphirow =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*discret.dof_row_map(), nsd);
  gradphirow->PutScalar(0.0);

  // map of pointers to nodes which must be reconstructed by this processor <local id, node>
  std::map<int, const Core::Nodes::Node*> nodesToReconstruct;
  nodesToReconstruct.clear();

  //--------------------------------------------------------------------------------------------
  // PART I:  loop over all elements in column map to find all nodes which must be reconstructed
  // remark: intersected elements at processor boundary must be seen by all processors
  //--------------------------------------------------------------------------------------------
  for (int iele = 0; iele < discret.num_my_col_elements(); iele++)
  {
    // get element from fluid discretization
    const Core::Elements::Element* actele = discret.l_col_element(iele);

    // get number of nodes of this element (number of vertices)
    const int numberOfNodes = actele->num_node();
    // get vector of pointers of node (for this element)
    const Core::Nodes::Node* const* ele_vecOfPtsToNode = actele->nodes();

    // loop nodes of this element
    for (int vec_it = 0; vec_it < numberOfNodes; vec_it++)
    {
      // get owner of the node to compare with my_rank
      int node_owner = (ele_vecOfPtsToNode[vec_it])->owner();
      // check whether this node is a row node, compare with actual processor id
      if (node_owner == Core::Communication::my_mpi_rank(discret.get_comm()))
      {
        // insert in map (overwrite existing entry)
        int lid = ele_vecOfPtsToNode[vec_it]->lid();
        nodesToReconstruct[lid] = ele_vecOfPtsToNode[vec_it];
      }
      // remark: all non-row nodes are reconstructed by another processor
    }
  }  // end for loop over row elements

  //-------------------------------------------------------------------------------------------
  // PART II: reconstruct nodes
  // remark: row nodes adjacent to intersected elements must be reconstructed by this processor
  //-------------------------------------------------------------------------------------------
  // loop over all nodes inserted into map 'nodesToReconstruct'
  for (auto& it_node : nodesToReconstruct)
  {
    static Core::LinAlg::Matrix<nsd, 1>
        node_gradphi_smoothed;  // set whole 3D vector also for 2D examples
    node_gradphi_smoothed.clear();

    // get local processor id of current node and pointer to current node
    // int lid_node = it_node->first;
    const Core::Nodes::Node* ptToNode = it_node.second;
    const int nodegid = ptToNode->id();

    // vector of elements located around this node
    std::vector<const Core::Elements::Element*> elements;

    // get adjacent elements for this node
    const Core::Elements::Element* const* adjelements = ptToNode->elements();

    const Core::FE::CellType DISTYPE = adjelements[0]->shape();  // Core::FE::CellType::hex8;

    for (int iele = 0; iele < ptToNode->num_element(); iele++)
    {
      if (DISTYPE != adjelements[iele]->shape())
        FOUR_C_THROW("discretization not with same elements!!!");

      elements.push_back(adjelements[iele]);
    }

    // -------------------------------------------------------------
    //--------------------------------------
    // add elements along perodic boundaries
    //--------------------------------------
    // boolean indicating whether this node is a pbc node
    bool pbcnode = false;
    std::set<int> coupnodegid;
    // loop all nodes with periodic boundary conditions (master nodes)
    std::map<int, std::vector<int>>* pbccolmap = discret.get_all_pbc_coupled_col_nodes();

    if (pbcnode)
    {
      for (const auto& [master_gid, slave_gids] : *pbccolmap)
      {
        if (master_gid == nodegid)  // node is a pbc master node
        {
          pbcnode = true;
          // coupled node is the slave node; there can be more than one per master node
          for (int i : slave_gids) coupnodegid.insert(i);
        }
        else
        {
          // loop all slave nodes
          for (size_t islave = 0; islave < slave_gids.size(); islave++)
          {
            if (slave_gids[islave] == nodegid)  // node is a pbc slave node
            {
              pbcnode = true;
              // coupled node is the master node
              coupnodegid.insert(master_gid);

              // there can be multiple slaves -> add all other slaves
              for (size_t i = 0; i < slave_gids.size(); ++i)
              {
                if (slave_gids[islave] != slave_gids[i]) coupnodegid.insert(slave_gids[i]);
              }
            }
          }
        }
      }
    }

    // add elements located around the coupled pbc node
    if (pbcnode)
    {
      for (int icoupnode : coupnodegid)
      {
        // get coupled pbc node (master or slave)
        const Core::Nodes::Node* ptToCoupNode = discret.g_node(icoupnode);
        // get adjacent elements of this node
        const Core::Elements::Element* const* pbcelements = ptToCoupNode->elements();
        // add elements to list
        for (int iele = 0; iele < ptToCoupNode->num_element(); iele++)  // = ptToNode->Elements();
        {
          elements.push_back(pbcelements[iele]);
        }
      }
    }

    // -------------------------------------------------------------

    // FOR OTHER TYPES OF ELEMENT THAN HEX -> One needs to change DoMeanValueAveraging - code as
    // well.
    if (DISTYPE == Core::FE::CellType::hex8)
    {
      node_gradphi_smoothed =
          do_mean_value_averaging_of_element_gradient_node<nsd, Core::FE::CellType::hex8>(
              discret, elements, phinp_col, nodegid, scatra_dofid);
    }
    else if (DISTYPE == Core::FE::CellType::hex27)
    {
      node_gradphi_smoothed =
          do_mean_value_averaging_of_element_gradient_node<nsd, Core::FE::CellType::hex27>(
              discret, elements, phinp_col, nodegid, scatra_dofid);
    }
    else
      FOUR_C_THROW("Element type not supported yet!");

    //----------------------------------------------------------------------------------------------------
    // set the global vector gradphirow holding the new reconstructed values of gradient of phi in
    // row map
    //----------------------------------------------------------------------------------------------------

    const std::vector<int> lm = discret.dof(scatra_dofid, (it_node.second));
    if (lm.size() != 1) FOUR_C_THROW("assume a unique level-set dof in ScaTra DoFset");

    int GID = lm[0];  // Global ID of DoF Map
    // get local processor id according to global node id
    const int lid = (*gradphirow).Map().LID(GID);
    if (lid < 0)
      FOUR_C_THROW("Proc %d: Cannot find gid=%d in Core::LinAlg::Vector<double>",
          Core::Communication::my_mpi_rank((*gradphirow).Comm()), GID);

    const int numcol = (*gradphirow).NumVectors();
    if (numcol != (int)nsd)
      FOUR_C_THROW(
          "number of columns in Core::LinAlg::MultiVector<double> is not identically to nsd");

    // loop over dimensions (= number of columns in multivector)
    for (int col = 0; col < numcol; col++)
    {
      // set smoothed gradient entry of phi into column of global multivector
      (*gradphirow)(col)[lid] = node_gradphi_smoothed(col, 0);
    }
  }  // end loop over nodes

  return gradphirow;
}

template std::shared_ptr<Core::LinAlg::MultiVector<double>>
ScaTra::ScaTraUtils::compute_gradient_at_nodes_mean_average<3>(Core::FE::Discretization& discret,
    const Core::LinAlg::Vector<double>& state, const int scatra_dofid);



template <const int dim, Core::FE::CellType distype>
Core::LinAlg::Matrix<dim, 1> ScaTra::ScaTraUtils::do_mean_value_averaging_of_element_gradient_node(
    Core::FE::Discretization& discret, std::vector<const Core::Elements::Element*> elements,
    Core::LinAlg::Vector<double>& phinp_node, const int nodegid, const int scatra_dofid)
{
  // number of nodes of this element for interpolation
  const int numnode = Core::FE::num_nodes<distype>;
  Core::LinAlg::Matrix<dim, 1> node_gradphi_smoothed(true);

  // number of elements located around this node
  const int numberOfElements = static_cast<int>(elements.size());
  {
    //--------------------------------------------------------------------------
    // average (mean value) reconstruction for boundary nodes and as alternative
    //--------------------------------------------------------------------------
    // loop over elements located around this node
    for (int ele_current = 0; ele_current < numberOfElements; ele_current++)
    {
      // get current element
      const Core::Elements::Element* ele_adj = elements[ele_current];

      const int* ptToNodeIds_adj = ele_adj->node_ids();

      // get phi-values of current adjacent element ele_adj
      // create vector "ephinp" holding scalar phi values for this element
      Core::LinAlg::SerialDenseVector ephinp(
          numnode);  // local vector phi-values of adjacent element

      // which node in param space of element ele_adj has actnode
      int ID_param_space = -1;

      // get vector of node GIDs of this adjacent element -> needed for extract_my_values
      std::vector<int> nodeID_adj(numnode);
      std::vector<int> nodeDOFID_adj(numnode);
      for (int inode = 0; inode < numnode; inode++)
      {
        nodeID_adj[inode] = ptToNodeIds_adj[inode];

        const std::vector<int> lm = discret.dof(scatra_dofid, (ele_adj->nodes()[inode]));
        if (lm.size() != 1) FOUR_C_THROW("assume a unique level-set dof in cutterdis-Dofset");
        nodeDOFID_adj[inode] = lm[0];

        // get local number of node actnode in ele_adj
        if (nodegid == ptToNodeIds_adj[inode]) ID_param_space = inode;
      }
      if (ID_param_space < 0) FOUR_C_THROW("node not found in element");

      // extract the phi-values of adjacent element with local ids from global vector *phinp
      // get pointer to vector holding G-function values at the fluid nodes
      Core::FE::extract_my_values(phinp_node, ephinp, nodeDOFID_adj);
      Core::LinAlg::Matrix<numnode, 1> ephi_adj(ephinp);

      //-------------------------------------
      // compute gradient of phi at this node
      //-------------------------------------
      // get derivatives of shape functions evaluated at node in XYZ-coordinates
      static Core::LinAlg::Matrix<dim, numnode> deriv3Dele_xyz;
      // get derivatives of shape functions evaluates at node in Xi-coordinates
      static Core::LinAlg::Matrix<dim, numnode> deriv3Dele;

      // TODO: Implement for other elements than HEX
      // get Xi-coordinates of current node in current adjacent element
      static Core::LinAlg::Matrix<dim, 1> node_Xicoordinates;
      node_Xicoordinates.clear();
      for (int icomp = 0; icomp < dim; ++icomp)
      {
        node_Xicoordinates(icomp) =
            Core::FE::eleNodeNumbering_hex27_nodes_reference[ID_param_space][icomp];
      }

      // get derivatives of shape functions at node
      switch (dim)
      {
        case 3:
        {
          Core::FE::shape_function_3d_deriv1(deriv3Dele, node_Xicoordinates(0),
              node_Xicoordinates(1), node_Xicoordinates(2), distype);
          break;
        }
        case 2:
        {
          Core::FE::shape_function_2d_deriv1(
              deriv3Dele, node_Xicoordinates(0), node_Xicoordinates(1), distype);
          break;
        }
        case 1:
        {
          Core::FE::shape_function_1d_deriv1(deriv3Dele, node_Xicoordinates(0), distype);
          break;
        }
        default:
          FOUR_C_THROW("Only spacial dimension 1,2,3 are allowed!");
      }

      // reconstruct XYZ-gradient
      // get node coordinates of this element
      static Core::LinAlg::Matrix<dim, numnode> xyze_adj;
      Core::Geo::fill_initial_position_array<distype>(ele_adj, xyze_adj);

      // get Jacobi-Matrix for transformation
      static Core::LinAlg::Matrix<dim, dim> xjm_ele_XiToXYZ;
      xjm_ele_XiToXYZ.multiply_nt(deriv3Dele, xyze_adj);

      // inverse of jacobian
      static Core::LinAlg::Matrix<dim, dim> xji_ele_XiToXYZ;
      xji_ele_XiToXYZ.invert(xjm_ele_XiToXYZ);

      // set XYZ-derivates of shapefunctions
      deriv3Dele_xyz.multiply(xji_ele_XiToXYZ, deriv3Dele);

      //----------------------------------------------------
      // compute gradient of phi at node for current element
      //----------------------------------------------------
      static Core::LinAlg::Matrix<dim, 1> nodal_grad_tmp;
      nodal_grad_tmp.clear();

      // get xyz-gradient
      nodal_grad_tmp.multiply(deriv3Dele_xyz, ephi_adj);

      //===============================================================================
      // add to vector with smoothed vector

      node_gradphi_smoothed.update(1.0, nodal_grad_tmp, 1.0);

    }  // end loop over all adjacent elements

    // weight sum of nodal_grad_tmp 1/number_of_vectors to get an average value
    node_gradphi_smoothed.scale(1.0 / numberOfElements);
  }
  return node_gradphi_smoothed;
}

// Templates for Mean value averaging -- For now only HEX-type elements allowed!
template Core::LinAlg::Matrix<3, 1>
ScaTra::ScaTraUtils::do_mean_value_averaging_of_element_gradient_node<3, Core::FE::CellType::hex8>(
    Core::FE::Discretization& discret, std::vector<const Core::Elements::Element*> elements,
    Core::LinAlg::Vector<double>& phinp_node, const int nodegid, const int scatra_dofid);

template Core::LinAlg::Matrix<3, 1>
ScaTra::ScaTraUtils::do_mean_value_averaging_of_element_gradient_node<3, Core::FE::CellType::hex27>(
    Core::FE::Discretization& discret, std::vector<const Core::Elements::Element*> elements,
    Core::LinAlg::Vector<double>& phinp_node, const int nodegid, const int scatra_dofid);

FOUR_C_NAMESPACE_CLOSE
