// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_superconvergent_patch_recovery.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_gauss.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::FE::compute_superconvergent_patch_recovery(
    Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& state,
    const std::string& statename, const int numvec, Teuchos::ParameterList& params)
{
  const int dimp = dim + 1;
  const int myrank = Core::Communication::my_mpi_rank(dis.get_comm());

  // check whether action type is set
  if (params.getEntryRCP("action") == Teuchos::null)
    FOUR_C_THROW("action type for element is missing");

  // decide whether a dof or an element based map is given
  FOUR_C_ASSERT(state.get_map().point_same_as(*dis.dof_row_map()), "Only works for same maps.");

  // handle pbcs if existing
  // build inverse map from slave to master nodes
  std::map<int, int> slavetomastercolnodesmap;
  std::map<int, std::vector<int>>* allcoupledcolnodes = dis.get_all_pbc_coupled_col_nodes();

  if (allcoupledcolnodes)
  {
    for (const auto& [master_gid, slave_gids] : *allcoupledcolnodes)
    {
      for (const auto slave_gid : slave_gids)
      {
        slavetomastercolnodesmap[slave_gid] = master_gid;
      }
    }
  }

  // set up reduced node row map of fluid field
  std::vector<int> reducednoderowmap;
  std::vector<int> reducednodecolmap;
  const Core::LinAlg::Map* fullnoderowmap = dis.node_row_map();
  const Core::LinAlg::Map* fullnodecolmap = dis.node_col_map();

  // a little more memory than necessary is possibly reserved here
  reducednoderowmap.reserve(fullnoderowmap->num_my_elements());
  reducednodecolmap.reserve(fullnodecolmap->num_my_elements());

  for (int i = 0; i < fullnodecolmap->num_my_elements(); ++i)
  {
    const int nodeid = fullnodecolmap->gid(i);
    // do not add slave pbc nodes to reduced node maps
    if (slavetomastercolnodesmap.count(nodeid) == 0)
    {
      // fill reduced node col map
      reducednodecolmap.push_back(nodeid);
      // fill reduced node row map
      if (fullnoderowmap->my_gid(nodeid)) reducednoderowmap.push_back(nodeid);
    }
  }

  // build node row map which does not include slave pbc nodes
  Core::LinAlg::Map noderowmap(
      -1, (int)reducednoderowmap.size(), reducednoderowmap.data(), 0, fullnoderowmap->get_comm());
  // build node col map which does not include slave pbc nodes
  Core::LinAlg::Map nodecolmap(
      -1, (int)reducednodecolmap.size(), reducednodecolmap.data(), 0, fullnodecolmap->get_comm());


  // step 1: get state to be reconstruced (e.g. velocity gradient) at element
  // centers (for linear elements the centers are the superconvergent sampling points!)
  dis.clear_state();
  // Set ALE displacements here
  dis.set_state(statename, state);

  const Core::LinAlg::Map* elementrowmap = dis.element_row_map();
  Core::LinAlg::MultiVector<double> elevec_toberecovered(*elementrowmap, numvec, true);
  Core::LinAlg::MultiVector<double> centercoords(*elementrowmap, dim, true);

  std::vector<int> lm;
  std::vector<int> lmowner;
  std::vector<int> lmstride;
  Core::Elements::LocationArray la(dis.num_dof_sets());

  // define element matrices and vectors
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // get number of elements
  const int numele = dis.num_my_row_elements();

  // loop only row elements
  for (int i = 0; i < numele; ++i)
  {
    Core::Elements::Element* actele = dis.l_row_element(i);

    // get element location vector
    // Core::Elements::LocationArray la(1);
    actele->location_vector(dis, la);

    // Reshape element matrices and vectors and initialize to zero
    elevector1.size(numvec);
    elevector2.size(3);

    // call the element specific evaluate method (elevec1 = velocity gradient, elevec2 = element
    // centroid)
    actele->evaluate(params, dis, la, elematrix1, elematrix2, elevector1, elevector2, elevector3);

    // store computed values (e.g. velocity gradient) for each element
    for (int j = 0; j < numvec; ++j)
    {
      double val = elevector1(j);

      elevec_toberecovered.replace_local_value(i, j, val);
    }

    // store corresponding element centroid
    for (int d = 0; d < dim; ++d)
    {
      centercoords.replace_local_value(i, d, elevector2(d));
    }
  }  // end element loop

  Core::LinAlg::MultiVector<double> elevec_toberecovered_col(
      *(dis.element_col_map()), numvec, true);
  Core::LinAlg::export_to(elevec_toberecovered, elevec_toberecovered_col);
  Core::LinAlg::MultiVector<double> centercoords_col(*(dis.element_col_map()), dim, true);
  Core::LinAlg::export_to(centercoords, centercoords_col);

  // step 2: use precalculated (velocity) gradient for patch-recovery of gradient
  // solution vector based on reduced node row map
  Core::LinAlg::FEVector<double> nodevec(noderowmap, numvec, false);

  std::vector<const Core::Conditions::Condition*> conds;
  dis.get_condition("SPRboundary", conds);

  // SPR boundary condition must be set for all boundaries except pbc
  if (conds.size() != 1 && conds.size() != 0)
    FOUR_C_THROW("exactly one boundary including all outer nodes expected");

  if (allcoupledcolnodes->begin() == allcoupledcolnodes->end() && conds.size() == 0)
    FOUR_C_THROW(
        "Neither periodic boundary conditions nor an SPRboundary is specified! Missing bc?");

  // loop all nodes
  for (int i = 0; i < nodecolmap.num_my_elements(); ++i)
  {
    const int nodegid = nodecolmap.gid(i);
    const Core::Nodes::Node* node = dis.g_node(nodegid);
    if (!node) FOUR_C_THROW("Cannot find with gid: {}", nodegid);

    // distinction between inner nodes and boundary nodes
    if (conds.size() == 0 || !conds[0]->contains_node(nodegid))
    {
      // continue with next node in case a ghost node is inner node
      if (node->owner() != myrank) continue;

      // distinction between normal inner node and pbc master node
      if (allcoupledcolnodes->find(nodegid) == allcoupledcolnodes->end())
      {
        //---------------------------------------------
        // we have an inner node here
        //---------------------------------------------

        // patch-recovery for each entry of the velocity gradient
        for (int j = 0; j < numvec; ++j)
        {
          static Core::LinAlg::Matrix<dimp, 1> p;
          p(0) = 1.0;
          static Core::LinAlg::Matrix<dimp, dimp> A;
          static Core::LinAlg::Matrix<dimp, 1> x;
          static Core::LinAlg::Matrix<dimp, 1> b;

          A.clear();
          b.clear();

          // loop over all surrounding elements
          for (auto adj_ele : node->adjacent_elements())
          {
            const int elelid = elevec_toberecovered_col.get_map().lid(adj_ele.global_id());
            for (int d = 0; d < dim; ++d)
              p(d + 1) = centercoords_col(d)[elelid] - node->x()[d] /* + ALE_DISP*/;

            // compute outer product of p x p and add to A
            A.multiply_nt(1.0, p, p, 1.0);

            b.update((elevec_toberecovered_col(j))[elelid], p, 1.0);
          }

          // solve for coefficients of interpolation
          const double det = Core::LinAlg::scaled_gauss_elimination<dimp>(A, b, x);
          if (det < 1.0e-14) FOUR_C_THROW("system singular, at inner node");

          // patch-recovery interpolation -> only first entry necessary, remaining ones are zero
          const double recoveredgradient = p(0) * x(0);

          // write solution vector
          nodevec.replace_global_values(1, &nodegid, &recoveredgradient, j);
        }
      }  // end normal inner node
      else
      {
        //---------------------------------------------
        // we have a pbc master node which is inner node
        //---------------------------------------------

        // get master nodes and corresponding slave nodes
        std::map<int, std::vector<int>>::const_iterator masternode =
            allcoupledcolnodes->find(nodegid);
        std::vector<int> slavenodeids = masternode->second;
        const int numslavenodes = (int)(masternode->second.size());
        // containers for adjacent elements to slave+master nodes
        std::vector<IteratorRange<DiscretizationIterator<ConstElementRef>>> adjacenteles;
        std::vector<double> offset(dim, 0.0);
        std::vector<std::vector<double>> eleoffsets(numslavenodes + 1, offset);
        for (int s = 0; s < numslavenodes; ++s)
        {
          const Core::Nodes::Node* slavenode = dis.g_node(slavenodeids[s]);
          // compute offset for slave elements
          for (int d = 0; d < dim; ++d)
            eleoffsets[s][d] = (node->x()[d] - slavenode->x()[d]) /* + ALE DISP */;

          // add adjacent elements of slave nodes to vector
          adjacenteles.emplace_back(slavenode->adjacent_elements());
        }
        // add elements connected to master node -> offset is zero for master elements
        adjacenteles.emplace_back(node->adjacent_elements());

        // patch-recovery for each entry of the velocity gradient
        for (int j = 0; j < numvec; ++j)
        {
          static Core::LinAlg::Matrix<dimp, 1> p;
          p(0) = 1.0;
          static Core::LinAlg::Matrix<dimp, dimp> A;
          static Core::LinAlg::Matrix<dimp, 1> x;
          static Core::LinAlg::Matrix<dimp, 1> b;

          A.clear();
          b.clear();

          // loop over all surrounding elements
          for (size_t s = 0; s < adjacenteles.size(); ++s)
          {
            for (auto ele : adjacenteles[s])
            {
              const int elelid = elevec_toberecovered_col.get_map().lid(ele.global_id());
              for (int d = 0; d < dim; ++d)
                p(d + 1) =
                    (centercoords_col(d))[elelid] + eleoffsets[s][d] - node->x()[d] /* + ALE_DISP*/;

              // compute outer product of p x p and add to A
              A.multiply_nt(1.0, p, p, 1.0);

              b.update((elevec_toberecovered_col(j))[elelid], p, 1.0);
            }
          }

          // solve for coefficients of interpolation
          const double det = Core::LinAlg::scaled_gauss_elimination<dimp>(A, b, x);
          if (det < 1.0e-14) FOUR_C_THROW("system singular, at pbc inner node");

          // patch-recovery interpolation -> only first entry necessary, remaining ones are zero
          const double recoveredgradient = p(0) * x(0);

          // write solution vector
          nodevec.replace_global_values(1, &nodegid, &recoveredgradient, j);
        }
      }  // end inner pbc master node
    }  // end inner nodes
    else
    {
      // we have a boundary node here -> patch is set up for closest inner node

      // distinction between normal boundary node and pbc master boundary node
      if (allcoupledcolnodes->find(nodegid) == allcoupledcolnodes->end())
      {
        //---------------------------------------------
        // we have a normal node at the boundary
        //---------------------------------------------

        // get all neighboring nodes of boundary node and find closest one
        double distance = 1.0e12;
        int closestnodeid = -1;
        for (auto adjacent_ele : node->adjacent_elements())
        {
          for (auto adjacent_node : adjacent_ele.nodes())
          {
            // continue with next node in case the neighbor is also on the boundary
            if (conds[0]->contains_node(adjacent_node.global_id())) continue;

            const auto& pos = adjacent_node.x(); /* + ALE DISP */
            static Core::LinAlg::Matrix<dim, 1> dist;
            for (int d = 0; d < dim; ++d) dist(d) = pos[d] - node->x()[d]; /* + ALE DISP */
            const double tmp = dist.norm2();
            if (tmp < distance and tmp > 1.0e-14)
            {
              distance = tmp;
              closestnodeid = adjacent_node.global_id();
            }
          }
        }

        if (closestnodeid == -1)
          FOUR_C_THROW(
              "no closest node not lying on a boundary could be found. The problem seems very "
              "small (at least in one direction)");

        // build patch for closest node and evaluate patch at boundary node
        const Core::Nodes::Node* closestnode = dis.g_node(closestnodeid);

        // leave here in case the closest node is a ghost node
        // only row nodes have all neighboring elements on this proc
        // this will result in off processor assembling (boundary node as ghost node)
        if (closestnode->owner() != myrank) continue;

        // patch-recovery for each entry of the velocity gradient
        for (int j = 0; j < numvec; ++j)
        {
          static Core::LinAlg::Matrix<dimp, 1> p;
          p(0) = 1.0;
          static Core::LinAlg::Matrix<dimp, dimp> A;
          static Core::LinAlg::Matrix<dimp, 1> x;
          static Core::LinAlg::Matrix<dimp, 1> b;

          A.clear();
          b.clear();

          // loop over all surrounding elements
          for (auto closest_node_adjacent_ele : closestnode->adjacent_elements())
          {
            const int elelid =
                elevec_toberecovered_col.get_map().lid(closest_node_adjacent_ele.global_id());
            for (int d = 0; d < dim; ++d)
              p(d + 1) = (centercoords_col(d))[elelid] - closestnode->x()[d]; /* + ALE_DISP*/

            // compute outer product of p x p and add to A
            A.multiply_nt(1.0, p, p, 1.0);

            b.update((elevec_toberecovered_col(j))[elelid], p, 1.0);
          }

          // solve for coefficients of interpolation
          const double det = Core::LinAlg::scaled_gauss_elimination<dimp>(A, b, x);
          if (det < 1.0e-14) FOUR_C_THROW("system singular, at boundary node");

          // patch-recovery interpolation for boundary point
          double recoveredgradient = p(0) * x(0);
          for (int d = 0; d < dim; ++d)
          {
            p(d + 1) = node->x()[d] - closestnode->x()[d] /* + ALE_DISP*/;
            recoveredgradient += p(d + 1) * x(d + 1);
          }

          // write solution vector
          nodevec.replace_global_values(1, &nodegid, &recoveredgradient, j);
        }
      }  // end normal boundary node
      else
      {
        //---------------------------------------------
        // we have a pbc master node at the boundary
        //---------------------------------------------

        // often bounds are axis aligned -> another pbc (master) node is closest node
        auto adjacentele = node->adjacent_elements();

        // leave here if the boundary node is a ghost node and has no adjacent elements on this proc
        // only boundary ghost nodes which have an inner node as a row node have all neighboring
        // elements on this proc this will result in off processor assembling (boundary ghost node
        // but closest node as row node)
        if (node->owner() != myrank && adjacentele.size() == 0) continue;

        double distance = 1.0e12;
        int closestnodeid = -1;
        for (auto ele : adjacentele)
        {
          for (auto adj_node : ele.nodes())
          {
            // continue with next node in case the neighbor is also on the boundary
            if (conds[0]->contains_node(adj_node.global_id())) continue;

            const auto& pos = adj_node.x(); /* + ALE DISP */
            static Core::LinAlg::Matrix<dim, 1> dist;
            for (int d = 0; d < dim; ++d) dist(d) = pos[d] - node->x()[d]; /* + ALE DISP */
            const double tmp = dist.norm2();
            if (tmp < distance and tmp > 1.0e-14)
            {
              distance = tmp;
              closestnodeid = adj_node.global_id();
            }
          }
        }

        if (closestnodeid == -1)
          FOUR_C_THROW(
              "no closest node _not_ lying on a boundary could be found. The problem seems very "
              "small (at least in one direction)");

        // build patch for closest node and evaluate patch at boundary node

        // get master nodes and corresponding slave nodes
        const Core::Nodes::Node* closestnode = dis.g_node(closestnodeid);

        // leave here in case the closest node is a ghost node
        // only row nodes have all neighboring elements on this proc
        // this will result in off processor assembling (boundary node as ghost node)
        if (closestnode->owner() != myrank) continue;

        std::map<int, std::vector<int>>::iterator masternode =
            allcoupledcolnodes->find(closestnodeid);

        int numslavenodes = -1;
        if (masternode != allcoupledcolnodes->end())
        {
          // closest node is (as expected) a master node
          numslavenodes = (int)(masternode->second.size());
        }
        else if (slavetomastercolnodesmap.count(closestnodeid) != 0)
        {
          // closest node is (surprisingly) a slave node
          int mastergid = slavetomastercolnodesmap[closestnodeid];
          masternode = allcoupledcolnodes->find(mastergid);
          numslavenodes = (int)(masternode->second.size());
        }
        else
        {
          // closest node is a standard node
          numslavenodes = 0;
        }

        // containers for adjacent elements to slave+master nodes
        std::vector<IteratorRange<DiscretizationIterator<ConstElementRef>>> closestnodeadjacenteles;
        std::vector<double> offset(dim, 0.0);
        std::vector<std::vector<double>> eleoffsets(numslavenodes + 1, offset);
        for (int s = 0; s < numslavenodes; ++s)
        {
          const Core::Nodes::Node* slavenode = dis.g_node(masternode->second[s]);
          // compute offset for slave elements
          for (int d = 0; d < dim; ++d)
            eleoffsets[s][d] = (closestnode->x()[d] - slavenode->x()[d]); /* + ALE DISP */

          // add adjacent elements of slave nodes to vectors
          closestnodeadjacenteles.emplace_back(slavenode->adjacent_elements());
        }
        // add elements connected to master node -> offset is zero for master elements
        closestnodeadjacenteles.emplace_back(closestnode->adjacent_elements());

        // patch-recovery for each entry of the velocity gradient
        for (int j = 0; j < numvec; ++j)
        {
          static Core::LinAlg::Matrix<dimp, 1> p;
          p(0) = 1.0;
          static Core::LinAlg::Matrix<dimp, dimp> A;
          static Core::LinAlg::Matrix<dimp, 1> x;
          static Core::LinAlg::Matrix<dimp, 1> b;

          A.clear();
          b.clear();

          // loop over all surrounding elements
          for (size_t s = 0; s < closestnodeadjacenteles.size(); ++s)
          {
            for (auto ele : closestnodeadjacenteles[s])
            {
              const int elelid = elevec_toberecovered_col.get_map().lid(ele.global_id());
              for (int d = 0; d < dim; ++d)
                p(d + 1) = (centercoords_col(d))[elelid] + eleoffsets[s][d] -
                           closestnode->x()[d]; /* + ALE_DISP*/

              // compute outer product of p x p and add to A
              A.multiply_nt(1.0, p, p, 1.0);

              b.update((elevec_toberecovered_col(j))[elelid], p, 1.0);
            }
          }

          // solve for coefficients of interpolation
          const double det = Core::LinAlg::scaled_gauss_elimination<dimp>(A, b, x);
          if (det < 1.0e-14) FOUR_C_THROW("system singular, at pbc boundary node");

          // patch-recovery interpolation for boundary point
          double recoveredgradient = p(0) * x(0);
          for (int d = 0; d < dim; ++d)
          {
            p(d + 1) = node->x()[d] - closestnode->x()[d] /* + ALE_DISP*/;
            recoveredgradient += p(d + 1) * x(d + 1);
          }

          // write solution vector
          nodevec.replace_global_values(1, &nodegid, &recoveredgradient, j);
        }
      }  // end boundary master pbc node
    }  // end boundary nodes

  }  // end loop over all nodes

  // call global assemble
  nodevec.complete(Insert, false);

  // if no pbc are involved leave here
  if (noderowmap.point_same_as(*fullnoderowmap))
    return std::make_shared<Core::LinAlg::MultiVector<double>>(
        nodevec.get_ref_of_epetra_fevector());

  // solution vector based on full row map in which the solution of the master node is inserted into
  // slave nodes
  std::shared_ptr<Core::LinAlg::MultiVector<double>> fullnodevec =
      std::make_shared<Core::LinAlg::MultiVector<double>>(*fullnoderowmap, numvec);

  for (int i = 0; i < fullnoderowmap->num_my_elements(); ++i)
  {
    const int nodeid = fullnoderowmap->gid(i);

    std::map<int, int>::iterator slavemasterpair = slavetomastercolnodesmap.find(nodeid);
    if (slavemasterpair != slavetomastercolnodesmap.end())
    {
      const int mastergid = slavemasterpair->second;
      const int masterlid = noderowmap.lid(mastergid);
      for (int j = 0; j < numvec; ++j)
        fullnodevec->replace_local_value(i, j, ((*(nodevec)(j))[masterlid]));
    }
    else
    {
      const int lid = noderowmap.lid(nodeid);
      for (int j = 0; j < numvec; ++j)
        fullnodevec->replace_local_value(i, j, ((*(nodevec)(j))[lid]));
    }
  }

  return fullnodevec;
}

template std::shared_ptr<Core::LinAlg::MultiVector<double>>
Core::FE::compute_superconvergent_patch_recovery<1>(Core::FE::Discretization&,
    const Core::LinAlg::Vector<double>&, const std::string&, const int, Teuchos::ParameterList&);
template std::shared_ptr<Core::LinAlg::MultiVector<double>>
Core::FE::compute_superconvergent_patch_recovery<2>(Core::FE::Discretization&,
    const Core::LinAlg::Vector<double>&, const std::string&, const int, Teuchos::ParameterList&);
template std::shared_ptr<Core::LinAlg::MultiVector<double>>
Core::FE::compute_superconvergent_patch_recovery<3>(Core::FE::Discretization&,
    const Core::LinAlg::Vector<double>&, const std::string&, const int, Teuchos::ParameterList&);

FOUR_C_NAMESPACE_CLOSE
